/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file schedule_dataflow_rewrite.cc
 */

/*
 * 2019.12.30 - Add new conditions for compute op.
 * 2020.10.27 - For the body of the factor_op in rfactor, perform the additional
 *              processing when the reduce axis is empty.
 * 2021.10.28 - Add NormalizeBody step and Inline Inject logic for hybrid and extern op
 * 2021.12.15 - Add TmpVarInline logic in NormalizeBody for hybrid op
 * 2022.3.28 - Support inline for CSR operations.
 */

#include <tvm/schedule.h>
#include <tvm/operation.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include <unordered_map>
#include <unordered_set>
#include "message_passing.h"
#include "../pass/ir_util.h"
#include "../arithmetic/compute_expr.h"
#include "tvm/expr.h"

namespace air {
using namespace ir;

struct ExprLess {
  bool operator()(const Expr& l, const Expr& r) const { return Compare(l, r) < 0; }
};

// find first occurance location in leaf
template<typename T>
size_t FindNodeRef(ArrayNode* array_node, const T& v) {
  const Node* n = v.get();
  for (size_t i = 0; i < array_node->data.size(); ++i) {
    if (array_node->data[i].get() == n) return i;
  }
  return array_node->data.size();
}

// The replacer of cache.
class VarReplacer : public ir::IRMutator {
 public:
  explicit VarReplacer(
      const std::unordered_map<const Variable*, Expr>& vsub)
      : vsub_(vsub) {}
  Expr Mutate_(const Variable* op, const Expr& e) {
    auto it = vsub_.find(op);
    if (it != vsub_.end()) return it->second;
    return e;
  }

  ir::CommReducer MutateCommReducer(ir::CommReducer combiner) {
    // Replace free variables in combiner
    auto new_identity = ir::UpdateArray(combiner->identity_element, [this] (const Expr& e) {
      return this->Mutate(e);
      });
    auto new_result = ir::UpdateArray(combiner->result, [this] (const Expr& e) {
      return this->Mutate(e);
      });

    if (combiner->identity_element.same_as(new_identity) &&
        combiner->identity_element.same_as(new_result)) {
      return combiner;
    } else {
      return ir::CommReducerNode::make(
        combiner->lhs, combiner->rhs, new_result, new_identity);
    }
  }

  Expr Mutate_(const ir::Reduce* op, const Expr& e) {
    Expr new_e = IRMutator::Mutate_(op, e);
    const ir::Reduce* new_reduce = new_e.as<ir::Reduce>();
    ir::CommReducer new_combiner = MutateCommReducer(op->combiner);
    if (op->combiner.same_as(new_combiner)) {
      return new_e;
    } else {
      return ir::Reduce::make(
        new_combiner,
        new_reduce->source,
        new_reduce->axis,
        new_reduce->condition,
        new_reduce->value_index);
    }
  }

 private:
  const std::unordered_map<const Variable*, Expr>& vsub_;
};

Expr InjectPredicate(const Array<Expr>& predicates,
                     Expr body) {
  using ir::Reduce;
  using ir::Select;
  if (predicates.size() == 0) return body;
  const Reduce* reduce = body.as<Reduce>();
  if (reduce) {
    auto n = make_node<Reduce>(*reduce);
    n->condition = n->condition && arith::ComputeReduce<ir::And>(predicates, Expr());
    return Expr(n);
  }
  return Select::make(arith::ComputeReduce<ir::And>(predicates, Expr()),
                      body,
                      make_zero(body.type()));
}

// Replace data flow appears in all stages given the tensor change.
// Also update vmap if subsequent dataflow need to be replaced.
// Need to keep an update to the date transitive closure property on the vmap by a reverse map.
void ReplaceDataFlow(const Array<Stage>& stages,
                     std::unordered_map<Tensor, Tensor>* vmap,
                     std::unordered_map<Tensor, Tensor>* rvmap) {
  for (Stage s : stages) {
    Operation op = s->op->ReplaceInputs(s->op, *vmap);
    if (!op.same_as(s->op)) {
      for (int i = 0; i < op->num_outputs(); ++i) {
        auto it = rvmap->find(s->op.output(i));
        if (it != rvmap->end()) {
          (*vmap)[it->second] = op.output(i);
        } else {
          (*vmap)[s->op.output(i)] = op.output(i);
          (*rvmap)[op.output(i)] = s->op.output(i);
        }
      }
      s->op = op;
    }
  }
}

inline bool ReduceEqual(const ir::Reduce* a, const ir::Reduce* b) {
  return (a->combiner.same_as(b->combiner)) &&
         (a->source.same_as(b->source)) &&
         (a->axis.same_as(b->axis)) &&
         (a->condition.same_as(b->condition));
}

Tensor Schedule::cache_read(const Tensor& tensor,
                            const std::string& scope,
                            const Array<Operation>& readers) {
  (*this)->InvalidateCache();
  // create identity mapping.
  std::ostringstream os;
  os << tensor->op->name;
  if (tensor->op->num_outputs() != 1) {
    os << ".v" << tensor->value_index;
  }
  os << "." << scope;

  std::unordered_map<Tensor, Tensor> vsub;
  Stage s = operator[](tensor->op);
  Tensor sugar_tensor = s->op.output(tensor->value_index);
  Tensor cache = compute(sugar_tensor->shape, [&sugar_tensor](const Array<Var>& i) {
      return sugar_tensor(Array<Expr>(i.begin(), i.end()));
    }, os.str());
  vsub[sugar_tensor] = cache;

  std::unordered_map<Tensor, Tensor> vmap;
  std::unordered_map<Tensor, Tensor> rvmap;
  for (Operation op : readers) {
    Stage s = operator[](op);
    Operation repl_op = s->op->ReplaceInputs(s->op, vsub);
    CHECK(!repl_op.same_as(s->op))
        << "Cannot find " << tensor
        << " in the inputs of " << s->op;
    vmap[s->op.output(0)] = repl_op.output(0);
    rvmap[repl_op.output(0)] = s->op.output(0);
    s->op = repl_op;
  }
  ReplaceDataFlow((*this)->stages, &vmap, &rvmap);
  ArrayNode* stages = (*this)->stages.CopyOnWrite();
  Stage op_stage = operator[](tensor->op);
  size_t pos = FindNodeRef(stages, op_stage);
  Stage cache_stage = Stage(cache->op);
  cache_stage.set_scope(scope);
  CHECK_LT(pos, stages->data.size());
  stages->data.insert(stages->data.begin() + pos + 1,
                      cache_stage);
  (*this)->stage_map.Set(cache->op, cache_stage);
  // Update group
  cache_stage->group = op_stage->group;
  if (cache_stage->group.defined()) {
    ++cache_stage->group->num_child_stages;
  }
  return cache;
}

template<typename OpType>
void PrepareAxisMapping(Stage orig_stage,
                        OpType* op,
                        std::unordered_set<IterVar>* p_red_axis,
                        Array<IterVar>* p_new_axis,
                        std::unordered_map<IterVar, Range>* p_dom_map,
                        std::unordered_map<const Variable*, Expr>* p_vsub,
                        std::unordered_map<const Variable*, Expr>* p_vsub2newvar,
                        std::vector<Expr>* p_predicates) {
  auto& red_axis = *p_red_axis;
  auto& new_axis = *p_new_axis;
  auto& dom_map = *p_dom_map;
  auto& vsub = *p_vsub;
  auto& vsub2newvar = *p_vsub2newvar;
  auto& predicates = *p_predicates;
  arith::Analyzer analyzer;

  for (IterVar iv : op->reduce_axis) {
    red_axis.insert(iv);
  }
  for (IterVar iv : op->axis) {
    dom_map[iv] = iv->dom;
    analyzer.Bind(iv->var, iv->dom);
  }
  schedule::PassDownDomain(orig_stage, &dom_map, &analyzer, true);
  {
    // The source->cache
    std::unordered_map<IterVar, Expr> value_map;
    for (IterVar iv : orig_stage->leaf_iter_vars) {
      if (red_axis.count(iv)) continue;
      CHECK_EQ(iv->iter_type, kDataPar)
          << "Can only relayout with in data parallel dimensions";
      Range dom = dom_map.at(iv);
      IterVar new_iv = IterVarNode::make(
          dom, iv->var.copy_with_suffix(".c"), iv->iter_type);
      new_axis.push_back(new_iv);
      if (is_one(dom->min)) {
        value_map[iv] = dom->min;
      } else {
        value_map[iv] = iv->var;
        vsub2newvar[iv->var.get()] = new_iv->var;
      }
    }
    // skip reduction iteration.
    std::unordered_set<IterVar> skip_bound_check;
    for (IterVar iv : op->reduce_axis) {
      skip_bound_check.insert(iv);
    }
    schedule::PassUpIndex(orig_stage, dom_map, &value_map, true);
    predicates = schedule::MakeBoundCheck(
        orig_stage, dom_map, value_map, true, skip_bound_check);
    // The root axis
    for (IterVar iv : op->axis) {
      if (value_map.count(iv)) {
        vsub[iv->var.get()] = value_map.at(iv);
      }  // to handle tensor axis
    }
  }
}

Array<Tensor> ReplaceOriginalOp(Schedule sch,
                                Stage orig_stage,
                                const std::string& scope,
                                Operation cache_op,
                                Operation orig_new_op,
                                size_t tensor_size) {
  Array<Tensor> cache_tensor_list;
  for (size_t i = 0; i < tensor_size; i++) {
    Tensor cache_tensor = cache_op.output(i);
    cache_tensor_list.push_back(cache_tensor);
  }
  // The replace of the dataflow
  std::unordered_map<Tensor, Tensor> vmap;
  std::unordered_map<Tensor, Tensor> rvmap;
  vmap[orig_stage->op.output(0)] = orig_new_op.output(0);
  rvmap[orig_new_op.output(0)] = orig_stage->op.output(0);
  for (size_t i = 0; i < tensor_size; i++) {
    vmap[orig_stage->op.output(0)] = orig_new_op.output(0);
    rvmap[orig_new_op.output(0)] = orig_stage->op.output(0);
  }
  ReplaceDataFlow(sch->stages, &vmap, &rvmap);
  // mutate orig stage
  orig_stage->op = orig_new_op;
  orig_stage->all_iter_vars = orig_stage->op->root_iter_vars();
  orig_stage->leaf_iter_vars = orig_stage->all_iter_vars;
  orig_stage->relations = Array<IterVarRelation>();
  // create schedule for new cached stage.
  ArrayNode* stages = sch->stages.CopyOnWrite();
  size_t pos = FindNodeRef(stages, orig_stage);
  Stage cache_stage = Stage(cache_op);
  cache_stage.set_scope(scope);
  CHECK_LT(pos, stages->data.size());
  stages->data.insert(stages->data.begin() + pos,
                      cache_stage);
  sch->stage_map.Set(cache_op, cache_stage);
  // Update group
  cache_stage->group = orig_stage->group;
  if (cache_stage->group.defined()) {
    ++cache_stage->group->num_child_stages;
  }
  return cache_tensor_list;
}


// Cache write and relayout the data according to loop pattern
Array<Tensor> CacheWriteWithReLayout(Schedule sch,
                                     const Array<Tensor>& tensor_array,
                                     const std::string& scope) {
  size_t tensor_size = tensor_array.size();
  sch->InvalidateCache();
  Tensor tensor = tensor_array[0];
  Stage orig_stage = sch[tensor->op];
  const ComputeOpNode* compute = orig_stage->op.as<ComputeOpNode>();

  std::unordered_set<IterVar> red_axis;
  Array<IterVar> new_axis;
  std::unordered_map<IterVar, Range> dom_map;

  std::unordered_map<const Variable*, Expr> vsub;
  std::unordered_map<const Variable*, Expr> vsub2newvar;
  std::vector<Expr> predicates;

  PrepareAxisMapping(orig_stage, compute,
    &red_axis, &new_axis, &dom_map, &vsub, &vsub2newvar, &predicates);

  Expr body;
  Array<Expr> body_list;
  const ir::Reduce* first_reduce = nullptr;
  for (auto cbody : compute->body) {
    body = VarReplacer(vsub).Mutate(cbody);
    body = InjectPredicate(predicates, body);
    body = VarReplacer(vsub2newvar).Mutate(body);
    // Reduce nodes in ONE computeOp must be the same except value_index
    // This is right only if the original body ensures Reduce nodes are the same
    if (body->IsInstance<ir::Reduce>()) {
      const ir::Reduce* reduce_body = body.as<ir::Reduce>();
      if (first_reduce != nullptr) {
        CHECK(ReduceEqual(reduce_body, first_reduce));
        body = ir::Reduce::make(first_reduce->combiner,
                                first_reduce->source,
                                first_reduce->axis,
                                first_reduce->condition,
                                reduce_body->value_index);
      } else {
        first_reduce = reduce_body;
      }
    } else {
      CHECK(first_reduce == nullptr)
        << "cannot mix reduce and other node in ONE compute bodys";
    }
    body_list.push_back(body);
  }
  // The reader args
  Array<Expr> args;
  {
    // cache->compute
    std::unordered_map<IterVar, Expr> value_map;
    for (IterVar iv : compute->axis) {
      value_map[iv] = iv->var;
    }
    schedule::PassDownIndex(orig_stage, dom_map, &value_map, true);
    for (IterVar iv : orig_stage->leaf_iter_vars) {
      if (red_axis.count(iv)) continue;
      args.push_back(value_map.at(iv));
    }
  }
  Operation cache_op = ComputeOpNode::make(
      compute->name + "." + scope, compute->tag, compute->attrs,
      new_axis, body_list);

  Array<Expr> cache_expr_list;
  for (size_t i = 0; i < tensor_size; i++) {
    Tensor cache_tensor = cache_op.output(i);
    cache_expr_list.push_back(cache_tensor(args));
  }
  Operation orig_new_op = ComputeOpNode::make(
      compute->name, compute->tag, compute->attrs,
      compute->axis, cache_expr_list);
  return ReplaceOriginalOp(sch, orig_stage, scope,
    cache_op, orig_new_op, tensor_size);
}


// for tensor compute op
Array<Tensor> CacheWriteWithReLayoutTensor(Schedule sch,
                                           const Array<Tensor>& tensor_array,
                                           const std::string& scope) {
  size_t tensor_size = tensor_array.size();
  sch->InvalidateCache();
  Tensor tensor = tensor_array[0];
  Stage orig_stage = sch[tensor->op];
  const TensorComputeOpNode* tensor_op = orig_stage->op.as<TensorComputeOpNode>();
  CHECK_EQ(tensor_op->num_outputs(), 1)
      << "cache write only support single output tensor_compute_op";

  std::unordered_set<IterVar> red_axis;
  Array<IterVar> new_axis;
  std::unordered_map<IterVar, Range> dom_map;

  std::unordered_map<const Variable*, Expr> vsub;
  std::unordered_map<const Variable*, Expr> vsub2newvar;
  std::vector<Expr> predicates;

  PrepareAxisMapping(orig_stage, tensor_op,
    &red_axis, &new_axis, &dom_map, &vsub, &vsub2newvar, &predicates);


  for (int i = tensor_op->schedulable_ndim; i < static_cast<int>(tensor_op->axis.size()); ++i) {
    IterVar iv = tensor_op->axis[i];
    IterVar new_iv = IterVarNode::make(
      iv->dom, iv->var.copy_with_suffix(".c"), iv->iter_type);
    new_axis.push_back(new_iv);
  }
  Array<Region> new_regions;
  for (Region old_region : tensor_op->input_regions) {
    Region region;
    for (Range r : old_region) {
      Expr min = VarReplacer(vsub2newvar).Mutate(r->min);
      Expr extent = VarReplacer(vsub2newvar).Mutate(r->extent);
      region.push_back(Range::make_by_min_extent(min, extent));
    }
    new_regions.push_back(region);
  }

  Array<Expr> new_scalar_inputs;
  for (Expr old_input : tensor_op->scalar_inputs) {
    new_scalar_inputs.push_back(VarReplacer(vsub2newvar).Mutate(old_input));
  }

  Operation cache_op = TensorComputeOpNode::make(
      tensor_op->name + "." + scope, tensor_op->tag, new_axis,
      tensor_op->reduce_axis, tensor_op->schedulable_ndim,
      tensor_op->intrin, tensor_op->inputs, new_regions, new_scalar_inputs);

  // axis will be used in generating compute op
  Array<IterVar> compute_axis = tensor_op->axis;
  for (size_t i = tensor_op->schedulable_ndim; i < tensor_op->axis.size(); ++i) {
    IterVar iv = tensor_op->axis[i];
    IterVar aiv = IterVarNode::make(iv->dom, iv->var, kDataPar);
    compute_axis.Set(i, aiv);
  }

  // The reader args
  Array<Expr> args;
  {
    // cache->compute
    std::unordered_map<IterVar, Expr> value_map;
    for (IterVar iv : compute_axis) {
      value_map[iv] = iv->var;
    }
    schedule::PassDownIndex(orig_stage, dom_map, &value_map, true);
    for (IterVar iv : orig_stage->leaf_iter_vars) {
      if (red_axis.count(iv)) continue;
      args.push_back(value_map.at(iv));
    }
    // tensorized region axis
    for (size_t i = tensor_op->schedulable_ndim; i < tensor_op->axis.size(); ++i) {
      IterVar iv = compute_axis[i];
      args.push_back(value_map.at(iv));
    }
  }

  Array<Expr> cache_expr_list;
  for (size_t i = 0; i < tensor_size; i++) {
    Tensor cache_tensor = cache_op.output(i);
    cache_expr_list.push_back(cache_tensor(args));
  }
  Operation orig_new_op = ComputeOpNode::make(
      tensor_op->name, tensor_op->tag, {},
      compute_axis, cache_expr_list);
  return ReplaceOriginalOp(sch, orig_stage, scope,
    cache_op, orig_new_op, tensor_size);
}


Array<Tensor> Schedule::cache_write(const Array<Tensor>& tensor_array,
                             const std::string& scope) {
  (*this)->InvalidateCache();
  CHECK(tensor_array.size() > 0)
      << "size of tensor_array must be greater than 0";
  Tensor tensor = tensor_array[0];
  Stage orig_stage = operator[](tensor->op);
  const ComputeOpNode* compute = tensor->op.as<ComputeOpNode>();
  CHECK(static_cast<size_t>(compute->num_outputs()) == tensor_array.size())
      << "size of input tensor list must be same as number of stage outputs";
  for (size_t i = 1; i < tensor_array.size(); i++) {
    Stage tmp_stage = operator[](tensor_array[i]->op);
    CHECK(orig_stage.same_as(tmp_stage))
        << "Input tensor list must be generated by ONE computeOp";
  }
  return CacheWriteWithReLayout(*this, tensor_array, scope);
}


Tensor Schedule::cache_write(const Tensor& tensor,
                             const std::string& scope) {
  // support original compute and tensor compute both
  (*this)->InvalidateCache();
  if (tensor->op.as<ComputeOpNode>()) {
    return (CacheWriteWithReLayout(*this, {tensor}, scope))[0];
  } else if (tensor->op.as<TensorComputeOpNode>()) {
    return (CacheWriteWithReLayoutTensor(*this, {tensor}, scope))[0];
  } else {
    LOG(FATAL) << "cache write only take ComputeOp or TensorComputeOp as writers";
    return Tensor();
  }
}


void RebaseNonZeroMinLoop(const Schedule& sch) {
  std::unordered_map<IterVar, IterVar> rebase_map;
  for (Stage s : sch->stages) {
    if (s->attach_type == kInlinedAlready) continue;

    auto root_iter_vars = s->op->root_iter_vars();
    ArrayNode* leaf_vars = s->leaf_iter_vars.CopyOnWrite();
    for (IterVar iv : root_iter_vars) {
      size_t idx = FindNodeRef(leaf_vars, iv);
      auto it  = s->iter_var_attrs.find(iv);
      // don;t need to rebase path that are binded.
      if (it != s->iter_var_attrs.end() &&
          (*it).second->bind_thread.defined()) {
        continue;
      }
      if (idx < leaf_vars->data.size()) {
        // insert rebase
        IterVar rebased = IterVarNode::make(
            Range(), iv->var.copy_with_suffix(""), iv->iter_type);
        s->relations.push_back(RebaseNode::make(iv, rebased));
        if (s->iter_var_attrs.count(iv)) {
          s->iter_var_attrs.Set(rebased, s->iter_var_attrs.at(iv));
        }
        leaf_vars->data[idx] = rebased;
        rebase_map[iv] = rebased;
      }
    }
  }
  // remap the parent relation
  for (Stage s : sch->stages) {
    if (s->attach_type != kScope) continue;
    if (rebase_map.count(s->attach_ivar)) {
      s->attach_ivar = rebase_map.at(s->attach_ivar);
    }
  }
  for (Stage s : sch->groups) {
    if (s->attach_type != kScope) continue;
    if (rebase_map.count(s->attach_ivar)) {
      s->attach_ivar = rebase_map.at(s->attach_ivar);
    }
  }
}

class BufferAccess2Tensor : public IRMutator {
 public:
  BufferAccess2Tensor(Array<Tensor> inputs, Tensor output, Array<Buffer> input_placeholders,
                      Buffer output_placeholder) {
    CHECK_EQ(inputs.size(), input_placeholders.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      tensors_.emplace(input_placeholders[i]->data.get(), inputs[i]);
    }
    tensors_.emplace(output_placeholder->data.get(), output);
  };

 private:
  Stmt Mutate_(const AttrStmt* op, const Stmt& s) override {
    if (op->attr_key == "buffer_bind_scope") {
      Array<NodeRef> bind_spec = Downcast<Array<NodeRef>>(op->node);
      Buffer buffer = Downcast<Buffer>(bind_spec[0]);
      Tensor tensor = Downcast<Tensor>(bind_spec[1]);
      tensors_.emplace(buffer->data.get(), tensor);
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Call* op, const Expr& e) override {
    Expr expr = IRMutator::Mutate_(op, e);
    op = expr.as<Call>();
    if (op != nullptr && op->name == "tensor_load") {
      auto it = tensors_.find(op->args[0].as<Variable>());
      CHECK(it != tensors_.end());
      Tensor t = it->second;
      Array<Expr> args;
      for (size_t i = 1; i < op->args.size(); ++i) {
        args.push_back(op->args[i]);
      }
      return Call::make(t->dtype, t->op->name, args, Call::CallType::Halide, t->op, t->value_index);
    }
    return expr;
  }

  Stmt Mutate_(const Evaluate* op, const Stmt& s) override {
    const Call* call = op->value.as<Call>();
    if (call != nullptr && call->name == "tensor_store") {
      Expr expr = IRMutator::Mutate(op->value);
      call = expr.as<Call>();
      auto it = tensors_.find(call->args[0].as<Variable>());
      CHECK(it != tensors_.end());
      Expr value = call->args[1];
      Array<Expr> args;
      for (size_t i = 2; i < call->args.size(); ++i) {
        args.push_back(call->args[i]);
      }
      return Provide::make(it->second->op, 0, value, args);
    }
    return IRMutator::Mutate_(op, s);
  }

  std::unordered_map<const Variable*, Tensor> tensors_;
};

class ProvideBody : public IRVisitor {
 public:
  explicit ProvideBody(FunctionRef output_tensors, bool csr_op = false) :
    output_tensors_(std::move(output_tensors)), csr_op_(csr_op) {}

  void Visit(const NodeRef& e) final {
    if (!can_inline_) return;
    IRVisitor::Visit(e);
  }

  void Visit_(const Block* op) final {
    can_inline_ = false;
    return;
  }

  void Visit_(const IfThenElse* op) final {
    if (csr_op_) {
      return IRVisitor::Visit_(op);
    }
    can_inline_ = false;
    return;
  }

  void Visit_(const LetStmt* op) final {
    can_inline_ = false;
    return;
  }

  void Visit_(const For* op) final {
    iter_var_set_.emplace(op->loop_var);
    IRVisitor::Visit(op->body);
  }

  void Visit_(const Load* op) final {
    can_inline_ = false;
    return;
  }

  void Visit_(const Store* op) final {
    can_inline_ = false;
    return;
  }

  void Visit_(const Let* op) final {
    can_inline_ = false;
    return;
  }

  void Visit_(const Provide* op) override {
    if (multi_provide_ || op->func != output_tensors_) {
      can_inline_ = false;
      return;
    }

    if (op->value.type().is_int() || op->value.type().is_uint()) {
      // it is better not to inline computation for index
      // so here we prevent to inline any int type body expr
      can_inline_ = false;
      return;
    }

    Array<Expr> op_args;
    if (csr_op_) {
      op_args = VarsFromArgs(op->args);
    } else {
      op_args = op->args;
    }

    if (op_args.size() != iter_var_set_.size()) {
      can_inline_ = false;
      return;
    }

    for (auto arg : op_args) {
      if (iter_var_set_.count(arg) == 0) {
        can_inline_ = false;
        return;
      }
    }

    multi_provide_ = true;

    body_ = op->value;
    args_ = op->args;

    IRVisitor::Visit_(op);
  }

  void Visit_(const Call* op) override {
    if (op->func == output_tensors_) {
      can_inline_ = false;
      return;
    }

    IRVisitor::Visit_(op);
  }

  const Expr body() { return body_; }
  const Array<Expr> args() { return args_; }
  const bool can_inline() { return can_inline_; }

 private:
  FunctionRef output_tensors_;
  Expr body_;
  Array<Expr> args_;
  bool can_inline_{true};
  bool multi_provide_{false};
  std::set<Expr, ExprLess> iter_var_set_;
  bool csr_op_;
};

class LetVarReplace : public IRMutator {
 public:
  Stmt Mutate_(const LetStmt* op, const Stmt& s) final {
    if (!HasSideEffect(op->value)) {
      var_value_[op->var.get()] = Mutate(op->value);
      return this->Mutate(op->body);
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (op->attr_key == attr::loop_scope || op->attr_key == attr::scan_init_scope) {
      return this->Mutate(op->body);
    } else if (op->attr_key == attr::scan_update_scope) {
      const ScanOpNode* scan = op->node.as<ScanOpNode>();
      CHECK(scan);
      var_value_[scan->scan_axis->var.get()] = op->value;
      return this->Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Variable* op, const Expr& e) final {
    auto it = var_value_.find(op);
    if (it != var_value_.end()) {
      return it->second;
    } else {
      return e;
    }
  }

 private:
  // The scan value
  std::unordered_map<const Variable*, Expr> var_value_;
};

class IfThenElseReplace : public IRMutator {
 public:
  Stmt Mutate_(const IfThenElse* op, const Stmt& s) final {
    Stmt then_branch = IRMutator::Mutate(op->then_case);
    Stmt else_branch;
    if (op->else_case.defined()) {
      else_branch = IRMutator::Mutate(op->else_case);
      auto then_provide = then_branch.as<Provide>();
      auto else_provide = else_branch.as<Provide>();

      if (CheckProvide(then_provide, else_provide)) {
        Expr select_body = Select::make(op->condition, then_provide->value, else_provide->value);
        return Provide::make(then_provide->func, then_provide->value_index, select_body,
                             then_provide->args);
      }
    }

    return IfThenElse::make(op->condition, then_branch, else_branch);
  }

 private:
  bool CheckProvide(const Provide* then_provide, const Provide* else_provide) {
    if ((then_provide == nullptr) || (else_provide == nullptr)) return false;

    if (!then_provide->func.same_as(else_provide->func)) return false;

    if (then_provide->value_index != else_provide->value_index) return false;

    if (then_provide->args.size() != else_provide->args.size()) return false;

    for (size_t i = 0; i < then_provide->args.size(); i++) {
      if (!then_provide->args[i].same_as(else_provide->args[i])) return false;
    }

    return true;
  }
};


/*
This pass is to inline tmp var defined in "block_realize" scope inside a hybrid op.
The example:
realize cube<float32>([0, 4], [0, 4]) {
  produce cube {
    // attr [[buffer(buffer, 0x55c34266f3a0), Tensor(shape=[4, 4], op.name=input_0)]]
      buffer_bind_scope = tvm_tuple(0, 4, 0, 4):handle:I
    // attr [[buffer(buffer, 0x55c34266f400), Tensor(shape=[4, 4], op.name=cube)]] buffer_bind_scope
      = tvm_tuple(0, 4, 0, 4):handle:I
    // attr [0] extern_scope = 0
    // attr [placeholder(b, 0x55c34273aaf0)] realize_scope = "local"
    realize b<float32>([0, 4], [0, 4]) {
      // attr [placeholder(b, 0x55c34273aaf0)] block_realize = (bool)1
      for (i0, 0, 4) {
        for (i1, 0, 4) {
          b(i0, i1) = (input_0(i0, i1)*input_0(i0, i1))
        }
      }
      for (i0, 0, 4) {
        for (i1, 0, 4) {
          cube(i0, i1) = (b(i0, i1)*input_0(i0, i1))
        }
      }
    }
  }
}
.....................
============>
realize cube<float32>([0, 4], [0, 4]) {
  produce cube {
    // attr [[buffer(buffer, 0x55f7634071f0), Tensor(shape=[4, 4], op.name=input_0)]]
      buffer_bind_scope = tvm_tuple(0, 4, 0, 4):handle:I
    // attr [[buffer(buffer, 0x55f7633b2940), Tensor(shape=[4, 4], op.name=cube)]] buffer_bind_scope
      = tvm_tuple(0, 4, 0, 4):handle:I
    // attr [0] extern_scope = 0
    for (i0, 0, 4) {
      for (i1, 0, 4) {
        cube(i0, i1) = ((input_0(i0, i1)*input_0(i0, i1))*input_0(i0, i1))
      }
    }
  }
}
*/
class TmpVarInline : public IRMutator {
 public:
  Stmt Mutate_(const Block* op, const Stmt& s) final {
    Stmt first = Mutate(op->first);
    Stmt rest = Mutate(op->rest);
    if (const auto attr = first.as<AttrStmt>()) {
      if (attr->attr_key == "block_realize") {
        auto func = Downcast<FunctionRef>(attr->node);
        auto find_body = ProvideBody(func);
        find_body.Visit(attr->body);

        if (find_body.can_inline()) {
          bool is_lvalue = false;

          PostOrderVisit(rest, [&is_lvalue, &func](const NodeRef& n) {
            if (auto provide = n.as<Provide>()) {
              if (provide->func == func) {
                is_lvalue = true;
              }
            }
          });

          if (!is_lvalue) {
            Array<Var> args;
            auto body = find_body.body();
            for (auto arg : find_body.args()) {
              args.push_back(Downcast<Var>(arg));
            }
            inlined_[func] = true;
            return ir::Inline(rest, func, args, body);
          }
        }
        inlined_[func] = false;
      }
    }
    return Block::make(first, rest);
  }

  Stmt Mutate_(const AttrStmt* op, const Stmt& s) {
    Stmt body = this->Mutate(op->body);
    FunctionRef func = Downcast<FunctionRef>(op->node);
    if (op->attr_key == "realize_scope"){
      if (auto attr_value = op->value.as<StringImm>()) {
        if (attr_value->value == "local" && inlined_.count(func) > 0 && inlined_[func]) {
          return body;
        }
      }
    }
    return AttrStmt::make(op->node, op->attr_key, op->value, body);
  }

  Stmt Mutate_(const Realize* op, const Stmt& s) {
    FunctionRef func = op->func;
    Stmt body = Mutate(op->body);
    if (inlined_.count(func) > 0 && inlined_[func]) {
      return body;
    } else {
      return Realize::make(op->func, op->value_index, op->type, op->bounds, op->condition, body);
    }
  }

 private:
  std::unordered_map<FunctionRef, bool, NodeHash, NodeEqual> inlined_;
};

void UpdateInlineInputs(std::set<Tensor>& new_inputs, const Array<Tensor>& inputs,
                        const std::unordered_map<Tensor, Operation>& inline_tensor_inputs) {
  for (auto input : inputs) {
    if (inline_tensor_inputs.count(input)) {
      UpdateInlineInputs(new_inputs, inline_tensor_inputs.at(input)->InputTensors(),
                         inline_tensor_inputs);
    } else {
      new_inputs.emplace(input);
    }
  }
}

class ResolveConditional : public IRMutator {
  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    Stmt stmt = s;
    auto cond = Simplify(op->condition);
    if (auto uint_cond = cond.as<UIntImm>()) {
      if (uint_cond->value) {
        return IRMutator::Mutate(op->then_case);
      }
      if (op->else_case.defined()) {
        return IRMutator::Mutate(op->else_case);
      }
    }
    return IRMutator::Mutate_(op, s);
  }
};

void NormalizeBody(ScheduleNode* sch) {
  // in this function, we will normalize the body of hybrid and extern ops
  // 1. eliminate let stmt
  // 2. rewrite some IfThenElse stmt as Select stmt
  // 3. for extern op, rewrite buffer access in tensor load/store as Tensor call
  std::unordered_map<Tensor, Tensor> repl;
  for (size_t i = 0; i < sch->stages.size(); ++i) {
    Stage s = sch->stages[i];
    Operation op = s->op;

    // Will skip all PlaceholderOpNode as they have neither body nor inputs
    if (s->op->IsInstance<PlaceholderOpNode>()) continue;

    const ExternOpNode* extern_op = s->op.as<ExternOpNode>();
    const HybridOpNode* hybird_op = s->op.as<HybridOpNode>();

    // Deal with the body of extern op and hybrid op respectively
    // create new op if the body is changed
    if (extern_op) {
      Stmt op_body = extern_op->body;
      auto buffer_mutator =
          BufferAccess2Tensor(extern_op->inputs, s->op.output(0), extern_op->input_placeholders,
                              extern_op->output_placeholders[0]);
      op_body = buffer_mutator.Mutate(op_body);
      op_body = LetVarReplace().Mutate(op_body);
      op_body = IfThenElseReplace().Mutate(op_body);
      op_body = ResolveConditional().Mutate(op_body);

      if (!extern_op->body.same_as(op_body)) {
        op = ExternOpNode::make(extern_op->name, extern_op->tag, extern_op->attrs,
                                extern_op->inputs, extern_op->input_placeholders,
                                extern_op->output_placeholders, op_body);
      }
    } else if (hybird_op) {
      Stmt op_body = hybird_op->body;
      op_body = LetVarReplace().Mutate(op_body);
      op_body = IfThenElseReplace().Mutate(op_body);
      op_body = TmpVarInline().Mutate(op_body);
      if (!hybird_op->body.same_as(op_body)) {
        op = HybridOpNode::make(hybird_op->name, hybird_op->tag, hybird_op->attrs,
                                hybird_op->inputs, hybird_op->outputs, hybird_op->input_buffers_,
                                hybird_op->output_buffers_, hybird_op->input_regions_,
                                hybird_op->output_regions_, op_body);
      }
    }

    op = op->ReplaceInputs(op, repl);
    if (!op.same_as(s->op)) {
      // update the stage op if new op is created
      // old op is stored in stage->origin_op
      for (int idx = 0; idx < s->op->num_outputs(); ++idx) {
        repl[s->op.output(idx)] = op.output(idx);
      }
      s->op = op;
      // update stage axis
      s->all_iter_vars = op->root_iter_vars();
      Array<IterVar> clean;
      for (IterVar iv : s->all_iter_vars) {
        if (iv->iter_type != kOpaque) clean.push_back(iv);
      }
      s->leaf_iter_vars = clean;
    }
  }
}

Array<Expr> GetFuncArgs(const Stmt &s, const FunctionRef &f) {
  Array<Expr> args;
  PostOrderVisit(s, [&f, &args] (const NodeRef &e) {
    if (auto pro = e.as<Provide>()) {
      if (pro->func.defined() && pro->func->func_name() == f->func_name()) {
        args = pro->args;
      }
    }
  });
  return args;
}

class MapCsrArgs : public IRVisitor {
 public:
  explicit MapCsrArgs(Array<Expr> csr_args, const FunctionRef &f) : csr_args_(csr_args), f_(f) {}

  void Visit_(const Call *op) final {
    if (op->func.same_as(f_)) {
      CHECK(op->args.size() == csr_args_.size());
      vmap_.Set(Downcast<Var>(op->args[0]), csr_args_[0]);
    } else {
      IRVisitor::Visit_(op);
    }
  }

  Array<Expr> csr_args_;
  const FunctionRef &f_;
  Map<Var, Expr> vmap_;
};

class ReplaceCsrSchedule : public IRMutator {
 public:
  explicit ReplaceCsrSchedule(Stmt &output_provide) : output_provide_(output_provide) {}

 private:
  Stmt Mutate_(const For *op, const Stmt &s) final {
    auto extent_sub = op->extent.as<Sub>();
    if (extent_sub != nullptr && extent_sub->a.as<Call>() != nullptr && extent_sub->b.as<Call>() != nullptr) {
      csr_idx_ = Add::make(extent_sub->b, op->loop_var);
    } else if (csr_idx_.defined()) {
      feature_var_.push_back(op->loop_var);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    size_t var_idx = 0;
    auto output_op = output_provide_.as<Provide>();
    CHECK(output_op != nullptr);
    for (auto e: output_op->args) {
      Array<Var> vars;
      PostOrderVisit(e, [&vars](const NodeRef &n) {
        if (n.as<Variable>() != nullptr) {
          vars.push_back(Downcast<Var>(n));
        }
      });
      if (vars.empty()) continue;
      CHECK(vars.size() == 1);
      if (var_idx == 0) {
        CHECK(csr_idx_.defined());
        vmap_.Set(vars[0], csr_idx_);
      } else {
        CHECK(var_idx - 1 < feature_var_.size());
        vmap_.Set(vars[0], feature_var_[var_idx - 1]);
      }
      ++var_idx;
    }
    return Substitute(output_provide_, vmap_);
  }

  Stmt output_provide_;
  Expr csr_idx_;
  Map<Var, Expr> vmap_;
  Array<Var> feature_var_;
};

class CollectAxes : public IRVisitor {
  void Visit_(const For *op) final {
    auto range = Range::make_by_min_extent(op->min, op->extent);
    auto axis = IterVarNode::make(range, op->loop_var, kDataPar);
    axes_.push_back(axis);
    IRVisitor::Visit_(op);
  }

 public:
  Array<IterVar> axes_;
};

void InlineReduce(size_t j, const Expr &new_value, std::vector<Array<Expr>> &new_body, std::vector<bool> &changed) {
  if (!new_value.same_as(new_body[j][0])) {
    changed[j] = true;
    const ir::Reduce* r = new_value.as<ir::Reduce>();
    CHECK(r != nullptr);
    CHECK_EQ(new_body[j].size(), r->source.size());
    for (size_t k = 0; k < new_body[j].size(); ++k) {
      auto n = make_node<ir::Reduce>(*r);
      n->value_index = static_cast<int>(k);
      n->type = r->source[k].type();
      new_body[j].Set(k, Expr(n));
    }
  }
}

void UpdateReduceInlineStage(const Reduce *reduce, const ComputeOpNode *compute, const Array<IterVar> &inline_axes,
                             const Array<Expr> &new_body, Stage &s) {
  Array<IterVar> injective_axes;
  for (auto axis: inline_axes) {
    bool is_injective = true;
    for (auto reduce_axis: reduce->axis) {
      if (axis.same_as(reduce_axis)) {
        is_injective = false;
        break;
      }
    }
    if (is_injective) {
      injective_axes.push_back(axis);
    }
  }
  s->op = ComputeOpNode::make(compute->name, compute->tag, compute->attrs, injective_axes,
                              new_body);
  s->all_iter_vars = inline_axes;
  s->leaf_iter_vars = inline_axes;
}

bool InlineCsrWithCompute(size_t j, const Array<Expr> &csr_args, const FunctionRef &func,
                          std::vector<Array<Expr>> &new_body) {
  bool map_csr = false;
  Array<Expr> new_csr_body;
  for (size_t k = 0; k < new_body[j].size(); ++k) {
    auto body = new_body[j][k];
    auto map_csr_args = MapCsrArgs(csr_args, func);
    map_csr_args.Visit(body);
    auto compute_body = Substitute(body, map_csr_args.vmap_);
    if (!body.same_as(compute_body)) {
      map_csr = true;
      new_body[j].Set(k, compute_body);
    }
  }
  return map_csr;
}

void UpdateComputeInlineStage(const ComputeOpNode *compute, const Array<IterVar> &inline_axes,
                              const Array<Expr> &new_body, Stage &s) {
  constexpr int csr_arg_size = 2;
  Array<IterVar> injective_axes;
  Array<IterVar> new_axes;
  CHECK(inline_axes.size() >= compute->root_iter_vars().size());
  for (size_t i = 0; i < inline_axes.size(); ++i) {
    bool is_injective = true;
    IterVar axis;
    if (i < csr_arg_size) {
      axis = inline_axes[i];
    } else {
      axis = compute->root_iter_vars()[i - (inline_axes.size() - compute->root_iter_vars().size())];
    }
    new_axes.push_back(axis);
    for (auto reduce_axis: compute->reduce_axis) {
      if (reduce_axis.same_as(axis)) {
        is_injective = false;
        break;
      }
    }
    if (is_injective) {
      injective_axes.push_back(axis);
    }
  }
  s->op = ComputeOpNode::make(compute->name, compute->tag, compute->attrs, injective_axes,
                              new_body);
  s->all_iter_vars = new_axes;
  s->leaf_iter_vars = new_axes;
}

void InjectInline(ScheduleNode* sch) {
  sch->InvalidateCache();

  std::vector<Array<Expr>> new_body(sch->stages.size());
  std::vector<bool> changed(sch->stages.size(), false);
  std::vector<Stmt> new_hybrid_body(sch->stages.size());
  std::vector<bool> hybrid_changed(sch->stages.size(), false);
  std::vector<Stmt> new_extern_body(sch->stages.size());
  std::vector<bool> extern_changed(sch->stages.size(), false);
  std::unordered_map<Tensor, Operation> inline_tensor_inputs;
  std::unordered_set<size_t> map_csr_stage;
  // inline all the ops
  for (size_t i = sch->stages.size(); i != 0; --i) {
    Stage stage = sch->stages[i - 1];
    if (auto extern_op = stage->op.as<ExternOpNode>()) {
      if (extern_op->attrs.count("csr_op")) {
        map_csr_stage.insert(i - 1);
      }
    }

    if (stage->is_output || stage->no_inline_inject) continue;

    Array<Var> args;
    Expr body;
    Array<Expr> csr_args;
    Stmt csr_body;
    Array<IterVar> inline_axes;

    if (stage->attach_type == kInline) {

      stage->attach_type = kInlinedAlready;
      {
        // setup args
        const ComputeOpNode* compute = stage->op.as<ComputeOpNode>();
        CHECK(compute)
            << "can only inline compute op";
        for (auto iv : compute->axis) {
          args.push_back(iv->var);
        }
        inline_axes = compute->axis;
        CHECK_EQ(compute->body.size(), 1U)
            << "can only inline compute op with 1 output";
        body = compute->body[0];
        inline_tensor_inputs.emplace(stage->op.output(0), stage->op);
      }
    } else {
      // can only inline compute op with 1 output
      if (stage->op->num_outputs() > 1) continue;

      const ExternOpNode* extern_op = stage->op.as<ExternOpNode>();
      const HybridOpNode* hybird_op = stage->op.as<HybridOpNode>();

      Stmt op_body;
      Operation output_op;

      if (extern_op) {
        if (extern_op->attrs.count("disable_inline"))
          continue;

        op_body = extern_op->body;
        output_op = stage->origin_op;
      } else if (hybird_op) {
        if (hybird_op->attrs.count("disable_inline"))
          continue;

        op_body = hybird_op->body;
        output_op = hybird_op->outputs[0]->op;
      } else {
        continue;
      }

      auto find_body = ProvideBody(output_op, map_csr_stage.count(i - 1) > 0);
      find_body.Visit(op_body);

      if (!find_body.can_inline()) continue;

      stage->attach_type = kInlinedAlready;
      body = find_body.body();

      Array<Expr> tmp_args;
      if (extern_op != nullptr && extern_op->attrs.count("csr_op")) {
        csr_args = find_body.args();
        tmp_args = VarsFromArgs(csr_args);
        csr_body = op_body;
        auto collect_axes = CollectAxes();
        collect_axes.Visit(csr_body);
        inline_axes = collect_axes.axes_;
      } else {
        tmp_args = find_body.args();
      }
      for (auto arg : tmp_args) {
        args.push_back(Downcast<Var>(arg));
      }
      inline_tensor_inputs.emplace(stage->op.output(0), stage->op);
    }

    if (body.defined()) {
      for (size_t j = i; j < sch->stages.size(); ++j) {
        Stage s = sch->stages[j];
        const ComputeOpNode* compute = s->op.as<ComputeOpNode>();
        const HybridOpNode* hybrid = s->op.as<HybridOpNode>();
        const ExternOpNode* extern_op = s->op.as<ExternOpNode>();
        if (compute) {
          if (!new_body[j].size()) {
            new_body[j] = compute->body;
          }
          if (body->IsInstance<ir::Reduce>()) {
            auto reduce = body.as<ir::Reduce>();
            Array<Expr> new_source;
            for (size_t k = 0; k < reduce->source.size(); ++k) {
              auto new_source_value = ir::Inline(ir::Evaluate::make(new_body[j][k]),
                                                 stage->op, args, reduce->source[k]).as<ir::Evaluate>()->value;
              new_source.push_back(new_source_value);
            }
            auto new_value = Reduce::make(
              reduce->combiner, new_source, reduce->axis, reduce->condition, reduce->value_index);
            InlineReduce(j, new_value, new_body, changed);
            UpdateReduceInlineStage(reduce, compute, inline_axes, new_body[j], s);
          }
          if (csr_body.defined() && !map_csr_stage.count(j) && InlineCsrWithCompute(j, csr_args, stage->op, new_body)) {
            map_csr_stage.insert(j);
            Array<Array<Expr>> csr_output_shape;
            for (size_t k = 0; k < static_cast<size_t>(s->op->num_outputs()); ++k) {
              csr_output_shape.push_back(s->op->output_shape(k));
            }
            s->csr_output_shape = csr_output_shape;
            CHECK(!csr_args.empty());
            s->csr_access = csr_args[0];
            UpdateComputeInlineStage(compute, inline_axes, new_body[j], s);
          }
          if (new_body[j][0]->IsInstance<ir::Reduce>()) {
            // specially handle reduction inline for multiplre reductions.
            const ir::Reduce* reduce = new_body[j][0].as<ir::Reduce>();
            for (size_t k = 1; k < new_body[j].size(); ++k) {
              const ir::Reduce* reduce_ = new_body[j][k].as<ir::Reduce>();
              CHECK(reduce_);
              CHECK(ReduceEqual(reduce_, reduce))
                  << "The Reduce inputs of ComputeOp should "
                  << "have the same attribute except value_index";
            }
            Expr new_value = ir::Inline(ir::Evaluate::make(new_body[j][0]),
                                        stage->op, args, body, csr_body.defined()).as<ir::Evaluate>()->value;
            if (!new_value.same_as(new_body[j][0])) {
              changed[j] = true;
              const ir::Reduce* r = new_value.as<ir::Reduce>();
              CHECK(r != nullptr);
              CHECK_EQ(new_body[j].size(), r->source.size());
              for (size_t k = 0; k < new_body[j].size(); ++k) {
                auto n = make_node<ir::Reduce>(*r);
                n->value_index = static_cast<int>(k);
                n->type = r->source[k].type();
                new_body[j].Set(k, Expr(n));
              }
            }
          } else {
            for (size_t k = 0; k < new_body[j].size(); ++k) {
              Expr new_value = ir::Inline(ir::Evaluate::make(new_body[j][k]),
                                          stage->op, args, body, csr_body.defined()).as<ir::Evaluate>()->value;
              if (!new_value.same_as(new_body[j][k])) {
                new_body[j].Set(k, new_value);
                changed[j] = true;
              }
            }
          }
        } else if (hybrid) {
          if (!new_hybrid_body[j].defined()) {
            new_hybrid_body[j] = hybrid->body;
          }
          Stmt new_stmt = ir::Inline(new_hybrid_body[j], stage->op, args, body, csr_body.defined());
          if (!new_stmt.same_as(new_hybrid_body[j])) {
            new_hybrid_body[j] = new_stmt;
            hybrid_changed[j] = true;
          }
        } else if (extern_op) {
          if (!new_extern_body[j].defined()) {
            new_extern_body[j] = extern_op->body;
          }

          if (csr_body.defined() && !map_csr_stage.count(j)) {
            Stmt provide;
            PostOrderVisit(new_extern_body[j], [&provide](const NodeRef &n) {
              if (n.as<Provide>() != nullptr) {
                provide = Downcast<Stmt>(n);
              }
            });
            auto replace_csr_schedule = ReplaceCsrSchedule(provide);
            auto output_body = replace_csr_schedule.Mutate(csr_body);
            new_extern_body[j] = output_body;
            map_csr_stage.insert(j);
          }
          Stmt new_stmt = ir::Inline(new_extern_body[j], stage->op, args, body, csr_body.defined());
          if (!new_stmt.same_as(new_extern_body[j])) {
            new_extern_body[j] = new_stmt;
            extern_changed[j] = true;
          }
        }
      }
    }
  }
  std::unordered_map<Tensor, Tensor> repl;
  // rewrite dataflow
  for (size_t i = 0; i < sch->stages.size(); ++i) {
    Stage s = sch->stages[i];
    if (s->attach_type == kInlinedAlready) continue;
    if (new_body[i].size()) {
      // Logics from ReplaceDataFlow
      const ComputeOpNode* compute = sch->stages[i]->op.as<ComputeOpNode>();
      Operation op = s->op;
      if (changed[i]) {
        op = ComputeOpNode::make(compute->name, compute->tag, compute->attrs, compute->axis,
                                 new_body[i]);
      }
      op = op->ReplaceInputs(op, repl);
      if (!op.same_as(s->op)) {
        for (int idx = 0; idx < s->op->num_outputs(); ++idx) {
          repl[s->op.output(idx)] = op.output(idx);
        }
        s->op = op;
      }
    } else if (hybrid_changed[i]) {
      const HybridOpNode* hybrid = sch->stages[i]->op.as<HybridOpNode>();
      Array<Tensor> new_inputs;
      std::set<Tensor> new_inputs_set;
      // we use a set here to avoid duplicated inputs
      UpdateInlineInputs(new_inputs_set, hybrid->inputs, inline_tensor_inputs);
      for (auto new_input : new_inputs_set) {
        new_inputs.push_back(new_input);
      }

      Operation op =
          HybridOpNode::make(hybrid->name, hybrid->tag, hybrid->attrs, new_inputs, hybrid->outputs,
                             hybrid->input_buffers_, hybrid->output_buffers_,
                             hybrid->input_regions_, hybrid->output_regions_, new_hybrid_body[i]);
      op = op->ReplaceInputs(op, repl);
      for (int idx = 0; idx < s->op->num_outputs(); ++idx) {
        repl[s->op.output(idx)] = op.output(idx);
      }
      s->op = op;
      // update stage axis
      s->all_iter_vars = op->root_iter_vars();
      Array<IterVar> clean;
      for (IterVar iv : s->all_iter_vars) {
        if (iv->iter_type != kOpaque) clean.push_back(iv);
      }
      s->leaf_iter_vars = clean;
    } else if (extern_changed[i]) {
      const ExternOpNode* extern_op = sch->stages[i]->op.as<ExternOpNode>();

      Array<Tensor> new_inputs;
      std::set<Tensor> new_inputs_set;
      Map<Tensor, Buffer> tensor_binds;
      for (size_t i = 0; i < extern_op->inputs.size(); i++){
        tensor_binds.Set(extern_op->inputs[i], extern_op->input_placeholders[i]);
      }

      // we use a set here to avoid duplicated inputs
      UpdateInlineInputs(new_inputs_set, extern_op->inputs, inline_tensor_inputs);
      for (auto new_input : new_inputs_set) {
        new_inputs.push_back(new_input);
      }

      Array<Buffer> new_input_placeholders;
      for (auto input : new_inputs) {
        if (tensor_binds.count(input)){
          new_input_placeholders.push_back(tensor_binds.at(input));
        } else {
          // after inline, some new inputs might be introduced to the body
          // An extern op needs a buffer for each inptut
          // here the input introduced by inlined is in a form a tensor, not a buffer
          // so we create a fake buffer for the input only for the legality of an extern op
          std::string name = input->op->name;
          Array<Expr> shape = input->shape;
          Type dtype = input->dtype;
          Array<Expr> strides;

          auto data = Variable::make(Handle(), name);
          auto fake_buffer = BufferNode::make(data, dtype, shape, strides, Expr(), name, "", 0, 0,
                                              BufferType::kDefault);
          new_input_placeholders.push_back(fake_buffer);
        }
      }
      Operation op = ExternOpNode::make(extern_op->name, extern_op->tag, extern_op->attrs,
                                        new_inputs, new_input_placeholders, extern_op->output_placeholders, new_extern_body[i]);
      op = op->ReplaceInputs(op, repl);
      for (int idx = 0; idx < s->op->num_outputs(); ++idx) {
        repl[s->op.output(idx)] = op.output(idx);
      }
      s->op = op;
      // update stage axis
      s->all_iter_vars = op->root_iter_vars();
      Array<IterVar> clean;
      for (IterVar iv : s->all_iter_vars) {
        if (iv->iter_type != kOpaque) clean.push_back(iv);
      }
      s->leaf_iter_vars = clean;
    } else {
      Operation op = s->op->ReplaceInputs(s->op, repl);
      if (!op.same_as(s->op)) {
        for (int j = 0; j < op->num_outputs(); ++j) {
          repl[s->op.output(j)] = op.output(j);
        }
        s->op = op;
      }
    }
  }
}

Schedule Schedule::normalize() {
  Schedule sn = copy();
  NormalizeBody(sn.operator->());
  InjectInline(sn.operator->());
  RebaseNonZeroMinLoop(sn);
  return sn;
}

// Handle reduction factor.
Array<Tensor> Schedule::rfactor(const Tensor& tensor,
                                const IterVar& axis,
                                int factor_axis) {
  (*this)->InvalidateCache();
  using ir::Reduce;
  CHECK_EQ(axis->iter_type, kCommReduce)
      << "Can only factor reduction axis";
  Stage reduce_stage = operator[](tensor->op);
  const ComputeOpNode* compute_op = reduce_stage->op.as<ComputeOpNode>();
  CHECK(compute_op) << "Can only factor ComputeOp";
  ArrayNode* leaf_vars = reduce_stage->leaf_iter_vars.CopyOnWrite();
  {
    size_t axis_pos = FindNodeRef(leaf_vars, axis);
    CHECK_NE(axis_pos, leaf_vars->data.size())
        << "Cannot find IterVar " << axis << " in leaf iter vars";
  }
  // Find touched reduction axis.
  std::unordered_map<IterVar, int> touch_map;
  touch_map[axis] = 1;
  schedule::PassUpBitMaskOr(reduce_stage, &touch_map, true);
  schedule::PassDownBitMaskOr(reduce_stage, &touch_map, true);
  // skip reduction iteration.
  std::unordered_set<IterVar> skip_bound_check;
  // Verify normal axis are not touched.
  for (IterVar iv : compute_op->axis) {
    CHECK(!touch_map.count(iv))
        << "Factor axis touches normal axis.";
    skip_bound_check.insert(iv);
  }
  // get analyzer.
  arith::Analyzer analyzer;
  // Get the replace index
  std::unordered_map<IterVar, Range> dom_map;
  std::unordered_map<IterVar, Expr> value_map;
  for (IterVar iv : compute_op->reduce_axis) {
    if (touch_map.count(iv)) {
      dom_map[iv] = iv->dom;
    } else {
      skip_bound_check.insert(iv);
    }
    analyzer.Bind(iv->var, iv->dom);
  }
  schedule::PassDownDomain(reduce_stage, &dom_map, &analyzer, true);
  for (IterVar iv : reduce_stage->leaf_iter_vars) {
    if (touch_map.count(iv)) {
      Range dom = dom_map.at(iv);
      if (is_one(dom->extent)) {
        value_map[iv] = dom->min;
      } else {
        value_map[iv] = iv->var;
      }
    }
  }
  schedule::PassUpIndex(reduce_stage, dom_map, &value_map, true);
  std::vector<Expr> predicates = schedule::MakeBoundCheck(
      reduce_stage, dom_map, value_map, true, skip_bound_check);

  // Get the factored op node.
  const int factor_axis_pos = \
      factor_axis >= 0 ? factor_axis : static_cast<int>(compute_op->axis.size() + 1) + factor_axis;
  CHECK_LE(factor_axis_pos, compute_op->axis.size());
  auto n = make_node<ComputeOpNode>();
  n->name = compute_op->name + ".rf";
  {
    // axis relacement.
    auto iv_node = make_node<IterVarNode>();
    iv_node->dom = dom_map.at(axis);
    CHECK(is_zero(iv_node->dom->min))
        << "Can only factor reduction domain starting from 0";
    iv_node->var = axis->var;
    iv_node->iter_type = kDataPar;

    const int size = compute_op->axis.size();
    for (int idx = 0; idx < size; ++idx) {
      if (factor_axis_pos == idx) {
        n->axis.push_back(IterVar(iv_node));
      }
      n->axis.push_back(compute_op->axis[idx]);
    }
    if (factor_axis_pos == size) {
      n->axis.push_back(IterVar(iv_node));
    }
  }
  // predicate generation, copy not touched axis.
  int idx = tensor->value_index;
  const Reduce* reduce = compute_op->body[idx].as<Reduce>();
  CHECK(reduce) << "Can only rfactor non-inline reductions";
  predicates.push_back(reduce->condition);
  Expr predicate = likely(arith::ComputeReduce<ir::And>(predicates, Expr()));

  std::unordered_map<const Variable*, Expr> vsub;

  for (IterVar iv : compute_op->reduce_axis) {
    if (!touch_map.count(iv)) {
      n->reduce_axis.push_back(iv);
    } else {
      CHECK(value_map.count(iv));
      Expr index = value_map.at(iv);
      vsub[iv->var.get()] = index;
    }
  }

  // Copy touched axis.
  for (IterVar iv : reduce_stage->leaf_iter_vars) {
    if (touch_map.count(iv) && !iv.same_as(axis)) {
      CHECK_EQ(iv->iter_type, kCommReduce);
      auto ncpy = make_node<IterVarNode>(*iv.operator->());
      ncpy->dom = dom_map.at(iv);
      n->reduce_axis.push_back(IterVar(ncpy));
    }
  }
  VarReplacer replacer(vsub);
  Array<Expr> new_source = ir::UpdateArray(reduce->source,
    [&replacer] (const Expr& e) { return replacer.Mutate(e); });

  Expr new_pred = replacer.Mutate(predicate);

  std::vector<Expr> body;
  for (size_t idx = 0; idx < reduce->source.size(); ++idx) {
    if (!n->reduce_axis.empty()) {
      body.emplace_back(Reduce::make(reduce->combiner,
                                     new_source,
                                     n->reduce_axis,
                                     new_pred,
                                     idx));
    } else {
      body.emplace_back(new_source[idx]);
    }
  }
  n->body = Array<Expr>(body);
  // refresh relations, keep the un-touched relations.
  Array<IterVarRelation> rels;
  for (IterVarRelation rel : reduce_stage->relations) {
    bool touched = false;
    if (const SplitNode* r = rel.as<SplitNode>()) {
      if (touch_map.count(r->parent)) touched = true;
    } else if (const FuseNode* r = rel.as<FuseNode>()) {
      if (touch_map.count(r->fused)) touched = true;
    } else if (const RebaseNode* r = rel.as<RebaseNode>()) {
      if (touch_map.count(r->parent)) touched = true;
    } else {
      LOG(FATAL) << "unknown relation type";
    }
    if (!touched) {
      rels.push_back(rel);
    }
  }
  // initialize the factored stage.
  Operation factor_op(n);
  ArrayNode* stages = (*this)->stages.CopyOnWrite();
  size_t stage_pos = FindNodeRef(stages, reduce_stage);
  Stage factor_stage = Stage(factor_op);
  factor_stage->relations = rels;
  CHECK_LT(stage_pos, stages->data.size());
  stages->data.insert(stages->data.begin() + stage_pos,
                      factor_stage);
  (*this)->stage_map.Set(factor_op, factor_stage);
  factor_stage->group = reduce_stage->group;
  if (factor_stage->group.defined()) {
    ++factor_stage->group->num_child_stages;
  }
  // Replace the old reduction.
  IterVar repl_red_axis = reduce_axis(
      dom_map.at(axis), axis->var->name_hint + ".v");
  Array<Tensor> factor_tensors;
  Array<Tensor> old_tensors;
  int size = factor_op->num_outputs();
  for (int idx = 0; idx < size; ++idx) {
    factor_tensors.push_back(factor_op.output(idx));
    old_tensors.push_back(reduce_stage->op.output(idx));
  }
  Array<Tensor> repl_tensors = compute(old_tensors[0]->shape,
    [&](const Array<Var>& i) {
      Array<Expr> indices;
      const int idx_size = static_cast<int>(i.size());
      for (int idx = 0; idx < idx_size; ++idx) {
        if (factor_axis_pos == idx) {
          indices.push_back(repl_red_axis->var);
        }
        indices.push_back(i[idx]);
      }
      if (factor_axis_pos == idx_size) {
          indices.push_back(repl_red_axis->var);
      }
      Array<Expr> factor_exprs;
      for (int idx = 0; idx < size; ++idx) {
        factor_exprs.push_back(factor_tensors[idx](indices));
      }
      Array<Expr> reductions;
      Array<IterVar> axis = {repl_red_axis};
      Expr cond = const_true();
      for (int idx = 0; idx < size; ++idx) {
        reductions.push_back(Reduce::make(reduce->combiner,
          factor_exprs, axis, cond, idx));
      }
      return reductions;
    }, reduce_stage->op->name + ".repl");

  std::unordered_map<Tensor, Tensor> vmap;
  std::unordered_map<Tensor, Tensor> rvmap;
  for (int idx = 0; idx < size; ++idx) {
    vmap[old_tensors[idx]] = repl_tensors[idx];
    rvmap[repl_tensors[idx]] = old_tensors[idx];
  }
  ReplaceDataFlow((*this)->stages, &vmap, &rvmap);
  // revamp the reduction stage.
  reduce_stage->op = repl_tensors[0]->op;
  reduce_stage->all_iter_vars = repl_tensors[0]->op->root_iter_vars();
  reduce_stage->leaf_iter_vars = reduce_stage->all_iter_vars;
  reduce_stage->relations = Array<IterVarRelation>();
  return factor_tensors;
}

}  // namespace air
