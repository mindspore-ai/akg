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

/*
 * 2019.12.30 - Add a class for get name_hint of loop_var and some methods for
 *              Op base compute such as compute realize bounds, build realize.
 * 2022.3.28 - Add a parameter to support inline of CSR tensors.
 */

/*!
 * \brief Compute Op.
 * \file compute_op.cc
 */
#include <tvm/operation.h>
#include <tvm/arithmetic.h>
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include <unordered_set>
#include <string>
#include <utility>
#include "compute_op.h"
#include "op_util.h"
#include "../schedule/message_passing.h"
#include "../arithmetic/compute_expr.h"
#include "../arithmetic/int_set.h"

namespace air {

using namespace ir;

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<ComputeOpNode>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const ComputeOpNode*>(node.get());
    p->stream << "compute(" << op->name << ", " << op << ")";
});

TVM_REGISTER_NODE_TYPE(ComputeOpNode);

/// Verify if ComputeOp is valid with respect to Reduce operations.
static void VerifyComputeOp(const ComputeOpNode *op);

inline bool ReduceEqual(const ir::Reduce* a, const ir::Reduce* b) {
  return (a->combiner.same_as(b->combiner)) &&
         (a->source.same_as(b->source)) &&
         (a->axis.same_as(b->axis)) &&
         (a->condition.same_as(b->condition));
}

int ComputeOpNode::num_outputs() const {
  return body.size();
}

Array<IterVar> BaseComputeOpNode::root_iter_vars() const {
  if (reduce_axis.size() == 0) return axis;
  Array<IterVar> ret = axis;
  for (IterVar iv : reduce_axis) {
    ret.push_back(iv);
  }
  return ret;
}

Type ComputeOpNode::output_dtype(size_t idx) const {
  CHECK_LT(idx, num_outputs());
  return body[idx].type();
}

Array<Expr> BaseComputeOpNode::output_shape(size_t idx) const {
  CHECK_LT(idx, num_outputs());
  // for now, all outputs of a BaseComputeOp have the same shape
  Array<Expr> shape;
  for (const auto& ivar : this->axis) {
    const Range& r = ivar->dom;
    shape.push_back(r->extent);
  }
  return shape;
}

Tensor compute(Array<Expr> shape,
               FCompute fcompute,
               std::string name,
               std::string tag,
               Map<std::string, NodeRef> attrs) {
  auto op_node = make_node<ComputeOpNode>();
  // compute dimension.
  size_t ndim = shape.size();
  std::vector<IterVar> axis;
  std::vector<Var> args;
  for (size_t i = 0; i < ndim; ++i) {
    std::ostringstream os;
    os << "ax" << i;
    axis.emplace_back(IterVarNode::make(
        Range(0, shape[i]), Var(os.str(), shape[i].type()), kDataPar));
    args.push_back(axis.back()->var);
  }

  return ComputeOpNode::make(
      name, tag, attrs, axis, {fcompute(args)}).output(0);
}

Array<Tensor> compute(Array<Expr> shape,
                      FBatchCompute fcompute,
                      std::string name,
                      std::string tag,
                      Map<std::string, NodeRef> attrs) {
  auto op_node = make_node<ComputeOpNode>();
  // compute dimension.
  size_t ndim = shape.size();
  std::vector<IterVar> axis;
  std::vector<Var> args;
  for (size_t i = 0; i < ndim; ++i) {
    std::ostringstream os;
    os << "ax" << i;
    axis.emplace_back(IterVarNode::make(
        Range(0, shape[i]), Var(os.str(), shape[i].type()), kDataPar));
    args.push_back(axis.back()->var);
  }

  Operation op = ComputeOpNode::make(name, tag, attrs, axis, fcompute(args));
  Array<Tensor> outputs;
  for (int idx = 0; idx < op->num_outputs(); ++idx) {
    outputs.push_back(op.output(idx));
  }
  return outputs;
}

Operation ComputeOpNode::make(std::string name,
                              std::string tag,
                              Map<std::string, NodeRef> attrs,
                              Array<IterVar> axis,
                              Array<Expr> body) {
  if (!attrs.defined()) {
    attrs = Map<std::string, NodeRef>();
  }
  auto n = make_node<ComputeOpNode>();
  n->name = std::move(name);
  n->tag = std::move(tag);
  n->attrs = std::move(attrs);
  n->axis = std::move(axis);
  n->body = std::move(body);
  if (n->body[0]->IsInstance<ir::Reduce>()) {
    const ir::Reduce* reduce = n->body[0].as<ir::Reduce>();
    n->reduce_axis = reduce->axis;
  }
  VerifyComputeOp(n.get());
  return Operation(n);
}

// The schedule related logics
Array<Tensor> ComputeOpNode::InputTensors() const {
  Array<Tensor> ret;
  std::unordered_set<Tensor> visited;
  for (auto& e : body) {
    ir::PostOrderVisit(e, [&ret, &visited](const NodeRef& n) {
        const ir::Call *call = n.as<ir::Call>();
        if (call != nullptr && call->func.defined()) {
          Tensor t = Downcast<Operation>(call->func).output(call->value_index);
          if (!visited.count(t)) {
            ret.push_back(t);
            visited.insert(t);
          }
        }
      });
  }
  return ret;
}

Operation ComputeOpNode::ReplaceInputs(
    const Operation& self,
    const std::unordered_map<Tensor, Tensor>& rmap) const {
  CHECK_EQ(self.operator->(), this);
  VerifyComputeOp(this);
  Array<Expr> arr;
  if (this->body[0]->IsInstance<ir::Reduce>()) {
    // Specially handle reduce so the replaced op
    // still share all the components
    Expr new_reduce = op::ReplaceTensor(this->body[0], rmap);
    if (!new_reduce.same_as(this->body[0])) {
      const ir::Reduce* r = new_reduce.as<ir::Reduce>();
      for (size_t k = 0; k < this->body.size(); ++k) {
        auto n = make_node<ir::Reduce>(*r);
        n->value_index = static_cast<int>(k);
        n->type = r->source[k].type();
        arr.push_back(Expr(n));
      }
    } else {
      arr = this->body;
    }
  } else {
    arr = UpdateArray(this->body, [&rmap] (const Expr& e) {
        return op::ReplaceTensor(e, rmap);
      });
  }
  if (!arr.same_as(this->body)) {
    return ComputeOpNode::make(
        this->name, this->tag, this->attrs, this->axis, arr);
  } else {
    return self;
  }
}

void ComputeOpNode::PropBoundToInputs(
    const Operation& self,
    arith::Analyzer* analyzer,
    const std::unordered_map<const Variable*, IntSet>& dom_map,
    std::unordered_map<Tensor, TensorDom>* out_dom_map) const {
  CHECK_EQ(self.operator->(), this);
  auto fvisit = [&dom_map, out_dom_map, analyzer](const NodeRef& n) {
    auto *call = n.as<ir::Call>();
    if (call != nullptr && call->func.defined()) {
      Tensor t = Downcast<Operation>(call->func).output(call->value_index);
      if (t->op.defined() && out_dom_map->count(t)) {
        TensorDom& dom = out_dom_map->at(t);
        for (size_t i = 0; i < t.ndim(); ++i) {
          // We assume that the value of the argument cannot be out of bounds (otherwise it is
          // undefined behaviour), so we can intersect the estimated set of the argument with the
          // range expected by the tensor. However, intersection may result in overly complex
          // expressions, so we perform a more relaxed form of intersection.
          IntSet arg_intset = EvalSet(call->args[i], dom_map);
          const arith::IntervalSetNode* arg_interval = arg_intset.as<arith::IntervalSetNode>();
          if (arg_interval) {
            Expr shape_i_min_value = make_zero(t->shape[i].type());
            Expr shape_i_max_value = t->shape[i] - 1;
            Expr min_value = arg_interval->min_value;
            Expr max_value = arg_interval->max_value;
            // Prefer the shape bounds only when we can prove they are tighter.
            if (arith::is_neg_inf(min_value) ||
                analyzer->CanProve(shape_i_min_value >= min_value)) {
              min_value = shape_i_min_value;
            }
            if (arith::is_pos_inf(max_value) ||
                analyzer->CanProve(shape_i_max_value <= max_value)) {
              max_value = shape_i_max_value;
            }
            dom.data[i].push_back(IntSet::interval(min_value, max_value));
          } else {
            dom.data[i].push_back(arg_intset);
          }
        }
      }
    }
  };
  for (auto& e : body) ir::PostOrderVisit(e, fvisit);
}

void BaseComputeOpNode::GatherBound(
    const Operation& self,
    const std::unordered_map<Tensor, TensorDom>& tensor_dom,
    std::unordered_map<IterVar, Range>* out_dom_map) const {
  CHECK_EQ(self.operator->(), this);
  const TensorDom& tdom = tensor_dom.at(self.output(0));
  for (size_t i = 0; i < this->axis.size(); ++i) {
    Range r = arith::Union(tdom.data.at(i)).cover_range(this->axis[i]->dom);
    CHECK(!out_dom_map->count(this->axis[i]));
    (*out_dom_map)[this->axis[i]] = r;
  }
  for (size_t i = 0; i < this->reduce_axis.size(); ++i) {
    CHECK(!out_dom_map->count(this->reduce_axis[i]));
    (*out_dom_map)[this->reduce_axis[i]] = this->reduce_axis[i]->dom;
  }
}

// visit for-loops in the first produce stmt in realize_body and skip the following stmt
// get the name_hint of loop_var which matches the pattern: "axis->name_hint + ... + (.outer | .inner)"
class GetInnermostIterVars : public IRVisitor {
 public:
  explicit GetInnermostIterVars(const Array<IterVar>& axis)
    : axis_(axis) {}
  using IRVisitor::Visit;

  bool loopVarIsSplitAxisOfIterVar(const std::string& loop_var_name, const std::string& iter_var_name) {
    if (loop_var_name == iter_var_name) {
      return true;
    } else if (loop_var_name.compare(0, iter_var_name.length(), iter_var_name) != 0) {
      return false;
    } else {
      return (loopVarIsSplitAxisOfIterVar(loop_var_name, iter_var_name + ".inner")
        || loopVarIsSplitAxisOfIterVar(loop_var_name, iter_var_name + ".outer"));
    }
  }

  void Visit_(const ProducerConsumer *op) {
    if (producer_fist_visit) {
      this->Visit(op->body);
      producer_fist_visit = false;
    }
  }

  void Visit_(const For *op) {
    if (producer_fist_visit) {
      for (const auto& iv : axis_) {
        if (loopVarIsSplitAxisOfIterVar(op->loop_var->name_hint, iv->var->name_hint)) {
          innermost_iter_vars_name[iv] = op->loop_var->name_hint;
          if (iter_vars_split_names.count(iv) != 1) {
            iter_vars_split_names[iv] = {Var(op->loop_var)};
          } else {
            iter_vars_split_names[iv].push_back(Var(op->loop_var));
          }
        }
      }
      this->Visit(op->body);
    }
  }

  std::unordered_map<IterVar, std::string> innermost_iter_vars_name;
  std::unordered_map<IterVar, Array<Var> > iter_vars_split_names;

 private:
  bool producer_fist_visit{true};
  const Array<IterVar>& axis_;
};

enum IRNodeType {
  EQ, NE, GT, LT, GE, LE, A, Or, Not, Add, Mul, Div, Sub
};

// Perform ComputeRealizeBounds only for the schedulable dimensions
void BaseComputeOpNode::ComputeRealizeBounds(
  const Stage& stage,
  const std::unordered_map<IterVar, Range>& realize_map,
  const Stmt& realize_body,
  Region* bounds) const
{
  const bool enable_opt = true;
  if (!enable_opt) {
    // Only for the schedulable dimensions!
    for (size_t i = 0; i < num_schedulable_dims(); ++i) {
      bounds->push_back(realize_map.at(this->axis[i]));
    }
    return;
  }
  const bool debug_output = false;
  // get the name_hint of the innermost loop_vars of each axis
  GetInnermostIterVars getInnermostIterVars = GetInnermostIterVars(this->axis);
  getInnermostIterVars.Visit(realize_body);

  // get the IterVars matching the name_hint of the innermost loop_vars from realize_map
  std::unordered_map<IterVar, IterVar> axis_to_innermost_iter_var;
  std::unordered_set<IterVar> all_iter_vars;
  for (const auto& axis_to_loop_var: getInnermostIterVars.innermost_iter_vars_name) {
    if (axis_to_loop_var.first->var->name_hint == axis_to_loop_var.second) {
      axis_to_innermost_iter_var[axis_to_loop_var.first] = axis_to_loop_var.first;
    } else {
      IterVar innermost_iter_var;
      for (const auto &kv : realize_map) {
        if (kv.first->var->name_hint == axis_to_loop_var.second) {
          innermost_iter_var = kv.first;
        }
      }
      CHECK(innermost_iter_var.defined());
      all_iter_vars.insert(innermost_iter_var);
      axis_to_innermost_iter_var[axis_to_loop_var.first] = innermost_iter_var;
    }
  }

  ComputeLoopNest ret;
  // make main loop nest
  bool debug_keep_trivial_loop = false;
  // The goal is to create expressions for the lower and upper bounds of each IV.
  // The lowerbound will be in the form of max(expr1, expr2, ...) and the upperbound
  // in the form of min(expr1, expr2, ...). The lowerbound is inclusive and the upperbound
  // will be non-inclusive, i.e., lowerbound <= IV < upperbound
  // Finally, we compute the size of each IV as upperbound - lowerbound.
  // where we only iterate over the schedulable dimensions.
  for (size_t i = 0; i < num_schedulable_dims(); ++i) {
    const IterVar &iv_org = this->axis[i];
    if (axis_to_innermost_iter_var.count(iv_org) != 1) {
      bounds->push_back(realize_map.at(iv_org));
      continue;
    }
    IterVar iv = axis_to_innermost_iter_var[iv_org];
    // assign the original values of the lowerbound and upperbound, i.e.,
    // the ones computed by original TVM code
    auto max = realize_map.at(iv_org)->extent;
    auto min = make_zero(Int(32));
    std::unordered_set<IterVar> skip_iter_vars = all_iter_vars;
    skip_iter_vars.erase(iv);
    // generate the rest of the expressions for lowerbound and upperbound using
    // loopnest predicates
    ret.main_nest = op::MakeLoopNest(
      stage, realize_map, 0, false, skip_iter_vars, &ret.main_vmap,
      debug_keep_trivial_loop);
    ret.main_predicates = schedule::MakeBoundCheck(
      stage, realize_map, ret.main_vmap,
      false, std::unordered_set<IterVar>());

    if (debug_output) {
      std::cout << "********************************" << std::endl << "IterVar: " << iv <<std::endl;
      std::cout << "vmap {" << std::endl;
      for (const auto &elem : ret.main_vmap) {
        std::cout << elem.first << "(" << elem.first->var.get() << ")  "
                  << elem.second << std::endl;
      }
      std::cout << "}" << std::endl;
      std::cout<< "predicates {" <<std::endl;
      for (const auto &e : ret.main_predicates) {
        std::cout<< e <<std::endl;
      }
      std::cout << "}" << std::endl;
    }

    Array<Var> matched_loop_vars;
    // hack: We need to map the IterVar of the ComputeOpNode to the IterVar used in
    // the loops and predicates generated by MakeLoopNest and MakeBoundCheck.
    // Currently, I'm using variable names to do this which is hacky.
    // Note that in general, a single IterVar of the ComputeNode may correspond
    // to multiple variables in the generated loop (e.g., when we split a loop)
    // For those IterVars we won't modify the original values of the bounds.
    for (const auto &map : ret.main_vmap) {
      if (!map.first->dom.defined() &&
        map.first->var->name_hint == iv->var->name_hint) {
        if (debug_output) std::cout << map.second <<" " << map.first->var.get() << " " << iv->var.get() << std::endl;
        if (map.second.as<Variable>() != nullptr) {
          matched_loop_vars.push_back(Downcast<Var>(map.second));
        } else if (const IntImm *i = map.second.as<IntImm>()) {
          CHECK(i->value == 0);
        } else {
          LOG(FATAL) << "unhandled situation";
        }
      }
    }
    CHECK(matched_loop_vars.size() <= 1);
    if (matched_loop_vars.size() == 1) {
      auto loop_var = matched_loop_vars[0];
      Array<Var> loop_vars=getInnermostIterVars.iter_vars_split_names[iv_org];
      if (debug_output) std::cout << "loop var: " << loop_var << std::endl;
      for (auto &pred : ret.main_predicates) {
        // For each predicate, figure out whether it determines a lower bound
        // or an upper bound for the IV (say x) and compute that bound
        if (debug_output) std::cout << "pred: " << pred << std::endl;
        Expr diff;
        IRNodeType node_type = IRNodeType::Not;
        // Take right side of the inequality to the left side and make and expression
        // of the difference
        if (const auto *ge_pred = pred.as<ir::GE>()) {
          node_type = IRNodeType::GE;
          diff = Sub::make(ge_pred->a, ge_pred->b);
        } else if (const auto *gt_pred = pred.as<ir::GT>()) {
          diff = Sub::make(gt_pred->a, gt_pred->b);
          node_type = IRNodeType::GT;
        } else if (const auto *lt_pred = pred.as<ir::LT>()) {
          diff = Sub::make(lt_pred->a, lt_pred->b);
          node_type = IRNodeType::LT;
        } else if (const auto *le_pred = pred.as<ir::LE>()) {
          diff = Sub::make(le_pred->a, le_pred->b);
          node_type = IRNodeType::LE;
        } else {
          LOG(FATAL) << "This was unexpected";
        }
        if (debug_output) std::cout << "diff: " << diff << std::endl;
        // Find integer a and expression b such that diff = ax + b
        auto coeffs = arith::DetectLinearEquation(diff, loop_vars);
        if (debug_output) {
          std::cout << "coeffs for " << loop_var->name_hint << ":" << std::endl;
          for (auto &coeff : coeffs) {
            std::cout << coeff << "  ";
          }
          std::cout << std::endl;
        }
        CHECK(coeffs.size() == loop_vars.size() + 1);
        CHECK(coeffs[loop_vars.size() - 1].as<ir::IntImm>() != nullptr);
        auto b = coeffs[loop_vars.size()];
        int64_t a = coeffs[loop_vars.size() - 1].as<ir::IntImm>()->value;
        if (a == 0) {
          continue;
        }
        for (int i = loop_vars.size() - 2; i >= 0; i--) {
          CHECK(coeffs[i].as<ir::IntImm>() != nullptr);
        }
        // Convert all inequality forms to either ax + b > 0 or ax + b <= 0
        // where a is positive
        // First: if a is negative multiple both sides of the inequality by -1
        if (a < 0) {
          a = -a;
          b = -b;
          switch (node_type) {
            case IRNodeType::GE: node_type = IRNodeType::LE;
              break;
            case IRNodeType::LE: node_type = IRNodeType::GE;
              break;
            case IRNodeType::GT: node_type = IRNodeType::LT;
              break;
            case IRNodeType::LT: node_type = IRNodeType::GT;
              break;
            default:LOG(FATAL) << "impossible";
          }
        }
        CHECK(a > 0);
        // Second: change >= to > and < to <=.
        if (node_type == IRNodeType::GE) {
          b = b + 1;
          node_type = IRNodeType::GT;
        } else if (node_type == IRNodeType::LT) {
          b = b + 1;
          node_type = IRNodeType::LE;
        }
        auto new_bound = truncdiv(-b, IntImm::make(Int(32), a)) + 1;
        if (node_type == IRNodeType::LE) {
          max = Min::make(max, new_bound);
        } else if (node_type == IRNodeType::GT) {
          min = Max::make(min, new_bound);
        }
      }
    }
    if (debug_output) {
      std::cout << "min: " << min << "  max: " << max << std::endl;
      std::cout << "iv->dom->min: " << iv_org->dom << std::endl;
      std::cout << "realize_map.at(iv_org): " << realize_map.at(iv_org) << std::endl;
      std::cout << "realize_map.at(iv): " << realize_map.at(iv) << std::endl;
    }
    Map<Var, Range> vrange;
    for (auto &e : realize_map) {
      vrange.Set(e.first->var, e.second);
    }
    auto extent = Simplify(max - min, vrange);
    bounds->push_back(Range::make_by_min_extent(Simplify(Max::make(realize_map.at(iv_org)->min, iv_org->dom->min), vrange), extent));
  }
  if (debug_output) {
    std::cout << "old bounds are " << std::endl;
    for (IterVar iv : this->axis) {
      std::cout << "[" << realize_map.at(iv)->min << ", " << realize_map.at(iv)->extent << "] ";
    }
    std::cout << std::endl << "new bounds are " << std::endl;
    for (const auto &bound : *bounds) {
      std::cout << "[" << bound->min << ", " << bound->extent << "] ";
    }
    std::cout<<std::endl;
  }
}


Stmt BaseComputeOpNode::BuildRealize(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& realize_map,
    const Stmt& realize_body) const {
  CHECK_EQ(stage->op.get(), this);
  Region bounds;
  ComputeRealizeBounds(stage, realize_map, realize_body, &bounds);
  for (size_t i = num_schedulable_dims(); i < this->axis.size(); i++) {
    bounds.push_back(realize_map.at(this->axis[i]));
  }
  Stmt realize = realize_body;
  for (int i = this->num_outputs(); i > 0; --i) {
    Tensor t = stage->op.output(i-1);
    realize = ir::Realize::make(t->op, t->value_index,
      t->dtype, bounds, const_true(), realize);
    // alignment requirement, only useful for compute
    for (size_t i = 0; i < num_schedulable_dims(); ++i) {
      auto it = stage->iter_var_attrs.find(this->axis[i]);
      if (it != stage->iter_var_attrs.end()) {
        IterVarAttr attr = (*it).second;
        if (attr->dim_align_factor != 0) {
          Array<Expr> tuple = {static_cast<int>(i),
                               attr->dim_align_factor,
                               attr->dim_align_offset};
          realize = ir::AttrStmt::make(
              t, ir::attr::buffer_dim_align,
              Call::make(Handle(), ir::intrinsic::tvm_tuple, tuple, Call::Intrinsic),
              realize);
        }
      }
    }
  }
  return realize;
}

size_t ComputeOpNode::num_schedulable_dims() const {
  return axis.size();
}

// Build a reduction body.
void MakeReduction(const ComputeOpNode* op,
                   const Array<Tensor>& tensors,
                   Stmt* init,
                   Stmt* provide,
                   Expr csr_access) {
  Array<Expr>  args;
  if (csr_access.defined()) {
    args.push_back(csr_access);
    for (size_t i = 2; i < op->axis.size(); ++i) {
      args.push_back(op->axis[i]->var);
    }
  } else {
    for (IterVar iv : op->axis) {
      args.push_back(iv->var);
    }
  }
  std::vector<Stmt> inits, provides;

  size_t size = op->body.size();
  const Reduce* reduce = op->body[0].as<Reduce>();
  CHECK(reduce);
  const CommReducerNode* combiner = reduce->combiner.as<CommReducerNode>();
  CHECK(combiner);
  Array<Expr> lhs;
  for (size_t i = 0; i < size; ++i) {
    lhs.push_back(tensors[i](args));
  }
  Array<Expr> init_value = combiner->identity_element;
  Array<Expr> update_value = (*combiner)(lhs, reduce->source);
  for (size_t i = 0; i < size; ++i) {
    Tensor t = tensors[i];
    inits.emplace_back(Provide::make(
          t->op, t->value_index, init_value[i], args));
    provides.emplace_back(Provide::make(
          t->op, t->value_index, update_value[i], args));

    // Annotate reduction op for Poly pass
    //@{
    provides.back() = AttrStmt::make(op->reduce_axis, attr::reduce_update,
                                     Expr(""), provides.back());
    //@}
  }
  *init = Block::make(inits);
  *provide = Block::make(provides);
  if (!is_one(reduce->condition)) {
    *provide = IfThenElse::make(reduce->condition, *provide);
  }
}

// Normal computation.
Stmt MakeProvide(const ComputeOpNode* op,
                 const Tensor& t) {
  Array<Expr> args;
  for (IterVar iv : op->axis) {
    args.push_back(iv->var);
  }
  return Provide::make(t->op, t->value_index, op->body[t->value_index], args);
}

Stmt MakeComputeStmt(const ComputeOpNode* self,
                     const Stage& stage,
                     const std::unordered_map<IterVar, Range>& dom_map,
                     bool debug_keep_trivial_loop) {
  // grab the nest structure
  ComputeLoopNest n = ComputeLoopNest::make(self, stage, dom_map, debug_keep_trivial_loop);
  // Normal loop structure
  n.init_nest.emplace_back(op::MakeIfNest(n.init_predicates));
  n.main_nest.emplace_back(op::MakeIfNest(n.main_predicates));
  if (self->reduce_axis.size() != 0) {
    // make reduction.
    Stmt init, provide;
    Array<Tensor> source;
    for (size_t i = 0; i < self->body.size(); ++i) {
      if (stage->csr_access.defined()) {
        auto node = make_node<TensorNode>();
        node->op = stage->op;
        node->value_index = i;
        node->dtype = stage->op->output_dtype(i);
        CHECK_LT(i, stage->csr_output_shape.size());
        node->shape = stage->csr_output_shape[i];
        source.push_back(Tensor(node));
      } else {
        source.push_back(stage->op.output(i));
      }
    }
    MakeReduction(self, source, &init, &provide, stage->csr_access);
    init = MergeNest(n.init_nest, init);
    init = op::Substitute(init, n.init_vmap);
    // common nest
    std::vector<std::vector<Stmt> > common(
        n.main_nest.begin(), n.main_nest.begin() + n.num_common_loop + 1);
    std::vector<std::vector<Stmt> > reduce(
        n.main_nest.begin() + n.num_common_loop + 1, n.main_nest.end());
    provide = MergeNest(reduce, provide);
    if (debug_keep_trivial_loop || stage->csr_access.defined()) {
      // for reduce stmt inlined with csr tensors, init is handled by inplace assign.
      provide = MergeNest(common, provide);
    } else {
      provide = MergeNest(common, Block::make(init, provide));
    }
    // run substitution in the on the full nest, because  loop condition
    // could depend on outer loops.
    return op::Substitute(provide, n.main_vmap);
  } else {
    std::vector<Stmt> provides;
    for (size_t i = 0; i < self->body.size(); ++i) {
      provides.emplace_back(MakeProvide(self, stage->op.output(i)));
    }
    Stmt provide = Block::make(provides);
    provide = MergeNest(n.main_nest, provide);
    // run substitution in the on the full nest, because  loop condition
    // could depend on outer loops.
    return op::Substitute(provide, n.main_vmap);
  }
}

enum class ComputeType {
  kNormal,
  kCrossThreadReduction,
  kTensorize
};

ComputeType DetectComputeType(const ComputeOpNode* self,
                              const Stage& stage) {
  // Verify correctness of leaf nest.
  int normal_red = 0, thread_red = 0, tensorize = 0;

  for (IterVar iv : stage->leaf_iter_vars) {
    IterVarAttr attr;
    auto it = stage->iter_var_attrs.find(iv);
    if (it != stage->iter_var_attrs.end()) {
      attr = (*it).second;
    }
    if (attr.defined() && attr->iter_type == kTensorized) {
      ++tensorize;
    }
    if (iv->iter_type == kCommReduce) {
      if (attr.defined() && attr->bind_thread.defined()) {
        ++thread_red;
      } else {
        ++normal_red;
      }
    } else {
      CHECK_EQ(thread_red, 0)
          << "Cross thread reduce cannot swap with normal data axis";
    }
  }
  if (tensorize != 0) {
    CHECK(thread_red == 0)
        << "Cannot mix cross thread reduction with Tensorize";
    return ComputeType::kTensorize;
  }
  CHECK(normal_red == 0 || thread_red == 0)
      << "Cannot mix normal reduction with thread reduce";
  if (thread_red != 0) {
    return ComputeType::kCrossThreadReduction;
  } else {
    return ComputeType::kNormal;
  }
}

// implement the provide utility.
Stmt ComputeOpNode::BuildProvide(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    bool debug_keep_trivial_loop) const {
  CHECK_EQ(stage->op.operator->(), this);
  ComputeType ctype = DetectComputeType(this, stage);
  if (ctype == ComputeType::kCrossThreadReduction) {
    // specially handle cross thread reduction.
    return MakeCrossThreadReduction(this, stage, dom_map, debug_keep_trivial_loop);
  } else if (ctype == ComputeType::kTensorize) {
    return MakeTensorize(this, stage, dom_map, debug_keep_trivial_loop);
  } else {
    return MakeComputeStmt(this, stage, dom_map, debug_keep_trivial_loop);
  }
}

ComputeLoopNest ComputeLoopNest::make(
    const BaseComputeOpNode* self,
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    bool debug_keep_trivial_loop) {
  CHECK_EQ(stage->op.operator->(), self);
  ComputeLoopNest ret;
  // make main loop nest
  ret.main_nest = op::MakeLoopNest(
      stage, dom_map, 0, false, std::unordered_set<IterVar>(), &ret.main_vmap,
      debug_keep_trivial_loop);
  ret.main_predicates = schedule::MakeBoundCheck(
      stage, dom_map, ret.main_vmap, false,
      std::unordered_set<IterVar>());
  for (auto& e : ret.main_predicates) {
    e = likely(e);
  }
  if (stage->store_predicate.defined()) {
    ret.main_predicates.push_back(stage->store_predicate);
  }
  if (self->reduce_axis.size() != 0) {
    // try to find the location to insert the initialization.
    // Fuse the initialization and provide loop when possible.
    std::unordered_map<IterVar, int> update_state;
    for (IterVar iv : self->reduce_axis) {
      update_state[iv] = 2;
    }
    for (size_t i = 0; i < self->num_schedulable_dims(); ++i) {
      update_state[self->axis[i]] = 1;
    }
    // find which iter var is related to reduction and which is related to axis.
    schedule::PassDownBitMaskOr(stage, &update_state);
    auto leaf_iter_vars = stage->leaf_iter_vars;
    // first first loop that is related to reduction.
    size_t begin_loop = leaf_iter_vars.size();
    for (size_t i = 0; i < leaf_iter_vars.size(); ++i) {
      auto iv = leaf_iter_vars[i];
      int flag = update_state.at(iv);
      if ((flag & 2) != 0) {
        begin_loop = i; break;
      }
      ret.init_vmap[iv] = ret.main_vmap.at(iv);
    }
    ret.num_common_loop = begin_loop;
    // skip loops that are related to reduction and are unrelated to axis.
    std::unordered_set<IterVar> skip_iter;
    for (auto kv : update_state) {
      int flag = kv.second;
      if (flag == 2) skip_iter.insert(kv.first);
    }
    ret.init_nest = op::MakeLoopNest(
        stage, dom_map, begin_loop, true,
        skip_iter, &(ret.init_vmap), debug_keep_trivial_loop);
    ret.init_predicates = schedule::MakeBoundCheck(
        stage, dom_map, ret.init_vmap, true, skip_iter);
    for (auto& e : ret.init_predicates) {
      e = likely(e);
    }
  } else {
    CHECK_EQ(ret.main_nest.size(), stage->leaf_iter_vars.size() + 1);
    ret.num_common_loop = stage->leaf_iter_vars.size();
  }
  // copy elison here.
  return ret;
}

namespace {
/*!
 * \brief Verify if ComputeOp is valid with respect to Reduce operations.
 *
 *  The following two properties are verified:
 *  (1) All Reduce operations must exist at top level.
 *  (2) For a list of operations, if one is Reduce, then the others
 *      must be Reduce as well; and their inputs should have the
 *      same attribute except value_index.
 */
class ComputeVerifier final : protected ir::IRVisitor {
 public:
  /// Special member functions
  //@{
  explicit ComputeVerifier(const ComputeOpNode* compute)
      : compute_(compute), reduce_(compute->body[0].as<ir::Reduce>()) {}
  virtual ~ComputeVerifier() = default;
  ComputeVerifier(const ComputeVerifier&) = delete;
  ComputeVerifier(ComputeVerifier&&) = delete;
  ComputeVerifier& operator=(const ComputeVerifier&) = delete;
  ComputeVerifier& operator=(ComputeVerifier&&) = delete;
  //@}

  /// Interface to perform compute verification
  void Run() {
    for (const Expr e : compute_->body) {
      // Check for consistency of top level reductions
      const ir::Reduce* reduce = e.as<ir::Reduce>();
      CHECK((reduce && reduce_) || (!reduce && !reduce_))
          << "All ComputeOp should be consistent "
          << "with being Reduce operation or not.";

      if (reduce && reduce_) {
        CHECK(ReduceEqual(reduce, reduce_))
            << "The Reduce inputs of ComputeOp should "
            << "have the same attribute except value_index";
      }

      level_ = 0;
      ir::IRVisitor::Visit(e);
    }
  }

 protected:
  /// Visitor implementation
  //@{
  void Visit(const NodeRef& n) final {
    ++level_;
    ir::IRVisitor::Visit(n);
    --level_;
  }

  void Visit_(const ir::Reduce* op) final {
    // Check for non top level reductions
    CHECK(0 == level_)
        << "Reductions are only allowed at the top level of compute. "
        << "Please create another tensor for further composition.";
  }
  //@}

 private:
  const ComputeOpNode* compute_{nullptr};  ///< ComputeOpNode to verify
  const ir::Reduce* reduce_{nullptr};      ///< Top level Reduce operation
  int level_{0};                           ///< Level of op being processed
};
}  // namespace

/// Verify if ComputeOp is valid with respect to Reduce operations.
static void VerifyComputeOp(const ComputeOpNode* op) {
  ComputeVerifier v(op);
  v.Run();
}

Stmt TransformUpdate(const Stage& stage,
                     const std::unordered_map<IterVar, Range>& dom_map,
                     const ComputeLoopNest& n,
                     Stmt body,
                     Stmt update) {
  Array<Expr> conds;
  std::unordered_set<const Variable*> banned;
  for (size_t i = 0; i < stage->leaf_iter_vars.size(); ++i) {
    IterVar iv = stage->leaf_iter_vars[i];
    auto iit = stage->iter_var_attrs.find(iv);
    if (iit != stage->iter_var_attrs.end()) {
      const IterVarAttr& attr = (*iit).second;
      if (attr->iter_type == kTensorized) {
        break;
      }
    }
    if (iv->iter_type == kCommReduce) {
      auto vit = dom_map.find(iv);
      CHECK(vit != dom_map.end());
      const Range& vrange = vit->second;
      conds.push_back(likely(iv->var > vrange->min));
      banned.insert(iv->var.get());
    }
  }
  for (const Expr& pred : n.main_predicates) {
    if (ir::ExprUseVar(pred, banned)) {
      LOG(FATAL) << "Tensorize update transform failed, the condition "
                 << pred << " has a conflict with the reset condition";
    }
  }

  return IfThenElse::make(arith::ComputeReduce<ir::Or>(conds, const_true(1)),
                          update, body);
}
}  // namespace air
