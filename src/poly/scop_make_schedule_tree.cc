/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pass/utils.h"
#include "construct_poly_accesses.h"
#include "poly/scop_builder.h"
#include "poly/schedule_tree_util.h"

namespace akg {
namespace ir {
namespace poly {
class ExtractCond : protected IRVisitor {
 public:
  ExtractCond() {}
  ~ExtractCond() override = default;
  std::vector<Expr> run(const Expr expr) {
    IRVisitor::Visit(Simplify_cce(expr));
    return result;
  }
  bool hasBothOrAndAnd() const { return (or_num && and_num); }
  bool IsOr() {
    if (!or_num && !and_num && result.size() > 1) {
      LOG(INFO) << "  result.size() > 1 and or(and)_num = 0";
    }
    return (or_num > 0);
  }
  std::vector<Expr> result;

 protected:
#define COMOP_VISIT_(OP)                    \
  void Visit_(const OP *op) final {         \
    has_tensor = false;                     \
    this->Visit(op->a);                     \
    this->Visit(op->b);                     \
    if (!has_tensor) {                      \
      Expr expr(GetRef<Expr>(op));          \
      result.push_back(Simplify_cce(expr)); \
    }                                       \
  }
  COMOP_VISIT_(EQ)
  COMOP_VISIT_(NE)
  COMOP_VISIT_(LT)
  COMOP_VISIT_(LE)
  COMOP_VISIT_(GT)
  COMOP_VISIT_(GE)

  void Visit_(const Call *op) final {
    IRVisitor::Visit_(op);
    if (op->call_type == Call::Halide) {
      has_tensor = true;
    }
  }
  void Visit_(const And *op) final {
    and_num++;
    this->Visit(op->a);
    this->Visit(op->b);
  }

  void Visit_(const Or *op) final {
    or_num++;
    this->Visit(op->a);
    this->Visit(op->b);
  }

  void Visit_(const Not *op) final {
    Expr expr(GetRef<Expr>(op));
    LOG(FATAL) << expr << " so far NOT is handled, please modify DSL";
  }

 private:
  int or_num{0};
  int and_num{0};
  bool has_tensor{false};
};

class CutSetTopDown final : protected IRVisitor {
 public:
  CutSetTopDown() {}
  ~CutSetTopDown() override = default;

  const isl::union_map Run(const Expr &expr, const isl::multi_id &tuple_, const isl::union_map &accesses_,
                           const isl::set &read_set_) {
    accesses = accesses_;
    read_set = read_set_;
    tuple = tuple_;
    Visit(expr);
    return accesses;
  }

 private:
  static std::unordered_set<std::string> GatherCallTensors(const Expr &e) {
    std::unordered_set<std::string> tensor_names;
    PostOrderVisit(e, [&](const NodeRef &node) -> void {
      if (auto op = node.as<Call>()) {
        if (op->call_type == Call::CallType::Halide) {
          tensor_names.insert(op->func->func_name());
        }
      }
    });
    return tensor_names;
  }

  void CutAccesses(const Expr &value, const std::vector<Expr> &conds, bool is_else, bool is_or) {
    auto may_access_tensors = GatherCallTensors(value);
    isl::union_map must_access = isl::union_map::empty(accesses.space());
    isl::union_map may_access = isl::union_map::empty(accesses.space());
    accesses.foreach_map([&](const isl::map &map) {
      auto tensor = map.get_tuple_id(isl_dim_out).get_name();
      if (may_access_tensors.count(tensor) == 0) {
        must_access = must_access.add_map(map);
      } else {
        may_access = may_access.add_map(map);
      }
    });
    read_set = CutSet(conds, read_set, is_else, is_or);
    auto cut_may_access = may_access.curry().intersect_domain(read_set.unbind_params(tuple)).uncurry();
    accesses = must_access.unite(cut_may_access);
  }

  void Visit_(const Select *sel) final {
    auto ec = ExtractCond();
    std::vector<Expr> conds = ec.run(sel->condition);
    if (!ec.hasBothOrAndAnd()) {
      if (isImm(sel->true_value)) {
        CutAccesses(sel->false_value, conds, true, !ec.IsOr());
      } else if (isImm(sel->false_value)) {
        CutAccesses(sel->true_value, conds, false, ec.IsOr());
      }
    }
  }

  isl::union_map accesses;
  isl::set read_set;
  isl::multi_id tuple;
};

class ScopMakeScheduleTree final : protected IRVisitor {
 public:
  ScopMakeScheduleTree(const NodeRef s, ScopInfo &scop_info, const isl::set set, const isl::id_list outer,
                       ssize_t macro_stmt)
      : s(s), scop_info_(scop_info), set(set), outer(outer), macro_stmt(macro_stmt) {
    IRVisitor::Visit(s);
  }
  ~ScopMakeScheduleTree() override = default;

  /// Visitor implementation
  void Visit_(const Provide *op) final {
    size_t stmt_index = scop_info_.analysis_result_.GetStatementMap().size();
    isl::id id(set.ctx(), macro_stmt >= 0 ? kStatementLabel + std::to_string(macro_stmt)
                                          : kStatementLabel + std::to_string(stmt_index));
    scop_info_.analysis_result_.RecordStatement(id, op);

    auto tuple_space = isl::space(set.ctx(), 0);
    tuple_space = tuple_space.add_named_tuple_id_ui(id, static_cast<unsigned int>(outer.size()));
    OperatorDomainSpace op_domain;
    op_domain.param_space = set.get_space();
    op_domain.tuple = isl::multi_id(tuple_space, outer);
    scop_info_.analysis_result_.RecordOperatorDomain(id, op_domain);
    auto domain = set.unbind_params(op_domain.tuple);
    sch = isl::schedule::from_domain(domain);

    isl::union_map new_reads, new_writes, new_to_inner;
    isl::union_map new_reads_with_conds, new_writes_with_conds;
    isl::set read_set = set;
    isl::set write_set = set;
    isl::set reduction_set = set;
    Stmt stmt = Downcast<Stmt>(s);
    std::tie(new_reads, new_writes, new_to_inner) =
      ConstructPolyAccesses(op_domain, stmt, scop_info_.analysis_result_.GetAccessMap());

    new_reads_with_conds = new_reads.curry().intersect_domain(read_set.unbind_params(op_domain.tuple)).uncurry();
    /// has Select
#if (SELECT_DOMAIN_OPT)
    if (scop_info_.user_config_.GetTarget() == TARGET_CCE) {
      new_reads_with_conds = CutSetTopDown().Run(op->value, op_domain.tuple, new_reads_with_conds, read_set);
    }
#endif
    new_writes_with_conds = new_writes.curry().intersect_domain(write_set.unbind_params(op_domain.tuple)).uncurry();

    ParseStmtOps(id, op, scop_info_.analysis_result_, new_reads, new_writes);

    // The parameters should be added as constraints of the reads/writes sets
    // otherwise isl may not be able to obtain a fixed box.
    if (macro_stmt >= 0) {
      auto params = domain.params();
      new_reads = new_reads.curry().intersect_domain(params).uncurry();
      new_writes = new_writes.curry().intersect_domain(params).uncurry();

      new_reads_with_conds = new_reads_with_conds.curry().intersect_domain(params).uncurry();
      new_writes_with_conds = new_writes_with_conds.curry().intersect_domain(params).uncurry();
    }
    scop_info_.analysis_result_.RecordReads(scop_info_.analysis_result_.GetReads().unite(new_reads_with_conds));
    scop_info_.analysis_result_.RecordWrites(scop_info_.analysis_result_.GetWrites().unite(new_writes_with_conds));
    found = true;
  }

  void Visit_(const Block *op) final {
    auto sch_first = MakeScheduleTreeHelper(op->first, scop_info_, set, outer, macro_stmt);
    auto sch_rest = MakeScheduleTreeHelper(op->rest, scop_info_, set, outer, macro_stmt);
    if (macro_stmt >= 0)
      sch = sch_first;
    else
      sch = sch_first.sequence(sch_rest);
    found = true;
  }

  void Visit_(const IfThenElse *op) final {
    Expr cond = op->condition;

    size_t stmt_index = scop_info_.analysis_result_.GetStatementMap().size();
    isl::id id(set.ctx(), macro_stmt >= 0 ? kStatementLabel + std::to_string(macro_stmt)
                                          : kStatementLabel + std::to_string(stmt_index));
    scop_info_.analysis_result_.RecordStatement(id, op);
    auto tuple_space = isl::space(set.ctx(), 0);
    tuple_space = tuple_space.add_named_tuple_id_ui(id, static_cast<unsigned int>(outer.size()));
    OperatorDomainSpace op_domain;
    op_domain.param_space = set.get_space();
    op_domain.tuple = isl::multi_id(tuple_space, outer);
    scop_info_.analysis_result_.RecordOperatorDomain(id, op_domain);
    auto domain = set.unbind_params(op_domain.tuple);
    sch = isl::schedule::from_domain(domain);

    isl::union_map new_reads, new_writes, new_to_inner;

    // Update the reads/writes sets of scop_info by analyzing the condition
    Stmt condition = Stmt(GetObjPtr<Object>(cond.get()));
    std::tie(new_reads, new_writes, new_to_inner) =
      ConstructPolyAccesses(op_domain, condition, scop_info_.analysis_result_.GetAccessMap());
    StmtOpInfo stmt_op_Info;
    for (auto a : new_reads.get_map_list()) {
      auto tensor_id = a.get_tuple_id(isl_dim_out);
      stmt_op_Info.readtensors.push_back(tensor_id);
    }
    scop_info_.analysis_result_.RecordStmtOpInfo(id, stmt_op_Info);
    ParseStmtOps(id, cond, scop_info_.analysis_result_, FunctionRef(GetObjPtr(cond.get())));
    scop_info_.analysis_result_.RecordReads(scop_info_.analysis_result_.GetReads().unite(new_reads));
    scop_info_.analysis_result_.RecordWrites(scop_info_.analysis_result_.GetWrites().unite(new_writes));

    // Update the flag for recording a macro statement
    if (macro_stmt < 0) macro_stmt = static_cast<int64_t>(stmt_index);
    // Build schedule for the then case without updating the schedule

    isl::set cut_set = set;

#if (SELECT_DOMAIN_OPT)
    auto ec = ExtractCond();
    std::vector<Expr> cond_vec;
    if (scop_info_.user_config_.GetTarget() == TARGET_CCE) {
      cond_vec = ec.run(cond);
      if (!ec.hasBothOrAndAnd()) {
        cut_set = CutSet(cond_vec, set, false, ec.IsOr());
      }
    }
#endif
    static_cast<void>(MakeScheduleTreeHelper(op->then_case, scop_info_, cut_set, outer, macro_stmt));

    // Build schedule for the else case without updating the schedule if defined
    if (op->else_case.defined()) {
#if (SELECT_DOMAIN_OPT)
      if (scop_info_.user_config_.GetTarget() == TARGET_CCE) {
        if (!ec.hasBothOrAndAnd()) {
          cut_set = CutSet(cond_vec, set, true, !ec.IsOr());
        }
      }
#endif
      static_cast<void>(MakeScheduleTreeHelper(op->else_case, scop_info_, cut_set, outer, macro_stmt));
    }

    found = true;
  }

  void Visit_(const Evaluate *op) final {
    const Call *call_op = op->value.as<Call>();
    if (call_op && call_op->name == CALL_IM2COL_UB) {
      size_t stmt_index = scop_info_.analysis_result_.GetStatementMap().size();
      isl::id id(set.ctx(), macro_stmt >= 0 ? kStatementLabel + std::to_string(macro_stmt)
                                            : kStatementLabel + std::to_string(stmt_index));
      scop_info_.analysis_result_.RecordStatement(id, op);
      auto tuple_space = isl::space(set.ctx(), 0);
      tuple_space = tuple_space.add_named_tuple_id_ui(id, static_cast<unsigned int>(outer.size()));

      OperatorDomainSpace op_domain;
      op_domain.param_space = set.get_space();
      op_domain.tuple = isl::multi_id(tuple_space, outer);
      scop_info_.analysis_result_.RecordOperatorDomain(id, op_domain);

      auto domain = set.unbind_params(op_domain.tuple);
      sch = isl::schedule::from_domain(domain);

      isl::union_map new_reads, new_writes, new_to_inner;
      Stmt stmt = Downcast<Stmt>(s);
      for (auto item : scop_info_.analysis_result_.GetAttrStmt()) {
        if (item->attr_key == ATTR_IM2COL_KEY) {
          stmt = AttrStmt::make(item->node, item->attr_key, item->value, stmt);
        }
      }
      std::tie(new_reads, new_writes, new_to_inner) =
        ConstructPolyAccesses(op_domain, stmt, scop_info_.analysis_result_.GetAccessMap());

      ParseStmtOps(id, op, scop_info_.analysis_result_, new_reads, new_writes);

      if (macro_stmt >= 0) {
        auto params = domain.params();
        new_reads = new_reads.curry().intersect_domain(params).uncurry();
        new_writes = new_writes.curry().intersect_domain(params).uncurry();
        new_to_inner = new_to_inner.curry().intersect_domain(params).uncurry();
      }
      scop_info_.analysis_result_.RecordReads(scop_info_.analysis_result_.GetReads().unite(new_reads));
      scop_info_.analysis_result_.RecordWrites(scop_info_.analysis_result_.GetWrites().unite(new_writes));
      found = true;
    }
  }

  void AddLoopBoundConstraints(const isl::aff &loop_var, const isl::space &space, const Expr &expr, bool permit_min,
                               bool permit_max) {
    auto constraint_bounds = Expr2AffChecked(space, expr, permit_min, permit_max);
    if (constraint_bounds.size() == 0u) LOG(INFO) << "could not obtain polyhedral lower / upper bounds from " << expr;
    for (const auto &item : constraint_bounds) {
      if (!permit_min && permit_max) {
        set = set.intersect(loop_var.ge_set(item));
      } else if (permit_min && !permit_max) {
        set = set.intersect(item.ge_set(loop_var));
      }
    }
  }

  isl::union_pw_aff GetUnionPwAffAtDomain(const isl::aff &f, const isl::union_set &domain,
                                          const OperatorDomainMap &map) {
    auto upa = isl::union_pw_aff::empty(domain.space());
    for (auto set : domain.get_set_list()) {
      upa = upa.union_add(isl::union_pw_aff(f.unbind_params_insert_domain(map.at(set.tuple_id()).tuple)));
    }
    return upa;
  }

  void Visit_(const For *op) final {
    auto loop_var_id = isl::id(set.ctx(), op->loop_var->name_hint);
    auto space = set.get_space().add_param(loop_var_id);

    auto loop_var = isl::aff::param_on_domain(space, loop_var_id);

    // Add lower/upper loop bound constraints.
    AddLoopBoundConstraints(loop_var, space, op->min, false, true);
    Expr max = Simplify_cce(op->min + op->extent - 1);
    AddLoopBoundConstraints(loop_var, space, max, true, false);

    auto outer_add = outer.add(loop_var_id);
    auto outer_list = macro_stmt >= 0 ? outer : outer_add;
    auto body_schedule = MakeScheduleTreeHelper(op->body, scop_info_, set, outer_list, macro_stmt);

    auto multi_union_pw_aff_func = isl::multi_union_pw_aff(
      GetUnionPwAffAtDomain(isl::aff::param_on_domain(space, loop_var_id), body_schedule.get_domain(),
                            scop_info_.analysis_result_.GetOperatorDomainMap()));

    sch = body_schedule.insert_partial_schedule(multi_union_pw_aff_func);
    found = true;
  }

  void Visit_(const Realize *op) final {
    auto name = op->func->func_name();
    CHECK_EQ(op->func->num_outputs(), 1);
    auto type = op->type;

    Array<Expr> shapes;
    for (auto i : op->bounds) {
      shapes.push_back(i->extent);
    }
    Tensor tensor = placeholder(shapes, type, name);
    const Buffer buffer = decl_buffer(shapes, type, name);

    scop_info_.user_config_.RecordRealizeTensors(tensor);
    IRVisitor::Visit_(op);

    /// add old realize
    scop_info_.user_config_.InsertRealizeFromInput(isl::id(scop_info_.GetCtx(), name));

    auto binds = scop_info_.user_config_.GetBind();
    for (auto i : binds) {
      if (i.first->op->name == name) return;
    }

    // add Realize's buf into binds
    scop_info_.user_config_.SetBind(tensor, buffer);
  }

  void Visit_(const ProducerConsumer *op) final {
    sch = MakeScheduleTreeHelper(op->body, scop_info_, set, outer, macro_stmt);
    found = true;
  }

  void Op_buffer_bind_scope(const AttrStmt *op) {
    /* ******************************************
     * parse attr like below
     * // attr [[buffer(Abuf, 0x29ff3b0), Tensor(shape=[1, 32, 7, 7, 16], op.name=fmap)]] buffer_bind_scope =
     * tvm_tuple(0, 1, floordiv(k, 9), 1, ((j*16)/5), 3, 0, 7, 0, 16):handle:I
     * *******************************************/
    Array<NodeRef> array = Downcast<Array<NodeRef>>(op->node);
    Buffer buffer = Downcast<Buffer>(array[0]);
    Tensor tensor = Downcast<Tensor>(array[1]);
    Array<NodeRef> update_array;
    std::string update_name = tensor->op->name;
    std::string update_scope;
    if (tensor->op.as<PlaceholderOpNode>()) {
      update_name += LOCAL_C1;
      update_scope = DOT_LOCAL_C1;
    } else {
      update_name += LOCAL_BUF;
      update_scope = DOT_LOCAL_BUF;
    }
    Buffer update_buffer =
      BufferNode::make(buffer->data, buffer->dtype, buffer->shape, buffer->strides, buffer->elem_offset, buffer->name,
                       update_scope, buffer->data_alignment, buffer->offset_factor, buffer->buffer_type);
    Tensor update_tensor = placeholder(tensor->shape, tensor->dtype, update_name);
    update_array.push_back(update_buffer);
    update_array.push_back(update_tensor);
    scop_info_.analysis_result_.RecordUpdateTensor(update_tensor);
    scop_info_.analysis_result_.RecordBufferBindVec(std::make_pair(update_array, op->value));
    scop_info_.analysis_result_.RecordAccess(op, isl::id(set.ctx(), tensor->op->name));
  }

  void SetTensorOfTensorInfo(const AttrStmt *op) {
    if (op->attr_key == AKG_ATOMIC_TOT) {
      size_t stmt_index = scop_info_.analysis_result_.GetStatementMap().size();
      isl::id id(set.ctx(), macro_stmt >= 0 ? kStatementLabel + std::to_string(macro_stmt)
                                            : kStatementLabel + std::to_string(stmt_index));
      CHECK(op->value.as<StringImm>());
      scop_info_.analysis_result_.RecordTensorOfTensorStmt(id.get_name(), op->value.as<StringImm>()->value);
    } else if (op->attr_key == AKG_TENSOR_OF_TENSOR) {
      scop_info_.analysis_result_.SetTensorOfTensor(true);
    } else if (op->attr_key == AKG_TENSOR_NOT_PROMOTE) {
      CHECK(op->value.as<StringImm>());
      scop_info_.analysis_result_.RecordTensorsNotPromote(op->value.as<StringImm>()->value);
    } else if (op->attr_key == AKG_INNER_TENSOR) {
      CHECK(op->value.as<StringImm>());
      scop_info_.analysis_result_.RecordInnerTensor(op->value.as<StringImm>()->value);
    }
  }

  void Visit_(const AttrStmt *op) final {
    if (AkgSupportedTotOp.count(op->attr_key) != 0) {
      SetTensorOfTensorInfo(op);
    } else if (op->attr_key == air::ir::attr::reduce_update) {
      Array<IterVar> red = Downcast<Array<IterVar>>(op->node);
      const auto pro = op->body.as<Provide>();
      if (pro) {
        scop_info_.analysis_result_.RecordReduce(pro, red);
      } else {
        auto blo = op->body.as<Block>();
        if (blo) {
          while (blo->rest.defined() && blo->rest.as<Block>()) {
            blo = blo->rest.as<Block>();
          }
          const auto pro_first = blo->first.as<Provide>();
          const auto pro_rest = blo->rest.as<Provide>();
          if (pro_rest) {
            scop_info_.analysis_result_.RecordReduce(pro_rest, red);
          } else if (pro_first) {
            scop_info_.analysis_result_.RecordReduce(pro_first, red);
          }
        }
      }
    } else if (op->attr_key == air::ir::attr::buffer_bind_scope) {
      Op_buffer_bind_scope(op);
    } else if (op->attr_key == ATTR_IM2COL_KEY) {
      scop_info_.analysis_result_.RecordAttrStmt(op);
    }

    sch = MakeScheduleTreeHelper(op->body, scop_info_, set, outer, macro_stmt);
    found = true;
  }

  isl::schedule sch;
  bool found{false};

 private:
  const NodeRef s;
  ScopInfo &scop_info_;
  isl::set set;
  isl::id_list outer;
  ssize_t macro_stmt{-1};
};

isl::schedule MakeScheduleTreeHelper(const NodeRef &s, ScopInfo &scop_info, const isl::set &set,
                                     const isl::id_list &outer, ssize_t macro_stmt) {
  ScopMakeScheduleTree schedule_tree(s, scop_info, set, outer, macro_stmt);
  if (!schedule_tree.found) {
    LOG(FATAL) << "Unhandled " << s.get()->GetTypeKey() << " : " << s;
  }
  return schedule_tree.sch;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
