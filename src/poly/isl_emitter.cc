/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "poly/isl_emitter.h"

#include <tvm/expr.h>
#include <tvm/ir.h>
#include <tvm/node/node.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm/operation.h>

#include <algorithm>
#include <utility>

#include "ir_pass.h"
#include "poly/scop_builder.h"
#include "poly/dma_inject.h"
#include "poly/isl.h"
#include "pass/utils.h"

using std::placeholders::_1;
using std::placeholders::_2;
using std::placeholders::_3;
using std::placeholders::_4;

namespace akg {
namespace ir {
namespace poly {
bool AOutThanB(std::vector<const Node *> a, std::vector<const Node *> b) {
  if (a.size() > b.size()) {
    return false;
  }
  for (unsigned int i = 0; i < a.size(); i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

/*
 * Divide two integers, rounding towards zero. This is the typical behavior of most hardware architectures, which
 * differs from Halide's division operator, which is Euclidean (rounds towards -infinity).
 */
inline Expr DivRoundToZero(const Expr &x, const Expr &y) {
  CHECK_NE(GetIntConst(y), 0);
  return x / y;
}

int ToSInt(const isl::val &v) {
  CHECK(v.is_int());
  static_assert(sizeof(long) <= EIGHT_BYTES, "long is assumed to fit into 64bits");
  return v.get_num_si();
}

int IslExprToSInt(const isl::ast_expr &e) {
  auto int_expr = e.as<isl::ast_expr_int>();
  CHECK(int_expr);
  return ToSInt(int_expr.get_val());
}

Expr IslEmitter::InterpretBinaryOp(const isl::ast_expr_op &e) {
  auto left = Interpret(e.get_arg(0));
  auto right = Interpret(e.get_arg(1));
  if (e.as<isl::ast_expr_op_add>()) {
    return left + right;
  } else if (e.as<isl::ast_expr_op_sub>()) {
    return left - right;
  } else if (e.as<isl::ast_expr_op_mul>()) {
    return left * right;
  } else if (e.as<isl::ast_expr_op_div>()) {
    return DivRoundToZero(left, right);
  } else if (e.as<isl::ast_expr_op_fdiv_q>()) {
    return left / right;
  } else if (e.as<isl::ast_expr_op_pdiv_q>()) {
    return left / right;
  } else if (e.as<isl::ast_expr_op_pdiv_r>()) {
    return left % right;
  } else if (e.as<isl::ast_expr_op_zdiv_r>()) {
    return left % right;
  } else if (e.as<isl::ast_expr_op_max>()) {
    return max(left, right);
  } else if (e.as<isl::ast_expr_op_min>()) {
    return min(left, right);
  } else if (e.as<isl::ast_expr_op_gt>()) {
    return left > right;
  } else if (e.as<isl::ast_expr_op_ge>()) {
    return left >= right;
  } else if (e.as<isl::ast_expr_op_lt>()) {
    return left < right;
  } else if (e.as<isl::ast_expr_op_le>()) {
    return left <= right;
  } else if (e.as<isl::ast_expr_op_eq>()) {
    return left == right;
  } else if (e.as<isl::ast_expr_op_and>()) {
    return left && right;
  } else if (e.as<isl::ast_expr_op_or>()) {
    return left || right;
  } else {
    CHECK(false) << "NYI: " << e;
    return 0;
  }
}

Expr IslEmitter::InterpretUnaryOp(const isl::ast_expr_op &e) {
  auto val = Interpret(e.get_arg(0));
  if (e.as<isl::ast_expr_op_minus>()) {
    return -val;
  } else {
    CHECK(false) << "NYI";
    return 0;
  }
}

Expr IslEmitter::InterpretMultiargsOp(const isl::ast_expr_op &e) {
  if (e.as<isl::ast_expr_op_max>()) {
    Expr left = Interpret(e.get_arg(0));
    for (unsigned int i = 1; i < e.get_n_arg(); ++i) {
      auto right = Interpret(e.get_arg(i));
      left = max(left, right);
    }
    return left;
  }
  if (e.as<isl::ast_expr_op_min>()) {
    Expr left = Interpret(e.get_arg(0));
    for (unsigned int i = 1; i < e.get_n_arg(); ++i) {
      auto right = Interpret(e.get_arg(i));
      left = min(left, right);
    }
    return left;
  }
  if (e.as<isl::ast_expr_op_select>()) {
    return if_then_else(Interpret(e.get_arg(0)), Interpret(e.get_arg(1)), Interpret(e.get_arg(2)));
  }
  CHECK(false) << "NYI: " << e;
  return 0;
}

Expr IslEmitter::InterpretOp(const isl::ast_expr_op &e) {
  switch (e.get_n_arg()) {
    case ONE_ARGS:
      return InterpretUnaryOp(e);
    case TWO_ARGS:
      return InterpretBinaryOp(e);
    default:
      return InterpretMultiargsOp(e);
  }
}

Expr IslEmitter::Interpret(const isl::ast_expr &e) {
  if (auto int_expr = e.as<isl::ast_expr_int>()) {
    return Expr(IslExprToSInt(int_expr));
  } else if (auto id_expr = e.as<isl::ast_expr_id>()) {
    // If this variable is defined by loop index, we need sharing it.
    const Variable *var = GetIterByName(id_expr.get_id().get_name());
    if (var)
      return VarExpr(GetObjPtr(var));
    else
      return VarExpr(id_expr.get_id().to_str());
  } else if (auto op_expr = e.as<isl::ast_expr_op>()) {
    return InterpretOp(op_expr);
  } else {
    LOG(FATAL) << "NYI " << e;
    return 0;
  }
}

Stmt IslEmitter::EmitFor(const isl::ast_node_for &node) {
  isl::id isl_iter_id = node.get_iterator().as<isl::ast_expr_id>().get_id();
  VarExpr iter_expr(isl_iter_id.to_str());
  PushIter(iter_expr.get());

  Expr init_expr = Interpret(node.get_init());

  auto isl_cond = node.get_cond().as<isl::ast_expr_op>();
  CHECK(isl_cond.as<isl::ast_expr_op_lt>() || isl_cond.as<isl::ast_expr_op_le>());
  auto cond_lhs = isl_cond.get_arg(0).as<isl::ast_expr_id>();
  CHECK(cond_lhs);
  CHECK_EQ(cond_lhs.get_id(), isl_iter_id);
  Expr cond_expr = Interpret(isl_cond.get_arg(1)) - init_expr;
  if (isl_cond.as<isl::ast_expr_op_le>()) {
    cond_expr = Simplify_cce(cond_expr + 1);
  }

  int64_t inc = static_cast<int64_t>(WrappedStrtol(node.get_inc().to_C_str()));
  CHECK_EQ(inc, 1) << "We guarantee stride=1 by making scale=false in poly.";

  Stmt body_stmt = EmitAst(node.get_body());
  PopIter(iter_expr.get());
  return For::make(iter_expr, init_expr, cond_expr, ForType::Serial, DeviceAPI::None, body_stmt);
}

Stmt IslEmitter::EmitIf(const isl::ast_node_if &node) {
  Expr cond_expr = Interpret(node.get_cond());
  cur_if_list_.push_back(cond_expr.get());
  Stmt then_case = EmitAst(node.get_then_node());
  Stmt else_case;
  if (node.has_else_node()) {
    else_case = EmitAst(node.get_else_node());
  }
  cur_if_list_.pop_back();
  return IfThenElse::make(cond_expr, then_case, else_case);
}

Stmt IslEmitter::EmitMark(const isl::ast_node_mark &node) { return EmitAst(node.get_node()); }

Stmt IslEmitter::EmitBlock(const isl::ast_node_block &node) {
  std::vector<Stmt> children_stmt;
  for (auto child : node.get_children()) {
    Stmt stmt = EmitAst(child);
    if (stmt.defined()) {
      children_stmt.push_back(stmt);
    }
  }
  if (children_stmt.empty()) {
    return Stmt();
  } else if (children_stmt.size() == 1) {
    return children_stmt[0];
  } else {
    return Block::make(children_stmt);
  }
}

isl::space IslEmitter::GetDomainSpace(const isl::id &node_id) {
  auto dom = isl::union_set(info_.analysis_result_.Domain());
  auto space = isl::space();
  dom.foreach_set([&node_id, &space](const isl::set &s) -> void {
    if (s.get_tuple_id() == node_id) {
      space = s.get_space();
    }
  });
  return space;
}

isl::space IslEmitter::GetSpace(const isl::id &tensor_id, const Array<Expr> &tensor_index, const isl::id &stmt_id) {
  auto domain_space = GetDomainSpace(stmt_id);
  auto tensor_space = domain_space.params().add_named_tuple_id_ui(tensor_id, tensor_index.size());
  auto space = domain_space.map_from_domain_and_range(tensor_space);
  return space;
}

isl::multi_aff IslEmitter::TensorAccessMultAff(isl::id &tensor_id, const Array<Expr> &tensor_index,
                                               const isl::id &node_id) {
  CHECK_NE(tensor_index.size(), 0u);
  isl::pw_multi_aff iter_map = node_info_map_.at(node_id).iterator_map;
  isl::id stmt_id = iter_map.get_tuple_id(isl_dim_out);
  OperatorDomainSpace domain_space = info_.analysis_result_.GetOperatorDomainMap().at(stmt_id);
  isl::multi_aff ma = isl::multi_aff::zero(GetSpace(tensor_id, tensor_index, stmt_id));
  for (size_t i = 0; i < tensor_index.size(); ++i) {
    auto aff = Expr2Aff(domain_space.param_space, tensor_index[i]).unbind_params_insert_domain(domain_space.tuple);
    ma = ma.set_aff(i, aff);
  }
  return ma;
}

class EmitExpr : public air::ir::IRMutator {
 public:
  EmitExpr(const std::function<Stmt(const std::string &, const Node *, const Array<Expr> &, VarMap)> &f, VarMap vm)
      : func(f), var_map(std::move(vm)) {}
  ~EmitExpr() override = default;

  Expr Mutate(Expr e) final {
    // use cache first
    for (auto i : cache_) {
      if (Equal(e, i.first)) {
        return i.second;
      }
    }

    Expr new_e = IRMutator::Mutate(e);
    cache_.Set(e, new_e);
    return new_e;
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->call_type == Call::Halide) {
      Stmt stmt = func(op->name, op, op->args, var_map);
      const auto eval = stmt.as<Evaluate>();
      return eval ? eval->value : Expr("error");
    } else {
      return air::ir::IRMutator::Mutate_(op, e);
    }
  }

  // replace var if need
  Expr Mutate_(const Variable *op, const Expr &e) final {
    for (auto &i : var_map) {
      if (op->name_hint == i.first.get_name()) {
        return i.second;
      }
    }
    return e;
  }

  Expr Mutate_(const Select *op, const Expr &e) final {
    Expr cond = this->Mutate(op->condition);
    Expr f = this->Mutate(op->false_value);
    Expr t = this->Mutate(op->true_value);
    if (cond.same_as(op->condition) && t.same_as(op->true_value) && f.same_as(op->false_value)) {
      return e;
    } else {
      return Select::make(cond, t, f);
    }
  }
  const std::function<Stmt(std::string, const Node *, const Array<Expr> &, VarMap)> func;
  VarMap var_map;

 private:
  Map<Expr, Expr> cache_;
};

BufferedFootPrintInfo FindBufferFootprintById(const std::vector<BufferedFootPrintInfo> &active_buf_footprints,
                                              const isl::id &fp_id) {
  BufferedFootPrintInfo buffer_footprint_info;
  for (const auto &act_buf_fp : active_buf_footprints) {
    if (act_buf_fp.cluster != nullptr) {
      for (const auto &fp : act_buf_fp.cluster->tensor_foot_prints) {
        if (fp->id == fp_id) {
          buffer_footprint_info = act_buf_fp;
          break;
        }
      }
    }
  }
  return buffer_footprint_info;
}

bool IslEmitter::IsTransferStmt() {
  if (info_.analysis_result_.GetIsTiled()) {
    isl::union_set transfer_stmt = info_.analysis_result_.GetTransferStmt();
    if (!transfer_stmt.is_empty()) {
      bool name_match = false;
      auto stmt_id = stmt_id_;
      transfer_stmt.foreach_set([&name_match, &stmt_id](const isl::set &s) -> void {
        if (s.get_tuple_name() == stmt_id.get_name()) {
          name_match = true;
        }
      });
      if (name_match) return true;
    }
  }
  return false;
}

Stmt IslEmitter::EmitAccessNodeProvide(const Node *node, const VarMap &var_map_tmp,
                                       BufferedFootPrintInfo &buffer_footprint_info) {
  const auto provide = static_cast<const Provide *>(node);
  Expr value = ReplaceLoopVar(var_map_tmp).Mutate(provide->value);
  Array<Expr> args;
  for (auto iv : provide->args) {
    args.push_back(ReplaceLoopVar(var_map_tmp).Mutate(iv));
  }
  // Not hoisted, emitting just the mapped subscript.
  if (!buffer_footprint_info.cluster_id) {
    return Provide::make(provide->func, provide->value_index, value, args);
  }
  return Stmt();
}

Stmt IslEmitter::EmitAccessNodeCall(const Node *node, const VarMap &var_map_tmp,
                                    BufferedFootPrintInfo &buffer_footprint_info) {
  const Call *call = static_cast<const Call *>(node);
  Array<Expr> args;
  for (auto iv : call->args) {
    args.push_back(ReplaceLoopVar(var_map_tmp).Mutate(iv));
  }
  // Not hoisted, emitting just the mapped subscript.
  if (!buffer_footprint_info.cluster_id) {
    return Evaluate::make(Call::make(call->type, call->name, args, call->call_type, call->func, call->value_index));
  }
  return Stmt();
}

bool IslEmitter::IsCopyinFromAnotherBand(isl::multi_aff &access) {
  for (isl::map inter_band_dependency : info_.analysis_result_.GetInnerBandDependency().get_map_list()) {
    if (inter_band_dependency.get_tuple_id(isl_dim_out) == access.get_tuple_id(isl_dim_out)) {
      return true;
    }
  }
  return false;
}

isl::pw_multi_aff &AffSubForAstToSchedule(isl::pw_multi_aff &ast_to_schedule, bool is_transfer_stmt,
                                          bool is_copyin_from_another_band) {
  if (is_transfer_stmt || is_copyin_from_another_band) {
    isl_pw_multi_aff *pma1 = ast_to_schedule.copy();
    isl_pw_multi_aff *pma2 = ast_to_schedule.copy();
    isl_pw_multi_aff *pma = isl_pw_multi_aff_sub(pma1, pma2);
    ast_to_schedule = isl::manage(pma);
  }
  return ast_to_schedule;
}

Stmt IslEmitter::EmitAccessNodeFromPromoteAcsProvide(isl::id var, const Node *node, Array<Expr> &args) {
  const auto provide = static_cast<const Provide *>(node);
  Tensor t = info_.FindTensor(var);
  if (info_.analysis_result_.CountBufferDefInfo(var)) {
    realize_may_def_.insert(var);
    if_map_[var] = cur_if_list_;
    if (cur_if_list_.empty()) {
      realize_must_def_.insert(var);
    }
  }
  Stmt s = Provide::make(t->op, 0, provide->value, args);
  return s;
}

Stmt IslEmitter::EmitAccessNodeFromPromoteAcsCall(isl::id var, const Node *node, Array<Expr> &args) {
  const Call *call = static_cast<const Call *>(node);
  Tensor t = info_.FindTensor(var);
  if (info_.analysis_result_.CountBufferDefInfo(var)) {
    realize_use_.insert(var);
    if (!if_map_.count(var) || !AOutThanB(if_map_.at(var), cur_if_list_)) {
      realize_use_with_may_def_.insert(var);
    }
  }
  return Evaluate::make(Call::make(call->type, var.get_name(), args, call->call_type, t->op, t->value_index));
}

Stmt IslEmitter::EmitAccessNode(const std::string &name, const Node *node, const Array<Expr> &tensor_index,
                                const VarMap &var_map_tmp) {
  // Scalars are not hoisted or remapped.
  if (tensor_index.empty()) {
    LOG(FATAL) << " Scalar " << name << " not in any buffers";
    return Evaluate::make(Expr("error"));
  }

  auto build = node_info_map_.at(node_id_).build;
  auto iterator_map = node_info_map_.at(node_id_).iterator_map;

  CHECK_EQ(info_.analysis_result_.GetAccessMap().count(node), 1u)
    << "generating tensor " << name << " not in Scop" << node << " not allowed ";
  auto fp_id = info_.analysis_result_.GetAccessMap().at(node);

  std::vector<BufferedFootPrintInfo> active_buf_footprint;
  for (const auto &kv : info_.analysis_result_.ActiveBufferFootprints()) {
    if (kv.first.intersect(isl::union_set(Domain())).is_empty()) {
      continue;
    }
    active_buf_footprint.emplace_back(kv.second);
  }
  BufferedFootPrintInfo buffer_footprint_info = FindBufferFootprintById(active_buf_footprint, fp_id);

  if (node->IsInstance<Provide>()) {
    auto stmt = EmitAccessNodeProvide(node, var_map_tmp, buffer_footprint_info);
    if (stmt.defined()) return stmt;
  }

  if (node->IsInstance<Call>()) {
    auto stmt = EmitAccessNodeCall(node, var_map_tmp, buffer_footprint_info);
    if (stmt.defined()) return stmt;
  }

  auto buf_def = info_.analysis_result_.GetBufferDefInfo(buffer_footprint_info.cluster_id);

  auto access = TensorAccessMultAff(buf_def.tensor_id, tensor_index, node_id_);

  bool is_copyin_from_another_band = IsCopyinFromAnotherBand(access);

  auto memory_hoist = buffer_footprint_info.cluster->ComputeBufferedFootprints();
  if (is_copyin_from_another_band) {
    memory_hoist = buffer_footprint_info.cluster->IdentityBufferFootprint();
  }

  // split read-only or write-only input tensor memory_hoists
  // we need to find tensor by name because tensor_id is a fake isl::id
  bool is_input_tensor = info_.FindTensorInOrig(buf_def.tensor_id.name()).defined();
  if (is_input_tensor && buffer_footprint_info.cluster->foot_print_.should_split) {
    memory_hoist = buffer_footprint_info.cluster->UnshiftedBufferFootprint(memory_hoist, fp_id);
  }
  memory_hoist = memory_hoist.set_tuple_id(isl_dim_out, buffer_footprint_info.cluster_id);

  auto schedule = isl::map::from(buffer_footprint_info.outer_schedule.intersect_domain(Domain()));
  CHECK(schedule.is_single_valued()) << schedule << " is not single-valued schedule";
  auto ast_to_schedule = isl::pw_multi_aff(schedule).pullback(iterator_map);
  ast_to_schedule = AffSubForAstToSchedule(ast_to_schedule, IsTransferStmt(), is_copyin_from_another_band);

  auto ast_to_original = isl::pw_multi_aff(access).pullback(iterator_map);
  auto ast_to_scheduled_original = ast_to_schedule.range_product(ast_to_original);
  auto ast_to_hoisted = isl::pw_multi_aff(memory_hoist).pullback(ast_to_scheduled_original);
  auto hoist_acs = build.access_from(ast_to_hoisted);
  if (auto op = hoist_acs.as<isl::ast_expr_op>()) {
    if (op.as<isl::ast_expr_op_access>()) {
      Array<Expr> args;
      for (int i = 1; i < static_cast<int>(op.get_n_arg()); ++i) {
        args.push_back(Interpret(op.get_arg(i)));
      }
      if (node->IsInstance<Provide>())
        return EmitAccessNodeFromPromoteAcsProvide(op.get_arg(0).as<isl::ast_expr_id>().get_id(), node, args);
      if (node->IsInstance<Call>())
        return EmitAccessNodeFromPromoteAcsCall(op.get_arg(0).as<isl::ast_expr_id>().get_id(), node, args);
    }
  }
  return Evaluate::make(Expr("todo EmitAst"));
}

Stmt IslEmitter::EmitUserStmtContent(const For *for_node) {
  Stmt body = EmitUserStmtContent(for_node->body.get());
  return For::make(for_node->loop_var, for_node->min, for_node->extent, for_node->for_type, DeviceAPI::None, body);
}

Stmt IslEmitter::EmitUserStmtContent(const Evaluate *eva_node) {
  const Call *call = eva_node->value.as<Call>();
  Array<Expr> args;
  for (auto iv : call->args) {
    args.push_back(ReplaceLoopVar(var_map_).Mutate(iv));
  }
  auto im2col = Call::make(call->type, call->name, args, call->call_type);
  Stmt res = Evaluate::make(im2col);
  // add AttrStmt to im2col
  for (const auto &item : info_.analysis_result_.GetBufferBindVec()) {
    Expr replaced = ReplaceLoopVar(var_map_).Mutate(item.second);
    res = AttrStmt::make(item.first, air::ir::attr::buffer_bind_scope, replaced, res);
  }
  return res;
}

class SubstituteByNameMutator : public IRMutator {
 public:
  explicit SubstituteByNameMutator(const VarMap &var_map) {
    for (const auto &pair : var_map) {
      var_map_[pair.first.name()] = pair.second;
    }
  }
  ~SubstituteByNameMutator() override = default;

  Expr Mutate_(const Variable *op, const Expr &e) override {
    if (var_map_.count(op->name_hint)) return var_map_[op->name_hint];
    return e;
  }

 private:
  std::unordered_map<std::string, Expr> var_map_;
};

/*
 * For conditional write tensors, we cannot generate copy out memory_hoist as usual
 * because some elements of the tensor may be undefined.
 * So, we need to sink the copy out statement into the innermost "if",
 * i.e., copy out immediately after each computation.
 */
static Stmt GenerateCopyOut(const ScopInfo &info, const Provide *original, const Provide *hoisted,
                            const VarMap &var_map) {
  auto call_type = info.GetDtypeOf(hoisted->func->func_name());
  Expr call_expr = Call::make(call_type, hoisted->func->func_name(), hoisted->args, Call::CallType::Halide,
                              hoisted->func, hoisted->value_index);
  Array<Expr> new_args;
  for (auto arg : original->args) {
    new_args.push_back(SubstituteByNameMutator(var_map).Mutate(arg));
  }
  return Provide::make(original->func, original->value_index, call_expr, new_args);
}

Stmt IslEmitter::EmitUserStmtContent(const Provide *provide_node) {
  std::string write_tensor = provide_node->func->func_name();
  Stmt op = EmitAccessNode(write_tensor, provide_node, provide_node->args, var_map_);
  const auto provide_new = op.as<Provide>();
  CHECK(provide_new);
  const std::function<Stmt(std::string, const Node *, const Array<Expr> &, VarMap)> f =
    std::bind(&IslEmitter::EmitAccessNode, this, _1, _2, _3, _4);
  Expr value = EmitExpr(f, var_map_).Mutate(provide_node->value);
  Stmt provide_stmt = Provide::make(provide_new->func, provide_new->value_index, value, provide_new->args);

  if (info_.analysis_result_.GetConditionalWriteBufferFootprints().count(write_tensor)) {
    return Block::make(provide_stmt, GenerateCopyOut(info_, provide_node, provide_new, var_map_));
  }
  return provide_stmt;
}

Stmt IslEmitter::EmitUserStmtContent(const IfThenElse *if_node) {
  Stmt then_stmt, else_stmt;
  if (if_node->then_case.defined()) {
    then_stmt = EmitUserStmtContent(if_node->then_case.get());
  }

  if (if_node->else_case.defined()) {
    else_stmt = EmitUserStmtContent(if_node->else_case.get());
  }

  const std::function<Stmt(std::string, const Node *, const Array<Expr> &, VarMap)> f =
    std::bind(&IslEmitter::EmitAccessNode, this, _1, _2, _3, _4);
  Expr cond = EmitExpr(f, var_map_).Mutate(if_node->condition);
  Stmt stmt = IfThenElse::make(cond, then_stmt, else_stmt);
  return stmt;
}

Stmt IslEmitter::EmitUserStmtContent(const Block *block_node) {
  Stmt first_stmt, rest_stmt;
  if (block_node->first.defined()) {
    first_stmt = EmitUserStmtContent(block_node->first.get());
  }

  if (block_node->rest.defined()) {
    rest_stmt = EmitUserStmtContent(block_node->rest.get());
  }

  Stmt stmt = Block::make(first_stmt, rest_stmt);
  return stmt;
}

Stmt IslEmitter::EmitUserStmtContent(const Node *node) {
  if (node->IsInstance<Provide>()) {
    const auto op = static_cast<const Provide *>(node);
    return EmitUserStmtContent(op);
  } else if (node->IsInstance<IfThenElse>()) {
    const auto op = static_cast<const IfThenElse *>(node);
    return EmitUserStmtContent(op);
  } else if (node->IsInstance<For>()) {
    LOG(WARNING) << "found For in isl::ast_node_user";
    const auto op = static_cast<const For *>(node);
    return EmitUserStmtContent(op);
  } else if (node->IsInstance<Block>()) {
    LOG(WARNING) << "found Block in isl::ast_node_user";
    const auto op = static_cast<const Block *>(node);
    return EmitUserStmtContent(op);
  } else if (node->IsInstance<Evaluate>()) {
    LOG(WARNING) << "found Evaluate in isl::ast_node_user";
    const auto op = static_cast<const Evaluate *>(node);
    return EmitUserStmtContent(op);
  } else {
    CHECK(false) << "unknown node type in isl::ast_node_user: " << node << " " << node->_type_key;
    return Stmt();
  }
}

Stmt IslEmitter::EmitUserStmt(const isl::ast_node_user &node) {
  CHECK(node.get_expr().isa<isl::ast_expr_op>());
  isl::ast_expr_op usr_expr = node.get_expr().as<isl::ast_expr_op>();
  stmt_id_ = usr_expr.get_arg(0).as<isl::ast_expr_id>().get_id();
  node_id_ = node.get_annotation();
  const Node *stmt_node = info_.analysis_result_.GetStatementMap().at(stmt_id_);
  CHECK(stmt_node);
  // compute VarMap to replace old iterators
  auto build = node_info_map_.at(node_id_).build;
  auto tuple = info_.analysis_result_.GetOperatorDomainMap().at(stmt_id_).tuple;
  auto iterator_map = node_info_map_.at(node_id_).iterator_map;

  var_map_.clear();
  for (unsigned int i = 0; i < tuple.size(); ++i) {
    isl::id isl_old_iter = tuple.get_id(i);
    auto isl_expr = build.expr_from(iterator_map.get_pw_aff(i));
    Expr halide_new_iter = Interpret(isl_expr);
    var_map_.emplace(isl_old_iter, halide_new_iter);
  }

  return EmitUserStmtContent(stmt_node);
}

Stmt IslEmitter::EmitStmt(const isl::ast_node_user &node) {
  CHECK(node.get_expr().isa<isl::ast_expr_op>());
  isl::ast_expr_op usr_expr = node.get_expr().as<isl::ast_expr_op>();
  CHECK(usr_expr);
  auto stmt_id = usr_expr.get_arg(0).as<isl::ast_expr_id>().get_id();
  if (info_.IsRead(stmt_id)) {
    return Evaluate::make(Expr("todo EmitRead"));
  }
  if (info_.IsWrite(stmt_id)) {
    return Evaluate::make(Expr("todo EmitWrite"));
  }
  return EmitUserStmt(node);
}

Stmt IslEmitter::EmitAst(const isl::ast_node &node) {
  Stmt s;
  std::string info;
  if (auto for_node = node.as<isl::ast_node_for>()) {
    info = "[FOR_NODE]";
    s = EmitFor(for_node);
  } else if (auto if_node = node.as<isl::ast_node_if>()) {
    info = "[IF_NODE]";
    s = EmitIf(if_node);
  } else if (auto block_node = node.as<isl::ast_node_block>()) {
    info = "[BLOCK_NODE]";
    s = EmitBlock(block_node);
  } else if (auto mark_node = node.as<isl::ast_node_mark>()) {
    info = "[MARK_NODE]";
    s = EmitMark(mark_node);
  } else if (auto user_node = node.as<isl::ast_node_user>()) {
    info = "[USER_NODE]";
    s = EmitStmt(user_node);
  } else {
    s = Evaluate::make(Expr("todo EmitAst"));
  }
  if (PRINT_EMITTER) {
    LOG(INFO) << ">>>>>>>>>>>>INPUT AST_NODE" << info << "<<<<<<<<<<<<<<\n" << node;
    LOG(INFO) << ">>>>>>>>>>>>OUTPUT STMT<<<<<<<<<<<<\n" << s;
  }
  return s;
}

Stmt IslEmitter::Emit(const isl::ast_node &node) { return EmitAst(node); }

void IslEmitter::PushIter(const Variable *iter) { iters_.push_back(iter); }

void IslEmitter::PopIter(const Variable *iter) {
  CHECK_EQ(iters_.back(), iter);
  iters_.pop_back();
}

bool IslEmitter::FindIter(const Variable *iter) const {
  return std::find(iters_.begin(), iters_.end(), iter) != iters_.end();
}

const Variable *IslEmitter::GetIterByName(const std::string &id) const {
  for (auto var : iters_) {
    if (var->name_hint == id) {
      return var;
    }
  }
  return nullptr;
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
