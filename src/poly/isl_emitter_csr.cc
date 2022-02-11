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

#include "build_module.h"
#include "gpu_emit/emit_pass.h"
#include "isl_emitter_csr.h"

namespace akg {
namespace ir {
namespace poly {
int64_t ToSInt64(const isl::val &v) {
  CHECK(v.is_int());
  static_assert(sizeof(long) <= EIGHT_BYTES, "long is assumed to fit into 64bits");
  return static_cast<int64_t>(v.get_num_si());
}

int64_t IslExprToSInt64(const isl::ast_expr &e) {
  auto int_expr = e.as<isl::ast_expr_int>();
  CHECK(int_expr);
  return ToSInt64(int_expr.get_val());
}

bool CallEqual(const Call *a, const Call *b) {
  if (a == nullptr || b == nullptr) {
    return false;
  }
  if (a->name != b->name || a->call_type != b->call_type || a->value_index != b->value_index || a->func != b->func) {
    return false;
  }
  for (auto e1: a->args) {
    if (std::none_of(b->args.begin(), b->args.end(), [&e1](Expr e2){ return Equal(e1, e2); })) {
      return false;
    }
  }
  return true;
}

void ReplaceCsrCall(const Node *node, const Stmt &s, int max_var_id = 0) {
  if (auto eval = s.as<Evaluate>()) {
    auto call = static_cast<const Call *>(node);
    auto replaced = eval->value;
    for (const auto &pair: g_csr) {
      auto origin = pair.first.as<Call>();
      if (origin != nullptr && CallEqual(call, origin) && max_var_id >= 0) {
        auto replace_arr = Downcast<Array<Expr>>(pair.second);
        int replace_size = static_cast<int>(replace_arr.size());
        CHECK(max_var_id <= replace_size);
        if (max_var_id == replace_size) {
          replace_arr.push_back(replaced);
        } else {
          replace_arr.Set(max_var_id, replaced);
        }
        g_csr.Set(pair.first, replace_arr);
      }
    }
  }
}

class CheckCsrCond : public IRVisitor {
 public:
  explicit CheckCsrCond(ScopInfo &info) : info_(info) {}

  bool has_csr_cond_{false};

 private:
  void Visit_(const Variable *op) {
    if (info_.analysis_result_.IsCsrDynamicExtent(op)) {
      has_csr_cond_ = true;
    }
  }

  ScopInfo &info_;
};

bool ContainsCsrCond(const Expr &e, ScopInfo &info) {
  auto check_csr_cond = CheckCsrCond(info);
  check_csr_cond.Visit(e);
  return check_csr_cond.has_csr_cond_;
}

Stmt IslEmitterCsr::EmitAccessNodeCall(
  const Node *node, const VarMap &var_map_tmp, BufferedFootPrintInfo &buffer_fp_info) {
  auto stmt = IslEmitter::EmitAccessNodeCall(node, var_map_tmp, buffer_fp_info);
  ReplaceCsrCall(node, stmt, max_var_id_);
  return stmt;
}

Stmt IslEmitterCsr::EmitAccessNodeFromPromoteAcsCall(isl::id var, const Node *node, Array<Expr> &args) {
  auto stmt = IslEmitter::EmitAccessNodeFromPromoteAcsCall(var, node, args);
  ReplaceCsrCall(node, stmt, max_var_id_);
  return stmt;
}

Expr CpuIslEmitterCsr::Interpret(const isl::ast_expr &e) {
  if (auto int_expr = e.as<isl::ast_expr_int>()) {
    if (dtype_ == Int(64)) return Expr(IslExprToSInt64(int_expr));
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

Stmt CpuIslEmitterCsr::EmitFor(const isl::ast_node_for &node) {
  Expr init_expr = Interpret(node.get_init());
  auto isl_cond = node.get_cond().as<isl::ast_expr_op>();
  CHECK(isl_cond.as<isl::ast_expr_op_lt>() || isl_cond.as<isl::ast_expr_op_le>());
  auto cond_lhs = isl_cond.get_arg(0).as<isl::ast_expr_id>();
  CHECK(cond_lhs);
  Expr cond_expr = Interpret(isl_cond.get_arg(1)) - init_expr;
  if (isl_cond.as<isl::ast_expr_op_le>()) {
    cond_expr = Simplify_cce(cond_expr + 1);
  }
  if (ContainsCsrCond(cond_expr, info_)) {
    ++max_var_id_;
    if (depth_ == 0) {
      unscoped_max_var_id_ = max_var_id_;
    }
  }

  int inc = 0;
  auto imm = cond_expr.as<IntImm>();
  if (imm == nullptr || imm->value != 1) {
    inc = 1;
  }
  int tmp_max_var_id = max_var_id_;
  depth_ += inc;
  Stmt stmt = CpuIslEmitter::EmitFor(node);
  depth_ -= inc;
  if (inc > 0 && max_var_id_ > tmp_max_var_id) {
    stmt = AttrStmt::make(Expr("INFO"), "max_var_id", max_var_id_, stmt);
  }
  return stmt;
}

Stmt CpuIslEmitterCsr::EmitInfo(const Stmt &stmt) {
  Stmt s = CpuIslEmitter::EmitInfo(stmt);
  if (unscoped_max_var_id_ >= 0) {
    s = AttrStmt::make(Expr("INFO"), "max_var_id", unscoped_max_var_id_, s);
  }
  return s;
}

Stmt GpuIslEmitterCsr::EmitAccessNodeCall(
  const Node *node, const VarMap &var_map_tmp, BufferedFootPrintInfo &buffer_fp_info) {
  auto stmt = GpuIslEmitter::EmitAccessNodeCall(node, var_map_tmp, buffer_fp_info);
  if (csr_dynamic_scope_) {
    ReplaceCsrCall(node, stmt);
  }
  return stmt;
}

Stmt GpuIslEmitterCsr::EmitAccessNodeFromPromoteAcsCall(isl::id var, const Node *node, Array<Expr> &args) {
  auto stmt = GpuIslEmitter::EmitAccessNodeFromPromoteAcsCall(var, node, args);
  if (csr_dynamic_scope_) {
    ReplaceCsrCall(node, stmt);
  }
  return stmt;
}

Stmt GpuIslEmitterCsr::EmitFor(const isl::ast_node_for &node) {
  auto isl_cond = node.get_cond().as<isl::ast_expr_op>();
  CHECK(isl_cond.as<isl::ast_expr_op_lt>() || isl_cond.as<isl::ast_expr_op_le>());
  auto cond_lhs = isl_cond.get_arg(0).as<isl::ast_expr_id>();
  CHECK(cond_lhs);
  Expr cond_expr = Interpret(isl_cond.get_arg(1));
  auto check_csr_cond = CheckCsrCond(info_);
  check_csr_cond.Visit(cond_expr);
  bool tmp_csr_dynamic_scope = csr_dynamic_scope_;
  if (ContainsCsrCond(cond_expr, info_)) {
    csr_dynamic_scope_ = true;
  }
  auto stmt = GpuIslEmitter::EmitFor(node);
  csr_dynamic_scope_ = tmp_csr_dynamic_scope;
  return stmt;
}

Stmt GpuIslEmitterCsr::SubstituteTensorStmt(const Stmt &s, Tensor origin, Tensor replaced) {
  for (const auto &pair: g_csr) {
    if (pair.first.as<Call>() != nullptr) {
      auto replace_arr = Downcast<Array<Expr>>(pair.second);
      for (size_t i = 0; i < replace_arr.size(); i ++) {
        auto value = TensorSubstitute(replace_arr[i], origin->op, replaced->op, replaced->value_index);
        value = TensorStringSubstitute(value, replaced->op->func_name(), replaced->op, replaced->value_index);
        replace_arr.Set(i, value);
      }
      g_csr.Set(pair.first, replace_arr);
    }
  }
  return GpuIslEmitter::SubstituteTensorStmt(s, origin, replaced);
}

Stmt GpuIslEmitterCsr::EmitTensorOfTensorStmt(const Stmt &s) {
  Stmt stmt = LowerWith(s);
  if (info_.analysis_result_.GetOpTemplate() == Template::REDUCTION) {
    stmt = AtomicReturnStmtEmit(info_).Mutate(stmt);
  }
  stmt = AttrStmt::make(Expr("INFO"), REDUCE_LIB_TYPE_FLAG, info_.user_config_.GetReduceLibType(), stmt);
  return stmt;
}
}  // namespace poly
}  // namespace ir
}  // namespace akg