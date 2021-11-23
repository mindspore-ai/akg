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
#include "emit_pass.h"
#include "gpu_isl_emitter_csr.h"

namespace akg {
namespace ir {
namespace poly {

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

void ReplaceCsrCall(const Node *node, const Stmt &s) {
  if (auto eval = s.as<Evaluate>()) {
    auto call = static_cast<const Call *>(node);
    auto replaced = eval->value;
    for (const auto &pair: g_csr) {
      auto origin = pair.first.as<Call>();
      if (origin != nullptr && CallEqual(call, origin)) {
        g_csr.Set(pair.first, replaced);
      }
    }
  }
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

Stmt GpuIslEmitterCsr::EmitFor(const isl::ast_node_for &node) {
  auto isl_cond = node.get_cond().as<isl::ast_expr_op>();
  CHECK(isl_cond.as<isl::ast_expr_op_lt>() || isl_cond.as<isl::ast_expr_op_le>());
  auto cond_lhs = isl_cond.get_arg(0).as<isl::ast_expr_id>();
  CHECK(cond_lhs);
  Expr cond_expr = Interpret(isl_cond.get_arg(1));
  auto check_csr_cond = CheckCsrCond(info_);
  check_csr_cond.Visit(cond_expr);
  bool tmp_csr_dynamic_scope = csr_dynamic_scope_;
  if (check_csr_cond.has_csr_cond_) {
    csr_dynamic_scope_ = true;
  }
  auto stmt = GpuIslEmitter::EmitFor(node);
  csr_dynamic_scope_ = tmp_csr_dynamic_scope;
  return stmt;
}

Stmt GpuIslEmitterCsr::SubstituteTensorStmt(const Stmt &s, Tensor origin, Tensor replaced) {
  for (const auto &pair: g_csr) {
    if (pair.first.as<Call>() != nullptr) {
      auto value = TensorSubstitute(air::Downcast<Expr>(pair.second), origin->op, replaced->op, replaced->value_index);
      value = TensorStringSubstitute(value, replaced->op->func_name(), replaced->op, replaced->value_index);
      g_csr.Set(pair.first, value);
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