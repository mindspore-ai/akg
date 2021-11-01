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
  ReplaceCsrCall(node, stmt);
  return stmt;
}

Stmt GpuIslEmitterCsr::EmitAccessNodeFromPromoteAcsCall(isl::id var, const Node *node, Array<Expr> &args) {
  auto stmt = GpuIslEmitter::EmitAccessNodeFromPromoteAcsCall(var, node, args);
  ReplaceCsrCall(node, stmt);
  return stmt;
}

class RemoveCsrCond : public IRMutator {
  Expr Mutate_(const Variable *op, const Expr &e) final {
    for (const auto &pair: g_csr) {
      auto csr_var = pair.first.as<Variable>();
      if (csr_var != nullptr && op->name_hint == csr_var->name_hint) {
        return Expr();
      }
    }
    return e;
  }

  Expr Mutate_(const And *op, const Expr &e) final { return MutateLogicOp(op, e); }
  Expr Mutate_(const Or *op, const Expr &e) final { return MutateLogicOp(op, e); }
  Expr Mutate_(const EQ *op, const Expr &e) final { return MutateCmpOp(op, e); }
  Expr Mutate_(const NE *op, const Expr &e) final { return MutateCmpOp(op, e); }
  Expr Mutate_(const GE *op, const Expr &e) final { return MutateCmpOp(op, e); }
  Expr Mutate_(const GT *op, const Expr &e) final { return MutateCmpOp(op, e); }
  Expr Mutate_(const LE *op, const Expr &e) final { return MutateCmpOp(op, e); }
  Expr Mutate_(const LT *op, const Expr &e) final { return MutateCmpOp(op, e); }

  template<typename T>
  Expr MutateLogicOp(const T *op, const Expr &e) {
    auto a = IRMutator::Mutate(op->a);
    auto b = IRMutator::Mutate(op->b);
    if (!a.defined()) return b;
    if (!b.defined()) return a;
    if (!a.same_as(op->a) || !b.same_as(op->b)) {
      return T::make(a, b);
    }
    return e;
  }

  template<typename T>
  Expr MutateCmpOp(const T *op, const Expr &e) {
    auto a = IRMutator::Mutate(op->a);
    auto b = IRMutator::Mutate(op->b);
    if (!a.defined() || !b.defined()) {
      return Expr();
    }
    return e;
  }
};

Stmt GpuIslEmitterCsr::EmitIf(const isl::ast_node_if &node) {
  Expr cond_expr = Interpret(node.get_cond());
  Expr new_cond_expr = RemoveCsrCond().Mutate(cond_expr);
  if (cond_expr.defined() && !new_cond_expr.defined()) {
    return EmitAst(node.get_then_node());
  }
  Stmt s = GpuIslEmitter::EmitIf(node);
  if (new_cond_expr.defined() && !new_cond_expr.same_as(cond_expr)) {
    auto if_stmt = s.as<IfThenElse>();
    if (if_stmt != nullptr) {
      return IfThenElse::make(new_cond_expr, if_stmt->then_case, if_stmt->else_case);
    }
  }
  return s;
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