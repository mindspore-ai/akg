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
#include "poly/poly_util.h"

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

void ReplaceCsrCall(const Node *node, const Stmt &s, int max_var_id, std::string max_var = "") {
  if (max_var_id < 0) {
    return;
  }
  // id 0 is reserved for the original expression
  max_var_id += 1;
  auto eval = s.as<Evaluate>();
  if (eval == nullptr) {
    return;
  }
  auto call = static_cast<const Call *>(node);
  CHECK(call);
  for (const auto &pair: g_csr) {
    auto var = pair.first.as<Variable>();
    CHECK(var);
    if (!max_var.empty() && var->name_hint != max_var) continue;
    auto replace_arr = Downcast<Array<Expr>>(pair.second);
    int replace_size = static_cast<int>(replace_arr.size());
    CHECK(replace_size > 0);
    CHECK(max_var_id <= replace_size);
    auto origin = replace_arr[0].as<Sub>();
    CHECK(origin);
    auto replaced = origin;
    if (max_var_id < replace_size) {
      replaced = replace_arr[max_var_id].as<Sub>();
    }
    auto replace_a = replaced->a;
    auto replace_b = replaced->b;
    if (max_var_id > 0) {
      if (CallEqual(call, origin->a.as<Call>())) {
        replace_a = eval->value;
      } else if (CallEqual(call, origin->b.as<Call>())) {
        replace_b = eval->value;
      }
    }
    if (!replace_a.same_as(replaced->a) || !replace_b.same_as(replaced->b)) {
      auto new_replaced = Sub::make(replace_a, replace_b);
      if (max_var_id == replace_size) {
        replace_arr.push_back(new_replaced);
      } else {
        replace_arr.Set(max_var_id, new_replaced);
      }
      g_csr.Set(pair.first, replace_arr);
    }
  }
}

class CheckCsrCond : public IRVisitor {
 public:
  explicit CheckCsrCond(ScopInfo &info) : info_(info) {}

  bool has_csr_cond_{false};
  std::string max_var_;

 private:
  void Visit_(const Variable *op) {
    if (info_.analysis_result_.IsCsrDynamicExtent(op)) {
      has_csr_cond_ = true;
      max_var_ = op->name_hint;
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
    if (dtype_ == Int(64)) {
      return Expr(IslExprToSInt64(int_expr));
    }
  }
  return IslEmitter::Interpret(e);
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
  bool tmp_csr_dynamic_scope = csr_dynamic_scope_;
  if (ContainsCsrCond(cond_expr, info_) && !csr_dynamic_scope_) {
    csr_dynamic_scope_ = true;
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
  csr_dynamic_scope_ = tmp_csr_dynamic_scope;
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
    ReplaceCsrCall(node, stmt, 0, max_var_);
  }
  return stmt;
}

Stmt GpuIslEmitterCsr::EmitAccessNodeFromPromoteAcsCall(isl::id var, const Node *node, Array<Expr> &args) {
  auto stmt = GpuIslEmitter::EmitAccessNodeFromPromoteAcsCall(var, node, args);
  if (csr_dynamic_scope_) {
    ReplaceCsrCall(node, stmt, 0, max_var_);
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
  if (check_csr_cond.has_csr_cond_) {
    csr_dynamic_scope_ = true;
    max_var_ = check_csr_cond.max_var_;
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

class RemoveAtomic : public IRMutator {
 public:
  explicit RemoveAtomic(ScopInfo &info) : info_(info) {}

 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (IsStartsWith(op->attr_key, ATOMIC_MARKER)) {
      need_check_ = true;
      auto body = IRMutator::Mutate(op->body);
      need_check_ = false;
      if (need_remove_) {
        need_remove_ = false;
        return body;
      }
      if (!body.same_as(op->body)) {
        return AttrStmt::make(op->node, op->attr_key, op->value, body);
      }
    } else {
      auto body = IRMutator::Mutate(op->body);
      auto attr_value = op->value.as<StringImm>();
      if (need_check_ && need_remove_ && attr_value &&
          (AkgSupportedReduceOp.count(attr_value->value) || attr_value->value == AKG_REDUCE_UNSUPPORTED)) {
        return body;
      }
      if (!body.same_as(op->body)) {
        return AttrStmt::make(op->node, op->attr_key, op->value, body);
      }
    }
    return s;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (ContainsCsrCond(op->extent, info_)) {
      csr_vars_.insert(op->loop_var.get());
      auto stmt = IRMutator::Mutate_(op, s);
      csr_vars_.erase(op->loop_var.get());
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (!need_check_) return s;
    std::unordered_set<const Variable *> dst_vars, src_vars;
    bool has_func = false;
    bool is_reduce = false;
    bool reduce_csr = false;
    auto func = op->func;
    CHECK(func.defined());
    for (auto arg : op->args) {
      PostOrderVisit(arg, [&dst_vars] (const NodeRef &n) {
        auto var = n.as<Variable>();
        if (var != nullptr && !IsStartsWith(var->name_hint, "blockIdx") && !IsStartsWith(var->name_hint, "threadIdx")) {
          dst_vars.insert(var);
        }
      });
    }
    PostOrderVisit(op->value, [&func, &has_func] (const NodeRef &n) {
      auto call = n.as<Call>();
      if (call != nullptr && call->func.defined() && call->func->func_name() == func->func_name()) {
        has_func = true;
      }
    });
    if (has_func) {
      PostOrderVisit(op->value, [&src_vars] (const NodeRef &n) {
        auto var = n.as<Variable>();
        if (var != nullptr && !IsStartsWith(var->name_hint, "blockIdx") && !IsStartsWith(var->name_hint, "threadIdx")) {
          src_vars.insert(var);
        }
      });
      for (auto v : src_vars) {
        if (dst_vars.count(v) == 0) {
          is_reduce = true;
          if (csr_vars_.count(v) > 0) {
            // reduce axis is in src vars but not dst vars
            reduce_csr = true;
          }
        }
      }
    }
    if (is_reduce) {
      if (!reduce_csr) {
        need_remove_ = true;
      }
    } else if (!has_func) {
      need_remove_ = true;
    }
    return s;
  }

  ScopInfo &info_;
  std::unordered_set<const Variable *> csr_vars_;
  bool need_check_{false};
  bool need_remove_{false};
};

Stmt GpuIslEmitterCsr::EmitTensorOfTensorStmt(const Stmt &s) {
  Stmt stmt = LowerWith(s);
  stmt = RemoveAtomic(info_).Mutate(stmt);
  if (info_.analysis_result_.GetOpTemplate() == Template::REDUCTION || info_.user_config_.GetEnableAtomicAdd()) {
    stmt = AtomicReturnStmtEmit(info_).Mutate(stmt);
  }
  stmt = AttrStmt::make(Expr("INFO"), REDUCE_LIB_TYPE_FLAG, info_.user_config_.GetReduceLibType(), stmt);
  return stmt;
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
