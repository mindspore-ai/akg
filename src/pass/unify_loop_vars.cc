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
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <ir_pass.h>
#include "pass/utils.h"

namespace akg {
namespace ir {
/*
 * TVM may generate different instances of loop variable with a same name,
 * leading to ambiguity in later passes.
 * This pass unifies the loop variables according to the name in definition,
 * and reports error for undefined variables.
 */

class UnifyLoopVarsMutator : public IRMutator {
 public:
  UnifyLoopVarsMutator(const Map<Tensor, Buffer> &extern_buffer, const Array<NodeRef> &arg_list) {
    std::unordered_set<Var, NodeHash, NodeEqual> vars;
    for (const auto &it : extern_buffer) {
      for (const auto &expr : it.first->shape) {
        GatherVars(expr, &vars);
      }
      for (const auto &expr : it.second->shape) {
        GatherVars(expr, &vars);
      }
      keep_vars_.insert(it.second->data.get());
    }
    for (const auto &arg : arg_list) {
      if (arg.as<Variable>()) {
        vars.insert(ktvm::Downcast<Var>(arg));
      }
    }
    for (const auto &var : vars) {
      define_var(var);
    }
  }
  ~UnifyLoopVarsMutator() override = default;

 private:
  void define_var(const VarExpr &var) {
    if (var_map_.count(var->name_hint) != 0) {
      LOG(FATAL) << "redefinition of variable: " << var;
    }
    var_map_[var->name_hint] = var;
  }

  void undefine_var(const VarExpr &var) {
    CHECK_EQ(var_map_.count(var->name_hint), 1);
    var_map_.erase(var->name_hint);
  }

  Stmt Mutate_(const For *op, const Stmt &s) override {
    define_var(op->loop_var);
    Stmt stmt = IRMutator::Mutate_(op, s);
    undefine_var(op->loop_var);
    return stmt;
  }

  Stmt Mutate_(const LetStmt *op, const Stmt &s) override {
    define_var(op->var);
    Stmt stmt = IRMutator::Mutate_(op, s);
    undefine_var(op->var);
    return stmt;
  }

  Expr Mutate_(const Let *op, const Expr &e) override {
    define_var(op->var);
    Expr expr = IRMutator::Mutate_(op, e);
    undefine_var(op->var);
    return expr;
  }

  Expr Mutate_(const Call *op, const Expr &e) override {
    // cce_img2col_ub() includes undefined vars that will be eliminated by StorageFlatten
    if (op->call_type != Call::CallType::Halide && op->name == "cce_img2col_ub") {
      return e;
    } else {
      return IRMutator::Mutate_(op, e);
    }
  }

  Stmt Mutate_(const Allocate *op, const Stmt &s) override {
    keep_vars_.insert(op->buffer_var.get());
    Stmt stmt = IRMutator::Mutate_(op, s);
    keep_vars_.erase(op->buffer_var.get());
    return stmt;
  }

  Expr Mutate_(const Variable *op, const Expr &e) override {
    if (keep_vars_.count(op) > 0) {
      return e;
    }
    if (var_map_.count(op->name_hint) == 0) {
      if (op->name_hint != "blockIdx.x" && op->name_hint.find(".db") == std::string::npos)
        LOG(FATAL) << "found undefined variable: " << op->name_hint;
      return e;
    }
    return var_map_[op->name_hint];
  }

  std::unordered_map<std::string, VarExpr> var_map_;
  std::unordered_set<const Variable *> keep_vars_;
};

Stmt UnifyLoopVars(const Stmt &stmt, const Map<Tensor, Buffer> &extern_buffer, const Array<NodeRef> &arg_list) {
  return UnifyLoopVarsMutator(extern_buffer, arg_list).Mutate(stmt);
}
}  // namespace ir
}  // namespace akg
