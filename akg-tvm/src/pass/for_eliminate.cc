/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include "pass/utils.h"

namespace akg {
namespace ir {
class LoopEliminater : public IRMutator {
 public:
  LoopEliminater() = default;
  ~LoopEliminater() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_emit_insn" && op->value.as<StringImm>() && op->body.as<For>()) {
      is_match_ = true;
    }

    auto attr = IRMutator::Mutate_(op, s);

    if (is_match_) {
      is_match_ = false;
      for_loop_vars_.clear();
    }
    return attr;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (is_match_) {
      for_loop_vars_.push_back(op->loop_var);
    }

    auto stmt = IRMutator::Mutate_(op, s);

    if (!is_match_) {
      return stmt;
    }

    auto f = stmt.as<For>();
    CHECK(f);
    if (!air::ir::StmtUseVar(f->body, f->loop_var)) {
      stmt = f->body;
    }

    return stmt;
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    auto stmt = IRMutator::Mutate_(op, s);

    if (!is_match_) {
      return stmt;
    }

    auto if_then_else = stmt.as<IfThenElse>();
    // Only handle no else case
    CHECK(if_then_else);
    if (!if_then_else->else_case.defined()) {
      // catch the condition vars
      std::vector<Expr> cond_vars;
      auto GetCondVars = [&cond_vars](const NodeRef &op) {
        if (op.as<Variable>()) {
          cond_vars.emplace_back(Downcast<Var>(op));
        }
      };
      PostOrderVisit(if_then_else->condition, GetCondVars);

      // loop_vars should cover condition var
      if (IsCover(for_loop_vars_, cond_vars)) {
        // all loop vars should not be used in then_case
        bool flag = true;
        for (auto &f : for_loop_vars_) {
          flag = flag && !air::ir::StmtUseVar(if_then_else->then_case, Downcast<Var>(f));
        }

        if (flag) {
          stmt = if_then_else->then_case;
        }
      }
    }

    return stmt;
  }

 private:
  bool is_match_{false};
  std::vector<Expr> for_loop_vars_;
};

Stmt ForEliminate(Stmt stmt) {
  stmt = LoopEliminater().Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
