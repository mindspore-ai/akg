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

#include <tvm/expr.h>
#include <tvm/operation.h>
#include <tvm/ir.h>
#include <tvm/tensor.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include <pass/ir_util.h>
#include "pass/utils.h"

/*
 * Example before this pass:

   realize reduce0([0, 100]) {
    realize selected([0, 100], [0, 10]) {
      for (cc0, 0, 100) {
        for (cc1, 0, 10) {
          selected(cc0, cc1) = 0h;
        }
      }
      for (cc0, 0, 10) {
        reduce0(cc0) = 0h;
      }
      for (cc0, 0, 100) {
        for (cc1, 0, 10) {
          if (var1(cc0) <= cc1 && cc1 < var1(cc0) + var2(cc0)) {
            selected(cc0, cc1) = var3(cc0, cc1);
          }
        }
      }
      for (cc0, 0, 100) {
        for (cc1, 0, 10) {
          if (cc1 < var1(cc0)) {
            reduce0(cc0) = reduce0(cc0) + var3(cc0, cc1);
          }
        }
      }
    }
  }

  After this pass:

  realize reduce0([0, 100]) {
    realize selected([0, 100], [0, 10]) {
      for (cc0, 0, 100) {
        for (cc1, 0, 10) {
          selected(cc0, cc1) = 0;
        }
      }
      for (cc0, 0, 10) {
        reduce0(cc0) = 0;
      }
      for (cc0, 0, 100) {
        // attr [cc1] loop_var_lower_bound = 0
        // attr [cc1] loop_var_max_extent = 10
        for (cc1, var1(cc0), var2(cc0)) {
          selected(cc0, cc1) = var3(cc0, cc1);
        }
      }
      for (cc0, 0, 100) {
        // attr [cc1] loop_var_max_extent = 10
        for (cc1, 0, var1(cc0)) {
          reduce0(cc0) = reduce0(cc0) + var3(cc0, cc1);
        }
      }
    }
  }
 */

namespace akg {
namespace ir {
using Region = Array<Range>;

/*
 * Solves a variable (keepExpr) from an inequality.
 *
 * Assume keepExpr = a.
 * before: (a + b) < 128, after: a < 128 - b
 * before: (a * 2) < 128, after: a < truncdiv(128 / 2)
 * We will need a more generic solver.
 */
class IfConditionReorder : public IRMutator {
 public:
  explicit IfConditionReorder(const Expr &keepExpr) : keepExpr_(keepExpr) {}
  ~IfConditionReorder() override = default;

 private:
  Expr Mutate_(const LT *op, const Expr &e) override {
    if (keepExpr_.defined()) {
      auto children = GetBinaryOpExprChildren(op->a);
      size_t idx = 0;
      if (GetIndexOfElement(children, keepExpr_, idx)) {
        if (op->a->IsInstance<Add>()) {
          CHECK_GE(children.size(), 1 + idx);
          return LT::make(keepExpr_, op->b - children[children.size() - 1 - idx]);
        } else if (op->a->IsInstance<Mul>()) {
          CHECK_GE(children.size(), 1 + idx);
          return LT::make(keepExpr_, truncdiv(op->b, children[children.size() - 1 - idx]));
        }
      }
    }

    return IRMutator::Mutate_(op, e);
  }

  Expr keepExpr_;
};

class ConvertCondToExtentMutator : public IRMutator {
 private:
  Stmt Mutate_(const For *op, const Stmt &s) override {
    const Variable *loop_var = op->loop_var.get();
    CHECK_EQ(loop_vars.count(loop_var), 0) << "loop var " << loop_var->name_hint << " is redefined";
    loop_vars.insert(loop_var);
    ordered_loop_vars.push_back(loop_var);
    loop_var_is_exclusive_block[loop_var] = true;
    loop_var_range[loop_var] = Range::make_by_min_extent(op->min, op->extent);
    innermost_for_var = op->loop_var;

    Stmt stmt = IRMutator::Mutate_(op, s);

    innermost_for_var = Expr();
    loop_vars.erase(loop_var);
    ordered_loop_vars.pop_back();
    loop_var_is_exclusive_block.erase(loop_var);
    loop_var_range.erase(loop_var);

    if (loop_var_min_cond.count(loop_var) > 0 || loop_var_max_cond.count(loop_var) > 0) {
      Expr loop_var_min;
      if (loop_var_min_cond.count(loop_var) > 0) {
        loop_var_min = Simplify(Max::make(op->min, loop_var_min_cond[loop_var]));
      } else {
        loop_var_min = op->min;
      }

      Expr loop_var_extent;
      if (loop_var_max_cond.count(loop_var) > 0) {
        loop_var_extent = Simplify(Min::make(op->extent, loop_var_max_cond[loop_var] - loop_var_min));
      } else {
        loop_var_extent = op->extent;
      }

      op = stmt.as<For>();
      CHECK(op != nullptr);
      Stmt for_stmt = For::make(op->loop_var, loop_var_min, loop_var_extent, op->for_type, op->device_api, op->body);
      Stmt attr_stmt = for_stmt;
      if (loop_var_min_cond.count(loop_var) > 0) {
        attr_stmt = AttrStmt::make(op->loop_var, "loop_var_lower_bound", op->min, attr_stmt);
      }
      if (loop_var_max_cond.count(loop_var) > 0) {
        attr_stmt = AttrStmt::make(op->loop_var, "loop_var_max_extent", op->extent, attr_stmt);
      }

      loop_var_min_cond.erase(loop_var);
      loop_var_max_cond.erase(loop_var);
      return attr_stmt;
    } else {
      return stmt;
    }
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) override {
    if (!is_no_op(op->else_case)) {
      return IRMutator::Mutate_(op, s);
    }

    in_condition = true;
    Expr simplified_condition = IfConditionReorder(innermost_for_var).Mutate(Simplify(op->condition));
    Expr condition = IRMutator::Mutate(simplified_condition);
    in_condition = false;
    Stmt then_case = IRMutator::Mutate(op->then_case);
    if (is_positive_const(condition)) {
      return then_case;
    } else {
      return IfThenElse::make(condition, then_case, Stmt());
    }
  }

  Stmt Mutate_(const Block *op, const Stmt &s) override {
    auto backup_loop_var_is_exclusive_block = loop_var_is_exclusive_block;
    for (const auto &var : loop_vars) {
      loop_var_is_exclusive_block[var] = false;
    }
    Stmt stmt = IRMutator::Mutate_(op, s);
    loop_var_is_exclusive_block = backup_loop_var_is_exclusive_block;
    return stmt;
  }

  const bool condition_defined_after_var(const Expr &condition, const Variable *loop_var) {
    bool loop_var_defined = false;
    std::unordered_set<const Variable *> inner_loop_vars;
    for (auto var : ordered_loop_vars) {
      if (loop_var_defined) inner_loop_vars.insert(var);
      if (var == loop_var) loop_var_defined = true;
    }
    if (inner_loop_vars.empty()) return false;

    return ExprUseVar(condition, inner_loop_vars);
  }

  template <class T>
  Expr add_max_cond(const Expr &var_expr, const Expr &max_value, const T *op, const Expr &e) {
    Expr condition = Simplify(max_value);
    CHECK(var_expr.as<Variable>());
    const auto var = var_expr.as<Variable>();
    if (loop_vars.count(var) > 0 && loop_var_is_exclusive_block[var] && !condition_defined_after_var(condition, var)) {
      if (loop_var_max_cond.count(var) == 0) {
        loop_var_max_cond[var] = condition;
      } else {
        Expr orig_cond = loop_var_max_cond[var];
        Expr max = Max::make(condition, orig_cond);
        loop_var_max_cond[var] = SimplifyExpr(max, loop_var_range);
      }
      return const_true();
    } else {
      return IRMutator::Mutate_(op, e);
    }
  }

  Expr Mutate_(const LT *op, const Expr &e) override {
    if (in_condition) {
      if (op->a.as<Variable>()) {
        return add_max_cond<LT>(op->a, op->b, op, e);
      }
    }
    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const GT *op, const Expr &e) override {
    if (in_condition) {
      if (op->b.as<Variable>()) {
        return add_max_cond<GT>(op->b, op->a, op, e);
      }
    }
    return IRMutator::Mutate_(op, e);
  }

  template <class T>
  Expr add_min_cond(const Expr &var_expr, const Expr &min_value, const T *op, const Expr &e) {
    Expr condition = Simplify(min_value);
    CHECK(var_expr.as<Variable>());
    const auto var = var_expr.as<Variable>();
    if (loop_vars.count(var) > 0 && loop_var_is_exclusive_block[var] && !condition_defined_after_var(condition, var)) {
      if (loop_var_min_cond.count(var) == 0) {
        loop_var_min_cond[var] = condition;
      } else {
        Expr orig_cond = loop_var_min_cond[var];
        Expr min = Min::make(condition, orig_cond);
        loop_var_min_cond[var] = SimplifyExpr(min, loop_var_range);
      }
      return const_true();
    } else {
      return IRMutator::Mutate_(op, e);
    }
  }

  Expr Mutate_(const LE *op, const Expr &e) override {
    if (in_condition) {
      if (op->b.as<Variable>()) {
        return add_min_cond<LE>(op->b, op->a, op, e);
      }
    }
    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const GE *op, const Expr &e) override {
    if (in_condition) {
      if (op->a.as<Variable>()) {
        return add_min_cond<GE>(op->a, op->b, op, e);
      }
    }
    return IRMutator::Mutate_(op, e);
  }

 public:
  Stmt run(const Stmt &stmt) { return IRMutator::Mutate(stmt); }

 private:
  bool in_condition{false};
  Expr innermost_for_var;
  std::unordered_set<const Variable *> loop_vars;
  std::vector<const Variable *> ordered_loop_vars;
  std::unordered_map<const Variable *, Expr> loop_var_min_cond;
  std::unordered_map<const Variable *, Expr> loop_var_max_cond;
  std::unordered_map<const Variable *, bool> loop_var_is_exclusive_block;
  std::unordered_map<const Variable *, Range> loop_var_range;
};

Stmt ConvertCondToExtent(Stmt stmt) {
  stmt = ConvertCondToExtentMutator().run(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
