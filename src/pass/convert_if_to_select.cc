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

/**
 * Convert innermost "if" stmt to a "select" stmt, if loop var in the condition is the last axis.
 * Canonicalize the select condition to "cc1 < xxx && cc1 >= xxx && xxx"
 */
#include <pass/ir_util.h>
#include "pass/utils.h"

/*
 * Example before this pass:

  realize reduce0([0, 16], [0, 16]) {
    for (cc0, 0, 16) {
      for (cc1, 0, 16) {
        if (cc1 < cc0 && cc1 > cc0 - 5) {
          if (cc1 < 15) {
            reduce0(cc0, cc1) = input(cc0, cc1);
          }
        }
      }
    }
  }

  After this pass:

  realize reduce0([0, 16], [0, 16]) {
    for (cc0, 0, 16) {
      for (cc1, 0, 16) {
        reduce0(cc0, cc1) = select_loop_var(cc1 < min(cc0, 15) && cc1 >= cc0 - 4,
                                   input(cc0, cc1), reduce0(cc0, cc1));
      }
    }
  }
 */

namespace akg {
namespace ir {
class ConvertIfToSelectMutator : public IRMutator {
 private:
  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    if_stmts.push_back(op->condition);
    Stmt then_stmt = Stmt();
    if (op->then_case.defined()) {
      then_stmt = Mutate(op->then_case);
    }
    if_stmts.pop_back();

    if_stmts.push_back(Simplify(!op->condition));
    Stmt else_stmt = Stmt();
    if (op->else_case.defined()) {
      else_stmt = Mutate(op->else_case);
    }
    if_stmts.pop_back();

    if (then_stmt.defined() && else_stmt.defined()) {
      return Block::make(then_stmt, else_stmt);
    } else if (then_stmt.defined()) {
      return then_stmt;
    } else if (else_stmt.defined()) {
      return else_stmt;
    } else {
      return Stmt();
    }
  }
  void HandleLTCond(const Expr &cond, std::unordered_map<const Variable *, std::vector<Expr>> &upper_bounds,
                    std::unordered_map<const Variable *, std::vector<Expr>> &lower_bounds,
                    std::unordered_map<const Variable *, Expr> &var_expr_map, std::vector<Expr> &irregular_conds) {
    auto lt_cond = cond.as<LT>();
    CHECK(lt_cond);
    if (auto var = lt_cond->a.as<Variable>()) {
      if (last_axis_iterators.count(var) > 0) {
        upper_bounds[var].push_back(lt_cond->b);
        var_expr_map[var] = lt_cond->a;
      }
    }
    if (auto var = lt_cond->b.as<Variable>()) {
      if (last_axis_iterators.count(var) > 0) {
        lower_bounds[var].push_back(lt_cond->a + 1);
        var_expr_map[var] = lt_cond->b;
      }
    }
    irregular_conds.push_back(cond);
  }

  void HandleGTCond(const Expr &cond, std::unordered_map<const Variable *, std::vector<Expr>> &upper_bounds,
                    std::unordered_map<const Variable *, std::vector<Expr>> &lower_bounds,
                    std::unordered_map<const Variable *, Expr> &var_expr_map, std::vector<Expr> &irregular_conds) {
    auto gt_cond = cond.as<GT>();
    CHECK(gt_cond);
    if (auto var = gt_cond->a.as<Variable>()) {
      if (last_axis_iterators.count(var) > 0) {
        lower_bounds[var].push_back(gt_cond->b + 1);
        var_expr_map[var] = gt_cond->a;
      }
    }
    if (auto var = gt_cond->b.as<Variable>()) {
      if (last_axis_iterators.count(var) > 0) {
        upper_bounds[var].push_back(gt_cond->a);
        var_expr_map[var] = gt_cond->b;
      }
    }
    irregular_conds.push_back(cond);
  }

  void HandleLECond(const Expr &cond, std::unordered_map<const Variable *, std::vector<Expr>> &upper_bounds,
                    std::unordered_map<const Variable *, std::vector<Expr>> &lower_bounds,
                    std::unordered_map<const Variable *, Expr> &var_expr_map, std::vector<Expr> &irregular_conds) {
    auto le_cond = cond.as<LE>();
    CHECK(le_cond);
    if (auto var = le_cond->a.as<Variable>()) {
      if (last_axis_iterators.count(var) > 0) {
        upper_bounds[var].push_back(le_cond->b + 1);
        var_expr_map[var] = le_cond->a;
      }
    }
    if (auto var = le_cond->b.as<Variable>()) {
      if (last_axis_iterators.count(var) > 0) {
        lower_bounds[var].push_back(le_cond->a);
        var_expr_map[var] = le_cond->b;
      }
    }
    irregular_conds.push_back(cond);
  }

  void HandleGECond(const Expr &cond, std::unordered_map<const Variable *, std::vector<Expr>> &upper_bounds,
                    std::unordered_map<const Variable *, std::vector<Expr>> &lower_bounds,
                    std::unordered_map<const Variable *, Expr> &var_expr_map, std::vector<Expr> &irregular_conds) {
    auto ge_cond = cond.as<GE>();
    CHECK(ge_cond);
    if (auto var = ge_cond->a.as<Variable>()) {
      if (last_axis_iterators.count(var) > 0) {
        lower_bounds[var].push_back(ge_cond->b);
        var_expr_map[var] = ge_cond->a;
      }
    }
    if (auto var = ge_cond->b.as<Variable>()) {
      if (last_axis_iterators.count(var) > 0) {
        upper_bounds[var].push_back(ge_cond->a + 1);
        var_expr_map[var] = ge_cond->b;
      }
    }
    irregular_conds.push_back(cond);
  }

  Array<Expr> MergeIfConditions(std::vector<Expr> if_conds, bool &found_irregular_conds) {
    std::unordered_map<const Variable *, std::vector<Expr>> upper_bounds;  // exclusive, cc1 < upper_bound
    std::unordered_map<const Variable *, std::vector<Expr>> lower_bounds;  // inclusive, cc1 >= lower_bound
    std::vector<Expr> irregular_conds;
    std::unordered_map<const Variable *, Expr> var_expr_map;

    for (size_t i = 0; i < if_conds.size(); i++) {
      Expr cond = Simplify(if_conds[i]);
      if (auto and_cond = cond.as<And>()) {
        if_conds.push_back(and_cond->a);
        if_conds.push_back(and_cond->b);
      } else {
        if (cond.as<LT>()) {
          HandleLTCond(cond, upper_bounds, lower_bounds, var_expr_map, irregular_conds);
        } else if (cond.as<GT>()) {
          HandleGTCond(cond, upper_bounds, lower_bounds, var_expr_map, irregular_conds);
        } else if (cond.as<LE>()) {
          HandleLECond(cond, upper_bounds, lower_bounds, var_expr_map, irregular_conds);
        } else if (cond.as<GE>()) {
          HandleLECond(cond, upper_bounds, lower_bounds, var_expr_map, irregular_conds);
        } else {
          irregular_conds.push_back(cond);
        }
      }
    }

    Array<Expr> joined_conds;

    for (auto var_upper_bound : upper_bounds) {
      auto var = var_upper_bound.first;
      CHECK_GE(var_upper_bound.second.size(), 1);
      Expr joined_upper_bound = var_upper_bound.second[0];
      for (size_t i = 1; i < var_upper_bound.second.size(); i++) {
        Expr min = Min::make(var_upper_bound.second[i], joined_upper_bound);
        joined_upper_bound = SimplifyExpr(min, loop_var_range);
      }
      Expr cond = (var_expr_map[var] < joined_upper_bound);
      joined_conds.push_back(cond);
    }

    for (auto var_lower_bound : lower_bounds) {
      auto var = var_lower_bound.first;
      CHECK_GE(var_lower_bound.second.size(), 1);
      Expr joined_lower_bound = var_lower_bound.second[0];
      for (size_t i = 1; i < var_lower_bound.second.size(); i++) {
        Expr max = Max::make(var_lower_bound.second[i], joined_lower_bound);
        joined_lower_bound = SimplifyExpr(max, loop_var_range);
      }
      Expr cond = (var_expr_map[var] >= joined_lower_bound);
      joined_conds.push_back(cond);
    }

    found_irregular_conds = !irregular_conds.empty();
    std::copy(irregular_conds.begin(), irregular_conds.end(), std::back_inserter(joined_conds.CopyOnWrite()->data));
    return joined_conds;
  }

  void GatherVars(const Expr &expr) {
    PostOrderVisit(expr, [this](const NodeRef &node) {
      if (const auto var = node.as<Variable>()) {
        last_axis_iterators.insert(var);
      }
    });
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (if_stmts.empty()) {
      return IRMutator::Mutate_(op, s);
    }

    last_axis_iterators.clear();
    if (!op->args.empty()) {
      GatherVars(op->args[op->args.size() - 1]);
    }

    // peel select and move the condition to if_stmts
    Expr true_value = op->value;
    std::vector<Expr> if_and_select_conds = if_stmts;
    while (auto select_expr = true_value.as<Select>()) {
      if (!select_expr->false_value.defined()) {
        if_and_select_conds.push_back(select_expr->condition);
        true_value = select_expr->true_value;
      } else {
        break;
      }
    }

    true_value = IRMutator::Mutate(true_value);
    Expr false_value =
      Call::make(true_value.type(), op->func->func_name(), op->args, Call::CallType::Halide, op->func, op->value_index);

    bool found_irregular_conds{false};
    Array<Expr> merged_conds = MergeIfConditions(if_and_select_conds, found_irregular_conds);
    Expr select_cond = Expr(true);
    if (!merged_conds.empty()) {
      select_cond = merged_conds[0];
      for (size_t i = 1; i < merged_conds.size(); i++) {
        select_cond = And::make(select_cond, merged_conds[i]);
      }
    }
    // We should not simplify the condition here because
    // Simplify may reorder the condition and break the cc1 < max, cc1 >= min form.

    Expr select_expr;
    if (is_const_true(select_cond)) {
      select_expr = true_value;
    } else {
      if (found_irregular_conds) {
        select_expr = Select::make(select_cond, true_value, false_value);
      } else {
        Array<Expr> select_args = {select_cond, true_value, false_value};
        select_expr = Call::make(op->value.type(), "select_loop_var", select_args, Call::CallType::PureExtern);
      }
    }
    return Provide::make(op->func, op->value_index, select_expr, op->args);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    loop_var_range[op->loop_var.get()] = Range::make_by_min_extent(op->min, op->extent);
    Stmt stmt = IRMutator::Mutate_(op, s);
    loop_var_range.erase(op->loop_var.get());
    return stmt;
  }

  std::vector<Expr> if_stmts;
  std::unordered_set<const Variable *> last_axis_iterators;
  std::unordered_map<const Variable *, Range> loop_var_range;
};

Stmt ConvertIfToSelect(Stmt stmt) {
  stmt = ConvertIfToSelectMutator().Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
