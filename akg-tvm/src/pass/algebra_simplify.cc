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

#include <regex>
#include <limits.h>

#include "pass/utils.h"
#include "pass/expr_alg_simplify.h"

namespace akg {
namespace ir {
class ReorderConsecutiveMulMutator : public IRMutator {
 public:
  explicit ReorderConsecutiveMulMutator(const std::unordered_map<const Variable *, size_t> &var_depth)
      : var_depth_(var_depth) {}
  ~ReorderConsecutiveMulMutator() override = default;

 private:
  Expr Mutate_(const Mul *op, const Expr &e) final {
    std::multimap<size_t, Expr> ordered_items;
    std::vector<Expr> items_to_split;
    items_to_split.push_back(op->a);
    items_to_split.push_back(op->b);
    while (!items_to_split.empty()) {
      Expr curr_item = items_to_split.back();
      items_to_split.pop_back();
      if (auto mul = curr_item.as<Mul>()) {
        items_to_split.push_back(mul->a);
        items_to_split.push_back(mul->b);
      } else {
        size_t priority = GetAssocPriority(curr_item);
        ordered_items.insert(std::pair<size_t, Expr>(priority, Mutate(curr_item)));
      }
    }

    CHECK_GE(ordered_items.size(), 2);
    Expr ordered_expr;
    bool first = true;
    for (const auto &item : ordered_items) {
      if (first) {
        ordered_expr = item.second;
        first = false;
      } else {
        ordered_expr = Mul::make(ordered_expr, item.second);
      }
    }
    return ordered_expr;
  }

  size_t GetAssocPriority(const Expr &e) const {
    // lower priority value means that they are associated closer, i.e., in the inner scope
    size_t priority = 0;
    constexpr size_t depth_begin = 1;
    PostOrderVisit(e, [&](const NodeRef &node) {
      if (node.as<IntImm>() || node.as<UIntImm>() || node.as<FloatImm>()) {
        return;
      } else if (node.as<Cast>()) {
        return;
      } else if (auto var = node.as<Variable>()) {
        if (var_depth_.count(var) > 0) {
          size_t curr_priority = depth_begin + var_depth_.at(var);
          if (curr_priority > priority) {
            priority = curr_priority;
          }
        } else {  // unknown vars, regarded as close to const
          priority = depth_begin;
        }
      } else if (node.as<Load>()) {
        priority = INT_MAX;
      }
    });
    return priority;
  }

  const std::unordered_map<const Variable *, size_t> &var_depth_;
};

class AlgebraSimplifyMutator : public IRMutator {
 private:
  Expr SimplifyExpr(const Expr &e) {
    if (is_const(e)) return e;
    if (e.as<Variable>()) return e;

    if (e.type().is_bool()) {
      return ReorderConsecutiveMul(Simplify(e));
    } else {
      return ReorderConsecutiveMul(expr_simplifier_.Simplify(e, constraints_));
    }
  }

  Expr ReorderConsecutiveMul(const Expr &e) const { return ReorderConsecutiveMulMutator(var_depth_).Mutate(e); }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    Expr min = SimplifyExpr(op->min);
    Expr extent = SimplifyExpr(op->extent);
    constraints_.push_back(op->loop_var >= min);
    constraints_.push_back(op->loop_var < SimplifyExpr(min + extent));
    loop_vars_.insert(op->loop_var);

    Stmt stmt;
    if (is_const(extent) && GetIntConst(extent) == 0) {
      stmt = Evaluate::make(0);
    } else if (is_const(extent) && GetIntConst(extent) == 1) {
      Map<Var, Expr> var_map;
      var_map.Set(op->loop_var, min);
      stmt = Mutate(Substitute(op->body, var_map));
    } else {
      stmt = For::make(op->loop_var, min, extent, op->for_type, op->device_api, Mutate(op->body));
    }
    constraints_.pop_back();
    constraints_.pop_back();
    loop_vars_.erase(op->loop_var);
    return stmt;
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    Array<Expr> args = op->args;
    for (auto i = 0u; i < args.size(); ++i) {
      args.Set(i, SimplifyExpr(args[i]));
    }
    return Call::make(op->type, op->name, args, op->call_type, op->func, op->value_index);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    Array<Expr> args = op->args;
    for (auto i = 0u; i < args.size(); ++i) {
      args.Set(i, SimplifyExpr(args[i]));
    }
    return Provide::make(op->func, op->value_index, Mutate(op->value), args);
  }

  Expr Mutate_(const Load *op, const Expr &e) final {
    return Load::make(op->type, op->buffer_var, SimplifyExpr(op->index), op->predicate);
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    return Store::make(op->buffer_var, Mutate(op->value), SimplifyExpr(op->index), op->predicate);
  }

  Stmt Mutate_(const Allocate *op, const Stmt &s) final {
    Array<Expr> extents;
    for (auto &extent : op->extents) {
      extents.push_back(SimplifyExpr(extent));
    }
    return Allocate::make(op->buffer_var, op->type, extents, op->condition, Mutate(op->body),
                          (op->new_expr.defined() ? SimplifyExpr(op->new_expr) : op->new_expr), op->free_function);
  }

  Stmt Mutate_(const LetStmt *op, const Stmt &s) final {
    var_depth_[op->var.get()] = ++scope_depth_;
    Expr value = SimplifyExpr(op->value);
    Stmt stmt;
    if (is_const(value) || value.as<Variable>()) {
      Map<Var, Expr> var_map;
      var_map.Set(op->var, value);
      stmt = Mutate(Substitute(op->body, var_map));
    } else {
      stmt = LetStmt::make(op->var, value, Mutate(op->body));
    }
    --scope_depth_;
    var_depth_.erase(op->var.get());
    return stmt;
  }

  std::vector<Expr> Expandconstraints_(const std::vector<Expr> &conds) {
    std::vector<Expr> expanded_conds = conds;
    std::unordered_set<Var, NodeHash, NodeEqual> vars;
    std::unordered_set<Var, NodeHash, NodeEqual> diff_vars;
    for (const auto &cond : conds) {
      GatherVars(cond, &vars);
    }

    for (const auto &var : vars) {
      if (loop_vars_.find(var) == loop_vars_.end()) {
        diff_vars.emplace(var);
      }
    }
    for (const auto &var : diff_vars) {
      expanded_conds.push_back(var > 0);
    }

    return expanded_conds;
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &e) final {
    // ues tvm::Simplify for simple cases
    Expr condition = Simplify(op->condition);
    if (is_const(condition)) {
      if (is_const_true(condition)) {
        return Mutate(op->then_case);
      } else if (op->else_case.defined()) {
        return Mutate(op->else_case);
      } else {
        return Evaluate::make(0);
      }
    }

    auto new_constraints = Expandconstraints_(constraints_);
    if (cond_simplifier_.CanProveValid(op->condition, new_constraints)) {
      return Mutate(op->then_case);
    }
    Stmt then_case = Mutate(op->then_case);
    if (op->else_case.defined()) {
      return IfThenElse::make(condition, then_case, Mutate(op->else_case));
    } else {
      return IfThenElse::make(condition, then_case, op->else_case);
    }
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    const std::regex memlimit_regex("\\[MemoryLimit_([A-Za-z0-9]+)\\]");
    if (std::regex_match(op->attr_key, memlimit_regex)) {
      constraints_.push_back(op->value);
      Stmt stmt = IRMutator::Mutate_(op, s);
      constraints_.pop_back();
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }

  std::unordered_map<const Variable *, size_t> var_depth_;
  std::vector<Expr> constraints_;
  std::unordered_set<Var, NodeHash, NodeEqual> loop_vars_;
  size_t scope_depth_{0};
  ExprSimplifier expr_simplifier_;
  SimplifyIfCondClass cond_simplifier_;
};

Stmt AlgebraSimplify(const Stmt &stmt) { return AlgebraSimplifyMutator().Mutate(stmt); }
}  // namespace ir
}  // namespace akg
