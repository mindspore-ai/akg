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
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include <ir_pass.h>
#include "pass/ir_util.h"
#include "pass/utils.h"

namespace akg {
namespace ir {
constexpr size_t kCommonExprCountThreshold = 2;

class ExprInfo {
 public:
  size_t count;
  size_t size;
  Expr expr;
};

using ExprCount = std::unordered_map<std::string, ExprInfo>;
using ExprToVarMap = std::unordered_map<std::string, Var>;

void ExtractSubExprsCount(const Expr &expr, const std::unordered_set<const Variable *> &defined_vars,
                          ExprCount *expr_count) {
  PostOrderVisit(expr, [&expr_count, &defined_vars](const NodeRef &node) {
    auto e = air::Downcast<Expr>(node);
    // do not extract simple exprs
    if (is_const(e)) return;
    if (e.as<FloatImm>()) return;
    if (e.as<Variable>()) return;
    if (e.as<Cast>() && is_const(e.as<Cast>()->value)) return;
    if (e.as<Cast>() && e.as<Cast>()->value.as<Variable>()) return;

    size_t expr_size = 0;
    bool invalid = false;
    PostOrderVisit(e, [&](const NodeRef &descendant) {
      ++expr_size;

      // remove exprs with variables that are defined inside the stmt scope
      if (auto var = descendant.as<Variable>()) {
        if (defined_vars.count(var)) invalid = true;
      }

      // invalid because the expr may have side-effects or its value depends on memory data
      if (descendant.as<Load>()) invalid = true;
      if (auto call = descendant.as<Call>()) {
        if (call->call_type != Call::CallType::PureIntrinsic && call->call_type != Call::CallType::PureExtern) {
          invalid = true;
        }
      }

      // string imms are not allowed to be promoted
      if (descendant.as<StringImm>()) invalid = true;
    });

    if (invalid) return;

    // record count of each identical expr
    auto e_str = ExprToString(e);
    if (expr_count->count(e_str) == 0) {
      expr_count->operator[](e_str) = {1, expr_size, e};
    } else {
      ExprInfo info = expr_count->at(e_str);
      if (!Equal(info.expr, e)) {
        LOG(FATAL) << "found non-equal expr with same string, please check for vars with same name: " << e;
      }
      ++info.count;
      expr_count->operator[](e_str) = info;
    }
  });
}

void ExtractSubExprsCount(const Expr &expr, ExprCount *expr_count) {
  std::unordered_set<const Variable *> defined_vars;
  ExtractSubExprsCount(expr, defined_vars, expr_count);
}

class ExtractExprs : public IRVisitor {
 public:
  ExprCount expr_count;

 private:
  void RegisterExpr(const Expr &e) { ExtractSubExprsCount(e, defined_vars, &expr_count); }

  void Visit_(const For *op) {
    defined_vars.insert(op->loop_var.get());
    RegisterExpr(op->min);
    RegisterExpr(op->extent);
    Visit(op->body);
    defined_vars.erase(op->loop_var.get());
  }

  void Visit_(const LetStmt *op) {
    defined_vars.insert(op->var.get());
    RegisterExpr(op->value);
    Visit(op->body);
    defined_vars.erase(op->var.get());
  }

  void Visit_(const IfThenElse *op) {
    RegisterExpr(op->condition);
    Visit(op->then_case);
    Visit(op->else_case);
  }

  void Visit_(const Provide *op) {
    for (auto arg : op->args) {
      RegisterExpr(arg);
    }
    Visit(op->value);
  }

  void Visit_(const Call *op) {
    for (auto arg : op->args) {
      RegisterExpr(arg);
    }
  }

  void Visit_(const Load *op) { RegisterExpr(op->index); }

  void Visit_(const Store *op) { RegisterExpr(op->index); }

  void Visit_(const Evaluate *op) { RegisterExpr(op->value); }

  void Visit_(const Allocate *op) {
    for (auto &extent : op->extents) {
      RegisterExpr(extent);
    }
    if (op->new_expr.defined()) {
      RegisterExpr(op->new_expr);
    }
    Visit(op->body);
  }

  std::unordered_set<const Variable *> defined_vars;
};

class ReplaceExprsInStmtMutator : public IRMutator {
 public:
  ReplaceExprsInStmtMutator(const ExprToVarMap &expr_to_var_map, const ExprCount &expr_count)
      : expr_to_var_map(expr_to_var_map), expr_count(expr_count) {}
  ~ReplaceExprsInStmtMutator() override = default;

  Stmt Mutate(Stmt stmt) override { return IRMutator::Mutate(stmt); }

  Expr Mutate(Expr expr) override {
    auto expr_str = ExprToString(expr);
    auto it = expr_to_var_map.find(expr_str);
    if (it != expr_to_var_map.end()) {
      auto count_it = expr_count.find(expr_str);
      if (count_it != expr_count.end()) {
        if (Equal(count_it->second.expr, expr)) {
          return it->second;
        }
      }
    }
    return IRMutator::Mutate(expr);
  }

 private:
  const ExprToVarMap &expr_to_var_map;
  const ExprCount &expr_count;
};

class PromoteCommonExprMutator : public IRMutator {
 public:
  Stmt PromoteCommonExprInScope(Stmt stmt) {
    if (defined_names.empty()) {
      GatherDefinedNames(stmt);
    }

    std::vector<Stmt> outer_let, outer_attr;
    // peel outer LetStmt because they are definitions
    stmt = PeelOuterLet(stmt, outer_let, outer_attr);
    auto e = ExtractExprs();
    e.Visit(stmt);
    auto expr_count = e.expr_count;
    while (!expr_count.empty()) {
      auto expr_to_var_map = FindLongestCommonExpr(expr_count);
      if (expr_to_var_map.empty()) break;

      RemoveReplacedExpr(expr_to_var_map, expr_count);
      stmt = ReplaceExprsInStmtMutator(expr_to_var_map, expr_count).Mutate(stmt);
      for (auto it : expr_to_var_map) {
        auto count_it = expr_count.find(it.first);
        CHECK(count_it != expr_count.end());
        stmt = LetStmt::make(it.second, count_it->second.expr, stmt);
      }
    }

    // peel LetStmt that have been added
    stmt = PeelOuterLet(stmt, outer_let, outer_attr);
    // promote common expr in internal scopes
    stmt = IRMutator::Mutate(stmt);
    // merge LetStmt and AttrStmt
    return air::ir::MergeNest(outer_let, air::ir::MergeNest(outer_attr, stmt));
  }

 private:
  Stmt PeelOuterLet(const Stmt &s, std::vector<Stmt> &outer_let, std::vector<Stmt> &outer_attr) {
    auto body = s;
    while (body.as<LetStmt>() || body.as<AttrStmt>()) {
      if (auto let = body.as<LetStmt>()) {
        outer_let.push_back(LetStmt::make(let->var, let->value, Evaluate::make(0)));
        body = let->body;
      } else if (auto attr = body.as<AttrStmt>()) {
        outer_attr.push_back(AttrStmt::make(attr->node, attr->attr_key, attr->value, Evaluate::make(0)));
        body = attr->body;
      }
    }
    return body;
  }

  void GatherDefinedNames(const Stmt &s) {
    PostOrderVisit(s, [&](const NodeRef &node) {
      if (auto op = node.as<Variable>()) {
        defined_names.insert(op->name_hint);
      }
    });
  }

  ExprToVarMap FindLongestCommonExpr(ExprCount &expr_count) {
    ExprToVarMap expr_to_var_map;
    size_t max_duplicate_expr_size = 0;
    for (auto it : expr_count) {
      bool has_duplicate = (it.second.count >= kCommonExprCountThreshold);
      if (has_duplicate && it.second.size > max_duplicate_expr_size) {
        max_duplicate_expr_size = it.second.size;
      }
    }
    if (max_duplicate_expr_size == 0) {
      return expr_to_var_map;  // return empty map
    }
    for (auto it : expr_count) {
      bool has_duplicate = (it.second.count >= kCommonExprCountThreshold);
      if (has_duplicate && it.second.size == max_duplicate_expr_size) {
        Var new_var = CreateNewVar(it.second.expr);
        CHECK(expr_to_var_map.count(it.first) == 0) << "duplicate promoted expr " << it.first;
        expr_to_var_map.emplace(it.first, new_var);
        // LOG(INFO) << "common expr repeat " << it.second.count << ": let " << new_var << " = " << it.second.expr;
      }
    }
    return expr_to_var_map;
  }

  static void RemoveSubExprsCount(const ExprCount &sub_exprs_count, ExprCount &expr_count, size_t outer_duplicates) {
    for (auto sub_it : sub_exprs_count) {
      auto info_it = expr_count.find(sub_it.first);
      CHECK(info_it != expr_count.end()) << sub_it.first;
      auto info = info_it->second;
      auto num_duplicates = outer_duplicates * sub_it.second.count;
      CHECK(info.count >= num_duplicates)
        << "assertion " << info.count << " >= " << num_duplicates << " failed in expr " << sub_it.first;
      CHECK_GE(outer_duplicates, 1);
      auto removed_duplicate_count = (outer_duplicates - 1) * sub_it.second.count;
      info.count -= removed_duplicate_count;
      expr_count[sub_it.first] = info;
    }
  }

  static void RemoveReplacedExpr(const ExprToVarMap &expr_to_var_map, ExprCount &expr_count) {
    for (auto it : expr_to_var_map) {
      auto count_it = expr_count.find(it.first);
      CHECK(count_it != expr_count.end());
      ExprCount sub_exprs_count;
      ExtractSubExprsCount(count_it->second.expr, &sub_exprs_count);
      auto outer_duplicates = count_it->second.count;
      RemoveSubExprsCount(sub_exprs_count, expr_count, outer_duplicates);
    }
  }

  Var CreateNewVar(const Expr &expr) {
    auto name = ExprToVarName(expr);
    if (defined_names.count(name) > 0) {
      int suffix = 1;
      while (defined_names.count(name + std::to_string(suffix)) > 0) {
        ++suffix;
      }
      name = name + std::to_string(suffix);
    }
    defined_names.insert(name);
    return Variable::make(expr.type(), name);
  }

  Stmt Mutate_(const For *op, const Stmt &s) {
    Stmt body = PromoteCommonExprInScope(op->body);
    return For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, body);
  }

  Stmt Mutate_(const LetStmt *op, const Stmt &s) {
    Stmt body = PromoteCommonExprInScope(op->body);
    return LetStmt::make(op->var, op->value, body);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) {
    // do not extract scoped expr inside an instruction because EmitInsn may have problem
    if (op->attr_key == "pragma_emit_insn") {
      return s;
    }
    return IRMutator::Mutate_(op, s);
  }

  std::unordered_set<std::string> defined_names;
};

Stmt PromoteCommonExpr(const Stmt &stmt) { return PromoteCommonExprMutator().PromoteCommonExprInScope(stmt); }
}  // namespace ir
}  // namespace akg
