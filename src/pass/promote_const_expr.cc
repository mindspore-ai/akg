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
using ExprList = std::unordered_map<std::string, Expr>;
using ExprToVarMap = std::unordered_map<std::string, Var>;

bool ExtractConstExpr(const Expr &e, const std::unordered_set<const Variable *> &defined_vars, ExprList *const_exprs) {
  // do not extract simple exprs
  if (is_const(e)) return false;
  if (e.as<FloatImm>()) return false;
  if (e.as<Variable>()) return false;
  if (e.as<Cast>() && is_const(e.as<Cast>()->value)) return false;
  if (e.as<Cast>() && e.as<Cast>()->value.as<Variable>()) return false;

  bool invalid = false;
  PostOrderVisit(e, [&](const NodeRef &descendant) {
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

  if (invalid) return false;

  // record count of each identical expr
  auto e_str = ExprToString(e);
  if (const_exprs->count(e_str) == 0) {
    const_exprs->operator[](e_str) = e;
  } else {
    if (!Equal(const_exprs->at(e_str), e)) {
      LOG(FATAL) << "found non-equal expr with same string, please check for vars with same name: " << e;
    }
  }
  return true;
}

class ExtractConstExprs : public IRVisitor {
 public:
  ExprList const_exprs;

 private:
  void RegisterExpr(const Expr &e) {
    if (!ExtractConstExpr(e, defined_vars, &const_exprs)) {
      // if not registered, descend into sub-expr
      IRVisitor::Visit(e);
    }
  }

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
  ReplaceExprsInStmtMutator(const ExprToVarMap &expr_to_var_map, const ExprList &expr_list)
      : expr_to_var_map(expr_to_var_map), expr_list(expr_list) {}
  ~ReplaceExprsInStmtMutator() override = default;

  Stmt Mutate(Stmt stmt) override { return IRMutator::Mutate(stmt); }

  Expr Mutate(Expr expr) override {
    auto expr_str = ExprToString(expr);
    auto it = expr_to_var_map.find(expr_str);
    if (it != expr_to_var_map.end()) {
      auto count_it = expr_list.find(expr_str);
      if (count_it != expr_list.end()) {
        if (Equal(count_it->second, expr)) {
          return it->second;
        }
      }
    }
    return IRMutator::Mutate(expr);
  }

 private:
  const ExprToVarMap &expr_to_var_map;
  const ExprList &expr_list;
};

class PromoteConstExprMutator : public IRMutator {
 public:
  Stmt PromoteConstExprInScope(Stmt stmt) {
    if (defined_names.empty()) {
      GatherDefinedNames(stmt);
    }
    if (!StmtContainsLoop(stmt)) {
      return stmt;
    }

    std::vector<Stmt> outer_let, outer_attr;
    // peel outer LetStmt because they are definitions
    stmt = PeelOuterLet(stmt, outer_let, outer_attr);
    auto e = ExtractConstExprs();
    e.Visit(stmt);
    auto const_exprs = e.const_exprs;
    ExprToVarMap expr_to_var_map;
    for (auto it : const_exprs) {
      Var new_var = CreateNewVar(it.second);
      CHECK(expr_to_var_map.count(it.first) == 0) << "duplicate promoted expr " << it.first;
      expr_to_var_map.emplace(it.first, new_var);
      outer_let.push_back(LetStmt::make(new_var, it.second, Evaluate::make(0)));
      // LOG(INFO) << "const expr: let " << new_var << " = " << it.second;
    }
    stmt = ReplaceExprsInStmtMutator(expr_to_var_map, const_exprs).Mutate(stmt);

    // peel LetStmt that have been added
    stmt = PeelOuterLet(stmt, outer_let, outer_attr);
    // promote common expr in internal scopes
    stmt = IRMutator::Mutate(stmt);
    // merge LetStmt and AttrStmt
    return ktvm::ir::MergeNest(outer_let, ktvm::ir::MergeNest(outer_attr, stmt));
  }

 private:
  static bool StmtContainsLoop(const Stmt &s) {
    bool found_loop = false;
    PostOrderVisit(s, [&](const NodeRef &node) {
      if (node.as<For>()) found_loop = true;
    });
    return found_loop;
  }

  static Stmt PeelOuterLet(const Stmt &s, std::vector<Stmt> &outer_let, std::vector<Stmt> &outer_attr) {
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
    Stmt body = PromoteConstExprInScope(op->body);
    return For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, body);
  }

  Stmt Mutate_(const LetStmt *op, const Stmt &s) {
    Stmt body = PromoteConstExprInScope(op->body);
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

Stmt PromoteConstExpr(const Stmt &stmt) { return PromoteConstExprMutator().PromoteConstExprInScope(stmt); }
}  // namespace ir
}  // namespace akg
