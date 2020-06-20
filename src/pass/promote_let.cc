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
#include "pass/utils.h"

namespace akg {
namespace ir {
class PromoteLetStmtMutator : public IRMutator {
 public:
  Stmt Promote(const Stmt &s, const Array<NodeRef> &arg_list) {
    GatherExternVars(arg_list);
    Stmt stmt_without_let = Mutate(s);
    GenerateNewLetVars();
    return WrapLetStmts(Substitute(stmt_without_let, var_map));
  }

 private:
  void GatherExternVars(const Array<NodeRef> &arg_list) {
    for (auto arg : arg_list) {
      if (arg.as<Variable>() != nullptr) {
        defined_vars.insert(ktvm::Downcast<Var>(arg));
      }
    }
  }

  void GenerateNewLetVars() {
    for (auto let : let_stmts) {
      Expr new_value = Substitute(let->value, var_map);
      // search value expr in defined let stmts
      bool found = false;
      Var reuse_var;
      for (auto it : new_let_stmts) {
        if (Equal(it.second, new_value)) {
          found = true;
          reuse_var = it.first;
        }
      }
      if (found) {
        // unify vars, discard duplicate LetStmt
        var_map.emplace(let->var, reuse_var);
      } else {
        // create a new LetStmt, and make sure the var name does not overlap
        Var new_var = let->var;
        if (used_var_names.count(let->var->name_hint)) {
          std::string new_var_name;
          int suffix = 0;
          do {
            new_var_name = let->var->name_hint + std::to_string(++suffix);
          } while (used_var_names.count(new_var_name));

          new_var = Variable::make(let->var.type(), new_var_name);
          var_map.emplace(let->var, new_var);
        }
        used_var_names.insert(new_var->name_hint);
        new_let_stmts.push_back(std::pair<Var, Expr>(new_var, new_value));
      }
    }
  }

  Stmt WrapLetStmts(Stmt stmt) {
    auto num_lets = new_let_stmts.size();
    for (auto i = num_lets; i > 0; --i) {
      auto let_info = new_let_stmts[i - 1];
      stmt = LetStmt::make(let_info.first, let_info.second, stmt);
    }
    return stmt;
  }

  bool IsStaticExpr(const Expr &e) const {
    bool found_free_var = false;
    bool found_mem_load = false;
    PostOrderVisit(e, [&](const NodeRef &node) {
      if (node.as<Variable>() != nullptr) {
        if (non_static_vars.count(ktvm::Downcast<Var>(node)) > 0) {
          found_free_var = true;
        } else if (defined_vars.count(ktvm::Downcast<Var>(node)) == 0) {
          // blockIdx.x is a special global var that indicates the thread number
          if (node.as<Variable>()->name_hint != "blockIdx.x") {
            LOG(INFO) << "possibly undefined var " << node << " " << node.as<Variable>() << " found in LetStmt " << e;
          }
        }
      } else if (auto op = node.as<Call>()) {
        if (op->call_type == Call::CallType::Halide) {
          found_mem_load = true;
        }
      } else if (node.as<Load>() != nullptr) {
        found_mem_load = true;
      }
    });
    return (!found_free_var) && (!found_mem_load);
  }

  Stmt Mutate_(const LetStmt *op, const Stmt &s) {
    if (IsStaticExpr(op->value)) {
      let_stmts.push_back(op);
      defined_vars.insert(op->var);
      Stmt body = Mutate(op->body);
      defined_vars.erase(op->var);
      return body;
    } else {
      non_static_vars.insert(op->var);
      Stmt stmt = IRMutator::Mutate_(op, s);
      non_static_vars.erase(op->var);
      return stmt;
    }
  }

  Stmt Mutate_(const For *op, const Stmt &s) {
    non_static_vars.insert(op->loop_var);
    Stmt stmt = IRMutator::Mutate_(op, s);
    non_static_vars.erase(op->loop_var);
    return stmt;
  }

  std::vector<const LetStmt *> let_stmts;
  std::unordered_map<Var, Expr, NodeHash, NodeEqual> var_map;
  std::unordered_set<std::string> used_var_names;
  std::vector<std::pair<Var, Expr>> new_let_stmts;
  std::unordered_set<Var, NodeHash, NodeEqual> defined_vars;
  std::unordered_set<Var, NodeHash, NodeEqual> non_static_vars;
};

Stmt PromoteLetStmt(const Stmt &stmt, const Array<NodeRef> &arg_list) {
  return PromoteLetStmtMutator().Promote(stmt, arg_list);
}
}  // namespace ir
}  // namespace akg
