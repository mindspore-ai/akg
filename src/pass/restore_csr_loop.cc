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
#include <pass/storage_access.h>
#include "tvm.h"
#include "build_module.h"
#include "utils.h"


namespace akg {
namespace ir {

Expr GetCsrDynamicExtent(const Variable *op) {
  for (const auto &it: g_csr) {
    auto var = it.first.as<Variable>();
    if (var != nullptr && var->name_hint == op->name_hint) {
      return air::Downcast<Expr>(it.second);
    }
  }
  return Expr();
}

class RecordVar : public IRVisitor {
 public:
  void Visit_(const For *op) final {
    check_csr_ = true;
    IRVisitor::Visit(op->extent);
    check_csr_ = false;
    IRVisitor::Visit(op->body);
    in_scope_ = false;
  }

  void Visit(const NodeRef &node) final{
    if (node->IsInstance<ExprNode>()) {
      auto e = air::Downcast<Expr>(node);
      if (auto var = e.as<Variable>()) {
        if (check_csr_ && GetCsrDynamicExtent(var).defined()) {
          in_scope_ = true;
        } else if (in_scope_) {
          var_map.Set(var->name_hint, e);
        }
        return;
      }
    }
    IRVisitor::Visit(node);
  }

  Map<std::string, Expr> var_map;

 private:
  bool check_csr_{false};
  bool in_scope_{false};
};

class RestoreMaxVar : public IRMutator {
 public:
  explicit RestoreMaxVar(Map<std::string, Expr> var_map) : var_map_(var_map){};
  ~RestoreMaxVar() = default;

  Stmt Mutate_(const For *op, const Stmt &s) final {
    auto stmt = IRMutator::Mutate_(op, s);
    auto new_op = stmt.as<For>();
    if (new_op != nullptr) {
      Expr extent = IRMutator::Mutate(new_op->extent);
      if (!extent.same_as(new_op->extent)) {
        return For::make(new_op->loop_var, new_op->min, extent, new_op->for_type, new_op->device_api, new_op->body);
      }
    }
    return stmt;
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (g_csr.count(e) > 0) {
      auto new_expr = g_csr.at(e);
      if (auto new_call = new_expr.as<Call>()) {
        Array<Expr> args;
        for (auto orig_arg: new_call->args) {
          args.push_back(IRMutator::Mutate(orig_arg));
        }
        return Call::make(
          new_call->type, new_call->name, args, new_call->call_type, new_call->func, new_call->value_index);
      }
    }
    return e;
  }

  Expr Mutate_(const Variable *op, const Expr &e) final {
    Expr extent = GetCsrDynamicExtent(op);
    if (extent.defined()) {
      in_scope_ = true;
      auto new_expr = IRMutator::Mutate(air::Downcast<Expr>(extent));
      in_scope_ = false;
      return new_expr;
    }
    if (in_scope_ && var_map_.count(op->name_hint) > 0) {
      return var_map_.at(op->name_hint);
    }
    return e;
  }

 private:
  Map<std::string, Expr> var_map_;
  bool in_scope_{false};
};

Stmt RestoreCsrLoop(Stmt stmt) {
  auto record_var = RecordVar();
  record_var.Visit(stmt);
  stmt = RestoreMaxVar(record_var.var_map).Mutate(stmt);
  g_csr.Clear();
  return stmt;
}

}  // namespace ir
}  // namespace akg