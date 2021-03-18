/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

namespace akg {
namespace ir {

class DoubleBufferScopeInjector : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == air::ir::attr::storage_scope && op->value.as<StringImm>()->value == "shared") {
      touched_.insert(op->node.as<Variable>());
    }
    if (op->attr_key == "promote_vectorization" && op->value.as<StringImm>()->value == "promote_vectorization") {
      is_vectorize_ = true;
      auto res = Mutate(op->body);
      if (need_db_) {
        res = AttrStmt::make(op->node, op->attr_key, op->value, res);
        res = AttrStmt::make(db_var_, air::ir::attr::double_buffer_scope, 1, res);
        need_db_ = false;
      }
      is_vectorize_ = false;
      return res;
    }
    return IRMutator::Mutate_(op, s);
  }

  bool IsDbFetchBlock(const Stmt &s) {
    if (auto store = s.as<Store>()) {
      auto it = touched_.find(store->buffer_var.get());
      if (it != touched_.end()) {
        db_var_ = store->buffer_var;
        return true;
      }
    } else if (auto loop = s.as<For>()) {
      if (IsDbFetchBlock(loop->body)) {
        return true;
      }
    } else if (auto attr = s.as<AttrStmt>()) {
      if (IsDbFetchBlock(attr->body)) {
        return true;
      }
    }
    return false;
  }
  bool HasOuterLoop() { return !loop_nest_.empty(); }
  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (IsDbFetchBlock(s) && HasOuterLoop()) {
      if (is_vectorize_) {
        need_db_ = true;
        return s;
      }
      return AttrStmt::make(db_var_, air::ir::attr::double_buffer_scope, 1, s);
    } else {
      loop_nest_.push_back(op);
      auto stmt = IRMutator::Mutate_(op, s);
      loop_nest_.pop_back();
      return stmt;
    }
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    if (IsDbFetchBlock(s) && HasOuterLoop()) {
      return AttrStmt::make(db_var_, air::ir::attr::double_buffer_scope, 1, s);
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

 private:
  std::unordered_set<const Variable *> touched_;
  VarExpr db_var_;
  std::vector<const For *> loop_nest_;
  bool need_db_{false};
  bool is_vectorize_{false};
};

Stmt InjectDoubleBufferScopeOnGpu(Stmt stmt) {
  stmt = DoubleBufferScopeInjector().Mutate(stmt);
  return stmt;
}

}  // namespace ir
}  // namespace akg