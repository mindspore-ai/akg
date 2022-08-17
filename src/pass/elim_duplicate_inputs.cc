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
#include "tvm/ir.h"
#include "tvm/ir_mutator.h"
#include "tvm/ir_pass.h"

#include "composite/lower_tree/sync_process.h"
#include "composite/utils/util.h"

namespace air {
namespace ir {
class DuplicateInputsEliminator : public IRMutator {
 public:
  explicit DuplicateInputsEliminator(const Array<NodeRef>& inputs)
      : names_(akg::GetNames(inputs)){};
  Stmt Run(Stmt& stmt) {
    is_mutate_ = false;
    static_cast<void>(Mutate(stmt));
    is_mutate_ = true;
    return Mutate(stmt);
  }

 private:
  Expr Mutate_(const Load* op, const Expr& e) {
    Var var = op->buffer_var;
    auto name = var->name_hint;
    if (std::find(names_.begin(), names_.end(), name) != names_.end()) {
      auto it = vars_.find(name);
      if (it != vars_.end()) {
        if (is_mutate_)
          return Load::make(op->type, it->second, this->Mutate(op->index), op->predicate);
      } else {
        vars_[name] = var;
      }
    }
    return IRMutator::Mutate_(op, e);
  }
  Stmt Mutate_(const Store* op, const Stmt& s) {
    Var var = op->buffer_var;
    auto name = var->name_hint;
    if (std::find(names_.begin(), names_.end(), name) != names_.end()) {
      auto it = vars_.find(name);
      if (it != vars_.end()) {
        if (is_mutate_)
          return Store::make(it->second, this->Mutate(op->value), this->Mutate(op->index),
                             op->predicate);
      } else {
        vars_[name] = var;
      }
    }
    return IRMutator::Mutate_(op, s);
  }
  bool is_mutate_{false};
  std::unordered_map<std::string, Var> vars_;
  std::vector<std::string> names_;
};

Stmt ElimDuplicateInputsPass(Stmt stmt, Array<NodeRef> inputs_list) {
  Array<NodeRef> inputs;
  for (auto& buf : inputs_list) {
    auto name = buf.as<BufferNode>()->name;
    inputs.push_back(Expr(name));
  }
  auto elim_func = DuplicateInputsEliminator(inputs);
  stmt = elim_func.Run(stmt);
  stmt = akg::ir::ProcessSyncInnerThread(stmt);
  stmt = elim_func.Run(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace air