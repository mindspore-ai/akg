/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "emit_insn/insn_info.h"
#include "emit_insn/cce_params.h"

namespace akg {
namespace ir {
class AtomicDmaEliminator : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_emit_insn" && op->value.as<StringImm>()) {
      if (op->value.as<StringImm>()->value == "broadcast") {
        in_broadcast = true;
        auto body = this->Mutate(op->body);
        in_broadcast = false;
        return AttrStmt::make(op->node, op->attr_key, op->value, body);
      } else if (op->value.as<StringImm>()->value == "dma_atomic_add") {
        in_atomic_add = true;
        auto body = this->Mutate(op->body);
        in_atomic_add = false;
        if (need_elim) {
          need_elim = false;
          return Evaluate::make(Expr(0));
        }
        return AttrStmt::make(op->node, op->attr_key, op->value, body);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    if (in_broadcast) {
      not_assign_addr.push_back(op->buffer_var.get());
    } else {
      auto iter = std::find(not_assign_addr.begin(), not_assign_addr.end(), op->buffer_var.get());
      if (iter != not_assign_addr.end()) {
        not_assign_addr.erase(iter);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Load *op, const Expr &e) final {
    if (in_atomic_add && GetBufScope(op->buffer_var->name_hint) == SCOPE_UBUF &&
        std::find(not_assign_addr.begin(), not_assign_addr.end(), op->buffer_var.get()) != not_assign_addr.end()) {
      need_elim = true;
    }
    return IRMutator::Mutate_(op, e);
  }

 private:
  bool in_broadcast{false};
  bool in_atomic_add{false};
  bool need_elim{false};
  std::vector<const Variable *> not_assign_addr;
};

Stmt EliminateAtomicDma(Stmt stmt) { return AtomicDmaEliminator().Mutate(std::move(stmt)); }
}  // namespace ir
}  // namespace akg
