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
#include <tvm/packed_func_ext.h>

#include <numeric>

#include "pass/ir_util.h"
#include "pass/storage_access.h"
#include "ir_pass.h"

namespace akg {
namespace ir {
using air::runtime::PackedFunc;

class EmitInsnDebugger {
 public:
  EmitInsnDebugger() {}
  ~EmitInsnDebugger() {}

  class FindPragmaAttrs : public IRMutator {
   public:
    FindPragmaAttrs() {}
    ~FindPragmaAttrs() override = default;

    Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
      if (op->attr_key == "pragma_emit_insn") {
        attr_stmts_.push_back(s);
      }

      return IRMutator::Mutate_(op, s);
    }
    Array<Stmt> attr_stmts_;
  };

  Stmt Emit(Stmt stmt) {
    FindPragmaAttrs finder;
    static_cast<void>(finder.Mutate(stmt));
    attr_stmts_ = std::move(finder.attr_stmts_);
    if (dumpJson) {
      const PackedFunc *f = air::runtime::Registry::Get("tvm.intrin.cce.writeToJson");
      CHECK(f);
      static_cast<void>((*f)(attr_stmts_));
    }

    return stmt;
  }

 private:
  Array<Stmt> attr_stmts_;
  bool dumpJson{true};
};

Stmt EmitInsnDebug(Stmt stmt) {
  stmt = EmitInsnDebugger().Emit(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
