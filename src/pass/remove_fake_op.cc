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
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>

#include <regex>

#include "ir_pass.h"
#include "pass/utils.h"

namespace akg {
namespace ir {

class FakeOpRemover : public IRMutator {
  public:
   Expr fake_node_value_;

  private:
    Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (op->attr_key == "pragma_fake_node") {
      fake_node_value_ = op->value;
      Stmt stmt = IRMutator::Mutate_(op, s);
      return stmt.as<AttrStmt>()->body;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final { 
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Provide>();
    if (fake_node_value_.defined()) {
      if (Equal(fake_node_value_, op->func->func_name())) {
        return Evaluate::make(0);
      }
    }
    return stmt;
  }

  
};

Stmt RemoveFakeOp(const Stmt &stmt) {
  return FakeOpRemover().Mutate(stmt);
}
}  // namespace ir
}  // namespace akg
