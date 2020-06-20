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

#include <tvm/ir_mutator.h>
#include <ir_pass.h>
#include <pass/ir_util.h>

namespace akg {
namespace ir {
class RewriteFloorDivMutator : public IRMutator {
 private:
  Expr Mutate_(const FloorDiv *op, const Expr &e) { return Div::make(Mutate(op->a), Mutate(op->b)); }

  Expr Mutate_(const FloorMod *op, const Expr &e) { return Mod::make(Mutate(op->a), Mutate(op->b)); }
};

Stmt RewriteFloorDiv(const Stmt &stmt) { return RewriteFloorDivMutator().Mutate(stmt); }
}  // namespace ir
}  // namespace akg
