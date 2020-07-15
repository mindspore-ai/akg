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
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/arithmetic.h>
#include <ir_pass.h>
#include "pass/utils.h"

namespace akg {
namespace ir {
class LoopExtentFixer : public IRMutator {
 public:
  LoopExtentFixer() {}
  ~LoopExtentFixer() override = default;

  Stmt Mutate_(const For *op, const Stmt &s) final {
    Var loop_var(op->loop_var);
    Stmt stmt;
    if (IsConstExpr(op->extent)) {
      analyzer.Bind(loop_var, Range::make_by_min_extent(op->min, op->extent));
      stmt = IRMutator::Mutate_(op, s);
    } else {
      auto range = analyzer.const_int_bound(op->extent);
      int extent = static_cast<int>(range->max_value);
      analyzer.Bind(loop_var, Range::make_by_min_extent(op->min, Expr(extent)));
      Expr cond = LT::make(op->loop_var, op->min + op->extent);
      Stmt body = this->Mutate(op->body);
      Stmt newBody = IfThenElse::make(cond, body);
      stmt = For::make(op->loop_var, op->min, Expr(extent), op->for_type, op->device_api, newBody);
    }
    return stmt;
  }

 private:
  air::arith::Analyzer analyzer;
};

Stmt FixLoopExtent(Stmt stmt) {
  stmt = LoopExtentFixer().Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
