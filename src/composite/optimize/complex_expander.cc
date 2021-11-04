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

#include "composite/optimize/complex_expander.h"

namespace akg {

class ComplexExpandMutator : public IRMutator {
 public:
  Stmt Mutate_(const Provide *op, const Stmt &s) {
    auto prim_op = op->value.as<Call>();
    CHECK(prim_op);
    if (prim_op->name == "CReal" || prim_op->name == "CImag") {
      CHECK(prim_op->args.size() == 1);
      auto c_tensor = prim_op->args[0].as<Call>();
      Array<Expr> args;
      auto shape = ExpandShape(c_tensor->args);
      air::DataType type(c_tensor->type.code(), c_tensor->type.bits(), 1);
      args.push_back(Call::make(type, c_tensor->name, shape, c_tensor->call_type, c_tensor->func));
      auto prim_expr = Call::make(prim_op->type, prim_op->name, args, prim_op->call_type, prim_op->func);
      return Provide::make(op->func, op->value_index, prim_expr, op->args);
    }
    if (prim_op->name == "Complex") {
      auto shape = ExpandShape(op->args);
      return Provide::make(op->func, op->value_index, op->value, shape);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  Array<Expr> ExpandShape(Array<Expr> shape) {
    Array<Expr> new_shape = shape;
    new_shape.push_back(make_const(Int(32), 2));
    return new_shape;
  }
};

Stmt ComplexExpander::Run(const Stmt &s) { return ComplexExpandMutator().Mutate(s); }
}  // namespace akg
