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
#include <tvm/ir_pass.h>
#include <ir_pass.h>

namespace akg {
namespace ir {
class CastNormalizeMutator : public ktvm::ir::IRMutator {
 public:
  Expr Execute(const Expr &e, const ktvm::DataType castType) {
    castType_ = castType;
    return Mutate(e);
  }

 private:
  Expr Mutate_(const Cast *op, const Expr &e) final { return Mutate(op->value); }

  Expr Mutate_(const IntImm *op, const Expr &e) final {
    if (op->type == castType_) return e;
    return make_const(castType_, op->value);
  }

  Expr Mutate_(const UIntImm *op, const Expr &e) final {
    if (op->type == castType_) return e;
    return make_const(castType_, op->value);
  }

  Expr Mutate_(const Variable *op, const Expr &e) final { return Cast::make(castType_, e); }

 private:
  ktvm::DataType castType_;
};

Expr CastNormalize(const Expr &expr, const ktvm::DataType castType) {
  return CastNormalizeMutator().Execute(expr, castType);
}
}  // namespace ir
}  // namespace akg
