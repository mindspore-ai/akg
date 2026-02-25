/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "mfusion/Analysis/SymbolicShape/SymExprBuilder.h"

#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/BuiltinTypes.h"
#include "symengine/add.h"
#include "symengine/functions.h"
#include "symengine/integer.h"
#include "symengine/mul.h"
#include "symengine/symbol.h"

namespace mfusion {

SymExprBuilder::SymExpr SymExprBuilder::makeSymbol(const std::string &name) const { return SymEngine::symbol(name); }

SymExprBuilder::SymExpr SymExprBuilder::makeInteger(int64_t value) const { return SymEngine::integer(value); }

SymExprBuilder::SymExpr SymExprBuilder::makeAdd(const SymExpr &lhs, const SymExpr &rhs) const {
  return SymEngine::add(lhs, rhs);
}

SymExprBuilder::SymExpr SymExprBuilder::makeMul(const SymExpr &lhs, const SymExpr &rhs) const {
  return SymEngine::mul(lhs, rhs);
}

SymExprBuilder::SymExpr SymExprBuilder::makeDiv(const SymExpr &lhs, const SymExpr &rhs) const {
  return SymEngine::div(lhs, rhs);
}

SymExprBuilder::SymExpr SymExprBuilder::makeCeil(const SymExpr &expr) const { return SymEngine::ceiling(expr); }

llvm::SmallVector<SymExprBuilder::SymExpr> SymExprBuilder::buildSymExprsFromStaticShape(
  llvm::ArrayRef<int64_t> shape) const {
  llvm::SmallVector<SymExpr> exprs;
  exprs.reserve(shape.size());
  for (int64_t dim : shape) {
    if (dim == mlir::ShapedType::kDynamic) {
      llvm::report_fatal_error("buildSymExprsFromStaticShape expects static shape");
    }
    exprs.push_back(makeInteger(dim));
  }
  return exprs;
}

}  // namespace mfusion
