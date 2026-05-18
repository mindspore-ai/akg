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

#ifndef MFUSION_ANALYSIS_SYMBOLIC_SHAPE_SYMENGINE_ANALYSIS_H
#define MFUSION_ANALYSIS_SYMBOLIC_SHAPE_SYMENGINE_ANALYSIS_H

#include <functional>
#include <string>

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "symengine/basic.h"
#include "mfusion/Analysis/SymbolicShape/SymExprBuilder.h"

namespace mfusion {

// Symbolic shape analysis helper using SymEngine expressions.
class SymEngineAnalysis {
 public:
  using SymExpr = SymExprBuilder::SymExpr;
  using SymbolNameResolver = std::function<mlir::FailureOr<std::string>(mlir::Value)>;

  // Convert an MLIR affine expression into a SymEngine expression.
  mlir::FailureOr<SymExpr> convertAffineExpr(mlir::AffineExpr expr, llvm::ArrayRef<SymExpr> dimSymbols,
                                             llvm::ArrayRef<SymExpr> symbolSymbols) const;

  // Apply an affine map to symbolic operands and return SymEngine results.
  mlir::FailureOr<llvm::SmallVector<SymExpr>> applyAffineMap(mlir::AffineMap map, mlir::ValueRange symbols,
                                                             const SymbolNameResolver &resolver);

  // Try to extract an integer value from a SymEngine expression.
  mlir::FailureOr<int64_t> tryExtractInt64(const SymExpr &expr) const;

  // SymEngine structural equality helpers (exact structural match).
  bool isStructurallyEqual(const SymExpr &lhs, const SymExpr &rhs) const;
  bool isStructurallyNotEqual(const SymExpr &lhs, const SymExpr &rhs) const;

  void reset();

 private:
  // Map an SSA value to a SymEngine symbol using the provided name resolver.
  mlir::FailureOr<SymExpr> getOrAssignSymbol(mlir::Value value, const SymbolNameResolver &resolver);

  SymExprBuilder builder_;
  llvm::DenseMap<mlir::Value, SymExpr> valueToSymMap_;
};

}  // namespace mfusion

#endif  // MFUSION_ANALYSIS_SYMBOLIC_SHAPE_SYMENGINE_ANALYSIS_H
