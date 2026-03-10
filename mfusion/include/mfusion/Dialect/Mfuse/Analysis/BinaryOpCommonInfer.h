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

#ifndef MFUSION_DIALECT_MFUSE_ANALYSIS_BINARY_OP_COMMON_INFER_H
#define MFUSION_DIALECT_MFUSE_ANALYSIS_BINARY_OP_COMMON_INFER_H

#include <algorithm>
#include <cstdint>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include "mfusion/Analysis/SymbolicShape/SymExprBuilder.h"
#include "mfusion/Dialect/Mfuse/Support/SymbolAttrUtils.h"

namespace mlir {
namespace mfuse {

/// Common type/shape inference for broadcastable binary ops (NumPy-style broadcasting).
class BinaryOpCommonInfer {
 public:
  /// Infers broadcast shape for binary ops using NumPy-style broadcasting.
  template <typename T, typename CompareFunc, typename MergeFunc>
  static std::vector<T> inferShape(const std::vector<T> &lhsShape, const std::vector<T> &rhsShape, CompareFunc isOne,
                                   MergeFunc mergeDims) {
    size_t lhsRank = lhsShape.size();
    size_t rhsRank = rhsShape.size();
    size_t maxRank = std::max(lhsRank, rhsRank);

    std::vector<T> resultShape;
    resultShape.reserve(maxRank);
    for (size_t i = 0; i < maxRank; ++i) {
      int64_t lhsIdx = static_cast<int64_t>(lhsRank) - 1 - static_cast<int64_t>(i);
      int64_t rhsIdx = static_cast<int64_t>(rhsRank) - 1 - static_cast<int64_t>(i);

      if (lhsIdx >= 0 && rhsIdx < 0) {
        resultShape.push_back(lhsShape[lhsIdx]);
        continue;
      }
      if (lhsIdx < 0 && rhsIdx >= 0) {
        resultShape.push_back(rhsShape[rhsIdx]);
        continue;
      }

      T lhsDim = lhsShape[lhsIdx];
      T rhsDim = rhsShape[rhsIdx];
      if (isOne(lhsDim)) {
        resultShape.push_back(rhsDim);
      } else if (isOne(rhsDim)) {
        resultShape.push_back(lhsDim);
      } else {
        resultShape.push_back(mergeDims(lhsDim, rhsDim));
      }
    }

    std::reverse(resultShape.begin(), resultShape.end());
    return resultShape;
  }

  /// Infers result type with symbolic shape for broadcastable binary ops.
  static mlir::Type inferSymbolicShape(mlir::OpBuilder &builder, mlir::Type baseType, mlir::Value lhs,
                                       mlir::Value rhs) {
    auto rankedResult = baseType.dyn_cast<mlir::RankedTensorType>();
    if (!rankedResult) return baseType;

    auto maybeLhsExprs = SymbolAttrUtils::getSymbolicShapeExprs(lhs.getType());
    auto maybeRhsExprs = SymbolAttrUtils::getSymbolicShapeExprs(rhs.getType());
    if (mlir::failed(maybeLhsExprs) || mlir::failed(maybeRhsExprs)) {
      return baseType;
    }

    auto lhsExprs = *maybeLhsExprs;
    auto rhsExprs = *maybeRhsExprs;

    std::vector<SymbolAttrUtils::SymExpr> lhsShapeVec(lhsExprs.begin(), lhsExprs.end());
    std::vector<SymbolAttrUtils::SymExpr> rhsShapeVec(rhsExprs.begin(), rhsExprs.end());

    mfusion::SymExprBuilder symBuilder;
    auto resultExprs = inferShape(
      lhsShapeVec, rhsShapeVec, [](SymbolAttrUtils::SymExpr dim) { return dim->__str__() == "1"; },
      [&symBuilder](SymbolAttrUtils::SymExpr lhsDim, SymbolAttrUtils::SymExpr rhsDim) {
        return symBuilder.makeMax(lhsDim, rhsDim);
      });

    auto combinedAttr = SymbolAttrUtils::createSymbolicShapeAttr(builder, resultExprs);
    if (!combinedAttr) return baseType;

    return SymbolAttrUtils::withSymbolicAttr(rankedResult, combinedAttr);
  }

  /// Infers result type for broadcastable binary ops using NumPy-style broadcasting.
  static mlir::Type inferResultType(mlir::Value lhs, mlir::Value rhs, bool isCompareOp) {
    auto lhsType = llvm::dyn_cast<mlir::RankedTensorType>(lhs.getType());
    auto rhsType = llvm::dyn_cast<mlir::RankedTensorType>(rhs.getType());
    if (!lhsType || !rhsType) {
      return {};
    }

    mlir::Type elementType;
    if (isCompareOp) {
      elementType = mlir::IntegerType::get(lhs.getContext(), 1);
    } else {
      elementType = lhsType.getElementType();
    }

    std::vector<int64_t> lhsShapeVec(lhsType.getShape().begin(), lhsType.getShape().end());
    std::vector<int64_t> rhsShapeVec(rhsType.getShape().begin(), rhsType.getShape().end());

    auto resultShapeVec = inferShape(
      lhsShapeVec, rhsShapeVec, [](int64_t dim) { return dim == 1; },
      [](int64_t lhsDim, int64_t rhsDim) -> int64_t { return std::max(lhsDim, rhsDim); });

    llvm::ArrayRef<int64_t> resultShape(resultShapeVec);
    return mlir::RankedTensorType::get(resultShape, elementType);
  }
};

}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_DIALECT_MFUSE_ANALYSIS_BINARY_OP_COMMON_INFER_H
