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
#include "mfusion/Dialect/Mfuse/IR/MfuseAttributes.h"
#include "mfusion/Dialect/Mfuse/Support/SymbolAttrUtils.h"

namespace mlir {
namespace mfuse {

enum class TypePromotePriority : int {
  BOOL = 0,
  INT8 = 1,
  UINT8 = 2,
  INT16 = 3,
  UINT16 = 4,
  INT32 = 5,
  UINT32 = 6,
  INT64 = 7,
  UINT64 = 8,
  BFLOAT16 = 9,
  FLOAT16 = 10,
  FLOAT32 = 11,
  FLOAT64 = 12,
  COMPLEX64 = 13,
  COMPLEX128 = 14,
};

inline int getTypePromotePriority(mlir::Type type) {
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    int w = intType.getWidth();
    if (w == 1) return static_cast<int>(TypePromotePriority::BOOL);
    if (intType.isUnsigned()) {
      if (w == 8) return static_cast<int>(TypePromotePriority::UINT8);
      if (w <= 16) return static_cast<int>(TypePromotePriority::UINT16);
      if (w <= 32) return static_cast<int>(TypePromotePriority::UINT32);
      return static_cast<int>(TypePromotePriority::UINT64);
    } else {
      if (w == 8) return static_cast<int>(TypePromotePriority::INT8);
      if (w <= 16) return static_cast<int>(TypePromotePriority::INT16);
      if (w <= 32) return static_cast<int>(TypePromotePriority::INT32);
      return static_cast<int>(TypePromotePriority::INT64);
    }
  }
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
    if (floatType.isBF16()) return static_cast<int>(TypePromotePriority::BFLOAT16);
    if (floatType.isF16()) return static_cast<int>(TypePromotePriority::FLOAT16);
    if (floatType.isF32()) return static_cast<int>(TypePromotePriority::FLOAT32);
    if (floatType.isF64()) return static_cast<int>(TypePromotePriority::FLOAT64);
  }
  if (auto complexType = mlir::dyn_cast<mlir::ComplexType>(type)) {
    if (auto floatType = mlir::dyn_cast<mlir::FloatType>(complexType.getElementType())) {
      return floatType.isF64() ? static_cast<int>(TypePromotePriority::COMPLEX128)
                               : static_cast<int>(TypePromotePriority::COMPLEX64);
    }
    return static_cast<int>(TypePromotePriority::COMPLEX64);
  }
  llvm::report_fatal_error("Unsupported type");
}

inline mlir::Type inferIntegerResultType(mlir::Type lhs, mlir::Type rhs) {
  auto lhsInt = mlir::cast<mlir::IntegerType>(lhs), rhsInt = mlir::cast<mlir::IntegerType>(rhs);
  auto lhsW = lhsInt.getWidth(), rhsW = rhsInt.getWidth();
  if (lhsW == 1) return rhs;
  if (rhsW == 1) return lhs;
  int w = std::max(lhsW, rhsW);
  bool lu = lhsInt.isUnsigned(), ru = rhsInt.isUnsigned();
  if (lu && ru) {
    if (lhsW == rhsW) {
      return mlir::IntegerType::get(lhs.getContext(), w, mlir::IntegerType::Unsigned);
    }
    llvm::report_fatal_error("Unsupported integer type for different width unsigned integers");
  }
  if ((lu && lhsW >= 16) || (ru && rhsW >= 16)) {
    llvm::report_fatal_error("Unsupported integer type for uint16, uint32, uint64 types");
  }
  w = std::max(w, 16);
  return mlir::IntegerType::get(lhs.getContext(), w, mlir::IntegerType::Signed);
}

inline mlir::Type inferFloatResultType(mlir::Type type, int resultPriority) {
  if (resultPriority >= static_cast<int>(TypePromotePriority::FLOAT64))
    return mlir::FloatType::getF64(type.getContext());
  if (resultPriority >= static_cast<int>(TypePromotePriority::FLOAT32))
    return mlir::FloatType::getF32(type.getContext());
  if (resultPriority >= static_cast<int>(TypePromotePriority::FLOAT16))
    return mlir::FloatType::getF16(type.getContext());
  return mlir::FloatType::getBF16(type.getContext());
}

inline bool isScalarType(mlir::Type type) {
  auto rankedType = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!rankedType) return false;
  auto encoding = rankedType.getEncoding();
  if (!encoding) return false;
  auto dictAttr = mlir::dyn_cast<mlir::DictionaryAttr>(encoding);
  return dictAttr && dictAttr.contains(kScalarMarkerAttr);
}

inline mlir::Type inferScalarType(mlir::Type scalarType, mlir::Type lhsType) {
  if (auto lhsFloat = mlir::dyn_cast<mlir::FloatType>(lhsType)) {
    return lhsType;
  } else if (auto lhsInt = mlir::dyn_cast<mlir::IntegerType>(lhsType)) {
    if (auto scalarInt = mlir::dyn_cast<mlir::IntegerType>(scalarType)) {
      return lhsType;
    }
    return mlir::FloatType::getF32(lhsType.getContext());
  } else {
    return scalarType;
  }
}

inline mlir::Type inferBinaryOpResultType(mlir::Type lhs, mlir::Type rhs) {
  int lhsPriority = getTypePromotePriority(lhs);
  int rhsPriority = getTypePromotePriority(rhs);
  if (lhsPriority == rhsPriority) return lhs;

  int resultPriority = std::max(lhsPriority, rhsPriority);
  bool lhsIsInt = lhsPriority <= static_cast<int>(TypePromotePriority::UINT64);
  bool rhsIsInt = rhsPriority <= static_cast<int>(TypePromotePriority::UINT64);
  bool lhsIsFloat = lhsPriority >= static_cast<int>(TypePromotePriority::BFLOAT16) &&
                    lhsPriority <= static_cast<int>(TypePromotePriority::FLOAT64);
  bool rhsIsFloat = rhsPriority >= static_cast<int>(TypePromotePriority::BFLOAT16) &&
                    rhsPriority <= static_cast<int>(TypePromotePriority::FLOAT64);

  if (lhsIsInt && rhsIsInt) return inferIntegerResultType(lhs, rhs);
  if ((lhsIsFloat && rhsIsFloat) || (lhsIsFloat && rhsIsInt) || (lhsIsInt && rhsIsFloat))
    return inferFloatResultType(lhs, resultPriority);

  bool lhsIsComplex = lhsPriority >= static_cast<int>(TypePromotePriority::COMPLEX64);
  bool rhsIsComplex = rhsPriority >= static_cast<int>(TypePromotePriority::COMPLEX64);
  if (lhsIsComplex || rhsIsComplex) {
    return mlir::ComplexType::get(resultPriority >= static_cast<int>(TypePromotePriority::COMPLEX128)
                                    ? mlir::FloatType::getF64(lhs.getContext())
                                    : mlir::FloatType::getF32(lhs.getContext()));
  }
  return lhs;
}

inline mlir::Type inferResultElementType(mlir::Value lhs, mlir::Value rhs) {
  auto lhsType = llvm::dyn_cast<mlir::RankedTensorType>(lhs.getType());
  auto rhsType = llvm::dyn_cast<mlir::RankedTensorType>(rhs.getType());
  if (!lhsType || !rhsType) return lhs.getType();

  auto lhsElemType = lhsType.getElementType();
  auto rhsElemType = rhsType.getElementType();

  if (isScalarType(rhsType)) {
    rhsElemType = inferScalarType(rhsElemType, lhsElemType);
  }
  return inferBinaryOpResultType(lhsElemType, rhsElemType);
}

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
    auto rankedResult = mlir::dyn_cast<mlir::RankedTensorType>(baseType);
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
      elementType = inferResultElementType(lhs, rhs);
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
