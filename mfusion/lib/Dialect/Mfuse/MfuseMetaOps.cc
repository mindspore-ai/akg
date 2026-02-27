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

#include <algorithm>
#include <cstdint>
#include <vector>

#include "mfusion/Dialect/Mfuse/Mfuse.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "mfusion/Analysis/SymbolicShape/SymEngineAnalysis.h"
#include "mfusion/Dialect/Mfuse/Analysis/BinaryOpCommonInfer.h"
#include "mfusion/Dialect/Mfuse/Utils/SymbolAttrUtils.h"
#include "mfusion/Analysis/SymbolicShape/SymExprBuilder.h"

namespace mlir::mfuse {

// Implementation of getHigherPrecisionType
mlir::Type getHigherPrecisionType(mlir::Type typeA, mlir::Type typeB) {
  // Helper function to get bit width of a type
  auto getBitWidth = [](mlir::Type type) -> int {
    if (auto floatType = type.dyn_cast<mlir::FloatType>()) {
      return floatType.getWidth();
    }
    if (auto integerType = type.dyn_cast<mlir::IntegerType>()) {
      return integerType.getWidth();
    }
    return 0;
  };

  bool isFloatA = typeA.isa<mlir::FloatType>();
  bool isFloatB = typeB.isa<mlir::FloatType>();

  // If one is float and the other is integer, prefer float
  if (isFloatA && !isFloatB) {
    return typeA;
  }
  if (!isFloatA && isFloatB) {
    return typeB;
  }

  // If both are same type, compare bit widths
  int bitWidthA = getBitWidth(typeA);
  int bitWidthB = getBitWidth(typeB);

  // Return the type with higher bit width
  return bitWidthA >= bitWidthB ? typeA : typeB;
}

mlir::Type CastOp::inferResultType(mlir::Value input, mlir::Type elementType) {
  auto inType = llvm::dyn_cast<mlir::RankedTensorType>(input.getType());
  if (!inType) {
    return {};
  }
  return mlir::RankedTensorType::get(inType.getShape(), elementType, inType.getEncoding());
}

mlir::OpFoldResult CastOp::fold(FoldAdaptor adaptor) {
  if (getInput().getType() == getResult().getType()) {
    return getInput();
  }
  return {};
}

mlir::LogicalResult CastOp::verify() {
  auto inType = mlir::dyn_cast<mlir::RankedTensorType>(getInput().getType());
  auto outType = mlir::dyn_cast<mlir::RankedTensorType>(getResult().getType());
  if (!inType || !outType) {
    return emitOpError("both input and result must be ranked tensors");
  }

  if (inType.getShape() != outType.getShape()) {
    return emitOpError("input and result must have the same shape");
  }

  if (inType.getEncoding() != outType.getEncoding()) {
    return emitOpError("input and result must have the same encoding");
  }

  return mlir::success();
}

mlir::LogicalResult ReshapeOp::verify() {
  auto inType = mlir::dyn_cast<mlir::RankedTensorType>(getInput().getType());
  auto outType = mlir::dyn_cast<mlir::RankedTensorType>(getResult().getType());
  if (!outType) {
    return emitOpError("result must be a ranked tensor");
  }
  int64_t dynamicDims = std::count_if(outType.getShape().begin(), outType.getShape().end(),
                                      [](int64_t d) { return d == mlir::ShapedType::kDynamic; });
  if (dynamicDims > 1) {
    return emitOpError("only semi-static output is supported (at most one dynamic dimension)");
  }
  // When both input and output shapes are static, verify total size equality.
  if (inType && inType.hasStaticShape() && outType.hasStaticShape()) {
    int64_t inNum = inType.getNumElements();
    int64_t outNum = outType.getNumElements();
    if (inNum != outNum) {
      return emitOpError("input and output must have the same total size for static shapes, got ")
             << inNum << " vs " << outNum;
    }
  }
  return mlir::success();
}

mlir::LogicalResult BroadcastToOp::verify() {
  auto inType = mlir::dyn_cast<mlir::RankedTensorType>(getInput().getType());
  auto outType = mlir::dyn_cast<mlir::RankedTensorType>(getOutput().getType());
  if (!inType || !outType) {
    return emitOpError("both input and output must be ranked tensors");
  }

  auto inShape = inType.getShape();
  auto outShape = outType.getShape();
  int64_t inRank = inType.getRank();
  int64_t outRank = outType.getRank();

  if (outRank < inRank) {
    return emitOpError("output rank must be >= input rank for broadcast, got ") << outRank << " vs " << inRank;
  }

  // Check newly added leading dimensions.
  int64_t leading = outRank - inRank;
  for (int64_t i = 0; i < leading; ++i) {
    int64_t dim = outShape[i];
    if (dim == mlir::ShapedType::kDynamic) {
      return emitOpError("newly added leading dimension at index ") << i << " must be static";
    }
    if (dim <= 0) {
      return emitOpError("newly added leading dimension at index ") << i << " must be positive, got " << dim;
    }
  }

  // Check aligned dimensions from the right.
  for (int64_t i = 0; i < inRank; ++i) {
    int64_t inIdx = inRank - 1 - i;
    int64_t outIdx = outRank - 1 - i;
    int64_t inDim = inShape[inIdx];
    int64_t outDim = outShape[outIdx];

    if (inDim == mlir::ShapedType::kDynamic) {
      continue;
    }
    // If input dimension is static, require output dimension to be static and
    // satisfy standard broadcasting constraints.
    if (outDim == mlir::ShapedType::kDynamic) {
      return emitOpError("output dimension at index ") << outIdx << " must be static when input dimension is static";
    }
    if (!(outDim == inDim || inDim == 1)) {
      return emitOpError("invalid broadcast from input dimension ")
             << inDim << " to output dimension " << outDim << " at index " << outIdx
             << " (expected equal or broadcasting from 1)";
    }
    if (outDim <= 0) {
      return emitOpError("output dimension at index ") << outIdx << " must be positive, got " << outDim;
    }
  }

  return mlir::success();
}

mlir::FailureOr<mlir::Type> ReshapeOp::inferSymbolicShapes(mlir::OpBuilder &builder, const mlir::OperationState &state,
                                                           mlir::Type resultType) {
  if (state.operands.empty()) {
    return mlir::failure();
  }

  auto inType = mlir::ValueRange(state.operands).front().getType().dyn_cast<mlir::RankedTensorType>();
  auto outType = resultType.dyn_cast<mlir::RankedTensorType>();
  if (!outType || !inType) {
    return mlir::failure();
  }

  auto outShape = outType.getShape();
  int64_t outRank = outType.getRank();

  int64_t dynamicDims = 0;
  int64_t dynamicDimIndex = -1;
  for (int64_t i = 0; i < outRank; ++i) {
    if (outShape[i] == mlir::ShapedType::kDynamic) {
      ++dynamicDims;
      dynamicDimIndex = i;
    }
  }
  if (dynamicDims != 1) {
    return mlir::failure();
  }

  mfusion::SymExprBuilder symBuilder;
  // SymEngineAnalysis is responsible for symbolic reasoning and expression
  // operations; it does not attach IR attributes itself.
  mfusion::SymEngineAnalysis symAnalysis;
  auto maybeInExprs = SymbolAttrUtils::getSymbolicShapeExprs(inType);
  if (mlir::failed(maybeInExprs)) {
    return mlir::failure();
  }
  auto inExprs = std::move(*maybeInExprs);

  auto prodIn = symBuilder.makeInteger(1);
  for (const auto &expr : inExprs) {
    prodIn = symBuilder.makeMul(prodIn, expr);
  }

  llvm::SmallVector<SymbolAttrUtils::SymExpr> outExprs;
  outExprs.resize(outRank);

  auto prodOut = symBuilder.makeInteger(1);
  for (int64_t i = 0; i < outRank; ++i) {
    if (i == dynamicDimIndex) {
      continue;
    }
    outExprs[i] = symBuilder.makeInteger(outShape[i]);
    prodOut = symBuilder.makeMul(prodOut, outExprs[i]);
  }

  auto res = symBuilder.makeDiv(prodIn, prodOut);
  auto staticDimValue = symAnalysis.tryExtractInt64(res);
  if (mlir::succeeded(staticDimValue)) {
    llvm::SmallVector<int64_t> staticOutShape(outShape.begin(), outShape.end());
    staticOutShape[dynamicDimIndex] = *staticDimValue;
    return mlir::RankedTensorType::get(staticOutShape, outType.getElementType(), outType.getEncoding());
  } else {
    outExprs[dynamicDimIndex] = res;
    // SymbolAttrUtils is only responsible for wiring symbolic attributes onto
    // IR types; the symbolic analysis itself is handled by SymEngineAnalysis.
    return SymbolAttrUtils::withSymbolicAttr(outType, builder, outExprs);
  }
}

// Implementation of promoteBinaryOperands template function
template <typename ConcreteOp>
std::pair<mlir::Value, mlir::Value> promoteBinaryOperands(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                                                          mlir::Value rhs) {
  auto type0 = lhs.getType();
  auto type1 = rhs.getType();
  auto elemType0 = mlir::getElementTypeOrSelf(type0);
  auto elemType1 = mlir::getElementTypeOrSelf(type1);

  // If element types already match, return original inputs
  if (elemType0 == elemType1) {
    return {lhs, rhs};
  }

  // Determine the target high-precision element type
  mlir::Type higherElemType = getHigherPrecisionType(elemType0, elemType1);
  mlir::Value newLhs = lhs;
  mlir::Value newRhs = rhs;

  // Insert CastOp for LHS if needed
  if (elemType0 != higherElemType) {
    newLhs = builder.create<mfuse::CastOp>(loc, lhs, higherElemType);
  }

  // Insert CastOp for RHS if needed
  if (elemType1 != higherElemType) {
    newRhs = builder.create<mfuse::CastOp>(loc, rhs, higherElemType);
  }

  return {newLhs, newRhs};
}

// Macro to implement functions for Mfuse_BroadcastableBinaryOp
#define IMPL_BINARY_OP_FUNCTION(OpName, IsComparisonOp)                                                                \
  mlir::FailureOr<mlir::Type> OpName::inferSymbolicShapes(mlir::OpBuilder &builder, const mlir::OperationState &state, \
                                                          mlir::Type resultType) {                                     \
    if (state.operands.size() != 2) return mlir::failure();                                                            \
    return BinaryOpCommonInfer::inferSymbolicShape(builder, resultType, state.operands[0], state.operands[1]);         \
  }                                                                                                                    \
  mlir::Type OpName::inferResultType(mlir::Value lhs, mlir::Value rhs) {                                               \
    return BinaryOpCommonInfer::inferResultType(lhs, rhs, IsComparisonOp);                                             \
  }                                                                                                                    \
  template std::pair<mlir::Value, mlir::Value> mlir::mfuse::promoteBinaryOperands<OpName>(                             \
    mlir::OpBuilder &, mlir::Location, mlir::Value, mlir::Value);

IMPL_BINARY_OP_FUNCTION(AddOp, false)
IMPL_BINARY_OP_FUNCTION(DivOp, false)
IMPL_BINARY_OP_FUNCTION(EqOp, true)
IMPL_BINARY_OP_FUNCTION(GeOp, true)
IMPL_BINARY_OP_FUNCTION(GtOp, true)
IMPL_BINARY_OP_FUNCTION(LeOp, true)
IMPL_BINARY_OP_FUNCTION(LogicalAndOp, true)
IMPL_BINARY_OP_FUNCTION(LogicalOrOp, true)
IMPL_BINARY_OP_FUNCTION(LtOp, true)
IMPL_BINARY_OP_FUNCTION(MaximumOp, false)
IMPL_BINARY_OP_FUNCTION(MinimumOp, false)
IMPL_BINARY_OP_FUNCTION(MulOp, false)
IMPL_BINARY_OP_FUNCTION(NeOp, false)
IMPL_BINARY_OP_FUNCTION(PowOp, false)
IMPL_BINARY_OP_FUNCTION(RealDivOp, false)
IMPL_BINARY_OP_FUNCTION(SubOp, false)

}  // namespace mlir::mfuse
