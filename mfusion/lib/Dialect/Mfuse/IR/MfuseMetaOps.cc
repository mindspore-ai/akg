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
#include <type_traits>

#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/APSInt.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mfusion/Analysis/SymbolicShape/SymEngineAnalysis.h"
#include "mfusion/Dialect/Mfuse/Analysis/BinaryOpCommonInfer.h"
#include "mfusion/Dialect/Mfuse/Analysis/ReduceOpCommonInfer.h"
#include "mfusion/Dialect/Mfuse/Support/SymbolAttrUtils.h"
#include "mfusion/Analysis/SymbolicShape/SymExprBuilder.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

namespace TorchD = mlir::torch::Torch;

namespace {

// Implementation of getHigherPrecisionType
mlir::Type getHigherPrecisionType(mlir::Type typeA, mlir::Type typeB, bool rhsIsScalar = false) {
  // Helper function to get bit width of a type
  auto getBitWidth = [](mlir::Type type) -> int {
    if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
      return floatType.getWidth();
    }

    if (auto integerType = mlir::dyn_cast<mlir::IntegerType>(type)) {
      return integerType.getWidth();
    }
    return 0;
  };

  bool isFloatA = mlir::isa<mlir::FloatType>(typeA);
  bool isFloatB = mlir::isa<mlir::FloatType>(typeB);

  if (rhsIsScalar) {
    if (isFloatB) {
      if (!isFloatA) {
        // Int64 + float64(scalar) ==> float32
        return getHigherPrecisionType(typeA, mlir::FloatType::getF32(typeB.getContext()));
      } else {
        // Float16 + float64(scalar) ==> float16
        return typeA;
      }
    } else {
      return typeA;
    }
  }

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

mlir::LogicalResult verifyReduceDimensions(mlir::Operation *op, mlir::ArrayAttr dimensions) {
  auto input = op->getOperand(0);
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  if (!inputType || !inputType.hasRank()) {
    return op->emitOpError("input must be a ranked tensor");
  }

  int64_t rank = inputType.getRank();
  if (dimensions.empty()) {
    if (rank == 0) {
      return mlir::success();
    }
    return op->emitOpError("dimensions must not be empty");
  }

  llvm::DenseSet<int64_t> seenDims;
  for (auto dimAttr : dimensions.getValue()) {
    auto dim = mlir::cast<mlir::IntegerAttr>(dimAttr).getValue().getSExtValue();
    if (dim < 0) {
      return op->emitOpError("dimensions must be non-negative, got ") << dim;
    }
    if (dim >= rank) {
      return op->emitOpError("dimension out of range, got ") << dim << " for input rank " << rank;
    }
    if (!seenDims.insert(dim).second) {
      return op->emitOpError("duplicate reduction dimensions are not supported, got ") << dim;
    }
  }
  return mlir::success();
}

}  // namespace

namespace mlir::mfuse {

// Helper function to fold unrealized conversion cast operations
// Returns a new UnrealizedConversionCastOp with the converted type, or std::nullopt on failure
static std::optional<mlir::Value> foldUnrealizedConversionCast(mlir::OpBuilder &b,
                                                               mlir::UnrealizedConversionCastOp castOp,
                                                               mlir::Type resultType, mlir::Location loc) {
  // Get the original input type (before conversion)
  mlir::Value origInput = castOp->getOperand(0);
  mlir::Type origInputType = origInput.getType();

  if (!mlir::isa<TorchD::FloatType>(origInputType) && !mlir::isa<TorchD::IntType>(origInputType)) {
    return std::nullopt;
  }

  // Create a new dictionary encoding with the scalar marker that includes original type info
  mlir::MLIRContext *ctx = b.getContext();
  mlir::SmallVector<mlir::NamedAttribute> attrs;

  // Set is_scalar to the original torch type if it's a torch type, otherwise empty string
  mlir::Attribute scalarValue;
  if (mlir::isa<TorchD::FloatType>(origInputType)) {
    scalarValue = mlir::StringAttr::get(ctx, "!torch.float");
  } else if (mlir::isa<TorchD::IntType>(origInputType)) {
    scalarValue = mlir::StringAttr::get(ctx, "!torch.int");
  } else {
    scalarValue = mlir::StringAttr::get(ctx, "");
  }

  attrs.emplace_back(mlir::StringAttr::get(ctx, mlir::mfuse::kScalarMarkerAttr), scalarValue);

  // Create the new encoding
  auto newEncoding = mlir::DictionaryAttr::get(ctx, attrs);

  // Create new tensor type with the new encoding
  auto resultTensorType = mlir::cast<mlir::RankedTensorType>(resultType);
  auto newResultType =
    mlir::RankedTensorType::get(resultTensorType.getShape(), resultTensorType.getElementType(), newEncoding);

  // Create new UnrealizedConversionCastOp with the new type
  auto newCast = b.create<mlir::UnrealizedConversionCastOp>(loc, newResultType, castOp->getOperands());
  return newCast.getResult(0);
}

// Helper function to fold constant cast operations
// Returns a new DenseElementsAttr with converted values, or std::nullopt on failure
static std::optional<mlir::DenseElementsAttr> foldConstantCast(mlir::DenseElementsAttr dense, mlir::Type srcType,
                                                               mlir::Type dstType, mlir::RankedTensorType resultType) {
  // Type categories
  bool srcIsInt = mlir::isa<mlir::IntegerType>(srcType);
  bool dstIsInt = mlir::isa<mlir::IntegerType>(dstType);
  bool srcIsFloat = mlir::isa<mlir::FloatType>(srcType);
  bool dstIsFloat = mlir::isa<mlir::FloatType>(dstType);

  // Integer -> Integer: adjust width
  if (srcIsInt && dstIsInt) {
    auto srcIntTy = mlir::cast<mlir::IntegerType>(srcType);
    auto dstIntTy = mlir::cast<mlir::IntegerType>(dstType);
    auto convertInt = [&](const llvm::APInt &val) -> llvm::APInt {
      if (dstIntTy.getWidth() > srcIntTy.getWidth()) {
        return srcIntTy.isSigned() ? val.sext(dstIntTy.getWidth()) : val.zext(dstIntTy.getWidth());
      } else if (dstIntTy.getWidth() < srcIntTy.getWidth()) {
        return val.trunc(dstIntTy.getWidth());
      }
      return val;
    };
    if (dense.isSplat()) {
      return std::optional<mlir::DenseElementsAttr>(
        mlir::DenseElementsAttr::get(resultType, convertInt(dense.getSplatValue<llvm::APInt>())));
    } else {
      llvm::SmallVector<llvm::APInt> vals;
      vals.reserve(dense.getNumElements());
      auto intVals = dense.getValues<llvm::APInt>();
      std::transform(intVals.begin(), intVals.end(), vals.begin(), convertInt);
      return std::optional<mlir::DenseElementsAttr>(mlir::DenseElementsAttr::get(resultType, vals));
    }
  }

  // Float -> Float: convert precision
  if (srcIsFloat && dstIsFloat) {
    auto dstFloatTy = mlir::cast<mlir::FloatType>(dstType);
    const llvm::fltSemantics &dstSemantics = dstFloatTy.getFloatSemantics();
    auto convertFloat = [&](const llvm::APFloat &val) -> llvm::APFloat {
      llvm::APFloat result(val);
      bool losesInfo = false;
      result.convert(dstSemantics, llvm::APFloat::rmNearestTiesToEven, &losesInfo);
      return result;
    };
    if (dense.isSplat()) {
      return std::optional<mlir::DenseElementsAttr>(
        mlir::DenseElementsAttr::get(resultType, convertFloat(dense.getSplatValue<llvm::APFloat>())));
    } else {
      llvm::SmallVector<llvm::APFloat> vals;
      vals.reserve(dense.getNumElements());
      auto floatVals = dense.getValues<llvm::APFloat>();
      std::transform(floatVals.begin(), floatVals.end(), vals.begin(), convertFloat);
      return std::optional<mlir::DenseElementsAttr>(mlir::DenseElementsAttr::get(resultType, vals));
    }
  }

  // Integer -> Float
  if (srcIsInt && dstIsFloat) {
    auto dstFloatTy = mlir::cast<mlir::FloatType>(dstType);
    const llvm::fltSemantics &dstSemantics = dstFloatTy.getFloatSemantics();
    auto convertIntToFloat = [&](const llvm::APInt &val) -> llvm::APFloat {
      llvm::APFloat result = llvm::APFloat::getZero(dstSemantics);
      result.convertFromAPInt(val, /*isSigned=*/true, llvm::APFloat::rmNearestTiesToEven);
      return result;
    };
    if (dense.isSplat()) {
      return std::optional<mlir::DenseElementsAttr>(
        mlir::DenseElementsAttr::get(resultType, convertIntToFloat(dense.getSplatValue<llvm::APInt>())));
    } else {
      llvm::SmallVector<llvm::APFloat> vals;
      vals.reserve(dense.getNumElements());
      auto intVals = dense.getValues<llvm::APInt>();
      std::transform(intVals.begin(), intVals.end(), vals.begin(), convertIntToFloat);
      return std::optional<mlir::DenseElementsAttr>(mlir::DenseElementsAttr::get(resultType, vals));
    }
  }

  // Float -> Integer
  if (srcIsFloat && dstIsInt) {
    auto dstIntTy = mlir::cast<mlir::IntegerType>(dstType);
    auto convertFloatToInt = [&](const llvm::APFloat &val) -> llvm::APInt {
      llvm::APSInt result(dstIntTy.getWidth(), !dstIntTy.isSigned());
      bool isExact = false;
      val.convertToInteger(result, llvm::APFloat::rmNearestTiesToEven, &isExact);
      return result;
    };

    if (dense.isSplat()) {
      return std::optional<mlir::DenseElementsAttr>(
        mlir::DenseElementsAttr::get(resultType, convertFloatToInt(dense.getSplatValue<llvm::APFloat>())));
    } else {
      llvm::SmallVector<llvm::APInt> vals;
      vals.reserve(dense.getNumElements());
      auto floatVals = dense.getValues<llvm::APFloat>();
      std::transform(floatVals.begin(), floatVals.end(), vals.begin(), convertFloatToInt);
      return std::optional<mlir::DenseElementsAttr>(mlir::DenseElementsAttr::get(resultType, vals));
    }
  }

  // For other type conversions, try reshape
  auto reshaped = dense.reshape(resultType);
  if (reshaped) return std::optional<mlir::DenseElementsAttr>(reshaped);
  return std::nullopt;
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

mlir::FailureOr<mlir::Type> BroadcastToOp::inferSymbolicShapes(mlir::OpBuilder &builder,
                                                               const mlir::OperationState &state,
                                                               mlir::Type resultType) {
  if (state.operands.empty()) {
    return mlir::failure();
  }

  auto inType = mlir::dyn_cast<mlir::RankedTensorType>(state.operands[0].getType());
  auto outType = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
  if (!outType || !inType) {
    return mlir::failure();
  }

  auto maybeInExprs = SymbolAttrUtils::getSymbolicShapeExprs(inType);
  if (mlir::failed(maybeInExprs)) {
    return mlir::failure();
  }
  auto inExprs = std::move(*maybeInExprs);

  mfusion::SymExprBuilder symBuilder;
  auto outShape = outType.getShape();
  int64_t outRank = outType.getRank();
  int64_t inRank = inType.getRank();
  int64_t leading = outRank - inRank;

  llvm::SmallVector<SymbolAttrUtils::SymExpr> outExprs;
  outExprs.resize(outRank);

  for (int64_t outIdx = 0; outIdx < leading; ++outIdx) {
    outExprs[outIdx] = symBuilder.makeInteger(outShape[outIdx]);
  }
  for (int64_t outIdx = leading; outIdx < outRank; ++outIdx) {
    int64_t inIdx = outIdx - leading;
    if (outShape[outIdx] == mlir::ShapedType::kDynamic) {
      outExprs[outIdx] = inExprs[inIdx];
    } else {
      outExprs[outIdx] = symBuilder.makeInteger(outShape[outIdx]);
    }
  }

  return SymbolAttrUtils::withSymbolicAttr(outType, builder, outExprs);
}

static mlir::FailureOr<llvm::SmallVector<int64_t>> getPermuteAxes(const mlir::OperationState &state, int64_t rank) {
  auto permAttr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(state.attributes.get("perm"));
  if (!permAttr || permAttr.size() != static_cast<size_t>(rank)) {
    return mlir::failure();
  }

  llvm::SmallVector<int64_t> permVals;
  permVals.reserve(permAttr.size());
  for (mlir::Attribute attr : permAttr) {
    auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr);
    if (!intAttr) {
      return mlir::failure();
    }
    permVals.push_back(intAttr.getInt());
  }
  return permVals;
}

static mlir::LogicalResult refinePermutedDim(llvm::ArrayRef<int64_t> inShape,
                                             llvm::SmallVectorImpl<int64_t> &refinedShape, int64_t outIdx,
                                             int64_t axis) {
  int64_t inDim = inShape[axis];
  int64_t outDim = refinedShape[outIdx];
  bool inDynamic = mlir::ShapedType::isDynamic(inDim);
  bool outDynamic = mlir::ShapedType::isDynamic(outDim);

  if (!inDynamic && outDynamic) {
    refinedShape[outIdx] = inDim;
    return mlir::success();
  }
  if (!inDynamic && !outDynamic && inDim != outDim) {
    return mlir::failure();
  }
  if (inDynamic && !outDynamic) {
    return mlir::failure();
  }
  return mlir::success();
}

static mlir::FailureOr<mlir::RankedTensorType> refinePermuteResultType(mlir::RankedTensorType inType,
                                                                       mlir::RankedTensorType outType,
                                                                       llvm::ArrayRef<int64_t> permVals) {
  llvm::SmallVector<int64_t> refinedShape(outType.getShape().begin(), outType.getShape().end());
  auto inShape = inType.getShape();
  int64_t rank = inType.getRank();
  llvm::SmallVector<bool> seen(static_cast<size_t>(rank), false);
  for (auto [outIdx, axis] : llvm::enumerate(permVals)) {
    if (axis < 0 || axis >= rank || seen[static_cast<size_t>(axis)]) {
      return mlir::failure();
    }
    seen[static_cast<size_t>(axis)] = true;

    if (mlir::failed(refinePermutedDim(inShape, refinedShape, outIdx, axis))) {
      return mlir::failure();
    }
  }

  return mlir::RankedTensorType::get(refinedShape, outType.getElementType(), outType.getEncoding());
}

mlir::FailureOr<mlir::Type> PermuteOp::inferSymbolicShapes(mlir::OpBuilder &builder, const mlir::OperationState &state,
                                                           mlir::Type resultType) {
  if (state.operands.size() != 1) {
    return mlir::failure();
  }

  auto inType = mlir::dyn_cast<mlir::RankedTensorType>(state.operands.front().getType());
  auto outType = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
  if (!outType || !inType || outType.getRank() != inType.getRank()) {
    return mlir::failure();
  }

  auto maybePermVals = getPermuteAxes(state, inType.getRank());
  if (mlir::failed(maybePermVals)) {
    return mlir::failure();
  }
  auto permVals = std::move(*maybePermVals);

  auto maybeOutExprs = SymbolAttrUtils::permuteSymbolicShapeExprs(inType, permVals);
  if (mlir::failed(maybeOutExprs)) {
    return mlir::failure();
  }

  auto maybeOutType = refinePermuteResultType(inType, outType, permVals);
  if (mlir::failed(maybeOutType)) {
    return mlir::failure();
  }
  return SymbolAttrUtils::withSymbolicAttr(*maybeOutType, builder, *maybeOutExprs);
}

mlir::LogicalResult PermuteOp::verify() {
  for (auto [idx, attr] : llvm::enumerate(getPermAttr().getValue())) {
    auto axis = mlir::cast<mlir::IntegerAttr>(attr).getInt();
    if (axis < 0) {
      return emitOpError("perm dimensions must be non-negative, got ") << axis << " at index " << idx;
    }
  }
  return mlir::success();
}

mlir::Type CastOp::inferResultType(mlir::Value input, mlir::Type elementType) {
  auto inType = llvm::dyn_cast<mlir::RankedTensorType>(input.getType());
  if (!inType) {
    return {};
  }
  return mlir::RankedTensorType::get(inType.getShape(), elementType, inType.getEncoding());
}

namespace {

static mlir::Type getRankedElementType(mlir::Type type) {
  auto rt = mlir::dyn_cast<mlir::RankedTensorType>(type);
  return rt ? rt.getElementType() : mlir::Type{};
}

/// True when cast(cast(x, Tmid), Tout) is equivalent to cast(x, Tout) without extra narrowing:
/// integer chains with matching signedness and non-decreasing bit width, or float chains
/// with non-decreasing width where equal-width steps use the same element type (e.g. no f16 vs bf16 mix).
static bool canComposeWideningCasts(mlir::Type elemIn, mlir::Type elemMid, mlir::Type elemOut) {
  auto iIn = mlir::dyn_cast<mlir::IntegerType>(elemIn);
  auto iMid = mlir::dyn_cast<mlir::IntegerType>(elemMid);
  auto iOut = mlir::dyn_cast<mlir::IntegerType>(elemOut);
  if (iIn && iMid && iOut) {
    if (iIn.getSignedness() != iMid.getSignedness() || iMid.getSignedness() != iOut.getSignedness()) {
      return false;
    }
    return iIn.getWidth() <= iMid.getWidth() && iMid.getWidth() <= iOut.getWidth();
  }
  auto fIn = mlir::dyn_cast<mlir::FloatType>(elemIn);
  auto fMid = mlir::dyn_cast<mlir::FloatType>(elemMid);
  auto fOut = mlir::dyn_cast<mlir::FloatType>(elemOut);
  if (fIn && fMid && fOut) {
    const unsigned wIn = fIn.getWidth();
    const unsigned wMid = fMid.getWidth();
    const unsigned wOut = fOut.getWidth();
    if (wIn > wMid || wMid > wOut) {
      return false;
    }
    if (wIn == wMid && elemIn != elemMid) {
      return false;
    }
    if (wMid == wOut && elemMid != elemOut) {
      return false;
    }
    return true;
  }
  return false;
}

/// Read mfuse.permute's I64 array attribute into `out` (Torch-style: output dim i comes from input dim perm[i]).
static bool readPermI64(PermuteOp op, llvm::SmallVectorImpl<int64_t> &out) {
  auto permAttr = op.getPermAttr();
  if (!permAttr) {
    return false;
  }
  out.clear();
  for (mlir::Attribute a : permAttr.getValue()) {
    auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a);
    if (!ia) {
      return false;
    }
    out.push_back(ia.getInt());
  }
  return true;
}

/// True when `perm` lists each axis in [0, rank) exactly once (valid Torch-style perm).
static bool isValidPermutation(llvm::ArrayRef<int64_t> perm, int64_t rank) {
  if (rank < 0 || perm.size() != static_cast<size_t>(rank)) {
    return false;
  }
  llvm::SmallVector<bool> seen(rank, false);
  for (int64_t ax : perm) {
    if (ax < 0 || ax >= rank || seen[static_cast<size_t>(ax)]) {
      return false;
    }
    seen[static_cast<size_t>(ax)] = true;
  }
  return true;
}

static bool isIdentityPermutation(llvm::ArrayRef<int64_t> perm) {
  for (auto [i, ax] : llvm::enumerate(perm)) {
    if (ax != static_cast<int64_t>(i)) {
      return false;
    }
  }
  return true;
}

}  // namespace

namespace {

/// Try to commute cast with permute: cast(permute(x)) -> permute(cast(x))
static mlir::LogicalResult tryCommuteWithPermute(CastOp op, mlir::PatternRewriter &rewriter) {
  mlir::Value input = op.getInput();
  mlir::Value outerRes = op.getResult();
  auto permOp = mlir::dyn_cast<PermuteOp>(input.getDefiningOp());
  if (!permOp) {
    return mlir::failure();
  }

  llvm::SmallVector<int64_t, 8> permVals;
  if (!readPermI64(permOp, permVals)) {
    return mlir::failure();
  }

  mlir::Value x = permOp.getInput();
  auto xTy = mlir::dyn_cast<mlir::RankedTensorType>(x.getType());
  auto castOutTy = mlir::dyn_cast<mlir::RankedTensorType>(outerRes.getType());
  if (!xTy || !castOutTy || xTy.getRank() != castOutTy.getRank()) {
    return mlir::failure();
  }

  const int64_t rank = xTy.getRank();
  if (!isValidPermutation(permVals, rank)) {
    return mlir::failure();
  }

  mlir::Type elemOut = castOutTy.getElementType();
  mlir::Type castOnXTy = CastOp::inferResultType(x, elemOut);
  auto castOnXRT = mlir::dyn_cast_or_null<mlir::RankedTensorType>(castOnXTy);
  if (!castOnXRT || castOnXRT.getRank() != rank) {
    return mlir::failure();
  }

  mlir::Value castX = rewriter.create<CastOp>(op.getLoc(), x, elemOut).getResult();
  rewriter.replaceOpWithNewOp<PermuteOp>(op, castOutTy, castX, permOp.getPermAttr());
  if (permOp->use_empty()) {
    rewriter.eraseOp(permOp);
  }
  return mlir::success();
}

/// Try to compose cast with inner cast: cast(cast(x)) -> cast(x)
static mlir::LogicalResult tryComposeWithInnerCast(CastOp op, mlir::PatternRewriter &rewriter) {
  mlir::Value input = op.getInput();
  mlir::Value outerRes = op.getResult();
  auto innerCast = mlir::dyn_cast<CastOp>(input.getDefiningOp());
  if (!innerCast) {
    return mlir::failure();
  }

  mlir::Value innerIn = innerCast.getInput();
  mlir::Value innerRes = innerCast.getResult();
  mlir::Type elemIn = getRankedElementType(innerIn.getType());
  mlir::Type elemMid = getRankedElementType(innerRes.getType());
  mlir::Type elemOut = getRankedElementType(outerRes.getType());
  if (!elemIn || !elemMid || !elemOut) {
    return mlir::failure();
  }

  // Round-trip: result type equals x (narrowing in the middle is never covered by compose rule).
  if (innerIn.getType() == outerRes.getType()) {
    rewriter.replaceOp(op, innerIn);
    if (innerCast->use_empty()) {
      rewriter.eraseOp(innerCast);
    }
    return mlir::success();
  }

  // Redundant outer: inner result tensor type already equals outer result (e.g. f32→f16→f16).
  if (innerRes.getType() == outerRes.getType()) {
    rewriter.replaceOp(op, innerRes);
    return mlir::success();
  }

  // Widening-only composition: cast(cast(x, T1), T2) → cast(x, T2).
  if (canComposeWideningCasts(elemIn, elemMid, elemOut)) {
    auto fused = rewriter.create<CastOp>(op.getLoc(), innerIn, elemOut);
    rewriter.replaceOp(op, fused.getResult());
    if (innerCast->use_empty()) {
      rewriter.eraseOp(innerCast);
    }
    return mlir::success();
  }
  return mlir::failure();
}

}  // namespace

mlir::LogicalResult CastOp::canonicalize(CastOp op, mlir::PatternRewriter &rewriter) {
  mlir::Value input = op.getInput();
  if (!input.getDefiningOp()) {
    return mlir::failure();
  }

  // Try to commute cast with permute
  if (mlir::succeeded(tryCommuteWithPermute(op, rewriter))) {
    return mlir::success();
  }

  // Try to compose cast with inner cast
  if (mlir::succeeded(tryComposeWithInnerCast(op, rewriter))) {
    return mlir::success();
  }

  return mlir::failure();
}

mlir::OpFoldResult CastOp::fold(FoldAdaptor adaptor) {
  auto input = getInput();
  if (input.getType() == getResult().getType()) {
    return input;
  }

  if (auto cst = input.getDefiningOp<mlir::mfuse::ConstantOp>()) {
    auto resultType = mlir::cast<mlir::RankedTensorType>(getResult().getType());
    auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
    if (!dense) return {};

    auto srcType = dense.getElementType();
    auto dstType = resultType.getElementType();

    // Use the helper function to fold constant cast
    auto newDenseOpt = foldConstantCast(dense, srcType, dstType, resultType);
    if (!newDenseOpt) return {};

    mlir::OpBuilder b(getOperation());
    auto newCst = b.create<mlir::mfuse::ConstantOp>(getLoc(), resultType, *newDenseOpt);
    return newCst.getResult();
  } else if (auto cst = input.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    // Fold through UnrealizedConversionCast using the helper function
    mlir::OpBuilder b(getOperation());
    auto newCastOpt = foldUnrealizedConversionCast(b, cst, getResult().getType(), getLoc());
    if (newCastOpt) {
      return *newCastOpt;
    }
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

mlir::FailureOr<mlir::Type> ReshapeOp::inferSymbolicShapes(mlir::OpBuilder &builder, const mlir::OperationState &state,
                                                           mlir::Type resultType) {
  if (state.operands.empty()) {
    return mlir::failure();
  }

  auto inType = mlir::dyn_cast<mlir::RankedTensorType>(state.operands[0].getType());
  auto outType = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
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

// Canonicalize reshape(reshape(a, shape1), shape2) -> reshape(a, shape2)
mlir::LogicalResult ReshapeOp::canonicalize(ReshapeOp op, mlir::PatternRewriter &rewriter) {
  auto innerReshape = op.getInput().getDefiningOp<ReshapeOp>();
  if (!innerReshape) {
    return mlir::failure();
  }

  // Create a new ReshapeOp that directly reshapes the inner ReshapeOp's input to the outer shape
  rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getResult().getType(), innerReshape.getInput());
  return mlir::success();
}

// Canonicalize reciprocal(sqrt(x)) -> rsqrt(x)
mlir::LogicalResult ReciprocalOp::canonicalize(ReciprocalOp op, mlir::PatternRewriter &rewriter) {
  auto sqrtOp = op.getInput().getDefiningOp<SqrtOp>();
  if (!sqrtOp || !sqrtOp->hasOneUse()) {
    return mlir::failure();
  }

  auto rsqrt = rewriter.create<RsqrtOp>(op.getLoc(), op.getResult().getType(), sqrtOp.getInput());
  rewriter.replaceOp(op, rsqrt.getResult());
  return mlir::success();
}

// Canonicalize: drop identity permute; fuse permute(permute(x, inner), outer) -> permute(x, fused) or x.
mlir::LogicalResult PermuteOp::canonicalize(PermuteOp op, mlir::PatternRewriter &rewriter) {
  llvm::SmallVector<int64_t, 8> outerPerm;
  if (!readPermI64(op, outerPerm)) {
    return mlir::failure();
  }

  auto inTy = mlir::dyn_cast<mlir::RankedTensorType>(op.getInput().getType());
  auto outTy = mlir::dyn_cast<mlir::RankedTensorType>(op.getResult().getType());
  if (!inTy || !outTy) {
    return mlir::failure();
  }
  const int64_t rank = inTy.getRank();
  if (rank != outTy.getRank() || !isValidPermutation(outerPerm, rank)) {
    return mlir::failure();
  }

  if (isIdentityPermutation(outerPerm)) {
    rewriter.replaceOp(op, op.getInput());
    return mlir::success();
  }

  auto inner = op.getInput().getDefiningOp<PermuteOp>();
  if (!inner) {
    return mlir::failure();
  }

  llvm::SmallVector<int64_t, 8> innerPerm;
  if (!readPermI64(inner, innerPerm)) {
    return mlir::failure();
  }
  auto innerInTy = mlir::dyn_cast<mlir::RankedTensorType>(inner.getInput().getType());
  auto innerOutTy = mlir::dyn_cast<mlir::RankedTensorType>(inner.getResult().getType());
  if (!innerInTy || !innerOutTy || innerInTy.getRank() != rank || innerOutTy.getRank() != rank) {
    return mlir::failure();
  }
  if (!isValidPermutation(innerPerm, rank)) {
    return mlir::failure();
  }

  llvm::SmallVector<int64_t, 8> fused;
  fused.reserve(static_cast<size_t>(rank));
  for (int64_t i = 0; i < rank; ++i) {
    fused.push_back(innerPerm[static_cast<size_t>(outerPerm[static_cast<size_t>(i)])]);
  }
  if (!isValidPermutation(fused, rank)) {
    return mlir::failure();
  }

  mlir::Value root = inner.getInput();
  if (isIdentityPermutation(fused)) {
    rewriter.replaceOp(op, root);
    if (inner->use_empty()) {
      rewriter.eraseOp(inner);
    }
    return mlir::success();
  }

  rewriter.replaceOpWithNewOp<PermuteOp>(op, op.getResult().getType(), root, rewriter.getI64ArrayAttr(fused));
  if (inner->use_empty()) {
    rewriter.eraseOp(inner);
  }
  return mlir::success();
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

  bool rhsIsScalar = false;
  if (auto rhsEnc = mlir::dyn_cast<mlir::RankedTensorType>(type1).getEncoding()) {
    if (auto dictAttr = mlir::dyn_cast<mlir::DictionaryAttr>(rhsEnc)) {
      rhsIsScalar = dictAttr.contains(mlir::mfuse::kScalarMarkerAttr);
    }
  }

  // Determine the target high-precision element type
  mlir::Type higherElemType = getHigherPrecisionType(elemType0, elemType1, rhsIsScalar);

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

mlir::Type ReduceMeanOp::inferResultType(mlir::Value input, mlir::ArrayAttr dimensions, mlir::BoolAttr keepdim,
                                         mlir::Type elementType) {
  return ReduceOpCommonInfer::inferResultType(input, dimensions, keepdim, elementType);
}

mlir::FailureOr<mlir::Type> ReduceMeanOp::inferSymbolicShapes(mlir::OpBuilder &builder,
                                                              const mlir::OperationState &state,
                                                              mlir::Type resultType) {
  return ReduceOpCommonInfer::inferSymbolicShapes(builder, state, resultType);
}

mlir::LogicalResult ReduceMeanOp::verify() {
  if (failed(verifyReduceDimensions(getOperation(), getDimensions()))) {
    return mlir::failure();
  }

  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(getResult().getType());
  if (!resultType || !resultType.hasRank()) {
    return emitOpError("result must be a ranked tensor");
  }

  if (!mlir::isa<mlir::FloatType>(resultType.getElementType())) {
    return emitOpError("result element type must be floating point");
  }
  return mlir::success();
}

mlir::Type ReduceSumOp::inferResultType(mlir::Value input, mlir::ArrayAttr dimensions, mlir::BoolAttr keepdim,
                                        mlir::Type elementType) {
  return ReduceOpCommonInfer::inferResultType(input, dimensions, keepdim, elementType);
}

mlir::FailureOr<mlir::Type> ReduceSumOp::inferSymbolicShapes(mlir::OpBuilder &builder,
                                                             const mlir::OperationState &state, mlir::Type resultType) {
  return ReduceOpCommonInfer::inferSymbolicShapes(builder, state, resultType);
}

mlir::LogicalResult ReduceSumOp::verify() {
  auto dimensions = getDimensions();
  for (auto dimAttr : dimensions.getValue()) {
    auto dim = mlir::cast<mlir::IntegerAttr>(dimAttr).getValue().getSExtValue();
    if (dim < 0) {
      return emitOpError("dimensions must be non-negative, got ") << dim;
    }
  }
  return mlir::success();
}

mlir::LogicalResult NumToTensorOp::verify() {
  // Verify that the input is a numeric type or a scalar tensor of numeric type
  auto inputType = getValue().getType();
  bool isNumericType = false;
  if (mlir::isa<mlir::FloatType>(inputType) || mlir::isa<mlir::IntegerType>(inputType)) {
    isNumericType = true;
  } else if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(inputType)) {
    if (tensorType.getRank() == 0) {  // It's a scalar tensor
      auto elementType = tensorType.getElementType();
      if (mlir::isa<mlir::FloatType>(elementType) || mlir::isa<mlir::IntegerType>(elementType)) {
        isNumericType = true;
      }
    }
  }
  if (!isNumericType) {
    return emitOpError("input must be a numeric type or a scalar tensor of numeric type, got ") << inputType;
  }
  // Verify that the result is a tensor type
  auto resultType = getResult().getType();
  if (!mlir::isa<mlir::RankedTensorType>(resultType)) {
    return emitOpError("result must be a ranked tensor type, got ") << resultType;
  }
  return mlir::success();
}

}  // namespace mlir::mfuse
