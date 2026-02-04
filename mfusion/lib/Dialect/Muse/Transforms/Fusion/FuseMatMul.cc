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

#include "mfusion/Dialect/Muse/Transforms/Fusion/FuseMatMul.h"

#include <numeric>

#include "mfusion/Dialect/Muse/Muse.h"
#include "mfusion/Dialect/Muse/Transforms/Passes.h"
#include "mfusion/Dialect/Muse/Utils/ArithUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace muse {
namespace {

// Constants for dimension and alignment checks
constexpr int kRank1D = 1;           // Rank for 1D tensors
constexpr int kRank2D = 2;           // Minimum rank for transpose operations
constexpr int kRank0D = 0;           // Rank for scalar/rank-0 tensors
constexpr int kBytesPerByte = 8;     // Bits per byte
constexpr int kBitWidthRound = 7;    // Rounding factor for bit width to byte conversion
constexpr int kPermStartIndex = 0;   // Starting index for permutation array
constexpr int kPermSecondLast = 2;   // Offset for second-to-last dimension
constexpr int kPermLast = 1;         // Offset for last dimension
constexpr int kUnsqueezeDim = 1;      // Dimension value for unsqueeze operation
constexpr int kSmallVectorDefaultSize = 4;  // Default size for SmallVector
constexpr int kBiasSizeInit = 1;      // Initial value for bias size calculation

//===----------------------------------------------------------------------===//
// Helper functions for common matmul fusion logic
//===----------------------------------------------------------------------===//

/// Check if a value has a single CastOp user that casts f16 to f32.
/// Returns the CastOp if found, nullptr otherwise.
static CastOp getF16ToF32Cast(Value value) {
  if (!value.hasOneUse()) {
    return nullptr;
  }
  auto castOp = dyn_cast<CastOp>(*value.user_begin());
  if (!castOp) {
    return nullptr;
  }
  auto castDtype = dyn_cast_or_null<FloatType>(castOp.getDtype());
  if (!castDtype || !castDtype.isF32()) {
    return nullptr;
  }
  auto inType = dyn_cast<RankedTensorType>(value.getType());
  if (!inType || !isa<Float16Type>(inType.getElementType())) {
    return nullptr;
  }
  return castOp;
}

/// Create a 1D tensor Value (shape) from ArrayRef<int64_t> for muse.reshape.
static Value createShapeValue(PatternRewriter &rewriter, Location loc,
                               ArrayRef<int64_t> shape) {
  auto tensorType = RankedTensorType::get(
      {static_cast<int64_t>(shape.size())},
      rewriter.getIntegerType(64));
  auto attr =
      mlir::DenseElementsAttr::get(tensorType, llvm::ArrayRef<int64_t>(shape));
  return rewriter.create<arith::ConstantOp>(loc, attr);
}

/// Compute reshape shapes for 1D inputs: self (1D) -> [1, N], other (1D) -> [N, 1].
static void computeReshapeShapesFor1D(RankedTensorType selfType, RankedTensorType otherType,
                                      SmallVectorImpl<int64_t> &newShapeSelf,
                                      SmallVectorImpl<int64_t> &newShapeOther) {
  int rSelf = selfType.getRank();
  int rOther = otherType.getRank();
  auto shapeSelf = selfType.getShape();
  auto shapeOther = otherType.getShape();

  if (rSelf == kRank1D) {
    newShapeSelf.clear();
    newShapeSelf.push_back(kUnsqueezeDim);
    newShapeSelf.push_back(shapeSelf[0]);
  } else {
    newShapeSelf.assign(shapeSelf.begin(), shapeSelf.end());
  }
  if (rOther == kRank1D) {
    newShapeOther.clear();
    newShapeOther.push_back(shapeOther[0]);
    newShapeOther.push_back(kUnsqueezeDim);
  } else {
    newShapeOther.assign(shapeOther.begin(), shapeOther.end());
  }
}

/// Create a permute op that swaps the last two dimensions.
/// Used for transpose operations to align memory layout for matmul operations.
static Value createLastTwoDimsSwapPermute(PatternRewriter &rewriter, Location loc,
                                          RankedTensorType inputType, Value input) {
  int rank = inputType.getRank();
  SmallVector<int64_t> perm(rank);
  std::iota(perm.begin(), perm.end(), kPermStartIndex);
  std::swap(perm[rank - kPermSecondLast], perm[rank - kPermLast]);
  SmallVector<int64_t> outShape(rank);
  for (size_t i = 0; i < perm.size(); ++i) {
    outShape[i] = inputType.getShape()[perm[i]];
  }
  auto outType = RankedTensorType::get(outShape, inputType.getElementType());
  return rewriter.create<PermuteOp>(loc, outType, input, rewriter.getI64ArrayAttr(perm));
}

/// Helper to get result value from aclnn matmul-like ops.
//===----------------------------------------------------------------------===//
// FuseMatMulCast: MatMul/Mm (f16) -> Cast (f32) => MatMul/Mm with f32 result
//===----------------------------------------------------------------------===//

/// Pattern to fuse MatMul followed by f16->f32 cast into MatMul with f32 output.
/// This eliminates redundant cast operations by computing matmul directly in f32.
class FuseMatMulCastMatmulPattern : public OpRewritePattern<MatmulOp> {
 public:
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp matmulOp, PatternRewriter &rewriter) const override {
    auto castOp = getF16ToF32Cast(matmulOp.getResult());
    if (!castOp) {
      return failure();
    }

    Type outType = castOp.getResult().getType();
    Value newMatmul = rewriter.create<MatmulOp>(
        matmulOp.getLoc(), outType, matmulOp.getSelf(), matmulOp.getOther(),
        matmulOp.getTransX1Attr(), matmulOp.getTransX2Attr());
    rewriter.replaceOp(castOp, newMatmul);
    rewriter.eraseOp(matmulOp);
    return success();
  }
};

/// Pattern to fuse MatMulWithBias followed by f16->f32 cast into MatMulWithBias with f32 output.
class FuseMatMulCastMatmulWithBiasPattern : public OpRewritePattern<MatmulWithBiasOp> {
 public:
  using OpRewritePattern<MatmulWithBiasOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulWithBiasOp op, PatternRewriter &rewriter) const override {
    auto castOp = getF16ToF32Cast(op.getResult());
    if (!castOp) {
      return failure();
    }

    Type outType = castOp.getResult().getType();
    Value newMatmul = rewriter.create<MatmulWithBiasOp>(
        op.getLoc(), outType, op.getSelf(), op.getOther(), op.getBias(),
        op.getTransX1Attr(), op.getTransX2Attr());
    rewriter.replaceOp(castOp, newMatmul);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// FuseMatmulUnsqueezeSqueeze: 1D inputs => reshape before/after
//===----------------------------------------------------------------------===//

/// Pattern to handle 1D tensor inputs for matmul by inserting reshape operations.
/// Reshapes 1D inputs to 2D (e.g., [N] -> [1, N] or [N, 1]) before matmul, then reshapes back.
class FuseMatmulUnsqueezeSqueezeMatmulPattern : public OpRewritePattern<MatmulOp> {
 public:
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp matmulOp, PatternRewriter &rewriter) const override {
    auto selfType = dyn_cast<RankedTensorType>(matmulOp.getSelf().getType());
    auto otherType = dyn_cast<RankedTensorType>(matmulOp.getOther().getType());
    if (!selfType || !otherType) {
      return failure();
    }

    int rSelf = selfType.getRank();
    int rOther = otherType.getRank();
    if (rSelf != kRank1D && rOther != kRank1D) {
      return failure();
    }

    Value self = matmulOp.getSelf();
    Value other = matmulOp.getOther();
    Location loc = matmulOp.getLoc();

    SmallVector<int64_t, kSmallVectorDefaultSize> newShapeSelf, newShapeOther;
    computeReshapeShapesFor1D(selfType, otherType, newShapeSelf, newShapeOther);

    auto newSelfType = RankedTensorType::get(newShapeSelf, selfType.getElementType());
    auto newOtherType = RankedTensorType::get(newShapeOther, otherType.getElementType());
    Value newSelf = rewriter.create<ReshapeOp>(loc, newSelfType, self, createShapeValue(rewriter, loc, newShapeSelf));
    Value newOther = rewriter.create<ReshapeOp>(loc, newOtherType, other, createShapeValue(rewriter, loc, newShapeOther));

    Type resultType = matmulOp.getResult().getType();
    auto resultTensorType = dyn_cast<RankedTensorType>(resultType);
    if (!resultTensorType) {
      return failure();
    }
    Value matmulResult = rewriter.create<MatmulOp>(
        loc, resultTensorType, newSelf, newOther,
        matmulOp.getTransX1Attr(), matmulOp.getTransX2Attr());
    Value outReshape = rewriter.create<ReshapeOp>(
        loc, resultType, matmulResult, createShapeValue(rewriter, loc, resultTensorType.getShape()));
    rewriter.replaceOp(matmulOp, outReshape);
    return success();
  }
};

/// Pattern to handle 1D tensor inputs for MatMulWithBias by inserting reshape operations.
class FuseMatmulUnsqueezeSqueezeMatmulWithBiasPattern : public OpRewritePattern<MatmulWithBiasOp> {
 public:
  using OpRewritePattern<MatmulWithBiasOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulWithBiasOp matmulOp, PatternRewriter &rewriter) const override {
    auto selfType = dyn_cast<RankedTensorType>(matmulOp.getSelf().getType());
    auto otherType = dyn_cast<RankedTensorType>(matmulOp.getOther().getType());
    if (!selfType || !otherType) {
      return failure();
    }

    int rSelf = selfType.getRank();
    int rOther = otherType.getRank();
    if (rSelf != kRank1D && rOther != kRank1D) {
      return failure();
    }

    Value self = matmulOp.getSelf();
    Value other = matmulOp.getOther();
    Location loc = matmulOp.getLoc();

    SmallVector<int64_t, kSmallVectorDefaultSize> newShapeSelf, newShapeOther;
    computeReshapeShapesFor1D(selfType, otherType, newShapeSelf, newShapeOther);

    auto newSelfType = RankedTensorType::get(newShapeSelf, selfType.getElementType());
    auto newOtherType = RankedTensorType::get(newShapeOther, otherType.getElementType());
    Value newSelf = rewriter.create<ReshapeOp>(loc, newSelfType, self, createShapeValue(rewriter, loc, newShapeSelf));
    Value newOther = rewriter.create<ReshapeOp>(loc, newOtherType, other, createShapeValue(rewriter, loc, newShapeOther));

    Type resultType = matmulOp.getResult().getType();
    auto resultTensorType = dyn_cast<RankedTensorType>(resultType);
    if (!resultTensorType) {
      return failure();
    }
    Value matmulResult = rewriter.create<MatmulWithBiasOp>(
        loc, resultTensorType, newSelf, newOther, matmulOp.getBias(),
        matmulOp.getTransX1Attr(), matmulOp.getTransX2Attr());
    Value outReshape = rewriter.create<ReshapeOp>(
        loc, resultType, matmulResult, createShapeValue(rewriter, loc, resultTensorType.getShape()));
    rewriter.replaceOp(matmulOp, outReshape);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// FuseMatmulTransposeWeight: insert permute when inner axis not 512B aligned
//===----------------------------------------------------------------------===//

constexpr int64_t kAlignBytes = 512;

/// Check if the inner (last) dimension of a tensor is 512-byte aligned.
/// Returns true if aligned, false otherwise. Used for memory alignment optimization.
static bool isInnerAxisAligned(RankedTensorType type) {
  if (!type || type.getRank() == kRank0D) {
    return true;
  }
  int64_t lastDim = type.getShape().back();
  int64_t elemSize = (type.getElementType().getIntOrFloatBitWidth() + kBitWidthRound) / kBytesPerByte;
  return (lastDim * elemSize) % kAlignBytes == 0;
}

/// Check if a value is the output of a PermuteOp and trans is true.
/// Used to avoid redundant permute operations.
static bool isPermuteOutputWithTrans(Value v, bool trans) {
  if (!trans) {
    return false;
  }
  auto permuteOp = v.getDefiningOp<PermuteOp>();
  return permuteOp != nullptr;
}

/// Pattern to insert permute operations when inner axis is not 512-byte aligned.
/// This optimizes memory access patterns for matmul operations.
class FuseMatmulTransposeWeightMatmulPattern : public OpRewritePattern<MatmulOp> {
 public:
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp op, PatternRewriter &rewriter) const override {
    Value self = op.getSelf();
    Value other = op.getOther();
    auto selfType = dyn_cast<RankedTensorType>(self.getType());
    auto otherType = dyn_cast<RankedTensorType>(other.getType());
    if (!selfType || !otherType) {
      return failure();
    }

    bool transX1 = op.getTransX1();
    bool transX2 = op.getTransX2();
    if (isPermuteOutputWithTrans(self, transX1) || isPermuteOutputWithTrans(other, transX2)) {
      return failure();
    }

    bool needTrans1 = !isInnerAxisAligned(selfType) && !transX1;
    bool needTrans2 = !isInnerAxisAligned(otherType) && !transX2;
    if (!needTrans1 && !needTrans2) {
      return failure();
    }

    Location loc = op.getLoc();
    Value newSelf = self;
    Value newOther = other;
    bool newTransX1 = transX1;
    bool newTransX2 = transX2;

    if (needTrans1 && selfType.getRank() >= kRank2D) {
      newSelf = createLastTwoDimsSwapPermute(rewriter, loc, selfType, self);
      newTransX1 = true;
    }
    if (needTrans2 && otherType.getRank() >= kRank2D) {
      newOther = createLastTwoDimsSwapPermute(rewriter, loc, otherType, other);
      newTransX2 = true;
    }

    Value newOp = rewriter.create<MatmulOp>(loc, op.getResult().getType(), newSelf, newOther,
                                            rewriter.getBoolAttr(newTransX1), rewriter.getBoolAttr(newTransX2));
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

/// Pattern to insert permute operations for MatMulWithBias when inner axis is not aligned.
class FuseMatmulTransposeWeightMatmulWithBiasPattern : public OpRewritePattern<MatmulWithBiasOp> {
 public:
  using OpRewritePattern<MatmulWithBiasOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulWithBiasOp op, PatternRewriter &rewriter) const override {
    Value self = op.getSelf();
    Value other = op.getOther();
    auto selfType = dyn_cast<RankedTensorType>(self.getType());
    auto otherType = dyn_cast<RankedTensorType>(other.getType());
    if (!selfType || !otherType) {
      return failure();
    }

    bool transX1 = op.getTransX1();
    bool transX2 = op.getTransX2();
    if (isPermuteOutputWithTrans(self, transX1) || isPermuteOutputWithTrans(other, transX2)) {
      return failure();
    }

    bool needTrans1 = !isInnerAxisAligned(selfType) && !transX1;
    bool needTrans2 = !isInnerAxisAligned(otherType) && !transX2;
    if (!needTrans1 && !needTrans2) {
      return failure();
    }

    Location loc = op.getLoc();
    Value newSelf = self;
    Value newOther = other;
    bool newTransX1 = transX1;
    bool newTransX2 = transX2;

    if (needTrans1 && selfType.getRank() >= kRank2D) {
      newSelf = createLastTwoDimsSwapPermute(rewriter, loc, selfType, self);
      newTransX1 = true;
    }
    if (needTrans2 && otherType.getRank() >= kRank2D) {
      newOther = createLastTwoDimsSwapPermute(rewriter, loc, otherType, other);
      newTransX2 = true;
    }

    Value newOp = rewriter.create<MatmulWithBiasOp>(loc, op.getResult().getType(), newSelf, newOther, op.getBias(),
                                                   rewriter.getBoolAttr(newTransX1), rewriter.getBoolAttr(newTransX2));
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// FuseMatmulReshapeBiasAdd: MatMul (with or without bias) -> Reshape -> Add(bias)
//   => MatMulWithBias -> Reshape
//===----------------------------------------------------------------------===//

/// Pattern to fuse MatMul -> Reshape -> Add into MatMulWithBias -> Reshape.
/// This eliminates the intermediate Add operation by incorporating bias into matmul.
class FuseMatmulReshapeBiasAddPattern : public OpRewritePattern<AddOp> {
 public:
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp addOp, PatternRewriter &rewriter) const override {
    Value addLhs = addOp.getX();
    Value addRhs = addOp.getY();
    Value reshapeOut = nullptr;
    Value addBias = nullptr;
    
    bool lhsIsReshape = addLhs.getDefiningOp<ReshapeOp>() != nullptr;
    bool rhsIsReshape = addRhs.getDefiningOp<ReshapeOp>() != nullptr;
    
    // Exactly one input must be a Reshape output, and the other must not be a Reshape.
    if (lhsIsReshape && !rhsIsReshape) {
      reshapeOut = addLhs;
      addBias = addRhs;
    } else if (!lhsIsReshape && rhsIsReshape) {
      reshapeOut = addRhs;
      addBias = addLhs;
    } else {
      // Both are Reshape or neither is Reshape - not a valid pattern.
      return failure();
    }

    auto reshapeOp = reshapeOut.getDefiningOp<ReshapeOp>();
    Value matmulOut = reshapeOp.getInput();

    auto matmulType = dyn_cast<RankedTensorType>(matmulOut.getType());
    auto reshapeType = dyn_cast<RankedTensorType>(reshapeOut.getType());
    auto biasType = dyn_cast<RankedTensorType>(addBias.getType());
    if (!matmulType || !reshapeType || !biasType) {
      return failure();
    }

    if (matmulType.getRank() < kRank1D || reshapeType.getRank() < kRank1D) {
      return failure();
    }
    int64_t lastMatmul = matmulType.getShape().back();
    SmallVector<int64_t> biasShape(biasType.getShape());
    int64_t biasSize = kBiasSizeInit;
    for (int64_t s : biasShape) {
      biasSize *= s;
    }
    // Bias must have the same size as matmul's last dimension (broadcast along last axis).
    if (biasSize != lastMatmul) {
      return failure();
    }

    Location loc = addOp.getLoc();
    Type newMatmulResultType = matmulType;
    Value fusedBias = addBias;
    Operation *matmulDef = matmulOut.getDefiningOp();

    if (auto matmulOp = dyn_cast<MatmulOp>(matmulDef)) {
      // MatmulOp -> Reshape -> Add => MatmulWithBiasOp -> Reshape
      Value newMatmul = rewriter.create<MatmulWithBiasOp>(
          loc, newMatmulResultType, matmulOp.getSelf(), matmulOp.getOther(), fusedBias,
          matmulOp.getTransX1Attr(), matmulOp.getTransX2Attr());
      Value newReshape = rewriter.create<ReshapeOp>(
          reshapeOp.getLoc(), reshapeOp.getResult().getType(), newMatmul, reshapeOp.getShape());
      rewriter.replaceOp(addOp, newReshape);
      rewriter.eraseOp(reshapeOp);
      rewriter.eraseOp(matmulOp);
      return success();
    }

    if (auto matmulWithBiasOp = dyn_cast<MatmulWithBiasOp>(matmulDef)) {
      // MatmulWithBiasOp -> Reshape -> Add => MatmulWithBiasOp(..., old_bias + add_bias) -> Reshape
      Value oldBias = matmulWithBiasOp.getBias();
      fusedBias = rewriter.create<AddOp>(loc, oldBias.getType(), oldBias, addBias);
      Value newMatmul = rewriter.create<MatmulWithBiasOp>(
          loc, newMatmulResultType, matmulWithBiasOp.getSelf(), matmulWithBiasOp.getOther(),
          fusedBias, matmulWithBiasOp.getTransX1Attr(), matmulWithBiasOp.getTransX2Attr());
      Value newReshape = rewriter.create<ReshapeOp>(
          reshapeOp.getLoc(), reshapeOp.getResult().getType(), newMatmul, reshapeOp.getShape());
      rewriter.replaceOp(addOp, newReshape);
      rewriter.eraseOp(reshapeOp);
      rewriter.eraseOp(matmulWithBiasOp);
      return success();
    }

    return failure();
  }
};

}  // namespace

// Include generated pass definitions
#define GEN_PASS_DEF_FUSEMATMULCAST
#include "mfusion/Dialect/Muse/Transforms/Passes.h.inc"

struct FuseMatMulCastPass : public impl::FuseMatMulCastBase<FuseMatMulCastPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseMatMulCastMatmulPattern, FuseMatMulCastMatmulWithBiasPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

#undef GEN_PASS_DEF_FUSEMATMULCAST
#define GEN_PASS_DEF_FUSEMATMULUNSQUEEZESQUEEZE
#include "mfusion/Dialect/Muse/Transforms/Passes.h.inc"

struct FuseMatmulUnsqueezeSqueezePass : public impl::FuseMatmulUnsqueezeSqueezeBase<FuseMatmulUnsqueezeSqueezePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseMatmulUnsqueezeSqueezeMatmulPattern, FuseMatmulUnsqueezeSqueezeMatmulWithBiasPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

#undef GEN_PASS_DEF_FUSEMATMULUNSQUEEZESQUEEZE
#define GEN_PASS_DEF_FUSEMATMULTRANSPOSEWEIGHT
#include "mfusion/Dialect/Muse/Transforms/Passes.h.inc"

struct FuseMatmulTransposeWeightPass : public impl::FuseMatmulTransposeWeightBase<FuseMatmulTransposeWeightPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseMatmulTransposeWeightMatmulPattern, FuseMatmulTransposeWeightMatmulWithBiasPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

#undef GEN_PASS_DEF_FUSEMATMULTRANSPOSEWEIGHT
#define GEN_PASS_DEF_FUSEMATMULRESHAPEBIASADD
#include "mfusion/Dialect/Muse/Transforms/Passes.h.inc"

struct FuseMatmulReshapeBiasAddPass : public impl::FuseMatmulReshapeBiasAddBase<FuseMatmulReshapeBiasAddPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseMatmulReshapeBiasAddPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

#undef GEN_PASS_DEF_FUSEMATMULRESHAPEBIASADD

}  // namespace muse

std::unique_ptr<Pass> createFuseMatMulCastPass() {
  return std::make_unique<muse::FuseMatMulCastPass>();
}

std::unique_ptr<Pass> createFuseMatmulUnsqueezeSqueezePass() {
  return std::make_unique<muse::FuseMatmulUnsqueezeSqueezePass>();
}

std::unique_ptr<Pass> createFuseMatmulTransposeWeightPass() {
  return std::make_unique<muse::FuseMatmulTransposeWeightPass>();
}

std::unique_ptr<Pass> createFuseMatmulReshapeBiasAddPass() {
  return std::make_unique<muse::FuseMatmulReshapeBiasAddPass>();
}

}  // namespace mlir
