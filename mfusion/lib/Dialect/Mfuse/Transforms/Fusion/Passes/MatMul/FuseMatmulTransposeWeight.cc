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

#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/MatMul/FuseMatmulTransposeWeight.h"

#include <numeric>

#include "mfusion/Dialect/Mfuse/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Utils/ArithUtils.h"
#include "mfusion/Dialect/Mfuse/Utils/OpConstants.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/FusionPassMacros.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_FUSEMATMULTRANSPOSEWEIGHT
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {
namespace {

// Constants for dimension and alignment checks
constexpr int kBytesPerByte = 8;    // Bits per byte
constexpr int kBitWidthRound = 7;   // Rounding factor for bit width to byte conversion
constexpr int kPermStartIndex = 0;  // Starting index for permutation array
constexpr int kPermSecondLast = 2;  // Offset for second-to-last dimension
constexpr int kPermLast = 1;        // Offset for last dimension
constexpr int64_t kAlignBytes = 512;

/// Check if the inner (last) dimension of a tensor is 512-byte aligned.
/// Returns true if aligned, false otherwise. Used for memory alignment optimization.
static bool isInnerAxisAligned(RankedTensorType type) {
  if (!type || type.getRank() == kDim0) {
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

/// Create a permute op that swaps the last two dimensions.
/// Used for transpose operations to align memory layout for matmul operations.
static Value createLastTwoDimsSwapPermute(PatternRewriter &rewriter, Location loc, RankedTensorType inputType,
                                          Value input) {
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

    if (needTrans1 && selfType.getRank() >= kDim2) {
      newSelf = createLastTwoDimsSwapPermute(rewriter, loc, selfType, self);
      newTransX1 = true;
    }
    if (needTrans2 && otherType.getRank() >= kDim2) {
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

    if (needTrans1 && selfType.getRank() >= kDim2) {
      newSelf = createLastTwoDimsSwapPermute(rewriter, loc, selfType, self);
      newTransX1 = true;
    }
    if (needTrans2 && otherType.getRank() >= kDim2) {
      newOther = createLastTwoDimsSwapPermute(rewriter, loc, otherType, other);
      newTransX2 = true;
    }

    Value newOp = rewriter.create<MatmulWithBiasOp>(loc, op.getResult().getType(), newSelf, newOther, op.getBias(),
                                                    rewriter.getBoolAttr(newTransX1), rewriter.getBoolAttr(newTransX2));
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

}  // namespace

DEFINE_MFUSE_FUSION_PASS(FuseMatmulTransposeWeight, FuseMatmulTransposeWeightMatmulPattern,
                        FuseMatmulTransposeWeightMatmulWithBiasPattern)

}  // namespace mfuse

}  // namespace mlir
