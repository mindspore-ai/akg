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

#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/MatMul/FuseMatmulUnsqueezeSqueeze.h"

#include <numeric>

#include "mfusion/Dialect/Mfuse/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Utils/ArithUtils.h"
#include "mfusion/Dialect/Mfuse/Utils/OpConstants.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/FusionPassMacros.h"
#include "mfusion/Support/Logging.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_FUSEMATMULUNSQUEEZESQUEEZE
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {
namespace {

// Constants for dimension and alignment checks
constexpr int kPermStartIndex = 0;          // Starting index for permutation array
constexpr int kPermSecondLast = 2;          // Offset for second-to-last dimension
constexpr int kPermLast = 1;                // Offset for last dimension
constexpr int kUnsqueezeDim = 1;            // Dimension value for unsqueeze operation
constexpr int kSmallVectorDefaultSize = 4;  // Default size for SmallVector

/// Compute reshape shapes for 1D inputs: self (1D) -> [1, N], other (1D) -> [N, 1].
static void computeReshapeShapesFor1D(RankedTensorType selfType, RankedTensorType otherType,
                                      SmallVectorImpl<int64_t> &newShapeSelf, SmallVectorImpl<int64_t> &newShapeOther) {
  int rSelf = selfType.getRank();
  int rOther = otherType.getRank();
  auto shapeSelf = selfType.getShape();
  auto shapeOther = otherType.getShape();

  if (rSelf == kDim1) {
    newShapeSelf.clear();
    newShapeSelf.push_back(kUnsqueezeDim);
    newShapeSelf.push_back(shapeSelf[0]);
  } else {
    newShapeSelf.assign(shapeSelf.begin(), shapeSelf.end());
  }
  if (rOther == kDim1) {
    newShapeOther.clear();
    newShapeOther.push_back(shapeOther[0]);
    newShapeOther.push_back(kUnsqueezeDim);
  } else {
    newShapeOther.assign(shapeOther.begin(), shapeOther.end());
  }
}

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
    if (rSelf != kDim1 && rOther != kDim1) {
      return failure();
    }

    // Log pattern match
    MLOG(DEBUG) << "FuseMatmulUnsqueezeSqueezeMatmulPattern matched MatmulOp with 1D input(s), self rank=" << rSelf
                << ", other rank=" << rOther;

    Value self = matmulOp.getSelf();
    Value other = matmulOp.getOther();
    Location loc = matmulOp.getLoc();

    SmallVector<int64_t, kSmallVectorDefaultSize> newShapeSelf, newShapeOther;
    computeReshapeShapesFor1D(selfType, otherType, newShapeSelf, newShapeOther);

    // Log reshape shapes
    std::string selfShapeStr, otherShapeStr;
    for (int64_t dim : newShapeSelf) {
      selfShapeStr += std::to_string(dim) + " ";
    }
    for (int64_t dim : newShapeOther) {
      otherShapeStr += std::to_string(dim) + " ";
    }
    MLOG(DEBUG) << "Computed reshape shapes: self=" << selfShapeStr << ", other=" << otherShapeStr;

    auto newSelfType = RankedTensorType::get(newShapeSelf, selfType.getElementType());
    auto newOtherType = RankedTensorType::get(newShapeOther, otherType.getElementType());
    Value newSelf = rewriter.create<ReshapeOp>(loc, newSelfType, self);
    Value newOther = rewriter.create<ReshapeOp>(loc, newOtherType, other);

    // Log reshape operation creation
    MLOG(DEBUG) << "Created reshape operations for 1D inputs";

    Type resultType = matmulOp.getResult().getType();
    auto resultTensorType = dyn_cast<RankedTensorType>(resultType);
    if (!resultTensorType) {
      return failure();
    }
    Value matmulResult = rewriter.create<MatmulOp>(loc, resultTensorType, newSelf, newOther, matmulOp.getTransX1Attr(),
                                                   matmulOp.getTransX2Attr());
    Value outReshape = rewriter.create<ReshapeOp>(loc, resultType, matmulResult);

    // Log output reshape creation
    MLOG(DEBUG) << "Created output reshape operation";
    rewriter.replaceOp(matmulOp, outReshape);

    // Log operation replacement
    MLOG(DEBUG) << "Replaced original MatmulOp with reshaped version";

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
    if (rSelf != kDim1 && rOther != kDim1) {
      return failure();
    }

    Value self = matmulOp.getSelf();
    Value other = matmulOp.getOther();
    Location loc = matmulOp.getLoc();

    SmallVector<int64_t, kSmallVectorDefaultSize> newShapeSelf, newShapeOther;
    computeReshapeShapesFor1D(selfType, otherType, newShapeSelf, newShapeOther);

    auto newSelfType = RankedTensorType::get(newShapeSelf, selfType.getElementType());
    auto newOtherType = RankedTensorType::get(newShapeOther, otherType.getElementType());
    Value newSelf = rewriter.create<ReshapeOp>(loc, newSelfType, self);
    Value newOther = rewriter.create<ReshapeOp>(loc, newOtherType, other);

    Type resultType = matmulOp.getResult().getType();
    auto resultTensorType = dyn_cast<RankedTensorType>(resultType);
    if (!resultTensorType) {
      return failure();
    }
    Value matmulResult = rewriter.create<MatmulWithBiasOp>(loc, resultTensorType, newSelf, newOther, matmulOp.getBias(),
                                                           matmulOp.getTransX1Attr(), matmulOp.getTransX2Attr());
    Value outReshape = rewriter.create<ReshapeOp>(loc, resultType, matmulResult);
    rewriter.replaceOp(matmulOp, outReshape);
    return success();
  }
};

}  // namespace

DEFINE_MFUSE_FUSION_PASS(FuseMatmulUnsqueezeSqueeze, FuseMatmulUnsqueezeSqueezeMatmulPattern,
                         FuseMatmulUnsqueezeSqueezeMatmulWithBiasPattern)

}  // namespace mfuse

}  // namespace mlir
