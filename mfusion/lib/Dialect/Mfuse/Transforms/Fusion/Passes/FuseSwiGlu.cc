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

#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/FuseSwiGlu.h"

#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Support/ArithUtils.h"
#include "mfusion/Dialect/Mfuse/Support/OpConstants.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/FusionPassMacros.h"
#include "mfusion/Support/Logging.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_FUSESWIGLU
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {

/**
 * @brief Fuse SplitWithSize, Silu and Mul into SwiGlu.
 * @example
 * %0 = SplitWithSize(%input, %splitSize, %dim)
 * %1 = TupleGetItem(%0, 0)
 * %2 = TupleGetItem(%0, 1)
 * %3 = Silu(%2)
 * %4 = Mul(%1, %3)
 * return %4
 * --->
 * %0 = SwiGlu(%input, %dim)
 * return %0
 */
class FuseSwiGluPattern : public OpRewritePattern<MulOp> {
 public:
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp mulOp, PatternRewriter &rewriter) const override {
    Value mulLhs = mulOp.getLhs();
    Value mulRhs = mulOp.getRhs();

    // Match pattern: one operand is Silu op, the other comes from SplitWithSize
    AclnnSiluOp siluOp;
    AclnnSplitWithSizeOp splitOp;
    if (!matchCommutativeOperands(mulLhs, mulRhs, siluOp, splitOp)) {
      return failure();
    }

    // Verify split has exactly 2 results and proper connections
    if (splitOp.getNumResults() != kOutputSize2) {
      return failure();
    }
    // Silu should be applied to split's second output
    if (siluOp.getSelf() != splitOp.getResult(kIndex1)) {
      return failure();
    }
    // Mul's other operand should be split's first output
    Value nonSiluOperand = (mulLhs == siluOp.getResult()) ? mulRhs : mulLhs;
    if (splitOp.getResult(kIndex0) != nonSiluOperand) {
      return failure();
    }

    MLOG(DEBUG) << "FuseSwiGluPattern matched MulOp, fusing with SplitWithSize and Silu";

    // Create SwiGlu op
    auto swiGluOp =
      rewriter.create<AclnnSwiGluOp>(mulOp.getLoc(), mulOp.getResult().getType(), splitOp.getSelf(), splitOp.getDim());
    MLOG(DEBUG) << "Created new AclnnSwiGluOp";

    rewriter.replaceOp(mulOp, swiGluOp.getResult());
    MLOG(DEBUG) << "Replaced original MulOp, SiluOp and SplitWithSizeOp with new AclnnSwiGluOp";
    return success();
  }
};

/**
 * @brief Fuse Reshape, SplitWithSize, Reshape, Silu and Mul into SwiGlu.
 * @example
 * %0 = Reshape(%input, %shape)
 * %1 = SplitWithSize(%0, %splitSize, %dim)
 * %2 = TupleGetItem(%1, 0)
 * %3 = TupleGetItem(%1, 1)
 * %4 = Reshape(%3, %shape1)
 * %5 = Reshape(%2, %shape2)
 * %6 = Silu(%5)
 * %7 = Mul(%4, %6)
 * return %7
 * --->
 * %0 = SwiGlu(%input, %dim)
 * return %0
 */
class FuseSwiGluWithReshapePattern : public OpRewritePattern<MulOp> {
 public:
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp mulOp, PatternRewriter &rewriter) const override {
    auto matchResult = MatchPattern(mulOp);
    if (failed(matchResult)) {
      return failure();
    }
    auto [originalInput, splitDim] = *matchResult;

    MLOG(DEBUG) << "FuseSwiGluWithReshapePattern matched MulOp, fusing with Reshape, SplitWithSize and Silu";

    // Create SwiGlu op
    auto swiGluOp =
      rewriter.create<AclnnSwiGluOp>(mulOp.getLoc(), mulOp.getResult().getType(), originalInput, splitDim);
    MLOG(DEBUG) << "Created new AclnnSwiGluOp";

    rewriter.replaceOp(mulOp, swiGluOp.getResult());
    MLOG(DEBUG) << "Replaced original MulOp, SiluOp, ReshapeOps and SplitWithSizeOp with new AclnnSwiGluOp";
    return success();
  }

 private:
  FailureOr<std::pair<Value, Value>> MatchPattern(MulOp mulOp) const {
    Value mulLhs = mulOp.getLhs();
    Value mulRhs = mulOp.getRhs();

    // Match Pattern: Mul(Reshape(SplitWithSize[1]), Silu(Reshape(SplitWithSize[0])))
    ReshapeOp reshapeOpLeft;
    AclnnSiluOp siluOp;
    if (!matchCommutativeOperands(mulLhs, mulRhs, reshapeOpLeft, siluOp)) {
      return failure();
    }

    // Match ReshapeOp from SiluOp
    auto reshapeOpRight = siluOp.getSelf().getDefiningOp<ReshapeOp>();
    if (!reshapeOpRight) {
      return failure();
    }
    Value splitOut0 = reshapeOpRight.getInput();
    Value splitOut1 = reshapeOpLeft.getInput();

    // Verify SplitWithSize has exactly 2 results
    auto splitOp = splitOut0.getDefiningOp<AclnnSplitWithSizeOp>();
    if (!splitOp || splitOp.getNumResults() != kOutputSize2) {
      return failure();
    }
    if (splitOp.getResult(kIndex0) != splitOut0 || splitOp.getResult(kIndex1) != splitOut1) {
      return failure();
    }

    // Restrict the input shape and dtype. This can be adjusted in the future according to actual needs.
    auto inputReshapeOp = splitOp.getSelf().getDefiningOp<ReshapeOp>();
    if (!inputReshapeOp) {
      return failure();
    }
    // Check if the reshape output has more than 3 dimensions
    auto reshapeOutputType = dyn_cast<RankedTensorType>(inputReshapeOp.getResult().getType());
    if (!reshapeOutputType || reshapeOutputType.getRank() > kDim3) {
      return failure();
    }
    Value originalInput = inputReshapeOp.getInput();
    auto inputType = dyn_cast<RankedTensorType>(originalInput.getType());
    if (!inputType) {
      return failure();
    }
    auto elemType = inputType.getElementType();
    if (!elemType.isF16() && !elemType.isBF16()) {
      return failure();
    }

    Value splitDim = splitOp.getDim();
    return std::make_pair(originalInput, splitDim);
  }
};

DEFINE_MFUSE_FUSION_PASS(FuseSwiGlu, FuseSwiGluPattern, FuseSwiGluWithReshapePattern)

}  // namespace mfuse

}  // namespace mlir
