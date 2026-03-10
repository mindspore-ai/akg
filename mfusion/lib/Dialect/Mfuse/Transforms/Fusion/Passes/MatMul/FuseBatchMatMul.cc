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

// Relationship with decompose (AclnnOps.cc):
// Decompose converts aclnn.batch_matmul -> mfuse.matmul (trans_x1=false, trans_x2=false).
// - If decompose runs before this pass: the IR contains mfuse.matmul only, so patterns that
//   match mfuse.batch_matmul (FuseBatchMatMulTransposeBatchMatmulPattern,
//   FuseBatchMatMulToMatmulPattern) will NOT be triggered; only the pattern that matches
//   mfuse.matmul (FuseBatchMatMulTransposeMatmulPattern) can trigger.
// - If decompose does not run (e.g. no aclnn.batch_matmul in the pipeline): mfuse.batch_matmul
//   may exist, and the BatchMatmulOp patterns above will be triggered when applicable.

#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/MatMul/FuseBatchMatMul.h"

#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/FusionPassMacros.h"
#include "mfusion/Dialect/Mfuse/Support/OpConstants.h"
#include "mfusion/Support/Logging.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_FUSEBATCHMATMUL
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {
namespace {

/// When both BatchMatMul inputs are 2D, convert to MatMul (rank 2).
constexpr int64_t kRank2D = static_cast<int64_t>(kDim2);

/// Returns true if the permute only swaps the last two dimensions:
/// input (..., a, b) -> output (..., b, a); batch dims unchanged.
static bool isPermuteSwapLastTwoDims(PermuteOp permuteOp) {
  auto inputType = dyn_cast<RankedTensorType>(permuteOp.getInput().getType());
  if (!inputType) {
    return false;
  }
  int64_t rank = inputType.getRank();
  if (rank < kRank2D) {
    return false;
  }
  auto permAttr = permuteOp.getPermAttr();
  if (!permAttr) {
    return false;
  }
  auto permValues = permAttr.getValue();
  if (permValues.size() != static_cast<size_t>(rank)) {
    return false;
  }
  int64_t lastIdx = rank - 1;
  int64_t secondLastIdx = rank - 2;
  for (int64_t i = 0; i < rank; ++i) {
    auto intAttr = dyn_cast<IntegerAttr>(permValues[i]);
    if (!intAttr) {
      return false;
    }
    int64_t p = intAttr.getInt();
    if (i < secondLastIdx) {
      if (p != i) {
        return false;
      }
    } else if (i == secondLastIdx) {
      if (p != lastIdx) {
        return false;
      }
    } else {
      if (p != secondLastIdx) {
        return false;
      }
    }
  }
  return true;
}

/// Matmul input dtypes only support float16, float32, bfloat16.
static bool isSupportedMatmulDtype(Type type) {
  auto elemType = dyn_cast<FloatType>(type);
  if (!elemType) return false;
  return isa<Float16Type>(elemType) || elemType.isF32() || isa<BFloat16Type>(elemType);
}

/// Mode 1: Eliminate Permute (swap last two dims) into MatmulOp by using permute
/// input and flipping trans_x1/trans_x2.
class FuseBatchMatMulTransposeMatmulPattern : public OpRewritePattern<MatmulOp> {
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
    if (!isSupportedMatmulDtype(selfType.getElementType()) ||
        !isSupportedMatmulDtype(otherType.getElementType())) {
      return failure();
    }

    auto permSelf = self.getDefiningOp<PermuteOp>();
    auto permOther = other.getDefiningOp<PermuteOp>();
    bool selfIsPermute = permSelf && isPermuteSwapLastTwoDims(permSelf);
    bool otherIsPermute = permOther && isPermuteSwapLastTwoDims(permOther);
    if (!selfIsPermute && !otherIsPermute) {
      return failure();
    }

    MLOG(DEBUG) << "FuseBatchMatMulTransposeMatmulPattern matched MatmulOp@" << op.getLoc()
               << " (transpose elimination: selfIsPermute=" << selfIsPermute
               << ", otherIsPermute=" << otherIsPermute << ")";

    Value newSelf = selfIsPermute ? permSelf.getInput() : self;
    Value newOther = otherIsPermute ? permOther.getInput() : other;
    bool newTransX1 = op.getTransX1() ^ selfIsPermute;
    bool newTransX2 = op.getTransX2() ^ otherIsPermute;

    Value newMatmul = rewriter.create<MatmulOp>(op.getLoc(), op.getResult().getType(), newSelf, newOther,
                                                rewriter.getBoolAttr(newTransX1), rewriter.getBoolAttr(newTransX2));
    MLOG(DEBUG) << "FuseBatchMatMul: created MatmulOp@" << newMatmul.getDefiningOp()->getLoc()
                << " trans_x1=" << newTransX1 << " trans_x2=" << newTransX2;
    rewriter.replaceOp(op, newMatmul);
    MLOG(DEBUG) << "FuseBatchMatMul: replaced MatmulOp with new MatmulOp (transpose eliminated)";
    return success();
  }
};

/// Mode 1: Eliminate Permute into BatchMatmulOp by flipping transpose_a/transpose_b.
/// Note: If decompose (aclnn.batch_matmul -> mfuse.matmul) has run, this pattern is not triggered.
class FuseBatchMatMulTransposeBatchMatmulPattern : public OpRewritePattern<BatchMatmulOp> {
 public:
  using OpRewritePattern<BatchMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BatchMatmulOp op, PatternRewriter &rewriter) const override {
    Value self = op.getSelf();
    Value mat2 = op.getMat2();
    auto selfType = dyn_cast<RankedTensorType>(self.getType());
    auto mat2Type = dyn_cast<RankedTensorType>(mat2.getType());
    if (!selfType || !mat2Type) {
      return failure();
    }
    if (!isSupportedMatmulDtype(selfType.getElementType()) ||
        !isSupportedMatmulDtype(mat2Type.getElementType())) {
      return failure();
    }

    auto permSelf = self.getDefiningOp<PermuteOp>();
    auto permMat2 = mat2.getDefiningOp<PermuteOp>();
    bool selfIsPermute = permSelf && isPermuteSwapLastTwoDims(permSelf);
    bool mat2IsPermute = permMat2 && isPermuteSwapLastTwoDims(permMat2);
    if (!selfIsPermute && !mat2IsPermute) {
      return failure();
    }

    MLOG(DEBUG) << "FuseBatchMatMulTransposeBatchMatmulPattern matched BatchMatmulOp@" << op.getLoc()
               << " (transpose elimination: selfIsPermute=" << selfIsPermute
               << ", mat2IsPermute=" << mat2IsPermute << ")";

    Value newSelf = selfIsPermute ? permSelf.getInput() : self;
    Value newMat2 = mat2IsPermute ? permMat2.getInput() : mat2;
    bool newTransA = op.getTransposeA() ^ selfIsPermute;
    bool newTransB = op.getTransposeB() ^ mat2IsPermute;

    Value newBmm = rewriter.create<BatchMatmulOp>(op.getLoc(), op.getResult().getType(), newSelf, newMat2,
                                                  rewriter.getBoolAttr(newTransA), rewriter.getBoolAttr(newTransB));
    MLOG(DEBUG) << "FuseBatchMatMul: created BatchMatmulOp@" << newBmm.getDefiningOp()->getLoc()
                << " transpose_a=" << newTransA << " transpose_b=" << newTransB;
    rewriter.replaceOp(op, newBmm);
    MLOG(DEBUG) << "FuseBatchMatMul: replaced BatchMatmulOp with new BatchMatmulOp (transpose eliminated)";
    return success();
  }
};

/// Mode 2: When BatchMatMul both inputs are 2D, convert to MatMul.
/// When both BatchMatMul inputs are 2D, convert to MatMul (rank 2).
/// Note: If decompose (aclnn.batch_matmul -> mfuse.matmul) has run, this pattern is not triggered.
class FuseBatchMatMulToMatmulPattern : public OpRewritePattern<BatchMatmulOp> {
 public:
  using OpRewritePattern<BatchMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BatchMatmulOp op, PatternRewriter &rewriter) const override {
    Value self = op.getSelf();
    Value mat2 = op.getMat2();
    auto selfType = dyn_cast<RankedTensorType>(self.getType());
    auto mat2Type = dyn_cast<RankedTensorType>(mat2.getType());
    if (!selfType || !mat2Type) {
      return failure();
    }
    if (selfType.getRank() != kRank2D || mat2Type.getRank() != kRank2D) {
      return failure();
    }

    MLOG(DEBUG) << "FuseBatchMatMulToMatmulPattern matched BatchMatmulOp@" << op.getLoc()
               << " (both inputs 2D -> MatMul)";

    Value newMatmul = rewriter.create<MatmulOp>(
        op.getLoc(), op.getResult().getType(), self, mat2,
        rewriter.getBoolAttr(op.getTransposeA()), rewriter.getBoolAttr(op.getTransposeB()));
    MLOG(DEBUG) << "FuseBatchMatMul: created MatmulOp@" << newMatmul.getDefiningOp()->getLoc()
                << " (from BatchMatmul 2D)";
    rewriter.replaceOp(op, newMatmul);
    MLOG(DEBUG) << "FuseBatchMatMul: replaced BatchMatmulOp with MatmulOp";
    return success();
  }
};

}  // namespace

DEFINE_MFUSE_FUSION_PASS(FuseBatchMatMul, FuseBatchMatMulTransposeMatmulPattern,
                         FuseBatchMatMulTransposeBatchMatmulPattern, FuseBatchMatMulToMatmulPattern)

}  // namespace mfuse
}  // namespace mlir
