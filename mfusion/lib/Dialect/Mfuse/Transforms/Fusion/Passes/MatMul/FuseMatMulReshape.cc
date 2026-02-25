//===----------------------------------------------------------------------===//
//
// Copyright 2026 Huawei Technologies Co., Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/MatMul/FuseMatMulReshape.h"

#include "mfusion/Dialect/Mfuse/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/FusionPassMacros.h"
#include "mfusion/Dialect/Mfuse/Utils/ArithUtils.h"
#include "mfusion/Dialect/Mfuse/Utils/OpConstants.h"
#include "mfusion/Support/Logging.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

#define GEN_PASS_DEF_FUSEMATMULRESHAPE
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {
namespace {

// Constants for fusion conditions
// FP16/BF16 reshape fusion is only applied when K < 27392 (hardware/backend limit).
constexpr int64_t kMaxKForFp16 = 27391;   // K < 27392
constexpr int64_t kNSize = 1;             // N axis size must be 1
constexpr int64_t kMinRankForMatmul = 2;  // Matmul requires at least rank 2 (2D matrices)
constexpr int64_t kRank2D = 2;            // Rank for 2D tensors

/// Helper function to check if the data type is supported
static bool isSupportedDataType(mlir::Type type) { return type.isBF16() || type.isF16() || type.isF32(); }

/// Helper function to check if the data type requires K size check
static bool requiresKSizeCheck(mlir::Type type) { return type.isBF16() || type.isF16(); }

/// Common matching logic for MatmulOp and MatmulWithBiasOp reshape fusion.
/// Checks if N=1, K size is valid, and data types are supported.
template <typename MatmulOpType>
static bool matchMatMulReshapeCommon(MatmulOpType op, int64_t &nDimSize, int64_t &K) {
  auto selfType = mlir::dyn_cast<mlir::RankedTensorType>(op.getSelf().getType());
  auto otherType = mlir::dyn_cast<mlir::RankedTensorType>(op.getOther().getType());

  if (!selfType || !otherType) {
    return false;
  }

  if (selfType.getRank() != kRank2D || otherType.getRank() != kRank2D) {
    return false;
  }

  // Check N dimension
  if (op.getTransX2()) {
    nDimSize = otherType.getShape()[0];
  } else {
    nDimSize = otherType.getShape()[1];
  }
  if (nDimSize != kNSize) {
    return false;
  }

  // Check data types
  auto selfElementType = selfType.getElementType();
  auto otherElementType = otherType.getElementType();
  if (!isSupportedDataType(selfElementType) || !isSupportedDataType(otherElementType)) {
    return false;
  }

  // Get K size
  if (op.getTransX1()) {
    K = selfType.getShape()[0];
  } else {
    K = selfType.getShape()[1];
  }
  if (K <= 0) {
    return false;
  }

  // K size check for FP16/BF16
  if (requiresKSizeCheck(selfElementType) || requiresKSizeCheck(otherElementType)) {
    if (K > kMaxKForFp16) {
      return false;
    }
  }

  // Check for existing reshape to avoid loops
  if (op.getOther().template getDefiningOp<mfuse::ReshapeOp>() != nullptr) {
    return false;
  }

  return true;
}

/// Helper function to create a reshape operation for the second input.
///
/// Note: This function creates a reshape even if the shape is unchanged.
/// Some hardware backends may require explicit reshape operations for
/// alignment or other reasons. If optimization is desired, consider checking
/// if reshape is actually needed before calling this function.
///
/// \param input The input value to reshape (must be rank-2)
/// \param loc Location for the new operations
/// \param rewriter Pattern rewriter
/// \returns Reshaped value (or original if input is not rank-2)
static mlir::Value createReshapeForSecondInput(mlir::Value input, mlir::Location loc, mlir::PatternRewriter &rewriter) {
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  if (!inputType || inputType.getRank() != kRank2D) {
    return input;  // Return unchanged if not rank-2 tensor
  }

  auto shape = inputType.getShape();
  if (shape.size() < kMinRankForMatmul) {
    return input;  // Safety check
  }

  // Create reshape operation
  // Even if shape is unchanged, this reshape may be required by backend
  return rewriter.create<mfuse::ReshapeOp>(loc, inputType, input);
}

/// Pattern to fuse MatMul with N=1 to add Reshape to second input
class FuseMatMulReshapePattern : public mlir::OpRewritePattern<mfuse::MatmulOp> {
 public:
  using OpRewritePattern<mfuse::MatmulOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mfuse::MatmulOp op, mlir::PatternRewriter &rewriter) const override {
    int64_t nDimSize, K;
    if (!matchMatMulReshapeCommon(op, nDimSize, K)) {
      return mlir::failure();
    }

    // Create reshape for the second input
    mlir::Value reshapedOther = createReshapeForSecondInput(op.getOther(), op.getLoc(), rewriter);

    // Create new MatmulOp with reshaped second input
    auto newMatmul = rewriter.create<mfuse::MatmulOp>(op.getLoc(), op.getResult().getType(), op.getSelf(),
                                                      reshapedOther, op.getTransX1Attr(), op.getTransX2Attr());

    // Replace the original operation
    rewriter.replaceOp(op, newMatmul.getResult());
    MLOG(DEBUG) << "FuseMatMulReshape: replaced MatmulOp@" << op.getLoc() << " with reshaped second input (N=1)";
    return mlir::success();
  }
};

/// Pattern to fuse MatmulWithBiasOp with N=1 to add Reshape to second input
class FuseMatMulWithBiasReshapePattern : public mlir::OpRewritePattern<mfuse::MatmulWithBiasOp> {
 public:
  using OpRewritePattern<mfuse::MatmulWithBiasOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mfuse::MatmulWithBiasOp op, mlir::PatternRewriter &rewriter) const override {
    int64_t nDimSize, K;
    if (!matchMatMulReshapeCommon(op, nDimSize, K)) {
      return mlir::failure();
    }

    // Create reshape for the second input
    mlir::Value reshapedOther = createReshapeForSecondInput(op.getOther(), op.getLoc(), rewriter);

    // Create new MatmulWithBiasOp with reshaped second input
    auto newMatmulWithBias =
      rewriter.create<mfuse::MatmulWithBiasOp>(op.getLoc(), op.getResult().getType(), op.getSelf(), reshapedOther,
                                               op.getBias(), op.getTransX1Attr(), op.getTransX2Attr());

    // Replace the original operation
    rewriter.replaceOp(op, newMatmulWithBias.getResult());
    MLOG(DEBUG) << "FuseMatMulReshape: replaced MatmulWithBiasOp@" << op.getLoc()
                << " with reshaped second input (N=1)";
    return mlir::success();
  }
};

}  // namespace

DEFINE_MFUSE_FUSION_PASS(FuseMatMulReshape, FuseMatMulReshapePattern, FuseMatMulWithBiasReshapePattern)
}  // namespace mfuse
}  // namespace mlir
