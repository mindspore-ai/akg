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

#include "mfusion/Dialect/Mfuse/Transforms/Recompose/RecomposePatterns.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::mfuse {
namespace {

constexpr int64_t kRankBatchMatmul = 3;  // rank >= 3 -> aclnn.batch_matmul
constexpr int64_t kAlphaOne = 1;

/// Shared helpers for matmul_with_bias (bias reshape only).
class MfuseMetaOpsToAclnnPattern {
 protected:
  static constexpr int64_t kRank2DVal = 2;

  /// Reshape bias to be compatible with result shape for broadcasting.
  static Value reshapeBiasForBroadcast(Value bias, RankedTensorType biasType, RankedTensorType resultType,
                                       PatternRewriter &rewriter) {
    Location loc = bias.getLoc();
    auto resultShape = resultType.getShape();
    auto biasShape = biasType.getShape();
    int64_t resultRank = resultShape.size();
    int64_t biasRank = biasShape.size();

    if (resultRank == biasRank && resultShape == biasShape) {
      return bias;
    }

    SmallVector<int64_t> broadcastShape(resultRank);
    int64_t leadingOnes = resultRank - biasRank;
    for (int64_t i = 0; i < leadingOnes; ++i) {
      broadcastShape[i] = 1;
    }
    for (int64_t i = 0; i < biasRank; ++i) {
      broadcastShape[leadingOnes + i] = biasShape[i];
    }

    auto broadcastType = RankedTensorType::get(broadcastShape, biasType.getElementType());
    return rewriter.create<ReshapeOp>(loc, broadcastType, bias);
  }
};

/// Lower mfuse.matmul to aclnn.matmul, aclnn.mm (2D), or aclnn.batch_matmul.
/// Transpose flags are carried on aclnn ops (no mfuse.permute).
class MatmulToAclnnPattern : public OpRewritePattern<MatmulOp>, public MfuseMetaOpsToAclnnPattern {
 public:
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp op, PatternRewriter &rewriter) const override {
    auto selfType = dyn_cast<RankedTensorType>(op.getSelf().getType());
    auto otherType = dyn_cast<RankedTensorType>(op.getOther().getType());
    if (!selfType || !otherType || !selfType.hasRank() || !otherType.hasRank()) {
      return failure();
    }

    Location loc = op.getLoc();
    Value self = op.getSelf();
    Value other = op.getOther();
    auto trans1 = rewriter.getBoolAttr(op.getTransX1());
    auto trans2 = rewriter.getBoolAttr(op.getTransX2());

    Type resultType = op.getResult().getType();
    int rank = selfType.getRank();
    Value result;
    if (rank == kRank2DVal && otherType.getRank() == kRank2DVal) {
      result = rewriter.create<AclnnMmOp>(loc, resultType, self, other, trans1, trans2);
    } else if (rank >= kRankBatchMatmul && otherType.getRank() >= kRankBatchMatmul) {
      result = rewriter.create<AclnnBatchMatmulOp>(loc, resultType, self, other, trans1, trans2);
    } else {
      result = rewriter.create<AclnnMatmulOp>(loc, resultType, self, other, trans1, trans2);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Lower mfuse.matmul_with_bias to aclnn.matmul/aclnn.mm/batch_matmul + aclnn.add.
class MfuseMetaOpsMatMulWithBiasToAclnnPattern : public OpRewritePattern<MatmulWithBiasOp>,
                                                 public MfuseMetaOpsToAclnnPattern {
 public:
  using OpRewritePattern<MatmulWithBiasOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulWithBiasOp op, PatternRewriter &rewriter) const override {
    auto selfType = dyn_cast<RankedTensorType>(op.getSelf().getType());
    auto otherType = dyn_cast<RankedTensorType>(op.getOther().getType());
    if (!selfType || !otherType || !selfType.hasRank() || !otherType.hasRank()) {
      return failure();
    }

    Location loc = op.getLoc();
    Value self = op.getSelf();
    Value other = op.getOther();
    auto trans1 = rewriter.getBoolAttr(op.getTransX1());
    auto trans2 = rewriter.getBoolAttr(op.getTransX2());
    Value bias = op.getBias();

    Type resultType = op.getResult().getType();
    Value matmulResult;
    int rank = selfType.getRank();
    if (rank == kRank2DVal && otherType.getRank() == kRank2DVal) {
      matmulResult = rewriter.create<AclnnMmOp>(loc, resultType, self, other, trans1, trans2);
    } else if (rank >= kRankBatchMatmul && otherType.getRank() >= kRankBatchMatmul) {
      matmulResult = rewriter.create<AclnnBatchMatmulOp>(loc, resultType, self, other, trans1, trans2);
    } else {
      matmulResult = rewriter.create<AclnnMatmulOp>(loc, resultType, self, other, trans1, trans2);
    }

    auto matmulResultType = dyn_cast<RankedTensorType>(matmulResult.getType());
    auto biasType = dyn_cast<RankedTensorType>(bias.getType());
    if (!matmulResultType || !biasType) {
      return failure();
    }

    Value biasForAdd = bias;
    if (matmulResultType.getShape() != biasType.getShape()) {
      biasForAdd = reshapeBiasForBroadcast(bias, biasType, matmulResultType, rewriter);
    }

    auto scalarType = RankedTensorType::get({}, rewriter.getI64Type());
    Value alphaOne = rewriter.create<mfuse::ConstantOp>(loc, scalarType, DenseElementsAttr::get(scalarType, kAlphaOne));
    Value addResult = rewriter.create<AclnnAddOp>(loc, resultType, matmulResult, biasForAdd, alphaOne);
    rewriter.replaceOp(op, addResult);
    return success();
  }
};

}  // namespace

void registerMfuseMetaOpsToAclnnPatterns(RewritePatternSet &patterns) {
  patterns.add<MatmulToAclnnPattern, MfuseMetaOpsMatMulWithBiasToAclnnPattern>(patterns.getContext());
}

}  // namespace mlir::mfuse
