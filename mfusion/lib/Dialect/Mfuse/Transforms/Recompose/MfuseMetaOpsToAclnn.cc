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
#include <numeric>

#include "llvm/ADT/SmallVector.h"
#include "mfusion/Dialect/Mfuse/Transforms/Recompose/RecomposePatterns.h"
#include "mfusion/Dialect/Mfuse/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Utils/ArithUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::mfuse {
namespace {

constexpr int64_t kRank2D = 2;
constexpr int64_t kRankBatchMatmul = 3;  // rank >= 3 -> aclnn.batch_matmul
constexpr int64_t kPermStartIndex = 0;
constexpr int64_t kPermDimSecondLast = 2;  // index offset: rank - 2
constexpr int64_t kPermDimLast = 1;        // index offset: rank - 1
constexpr int64_t kAlphaOne = 1;

/// Base class for Mfuse meta op to aclnn patterns; provides shared trans/permute helpers.
class MfuseMetaOpsToAclnnPattern {
 protected:
  static constexpr int64_t kRank2DVal = 2;

  /// When trans is true, return perm that swaps the last two dimensions; else identity.
  static SmallVector<int64_t> permForTrans(int rank, bool trans) {
    SmallVector<int64_t> perm(rank);
    std::iota(perm.begin(), perm.end(), kPermStartIndex);
    if (trans && rank >= kRank2D) {
      std::swap(perm[rank - kPermDimSecondLast], perm[rank - kPermDimLast]);
    }
    return perm;
  }

  /// If trans is true, insert PermuteOp to swap last two dimensions and return new value; else return v.
  static Value applyTransToValue(Value v, RankedTensorType type, bool trans, PatternRewriter &rewriter) {
    if (!trans || type.getRank() < kRank2D) return v;
    SmallVector<int64_t> perm = permForTrans(type.getRank(), true);
    SmallVector<int64_t> outShape(type.getRank());
    for (size_t i = kPermStartIndex; i < perm.size(); ++i) {
      outShape[i] = type.getShape()[perm[i]];
    }
    auto outType = RankedTensorType::get(outShape, type.getElementType());
    return rewriter.create<PermuteOp>(v.getLoc(), outType, v, rewriter.getI64ArrayAttr(perm));
  }

  /// Check if two shapes are compatible for broadcasting (bias can broadcast to result).
  /// Returns true if shapes are compatible, false otherwise.
  static bool areShapesCompatibleForBroadcast(ArrayRef<int64_t> resultShape, ArrayRef<int64_t> biasShape) {
    if (resultShape == biasShape) {
      return true;
    }
    // Check if bias can broadcast to result shape (bias should match trailing dimensions).
    int64_t resultRank = resultShape.size();
    int64_t biasRank = biasShape.size();
    if (biasRank > resultRank) {
      return false;
    }
    // Check trailing dimensions match.
    for (int64_t i = 0; i < biasRank; ++i) {
      int64_t resultDim = resultShape[resultRank - biasRank + i];
      int64_t biasDim = biasShape[i];
      if (resultDim != biasDim && resultDim != ShapedType::kDynamic && biasDim != ShapedType::kDynamic) {
        return false;
      }
    }
    return true;
  }

  /// Reshape bias to be compatible with result shape for broadcasting.
  /// If bias shape already matches trailing dimensions, insert Reshape to add leading 1s.
  static Value reshapeBiasForBroadcast(Value bias, RankedTensorType biasType, RankedTensorType resultType,
                                       PatternRewriter &rewriter) {
    Location loc = bias.getLoc();
    auto resultShape = resultType.getShape();
    auto biasShape = biasType.getShape();
    int64_t resultRank = resultShape.size();
    int64_t biasRank = biasShape.size();

    // If ranks are equal and shapes match, no reshape needed.
    if (resultRank == biasRank && resultShape == biasShape) {
      return bias;
    }

    // Build broadcast shape: [1, ..., 1, bias_shape...]
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

/// Lower mfuse.matmul to aclnn.matmul, aclnn.mm (2D), or aclnn.batch_matmul. Inherits base for trans helpers.
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
    Value self = applyTransToValue(op.getSelf(), selfType, op.getTransX1(), rewriter);
    Value other = applyTransToValue(op.getOther(), otherType, op.getTransX2(), rewriter);
    if (self != op.getSelf()) selfType = cast<RankedTensorType>(self.getType());
    if (other != op.getOther()) otherType = cast<RankedTensorType>(other.getType());

    Type resultType = op.getResult().getType();
    int rank = selfType.getRank();
    Value result;
    if (rank == kRank2DVal && otherType.getRank() == kRank2DVal) {
      result = rewriter.create<AclnnMmOp>(loc, resultType, self, other);
    } else if (rank >= kRankBatchMatmul && otherType.getRank() >= kRankBatchMatmul) {
      result = rewriter.create<AclnnBatchMatmulOp>(loc, resultType, self, other);
    } else {
      result = rewriter.create<AclnnMatmulOp>(loc, resultType, self, other);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Lower mfuse.matmul_with_bias to aclnn.matmul/aclnn.mm/batch_matmul + aclnn.add. Inherits base for trans helpers.
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
    Value self = applyTransToValue(op.getSelf(), selfType, op.getTransX1(), rewriter);
    Value other = applyTransToValue(op.getOther(), otherType, op.getTransX2(), rewriter);
    if (self != op.getSelf()) selfType = cast<RankedTensorType>(self.getType());
    if (other != op.getOther()) otherType = cast<RankedTensorType>(other.getType());
    Value bias = op.getBias();

    Type resultType = op.getResult().getType();
    Value matmulResult;
    int rank = selfType.getRank();
    if (rank == kRank2DVal && otherType.getRank() == kRank2DVal) {
      matmulResult = rewriter.create<AclnnMmOp>(loc, resultType, self, other);
    } else if (rank >= kRankBatchMatmul && otherType.getRank() >= kRankBatchMatmul) {
      matmulResult = rewriter.create<AclnnBatchMatmulOp>(loc, resultType, self, other);
    } else {
      matmulResult = rewriter.create<AclnnMatmulOp>(loc, resultType, self, other);
    }

    // Check if matmulResult and bias have compatible shapes for Add operation.
    auto matmulResultType = dyn_cast<RankedTensorType>(matmulResult.getType());
    auto biasType = dyn_cast<RankedTensorType>(bias.getType());
    if (!matmulResultType || !biasType) {
      return failure();
    }

    // Reshape bias if shapes are not identical to ensure shape compatibility for broadcasting.
    Value biasForAdd = bias;
    if (matmulResultType.getShape() != biasType.getShape()) {
      // Reshape bias to be compatible with matmulResult's shape for broadcasting.
      biasForAdd = reshapeBiasForBroadcast(bias, biasType, matmulResultType, rewriter);
    }

    auto scalarType = RankedTensorType::get({}, rewriter.getI64Type());
    Value alphaOne = rewriter.create<arith::ConstantOp>(loc, scalarType, DenseElementsAttr::get(scalarType, kAlphaOne));
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
