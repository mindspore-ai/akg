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

constexpr int64_t kRank2DVal = 2;
constexpr int64_t kRankBatchMatmul = 3;  // rank >= 3 -> aclnn.batch_matmul

/// Lower mfuse.matmul to aclnn.matmul, aclnn.mm (2D), or aclnn.batch_matmul.
/// Transpose flags are carried on aclnn ops (no mfuse.permute).
class MatmulToAclnnPattern : public OpRewritePattern<MatmulOp> {
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

}  // namespace

void registerMfuseMetaOpsToAclnnPatterns(RewritePatternSet &patterns) {
  patterns.add<MatmulToAclnnPattern>(patterns.getContext());
}

}  // namespace mlir::mfuse
