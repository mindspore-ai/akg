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

#include "mfusion/Dialect/Muse/Muse.h"
#include "mfusion/Dialect/Muse/Transforms/Passes.h"
#include "mfusion/Dialect/Muse/Utils/ArithUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTACLNNMATMULTOMATMUL
#include "mfusion/Dialect/Muse/Transforms/Passes.h.inc"

namespace muse {
namespace {

/// All aclnn matmul-like ops (mm/matmul/batch_matmul) use getOut().
template <typename Op>
static Value getMatmulResult(Op op) {
  return op.getOut();
}

/// Try to find aclnn matmul-like op (mm/matmul/batch_matmul) from add operands.
/// Returns the matmul op and sets bias to the other operand.
template <typename AclnnMatmulOp>
static Operation *findAclnnMatmulFromAddOperands(Value x, Value y, Value &bias) {
  if (auto mm = x.getDefiningOp<AclnnMatmulOp>()) {
    if (getMatmulResult(mm).hasOneUse()) {
      bias = y;
      return mm.getOperation();
    }
  }
  if (auto mm = y.getDefiningOp<AclnnMatmulOp>()) {
    if (getMatmulResult(mm).hasOneUse()) {
      bias = x;
      return mm.getOperation();
    }
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// ConvertAclnnMatmulToMatmul: aclnn.mm / aclnn.matmul / aclnn.batch_matmul =>
// muse.matmul (trans_x1=false, trans_x2=false)
//===----------------------------------------------------------------------===//

/// Template pattern to convert aclnn matmul-like ops (mm/matmul/batch_matmul)
/// to muse.matmul. This allows aclnn-specific operations to be converted to
/// generic muse operations for further fusion.
template <typename AclnnMatmulLikeOp>
class ConvertAclnnMatmulLikeToMatMulPattern : public OpRewritePattern<AclnnMatmulLikeOp> {
 public:
  using OpRewritePattern<AclnnMatmulLikeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AclnnMatmulLikeOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    Value mat2 = op.getMat2();
    Type resultType = getMatmulResult(op).getType();
    auto newMatmul = rewriter.create<MatmulOp>(loc, resultType, self, mat2,
                                               rewriter.getBoolAttr(false), rewriter.getBoolAttr(false));
    rewriter.replaceOp(op, newMatmul.getResult());
    return success();
  }
};

using ConvertAclnnMmToMatMulPattern = ConvertAclnnMatmulLikeToMatMulPattern<AclnnMmOp>;
using ConvertAclnnMatmulToMatMulPattern = ConvertAclnnMatmulLikeToMatMulPattern<AclnnMatmulOp>;
using ConvertAclnnBatchMatmulToMatMulPattern =
    ConvertAclnnMatmulLikeToMatMulPattern<AclnnBatchMatmulOp>;

/// Pattern to fuse aclnn matmul + aclnn.add into muse.matmul_with_bias.
/// Converts aclnn.mm/matmul/batch_matmul + aclnn.add (with alpha=1) to muse.matmul_with_bias.
/// Higher benefit pattern, so it runs before converting standalone matmul operations.
class ConvertAclnnMatmulAddToMatMulWithBiasPattern : public OpRewritePattern<AclnnAddOp> {
 public:
  using OpRewritePattern<AclnnAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AclnnAddOp addOp, PatternRewriter &rewriter) const override {
    if (!muse::isConstOne(addOp.getAlpha())) {
      return failure();
    }
    Value x = addOp.getX();
    Value y = addOp.getY();
    Value bias;
    Operation *matmulOp = nullptr;

    if (!matmulOp) {
      matmulOp = findAclnnMatmulFromAddOperands<AclnnMmOp>(x, y, bias);
    }
    if (!matmulOp) {
      matmulOp = findAclnnMatmulFromAddOperands<AclnnMatmulOp>(x, y, bias);
    }
    if (!matmulOp) {
      matmulOp = findAclnnMatmulFromAddOperands<AclnnBatchMatmulOp>(x, y, bias);
    }
    if (!matmulOp) {
      return failure();
    }

    Value self = matmulOp->getOperand(0);
    Value other = matmulOp->getOperand(1);
    Type resultType = addOp.getResult().getType();
    Location loc = addOp.getLoc();
    auto newOp = rewriter.create<MatmulWithBiasOp>(
        loc, resultType, self, other, bias, rewriter.getBoolAttr(false), rewriter.getBoolAttr(false));
    rewriter.replaceOp(addOp, newOp.getResult());
    rewriter.eraseOp(matmulOp);
    return success();
  }
};

}  // namespace

struct ConvertAclnnMatmulToMatmulPass : public impl::ConvertAclnnMatmulToMatmulBase<ConvertAclnnMatmulToMatmulPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertAclnnMatmulAddToMatMulWithBiasPattern>(&getContext(), 2);
    patterns.add<ConvertAclnnMmToMatMulPattern, ConvertAclnnMatmulToMatMulPattern,
                 ConvertAclnnBatchMatmulToMatMulPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace muse

std::unique_ptr<Pass> createConvertAclnnMatmulToMatmulPass() {
  return std::make_unique<muse::ConvertAclnnMatmulToMatmulPass>();
}

}  // namespace mlir
