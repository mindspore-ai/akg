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

#include "mfusion/Dialect/Mfuse/Transforms/HighLevelOpt/Passes.h"

#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Transforms/Outlining/FusionAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

#define GEN_PASS_DEF_FOLDREDUNDANTMATMULTRANSPOSEPASS
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {
namespace {

static bool isSwapLastTwoDimsPermute(PermuteOp permuteOp) {
  auto inputType = dyn_cast<RankedTensorType>(permuteOp.getInput().getType());
  if (!inputType) {
    return false;
  }

  int64_t rank = inputType.getRank();
  if (rank < 2) {
    return false;
  }

  ArrayAttr permAttr = permuteOp.getPermAttr();
  if (!permAttr || static_cast<int64_t>(permAttr.size()) != rank) {
    return false;
  }

  int64_t secondLast = rank - 2;
  int64_t last = rank - 1;
  for (int64_t i = 0; i < rank; ++i) {
    auto intAttr = dyn_cast<IntegerAttr>(permAttr[static_cast<size_t>(i)]);
    if (!intAttr) {
      return false;
    }

    int64_t perm = intAttr.getInt();
    if (i < secondLast && perm != i) {
      return false;
    }
    if (i == secondLast && perm != last) {
      return false;
    }
    if (i == last && perm != secondLast) {
      return false;
    }
  }

  return true;
}

static bool foldImmediateTransposePair(Value value, bool trans, Value &replacement) {
  if (!trans) {
    return false;
  }

  auto permuteOp = value.getDefiningOp<PermuteOp>();
  if (!permuteOp || !isSwapLastTwoDimsPermute(permuteOp)) {
    return false;
  }

  replacement = permuteOp.getInput();
  return true;
}

static bool isInsideNestedMfuseFusedOp(Operation *op) { return op->getParentOfType<FusedOp>() != nullptr; }

static bool isDvmFusionFunction(func::FuncOp func) {
  if (!func) {
    return false;
  }
  auto fusionType = func->getAttrOfType<StringAttr>(mfusion_attrs::kFusionType);
  return fusionType && fusionType.getValue() == "dvm";
}

struct FoldMatmulPattern : public OpRewritePattern<MatmulOp> {
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp op, PatternRewriter &rewriter) const override {
    if (isInsideNestedMfuseFusedOp(op)) {
      return failure();
    }

    Value newSelf = op.getSelf();
    Value newOther = op.getOther();
    bool foldSelf = foldImmediateTransposePair(op.getSelf(), op.getTransX1(), newSelf);
    bool foldOther = foldImmediateTransposePair(op.getOther(), op.getTransX2(), newOther);
    if (!foldSelf && !foldOther) {
      return failure();
    }

    auto newOp = rewriter.create<MatmulOp>(op.getLoc(), op.getResult().getType(), newSelf, newOther,
                                           rewriter.getBoolAttr(foldSelf ? false : op.getTransX1()),
                                           rewriter.getBoolAttr(foldOther ? false : op.getTransX2()));
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct FoldMatmulWithBiasPattern : public OpRewritePattern<MatmulWithBiasOp> {
  using OpRewritePattern<MatmulWithBiasOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulWithBiasOp op, PatternRewriter &rewriter) const override {
    if (isInsideNestedMfuseFusedOp(op)) {
      return failure();
    }

    Value newSelf = op.getSelf();
    Value newOther = op.getOther();
    bool foldSelf = foldImmediateTransposePair(op.getSelf(), op.getTransX1(), newSelf);
    bool foldOther = foldImmediateTransposePair(op.getOther(), op.getTransX2(), newOther);
    if (!foldSelf && !foldOther) {
      return failure();
    }

    auto newOp = rewriter.create<MatmulWithBiasOp>(
        op.getLoc(), op.getResult().getType(), newSelf, newOther, op.getBias(),
        rewriter.getBoolAttr(foldSelf ? false : op.getTransX1()),
        rewriter.getBoolAttr(foldOther ? false : op.getTransX2()));
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct FoldBatchMatmulPattern : public OpRewritePattern<BatchMatmulOp> {
  using OpRewritePattern<BatchMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BatchMatmulOp op, PatternRewriter &rewriter) const override {
    if (isInsideNestedMfuseFusedOp(op)) {
      return failure();
    }

    Value newSelf = op.getSelf();
    Value newMat2 = op.getMat2();
    bool foldSelf = foldImmediateTransposePair(op.getSelf(), op.getTransposeA(), newSelf);
    bool foldMat2 = foldImmediateTransposePair(op.getMat2(), op.getTransposeB(), newMat2);
    if (!foldSelf && !foldMat2) {
      return failure();
    }

    auto newOp = rewriter.create<BatchMatmulOp>(
        op.getLoc(), op.getResult().getType(), newSelf, newMat2,
        rewriter.getBoolAttr(foldSelf ? false : op.getTransposeA()),
        rewriter.getBoolAttr(foldMat2 ? false : op.getTransposeB()));
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

template <typename OpTy>
struct FoldAclnnTransXMatmulPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
    if (isInsideNestedMfuseFusedOp(op)) {
      return failure();
    }

    Value newSelf = op.getSelf();
    Value newMat2 = op.getMat2();
    bool foldSelf = foldImmediateTransposePair(op.getSelf(), op.getTransX1(), newSelf);
    bool foldMat2 = foldImmediateTransposePair(op.getMat2(), op.getTransX2(), newMat2);
    if (!foldSelf && !foldMat2) {
      return failure();
    }

    auto newOp = rewriter.create<OpTy>(op.getLoc(), op.getResult().getType(), newSelf, newMat2,
                                       rewriter.getBoolAttr(foldSelf ? false : op.getTransX1()),
                                       rewriter.getBoolAttr(foldMat2 ? false : op.getTransX2()));
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct FoldRedundantMatmulTransposePass
    : public impl::FoldRedundantMatmulTransposePassBase<FoldRedundantMatmulTransposePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (isDvmFusionFunction(func)) {
        continue;
      }

      RewritePatternSet patterns(func.getContext());
      patterns.add<FoldMatmulPattern, FoldMatmulWithBiasPattern, FoldBatchMatmulPattern,
                   FoldAclnnTransXMatmulPattern<AclnnBatchMatmulOp>,
                   FoldAclnnTransXMatmulPattern<AclnnMatmulOp>, FoldAclnnTransXMatmulPattern<AclnnMmOp>>(
          func.getContext());
      if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createFoldRedundantMatmulTransposePass() {
  return std::make_unique<FoldRedundantMatmulTransposePass>();
}

}  // namespace mfuse
}  // namespace mlir
