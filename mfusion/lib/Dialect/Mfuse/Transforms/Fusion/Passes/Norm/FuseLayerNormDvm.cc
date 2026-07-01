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

#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/Norm/FuseLayerNormDvm.h"

#include "mfusion/Analysis/FusionRegion/FusionRegionTag.h"
#include "mfusion/Analysis/DvmFusion/LayerNorm/LayerNormDvmUtils.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/FusionPassMacros.h"
#include "mfusion/Support/Logging.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_FUSELAYERNORMDVM
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {
namespace {

bool isLayerNormDvmTagged(Operation *op) { return fusion_region::isTagged(op); }

bool closureHasLayerNormDvmTag(ArrayRef<Operation *> ops) {
  return llvm::any_of(ops, [](Operation *taggedOp) { return isLayerNormDvmTagged(taggedOp); });
}

template <typename OpTy, typename MatchFn>
LogicalResult tagLayerNormDvmVectorMatch(OpTy op, MatchFn &&matchFn, PatternRewriter &rewriter) {
  if (isLayerNormDvmTagged(op.getOperation())) {
    return failure();
  }
  auto matchResult = matchFn();
  if (!matchResult.matched) {
    return rewriter.notifyMatchFailure(op, "backward vector subgraph pattern mismatch");
  }
  std::string groupId = fusion_region::allocateGroupId(fusion_region::kLayerNormFuseKind);
  unsigned tagged = layernorm_dvm::tagLayerNormDvmBackwardOpsExclusive(matchResult.ops, groupId);
  if (tagged == 0 && !closureHasLayerNormDvmTag(matchResult.ops)) {
    return rewriter.notifyMatchFailure(op, "no exclusive backward vector ops to tag (shared rstd/neg?)");
  }
  layernorm_dvm::tagLayerNormDvmOp(op.getOperation(), groupId);
  return success();
}

class FuseLayerNormDvmPattern : public OpRewritePattern<AddOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp addOp, PatternRewriter &rewriter) const override {
    if (layernorm_dvm::hasLayerNormDvmAttr(addOp)) {
      return failure();
    }
    auto matchResult = layernorm_dvm::matchLayerNormDvmFromBetaAdd(addOp);
    if (!matchResult.matched) {
      return rewriter.notifyMatchFailure(addOp, "not a decomposed LayerNorm beta-add anchor");
    }
    MLOG(DEBUG) << "FuseLayerNormDvmPattern: tagged sum/mean LayerNorm subgraph for DVM";
    layernorm_dvm::tagLayerNormDvmForwardOps(matchResult.ops);
    return success();
  }
};

class FuseLayerNormDvmBwdPattern : public OpRewritePattern<DivOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DivOp divOp, PatternRewriter &rewriter) const override {
    if (isLayerNormDvmTagged(divOp.getOperation())) {
      return failure();
    }
    auto matchResult = layernorm_dvm::matchLayerNormDvmBackwardFromGradDiv(divOp);
    if (!matchResult.matched) {
      return rewriter.notifyMatchFailure(divOp, "not a LayerNorm backward grad-div anchor");
    }
    std::string groupId = fusion_region::allocateGroupId(fusion_region::kLayerNormFuseKind);
    unsigned tagged = layernorm_dvm::tagLayerNormDvmBackwardOpsExclusive(matchResult.ops, groupId);
    if (tagged == 0 && !closureHasLayerNormDvmTag(matchResult.ops)) {
      return rewriter.notifyMatchFailure(divOp, "no exclusive backward grad-div ops to tag");
    }
    layernorm_dvm::tagLayerNormDvmOp(divOp.getOperation(), groupId);
    return success();
  }
};

class FuseLayerNormDvmBwdCqxnPattern : public OpRewritePattern<ReduceSumOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceSumOp sumOp, PatternRewriter &rewriter) const override {
    return tagLayerNormDvmVectorMatch(sumOp,
                                      [&]() { return layernorm_dvm::matchLayerNormDvmBackwardFromCqxnSum(sumOp); },
                                      rewriter);
  }
};

class FuseLayerNormDvmBwdCuahirPattern : public OpRewritePattern<ReduceSumOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceSumOp sumOp, PatternRewriter &rewriter) const override {
    return tagLayerNormDvmVectorMatch(sumOp,
                                      [&]() { return layernorm_dvm::matchLayerNormDvmBackwardFromCuahirSum(sumOp); },
                                      rewriter);
  }
};

void registerLayerNormDvmRegionPatternsImpl(RewritePatternSet &patterns) {
  patterns.add<FuseLayerNormDvmPattern, FuseLayerNormDvmBwdPattern, FuseLayerNormDvmBwdCqxnPattern,
               FuseLayerNormDvmBwdCuahirPattern>(patterns.getContext());
}

struct FuseLayerNormDvmPass : public impl::FuseLayerNormDvmBase<FuseLayerNormDvmPass> {
  void runOnOperation() override {
    fusion_region::resetGroupIdAllocator();
    RewritePatternSet patterns(&getContext());
    layernorm_dvm::registerLayerNormDvmRegionPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

namespace layernorm_dvm {

void registerLayerNormDvmRegionPatterns(RewritePatternSet &patterns) {
  registerLayerNormDvmRegionPatternsImpl(patterns);
}

}  // namespace layernorm_dvm

std::unique_ptr<Pass> createFuseLayerNormDvmPass() { return std::make_unique<FuseLayerNormDvmPass>(); }

}  // namespace mfuse
}  // namespace mlir
