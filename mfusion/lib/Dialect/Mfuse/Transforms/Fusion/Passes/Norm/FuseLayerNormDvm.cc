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

#include <cstdlib>

#include "mfusion/Analysis/DvmFusion/LayerNorm/LayerNormDvmMaterializer.h"
#include "mfusion/Analysis/DvmFusion/LayerNorm/LayerNormDvmMatcher.h"
#include "mfusion/Analysis/DvmFusion/LayerNorm/LayerNormDvmUtils.h"
#include "mfusion/Analysis/FusionRegion/FusionRegionTag.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Support/VarianceUtils.h"
#include "mfusion/Dialect/Mfuse/Support/ArithUtils.h"
#include "mfusion/Dialect/Mfuse/Transforms/Decompose/DecomposePatterns.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_FUSELAYERNORMDVM
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {
namespace {

bool lnDvmDebugEnabled() {
  if (const char *env = std::getenv("MFUSION_DEBUG_LN_DVM")) {
    return env[0] != '\0' && env[0] != '0';
  }
  return false;
}

bool isAlreadyProcessed(Operation *op) {
  return layernorm_dvm::isLayerNormDvmMaterialized(op) || layernorm_dvm::isOpInFusedIsland(op);
}

DivOp getForwardNormDiv(AddOp addOp) {
  if (auto div = addOp.getX().getDefiningOp<DivOp>()) {
    return div;
  }
  return addOp.getY().getDefiningOp<DivOp>();
}

SubOp getForwardCenterSub(DivOp normDiv) {
  if (!normDiv) {
    return {};
  }
  auto gammaMul = normDiv.getSelf().getDefiningOp<MulOp>();
  if (!gammaMul) {
    return {};
  }
  if (auto center = gammaMul.getLhs().getDefiningOp<SubOp>()) {
    return center;
  }
  return gammaMul.getRhs().getDefiningOp<SubOp>();
}

SqrtOp getForwardVarianceSqrt(DivOp normDiv) {
  if (!normDiv) {
    return {};
  }
  Value denominator = peelBroadcast(normDiv.getOther());
  auto addEps = denominator.getDefiningOp<AddOp>();
  if (!addEps) {
    return {};
  }
  if (auto sqrt = addEps.getX().getDefiningOp<SqrtOp>()) {
    return sqrt;
  }
  return addEps.getY().getDefiningOp<SqrtOp>();
}

void decomposeCandidateVariance(AddOp anchor, PatternRewriter &rewriter) {
  DivOp normDiv = getForwardNormDiv(anchor);
  SubOp centerSub = getForwardCenterSub(normDiv);
  SqrtOp sqrtOp = getForwardVarianceSqrt(normDiv);
  if (!centerSub || !sqrtOp) {
    return;
  }

  Value meanValue = peelBroadcast(centerSub.getY());
  Value varianceValue = peelBroadcast(sqrtOp.getInput());
  AclnnVarMeanOp varMean = meanValue.getDefiningOp<AclnnVarMeanOp>();
  if (!varMean) {
    varMean = varianceValue.getDefiningOp<AclnnVarMeanOp>();
  }

  if (varMean) {
    rewriter.setInsertionPoint(varMean);
    auto resultsOr = decomposeAclnnVarMean(varMean, rewriter);
    if (succeeded(resultsOr)) {
      rewriter.replaceOp(varMean, {resultsOr->first, resultsOr->second});
    }
  } else if (auto var = varianceValue.getDefiningOp<AclnnVarOp>()) {
    rewriter.setInsertionPoint(var);
    auto varianceOr = decomposeAclnnVar(var, rewriter);
    if (succeeded(varianceOr)) {
      rewriter.replaceOp(var, *varianceOr);
    }
  }
}

void decomposeCandidateMean(AddOp anchor, PatternRewriter &rewriter) {
  SubOp centerSub = getForwardCenterSub(getForwardNormDiv(anchor));
  if (!centerSub) {
    return;
  }
  auto reduceMean = peelBroadcast(centerSub.getY()).getDefiningOp<ReduceMeanOp>();
  if (!reduceMean) {
    return;
  }
  rewriter.setInsertionPoint(reduceMean);
  (void)decomposeReduceMean(reduceMean, rewriter);
}

void mergeDuplicateCandidateCenterSub(AddOp anchor);

void runLayerNormDvmPrepDecompose(func::FuncOp func) {
  SmallVector<AddOp> anchors;
  func.walk([&](AddOp addOp) {
    if (!addOp->getParentOfType<FusedOp>() && getForwardNormDiv(addOp)) {
      anchors.push_back(addOp);
    }
  });

  PatternRewriter rewriter(func.getContext());
  for (AddOp anchor : anchors) {
    decomposeCandidateVariance(anchor, rewriter);
    decomposeCandidateMean(anchor, rewriter);
    mergeDuplicateCandidateCenterSub(anchor);
  }
}

void mergeDuplicateCandidateCenterSub(AddOp anchor) {
  SubOp centerSub = getForwardCenterSub(getForwardNormDiv(anchor));
  if (!centerSub) {
    return;
  }

  Value x = getCanonicalFusionTensor(centerSub.getX());
  Value mean = peelBroadcast(centerSub.getY());
  SubOp dominatingSub;
  for (Operation &op : *centerSub->getBlock()) {
    auto candidate = dyn_cast<SubOp>(&op);
    if (!candidate || candidate == centerSub) {
      continue;
    }
    if (!candidate->isBeforeInBlock(centerSub.getOperation())) {
      break;
    }
    if (getCanonicalFusionTensor(candidate.getX()) == x && peelBroadcast(candidate.getY()) == mean) {
      dominatingSub = candidate;
    }
  }
  if (!dominatingSub) {
    return;
  }
  centerSub.getResult().replaceAllUsesWith(dominatingSub.getResult());
  if (centerSub->use_empty()) {
    centerSub.erase();
  }
}



LogicalResult canonicalizeFunc(func::FuncOp func) {
  MLIRContext *ctx = func.getContext();
  RewritePatternSet patterns(ctx);
  for (Dialect *dialect : ctx->getLoadedDialects()) {
    dialect->getCanonicalizationPatterns(patterns);
  }
  return applyPatternsAndFoldGreedily(func, std::move(patterns));
}

bool fuseForwardAnchors(func::FuncOp func, MLIRContext *ctx) {
  bool changed = false;
  SmallVector<AddOp> anchors;
  func.walk([&](AddOp addOp) {
    if (addOp->getParentOfType<FusedOp>() || isAlreadyProcessed(addOp.getOperation())) {
      return;
    }
    anchors.push_back(addOp);
  });
  for (AddOp addOp : anchors) {
    auto matchResult = layernorm_dvm::matchLayerNormDvmFromBetaAdd(addOp);
    if (!matchResult.matched) {
      if (lnDvmDebugEnabled() && (addOp.getX().getDefiningOp<DivOp>() || addOp.getY().getDefiningOp<DivOp>())) {
        llvm::errs() << "[LN-DVM] beta_add match failed in "
                     << addOp->getParentOfType<func::FuncOp>().getName() << "\n";
      }
      continue;
    }
    PatternRewriter rewriter(ctx);
    if (succeeded(layernorm_dvm::fuseLayerNormDvmForward(matchResult, rewriter))) {
      changed = true;
    } else if (lnDvmDebugEnabled()) {
      llvm::errs() << "[LN-DVM] materialize failed after match in "
                   << addOp->getParentOfType<func::FuncOp>().getName() << "\n";
    }
  }
  return changed;
}

bool fuseBackwardAnchors(func::FuncOp func, MLIRContext *ctx) {
  bool changed = false;
  SmallVector<Operation *> anchors;
  func.walk([&](Operation *op) {
    if (op->getParentOfType<FusedOp>() || isAlreadyProcessed(op)) {
      return;
    }
    if (isa<DivOp, ReduceSumOp>(op)) {
      anchors.push_back(op);
    }
  });
  for (Operation *op : anchors) {
    if (isAlreadyProcessed(op)) {
      continue;
    }
    PatternRewriter rewriter(ctx);
    if (auto divOp = dyn_cast<DivOp>(op)) {
      auto matchResult = layernorm_dvm::matchLayerNormDvmBackwardFromGradDiv(divOp);
      if (matchResult.matched &&
          succeeded(layernorm_dvm::fuseLayerNormDvmBackwardGradDiv(matchResult, rewriter))) {
        changed = true;
        continue;
      }
    }
    if (auto sumOp = dyn_cast<ReduceSumOp>(op)) {
      auto cqxnMatch = layernorm_dvm::matchLayerNormDvmBackwardFromCqxnSum(sumOp);
      if (cqxnMatch.matched &&
          succeeded(layernorm_dvm::fuseLayerNormDvmBackwardVector(cqxnMatch, rewriter))) {
        changed = true;
        continue;
      }
      auto cuahirMatch = layernorm_dvm::matchLayerNormDvmBackwardFromCuahirSum(sumOp);
      if (cuahirMatch.matched &&
          succeeded(layernorm_dvm::fuseLayerNormDvmBackwardVector(cuahirMatch, rewriter))) {
        changed = true;
      }
    }
  }
  return changed;
}

LogicalResult fuseLayerNormDvmInFunc(func::FuncOp func, MLIRContext *ctx) {
  // Candidate tracing keeps preparation local; unrelated mean/variance graphs are untouched.
  runLayerNormDvmPrepDecompose(func);

  fusion_region::resetGroupIdAllocator();

  bool changed = true;
  while (changed) {
    bool forwardChanged = fuseForwardAnchors(func, ctx);
    bool backwardChanged = fuseBackwardAnchors(func, ctx);
    changed = forwardChanged || backwardChanged;
  }
  return canonicalizeFunc(func);
}

}  // namespace

struct FuseLayerNormDvmPass : public impl::FuseLayerNormDvmBase<FuseLayerNormDvmPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto walkResult = module.walk([&](func::FuncOp func) {
      if (failed(fuseLayerNormDvmInFunc(func, &getContext()))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createFuseLayerNormDvmPass() { return std::make_unique<FuseLayerNormDvmPass>(); }

}  // namespace mfuse
}  // namespace mlir
