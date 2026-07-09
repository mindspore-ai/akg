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

#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/Softmax/FuseSafeSoftmaxDvm.h"

#include "mfusion/Analysis/DvmFusion/SafeSoftmax/SafeSoftmaxDvmUtils.h"
#include "mfusion/Analysis/FusionRegion/FusionRegionTag.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
class RewritePatternSet;

#define GEN_PASS_DEF_FUSESAFESOFTMAXDVM
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {

void registerFuseSafeSoftmaxDvmPatterns(RewritePatternSet &patterns);

namespace {

struct FuseSafeSoftmaxDvmPass : public impl::FuseSafeSoftmaxDvmBase<FuseSafeSoftmaxDvmPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!safe_softmax_dvm::hasSafeSoftmaxCandidate(module)) {
      return;
    }
    safe_softmax_dvm::markSafeSoftmaxPipelineActive(module);

    fusion_region::resetGroupIdAllocator();
    MLIRContext *ctx = module.getContext();
    RewritePatternSet fusePatterns(ctx);
    registerFuseSafeSoftmaxDvmPatterns(fusePatterns);
    if (failed(applyPatternsAndFoldGreedily(module, std::move(fusePatterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createFuseSafeSoftmaxDvmPass() {
  return std::make_unique<FuseSafeSoftmaxDvmPass>();
}

}  // namespace mfuse
}  // namespace mlir
