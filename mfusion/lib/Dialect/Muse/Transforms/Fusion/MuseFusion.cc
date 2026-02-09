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

#include "mfusion/Dialect/Muse/Transforms/Fusion/MuseFusion.h"

#include "mfusion/Dialect/Muse/MuseDialect.h"
#include "mfusion/Dialect/Muse/Transforms/Fusion/FusionPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
#define GEN_PASS_DEF_MUSEFUSION
#include "mfusion/Dialect/Muse/Transforms/Passes.h.inc"

namespace muse {

struct MuseFusionPass : public impl::MuseFusionBase<MuseFusionPass> {
  void runOnOperation() override {
    // Create a pass manager to run all fusion passes in order
    PassManager pm(&getContext());

    pm.addPass(createFuseMatMulCastPass());
    pm.addPass(createFuseMatmulUnsqueezeSqueezePass());
    pm.addPass(createFuseMatmulTransposeWeightPass());
    pm.addPass(createFuseMatmulReshapeBiasAddPass());
    pm.addPass(createFuseGeluPass());
    pm.addPass(createFuseAddRmsNormPass());
    pm.addPass(createFuseSwiGluPass());

    // Run the pipeline
    if (failed(pm.run(getOperation()))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createMuseFusionPass() { return std::make_unique<muse::MuseFusionPass>(); }

}  // namespace muse
}  // namespace mlir
