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

#include "mfusion/Dialect/Mfuse/Transforms/Fusion/MfuseFusion.h"

#include <functional>
#include <utility>
#include <vector>

#include "mfusion/Dialect/Mfuse/IR/MfuseDialect.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/FusionPasses.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
#define GEN_PASS_DEF_MFUSEFUSION
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {

struct MfuseFusionPass : public impl::MfuseFusionBase<MfuseFusionPass> {
  void runOnOperation() override {
    using PassCreator = std::function<std::unique_ptr<Pass>()>;
    std::vector<std::pair<const char *, PassCreator>> preMatmulPasses = {
      // Must run before fuse-num-to-tensor so scalar binary operands are not materialized as mfuse.full.
      {"mfuse-canonicalize-binary-scalar-operands",
       []() { return createCanonicalizeBinaryScalarOperandsPass(); }},
      // Conv-related fusion passes:
      {"fuse-biasadd-conv", []() { return createFuseBiasaddConvPass(); }},
      {"fuse-conv2d-cast", []() { return createFuseConv2DCastPass(); }},
      {"fuse-batch-norm", []() { return createFuseBatchNormPass(); }},
      {"fuse-conv-batchnorm", []() { return createFuseConvBatchNormPass(); }},
    };
    std::vector<std::pair<const char *, PassCreator>> postMatmulPasses = {
      {"fuse-gelu", []() { return createFuseGeluPass(); }},
      {"fuse-logical-not-compare", []() { return createFuseLogicalNotComparePass(); }},
      // RmsNorm is fused on Torch dialect (torch-fusion) before convert-torch-to-mfuse.
      // fuse-addrmsnorm can then fold adjacent add ops with the resulting aclnn.rms_norm.
      {"fuse-addrmsnorm", []() { return createFuseAddRmsNormPass(); }},
      {"fuse-layernorm", []() { return createFuseLayerNormPass(); }},
      {"fuse-swi-glu", []() { return createFuseSwiGluPass(); }},
      {"fuse-num-to-tensor", []() { return createFuseNumToTensorPass(); }},
      // Safe-softmax fusion must run last: it relies on all preceding fusion
      // passes having completed so softmax producers are in their final form,
      // and it creates its own mfuse.fused regions.
      {"fuse-safe-softmax-dvm", []() { return createFuseSafeSoftmaxDvmPass(); }},
    };

    PassManager pm(&getContext());
    for (const auto &[name, creator] : preMatmulPasses) {
      (void)name;
      pm.addPass(creator());
    }
    // MatMul-related fusion passes via MatMulOptimizationManager (recommended order):
    // 1. FuseMatMulCast  2. FuseMatmulUnsqueezeSqueeze  3. FuseMatmulPermute
    // 4. FuseMatmulK1ToMul  5. FuseMatMulBiasAdd  6. FuseMatmulReshapeBiasAdd
    MatMulOptimizationManager::registerPasses(pm);
    for (const auto &[name, creator] : postMatmulPasses) {
      (void)name;
      pm.addPass(creator());
    }
    if (failed(pm.run(getOperation()))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createMfuseFusionPass() { return std::make_unique<mfuse::MfuseFusionPass>(); }

}  // namespace mfuse
}  // namespace mlir
