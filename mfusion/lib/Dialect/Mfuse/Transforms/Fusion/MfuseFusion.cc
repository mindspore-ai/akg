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
    std::vector<std::pair<const char *, PassCreator>> passes = {
        // Conv-related fusion passes:
        {"fuse-biasadd-conv", []() { return createFuseBiasaddConvPass(); }},
        {"fuse-conv2d-cast", []() { return createFuseConv2DCastPass(); }},

        // MatMul-related fusion passes (order by dependency):
        // FuseMatMulCast: matmul+cast -> matmul (output type); no deps.
        // FuseMatMulBiasAdd: matmul/batch_matmul+add(bias) -> matmul_with_bias;
        // before reshape so direct matmul+add is fused.
        // FuseMatmulUnsqueezeSqueeze: normalize 1D inputs (reshape); after Cast for stable type.
        // FuseMatmulTransposeWeight: alignment (permute/trans); after shape normalization.
        // FuseBatchMatMul: transpose elimination (permute into trans); BatchMatMul 2D -> MatMul.
        // FuseBatchMatMulToMul: matmul/batch_matmul (k=1) -> mul; after shape normalization.
        // FuseMatmulReshapeBiasAdd: matmul->reshape->add -> matmul_with_bias; last so it sees final matmul form.
        {"fuse-matmul-cast", []() { return createFuseMatMulCastPass(); }},
        {"fuse-matmul-bias-add", []() { return createFuseMatMulBiasAddPass(); }},
        {"fuse-matmul-unsqueeze-squeeze",
         []() { return createFuseMatmulUnsqueezeSqueezePass(); }},
        {"fuse-matmul-transpose-weight",
         []() { return createFuseMatmulTransposeWeightPass(); }},
        {"fuse-batch-matmul",
         []() { return createFuseBatchMatMulPass(); }},
        {"fuse-batchmatmul-to-mul", []() { return createFuseBatchMatMulToMulPass(); }},
        {"fuse-matmul-reshape-bias-add",
         []() { return createFuseMatmulReshapeBiasAddPass(); }},

        {"fuse-gelu", []() { return createFuseGeluPass(); }},
        // RmsNorm is fused on Torch dialect (torch-fusion) before convert-torch-to-mfuse.
        // fuse-add-rms-norm can then fold adjacent add ops with the resulting aclnn.rms_norm.
        {"fuse-addrmsnorm", []() { return createFuseAddRmsNormPass(); }},
        {"fuse-swi-glu", []() { return createFuseSwiGluPass(); }},
    };

    PassManager pm(&getContext());
    for (const auto &[name, creator] : passes) {
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
