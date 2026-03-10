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

#include <cstdlib>
#include <functional>
#include <utility>
#include <vector>

#include "llvm/Support/raw_ostream.h"
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

namespace {

const char kPrintIrEnvName[] = "MFUSION_PRINT_IR";
const char kPrintIrEnvValue[] = "1";

/// Returns true when MFUSION_PRINT_IR is set to "1" (enable per-pass IR dump).
inline bool isPrintIrEnabled() {
  const char* v = std::getenv(kPrintIrEnvName);
  return v && (std::string(v) == kPrintIrEnvValue);
}

/// Prints the current module IR to llvm::outs() with a short title.
void printIrAfterPass(llvm::StringRef passName, Operation* op) {
  llvm::outs() << "\n" << std::string(80, '=') << "\n";
  llvm::outs() << "MFuse Fusion IR after pass: " << passName << "\n";
  llvm::outs() << std::string(80, '=') << "\n";
  OpPrintingFlags flags;
  op->print(llvm::outs(), flags);
  llvm::outs() << "\n";
}

}  // namespace

struct MfuseFusionPass : public impl::MfuseFusionBase<MfuseFusionPass> {
  void runOnOperation() override {
    using PassCreator = std::function<std::unique_ptr<Pass>()>;
    std::vector<std::pair<const char*, PassCreator>> passes = {
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
        {"fuse-batch-matmul-to-mul", []() { return createFuseBatchMatMulToMulPass(); }},
        {"fuse-matmul-reshape-bias-add",
         []() { return createFuseMatmulReshapeBiasAddPass(); }},

        {"fuse-gelu", []() { return createFuseGeluPass(); }},
        {"fuse-add-rms-norm", []() { return createFuseAddRmsNormPass(); }},
        {"fuse-swi-glu", []() { return createFuseSwiGluPass(); }},
    };

    const bool printIr = isPrintIrEnabled();

    if (printIr) {
      Operation* op = getOperation();
      for (const auto& [name, creator] : passes) {
        PassManager pm(&getContext());
        pm.addPass(creator());
        if (failed(pm.run(op))) {
          signalPassFailure();
          return;
        }
        printIrAfterPass(name, op);
      }
      return;
    }

    PassManager pm(&getContext());
    for (const auto& [name, creator] : passes) {
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
