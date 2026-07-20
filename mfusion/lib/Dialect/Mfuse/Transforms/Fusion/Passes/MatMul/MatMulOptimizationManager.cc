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

#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/MatMul/MatMulOptimizationManager.h"

#include "mfusion/Dialect/Mfuse/IR/MfuseDialect.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/MatMul/FuseMatMulBiasAdd.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/MatMul/FuseMatMulCast.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/MatMul/FuseMatmulK1ToMul.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/MatMul/FuseMatmulPermute.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/MatMul/FuseMatmulReshapeBiasAdd.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/MatMul/FuseMatmulUnsqueezeSqueeze.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_MATMULOPTIMIZATION
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {

void MatMulOptimizationManager::registerPasses(PassManager &pm) {
  // 1. Absorb f16->f32 cast into matmul before Unsqueeze inserts reshape.
  // Cast only matches MatmulOp/MatmulWithBiasOp -> Cast; if Unsqueeze runs
  // first, 1D cases become matmul -> reshape -> cast and Cast/ReshapeBiasAdd miss.
  pm.addPass(createFuseMatMulCastPass());

  // 2. Normalize 1D inputs via reshape (after Cast so cast can still fuse).
  pm.addPass(createFuseMatmulUnsqueezeSqueezePass());

  // 3. Eliminate permute into trans flags (after types/shapes normalized).
  pm.addPass(createFuseMatmulPermutePass());

  // 4. K=1 reduction (after permute elimination).
  // Must run before BiasAdd: FuseMatmulK1ToMul only matches MatmulOp.
  // If BiasAdd runs first, ND K=1 matmul+bias becomes MatmulWithBiasOp and K1
  // misses it; AFTER_MANUAL_FUSION later splits back to matmul+add with no mul.
  pm.addPass(createFuseMatmulK1ToMulPass());

  // 5. Bias fusion (after K1 so K=1 reduction is not blocked by MatmulWithBiasOp).
  pm.addPass(createFuseMatMulBiasAddPass());

  // 6. Reshape-path bias fusion (after the matmul chain is settled).
  // Also recovers 1D matmul+cast+bias: Cast then Unsqueeze yields
  // matmul(f32)->reshape->add, which this pass fuses.
  pm.addPass(createFuseMatmulReshapeBiasAddPass());
}

namespace {
struct MatMulOptimizationPass : public impl::MatMulOptimizationBase<MatMulOptimizationPass> {
  using MatMulOptimizationBase::MatMulOptimizationBase;

  void runOnOperation() override {
    PassManager pm(&getContext());
    MatMulOptimizationManager::registerPasses(pm);
    if (failed(pm.run(getOperation()))) {
      signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<Pass> createMatMulOptimizationPass() {
  return std::make_unique<MatMulOptimizationPass>();
}

}  // namespace mfuse
}  // namespace mlir
