/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "akg/Dialect/Tensor/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_DECOMPOSETENSOR
#include "akg/Dialect/Tensor/Passes.h.inc"
}  // namespace mlir

namespace mlir {
namespace {
/// Pass that enables broadcast by making all input arrays have the same
/// number of dimensions. Insert RESHAPE operations to lower rank operand
struct DecomposeTensor : public impl::DecomposeTensorBase<DecomposeTensor> {
 public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    tensor::populateDecomposeTensorConcatPatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace
}  // namespace mlir

std::unique_ptr<mlir::Pass> mlir::createDecomposeTensorPass() { return std::make_unique<mlir::DecomposeTensor>(); }
