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

#include "akg/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DECL_ERASEUNUSEDOPERANDSANDRESULTS
#define GEN_PASS_DEF_ERASEUNUSEDOPERANDSANDRESULTS
#include "akg/Dialect/Linalg/Passes.h.inc"
}  // namespace mlir

namespace mlir {
namespace {

struct EraseUnusedOperandsAndResults
    : public impl::EraseUnusedOperandsAndResultsBase<EraseUnusedOperandsAndResults> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createEraseUnusedOperandsAndResultsPass() {
  return std::make_unique<EraseUnusedOperandsAndResults>();
}

}  // namespace mlir
