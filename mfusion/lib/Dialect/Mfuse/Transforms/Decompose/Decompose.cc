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

#include "mfusion/Dialect/Mfuse/Transforms/Decompose/Decompose.h"

#include "mfusion/Dialect/Mfuse/Transforms/Decompose/DecomposePatterns.h"
#include "mfusion/Dialect/Mfuse/Mfuse.h"
#include "mfusion/Dialect/Mfuse/MfuseDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"

namespace mlir {

#define GEN_PASS_DEF_DECOMPOSE
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {
struct DecomposePass : public impl::DecomposeBase<DecomposePass> {
  using DecomposeBase::DecomposeBase;

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::mfuse::MfuseDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    // Create patterns for this run - modern compilers optimize this well
    // and in practice, decompose passes are rarely run multiple times
    RewritePatternSet patterns(ctx);
    registerDecomposePatterns(patterns);

    // Apply the patterns using the greedy pattern rewrite driver
    // OpRewritePattern will automatically match GELU and Tanh operations and decompose them
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      llvm::errs() << "Failed to apply decompose patterns\n";
      signalPassFailure();
    }
  }
};

}  // namespace mfuse

std::unique_ptr<Pass> createDecomposePass() { return std::make_unique<mfuse::DecomposePass>(); }

}  // namespace mlir
