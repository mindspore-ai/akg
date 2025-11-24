/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
//===- SCFToGPUPass.cpp - Convert a loop nest to a GPU kernel -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "akg/Conversion/SCFToGPUExt/SCFToGPUPassExt.h"
#include "akg/Conversion/SCFToGPUExt/SCFToGPUExt.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTAKGAFFINEFORTOGPU
#define GEN_PASS_DEF_CONVERTAKGPARALLELLOOPTOGPU
#include "akg/Conversion/Passes.h.inc"
}  // namespace mlir

using namespace mlir;
using namespace mlir::scf;

namespace {
// A pass that traverses top-level loops in the function and converts them to
// GPU launch operations.  Nested launches are not allowed, so this does not
// walk the function recursively to avoid considering nested loops.
struct ForLoopMapper : public impl::ConvertAKGAffineForToGPUBase<ForLoopMapper> {
  ForLoopMapper() = default;
  ForLoopMapper(unsigned numBlockDims, unsigned numThreadDims) {
    this->numBlockDims = numBlockDims;
    this->numThreadDims = numThreadDims;
  }

  void runOnOperation() override {
    for (Operation &op : llvm::make_early_inc_range(getOperation().getFunctionBody().getOps())) {
      if (auto forOp = dyn_cast<affine::AffineForOp>(&op)) {
        if (failed(convertAffineLoopNestToGPULaunch(forOp, numBlockDims, numThreadDims))) {
          signalPassFailure();
        }
      }
    }
  }
};

struct ParallelLoopToGpuPass : public impl::ConvertAKGParallelLoopToGpuBase<ParallelLoopToGpuPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateParallelLoopToGPUPatterns(patterns);
    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    configureParallelLoopToGPULegality(target);
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
    finalizeParallelLoopToGPUConversion(getOperation());
  }
};

}  // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> mlir::createAKGAffineForToGPUPass(unsigned numBlockDims,
                                                                                      unsigned numThreadDims) {
  return std::make_unique<ForLoopMapper>(numBlockDims, numThreadDims);
}
std::unique_ptr<InterfacePass<FunctionOpInterface>> mlir::createAKGAffineForToGPUPass() {
  return std::make_unique<ForLoopMapper>();
}

std::unique_ptr<Pass> mlir::createAKGParallelLoopToGpuPass() { return std::make_unique<ParallelLoopToGpuPass>(); }
