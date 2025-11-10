/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//===----------------------------------------------------------------------===//
// AffineForVectorize.cpp
//
// Vectorize every innermost affine.for loop:
//   • If the upper bound is a constant and > 1: VF = upper-bound
//   • Otherwise                               : VF = --default-vf (default 512)
//
// Supports iter_args reductions; detect and skip broadcast-write
// (otherwise the verifier will complain).
//===----------------------------------------------------------------------===//

#include "akg/Dialect/Affine/Transforms/AffineForVectorize.h"
#include "akg/Dialect/Affine/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "affine-for-vectorize"

namespace mlir {
#define GEN_PASS_DECL_AFFINEFORVECTPASS
#define GEN_PASS_DEF_AFFINEFORVECTPASS
#include "akg/Dialect/Affine/Passes.h.inc"
}  // namespace mlir

using namespace mlir;          // NOLINT(build/namespaces)
using namespace mlir::affine;  // NOLINT(build/namespaces)

namespace {

/// -------------------------------------------------------------------------
/// helper: If any AffineStoreOp's index map operands do NOT contain the
///         induction variable, it will produce a broadcast-write; return false.
/// -------------------------------------------------------------------------
static bool allStoresUseIV(AffineForOp loop) {
  Value iv = loop.getInductionVar();
  bool ok = true;
  loop.walk([&](AffineStoreOp st) {
    auto mapOpnds = st.getMapOperands();
    if (llvm::none_of(mapOpnds, [&](Value v) { return v == iv; })) {
      ok = false;
    }
  });
  return ok;
}

/// Check if innermost
static bool isInnermostLoop(AffineForOp loop) {
  bool hasNested = false;
  loop.walk([&](AffineForOp nested) {
    if (nested != loop) hasNested = true;
  });
  return !hasNested;
}

/// Decide vector factor; return {VF, skipFlag}
static std::pair<int64_t, bool> decideVectorFactor(AffineForOp loop, int64_t defaultVF) {
  if (loop.getStep() != 1) return {0, true};

  // Constant upper bound
  if (loop.hasConstantUpperBound()) {
    int64_t ub = loop.getConstantUpperBound();
    if (ub > 1) return {ub, false};
    if (ub == 1 && allStoresUseIV(loop)) return {defaultVF, false};
    return {0, true};
  }

  // Dynamic upper bound
  if (!allStoresUseIV(loop)) return {0, true};
  return {defaultVF, false};
}

/// Build 1-D VectorizationStrategy
static VectorizationStrategy build1DStrategy(AffineForOp loop, int64_t vecSize) {
  VectorizationStrategy s;
  s.vectorSizes = {vecSize};
  s.loopToVectorDim[loop] = 0;

  SmallVector<LoopReduction, 2> reductions;
  if (isLoopParallel(loop, &reductions) && !reductions.empty())
    s.reductionLoops[loop] = reductions;
  return s;
}

class AffineForVectPass
    : public mlir::impl::AffineForVectPassBase<AffineForVectPass> {
 public:
  AffineForVectPass() = default;
  AffineForVectPass(const AffineForVectPass &) = default;

  StringRef getArgument() const override { return "affine-for-vectorize"; }
  StringRef getDescription() const override {
    return "Vectorize each innermost affine.for loop (1-D). "
           "For dynamic trip count or ub==1, use --default-vf.";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry
        .insert<AffineDialect, vector::VectorDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    SmallVector<AffineForOp> leafLoops;
    funcOp.walk([&](AffineForOp forOp) {
      if (isInnermostLoop(forOp)) leafLoops.push_back(forOp);
    });

    for (AffineForOp loop : leafLoops) {
      auto [vf, skip] = decideVectorFactor(loop, defaultVF);
      if (skip || vf <= 0) {
        loop.emitRemark(DEBUG_TYPE) << "skip: not vectorizable";
        continue;
      }

      VectorizationStrategy strategy = build1DStrategy(loop, vf);
      std::vector<SmallVector<AffineForOp, 2>> nest{
          SmallVector<AffineForOp, 2>{loop}};
      if (failed(vectorizeAffineLoopNest(nest, strategy)))
        loop.emitRemark(DEBUG_TYPE) << "vectorization failed";
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::affine::createAffineForVectPass() {
  return std::make_unique<AffineForVectPass>();
}
