/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "akg/Dialect/Affine/Transforms/AKGLoopParallelize.h"
#include "akg/Dialect/Affine/Analysis/AffineAnalysis.h"

#include <deque>
#include <map>

#include "akg/Utils/AnalysisCommon.hpp"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h.inc"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
#define GEN_PASS_DEF_AKGAFFINELOOPPARALLELIZE
#define GEN_PASS_DECL_AKGAFFINELOOPPARALLELIZE
#include "akg/Dialect/Affine/Passes.h.inc"
}  // namespace mlir

#define DEBUG_TYPE "akg-affine-loop-parallelize"

using namespace mlir;
using namespace mlir::akg;

namespace {
/// Convert all parallel affine.for op into 1-D affine.parallel op.
struct AKGLoopParallelize : public impl::AKGAffineLoopParallelizeBase<AKGLoopParallelize> {
  AKGLoopParallelize() {}
  AKGLoopParallelize(const bool enableParallel) : enableParallel(enableParallel) {}

  void runOnOperation() override;
  bool isAncestorLoopParallel(SmallVector<affine::AffineForOp, 6> nonParallelizableLoops, Operation *op);

  bool enableParallel = true;
  static constexpr long PARALLEL_SIZE = 65536;
};

/// Descriptor of a potentially parallelizable loop.
struct ParallelizationCandidate {
  ParallelizationCandidate(affine::AffineForOp l, SmallVector<affine::LoopReduction> &&r)
      : loop(l), reductions(std::move(r)) {}

  /// The potentially parallelizable loop.
  affine::AffineForOp loop;
  /// Desciprtors of reductions that can be parallelized in the loop.
  SmallVector<affine::LoopReduction> reductions;
};
}  // namespace

bool AKGLoopParallelize::isAncestorLoopParallel(SmallVector<affine::AffineForOp, 6> nonParallelizableLoops,
                                                Operation *op) {
  for (auto nonParallelizable : nonParallelizableLoops) {
    if (nonParallelizable.getBody()->findAncestorOpInBlock(*op) != nullptr) {
      return false;
    }
  }
  return true;
}

void AKGLoopParallelize::runOnOperation() {
  if (!enableParallel) {
    return;
  }
  // todo(lijintao): multi bands
  func::FuncOp funcOp = getOperation();
  std::map<affine::AffineForOp, bool> smallLoop;
  for (auto band : funcOp.getOps<affine::AffineForOp>()) {
    int64_t upperSize = 1;
    band->walk([&](affine::AffineForOp forOp) {
      if (!forOp.hasConstantBounds()) {
        return;
      }
      upperSize *= (forOp.getConstantUpperBound() - forOp.getConstantLowerBound());
    });
    if (upperSize > PARALLEL_SIZE) {
      continue;
    }
    band->walk([&](affine::AffineForOp forOp) { smallLoop[forOp] = true; });
  }

  // todo: tiling reduction axis to support parallelism for reduce op

  // The walker proceeds in pre-order to process the outer loops first
  // and control the number of outer parallel loops.
  std::vector<ParallelizationCandidate> parallelizableLoops;
  SmallVector<affine::AffineForOp, 6> nonParallelizableLoops;
  funcOp.walk<WalkOrder::PreOrder>([&](affine::AffineForOp loop) {
    SmallVector<affine::LoopReduction> reductions;
    if (mlir::affine::isLoopParallelAKG(loop, parallelReductions ? &reductions : nullptr) && !smallLoop[loop]) {
      parallelizableLoops.emplace_back(loop, std::move(reductions));
    } else {
      nonParallelizableLoops.push_back(loop);
    }
  });

  for (const ParallelizationCandidate &candidate : parallelizableLoops) {
    affine::AffineForOp loop = candidate.loop;
    if (!isAncestorLoopParallel(nonParallelizableLoops, loop)) {
      continue;
    }
    unsigned numParentParallelOps = 0;
    for (Operation *op = loop->getParentOp(); op != nullptr && !op->hasTrait<OpTrait::AffineScope>();
         op = op->getParentOp()) {
      if (isa<affine::AffineParallelOp>(op)) {
        ++numParentParallelOps;
      }
    }

    if (numParentParallelOps < maxNested) {
      if (failed(affineParallelize(loop, candidate.reductions))) {
        LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] failed to parallelize\n" << loop);
      }
    } else {
      LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] too many nested loops\n" << loop);
    }
  }
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createAKGLoopParallelizePass() {
  return std::make_unique<AKGLoopParallelize>();
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createAKGLoopParallelizePass(const bool enableParallel) {
  return std::make_unique<AKGLoopParallelize>(enableParallel);
}
