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

#include "akg/Dialect/Affine/Transforms/AKGVectorize.h"

#include "akg/Utils/AKGGlobalVars.hpp"
#include "akg/Utils/AnalysisCommon.hpp"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/NestedMatcher.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

using namespace akgglobal;
namespace mlir {
#ifndef GEN_PASS_DEF_AKGAFFINEVECTORIZE
#define GEN_PASS_DEF_AKGAFFINEVECTORIZE
#ifndef GEN_PASS_DECL_AKGAFFINEVECTORIZE
#define GEN_PASS_DECL_AKGAFFINEVECTORIZE
#include "akg/Dialect/Affine/Passes.h.inc"
#endif
#endif
}  // namespace mlir

using namespace mlir;
using namespace vector;
using namespace mlir::akg;

#define DEBUG_TYPE "akg-affine-vectorize"

using llvm::dbgs;

namespace {

/// Base state for the vectorize pass.
/// Command line arguments are preempted by non-empty pass arguments.
struct AKGVectorize : public impl::AKGAffineVectorizeBase<AKGVectorize> {
  AKGVectorize() {}
  AKGVectorize(const std::string &target, const std::string &feature) : target(target), feature(feature) {}

  void vectorizeLoopsAccordingToStep(Operation *parentOp, const DenseSet<Operation *> &loops,
                                     const ReductionLoopMap &reductionLoops);
  void runOnOperation() override;
  void runCpuOperation();

  std::string target = kTargetCpu;
  std::string feature = kNEONInstructionSet;
};

}  // namespace

static int64_t getVectorSize(Operation *cpuOp, const int64_t instructionSetBit = kVectorize128Bit) {
  int64_t vectorSize = instructionSetBit;
  cpuOp->walk([&vectorSize, instructionSetBit](Operation *op) {
    if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
      MemRefType memRefType = loadOp.getMemRefType();
      Type elementType = memRefType.getElementType();
      int64_t elementBit = static_cast<int64_t>(elementType.getIntOrFloatBitWidth());
      vectorSize = std::min(vectorSize, instructionSetBit / elementBit);
    }
  });
  return vectorSize;
}

// To support tail block vectorization, set the final vectorization size according to the trip count of the for loop and
// vectorSize.
void AKGVectorize::vectorizeLoopsAccordingToStep(Operation *parentOp, const DenseSet<Operation *> &loops,
                                                 const ReductionLoopMap &reductionLoops) {
  // Compute 1-D loop pattern to be matched on the target loops.
  Optional<NestedPattern> pattern = makePatternAKG(loops, 1, {0});
  if (!pattern) {
    LLVM_DEBUG(dbgs() << "\n[early-vect] pattern couldn't be computed\n");
    return;
  }

  unsigned patternDepth = pattern->getDepth();

  // Compute all the pattern matches and classify them into buckets of
  // intersecting matches.
  SmallVector<NestedMatch, 32> allMatches;
  pattern->match(parentOp, &allMatches);
  std::vector<SmallVector<NestedMatch, 8>> intersectionBuckets;
  computeIntersectionBucketsAKG(allMatches, intersectionBuckets);

  // Iterate over all buckets and vectorize the matches eagerly. We can only
  // vectorize one match from each bucket since all the matches within a bucket
  // intersect.
  for (auto &intersectingMatches : intersectionBuckets) {
    for (NestedMatch &match : intersectingMatches) {
      // Set the final vectorization size based on the trip count of the for loop and vectorSize.
      std::vector<SmallVector<AffineForOp, 2>> loopsToVectorize;
      getMatchedAffineLoopsAKG(match, loopsToVectorize);
      AffineForOp forOp = loopsToVectorize[0][0];
      Optional<uint64_t> tripCount = getConstantTripCount(forOp);
      if (!tripCount) {
        continue;
      }

      // If the trip count is less than vectorSize, vectorization is not performed on the current tail block.
      // todo: tripCount = 3
      if (*tripCount % static_cast<uint64_t>(vectorSize) != 0) {
        continue;
      }

      VectorizationStrategy strategy;
      strategy.vectorSizes.push_back(vectorSize);
      strategy.reductionLoops = reductionLoops;
      if (failed(analyzeProfitabilityAKG(match.getMatchedChildren(), 1, patternDepth, &strategy))) {
        continue;
      }
      vectorizeLoopIfProfitableAKG(match.getMatchedOperation(), 0, patternDepth, &strategy);
      // Vectorize match. Skip the rest of intersecting matches in the bucket if
      // vectorization succeeded.
      if (succeeded(vectorizeLoopNestAKG(loopsToVectorize, strategy))) {
        break;
      }
    }
  }
}

void AKGVectorize::runCpuOperation() {
  Operation *cpuOp = getOperation();
  ReduceDirection reduceDirection = CommonUtils::getReduceDirection(cpuOp);

  if (reduceDirection == ReduceDirection::Y) {
    vectorizeReductions = false;
  }

  DenseSet<Operation *> parallelLoops;
  ReductionLoopMap reductionLoops;

  // If 'vectorize-reduction=true' is provided, we also populate the
  // `reductionLoops` map.
  if (vectorizeReductions) {
    cpuOp->walk([&parallelLoops, &reductionLoops](const AffineForOp loop) {
      SmallVector<LoopReduction, 2> reductions;
      if (mlir::isLoopParallel(loop, &reductions)) {
        (void)parallelLoops.insert(loop);
        // If it's not a reduction loop, adding it to the map is not necessary.
        if (!reductions.empty()) {
          reductionLoops[loop] = reductions;
        }
      }
    });
  } else {
    cpuOp->walk([&parallelLoops](AffineForOp loop) {
      Operation *curOp = loop.getOperation();
      if (mlir::isLoopParallel(loop) && !curOp->getAttr("reduceLoop")) {
        (void)parallelLoops.insert(loop);
      }
    });
  }

  if (vectorSize == 0) {
    auto iter = cpuInstructionSetMap.find(feature);
    if (iter == cpuInstructionSetMap.end()) {
      cpuOp->emitError(
        "The instruction set supported by the cpu only includes "
        "sse, avx, avx2, avx512 and neon.\n");
      return;
    }
    int64_t instructionSetBit = iter->second;
    vectorSize = getVectorSize(cpuOp, instructionSetBit);
  }

  // Thread-safe RAII local context, BumpPtrAllocator freed on exit.
  NestedPatternContext mlContext;
  vectorizeLoopsAccordingToStep(cpuOp, parallelLoops, reductionLoops);
}

/// Applies vectorization to the current function by searching over a bunch of
/// predetermined patterns.
void AKGVectorize::runOnOperation() {
  if (target == kTargetCpu) {
    runCpuOperation();
  } else if (GpuScheduleTool::getInstance().enableVectorize) {
    auto gpuOp = getOperation();
    DenseSet<Operation *> parallelLoops;
    ReductionLoopMap reductionLoops;
    gpuOp->walk([&parallelLoops](AffineForOp loop) {
      Operation *curOp = loop.getOperation();
      if (mlir::isLoopParallel(loop) && !curOp->getAttr("reduceLoop")) {
        (void)parallelLoops.insert(loop);
      }
    });
    mlir::vectorizeAffineLoops(gpuOp, parallelLoops, {GpuScheduleTool::getInstance().vectorSize}, {0}, reductionLoops);
  }
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createAKGVectorizePass() { return std::make_unique<AKGVectorize>(); }

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createAKGVectorizePass(const std::string &target,
                                                                          const std::string &feature) {
  return std::make_unique<AKGVectorize>(target, feature);
}
