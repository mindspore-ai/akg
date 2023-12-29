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

#include "akg/Dialect/Affine/Transforms/AffineTailBlockTiling.h"

#include "akg/Utils/AnalysisCommon.hpp"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/NestedMatcher.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {
#define GEN_PASS_DECL_AFFINETAILBLOCKTILING
#define GEN_PASS_DEF_AFFINETAILBLOCKTILING
#include "akg/Dialect/Affine/Passes.h.inc"
}  // namespace mlir

#define DEBUG_TYPE "affine-tail-block-tiling"

using namespace mlir;

namespace mlir {

struct AffineTailBlockTiling : public impl::AffineTailBlockTilingBase<AffineTailBlockTiling> {
  AffineTailBlockTiling() {}
  AffineTailBlockTiling(const std::string &target, const std::string &feature) : target(target), feature(feature) {}

  void runOnOperation() override;
  LogicalResult tailBlockTiling(func::FuncOp func, AffineForOp rootLoop);

  std::string target = kTargetCpu;
  std::string feature = kNEONInstructionSet;
};
}  // namespace mlir

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

static int64_t getDifferenceUbAndLb(AffineMap ubMap, AffineMap lbMap) {
  auto lbMapDim = lbMap.getNumDims();
  auto lbMapSymbol = lbMap.getNumSymbols();
  if (lbMapDim != ubMap.getNumDims() || lbMapSymbol != ubMap.getNumSymbols()) {
    return -1;
  }
  // todo: extend this to handle multiple result maps.
  if (lbMap.getNumResults() != 1 || ubMap.getNumResults() != 1) {
    return -1;
  }

  auto constExpr = simplifyAffineExpr(ubMap.getResult(0) - lbMap.getResult(0), lbMapDim, lbMapSymbol);
  if (auto cExpr = constExpr.dyn_cast<AffineConstantExpr>()) {
    return cExpr.getValue();
  }
  return -1;
}

// Updates the upper bound of all users of the trailing block for loop.
static void updateForOpUsers(AffineForOp forOp, int64_t newSize) {
  if (!newSize) {
    return;
  }
  for (OpOperand &use : forOp.getInductionVar().getUses()) {
    if (auto tiledOp = dyn_cast<AffineForOp>(use.getOwner())) {
      auto ubMap = tiledOp.getUpperBoundMap();
      auto newExpr = tiledOp.getLowerBoundMap().getResult(0) + newSize;
      ubMap = ubMap.replace(ubMap.getResult(0), newExpr, ubMap.getNumDims(), ubMap.getNumSymbols());
      tiledOp.setUpperBoundMap(ubMap);
    }
  }
}

LogicalResult AffineTailBlockTiling::tailBlockTiling(func::FuncOp func, AffineForOp rootLoop) {
  OperatorTemplate opType = CommonUtils::getOperatorType(func);
  ReduceDirection reduceDirection = CommonUtils::getReduceDirection(func);
  AffineForOp tileLoop = nullptr;
  if (reduceDirection == ReduceDirection::ALL) {
    tileLoop = rootLoop;
  } else if (reduceDirection == ReduceDirection::Y) {
    SmallVector<Operation *, 8> reduceLoops = CommonUtils::collectReductionAxes(func);
    tileLoop = dyn_cast<AffineForOp>(reduceLoops[0]->getParentOp());
  } else {
    rootLoop.walk([&](AffineForOp op) {
      if (tileLoop == nullptr) {
        tileLoop = op;
      }
    });
  }

  auto origUbMap = tileLoop.getUpperBoundMap();
  auto origLbMap = tileLoop.getLowerBoundMap();
  auto iter = cpuInstructionSetMap.find(feature);
  int64_t instructionSetBit = iter->second;
  int64_t vectorSize = getVectorSize(func, instructionSetBit);
  int64_t differenceUbAndLb = getDifferenceUbAndLb(origUbMap, origLbMap);
  if (differenceUbAndLb < 0) {
    tileLoop.emitError("Error: Could not get the difference between upper and lower bounds of the loop.");
    return failure();
  }
  int64_t tailSize = differenceUbAndLb % vectorSize;
  if (tailSize == 0 || (differenceUbAndLb < vectorSize)) {
    return success();
  }
  auto ubMap = origUbMap;
  AffineExpr newExpr = ubMap.getResult(0) - tailSize;
  ubMap = ubMap.replace(ubMap.getResult(0), newExpr, ubMap.getNumDims(), ubMap.getNumSymbols());
  auto newDifferenceUbAndLb = getDifferenceUbAndLb(origUbMap, ubMap);
  if (newDifferenceUbAndLb < 0) {
    tileLoop.emitError("Error: Could not get the difference between upper and lower bounds of the loop.");
    return failure();
  }
  // differenceUbAndLb < vectorSize: If the difference between the upper and lower bounds is less than step, the
  // body block does not need to be inserted.
  // !newDifferenceUbAndLb: If the difference between the upper and lower bounds of the new trailing block is 0, the
  // trailing block does not need to be inserted.
  if (differenceUbAndLb < vectorSize && newDifferenceUbAndLb) {
    // Only insert the tail tiles.
    tileLoop.setLowerBoundMap(ubMap);
    tileLoop.setUpperBoundMap(origUbMap);
    updateForOpUsers(tileLoop, tailSize);
  } else if (differenceUbAndLb >= vectorSize) {
    // Insert the full tiles
    tileLoop.setUpperBoundMap(ubMap);
  }

  OpBuilder b(tileLoop);
  b.setInsertionPointAfter(tileLoop);
  AffineForOp tailLoop = dyn_cast<AffineForOp>(b.clone(*tileLoop.getOperation()));
  tailLoop.setLowerBoundMap(ubMap);
  tailLoop.setUpperBoundMap(origUbMap);
  replaceAllUsesInRegionWith(tailLoop.getInductionVar(), tailLoop.getInductionVar(), tailLoop.getRegion());
  updateForOpUsers(tailLoop, tailSize);
  tailLoop.getOperation()->setAttr("tailBlock", b.getUnitAttr());
  return success();
}

void AffineTailBlockTiling::runOnOperation() {
  func::FuncOp func = getOperation();
  SmallVector<AffineForOp, 6> rootLoops;
  for (auto rootLoop : func.getOps<AffineForOp>()) {
    rootLoops.push_back(rootLoop);
  }
  for (auto rootLoop : rootLoops) {
    tailBlockTiling(func, rootLoop);
  }
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createAffineTailBlockTilingPass() {
  return std::make_unique<AffineTailBlockTiling>();
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createAffineTailBlockTilingPass(const std::string &target,
                                                                                   const std::string &feature) {
  return std::make_unique<AffineTailBlockTiling>(target, feature);
}
