/**
 * Copyright 2023-2026 Huawei Technologies Co., Ltd
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

#include <utility>
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
#include "akg/Utils/SmallVectorSize.h"

namespace mlir {
#define GEN_PASS_DECL_AFFINETAILBLOCKTILING
#define GEN_PASS_DEF_AFFINETAILBLOCKTILING
#include "akg/Dialect/Affine/Passes.h.inc"

}  // namespace mlir

namespace {
using mlir::AffineExpr;
using mlir::AffineMap;
using mlir::cast;
using mlir::CommonUtils;
using mlir::cpuInstructionSetMap;
using mlir::dyn_cast;
using mlir::failure;
using mlir::isa;
using mlir::kNEONInstructionSet;
using mlir::kSmallVectorSizeEight;
using mlir::kSmallVectorSizeSix;
using mlir::kTargetCpu;
using mlir::kVectorize128Bit;
using mlir::LogicalResult;
using mlir::MemRefType;
using mlir::Operation;
using mlir::OperationPass;
using mlir::ReduceDirection;
using mlir::replaceAllUsesInRegionWith;
using mlir::simplifyAffineExpr;
using mlir::SmallVector;
using mlir::SmallVectorImpl;
using mlir::success;
using mlir::Type;
}  // namespace

namespace mlir {

struct AffineTailBlockTiling : public mlir::impl::AffineTailBlockTilingBase<AffineTailBlockTiling> {
  AffineTailBlockTiling() {}
  AffineTailBlockTiling(std::string target, std::string feature)
      : target(std::move(target)), feature(std::move(feature)) {}

  void runOnOperation() override;
  LogicalResult tailBlockTiling(mlir::func::FuncOp func, mlir::affine::AffineForOp rootLoop);

  std::string target = kTargetCpu;
  std::string feature = kNEONInstructionSet;
};
}  // namespace mlir

static int64_t getVectorSize(Operation *cpuOp, const int64_t instructionSetBit = kVectorize128Bit) {
  int64_t vectorSize = instructionSetBit;
  cpuOp->walk([&vectorSize, instructionSetBit](Operation *op) {
    if (auto loadOp = dyn_cast<mlir::affine::AffineLoadOp>(op)) {
      MemRefType memRefType = loadOp.getMemRefType();
      Type elementType = memRefType.getElementType();
      auto elementBit = static_cast<int64_t>(elementType.getIntOrFloatBitWidth());
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
  // Extend this to handle multiple result maps.
  if (lbMap.getNumResults() != 1 || ubMap.getNumResults() != 1) {
    return -1;
  }

  auto constExpr = simplifyAffineExpr(ubMap.getResult(0) - lbMap.getResult(0), lbMapDim, lbMapSymbol);
  if (auto cExpr = llvm::dyn_cast<mlir::AffineConstantExpr>(constExpr)) {
    return cExpr.getValue();
  }
  return -1;
}

// Updates the upper bound of all users of the trailing block for loop.
static void updateForOpUsers(mlir::affine::AffineForOp forOp, int64_t newSize) {
  if (newSize == 0) {
    return;
  }
  for (mlir::OpOperand &use : forOp.getInductionVar().getUses()) {
    if (auto tiledOp = dyn_cast<mlir::affine::AffineForOp>(use.getOwner())) {
      auto ubMap = tiledOp.getUpperBoundMap();
      auto newExpr = tiledOp.getLowerBoundMap().getResult(0) + newSize;
      ubMap = ubMap.replace(ubMap.getResult(0), newExpr, ubMap.getNumDims(), ubMap.getNumSymbols());
      tiledOp.setUpperBoundMap(ubMap);
    }
  }
}

LogicalResult mlir::AffineTailBlockTiling::tailBlockTiling(mlir::func::FuncOp func,
                                                           mlir::affine::AffineForOp rootLoop) {
  ReduceDirection reduceDirection = CommonUtils::getReduceDirection(func);
  mlir::affine::AffineForOp tileLoop = nullptr;
  if (reduceDirection == ReduceDirection::ALL) {
    tileLoop = rootLoop;
  } else if (reduceDirection == ReduceDirection::Y) {
    SmallVector<Operation *, kSmallVectorSizeEight> reduceLoops = CommonUtils::collectReductionAxes(func);
    tileLoop = dyn_cast<mlir::affine::AffineForOp>(reduceLoops[0]->getParentOp());
  } else {
    rootLoop.walk([&tileLoop](mlir::affine::AffineForOp op) {
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
  if (differenceUbAndLb < vectorSize && (newDifferenceUbAndLb != 0)) {
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
  mlir::affine::AffineForOp tailLoop = dyn_cast<mlir::affine::AffineForOp>(b.clone(*tileLoop.getOperation()));
  tailLoop.setLowerBoundMap(ubMap);
  tailLoop.setUpperBoundMap(origUbMap);
  replaceAllUsesInRegionWith(tailLoop.getInductionVar(), tailLoop.getInductionVar(), tailLoop.getRegion());
  updateForOpUsers(tailLoop, tailSize);
  tailLoop.getOperation()->setAttr("tailBlock", b.getUnitAttr());
  return success();
}

void mlir::AffineTailBlockTiling::runOnOperation() {
  mlir::func::FuncOp func = getOperation();
  SmallVector<mlir::affine::AffineForOp, kSmallVectorSizeSix> rootLoops;
  std::copy(func.getOps<mlir::affine::AffineForOp>().begin(), func.getOps<mlir::affine::AffineForOp>().end(),
            std::back_inserter(rootLoops));
  if (std::any_of(rootLoops.begin(), rootLoops.end(),
                  [&](mlir::affine::AffineForOp rootLoop) { return failed(tailBlockTiling(func, rootLoop)); })) {
    return signalPassFailure();
  }
}

namespace mlir {
std::unique_ptr<OperationPass<mlir::func::FuncOp>> createAffineTailBlockTilingPass() {
  return std::make_unique<AffineTailBlockTiling>();
}
}  // namespace mlir

namespace mlir {
std::unique_ptr<OperationPass<mlir::func::FuncOp>> createAffineTailBlockTilingPass(const std::string &target,
                                                                                   const std::string &feature) {
  return std::make_unique<AffineTailBlockTiling>(target, feature);
}
}  // namespace mlir
