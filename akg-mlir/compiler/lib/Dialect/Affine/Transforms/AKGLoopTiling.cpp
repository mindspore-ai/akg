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

#include "akg/Dialect/Affine/Transforms/AKGLoopTiling.h"

#include <unordered_set>
#include "akg/Dialect/Affine/Analysis/AutoTiling.h"
#include "akg/Utils/AKGGlobalVars.hpp"
#include "akg/Utils/AnalysisCommon.hpp"
#include "akg/Utils/AnalysisForGpu.hpp"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {
#define GEN_PASS_DECL_AKGAFFINELOOPTILING
#define GEN_PASS_DEF_AKGAFFINELOOPTILING
#include "akg/Dialect/Affine/Passes.h.inc"
}  // namespace mlir

using namespace mlir;
using namespace akg::autotiling;
using namespace akgglobal;
using namespace mlir::akg::utils;

#define DEBUG_TYPE "akg-affine-loop-tile"

namespace {

/// A pass to perform loop tiling on all suitable loop nests of a Function.
class AKGLoopTiling : public impl::AKGAffineLoopTilingBase<AKGLoopTiling> {
 public:
  AKGLoopTiling() = default;
  AKGLoopTiling(uint64_t cacheSizeBytes, bool avoidMaxMinBounds = true) : avoidMaxMinBounds(avoidMaxMinBounds) {
    this->cacheSizeInKiB = cacheSizeBytes / 1024;
  }

  AKGLoopTiling(const std::string &target, bool useAutoTiling = false) : target(target) {
    this->useAutoTiling = useAutoTiling;
  }

  AKGLoopTiling(const std::string &target, bool useAutoTiling, const std::string &tilingMode) : target(target) {
    this->useAutoTiling = useAutoTiling;
    this->tilingMode = tilingMode;
  }

  AKGLoopTiling(const std::string &target, const std::string &feature, bool useAutoTiling = false)
      : target(target), feature(feature) {
    this->useAutoTiling = useAutoTiling;
  }

  void runOnOperation() override;

 private:
  void runCpuOperation();
  void runCudaOperation();
  void BandCheck(const std::vector<SmallVector<affine::AffineForOp, 6>> &bands);
  void getTileSizes();
  std::string getHardware();
  bool isDynamicShape() const;

  // core tiling function
  void tileEachBand();
  // initial tiling function
  void constructTiledLoop(affine::AffineForOp rootAffineForOp, unsigned width,
                          MutableArrayRef<affine::AffineForOp> tiledLoops);
  void constructTiledIndex(MutableArrayRef<affine::AffineForOp> newLoops);
  void setInsertInequality(int curTile, bool &insertInequality);
  void setNewUpperBound(MutableArrayRef<affine::AffineForOp> newLoops, int curTile, bool insertInequality = true);

  // tile tail block
  void updateForOpUsers(affine::AffineForOp forOp, int64_t newSize = 0);
  LogicalResult createTailBlockForBody(affine::AffineForOp forOp);
  LogicalResult createTailBlock(affine::AffineForOp forOp);
  LogicalResult createFullBlock(MutableArrayRef<affine::AffineForOp> tiledLoops,
                                SmallVectorImpl<affine::AffineForOp> &fullTileLoops);
  LogicalResult createTailBlockDynamic(affine::AffineForOp forOp, AffineSymbolExpr sExpr);
  LogicalResult createTailBlockStatic(affine::AffineForOp forOp, int64_t differenceUbAndLb);

  LogicalResult separateFullTilesNoIf(SmallVector<affine::AffineForOp, 6> tiledLoops);

  LogicalResult perfectlyNestedWithIf(SmallVector<affine::AffineForOp, 6> tiledLoops);
  void updateInsertIfLoops(SmallVector<affine::AffineForOp, 6> &newTiledLoops,
                           std::unordered_set<unsigned> inequalityForIndex);
  affine::AffineIfOp createperfectlyNestedCondition(SmallVector<affine::AffineForOp, 6> tiledLoops, OpBuilder b);

  // If true, tile sizes are set to avoid max/min in bounds if possible.
  bool avoidMaxMinBounds{true};
  // hardware information
  std::string target{kTargetCpu};
  std::string feature{kNEONInstructionSet};
  std::string tilingMode{"auto"};

  TilingSolverPtr solver{nullptr};
  size_t levelToTile{1};

  SmallVector<unsigned, 6> bandTileSizes;
  MutableArrayRef<affine::AffineForOp> band;
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createAKGLoopTilingPass() {
  return std::make_unique<AKGLoopTiling>();
}

/// Creates a pass to perform loop tiling on all suitable loop nests of a
/// Function.
std::unique_ptr<OperationPass<func::FuncOp>> mlir::createAKGLoopTilingPass(uint64_t cacheSizeBytes) {
  return std::make_unique<AKGLoopTiling>(cacheSizeBytes);
}
/// Creates a pass to perform loop tiling using auto-tiling strategy
std::unique_ptr<OperationPass<func::FuncOp>> mlir::createAKGLoopTilingPass(const std::string &target,
                                                                           bool useAutoTiling = false) {
  return std::make_unique<AKGLoopTiling>(target, useAutoTiling);
}

/// Creates a pass to perform loop tiling using auto-tiling strategy for dynamic shape
std::unique_ptr<OperationPass<func::FuncOp>> mlir::createAKGLoopTilingPass(const std::string &target,
                                                                           bool useAutoTiling = true,
                                                                           const std::string &tilingMode = "auto") {
  return std::make_unique<AKGLoopTiling>(target, useAutoTiling, tilingMode);
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createAKGLoopTilingPass(const std::string &target,
                                                                           const std::string &feature,
                                                                           bool useAutoTiling = false) {
  return std::make_unique<AKGLoopTiling>(target, feature, useAutoTiling);
}

static void moveLoopBody(affine::AffineForOp src, affine::AffineForOp dest) {
  Block::iterator loc = dest.getBody()->begin();
  auto &ops = src.getBody()->getOperations();
  dest.getBody()->getOperations().splice(loc, ops, ops.begin(), std::prev(ops.end()));
}

static AffineExpr getDifferenceUbAndLb(AffineMap ubMap, AffineMap lbMap) {
  auto maxDim = std::max(lbMap.getNumDims(), ubMap.getNumDims());
  auto maxSymbol = std::max(lbMap.getNumSymbols(), ubMap.getNumSymbols());
  // TODO: extend this to handle multiple result maps.
  return simplifyAffineExpr(ubMap.getResult(0) - lbMap.getResult(0), maxDim, maxSymbol);
}

// Add new axis and initialize lowerbound/upperbound to 0
// The number of axes is determined by the number of tileSizes and the number of for loops,
// which is tileSizes.size() * band.size().

// ```mlir (tileSizes.size() = 2)
// affine.for %arg0 = 0 to 256 step 32 {
//   "test.foo"(%arg1) : (index) -> ()
// }
// -->
// affine.for %arg0 = 0 to 0 {
//   affine.for %arg1 = 0 to 0 {
//     affine.for %arg2 = 0 to 0 {
//       affine.for %arg3 = 0 to 256 step 32 {
//         "test.foo"(%arg3) : (index) -> ()
//     }
//   }
// }
// ```
void AKGLoopTiling::constructTiledLoop(affine::AffineForOp rootAffineForOp, unsigned width,
                                       MutableArrayRef<affine::AffineForOp> tiledLoops) {
  Location loc = rootAffineForOp.getLoc();

  Operation *topLoop = rootAffineForOp.getOperation();
  affine::AffineForOp innermostPointLoop;

  for (unsigned i = 0; i < width; ++i) {
    OpBuilder b(topLoop);
    // Loop bounds will be set later.
    affine::AffineForOp pointLoop = b.create<affine::AffineForOp>(loc, 0, 0);
    pointLoop.getBody()->getOperations().splice(pointLoop.getBody()->begin(), topLoop->getBlock()->getOperations(),
                                                topLoop);
    tiledLoops[width - 1 - i] = pointLoop;
    topLoop = pointLoop.getOperation();
    if (i == 0) {
      innermostPointLoop = pointLoop;
    }
  }

  moveLoopBody(band.back(), innermostPointLoop);
}

// Constructs and sets new loop bounds after tiling for the case of
// index sets, the algorithm is as follow:

// ```mlir (tile-sizes=32,4)
// affine.for %arg0 = 0 to 0 {
//   affine.for %arg1 = 0 to 0 {
//     affine.for %arg2 = 0 to 0 {
//       affine.for %arg3 = 0 to 256 step 32 {
//         "test.foo"(%arg3) : (index) -> ()
//       }
//     }
//   }
// }
// -->
// #map = affine_map<(d0) -> (d0)>
// #map1 = affine_map<(d0) -> (d0 + 32)>
// #map2 = affine_map<(d0) -> (d0 + 4)>
// affine.for %arg3 = 0 to 1024 step 32 {
//   affine.for %arg4 = #map(%arg3) to #map1(%arg3) step 4 {
//     affine.for %arg5 = #map(%arg4) to #map2(%arg4) {
//       "test.foo"(%arg5) : (index) -> ()
//     }
//   }
// }
// ```
void AKGLoopTiling::constructTiledIndex(MutableArrayRef<affine::AffineForOp> newLoops) {
  int bandSize = band.size();
  if (bandSize == 0) {
    return;
  }
  OpBuilder b(band[0].getOperation());
  int tileNum = bandTileSizes.size() / bandSize;
  // the i-th tiling
  for (int i = 0; i <= tileNum; ++i) {
    // the j-th axis
    for (int j = 0; j < bandSize; ++j) {
      // the i-th tiling of the j-th axis
      int curTile = i * bandSize + j;
      // the i-th tiling of the (j-1)th axis
      int lastTile = curTile - bandSize;
      if (i == 0) {
        // first tile
        OperandRange newLbOperands = band[j].getLowerBoundOperands();
        OperandRange newUbOperands = band[j].getUpperBoundOperands();
        newLoops[curTile].setLowerBound(newLbOperands, band[j].getLowerBoundMap());
        newLoops[curTile].setUpperBound(newUbOperands, band[j].getUpperBoundMap());
        newLoops[curTile].setStep(bandTileSizes[curTile]);
      } else if (i == tileNum) {
        // last tile
        AffineMap lbMap = b.getDimIdentityMap();
        newLoops[curTile].setLowerBound(newLoops[lastTile].getInductionVar(), lbMap);
        newLoops[curTile].setStep(1);

        setNewUpperBound(newLoops, curTile, true);
      } else {
        // middle tile
        AffineMap lbMap = b.getDimIdentityMap();
        newLoops[curTile].setLowerBound(newLoops[lastTile].getInductionVar(), lbMap);
        newLoops[curTile].setStep(bandTileSizes[curTile]);

        setNewUpperBound(newLoops, curTile, separateNoIf);
      }
    }
  }
  return;
}

// If the previous tile_size or the original shape of a certain axis cannot be evenly divided by the next tile_size,
// then all subsequent tiling, that axis must add an out-of-bounds check.
void AKGLoopTiling::setInsertInequality(int curTile, bool &insertInequality) {
  if (!insertInequality) {
    return;
  }

  int bandSize = band.size();
  int lastTile = curTile - bandSize;
  insertInequality = false;
  // Check whether it can be divided from the inside out.
  while (lastTile >= bandSize) {
    insertInequality |= (bandTileSizes[lastTile - bandSize] % bandTileSizes[lastTile] != 0);
    lastTile -= bandSize;
    if (insertInequality) {
      return;
    }
  }
  int64_t largestDiv = getLargestDivisorOfTripCount(band[lastTile]);
  insertInequality |= (largestDiv % bandTileSizes[lastTile] != 0);
}

// Set the upper bound of newLoops[curTile]. If insertInequality is true, the upper bound of the loop should be inserted
// with min/max based on whether the tile_size is divisible, otherwise min/max should not be inserted.
// ```mlir (tile-sizes=32,2)
// affine.for %arg1 = 0 to 145 {
//   "test.foo"(%arg1) : (index) -> ()
// }
// --> first tile
// #map = affine_map<(d0) -> (d0)>
// #map1 = affine_map<(d0) -> (d0 + 32, 145)>
// affine.for %arg1 = 0 to 145 step 32 {
//   affine.for %arg2 = #map(%arg1) to min #map1(%arg1) step 2 {
//     "test.foo"(%arg2) : (index) -> ()
//   }
// }
// --> second tile
// #map = affine_map<(d0) -> (d0)>
// #map1 = affine_map<(d0) -> (d0 + 32, 145)>
// #map2 = affine_map<(d0, d1) -> (d1 + 2, d0 + 32, 145)>
// affine.for %arg1 = 0 to 145 step 32 {
//   affine.for %arg2 = #map(%arg1) to min #map1(%arg1) step 2 {
//     affine.for %arg3 =  #map(%arg2) to min #map2(%arg1, %arg2) {
//       "test.foo"(%arg3) : (index) -> ()
//     }
//   }
// }
void AKGLoopTiling::setNewUpperBound(MutableArrayRef<affine::AffineForOp> newLoops, int curTile,
                                     bool insertInequality) {
  // Set the upper bound.
  int bandSize = band.size();
  int lastTile = curTile - bandSize;
  OpBuilder b(newLoops[0].getOperation());
  int64_t largestDiv = getLargestDivisorOfTripCount(band[curTile % bandSize]);

  setInsertInequality(curTile, insertInequality);
  if (insertInequality) {
    affine::AffineBound lastTileUb = newLoops[lastTile].getUpperBound();
    AffineMap lastTileUbMap = lastTileUb.getMap();
    SmallVector<Value, 4> ubOperands;
    ubOperands.reserve(lastTileUb.getNumOperands() + 1);
    // Add dim operands from upper bound of the last tile.
    for (unsigned k = 0; k < lastTileUbMap.getNumDims(); ++k) {
      ubOperands.push_back(lastTileUb.getOperand(k));
    }

    // Add dim operand for new loop upper bound.
    ubOperands.push_back(newLoops[lastTile].getInductionVar());

    // Add symbol operands from upper bound of the last tile.
    for (unsigned k = 0; k < lastTileUbMap.getNumSymbols(); ++k) {
      ubOperands.push_back(lastTileUb.getOperand(lastTileUbMap.getNumDims() + k));
    }

    // To ensure the correctness of the result, when inserting the out-of-bounds check, the upper bound of this axis
    // needs to be inserted.
    bool insertLoopUb = true;
    for (auto ubMap : lastTileUbMap.getResults()) {
      if (llvm::isa<AffineConstantExpr>(ubMap)) {
        insertLoopUb = false;
      }
    }
    SmallVector<AffineExpr, 4> ubExprs;
    unsigned newExprSize = 1 + lastTileUbMap.getNumResults();
    if (insertLoopUb) {
      ++newExprSize;
    }
    ubExprs.reserve(newExprSize);

    AffineExpr dimExpr = b.getAffineDimExpr(lastTileUbMap.getNumDims());
    ubExprs.push_back(dimExpr + bandTileSizes[lastTile]);
    ubExprs.append(lastTileUbMap.getResults().begin(), lastTileUbMap.getResults().end());
    if (insertLoopUb) {
      ubExprs.push_back(b.getAffineConstantExpr(largestDiv));
    }
    AffineMap ubMap =
      AffineMap::get(lastTileUbMap.getNumDims() + 1, lastTileUbMap.getNumSymbols(), ubExprs, b.getContext());
    newLoops[curTile].setUpperBound(ubOperands, ubMap);
  } else {
    AffineExpr dim = b.getAffineDimExpr(0);
    AffineMap ubMap = AffineMap::get(1, 0, dim + newLoops[lastTile].getStepAsInt());
    newLoops[curTile].setUpperBound(newLoops[lastTile].getInductionVar(), ubMap);
  }
}

void AKGLoopTiling::getTileSizes() {
  // TODO: Separately tile axis
  if (useAutoTiling && solver) {
    // TODO: remove levelToTile
    SmallVector<affine::AffineForOp, 6> curband;
    curband.assign(band.begin(), band.end());
    for (size_t level = 0; level < levelToTile; ++level) {
      // TODO: Multiple band
      getTileSizeWithSolver(solver, curband, &bandTileSizes, TilingTaskDesc(0, level));
    }
  } else {
    if (!tileSizes.empty() && tileSize == 1) {
      for (auto it : tileSizes) {
        bandTileSizes.push_back(it);
      }
    } else {
      bandTileSizes.assign(band.size(), tileSize);
    }
  }
  return;
}

// Remove min/max from AffineFor to generate Remove min/max from affinefor to generate the full tile loop nest.
// ```mlir (tile-sizes=32,2)
// #map = affine_map<(d0) -> (d0)>
// #map1 = affine_map<(d0) -> (d0 + 32, 145)>
// #map2 = affine_map<(d0, d1) -> (d1 + 2, d0 + 32, 145)>
// affine.for %arg1 = 0 to 145 step 32 {
//   affine.for %arg2 = #map(%arg1) to min #map1(%arg1) step 2 {
//     affine.for %arg3 =  #map(%arg2) to min #map2(%arg1, %arg2) {
//       "test.foo"(%arg3) : (index) -> ()
//     }
//   }
// }
// -->
// #map = affine_map<(d0) -> (d0)>
// #map1 = affine_map<(d0) -> (d0 + 32)>
// #map2 = affine_map<(d0) -> (d0 + 2)>
// affine.for %arg1 = 0 to 145 step 32 {
//   affine.for %arg2 = #map(%arg1) to #map1(%arg1) step 2 {
//     affine.for %arg3 =  #map(%arg2) to #map2(%arg2) {
//       "test.foo"(%arg3) : (index) -> ()
//     }
//   }
// }
// ```
LogicalResult AKGLoopTiling::createFullBlock(MutableArrayRef<affine::AffineForOp> intraTileLoops,
                                             SmallVectorImpl<affine::AffineForOp> &fullTileLoops) {
  if (intraTileLoops.size() == 0) {
    return success();
  }
  OpBuilder b(intraTileLoops[0]);
  fullTileLoops.reserve(intraTileLoops.size());

  // For each loop in the original nest identify a lower/upper bound pair such
  // that their difference is a constant.
  affine::FlatAffineValueConstraints cst;
  for (auto loop : intraTileLoops) {
    SmallVector<Operation *, 1> loopOp{loop.getOperation()};
    (void)getIndexSet(loopOp, &cst);
    // We will mark everything other than this loop IV as symbol for getting a
    // pair of <lb, ub> with a constant difference.
    cst.setDimSymbolSeparation(cst.getNumDimAndSymbolVars() - 1);
    unsigned lbPos, ubPos;
    if (!cst.getConstantBoundOnDimSize(0, nullptr, nullptr, nullptr, &lbPos, &ubPos) || lbPos == ubPos) {
      LLVM_DEBUG(llvm::dbgs() << "[tile separation] Can't get constant diff / "
                                 "equalities not yet handled\n");
      return failure();
    }

    // Set all variables as dimensions uniformly since some of those marked as
    // symbols above could be outer loop IVs (corresponding tile space IVs).
    cst.setDimSymbolSeparation(0);

    affine::AffineValueMap lbVmap, ubVmap;
    cst.getIneqAsAffineValueMap(0, lbPos, lbVmap, b.getContext());
    cst.getIneqAsAffineValueMap(0, ubPos, ubVmap, b.getContext());

    affine::AffineForOp fullTileLoop =
      affine::createCanonicalizedAffineForOp(b, loop.getLoc(), lbVmap.getOperands(), lbVmap.getAffineMap(),
                                             ubVmap.getOperands(), ubVmap.getAffineMap(), loop.getStepAsInt());
    b = OpBuilder::atBlockTerminator(fullTileLoop.getBody());
    fullTileLoops.push_back(fullTileLoop);
  }

  // Add the body for the full tile loop nest.
  IRMapping operandMap;
  for (const auto &loopEn : llvm::enumerate(intraTileLoops)) {
    operandMap.map(loopEn.value().getInductionVar(), fullTileLoops[loopEn.index()].getInductionVar());
  }
  b = OpBuilder::atBlockTerminator(fullTileLoops.back().getBody());
  for (auto &op : intraTileLoops.back().getBody()->without_terminator()) {
    b.clone(op, operandMap);
  }
  // Add the body for the full tile loop nest.
  for (const auto &loopEn : llvm::enumerate(intraTileLoops)) {
    replaceAllUsesInRegionWith(loopEn.value().getInductionVar(), fullTileLoops[loopEn.index()].getInductionVar(),
                               fullTileLoops[loopEn.index()].getRegion());
  }

  // insert the full block, replacing the original nested for
  Block *intraBlock = intraTileLoops[0].getOperation()->getBlock();
  affine::AffineForOp outermostFullTileLoop = fullTileLoops[0];
  intraBlock->getOperations().splice(std::prev(intraBlock->end()), outermostFullTileLoop->getBlock()->getOperations(),
                                     Block::iterator(outermostFullTileLoop));
  intraTileLoops[0].erase();
  return success();
}

LogicalResult AKGLoopTiling::createTailBlockForBody(affine::AffineForOp forOp) {
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (auto bodyOp = dyn_cast<affine::AffineForOp>(op)) {
      if (failed(createTailBlock(bodyOp))) {
        return failure();
      }
    }
  }
  return success();
}

// Updates the upper bound of all users of the trailing block for loop.
void AKGLoopTiling::updateForOpUsers(affine::AffineForOp forOp, int64_t newSize) {
  if (!newSize) {
    return;
  }
  for (OpOperand &use : forOp.getInductionVar().getUses()) {
    if (auto tiledOp = dyn_cast<affine::AffineForOp>(use.getOwner())) {
      auto ubMap = tiledOp.getUpperBoundMap();
      auto newExpr = tiledOp.getLowerBoundMap().getResult(0) + newSize;
      ubMap = ubMap.replace(ubMap.getResult(0), newExpr, ubMap.getNumDims(), ubMap.getNumSymbols());
      tiledOp.setUpperBoundMap(ubMap);
    }
  }
}

// According to the upper and lower bounds of the loop and the corresponding step, the perfect nested loop is split into
// two parallel loops.

// ```mlir (tile-sizes=32,2)
// #map = affine_map<(d0) -> (d0)>
// #map1 = affine_map<(d0) -> (d0 + 32)>
// #map2 = affine_map<(d0) -> (d0 + 2)>
// affine.for %arg1 = 0 to 145 step 32 {
//   affine.for %arg2 = #map(%arg1) to #map1(%arg1) step 2 {
//     affine.for %arg3 =  #map(%arg2) to #map2(%arg2) {
//       "test.foo"(%arg3) : (index) -> ()
//     }
//   }
// }
//   -->
// #map = affine_map<(d0) -> (d0)>
// #map1 = affine_map<(d0) -> (d0 + 32)>
// #map2 = affine_map<(d0) -> (d0 + 2)>
// #map3 = affine_map<(d0) -> (d0 + 16)>
// #map4 = affine_map<(d0) -> (d0 + 17)>
// #map5 = affine_map<(d0) -> (d0 + 1)>
// affine.for %arg1 = 0 to 128 step 32 {
//   affine.for %arg2 = #map(%arg1) to #map1(%arg1) step 2 {
//     affine.for %arg3 =  #map(%arg2) to #map2(%arg2) {
//       "test.foo"(%arg3) : (index) -> ()
//     }
//   }
// }
// affine.for %arg1 = 128 to 145 step 17 {
//   affine.for %arg2 = #map(%arg1) to #map3(%arg1) step 2 {
//     affine.for %arg3 =  #map(%arg2) to #map2(%arg2) {
//       "test.foo"(%arg3) : (index) -> ()
//     }
//   }
//   affine.for %arg2 = #map3(%arg1) to #map4(%arg1) step 2 {
//     affine.for %arg3 =  #map(%arg2) to #map5(%arg2) {
//       "test.foo"(%arg3) : (index) -> ()
//     }
//   }
// }
// ```
LogicalResult AKGLoopTiling::createTailBlock(affine::AffineForOp forOp) {
  auto origUbMap = forOp.getUpperBoundMap();
  auto origLbMap = forOp.getLowerBoundMap();
  AffineExpr differenceExpr = getDifferenceUbAndLb(origUbMap, origLbMap);
  if (auto cExpr = llvm::dyn_cast<AffineConstantExpr>(differenceExpr)) {
    return createTailBlockStatic(forOp, cExpr.getValue());
  } else if (auto sExpr = llvm::dyn_cast<AffineSymbolExpr>(differenceExpr)) {
    return createTailBlockDynamic(forOp, sExpr);
  }
  return success();
}

LogicalResult AKGLoopTiling::createTailBlockStatic(affine::AffineForOp forOp, int64_t differenceUbAndLb) {
  auto origUbMap = forOp.getUpperBoundMap();
  auto origLbMap = forOp.getLowerBoundMap();
  int64_t origStep = forOp.getStepAsInt();
  int64_t tailSize = differenceUbAndLb % origStep;
  if (tailSize == 0) {
    // Recursively processes the forOp body.
    if (failed(createTailBlockForBody(forOp))) {
      return failure();
    }
    return success();
  }

  auto ubMap = origUbMap;
  AffineExpr newExpr = ubMap.getResult(0) - tailSize;
  ubMap = ubMap.replace(ubMap.getResult(0), newExpr, ubMap.getNumDims(), ubMap.getNumSymbols());
  int64_t newDifferenceUbAndLb = 0;
  AffineExpr differenceExpr = getDifferenceUbAndLb(origUbMap, ubMap);
  if (auto cExpr = llvm::dyn_cast<AffineConstantExpr>(differenceExpr)) {
    newDifferenceUbAndLb = cExpr.getValue();
  } else {
    forOp.emitError("Error: Could not get the difference between upper and lower bounds of the loop.");
    return failure();
  }
  // differenceUbAndLb < origStep: If the difference between the upper and lower bounds is less than step, the
  // body block does not need to be inserted.
  // !newDifferenceUbAndLb: If the difference between the upper and lower bounds of the new trailing block is 0, the
  // trailing block does not need to be inserted.
  if (differenceUbAndLb < origStep && newDifferenceUbAndLb) {
    // Only insert the tail tiles.
    forOp.setLowerBoundMap(ubMap);
    forOp.setUpperBoundMap(origUbMap);
    forOp.setStep(tailSize);
    updateForOpUsers(forOp, tailSize);
  } else if (differenceUbAndLb >= origStep) {
    // Insert the full tiles
    forOp.setUpperBoundMap(ubMap);
  }

  // Recursively processes the forOp body.
  if (failed(createTailBlockForBody(forOp))) {
    return failure();
  }

  // differenceUbAndLb < origStep: If the difference between the upper and lower bounds is less than step, the
  // body block does not need to be inserted.
  // !newDifferenceUbAndLb: If the difference between the upper and lower bounds of the new trailing block is 0, the
  // trailing block does not need to be inserted.
  // isEqualToBlock: If the upper and lower bounds and steps of the body block and tail block are the same, the tail
  // block does not need to be inserted.
  bool isEqualToBlock = (ubMap == origLbMap) && (origUbMap == ubMap) && (tailSize == origStep);
  if (differenceUbAndLb < origStep || !newDifferenceUbAndLb || isEqualToBlock) {
    return success();
  }

  // Insert the tail tiles
  OpBuilder b(forOp);
  b.setInsertionPointAfter(forOp);
  auto tailOp = b.clone(*forOp.getOperation());
  auto tailForOp = dyn_cast<affine::AffineForOp>(tailOp);
  tailForOp.setLowerBoundMap(ubMap);
  tailForOp.setUpperBoundMap(origUbMap);
  tailForOp.setStep(tailSize);
  replaceAllUsesInRegionWith(forOp.getInductionVar(), tailForOp.getInductionVar(), tailForOp.getRegion());
  updateForOpUsers(tailForOp, tailSize);

  // Recursively processes the tailForOp body.
  if (failed(createTailBlockForBody(tailForOp))) {
    return failure();
  }
  return success();
}

LogicalResult AKGLoopTiling::createTailBlockDynamic(affine::AffineForOp forOp, AffineSymbolExpr sExpr) {
  if (band.size() != bandTileSizes.size()) {
    forOp.emitError("Dynamic shape supports only one tiling.");
    return failure();
  }
  auto origUbMap = forOp.getUpperBoundMap();
  auto origUbOp = forOp.getUpperBoundOperands();
  int64_t origStep = forOp.getStepAsInt();
  // When the dynamic shape step is 1, the tail block does not need to be processed.
  // affine.for %arg4 = 0 to %dim
  if (origStep == 1) {
    // Recursively processes the forOp body.
    if (failed(createTailBlockForBody(forOp))) {
      return failure();
    }
    return success();
  }

  auto ubMap = origUbMap;
  AffineExpr newExpr = ubMap.getResult(0) - sExpr % origStep;
  ubMap = ubMap.replace(ubMap.getResult(0), newExpr, ubMap.getNumDims(), ubMap.getNumSymbols());
  forOp.setUpperBoundMap(ubMap);

  // Recursively processes the forOp body.
  if (failed(createTailBlockForBody(forOp))) {
    return failure();
  }

  // Insert the tail tiles
  OpBuilder b(forOp);
  b.setInsertionPointAfter(forOp);
  affine::AffineForOp tailForOp =
    b.create<affine::AffineForOp>(forOp.getLoc(), forOp.getUpperBoundOperands(), ubMap, origUbOp, origUbMap, 1);

  // Add the body for the full tile loop nest.
  b = OpBuilder::atBlockTerminator(tailForOp.getBody());
  for (auto &op : forOp.getBody()->without_terminator()) {
    b.clone(op);
  }
  // Add the body for the full tile loop nest.
  replaceAllUsesInRegionWith(forOp.getInductionVar(), tailForOp.getInductionVar(), tailForOp.getRegion());
  updateForOpUsers(tailForOp, 1);

  // Recursively processes the tailForOp body.
  if (failed(createTailBlockForBody(tailForOp))) {
    return failure();
  }
  return success();
}

LogicalResult AKGLoopTiling::separateFullTilesNoIf(SmallVector<affine::AffineForOp, 6> tiledLoops) {
  // full block
  auto intraTileLoops = MutableArrayRef<affine::AffineForOp>(tiledLoops).drop_front(band.size());
  SmallVector<affine::AffineForOp, 4> fullTileLoops;
  if (failed(createFullBlock(intraTileLoops, fullTileLoops))) {
    if (!fullTileLoops.empty()) {
      fullTileLoops.front().erase();
    }
    return failure();
  }
  if (fullTileLoops.size() == 0) {
    return success();
  }

  // tail block
  if (failed(createTailBlock(tiledLoops[0]))) {
    return failure();
  }

  return success();
}

// Insert the corresponding if statement based on the upper bound of the for loop to ensure the correctness of the
// result.
// ```mlir (tile-sizes=32,5)
// #map = affine_map<(d0) -> (d0)>
// #map1 = affine_map<(d0) -> (d0 + 32)>
// #map2 = affine_map<(d0) -> (d0 + 5)>
// affine.for %arg3 = 0 to 1024 step 32 {
//   affine.for %arg4 = #map(%arg3) to #map1(%arg3) step 5 {
//     affine.for %arg5 = #map(%arg4) to #map2(%arg4) {
//       "test.foo"(%arg5) : (index) -> ()
//     }
//   }
// }
//   -->
// #map = affine_map<(d0) -> (d0)>
// #map1 = affine_map<(d0) -> (d0 + 32)>
// #map2 = affine_map<(d0) -> (d0 + 5)>
// #set = affine_set<(d0) : (-d0 + 1023 >= 0)>
// affine.for %arg3 = 0 to 1024 step 32 {
//   affine.for %arg4 = #map(%arg3) to #map1(%arg3) step 5 {
//     affine.for %arg5 = #map(%arg4) to #map2(%arg4) {
//       affine.if #set(%arg5) {
//       }
//       "test.foo"(%arg5) : (index) -> ()
//     }
//   }
// }
// ```
affine::AffineIfOp AKGLoopTiling::createperfectlyNestedCondition(SmallVector<affine::AffineForOp, 6> tiledLoops,
                                                                 OpBuilder b) {
  if (tiledLoops.empty()) {
    return nullptr;
  }

  SmallVector<AffineExpr, 4> exprs;
  SmallVector<bool, 4> eqFlags;
  SmallVector<mlir::Value, 4> dimOperands;
  SmallVector<mlir::Value, 4> symbolOperands;
  int64_t symbolNum = 0;
  int64_t dimNum = 0;
  for (auto loop : tiledLoops) {
    assert(loop.getStepAsInt() == 1 && "point loop step expected to be one");

    // Find the outermost for loop, that is, the for loop before tiling.
    Operation *outerOp = nullptr;
    for (auto value : loop.getOperands()) {
      SmallVector<Operation *, 8> opAxes;
      CommonUtils::collectRelatedAxes(value, opAxes);
      if (opAxes.empty()) {
        continue;
      }
      for (auto axes : opAxes) {
        outerOp = CommonUtils::getInnerOrOuterOp(outerOp, axes, false);
      }
    }

    if (!outerOp || !isa<affine::AffineForOp>(outerOp)) {
      continue;
    }

    auto outerForOp = dyn_cast<affine::AffineForOp>(outerOp);
    auto context = loop.getContext();
    AffineExpr upperExpr = outerForOp.getUpperBoundMap().getResult(0);
    // Adapts to dynamic shapes.
    if (llvm::isa<AffineSymbolExpr>(upperExpr)) {
      for (auto operand : outerForOp->getOperands()) {
        if (Operation *parentOp = operand.getDefiningOp()) {
          symbolOperands.push_back(parentOp->getResult(0));
          upperExpr = mlir::getAffineSymbolExpr(symbolNum++, context);
        }
      }
    }
    // Make sure that the dim variable is incremented each time.
    AffineExpr newExpr = upperExpr - 1 - mlir::getAffineDimExpr(dimNum++, context);
    exprs.push_back(newExpr);
    eqFlags.push_back(false);
    dimOperands.push_back(loop.getInductionVar());
  }

  // Adapts to dynamic shapes: add symbol operands.
  if (symbolNum > 0) {
    dimOperands.insert(dimOperands.end(), symbolOperands.begin(), symbolOperands.end());
  }

  IntegerSet ifCondSet = IntegerSet::get(tiledLoops.size(), symbolNum, exprs, eqFlags);
  affine::canonicalizeSetAndOperands(&ifCondSet, &dimOperands);
  return b.create<affine::AffineIfOp>(tiledLoops[0].getLoc(), ifCondSet, dimOperands, false);
  ;
}

void AKGLoopTiling::updateInsertIfLoops(SmallVector<affine::AffineForOp, 6> &newTiledLoops,
                                        std::unordered_set<unsigned> inequalityForIndex) {
  auto it = newTiledLoops.begin();
  unsigned i = 0;
  while (it != newTiledLoops.end()) {
    if (inequalityForIndex.count(i)) {
      it++;
    } else {
      it = newTiledLoops.erase(it);
    }
    ++i;
  }
}

// Insert the corresponding if statement based on the upper bound of the for loop to ensure the correctness of the
// result.
// ```mlir (tile-sizes=32,5)
// #map = affine_map<(d0) -> (d0)>
// #map1 = affine_map<(d0) -> (d0 + 32)>
// #map2 = affine_map<(d0, d1) -> (d1 + 5, d0 + 32, 1024)>
// affine.for %arg3 = 0 to 1024 step 32 {
//   affine.for %arg4 = #map(%arg3) to #map1(%arg3) step 5 {
//     affine.for %arg5 = #map(%arg4) to min #map2(%arg3, %arg4) {
//       "test.foo"(%arg5) : (index) -> ()
//     }
//   }
// }
//   -->
// #map = affine_map<(d0) -> (d0)>
// #map1 = affine_map<(d0) -> (d0 + 32)>
// #map2 = affine_map<(d0) -> (d0 + 5)>
// #set = affine_set<(d0) : (-d0 + 1023 >= 0)>
// affine.for %arg3 = 0 to 1024 step 32 {
//   affine.for %arg4 = #map(%arg3) to #map1(%arg3) step 5 {
//     affine.for %arg5 = #map(%arg4) to #map2(%arg4) {
//       affine.if #set(%arg5) {
//         "test.foo"(%arg5) : (index) -> ()
//       }
//     }
//   }
// }
// ```
LogicalResult AKGLoopTiling::perfectlyNestedWithIf(SmallVector<affine::AffineForOp, 6> tiledLoops) {
  unsigned forNum = band.size();
  unsigned tileSizesNum = bandTileSizes.size();
  unsigned allTiledForNum = forNum + tileSizesNum;
  if (tiledLoops.size() != allTiledForNum) {
    return failure();
  }

  // Records the index whose tile_size cannot be divisible by the for loop.
  std::unordered_set<unsigned> inequalityForIndex;
  for (unsigned i = tileSizesNum; i < tileSizesNum + forNum; ++i) {
    auto ubMap = tiledLoops[i].getUpperBoundMap();
    if (ubMap.getResults().size() > 1) {
      inequalityForIndex.insert(i);
    }
  }

  if (inequalityForIndex.empty()) {
    return success();
  }

  // Gets all for loops except the first tiling.
  auto tileLoops = MutableArrayRef<affine::AffineForOp>(tiledLoops).drop_front(forNum);
  SmallVector<affine::AffineForOp, 4> fullTileLoops;

  if (failed(createFullBlock(tileLoops, fullTileLoops))) {
    if (!fullTileLoops.empty()) {
      fullTileLoops.front().erase();
    }
    tileLoops[0].emitError("Cannot construct the body block.");
    return failure();
  }
  if (fullTileLoops.empty()) {
    return success();
  }

  SmallVector<affine::AffineForOp, 6> newTiledLoops;
  getPerfectlyNestedLoops(newTiledLoops, tiledLoops[0]);
  Block *forBody = newTiledLoops[newTiledLoops.size() - 1].getBody();
  OpBuilder b(forBody, forBody->begin());
  SmallVector<Operation *, 6> bodyOp;
  // Records all ops in the for loop to facilitate the insertion of if statements.
  for (auto it = forBody->begin(); it != forBody->end(); ++it) {
    Operation *op = &*it;
    if (isa<affine::AffineYieldOp>(op)) {
      continue;
    }
    bodyOp.push_back(op);
  }
  updateInsertIfLoops(newTiledLoops, inequalityForIndex);
  affine::AffineIfOp ifOp = createperfectlyNestedCondition(newTiledLoops, b);
  if (!ifOp) {
    fullTileLoops.front().erase();
    newTiledLoops[0].emitError("Cannot construct an if statement.");
    return failure();
  }

  Block *thenBlock = ifOp.getThenBlock();
  for (auto op : bodyOp) {
    thenBlock->getOperations().splice(std::prev(thenBlock->end()), op->getBlock()->getOperations(),
                                      Block::iterator(op));
  }
  return success();
}

void AKGLoopTiling::tileEachBand() {
  getTileSizes();

  unsigned forNum = band.size();
  unsigned tileSizesNum = bandTileSizes.size();
  if (forNum == tileSizesNum) {
    SmallVector<affine::AffineForOp, 6> tiledNest;
    if (failed(tilePerfectlyNested(band, bandTileSizes, &tiledNest))) {
      // An empty band always succeeds.
      assert(!band.empty() && "guaranteed to succeed on empty bands");
      LLVM_DEBUG(band.front()->emitRemark("loop tiling failed!\n"));
    }

    if (separate) {
      auto intraTileLoops = MutableArrayRef<affine::AffineForOp>(tiledNest).drop_front(forNum);
      if (failed(separateFullTiles(intraTileLoops))) {
        assert(!intraTileLoops.empty() && "guaranteed to succeed on empty bands");
        LLVM_DEBUG(intraTileLoops.front()->emitRemark("separation post tiling failed!\n"));
      }
    } else if (separateNoIf) {
      if (failed(separateFullTilesNoIf(tiledNest))) {
        llvm::errs() << "separation-no-if tiling failed!\n";
      }
    } else if (inequalityConvertToIf) {
      if (failed(perfectlyNestedWithIf(tiledNest))) {
        llvm::errs() << "inequality-converted-to-if failed!\n";
      }
    }
  } else if (tileSizesNum > forNum && tileSizesNum % forNum == 0) {
    // Tiles the specified band of perfectly nested loops.
    affine::AffineForOp rootAffineForOp = band[0];
    unsigned width = tileSizesNum + forNum;
    SmallVector<affine::AffineForOp, 6> tiledLoops(width);
    constructTiledLoop(rootAffineForOp, width, tiledLoops);
    constructTiledIndex(tiledLoops);
    SmallVector<Value, 8> origLoopIVs;
    extractForInductionVars(band, &origLoopIVs);

    for (unsigned i = 0; i < forNum; i++) {
      origLoopIVs[i].replaceAllUsesWith(tiledLoops[i + tileSizesNum].getInductionVar());
    }
    rootAffineForOp.erase();

    if (separateNoIf) {
      if (failed(separateFullTilesNoIf(tiledLoops))) {
        llvm::errs() << "separation-no-if tiling failed!\n";
      }
    } else if (inequalityConvertToIf) {
      if (failed(perfectlyNestedWithIf(tiledLoops))) {
        llvm::errs() << "inequality-converted-to-if failed!\n";
      }
    }
  } else {
    llvm::errs() << "tileSizes must be a multiple of band!\n";
  }
}

void AKGLoopTiling::runCpuOperation() {
  separateNoIf = true;
  tileEachBand();

  func::FuncOp funcOp = getOperation();
  auto opType = CommonUtils::getOperatorType(funcOp);
  OpBuilder b(funcOp);
  if (opType == OperatorTemplate::Reduce) {
    SmallVector<Operation *, 8> reduceLoops = CommonUtils::collectReductionAxes(funcOp);
    for (auto reduceLoop : reduceLoops) {
      reduceLoop->setAttr("reduceLoop", b.getUnitAttr());
    }
  } else if (opType == OperatorTemplate::Broadcast) {
    llvm::SmallSet<affine::AffineForOp, 6> allBroadcastFor;
    funcOp.walk([&](affine::AffineForOp forOp) {
      llvm::SmallSet<affine::AffineForOp, 6> multiFor;
      for (auto op : forOp->getBlock()->getOps<affine::AffineForOp>()) {
        multiFor.insert(op);
      }

      if (multiFor.size() > 1) {
        allBroadcastFor.insert(multiFor.begin(), multiFor.end());
      }
    });

    llvm::SmallSet<Operation *, 8> broadcastLoops;
    if (allBroadcastFor.empty()) {
      CommonUtils::collectBroadcastAxes(funcOp, broadcastLoops);
    } else {
      for (auto i : allBroadcastFor) {
        CommonUtils::collectBroadcastAxes(i, broadcastLoops);
      }
    }
    for (auto broadcastLoop : broadcastLoops) {
      broadcastLoop->setAttr("broadcastLoop", b.getUnitAttr());
    }
  }
}

bool AKGLoopTiling::isDynamicShape() const { return ShapeAlignTool::getInstance().getFuncArgSizes() > 0; }

void AKGLoopTiling::runCudaOperation() {
  func::FuncOp funcOp = getOperation();
  auto opType = CommonUtils::getOperatorType(funcOp);
  if (!isDynamicShape() || opType == OperatorTemplate::Reduce) {
    inequalityConvertToIf = true;
  }
  tileEachBand();

  if (useAutoTiling && solver) {
    auto configSolver = std::make_shared<GlobalConfigSolver>(solver);

    configSolver->solve(funcOp);

    if (configSolver->modelGraph->globalConfigs.find(akg::utils::kEnableAtomicAdd) !=
        configSolver->modelGraph->globalConfigs.end()) {
      funcOp->setAttr(akg::utils::kEnableAtomicAdd,
                      configSolver->modelGraph->globalConfigs[akg::utils::kEnableAtomicAdd]);
    }
    funcOp->walk([&](Operation *curOp) {
      if (curOp->hasAttr(kReductionAxesStr)) {
        for (auto it : configSolver->modelGraph->globalConfigs) {
          curOp->setAttr(it.first, it.second);
        }
        // kEnableParallelReduce
        if (configSolver->modelGraph->globalConfigs.find(akg::utils::kEnableParallelReduce) !=
            configSolver->modelGraph->globalConfigs.end()) {
          curOp->setAttr(akg::utils::kEnableParallelReduce,
                         configSolver->modelGraph->globalConfigs[akg::utils::kEnableParallelReduce]);
        }
      }
    });
  }
}

void AKGLoopTiling::BandCheck(const std::vector<SmallVector<affine::AffineForOp, 6>> &bands) {
  func::FuncOp funcOp = getOperation();

  // A common checker for CPU/GPU that disables reshape op with empty band.
  if (bands.empty() && (CommonUtils::getOperatorType(funcOp) == OperatorTemplate::Reshape ||
                        CommonUtils::getOperatorType(funcOp) == OperatorTemplate::Transpose)) {
    getOperation()->walk([&](Operation *copyOp) {
      if (isa<CopyOpInterface>(copyOp)) {
        llvm::report_fatal_error(
          llvm::StringRef("[BandCheck]: No loop for reshape op, may have performance issue in static shape."));
      }
    });
  }

  // Rest are checkers for GPU only.
  if (target == kTargetCpu) {
    return;
  }

  // 2. multi-band checker
  if (bands.size() > 1) {
    llvm::report_fatal_error(llvm::StringRef("[BandCheck]: GPU cannot support multi-bands."));
  }

  // 3. single-band + dynamic axis checker cause we cannot map the dynamic axis
  getOperation()->walk([&](affine::AffineForOp forOp) {
    if (!forOp.hasConstantLowerBound() || !forOp.hasConstantUpperBound()) {
      llvm::report_fatal_error(
        llvm::StringRef("[BandCheck]: Dynamic bound, may have performance issue in static shape."));
    }
  });
}

void AKGLoopTiling::runOnOperation() {
  // Bands of loops to tile.
  func::FuncOp funcOp = getOperation();
  std::vector<SmallVector<affine::AffineForOp, 6>> bands;
  getTileableBands(funcOp, &bands);
  if (getOperation()->getAttr("process")) {
    target = getOperation()->getAttr("process").dyn_cast<StringAttr>().getValue().str();
  }

  if (!isDynamicShape()) {
    BandCheck(bands);
  }

  if (useAutoTiling) {
    auto initGraph = parseIr(funcOp, bands);
    initGraph->setHardware(target);
    initGraph->setFeature(feature);
    initGraph->setIsDynamicShape(isDynamicShape());
    initGraph->setTilingMode(tilingMode);
    auto modelGraph = buildModelGraph(initGraph);
    solver = getHeuristicTilingSolver(modelGraph);
    levelToTile = modelGraph->levelToTile;
  }

  // Tile each band.
  for (auto &curBand : bands) {
    band = MutableArrayRef<affine::AffineForOp>(curBand);
    if (target == kTargetCpu) {
      runCpuOperation();
    } else if (target == kTargetCuda) {
      runCudaOperation();
    } else {
      llvm::errs() << "Currently, only cpu and cuda backends are supported.\n";
    }
  }
}
