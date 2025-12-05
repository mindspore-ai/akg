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

#include <algorithm>
#include <iterator>
#include <memory>
#include <unordered_set>
#include "akg/Dialect/Affine/Analysis/AutoTiling.h"
#include "akg/Dialect/Affine/Analysis/BufferAnalysis.h"
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

using llvm::SmallVector;
using llvm::SmallVectorImpl;

#define DEBUG_TYPE "akg-affine-loop-tile"

namespace {

/// A pass to perform loop tiling on all suitable loop nests of a Function.
class AKGLoopTiling : public impl::AKGAffineLoopTilingBase<AKGLoopTiling> {
 public:
  AKGLoopTiling() = default;
  explicit AKGLoopTiling(uint64_t cacheSizeBytes, bool avoidMaxMinBounds = true)
      : avoidMaxMinBounds(avoidMaxMinBounds) {
    this->cacheSizeInKiB = cacheSizeBytes / 1024;
  }

  explicit AKGLoopTiling(const std::string &target, bool useAutoTiling = false) : target(target) {
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

  AKGLoopTiling(const std::string &target, bool useAutoTiling, const std::string &arch, const std::string &feature)
      : target(target), arch(arch), feature(feature) {
    this->useAutoTiling = useAutoTiling;
  }

  void runOnOperation() override;

 private:
  void runCpuOperation();
  void runCudaOperation();
  void runNpuOperation();
  void BandCheck(const std::vector<SmallVector<mlir::affine::AffineForOp, 6>> &bands);
  void getTileSizes();
  std::string getHardware();
  bool isDynamicShape() const;

  // core tiling function
  void tileEachBand();
  // initial tiling function
  void constructTiledLoop(mlir::affine::AffineForOp rootAffineForOp, unsigned width,
                          mlir::MutableArrayRef<mlir::affine::AffineForOp> tiledLoops);
  void constructTiledIndex(mlir::MutableArrayRef<mlir::affine::AffineForOp> newLoops);
  void setInsertInequality(int curTile, bool &insertInequality);
  void setNewUpperBound(mlir::MutableArrayRef<mlir::affine::AffineForOp> newLoops, int curTile,
                        bool insertInequality = true);

  // tile tail block
  void updateForOpUsers(mlir::affine::AffineForOp forOp, int64_t newSize = 0);
  mlir::LogicalResult createTailBlockForBody(mlir::affine::AffineForOp forOp);
  mlir::LogicalResult createTailBlock(mlir::affine::AffineForOp forOp);
  mlir::LogicalResult createFullBlock(mlir::MutableArrayRef<mlir::affine::AffineForOp> tiledLoops,
                                      SmallVectorImpl<mlir::affine::AffineForOp> &fullTileLoops);
  mlir::LogicalResult createTailBlockDynamic(mlir::affine::AffineForOp forOp, mlir::AffineSymbolExpr sExpr);
  mlir::LogicalResult createTailBlockStatic(mlir::affine::AffineForOp forOp, int64_t differenceUbAndLb);

  mlir::LogicalResult separateFullTilesNoIf(SmallVector<mlir::affine::AffineForOp, 6> tiledLoops);

  mlir::LogicalResult perfectlyNestedWithIf(SmallVector<mlir::affine::AffineForOp, 6> tiledLoops);
  void updateInsertIfLoops(SmallVector<mlir::affine::AffineForOp, 6> &newTiledLoops,
                           std::unordered_set<unsigned> inequalityForIndex);
  mlir::affine::AffineIfOp createperfectlyNestedCondition(SmallVector<mlir::affine::AffineForOp, 6> tiledLoops,
                                                          mlir::OpBuilder b);

  // If true, tile sizes are set to avoid max/min in bounds if possible.
  bool avoidMaxMinBounds{true};
  // hardware information
  std::string target{mlir::kTargetCpu};
  std::string tilingMode{"auto"};
  [[maybe_unused]] std::string arch{};
  std::string feature{mlir::kNEONInstructionSet};

  mlir::akg::autotiling::TilingSolverPtr solver{nullptr};
  size_t levelToTile{1};

  SmallVector<unsigned, 6> bandTileSizes;
  mlir::MutableArrayRef<mlir::affine::AffineForOp> band;
};
}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::createAKGLoopTilingPass() {
  return std::make_unique<AKGLoopTiling>();
}

/// Creates a pass to perform loop tiling on all suitable loop nests of a
/// Function.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::createAKGLoopTilingPass(uint64_t cacheSizeBytes) {
  return std::make_unique<AKGLoopTiling>(cacheSizeBytes);
}
/// Creates a pass to perform loop tiling using auto-tiling strategy
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::createAKGLoopTilingPass(const std::string &target,
                                                                                       bool useAutoTiling) {
  return std::make_unique<AKGLoopTiling>(target, useAutoTiling);
}

/// Creates a pass to perform loop tiling using auto-tiling strategy for dynamic shape
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::createAKGLoopTilingPass(const std::string &target,
                                                                                       bool useAutoTiling,
                                                                                       const std::string &tilingMode) {
  return std::make_unique<AKGLoopTiling>(target, useAutoTiling, tilingMode);
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::createAKGLoopTilingPass(const std::string &target,
                                                                                       const std::string &feature,
                                                                                       bool useAutoTiling) {
  return std::make_unique<AKGLoopTiling>(target, feature, useAutoTiling);
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::createAKGLoopTilingPass(const std::string &target,
                                                                                       bool useAutoTiling,
                                                                                       const std::string &arch,
                                                                                       const std::string &feature) {
  return std::make_unique<AKGLoopTiling>(target, useAutoTiling, arch, feature);
}

static void moveLoopBody(mlir::affine::AffineForOp src, mlir::affine::AffineForOp dest) {
  mlir::Block::iterator loc = dest.getBody()->begin();
  auto &ops = src.getBody()->getOperations();
  dest.getBody()->getOperations().splice(loc, ops, ops.begin(), std::prev(ops.end()));
}

// Finds the innermost loop with the maximum depth in the given loop.
// work even in the case of not perfectly nested loops.
static std::pair<int, mlir::affine::AffineForOp> findInnermostLoopWithDepth(mlir::affine::AffineForOp loop) {
  std::pair<int, mlir::affine::AffineForOp> best{1, loop};
  for (mlir::Operation &op : loop.getBody()->without_terminator()) {
    if (auto innerLoop = mlir::dyn_cast<mlir::affine::AffineForOp>(&op)) {
      auto candidate = findInnermostLoopWithDepth(innerLoop);
      int depth = candidate.first + 1;
      if (depth > best.first) {
        best.first = depth;
        best.second = candidate.second;
      }
    }
  }
  return best;
}

static mlir::AffineExpr getDifferenceUbAndLb(mlir::AffineMap ubMap, mlir::AffineMap lbMap) {
  auto maxDim = std::max(lbMap.getNumDims(), ubMap.getNumDims());
  auto maxSymbol = std::max(lbMap.getNumSymbols(), ubMap.getNumSymbols());
  // TODO(akg-dev): extend this to handle multiple result maps.
  return mlir::simplifyAffineExpr(ubMap.getResult(0) - lbMap.getResult(0), maxDim, maxSymbol);
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
void AKGLoopTiling::constructTiledLoop(mlir::affine::AffineForOp rootAffineForOp, unsigned width,
                                       mlir::MutableArrayRef<mlir::affine::AffineForOp> tiledLoops) {
  mlir::Location loc = rootAffineForOp.getLoc();

  mlir::Operation *topLoop = rootAffineForOp.getOperation();
  mlir::affine::AffineForOp innermostPointLoop;

  for (unsigned i = 0; i < width; ++i) {
    mlir::OpBuilder b(topLoop);
    // Loop bounds will be set later.
    mlir::affine::AffineForOp pointLoop = b.create<mlir::affine::AffineForOp>(loc, 0, 0);
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
void AKGLoopTiling::constructTiledIndex(mlir::MutableArrayRef<mlir::affine::AffineForOp> newLoops) {
  int bandSize = band.size();
  if (bandSize == 0) {
    return;
  }
  mlir::OpBuilder b(band[0].getOperation());
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
        mlir::OperandRange newLbOperands = band[j].getLowerBoundOperands();
        mlir::OperandRange newUbOperands = band[j].getUpperBoundOperands();
        newLoops[curTile].setLowerBound(newLbOperands, band[j].getLowerBoundMap());
        newLoops[curTile].setUpperBound(newUbOperands, band[j].getUpperBoundMap());
        newLoops[curTile].setStep(bandTileSizes[curTile]);
      } else if (i == tileNum) {
        // last tile
        mlir::AffineMap lbMap = b.getDimIdentityMap();
        newLoops[curTile].setLowerBound(newLoops[lastTile].getInductionVar(), lbMap);
        newLoops[curTile].setStep(1);

        setNewUpperBound(newLoops, curTile, true);
      } else {
        // middle tile
        mlir::AffineMap lbMap = b.getDimIdentityMap();
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
void AKGLoopTiling::setNewUpperBound(mlir::MutableArrayRef<mlir::affine::AffineForOp> newLoops, int curTile,
                                     bool insertInequality) {
  // Set the upper bound.
  int bandSize = band.size();
  int lastTile = curTile - bandSize;
  mlir::OpBuilder b(newLoops[0].getOperation());
  int64_t largestDiv = getLargestDivisorOfTripCount(band[curTile % bandSize]);

  setInsertInequality(curTile, insertInequality);
  if (insertInequality) {
    mlir::affine::AffineBound lastTileUb = newLoops[lastTile].getUpperBound();
    mlir::AffineMap lastTileUbMap = lastTileUb.getMap();
    SmallVector<mlir::Value, 4> ubOperands;
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
      if (llvm::isa<mlir::AffineConstantExpr>(ubMap)) {
        insertLoopUb = false;
      }
    }
    SmallVector<AffineExpr, 4> ubExprs;
    unsigned newExprSize = 1 + lastTileUbMap.getNumResults();
    if (insertLoopUb) {
      ++newExprSize;
    }
    ubExprs.reserve(newExprSize);

    mlir::AffineExpr dimExpr = b.getAffineDimExpr(lastTileUbMap.getNumDims());
    ubExprs.push_back(dimExpr + bandTileSizes[lastTile]);
    ubExprs.append(lastTileUbMap.getResults().begin(), lastTileUbMap.getResults().end());
    if (insertLoopUb) {
      ubExprs.push_back(b.getAffineConstantExpr(largestDiv));
    }
    mlir::AffineMap ubMap =
      mlir::AffineMap::get(lastTileUbMap.getNumDims() + 1, lastTileUbMap.getNumSymbols(), ubExprs, b.getContext());
    newLoops[curTile].setUpperBound(ubOperands, ubMap);
  } else {
    mlir::AffineExpr dim = b.getAffineDimExpr(0);
    mlir::AffineMap ubMap = mlir::AffineMap::get(1, 0, dim + newLoops[lastTile].getStepAsInt());
    newLoops[curTile].setUpperBound(newLoops[lastTile].getInductionVar(), ubMap);
  }
}

void AKGLoopTiling::getTileSizes() {
  // TODO(akg-dev): Separately tile axis
  if (useAutoTiling && solver) {
    // TODO(akg-dev): remove levelToTile
    SmallVector<mlir::affine::AffineForOp, 6> curband;
    curband.assign(band.begin(), band.end());
    for (size_t level = 0; level < levelToTile; ++level) {
      // TODO(akg-dev): Multiple band
      mlir::akg::autotiling::getTileSizeWithSolver(solver, curband, &bandTileSizes,
                                                   mlir::akg::autotiling::TilingTaskDesc(0, level));
    }
  } else {
    if (!tileSizes.empty() && tileSize == 1) {
      bandTileSizes.insert(bandTileSizes.end(), tileSizes.begin(), tileSizes.end());
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
mlir::LogicalResult AKGLoopTiling::createFullBlock(mlir::MutableArrayRef<mlir::affine::AffineForOp> intraTileLoops,
                                                   SmallVectorImpl<mlir::affine::AffineForOp> &fullTileLoops) {
  if (intraTileLoops.size() == 0) {
    return mlir::success();
  }
  mlir::OpBuilder b(intraTileLoops[0]);
  fullTileLoops.reserve(intraTileLoops.size());

  // For each loop in the original nest identify a lower/upper bound pair such
  // that their difference is a constant.
  mlir::affine::FlatAffineValueConstraints cst;
  for (auto loop : intraTileLoops) {
    SmallVector<mlir::Operation *, 1> loopOp{loop.getOperation()};
    (void)mlir::affine::getIndexSet(loopOp, &cst);
    // We will mark everything other than this loop IV as symbol for getting a
    // pair of <lb, ub> with a constant difference.
    cst.setDimSymbolSeparation(cst.getNumDimAndSymbolVars() - 1);
    unsigned lbPos, ubPos;
    if (!cst.getConstantBoundOnDimSize(0, nullptr, nullptr, nullptr, &lbPos, &ubPos) || lbPos == ubPos) {
      LLVM_DEBUG(llvm::dbgs() << "[tile separation] Can't get constant diff / equalities not yet handled\n");
      return mlir::failure();
    }

    // Set all variables as dimensions uniformly since some of those marked as
    // symbols above could be outer loop IVs (corresponding tile space IVs).
    cst.setDimSymbolSeparation(0);

    mlir::affine::AffineValueMap lbVmap, ubVmap;
    cst.getIneqAsAffineValueMap(0, lbPos, lbVmap, b.getContext());
    cst.getIneqAsAffineValueMap(0, ubPos, ubVmap, b.getContext());

    mlir::affine::AffineForOp fullTileLoop =
      mlir::affine::createCanonicalizedAffineForOp(b, loop.getLoc(), lbVmap.getOperands(), lbVmap.getAffineMap(),
                                                   ubVmap.getOperands(), ubVmap.getAffineMap(), loop.getStepAsInt());
    b = mlir::OpBuilder::atBlockTerminator(fullTileLoop.getBody());
    fullTileLoops.push_back(fullTileLoop);
  }

  // Add the body for the full tile loop nest.
  mlir::IRMapping operandMap;
  for (const auto &loopEn : llvm::enumerate(intraTileLoops)) {
    operandMap.map(loopEn.value().getInductionVar(), fullTileLoops[loopEn.index()].getInductionVar());
  }
  b = mlir::OpBuilder::atBlockTerminator(fullTileLoops.back().getBody());
  for (auto &op : intraTileLoops.back().getBody()->without_terminator()) {
    b.clone(op, operandMap);
  }
  // Add the body for the full tile loop nest.
  for (const auto &loopEn : llvm::enumerate(intraTileLoops)) {
    mlir::replaceAllUsesInRegionWith(loopEn.value().getInductionVar(), fullTileLoops[loopEn.index()].getInductionVar(),
                                     fullTileLoops[loopEn.index()].getRegion());
  }

  // insert the full block, replacing the original nested for
  mlir::Block *intraBlock = intraTileLoops[0].getOperation()->getBlock();
  mlir::affine::AffineForOp outermostFullTileLoop = fullTileLoops[0];
  intraBlock->getOperations().splice(std::prev(intraBlock->end()), outermostFullTileLoop->getBlock()->getOperations(),
                                     mlir::Block::iterator(outermostFullTileLoop));
  intraTileLoops[0].erase();
  return mlir::success();
}

mlir::LogicalResult AKGLoopTiling::createTailBlockForBody(mlir::affine::AffineForOp forOp) {
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (auto bodyOp = mlir::dyn_cast<mlir::affine::AffineForOp>(op)) {
      if (mlir::failed(createTailBlock(bodyOp))) {
        return mlir::failure();
      }
    }
  }
  return mlir::success();
}

// Updates the upper bound of all users of the trailing block for loop.
void AKGLoopTiling::updateForOpUsers(mlir::affine::AffineForOp forOp, int64_t newSize) {
  if (!newSize) {
    return;
  }
  for (mlir::OpOperand &use : forOp.getInductionVar().getUses()) {
    if (auto tiledOp = mlir::dyn_cast<mlir::affine::AffineForOp>(use.getOwner())) {
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
mlir::LogicalResult AKGLoopTiling::createTailBlock(mlir::affine::AffineForOp forOp) {
  auto origUbMap = forOp.getUpperBoundMap();
  auto origLbMap = forOp.getLowerBoundMap();
  mlir::AffineExpr differenceExpr = getDifferenceUbAndLb(origUbMap, origLbMap);
  if (auto cExpr = llvm::dyn_cast<mlir::AffineConstantExpr>(differenceExpr)) {
    return createTailBlockStatic(forOp, cExpr.getValue());
  } else if (auto sExpr = llvm::dyn_cast<mlir::AffineSymbolExpr>(differenceExpr)) {
    return createTailBlockDynamic(forOp, sExpr);
  }
  return mlir::success();
}

mlir::LogicalResult AKGLoopTiling::createTailBlockStatic(mlir::affine::AffineForOp forOp, int64_t differenceUbAndLb) {
  auto origUbMap = forOp.getUpperBoundMap();
  auto origLbMap = forOp.getLowerBoundMap();
  int64_t origStep = forOp.getStepAsInt();
  int64_t tailSize = differenceUbAndLb % origStep;
  if (tailSize == 0) {
    // Recursively processes the forOp body.
    if (mlir::failed(createTailBlockForBody(forOp))) {
      return mlir::failure();
    }
    return mlir::success();
  }

  auto ubMap = origUbMap;
  mlir::AffineExpr newExpr = ubMap.getResult(0) - tailSize;
  ubMap = ubMap.replace(ubMap.getResult(0), newExpr, ubMap.getNumDims(), ubMap.getNumSymbols());
  int64_t newDifferenceUbAndLb = 0;
  mlir::AffineExpr differenceExpr = getDifferenceUbAndLb(origUbMap, ubMap);
  if (auto cExpr = llvm::dyn_cast<mlir::AffineConstantExpr>(differenceExpr)) {
    newDifferenceUbAndLb = cExpr.getValue();
  } else {
    forOp.emitError("Error: Could not get the difference between upper and lower bounds of the loop.");
    return mlir::failure();
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
  if (mlir::failed(createTailBlockForBody(forOp))) {
    return mlir::failure();
  }

  // differenceUbAndLb < origStep: If the difference between the upper and lower bounds is less than step, the
  // body block does not need to be inserted.
  // !newDifferenceUbAndLb: If the difference between the upper and lower bounds of the new trailing block is 0, the
  // trailing block does not need to be inserted.
  // isEqualToBlock: If the upper and lower bounds and steps of the body block and tail block are the same, the tail
  // block does not need to be inserted.
  bool isEqualToBlock = (ubMap == origLbMap) && (origUbMap == ubMap) && (tailSize == origStep);
  if (differenceUbAndLb < origStep || !newDifferenceUbAndLb || isEqualToBlock) {
    return mlir::success();
  }

  // Insert the tail tiles
  mlir::OpBuilder b(forOp);
  b.setInsertionPointAfter(forOp);
  auto tailOp = b.clone(*forOp.getOperation());
  auto tailForOp = mlir::dyn_cast<mlir::affine::AffineForOp>(tailOp);
  tailForOp.setLowerBoundMap(ubMap);
  tailForOp.setUpperBoundMap(origUbMap);
  tailForOp.setStep(tailSize);
  mlir::replaceAllUsesInRegionWith(forOp.getInductionVar(), tailForOp.getInductionVar(), tailForOp.getRegion());
  updateForOpUsers(tailForOp, tailSize);

  // Recursively processes the tailForOp body.
  if (mlir::failed(createTailBlockForBody(tailForOp))) {
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult AKGLoopTiling::createTailBlockDynamic(mlir::affine::AffineForOp forOp,
                                                          mlir::AffineSymbolExpr sExpr) {
  if (band.size() != bandTileSizes.size()) {
    forOp.emitError("Dynamic shape supports only one tiling.");
    return mlir::failure();
  }
  auto origUbMap = forOp.getUpperBoundMap();
  auto origUbOp = forOp.getUpperBoundOperands();
  int64_t origStep = forOp.getStepAsInt();
  // When the dynamic shape step is 1, the tail block does not need to be processed.
  // affine.for %arg4 = 0 to %dim
  if (origStep == 1) {
    // Recursively processes the forOp body.
    if (mlir::failed(createTailBlockForBody(forOp))) {
      return mlir::failure();
    }
    return mlir::success();
  }

  auto ubMap = origUbMap;
  mlir::AffineExpr newExpr = ubMap.getResult(0) - sExpr % origStep;
  ubMap = ubMap.replace(ubMap.getResult(0), newExpr, ubMap.getNumDims(), ubMap.getNumSymbols());
  forOp.setUpperBoundMap(ubMap);

  // Recursively processes the forOp body.
  if (mlir::failed(createTailBlockForBody(forOp))) {
    return mlir::failure();
  }

  // Insert the tail tiles
  mlir::OpBuilder b(forOp);
  b.setInsertionPointAfter(forOp);
  mlir::affine::AffineForOp tailForOp =
    b.create<mlir::affine::AffineForOp>(forOp.getLoc(), forOp.getUpperBoundOperands(), ubMap, origUbOp, origUbMap, 1);

  // Add the body for the full tile loop nest.
  b = mlir::OpBuilder::atBlockTerminator(tailForOp.getBody());
  for (auto &op : forOp.getBody()->without_terminator()) {
    b.clone(op);
  }
  // Add the body for the full tile loop nest.
  mlir::replaceAllUsesInRegionWith(forOp.getInductionVar(), tailForOp.getInductionVar(), tailForOp.getRegion());
  updateForOpUsers(tailForOp, 1);

  // Recursively processes the tailForOp body.
  if (mlir::failed(createTailBlockForBody(tailForOp))) {
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult AKGLoopTiling::separateFullTilesNoIf(SmallVector<mlir::affine::AffineForOp, 6> tiledLoops) {
  // full block
  auto intraTileLoops = mlir::MutableArrayRef<mlir::affine::AffineForOp>(tiledLoops).drop_front(band.size());
  SmallVector<mlir::affine::AffineForOp, 4> fullTileLoops;
  if (mlir::failed(createFullBlock(intraTileLoops, fullTileLoops))) {
    if (!fullTileLoops.empty()) {
      fullTileLoops.front().erase();
    }
    return mlir::failure();
  }
  if (fullTileLoops.size() == 0) {
    return mlir::success();
  }

  // tail block
  if (mlir::failed(createTailBlock(tiledLoops[0]))) {
    return mlir::failure();
  }

  return mlir::success();
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
mlir::affine::AffineIfOp AKGLoopTiling::createperfectlyNestedCondition(
  SmallVector<mlir::affine::AffineForOp, 6> tiledLoops, mlir::OpBuilder b) {
  if (tiledLoops.empty()) {
    return nullptr;
  }

  SmallVector<mlir::AffineExpr, 4> exprs;
  SmallVector<bool, 4> eqFlags;
  SmallVector<mlir::Value, 4> dimOperands;
  SmallVector<mlir::Value, 4> symbolOperands;
  int64_t symbolNum = 0;
  int64_t dimNum = 0;
  for (auto loop : tiledLoops) {
    assert(loop.getStepAsInt() == 1 && "point loop step expected to be one");

    // Find the outermost for loop, that is, the for loop before tiling.
    mlir::Operation *outerOp = nullptr;
    for (auto value : loop.getOperands()) {
      SmallVector<mlir::Operation *, 8> opAxes;
      mlir::CommonUtils::collectRelatedAxes(value, opAxes);
      if (opAxes.empty()) {
        continue;
      }
      for (auto axes : opAxes) {
        outerOp = mlir::CommonUtils::getInnerOrOuterOp(outerOp, axes, false);
      }
    }

    if (!outerOp || !mlir::isa<mlir::affine::AffineForOp>(outerOp)) {
      continue;
    }

    auto outerForOp = mlir::dyn_cast<mlir::affine::AffineForOp>(outerOp);
    auto context = loop.getContext();
    mlir::AffineExpr upperExpr = outerForOp.getUpperBoundMap().getResult(0);
    // Adapts to dynamic shapes.
    if (llvm::isa<mlir::AffineSymbolExpr>(upperExpr)) {
      for (auto operand : outerForOp->getOperands()) {
        if (mlir::Operation *parentOp = operand.getDefiningOp()) {
          symbolOperands.push_back(parentOp->getResult(0));
          upperExpr = mlir::getAffineSymbolExpr(symbolNum++, context);
        }
      }
    }
    // Make sure that the dim variable is incremented each time.
    mlir::AffineExpr newExpr = upperExpr - 1 - mlir::getAffineDimExpr(dimNum++, context);
    exprs.push_back(newExpr);
    eqFlags.push_back(false);
    dimOperands.push_back(loop.getInductionVar());
  }

  // Adapts to dynamic shapes: add symbol operands.
  if (symbolNum > 0) {
    dimOperands.insert(dimOperands.end(), symbolOperands.begin(), symbolOperands.end());
  }

  mlir::IntegerSet ifCondSet = mlir::IntegerSet::get(tiledLoops.size(), symbolNum, exprs, eqFlags);
  mlir::affine::canonicalizeSetAndOperands(&ifCondSet, &dimOperands);
  return b.create<mlir::affine::AffineIfOp>(tiledLoops[0].getLoc(), ifCondSet, dimOperands, false);
}

void AKGLoopTiling::updateInsertIfLoops(SmallVector<mlir::affine::AffineForOp, 6> &newTiledLoops,
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
mlir::LogicalResult AKGLoopTiling::perfectlyNestedWithIf(SmallVector<mlir::affine::AffineForOp, 6> tiledLoops) {
  unsigned forNum = band.size();
  unsigned tileSizesNum = bandTileSizes.size();
  unsigned allTiledForNum = forNum + tileSizesNum;
  if (tiledLoops.size() != allTiledForNum) {
    return mlir::failure();
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
    return mlir::success();
  }

  // Gets all for loops except the first tiling.
  auto tileLoops = mlir::MutableArrayRef<mlir::affine::AffineForOp>(tiledLoops).drop_front(forNum);
  SmallVector<mlir::affine::AffineForOp, 4> fullTileLoops;

  if (mlir::failed(createFullBlock(tileLoops, fullTileLoops))) {
    if (!fullTileLoops.empty()) {
      fullTileLoops.front().erase();
    }
    tileLoops[0].emitError("Cannot construct the body block.");
    return mlir::failure();
  }
  if (fullTileLoops.empty()) {
    return mlir::success();
  }

  SmallVector<mlir::affine::AffineForOp, 6> newTiledLoops;
  mlir::affine::getPerfectlyNestedLoops(newTiledLoops, tiledLoops[0]);
  mlir::Block *forBody = newTiledLoops[newTiledLoops.size() - 1].getBody();
  mlir::OpBuilder b(forBody, forBody->begin());
  SmallVector<mlir::Operation *, 6> bodyOp;
  // Records all ops in the for loop to facilitate the insertion of if statements.
  for (auto it = forBody->begin(); it != forBody->end(); ++it) {
    mlir::Operation *op = &*it;
    if (mlir::isa<mlir::affine::AffineYieldOp>(op)) {
      continue;
    }
    bodyOp.push_back(op);
  }
  updateInsertIfLoops(newTiledLoops, inequalityForIndex);
  mlir::affine::AffineIfOp ifOp = createperfectlyNestedCondition(newTiledLoops, b);
  if (!ifOp) {
    fullTileLoops.front().erase();
    newTiledLoops[0].emitError("Cannot construct an if statement.");
    return mlir::failure();
  }

  mlir::Block *thenBlock = ifOp.getThenBlock();
  for (auto op : bodyOp) {
    thenBlock->getOperations().splice(std::prev(thenBlock->end()), op->getBlock()->getOperations(),
                                      mlir::Block::iterator(op));
  }
  return mlir::success();
}

void AKGLoopTiling::tileEachBand() {
  getTileSizes();

  unsigned forNum = band.size();
  unsigned tileSizesNum = bandTileSizes.size();
  if (forNum == tileSizesNum) {
    SmallVector<mlir::affine::AffineForOp, 6> tiledNest;
    if (mlir::failed(mlir::affine::tilePerfectlyNested(band, bandTileSizes, &tiledNest))) {
      // An empty band always succeeds.
      assert(!band.empty() && "guaranteed to succeed on empty bands");
      LLVM_DEBUG(band.front()->emitRemark("loop tiling failed!\n"));
    }

    if (separate) {
      auto intraTileLoops = mlir::MutableArrayRef<mlir::affine::AffineForOp>(tiledNest).drop_front(forNum);
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
    mlir::affine::AffineForOp rootAffineForOp = band[0];
    unsigned width = tileSizesNum + forNum;
    SmallVector<mlir::affine::AffineForOp, 6> tiledLoops(width);
    constructTiledLoop(rootAffineForOp, width, tiledLoops);
    constructTiledIndex(tiledLoops);
    SmallVector<mlir::Value, 8> origLoopIVs;
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

  mlir::func::FuncOp funcOp = getOperation();
  auto opType = mlir::CommonUtils::getOperatorType(funcOp);
  mlir::OpBuilder b(funcOp);
  if (opType == mlir::OperatorTemplate::Reduce) {
    SmallVector<mlir::Operation *, 8> reduceLoops = mlir::CommonUtils::collectReductionAxes(funcOp);
    for (auto reduceLoop : reduceLoops) {
      reduceLoop->setAttr(kReductionLoopAttr, b.getUnitAttr());
    }
  } else if (opType == mlir::OperatorTemplate::Broadcast) {
    llvm::SmallSet<mlir::affine::AffineForOp, 6> allBroadcastFor;
    funcOp.walk([&](mlir::affine::AffineForOp forOp) {
      llvm::SmallSet<mlir::affine::AffineForOp, 6> multiFor;
      for (auto op : forOp->getBlock()->getOps<mlir::affine::AffineForOp>()) {
        multiFor.insert(op);
      }

      if (multiFor.size() > 1) {
        allBroadcastFor.insert(multiFor.begin(), multiFor.end());
      }
    });

    llvm::SmallSet<mlir::Operation *, 8> broadcastLoops;
    if (allBroadcastFor.empty()) {
      mlir::CommonUtils::collectBroadcastAxes(funcOp, broadcastLoops);
    } else {
      for (auto i : allBroadcastFor) {
        mlir::CommonUtils::collectBroadcastAxes(i, broadcastLoops);
      }
    }
    for (auto broadcastLoop : broadcastLoops) {
      broadcastLoop->setAttr("broadcastLoop", b.getUnitAttr());
    }
  }
}

bool AKGLoopTiling::isDynamicShape() const { return akgglobal::ShapeAlignTool::getInstance().getFuncArgSizes() > 0; }

void AKGLoopTiling::runNpuOperation() {
  if (band.empty()) {
    return;
  }
  mlir::func::FuncOp funcOp = getOperation();
  auto opType = mlir::CommonUtils::getOperatorType(funcOp);
  // Use separateNoIf to handle tail blocks with separate for loops instead of if statements
  // This improves NPU performance by avoiding conditional branches
  if (!isDynamicShape() || opType == mlir::OperatorTemplate::Reduce) {
    separateNoIf = true;
  }
  tileEachBand();
  if (useAutoTiling && solver) {
    // TODO(ascend-tiling): annotate loops for cooperative scheduling
    return;
  }
  return;
}

void AKGLoopTiling::runCudaOperation() {
  mlir::func::FuncOp funcOp = getOperation();
  auto opType = mlir::CommonUtils::getOperatorType(funcOp);
  if (!isDynamicShape() || opType == mlir::OperatorTemplate::Reduce) {
    inequalityConvertToIf = true;
  }
  tileEachBand();

  if (useAutoTiling && solver) {
    auto configSolver = std::make_shared<mlir::akg::autotiling::GlobalConfigSolver>(solver);

    configSolver->solve(funcOp);

    if (configSolver->modelGraph->globalConfigs.find(akg::utils::kEnableAtomicAdd) !=
        configSolver->modelGraph->globalConfigs.end()) {
      funcOp->setAttr(akg::utils::kEnableAtomicAdd,
                      configSolver->modelGraph->globalConfigs[akg::utils::kEnableAtomicAdd]);
    }
    funcOp->walk([&](mlir::Operation *curOp) {
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

void AKGLoopTiling::BandCheck(const std::vector<SmallVector<mlir::affine::AffineForOp, 6>> &bands) {
  mlir::func::FuncOp funcOp = getOperation();

  // A common checker for CPU/GPU that disables reshape op with empty band.
  if (bands.empty() && (mlir::CommonUtils::getOperatorType(funcOp) == mlir::OperatorTemplate::Reshape ||
                        mlir::CommonUtils::getOperatorType(funcOp) == mlir::OperatorTemplate::Transpose)) {
    getOperation()->walk([&](mlir::Operation *copyOp) {
      if (mlir::isa<mlir::CopyOpInterface>(copyOp)) {
        llvm::report_fatal_error(
          llvm::StringRef("[BandCheck]: No loop for reshape op, may have performance issue in static shape."));
      }
    });
  }

  // Rest are checkers for GPU only.
  if (target == mlir::kTargetCpu || target == mlir::kTargetNpu) {
    return;
  }

  // 2. multi-band checker
  if (bands.size() > 1) {
    llvm::report_fatal_error(llvm::StringRef("[BandCheck]: GPU cannot support multi-bands."));
  }

  // 3. single-band + dynamic axis checker cause we cannot map the dynamic axis
  getOperation()->walk([&](mlir::affine::AffineForOp forOp) {
    if (!forOp.hasConstantLowerBound() || !forOp.hasConstantUpperBound()) {
      llvm::report_fatal_error(
        llvm::StringRef("[BandCheck]: Dynamic bound, may have performance issue in static shape."));
    }
  });
}

void AKGLoopTiling::runOnOperation() {
  // Bands of loops to tile.
  llvm::dbgs() << "arch: " << arch << "\n";
  mlir::func::FuncOp funcOp = getOperation();
  std::vector<SmallVector<mlir::affine::AffineForOp, 6>> bands;
  mlir::affine::getTileableBands(funcOp, &bands);
  if (getOperation()->getAttr("process")) {
    target = mlir::dyn_cast<mlir::StringAttr>(getOperation()->getAttr("process")).getValue().str();
  }

  if (!isDynamicShape()) {
    BandCheck(bands);
  }

  if (useAutoTiling) {
    mlir::akg::BufferAnalysisOptions options;
    options.enableDmaOpt = false;
    auto maxBuffer = countMaxBuffer(funcOp, options);
    llvm::outs() << "maxBuffer: " << maxBuffer << "\n";

    auto initGraph = mlir::akg::autotiling::parseIr(funcOp, bands);
    initGraph->setHardware(target);
    initGraph->setFeature(feature);
    initGraph->setIsDynamicShape(isDynamicShape());
    initGraph->setTilingMode(tilingMode);

    if (target == mlir::kTargetNpu && !this->multiTileSizes.empty()) {
      mlir::OpBuilder builder(funcOp);
      SmallVector<Attribute, 4> tileSizeAttrs;
      tileSizeAttrs.reserve(this->multiTileSizes.size());
      std::transform(this->multiTileSizes.begin(), this->multiTileSizes.end(), std::back_inserter(tileSizeAttrs),
                     [&builder](unsigned size) { return builder.getI32IntegerAttr(size); });
      funcOp->setAttr("npu.multiTileSizes", builder.getArrayAttr(tileSizeAttrs));
    }

    auto modelGraph = mlir::akg::autotiling::buildModelGraph(initGraph);
    solver = mlir::akg::autotiling::getHeuristicTilingSolver(modelGraph);
    levelToTile = modelGraph->levelToTile;
  }

  // Tile each band.
  for (auto &curBand : bands) {
    band = mlir::MutableArrayRef<mlir::affine::AffineForOp>(curBand);
    if (target == mlir::kTargetCpu) {
      runCpuOperation();
    } else if (target == mlir::kTargetCuda) {
      runCudaOperation();
    } else if (target == mlir::kTargetNpu) {
      runNpuOperation();
    } else {
      llvm::errs() << "Currently, only cpu, cuda and ascend backends are supported.\n" << "Current Hardware:" << target;
    }
  }
}
