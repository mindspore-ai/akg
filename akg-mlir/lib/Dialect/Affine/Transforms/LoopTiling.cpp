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

#include "akg/Dialect/Affine/Transforms/LoopTiling.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <unordered_set>
#include <utility>
#include "akg/Analysis/AutoTiling.h"
#include "akg/Utils/GlobalVars.hpp"
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
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {
#define GEN_PASS_DECL_AKGAFFINELOOPTILING
#define GEN_PASS_DEF_AKGAFFINELOOPTILING
#include "akg/Dialect/Affine/Passes.h.inc"

using llvm::SmallVector;
using llvm::SmallVectorImpl;

#define DEBUG_TYPE "akg-affine-loop-tile"

namespace {

static constexpr int64_t kBytesPerKiB = 1024;
static constexpr int32_t kDefaultNumKernels = 1;
static constexpr int32_t kMinNumKernels = 1;
static constexpr int32_t kTailBlockCount = 1;
static constexpr int32_t kNoTailBlockCount = 0;
static constexpr int64_t kPointLoopStep = 1;
static constexpr int64_t kPlaceholderLoopBound = 0;
static constexpr int64_t kKernelLoopLowerBound = 0;
static constexpr int64_t kKernelLoopStep = 1;
static constexpr unsigned kAffineSingleDimCount = 1;
static constexpr unsigned kAffineZeroSymbolCount = 0;
static constexpr unsigned kAffineFirstDimIndex = 0;
static constexpr int64_t kExclusiveBoundAdjustment = 1;
static constexpr int64_t kCountToIndexAdjustment = 1;
static constexpr int kInitialLoopDepth = 1;
static constexpr unsigned kAdditionalDimCount = 1;
static constexpr int64_t kCeilingDivAdjustment = 1;
static constexpr int32_t kMaxNumKernels = 40;
static constexpr int64_t kNoSizeUpdate = 0;

/// A pass to perform loop tiling on all suitable loop nests of a Function.
class LoopTiling : public impl::AKGAffineLoopTilingBase<LoopTiling> {
 public:
  LoopTiling() = default;
  explicit LoopTiling(uint64_t cacheSizeBytes, bool avoidMaxMinBounds = true) : avoidMaxMinBounds(avoidMaxMinBounds) {
    this->cacheSizeInKiB = cacheSizeBytes / kBytesPerKiB;
  }

  explicit LoopTiling(std::string target, bool useAutoTiling = false) : target(std::move(target)) {
    this->useAutoTiling = useAutoTiling;
  }

  LoopTiling(std::string target, bool useAutoTiling, const std::string &tilingMode) : target(std::move(target)) {
    this->useAutoTiling = useAutoTiling;
    this->tilingMode = tilingMode;
  }

  LoopTiling(std::string target, std::string feature, bool useAutoTiling = false)
      : target(std::move(target)), feature(std::move(feature)) {
    this->useAutoTiling = useAutoTiling;
  }

  LoopTiling(std::string target, bool useAutoTiling, std::string arch, std::string feature)
      : target(std::move(target)), arch(std::move(arch)), feature(std::move(feature)) {
    this->useAutoTiling = useAutoTiling;
  }

  LoopTiling(std::string target, bool useAutoTiling, std::string arch, std::string feature,
             const SmallVector<unsigned, 6> &inputTileSizes)
      : target(std::move(target)), arch(std::move(arch)), feature(std::move(feature)), inputTileSizes(inputTileSizes) {
    this->useAutoTiling = useAutoTiling;
  }

  void runOnOperation() override;

 private:
  void runCpuOperation();
  void runCudaOperation();
  void runNpuOperation();
  void addOuterKernelMappingLoop(affine::AffineForOp bandRootLoop);

  // Helper functions for runNpuOperation
  affine::AffineForOp findTiledBandRoot(Block *parentBlock, func::FuncOp funcOp);
  void collectAndMarkInnermostLoops(func::FuncOp funcOp);

  // Helper structures and functions for kernel mapping
  struct DimensionInfo {
    affine::AffineForOp fullBlock;  // Full tiled block
    affine::AffineForOp tailBlock;  // Tail block (may be null)
    int32_t numBlocks;              // Number of blocks (full blocks only, excluding tail)
    int32_t step;                   // Step size
    int32_t lb;                     // Lower bound of full block
    int32_t ub;                     // Upper bound of full block (may be adjusted)
    int32_t extent;                 // Extent of full block
    int32_t tailLb;                 // Lower bound of tail block
    int32_t tailUb;                 // Upper bound of tail block
    int32_t tailStep;               // Step size of tail block
    bool isReduceAxis;              // Whether this is a reduction axis
  };

  struct MappingContext {
    affine::AffineForOp fullBlock;
    affine::AffineForOp tailBlock;
    int32_t numBlocks;
    int32_t step;
    int32_t lb;
    int32_t ub;
    int32_t numKernels;
  };
  MappingContext collectAndSelectMappingDimension(affine::AffineForOp bandRootLoop);
  // Helper functions for collectAndSelectMappingDimension
  SmallVector<affine::AffineForOp> collectOutermostTiledLoops(affine::AffineForOp bandRootLoop);
  SmallVector<DimensionInfo> collectDimensionInfos(const SmallVector<affine::AffineForOp> &outermostTiledLoops,
                                                   int32_t &maxNumBlocks);
  DimensionInfo *selectMappingDimension(SmallVector<DimensionInfo> &dimInfos, int32_t maxNumBlocks);

  affine::AffineForOp createKernelLoopAndMapFullBlock(OpBuilder &builder, func::FuncOp funcOp, MappingContext &ctx);
  void mapTailBlockToKernel(OpBuilder &builder, affine::AffineForOp kernelLoop, affine::AffineForOp fullLoop,
                            affine::AffineForOp tailLoop, int32_t numKernels, Value kernelId);
  void BandCheck(const std::vector<SmallVector<affine::AffineForOp, 6>> &bands);
  void getTileSizes();
  std::string getHardware();
  [[nodiscard]] bool isDynamicShape() const;

  // core tiling function
  void tileEachBand();
  // initial tiling function
  void constructTiledLoop(affine::AffineForOp rootAffineForOp, unsigned width,
                          MutableArrayRef<affine::AffineForOp> tiledLoops);
  void constructTiledIndex(MutableArrayRef<affine::AffineForOp> newLoops);
  void setInsertInequality(int curTile, bool &insertInequality);
  void setNewUpperBound(MutableArrayRef<affine::AffineForOp> newLoops, int curTile, bool insertInequality = true);

  // tile tail block
  void updateForOpUsers(affine::AffineForOp forOp, int64_t newSize = kNoSizeUpdate);
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
  std::string tilingMode{"auto"};
  std::string arch{akg::kSoc910B2};
  std::string feature{kNEONInstructionSet};

  autotiling::TilingSolverPtr solver{nullptr};
  size_t levelToTile{1};

  SmallVector<unsigned, 6> bandTileSizes;
  SmallVector<unsigned, 6> inputTileSizes;
  size_t currentBandIdx = 0;
  MutableArrayRef<affine::AffineForOp> band;
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createAKGLoopTilingPass() { return std::make_unique<LoopTiling>(); }

/// Creates a pass to perform loop tiling on all suitable loop nests of a
/// Function.
std::unique_ptr<OperationPass<func::FuncOp>> createAKGLoopTilingPass(uint64_t cacheSizeBytes) {
  return std::make_unique<LoopTiling>(cacheSizeBytes);
}
/// Creates a pass to perform loop tiling using auto-tiling strategy
std::unique_ptr<OperationPass<func::FuncOp>> createAKGLoopTilingPass(const std::string &target, bool useAutoTiling) {
  return std::make_unique<LoopTiling>(target, useAutoTiling);
}

/// Creates a pass to perform loop tiling using auto-tiling strategy for dynamic shape
std::unique_ptr<OperationPass<func::FuncOp>> createAKGLoopTilingPass(const std::string &target, bool useAutoTiling,
                                                                     const std::string &tilingMode) {
  return std::make_unique<LoopTiling>(target, useAutoTiling, tilingMode);
}

std::unique_ptr<OperationPass<func::FuncOp>> createAKGLoopTilingPass(const std::string &target,
                                                                     const std::string &feature, bool useAutoTiling) {
  return std::make_unique<LoopTiling>(target, feature, useAutoTiling);
}

std::unique_ptr<OperationPass<func::FuncOp>> createAKGLoopTilingPass(const std::string &target, bool useAutoTiling,
                                                                     const std::string &arch,
                                                                     const std::string &feature) {
  return std::make_unique<LoopTiling>(target, useAutoTiling, arch, feature);
}

std::unique_ptr<OperationPass<func::FuncOp>> createAKGLoopTilingPass(
  const std::string &target, bool useAutoTiling, const std::string &arch, const std::string &feature,
  const llvm::SmallVector<unsigned, 6> &inputTileSizes) {
  return std::make_unique<LoopTiling>(target, useAutoTiling, arch, feature, inputTileSizes);
}

static void moveLoopBody(affine::AffineForOp src, affine::AffineForOp dest) {
  Block::iterator loc = dest.getBody()->begin();
  auto &ops = src.getBody()->getOperations();
  dest.getBody()->getOperations().splice(loc, ops, ops.begin(), std::prev(ops.end()));
}

// Finds the innermost loop with the maximum depth in the given loop.
// work even in the case of not perfectly nested loops.
static std::pair<int, affine::AffineForOp> findInnermostLoopWithDepth(affine::AffineForOp loop) {
  std::pair<int, affine::AffineForOp> best{kInitialLoopDepth, loop};
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (auto innerLoop = dyn_cast<affine::AffineForOp>(&op)) {
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

static AffineExpr getDifferenceUbAndLb(AffineMap ubMap, AffineMap lbMap) {
  auto maxDim = std::max(lbMap.getNumDims(), ubMap.getNumDims());
  auto maxSymbol = std::max(lbMap.getNumSymbols(), ubMap.getNumSymbols());
  // Extend this to handle multiple result maps.
  return simplifyAffineExpr(ubMap.getResult(kAffineFirstDimIndex) - lbMap.getResult(kAffineFirstDimIndex), maxDim,
                            maxSymbol);
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
void LoopTiling::constructTiledLoop(affine::AffineForOp rootAffineForOp, unsigned width,
                                    MutableArrayRef<affine::AffineForOp> tiledLoops) {
  Location loc = rootAffineForOp.getLoc();

  Operation *topLoop = rootAffineForOp.getOperation();
  affine::AffineForOp innermostPointLoop;

  for (unsigned i = 0; i < width; ++i) {
    OpBuilder b(topLoop);
    // Loop bounds will be set later.
    auto pointLoop = b.create<affine::AffineForOp>(loc, kPlaceholderLoopBound, kPlaceholderLoopBound);
    pointLoop.getBody()->getOperations().splice(pointLoop.getBody()->begin(), topLoop->getBlock()->getOperations(),
                                                topLoop);
    tiledLoops[width - kCountToIndexAdjustment - i] = pointLoop;
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
void LoopTiling::constructTiledIndex(MutableArrayRef<affine::AffineForOp> newLoops) {
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
        newLoops[curTile].setStep(kPointLoopStep);

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
void LoopTiling::setInsertInequality(int curTile, bool &insertInequality) {
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
void LoopTiling::setNewUpperBound(MutableArrayRef<affine::AffineForOp> newLoops, int curTile, bool insertInequality) {
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
    ubOperands.reserve(lastTileUb.getNumOperands() + kAdditionalDimCount);
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
    unsigned newExprSize = kAdditionalDimCount + lastTileUbMap.getNumResults();
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
    AffineMap ubMap = AffineMap::get(lastTileUbMap.getNumDims() + kAdditionalDimCount, lastTileUbMap.getNumSymbols(),
                                     ubExprs, b.getContext());
    newLoops[curTile].setUpperBound(ubOperands, ubMap);
  } else {
    AffineExpr dim = b.getAffineDimExpr(kAffineFirstDimIndex);
    AffineMap ubMap =
      AffineMap::get(kAffineSingleDimCount, kAffineZeroSymbolCount, dim + newLoops[lastTile].getStepAsInt());
    newLoops[curTile].setUpperBound(newLoops[lastTile].getInductionVar(), ubMap);
  }
}

void LoopTiling::getTileSizes() {
  // Separately tile axis
  if (useAutoTiling && solver) {
    // Remove levelToTiles
    SmallVector<affine::AffineForOp, 6> curBand;
    curBand.assign(band.begin(), band.end());

    for (size_t level = 0; level < levelToTile; ++level) {
      // Multiple band
      autotiling::getTileSizeWithSolver(solver, curBand, &bandTileSizes,
                                        autotiling::TilingTaskDesc(currentBandIdx, level));
    }
  } else {
    if (!tileSizes.empty() && tileSize == kPointLoopStep) {
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
LogicalResult LoopTiling::createFullBlock(MutableArrayRef<affine::AffineForOp> intraTileLoops,
                                          SmallVectorImpl<affine::AffineForOp> &fullTileLoops) {
  if (intraTileLoops.empty()) {
    return success();
  }
  OpBuilder b(intraTileLoops[0]);
  fullTileLoops.reserve(intraTileLoops.size());

  // For each loop in the original nest identify a lower/upper bound pair such
  // that their difference is a constant.
  affine::FlatAffineValueConstraints cst;
  for (auto loop : intraTileLoops) {
    SmallVector<Operation *, 1> loopOp{loop.getOperation()};
    (void)affine::getIndexSet(loopOp, &cst);
    // We will mark everything other than this loop IV as symbol for getting a
    // pair of <lb, ub> with a constant difference.
    cst.setDimSymbolSeparation(cst.getNumDimAndSymbolVars() - kCountToIndexAdjustment);
    unsigned lbPos, ubPos;
    if (!cst.getConstantBoundOnDimSize(kAffineFirstDimIndex, nullptr, nullptr, nullptr, &lbPos, &ubPos) ||
        lbPos == ubPos) {
      LLVM_DEBUG(llvm::dbgs() << "[tile separation] Can't get constant diff / equalities not yet handled\n");
      return failure();
    }

    // Set all variables as dimensions uniformly since some of those marked as
    // symbols above could be outer loop IVs (corresponding tile space IVs).
    cst.setDimSymbolSeparation(kAffineZeroSymbolCount);

    affine::AffineValueMap lbVmap, ubVmap;
    cst.getIneqAsAffineValueMap(kAffineFirstDimIndex, lbPos, lbVmap, b.getContext());
    cst.getIneqAsAffineValueMap(kAffineFirstDimIndex, ubPos, ubVmap, b.getContext());

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

LogicalResult LoopTiling::createTailBlockForBody(affine::AffineForOp forOp) {
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
void LoopTiling::updateForOpUsers(affine::AffineForOp forOp, int64_t newSize) {
  if (newSize == kNoSizeUpdate) {
    return;
  }
  for (OpOperand &use : forOp.getInductionVar().getUses()) {
    if (auto tiledOp = dyn_cast<affine::AffineForOp>(use.getOwner())) {
      auto ubMap = tiledOp.getUpperBoundMap();
      auto newExpr = tiledOp.getLowerBoundMap().getResult(kAffineFirstDimIndex) + newSize;
      ubMap = ubMap.replace(ubMap.getResult(kAffineFirstDimIndex), newExpr, ubMap.getNumDims(), ubMap.getNumSymbols());
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
LogicalResult LoopTiling::createTailBlock(affine::AffineForOp forOp) {
  auto origUbMap = forOp.getUpperBoundMap();
  auto origLbMap = forOp.getLowerBoundMap();
  AffineExpr differenceExpr = getDifferenceUbAndLb(origUbMap, origLbMap);
  if (auto cExpr = llvm::dyn_cast<AffineConstantExpr>(differenceExpr)) {
    return createTailBlockStatic(forOp, cExpr.getValue());
  }
  if (auto sExpr = llvm::dyn_cast<AffineSymbolExpr>(differenceExpr)) {
    return createTailBlockDynamic(forOp, sExpr);
  }
  return success();
}

LogicalResult LoopTiling::createTailBlockStatic(affine::AffineForOp forOp, int64_t differenceUbAndLb) {
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
  AffineExpr newExpr = ubMap.getResult(kAffineFirstDimIndex) - tailSize;
  ubMap = ubMap.replace(ubMap.getResult(kAffineFirstDimIndex), newExpr, ubMap.getNumDims(), ubMap.getNumSymbols());
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
  if (differenceUbAndLb < origStep && (newDifferenceUbAndLb != 0)) {
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
  if (differenceUbAndLb < origStep || (newDifferenceUbAndLb == 0) || isEqualToBlock) {
    return success();
  }

  // Insert the tail tiles
  // Ensure tail block is inserted after forOp in the same parent block, not inside forOp's body
  Block *parentBlock = forOp->getBlock();
  OpBuilder b(parentBlock, std::next(forOp->getIterator()));
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

LogicalResult LoopTiling::createTailBlockDynamic(affine::AffineForOp forOp, AffineSymbolExpr sExpr) {
  if (band.size() != bandTileSizes.size()) {
    forOp.emitError("Dynamic shape supports only one tiling.");
    return failure();
  }
  auto origUbMap = forOp.getUpperBoundMap();
  auto origUbOp = forOp.getUpperBoundOperands();
  int64_t origStep = forOp.getStepAsInt();
  // When the dynamic shape step is 1, the tail block does not need to be processed.
  // affine.for %arg4 = 0 to %dim
  if (origStep == kPointLoopStep) {
    // Recursively processes the forOp body.
    if (failed(createTailBlockForBody(forOp))) {
      return failure();
    }
    return success();
  }

  auto ubMap = origUbMap;
  AffineExpr newExpr = ubMap.getResult(kAffineFirstDimIndex) - sExpr % origStep;
  ubMap = ubMap.replace(ubMap.getResult(kAffineFirstDimIndex), newExpr, ubMap.getNumDims(), ubMap.getNumSymbols());
  forOp.setUpperBoundMap(ubMap);

  // Recursively processes the forOp body.
  if (failed(createTailBlockForBody(forOp))) {
    return failure();
  }

  // Insert the tail tiles
  // Ensure tail block is inserted after forOp in the same parent block, not inside forOp's body
  Block *parentBlock = forOp->getBlock();
  OpBuilder b(parentBlock, std::next(forOp->getIterator()));
  auto tailForOp = b.create<affine::AffineForOp>(forOp.getLoc(), forOp.getUpperBoundOperands(), ubMap, origUbOp,
                                                 origUbMap, kPointLoopStep);

  // Add the body for the full tile loop nest.
  b = OpBuilder::atBlockTerminator(tailForOp.getBody());
  for (auto &op : forOp.getBody()->without_terminator()) {
    b.clone(op);
  }
  // Add the body for the full tile loop nest.
  replaceAllUsesInRegionWith(forOp.getInductionVar(), tailForOp.getInductionVar(), tailForOp.getRegion());
  updateForOpUsers(tailForOp, kPointLoopStep);

  // Recursively processes the tailForOp body.
  if (failed(createTailBlockForBody(tailForOp))) {
    return failure();
  }
  return success();
}

LogicalResult LoopTiling::separateFullTilesNoIf(SmallVector<affine::AffineForOp, 6> tiledLoops) {
  // full block
  auto intraTileLoops = MutableArrayRef<affine::AffineForOp>(tiledLoops).drop_front(band.size());
  SmallVector<affine::AffineForOp, 4> fullTileLoops;
  if (failed(createFullBlock(intraTileLoops, fullTileLoops))) {
    if (!fullTileLoops.empty()) {
      fullTileLoops.front().erase();
    }
    return failure();
  }
  if (fullTileLoops.empty()) {
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
affine::AffineIfOp LoopTiling::createperfectlyNestedCondition(SmallVector<affine::AffineForOp, 6> tiledLoops,
                                                              OpBuilder b) {
  if (tiledLoops.empty()) {
    return nullptr;
  }

  SmallVector<AffineExpr, 4> exprs;
  SmallVector<bool, 4> eqFlags;
  SmallVector<Value, 4> dimOperands;
  SmallVector<Value, 4> symbolOperands;
  int64_t symbolNum = 0;
  int64_t dimNum = 0;
  for (auto loop : tiledLoops) {
    assert(loop.getStepAsInt() == kPointLoopStep && "point loop step expected to be one");

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

    if ((outerOp == nullptr) || !isa<affine::AffineForOp>(outerOp)) {
      continue;
    }

    auto outerForOp = dyn_cast<affine::AffineForOp>(outerOp);
    auto context = loop.getContext();
    AffineExpr upperExpr = outerForOp.getUpperBoundMap().getResult(kAffineFirstDimIndex);
    // Adapts to dynamic shapes.
    if (llvm::isa<AffineSymbolExpr>(upperExpr)) {
      for (auto operand : outerForOp->getOperands()) {
        if (Operation *parentOp = operand.getDefiningOp()) {
          symbolOperands.push_back(parentOp->getResult(0));
          upperExpr = getAffineSymbolExpr(symbolNum++, context);
        }
      }
    }
    // Make sure that the dim variable is incremented each time.
    AffineExpr newExpr = upperExpr - kExclusiveBoundAdjustment - getAffineDimExpr(dimNum++, context);
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
}

void LoopTiling::updateInsertIfLoops(SmallVector<affine::AffineForOp, 6> &newTiledLoops,
                                     std::unordered_set<unsigned> inequalityForIndex) {
  auto it = newTiledLoops.begin();
  unsigned i = 0;
  while (it != newTiledLoops.end()) {
    if (inequalityForIndex.count(i) != 0u) {
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
LogicalResult LoopTiling::perfectlyNestedWithIf(SmallVector<affine::AffineForOp, 6> tiledLoops) {
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
    if (ubMap.getResults().size() > kAffineSingleDimCount) {
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
  affine::getPerfectlyNestedLoops(newTiledLoops, tiledLoops[0]);
  Block *forBody = newTiledLoops[newTiledLoops.size() - 1].getBody();
  OpBuilder b(forBody, forBody->begin());
  SmallVector<Operation *, 6> bodyOp;
  // Records all ops in the for loop to facilitate the insertion of if statements.
  for (auto &it : *forBody) {
    Operation *op = &it;
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

void LoopTiling::tileEachBand() {
  getTileSizes();

  unsigned forNum = band.size();
  unsigned tileSizesNum = bandTileSizes.size();
  if (forNum == tileSizesNum) {
    SmallVector<affine::AffineForOp, 6> tiledNest;
    if (failed(affine::tilePerfectlyNested(band, bandTileSizes, &tiledNest))) {
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

void LoopTiling::runCpuOperation() {
  separateNoIf = true;
  tileEachBand();

  func::FuncOp funcOp = getOperation();
  auto opType = CommonUtils::getOperatorType(funcOp);
  OpBuilder b(funcOp);
  if (opType == OperatorTemplate::Reduction) {
    SmallVector<Operation *, 8> reduceLoops = CommonUtils::collectReductionAxes(funcOp);
    for (auto reduceLoop : reduceLoops) {
      reduceLoop->setAttr(kReductionLoopAttr, b.getUnitAttr());
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

bool LoopTiling::isDynamicShape() const { return akgglobal::ShapeAlignTool::getInstance().getFuncArgSizes() > 0; }

// Helper function to calculate nesting depth of a loop relative to funcOp
// Counts only affine.for loops in the path
static unsigned getLoopNestingDepth(affine::AffineForOp forOp, func::FuncOp funcOp) {
  unsigned depth = 0;
  Operation *currentOp = forOp.getOperation()->getParentOp();
  while ((currentOp != nullptr) && currentOp != funcOp.getOperation()) {
    if (isa<affine::AffineForOp>(currentOp)) {
      depth++;
    }
    currentOp = currentOp->getParentOp();
  }
  return depth;
}

// Helper function to check if a loop is innermost (no nested affine.for inside)
static bool isInnermostLoop(affine::AffineForOp forOp) {
  return !forOp.getBody()
            ->walk([&](affine::AffineForOp nestedForOp) { return WalkResult::interrupt(); })
            .wasInterrupted();
}

static bool isOutermostAffineForOp(affine::AffineForOp forOp) {
  Operation *parentOp = forOp->getParentOp();
  while (parentOp != nullptr) {
    if (isa<affine::AffineForOp>(parentOp)) {
      return false;
    }
    if (isa<func::FuncOp>(parentOp)) {
      break;
    }
    parentOp = parentOp->getParentOp();
  }
  return true;
}

static void moveTailBlockAfterFullLoop(affine::AffineForOp fullLoop, affine::AffineForOp tailBlock) {
  if (!tailBlock) {
    return;
  }
  Block *parentBlock = fullLoop->getBlock();
  if (tailBlock->getBlock() != parentBlock) {
    return;
  }
  bool foundFullLoop = false;
  for (auto it = parentBlock->begin(); it != parentBlock->end(); ++it) {
    if (&*it == fullLoop.getOperation()) {
      foundFullLoop = true;
    } else if (foundFullLoop && &*it == tailBlock.getOperation()) {
      tailBlock->moveBefore(parentBlock, parentBlock->end());
      break;
    }
  }
}

static Operation *findWrappingAffineIfOp(Operation *startOp, Block *stopBlock) {
  if (startOp == nullptr) {
    return nullptr;
  }
  if (auto ifOp = dyn_cast<affine::AffineIfOp>(startOp)) {
    return ifOp;
  }
  Operation *currentOp = startOp;
  while (currentOp != nullptr) {
    if (auto ifOp = dyn_cast<affine::AffineIfOp>(currentOp)) {
      return ifOp;
    }
    if (currentOp->getBlock() == stopBlock) {
      break;
    }
    currentOp = currentOp->getParentOp();
  }
  return nullptr;
}

// Collect outermost tiled loops that belong to the band starting from bandRootLoop
SmallVector<affine::AffineForOp> LoopTiling::collectOutermostTiledLoops(affine::AffineForOp bandRootLoop) {
  SmallVector<affine::AffineForOp> outermostTiledLoops;

  if (!bandRootLoop) {
    return outermostTiledLoops;
  }

  // bandRootLoop should be an outermost loop, so its block should contain all band outermost loops
  Block *bandBlock = bandRootLoop->getBlock();

  // Find the range of loops belonging to this band
  bool foundRoot = false;
  for (Operation &op : *bandBlock) {
    if (auto forOp = dyn_cast<affine::AffineForOp>(&op)) {
      // Check if we've reached the root
      if (forOp == bandRootLoop) {
        foundRoot = true;
      }

      if (foundRoot) {
        bool isOutermost = isOutermostAffineForOp(forOp);

        if (forOp->hasAttr(kTileForOneAttr)) {
          break;
        }
        if (!isOutermost) {
          continue;
        }
        if (!forOp.hasConstantBounds()) {
          break;
        }
        outermostTiledLoops.push_back(forOp);
      }
    } else if (foundRoot) {
      break;
    }
  }

  return outermostTiledLoops;
}

// Collect dimension information and find tail blocks for each outermost tiled loop
SmallVector<LoopTiling::DimensionInfo> LoopTiling::collectDimensionInfos(
  const SmallVector<affine::AffineForOp> &outermostTiledLoops, int32_t &maxNumBlocks) {
  SmallVector<DimensionInfo> dimInfos;
  maxNumBlocks = 0;

  for (auto fullLoop : outermostTiledLoops) {
    bool isReduceAxis = fullLoop->hasAttr(kReductionLoopAttr);
    int32_t lb = fullLoop.getConstantLowerBound();
    int32_t ub = fullLoop.getConstantUpperBound();
    int32_t step = fullLoop.getStepAsInt();
    int32_t extent = ub - lb;
    int32_t numBlocks = (extent + step - kCeilingDivAdjustment) / step;

    if (numBlocks <= 0) {
      continue;
    }

    // Find tail block: look for the next affine.for after this loop in the same parent block
    affine::AffineForOp tailBlock = nullptr;
    int32_t tailLb = 0, tailUb = 0, tailStep = 0;

    Operation *nextOp = fullLoop->getNextNode();
    if ((nextOp != nullptr) && isa<affine::AffineForOp>(nextOp)) {
      auto candidateTail = dyn_cast<affine::AffineForOp>(nextOp);
      if (candidateTail.hasConstantBounds() && candidateTail->getParentOp() == fullLoop->getParentOp()) {
        int32_t candidateLb = candidateTail.getConstantLowerBound();
        int32_t candidateUb = candidateTail.getConstantUpperBound();
        // Tail block typically starts at or after full block's upper bound
        if (candidateLb >= ub) {
          tailBlock = candidateTail;
          tailLb = candidateLb;
          tailUb = candidateUb;
          tailStep = candidateTail.getStepAsInt();
        }
      }
    }

    dimInfos.push_back({fullLoop, tailBlock, numBlocks, step, lb, ub, extent, tailLb, tailUb, tailStep, isReduceAxis});

    // Only consider non-reduction dimensions for max blocks calculation
    if (!isReduceAxis) {
      maxNumBlocks = std::max(maxNumBlocks, numBlocks);
    }
  }

  return dimInfos;
}

// Select the dimension to map: prefer non-reduction dimension with maximum blocks
LoopTiling::DimensionInfo *LoopTiling::selectMappingDimension(SmallVector<DimensionInfo> &dimInfos,
                                                              int32_t maxNumBlocks) {
  if (dimInfos.empty()) {
    return nullptr;
  }

  // Prefer non-reduction dimension with maximum blocks
  auto itMax = std::find_if(dimInfos.begin(), dimInfos.end(),
                            [&](const DimensionInfo &d) { return !d.isReduceAxis && d.numBlocks == maxNumBlocks; });
  if (itMax != dimInfos.end()) {
    return &(*itMax);
  }

  // If all dimensions are reduction axes, use first non-reduction or first dimension
  auto itNonReduce =
    std::find_if(dimInfos.begin(), dimInfos.end(), [](const DimensionInfo &d) { return !d.isReduceAxis; });
  if (itNonReduce != dimInfos.end()) {
    return &(*itNonReduce);
  }

  return &dimInfos[0];
}

// Collect outermost tiled loops, dimension info, select mapping dimension, and calculate numKernels
LoopTiling::MappingContext LoopTiling::collectAndSelectMappingDimension(affine::AffineForOp bandRootLoop) {
  MappingContext ctx = {};

  if (!bandRootLoop) {
    ctx.numKernels = kDefaultNumKernels;
    return ctx;
  }

  // Step 1: Collect outermost tiled loops
  SmallVector<affine::AffineForOp> outermostTiledLoops = collectOutermostTiledLoops(bandRootLoop);
  if (outermostTiledLoops.empty()) {
    ctx.numKernels = kDefaultNumKernels;
    return ctx;
  }

  // Step 2: Collect dimension information and find tail blocks
  int32_t maxNumBlocks = 0;
  SmallVector<DimensionInfo> dimInfos = collectDimensionInfos(outermostTiledLoops, maxNumBlocks);
  if (dimInfos.empty()) {
    ctx.numKernels = kDefaultNumKernels;
    return ctx;
  }

  // Step 3: Select the dimension to map
  DimensionInfo *selectedDim = selectMappingDimension(dimInfos, maxNumBlocks);
  if (selectedDim == nullptr) {
    ctx.numKernels = kDefaultNumKernels;
    return ctx;
  }

  // Step 4: Calculate number of kernels (limit to kMaxNumKernels)
  int32_t totalBlocks = selectedDim->numBlocks + (selectedDim->tailBlock ? kTailBlockCount : kNoTailBlockCount);
  ctx.fullBlock = selectedDim->fullBlock;
  ctx.tailBlock = selectedDim->tailBlock;
  ctx.numBlocks = selectedDim->numBlocks;
  ctx.step = selectedDim->step;
  ctx.lb = selectedDim->lb;
  ctx.ub = selectedDim->ub;
  ctx.numKernels = std::min(totalBlocks > 0 ? totalBlocks : static_cast<int32_t>(kMinNumKernels),
                            static_cast<int32_t>(kMaxNumKernels));
  return ctx;
}

// Create kernel loop and map full block to kernel iterations
affine::AffineForOp LoopTiling::createKernelLoopAndMapFullBlock(OpBuilder &builder, func::FuncOp funcOp,
                                                                MappingContext &ctx) {
  auto fullLoop = ctx.fullBlock;

  if (!fullLoop || ctx.numKernels <= 0) {
    llvm::report_fatal_error("Invalid inputs to createKernelLoopAndMapFullBlock");
  }

  // Verify fullLoop is still valid before using it
  if ((fullLoop.getOperation() == nullptr) || (fullLoop.getOperation()->getBlock() == nullptr)) {
    llvm::report_fatal_error("fullLoop is no longer valid in createKernelLoopAndMapFullBlock");
  }

  // Use fullLoop's location instead of funcOp's to ensure proper context
  Location loc = fullLoop.getLoc();

  // Insertion point should already be set in addOuterKernelMappingLoop
  auto kernelLoop = builder.create<affine::AffineForOp>(loc, kKernelLoopLowerBound, ctx.numKernels, kKernelLoopStep);

  kernelLoop->setAttr(kTileForOneAttr, builder.getUnitAttr());
  kernelLoop->setAttr(kMapForToForallAttr, builder.getUnitAttr());

  Block *kernelBody = kernelLoop.getBody();
  Value kernelId = kernelLoop.getInductionVar();

  // Move full block into kernel body
  if (fullLoop->getBlock() != kernelBody) {
    fullLoop->moveBefore(&kernelBody->front());
  }

  // Create affine maps for full block mapping
  auto kernelIdExpr = builder.getAffineDimExpr(kAffineFirstDimIndex);
  auto stepConstExpr = builder.getAffineConstantExpr(ctx.step);
  auto lbConstExpr = builder.getAffineConstantExpr(ctx.lb);
  auto startExpr = kernelIdExpr * stepConstExpr + lbConstExpr;
  auto endExpr = kernelIdExpr * stepConstExpr + stepConstExpr + lbConstExpr;

  auto startMap = AffineMap::get(kAffineSingleDimCount, kAffineZeroSymbolCount, startExpr, builder.getContext());
  auto endMap = AffineMap::get(kAffineSingleDimCount, kAffineZeroSymbolCount, endExpr, builder.getContext());

  // Map full block bounds to kernel_id, guarded so the last kernel (reserved for tail) skips full block
  if (ctx.tailBlock) {
    auto numBlocksConst = builder.getAffineConstantExpr(ctx.numBlocks);
    auto condExpr = numBlocksConst - builder.getAffineDimExpr(kAffineFirstDimIndex) -
                    builder.getAffineConstantExpr(kExclusiveBoundAdjustment);
    SmallVector<AffineExpr> exprs = {condExpr};
    SmallVector<bool> eqFlags = {false};
    auto condSet = IntegerSet::get(kAffineSingleDimCount, kAffineZeroSymbolCount, exprs, eqFlags);

    Block *parentBlock = fullLoop->getBlock();
    moveTailBlockAfterFullLoop(fullLoop, ctx.tailBlock);
    builder.setInsertionPoint(parentBlock, fullLoop->getIterator());
    (void)builder.create<affine::AffineIfOp>(fullLoop.getLoc(), condSet, ValueRange{kernelId}, /*hasElse=*/false);
  }
  fullLoop.setLowerBound(ValueRange{kernelId}, startMap);
  fullLoop.setUpperBound(ValueRange{kernelId}, endMap);

  return kernelLoop;
}

// Map tail block to the last kernel iteration
void LoopTiling::mapTailBlockToKernel(OpBuilder &builder, affine::AffineForOp kernelLoop, affine::AffineForOp fullLoop,
                                      affine::AffineForOp tailLoop, int32_t numKernels, Value kernelId) {
  if (!tailLoop) {
    return;
  }

  Block *kernelBody = kernelLoop.getBody();

  Operation *insertAfterOp = fullLoop.getOperation();
  Operation *wrappingIfOp = findWrappingAffineIfOp(fullLoop->getParentOp(), kernelBody);
  if (wrappingIfOp != nullptr) {
    insertAfterOp = wrappingIfOp;
  }
  // Move tail block into kernel body (after full block or its wrapping if)
  builder.setInsertionPointAfter(insertAfterOp);
  if (tailLoop->getBlock() != kernelBody) {
    tailLoop->moveBefore(builder.getInsertionBlock(), builder.getInsertionPoint());
  }

  // Create condition: kernel_id == numKernels - 1 (last iteration reserved for tail)
  auto lastKernelIdExpr = builder.getAffineDimExpr(kAffineFirstDimIndex);
  auto numKernelsConstExpr = builder.getAffineConstantExpr(numKernels);
  auto lastIterExpr =
    lastKernelIdExpr - (numKernelsConstExpr - builder.getAffineConstantExpr(kExclusiveBoundAdjustment));

  SmallVector<AffineExpr> exprs = {lastIterExpr};
  SmallVector<bool> eqFlags = {true};  // Equality condition: kernel_id == numKernels - 1
  auto lastIterSet = IntegerSet::get(kAffineSingleDimCount, kAffineZeroSymbolCount, exprs, eqFlags);

  builder.setInsertionPoint(tailLoop);
  auto tailIfOp =
    builder.create<affine::AffineIfOp>(tailLoop.getLoc(), lastIterSet, ValueRange{kernelId}, /*hasElse=*/false);

  tailLoop->moveBefore(&tailIfOp.getThenBlock()->front());
}

// Add outer kernel mapping loop to distribute work across NPU cores
void LoopTiling::addOuterKernelMappingLoop(affine::AffineForOp bandRootLoop) {
  MappingContext ctx = collectAndSelectMappingDimension(bandRootLoop);
  if (!ctx.fullBlock) {
    return;
  }

  // Verify fullBlock is still valid before using it
  if ((ctx.fullBlock.getOperation() == nullptr) || (ctx.fullBlock.getOperation()->getBlock() == nullptr)) {
    return;
  }

  MLIRContext *context = ctx.fullBlock.getOperation()->getContext();

  // Create builder with context from fullBlock, then set insertion point
  OpBuilder builder(context);
  builder.setInsertionPoint(ctx.fullBlock);

  // Create kernel loop and map full block
  // Note: funcOp parameter is not used in createKernelLoopAndMapFullBlock, but kept for API consistency
  auto funcOp = bandRootLoop->getParentOfType<func::FuncOp>();
  auto kernelLoop = createKernelLoopAndMapFullBlock(builder, funcOp, ctx);
  Value kernelId = kernelLoop.getInductionVar();

  // Map tail block to last kernel iteration
  mapTailBlockToKernel(builder, kernelLoop, ctx.fullBlock, ctx.tailBlock, ctx.numKernels, kernelId);
}

// Find the tiled band root loop after tiling transformation
affine::AffineForOp LoopTiling::findTiledBandRoot(Block *parentBlock, func::FuncOp funcOp) {
  for (auto &op : *parentBlock) {
    if (auto forOp = dyn_cast<affine::AffineForOp>(&op)) {
      // Skip if already processed
      if (forOp->hasAttr(kTileForOneAttr)) {
        continue;
      }

      // Check if it's outermost and has constant bounds
      Operation *parentOp = forOp->getParentOp();
      bool isOutermost = true;
      while ((parentOp != nullptr) && parentOp != funcOp.getOperation()) {
        if (isa<affine::AffineForOp>(parentOp)) {
          isOutermost = false;
          break;
        }
        parentOp = parentOp->getParentOp();
      }
      if (isOutermost && forOp.hasConstantBounds()) {
        // Found a candidate - use the first one (it should be the band root at or near the original position)
        return forOp;
      }
    }
  }
  return nullptr;
}

// Collect innermost loops with maximum depth and mark them with vector attribute
void LoopTiling::collectAndMarkInnermostLoops(func::FuncOp funcOp) {
  OpBuilder builder(funcOp);
  SmallVector<std::pair<affine::AffineForOp, unsigned>> innermostCandidates;

  // Collect all innermost loops with their depths
  funcOp->walk([&](affine::AffineForOp forOp) {
    unsigned depth = getLoopNestingDepth(forOp, funcOp);
    if (isInnermostLoop(forOp)) {
      innermostCandidates.push_back({forOp, depth});
    }
  });

  if (innermostCandidates.empty()) {
    return;
  }

  // Find maximum depth
  unsigned maxDepth = innermostCandidates[0].second;
  for (const auto &[loop, depth] : innermostCandidates) {
    if (depth > maxDepth) {
      maxDepth = depth;
    }
  }

  // Add vector attribute to innermost loops with maximum depth
  for (const auto &[loop, depth] : innermostCandidates) {
    if (depth == maxDepth) {
      loop->setAttr(kVectorAttr, builder.getUnitAttr());
    }
  }
}

void LoopTiling::runNpuOperation() {
  if (band.empty()) {
    return;
  }
  func::FuncOp funcOp = getOperation();

  // Use tail block without if
  separateNoIf = true;

  // Clear bandTileSizes to avoid pollution from previous bands
  bandTileSizes.clear();

  // Save the parent block before tiling to find the tiled root later
  Block *parentBlock = band[0]->getBlock();

  // Execute tiling transformation
  tileEachBand();

  // Find and process the tiled band root
  affine::AffineForOp tiledBandRoot = findTiledBandRoot(parentBlock, funcOp);
  if (tiledBandRoot) {
    addOuterKernelMappingLoop(tiledBandRoot);
  }

  // Collect and mark innermost loops with vector attribute
  collectAndMarkInnermostLoops(funcOp);

  if (useAutoTiling && solver) {
    // Annotate loops for cooperative scheduling
  }
}

void LoopTiling::runCudaOperation() {
  func::FuncOp funcOp = getOperation();
  auto opType = CommonUtils::getOperatorType(funcOp);
  if (!isDynamicShape() || opType == OperatorTemplate::Reduction) {
    inequalityConvertToIf = true;
  }
  tileEachBand();

  if (useAutoTiling && solver) {
    auto configSolver = std::make_shared<autotiling::GlobalConfigSolver>(solver);

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

void LoopTiling::BandCheck(const std::vector<SmallVector<affine::AffineForOp, 6>> &bands) {
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
  if (target == kTargetCpu || target == kTargetNpu) {
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

void LoopTiling::runOnOperation() {
  // Bands of loops to tile.
  func::FuncOp funcOp = getOperation();
  std::vector<SmallVector<affine::AffineForOp, 6>> bands;
  affine::getTileableBands(funcOp, &bands);
  if (getOperation()->getAttr("process")) {
    target = dyn_cast<StringAttr>(getOperation()->getAttr("process")).getValue().str();
  }

  if (!isDynamicShape()) {
    BandCheck(bands);
  }

  if (useAutoTiling) {
    auto initGraph = autotiling::parseIr(funcOp, bands);
    initGraph->setHardware(target);
    initGraph->setFeature(feature);
    initGraph->setArch(arch);
    initGraph->setIsDynamicShape(isDynamicShape());
    initGraph->setTilingMode(tilingMode);

    // If inputTileSizes is provided, use it to override multiTileSizes
    if (target == kTargetNpu && this->multiTileSizes.empty()) {
      this->multiTileSizes = this->inputTileSizes;
    }
    if (target == kTargetNpu && !this->multiTileSizes.empty()) {
      OpBuilder builder(funcOp);
      SmallVector<Attribute, 4> tileSizeAttrs;
      tileSizeAttrs.reserve(this->multiTileSizes.size());
      std::transform(this->multiTileSizes.begin(), this->multiTileSizes.end(), std::back_inserter(tileSizeAttrs),
                     [&builder](unsigned size) { return builder.getI32IntegerAttr(size); });
      funcOp->setAttr("npu.multiTileSizes", builder.getArrayAttr(tileSizeAttrs));
    }

    auto modelGraph = autotiling::buildModelGraph(initGraph);
    solver = autotiling::getHeuristicTilingSolver(modelGraph);
    levelToTile = modelGraph->levelToTile;
  }

  // Tile each band.
  for (size_t i = 0; i < bands.size(); ++i) {
    auto &curBand = bands[i];
    // Set current band index
    currentBandIdx = i;

    // Set current band
    band = MutableArrayRef<affine::AffineForOp>(curBand);
    if (target == kTargetCpu) {
      runCpuOperation();
    } else if (target == kTargetCuda) {
      runCudaOperation();
    } else if (target == kTargetNpu) {
      runNpuOperation();
    } else {
      llvm::errs() << "Currently, only cpu, cuda and ascend backends are supported.\n" << "Current Hardware:" << target;
    }
  }
}
}  // namespace mlir
