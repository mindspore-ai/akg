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

#include "akg/Analysis/TilingStrategy.h"

#include <algorithm>
#include <climits>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <utility>
#include "akg/Dialect/Affine/Analysis/GpuTemplateTilingSolver.h"
#include "akg/Utils/GlobalVars.hpp"
#include "akg/Utils/AnalysisCommon.hpp"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "akg/Utils/Constants.h"

namespace mlir {
namespace autotiling {
static constexpr const char *kNotInnerDimensionBroadcastLoopAttr = "not_inner_dimension_broadcast";
using akg::alignUpInt64;
using akg::ceilDivInt64;
using akg::computeBishengStrideAlignedStorageBytes;
using akg::getBishengLogicalStructuredStrideAlignDims;
using akg::getBishengStrideAlignTargetForBits;
using akg::getDefaultBishengStrideAlignDims;
using akg::multiplyAndCap;
using akg::utils::GpuInfo;
using akg::utils::StrategyHelper;
using llvm::SmallVector;
using mlir::akg::autotiling::GpuTemplateTilingSolver;

static constexpr int64_t kDynamicShapeValue = -1;
static constexpr int64_t kDynamicAllocationSize = -1;
static constexpr int64_t kInvalidAxisIndex = -1;
static constexpr int kDynamicTileMarker = -1;

static constexpr int kOneDynamicInnerTileSize = 1024;
static constexpr int kTwoDynamicInnerTileSizeOuter = 8;
static constexpr int kTwoDynamicInnerTileSizeInner = 32;
static constexpr int kMoreDynamicInnerTileSizeOuter = 4;
static constexpr int kMoreDynamicInnerTileSizeMiddle = 8;
static constexpr int kMoreDynamicInnerTileSizeInner = 32;

static constexpr int64_t kHeavyMathComplexityScore = 3;
static constexpr int64_t kReduceMathComplexityScore = 1;
static constexpr int64_t kMediumMathComplexityScore = 2;
static constexpr int64_t kMathComplexityCap = 64;

static constexpr int64_t kBlockCountScoreWeight = 10000;
static constexpr int64_t kRemainingTargetBonus = 500;

static constexpr size_t kMinTransposeAxisOrderSize = 2;
static constexpr size_t kMinPrefixTransposeDiffSize = 3;
static constexpr size_t kPairTransposeDimCount = 2;

static constexpr int64_t kDoubleAlignFactor = 2;
static constexpr size_t kNpuTileLevels = 2;
static constexpr int64_t kMinTileSizeMultiplier = 2;
static constexpr int64_t kUnrollShrinkFactor = 2;

static constexpr unsigned kMaxTileValue = UINT_MAX;
static constexpr int64_t kInitialMaxConstraintValue = INT_MAX;

bool TilingStrategy::IsRelevant(const AxisPtr &a, const InitGraphPtr graph) {
  if (a->axisType.find(workForAxisLabel) != a->axisType.end()) {
    return true;
  }
  // check workForOps
  if (IsRelevant(a, graph, this->workForOps)) {
    return true;
  }
  return false;
}

bool TilingStrategy::IsRelevant(const AxisPtr &a, const InitGraphPtr graph, std::unordered_set<std::string> ops) {
  if (ops.empty()) {
    return false;
  }
  return std::any_of(graph->nodes().begin(), graph->nodes().end(), [&a, &ops](const NodePtr &node) {
    if (ops.find(node->opType) == ops.end()) {
      return false;
    }
    const auto &loopNest = node->loopNest();
    return std::find(loopNest.begin(), loopNest.end(), a) != loopNest.end();
  });
}

void RepositoryStrategy::AddConstraint(ModelGraphPtr initGraph) {
  auto &tool = akgglobal::GpuScheduleTool::getInstance();
  initGraph->rootAxis->forEachAxisTopDown([this, &tool](const AxisPtr a) {
    for (auto axisInfo : tool.get(a->name)) {
      if (axisInfo.tileLevel <= 0 || axisInfo.constSize <= 0) {
        // tileLevel <= 0 is the other most axis and we don't need to tile it;
        // constSize <= 0 is dynamic tile and we don't support it for now.
        continue;
      }
      if (axisInfo.tileLevel == extraTileLevel) {
        a->doExtraTile();
      }
      auto tileCfg = a->tryGetConfig(axisInfo.tileLevel - 1);
      if (tileCfg == nullptr) {
        continue;
      }
      tileCfg->value = axisInfo.constSize;
    }
  });
}

Sketch DynamicShapeStrategy::SketchAnalysis(std::vector<int64_t> shapes) {
  Sketch gpuSketch = Sketch::kAllStatic;
  auto it = std::find(shapes.begin(), shapes.end(), kDynamicShapeValue);
  if (it == shapes.end()) {
    return gpuSketch;
  }
  if (shapes.back() == kDynamicShapeValue) {
    int dynCnt = 0;
    for (int i = static_cast<int>(shapes.size()) - 1; i >= 0; --i) {
      auto shape = shapes[static_cast<unsigned>(i)];
      if (shape != kDynamicShapeValue) {
        break;
      }
      ++dynCnt;
    }
    gpuSketch = static_cast<Sketch>(std::min<int>(dynCnt, static_cast<int>(Sketch::kMoreDynamicInner)));
  } else {
    int64_t mul = 1;
    for (int i = static_cast<int>(shapes.size()) - 1; i >= 0; --i) {
      auto shape = shapes[static_cast<unsigned>(i)];
      if (shape == kDynamicShapeValue) {
        break;
      }
      mul *= shape;
    }
    if (mul >= largeShapeLimit) {
      gpuSketch = Sketch::kLargeStaticInner;
    } else {
      gpuSketch = Sketch::kSmallStaticInner;
    }
  }
  return gpuSketch;
}

void DynamicShapeStrategy::AddCpuConstraint(CpuModelGraphPtr initGraph) {
  // This is a very naive strategy to tile a dynamic-shape axis with a fix size.
  initGraph->rootAxis->forEachAxisTopDown([this](const AxisPtr a) {
    if (a->axisType.find(mlir::autotiling::Axis::AxisLabel::kDynamic) != a->axisType.end()) {
      auto tile1_cons = Constraint({microTileSize});
      a->tryAddConstraint(0, tile1_cons);
    }
  });
}

void DynamicShapeStrategy::DoConstTile(const GpuModelGraphPtr initGraph, Sketch sketch) {
  std::vector<int> dynamicTiles;
  auto totalBlocks = GpuInfo::getInstance(initGraph->hardware).getTotalAvailableBlocks();
  if (sketch == Sketch::kOneDynamicInner) {
    dynamicTiles = {kOneDynamicInnerTileSize};
  } else if (sketch == Sketch::kTwoDynamicInner) {
    dynamicTiles = {kTwoDynamicInnerTileSizeOuter, kTwoDynamicInnerTileSizeInner};
  } else if (sketch == Sketch::kMoreDynamicInner) {
    dynamicTiles = {kMoreDynamicInnerTileSizeOuter, kMoreDynamicInnerTileSizeMiddle, kMoreDynamicInnerTileSizeInner};
  }
  int staticTiles = 1;
  initGraph->rootAxis->forEachAxisBottomUp([&dynamicTiles, &sketch, &totalBlocks, &staticTiles](const AxisPtr a) {
    if (a->axisType.find(mlir::autotiling::Axis::AxisLabel::kDynamic) != a->axisType.end()) {
      if (dynamicTiles.empty()) {
        if (sketch == Sketch::kSmallStaticInner && staticTiles < totalBlocks) {
          a->tryAddConstraint(0, Constraint({totalBlocks / staticTiles}));
        } else {
          a->tryAddConstraint(0, Constraint({1}));
        }
      } else {
        a->tryAddConstraint(0, Constraint({dynamicTiles.back()}));
        staticTiles *= dynamicTiles.back();
        dynamicTiles.pop_back();
      }
    } else {
      if (staticTiles < totalBlocks) {
        int tile = std::min<int64_t>(totalBlocks, a->range.second);
        a->tryAddConstraint(0, Constraint({tile}));
        staticTiles *= tile;
      } else {
        a->tryAddConstraint(0, Constraint({1}));
      }
    }
  });
}

void DynamicShapeStrategy::DoVariableTile(const GpuModelGraphPtr initGraph, Sketch sketch) {
  auto &gpuTool = akgglobal::GpuScheduleTool::getInstance();
  auto &tool = akgglobal::PrimeNumTool::getInstance();
  if (sketch != Sketch::kTwoDynamicInner) {
    DoConstTile(initGraph, sketch);
    return;
  }

  auto prime1 = tool.getOnePrimeWithIdxUpdate();
  auto prime2 = tool.getOnePrimeWithIdxUpdate();
  auto arg1 = gpuTool.addRuntimeArgument(static_cast<int64_t>(prime1));
  auto arg2 = gpuTool.addRuntimeArgument(static_cast<int64_t>(prime2));
  std::vector<akgglobal::RuntimeVar> dynamicTiles{arg1, arg2};
  size_t totalAxis = 0;
  initGraph->rootAxis->forEachAxisBottomUp([&totalAxis, &initGraph, currMapDim = size_t{0}](const AxisPtr a) mutable {
    ++totalAxis;
    if (a->axisType.find(mlir::autotiling::Axis::AxisLabel::kDynamic) != a->axisType.end()) {
      return;
    }
    // Static shape
    if (initGraph->gpuGrid.canApply(a->range.second)) {
      auto gridCfg = initGraph->gpuGrid.alloc(a, a->range.second);
      gridCfg->mapDim = static_cast<int>(currMapDim);
      currMapDim++;
    }
    a->tryAddConstraint(0, Constraint({1}));
  });

  size_t currDynAxis = 0;
  initGraph->rootAxis->forEachAxisBottomUp(
    [&dynamicTiles, &initGraph, &totalAxis, &currDynAxis, &gpuTool](const AxisPtr a) {
      if (a->axisType.find(mlir::autotiling::Axis::AxisLabel::kDynamic) == a->axisType.end()) {
        return;
      }
      // Dynamic shape
      if (dynamicTiles.empty()) {
        a->tryAddConstraint(0, Constraint({1}));
      } else {
        auto arg = dynamicTiles.back();
        auto prime = static_cast<int>(arg.prime);
        a->tryAddConstraint(0, Constraint({prime}));
        if (initGraph->gpuBlock.canApply(kDynamicAllocationSize)) {
          (void)initGraph->gpuBlock.alloc(a, kDynamicAllocationSize);
          auto blockCfg = std::make_shared<GpuBlock>("DynBlock");
          blockCfg->index = ConfigPos::kInner;
          blockCfg->value = prime;
          a->configs[blockCfg->type].push_back(blockCfg);
        }

        if ((totalAxis <= initGraph->gpuGrid.availbleSize.size() || currDynAxis == 1) &&
            initGraph->gpuGrid.canApply(kDynamicAllocationSize)) {
          (void)initGraph->gpuGrid.alloc(a, kDynamicAllocationSize);
          auto gridCfg = std::make_shared<GpuGrid>("DynGrid");
          gridCfg->index = ConfigPos::kOuter;
          gridCfg->value = 1;
          a->configs[gridCfg->type].push_back(gridCfg);
        }

        if (currDynAxis == 0) {
          arg.expr = "mod32 OR -1";
          gpuTool.updateRuntimeArgument(arg);
        } else if (currDynAxis == 1 && totalAxis >= initGraph->gpuGrid.availbleSize.size()) {
          arg.expr = "mod8 OR -1";
          gpuTool.updateRuntimeArgument(arg);
        }

        dynamicTiles.pop_back();
      }
      currDynAxis++;
    });
}

void DynamicShapeStrategy::AddGpuConstraint(GpuModelGraphPtr initGraph) {
  std::vector<int64_t> shapes;
  initGraph->rootAxis->forEachAxisTopDown([&shapes](const AxisPtr a) {
    if (a->axisType.find(mlir::autotiling::Axis::AxisLabel::kDynamic) != a->axisType.end()) {
      shapes.push_back(kDynamicShapeValue);
    } else if (a->range.second > 1) {
      shapes.push_back(a->range.second);
    }
  });
  auto sketch = SketchAnalysis(shapes);
  if (sketch == Sketch::kAllStatic) {
    return;
  }
  if (initGraph->tilingMode == "static") {
    DoConstTile(initGraph, sketch);
  } else {
    // `auto` tiling mode
    if (sketch == Sketch::kTwoDynamicInner) {
      DoVariableTile(initGraph, sketch);
    } else {
      DoConstTile(initGraph, sketch);
    }
  }
}

std::vector<int> rankArray(const std::vector<AxisPtr> &arr) {
  std::vector<std::pair<int, int>> pairs;
  for (size_t i = 0; i < arr.size(); i++) {
    auto axisLength = arr[i]->range.second;
    (void)pairs.emplace_back(axisLength, static_cast<int>(i));
  }

  std::sort(pairs.begin(), pairs.end(),
            [](const std::pair<int, int> &a, const std::pair<int, int> &b) { return a.first > b.first; });

  std::vector<int> ranks;
  for (const auto &p : pairs) {
    (void)ranks.emplace_back(p.second);
  }

  return ranks;
}

void TransposeStrategy::AddGpuConstraint(GpuModelGraphPtr gpuGraph) {
  std::deque<AxisPtr> sortByLoadAxes;
  NodePtr transposeRead;
  NodePtr transposeWrite;
  for (auto node : gpuGraph->nodes_) {
    auto isHigherRank = [&node](const NodePtr target) -> bool {
      return (target == nullptr || target->loopNest_.size() < node->loopNest_.size());
    };
    if (node->opType == "Load" && isHigherRank(transposeRead)) {
      transposeRead = node;
    }
    if (node->opType == "Store" && isHigherRank(transposeWrite)) {
      transposeWrite = node;
    }
  }
  if (!transposeRead || !transposeWrite || transposeWrite->loopNest_.size() < minRankForTranspose) {
    return;
  }

  sortByLoadAxes.assign(transposeWrite->loopNest_.rbegin(), transposeWrite->loopNest_.rend());
  std::sort(sortByLoadAxes.begin(), sortByLoadAxes.end(),
            [](const AxisPtr a1, const AxisPtr a2) { return a1->priority > a2->priority; });
  auto warpSize = GpuInfo::getInstance(gpuGraph->hardware).getWarpSizes();
  auto innerMostReadAxis = transposeRead->loopNest_.back();
  auto innerMostWriteAxis = transposeWrite->loopNest_.back();
  auto allocResource = [this, &warpSize, &gpuGraph](const AxisPtr axis) {
    auto blockSize = std::min<int64_t>(warpSize, axis->range.second);
    if (!gpuGraph->gpuBlock.canApply(blockSize)) {
      return;
    }
    auto blockcfg = gpuGraph->gpuBlock.alloc(axis, blockSize);
    blockcfg->index = ConfigPos::kInner;
    blockcfg->mapDim = static_cast<int>(gpuGraph->gpuBlock.currAllocDim()) - 1;
    auto outerSize = (axis->range.second - 1) / blockSize + 1;
    if (outerSize > maxExpectSeqPerAxis && outerSize % maxExpectSeqPerAxis == 0 &&
        gpuGraph->gpuGrid.canApply(outerSize / maxExpectSeqPerAxis)) {
      // we can do multi-tile for this divisible & large-shape case and alloc grids to the outer-most axis
      auto innerTile = axis->tryGetConfig(1);
      if (!innerTile) {
        axis->doExtraTile();
        innerTile = axis->tryGetConfig(1);
      }
      innerTile->value = static_cast<int>(blockSize);
      auto outerTile = axis->tryGetConfig(0);
      outerTile->value = innerTile->value * maxExpectSeqPerAxis;
      outerSize = outerSize / maxExpectSeqPerAxis;
      auto seqCfg = std::make_shared<GpuSeq>(maxExpectSeqPerAxis);
      seqCfg->index = ConfigPos::kMiddle;
      axis->configs[seqCfg->type].push_back(seqCfg);
      auto gridcfg = gpuGraph->gpuGrid.alloc(axis, outerSize);
      gridcfg->index = ConfigPos::kOuter;
    } else {
      // otherwise, we can only do single tile, and we alloc the outer-most axis to grids or seqs by condition
      auto tile = axis->tryGetConfig(0);
      tile->value = static_cast<int>(blockSize);
      if (outerSize <= maxExpectSeq) {
        auto seqCfg = std::make_shared<GpuSeq>(outerSize);
        seqCfg->index = ConfigPos::kOuter;
        axis->configs[seqCfg->type].push_back(seqCfg);
      } else {
        auto gridcfg = gpuGraph->gpuGrid.alloc(axis, outerSize);
        gridcfg->index = ConfigPos::kOuter;
      }
    }
  };
  allocResource(innerMostWriteAxis);
  allocResource(innerMostReadAxis);
  gpuGraph->sortedAxes = sortByLoadAxes;
  gpuGraph->updateMaxRankTensor(transposeWrite);
}

bool BroadcastStrategy::searchForSmallShape(const GpuModelGraphPtr gpuGraph, const AxisPtr a) {
  bool succ = false;
  auto warpSize = GpuInfo::getInstance(gpuGraph->hardware).getWarpSizes();
  for (auto expectSeq = minExpectSeq; expectSeq <= maxExpectSeq; ++expectSeq) {
    if (a->range.second % expectSeq != 0 || a->range.second / expectSeq < warpSize) {
      continue;
    }
    auto innerSize = a->range.second / expectSeq;
    if (!gpuGraph->gpuBlock.canApply(innerSize)) {
      continue;
    }
    auto blockcfg = gpuGraph->gpuBlock.alloc(a, innerSize);
    blockcfg->index = ConfigPos::kInner;
    auto tile = a->tryGetConfig(0);
    tile->value = static_cast<int>(innerSize);
    auto seqCfg = std::make_shared<GpuSeq>(expectSeq);
    seqCfg->index = ConfigPos::kOuter;
    a->configs[seqCfg->type].push_back(seqCfg);
    succ = true;
    break;
  }
  return succ;
}

bool BroadcastStrategy::searchForLargeShape(const GpuModelGraphPtr gpuGraph, const AxisPtr a) {
  bool succ = false;
  auto warpSize = GpuInfo::getInstance(gpuGraph->hardware).getWarpSizes();
  auto innerSize = warpSize;
  for (auto expectSeq = minExpectSeq; expectSeq <= maxExpectSeq; ++expectSeq) {
    auto middleSize = warpSize * expectSeq;
    if (a->range.second % middleSize != 0) {
      continue;
    }
    auto outerSize = a->range.second / middleSize;
    if (!gpuGraph->gpuGrid.canApply(outerSize)) {
      continue;
    }
    auto blockcfg = gpuGraph->gpuBlock.alloc(a, innerSize);
    blockcfg->index = ConfigPos::kInner;
    a->doExtraTile();
    auto tile1 = a->tryGetConfig(1);
    tile1->value = innerSize;

    auto tile = a->tryGetConfig(0);
    tile->value = middleSize;
    auto seqCfg = std::make_shared<GpuSeq>(expectSeq);
    seqCfg->index = ConfigPos::kMiddle;
    a->configs[seqCfg->type].push_back(seqCfg);

    auto gridcfg = gpuGraph->gpuGrid.alloc(a, outerSize);
    gridcfg->index = ConfigPos::kOuter;
    succ = true;
    break;
  }
  return succ;
}

NodePtr BroadcastStrategy::findMinRankNode(const GpuModelGraphPtr gpuGraph) {
  NodePtr minRankNode = nullptr;
  for (auto node : gpuGraph->nodes()) {
    if (node->loopNest_.empty() || (node->opType != "Load" && node->opType != "Store")) {
      continue;
    }
    if (minRankNode == nullptr || minRankNode->loopNest_.size() > node->loopNest_.size()) {
      minRankNode = node;
    }
  }
  return minRankNode;
}

int BroadcastStrategy::computeExpectedSeq(const GpuModelGraphPtr gpuGraph, const AxisPtr innerMostReadAxis) {
  auto maxAvailBlocks = gpuGraph->gpuBlock.totalAvailableSize;
  assert(maxExpectSeq > 0 && innerMostReadAxis->range.second > 0);
  auto maxBlocks = StrategyHelper::getLargestDivisor(maxAvailBlocks, innerMostReadAxis->range.second / maxExpectSeq);
  return std::min<int>(maxExpectSeq,
                       (maxBlocks * maxExpectSeq * blockWasteCoef - 1) / innerMostReadAxis->range.second + 1);
}

void BroadcastStrategy::searchSeqAxisFromInnerToOuter(const GpuModelGraphPtr gpuGraph, const NodePtr maxRankNode,
                                                      int expectSeq) {
  int currBlocks = 1;
  auto warpSize = GpuInfo::getInstance(gpuGraph->hardware).getWarpSizes();

  for (int i = static_cast<int>(maxRankNode->loopNest_.size()) - 1; i >= 0; --i) {
    auto a = maxRankNode->loopNest_[static_cast<unsigned>(i)];
    if (expectSeq <= 1 || currBlocks >= proposedBlock || currBlocks >= warpSize) {
      break;
    }
    auto currGrids = gpuGraph->problemSize / currBlocks / a->range.second;
    if ((currGrids * gridWasteCoef >= proposedGrid) && gpuGraph->gpuBlock.canApply(a->range.second / maxExpectSeq)) {
      if (searchForSmallShape(gpuGraph, a)) {
        expectSeq = 1;
      }
    } else if (!gpuGraph->isDynamicShape && a->range.second % warpSize == 0 && gpuGraph->gpuBlock.canApply(warpSize)) {
      if (searchForLargeShape(gpuGraph, a)) {
        expectSeq = 1;
      }
    }
    currBlocks *= static_cast<int>(a->range.second);
  }
}

void BroadcastStrategy::AddGpuConstraint(GpuModelGraphPtr gpuGraph) {
  NodePtr maxRankNode = gpuGraph->getMaxRankTensor();
  if (!maxRankNode || maxRankNode->loopNest_.size() <= 1) {
    return;
  }

  NodePtr minRankNode = findMinRankNode(gpuGraph);
  if (!minRankNode || minRankNode->loopNest_.size() == maxRankNode->loopNest_.size()) {
    return;
  }

  auto innerMostReadAxis = maxRankNode->loopNest_.back();
  int expectSeq = computeExpectedSeq(gpuGraph, innerMostReadAxis);

  std::tie(proposedGrid, proposedBlock) =
    StrategyHelper::getProposalParallelSize(gpuGraph->problemSize, gpuGraph->hardware);

  searchSeqAxisFromInnerToOuter(gpuGraph, maxRankNode, expectSeq);
}

void ParallelStrategy::InitProposalResource(const GpuModelGraphPtr gpuGraph) {
  std::tie(proposedGrid, proposedBlock) =
    StrategyHelper::getProposalParallelSize(gpuGraph->problemSize, gpuGraph->hardware);
}

bool ParallelStrategy::tryMapBlock(const GpuModelGraphPtr gpuGraph, const AxisPtr axis) {
  auto loopExtent = axis->getRestExtent();
  int64_t blockLimit = std::min<int64_t>(proposedBlock / gpuGraph->gpuBlock.currSize, gpuGraph->gpuBlock.rest());
  int64_t largestForBlock = StrategyHelper::getLargestDivisor(blockLimit, loopExtent);
  auto warpSize = GpuInfo::getInstance(gpuGraph->hardware).getWarpSizes();
  // If reduction axis cannot find a divisor for block, then directly use all rest blocks.
  // Then we will map the outer-axis (divisible part) to block;
  // while keep inner-axis (indivisible part) to sequential.
  if (largestForBlock == 1 && axis->axisType.find(Axis::AxisLabel::kReduction) != axis->axisType.end()) {
    largestForBlock = gpuGraph->gpuBlock.totalAvailableSize / gpuGraph->gpuBlock.currSize;
  } else if (loopExtent < axis->range.second) {
    largestForBlock = loopExtent;
  } else if (largestForBlock * blockWasteCoef <= blockLimit &&
             gpuGraph->gpuBlock.canApply(std::min<int64_t>(blockLimit, loopExtent))) {
    largestForBlock = std::min<int64_t>(blockLimit, loopExtent);
    currHasMinMax = true;
  } else if (axis->isInnerMost && largestForBlock % warpSize != 0 && largestForBlock * blockLimitCoef <= blockLimit &&
             gpuGraph->gpuBlock.canApply(std::min<int64_t>(blockLimit, loopExtent))) {
    largestForBlock = std::min<int64_t>(blockLimit, loopExtent);
    currHasMinMax = true;
  }
  if (!gpuGraph->gpuBlock.seen(axis) && gpuGraph->gpuBlock.canApply(largestForBlock)) {
    (void)gpuGraph->gpuBlock.alloc(axis, largestForBlock);
    if (currHasMinMax) {
      auto tile = axis->tryGetConfig(0);
      tile->value = static_cast<int>(largestForBlock);
    } else {
      auto consTile = Constraint(1, static_cast<int>(largestForBlock), 1);
      axis->tryAddConstraint(0, consTile);
    }
    return true;
  }
  return false;
}

bool ParallelStrategy::tryMapGrid(const GpuModelGraphPtr gpuGraph, const AxisPtr axis) {
  auto DisableGridMapping = [&gpuGraph](const AxisPtr &axis) -> bool {
    if (gpuGraph->globalConfigs.find(akg::utils::kEnableAtomicAdd) == gpuGraph->globalConfigs.end()) {
      return false;
    }
    if (axis->axisType.find(mlir::autotiling::Axis::AxisLabel::kReduction) == axis->axisType.end()) {
      return false;
    }
    return !dyn_cast<BoolAttr>(gpuGraph->globalConfigs[akg::utils::kEnableAtomicAdd]).getValue();
  };
  // Calculate rest extent after mapping to block (if any), and then try to map to grid.
  auto loopExtent = axis->getRestExtent();
  int64_t gridLimit = std::min<int64_t>(proposedGrid / gpuGraph->gpuGrid.currSize, gpuGraph->gpuGrid.rest());
  int64_t largestForGrid = StrategyHelper::getLargestDivisor(gridLimit, loopExtent);
  if (loopExtent < axis->range.second) {
    largestForGrid = loopExtent;
  }
  if (!currHasMinMax && largestForGrid * gridWasteCoef < gpuGraph->gpuGrid.rest() &&
      gpuGraph->gpuGrid.canApply(std::min<int64_t>(gpuGraph->gpuGrid.rest(), loopExtent))) {
    largestForGrid = std::min<int64_t>(gpuGraph->gpuGrid.rest(), loopExtent);
    currHasMinMax = true;
  }
  if (!gpuGraph->gpuGrid.seen(axis) && gpuGraph->gpuGrid.canApply(largestForGrid) && !DisableGridMapping(axis)) {
    (void)gpuGraph->gpuGrid.alloc(axis, largestForGrid);
    if (largestForGrid < loopExtent && loopExtent < axis->range.second) {
      axis->doExtraTile();
      auto consTile = Constraint({static_cast<int>(largestForGrid)});
      axis->tryAddConstraint(1, consTile);
    } else {
      auto consTile = Constraint(axis->range.second / largestForGrid, axis->range.second, 1);
      axis->tryAddConstraint(0, consTile);
    }
    return true;
  }
  return false;
}

void ParallelStrategy::AddGpuConstraint(GpuModelGraphPtr gpuGraph) {
  InitProposalResource(gpuGraph);
  for (auto axis : gpuGraph->sortedAxes) {
    currHasMinMax = false;
    auto loopExtent = axis->getRestExtent();
    if (loopExtent == 1) {
      continue;
    }
    auto succMapBlock = tryMapBlock(gpuGraph, axis);
    if (succMapBlock && gpuGraph->isDynamicShape) {
      // Map one axis to both block and grid is not supported for dynamic shape.
      continue;
    }
    (void)tryMapGrid(gpuGraph, axis);
    gpuGraph->hasMinMax |= currHasMinMax;
  }
}

// This reduction static strategy  deals with both reduce-axes & non-reduce-axes on gpu backend
void ReduceStrategy::AddGpuConstraint(GpuModelGraphPtr gpuGraph) {
  OpBuilder builder(gpuGraph->funcOp);
  std::vector<AxisPtr> axes;
  gpuGraph->rootAxis->forEachAxisTopDown([this, &axes, &gpuGraph](const AxisPtr a) {
    if (!this->IsRelevant(a, gpuGraph)) {
      return;
    }
    axes.push_back(a);
  });

  // todo(yanzhi): remove it later for dynamic cases
  gpuGraph->hasMinMax = true;

  bool enableParallelReduction = true;
  bool enableAtomicReduction = gpuGraph->funcOp->hasAttr(akg::utils::kEnableAtomicAdd) &&
                               dyn_cast<BoolAttr>(gpuGraph->funcOp->getAttr(akg::utils::kEnableAtomicAdd)).getValue();
  bool applyReorderPass = true;
  GpuTemplateTilingSolver::SolveScheduleForReductionOps(axes, enableParallelReduction, enableAtomicReduction,
                                                        applyReorderPass);

  gpuGraph->globalConfigs[akg::utils::kEnableParallelReduce] = builder.getBoolAttr(enableParallelReduction);
  gpuGraph->globalConfigs[akg::utils::kEnableAtomicAdd] = builder.getBoolAttr(enableAtomicReduction);
  gpuGraph->globalConfigs[akg::utils::kApplyReorderPass] = builder.getBoolAttr(applyReorderPass);
}

int64_t BroadcastStrategy::getMaxByte(const CpuModelGraphPtr cpuGraph, NodePtr &minRankNode) {
  int64_t maxByte = 0;
  for (auto node : cpuGraph->nodes()) {
    if (node->loopNest_.empty() || (node->opType != "Load" && node->opType != "Store")) {
      continue;
    }
    if (minRankNode == nullptr || minRankNode->loopNest_.size() > node->loopNest_.size()) {
      minRankNode = node;
    }
    if (!node->dataType) {
      continue;
    }
    maxByte = std::max(maxByte, static_cast<int64_t>(node->dataType.getIntOrFloatBitWidth()));
  }
  maxByte /= cpuGraph->bitUnit;
  return maxByte;
}

int BroadcastStrategy::getMaxBroadcastAxes(const CpuModelGraphPtr cpuGraph, std::vector<AxisPtr> maxloopNest) {
  NodePtr minRankNode = nullptr;
  int64_t maxByte = getMaxByte(cpuGraph, minRankNode);
  if (!minRankNode || minRankNode->loopNest_.size() == maxloopNest.size()) {
    return static_cast<int>(kInvalidAxisIndex);
  }

  if (cpuGraph->funcOp->hasAttr(kOperatorTypeStr)) {
    OpBuilder builder(cpuGraph->funcOp);
    (void)cpuGraph->funcOp->removeAttr(kOperatorTypeStr);
    Attribute opType = builder.getStringAttr("Broadcast");
    cpuGraph->funcOp->setAttr(kOperatorTypeStr, opType);
  }

  // Number of loads and stores, which are used to calculate the upper limit of bytes.
  int broadcastOpNum = 0;
  int nonBroadcastOpNum = 0;
  int estimatedDifferenceNum = 0;
  std::vector<AxisPtr> minloopNest = minRankNode->loopNest_;
  for (auto node : cpuGraph->nodes()) {
    if (node->loopNest_.empty() || (node->opType != "Load" && node->opType != "Store")) {
      continue;
    }
    if (maxloopNest.size() == node->loopNest_.size()) {
      ++broadcastOpNum;
    }
    if (minloopNest.size() == node->loopNest_.size()) {
      if (node->opType == "Store") {
        ++estimatedDifferenceNum;
      }
      ++nonBroadcastOpNum;
    }
  }
  nonBroadcastOpNum -= estimatedDifferenceNum;
  broadcastOpNum += estimatedDifferenceNum;

  int64_t broadcastSize = 1;
  for (int i = static_cast<int>(maxloopNest.size()) - 1; i >= 0; --i) {
    auto axis = maxloopNest[static_cast<unsigned>(i)];
    int64_t axisSize = axis->range.second;
    auto iter = std::find(minloopNest.begin(), minloopNest.end(), axis);
    if (iter == minloopNest.end()) {
      broadcastSize *= axisSize;
    }
  }

  // Tile multi non-broadcast
  // Size of bytes occupied by the broadcast forward fusion operator.
  int64_t nonBroadcastBytesSize = nonBroadcastOpNum * maxByte;
  // Maximum number of bytes of the broadcast operator.
  int64_t broadcastBytesSize = broadcastSize * broadcastOpNum * maxByte;
  // Upper limit of the total size of all non-broadcast axes.
  return static_cast<int>(cpuGraph->l1Cache / (broadcastBytesSize + nonBroadcastBytesSize + 1));
}

void BroadcastStrategy::AddCpuConstraint(CpuModelGraphPtr cpuGraph) {
  NodePtr maxRankNode = cpuGraph->getMaxRankTensor();
  if (!maxRankNode || maxRankNode->loopNest_.size() <= 1) {
    return;
  }
  std::vector<AxisPtr> maxloopNest = maxRankNode->loopNest_;
  int maxSize = static_cast<int>(maxloopNest.size());
  if (maxloopNest[maxSize - 1]->isInnerMost) {
    return;
  }

  int maxBroadcastAxes = getMaxBroadcastAxes(cpuGraph, maxloopNest);
  bool isNewBand = true;
  int pos = cpuGraph->tileNum;
  for (int i = static_cast<int>(maxloopNest.size()) - 1; i >= 0; --i) {
    auto axis = maxloopNest[i];
    if (pos != 0) {
      axis->doExtraTile();
    }
    int64_t tileSize = 1;
    int64_t axisSize = axis->range.second;
    if (axis->isInnerMost && isNewBand) {
      tileSize = StrategyHelper::getLargestDivisor(maxBroadcastAxes, axisSize);
      isNewBand = false;
    }
    tileSize = std::min(axisSize, tileSize);
    axis->tryAddConstraint(0, Constraint({static_cast<int>(tileSize)}), kTileCfg);
  }
  ++cpuGraph->tileNum;
}

void UnrollStrategy::AddCpuConstraint(CpuModelGraphPtr cpuGraph) {
  auto iter = cpuInstructionSetMap.find(cpuGraph->feature);
  if (iter == cpuInstructionSetMap.end()) {
    llvm::errs() << "The instruction set supported by the cpu only includes "
                    "sse, avx, avx2, avx512 and neon."
                 << "\n";
    return;
  }
  int64_t instructionSetBit = iter->second;
  int64_t vectorSize = kVectorize128Bit;
  for (auto node : cpuGraph->nodes_) {
    if (!node->dataType) {
      continue;
    }
    auto elementBit = static_cast<int64_t>(node->dataType.getIntOrFloatBitWidth());
    vectorSize = std::min(vectorSize, instructionSetBit / elementBit);
  }
  int pos = cpuGraph->tileNum;
  for (auto bandRoot : cpuGraph->rootAxis->children) {
    std::deque<AxisPtr> q = {bandRoot};
    bandRoot->forEachAxisTopDown([&q](const AxisPtr axis) { q.push_front(axis); });

    bool isNewBand = true;
    for (auto axis : q) {
      if (pos != 0) {
        axis->doExtraTile();
      }

      if (!axis->isInnerMost || !isNewBand) {
        axis->tryAddConstraint(pos, Constraint({1}), kTileCfg);
        continue;
      }

      if (axis->axisType.find(Axis::AxisLabel::kDynamic) != axis->axisType.end()) {
        axis->tryAddConstraint(pos, Constraint({(int32_t)vectorSize}), kTileCfg);
      } else {
        int64_t unrollSize = BEST_UNROLL_NUM;
        int64_t axisSize = axis->range.second;
        while (axisSize % unrollSize != 0 && unrollSize > MIN_UNROLL_NUM) {
          unrollSize /= kUnrollShrinkFactor;
        }

        if (axisSize % vectorSize == 0) {
          vectorSize = 1;
        } else if (axisSize > vectorSize) {
          vectorSize = axisSize - axisSize % vectorSize;
        }
        int64_t curVectorSize = std::min(axisSize, vectorSize);
        axis->tryAddConstraint(pos, Constraint({static_cast<int>(curVectorSize)}), kTileCfg);
      }
      // Vectorization and unroll are on the same axis.
      (void)axis->axisType.insert(mlir::autotiling::Axis::AxisLabel::kVectorization);
      isNewBand = false;
    }
  }
  ++cpuGraph->tileNum;
}

void ParallelStrategy::AddCpuConstraint(CpuModelGraphPtr cpuGraph) {
  for (auto bandRoot : cpuGraph->rootAxis->children) {
    int64_t dataSize = bandRoot->range.second;
    std::deque<AxisPtr> q = {bandRoot};
    bandRoot->forEachAxisTopDown([&q, &dataSize](const AxisPtr axis) {
      q.push_front(axis);
      dataSize *= axis->range.second;
    });

    for (auto axis : q) {
      // Keep the current single-axis parallel handling.
      if (axis->axisIdx != 0) {
        continue;
      }
      (void)axis->axisType.insert(mlir::autotiling::Axis::AxisLabel::kMultiCore);
      int64_t axisSize = axis->range.second;
      int parallelNum = BEST_PARALLEL_NUM;
      int unrollTileValue = 1;
      // Vectorization and unroll are on the same axis.
      bool isUnroll = axis->axisType.count(mlir::autotiling::Axis::AxisLabel::kVectorization) != 0u;
      if (isUnroll) {
        auto unrollConfig = axis->tryGetConfig(1, kTileCfg);
        unrollConfig->mergeConstraints();
        unrollTileValue = unrollConfig->getValidCandidates()[0];
        axisSize = unrollTileValue;
      }

      int evaluateNum = static_cast<int>(dataSize) / MIN_EXEC_NUM_PER_THREAD;
      if (evaluateNum >= BEST_PARALLEL_NUM) {
        parallelNum = std::min(axis->range.second, static_cast<int64_t>(BEST_PARALLEL_NUM));
      } else if (evaluateNum > 1) {
        while (parallelNum > 0 && axisSize % parallelNum != 0) {
          parallelNum -= PARALLEL_DECREASE_VALUE;
        }
      } else {
        parallelNum = 1;
      }
      if (parallelNum <= 0) {
        parallelNum = evaluateNum;
      }

      int paralleltileValue = static_cast<int>(axis->range.second) / parallelNum;
      // todo(hujiahui) : reduceY
      if (axis->range.second % parallelNum != 0) {
        paralleltileValue = static_cast<int>(axis->range.second);
      }

      if (paralleltileValue < MIN_UNROLL_NUM && isUnroll) {
        paralleltileValue = std::min(static_cast<int>(axis->range.second), MIN_UNROLL_NUM);
        unrollTileValue = paralleltileValue;
      }
      paralleltileValue = std::max(paralleltileValue, unrollTileValue);

      axis->tryAddConstraint(0, Constraint({paralleltileValue}));
      axis->tryAddConstraint(1, Constraint({unrollTileValue}));
    }
  }
}

unsigned getOuterTileSize(const AxisPtr axis, unsigned blockNumber) {
  const unsigned MIN_TILE_SIZE = 512;
  if (!axis || !axis->hasConstantBounds()) {
    return MIN_TILE_SIZE;
  }
  int64_t upperBound = axis->getConstantUpperBound();
  int64_t lowerBound = axis->getConstantLowerBound();
  auto extent = static_cast<unsigned>(upperBound - lowerBound);
  unsigned tileSizePerBlock = (extent + blockNumber - 1) / blockNumber;

  if (tileSizePerBlock < MIN_TILE_SIZE && extent >= MIN_TILE_SIZE) {
    tileSizePerBlock = MIN_TILE_SIZE;
  }
  return tileSizePerBlock;
}

namespace {
constexpr int64_t kBytesPerKb = 1024;
constexpr int64_t kNpuFallbackUbSizeKb = 192;
constexpr int64_t kUbGuardReserveKb = 64;

constexpr unsigned kNpuTargetBlocks = 48;
constexpr int64_t kNpuFallbackUbSizeInBytes = kNpuFallbackUbSizeKb * kBytesPerKb;
constexpr unsigned kNpuMinInnerTileSize = 512;
constexpr int64_t kDynamicUnknownExtentForAlignment = 2;
constexpr int64_t kDefaultTypeBits = 32;
constexpr int64_t kBitsPerByte = 8;
constexpr int64_t kUbAlignBytes = 32;
constexpr int64_t kUbAlignBits = kUbAlignBytes * kBitsPerByte;
constexpr int64_t kUbGuardReserveBytes = kUbGuardReserveKb * kBytesPerKb;
constexpr int64_t kHivmAutoMultiBufferFactor = 2;
constexpr int64_t kGenericHivmWorkspaceBuffers = 1;
constexpr int64_t kCompareHivmWorkspaceBuffers = 3;
constexpr int64_t kHeavyHivmWorkspaceBuffers = 3;
constexpr int64_t kReductionBackendExtraReserveBuffers = 1;
constexpr int64_t kReductionFixedReserveAlignUnits = 4;
constexpr int64_t kReductionBackendExtraFixedReserveBytes = kUbAlignBytes * kReductionFixedReserveAlignUnits;
constexpr int64_t kTransposeOrBroadcastFootprintBuffers = 4;
constexpr int64_t kSuffixPreservePointRows = 2;

struct TransposeCoLiveReserve {
  SmallVector<size_t, kSmallVectorSizeSix> axisOrder;
  SmallVector<int32_t, kSmallVectorSizeSix> alignDims;
  int64_t elementBits{kDefaultTypeBits};
};

struct VectorLiveReserve {
  SmallVector<size_t, kSmallVectorSizeSix> axisOrder;
  SmallVector<int32_t, kSmallVectorSizeSix> alignDims;
  int64_t elementBits{kDefaultTypeBits};
};

struct VectorPeakReserveSet {
  SmallVector<VectorLiveReserve, kSmallVectorSizeEight> liveBuffers;
  int64_t fixedBytes{0};
};

struct TransposeVectorInfo {
  SmallVector<size_t, kSmallVectorSizeSix> sourceAxisOrder;
  SmallVector<size_t, kSmallVectorSizeSix> targetAxisOrder;
  int64_t elementBits{kDefaultTypeBits};
  bool isF32{true};
};

struct NpuBandContext {
  size_t bandIdx{0};
  SmallVector<AxisPtr, kSmallVectorSizeFour> axes;
  SmallVector<int64_t, kSmallVectorSizeFour> extents;
  SmallVector<NodePtr, kSmallVectorSizeThirtyTwo> nodes;
  GraphTemplate graphTemplate{GraphTemplate::DEFAULT};
  int64_t rawUbElems{1};
  int64_t ubCapacityElems{1};
  int64_t vectorUbCapacityElems{1};
  int64_t vectorUbBitsPerElem{kDefaultTypeBits};
  int64_t vectorElementBits{kDefaultTypeBits};
  int64_t transposeElementBits{kDefaultTypeBits};
  bool transposeElementIsF32{true};
  SmallVector<int64_t, kSmallVectorSizeSix> axesAlignUnits;
  SmallVector<size_t, kSmallVectorSizeSix> transposeSourceAxisOrder;
  SmallVector<size_t, kSmallVectorSizeSix> transposeTargetAxisOrder;
  SmallVector<TransposeVectorInfo, kSmallVectorSizeSix> transposeInfos;
  SmallVector<SmallVector<TransposeCoLiveReserve, kSmallVectorSizeFour>, kSmallVectorSizeFour>
    transposeCoLiveReserveSets;
  SmallVector<VectorPeakReserveSet, kSmallVectorSizeThirtyTwo> vectorPeakReserveSets;
  int64_t fixedReserveBytes{0};
  int64_t targetBlocks{kNpuTargetBlocks};
  int64_t mathComplexityScore{0};
  int64_t smallestTypeBits{kDefaultTypeBits};
  bool hasDynamicAxis{false};
  bool hasReduction{false};
  bool lastAxisIsReduction{false};
};

struct BandTilePlan {
  SmallVector<unsigned, kSmallVectorSizeFour> outerTiles;
  SmallVector<unsigned, kSmallVectorSizeFour> innerTiles;
  // True for axes that participate in multi-dimensional vectorization.
  // When set, the corresponding inner-tile (point) loop receives
  // `kMultiVecLoopAttr` so the apply phase can sink them together and
  // consume them as a unified vec chain.
  SmallVector<bool, kSmallVectorSizeSix> multiVecAxisMask;
  // When true, the plan used multi-dimensional vectorization greedy search;
  // `tagMultiVecLoops` consumes `multiVecAxisMask` on the apply path.
  bool usesMultiVecScheme{false};
  bool preserveWholeBandTiles{false};
};

int64_t computeAxisAlignSize(size_t axisIdx, const NpuBandContext &ctx);
void applyF32LastDimDoubleAlign(const NpuBandContext &ctx, SmallVectorImpl<int64_t> &axisAlignUnits);
bool isParallelCandidateAxis(const NpuBandContext &ctx, size_t axisIdx);

static void applyFallbackAxisTiling(const AxisPtr axis, const SmallVector<unsigned, kSmallVectorSizeFour> &tileSizes,
                                    unsigned innerTileSize, unsigned blockNumber, size_t &maxLevelToTile,
                                    bool isFullTileAxis = false);

static SmallVector<unsigned, kSmallVectorSizeFour> computeFallbackTileSizes(
  const AxisPtr axis, const SmallVector<unsigned, kSmallVectorSizeFour> &tileSizes, unsigned innerTileSize,
  unsigned blockNumber, bool isFullTileAxis) {
  int64_t lowerBound = axis->getConstantLowerBound();
  int64_t upperBound = axis->getConstantUpperBound();
  int64_t extent = upperBound - lowerBound;

  SmallVector<unsigned, kSmallVectorSizeFour> currentTileSizes = tileSizes;
  if (isFullTileAxis) {
    unsigned fullTile = 1;
    if (extent > 0) {
      fullTile = extent > static_cast<int64_t>(kMaxTileValue) ? kMaxTileValue : static_cast<unsigned>(extent);
    }
    currentTileSizes = {fullTile, fullTile};
  }
  if (currentTileSizes.empty()) {
    currentTileSizes = {std::max(getOuterTileSize(axis, blockNumber), innerTileSize), innerTileSize};
  }

  SmallVector<unsigned, kSmallVectorSizeFour> usedTileSizes;
  int64_t currentSize = extent;
  for (size_t i = 0; i < currentTileSizes.size(); ++i) {
    auto tileSize = static_cast<int64_t>(currentTileSizes[i]);
    int64_t minRequired = (i == currentTileSizes.size() - 1) ? tileSize * kMinTileSizeMultiplier : tileSize;
    if (currentSize >= minRequired) {
      usedTileSizes.push_back(static_cast<unsigned>(tileSize));
      currentSize = tileSize;
    } else {
      usedTileSizes.push_back(static_cast<unsigned>(currentSize));
    }
  }
  return usedTileSizes;
}

static void applyTileConfigsToAxis(const AxisPtr axis, const SmallVector<unsigned, kSmallVectorSizeFour> &usedTileSizes,
                                   size_t &maxLevelToTile) {
  size_t numLevels = usedTileSizes.size();
  size_t currentTileLevel = axis->configs[kTileCfg].size();
  if (currentTileLevel > numLevels) {
    auto &cfgs = axis->configs[kTileCfg];
    cfgs.resize(numLevels);
    currentTileLevel = numLevels;
  }

  size_t levelsToAdd = (numLevels > currentTileLevel) ? (numLevels - currentTileLevel) : 0;
  for (size_t level = 0; level < levelsToAdd; ++level) {
    axis->doExtraTile();
  }

  maxLevelToTile = std::max(maxLevelToTile, numLevels);
  for (size_t level = 0; level < numLevels; ++level) {
    auto tileConfig = axis->tryGetConfig(static_cast<int>(level), kTileCfg);
    if (tileConfig != nullptr) {
      tileConfig->value = static_cast<int>(usedTileSizes[level]);
      axis->tryAddConstraint(static_cast<int>(level), Constraint({static_cast<int>(usedTileSizes[level])}));
    }
  }
}

inline bool isReductionAxis(const AxisPtr &axis) {
  if (!axis) {
    return false;
  }
  if (axis->axisType.count(mlir::autotiling::Axis::AxisLabel::kReduction) > 0) {
    return true;
  }
  Operation *loopOp = axis->getLoopOperation();
  return (loopOp != nullptr) && loopOp->hasAttr(kReductionLoopAttr);
}

inline bool isDynamicAxis(const AxisPtr &axis) {
  return axis && axis->axisType.count(mlir::autotiling::Axis::AxisLabel::kDynamic) > 0;
}

inline bool isTransposeAxis(const AxisPtr &axis) {
  if (!axis || !axis->loop) {
    return false;
  }
  Operation *loopOp = axis->getLoopOperation();
  return (loopOp != nullptr) && loopOp->hasAttr(kTransposeLoopAttr);
}

inline unsigned saturateToTileValue(int64_t value) {
  return static_cast<unsigned>(std::clamp<int64_t>(value, 1, kMaxTileValue));
}

// Complexity score contributed by a single node (for identifying heavy-math bands).
int64_t getNodeMathComplexity(const NodePtr &node) {
  if (!node) {
    return 0;
  }
  if (node->opType == "HeavyElem") {
    return kHeavyMathComplexityScore;
  }
  if (node->opType == "Reduce") {
    return kReduceMathComplexityScore;
  }
  Operation *op = node->op();
  if (op == nullptr) {
    return 0;
  }
  llvm::StringRef name = op->getName().getStringRef();
  if (name.contains("pow") || name.contains("rsqrt") || name.contains("exp") || name.contains("log") ||
      name.contains("tanh")) {
    return kHeavyMathComplexityScore;
  }
  if (name.contains("sqrt") || name.contains("div")) {
    return kMediumMathComplexityScore;
  }
  return 0;
}

struct UbCapacity {
  int64_t rawUbElems{1};
  int64_t ubCapacityElems{1};
  int64_t smallestTypeBits{kDefaultTypeBits};
};

int64_t alignUbElems(int64_t elems, int64_t typeBits) {
  int64_t safeTypeBits = std::max<int64_t>(typeBits, 1);
  int64_t bits = std::max<int64_t>(elems, 1) * safeTypeBits;
  return std::max<int64_t>((bits / kUbAlignBits) * kUbAlignBits / safeTypeBits, 1);
}

UbCapacity computeUbCapacity(const NpuModelGraphPtr &npuGraph) {
  UbCapacity c;
  if (!npuGraph) {
    return c;
  }
  int64_t ubBytes = (npuGraph->ubSize > 0) ? npuGraph->ubSize : kNpuFallbackUbSizeInBytes;
  ubBytes = std::max<int64_t>(ubBytes - kUbGuardReserveBytes, kUbAlignBytes);
  c.smallestTypeBits = std::max<int64_t>(static_cast<int64_t>(npuGraph->smallestTypeBits), 1);
  int64_t maxBufferCnt = std::max<int64_t>(npuGraph->maxBufferCnt, 1);
  int64_t rawElems = ubBytes * kBitsPerByte / c.smallestTypeBits;
  c.rawUbElems = alignUbElems(rawElems, c.smallestTypeBits);
  c.ubCapacityElems = alignUbElems(rawElems / maxBufferCnt, c.smallestTypeBits);
  return c;
}

inline int64_t ubCapacityForPointTile(const NpuBandContext &ctx) { return std::max<int64_t>(ctx.ubCapacityElems, 1); }

inline int64_t ubCapacityForVectorTile(const NpuBandContext &ctx) {
  return std::max<int64_t>(ctx.vectorUbCapacityElems, 1);
}

std::map<size_t, SmallVector<AxisPtr, kSmallVectorSizeFour>> groupAxesByBand(const SmallVector<AxisPtr> &axes) {
  std::map<size_t, SmallVector<AxisPtr, kSmallVectorSizeFour>> grouped;
  for (const auto &axis : axes) {
    if (axis) {
      grouped[axis->bandIdx].push_back(axis);
    }
  }
  for (auto &entry : grouped) {
    auto &bandAxes = entry.second;
    std::sort(bandAxes.begin(), bandAxes.end(),
              [](const AxisPtr &lhs, const AxisPtr &rhs) { return lhs->axisIdx < rhs->axisIdx; });
  }
  return grouped;
}

struct BandNodeFacts {
  int64_t mathComplexityScore{0}, maxVectorLoadBits{0};
  bool hasF32VectorLoad{false};
};
struct VectorUbFootprint {
  int64_t peakBitsPerElem{kDefaultTypeBits}, vectorElementBits{kDefaultTypeBits};
  int64_t fixedReserveBytes{0};
  SmallVector<SmallVector<TransposeCoLiveReserve, kSmallVectorSizeFour>, kSmallVectorSizeFour>
    transposeCoLiveReserveSets;
  SmallVector<VectorPeakReserveSet, kSmallVectorSizeThirtyTwo> vectorPeakReserveSets;
};
struct VectorUbLiveState {
  llvm::DenseMap<Value, int64_t> remainingUses;
  llvm::DenseMap<Value, bool> multiBufferedValues;
  llvm::DenseMap<Value, int64_t> liveBits;
  llvm::DenseMap<Value, SmallVector<size_t, kSmallVectorSizeSix>> liveAxisOrders;
  llvm::DenseMap<Value, SmallVector<int32_t, kSmallVectorSizeSix>> liveAlignDims;
  int64_t liveBitsSum{0};
  int64_t liveMultiBufferExtraBitsSum{0};
};
struct BandNodeInfo {
  BandNodeFacts facts;
  SmallVector<NodePtr, kSmallVectorSizeThirtyTwo> nodes;
  SmallVector<TransposeVectorInfo, kSmallVectorSizeSix> transposeInfos;
};
enum class VectorOpKind { None, Generic, Heavy, Compare, Select };
int64_t getValueElementBits(Value value);
inline bool isNodeInBand(const NodePtr &node, size_t bandIdx) {
  return node && !node->loopNest_.empty() && node->loopNest_.front() && node->loopNest_.front()->bandIdx == bandIdx;
}
std::optional<size_t> findAxisIndex(ArrayRef<AxisPtr> axes, const AxisPtr &target) {
  if (!target || !target->loop) {
    return std::nullopt;
  }
  Operation *targetLoop = target->getLoopOperation();
  for (size_t i = 0; i < axes.size(); ++i) {
    if (axes[i] && axes[i]->getLoopOperation() == targetLoop) {
      return i;
    }
  }
  return std::nullopt;
}
SmallVector<size_t, kSmallVectorSizeSix> getNodeAxisOrder(const NodePtr &node, ArrayRef<AxisPtr> axes) {
  SmallVector<size_t, kSmallVectorSizeSix> order;
  if (!node) {
    return order;
  }
  for (const AxisPtr &axis : node->loopNest_) {
    if (std::optional<size_t> idx = findAxisIndex(axes, axis)) {
      order.push_back(*idx);
    }
  }
  return order;
}
SmallVector<size_t, kSmallVectorSizeSix> intersectAxisOrder(ArrayRef<size_t> lhs, ArrayRef<size_t> rhs) {
  SmallVector<size_t, kSmallVectorSizeSix> order;
  for (size_t idx : lhs) {
    if (llvm::is_contained(rhs, idx)) {
      order.push_back(idx);
    }
  }
  return order;
}
bool isSameTransposeInfo(const TransposeVectorInfo &lhs, const TransposeVectorInfo &rhs) {
  return lhs.sourceAxisOrder == rhs.sourceAxisOrder && lhs.targetAxisOrder == rhs.targetAxisOrder &&
         lhs.elementBits == rhs.elementBits && lhs.isF32 == rhs.isF32;
}
SmallVector<size_t, kSmallVectorSizeSix> getMemrefAxisOrder(ValueRange indices, const NpuBandContext &ctx) {
  SmallVector<size_t, kSmallVectorSizeSix> order;
  Operation *leafParent = !ctx.axes.empty() && ctx.axes.back() && ctx.axes.back()->loop
                            ? ctx.axes.back()->getLoopOperation()->getParentOp()
                            : nullptr;
  for (Value idx : indices) {
    Value strippedIdx = idx;
    while (true) {
      if (auto castOp = strippedIdx.getDefiningOp<arith::IndexCastOp>()) {
        strippedIdx = castOp.getIn();
        continue;
      }
      if (auto castUIOp = strippedIdx.getDefiningOp<arith::IndexCastUIOp>()) {
        strippedIdx = castUIOp.getIn();
        continue;
      }
      break;
    }
    bool matched = false;
    for (size_t i = 0; i < ctx.axes.size(); ++i) {
      if (ctx.axes[i] && strippedIdx == ctx.axes[i]->getInductionVar()) {
        order.push_back(i);
        matched = true;
        break;
      }
    }
    if (matched) {
      continue;
    }
    if (auto arg = dyn_cast<BlockArgument>(strippedIdx)) {
      if (auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
          (leafParent != nullptr) && loop && loop->getParentOp() == leafParent) {
        order.push_back(ctx.axes.size() - 1);
      }
    }
  }
  return order;
}
// Walk the SSA use-def graph backwards from `value` and check whether `source`
// is reachable. In addition to the plain `defOp->getOperands()` chain this
// also descends into the region(s) of region-bearing ops (scf.for, scf.if,
// affine.if, scf.while, linalg.*, ...) by inspecting each block terminator's
// operands -- region ops surface their region results through `scf.yield` /
// `affine.yield` etc., so without this descent any Load living inside a
// reduce / if body would be invisible to the consumer-side Store and the
// corresponding transpose pair would be silently dropped (axes then miss
// their 32B size / 64B f32 double-align in chooseAlignedTransposeTiles).
// `visited` deduplicates the SSA DAG so cost is O(#unique values reachable),
// not exponential in the number of multi-use chains.
bool isValueDerivedFromImpl(Value value, Value source, llvm::SmallPtrSetImpl<Value> &visited) {
  if (value == source) {
    return true;
  }
  if (!visited.insert(value).second) {
    return false;
  }
  Operation *defOp = value.getDefiningOp();
  if (defOp == nullptr) {
    return false;
  }
  if (llvm::any_of(defOp->getOperands(),
                   [&source, &visited](Value operand) { return isValueDerivedFromImpl(operand, source, visited); })) {
    return true;
  }
  for (Region &region : defOp->getRegions()) {
    for (Block &block : region) {
      Operation *terminator = block.getTerminator();
      if ((terminator != nullptr) && llvm::any_of(terminator->getOperands(), [&source, &visited](Value v) {
            return isValueDerivedFromImpl(v, source, visited);
          })) {
        return true;
      }
    }
  }
  return false;
}
bool isValueDerivedFrom(Value value, Value source) {
  llvm::SmallPtrSet<Value, kSmallVectorSizeSixteen> visited;
  return isValueDerivedFromImpl(value, source, visited);
}
void pushUniqueTransposeInfo(SmallVectorImpl<TransposeVectorInfo> &infos, TransposeVectorInfo info) {
  if (llvm::any_of(infos,
                   [&info](const TransposeVectorInfo &existing) { return isSameTransposeInfo(existing, info); })) {
    return;
  }
  infos.push_back(std::move(info));
}
SmallVector<TransposeVectorInfo, kSmallVectorSizeSix> findTransposeVectorInfos(const NpuBandContext &ctx) {
  SmallVector<TransposeVectorInfo, kSmallVectorSizeSix> infos;
  const bool hasTaggedTransposeAxis = llvm::any_of(ctx.axes, [](const AxisPtr &axis) { return isTransposeAxis(axis); });
  auto tryAddInfo = [&hasTaggedTransposeAxis, &ctx, &infos](Operation *loadOp, ArrayRef<size_t> loadOrder,
                                                            Operation *storeOp, ArrayRef<size_t> storeOrder) {
    if (!isValueDerivedFrom(storeOp->getOperand(0), loadOp->getResult(0))) {
      return;
    }
    SmallVector<size_t, kSmallVectorSizeSix> loadCommon = intersectAxisOrder(loadOrder, storeOrder);
    SmallVector<size_t, kSmallVectorSizeSix> storeCommon = intersectAxisOrder(storeOrder, loadOrder);
    if (loadCommon.size() < kMinTransposeAxisOrderSize || loadCommon.size() != storeCommon.size() ||
        loadCommon == storeCommon || !std::is_permutation(loadCommon.begin(), loadCommon.end(), storeCommon.begin()) ||
        (hasTaggedTransposeAxis &&
         !llvm::any_of(loadCommon, [&ctx](size_t idx) { return isTransposeAxis(ctx.axes[idx]); }))) {
      return;
    }
    pushUniqueTransposeInfo(infos,
                            TransposeVectorInfo{loadCommon, storeCommon, getValueElementBits(loadOp->getResult(0)),
                                                loadOp->getResult(0).getType().isF32()});
  };
  Operation *scanRoot = nullptr;
  auto transposeAxisIt = llvm::find_if(ctx.axes, [](const AxisPtr &axis) { return isTransposeAxis(axis); });
  if (transposeAxisIt != ctx.axes.end()) {
    scanRoot = (*transposeAxisIt)->getLoopOperation();
  } else if (!ctx.axes.empty() && ctx.axes.front()) {
    scanRoot = ctx.axes.front()->getLoopOperation();
  }
  SmallVector<memref::LoadOp, kSmallVectorSizeEight> loads;
  SmallVector<memref::StoreOp, kSmallVectorSizeEight> stores;
  if (scanRoot != nullptr) {
    scanRoot->walk([&loads, &stores](Operation *op) {
      if (auto load = dyn_cast<memref::LoadOp>(op)) {
        loads.push_back(load);
      } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
        stores.push_back(store);
      }
    });
  }
  for (memref::LoadOp load : loads) {
    SmallVector<size_t, kSmallVectorSizeSix> loadOrder = getMemrefAxisOrder(load.getIndices(), ctx);
    for (memref::StoreOp store : stores) {
      tryAddInfo(load, loadOrder, store, getMemrefAxisOrder(store.getIndices(), ctx));
    }
  }
  std::stable_sort(infos.begin(), infos.end(), [](const TransposeVectorInfo &lhs, const TransposeVectorInfo &rhs) {
    if (lhs.sourceAxisOrder.size() != rhs.sourceAxisOrder.size()) {
      return lhs.sourceAxisOrder.size() > rhs.sourceAxisOrder.size();
    }
    if (lhs.elementBits != rhs.elementBits) {
      return lhs.elementBits > rhs.elementBits;
    }
    return std::lexicographical_compare(lhs.sourceAxisOrder.begin(), lhs.sourceAxisOrder.end(),
                                        rhs.sourceAxisOrder.begin(), rhs.sourceAxisOrder.end());
  });
  return infos;
}
int64_t getValueElementBits(Value value) {
  if (!value) {
    return 0;
  }
  Type type = value.getType();
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    type = shapedType.getElementType();
  }
  if (!type.isIntOrFloat()) {
    return 0;
  }
  return static_cast<int64_t>(type.getIntOrFloatBitWidth());
}
int64_t getValueUbBitsPerElem(Value value) {
  int64_t elementBits = getValueElementBits(value);
  return elementBits >= kBitsPerByte ? elementBits : 0;
}
bool hasTrackedVectorOperand(Operation *op, const llvm::DenseMap<Value, int64_t> &liveBits) {
  if (op == nullptr) {
    return false;
  }
  for (Value operand : op->getOperands()) {
    if (liveBits.find(operand) != liveBits.end()) {
      return true;
    }
  }
  return false;
}
bool inheritsMultiBufferFromOperand(Operation *op, VectorOpKind kind, const VectorUbLiveState &state) {
  if ((op == nullptr) || (kind != VectorOpKind::Generic && kind != VectorOpKind::Heavy)) {
    return false;
  }
  Value vectorOperand;
  for (Value operand : op->getOperands()) {
    auto liveIt = state.liveBits.find(operand);
    if (liveIt == state.liveBits.end() || liveIt->second < kBitsPerByte) {
      continue;
    }
    if (vectorOperand) {
      return false;
    }
    vectorOperand = operand;
  }
  return vectorOperand && state.multiBufferedValues.lookup(vectorOperand);
}
VectorOpKind classifyVectorOp(Operation *op) {
  if (op == nullptr) {
    return VectorOpKind::None;
  }
  llvm::StringRef name = op->getName().getStringRef();
  const llvm::StringRef compareOps[] = {"arith.cmpf", "arith.cmpi"};
  const llvm::StringRef heavyPatterns[] = {"arith.div", "math.exp", "math.log", "math.erf"};
  const llvm::StringRef genericPatterns[] = {"arith.add", "arith.sub", "arith.mul", "arith.max",
                                             "arith.min", "arith.and", "arith.or",  "arith.xor"};
  if (op->getNumResults() == 1 && llvm::is_contained(compareOps, name)) {
    return VectorOpKind::Compare;
  }
  if (name == "arith.select") {
    return VectorOpKind::Select;
  }
  if (op->getNumResults() == 0) {
    return VectorOpKind::None;
  }
  if (llvm::any_of(heavyPatterns, [&name](llvm::StringRef pattern) { return name.contains(pattern); })) {
    return VectorOpKind::Heavy;
  }
  if (llvm::any_of(genericPatterns, [&name](llvm::StringRef pattern) { return name.contains(pattern); }) ||
      name.starts_with("math.")) {
    return VectorOpKind::Generic;
  }
  return VectorOpKind::None;
}

// Whitelist of arith/math/vector ops that bishengir lowers as "structured" UB-resident
// vector ops, i.e. their UB buffers participate in stride-align padding driven by
// MarkStrideAlign. Keep this list in sync with the lowerable elementwise/math
// surface in `bishengir/lib/Dialect/HIVM/Transforms/AlignBuffer/MarkStrideAlign.cpp`
// and `Conversion/ArithToHIVM`. Missing an entry here silently disables stride-align
// merging for that op and can re-introduce the very alignment bug this analysis
// is intended to prevent.
// Note: `arith.constant` is intentionally included because vectorized splat
// constants flow through the same UB stride-align path as elementwise results.
bool isBishengStructuredVectorLoweringOp(Operation *op) {
  if ((op == nullptr) || op->getNumResults() == 0) {
    return false;
  }
  static const llvm::StringSet<> kStructuredOps = {"arith.addf",
                                                   "arith.addi",
                                                   "arith.mulf",
                                                   "arith.muli",
                                                   "arith.subf",
                                                   "arith.subi",
                                                   "arith.divf",
                                                   "arith.divsi",
                                                   "arith.divui",
                                                   "arith.maxsi",
                                                   "arith.maxui",
                                                   "arith.minsi",
                                                   "arith.minui",
                                                   "arith.andi",
                                                   "arith.ori",
                                                   "arith.xori",
                                                   "arith.remf",
                                                   "arith.remsi",
                                                   "arith.remui",
                                                   "arith.minnumf",
                                                   "arith.minimumf",
                                                   "arith.maxnumf",
                                                   "arith.maximumf",
                                                   "arith.shli",
                                                   "arith.shrsi",
                                                   "arith.shrui",
                                                   "arith.extf",
                                                   "arith.fptosi",
                                                   "arith.fptoui",
                                                   "arith.sitofp",
                                                   "arith.uitofp",
                                                   "arith.extsi",
                                                   "arith.extui",
                                                   "arith.trunci",
                                                   "arith.truncf",
                                                   "arith.cmpf",
                                                   "arith.cmpi",
                                                   "arith.select",
                                                   "arith.constant",
                                                   "arith.negf",
                                                   "math.log",
                                                   "math.absf",
                                                   "math.sqrt",
                                                   "math.rsqrt",
                                                   "math.tanh",
                                                   "math.sin",
                                                   "math.cos",
                                                   "math.erf",
                                                   "math.ceil",
                                                   "math.floor",
                                                   "math.absi",
                                                   "math.exp",
                                                   "vector.reduction",
                                                   "vector.broadcast",
                                                   "arith.mulsi_extended",
                                                   "arith.mului_extended"};
  return kStructuredOps.contains(op->getName().getStringRef());
}

bool isBishengStrideAlignedCandidate(const NodePtr &node, Operation *op) {
  if (!node || (op == nullptr)) {
    return false;
  }
  if (node->opType == "Load") {
    return isa<memref::LoadOp>(op);
  }
  return isBishengStructuredVectorLoweringOp(op);
}
SmallVector<int32_t, kSmallVectorSizeSix> getBishengStrideAlignDims(const NpuBandContext &ctx, const NodePtr &node,
                                                                    Operation *op, ArrayRef<size_t> axisOrder) {
  if (!isBishengStrideAlignedCandidate(node, op)) {
    return {};
  }
  // Mirror BiShengIR MarkStrideAlign.cpp::getLastUnContinuousDim for the
  // ordinary structured/load path: when the flattened operands have unit
  // innermost stride, the marked dim is the last non-continuous dim before it.
  // EnableStrideAlign's collectAlignUnits applies that mark to dim+1, i.e. the
  // logical innermost vector dimension for the shape seen by akg-opt.
  SmallVector<int32_t, kSmallVectorSizeSix> alignDims =
    getBishengLogicalStructuredStrideAlignDims(static_cast<int64_t>(axisOrder.size()));
  // Drop align dims whose entire prefix [0..dim] is degenerate (extent==1).
  // bishengir's MarkStrideAlign / flatten step collapses unit-extent prefixes
  // away; pretending they need stride padding inflates the live footprint and
  // produces tiles that bishengir would just flatten back. An out-of-range
  // axis index also falls into this branch as a defensive guard.
  llvm::erase_if(alignDims, [&axisOrder, &ctx](int32_t dim) {
    if (dim < 0 || static_cast<size_t>(dim) >= axisOrder.size()) {
      return true;
    }
    for (size_t i = 0; i <= static_cast<size_t>(dim); ++i) {
      size_t axisIdx = axisOrder[i];
      if (axisIdx >= ctx.extents.size() || ctx.extents[axisIdx] != 1) {
        return false;
      }
    }
    return true;
  });
  return alignDims;
}
// `alignDims` mirrors BiShengIR's raw stride-align dims. collectAlignUnits places
// the actual storage expansion on the inner sub-shape dim, so schedule hard-align
// uses rawDim+1 when that dimension exists in the vector shape.
void mergeStrideAlignUnits(NpuBandContext &ctx, ArrayRef<size_t> axisOrder, ArrayRef<int32_t> alignDims,
                           int64_t elementBits) {
  if (axisOrder.empty() || alignDims.empty() || elementBits < kBitsPerByte) {
    return;
  }
  int64_t unit = getBishengStrideAlignTargetForBits(elementBits);
  if (unit <= 1) {
    return;
  }
  for (int32_t alignDim : alignDims) {
    int64_t expandedDim = static_cast<int64_t>(alignDim) + 1;
    if (expandedDim < 0 || static_cast<size_t>(expandedDim) >= axisOrder.size()) {
      continue;
    }
    size_t axisIdx = axisOrder[static_cast<size_t>(expandedDim)];
    if (axisIdx >= ctx.axesAlignUnits.size()) {
      continue;
    }
    ctx.axesAlignUnits[axisIdx] = std::lcm(std::max<int64_t>(ctx.axesAlignUnits[axisIdx], 1), unit);
  }
}
int64_t getScalarVbrcBitsPerElem(Operation *op, VectorOpKind kind, const llvm::DenseMap<Value, int64_t> &liveBits) {
  if (!hasTrackedVectorOperand(op, liveBits)) {
    return 0;
  }
  if (kind != VectorOpKind::Compare && kind != VectorOpKind::Select) {
    return 0;
  }
  int64_t firstScalarOperand = kind == VectorOpKind::Select ? 1 : 0;
  int64_t bits = 0;
  for (int64_t i = firstScalarOperand; i < static_cast<int64_t>(op->getNumOperands()); ++i) {
    Value operand = op->getOperand(static_cast<unsigned>(i));
    if (liveBits.find(operand) == liveBits.end()) {
      bits += getValueUbBitsPerElem(operand);
    }
  }
  return bits;
}
int64_t getOpDataBitsPerElem(Operation *op, const llvm::DenseMap<Value, int64_t> &liveBits, int64_t resultBits,
                             int64_t scalarVbrcBits) {
  int64_t dataBits = std::max(resultBits, scalarVbrcBits);
  for (Value operand : op->getOperands()) {
    auto liveIt = liveBits.find(operand);
    int64_t operandBits = liveIt == liveBits.end() ? getValueUbBitsPerElem(operand) : liveIt->second;
    dataBits = std::max(dataBits, operandBits);
  }
  return dataBits;
}
int64_t getHivmWorkspaceBitsPerElem(VectorOpKind kind, int64_t dataBits) {
  if (dataBits <= 0) {
    return 0;
  }
  if (kind == VectorOpKind::Heavy) {
    return kHeavyHivmWorkspaceBuffers * dataBits;
  }
  if (kind == VectorOpKind::Compare) {
    return kCompareHivmWorkspaceBuffers * dataBits;
  }
  return kind == VectorOpKind::None ? 0 : kGenericHivmWorkspaceBuffers * dataBits;
}
int64_t getHivmWorkspaceBufferCount(VectorOpKind kind) {
  if (kind == VectorOpKind::Heavy) {
    return kHeavyHivmWorkspaceBuffers;
  }
  if (kind == VectorOpKind::Compare) {
    return kCompareHivmWorkspaceBuffers;
  }
  return kind == VectorOpKind::None ? 0 : kGenericHivmWorkspaceBuffers;
}
void appendVectorReserve(VectorPeakReserveSet &set, ArrayRef<size_t> axisOrder, ArrayRef<int32_t> alignDims,
                         int64_t elementBits) {
  if (axisOrder.empty() || elementBits < kBitsPerByte) {
    return;
  }
  VectorLiveReserve reserve;
  reserve.axisOrder.assign(axisOrder.begin(), axisOrder.end());
  reserve.alignDims.assign(alignDims.begin(), alignDims.end());
  reserve.elementBits = elementBits;
  set.liveBuffers.push_back(std::move(reserve));
}
int64_t getSelectTempBytes(VectorOpKind kind, bool hasVectorInput, int64_t resultBits) {
  return (kind == VectorOpKind::Select && hasVectorInput && resultBits >= kBitsPerByte) ? kUbAlignBytes : 0;
}
int64_t getNodeResultUbBitsPerElem(const NodePtr &node, Operation *op, VectorOpKind kind, bool hasVectorInput) {
  if (op->getNumResults() == 0) {
    return 0;
  }
  if (kind == VectorOpKind::Compare) {
    return 0;
  }
  if (node->opType != "Load" && !hasVectorInput) {
    return 0;
  }
  int64_t bits = 0;
  for (Value result : op->getResults()) {
    bits = std::max(bits, getValueUbBitsPerElem(result));
  }
  return bits;
}
bool hasBroadcastVectorAxis(const NpuBandContext &ctx) {
  for (const auto &axis : ctx.axes) {
    Operation *loopOp = axis ? axis->getLoopOperation() : nullptr;
    if ((loopOp != nullptr) &&
        (loopOp->hasAttr(kBroadcastLoopAttr) || loopOp->hasAttr(kNotInnerDimensionBroadcastLoopAttr))) {
      return true;
    }
  }
  return false;
}
void recordVectorUbPeakState(VectorUbFootprint &footprint, const NpuBandContext &ctx, const VectorUbLiveState &state,
                             const NodePtr &node, ArrayRef<size_t> nodeAxisOrder, int64_t scalarVbrcBits,
                             int64_t resultBits, int64_t resultBufferCopies, int64_t opDataBits, int64_t workspaceBits,
                             VectorOpKind kind, bool hasVectorInput) {
  footprint.peakBitsPerElem =
    std::max(footprint.peakBitsPerElem, state.liveBitsSum + state.liveMultiBufferExtraBitsSum + scalarVbrcBits +
                                          resultBits * resultBufferCopies + workspaceBits);

  VectorPeakReserveSet set;
  set.fixedBytes = getSelectTempBytes(kind, hasVectorInput, resultBits);
  Operation *op = node ? node->op() : nullptr;
  SmallVector<int32_t, kSmallVectorSizeSix> nodeAlignDims = getBishengStrideAlignDims(ctx, node, op, nodeAxisOrder);
  for (const auto &entry : state.liveBits) {
    auto axisIt = state.liveAxisOrders.find(entry.first);
    if (axisIt == state.liveAxisOrders.end()) {
      continue;
    }
    SmallVector<int32_t, kSmallVectorSizeSix> alignDims = state.liveAlignDims.lookup(entry.first);
    appendVectorReserve(set, axisIt->second, alignDims, entry.second);
    if (state.multiBufferedValues.lookup(entry.first)) {
      appendVectorReserve(set, axisIt->second, alignDims, entry.second);
    }
  }
  appendVectorReserve(set, nodeAxisOrder, nodeAlignDims, scalarVbrcBits);
  for (int64_t i = 0; i < resultBufferCopies; ++i) {
    appendVectorReserve(set, nodeAxisOrder, nodeAlignDims, resultBits);
  }
  for (int64_t i = 0, e = getHivmWorkspaceBufferCount(kind); i < e; ++i) {
    appendVectorReserve(set, nodeAxisOrder, nodeAlignDims, opDataBits);
  }
  if (!set.liveBuffers.empty() || set.fixedBytes > 0) {
    footprint.fixedReserveBytes = std::max(footprint.fixedReserveBytes, set.fixedBytes);
    footprint.vectorPeakReserveSets.push_back(std::move(set));
  }

  if (node->opType != "Load") {
    return;
  }
  SmallVector<size_t, kSmallVectorSizeSix> currentOrder = getNodeAxisOrder(node, ctx.axes);
  bool isKnownTransposeLoad = llvm::any_of(ctx.transposeInfos, [&currentOrder](const TransposeVectorInfo &info) {
    return currentOrder == info.sourceAxisOrder;
  });
  if (!isKnownTransposeLoad && (ctx.transposeSourceAxisOrder.empty() || currentOrder != ctx.transposeSourceAxisOrder)) {
    return;
  }
  SmallVector<TransposeCoLiveReserve, kSmallVectorSizeFour> liveSet;
  for (const auto &entry : state.liveBits) {
    if (entry.second < kBitsPerByte) {
      continue;
    }
    auto axisIt = state.liveAxisOrders.find(entry.first);
    if (axisIt == state.liveAxisOrders.end() || axisIt->second.empty()) {
      continue;
    }
    liveSet.push_back(TransposeCoLiveReserve{axisIt->second, state.liveAlignDims.lookup(entry.first), entry.second});
  }
  if (!liveSet.empty()) {
    footprint.transposeCoLiveReserveSets.push_back(std::move(liveSet));
  }
}

void updateVectorUbLiveState(VectorUbFootprint &footprint, const NpuBandContext &ctx, VectorUbLiveState &state,
                             const NodePtr &node, Operation *op, ArrayRef<size_t> nodeAxisOrder, int64_t resultBits,
                             VectorOpKind kind, bool hasVectorInput, bool inheritsMultiBuffer) {
  bool addLoadResult = node->opType == "Load";
  bool addVectorResult = node->opType != "Store" && hasVectorInput;
  if (addLoadResult || addVectorResult) {
    unsigned resultCount = addLoadResult ? std::min<unsigned>(op->getNumResults(), 1) : op->getNumResults();
    for (unsigned i = 0; i < resultCount; ++i) {
      Value result = op->getResult(i);
      int64_t bits = addLoadResult ? resultBits : (kind == VectorOpKind::Compare ? 0 : getValueUbBitsPerElem(result));
      auto liveIt = state.liveBits.find(result);
      if (liveIt != state.liveBits.end()) {
        state.liveBitsSum -= liveIt->second;
        state.liveMultiBufferExtraBitsSum -=
          static_cast<int64_t>(state.multiBufferedValues.lookup(result)) * liveIt->second;
      }
      if (inheritsMultiBuffer && bits >= kBitsPerByte) {
        state.multiBufferedValues[result] = true;
      }
      state.liveBits[result] = bits;
      state.liveAxisOrders[result] =
        SmallVector<size_t, kSmallVectorSizeSix>(nodeAxisOrder.begin(), nodeAxisOrder.end());
      state.liveAlignDims[result] = getBishengStrideAlignDims(ctx, node, op, nodeAxisOrder);
      state.liveBitsSum += bits;
      state.liveMultiBufferExtraBitsSum += static_cast<int64_t>(state.multiBufferedValues.lookup(result)) * bits;
      footprint.peakBitsPerElem =
        std::max(footprint.peakBitsPerElem, state.liveBitsSum + state.liveMultiBufferExtraBitsSum);
    }
  }

  SmallVector<Value, kSmallVectorSizeEight> valuesToErase;
  for (Value operand : op->getOperands()) {
    auto useIt = state.remainingUses.find(operand);
    if (useIt == state.remainingUses.end()) {
      continue;
    }
    if (--useIt->second != 0) {
      continue;
    }
    state.remainingUses.erase(useIt);
    valuesToErase.push_back(operand);
  }
  for (Value result : op->getResults()) {
    if (state.remainingUses.lookup(result) == 0) {
      valuesToErase.push_back(result);
    }
  }
  for (Value value : valuesToErase) {
    auto liveIt = state.liveBits.find(value);
    if (liveIt == state.liveBits.end()) {
      continue;
    }
    state.liveBitsSum -= liveIt->second;
    state.liveMultiBufferExtraBitsSum -= static_cast<int64_t>(state.multiBufferedValues.count(value)) * liveIt->second;
    state.liveBits.erase(liveIt);
    state.liveAxisOrders.erase(value);
    state.liveAlignDims.erase(value);
  }
}

VectorUbFootprint computeVectorUbFootprint(ArrayRef<NodePtr> bandNodes, const NpuBandContext &ctx) {
  VectorUbFootprint footprint;
  footprint.vectorElementBits = std::max<int64_t>(ctx.smallestTypeBits, 1);
  if (bandNodes.empty()) {
    return footprint;
  }

  VectorUbLiveState state;
  for (const auto &node : bandNodes) {
    Operation *op = node->op();
    for (Value result : op->getResults()) {
      footprint.vectorElementBits = std::max(footprint.vectorElementBits, getValueElementBits(result));
    }
    for (Value operand : op->getOperands()) {
      footprint.vectorElementBits = std::max(footprint.vectorElementBits, getValueElementBits(operand));
      ++state.remainingUses[operand];
    }
    if (node->opType == "Load" && op->getNumResults() > 0) {
      state.multiBufferedValues[op->getResult(0)] = true;
    }
    if (node->opType == "Store" && op->getNumOperands() > 0) {
      state.multiBufferedValues[op->getOperand(0)] = true;
    }
  }

  for (const auto &node : bandNodes) {
    Operation *op = node->op();
    VectorOpKind kind = classifyVectorOp(op);
    bool hasVectorInput = hasTrackedVectorOperand(op, state.liveBits);
    int64_t resultBits = getNodeResultUbBitsPerElem(node, op, kind, hasVectorInput);
    int64_t scalarVbrcBits = getScalarVbrcBitsPerElem(op, kind, state.liveBits);
    int64_t opDataBits = getOpDataBitsPerElem(op, state.liveBits, resultBits, scalarVbrcBits);
    int64_t workspaceBits = getHivmWorkspaceBitsPerElem(kind, opDataBits);
    SmallVector<size_t, kSmallVectorSizeSix> nodeAxisOrder = getNodeAxisOrder(node, ctx.axes);
    bool inheritsMultiBuffer = resultBits >= kBitsPerByte && inheritsMultiBufferFromOperand(op, kind, state);
    int64_t resultBufferCopies =
      (inheritsMultiBuffer ||
       llvm::any_of(op->getResults(), [&state](Value result) { return state.multiBufferedValues.lookup(result); }))
        ? kHivmAutoMultiBufferFactor
        : 1;
    recordVectorUbPeakState(footprint, ctx, state, node, nodeAxisOrder, scalarVbrcBits, resultBits, resultBufferCopies,
                            opDataBits, workspaceBits, kind, hasVectorInput);
    updateVectorUbLiveState(footprint, ctx, state, node, op, nodeAxisOrder, resultBits, kind, hasVectorInput,
                            inheritsMultiBuffer);
  }
  if (ctx.graphTemplate == GraphTemplate::TRANSPOSE_OP ||
      llvm::any_of(ctx.axes, [](const AxisPtr &axis) { return isTransposeAxis(axis); })) {
    footprint.peakBitsPerElem =
      std::max(footprint.peakBitsPerElem, kTransposeOrBroadcastFootprintBuffers * footprint.vectorElementBits);
  } else if (ctx.graphTemplate == GraphTemplate::BROADCAST_OP || hasBroadcastVectorAxis(ctx)) {
    footprint.peakBitsPerElem =
      std::max(footprint.peakBitsPerElem, kTransposeOrBroadcastFootprintBuffers * footprint.vectorElementBits);
  }
  return footprint;
}

BandNodeInfo collectBandNodeInfo(const NpuModelGraphPtr &npuGraph, const NpuBandContext &ctx) {
  BandNodeInfo info;
  if (!npuGraph) {
    return info;
  }

  for (const auto &node : npuGraph->nodes()) {
    if (!isNodeInBand(node, ctx.bandIdx)) {
      continue;
    }
    info.facts.mathComplexityScore =
      std::min<int64_t>(info.facts.mathComplexityScore + getNodeMathComplexity(node), kMathComplexityCap);
    Operation *op = node->op();
    if (op != nullptr) {
      info.nodes.push_back(node);
    }
    if (auto loadOp = dyn_cast_or_null<memref::LoadOp>(op)) {
      int64_t loadBits = getValueElementBits(loadOp.getResult());
      if (loadBits >= kBitsPerByte && loadBits <= kDefaultTypeBits) {
        info.facts.maxVectorLoadBits = std::max(info.facts.maxVectorLoadBits, loadBits);
        info.facts.hasF32VectorLoad |= loadOp.getResult().getType().isF32();
      }
    }
  }
  info.transposeInfos = findTransposeVectorInfos(ctx);
  return info;
}
// Common def-use walker for the BiSheng vector chain inside a band.
// Stride-align consumers share an identical "vector flows from Load down
// through structured ops, stops at Store" propagation pattern. This helper
// centralizes that logic so they can not drift apart.
// `fn` receives `(node, op, hasVectorResult, hasVectorInput, axisOrder)`.
// `vectorValues` is updated AFTER `fn` runs, so a node's hasVectorInput is
// decided before its own results join the set. Propagation is deliberately
// limited to loads and BiSheng stride-aligned candidates; unsupported ops break
// the chain instead of making later structured ops look vector-resident.
template <typename Fn>
void forEachBishengVectorNode(const NpuBandContext &ctx, Fn &&fn) {
  llvm::DenseSet<Value> vectorValues;
  for (const NodePtr &node : ctx.nodes) {
    Operation *op = node ? node->op() : nullptr;
    if (!op) {
      continue;
    }
    bool isLoad = node->opType == "Load";
    bool hasVectorInput =
      llvm::any_of(op->getOperands(), [&vectorValues](Value operand) { return vectorValues.contains(operand); });
    SmallVector<size_t, kSmallVectorSizeSix> axisOrder = getNodeAxisOrder(node, ctx.axes);
    bool hasVectorResult = isLoad || (hasVectorInput && isBishengStrideAlignedCandidate(node, op));
    fn(node, op, hasVectorResult, hasVectorInput, ArrayRef<size_t>(axisOrder));
    if (hasVectorResult) {
      for (Value result : op->getResults()) {
        vectorValues.insert(result);
      }
    }
  }
}

void mergeGenericStrideAlignUnits(NpuBandContext &ctx) {
  if (!ctx.transposeInfos.empty() ||
      llvm::any_of(ctx.axes, [](const AxisPtr &axis) { return isTransposeAxis(axis); })) {
    return;
  }
  // Shared loop tiles follow the widest byte type; narrower buffers are padded by their own reserve/mark path.
  int64_t maxElementBits = 0;
  forEachBishengVectorNode(
    ctx, [&maxElementBits](const NodePtr &, Operation *op, bool hasVectorResult, bool, ArrayRef<size_t>) {
      if (!hasVectorResult) {
        return;
      }
      for (Value result : op->getResults()) {
        maxElementBits = std::max(maxElementBits, getValueElementBits(result));
      }
    });
  if (maxElementBits < kBitsPerByte) {
    return;
  }
  forEachBishengVectorNode(ctx, [&ctx, &maxElementBits](const NodePtr &node, Operation *op, bool hasVectorResult, bool,
                                                        ArrayRef<size_t> axisOrder) {
    if (!hasVectorResult) {
      return;
    }
    mergeStrideAlignUnits(ctx, axisOrder, getBishengStrideAlignDims(ctx, node, op, axisOrder), maxElementBits);
  });
}

NpuBandContext buildNpuBandContext(const NpuModelGraphPtr &npuGraph, size_t bandIdx,
                                   const SmallVector<AxisPtr, kSmallVectorSizeFour> &bandAxes) {
  NpuBandContext ctx;
  ctx.bandIdx = bandIdx;
  ctx.axes = bandAxes;
  if (!npuGraph) {
    return ctx;
  }

  ctx.graphTemplate = npuGraph->graphTemplate;
  ctx.targetBlocks =
    (npuGraph->coreNum > 0) ? static_cast<int64_t>(npuGraph->coreNum) : static_cast<int64_t>(kNpuTargetBlocks);
  UbCapacity ub = computeUbCapacity(npuGraph);
  ctx.rawUbElems = ub.rawUbElems;
  ctx.ubCapacityElems = ub.ubCapacityElems;
  ctx.vectorUbCapacityElems = ub.ubCapacityElems;
  ctx.smallestTypeBits = ub.smallestTypeBits;

  for (const auto &axis : bandAxes) {
    bool hasDynamicExtent = !axis || isDynamicAxis(axis) || !axis->hasConstantBounds();
    ctx.extents.push_back(hasDynamicExtent ? kDynamicUnknownExtentForAlignment
                                           : std::max<int64_t>(axis->range.second, 1));
    ctx.hasDynamicAxis |= isDynamicAxis(axis);
    ctx.hasReduction |= isReductionAxis(axis);
  }
  ctx.lastAxisIsReduction = !bandAxes.empty() && isReductionAxis(bandAxes.back());
  BandNodeInfo nodeInfo = collectBandNodeInfo(npuGraph, ctx);
  const BandNodeFacts &nodeFacts = nodeInfo.facts;
  ctx.mathComplexityScore = nodeFacts.mathComplexityScore;
  ctx.transposeElementBits =
    nodeFacts.maxVectorLoadBits > 0 ? nodeFacts.maxVectorLoadBits : std::max<int64_t>(ctx.smallestTypeBits, 1);
  ctx.transposeElementIsF32 = nodeFacts.hasF32VectorLoad;
  if (!nodeInfo.transposeInfos.empty()) {
    ctx.transposeInfos = nodeInfo.transposeInfos;
    const TransposeVectorInfo &primary = ctx.transposeInfos.front();
    ctx.transposeSourceAxisOrder = primary.sourceAxisOrder;
    ctx.transposeTargetAxisOrder = primary.targetAxisOrder;
    ctx.transposeElementBits = std::max<int64_t>(primary.elementBits, 1);
    ctx.transposeElementIsF32 = primary.isF32;
  }
  ctx.axesAlignUnits.assign(ctx.axes.size(), 1);
  for (size_t i = 0; i < ctx.axes.size(); ++i) {
    ctx.axesAlignUnits[i] = std::max<int64_t>(computeAxisAlignSize(i, ctx), 1);
  }
  // computeVectorUbFootprint only reads ctx.{axes, extents, transposeInfos}, not axesAlignUnits.
  // Run it before moving nodes into ctx to avoid an extra SmallVector copy of NodePtr.
  VectorUbFootprint footprint = computeVectorUbFootprint(nodeInfo.nodes, ctx);
  ctx.nodes = std::move(nodeInfo.nodes);
  mergeGenericStrideAlignUnits(ctx);
  applyF32LastDimDoubleAlign(ctx, ctx.axesAlignUnits);
  ctx.vectorElementBits = std::max<int64_t>(footprint.vectorElementBits, ctx.smallestTypeBits);
  ctx.vectorUbBitsPerElem = std::max<int64_t>(footprint.peakBitsPerElem, ctx.vectorElementBits);
  ctx.transposeCoLiveReserveSets = footprint.transposeCoLiveReserveSets;
  ctx.vectorPeakReserveSets = std::move(footprint.vectorPeakReserveSets);
  ctx.fixedReserveBytes = footprint.fixedReserveBytes;
  int64_t rawUbBits = multiplyAndCap(ctx.rawUbElems, ctx.smallestTypeBits);
  int64_t fixedReserveBits = multiplyAndCap(footprint.fixedReserveBytes, kBitsPerByte);
  int64_t vectorUbBits = rawUbBits > fixedReserveBits ? rawUbBits - fixedReserveBits : kUbAlignBits;
  ctx.vectorUbCapacityElems = alignUbElems(vectorUbBits / ctx.vectorUbBitsPerElem, ctx.vectorElementBits);
  return ctx;
}

int64_t getSliceProduct(ArrayRef<int64_t> extents, size_t from, size_t to) {
  int64_t p = 1;
  for (size_t i = from; i < to && i < extents.size(); ++i) {
    p = multiplyAndCap(p, std::max<int64_t>(extents[i], 1));
  }
  return p;
}

inline bool isSmallSemanticAxis(const NpuBandContext &ctx, int64_t extent) {
  constexpr int64_t kMinSmallSemanticAxisLimit = 8;
  constexpr int64_t kSmallSemanticTargetBlockDivisor = 3;
  return extent > 1 &&
         extent <= std::max<int64_t>(kMinSmallSemanticAxisLimit, ctx.targetBlocks / kSmallSemanticTargetBlockDivisor);
}

size_t computeReductionSuffixStart(const NpuBandContext &ctx) {
  size_t suffixStart = ctx.axes.size() - 1;
  int64_t suffixElems = ctx.extents.back();
  while (suffixStart > 0) {
    int64_t candidate = ctx.extents[suffixStart - 1];
    if (!isSmallSemanticAxis(ctx, candidate)) {
      break;
    }
    if (suffixElems > ubCapacityForPointTile(ctx) / candidate) {
      break;
    }
    --suffixStart;
    suffixElems = multiplyAndCap(suffixElems, candidate);
  }
  return suffixStart;
}

size_t computeBroadcastSuffixStart(const NpuBandContext &ctx, int64_t tileBudget) {
  constexpr size_t kMinStructuredSuffixRank = 2;
  constexpr size_t kMaxStructuredSuffixRank = 3;
  constexpr int64_t kMinWideInnermostExtent = 32;
  constexpr int64_t kWideInnermostTargetBlockDivisor = 2;
  if (ctx.axes.size() <= kMinStructuredSuffixRank) {
    return ctx.axes.size() - 1;
  }

  int64_t wideThresh = std::max<int64_t>(kMinWideInnermostExtent, ctx.targetBlocks / kWideInnermostTargetBlockDivisor);
  int64_t innerExtent = ctx.extents.back();
  int64_t ubPerPoint = tileBudget / kSuffixPreservePointRows;

  size_t bestStart = ctx.axes.size() - 1;
  for (size_t start = ctx.axes.size() - 1; start > 0; --start) {
    size_t candidateStart = start - 1;
    size_t suffixRank = ctx.axes.size() - candidateStart;
    if (suffixRank < kMinStructuredSuffixRank || suffixRank > kMaxStructuredSuffixRank) {
      continue;
    }
    int64_t suffixElems = getSliceProduct(ctx.extents, candidateStart, ctx.extents.size());
    if (suffixElems > ubPerPoint) {
      break;
    }
    if (innerExtent < wideThresh) {
      continue;
    }
    bool hasSmall = false;
    for (size_t i = candidateStart; i + 1 < ctx.extents.size(); ++i) {
      if (isSmallSemanticAxis(ctx, ctx.extents[i])) {
        hasSmall = true;
        break;
      }
    }
    if (!hasSmall) {
      continue;
    }
    bestStart = candidateStart;
  }
  return bestStart;
}

static SmallVector<int64_t, kSmallVectorSizeSixteen> collectAlignAwareTileCandidates(int64_t extent,
                                                                                     int64_t alignUnit) {
  SmallVector<int64_t, kSmallVectorSizeSixteen> candidates;
  extent = std::max<int64_t>(extent, 1);
  alignUnit = std::max<int64_t>(alignUnit, 1);
  if (alignUnit == 1) {
    candidates.push_back(1);
    for (int64_t tile = 1; tile <= extent; ++tile) {
      if (extent % tile == 0) {
        candidates.push_back(tile);
      }
    }
  } else {
    candidates.push_back(1);
    for (int64_t tile = alignUnit; tile <= extent; tile += alignUnit) {
      candidates.push_back(tile);
    }
  }
  llvm::sort(candidates);
  candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
  return candidates;
}

SmallVector<unsigned, kSmallVectorSizeFour> assignPrefixOuterTiles(ArrayRef<int64_t> prefixExtents,
                                                                   ArrayRef<int64_t> prefixAlignUnits,
                                                                   int64_t targetBlocks) {
  constexpr int64_t kMaxSmallPrefixTileCount = 2;
  SmallVector<unsigned, kSmallVectorSizeFour> outerTiles;
  outerTiles.reserve(prefixExtents.size());
  int64_t producedTiles = 1;
  for (size_t i = 0; i < prefixExtents.size(); ++i) {
    int64_t extent = std::max<int64_t>(prefixExtents[i], 1);
    if (extent == 1) {
      outerTiles.push_back(1);
      continue;
    }
    int64_t alignUnit = i < prefixAlignUnits.size() ? std::max<int64_t>(prefixAlignUnits[i], 1) : 1;
    int64_t remaining = ceilDivInt64(targetBlocks, producedTiles);
    SmallVector<int64_t, kSmallVectorSizeSixteen> candidates = collectAlignAwareTileCandidates(extent, alignUnit);
    int64_t balancedTile = std::max<int64_t>(extent / std::max<int64_t>(remaining, 1), 1);
    candidates.push_back(alignUnit > 1 ? alignUpInt64(balancedTile, alignUnit) : balancedTile);

    int64_t bestTile = extent;
    int64_t bestScore = std::numeric_limits<int64_t>::min();
    for (int64_t tile : candidates) {
      int64_t blocks = ceilDivInt64(extent, tile);
      if (i > 0 && extent <= targetBlocks && blocks > kMaxSmallPrefixTileCount) {
        continue;
      }
      int64_t newProduced = multiplyAndCap(producedTiles, blocks);
      int64_t score = 0;
      if (newProduced <= targetBlocks) {
        score = newProduced * kBlockCountScoreWeight + blocks;
      } else {
        score = targetBlocks * kBlockCountScoreWeight - (newProduced - targetBlocks);
      }
      if (newProduced >= remaining) {
        score += kRemainingTargetBonus;
      }
      if (score > bestScore) {
        bestScore = score;
        bestTile = tile;
      }
    }
    outerTiles.push_back(saturateToTileValue(bestTile));
    producedTiles = multiplyAndCap(producedTiles, ceilDivInt64(extent, bestTile));
  }
  return outerTiles;
}

void initWholeBandPlan(const NpuBandContext &ctx, BandTilePlan &plan) {
  plan.outerTiles.clear();
  plan.innerTiles.clear();
  plan.outerTiles.reserve(ctx.extents.size());
  plan.innerTiles.reserve(ctx.extents.size());
  for (int64_t extent : ctx.extents) {
    unsigned tile = saturateToTileValue(extent);
    plan.outerTiles.push_back(tile);
    plan.innerTiles.push_back(tile);
  }
}

int64_t getVectorUbBytes(const NpuBandContext &ctx) {
  int64_t rawBytes = multiplyAndCap(ctx.rawUbElems, ctx.smallestTypeBits) / kBitsPerByte;
  return std::max<int64_t>(rawBytes - ctx.fixedReserveBytes, kUbAlignBytes);
}
int64_t computeReserveBytes(const NpuBandContext &ctx, const VectorLiveReserve &reserve, ArrayRef<int64_t> axisTiles) {
  SmallVector<int64_t, kSmallVectorSizeSix> shape;
  SmallVector<char, kSmallVectorSizeSix> staticDims;
  shape.reserve(reserve.axisOrder.size());
  staticDims.reserve(reserve.axisOrder.size());
  for (size_t axisIdx : reserve.axisOrder) {
    int64_t tile = axisIdx < axisTiles.size() ? axisTiles[axisIdx] : 1;
    int64_t extent = axisIdx < ctx.extents.size() ? ctx.extents[axisIdx] : tile;
    bool axisHasRuntimeExtent = axisIdx >= ctx.axes.size() || !ctx.axes[axisIdx] || isDynamicAxis(ctx.axes[axisIdx]) ||
                                !ctx.axes[axisIdx]->hasConstantBounds();
    shape.push_back(std::max<int64_t>(tile, 1));
    staticDims.push_back(static_cast<char>(!axisHasRuntimeExtent && tile >= extent));
  }
  return computeBishengStrideAlignedStorageBytes(shape, staticDims, reserve.alignDims, reserve.elementBits);
}
int64_t computeLogicalShapeBytes(ArrayRef<int64_t> shape, int64_t elementBits) {
  int64_t elems = 1;
  for (int64_t dim : shape) {
    elems = multiplyAndCap(elems, std::max<int64_t>(dim, 1));
  }
  return alignUpInt64(ceilDivInt64(multiplyAndCap(elems, elementBits), kBitsPerByte), kUbAlignBytes);
}
int64_t computeLogicalReserveBytes(const VectorLiveReserve &reserve, ArrayRef<int64_t> axisTiles) {
  SmallVector<int64_t, kSmallVectorSizeSix> shape;
  shape.reserve(reserve.axisOrder.size());
  for (size_t axisIdx : reserve.axisOrder) {
    int64_t tile = axisIdx < axisTiles.size() ? axisTiles[axisIdx] : 1;
    shape.push_back(std::max<int64_t>(tile, 1));
  }
  return computeLogicalShapeBytes(shape, reserve.elementBits);
}
int64_t computeReductionBackendExtraReserveBytes(const NpuBandContext &ctx, ArrayRef<int64_t> axisTiles) {
  if (ctx.hasDynamicAxis || !ctx.hasReduction || !ctx.lastAxisIsReduction ||
      ctx.graphTemplate == GraphTemplate::TRANSPOSE_OP) {
    return 0;
  }
  SmallVector<size_t, kSmallVectorSizeSix> axisOrder;
  axisOrder.reserve(axisTiles.size());
  for (size_t i = 0; i < axisTiles.size() && i < ctx.axes.size(); ++i) {
    axisOrder.push_back(i);
  }
  if (axisOrder.empty()) {
    return 0;
  }
  VectorLiveReserve reserve{axisOrder, getDefaultBishengStrideAlignDims(static_cast<int64_t>(axisOrder.size())),
                            ctx.vectorElementBits};
  int64_t extraBytes =
    multiplyAndCap(kReductionBackendExtraReserveBuffers, computeReserveBytes(ctx, reserve, axisTiles));
  return (extraBytes > LLONG_MAX - kReductionBackendExtraFixedReserveBytes)
           ? LLONG_MAX
           : extraBytes + kReductionBackendExtraFixedReserveBytes;
}
bool satisfiesStrideAlignWithoutExpansion(const NpuBandContext &ctx, const VectorLiveReserve &reserve,
                                          ArrayRef<int64_t> axisTiles) {
  return computeReserveBytes(ctx, reserve, axisTiles) <= computeLogicalReserveBytes(reserve, axisTiles);
}
int64_t computeVectorPeakReserveBytes(const NpuBandContext &ctx, ArrayRef<int64_t> axisTiles) {
  int64_t peakBytes = 0;
  for (const VectorPeakReserveSet &set : ctx.vectorPeakReserveSets) {
    int64_t bytes = 0;
    for (const VectorLiveReserve &reserve : set.liveBuffers) {
      int64_t reserveBytes = computeReserveBytes(ctx, reserve, axisTiles);
      bytes = (bytes > LLONG_MAX - reserveBytes) ? LLONG_MAX : bytes + reserveBytes;
    }
    peakBytes = std::max(peakBytes, bytes);
  }
  int64_t reductionExtraBytes = computeReductionBackendExtraReserveBytes(ctx, axisTiles);
  return (peakBytes > LLONG_MAX - reductionExtraBytes) ? LLONG_MAX : peakBytes + reductionExtraBytes;
}
int64_t computeAlignedShapeBytes(ArrayRef<int64_t> shape, ArrayRef<int64_t> alignBytes, int64_t elemBytes) {
  int64_t elems = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    int64_t alignUnit = alignBytes[i] > 0 ? std::max<int64_t>(alignBytes[i] / elemBytes, 1) : 1;
    elems = multiplyAndCap(elems, alignUpInt64(std::max<int64_t>(shape[i], 1), alignUnit));
  }
  return alignUpInt64(multiplyAndCap(elems, elemBytes), kUbAlignBytes);
}
bool satisfiesAlignedShapeWithoutExpansion(ArrayRef<int64_t> shape, ArrayRef<int64_t> alignBytes, int64_t elemBytes) {
  return computeAlignedShapeBytes(shape, alignBytes, elemBytes) <=
         computeLogicalShapeBytes(shape, multiplyAndCap(elemBytes, kBitsPerByte));
}
// Bundled parameters for computeLastDimTransposeAlignBytes to stay under readability-function-size_parameters limit.
struct LastDimTransposeAlignParams {
  ArrayRef<int64_t> srcShape;
  size_t dim0;
  size_t dim1;
  int64_t elemBytes;
  bool isF32Transpose;
};
void computeLastDimTransposeAlignBytes(const LastDimTransposeAlignParams &params, SmallVectorImpl<int64_t> &dstShape,
                                       SmallVectorImpl<int64_t> &srcAlign, SmallVectorImpl<int64_t> &dstAlign) {
  const ArrayRef<int64_t> &srcShape = params.srcShape;
  size_t dim0 = params.dim0;
  size_t dim1 = params.dim1;
  int64_t elemBytes = params.elemBytes;
  bool isF32Transpose = params.isF32Transpose;
  dstShape.assign(srcShape.begin(), srcShape.end());
  std::swap(dstShape[dim0], dstShape[dim1]);
  srcAlign.assign(srcShape.size(), 0);
  dstAlign.assign(dstShape.size(), 0);
  if (multiplyAndCap(srcShape[dim0], elemBytes) % kUbAlignBytes != 0) {
    srcAlign[dim0] = kUbAlignBytes;
  }
  if (multiplyAndCap(srcShape[dim1], elemBytes) % kUbAlignBytes != 0) {
    srcAlign[dim1] = kUbAlignBytes;
  }
  if (multiplyAndCap(dstShape[dim0], elemBytes) % kUbAlignBytes != 0) {
    dstAlign[dim0] = kUbAlignBytes;
  }
  if (multiplyAndCap(dstShape[dim1], elemBytes) % kUbAlignBytes != 0) {
    dstAlign[dim1] = kUbAlignBytes;
  }
  bool hasDoubleAlignedDim =
    alignUpInt64(multiplyAndCap(srcShape[dim0], elemBytes), kUbAlignBytes) % (kUbAlignBytes * kDoubleAlignFactor) ==
      0 ||
    alignUpInt64(multiplyAndCap(srcShape[dim1], elemBytes), kUbAlignBytes) % (kUbAlignBytes * kDoubleAlignFactor) == 0;
  if (isF32Transpose && !hasDoubleAlignedDim) {
    srcAlign[dim0] = kUbAlignBytes;
    srcAlign[dim1] = kUbAlignBytes * kDoubleAlignFactor;
    dstAlign[dim0] = kUbAlignBytes * kDoubleAlignFactor;
    dstAlign[dim1] = kUbAlignBytes;
  }
}
int64_t computeLastDimTransposeAllocBytes(ArrayRef<int64_t> srcShape, size_t dim0, size_t dim1, int64_t elemBytes,
                                          bool isF32Transpose) {
  SmallVector<int64_t, kSmallVectorSizeSix> dstShape;
  SmallVector<int64_t, kSmallVectorSizeSix> srcAlign;
  SmallVector<int64_t, kSmallVectorSizeSix> dstAlign;
  computeLastDimTransposeAlignBytes({srcShape, dim0, dim1, elemBytes, isF32Transpose}, dstShape, srcAlign, dstAlign);
  int64_t srcReserve =
    multiplyAndCap(kHivmAutoMultiBufferFactor, computeAlignedShapeBytes(srcShape, srcAlign, elemBytes));
  int64_t dstReserve =
    multiplyAndCap(kHivmAutoMultiBufferFactor, computeAlignedShapeBytes(dstShape, dstAlign, elemBytes));
  return (srcReserve > LLONG_MAX - dstReserve) ? LLONG_MAX : srcReserve + dstReserve;
}
int64_t computeTransposeAllocReserveBytes(ArrayRef<int64_t> srcShape, ArrayRef<size_t> sourceOrder,
                                          ArrayRef<size_t> targetOrder, int64_t elemBytes, bool isF32Transpose) {
  if (srcShape.size() != sourceOrder.size() || sourceOrder.size() != targetOrder.size()) {
    return 0;
  }

  int64_t peakBytes = 0;
  SmallVector<size_t, kSmallVectorSizeSix> transposeDims;
  for (size_t i = 0; i < sourceOrder.size(); ++i) {
    if (sourceOrder[i] != targetOrder[i]) {
      transposeDims.push_back(i);
    }
  }
  if (transposeDims.size() == kPairTransposeDimCount && llvm::is_contained(transposeDims, sourceOrder.size() - 1)) {
    return computeLastDimTransposeAllocBytes(srcShape, transposeDims[0], transposeDims[1], elemBytes, isF32Transpose);
  }
  if (transposeDims.size() <= kPairTransposeDimCount) {
    return peakBytes;
  }

  SmallVector<size_t, kSmallVectorSizeSix> currentOrder(sourceOrder.begin(), sourceOrder.end());
  SmallVector<int64_t, kSmallVectorSizeSix> currentShape(srcShape.begin(), srcShape.end());
  for (size_t targetPos = 0; targetPos < targetOrder.size(); ++targetPos) {
    auto it = llvm::find(currentOrder, targetOrder[targetPos]);
    if (it == currentOrder.end()) {
      return peakBytes;
    }
    for (auto i = static_cast<size_t>(it - currentOrder.begin()); i > targetPos; --i) {
      if (i == currentOrder.size() - 1) {
        peakBytes =
          std::max(peakBytes, computeLastDimTransposeAllocBytes(currentShape, i - 1, i, elemBytes, isF32Transpose));
      }
      std::swap(currentOrder[i - 1], currentOrder[i]);
      std::swap(currentShape[i - 1], currentShape[i]);
    }
  }
  return peakBytes;
}
bool satisfiesLastDimTransposeAllocAlign(ArrayRef<int64_t> srcShape, size_t dim0, size_t dim1, int64_t elemBytes,
                                         bool isF32Transpose) {
  SmallVector<int64_t, kSmallVectorSizeSix> dstShape;
  SmallVector<int64_t, kSmallVectorSizeSix> srcAlign;
  SmallVector<int64_t, kSmallVectorSizeSix> dstAlign;
  computeLastDimTransposeAlignBytes({srcShape, dim0, dim1, elemBytes, isF32Transpose}, dstShape, srcAlign, dstAlign);
  return satisfiesAlignedShapeWithoutExpansion(srcShape, srcAlign, elemBytes) &&
         satisfiesAlignedShapeWithoutExpansion(dstShape, dstAlign, elemBytes);
}
bool satisfiesTransposeAllocAlign(ArrayRef<int64_t> srcShape, ArrayRef<size_t> sourceOrder,
                                  ArrayRef<size_t> targetOrder, int64_t elemBytes, bool isF32Transpose) {
  if (srcShape.size() != sourceOrder.size() || sourceOrder.size() != targetOrder.size()) {
    return false;
  }

  SmallVector<size_t, kSmallVectorSizeSix> transposeDims;
  for (size_t i = 0; i < sourceOrder.size(); ++i) {
    if (sourceOrder[i] != targetOrder[i]) {
      transposeDims.push_back(i);
    }
  }
  if (transposeDims.size() == kPairTransposeDimCount && llvm::is_contained(transposeDims, sourceOrder.size() - 1)) {
    return satisfiesLastDimTransposeAllocAlign(srcShape, transposeDims[0], transposeDims[1], elemBytes, isF32Transpose);
  }
  if (transposeDims.size() <= kPairTransposeDimCount) {
    return true;
  }

  SmallVector<size_t, kSmallVectorSizeSix> currentOrder(sourceOrder.begin(), sourceOrder.end());
  SmallVector<int64_t, kSmallVectorSizeSix> currentShape(srcShape.begin(), srcShape.end());
  for (size_t targetPos = 0; targetPos < targetOrder.size(); ++targetPos) {
    auto it = llvm::find(currentOrder, targetOrder[targetPos]);
    if (it == currentOrder.end()) {
      return false;
    }
    for (auto i = static_cast<size_t>(it - currentOrder.begin()); i > targetPos; --i) {
      if (i == currentOrder.size() - 1 &&
          !satisfiesLastDimTransposeAllocAlign(currentShape, i - 1, i, elemBytes, isF32Transpose)) {
        return false;
      }
      std::swap(currentOrder[i - 1], currentOrder[i]);
      std::swap(currentShape[i - 1], currentShape[i]);
    }
  }
  return true;
}
int64_t computeStrideAlignedTransposeBytes(ArrayRef<int64_t> shape, ArrayRef<size_t> sourceOrder,
                                           ArrayRef<size_t> targetOrder, int64_t elemBytes) {
  if (shape.size() <= kMinTransposeAxisOrderSize || shape.size() != sourceOrder.size() ||
      sourceOrder.size() != targetOrder.size()) {
    return 0;
  }

  auto findPos = [](ArrayRef<size_t> order, size_t axis) -> std::optional<size_t> {
    auto it = llvm::find(order, axis);
    if (it == order.end()) {
      return std::nullopt;
    }
    return static_cast<size_t>(it - order.begin());
  };
  auto computePairBytes = [&sourceOrder, &shape, &elemBytes, &findPos](size_t lhsAxis, size_t rhsAxis) {
    std::optional<size_t> lhs = findPos(sourceOrder, lhsAxis);
    std::optional<size_t> rhs = findPos(sourceOrder, rhsAxis);
    if (!lhs || !rhs) {
      return int64_t{0};
    }
    SmallVector<int64_t, kSmallVectorSizeSix> alignBytes(shape.size(), 0);
    alignBytes[*lhs] = kUbAlignBytes;
    alignBytes[*rhs] = kUbAlignBytes;
    return computeAlignedShapeBytes(shape, alignBytes, elemBytes);
  };

  SmallVector<size_t, kSmallVectorSizeSix> current(sourceOrder.begin(), sourceOrder.end());
  int64_t peakBytes = 0;
  for (size_t targetPos = 0; targetPos < targetOrder.size(); ++targetPos) {
    std::optional<size_t> pos = findPos(current, targetOrder[targetPos]);
    if (!pos) {
      return peakBytes;
    }
    for (size_t i = *pos; i > targetPos; --i) {
      if (i == current.size() - 1) {
        peakBytes = std::max(peakBytes, computePairBytes(current[i - 1], current[i]));
      }
      std::swap(current[i - 1], current[i]);
    }
  }
  return peakBytes;
}
bool satisfiesStrideAlignedTranspose(ArrayRef<int64_t> shape, ArrayRef<size_t> sourceOrder,
                                     ArrayRef<size_t> targetOrder, int64_t elemBytes) {
  if (shape.size() <= kMinTransposeAxisOrderSize || shape.size() != sourceOrder.size() ||
      sourceOrder.size() != targetOrder.size()) {
    return true;
  }

  auto findPos = [](ArrayRef<size_t> order, size_t axis) -> std::optional<size_t> {
    auto it = llvm::find(order, axis);
    if (it == order.end()) {
      return std::nullopt;
    }
    return static_cast<size_t>(it - order.begin());
  };
  auto satisfiesPair = [&sourceOrder, &shape, &elemBytes, &findPos](size_t lhsAxis, size_t rhsAxis) {
    std::optional<size_t> lhs = findPos(sourceOrder, lhsAxis);
    std::optional<size_t> rhs = findPos(sourceOrder, rhsAxis);
    if (!lhs || !rhs) {
      return false;
    }
    SmallVector<int64_t, kSmallVectorSizeSix> alignBytes(shape.size(), 0);
    alignBytes[*lhs] = kUbAlignBytes;
    alignBytes[*rhs] = kUbAlignBytes;
    return satisfiesAlignedShapeWithoutExpansion(shape, alignBytes, elemBytes);
  };

  SmallVector<size_t, kSmallVectorSizeSix> current(sourceOrder.begin(), sourceOrder.end());
  for (size_t targetPos = 0; targetPos < targetOrder.size(); ++targetPos) {
    std::optional<size_t> pos = findPos(current, targetOrder[targetPos]);
    if (!pos) {
      return false;
    }
    for (size_t i = *pos; i > targetPos; --i) {
      if (i == current.size() - 1 && !satisfiesPair(current[i - 1], current[i])) {
        return false;
      }
      std::swap(current[i - 1], current[i]);
    }
  }
  return true;
}
bool fitsVectorUb(const NpuBandContext &ctx, ArrayRef<int64_t> axisTiles) {
  return computeVectorPeakReserveBytes(ctx, axisTiles) <= getVectorUbBytes(ctx);
}

int64_t findMaxFittingCommonFactorTile(const NpuBandContext &ctx, SmallVectorImpl<int64_t> &axisTiles, size_t axisIdx,
                                       int64_t candidate) {
  int64_t extent = axisIdx < ctx.extents.size() ? std::max<int64_t>(ctx.extents[axisIdx], 1) : 1;
  int64_t alignUnit = axisIdx < ctx.axesAlignUnits.size() ? std::max<int64_t>(ctx.axesAlignUnits[axisIdx], 1) : 1;
  int64_t fitTile = 1;
  for (int64_t q = std::max<int64_t>(candidate, 1); q >= 1;) {
    int64_t tile = candidate / q;
    q = candidate / (tile + 1);
    if (tile <= 1 || candidate % tile != 0 || extent % tile != 0) {
      continue;
    }
    if (tile % alignUnit != 0) {
      continue;
    }
    axisTiles[axisIdx] = tile;
    if (!fitsVectorUb(ctx, axisTiles)) {
      break;
    }
    fitTile = tile;
  }
  axisTiles[axisIdx] = 1;
  return fitTile;
}

int64_t findMaxFittingTile(const NpuBandContext &ctx, SmallVectorImpl<int64_t> &axisTiles, size_t axisIdx,
                           int64_t upper) {
  int64_t alignUnit = axisIdx < ctx.axesAlignUnits.size() ? std::max<int64_t>(ctx.axesAlignUnits[axisIdx], 1) : 1;
  int64_t safeUpper = std::max<int64_t>(upper, 1);
  // Enumerate by `alignUnit` cells. The last cell may be `alignUp(upper, alignUnit)`,
  // which is intentional: BiSheng pads stride-aligned vector buffers to that logical
  // tile shape, so the schedule-side tile must preserve the same aligned extent.
  int64_t high = ceilDivInt64(safeUpper, alignUnit);
  int64_t best = alignUnit;
  int64_t low = 1;
  while (low <= high) {
    int64_t cell = (low + high) / 2;
    int64_t mid = cell * alignUnit;
    axisTiles[axisIdx] = mid;
    if (fitsVectorUb(ctx, axisTiles)) {
      best = mid;
      low = cell + 1;
    } else {
      high = cell - 1;
    }
  }
  axisTiles[axisIdx] = best;
  return best;
}
void fillPrefixSuffixTiles(const NpuBandContext &ctx, ArrayRef<unsigned> prefixOuter, int64_t requestedPointRows,
                           BandTilePlan &plan) {
  for (size_t i = 0; i < prefixOuter.size(); ++i) {
    plan.outerTiles[i] = prefixOuter[i];
    plan.innerTiles[i] = prefixOuter[i];
  }
  size_t innermost = ctx.extents.size() - 1;
  SmallVector<int64_t, kSmallVectorSizeFour> axisTiles(plan.innerTiles.begin(), plan.innerTiles.end());
  int64_t maxPointRows = std::min<int64_t>(requestedPointRows, prefixOuter.front());
  for (int64_t pointRows = std::max<int64_t>(maxPointRows, 1); pointRows >= 1; --pointRows) {
    axisTiles.front() = pointRows;
    axisTiles[innermost] = ctx.extents[innermost];
    int64_t innerTile = findMaxFittingTile(ctx, axisTiles, innermost, ctx.extents[innermost]);
    if (fitsVectorUb(ctx, axisTiles)) {
      plan.innerTiles.front() = saturateToTileValue(pointRows);
      plan.innerTiles[innermost] = saturateToTileValue(innerTile);
      return;
    }
  }
  plan.innerTiles.front() = 1;
  plan.innerTiles[innermost] = 1;
}
int64_t computeTransposeCoLiveReserveBytes(const NpuBandContext &ctx, ArrayRef<int64_t> axisTiles) {
  if (ctx.transposeCoLiveReserveSets.empty()) {
    return 0;
  }

  int64_t peakBytes = 0;
  for (const auto &liveSet : ctx.transposeCoLiveReserveSets) {
    int64_t liveBytes = 0;
    for (const TransposeCoLiveReserve &reserve : liveSet) {
      VectorLiveReserve vectorReserve{reserve.axisOrder, reserve.alignDims, reserve.elementBits};
      int64_t bytes = computeReserveBytes(ctx, vectorReserve, axisTiles);
      liveBytes = (liveBytes > LLONG_MAX - bytes) ? LLONG_MAX : liveBytes + bytes;
    }
    peakBytes = std::max(peakBytes, liveBytes);
  }
  return peakBytes;
}
SmallVector<size_t, kSmallVectorSizeSix> normalizeTransposeTargetAxisOrder(ArrayRef<size_t> sourceAxisOrder,
                                                                           ArrayRef<size_t> targetAxisOrder) {
  SmallVector<size_t, kSmallVectorSizeSix> normalized(targetAxisOrder.begin(), targetAxisOrder.end());
  if (normalized.size() == sourceAxisOrder.size()) {
    return normalized;
  }

  normalized.assign(sourceAxisOrder.begin(), sourceAxisOrder.end());
  if (normalized.size() >= kMinTransposeAxisOrderSize) {
    std::swap(normalized[normalized.size() - kMinTransposeAxisOrderSize], normalized.back());
  }
  return normalized;
}
// Bundled parameters for computeTransposeTempReserveBytes to stay under readability-function-size_parameters limit.
struct TransposeTempReserveParams {
  const NpuBandContext &ctx;
  ArrayRef<int64_t> axisTiles;
  ArrayRef<size_t> sourceAxisOrder;
  ArrayRef<size_t> targetAxisOrder;
  int64_t elementBits;
  bool elementIsF32;
};
int64_t computeTransposeTempReserveBytes(const TransposeTempReserveParams &params) {
  const NpuBandContext &ctx = params.ctx;
  ArrayRef<int64_t> axisTiles = params.axisTiles;
  ArrayRef<size_t> sourceAxisOrder = params.sourceAxisOrder;
  ArrayRef<size_t> targetAxisOrder = params.targetAxisOrder;
  int64_t elementBits = params.elementBits;
  bool elementIsF32 = params.elementIsF32;
  if (sourceAxisOrder.empty()) {
    return 0;
  }
  SmallVector<size_t, kSmallVectorSizeSix> normalizedTargetAxisOrder =
    normalizeTransposeTargetAxisOrder(sourceAxisOrder, targetAxisOrder);

  SmallVector<size_t, kSmallVectorSizeSix> sourceOrder(sourceAxisOrder.begin(), sourceAxisOrder.end());
  VectorLiveReserve source{sourceOrder, getDefaultBishengStrideAlignDims(static_cast<int64_t>(sourceAxisOrder.size())),
                           elementBits};
  VectorLiveReserve target{normalizedTargetAxisOrder,
                           getDefaultBishengStrideAlignDims(static_cast<int64_t>(normalizedTargetAxisOrder.size())),
                           elementBits};
  int64_t sourceBytes = computeReserveBytes(ctx, source, axisTiles);
  int64_t targetBytes = computeReserveBytes(ctx, target, axisTiles);
  // BiSheng materializes the load layout and vtranspose result as separate UB allocations.
  // Both can carry multi_buffer, so the tiling model must count them co-live.
  int64_t strideReserveBytes = multiplyAndCap(
    kHivmAutoMultiBufferFactor, sourceBytes > LLONG_MAX - targetBytes ? LLONG_MAX : sourceBytes + targetBytes);

  int64_t elemBytes = std::max<int64_t>(ceilDivInt64(std::max<int64_t>(elementBits, 1), kBitsPerByte), 1);
  SmallVector<int64_t, kSmallVectorSizeSix> sourceShape;
  sourceShape.resize(sourceOrder.size());
  const auto &tilesRef = axisTiles;
  std::transform(sourceOrder.begin(), sourceOrder.end(), sourceShape.begin(),
                 [&tilesRef](size_t axisIdx) { return std::max<int64_t>(tilesRef[axisIdx], 1); });

  int64_t allocReserveBytes =
    computeTransposeAllocReserveBytes(sourceShape, sourceOrder, normalizedTargetAxisOrder, elemBytes, elementIsF32);
  int64_t adjacentStrideBytes =
    multiplyAndCap(kHivmAutoMultiBufferFactor + 1,
                   computeStrideAlignedTransposeBytes(sourceShape, sourceOrder, normalizedTargetAxisOrder, elemBytes));
  return std::max(strideReserveBytes, std::max(allocReserveBytes, adjacentStrideBytes));
}
bool satisfiesVectorReserveAlignConstraints(const NpuBandContext &ctx, ArrayRef<int64_t> axisTiles) {
  if (std::any_of(ctx.vectorPeakReserveSets.begin(), ctx.vectorPeakReserveSets.end(),
                  [&ctx, &axisTiles](const VectorPeakReserveSet &set) {
                    return std::any_of(set.liveBuffers.begin(), set.liveBuffers.end(),
                                       [&ctx, &axisTiles](const VectorLiveReserve &reserve) {
                                         return !satisfiesStrideAlignWithoutExpansion(ctx, reserve, axisTiles);
                                       });
                  })) {
    return false;
  }
  if (std::any_of(ctx.transposeCoLiveReserveSets.begin(), ctx.transposeCoLiveReserveSets.end(),
                  [&ctx, &axisTiles](const SmallVector<TransposeCoLiveReserve, kSmallVectorSizeFour> &liveSet) {
                    return std::any_of(
                      liveSet.begin(), liveSet.end(), [&ctx, &axisTiles](const TransposeCoLiveReserve &reserve) {
                        VectorLiveReserve vectorReserve{reserve.axisOrder, reserve.alignDims, reserve.elementBits};
                        return !satisfiesStrideAlignWithoutExpansion(ctx, vectorReserve, axisTiles);
                      });
                  })) {
    return false;
  }
  return true;
}
bool satisfiesSingleTransposeAlignConstraints(ArrayRef<int64_t> axisTiles, ArrayRef<size_t> sourceAxisOrder,
                                              ArrayRef<size_t> targetAxisOrder, int64_t elementBits,
                                              bool elementIsF32) {
  if (sourceAxisOrder.empty()) {
    return false;
  }

  SmallVector<size_t, kSmallVectorSizeSix> normalizedTargetAxisOrder =
    normalizeTransposeTargetAxisOrder(sourceAxisOrder, targetAxisOrder);
  int64_t elemBytes = std::max<int64_t>(ceilDivInt64(std::max<int64_t>(elementBits, 1), kBitsPerByte), 1);
  SmallVector<int64_t, kSmallVectorSizeSix> sourceShape;
  sourceShape.resize(sourceAxisOrder.size());
  std::transform(sourceAxisOrder.begin(), sourceAxisOrder.end(), sourceShape.begin(),
                 [&axisTiles](size_t axisIdx) { return std::max<int64_t>(axisTiles[axisIdx], 1); });
  return satisfiesTransposeAllocAlign(sourceShape, sourceAxisOrder, normalizedTargetAxisOrder, elemBytes,
                                      elementIsF32) &&
         satisfiesStrideAlignedTranspose(sourceShape, sourceAxisOrder, normalizedTargetAxisOrder, elemBytes);
}
bool satisfiesTransposeAlignConstraints(const NpuBandContext &ctx, ArrayRef<int64_t> axisTiles,
                                        ArrayRef<size_t> fallbackSourceAxisOrder) {
  if (!satisfiesVectorReserveAlignConstraints(ctx, axisTiles)) {
    return false;
  }
  if (ctx.transposeInfos.empty()) {
    return satisfiesSingleTransposeAlignConstraints(axisTiles, fallbackSourceAxisOrder, ctx.transposeTargetAxisOrder,
                                                    ctx.transposeElementBits, ctx.transposeElementIsF32);
  }

  return llvm::all_of(ctx.transposeInfos, [&axisTiles, &ctx](const TransposeVectorInfo &info) {
    return satisfiesSingleTransposeAlignConstraints(axisTiles, info.sourceAxisOrder, info.targetAxisOrder,
                                                    info.elementBits, info.isF32);
  });
}
int64_t getAnotherLastDimTransposeSizeAlignDim(const TransposeVectorInfo &info, size_t dim, bool &isInnerDim) {
  isInnerDim = false;
  if (info.sourceAxisOrder.size() != info.targetAxisOrder.size() || dim >= info.sourceAxisOrder.size()) {
    return kInvalidAxisIndex;
  }

  SmallVector<size_t, kSmallVectorSizeSix> transposeDims;
  for (size_t i = 0; i < info.sourceAxisOrder.size(); ++i) {
    if (info.sourceAxisOrder[i] != info.targetAxisOrder[i]) {
      transposeDims.push_back(i);
    }
  }
  if (transposeDims.size() == kPairTransposeDimCount) {
    if (!llvm::is_contained(transposeDims, info.sourceAxisOrder.size() - 1) ||
        !llvm::is_contained(transposeDims, dim)) {
      return kInvalidAxisIndex;
    }
    isInnerDim = dim == info.sourceAxisOrder.size() - 1;
    return static_cast<int64_t>(transposeDims[dim == transposeDims[0] ? 1 : 0]);
  }

  size_t sourceAxis = info.sourceAxisOrder[dim];
  SmallVector<size_t, kSmallVectorSizeSix> currentOrder(info.sourceAxisOrder.begin(), info.sourceAxisOrder.end());
  for (size_t targetPos = 0; targetPos < info.targetAxisOrder.size(); ++targetPos) {
    auto it = llvm::find(currentOrder, info.targetAxisOrder[targetPos]);
    if (it == currentOrder.end()) {
      return kInvalidAxisIndex;
    }
    for (auto i = static_cast<size_t>(it - currentOrder.begin()); i > targetPos; --i) {
      if (i == currentOrder.size() - 1 && (currentOrder[i - 1] == sourceAxis || currentOrder[i] == sourceAxis)) {
        isInnerDim = currentOrder[i] == sourceAxis;
        size_t anotherAxis = isInnerDim ? currentOrder[i - 1] : currentOrder[i];
        auto anotherIt = llvm::find(info.sourceAxisOrder, anotherAxis);
        return anotherIt == info.sourceAxisOrder.end() ? kInvalidAxisIndex
                                                       : static_cast<int64_t>(anotherIt - info.sourceAxisOrder.begin());
      }
      std::swap(currentOrder[i - 1], currentOrder[i]);
    }
  }
  return kInvalidAxisIndex;
}

int64_t getLastNonUnitDim(ArrayRef<size_t> sourceOrder, ArrayRef<size_t> targetOrder, const NpuBandContext &ctx,
                          int64_t startDim) {
  for (int64_t dim = startDim; dim >= 0; --dim) {
    size_t sourceAxis = sourceOrder[static_cast<size_t>(dim)];
    size_t targetAxis = targetOrder[static_cast<size_t>(dim)];
    int64_t sourceExtent = sourceAxis < ctx.extents.size() ? ctx.extents[sourceAxis] : 1;
    int64_t targetExtent = targetAxis < ctx.extents.size() ? ctx.extents[targetAxis] : 1;
    if (sourceExtent != 1 || targetExtent != 1) {
      return dim;
    }
  }
  return kInvalidAxisIndex;
}

int64_t getNonLastTransposeStrideAlignDim(ArrayRef<size_t> sourceOrder, ArrayRef<size_t> targetOrder,
                                          const NpuBandContext &ctx) {
  if (sourceOrder.size() < kMinTransposeAxisOrderSize || sourceOrder.size() != targetOrder.size()) {
    return kInvalidAxisIndex;
  }

  SmallVector<size_t, kSmallVectorSizeSix> transposeDims;
  for (size_t i = 0; i < sourceOrder.size(); ++i) {
    if (sourceOrder[i] != targetOrder[i]) {
      transposeDims.push_back(i);
    }
  }
  if (transposeDims.empty() || llvm::is_contained(transposeDims, sourceOrder.size() - 1)) {
    return kInvalidAxisIndex;
  }
  return getLastNonUnitDim(sourceOrder, targetOrder, ctx, static_cast<int64_t>(transposeDims.back()));
}

bool isNonLastTransposeStrideAlignAxis(const TransposeVectorInfo &info, size_t axisIdx, const NpuBandContext &ctx) {
  if (info.sourceAxisOrder.size() < kMinTransposeAxisOrderSize ||
      info.sourceAxisOrder.size() != info.targetAxisOrder.size()) {
    return false;
  }

  SmallVector<size_t, kSmallVectorSizeSix> transposeDims;
  for (size_t i = 0; i < info.sourceAxisOrder.size(); ++i) {
    if (info.sourceAxisOrder[i] != info.targetAxisOrder[i]) {
      transposeDims.push_back(i);
    }
  }
  auto matchesAxis = [&axisIdx](ArrayRef<size_t> sourceOrder, ArrayRef<size_t> targetOrder, int64_t dim) {
    return dim >= 0 &&
           (sourceOrder[static_cast<size_t>(dim)] == axisIdx || targetOrder[static_cast<size_t>(dim)] == axisIdx);
  };
  if (transposeDims.size() == kPairTransposeDimCount) {
    return matchesAxis(info.sourceAxisOrder, info.targetAxisOrder,
                       getNonLastTransposeStrideAlignDim(info.sourceAxisOrder, info.targetAxisOrder, ctx));
  }
  if (transposeDims.size() <= kPairTransposeDimCount) {
    return false;
  }

  SmallVector<size_t, kSmallVectorSizeSix> currentOrder(info.sourceAxisOrder.begin(), info.sourceAxisOrder.end());
  for (size_t targetPos = 0; targetPos < info.targetAxisOrder.size(); ++targetPos) {
    auto it = llvm::find(currentOrder, info.targetAxisOrder[targetPos]);
    if (it == currentOrder.end()) {
      return false;
    }
    for (auto i = static_cast<size_t>(it - currentOrder.begin()); i > targetPos; --i) {
      SmallVector<size_t, kSmallVectorSizeSix> nextOrder(currentOrder.begin(), currentOrder.end());
      std::swap(nextOrder[i - 1], nextOrder[i]);
      if (matchesAxis(currentOrder, nextOrder, getNonLastTransposeStrideAlignDim(currentOrder, nextOrder, ctx))) {
        return true;
      }
      currentOrder = std::move(nextOrder);
    }
  }
  return false;
}

// Aggregate the per-axis alignment requirement from every TransposeVectorInfo
// that mentions `axisIdx`. The returned value is in element units (not bytes):
// any tile size that is a multiple of it satisfies 32B size/stride alignment.
int64_t computeAxisAlignSize(size_t axisIdx, const NpuBandContext &ctx) {
  int64_t lcmUnit = 1;
  for (const TransposeVectorInfo &info : ctx.transposeInfos) {
    if (info.sourceAxisOrder.size() < kMinTransposeAxisOrderSize) {
      continue;
    }
    auto it = llvm::find(info.sourceAxisOrder, axisIdx);
    if (it == info.sourceAxisOrder.end()) {
      continue;
    }
    auto dim = static_cast<size_t>(it - info.sourceAxisOrder.begin());
    int64_t elemBytes = std::max<int64_t>(ceilDivInt64(std::max<int64_t>(info.elementBits, 1), kBitsPerByte), 1);
    bool isInnerDim = false;

    if (getAnotherLastDimTransposeSizeAlignDim(info, dim, isInnerDim) >= 0) {
      lcmUnit = std::lcm(lcmUnit, std::max<int64_t>(kUbAlignBytes / elemBytes, 1));
    }

    // Non-last-dim transpose uses BiShengIR MarkStrideAlign flattening, not alloc-size align.
    if (isNonLastTransposeStrideAlignAxis(info, axisIdx, ctx)) {
      lcmUnit = std::lcm(lcmUnit, std::max<int64_t>(kUbAlignBytes / elemBytes, 1));
    }
  }
  return std::max<int64_t>(lcmUnit, 1);
}

void applyF32LastDimDoubleAlign(const NpuBandContext &ctx, SmallVectorImpl<int64_t> &axisAlignUnits) {
  for (const TransposeVectorInfo &info : ctx.transposeInfos) {
    if (!info.isF32) {
      continue;
    }
    int64_t elemBytes = std::max<int64_t>(ceilDivInt64(std::max<int64_t>(info.elementBits, 1), kBitsPerByte), 1);
    int64_t doubleUnit = std::max<int64_t>((kUbAlignBytes * kDoubleAlignFactor) / elemBytes, 1);
    for (size_t dim = 0; dim < info.sourceAxisOrder.size(); ++dim) {
      bool isInnerDim = false;
      int64_t anotherDim = getAnotherLastDimTransposeSizeAlignDim(info, dim, isInnerDim);
      if (anotherDim < 0 || !isInnerDim) {
        continue;
      }
      size_t innerAxis = info.sourceAxisOrder[dim];
      size_t anotherAxis = info.sourceAxisOrder[static_cast<size_t>(anotherDim)];
      if (innerAxis >= axisAlignUnits.size() || anotherAxis >= axisAlignUnits.size()) {
        continue;
      }
      if (axisAlignUnits[innerAxis] % doubleUnit == 0 || axisAlignUnits[anotherAxis] % doubleUnit == 0) {
        continue;
      }
      axisAlignUnits[innerAxis] = std::lcm(axisAlignUnits[innerAxis], doubleUnit);
    }
  }
}

static SmallVector<int64_t, kSmallVectorSizeSix> buildTransposeSearchAxisTiles(const NpuBandContext &ctx,
                                                                               const BandTilePlan &plan,
                                                                               ArrayRef<size_t> searchAxisOrder,
                                                                               ArrayRef<int64_t> searchShape) {
  SmallVector<int64_t, kSmallVectorSizeSix> axisTiles;
  axisTiles.reserve(ctx.extents.size());
  for (size_t i = 0; i < ctx.extents.size(); ++i) {
    const int64_t fullExtent = i < ctx.extents.size() ? std::max<int64_t>(ctx.extents[i], 1) : int64_t{1};
    int64_t planned = i < plan.innerTiles.size() ? std::max<int64_t>(static_cast<int64_t>(plan.innerTiles[i]), 1)
                                                 : saturateToTileValue(fullExtent);
    int64_t tile = planned;
    if (!llvm::is_contained(searchAxisOrder, i)) {
      // Axes outside the DFS only matter when inner tiles still mirror full loop trips
      // (initWholeBandPlan). That multiplied every live footprint by Π extent and drowned
      // meaningful transpose tiles. Already-split shells (planned < extent, e.g. block-0
      // parallel slicing) stay in the UB model.
      if (planned >= fullExtent) {
        tile = 1;
      }
    }
    axisTiles.push_back(tile);
  }
  for (size_t i = 0; i < searchAxisOrder.size() && i < searchShape.size(); ++i) {
    axisTiles[searchAxisOrder[i]] = std::max<int64_t>(searchShape[i], 1);
  }
  return axisTiles;
}

static int64_t computeTransposeSearchCandidatePeak(const NpuBandContext &ctx, const BandTilePlan &plan,
                                                   ArrayRef<size_t> searchAxisOrder, ArrayRef<int64_t> searchShape) {
  SmallVector<int64_t, kSmallVectorSizeSix> axisTiles =
    buildTransposeSearchAxisTiles(ctx, plan, searchAxisOrder, searchShape);
  if (!satisfiesTransposeAlignConstraints(ctx, axisTiles, searchAxisOrder)) {
    return static_cast<int64_t>(LLONG_MAX);
  }
  int64_t vectorBytes = computeVectorPeakReserveBytes(ctx, axisTiles);
  int64_t transposeBytes = 0;
  if (ctx.transposeInfos.empty()) {
    transposeBytes = computeTransposeTempReserveBytes({ctx, axisTiles, searchAxisOrder, ctx.transposeTargetAxisOrder,
                                                       ctx.transposeElementBits, ctx.transposeElementIsF32});
  } else {
    for (const TransposeVectorInfo &info : ctx.transposeInfos) {
      transposeBytes = std::max(
        transposeBytes, computeTransposeTempReserveBytes(
                          {ctx, axisTiles, info.sourceAxisOrder, info.targetAxisOrder, info.elementBits, info.isF32}));
    }
  }
  int64_t coLiveBytes = computeTransposeCoLiveReserveBytes(ctx, axisTiles);
  transposeBytes = (transposeBytes > static_cast<int64_t>(LLONG_MAX) - coLiveBytes) ? static_cast<int64_t>(LLONG_MAX)
                                                                                    : transposeBytes + coLiveBytes;
  return std::max<int64_t>(vectorBytes, transposeBytes);
}

struct AlignedTransposeMinTileExceedsUbParams {
  const NpuBandContext &ctx;
  ArrayRef<size_t> searchAxisOrder;
  ArrayRef<int64_t> axisAlignSize;
  ArrayRef<int64_t> axisMaxTileSize;
  ArrayRef<int64_t> tiles;
  int64_t ubLimitBytes;
  int64_t peak;
};
static void emitAlignedTransposeMinTileExceedsUb(const AlignedTransposeMinTileExceedsUbParams &params) {
  const NpuBandContext &ctx = params.ctx;
  ArrayRef<size_t> searchAxisOrder = params.searchAxisOrder;
  ArrayRef<int64_t> axisAlignSize = params.axisAlignSize;
  ArrayRef<int64_t> axisMaxTileSize = params.axisMaxTileSize;
  ArrayRef<int64_t> tiles = params.tiles;
  int64_t ubLimitBytes = params.ubLimitBytes;
  int64_t peak = params.peak;
  llvm::errs() << "[TilingStrategy] chooseAlignedTransposeTiles: boundary-1 "
                  "(min tile exceeds UB)\n";
  llvm::errs() << "  ubLimitBytes = " << ubLimitBytes << "\n";
  llvm::errs() << "  peak(minTile) = " << peak << "\n";
  for (size_t i = 0; i < searchAxisOrder.size(); ++i) {
    llvm::errs() << "    axis[" << searchAxisOrder[i] << "] extent=" << ctx.extents[searchAxisOrder[i]]
                 << " axisAlignSize=" << axisAlignSize[i] << " axisMaxTileSize=" << axisMaxTileSize[i]
                 << " finalTile=" << tiles[i] << "\n";
  }
}

// Greedy transpose tile selection.
// Replaces the previous DFS over per-axis candidate sets. The new algorithm
// is built around `ctx.axesAlignUnits`, which aggregates the per-axis LCM
// alignment requirement across every TransposeVectorInfo
// (32B size-align on last-dim transpose dims, f32 64B double-align, 32B inner
// stride for non-last-dim transposes). Because every tile we emit is a multiple
// of `axisAlignSize`, satisfaction of `satisfiesTransposeAlignConstraints` is
// guaranteed by construction; we only assert it as an invariant.
// Strategy (matches the agreed plan)
//   1. Compute `axisAlignSize[i]` and `axisMaxTileSize[i] = alignUp(extent, alignSize)`
//      for every axis in `searchAxisOrder`.
//   2. Initialize tiles to `[axisAlignSize, ..., axisAlignSize, axisMaxTileSize_last]`
//      -- bias toward keeping the innermost axis full (best vector intrinsic
//      throughput).
//   3. If that already exceeds UB, shrink only the innermost axis by
//      `axisAlignSize_last` each step until it fits. If we hit
//      `tiles[n-1] == axisAlignSize_last` and still don't fit -- "boundary 1"
//      -- keep the minimum legal aligned tile instead of falling back to a
//      generic unaligned split.
//   4. Otherwise, walk from the second-innermost axis outward and try to lift
//      each tile to its `axisMaxTileSize`; revert per-axis if it would
//      exceed UB. No DFS; no LLONG_MAX silent fallback.
// Always returns the best aligned transpose tile it can construct.
bool chooseAlignedTransposeTiles(const NpuBandContext &ctx, BandTilePlan &plan, ArrayRef<size_t> searchAxisOrder,
                                 int64_t ubLimitBytes) {
  if (searchAxisOrder.empty()) {
    return true;
  }

  const size_t n = searchAxisOrder.size();
  SmallVector<int64_t, kSmallVectorSizeSix> axisAlignSize(n, 1);
  SmallVector<int64_t, kSmallVectorSizeSix> axisMaxTileSize(n, 1);
  for (size_t i = 0; i < n; ++i) {
    size_t axisIdx = searchAxisOrder[i];
    int64_t alignSize = axisIdx < ctx.axesAlignUnits.size() ? ctx.axesAlignUnits[axisIdx] : 1;
    axisAlignSize[i] = std::max<int64_t>(alignSize, 1);
  }
  for (size_t i = 0; i < n; ++i) {
    size_t axisIdx = searchAxisOrder[i];
    int64_t extent = std::max<int64_t>(ctx.extents[axisIdx], 1);
    axisMaxTileSize[i] = alignUpInt64(extent, axisAlignSize[i]);
  }

  // Reset plan tiles for search-axis-order before peak computation.
  // `computeTransposeSearchCandidatePeak` builds axisTiles from plan + the
  // provided searchShape; axes outside `searchAxisOrder` must already be set by
  // the caller (tryBuildTransposePlan handles axis 0 multi-core split there).
  for (size_t i = 0; i < n; ++i) {
    plan.outerTiles[searchAxisOrder[i]] = 1;
    plan.innerTiles[searchAxisOrder[i]] = 1;
  }

  // Initial tile: outer axes minimal, innermost maximal.
  SmallVector<int64_t, kSmallVectorSizeSix> tiles(n, 1);
  for (size_t i = 0; i + 1 < n; ++i) {
    tiles[i] = axisAlignSize[i];
  }
  tiles[n - 1] = axisMaxTileSize[n - 1];

  auto peakOf = [&ctx, &plan, &searchAxisOrder](ArrayRef<int64_t> shape) {
    return computeTransposeSearchCandidatePeak(ctx, plan, searchAxisOrder, shape);
  };

  int64_t peak = peakOf(tiles);
  if (peak > ubLimitBytes) {
    // Path A: shrink innermost only. The byte footprint is monotonic for
    // aligned candidates, but the stride-align predicate can have holes when a
    // candidate crosses the full-extent/static-dim boundary. If that happens,
    // fall back to the old descending scan to preserve exact tile selection.
    int64_t align = axisAlignSize[n - 1];
    int64_t highCell = std::max<int64_t>(axisMaxTileSize[n - 1] / align, 1);
    int64_t low = 1;
    int64_t high = highCell - 1;  // full aligned tile was checked above and does not fit.
    int64_t bestCell = 1;
    bool useLinearScan = peak == static_cast<int64_t>(LLONG_MAX);

    while (!useLinearScan && low <= high) {
      int64_t cell = low + (high - low) / 2;
      tiles[n - 1] = cell * align;
      int64_t candidatePeak = peakOf(tiles);
      if (candidatePeak == static_cast<int64_t>(LLONG_MAX)) {
        useLinearScan = true;
        break;
      }
      if (candidatePeak <= ubLimitBytes) {
        bestCell = cell;
        low = cell + 1;
      } else {
        high = cell - 1;
      }
    }

    if (useLinearScan) {
      tiles[n - 1] = axisMaxTileSize[n - 1];
      peak = peakOf(tiles);
      while (peak > ubLimitBytes && tiles[n - 1] > align) {
        tiles[n - 1] -= align;
        peak = peakOf(tiles);
      }
    } else {
      tiles[n - 1] = bestCell * align;
    }
    peak = peakOf(tiles);
    if (peak > ubLimitBytes) {
      // Boundary 1: even the minimum aligned tile exceeds the UB model. Keep it
      // to preserve transpose alignment instead of falling back to generic.
      emitAlignedTransposeMinTileExceedsUb(
        {ctx, searchAxisOrder, axisAlignSize, axisMaxTileSize, tiles, ubLimitBytes, peak});
    }
  } else {
    // Path B: walk from second-innermost outward, try lifting each tile to
    // its axisMaxTileSize.
    for (size_t i = n - 1; i-- > 0;) {
      int64_t saved = tiles[i];
      tiles[i] = axisMaxTileSize[i];
      int64_t newPeak = peakOf(tiles);
      if (newPeak > ubLimitBytes) {
        tiles[i] = saved;
      } else {
        peak = newPeak;
      }
    }
  }

  for (size_t i = 0; i < n; ++i) {
    unsigned tile = saturateToTileValue(tiles[i]);
    plan.outerTiles[searchAxisOrder[i]] = tile;
    plan.innerTiles[searchAxisOrder[i]] = tile;
  }
  return true;
}

int64_t getBroadcastSuffixPointRows(const NpuBandContext &ctx, size_t suffixStart, int64_t tileBudget) {
  constexpr int64_t kBroadcastSuffixPointRows = 3;
  constexpr int64_t kBroadcastCloneInnerBlocksFactor = 5;
  constexpr int64_t kBroadcastCloneUbDivisor = 4;
  int64_t suffixElems = getSliceProduct(ctx.extents, suffixStart, ctx.extents.size());
  bool broadcastCloneLike = ctx.extents.back() >= ctx.targetBlocks * kBroadcastCloneInnerBlocksFactor &&
                            suffixElems >= std::max<int64_t>(tileBudget / kBroadcastCloneUbDivisor, 1);
  return broadcastCloneLike ? kBroadcastSuffixPointRows : kSuffixPreservePointRows;
}

// Scan the loop body for `reduction_type` attribute markers and resolve the
// dominant reduction direction for `loop`. Mirrors the apply-side helper of the
// same name in LoopTiling.cpp so the strategy can decide which reduction shapes
// are eligible for multi-dim vectorization.
ReduceDirection inferReduceDirection(scf::ForOp loop) {
  if (!loop) {
    return ReduceDirection::UNKNOWN;
  }
  ReduceDirection result = ReduceDirection::UNKNOWN;
  loop.getBody()->walk([&result](Operation *op) {
    auto typeAttr = op->getAttrOfType<StringAttr>(kReductionTypeStr);
    if (!typeAttr) {
      return WalkResult::advance();
    }
    StringRef typeStr = typeAttr.getValue();
    if (typeStr == "all") {
      result = ReduceDirection::ALL;
      return WalkResult::interrupt();
    }
    if (typeStr == "y") {
      result = ReduceDirection::Y;
      return WalkResult::advance();
    }
    if (typeStr == "x" && result == ReduceDirection::UNKNOWN) {
      result = ReduceDirection::X;
    }
    return WalkResult::advance();
  });
  return result;
}

// The current vectorization path cannot multi-dim vectorize an innermost
// `reduce_y` (the reduction axis is *not* the contiguous one). When this is
// detected the reduce plan must fall back to legacy single-dim tiling.
bool isInnermostAxisReduceY(const NpuBandContext &ctx) {
  if (ctx.axes.empty()) {
    return false;
  }
  const AxisPtr &innerAxis = ctx.axes.back();
  if (!innerAxis || !isReductionAxis(innerAxis)) {
    return false;
  }
  Operation *loopOp = innerAxis->getLoopOperation();
  auto forOp = dyn_cast_or_null<scf::ForOp>(loopOp);
  if (!forOp) {
    return false;
  }
  return inferReduceDirection(forOp) == ReduceDirection::Y;
}

bool hasReduceYAxis(const NpuBandContext &ctx) {
  for (const AxisPtr &axis : ctx.axes) {
    Operation *loopOp = axis ? axis->getLoopOperation() : nullptr;
    auto forOp = dyn_cast_or_null<scf::ForOp>(loopOp);
    if (isReductionAxis(axis) && forOp && inferReduceDirection(forOp) == ReduceDirection::Y) {
      return true;
    }
  }
  return false;
}

// Pre-compute per-axis reduction labels for the multi-vec greedy search.
// Reduction axes are *always* eligible for vec (their inner tile becomes the
// per-vec-instruction chunk), even when their tile must be shrunk to fit UB.
SmallVector<bool, kSmallVectorSizeSix> collectReductionAxisMask(const NpuBandContext &ctx) {
  SmallVector<bool, kSmallVectorSizeSix> mask(ctx.axes.size(), false);
  for (size_t i = 0; i < ctx.axes.size(); ++i) {
    mask[i] = ctx.axes[i] ? isReductionAxis(ctx.axes[i]) : false;
  }
  return mask;
}

void markUnitAxesInsideMultiVecChain(BandTilePlan &plan) {
  bool hasOuterVecAxis = false;
  for (size_t i = 0; i < plan.innerTiles.size(); ++i) {
    if (hasOuterVecAxis && plan.innerTiles[i] == 1) {
      plan.multiVecAxisMask[i] = true;
    }
    hasOuterVecAxis |= plan.multiVecAxisMask[i];
  }
}

// Multi-vec scheme: from the physical vector axis outward, try to set each
// axis' inner tile equal to its outer tile (full block-local chunk). On UB
// pressure we shrink the current axis with `findMaxFittingTile` and stop
// expanding further outward. Returns a per-axis mask marking which axes should
// receive `kMultiVecLoopAttr` on their point loop.
void computeInnerTilesVecGreedy(const NpuBandContext &ctx, BandTilePlan &plan, ArrayRef<bool> reductionAxisMask,
                                std::optional<size_t> vectorAxis = std::nullopt) {
  const size_t n = ctx.axes.size();
  plan.innerTiles.assign(n, 1);
  plan.multiVecAxisMask.assign(n, false);
  if (n == 0) {
    return;
  }

  size_t primaryAxis = vectorAxis && *vectorAxis < n ? *vectorAxis : n - 1;
  SmallVector<size_t, kSmallVectorSizeSix> searchAxisOrder;
  for (size_t i = primaryAxis; i < n; ++i) {
    searchAxisOrder.push_back(i);
  }
  for (size_t i = primaryAxis; i > 0; --i) {
    searchAxisOrder.push_back(i - 1);
  }

  SmallVector<int64_t, kSmallVectorSizeSix> axisTiles(n, 1);
  bool stopped = false;
  bool hasInnerVecAxis = false;
  for (size_t idx : searchAxisOrder) {
    if (stopped) {
      break;
    }
    int64_t candidate = std::max<int64_t>(static_cast<int64_t>(plan.outerTiles[idx]), 1);
    bool isPrimaryAxis = idx == primaryAxis;
    if (!isPrimaryAxis) {
      int64_t fitTile = findMaxFittingCommonFactorTile(ctx, axisTiles, idx, candidate);
      plan.innerTiles[idx] = saturateToTileValue(fitTile);
      axisTiles[idx] = fitTile;
      plan.multiVecAxisMask[idx] = (fitTile > 1) || reductionAxisMask[idx];
      hasInnerVecAxis |= plan.multiVecAxisMask[idx];
      if (candidate > 1 && fitTile < candidate) {
        stopped = true;
      }
      continue;
    }

    axisTiles[idx] = candidate;
    if (fitsVectorUb(ctx, axisTiles)) {
      plan.innerTiles[idx] = saturateToTileValue(candidate);
      plan.multiVecAxisMask[idx] = (candidate > 1) || reductionAxisMask[idx] || isPrimaryAxis;
      hasInnerVecAxis |= plan.multiVecAxisMask[idx];
      continue;
    }

    axisTiles[idx] = 1;
    int64_t fitTile = std::max<int64_t>(findMaxFittingTile(ctx, axisTiles, idx, candidate), 1);
    if (fitTile > 1) {
      plan.innerTiles[idx] = saturateToTileValue(fitTile);
      axisTiles[idx] = fitTile;
      plan.multiVecAxisMask[idx] = true;
      hasInnerVecAxis = true;
    } else if (hasInnerVecAxis) {
      plan.innerTiles[idx] = saturateToTileValue(candidate);
      plan.multiVecAxisMask[idx] = false;
    } else {
      plan.innerTiles[idx] = 1;
      plan.multiVecAxisMask[idx] = false;
    }
    stopped = true;
  }
  // Keep the multi-vec axis chain contiguous: an inner axis with innerTile==1 sitting
  // between two vectorized axes must still carry the marker so the apply phase emits a
  // length-1 vector slot (shapes like [10, 1, 10]) instead of breaking the chain.
  markUnitAxesInsideMultiVecChain(plan);
}

bool hasActiveVectorSuffix(const BandTilePlan &plan, size_t suffixStart) {
  for (size_t i = suffixStart; i < plan.innerTiles.size(); ++i) {
    if (plan.innerTiles[i] <= 1) {
      return false;
    }
  }
  return true;
}

bool tryBuildTargetFirstMultiVecPlan(const NpuBandContext &ctx, BandTilePlan &plan, size_t targetAxis) {
  if (targetAxis + 1 >= ctx.axes.size()) {
    return false;
  }
  initWholeBandPlan(ctx, plan);
  computeInnerTilesVecGreedy(ctx, plan, collectReductionAxisMask(ctx), targetAxis);
  if (!hasActiveVectorSuffix(plan, targetAxis)) {
    return false;
  }
  plan.usesMultiVecScheme = true;
  return true;
}

std::optional<std::pair<size_t, size_t>> getPrefixTransposeDiffRange(const TransposeVectorInfo &info) {
  if (info.sourceAxisOrder.size() < kMinPrefixTransposeDiffSize ||
      info.sourceAxisOrder.size() != info.targetAxisOrder.size()) {
    return std::nullopt;
  }

  size_t firstDiff = info.sourceAxisOrder.size();
  size_t lastDiff = 0;
  for (size_t i = 0; i < info.sourceAxisOrder.size(); ++i) {
    if (info.sourceAxisOrder[i] == info.targetAxisOrder[i]) {
      continue;
    }
    if (firstDiff == info.sourceAxisOrder.size()) {
      firstDiff = i;
    }
    lastDiff = i;
  }
  if (firstDiff == info.sourceAxisOrder.size() || lastDiff + 1 >= info.sourceAxisOrder.size()) {
    return std::nullopt;
  }
  return std::make_pair(firstDiff, lastDiff);
}

void preserveDegeneratePrefixTransposeAxes(const NpuBandContext &ctx, BandTilePlan &plan) {
  if (ctx.transposeInfos.empty() || plan.innerTiles.size() != plan.multiVecAxisMask.size()) {
    return;
  }

  auto isAxisMarked = [&plan](size_t axisIdx) {
    return axisIdx < plan.multiVecAxisMask.size() && plan.multiVecAxisMask[axisIdx];
  };
  auto markUnitNonReductionAxis = [&plan, &ctx](size_t axisIdx) {
    if (axisIdx >= plan.innerTiles.size() || axisIdx >= ctx.axes.size()) {
      return;
    }
    if (plan.innerTiles[axisIdx] != 1 || isReductionAxis(ctx.axes[axisIdx])) {
      return;
    }
    plan.multiVecAxisMask[axisIdx] = true;
  };

  for (const TransposeVectorInfo &info : ctx.transposeInfos) {
    std::optional<std::pair<size_t, size_t>> diffRange = getPrefixTransposeDiffRange(info);
    if (!diffRange) {
      continue;
    }
    size_t firstDiff = diffRange->first;
    size_t lastDiff = diffRange->second;

    bool hasActivePermutedAxis = false;
    for (size_t dim = firstDiff; dim <= lastDiff; ++dim) {
      hasActivePermutedAxis |= isAxisMarked(info.sourceAxisOrder[dim]) || isAxisMarked(info.targetAxisOrder[dim]);
    }
    if (!hasActivePermutedAxis) {
      continue;
    }

    bool hasActiveCommonInnerSuffix = false;
    for (size_t dim = lastDiff + 1; dim < info.sourceAxisOrder.size(); ++dim) {
      if (info.sourceAxisOrder[dim] != info.targetAxisOrder[dim]) {
        hasActiveCommonInnerSuffix = false;
        break;
      }
      hasActiveCommonInnerSuffix |= isAxisMarked(info.sourceAxisOrder[dim]);
    }
    if (!hasActiveCommonInnerSuffix) {
      continue;
    }

    for (size_t dim = firstDiff; dim <= lastDiff; ++dim) {
      markUnitNonReductionAxis(info.sourceAxisOrder[dim]);
      markUnitNonReductionAxis(info.targetAxisOrder[dim]);
    }
  }
}

bool hasAxisAttr(const AxisPtr &axis, StringRef attrName) {
  Operation *loopOp = axis ? axis->getLoopOperation() : nullptr;
  return (loopOp != nullptr) && loopOp->hasAttr(attrName);
}

bool hasRuntimeExtent(const AxisPtr &axis) { return !axis || isDynamicAxis(axis) || !axis->hasConstantBounds(); }

std::optional<size_t> findOutermostTransposeAxis(const NpuBandContext &ctx) {
  for (size_t i = 0; i < ctx.axes.size(); ++i) {
    if (isTransposeAxis(ctx.axes[i])) {
      return i;
    }
  }
  return std::nullopt;
}

std::optional<size_t> findReduceYVectorTargetAxis(const NpuBandContext &ctx) {
  if (!isInnermostAxisReduceY(ctx) || ctx.axes.size() < kMinTransposeAxisOrderSize) {
    return std::nullopt;
  }
  for (int64_t i = static_cast<int64_t>(ctx.axes.size()) - static_cast<int64_t>(kMinTransposeAxisOrderSize); i >= 0;
       --i) {
    size_t axisIdx = static_cast<size_t>(i);
    if (!isReductionAxis(ctx.axes[axisIdx])) {
      return axisIdx;
    }
  }
  return std::nullopt;
}

std::optional<size_t> findDefaultVectorTargetAxis(const NpuBandContext &ctx) {
  if (ctx.axes.empty()) {
    return std::nullopt;
  }
  size_t innermost = ctx.axes.size() - 1;
  if (isReductionAxis(ctx.axes[innermost])) {
    if (isInnermostAxisReduceY(ctx)) {
      return findReduceYVectorTargetAxis(ctx);
    }
    return innermost;
  }

  for (int64_t i = static_cast<int64_t>(ctx.axes.size()) - 1; i >= 0; --i) {
    size_t axisIdx = static_cast<size_t>(i);
    if (!hasAxisAttr(ctx.axes[axisIdx], kNotInnerDimensionBroadcastLoopAttr)) {
      return axisIdx;
    }
  }
  return std::nullopt;
}

int64_t getAxisSearchUpper(const NpuBandContext &ctx, size_t axisIdx) {
  if (axisIdx >= ctx.axes.size() || hasRuntimeExtent(ctx.axes[axisIdx])) {
    int64_t alignUnit = axisIdx < ctx.axesAlignUnits.size() ? std::max<int64_t>(ctx.axesAlignUnits[axisIdx], 1) : 1;
    return std::max<int64_t>(ctx.vectorUbCapacityElems, alignUnit);
  }
  int64_t extent = axisIdx < ctx.extents.size() ? ctx.extents[axisIdx] : 1;
  return std::max<int64_t>(extent, 1);
}

void initDynamicSingleAxisPlan(const NpuBandContext &ctx, BandTilePlan &plan) {
  plan.outerTiles.assign(ctx.axes.size(), 1);
  plan.innerTiles.assign(ctx.axes.size(), 1);
  plan.multiVecAxisMask.assign(ctx.axes.size(), false);
  plan.usesMultiVecScheme = false;

  for (size_t i = 0; i < ctx.axes.size(); ++i) {
    const AxisPtr &axis = ctx.axes[i];
    bool hasRuntime = hasRuntimeExtent(axis);
    if (isReductionAxis(axis) && !hasRuntime) {
      continue;
    }
    plan.outerTiles[i] =
      hasRuntime ? static_cast<unsigned>(UINT_MAX) : saturateToTileValue(std::max<int64_t>(ctx.extents[i], 1));
  }
}

int64_t findMaxFittingTransposeTargetTile(const NpuBandContext &ctx, const BandTilePlan &plan,
                                          ArrayRef<size_t> searchAxisOrder, size_t targetAxis, int64_t ubLimitBytes) {
  auto targetPosIt = llvm::find(searchAxisOrder, targetAxis);
  if (targetPosIt == searchAxisOrder.end()) {
    return 1;
  }
  size_t targetPos = static_cast<size_t>(targetPosIt - searchAxisOrder.begin());

  SmallVector<int64_t, kSmallVectorSizeSix> axisAlignSize;
  SmallVector<int64_t, kSmallVectorSizeSix> axisMaxTileSize;
  SmallVector<int64_t, kSmallVectorSizeSix> searchShape;
  axisAlignSize.reserve(searchAxisOrder.size());
  axisMaxTileSize.reserve(searchAxisOrder.size());
  searchShape.reserve(searchAxisOrder.size());

  for (size_t axisIdx : searchAxisOrder) {
    int64_t alignUnit = axisIdx < ctx.axesAlignUnits.size() ? std::max<int64_t>(ctx.axesAlignUnits[axisIdx], 1) : 1;
    int64_t maxTile = (axisIdx == targetAxis) ? alignUpInt64(getAxisSearchUpper(ctx, axisIdx), alignUnit) : alignUnit;
    axisAlignSize.push_back(alignUnit);
    axisMaxTileSize.push_back(std::max<int64_t>(maxTile, alignUnit));
    searchShape.push_back(alignUnit);
  }

  int64_t targetAlign = axisAlignSize[targetPos];
  int64_t upperTile = axisMaxTileSize[targetPos];
  auto peakOf = [&ctx, &plan, &searchAxisOrder, &searchShape, targetAlign, targetPos](int64_t targetTile) {
    searchShape[targetPos] = std::max<int64_t>(targetTile, targetAlign);
    return computeTransposeSearchCandidatePeak(ctx, plan, searchAxisOrder, searchShape);
  };

  int64_t peak = peakOf(upperTile);
  if (peak <= ubLimitBytes) {
    return upperTile;
  }

  int64_t highCell = std::max<int64_t>(upperTile / targetAlign, 1);
  int64_t low = 1;
  int64_t high = highCell - 1;
  int64_t bestCell = 1;
  bool useLinearScan = peak == static_cast<int64_t>(LLONG_MAX);

  while (!useLinearScan && low <= high) {
    int64_t cell = low + (high - low) / 2;
    int64_t candidate = cell * targetAlign;
    int64_t candidatePeak = peakOf(candidate);
    if (candidatePeak == static_cast<int64_t>(LLONG_MAX)) {
      useLinearScan = true;
      break;
    }
    if (candidatePeak <= ubLimitBytes) {
      bestCell = cell;
      low = cell + 1;
    } else {
      high = cell - 1;
    }
  }

  if (useLinearScan) {
    searchShape[targetPos] = upperTile;
    peak = computeTransposeSearchCandidatePeak(ctx, plan, searchAxisOrder, searchShape);
    while (peak > ubLimitBytes && searchShape[targetPos] > targetAlign) {
      searchShape[targetPos] -= targetAlign;
      peak = computeTransposeSearchCandidatePeak(ctx, plan, searchAxisOrder, searchShape);
    }
  } else {
    searchShape[targetPos] = bestCell * targetAlign;
    peak = computeTransposeSearchCandidatePeak(ctx, plan, searchAxisOrder, searchShape);
  }

  if (peak > ubLimitBytes) {
    emitAlignedTransposeMinTileExceedsUb(
      {ctx, searchAxisOrder, axisAlignSize, axisMaxTileSize, searchShape, ubLimitBytes, peak});
  }
  return std::max<int64_t>(searchShape[targetPos], targetAlign);
}

bool tryBuildDynamicTransposeSingleAxisPlan(const NpuBandContext &ctx, BandTilePlan &plan) {
  std::optional<size_t> outermostTransposeIdx = findOutermostTransposeAxis(ctx);
  if (!ctx.hasDynamicAxis || !outermostTransposeIdx || ctx.axes.size() < kMinTransposeAxisOrderSize) {
    return false;
  }

  NpuBandContext transposeCtx = ctx;
  int64_t elemBytes =
    std::max<int64_t>(ceilDivInt64(std::max<int64_t>(transposeCtx.transposeElementBits, 1), kBitsPerByte), 1);
  transposeCtx.axesAlignUnits.back() =
    std::lcm(transposeCtx.axesAlignUnits.back(), std::max<int64_t>(kUbAlignBytes / elemBytes, 1));

  initDynamicSingleAxisPlan(transposeCtx, plan);

  SmallVector<size_t, kSmallVectorSizeSix> searchAxisOrder;
  for (size_t i = *outermostTransposeIdx; i < transposeCtx.axes.size(); ++i) {
    searchAxisOrder.push_back(i);
    plan.innerTiles[i] =
      saturateToTileValue(i < transposeCtx.axesAlignUnits.size() ? transposeCtx.axesAlignUnits[i] : 1);
  }

  size_t targetAxis = searchAxisOrder.back();
  int64_t targetTile =
    findMaxFittingTransposeTargetTile(transposeCtx, plan, searchAxisOrder, targetAxis, getVectorUbBytes(transposeCtx));
  plan.innerTiles[targetAxis] = saturateToTileValue(targetTile);
  return true;
}

bool tryBuildDynamicSingleAxisVectorPlan(const NpuBandContext &ctx, BandTilePlan &plan) {
  if (!ctx.hasDynamicAxis || ctx.axes.empty()) {
    return false;
  }
  if (findOutermostTransposeAxis(ctx)) {
    return tryBuildDynamicTransposeSingleAxisPlan(ctx, plan);
  }

  std::optional<size_t> targetAxis = findDefaultVectorTargetAxis(ctx);
  if (!targetAxis) {
    return false;
  }

  initDynamicSingleAxisPlan(ctx, plan);
  SmallVector<int64_t, kSmallVectorSizeSix> axisTiles(ctx.axes.size(), 1);
  int64_t tile = findMaxFittingTile(ctx, axisTiles, *targetAxis, getAxisSearchUpper(ctx, *targetAxis));
  plan.innerTiles[*targetAxis] = saturateToTileValue(tile);
  return true;
}

// Attach `kMultiVecLoopAttr` to the original (un-tiled) loop ops of axes that
// participate in multi-dim vectorization. `copySemanticLoopAttrsToPointLoop`
// later carries the marker to the resulting point loops, which is what the
// post-tiling sink / consume passes look for.
// The dynamic-shape pipeline re-enters the strategy for the same band twice
// (memref-size pre-pass + actual apply); clearing stale markers first keeps
// the second pass authoritative when the chosen plan changes.
void tagMultiVecLoops(const SmallVector<AxisPtr, kSmallVectorSizeFour> &bandAxes, const BandTilePlan &plan) {
  for (const AxisPtr &axis : bandAxes) {
    Operation *loopOp = axis ? axis->getLoopOperation() : nullptr;
    if ((loopOp != nullptr) && loopOp->hasAttr(kMultiVecLoopAttr)) {
      loopOp->removeAttr(kMultiVecLoopAttr);
    }
  }
  if (!plan.usesMultiVecScheme) {
    return;
  }
  for (size_t i = 0; i < bandAxes.size() && i < plan.multiVecAxisMask.size(); ++i) {
    if (!plan.multiVecAxisMask[i]) {
      continue;
    }
    Operation *loopOp = bandAxes[i] ? bandAxes[i]->getLoopOperation() : nullptr;
    if (loopOp == nullptr) {
      continue;
    }
    OpBuilder builder(loopOp);
    loopOp->setAttr(kMultiVecLoopAttr, builder.getUnitAttr());
  }
}

bool tryBuildReductionSuffixPlan(const NpuBandContext &ctx, BandTilePlan &plan) {
  if (ctx.hasDynamicAxis || !ctx.hasReduction || !ctx.lastAxisIsReduction ||
      ctx.graphTemplate == GraphTemplate::TRANSPOSE_OP) {
    return false;
  }
  bool singleReductionAxis = ctx.axes.size() == 1;
  if (!singleReductionAxis && ctx.axes.size() < kMinTransposeAxisOrderSize) {
    return false;
  }

  size_t suffixStart = computeReductionSuffixStart(ctx);
  if (suffixStart == 0 && !singleReductionAxis) {
    return false;
  }

  if (!isInnermostAxisReduceY(ctx)) {
    initWholeBandPlan(ctx, plan);
    computeInnerTilesVecGreedy(ctx, plan, collectReductionAxisMask(ctx));
    preserveDegeneratePrefixTransposeAxes(ctx, plan);
    plan.usesMultiVecScheme = true;
    return true;
  }

  if (std::optional<size_t> targetAxis = findReduceYVectorTargetAxis(ctx)) {
    if (tryBuildTargetFirstMultiVecPlan(ctx, plan, *targetAxis)) {
      return true;
    }
  }

  initWholeBandPlan(ctx, plan);

  size_t innermost = ctx.extents.size() - 1;
  SmallVector<int64_t, kSmallVectorSizeFour> axisTiles(plan.innerTiles.begin(), plan.innerTiles.end());
  for (size_t i = 0; i < suffixStart; ++i) {
    axisTiles[i] = 1;
  }
  unsigned innerTile = saturateToTileValue(findMaxFittingTile(ctx, axisTiles, innermost, ctx.extents[innermost]));
  plan.outerTiles[innermost] = innerTile;
  plan.innerTiles[innermost] = innerTile;
  return true;
}

bool tryBuildTransposePlan(const NpuBandContext &ctx, BandTilePlan &plan) {
  if (ctx.hasDynamicAxis || ctx.axes.size() < kMinTransposeAxisOrderSize) {
    return false;
  }

  int64_t outermostTransposeIdx = kInvalidAxisIndex;
  for (size_t i = 0; i < ctx.axes.size(); ++i) {
    if (isTransposeAxis(ctx.axes[i])) {
      outermostTransposeIdx = static_cast<int64_t>(i);
      break;
    }
  }
  if (outermostTransposeIdx < 0) {
    return false;
  }

  NpuBandContext transposeCtx = ctx;
  int64_t elemBytes =
    std::max<int64_t>(ceilDivInt64(std::max<int64_t>(transposeCtx.transposeElementBits, 1), kBitsPerByte), 1);
  transposeCtx.axesAlignUnits.back() =
    std::lcm(transposeCtx.axesAlignUnits.back(), std::max<int64_t>(kUbAlignBytes / elemBytes, 1));

  initWholeBandPlan(ctx, plan);
  unsigned firstAxisTile = saturateToTileValue(ceilDivInt64(ctx.extents.front(), ctx.targetBlocks));
  plan.outerTiles.front() = firstAxisTile;
  plan.innerTiles.front() = firstAxisTile;
  int64_t ubLimitBytes = getVectorUbBytes(ctx);
  SmallVector<size_t, kSmallVectorSizeSix> searchAxisOrder;
  for (auto i = static_cast<size_t>(outermostTransposeIdx); i < ctx.axes.size(); ++i) {
    searchAxisOrder.push_back(i);
  }
  // NPU vectorization materializes the full suffix starting at the outermost
  // transpose axis; non-transpose suffix axes participate in UB search with align=1.
  chooseAlignedTransposeTiles(transposeCtx, plan, searchAxisOrder, ubLimitBytes);
  return true;
}

bool tryBuildBroadcastSuffixPlan(const NpuBandContext &ctx, BandTilePlan &plan) {
  if (ctx.hasDynamicAxis || ctx.hasReduction || ctx.graphTemplate != GraphTemplate::BROADCAST_OP ||
      ctx.axes.size() < kMinTransposeAxisOrderSize) {
    return false;
  }

  if (std::optional<size_t> targetAxis = findDefaultVectorTargetAxis(ctx)) {
    if (tryBuildTargetFirstMultiVecPlan(ctx, plan, *targetAxis)) {
      return true;
    }
  }

  int64_t broadcastTileBudget = ubCapacityForVectorTile(ctx);
  size_t suffixStart = computeBroadcastSuffixStart(ctx, broadcastTileBudget);
  if (suffixStart == 0 || suffixStart >= ctx.axes.size() - 1) {
    return false;
  }

  initWholeBandPlan(ctx, plan);
  SmallVector<unsigned, kSmallVectorSizeFour> prefixPlaceholder(plan.outerTiles.begin(),
                                                                plan.outerTiles.begin() + suffixStart);
  fillPrefixSuffixTiles(ctx, prefixPlaceholder, getBroadcastSuffixPointRows(ctx, suffixStart, broadcastTileBudget),
                        plan);

  SmallVector<int64_t, kSmallVectorSizeFour> axisTiles(plan.innerTiles.begin(), plan.innerTiles.end());
  for (int64_t i = static_cast<int64_t>(ctx.axes.size()) - 1; i >= 0; --i) {
    auto idx = static_cast<size_t>(i);
    Operation *loopOp = ctx.axes[idx] ? ctx.axes[idx]->getLoopOperation() : nullptr;
    if ((loopOp == nullptr) ||
        (!loopOp->hasAttr(kBroadcastLoopAttr) && !loopOp->hasAttr(kNotInnerDimensionBroadcastLoopAttr))) {
      break;
    }
    int64_t extent = ctx.extents[idx];
    unsigned tile = saturateToTileValue(findMaxFittingTile(ctx, axisTiles, idx, extent));
    plan.outerTiles[idx] = tile;
    plan.innerTiles[idx] = tile;
  }
  return true;
}

bool tryBuildElementwisePlan(const NpuBandContext &ctx, BandTilePlan &plan) {
  if (ctx.hasDynamicAxis || ctx.hasReduction || ctx.axes.empty()) {
    return false;
  }

  initWholeBandPlan(ctx, plan);
  computeInnerTilesVecGreedy(ctx, plan, collectReductionAxisMask(ctx));
  plan.usesMultiVecScheme = true;
  return true;
}

bool isParallelCandidateAxis(const NpuBandContext &ctx, size_t axisIdx) {
  if (axisIdx >= ctx.axes.size()) {
    return false;
  }
  const AxisPtr &axis = ctx.axes[axisIdx];
  if (!axis || isReductionAxis(axis) || isDynamicAxis(axis) || !axis->hasConstantBounds() ||
      ctx.extents[axisIdx] <= 1) {
    return false;
  }
  Operation *loopOp = axis->getLoopOperation();
  return (loopOp == nullptr) || !loopOp->hasAttr(kNotInnerDimensionBroadcastLoopAttr);
}

bool isStrictPerfectAxisEdge(const NpuBandContext &ctx, size_t parentIdx) {
  if (parentIdx + 1 >= ctx.axes.size() || !ctx.axes[parentIdx] || !ctx.axes[parentIdx + 1]) {
    return false;
  }
  Operation *parentOp = ctx.axes[parentIdx]->getLoopOperation();
  Operation *childOp = ctx.axes[parentIdx + 1]->getLoopOperation();
  auto parentLoop = dyn_cast_or_null<scf::ForOp>(parentOp);
  if (!parentLoop || (childOp == nullptr)) {
    return false;
  }

  bool sawChild = false;
  for (Operation &op : parentLoop.getBody()->without_terminator()) {
    if (&op == childOp) {
      if (sawChild) {
        return false;
      }
      sawChild = true;
      continue;
    }
    return false;
  }
  return sawChild;
}

SmallVector<size_t, kSmallVectorSizeSix> collectParallelPrefixAxes(const NpuBandContext &ctx) {
  SmallVector<size_t, kSmallVectorSizeSix> axes;
  for (size_t i = 0; i < ctx.axes.size(); ++i) {
    if (!isParallelCandidateAxis(ctx, i)) {
      break;
    }
    axes.push_back(i);
    if (i + 1 == ctx.axes.size() || !isStrictPerfectAxisEdge(ctx, i)) {
      break;
    }
  }
  return axes;
}

void alignPlanTiles(const NpuBandContext &ctx, BandTilePlan &plan) {
  for (size_t i = 0; i < ctx.axes.size(); ++i) {
    int64_t alignUnit = i < ctx.axesAlignUnits.size() ? std::max<int64_t>(ctx.axesAlignUnits[i], 1) : 1;
    if (alignUnit <= 1) {
      continue;
    }
    plan.outerTiles[i] = saturateToTileValue(alignUpInt64(std::max<int64_t>(plan.outerTiles[i], 1), alignUnit));
    plan.innerTiles[i] = saturateToTileValue(alignUpInt64(std::max<int64_t>(plan.innerTiles[i], 1), alignUnit));
  }
}

bool tryBuildSmallMathLimitedCorePlan(const NpuBandContext &ctx, BandTilePlan &plan) {
  constexpr int64_t kSmallMathTargetCores = 3;
  int64_t minMathScore = ctx.hasReduction ? kReduceMathComplexityScore : kMediumMathComplexityScore;
  if (ctx.hasDynamicAxis || ctx.axes.empty() || ctx.graphTemplate == GraphTemplate::TRANSPOSE_OP ||
      ctx.graphTemplate == GraphTemplate::BROADCAST_OP || ctx.mathComplexityScore < minMathScore ||
      (ctx.hasReduction && hasReduceYAxis(ctx))) {
    return false;
  }
  initWholeBandPlan(ctx, plan);
  alignPlanTiles(ctx, plan);
  SmallVector<int64_t, kSmallVectorSizeSix> axisTiles(plan.innerTiles.begin(), plan.innerTiles.end());
  if (!fitsVectorUb(ctx, axisTiles)) {
    return false;
  }
  for (size_t i = 0; i < ctx.extents.size(); ++i) {
    if (isReductionAxis(ctx.axes[i])) {
      continue;
    }
    int64_t extent = std::max<int64_t>(ctx.extents[i], 1);
    int64_t alignUnit = i < ctx.axesAlignUnits.size() ? std::max<int64_t>(ctx.axesAlignUnits[i], 1) : 1;
    int64_t targetCoreTile = alignUpInt64(ceilDivInt64(extent, kSmallMathTargetCores), alignUnit);
    if (targetCoreTile < extent) {
      plan.outerTiles[i] = saturateToTileValue(targetCoreTile);
      plan.innerTiles[i] = plan.outerTiles[i];
      break;
    }
  }
  plan.multiVecAxisMask.assign(ctx.axes.size(), true);
  plan.usesMultiVecScheme = true;
  plan.preserveWholeBandTiles = true;
  return true;
}

bool isParallelPrefixVectorAxis(const NpuBandContext &ctx, const BandTilePlan &plan, size_t axisIdx) {
  if (axisIdx < plan.multiVecAxisMask.size()) {
    return plan.multiVecAxisMask[axisIdx];
  }
  return axisIdx + 1 == ctx.axes.size();
}

int64_t countParallelPrefixTasks(const NpuBandContext &ctx, const BandTilePlan &plan, ArrayRef<size_t> parallelAxes) {
  int64_t tasks = 1;
  for (size_t axisIdx : parallelAxes) {
    int64_t extent = std::max<int64_t>(ctx.extents[axisIdx], 1);
    int64_t tile = std::max<int64_t>(plan.outerTiles[axisIdx], 1);
    tasks = multiplyAndCap(tasks, ceilDivInt64(extent, tile));
  }
  return tasks;
}

void syncParallelPrefixInnerOuterTiles(const NpuBandContext &ctx, BandTilePlan &plan) {
  for (size_t i = 0; i < ctx.axes.size(); ++i) {
    if (!isParallelPrefixVectorAxis(ctx, plan, i)) {
      plan.innerTiles[i] = plan.outerTiles[i];
    }
    plan.outerTiles[i] = plan.innerTiles[i];
  }
}

void compensateParallelPrefixTaskCount(const NpuBandContext &ctx, BandTilePlan &plan, ArrayRef<size_t> parallelAxes) {
  int64_t coreNum = std::max<int64_t>(ctx.targetBlocks, 1);
  int64_t targetTasks = multiplyAndCap(coreNum, 10);
  int64_t tasks = countParallelPrefixTasks(ctx, plan, parallelAxes);
  if (tasks >= targetTasks || tasks % coreNum == 0) {
    return;
  }

  for (size_t axisIdx : parallelAxes) {
    int64_t extent = std::max<int64_t>(ctx.extents[axisIdx], 1);
    int64_t currentTile = std::max<int64_t>(plan.outerTiles[axisIdx], 1);
    if (isParallelPrefixVectorAxis(ctx, plan, axisIdx) || currentTile <= 1) {
      continue;
    }
    int64_t blocks = ceilDivInt64(extent, currentTile);
    int64_t otherTasks = std::max<int64_t>(tasks / blocks, 1);
    int64_t needBlocks = ceilDivInt64(targetTasks, otherTasks);
    if (needBlocks <= blocks) {
      continue;
    }

    int64_t tile = std::max<int64_t>(extent / needBlocks, 1);
    int64_t alignUnit = axisIdx < ctx.axesAlignUnits.size() ? std::max<int64_t>(ctx.axesAlignUnits[axisIdx], 1) : 1;
    if (alignUnit > 1 && tile > 1) {
      tile = std::max<int64_t>((tile / alignUnit) * alignUnit, 1);
    }
    if (tile >= currentTile) {
      continue;
    }

    plan.outerTiles[axisIdx] = plan.innerTiles[axisIdx] = saturateToTileValue(tile);
    tasks = multiplyAndCap(otherTasks, ceilDivInt64(extent, tile));
    if (tasks >= targetTasks || tasks % coreNum == 0) {
      break;
    }
  }
}

void adjustParallelPrefixOuterTiles(const NpuBandContext &ctx, BandTilePlan &plan) {
  if (ctx.hasDynamicAxis || plan.preserveWholeBandTiles || plan.outerTiles.size() < ctx.axes.size() ||
      plan.innerTiles.size() < ctx.axes.size()) {
    return;
  }
  alignPlanTiles(ctx, plan);

  SmallVector<size_t, kSmallVectorSizeSix> parallelAxes = collectParallelPrefixAxes(ctx);
  if (parallelAxes.empty()) {
    return;
  }

  SmallVector<int64_t, kSmallVectorSizeSix> parallelExtents;
  SmallVector<int64_t, kSmallVectorSizeSix> parallelAlignUnits;
  parallelExtents.reserve(parallelAxes.size());
  parallelAlignUnits.reserve(parallelAxes.size());
  for (size_t axisIdx : parallelAxes) {
    parallelExtents.push_back(std::max<int64_t>(ctx.extents[axisIdx], 1));
    parallelAlignUnits.push_back(axisIdx < ctx.axesAlignUnits.size() ? std::max<int64_t>(ctx.axesAlignUnits[axisIdx], 1)
                                                                     : 1);
  }
  SmallVector<unsigned, kSmallVectorSizeFour> outerCaps =
    assignPrefixOuterTiles(parallelExtents, parallelAlignUnits, ctx.targetBlocks);

  for (auto [pos, axisIdx] : llvm::enumerate(parallelAxes)) {
    int64_t alignUnit = axisIdx < ctx.axesAlignUnits.size() ? std::max<int64_t>(ctx.axesAlignUnits[axisIdx], 1) : 1;
    int64_t outerCap = pos < outerCaps.size() ? std::max<int64_t>(outerCaps[pos], 1) : 1;
    outerCap = alignUpInt64(outerCap, alignUnit);
    plan.outerTiles[axisIdx] = saturateToTileValue(outerCap);
  }

  // Greedy multi-vec planning runs before parallel outer caps are applied and may
  // leave innerTile==1 dispatch axes marked. Drop them so tagMultiVecLoops does not
  // emit a degenerate vector=1 chain leg alongside the real vectorized axis.
  for (size_t axisIdx : parallelAxes) {
    if (axisIdx < plan.multiVecAxisMask.size() && plan.innerTiles[axisIdx] == 1) {
      plan.multiVecAxisMask[axisIdx] = false;
    }
  }
  syncParallelPrefixInnerOuterTiles(ctx, plan);
  compensateParallelPrefixTaskCount(ctx, plan, parallelAxes);
}

static unsigned saturateReductionTileValue(int64_t alignUnit, const AxisPtr axis, unsigned tileValue) {
  if (alignUnit > 1 || !isReductionAxis(axis) || isTransposeAxis(axis) || !axis->hasConstantBounds()) {
    return tileValue;
  }
  int64_t fullExtent = axis->getConstantUpperBound() - axis->getConstantLowerBound();
  if (static_cast<int64_t>(tileValue) < fullExtent) {
    return tileValue;
  }
  return saturateToTileValue(std::max<int64_t>(fullExtent, 1));
}

void applyBandTilePlan(const NpuBandContext &ctx, const SmallVector<AxisPtr, kSmallVectorSizeFour> &bandAxes,
                       const BandTilePlan &plan, size_t &maxLevelToTile) {
  constexpr size_t kTileLevels = 2;
  for (size_t i = 0; i < bandAxes.size(); ++i) {
    const auto &axis = bandAxes[i];
    while (axis->configs[kTileCfg].size() < kTileLevels) {
      axis->doExtraTile();
    }
    if (axis->configs[kTileCfg].size() > kTileLevels) {
      axis->configs[kTileCfg].resize(kTileLevels);
    }

    for (size_t level = 0; level < kTileLevels; ++level) {
      auto tileConfig = axis->tryGetConfig(static_cast<int>(level), kTileCfg);
      if (tileConfig == nullptr) {
        continue;
      }
      tileConfig->constraints.clear();
      unsigned tileValue = (level == 0) ? plan.outerTiles[i] : plan.innerTiles[i];
      int64_t alignUnit = i < ctx.axesAlignUnits.size() ? std::max<int64_t>(ctx.axesAlignUnits[i], 1) : 1;
      tileValue = saturateReductionTileValue(alignUnit, axis, tileValue);
      tileConfig->value = static_cast<int>(tileValue);
      axis->tryAddConstraint(static_cast<int>(level), Constraint({static_cast<int>(tileValue)}));
    }
  }
  maxLevelToTile = std::max(maxLevelToTile, kTileLevels);
}

void applyDynamicFallbackAxisTiling(const AxisPtr axis, bool isFullTileAxis) {
  for (size_t i = axis->configs[kTileCfg].size(); i < kNpuTileLevels; ++i) {
    axis->doExtraTile();
  }

  auto tileConfig0 = axis->tryGetConfig(0, kTileCfg);
  auto tileConfig1 = axis->tryGetConfig(1, kTileCfg);

  if (isFullTileAxis) {
    if (tileConfig0 != nullptr) {
      tileConfig0->value = 1;
      axis->tryAddConstraint(0, Constraint({1}));
    }
    if (tileConfig1 != nullptr) {
      tileConfig1->value = 1;
      axis->tryAddConstraint(1, Constraint({1}));
    }
    return;
  }

  if (tileConfig0 != nullptr) {
    int value = static_cast<int>(kMaxTileValue);
    tileConfig0->value = value;
    axis->tryAddConstraint(0, Constraint({value}));
  }

  if (tileConfig1 != nullptr) {
    tileConfig1->value = static_cast<int>(kNpuMinInnerTileSize);
    axis->tryAddConstraint(1, Constraint({static_cast<int>(kNpuMinInnerTileSize)}));
  }
}

static void applyFallbackAxisTiling(const AxisPtr axis, const SmallVector<unsigned, kSmallVectorSizeFour> &tileSizes,
                                    unsigned innerTileSize, unsigned blockNumber, size_t &maxLevelToTile,
                                    bool isFullTileAxis) {
  bool hasStaticBounds = axis->hasConstantBounds();
  bool hasDynamicUpperBound = !axis->hasConstantUpperBound();
  if (hasDynamicUpperBound || !hasStaticBounds) {
    applyDynamicFallbackAxisTiling(axis, isFullTileAxis);
    maxLevelToTile = std::max(maxLevelToTile, kNpuTileLevels);
    return;
  }

  SmallVector<unsigned, kSmallVectorSizeFour> usedTileSizes =
    computeFallbackTileSizes(axis, tileSizes, innerTileSize, blockNumber, isFullTileAxis);
  applyTileConfigsToAxis(axis, usedTileSizes, maxLevelToTile);
}
}  // namespace

SmallVector<AxisPtr> NpuDefaultTileStrategy::collectAxes(const NpuModelGraphPtr npuGraph) {
  SmallVector<AxisPtr> axes;
  npuGraph->rootAxis->forEachAxisTopDown([&axes](const AxisPtr axis) {
    if (axis) {
      axes.push_back(axis);
    }
  });
  return axes;
}

SmallVector<unsigned, kSmallVectorSizeFour> NpuDefaultTileStrategy::parseTileSizesConfig(
  const NpuModelGraphPtr npuGraph) {
  SmallVector<unsigned, kSmallVectorSizeFour> tileSizes;
  auto appendTileSizes = [&tileSizes](Attribute attr) {
    auto arrayAttr = dyn_cast_or_null<ArrayAttr>(attr);
    if (arrayAttr) {
      for (auto attr : arrayAttr) {
        if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
          tileSizes.push_back(static_cast<unsigned>(intAttr.getInt()));
        }
      }
    }
  };

  if ((npuGraph->funcOp != nullptr) && npuGraph->funcOp->hasAttr("npu.multiTileSizes")) {
    appendTileSizes(npuGraph->funcOp->getAttr("npu.multiTileSizes"));
  }
  if (tileSizes.empty()) {
    auto tileSizesIt = npuGraph->globalConfigs.find("npu.multiTileSizes");
    if (tileSizesIt != npuGraph->globalConfigs.end()) {
      appendTileSizes(tileSizesIt->second);
    }
  }

  // Ensure tile sizes are in non-increasing order (largest to smallest)
  for (size_t i = 1; i < tileSizes.size(); ++i) {
    if (tileSizes[i] > tileSizes[i - 1]) {
      tileSizes[i] = tileSizes[i - 1];
    }
  }
  return tileSizes;
}

void NpuDefaultTileStrategy::applyTilingToAxes(const NpuModelGraphPtr npuGraph, const SmallVector<AxisPtr> &axes,
                                               const SmallVector<unsigned, kSmallVectorSizeFour> &tileSizes) {
  size_t maxLevelToTile = 1;
  constexpr unsigned innerTileSize = kNpuMinInnerTileSize;
  const unsigned blockNumber =
    (npuGraph && npuGraph->coreNum > 0) ? static_cast<unsigned>(npuGraph->coreNum) : kNpuTargetBlocks;

  if (!tileSizes.empty()) {
    for (const auto &axis : axes) {
      applyFallbackAxisTiling(axis, tileSizes, innerTileSize, blockNumber, maxLevelToTile, isReductionAxis(axis));
    }
    npuGraph->levelToTile = std::max(npuGraph->levelToTile, maxLevelToTile);
    return;
  }

  for (const auto &entry : groupAxesByBand(axes)) {
    const auto &bandIdx = entry.first;
    const auto &bandAxes = entry.second;
    NpuBandContext bandCtx = buildNpuBandContext(npuGraph, bandIdx, bandAxes);
    BandTilePlan bandPlan;
    bool matched = tryBuildTransposePlan(bandCtx, bandPlan) || tryBuildSmallMathLimitedCorePlan(bandCtx, bandPlan) ||
                   tryBuildReductionSuffixPlan(bandCtx, bandPlan) || tryBuildBroadcastSuffixPlan(bandCtx, bandPlan) ||
                   tryBuildElementwisePlan(bandCtx, bandPlan) || tryBuildDynamicSingleAxisVectorPlan(bandCtx, bandPlan);
    if (matched) {
      adjustParallelPrefixOuterTiles(bandCtx, bandPlan);
      applyBandTilePlan(bandCtx, bandAxes, bandPlan, maxLevelToTile);
      tagMultiVecLoops(bandAxes, bandPlan);
      continue;
    }

    for (const auto &axis : bandAxes) {
      applyFallbackAxisTiling(axis, tileSizes, innerTileSize, blockNumber, maxLevelToTile, isReductionAxis(axis));
    }
  }

  npuGraph->levelToTile = std::max(npuGraph->levelToTile, maxLevelToTile);
}

void NpuDefaultTileStrategy::AddNpuConstraint(NpuModelGraphPtr npuGraph) {
  if (npuGraph == nullptr || npuGraph->rootAxis == nullptr) {
    return;
  }

  SmallVector<AxisPtr> axes = collectAxes(npuGraph);
  SmallVector<unsigned, kSmallVectorSizeFour> tileSizes = parseTileSizesConfig(npuGraph);

  applyTilingToAxes(npuGraph, axes, tileSizes);

  if ((npuGraph->funcOp != nullptr) && npuGraph->funcOp->hasAttr("npu.multiTileSizes")) {
    (void)npuGraph->funcOp->removeAttr("npu.multiTileSizes");
  }
}

void TilingStrategyManager::processOn(const ModelGraphPtr modelGraph) {
  modelGraph->name = modelGraph->name + "_AfterStrategy";
  for (auto strategy : this->strategies_) {
    strategy->AddConstraint(modelGraph);
  }
}

void TilingStrategyManager::processOn(const GpuModelGraphPtr gpuGraph) {
  gpuGraph->name = gpuGraph->name + "_AfterStrategy";
  for (auto strategy : this->strategies_) {
    strategy->AddGpuConstraint(gpuGraph);
  }
}

void TilingStrategyManager::processOn(const CpuModelGraphPtr cpuGraph) {
  cpuGraph->name = cpuGraph->name + "_AfterStrategy";
  for (auto strategy : this->strategies_) {
    strategy->AddCpuConstraint(cpuGraph);
  }
}

void TilingStrategyManager::processOn(const NpuModelGraphPtr npuGraph) {
  npuGraph->name = npuGraph->name + "_AfterStrategy";
  for (auto strategy : this->strategies_) {
    strategy->AddNpuConstraint(npuGraph);
  }
}

// Helper function to find the first non-static axis index from innermost
int64_t VectorizationStrategy::findConsecutiveStaticEnd(const SmallVector<AxisPtr> &axes) {
  auto numAxes = static_cast<int64_t>(axes.size());
  for (int64_t dimIdx = numAxes - 1; dimIdx >= 0; --dimIdx) {
    auto axis = axes[static_cast<size_t>(dimIdx)];
    if (!axis) {
      return dimIdx;
    }
    if (axis->axisType.find(mlir::autotiling::Axis::AxisLabel::kDynamic) != axis->axisType.end()) {
      return dimIdx;
    }
  }
  return kInvalidAxisIndex;
}

// Helper function to handle dynamic axis for vectorization
void VectorizationStrategy::handleDynamicAxisForVectorization(const AxisPtr &axis, int pos, int64_t &ubRemainingNum) {
  if (ubRemainingNum > 1) {
    axis->tryAddConstraint(pos, Constraint(1, static_cast<int>(ubRemainingNum), 1));
    auto tileConfig = axis->tryGetConfig(pos, kTileCfg);
    if (tileConfig != nullptr) {
      tileConfig->value = kDynamicTileMarker;
    }
  } else {
    auto tileConfig = axis->tryGetConfig(pos, kTileCfg);
    if (tileConfig != nullptr) {
      tileConfig->value = 1;
    }
  }
}

// Helper function to handle consecutive static axis for vectorization
void VectorizationStrategy::handleConsecutiveStaticAxis(const AxisPtr &axis, int pos, int64_t dimSize,
                                                        int64_t &ubRemainingNum) {
  int64_t tileSize = 1;
  if (ubRemainingNum > 1) {
    if (dimSize <= ubRemainingNum) {
      tileSize = dimSize;
    } else {
      tileSize = ubRemainingNum;
    }
  }

  tileSize = std::max<int64_t>(tileSize, 1);
  auto vectorizationConfig = axis->tryGetConfig(pos, kTileCfg);
  if (vectorizationConfig != nullptr) {
    vectorizationConfig->value = static_cast<int>(tileSize);
  }

  if (axis->isInnerMost) {
    (void)axis->axisType.insert(mlir::autotiling::Axis::AxisLabel::kVectorization);
  }

  if (tileSize > 0) {
    ubRemainingNum = ubRemainingNum / tileSize;
    ubRemainingNum = std::max<int64_t>(ubRemainingNum, 1);
  }
}

// Helper function to handle non-consecutive static axis for vectorization
void VectorizationStrategy::handleNonConsecutiveStaticAxis(const AxisPtr &axis, int pos, int64_t dimSize,
                                                           int64_t &ubRemainingNum) {
  if (ubRemainingNum > 1 && dimSize > 1) {
    int64_t maxTileSize = std::min(ubRemainingNum, dimSize);
    axis->tryAddConstraint(pos, Constraint(1, static_cast<int>(maxTileSize), 1));
    auto tileConfig = axis->tryGetConfig(pos, kTileCfg);
    if (tileConfig != nullptr) {
      tileConfig->value = kDynamicTileMarker;
    }
    ubRemainingNum = ubRemainingNum / maxTileSize;
    ubRemainingNum = std::max<int64_t>(ubRemainingNum, 1);
  } else {
    auto tileConfig = axis->tryGetConfig(pos, kTileCfg);
    if (tileConfig != nullptr) {
      tileConfig->value = 1;
    }
  }
}

void VectorizationStrategy::applyVectorizationTiling(const SmallVector<AxisPtr> &axes, int64_t ubAvailableNum,
                                                     int pos) {
  auto numAxes = static_cast<int64_t>(axes.size());
  int64_t ubRemainingNum = std::max<int64_t>(ubAvailableNum, 1);

  int64_t consecutiveStaticEnd = findConsecutiveStaticEnd(axes);

  // Process axes from inner to outer (reverse order)
  for (int64_t dimIdx = numAxes - 1; dimIdx >= 0; --dimIdx) {
    auto axis = axes[static_cast<size_t>(dimIdx)];
    if (!axis) {
      continue;
    }

    if (pos != 0) {
      axis->doExtraTile();
    }

    bool isDynamic = axis->axisType.find(mlir::autotiling::Axis::AxisLabel::kDynamic) != axis->axisType.end();
    if (isDynamic) {
      handleDynamicAxisForVectorization(axis, pos, ubRemainingNum);
      continue;
    }

    int64_t dimSize = axis->range.second > 0 ? axis->range.second : 1;
    bool isInConsecutiveStatic = (consecutiveStaticEnd == kInvalidAxisIndex) || (dimIdx >= consecutiveStaticEnd);
    if (isInConsecutiveStatic) {
      handleConsecutiveStaticAxis(axis, pos, dimSize, ubRemainingNum);
    } else {
      handleNonConsecutiveStaticAxis(axis, pos, dimSize, ubRemainingNum);
    }
  }
}

void VectorizationStrategy::AddNpuConstraint(NpuModelGraphPtr npuGraph) {
  SmallVector<AxisPtr> axes;
  npuGraph->rootAxis->forEachAxisTopDown([&axes](const AxisPtr a) { axes.push_back(a); });

  if (axes.empty()) {
    return;
  }

  // Get buffer info from NpuModelGraph (populated by buffer analysis)
  int64_t maxBufferCnt = npuGraph->maxBufferCnt;
  auto smallestTypeBits = static_cast<int64_t>(npuGraph->smallestTypeBits);
  int64_t ubSizeInBytes = npuGraph->ubSize;

  // Calculate UB available number in terms of smallest type elements
  // UB available = UB size / (smallest type size * buffer count)
  int64_t ubMaxSizeInBits = ubSizeInBytes * kNumBitsInByte;
  int64_t ubAvailableNumInSmallestType = ubMaxSizeInBits / smallestTypeBits / maxBufferCnt;

  // Align down to alignment boundary
  int64_t alignedBufferSizeInBits = (ubAvailableNumInSmallestType * smallestTypeBits) /
                                    (kUBAlignSizeInBytes * kNumBitsInByte) * (kUBAlignSizeInBytes * kNumBitsInByte);
  ubAvailableNumInSmallestType = alignedBufferSizeInBits / smallestTypeBits;

  // Vectorization is the second tile level, parallel is the first
  applyVectorizationTiling(axes, ubAvailableNumInSmallestType, 1);
  ++npuGraph->tileNum;
}

void ParallelStrategy::collectAxesInfo(const SmallVector<AxisPtr> &axes, int pos) {
  totalParallelSize = 1;
  totalReduceSize = 1;
  isParallelAxis.clear();

  for (const auto &axis : axes) {
    if (!axis) {
      isParallelAxis.push_back(true);
      continue;
    }

    int64_t axisSize = axis->range.second > 0 ? axis->range.second : 1;

    // Check if it's a reduction axis
    bool isReduction = axis->axisType.find(mlir::autotiling::Axis::AxisLabel::kReduction) != axis->axisType.end();
    isParallelAxis.push_back(!isReduction);

    // Consider existing tile configuration
    // Vectorization is at pos=1, so we read from pos directly
    auto tileConfig = axis->tryGetConfig(pos);
    int64_t effectiveSize = axisSize;
    if (tileConfig && tileConfig->value > 0) {
      effectiveSize = (axisSize + tileConfig->value - 1) / tileConfig->value;
    }

    if (isReduction) {
      totalReduceSize *= effectiveSize;
    } else {
      totalParallelSize *= effectiveSize;
    }
  }
}

std::pair<int64_t, int64_t> ParallelStrategy::allocateCoresForAxes(int64_t totalCores) {
  int64_t coresForParallel = 1;
  int64_t coresForReduce = 1;

  // Case 1: If total parallel size >= total cores, allocate all cores to parallel axes
  if (totalParallelSize >= totalCores) {
    coresForParallel = totalCores;
    coresForReduce = 1;
  } else {
    // Case 2: If total parallel size < total cores, satisfy parallel axes first, then allocate remaining to reduce axes
    coresForParallel = totalParallelSize;
    int64_t remainingCores = totalCores / coresForParallel;
    // Reduce axis cores should not exceed its total size, and at least 1
    coresForReduce = std::min(remainingCores, totalReduceSize);
    coresForReduce = std::max<int64_t>(coresForReduce, 1);
  }

  return {coresForParallel, coresForReduce};
}

// Check if any axis is dynamic
bool ParallelStrategy::hasDynamicAxis(const SmallVector<AxisPtr> &axes) {
  for (const auto &axis : axes) {
    if (axis && axis->axisType.find(mlir::autotiling::Axis::AxisLabel::kDynamic) != axis->axisType.end()) {
      return true;
    }
  }
  return false;
}

static int64_t getVectorizationUpperBound(const ConfigPtr vectorizationTileConfig) {
  if (!vectorizationTileConfig) {
    return 1;
  }
  if (vectorizationTileConfig->value > 0) {
    return vectorizationTileConfig->value;
  }
  if (vectorizationTileConfig->constraints.empty()) {
    return 1;
  }
  int64_t minMax = kInitialMaxConstraintValue;
  for (const auto &cons : vectorizationTileConfig->constraints) {
    if (cons.max > 0 && cons.max < minMax) {
      minMax = cons.max;
    }
  }
  return (minMax != kInitialMaxConstraintValue) ? minMax : 1;
}

// Dynamic shape parallel tiling: simplified strategy
// Parallel constraint = vectorization upper bound * coreNum
void ParallelStrategy::applyDynamicParallelTiling(const SmallVector<AxisPtr> &axes, int64_t coreNum, int pos) {
  for (const auto &axis : axes) {
    if (!axis) {
      continue;
    }

    if (pos != 0) {
      axis->doExtraTile();
    }

    // Skip reduction axes - they don't participate in parallel tiling for dynamic shape
    bool isReduction = axis->axisType.find(mlir::autotiling::Axis::AxisLabel::kReduction) != axis->axisType.end();
    if (isReduction) {
      auto parallelTileConfig = axis->tryGetConfig(pos, kTileCfg);
      if (parallelTileConfig != nullptr) {
        parallelTileConfig->value = 1;
      }
      continue;
    }

    // Get vectorization constraint upper bound from pos+1 (vectorization level)
    auto vectorizationTileConfig = axis->tryGetConfig(pos + 1);
    int64_t vectorizationUpperBound = getVectorizationUpperBound(vectorizationTileConfig);

    // Check if vectorizationUpperBound equals dim size
    int64_t dimSize = axis->range.second > 0 ? axis->range.second : 1;
    bool shouldSetParallelValue = (vectorizationUpperBound == dimSize);

    // Parallel upper bound = vectorization upper bound * coreNum
    int64_t parallelUpperBound = vectorizationUpperBound * coreNum;
    parallelUpperBound = std::max<int64_t>(parallelUpperBound, 1);

    // Add constraint for parallel tiling
    auto tileConfig = axis->tryGetConfig(pos, kTileCfg);
    if (tileConfig != nullptr) {
      if (shouldSetParallelValue) {
        // If vectorizationUpperBound equals dim size, set parallel tile value to vectorizationUpperBound
        tileConfig->value = vectorizationUpperBound;
      } else {
        axis->tryAddConstraint(pos, Constraint(1, static_cast<int>(parallelUpperBound), 1));
        tileConfig->value = kDynamicTileMarker;
      }
    }
  }
}

// Helper function to calculate parallel tile size
int64_t ParallelStrategy::calculateParallelTileSize(int64_t outerSize, int64_t &remainingParallelCores) {
  if (remainingParallelCores <= 1 || outerSize <= 1) {
    return outerSize;
  }

  int64_t parallelTileSize = (outerSize + remainingParallelCores - 1) / remainingParallelCores;
  parallelTileSize = std::max<int64_t>(parallelTileSize, 1);

  if (parallelTileSize == 1 && outerSize > 1) {
    int64_t adjustedCores = std::max<int64_t>((outerSize + 1) / 2, 1);
    if (adjustedCores < remainingParallelCores) {
      remainingParallelCores = adjustedCores;
      parallelTileSize = (outerSize + remainingParallelCores - 1) / remainingParallelCores;
      parallelTileSize = std::max<int64_t>(parallelTileSize, 1);
    }
  }

  if (parallelTileSize < outerSize) {
    remainingParallelCores = (remainingParallelCores + parallelTileSize - 1) / parallelTileSize;
  } else {
    remainingParallelCores = 1;
  }

  return parallelTileSize;
}

// Helper function to calculate reduce tile size
int64_t ParallelStrategy::calculateReduceTileSize(int64_t outerSize, int64_t &remainingReduceCores) {
  if (remainingReduceCores <= 1 || outerSize <= 1) {
    return outerSize;
  }

  int64_t reduceTileSize = (outerSize + remainingReduceCores - 1) / remainingReduceCores;
  reduceTileSize = std::max<int64_t>(reduceTileSize, 1);

  if (reduceTileSize < outerSize) {
    remainingReduceCores = (remainingReduceCores + reduceTileSize - 1) / reduceTileSize;
  } else {
    remainingReduceCores = 1;
  }

  return reduceTileSize;
}

// Helper function to adjust tile size to ensure it doesn't exceed core limit
int64_t ParallelStrategy::adjustTileSizeForCoreLimit(int64_t axisSize, int64_t tileSize, int64_t coreNum) {
  int64_t finalTileSize = tileSize;
  int64_t numBlocks = (axisSize + finalTileSize - 1) / finalTileSize;
  if (numBlocks > coreNum) {
    finalTileSize = (axisSize + coreNum - 1) / coreNum;
    finalTileSize = std::max<int64_t>(finalTileSize, 1);
  }
  return finalTileSize;
}

// Apply static parallel tiling strategy: prioritize parallel axes, then process reduce axes
void ParallelStrategy::applyStaticParallelTiling(const SmallVector<AxisPtr> &axes, int64_t coresForParallel,
                                                 int64_t coresForReduce, int64_t coreNum, int pos) {
  int64_t remainingParallelCores = coresForParallel;
  int64_t remainingReduceCores = coresForReduce;

  for (size_t i = 0; i < axes.size(); ++i) {
    auto axis = axes[i];
    if (!axis) {
      continue;
    }

    if (pos != 0) {
      axis->doExtraTile();
    }

    auto vectorizationTileConfig = axis->tryGetConfig(pos + 1);
    int64_t innerTileSize = 1;
    if (vectorizationTileConfig && vectorizationTileConfig->value > 0) {
      innerTileSize = vectorizationTileConfig->value;
    }

    int64_t axisSize = axis->range.second > 0 ? axis->range.second : 1;
    int64_t outerSize = (axisSize + innerTileSize - 1) / innerTileSize;
    int64_t tileSize = outerSize;

    if (isParallelAxis[i] && remainingParallelCores > 1 && outerSize > 1) {
      tileSize = std::min(tileSize, calculateParallelTileSize(outerSize, remainingParallelCores));
    } else if (!isParallelAxis[i] && remainingReduceCores > 1 && outerSize > 1) {
      tileSize = std::min(tileSize, calculateReduceTileSize(outerSize, remainingReduceCores));
    }

    int64_t finalTileSize = tileSize * innerTileSize;
    finalTileSize = std::min(finalTileSize, axisSize);
    finalTileSize = std::max<int64_t>(finalTileSize, 1);
    finalTileSize = adjustTileSizeForCoreLimit(axisSize, finalTileSize, coreNum);

    auto parallelTileConfig = axis->tryGetConfig(pos, kTileCfg);
    if (parallelTileConfig != nullptr) {
      parallelTileConfig->value = static_cast<int>(finalTileSize);
    }
  }
}

void ParallelStrategy::AddNpuConstraint(NpuModelGraphPtr npuGraph) {
  // Get total available cores from NPU resource
  int64_t totalCores = npuGraph->coreNum;

  // Collect axes
  SmallVector<AxisPtr> axes;
  npuGraph->rootAxis->forEachAxisTopDown([&axes](const AxisPtr axis) {
    if (axis) {
      axes.push_back(axis);
    }
  });

  if (axes.empty()) {
    return;
  }
  // Check if this is a dynamic shape scenario
  bool isDynamicShape = hasDynamicAxis(axes);
  if (isDynamicShape) {
    // Dynamic shape: simplified parallel tiling
    // Parallel constraint = vectorization upper bound * coreNum
    // Only non-reduction axes participate in parallel tiling
    applyDynamicParallelTiling(axes, totalCores, 0);
  } else {
    // Static shape: detailed parallel tiling with core allocation
    collectAxesInfo(axes, 1);
    auto [coresForParallel, coresForReduce] = allocateCoresForAxes(totalCores);
    applyStaticParallelTiling(axes, coresForParallel, coresForReduce, totalCores, 0);
  }

  // Mark the outermost parallel axis for multi-core execution
  for (const auto &axis : axes) {
    if (axis && axis->axisType.find(mlir::autotiling::Axis::AxisLabel::kReduction) == axis->axisType.end()) {
      (void)axis->axisType.insert(mlir::autotiling::Axis::AxisLabel::kMultiCore);
      break;
    }
  }

  ++npuGraph->tileNum;
}
}  // namespace autotiling
}  // namespace mlir
