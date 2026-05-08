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

#include "akg/Analysis/TilingStrategy.h"

#include <algorithm>
#include <climits>
#include <map>
#include <numeric>
#include "akg/Dialect/Affine/Analysis/GpuTemplateTilingSolver.h"
#include "akg/Utils/AKGGlobalVars.hpp"
#include "akg/Utils/AnalysisCommon.hpp"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {
namespace autotiling {
static constexpr const char *kNotInnerDimensionBroadcastLoopAttr = "not_inner_dimension_broadcast";
using akg::utils::GpuInfo;
using akg::utils::StrategyHelper;
using llvm::SmallVector;
using mlir::akg::autotiling::GpuTemplateSolver;

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
  return std::any_of(graph->nodes().begin(), graph->nodes().end(), [&](const NodePtr &node) {
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
  auto it = std::find(shapes.begin(), shapes.end(), -1);
  if (it == shapes.end()) {
    return gpuSketch;
  }
  if (shapes.back() == -1) {
    int dynCnt = 0;
    for (int i = static_cast<int>(shapes.size()) - 1; i >= 0; --i) {
      auto shape = shapes[static_cast<unsigned>(i)];
      if (shape != -1) {
        break;
      }
      ++dynCnt;
    }
    gpuSketch = Sketch(std::min<int>(dynCnt, static_cast<int>(Sketch::kMoreDynamicInner)));
  } else {
    int64_t mul = 1;
    for (int i = static_cast<int>(shapes.size()) - 1; i >= 0; --i) {
      auto shape = shapes[static_cast<unsigned>(i)];
      if (shape == -1) {
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
    dynamicTiles = {1024};
  } else if (sketch == Sketch::kTwoDynamicInner) {
    dynamicTiles = {8, 32};
  } else if (sketch == Sketch::kMoreDynamicInner) {
    dynamicTiles = {4, 8, 32};
  }
  int staticTiles = 1;
  initGraph->rootAxis->forEachAxisBottomUp([&](const AxisPtr a) {
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
  initGraph->rootAxis->forEachAxisBottomUp([&, currMapDim = size_t{0}](const AxisPtr a) mutable {
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
  initGraph->rootAxis->forEachAxisBottomUp([&](const AxisPtr a) {
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
      if (initGraph->gpuBlock.canApply(-1)) {
        (void)initGraph->gpuBlock.alloc(a, -1);
        auto blockCfg = std::make_shared<GpuBlock>("DynBlock");
        blockCfg->index = ConfigPos::kInner;
        blockCfg->value = prime;
        a->configs[blockCfg->type].push_back(blockCfg);
      }

      if ((totalAxis <= initGraph->gpuGrid.availbleSize.size() || currDynAxis == 1) &&
          initGraph->gpuGrid.canApply(-1)) {
        (void)initGraph->gpuGrid.alloc(a, -1);
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
  initGraph->rootAxis->forEachAxisTopDown([&](const AxisPtr a) {
    if (a->axisType.find(mlir::autotiling::Axis::AxisLabel::kDynamic) != a->axisType.end()) {
      shapes.push_back(-1);
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
    (void)pairs.push_back({axisLength, static_cast<int>(i)});
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
    auto isHigherRank = [&](const NodePtr target) -> bool {
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
  auto allocResource = [&, this](const AxisPtr axis) {
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
    seqCfg->index = kOuter;
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
    seqCfg->index = kMiddle;
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
  GpuTemplateSolver::SolveScheduleForReductionOps(axes, enableParallelReduction, enableAtomicReduction,
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
    return -1;
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

  // todo: tile multi non-broadcast
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
    if (pos) {
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
    int64_t elementBit = static_cast<int64_t>(node->dataType.getIntOrFloatBitWidth());
    vectorSize = std::min(vectorSize, instructionSetBit / elementBit);
  }
  int pos = cpuGraph->tileNum;
  for (auto bandRoot : cpuGraph->rootAxis->children) {
    std::deque<AxisPtr> q = {bandRoot};
    bandRoot->forEachAxisTopDown([&q](const AxisPtr axis) { q.push_front(axis); });

    bool isNewBand = true;
    for (auto axis : q) {
      if (pos) {
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
          unrollSize /= 2;
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
      // todo: Multiple axes parallel
      if (axis->axisIdx != 0) {
        continue;
      }
      (void)axis->axisType.insert(mlir::autotiling::Axis::AxisLabel::kMultiCore);
      int64_t axisSize = axis->range.second;
      int parallelNum = BEST_PARALLEL_NUM;
      int unrollTileValue = 1;
      // Vectorization and unroll are on the same axis.
      bool isUnroll = axis->axisType.count(mlir::autotiling::Axis::AxisLabel::kVectorization);
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
  if (!axis || !axis->hasConstantBounds()) {
    const unsigned MIN_TILE_SIZE = 512;
    return MIN_TILE_SIZE;
  }
  int64_t upperBound = axis->getConstantUpperBound();
  int64_t lowerBound = axis->getConstantLowerBound();
  unsigned extent = static_cast<unsigned>(upperBound - lowerBound);
  // Simple ceiling division, no power of 2 alignment to avoid extreme cases
  unsigned tileSizePerBlock = (extent + blockNumber - 1) / blockNumber;

  // tileSizePerBlock = llvm::bit_ceil(tileSizePerBlock);
  const unsigned MIN_TILE_SIZE = 512;
  if (tileSizePerBlock < MIN_TILE_SIZE && extent >= MIN_TILE_SIZE) {
    tileSizePerBlock = MIN_TILE_SIZE;
  }
  return tileSizePerBlock;
}

namespace {
// ---------- Parallel / UB capacity ----------
// Fallback hardware values used only when NpuModelGraph has no arch info.
// Real values are sourced from NpuInfo::getInstance(arch) -> NpuModelGraph::{coreNum, ubSize}
// via NpuModelGraph::InitResource().
constexpr unsigned kNpuTargetBlocks = 48;
constexpr int64_t kNpuFallbackUbSizeInBytes = 192 * 1024;
constexpr unsigned kNpuMinInnerTileSize = 512;
constexpr int64_t kDefaultTypeBits = 32;
constexpr int64_t kBitsPerByte = 8;
constexpr int64_t kUbAlignBytes = 32;
constexpr int64_t kUbAlignBits = kUbAlignBytes * kBitsPerByte;
// Vectorized elementwise lowering can materialize extra UB-local temporaries that
// are not fully reflected by pre-tiling buffer analysis. Keep half UB as reserve.
constexpr int64_t kNpuPointTileUbReserveFactor = 2;

// ---------- Reduction regime thresholds ----------
constexpr int64_t kTinyReductionTargetBlocks = 64;
constexpr int64_t kHugeReductionTargetBlocksDelta = 6;
constexpr int64_t kTinyReductionRowsPerBlock = 8;
constexpr int64_t kHugeReductionRowsPerBlock = 2048;
constexpr int64_t kTinyReductionPointRows = 5;
constexpr int64_t kHeavyReductionPointRows = 4;
constexpr int64_t kLightReductionPointRows = 6;
constexpr int64_t kReductionComplexityThreshold = 7;

// ---------- Suffix preserve point-tile rows ----------
constexpr int64_t kSuffixPreservePointRows = 2;

struct NpuBandContext {
  size_t bandIdx{0};
  SmallVector<AxisPtr, 4> axes;
  SmallVector<int64_t, 4> extents;
  GraphTemplate graphTemplate{GraphTemplate::DEFAULT};
  int64_t rawUbElems{1};
  int64_t ubCapacityElems{1};
  int64_t targetBlocks{kNpuTargetBlocks};
  int64_t mathComplexityScore{0};
  int64_t distinctLoadTensorCount{0};
  int64_t smallestTypeBits{kDefaultTypeBits};
  bool hasDynamicAxis{false};
  bool hasReduction{false};
  bool lastAxisIsReduction{false};
};

struct BandTilePlan {
  SmallVector<unsigned, 4> outerTiles;
  SmallVector<unsigned, 4> innerTiles;
};

static void applyFallbackAxisTiling(const AxisPtr axis, const SmallVector<unsigned, 4> &tileSizes,
                                    unsigned innerTileSize, unsigned blockNumber, size_t &maxLevelToTile,
                                    bool isFullTileAxis = false);

// ===================== Axis predicates =====================
inline bool isReductionAxis(const AxisPtr &axis) {
  return axis && axis->axisType.count(mlir::autotiling::Axis::AxisLabel::kReduction) > 0;
}

inline bool isDynamicAxis(const AxisPtr &axis) {
  return axis && axis->axisType.count(mlir::autotiling::Axis::AxisLabel::kDynamic) > 0;
}

inline bool isNoSplitAxis(const AxisPtr &axis) {
  return isReductionAxis(axis);
}

inline bool isTransposeAxis(const AxisPtr &axis) {
  if (!axis || !axis->loop) return false;
  Operation *loopOp = axis->getLoopOperation();
  return loopOp && loopOp->hasAttr(kTransposeLoopAttr);
}

// ===================== Numeric helpers =====================
inline int64_t ceilDivInt64(int64_t lhs, int64_t rhs) { return (rhs <= 0) ? lhs : (lhs + rhs - 1) / rhs; }

inline int64_t multiplyAndCap(int64_t lhs, int64_t rhs) {
  if (lhs <= 0 || rhs <= 0) return 0;
  return (lhs > LLONG_MAX / rhs) ? LLONG_MAX : lhs * rhs;
}

inline unsigned saturateToTileValue(int64_t value) {
  return static_cast<unsigned>(std::clamp<int64_t>(value, 1, UINT_MAX));
}

inline int64_t getLargestFactorNotExceeding(int64_t extent, int64_t limit) {
  return StrategyHelper::getLargestDivisor(std::max<int64_t>(limit, 1), std::max<int64_t>(extent, 1));
}

// ===================== Op helpers =====================
Value getSourceMemRefValue(Value value) {
  while (value) {
    Operation *defOp = value.getDefiningOp();
    if (!defOp) break;
    if (auto op = dyn_cast<ViewLikeOpInterface>(defOp)) {
      value = op.getViewSource();
      continue;
    }
    if (auto op = dyn_cast<memref::CastOp>(defOp)) {
      value = op.getSource();
      continue;
    }
    if (auto op = dyn_cast<memref::MemorySpaceCastOp>(defOp)) {
      value = op.getSource();
      continue;
    }
    break;
  }
  return value;
}

// Complexity score contributed by a single node (for identifying heavy-math bands).
int64_t getNodeMathComplexity(const NodePtr &node) {
  if (!node) return 0;
  if (node->opType == "HeavyElem") return 3;
  if (node->opType == "Reduce") return 1;
  Operation *op = node->op();
  if (!op) return 0;
  llvm::StringRef name = op->getName().getStringRef();
  if (name.contains("pow") || name.contains("rsqrt") || name.contains("exp") || name.contains("log") ||
      name.contains("tanh"))
    return 3;
  if (name.contains("sqrt") || name.contains("div")) return 2;
  return 0;
}

// Aligned UB element capacity derived from graph config.
struct UbCapacity {
  int64_t rawUbElems{1};
  int64_t ubCapacityElems{1};
  int64_t smallestTypeBits{kDefaultTypeBits};
};

UbCapacity computeUbCapacity(const NpuModelGraphPtr &npuGraph) {
  UbCapacity c;
  if (!npuGraph) return c;
  int64_t ubBytes =
    (npuGraph->ubSize > 0) ? npuGraph->ubSize : kNpuFallbackUbSizeInBytes;
  c.smallestTypeBits = static_cast<int64_t>(npuGraph->smallestTypeBits);
  int64_t maxBufferCnt = npuGraph->maxBufferCnt;
  auto alignElems = [&](int64_t elems) {
    int64_t bits = std::max<int64_t>(elems, 1) * c.smallestTypeBits;
    return std::max<int64_t>((bits / kUbAlignBits) * kUbAlignBits / c.smallestTypeBits, 1);
  };
  int64_t rawElems = ubBytes * kBitsPerByte / c.smallestTypeBits;
  c.rawUbElems = alignElems(rawElems);
  c.ubCapacityElems = alignElems(rawElems / maxBufferCnt);
  return c;
}

inline int64_t ubCapacityForPointTile(const NpuBandContext &ctx) {
  return std::max<int64_t>(ctx.ubCapacityElems / kNpuPointTileUbReserveFactor, 1);
}

// ===================== Context building =====================
std::map<size_t, SmallVector<AxisPtr, 4>> groupAxesByBand(const SmallVector<AxisPtr> &axes) {
  std::map<size_t, SmallVector<AxisPtr, 4>> grouped;
  for (const auto &axis : axes) {
    if (axis) grouped[axis->bandIdx].push_back(axis);
  }
  for (auto &[_, bandAxes] : grouped) {
    std::sort(bandAxes.begin(), bandAxes.end(),
              [](const AxisPtr &lhs, const AxisPtr &rhs) { return lhs->axisIdx < rhs->axisIdx; });
  }
  return grouped;
}

struct BandNodeFacts {
  int64_t mathComplexityScore{0};
  int64_t distinctLoadTensorCount{0};
};

BandNodeFacts collectBandNodeFacts(const NpuModelGraphPtr &npuGraph, size_t bandIdx) {
  BandNodeFacts facts;
  if (!npuGraph) return facts;

  llvm::SmallPtrSet<void *, 8> loadTensors;
  for (const auto &node : npuGraph->nodes()) {
    if (!node || node->loopNest_.empty() || !node->loopNest_.front() || node->loopNest_.front()->bandIdx != bandIdx) {
      continue;
    }
    facts.mathComplexityScore = std::min<int64_t>(facts.mathComplexityScore + getNodeMathComplexity(node), 64);
    Operation *op = node->op();
    if (auto loadOp = dyn_cast_or_null<memref::LoadOp>(op)) {
      loadTensors.insert(getSourceMemRefValue(loadOp.getMemRef()).getAsOpaquePointer());
    }
  }
  facts.distinctLoadTensorCount = static_cast<int64_t>(loadTensors.size());
  return facts;
}

// Build a per-band context in a single pass over band axes plus one aggregated node scan.
NpuBandContext buildNpuBandContext(const NpuModelGraphPtr &npuGraph, size_t bandIdx,
                                   const SmallVector<AxisPtr, 4> &bandAxes) {
  NpuBandContext ctx;
  ctx.bandIdx = bandIdx;
  ctx.axes = bandAxes;
  if (!npuGraph) return ctx;

  ctx.graphTemplate = npuGraph->graphTemplate;
  ctx.targetBlocks =
    (npuGraph->coreNum > 0) ? static_cast<int64_t>(npuGraph->coreNum) : static_cast<int64_t>(kNpuTargetBlocks);
  UbCapacity ub = computeUbCapacity(npuGraph);
  ctx.rawUbElems = ub.rawUbElems;
  ctx.ubCapacityElems = ub.ubCapacityElems;
  ctx.smallestTypeBits = ub.smallestTypeBits;

  for (const auto &axis : bandAxes) {
    ctx.extents.push_back((axis && axis->range.second > 0) ? axis->range.second : 1);
    ctx.hasDynamicAxis |= isDynamicAxis(axis);
    ctx.hasReduction |= isReductionAxis(axis);
  }
  ctx.lastAxisIsReduction = !bandAxes.empty() && isReductionAxis(bandAxes.back());
  BandNodeFacts nodeFacts = collectBandNodeFacts(npuGraph, bandIdx);
  ctx.mathComplexityScore = nodeFacts.mathComplexityScore;
  ctx.distinctLoadTensorCount = nodeFacts.distinctLoadTensorCount;
  return ctx;
}

// ===================== Geometry helpers =====================
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
    if (!isSmallSemanticAxis(ctx, candidate)) break;
    if (suffixElems > ubCapacityForPointTile(ctx) / candidate) break;
    --suffixStart;
    suffixElems = multiplyAndCap(suffixElems, candidate);
  }
  return suffixStart;
}

// Largest broadcast suffix satisfying: rank is small, suffix fits UB per point,
// innermost axis is wide, and suffix contains a small-semantic axis before it.
size_t computeBroadcastSuffixStart(const NpuBandContext &ctx) {
  constexpr size_t kMinStructuredSuffixRank = 2;
  constexpr size_t kMaxStructuredSuffixRank = 3;
  constexpr int64_t kMinWideInnermostExtent = 32;
  constexpr int64_t kWideInnermostTargetBlockDivisor = 2;
  if (ctx.axes.size() <= kMinStructuredSuffixRank) return ctx.axes.size() - 1;

  int64_t wideThresh =
    std::max<int64_t>(kMinWideInnermostExtent, ctx.targetBlocks / kWideInnermostTargetBlockDivisor);
  int64_t innerExtent = ctx.extents.back();
  int64_t ubPerPoint = ubCapacityForPointTile(ctx) / kSuffixPreservePointRows;

  size_t bestStart = ctx.axes.size() - 1;
  for (size_t start = ctx.axes.size() - 1; start > 0; --start) {
    size_t candidateStart = start - 1;
    size_t suffixRank = ctx.axes.size() - candidateStart;
    if (suffixRank < kMinStructuredSuffixRank || suffixRank > kMaxStructuredSuffixRank) continue;
    int64_t suffixElems = getSliceProduct(ctx.extents, candidateStart, ctx.extents.size());
    if (suffixElems > ubPerPoint) break;
    if (innerExtent < wideThresh) continue;
    bool hasSmall = false;
    for (size_t i = candidateStart; i + 1 < ctx.extents.size(); ++i) {
      if (isSmallSemanticAxis(ctx, ctx.extents[i])) {
        hasSmall = true;
        break;
      }
    }
    if (!hasSmall) continue;
    bestStart = candidateStart;
  }
  return bestStart;
}

// ===================== Tile computation =====================
int64_t chooseBiasedTileSizeForTileCount(int64_t extent, int64_t tileCount) {
  constexpr int64_t kTileSizeBiasNumerator = 2;
  constexpr int64_t kTileSizeBiasDenominator = 3;
  constexpr int64_t kTwoTileCount = 2;
  if (extent <= 1 || tileCount <= 1) return extent;
  int64_t low = ceilDivInt64(extent, tileCount);
  int64_t high =
    (tileCount == kTwoTileCount) ? (extent - 1) : (ceilDivInt64(extent, tileCount - 1) - 1);
  high = std::max<int64_t>(high, low);
  return std::clamp(low + (high - low) * kTileSizeBiasNumerator / kTileSizeBiasDenominator, int64_t{1}, extent);
}

int64_t chooseTileSizeForTargetBlocks(int64_t extent, int64_t targetBlocks) {
  return ceilDivInt64(extent, targetBlocks);
}

SmallVector<unsigned, 4> assignPrefixOuterTiles(ArrayRef<int64_t> prefixExtents, int64_t targetBlocks) {
  constexpr int64_t kMaxSmallPrefixTileCount = 2;
  SmallVector<unsigned, 4> outerTiles;
  outerTiles.reserve(prefixExtents.size());
  int64_t producedTiles = 1;
  for (size_t i = 0; i < prefixExtents.size(); ++i) {
    int64_t extent = prefixExtents[i];
    if (extent == 1) {
      outerTiles.push_back(1);
      continue;
    }
    int64_t remaining = ceilDivInt64(targetBlocks, producedTiles);
    int64_t desired = std::min<int64_t>(extent, remaining);
    if (i > 0 && extent <= targetBlocks && desired > kMaxSmallPrefixTileCount) {
      desired = kMaxSmallPrefixTileCount;
    }
    int64_t tile = chooseBiasedTileSizeForTileCount(extent, desired);
    outerTiles.push_back(saturateToTileValue(tile));
    producedTiles = multiplyAndCap(producedTiles, ceilDivInt64(extent, tile));
  }
  return outerTiles;
}

// ===================== Plan construction =====================
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

void setFirstAxisTilePreservingRest(const NpuBandContext &ctx, int64_t firstAxisTile, BandTilePlan &plan) {
  initWholeBandPlan(ctx, plan);
  unsigned tile = saturateToTileValue(firstAxisTile);
  plan.outerTiles.front() = tile;
  plan.innerTiles.front() = tile;
}

// Fill `plan` with prefix outer tiles and limit innermost inner-tile by UB capacity.
// - prefixOuter: tile sizes for axes [0 .. prefixOuter.size())
// - suffixStart: first suffix axis index (== prefixOuter.size() by convention)
// - requestedPointRows: requested number of suffix "rows" co-resident in UB.
//   This function first clamps it by UB capacity, then propagates to tile[0]/inner[0]
//   and to the innermost inner-tile.
void fillPrefixSuffixTiles(const NpuBandContext &ctx, ArrayRef<unsigned> prefixOuter, size_t suffixStart,
                           int64_t requestedPointRows, BandTilePlan &plan) {
  int64_t suffixElems = getSliceProduct(ctx.extents, suffixStart, ctx.extents.size());
  int64_t ubRows = ubCapacityForPointTile(ctx) / suffixElems;
  int64_t pointRows = std::max<int64_t>(std::min(requestedPointRows, std::max<int64_t>(ubRows, 1)), 1);

  for (size_t i = 0; i < prefixOuter.size(); ++i) {
    plan.outerTiles[i] = prefixOuter[i];
    plan.innerTiles[i] = prefixOuter[i];
  }
  plan.innerTiles.front() =
    saturateToTileValue(std::min<int64_t>(pointRows, static_cast<int64_t>(prefixOuter.front())));
  size_t innermost = ctx.extents.size() - 1;
  int64_t preservedMid = getSliceProduct(ctx.extents, suffixStart, innermost);
  int64_t denom = multiplyAndCap(pointRows, preservedMid);
  int64_t maxInner = ubCapacityForPointTile(ctx) / denom;
  int64_t innerExtent = ctx.extents[innermost];
  plan.innerTiles[innermost] = saturateToTileValue(std::min(innerExtent, std::max<int64_t>(maxInner, 1)));
}

// ===================== Reduction regime =====================
enum class ReductionRowRegime { Tiny, Normal, Huge };

struct ReductionRegimeInfo {
  ReductionRowRegime regime{ReductionRowRegime::Normal};
  int64_t targetBlocks{kNpuTargetBlocks};
  int64_t pointRows{kLightReductionPointRows};
};

ReductionRegimeInfo computeReductionRegime(const NpuBandContext &ctx, size_t suffixStart) {
  ReductionRegimeInfo info;
  int64_t prefixRows = getSliceProduct(ctx.extents, 0, suffixStart);
  int64_t rowsPerBlock = ceilDivInt64(prefixRows, ctx.targetBlocks);
  if (rowsPerBlock <= kTinyReductionRowsPerBlock)
    info.regime = ReductionRowRegime::Tiny;
  else if (rowsPerBlock >= kHugeReductionRowsPerBlock)
    info.regime = ReductionRowRegime::Huge;

  if (info.regime == ReductionRowRegime::Tiny)
    info.targetBlocks = kTinyReductionTargetBlocks;
  else if (info.regime == ReductionRowRegime::Huge)
    info.targetBlocks = std::max<int64_t>(ctx.targetBlocks - kHugeReductionTargetBlocksDelta, 1);
  else
    info.targetBlocks = ctx.targetBlocks;

  if (suffixStart < ctx.axes.size() - 1)
    info.pointRows = kSuffixPreservePointRows;
  else if (info.regime == ReductionRowRegime::Tiny)
    info.pointRows = kTinyReductionPointRows;
  else if (info.regime == ReductionRowRegime::Huge)
    info.pointRows = kLightReductionPointRows;
  else
    info.pointRows =
      (ctx.mathComplexityScore >= kReductionComplexityThreshold) ? kHeavyReductionPointRows : kLightReductionPointRows;
  return info;
}

int64_t getBroadcastSuffixPointRows(const NpuBandContext &ctx, size_t suffixStart) {
  constexpr int64_t kBroadcastSuffixPointRows = 3;
  constexpr int64_t kBroadcastCloneInnerBlocksFactor = 5;
  constexpr int64_t kBroadcastCloneUbDivisor = 4;
  int64_t suffixElems = getSliceProduct(ctx.extents, suffixStart, ctx.extents.size());
  bool broadcastCloneLike =
    ctx.extents.back() >= ctx.targetBlocks * kBroadcastCloneInnerBlocksFactor &&
    suffixElems >= std::max<int64_t>(ubCapacityForPointTile(ctx) / kBroadcastCloneUbDivisor, 1);
  return broadcastCloneLike ? kBroadcastSuffixPointRows : kSuffixPreservePointRows;
}

int64_t estimatePureElemLiveBufferFactor(const NpuBandContext &ctx) {
  constexpr int64_t kPureElemWidenTargetBits = 32;
  int64_t inputCount = std::max<int64_t>(ctx.distinctLoadTensorCount, 1);
  int64_t widenFactor =
    std::max<int64_t>(ceilDivInt64(kPureElemWidenTargetBits, std::max<int64_t>(ctx.smallestTypeBits, 1)), 1);
  return inputCount + (inputCount + 1) * widenFactor;
}

int64_t getPureElemTileBudget(const NpuBandContext &ctx) {
  constexpr int64_t kPureElemUbBudgetNumerator = 95;
  constexpr int64_t kPureElemUbBudgetDenominator = 100;
  int64_t liveFactor = estimatePureElemLiveBufferFactor(ctx);
  int64_t usableUb = ctx.ubCapacityElems * kPureElemUbBudgetNumerator / kPureElemUbBudgetDenominator;
  return std::max<int64_t>(usableUb / liveFactor, 1);
}

size_t computePureElemSuffixStart(const NpuBandContext &ctx, int64_t tileBudget) {
  if (ctx.axes.size() <= 1) return 0;
  size_t suffixStart = ctx.axes.size() - 1;
  int64_t suffixElems = ctx.extents.back();
  while (suffixStart > 1) {
    int64_t candidate = ctx.extents[suffixStart - 1];
    if (suffixElems > tileBudget / candidate) break;
    --suffixStart;
    suffixElems = multiplyAndCap(suffixElems, candidate);
  }
  return suffixStart;
}

void buildPureElemPlan(const NpuBandContext &ctx, size_t suffixStart, int64_t tileBudget, BandTilePlan &plan) {
  initWholeBandPlan(ctx, plan);
  if (ctx.axes.size() == 1) {
    int64_t extent = ctx.extents.front();
    int64_t parallelBlocks = std::min<int64_t>(ctx.targetBlocks, ceilDivInt64(extent, tileBudget));
    int64_t outerTile = chooseTileSizeForTargetBlocks(extent, parallelBlocks);
    plan.outerTiles.front() = saturateToTileValue(outerTile);
    plan.innerTiles.front() = saturateToTileValue(std::min<int64_t>(outerTile, tileBudget));
    return;
  }

  int64_t suffixElems = getSliceProduct(ctx.extents, suffixStart, ctx.extents.size());
  int64_t prefixRows = getSliceProduct(ctx.extents, 0, suffixStart);
  int64_t pointRowsCap = std::max<int64_t>(tileBudget / suffixElems, 1);
  int64_t parallelBlocks = std::min<int64_t>(ctx.targetBlocks, ceilDivInt64(prefixRows, pointRowsCap));
  SmallVector<unsigned, 4> prefixOuterTiles = assignPrefixOuterTiles(
    ArrayRef<int64_t>(ctx.extents).take_front(suffixStart), parallelBlocks);

  int64_t preservedPrefixInner = 1;
  for (size_t i = 0; i < prefixOuterTiles.size(); ++i) {
    plan.outerTiles[i] = prefixOuterTiles[i];
    plan.innerTiles[i] = prefixOuterTiles[i];
    if (i > 0) preservedPrefixInner = multiplyAndCap(preservedPrefixInner, prefixOuterTiles[i]);
  }

  int64_t firstInner = std::max<int64_t>(tileBudget / suffixElems, 1);
  firstInner = std::max<int64_t>(firstInner / preservedPrefixInner, 1);
  plan.innerTiles.front() = saturateToTileValue(std::min<int64_t>(firstInner, prefixOuterTiles.front()));
}

bool tryBuildReductionSuffixPlan(const NpuBandContext &ctx, BandTilePlan &plan) {
  if (ctx.hasDynamicAxis || !ctx.hasReduction || !ctx.lastAxisIsReduction || ctx.axes.size() < 2 ||
      ctx.graphTemplate == GraphTemplate::TRANSPOSE_OP) {
    return false;
  }

  size_t suffixStart = computeReductionSuffixStart(ctx);
  if (suffixStart == 0) return false;

  initWholeBandPlan(ctx, plan);
  ArrayRef<int64_t> prefixExtents(ctx.extents);
  ReductionRegimeInfo info = computeReductionRegime(ctx, suffixStart);
  SmallVector<unsigned, 4> prefixOuterTiles =
    assignPrefixOuterTiles(prefixExtents.take_front(suffixStart), info.targetBlocks);
  fillPrefixSuffixTiles(ctx, prefixOuterTiles, suffixStart, info.pointRows, plan);
  return true;
}

bool tryBuildTransposePlan(const NpuBandContext &ctx, BandTilePlan &plan) {
  if (ctx.hasDynamicAxis || ctx.hasReduction || ctx.axes.size() < 2) {
    return false;
  }

  int64_t innermostTransposeIdx = -1;
  for (int64_t i = static_cast<int64_t>(ctx.axes.size()) - 1; i >= 0; --i) {
    if (isTransposeAxis(ctx.axes[static_cast<size_t>(i)])) {
      innermostTransposeIdx = i;
      break;
    }
  }
  if (innermostTransposeIdx < 0) {
    return false;
  }

  constexpr int64_t kTransposeTileProductLimit = 8192;
  setFirstAxisTilePreservingRest(ctx, chooseTileSizeForTargetBlocks(ctx.extents.front(), ctx.targetBlocks), plan);

  if (static_cast<size_t>(innermostTransposeIdx) == ctx.axes.size() - 1) {
    int64_t remainingProduct = kTransposeTileProductLimit;
    for (int64_t i = innermostTransposeIdx; i >= 0; --i) {
      if (!isTransposeAxis(ctx.axes[static_cast<size_t>(i)])) {
        break;
      }
      int64_t extent = ctx.extents[static_cast<size_t>(i)];
      unsigned tile = saturateToTileValue(getLargestFactorNotExceeding(extent, remainingProduct));
      plan.outerTiles[static_cast<size_t>(i)] = tile;
      plan.innerTiles[static_cast<size_t>(i)] = tile;
      remainingProduct = remainingProduct / tile;
    }
  } else {
    int64_t preservedSuffixElems = 1;
    for (size_t i = static_cast<size_t>(innermostTransposeIdx + 1); i < ctx.extents.size(); ++i) {
      preservedSuffixElems = multiplyAndCap(preservedSuffixElems, ctx.extents[i]);
    }
    int64_t maxTransposeTile = std::max<int64_t>(kTransposeTileProductLimit / preservedSuffixElems, 1);
    plan.outerTiles[static_cast<size_t>(innermostTransposeIdx)] = saturateToTileValue(getLargestFactorNotExceeding(
      ctx.extents[static_cast<size_t>(innermostTransposeIdx)], maxTransposeTile));
  }
  return true;
}

bool tryBuildBroadcastSuffixPlan(const NpuBandContext &ctx, BandTilePlan &plan) {
  if (ctx.hasDynamicAxis || ctx.hasReduction || ctx.graphTemplate != GraphTemplate::BROADCAST_OP ||
      ctx.axes.size() < 2) {
    return false;
  }

  size_t suffixStart = computeBroadcastSuffixStart(ctx);
  if (suffixStart == 0 || suffixStart >= ctx.axes.size() - 1) return false;

  initWholeBandPlan(ctx, plan);
  ArrayRef<int64_t> prefixExtents(ctx.extents);
  SmallVector<unsigned, 4> prefixOuterTiles =
    assignPrefixOuterTiles(prefixExtents.take_front(suffixStart), ctx.targetBlocks);
  fillPrefixSuffixTiles(ctx, prefixOuterTiles, suffixStart, getBroadcastSuffixPointRows(ctx, suffixStart), plan);

  constexpr int64_t kBroadcastTileProductLimit = 8192;
  int64_t remainingProduct = kBroadcastTileProductLimit;
  for (int64_t i = static_cast<int64_t>(ctx.axes.size()) - 1; i >= 0; --i) {
    size_t idx = static_cast<size_t>(i);
    Operation *loopOp = ctx.axes[idx] ? ctx.axes[idx]->getLoopOperation() : nullptr;
    if (!loopOp || (!loopOp->hasAttr(kBroadcastLoopAttr) && !loopOp->hasAttr(kNotInnerDimensionBroadcastLoopAttr))) {
      break;
    }
    int64_t extent = ctx.extents[idx];
    unsigned tile = saturateToTileValue(getLargestFactorNotExceeding(extent, remainingProduct));
    plan.outerTiles[idx] = tile;
    plan.innerTiles[idx] = tile;
    remainingProduct = remainingProduct / tile;
  }
  return true;
}

bool tryBuildElementwisePlan(const NpuBandContext &ctx, BandTilePlan &plan) {
  if (ctx.hasDynamicAxis || ctx.hasReduction || ctx.axes.empty() || ctx.graphTemplate != GraphTemplate::PURE_ELEM) {
    return false;
  }

  int64_t tileBudget = getPureElemTileBudget(ctx);
  int64_t alignElems = std::max<int64_t>(kUbAlignBits / ctx.smallestTypeBits, 1);
  if (tileBudget >= alignElems) tileBudget = (tileBudget / alignElems) * alignElems;
  size_t suffixStart = computePureElemSuffixStart(ctx, tileBudget);
  buildPureElemPlan(ctx, suffixStart, tileBudget, plan);
  return true;
}

void applyBandTilePlan(const SmallVector<AxisPtr, 4> &bandAxes, const BandTilePlan &plan, size_t &maxLevelToTile) {
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
      if (isNoSplitAxis(axis) && axis->hasConstantBounds()) {
        int64_t fullExtent = axis->getConstantUpperBound() - axis->getConstantLowerBound();
        tileValue = saturateToTileValue(std::max<int64_t>(fullExtent, 1));
      }
      tileConfig->value = static_cast<int>(tileValue);
      axis->tryAddConstraint(static_cast<int>(level), Constraint({static_cast<int>(tileValue)}));
    }
  }
  maxLevelToTile = std::max(maxLevelToTile, kTileLevels);
}

void applyDynamicFallbackAxisTiling(const AxisPtr axis, bool isFullTileAxis) {
  for (size_t i = axis->configs[kTileCfg].size(); i < 2; ++i) {
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
    unsigned value = UINT_MAX;
    tileConfig0->value = value;
    axis->tryAddConstraint(0, Constraint({value}));
  }

  if (tileConfig1 != nullptr) {
    tileConfig1->value = static_cast<int>(kNpuMinInnerTileSize);
    axis->tryAddConstraint(1, Constraint({static_cast<int>(kNpuMinInnerTileSize)}));
  }
}

static void applyFallbackAxisTiling(const AxisPtr axis, const SmallVector<unsigned, 4> &tileSizes,
                                    unsigned innerTileSize, unsigned blockNumber, size_t &maxLevelToTile,
                                    bool isFullTileAxis) {
  bool hasStaticBounds = axis->hasConstantBounds();
  bool hasDynamicUpperBound = !axis->hasConstantUpperBound();

  if (hasDynamicUpperBound || !hasStaticBounds) {
    applyDynamicFallbackAxisTiling(axis, isFullTileAxis);
    maxLevelToTile = std::max(maxLevelToTile, static_cast<size_t>(2));
    return;
  }

  int64_t lowerBound = axis->getConstantLowerBound();
  int64_t upperBound = axis->getConstantUpperBound();
  int64_t extent = upperBound - lowerBound;

  SmallVector<unsigned, 4> currentTileSizes = tileSizes;
  if (isFullTileAxis) {
    unsigned fullTile = 1;
    if (extent > 0) {
      fullTile = extent > static_cast<int64_t>(UINT_MAX) ? UINT_MAX : static_cast<unsigned>(extent);
    }
    currentTileSizes = {fullTile, fullTile};
  }
  if (currentTileSizes.empty()) {
    currentTileSizes = {std::max(getOuterTileSize(axis, blockNumber), innerTileSize), innerTileSize};
  }

  SmallVector<unsigned, 4> usedTileSizes;
  int64_t currentSize = extent;
  for (size_t i = 0; i < currentTileSizes.size(); ++i) {
    int64_t tileSize = static_cast<int64_t>(currentTileSizes[i]);
    int64_t minRequired = (i == currentTileSizes.size() - 1) ? tileSize * 2 : tileSize;
    if (currentSize >= minRequired) {
      usedTileSizes.push_back(static_cast<unsigned>(tileSize));
      currentSize = tileSize;
    } else {
      usedTileSizes.push_back(static_cast<unsigned>(currentSize));
    }
  }

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

SmallVector<unsigned, 4> NpuDefaultTileStrategy::parseTileSizesConfig(const NpuModelGraphPtr npuGraph) {
  SmallVector<unsigned, 4> tileSizes;
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

  if (npuGraph->funcOp && npuGraph->funcOp->hasAttr("npu.multiTileSizes")) {
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
                                               const SmallVector<unsigned, 4> &tileSizes) {
  size_t maxLevelToTile = 1;
  constexpr unsigned innerTileSize = kNpuMinInnerTileSize;
  const unsigned blockNumber =
    (npuGraph && npuGraph->coreNum > 0) ? static_cast<unsigned>(npuGraph->coreNum) : kNpuTargetBlocks;

  if (!tileSizes.empty()) {
    for (const auto &axis : axes) {
      bool axisIsFullTile = isNoSplitAxis(axis);
      applyFallbackAxisTiling(axis, tileSizes, innerTileSize, blockNumber, maxLevelToTile, axisIsFullTile);
    }
    npuGraph->levelToTile = std::max(npuGraph->levelToTile, maxLevelToTile);
    return;
  }

  for (const auto &[bandIdx, bandAxes] : groupAxesByBand(axes)) {
    NpuBandContext bandCtx = buildNpuBandContext(npuGraph, bandIdx, bandAxes);
    BandTilePlan bandPlan;
    bool matched = tryBuildReductionSuffixPlan(bandCtx, bandPlan) || tryBuildTransposePlan(bandCtx, bandPlan) ||
                   tryBuildBroadcastSuffixPlan(bandCtx, bandPlan) || tryBuildElementwisePlan(bandCtx, bandPlan);

    if (matched) {
      applyBandTilePlan(bandAxes, bandPlan, maxLevelToTile);
      continue;
    }

    for (const auto &axis : bandAxes) {
      bool axisIsFullTile = isNoSplitAxis(axis);
      applyFallbackAxisTiling(axis, tileSizes, innerTileSize, blockNumber, maxLevelToTile, axisIsFullTile);
    }
  }

  npuGraph->levelToTile = std::max(npuGraph->levelToTile, maxLevelToTile);
}

void NpuDefaultTileStrategy::AddNpuConstraint(NpuModelGraphPtr npuGraph) {
  if (npuGraph == nullptr || npuGraph->rootAxis == nullptr) {
    return;
  }

  SmallVector<AxisPtr> axes = collectAxes(npuGraph);
  SmallVector<unsigned, 4> tileSizes = parseTileSizesConfig(npuGraph);

  applyTilingToAxes(npuGraph, axes, tileSizes);

  if (npuGraph->funcOp && npuGraph->funcOp->hasAttr("npu.multiTileSizes")) {
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

SmallVector<int64_t> VectorizationStrategy::getDimSizes(const SmallVector<AxisPtr> &axes) {
  SmallVector<int64_t> dims;
  for (const auto &axis : axes) {
    if (axis && axis->hasConstantBounds()) {
      int64_t extent = axis->getConstantUpperBound() - axis->getConstantLowerBound();
      dims.push_back(extent > 0 ? extent : 1);
    } else if (axis) {
      dims.push_back(axis->range.second > 0 ? axis->range.second : 1);
    }
  }
  return dims;
}

// Helper function to find the first non-static axis index from innermost
int64_t VectorizationStrategy::findConsecutiveStaticEnd(const SmallVector<AxisPtr> &axes) {
  int64_t numAxes = static_cast<int64_t>(axes.size());
  for (int64_t dimIdx = numAxes - 1; dimIdx >= 0; --dimIdx) {
    auto axis = axes[static_cast<size_t>(dimIdx)];
    if (!axis) {
      return dimIdx;
    }
    if (axis->axisType.find(mlir::autotiling::Axis::AxisLabel::kDynamic) != axis->axisType.end()) {
      return dimIdx;
    }
  }
  return -1;  // All axes are static
}

// Helper function to handle dynamic axis for vectorization
void VectorizationStrategy::handleDynamicAxisForVectorization(const AxisPtr &axis, int pos, int64_t &ubRemainingNum) {
  if (ubRemainingNum > 1) {
    axis->tryAddConstraint(pos, Constraint(1, static_cast<int>(ubRemainingNum), 1));
    auto tileConfig = axis->tryGetConfig(pos, kTileCfg);
    if (tileConfig != nullptr) {
      tileConfig->value = -1;  // Mark as dynamic
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
      tileConfig->value = -1;  // Mark as dynamic
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
  int64_t numAxes = static_cast<int64_t>(axes.size());
  int64_t ubRemainingNum = std::max<int64_t>(ubAvailableNum, 1);

  int64_t consecutiveStaticEnd = findConsecutiveStaticEnd(axes);

  // Process axes from inner to outer (reverse order)
  for (int64_t dimIdx = numAxes - 1; dimIdx >= 0; --dimIdx) {
    auto axis = axes[static_cast<size_t>(dimIdx)];
    if (!axis) {
      continue;
    }

    if (pos) {
      axis->doExtraTile();
    }

    bool isDynamic = axis->axisType.find(mlir::autotiling::Axis::AxisLabel::kDynamic) != axis->axisType.end();
    if (isDynamic) {
      handleDynamicAxisForVectorization(axis, pos, ubRemainingNum);
      continue;
    }

    int64_t dimSize = axis->range.second > 0 ? axis->range.second : 1;
    bool isInConsecutiveStatic = (consecutiveStaticEnd == -1) || (dimIdx >= consecutiveStaticEnd);
    if (isInConsecutiveStatic) {
      handleConsecutiveStaticAxis(axis, pos, dimSize, ubRemainingNum);
    } else {
      handleNonConsecutiveStaticAxis(axis, pos, dimSize, ubRemainingNum);
    }
  }
}

void VectorizationStrategy::AddNpuConstraint(NpuModelGraphPtr npuGraph) {
  SmallVector<AxisPtr> axes;
  npuGraph->rootAxis->forEachAxisTopDown([this, &axes](const AxisPtr a) { axes.push_back(a); });

  if (axes.empty()) {
    return;
  }

  // Get buffer info from NpuModelGraph (populated by buffer analysis)
  int64_t maxBufferCnt = npuGraph->maxBufferCnt;
  int64_t smallestTypeBits = static_cast<int64_t>(npuGraph->smallestTypeBits);
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

// Dynamic shape parallel tiling: simplified strategy
// Parallel constraint = vectorization upper bound * coreNum
void ParallelStrategy::applyDynamicParallelTiling(const SmallVector<AxisPtr> &axes, int64_t coreNum, int pos) {
  for (const auto &axis : axes) {
    if (!axis) {
      continue;
    }

    if (pos) {
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
    int64_t vectorizationUpperBound = 1;

    if (vectorizationTileConfig) {
      if (vectorizationTileConfig->value > 0) {
        // Static axis within dynamic shape scenario: use value directly
        vectorizationUpperBound = vectorizationTileConfig->value;
      } else {
        // Dynamic axis: get upper bound from constraint
        if (!vectorizationTileConfig->constraints.empty()) {
          // Find the minimum max value from all constraints
          int64_t minMax = INT_MAX;
          for (const auto &cons : vectorizationTileConfig->constraints) {
            if (cons.max > 0 && cons.max < minMax) {
              minMax = cons.max;
            }
          }
          if (minMax != INT_MAX) {
            vectorizationUpperBound = minMax;
          }
        }
      }
    }

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
        tileConfig->value = -1;  // Mark as dynamic, constraint.max contains upper bound
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

    if (pos) {
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
