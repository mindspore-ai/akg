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

#include "akg/Dialect/Affine/Analysis/TilingStrategy.h"

#include <algorithm>
#include <numeric>
#include "akg/Utils/AKGGlobalVars.hpp"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace akg {
namespace autotiling {
using llvm::SmallVector;
using akg::utils::GpuInfo;
using akg::utils::StrategyHelper;

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
    if (a->axisType.find(Axis::AxisLabel::kDynamic) != a->axisType.end()) {
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
    if (a->axisType.find(Axis::AxisLabel::kDynamic) != a->axisType.end()) {
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
    if (a->axisType.find(Axis::AxisLabel::kDynamic) != a->axisType.end()) {
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
    if (a->axisType.find(Axis::AxisLabel::kDynamic) == a->axisType.end()) {
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
    if (a->axisType.find(Axis::AxisLabel::kDynamic) != a->axisType.end()) {
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

void BroadcastStrategy::AddGpuConstraint(GpuModelGraphPtr gpuGraph) {
  NodePtr maxRankNode = gpuGraph->getMaxRankTensor();
  if (!maxRankNode || maxRankNode->loopNest_.size() <= 1) {
    return;
  }
  NodePtr minRankNode = nullptr;
  for (auto node : gpuGraph->nodes()) {
    if (node->loopNest_.empty() || (node->opType != "Load" && node->opType != "Store")) {
      continue;
    }
    if (minRankNode == nullptr || minRankNode->loopNest_.size() > node->loopNest_.size()) {
      minRankNode = node;
    }
  }
  if (!minRankNode || minRankNode->loopNest_.size() == maxRankNode->loopNest_.size()) {
    return;
  }
  auto innerMostReadAxis = maxRankNode->loopNest_.back();
  auto maxAvailBlocks = gpuGraph->gpuBlock.totalAvailableSize;
  assert(maxExpectSeq > 0 && innerMostReadAxis->range.second > 0);
  auto maxBlocks = StrategyHelper::getLargestDivisor(maxAvailBlocks, innerMostReadAxis->range.second / maxExpectSeq);
  int expectSeq =
    std::min<int>(maxExpectSeq, (maxBlocks * maxExpectSeq * blockWasteCoef - 1) / innerMostReadAxis->range.second + 1);
  int currBlocks = 1;
  auto warpSize = GpuInfo::getInstance(gpuGraph->hardware).getWarpSizes();
  std::tie(proposedGrid, proposedBlock) =
    StrategyHelper::getProposalParallelSize(gpuGraph->problemSize, gpuGraph->hardware);

  // Search for seq-axis from inner to outer
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
    if (axis->axisType.find(Axis::AxisLabel::kReduction) == axis->axisType.end()) {
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
      (void)axis->axisType.insert(Axis::AxisLabel::kVectorization);
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
      (void)axis->axisType.insert(Axis::AxisLabel::kMultiCore);
      int64_t axisSize = axis->range.second;
      int parallelNum = BEST_PARALLEL_NUM;
      int unrollTileValue = 1;
      // Vectorization and unroll are on the same axis.
      bool isUnroll = axis->axisType.count(Axis::AxisLabel::kVectorization);
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
  unsigned upperBound = axis->loop->getConstantUpperBound();
  unsigned lowerBound = axis->loop->getConstantLowerBound();
  unsigned extent = upperBound - lowerBound;
  // Calculate tile size to get approximately blockNumber blocks
  unsigned tileSizePerBlock = (extent + blockNumber - 1) / blockNumber;
  // Simple ceiling division, no power of 2 alignment to avoid extreme cases
  // (e.g., only 21 blocks when kernel can use about half)

  // tileSizePerBlock = llvm::bit_ceil(tileSizePerBlock);
  const unsigned MIN_TILE_SIZE = 512;
  if (tileSizePerBlock < MIN_TILE_SIZE && extent >= MIN_TILE_SIZE) {
    tileSizePerBlock = MIN_TILE_SIZE;
  }
  return tileSizePerBlock;
}

// Helper function to process tiling for a single axis
static void processAxisTiling(const AxisPtr axis, const SmallVector<unsigned, 4> &tileSizes,
                              unsigned innerTileSize, unsigned blockNumber, size_t &maxLevelToTile,
                              bool isReduceAxis = false) {
  if (axis == nullptr) {
    return;
  }
  auto loop = axis->loop;

  // Skip tiling for reduce axes
  if (isReduceAxis) {
    auto tileConfig = axis->tryGetConfig(0, kTileCfg);
    if (tileConfig != nullptr) {
      tileConfig->value = 1;
      axis->tryAddConstraint(0, Constraint({1}));
    }
    return;
  }

  bool hasStaticBounds = loop && loop->hasConstantLowerBound() && loop->hasConstantUpperBound();
  bool isDynamic = axis->axisType.find(Axis::AxisLabel::kDynamic) != axis->axisType.end();
  if (!hasStaticBounds || isDynamic) {
    auto tileConfig = axis->tryGetConfig(0, kTileCfg);
    if (tileConfig != nullptr) {
      tileConfig->value = 1;
      axis->tryAddConstraint(0, Constraint({1}));
    }
    return;
  }
  int32_t lowerBound = loop->getConstantLowerBound();
  int32_t upperBound = loop->getConstantUpperBound();
  int32_t extent = upperBound - lowerBound;

  // Get default tile sizes if not specified by user
  SmallVector<unsigned, 4> currentTileSizes = tileSizes;
  if (currentTileSizes.empty()) {
    currentTileSizes = {std::max(getOuterTileSize(axis, blockNumber),
      innerTileSize), innerTileSize};
  }

  SmallVector<unsigned, 4> usedTileSizes;
  int32_t currentSize = extent;

  for (size_t i = 0; i < currentTileSizes.size(); ++i) {
    int32_t tileSize = static_cast<int32_t>(currentTileSizes[i]);
    int32_t minRequired = (i == currentTileSizes.size() - 1) ? tileSize * 2 : tileSize;
    if (currentSize >= minRequired) {
      usedTileSizes.push_back(tileSize);
      currentSize = tileSize;
    } else {
      usedTileSizes.push_back(currentSize);
    }
  }

  size_t numLevels = usedTileSizes.size();
  for (size_t level = 1; level < numLevels; ++level) {
    axis->doExtraTile();
  }
  maxLevelToTile = std::max(maxLevelToTile, numLevels);

  // Set tile sizes for each level
  for (size_t level = 0; level < numLevels; ++level) {
    auto tileConfig = axis->tryGetConfig(static_cast<int>(level), kTileCfg);
    if (tileConfig != nullptr) {
      tileConfig->value = static_cast<int>(usedTileSizes[level]);
      axis->tryAddConstraint(static_cast<int>(level), Constraint({static_cast<int>(usedTileSizes[level])}));
    }
  }
}

void NpuDefaultTileStrategy::AddGpuConstraint(GpuModelGraphPtr gpuGraph) {
  if (gpuGraph == nullptr || gpuGraph->rootAxis == nullptr) {
    return;
  }

  // Check if this is a Reduce operation
  bool isReduceOp = false;
  if (gpuGraph->funcOp) {
    auto opType = CommonUtils::getOperatorType(gpuGraph->funcOp);
    isReduceOp = (opType == OperatorTemplate::Reduce);
  }

  std::unordered_map<size_t, unsigned> bandRankMap;
  // get depth of each band
  gpuGraph->rootAxis->forEachAxisTopDown([&bandRankMap](const AxisPtr axis) {
    if (axis == nullptr) {
      return;
    }
    auto currentDepth = static_cast<unsigned>(axis->axisIdx + 1);
    auto it = bandRankMap.find(axis->bandIdx);
    if (it == bandRankMap.end() || currentDepth > it->second) {
      bandRankMap[axis->bandIdx] = currentDepth;
    }
  });

  SmallVector<unsigned, 4> tileSizes;
  auto tileSizesIt = gpuGraph->globalConfigs.find("npu.multiTileSizes");
  if (tileSizesIt != gpuGraph->globalConfigs.end()) {
    auto arrayAttr = dyn_cast<ArrayAttr>(tileSizesIt->second);
    if (arrayAttr) {
      for (auto attr : arrayAttr) {
        if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
          tileSizes.push_back(static_cast<unsigned>(intAttr.getInt()));
        }
      }
    }
  }

  // Ensure tile sizes are in non-increasing order (largest to smallest)
  for (size_t i = 1; i < tileSizes.size(); ++i) {
    if (tileSizes[i] > tileSizes[i-1]) {
      tileSizes[i] = tileSizes[i-1];
    }
  }

  size_t maxLevelToTile = 1;
  unsigned innerTileSize = 512;
  unsigned blockNumber = 40;

  gpuGraph->rootAxis->forEachAxisTopDown([&bandRankMap, &maxLevelToTile, tileSizes,
    innerTileSize, blockNumber, isReduceOp](const AxisPtr axis) {
    if (axis == nullptr) {
      return;
    }
    auto rankIt = bandRankMap.find(axis->bandIdx);
    if (rankIt == bandRankMap.end()) {
      return;
    }

    // Check if this axis is a reduction axis
    bool isReduceAxis = false;
    if (isReduceOp && axis->loop) {
      isReduceAxis = axis->loop->getOperation()->hasAttr(kReductionLoopAttr);
    }

    processAxisTiling(axis, tileSizes, innerTileSize, blockNumber, maxLevelToTile, isReduceAxis);
  });

  gpuGraph->levelToTile = std::max(gpuGraph->levelToTile, maxLevelToTile);

  if (gpuGraph->funcOp && gpuGraph->funcOp->hasAttr("npu.multiTileSizes")) {
    (void)gpuGraph->funcOp->removeAttr("npu.multiTileSizes");
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
}  // namespace autotiling
}  // namespace akg
}  // namespace mlir
