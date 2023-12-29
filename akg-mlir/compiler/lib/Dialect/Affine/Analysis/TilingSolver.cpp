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

#include "akg/Dialect/Affine/Analysis/TilingSolver.h"

#include <algorithm>
#include "akg/Utils/AKGGlobalVars.hpp"
#include "akg/Utils/AnalysisCommon.hpp"
#include "akg/Utils/AnalysisForGpu.hpp"
using akgglobal::AxisInfo;

namespace mlir {
namespace akg {
namespace autotiling {
static bool isDynamicShape() { return akgglobal::ShapeAlignTool::getInstance().getFuncArgSizes() > 0; }

void TilingSolver::initMinSize() {
  if (this->modelGraph->rootAxis == nullptr) {
    llvm::errs() << "Error: root axis is nullptr";
    return;
  }
  this->modelGraph->rootAxis->forEachAxisTopDown([this](const AxisPtr &a) {
    this->modelGraph->levelToTile = std::max<size_t>(this->modelGraph->levelToTile, a->configs[kTileCfg].size());
    (void)std::for_each(a->configs.begin(), a->configs.end(), [](auto it) {
      (void)std::for_each(it.second.begin(), it.second.end(), [](auto cfg) { cfg->mergeConstraints(); });
    });
  });
  this->modelGraph->rootAxis->forEachAxisTopDown([this](const AxisPtr &a) {
    while (this->modelGraph->levelToTile > a->configs[kTileCfg].size()) {
      a->doExtraTile();
      auto tile = a->tryGetConfig(static_cast<int>(a->configs[kTileCfg].size()) - 1);
      tile->value = 1;
      // Move last block config forward
      if (auto mapBlock = a->tryGetConfig(0, kGpuBlockCfg)) {
        mapBlock->index = ConfigPos(static_cast<int>(mapBlock->index) - 1);
      }
    }
  });
}

std::vector<AxisPtr> TilingSolver::sortAxis(size_t bandIdx) {
  std::vector<AxisPtr> sortedAxes;
  auto bandRoot = modelGraph->rootAxis->children[bandIdx];
  sortedAxes.push_back(bandRoot);
  bandRoot->forEachAxisTopDown([&sortedAxes](const AxisPtr a) { sortedAxes.push_back(a); });
  return sortedAxes;
}

void TilingSolver::solve(const AxisPtr a) {
  if (!genSolveTarget()) {
    llvm::errs() << "Solve target should be implement!\n";
    return;
  }
  auto tasks = sortSolveTask(a);
  for (auto t : tasks) {
    if (t->value != 0) {
      // If a task value is set to a positive number, then it is assumed to be solved.
      continue;
    }
    auto validCandidates = t->getValidCandidates();
    t->value = getTileSize(a, validCandidates);
  }
  (void)solved.insert(a);
}

int TilingSolver::getTileSize(const AxisPtr a, std::vector<int> candidates) {
  if (candidates.empty()) {
    llvm::errs() << "Error: candidates empty!!\n";
    return 1;
  }
  std::set<int> finalCandidates;
  std::map<int, int> candidateRanks;
  for (auto rule : target->rules) {
    auto res = rule(a, candidates);
    for (auto cand : res) {
      if (finalCandidates.find(cand) == finalCandidates.end()) {
        candidateRanks[candidateRanks.size()] = cand;
      }
    }
  }
  if (candidateRanks.empty()) {
    llvm::errs() << "Error, no candiate is chosen\n";
    return 1;
  }
  auto chosen = candidateRanks.begin()->second;
  return chosen;
}

// Here is a naive solve target that always return the max tile size in candidate
bool HeuristicTilingSolver::genSolveTarget() {
  target = std::make_shared<SolveTarget>("Heuristic");
  target->addRule([](const AxisPtr a, std::vector<int> &candidates) -> std::deque<int> {
    std::deque<int> chosen;
    if (a->axisType.find(Axis::AxisLabel::kMultiCore) != a->axisType.end()) {
      for (auto cand : candidates) {
        chosen.push_front(cand);
        if (a->range.second % cand == 0) {
          return chosen;
        }
      }
      return chosen;
    }
    if (a->axisType.find(Axis::AxisLabel::kReduction) != a->axisType.end()) {
      chosen.push_front(candidates.back());
      return chosen;
    }
    if (a->axisType.find(Axis::AxisLabel::kVectorization) != a->axisType.end()) {
      for (int i = static_cast<int>(candidates.size()) - 1; i >= 0; --i) {
        auto cand = candidates[i];
        chosen.push_front(cand);
        if (a->range.second % cand == 0) {
          return chosen;
        }
      }
      return chosen;
    }
    return {candidates.back()};
  });
  return true;
}

std::vector<ConfigPtr> HeuristicTilingSolver::sortSolveTask(const AxisPtr &axis) {
  std::vector<ConfigPtr> sortedConfigs;
  std::vector<std::string> orderList = {kGpuSeqCfg, kGpuBlockCfg, kGpuGridCfg, kTileCfg};
  for (auto configType : orderList) {
    if (axis->configs.find(configType) == axis->configs.end()) {
      continue;
    }
    (void)std::for_each(axis->configs[configType].begin(), axis->configs[configType].end(),
                        [&sortedConfigs](auto cfg) { sortedConfigs.push_back(cfg); });
  }
  return sortedConfigs;
}

void GlobalConfigSolver::setEnableVectorize() {
  if (modelGraph->graphTemplate == GraphTemplate::TRANSPOSE_OP) {
    akgglobal::GpuScheduleTool::getInstance().enableVectorize = false;
    return;
  }
  bool enableVec = true;
  int64_t innerAlignSize = -1;
  bool innerDivisible = false;
  modelGraph->rootAxis->forEachAxisTopDown([&](const AxisPtr innerMostAxis) {
    if (!enableVec || !innerMostAxis->isInnerMost) {
      return;
    }
    if (innerMostAxis->configs[kTileCfg].size() != 1) {
      enableVec = false;
      return;
    }
    auto innerTile = innerMostAxis->tryGetConfig(0, kTileCfg);
    innerAlignSize =
      innerMostAxis->axisType.find(Axis::AxisLabel::kDynamic) == innerMostAxis->axisType.end() ? innerTile->value : -1;
    innerDivisible = innerMostAxis->range.second % innerAlignSize == 0;
  });
  for (auto node : modelGraph->nodes()) {
    if (node->opType == "HeavyElem") {
      enableVec = false;
      break;
    }
  }
  if (innerAlignSize <= akgglobal::GpuScheduleTool::getInstance().minBlockSizesForVectorized ||
      innerAlignSize % akgglobal::GpuScheduleTool::getInstance().vectorSize != 0 || !innerDivisible) {
    enableVec = false;
  }
  akgglobal::GpuScheduleTool::getInstance().enableVectorize = enableVec;
}

void GlobalConfigSolver::solve(func::FuncOp funcOp) {
  if (modelGraph->hardware != kTargetCpu) {
    if (!isDynamicShape() || akgglobal::GpuScheduleTool::getInstance().runtimeArgSize() > 0 ||
        modelGraph->graphTemplate == GraphTemplate::REDUCTION) {
      if (!akgglobal::GpuScheduleTool::getInstance().getIsCustomConfig()) {
        akgglobal::GpuScheduleTool::getInstance().splitLoop(modelGraph->levelToTile + 1);
        UpdateGlobalInfo(funcOp);
      }
      akgglobal::GpuScheduleTool::getInstance().tagLoopWithAxisName(funcOp);
      if (modelGraph->graphTemplate == GraphTemplate::REDUCTION &&
          modelGraph->globalConfigs.find(akg::utils::kApplyReorderPass) != modelGraph->globalConfigs.end() &&
          modelGraph->globalConfigs[akg::utils::kApplyReorderPass].dyn_cast<BoolAttr>().getValue()) {
        akgglobal::GpuScheduleTool::getInstance().setReductionOrder();
      } else {
        setEnableVectorize();
      }
    } else {
      setEnableVectorize();
    }
  }
}

static std::pair<int, int> CollectAllAxesInfo(func::FuncOp funcOp, const ModelGraphPtr &modelGraph,
                                              const AxisPtr targetAxis) {
  auto getArgIndex = [&](const mlir::Value &memref) -> int {
    if (!memref) {
      return -1;
    }
    size_t i = 0;
    for (auto arg : funcOp.getBody().front().getArguments()) {
      if (arg == memref) {
        return static_cast<int>(i);
      }
      mlir::Value alloc = Value();
      akg::utils::GpuCommonUtils::findAllocOpForFuncArg(alloc, funcOp, arg);
      akg::utils::GpuCommonUtils::findExpandShapeOpForFuncArg(alloc, funcOp, arg);
      if (alloc && alloc == memref) {
        return static_cast<int>(i);
      }
      ++i;
    }
    return -1;
  };
  if (modelGraph->hasMinMax) {
    return std::make_pair(-1, -1);
  }
  for (auto node : modelGraph->nodes()) {
    int tensorId = -1;
    if (node->opType == "Load" && isa<AffineLoadOp>(node->op_)) {
      auto loadOp = dyn_cast<AffineLoadOp>(node->op_);
      tensorId = getArgIndex(loadOp.getMemref());
    } else if (node->opType == "Store" && isa<AffineStoreOp>(node->op_)) {
      if (auto storeOp = dyn_cast<AffineStoreOp>(node->op_)) {
        tensorId = getArgIndex(storeOp.getMemref());
      }
    }
    if (tensorId == -1) {
      continue;
    }
    for (size_t dimId = 0; dimId < node->loopNest_.size(); ++dimId) {
      auto axis = node->loopNest_[dimId];
      if (axis != targetAxis) {
        continue;
      }
      return std::make_pair(tensorId, dimId);
    }
  }
  return std::make_pair(-1, -1);
}

std::map<AxisPtr, std::vector<std::pair<std::string, int>>> GlobalConfigSolver::globalAlloc() {
  std::map<AxisPtr, std::vector<std::pair<std::string, int>>> globalRes;

  modelGraph->rootAxis->forEachAxisTopDown([&](const AxisPtr axis) { globalRes[axis] = solveMapResource(axis); });

  return globalRes;
}

std::vector<std::pair<std::string, int>> GlobalConfigSolver::solveMapResource(const AxisPtr axis) {
  std::map<int, std::pair<std::string, int>> tempMap;
  auto axisSizes = modelGraph->getLoopExtentsAfterTiling(axis);
  std::vector<std::pair<std::string, int>> allocResult;
  allocResult.reserve(axisSizes.size());
  int outerLoc = 0;
  int innerLoc = static_cast<int>(axisSizes.size()) - 1;
  int midLoc = static_cast<int>(axisSizes.size()) - 2;
  auto Load = [&](const ConfigPtr &config) {
    if (config->index == ConfigPos::kOuter) {
      tempMap[outerLoc] = std::make_pair(config->type, axisSizes[outerLoc]);
    } else if (config->index == ConfigPos::kInner) {
      tempMap[innerLoc] = std::make_pair(config->type, axisSizes[innerLoc]);
    } else if (config->index == ConfigPos::kMiddle && axisSizes.size() == 3) {
      tempMap[midLoc] = std::make_pair(config->type, axisSizes[midLoc]);
    }
  };
  if (auto mapGrid = axis->tryGetConfig(0, kGpuGridCfg)) {
    Load(mapGrid);
  }
  if (auto mapBlock = axis->tryGetConfig(0, kGpuBlockCfg)) {
    Load(mapBlock);
  }
  if (auto mapSeq = axis->tryGetConfig(0, kGpuSeqCfg)) {
    Load(mapSeq);
  }
  for (int i = 0; i < static_cast<int>(axisSizes.size()); ++i) {
    if (tempMap.find(i) == tempMap.end()) {
      tempMap[i] = std::make_pair(kGpuSeqCfg, axisSizes[static_cast<unsigned>(i)]);
    }
  }
  for (int i = 0; i < static_cast<int>(axisSizes.size()); ++i) {
    allocResult.push_back(tempMap[i]);
  }
  return allocResult;
}

void GlobalConfigSolver::UpdateGlobalInfo(func::FuncOp funcOp) {
  std::deque<AxisInfo> allAxesInfo;
  std::map<std::string, std::vector<bool>> mapLevelCnt = {
    {kGpuGridCfg, {false, false, false}},
    {kGpuBlockCfg, {false, false, false}},
  };

  for (auto axis : modelGraph->sortedAxes) {
    auto newAxisNames = akgglobal::GpuScheduleTool::getInstance().getNamesAfterTiling(axis->name);
    auto resources = solveMapResource(axis);
    assert(newAxisNames.size() == resources.size() &&
           "Length of tile-size after tiling should be equal to number of tile plus one");
    for (int i = static_cast<int>(newAxisNames.size()) - 1; i >= 0; --i) {
      auto newName = newAxisNames[static_cast<unsigned>(i)];
      auto axisInfo = AxisInfo(newName, CollectAllAxesInfo(funcOp, modelGraph, axis));
      axisInfo.size = std::to_string(resources[static_cast<unsigned>(i)].second);
      axisInfo.constSize = resources[static_cast<unsigned>(i)].second;
      axisInfo.mapLevel = resources[static_cast<unsigned>(i)].first;
      axisInfo.tileLevel = i;
      if (axisInfo.mapLevel == kGpuGridCfg) {
        auto mapGrid = axis->tryGetConfig(0, kGpuGridCfg);
        if (mapGrid && mapGrid->mapDim != -1) {
          axisInfo.mapDim = mapGrid->mapDim;
          mapLevelCnt[axisInfo.mapLevel][mapGrid->mapDim] = true;
        }
      } else if (axisInfo.mapLevel == kGpuBlockCfg) {
        auto mapBlock = axis->tryGetConfig(0, kGpuBlockCfg);
        if (mapBlock && mapBlock->mapDim != -1) {
          axisInfo.mapDim = mapBlock->mapDim;
          mapLevelCnt[axisInfo.mapLevel][mapBlock->mapDim] = true;
        }
      }
      allAxesInfo.push_back(axisInfo);
    }
  }

  if (modelGraph->graphTemplate != GraphTemplate::TRANSPOSE_OP) {
    std::sort(allAxesInfo.begin(), allAxesInfo.end(), [](const AxisInfo &a1, const AxisInfo &a2) {
      return a1.mapLevel == kGpuGridCfg && a2.mapLevel == kGpuGridCfg && a1.constSize > a2.constSize;
    });
  }

  // Alloc mapDim from tensor's low-rank to high-rank
  for (auto axisInfo : allAxesInfo) {
    if (axisInfo.mapDim == -1) {
      // Map from `x` to `z` from inner to outer to ensure coalesced access
      for (size_t i = 0; i < mapLevelCnt[axisInfo.mapLevel].size(); ++i) {
        auto used = mapLevelCnt[axisInfo.mapLevel][i];
        if (!used) {
          axisInfo.mapDim = i;
          mapLevelCnt[axisInfo.mapLevel][i] = true;
          break;
        }
      }
    }
    akgglobal::GpuScheduleTool::getInstance().add(axisInfo);
  }
}

}  // namespace autotiling
}  // namespace akg
}  // namespace mlir

