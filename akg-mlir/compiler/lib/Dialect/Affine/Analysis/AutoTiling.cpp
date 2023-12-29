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

#include "akg/Dialect/Affine/Analysis/AutoTiling.h"

#include "akg/Dialect/Affine/Analysis/Axis.h"
#include "akg/Utils/AKGGlobalVars.hpp"
#include "akg/Utils/AnalysisCommon.hpp"

namespace mlir {
namespace akg {
namespace autotiling {
InitGraphPtr parseIr(Operation *funcOp, const std::vector<SmallVector<AffineForOp, 6>> &bands) {
  auto initGraph = parseIr(bands);
  initGraph->funcOp = funcOp;
  return initGraph;
}

InitGraphPtr parseIr(const std::vector<SmallVector<AffineForOp, 6>> &bands) {
  auto initGraph = std::make_shared<InitGraph>("initGraph");
  initGraph->rootAxis = std::make_shared<Axis>("Root");
  for (size_t i = 0; i < bands.size(); ++i) {
    auto band = bands[i];
    AxisPtr currAxis = nullptr;
    std::vector<AxisPtr> loopNest;
    for (size_t j = 0; j < band.size(); ++j) {
      auto loop = band[j];
      auto axis = std::make_shared<Axis>(i, j, std::make_shared<AffineForOp>(loop));
      loopNest.push_back(axis);
      if (!akgglobal::GpuScheduleTool::getInstance().getIsCustomConfig()) {
        akgglobal::GpuScheduleTool::getInstance().recordLoopStructure(axis->name);
      }
      auto body = loop.getBody();
      bool isBasicBlock = dyn_cast<AffineForOp>(&body->front()) == nullptr;
      if (isBasicBlock) {
        body->walk([&](Operation *op) {
          auto node = std::make_shared<Node>(op, loopNest);
          initGraph->drawNode(node);
        });
      }
      if (currAxis == nullptr) {
        initGraph->rootAxis->children.push_back(axis);
      } else {
        currAxis->children.push_back(axis);
      }
      currAxis = axis;
    }
  }
  return initGraph;
}

ModelGraphPtr buildModelGraph(InitGraphPtr initGraph) {
  auto hardware = initGraph->hardware;
  auto operatorTypes = initGraph->funcOp->getAttr("OperatorType").dyn_cast<StringAttr>();
  initGraph->setGraphType(operatorTypes);

  if (initGraph->graphType == "Reduce") {
    auto funcReductionAxes = CommonUtils::collectReductionAxesEachDim(initGraph->funcOp);
    initGraph->rootAxis->forEachAxisTopDown([&initGraph, &funcReductionAxes](const AxisPtr a) {
      if (a == nullptr || a->loop.get() == nullptr) {
        return;
      }
      for (auto dim = 0; dim < funcReductionAxes.size(); ++dim) {
        if (CommonUtils::isReduceAxis(funcReductionAxes[dim], a->loop->getOperation())) {
          (void)a->axisType.insert(Axis::AxisLabel::kReduction);
        }
      }
    });
  }

  TilingStrategyManagerPtr tilingMgr = std::make_shared<TilingStrategyManager>();
  if (akgglobal::GpuScheduleTool::getInstance().getIsCustomConfig()) {
    tilingMgr->addStrategy(std::make_shared<RepositoryStrategy>());
    auto modelGraph = std::make_shared<ModelGraph>(initGraph);
    tilingMgr->processOn(modelGraph);
    return modelGraph;
  }
  if (hardware == kTargetCuda) {
    return buildGpuModelGraph(initGraph, tilingMgr);
  } else if (hardware == kTargetCpu) {
    return buildCpuModelGraph(initGraph, tilingMgr);
  } else {
    llvm::errs() << "Not impl model graph for hardware " << hardware;
    return std::make_shared<ModelGraph>(initGraph);
  }
}

void UniquePrimeCollect(Operation *op) {
  auto &tool = akgglobal::PrimeNumTool::getInstance();
  op->walk([&](mlir::arith::ConstantOp constOp) {
    int constValue = 0;
    auto constValueAttr = constOp.getOperation()->getAttr("value");
    if (isa<IntegerAttr>(constValueAttr)) {
      constValue = static_cast<int>(constValueAttr.dyn_cast<IntegerAttr>().getInt());
    }
    tool.updateVisited(constValue);
  });
}

GpuModelGraphPtr buildGpuModelGraph(InitGraphPtr initGraph, const TilingStrategyManagerPtr tilingMgr) {
  auto gpuGraph = std::make_shared<GpuModelGraph>(initGraph);
  gpuGraph->funcOp = initGraph->funcOp;
  UniquePrimeCollect(initGraph->funcOp);

  gpuGraph->AnalyzeGraphTemplate();
  if (initGraph->funcOp->hasAttr(akg::utils::kEnableAtomicAdd)) {
    gpuGraph->globalConfigs[akg::utils::kEnableAtomicAdd] = initGraph->funcOp->getAttr(akg::utils::kEnableAtomicAdd);
  } else {
    OpBuilder builder(initGraph->funcOp);
    gpuGraph->globalConfigs[akg::utils::kEnableAtomicAdd] = builder.getBoolAttr(false);
  }

  gpuGraph->InitResource();
  if (gpuGraph->graphTemplate == GraphTemplate::REDUCTION) {
    tilingMgr->addStrategy(std::make_shared<ReduceStrategy>());
  } else {
    if (gpuGraph->graphTemplate == GraphTemplate::BROADCAST_OP) {
      tilingMgr->addStrategy(std::make_shared<BroadcastStrategy>());
    }
    if (gpuGraph->graphTemplate == GraphTemplate::TRANSPOSE_OP) {
      tilingMgr->addStrategy(std::make_shared<TransposeStrategy>());
    }
    tilingMgr->addStrategy(std::make_shared<DynamicShapeStrategy>());
    tilingMgr->addStrategy(std::make_shared<ParallelStrategy>());
  }
  tilingMgr->processOn(gpuGraph);
  return gpuGraph;
}

CpuModelGraphPtr buildCpuModelGraph(InitGraphPtr initGraph, const TilingStrategyManagerPtr tilingMgr) {
  auto cpuGraph = std::make_shared<CpuModelGraph>(initGraph);
  cpuGraph->funcOp = initGraph->funcOp;
  cpuGraph->AnalyzeGraphTemplate();
  if (cpuGraph->isDynamicShape) {
    tilingMgr->addStrategy(std::make_shared<UnrollStrategy>());
  } else {
    if (cpuGraph->graphTemplate == GraphTemplate::BROADCAST_OP) {
      tilingMgr->addStrategy(std::make_shared<BroadcastStrategy>());
    }
    tilingMgr->addStrategy(std::make_shared<UnrollStrategy>());
  }
  tilingMgr->processOn(cpuGraph);
  return cpuGraph;
}

TilingSolverPtr getHeuristicTilingSolver(ModelGraphPtr modelGraph) {
  modelGraph->name += "_AfterSolve";
  auto solver = std::make_shared<HeuristicTilingSolver>(modelGraph);
  solver->initMinSize();
  return solver;
}

void getTileSizeWithSolver(const TilingSolverPtr &solver, SmallVector<AffineForOp, 6> band,
                           SmallVectorImpl<unsigned> *tileSizes, const TilingTaskDesc &taskDesc) {
  size_t level = taskDesc.level;
  size_t bandIdx = taskDesc.bandIdx;
  std::map<unsigned, unsigned> resMap;
  if (solver->modelGraph == nullptr || solver->modelGraph->rootAxis == nullptr) {
    llvm::errs() << "Create model graph before solve.";
    return;
  }
  auto getAxisIdx = [&band](const AxisPtr a) {
    if (a == nullptr || a->loop == nullptr || a->loop.get() == nullptr) {
      return -1;
    }
    for (size_t i = 0; i < band.size(); ++i) {
      if (band[i].getInductionVar() == a->loop->getInductionVar()) {
        return static_cast<int>(i);
      }
    }
    return -1;
  };
  if (bandIdx >= solver->modelGraph->rootAxis->children.size()) {
    return;
  }
  auto TrySolve = [&solver, &getAxisIdx, &resMap, &level](const AxisPtr a) {
    auto axisIdx = getAxisIdx(a);
    if (axisIdx == -1) {
      return;
    }
    if (solver->solved.find(a) == solver->solved.end()) {
      solver->solve(a);
    }
    if (level < a->configs[kTileCfg].size()) {
      resMap[axisIdx] = a->configs[kTileCfg][level]->value;
    } else {
      resMap[axisIdx] = 1;
    }
  };
  auto sortedAxes = solver->sortAxis(bandIdx);
  for (auto axis : sortedAxes) {
    TrySolve(axis);
  }
  for (auto it : resMap) {
    tileSizes->push_back(it.second);
  }
}
}  // namespace autotiling
}  // namespace akg
}  // namespace mlir

