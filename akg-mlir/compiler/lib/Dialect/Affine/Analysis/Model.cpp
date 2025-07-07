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

#include "akg/Dialect/Affine/Analysis/Model.h"
#include "akg/Dialect/Affine/Analysis/TilingStrategy.h"

namespace mlir {
namespace akg {
namespace autotiling {
// The score will sum up to axis's priority.
constexpr auto kLoadScore = 1.0;
constexpr auto kStoreScore = 1.5;
template <typename T>
void Tensor::SetLoadTensor(T loadOp, const std::vector<AxisPtr> &loopNest) {
  opType = "Load";
  dataType = loadOp.getMemRefType().getElementType();
  Value loadResult = loadOp.getResult();
  SmallVector<Operation *, 8> loadRelatedFor;
  CommonUtils::collectRelatedAxes(loadResult, loadRelatedFor);
  for (size_t i = 0; i < loadRelatedFor.size(); ++i) {
    for (auto axis : loopNest) {
      if (axis->loop->getOperation() == loadRelatedFor[i]) {
        loopNest_.push_back(axis);
        axis->isInnerMost = (i == (loadRelatedFor.size() - 1)) || axis->isInnerMost;
        if (axis->isInnerMost) {
          axis->priority = axis->priority + kLoadScore;
        }
        break;
      }
    }
  }
}

template <typename T>
void Tensor::SetStoreTensor(T storeOp, const std::vector<AxisPtr> &loopNest) {
  opType = "Store";
  dataType = storeOp.getMemRefType().getElementType();
  auto indices = storeOp.getIndices();
  for (size_t i = 0; i < indices.size(); ++i) {
    for (auto axis : loopNest) {
      auto forOp = dyn_cast<affine::AffineForOp>(axis->loop->getOperation());
      auto inductionVar = forOp.getInductionVar();
      if (indices[i] == inductionVar) {
        axis->isInnerMost = (i == (indices.size() - 1)) || axis->isInnerMost;
        if (axis->isInnerMost) {
          axis->priority = axis->priority + kStoreScore;
        }
        loopNest_.push_back(axis);
        break;
      }
    }
  }
}

Tensor::Tensor(mlir::Operation *op, const std::vector<AxisPtr> &loopNest) : op_(op) {
  // 1. get op_type
  if (op->hasAttr("OperatorType")) {
    opType = dyn_cast<StringAttr>(op->getAttr("OperatorType")).getValue().str();
  } else if (op->hasAttr("reduction_axes")) {
    opType = "Reduce";
    loopNest_ = loopNest;
  } else if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
    SetLoadTensor<affine::AffineLoadOp>(loadOp, loopNest);
  } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
    SetLoadTensor<memref::LoadOp>(loadOp, loopNest);
  } else if (auto loadOp = dyn_cast<affine::AffineStoreOp>(op)) {
    SetStoreTensor<affine::AffineStoreOp>(loadOp, loopNest);
  } else if (auto loadOp = dyn_cast<memref::StoreOp>(op)) {
    SetStoreTensor<memref::StoreOp>(loadOp, loopNest);
  } else {
    loopNest_ = loopNest;
    if (isa<mlir::math::ExpOp, mlir::math::LogOp, mlir::math::TanhOp, mlir::math::SqrtOp>(op)) {
      opType = "HeavyElem";
    }
  }
}

InitGraph::InitGraph(const std::string &name) : name{name} {}
InitGraph::InitGraph(const std::string &name, const std::vector<std::shared_ptr<Node>> &nodes,
                     const std::vector<std::shared_ptr<Node>> &inputs,
                     const std::vector<std::shared_ptr<Node>> &outputs)
    : name{name}, nodes_{nodes}, inputs_{inputs}, outputs_{outputs} {}

void InitGraph::setGraphType(const StringAttr &attrs) { graphType = attrs.getValue().str(); }

void InitGraph::setHardware(const std::string &hw) { hardware = hw; }
void InitGraph::setFeature(const std::string &fea) { feature = fea; }
void InitGraph::setIsDynamicShape(bool isDyn) { isDynamicShape = isDyn; }
void InitGraph::setTilingMode(const std::string &tm) { tilingMode = tm; }

void InitGraph::dump() {
  llvm::outs() << "------Start Dump Graph: name = " << this->name << "------\n";
  if (this->rootAxis) {
    this->rootAxis->forEachAxisTopDown([](const AxisPtr a) { llvm::outs() << a->toString(); });
  }
  llvm::outs() << "Graph type :" << graphType << "\n";
  if (!nodes_.empty()) {
    llvm::outs() << "Nodes:\n";
    for (auto node : nodes_) {
      llvm::outs() << node->toString();
    }
  }
}

NodePtr InitGraph::getMaxRankTensor() {
  if (maxRankNode) {
    return maxRankNode;
  }
  for (auto node : nodes_) {
    if (node->opType != "Load" && node->opType != "Store") {
      continue;
    }
    if (maxRankNode == nullptr || maxRankNode->loopNest_.size() < node->loopNest_.size() ||
        (maxRankNode->loopNest_.size() == node->loopNest_.size() && node->opType == "Store" &&
         maxRankNode->opType == "Load")) {
      maxRankNode = node;
    }
  }
  return maxRankNode;
}

ModelGraph::ModelGraph(const InitGraphPtr &initGraph)
    : InitGraph(initGraph->name, initGraph->nodes_, initGraph->inputs_, initGraph->outputs_) {
  rootAxis = initGraph->rootAxis;
  graphType = initGraph->graphType;
  hardware = initGraph->hardware;
  feature = initGraph->feature;
  tilingMode = initGraph->tilingMode;
  isDynamicShape = initGraph->isDynamicShape;
}

GraphTemplate ModelGraph::AnalyzeTransposeGraph() {
  std::set<size_t> rwRank;
  std::map<size_t, std::vector<NodePtr>> sameDimReadTensors;
  std::map<size_t, std::vector<NodePtr>> sameDimWriteTensors;
  for (auto node : nodes_) {
    if (node->loopNest_.empty()) {
      continue;
    }
    (void)rwRank.insert(node->loopNest_.size());
    if (node->opType == "Load") {
      (void)sameDimReadTensors[node->loopNest_.size()].emplace_back(node);
    }
    if (node->opType == "Store") {
      (void)sameDimWriteTensors[node->loopNest_.size()].emplace_back(node);
    }
  }
  for (auto it : sameDimWriteTensors) {
    auto wtensors = it.second;
    if (sameDimReadTensors.find(it.first) == sameDimReadTensors.end()) {
      continue;
    }
    auto rtensors = sameDimReadTensors[it.first];
    for (auto wt : wtensors) {
      for (auto rt : rtensors) {
        for (size_t j = 0; j < wt->loopNest_.size(); ++j) {
          if (wt->loopNest_[j] != rt->loopNest_[j]) {
            return GraphTemplate::TRANSPOSE_OP;
          }
        }
      }
    }
  }
  return rwRank.size() == 1 ? GraphTemplate::PURE_ELEM : GraphTemplate::BROADCAST_OP;
}

void ModelGraph::AnalyzeGraphTemplate() {
  if (graphTemplate == GraphTemplate::DEFAULT) {
    if (graphType == "Reduce") {
      graphTemplate = GraphTemplate::REDUCTION;
    } else if (graphType == "Transpose") {
      graphTemplate = AnalyzeTransposeGraph();
      if (graphTemplate != GraphTemplate::TRANSPOSE_OP) {
        OpBuilder builder(funcOp->getContext());
        if (graphTemplate == GraphTemplate::BROADCAST_OP) {
          Attribute opType = builder.getStringAttr("Broadcast");
          funcOp->setAttr(kOperatorTypeStr, opType);
        } else if (graphTemplate == GraphTemplate::PURE_ELEM) {
          Attribute opType = builder.getStringAttr("Elementwise");
          funcOp->setAttr(kOperatorTypeStr, opType);
        }
      }
    } else if (graphType == "Broadcast" || graphType == "Reshape") {
      graphTemplate = GraphTemplate::BROADCAST_OP;
    } else if (graphType == "Elementwise") {
      graphTemplate = GraphTemplate::PURE_ELEM;
    } else {
      llvm::errs() << "Get Unknown graph type: " << graphType << "\n";
    }
  }
  llvm::outs() << "ModelGraph Template : " << ShowGraphTemplate() << "\n";
}

std::vector<int> ModelGraph::getLoopExtentsAfterTiling(const AxisPtr axis) const {
  std::vector<int> axisSizes{static_cast<int>(axis->range.second)};
  for (auto tile : axis->configs[kTileCfg]) {
    auto innerSize = tile->value;
    if (innerSize > 0) {
      axisSizes[axisSizes.size() - 1] = (axisSizes.back() - 1) / innerSize + 1;
      axisSizes.push_back(innerSize);
    } else {
      axisSizes.push_back(innerSize);
    }
  }
  return axisSizes;
}

GpuModelGraph::GpuModelGraph(const InitGraphPtr &initGraph) : ModelGraph(initGraph) {}

void GpuModelGraph::InitResource() {
  if (this->funcOp->getAttr("compute_capability")) {
    auto compute_capability = dyn_cast<StringAttr>(funcOp->getAttr("compute_capability")).getValue().str();
    if (compute_capability == "8.0") {
      hardware = akg::utils::kA100Device;
    } else {
      hardware = akg::utils::kV100Device;
    }
  }
  sortedAxes.clear();
  problemSize = 1;
  auto maxRankTensor = getMaxRankTensor();
  if (!maxRankTensor) {
    return;
  }

  for (auto axis : maxRankTensor->loopNest_) {
    sortedAxes.push_front(axis);
    problemSize *= static_cast<int>(axis->range.second);
  }
  gpuGrid.availbleSize = akg::utils::GpuInfo::getInstance(hardware).getMaxGrids();
  gpuGrid.totalAvailableSize = gpuGrid.availbleSize.front();
  gpuGrid.resourceType = kGpuGridCfg;
  gpuBlock.availbleSize = akg::utils::GpuInfo::getInstance(hardware).getMaxBlocks();
  gpuBlock.totalAvailableSize = akg::utils::GpuInfo::getInstance(hardware).getTotalAvailableBlocks();
  gpuBlock.resourceType = kGpuBlockCfg;
}

CpuModelGraph::CpuModelGraph(const InitGraphPtr &initGraph) : ModelGraph(initGraph) { name = "cpuGraph"; }

int64_t Resource::rest() {
  if (currSize == 0) {
    return 0;
  }
  return std::min<int64_t>(totalAvailableSize / currSize, availbleSize[currAllocDim()]);
}

bool Resource::canApply(int64_t size) {
  size_t applyDim = currAllocDim();
  if (size == -1 && applyDim < availbleSize.size()) {
    return true;
  }
  auto res = size > 1 && currSize * size <= totalAvailableSize && applyDim < availbleSize.size() &&
             size <= availbleSize[applyDim];
  return res;
}

ConfigPtr Resource::alloc(const AxisPtr axis, int64_t size) {
  ConfigPtr config = nullptr;
  if (size == -1 && currAllocDim() < availbleSize.size()) {
    // that is a place holder
    allocSize[axis] = 1;
    return config;
  }
  if (!canApply(size)) {
    return config;
  }
  if (resourceType == kGpuGridCfg) {
    config = std::make_shared<GpuGrid>("AutoAlloc");
    (void)axis->axisType.insert(Axis::AxisLabel::kMultiCore);
  } else if (resourceType == kGpuBlockCfg) {
    config = std::make_shared<GpuBlock>("AutoAlloc");
    (void)axis->axisType.insert(Axis::AxisLabel::kVectorization);
  }
  if (config) {
    config->value = size;
    axis->configs[config->type].push_back(config);
  }
  allocSize[axis] = size;
  currSize *= size;
  return config;
}
}  // namespace autotiling
}  // namespace akg
}  // namespace mlir
