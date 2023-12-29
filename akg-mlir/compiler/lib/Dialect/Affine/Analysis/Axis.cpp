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

#include "akg/Dialect/Affine/Analysis/Axis.h"

#include <deque>
#include <stack>
#include <queue>

#include "akg/Utils/AnalysisForGpu.hpp"

namespace mlir {
namespace akg {
namespace autotiling {

Axis::Axis(size_t bandIdx, size_t axisIdx, const std::shared_ptr<AffineForOp> loop)
    : bandIdx(bandIdx), axisIdx(axisIdx), loop(loop) {
  auto nameSuffix = std::to_string(bandIdx) + "_" + std::to_string(axisIdx);
  this->name = "Axis_" + nameSuffix;
  this->initRange();
  this->initConfigs();
}

Axis::Axis(const std::string &name) : name(name) {}

void Axis::forEachAxisTopDown(const std::function<void(AxisPtr)> &fn) const {
  std::deque<AxisPtr> stack;
  for (auto &i : this->children) {
    stack.push_front(i);
  }
  while (!stack.empty()) {
    AxisPtr a = stack.back();
    if (a == nullptr) {
      return;
    }
    stack.pop_back();
    fn(a);
    for (int i = static_cast<int>(a->children.size()) - 1; i >= 0; --i) {
      stack.push_back(a->children[i]);
    }
  }
}

void Axis::forEachAxisBottomUp(const std::function<void(AxisPtr)> &fn) const {
  std::stack<AxisPtr> stack;
  std::queue<AxisPtr> queue;
  for (auto &i : this->children) {
    queue.push(i);
  }

  while (!queue.empty()) {
    AxisPtr a = queue.back();
    if (a == nullptr) {
      continue;
    }
    queue.pop();
    stack.push(a);
    for (int i = static_cast<int>(a->children.size()) - 1; i >= 0; --i) {
      queue.push(a->children[i]);
    }
  }

  while (!stack.empty()) {
    AxisPtr a = stack.top();
    stack.pop();
    fn(a);
  }
}

std::string Axis::toString() {
  std::stringstream ss;
  ss << "|Axis " << name << " axisType: [";
  for (auto type : axisType) {
    ss << std::to_string(static_cast<int>(type)) << ", ";
  }
  ss << "]\n";
  ss << "|-> Range: (" << range.first << ", " << range.second << ")\n";
  ss << "|->  [\n";
  for (auto it : configs) {
    ss << "|---> ConfigType: " << it.first << ": [\n";
    for (auto cfg : it.second) {
      ss << "        " << cfg->toString();
    }
    ss << "    ]\n";
  }
  ss << "|->  ]\n";
  return ss.str();
}

void Axis::initRange() {
  if (this->loop == nullptr) {
    return;
  }
  int fakeConst = 1;
  if (!this->loop->hasConstantLowerBound()) {
    this->range.first = fakeConst;
    (void)this->axisType.insert(AxisLabel::kDynamic);
  } else {
    this->range.first = this->loop->getConstantLowerBound();
  }
  if (!this->loop->hasConstantUpperBound()) {
    this->range.second = fakeConst;
    (void)this->axisType.insert(AxisLabel::kDynamic);
  } else {
    this->range.second = this->loop->getConstantUpperBound();
  }
}

void Axis::initConfigs() { doExtraTile(); }

void Axis::doExtraTile() {
  auto tileName = "lv_" + std::to_string(this->configs[kTileCfg].size());
  if (this->configs[kTileCfg].empty()) {
    this->configs[kTileCfg].push_back(std::make_shared<Tile>(tileName, this->range.second));
  } else {
    // Move last tile to middle and make current tile to inner
    this->configs[kTileCfg].back()->index = ConfigPos::kMiddle;
    this->configs[kTileCfg].push_back(std::make_shared<Tile>(tileName, this->configs[kTileCfg].back()->getMax()));
  }
}

void Axis::setMappings(const std::vector<std::string> &maps) { mappings = maps; }

void Axis::tryAddConstraint(int pos, const Constraint &cons, const std::string &configType) {
  auto config = tryGetConfig(pos, configType);
  if (config == nullptr) {
    llvm::errs() << "Add constraint fail, " << configType << " not in config map.\n";
    return;
  }
  config->constraints.push_back(cons);
}

int64_t Axis::getRestExtent() {
  int64_t loopExtent = range.second;
  auto mapSeq = tryGetConfig(0, kGpuSeqCfg);
  if (mapSeq && mapSeq->value > 0) {
    loopExtent /= mapSeq->value;
  }
  auto mapBlock = tryGetConfig(0, kGpuBlockCfg);
  if (mapBlock && mapBlock->value > 0) {
    loopExtent /= mapBlock->value;
  }
  auto mapGrid = tryGetConfig(0, kGpuGridCfg);
  if (mapGrid && mapGrid->value > 0) {
    loopExtent /= mapGrid->value;
  }
  return loopExtent;
}

ConfigPtr Axis::tryGetConfig(int pos, const std::string &configType) {
  if (configs.find(configType) == configs.end()) {
    return nullptr;
  }
  if (pos < 0) {
    pos += static_cast<int>(configs[configType].size());
  }
  if (pos >= static_cast<int>(configs[configType].size())) {
    return nullptr;
  }
  return configs[configType][pos];
}
}  // namespace autotiling
}  // namespace akg
}  // namespace mlir

