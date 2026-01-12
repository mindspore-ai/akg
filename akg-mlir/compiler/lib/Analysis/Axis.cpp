/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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

#include "akg/Analysis/Axis.h"

#include <algorithm>
#include <deque>
#include <iterator>
#include <stack>
#include <queue>
#include <sstream>
#include <optional>
#include <functional>

#include "akg/Utils/AnalysisForGpu.hpp"
#include "akg/Analysis/Config.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace autotiling {
// Use types from mlir::akg::autotiling namespace
using mlir::autotiling::Constraint;
using mlir::autotiling::ConfigPtr;
using mlir::autotiling::kTileCfg;
using mlir::autotiling::kGpuGridCfg;
using mlir::autotiling::kGpuBlockCfg;
using mlir::autotiling::kGpuSeqCfg;
using mlir::autotiling::ConfigPos;
using mlir::autotiling::Tile;

// Helper function to get constant index value from Value
static std::optional<int64_t> getConstantIndexValue(mlir::Value value) {
  if (auto constOp = value.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
    return constOp.value();
  }
  if (auto constOp = value.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
      return intAttr.getInt();
    }
  }
  return std::nullopt;
}

Axis::Axis(size_t bandIdx, size_t axisIdx, affine::AffineForOp affineLoop)
    : bandIdx(bandIdx), axisIdx(axisIdx) {
  auto nameSuffix = std::to_string(bandIdx) + "_" + std::to_string(axisIdx);
  this->name = "Axis_" + nameSuffix;
  // Use empty deleter since MLIR manages Operation lifetime
  this->loop = std::shared_ptr<mlir::Operation>(
    affineLoop.getOperation(),
    [](Operation*){ /* MLIR manages lifetime */ });
  this->initRange();
  this->initConfigs();
}

Axis::Axis(size_t bandIdx, size_t axisIdx, mlir::scf::ForOp scfLoop)
    : bandIdx(bandIdx), axisIdx(axisIdx) {
  auto nameSuffix = std::to_string(bandIdx) + "_" + std::to_string(axisIdx);
  this->name = "Axis_" + nameSuffix;
  // Use empty deleter since MLIR manages Operation lifetime
  this->loop = std::shared_ptr<mlir::Operation>(
    scfLoop.getOperation(),
    [](Operation*){ /* MLIR manages lifetime */ });
  this->initRange();
  this->initConfigs();
}

Axis::Axis(const std::string &name) : name(name) {}

void Axis::forEachAxisTopDown(const std::function<void(AxisPtr)> &fn) const {
  std::deque<AxisPtr> stack;
  std::copy(this->children.rbegin(), this->children.rend(), std::front_inserter(stack));
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
  if (!hasConstantLowerBound()) {
    this->range.first = fakeConst;
    (void)this->axisType.insert(AxisLabel::kDynamic);
  } else {
    this->range.first = getConstantLowerBound();
  }
  if (!hasConstantUpperBound()) {
    this->range.second = fakeConst;
    (void)this->axisType.insert(AxisLabel::kDynamic);
  } else {
    this->range.second = getConstantUpperBound();
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

// Unified access interface implementations
mlir::Operation* Axis::getLoopOperation() const {
  return loop.get();
}

bool Axis::hasConstantLowerBound() const {
  if (!loop) return false;
  if (auto affineFor = mlir::dyn_cast<affine::AffineForOp>(loop.get())) {
    return affineFor.hasConstantLowerBound();
  }
  if (auto scfFor = mlir::dyn_cast<mlir::scf::ForOp>(loop.get())) {
    return getConstantIndexValue(scfFor.getLowerBound()).has_value();
  }
  return false;
}

bool Axis::hasConstantUpperBound() const {
  if (!loop) return false;
  if (auto affineFor = mlir::dyn_cast<affine::AffineForOp>(loop.get())) {
    return affineFor.hasConstantUpperBound();
  }
  if (auto scfFor = mlir::dyn_cast<mlir::scf::ForOp>(loop.get())) {
    return getConstantIndexValue(scfFor.getUpperBound()).has_value();
  }
  return false;
}

bool Axis::hasConstantBounds() const {
  return hasConstantLowerBound() && hasConstantUpperBound();
}

int64_t Axis::getConstantLowerBound() const {
  if (!loop) return 0;
  if (auto affineFor = mlir::dyn_cast<affine::AffineForOp>(loop.get())) {
    return affineFor.getConstantLowerBound();
  }
  if (auto scfFor = mlir::dyn_cast<mlir::scf::ForOp>(loop.get())) {
    auto lb = getConstantIndexValue(scfFor.getLowerBound());
    return lb.value_or(0);
  }
  return 0;
}

int64_t Axis::getConstantUpperBound() const {
  if (!loop) return 0;
  if (auto affineFor = mlir::dyn_cast<affine::AffineForOp>(loop.get())) {
    return affineFor.getConstantUpperBound();
  }
  if (auto scfFor = mlir::dyn_cast<mlir::scf::ForOp>(loop.get())) {
    auto ub = getConstantIndexValue(scfFor.getUpperBound());
    return ub.value_or(0);
  }
  return 0;
}

mlir::Value Axis::getLowerBound() const {
  if (!loop) return nullptr;
  // AffineForOp bounds are AffineMap, not Value, so return nullptr for affine loops
  if (mlir::isa<affine::AffineForOp>(loop.get())) {
    return nullptr;
  }
  if (auto scfFor = mlir::dyn_cast<mlir::scf::ForOp>(loop.get())) {
    return scfFor.getLowerBound();
  }
  return nullptr;
}

mlir::Value Axis::getUpperBound() const {
  if (!loop) return nullptr;
  // AffineForOp bounds are AffineMap, not Value, so return nullptr for affine loops
  if (mlir::isa<affine::AffineForOp>(loop.get())) {
    return nullptr;
  }
  if (auto scfFor = mlir::dyn_cast<mlir::scf::ForOp>(loop.get())) {
    return scfFor.getUpperBound();
  }
  return nullptr;
}

mlir::Value Axis::getInductionVar() const {
  if (!loop) return nullptr;
  if (auto affineFor = mlir::dyn_cast<affine::AffineForOp>(loop.get())) {
    return affineFor.getInductionVar();
  }
  if (auto scfFor = mlir::dyn_cast<mlir::scf::ForOp>(loop.get())) {
    return scfFor.getInductionVar();
  }
  return nullptr;
}

mlir::Value Axis::getStep() const {
  if (!loop) return nullptr;
  // AffineForOp step is APInt, not Value, so return nullptr for affine loops
  if (mlir::isa<affine::AffineForOp>(loop.get())) {
    return nullptr;
  }
  if (auto scfFor = mlir::dyn_cast<mlir::scf::ForOp>(loop.get())) {
    return scfFor.getStep();
  }
  return nullptr;
}

}  // namespace autotiling
}  // namespace mlir

