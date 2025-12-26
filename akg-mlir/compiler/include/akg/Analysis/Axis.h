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

#ifndef COMPILER_INCLUDE_AKG_ANALYSIS_AXIS_H_
#define COMPILER_INCLUDE_AKG_ANALYSIS_AXIS_H_
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <functional>
#include "akg/Analysis/Config.h"

namespace mlir {
namespace affine {
class AffineForOp;
}  // namespace affine
}  // namespace mlir

namespace mlir {
namespace scf {
class ForOp;
}  // namespace scf
}  // namespace mlir

namespace mlir {
namespace autotiling {
// Use types from mlir::autotiling namespace
using mlir::autotiling::Constraint;
using mlir::autotiling::ConfigPtr;
using mlir::autotiling::kTileCfg;
using mlir::autotiling::kGpuGridCfg;
using mlir::autotiling::kGpuBlockCfg;
using mlir::autotiling::kGpuSeqCfg;
using mlir::autotiling::ConfigPos;
using mlir::autotiling::Tile;

class Axis;
using AxisPtr = std::shared_ptr<Axis>;
class Axis {
 public:
  enum AxisLabel { kDefault = 0, kMultiCore, kUnroll, kVectorization, kReduction, kDynamic };
  explicit Axis(const std::string &name);
  Axis(size_t bandIdx, size_t axisIdx, affine::AffineForOp affineLoop);
  Axis(size_t bandIdx, size_t axisIdx, mlir::scf::ForOp scfLoop);
  Axis() = default;
  void forEachAxisTopDown(const std::function<void(AxisPtr)> &fn) const;
  void forEachAxisBottomUp(const std::function<void(AxisPtr)> &fn) const;
  void initRange();
  void initConfigs();
  void doExtraTile();
  void tryAddConstraint(int pos, const Constraint &cons, const std::string &configType = kTileCfg);
  ConfigPtr tryGetConfig(int pos, const std::string &configType = kTileCfg);
  int64_t getRestExtent();
  std::string toString();
  void setMappings(const std::vector<std::string> &maps);

  // Unified access interfaces
  bool hasConstantLowerBound() const;
  bool hasConstantUpperBound() const;
  bool hasConstantBounds() const;
  int64_t getConstantLowerBound() const;
  int64_t getConstantUpperBound() const;
  mlir::Value getLowerBound() const;
  mlir::Value getUpperBound() const;
  mlir::Value getInductionVar() const;
  mlir::Value getStep() const;
  mlir::Operation* getLoopOperation() const;

  std::string name;
  size_t bandIdx;
  size_t axisIdx;
  bool isInnerMost{false};
  std::shared_ptr<mlir::Operation> loop{nullptr};
  std::pair<int64_t, int64_t> range{std::make_pair(0, 0)};
  std::vector<AxisPtr> children;
  std::set<AxisLabel> axisType;
  std::map<std::string, std::vector<mlir::autotiling::ConfigPtr>> configs;
  std::vector<std::string> mappings;
  float priority{0.0};
};
}  // namespace autotiling
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_ANALYSIS_AXIS_H_

