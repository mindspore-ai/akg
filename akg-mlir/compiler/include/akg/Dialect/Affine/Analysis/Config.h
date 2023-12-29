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

#ifndef COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_CONFIG_H_
#define COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_CONFIG_H_
#include <climits>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "mlir/Dialect/Affine/IR/AffineOps.h"

namespace mlir {
namespace akg {
namespace autotiling {
constexpr auto kTileCfg = "Tile";
constexpr auto kGpuGridCfg = "GpuGrid";
constexpr auto kGpuBlockCfg = "GpuBlock";
constexpr auto kGpuSeqCfg = "GpuSeq";
class Constraint {
 public:
  Constraint(int min, int max, int step, const std::vector<int> &candidates = {})
      : min(min), max(max), step(step), candidates(candidates) {}
  explicit Constraint(const std::vector<int> &candidates) : candidates(candidates) {}
  Constraint() = default;
  int min{1};
  int max{1};
  int step{1};
  std::vector<int> candidates;
  std::string toString() {
    std::stringstream ss;
    ss << "(" << min << " to " << max << " step " << step << ");";
    if (!candidates.empty()) {
      ss << " Candidates:[";
      for (auto cand : candidates) {
        ss << cand << ", ";
      }
      ss << "]";
    }
    return ss.str();
  }
};
enum ConfigPos { kOuter = 0, kMiddle, kInner, kAuto};
class Config;
using ConfigPtr = std::shared_ptr<Config>;
class Config {
 public:
  Config(const std::string &name, const std::string &type, int max = INT_MAX)
      : name(type + "_" + name), type(type), maxValue(max) {}
  virtual ~Config() = default;
  std::string name;
  std::string type;
  int value{0};
  int maxValue;
  ConfigPos index{ConfigPos::kAuto};
  Constraint finalConstraint{Constraint(1, maxValue, 1)};
  std::vector<int> validCandidates;
  std::vector<Constraint> constraints;
  int getMax() const { return maxValue; }
  virtual std::string toString() {
    std::stringstream ss;
    ss << type << ": " << name << " value " << value << " index " << std::to_string(static_cast<int>(index));
    ss << " final cons: " << finalConstraint.toString() << "\n";
    if (!constraints.empty()) {
      ss << "       All cons: [";
      for (auto cons : constraints) {
        ss << "            " << cons.toString() << "\n";
      }
      ss << "        ]\n";
    }
    return ss.str();
  }
  void mergeConstraints();
  std::vector<int> getValidCandidates();
  int mapDim{-1};
};

class Tile : public Config {
 public:
  explicit Tile(const std::string &name, int max = INT_MAX) : Config(name, kTileCfg, max) {
    this->index = ConfigPos::kInner;
  }
  virtual ~Tile() = default;
};
using TilePtr = std::shared_ptr<Tile>;

class GpuGrid : public Config {
 public:
  explicit GpuGrid(const std::string &name, int max = INT_MAX) : Config(name, kGpuGridCfg, max) {
    this->index = ConfigPos::kOuter;
  }
  virtual ~GpuGrid() = default;
};

class GpuBlock : public Config {
 public:
  explicit GpuBlock(const std::string &name, int max = INT_MAX) : Config(name, kGpuBlockCfg, max) {
    this->index = ConfigPos::kInner;
  }
  virtual ~GpuBlock() = default;
};

class GpuSeq : public Config {
 public:
  explicit GpuSeq(int max = INT_MAX) : Config("", kGpuSeqCfg, max) {
    this->value = max;
    this->index = ConfigPos::kInner;
  }
  virtual ~GpuSeq() = default;
};
}  // namespace autotiling
}  // namespace akg
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_CONFIG_H_

