/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MFUSION_ANALYSIS_SPLIT_SPLITMODEL_H
#define MFUSION_ANALYSIS_SPLIT_SPLITMODEL_H

#include <vector>
#include <list>
#include <memory>
#include <set>
#include <utility>
#include <unordered_map>
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mfusion/Analysis/Split/Area.h"
#include "mfusion/Analysis/Split/FusePattern.h"

namespace mlir {
namespace mfuse {
namespace split {

// Forward declarations
class FusePattern;

using FusePatternPtr = std::shared_ptr<FusePattern>;

/// ReachTable tracks reachability between areas and checks for circles
class ReachTable {
 public:
  explicit ReachTable(size_t size);
  virtual ~ReachTable() = default;

  /// Check if adding an edge between a and b would create a circle
  bool hasCircle(const AreaPtr &a, const AreaPtr &b) const;

  /// Link area from `from` to `to`
  void link(size_t from, size_t to);

  /// Fuse the area `target` and `other`. After that, the `other` area will be discarded
  void fuseArea(size_t target, size_t other);

 private:
  /// Check the reachability from `from` to `to`
  bool reachable(size_t from, size_t to) const { return reach_[from][to]; }

  size_t size_;
  std::vector<std::vector<bool>> reach_;
  std::set<size_t> alive_;
};

/// SplitModel is the base class for splitting models into areas
class SplitModel {
 public:
  void run(Block *block);
  const std::list<AreaPtr> &areas() const { return areas_; }
  SplitModel() = default;
  virtual ~SplitModel() = default;

 protected:
  /// Transform the function operation to areas, and initialize inner tables
  void initGraph(Block *block);

  /// Align shapes for block arguments and operations
  void alignShape(Block *block) const;

  /// Initialize fusion pattern list
  virtual void initFusePatterns() = 0;

  /// Run one fusion pattern
  bool runOnePattern(const FusePatternPtr &pattern);

  /// Fuse areas by pattern
  void runFusePatterns();

  /// Set default area mode when the area has only one node
  void setDefaultAreaMode(const AreaPtr &area) const { area->setMode(getDefaultAreaMode(area->dom())); }

  /// Get default area mode of the dominant node
  virtual AreaMode getDefaultAreaMode(Operation *node) const;

  /// Add new pattern
  void addPattern(const FusePatternPtr &pn, bool enable = true);

  /// Fuse areas
  void fuseAreas(const AreaPtr &dom, const std::vector<AreaPtr> &areas, FuseDirection direction);

  /// Create new area
  AreaPtr newArea(Operation *op, bool is_output = false);

  /// Limit the area's size
  void limitAreaSize(const AreaPtr &dom, std::vector<AreaPtr> *areas) const;

  /// Update the area's outputs
  void updateAreaOutput(const AreaPtr &area) const;

  std::list<AreaPtr> areas_;  // use std::list to accelerate the "erase"
  std::shared_ptr<ReachTable> reach_table_{nullptr};
  std::unordered_map<Operation *, AreaPtr> node_area_map_;

 private:
  size_t cur_area_id_{0};
  std::vector<std::pair<FusePatternPtr, bool>> patterns_;
};

}  // namespace split
}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_ANALYSIS_SPLIT_SPLITMODEL_H
