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
#ifndef COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_MODEL_H_
#define COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_MODEL_H_

#include <list>
#include <memory>
#include <set>
#include <utility>
#include <vector>
#include "akg/Dialect/MindSpore/Spliter/Area.h"
#include "akg/Dialect/MindSpore/Spliter/FusePattern.h"
#include "akg/Dialect/MindSpore/Spliter/LiteGraph.h"

namespace mlir::spliter {
class ReachTable : public CircleChecker {
 public:
  explicit ReachTable(size_t size);
  virtual ~ReachTable() = default;
  bool hasCircle(const AreaPtr &a, const AreaPtr &b) const override;

  // Link area from `from` to `to`.
  void link(size_t from, size_t to);

  // Fuse the area `target` and `other`. After that, the `other` area will be discarded.
  void fuseArea(size_t target, size_t other);

 private:
  // check the reachability from `from` to `to`
  bool reachable(size_t from, size_t to) const { return reach[from][to]; }

  size_t size;
  std::vector<std::vector<bool>> reach;
  std::set<size_t> alive;
};

class Model {
 public:
  void run(const LiteGraphPtr &litegraph);
  const std::list<AreaPtr> &getAreas() const { return areas; }
  Model() = default;
  virtual ~Model() = default;

 protected:
  // transform the litegraph to areas, and initialize inner tables.
  void initGraph(const LiteGraphPtr &litegraph);
  // Push leading "1" to shapes to facilitate pattern match.
  void alignShape(const LiteGraphPtr &litegraph) const;
  // initialize fusion pattern list.
  virtual void initFusePatterns() = 0;
  bool runOnePattern(const FusePatternPtr &pattern);
  // fuse areas by pattern
  void runFusePatterns();
  // set default area mode when the area has only one node.
  void setDefaultAreaMode(const AreaPtr &area) const { area->setMode(getDefaultAreaMode(area->dom())); }
  // get default area mode of the dominant node
  virtual AreaMode getDefaultAreaMode(const PrimOpPtr &node) const = 0;
  // add new pattern
  void addPattern(const std::shared_ptr<FusePattern> &pn, bool enable = true);
  // fuse areas
  void fuseAreas(const AreaPtr &dom, const std::vector<AreaPtr> &areas, FuseDirection direction);
  // create new area
  AreaPtr newArea(const PrimOpPtr &op, bool isOutput);
  // limit the area's size
  void limitAreaSize(const AreaPtr &dom, std::vector<AreaPtr> *areas, size_t maxSize = 200) const;

  std::list<AreaPtr> areas;  // use std::list to accelerate the "erase"
  std::shared_ptr<ReachTable> reachTable{nullptr};
  HashMap<NodePtr, AreaPtr> nodeAreaMap;

 private:
  size_t curAreaId{0};
  std::vector<std::pair<FusePatternPtr, bool>> patterns;
};
using ModelPtr = std::shared_ptr<Model>;
}  // namespace mlir::spliter
#endif  // COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_MODEL_H_
