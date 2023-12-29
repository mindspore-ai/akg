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
#ifndef COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_AREA_H_
#define COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_AREA_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "akg/Dialect/MindSpore/Spliter/OpNode.h"
#include "akg/Dialect/MindSpore/Spliter/Utils.h"

namespace mlir::spliter {
using NodePattern = PrimOp::ComputeType;
// EdgeRelation indicates the pattern of node's input edges.
// the INJECTIVE means the input is directly sent into the kernel,
// the BROADCAST means the input is implicit broadcasted.
//
// Note, it should be distinguished from the PrimOp::ComputeType,
// which indicates the INNER logic of kernels.
enum class EdgeRelation : int { INJECTIVE = 0, BROADCAST = 1 };

// AreaMode indicates the finally mode of kernels.
// the BASIC means the node(s) of area will be inlined into the main graph
// the COMPOSITE means the node(s) of area will be kept as a GraphKernel node.
enum class AreaMode { BASIC, COMPOSITE };

class Area;
using AreaPtr = std::shared_ptr<Area>;
using AreaWithRelation = std::pair<AreaPtr, EdgeRelation>;

// Area is used to maintain the operator set that was fused.
class Area : public std::enable_shared_from_this<Area> {
  // NodeHandle is used to maintain the input and user edges of areas.
  // The handle's inputs should be other areas' handle.
  //
  // This class is derived from PrimOp, to reuse the compute_type field
  // and to avoid overriding pure virtual functions (if exists).
  //
  // This class is not visible outside the class Area.
  class NodeHandle : public PrimOp {
   public:
    NodeHandle(Area *newArea, const PrimOpPtr &p) : PrimOp("", p->getComputeType()), area(newArea) {}
    ~NodeHandle() = default;
    using PrimOp::computeType;
    AreaPtr getArea() const { return area->shared_from_this(); }

   private:
    Area *const area;
  };  // class Area::NodeHandle

 public:
  Area(size_t id, const PrimOpPtr &prim_op, bool isOutput, const HashMap<NodePtr, AreaPtr> &node_area_map);
  ~Area() = default;

  size_t id() const { return uniqueId; }
  const AreaPtr &getInput(size_t i) const { return inputsWithRelation[i].first; }
  std::vector<AreaPtr> getInputs() const;
  EdgeRelation inputRelation(size_t i) const { return inputsWithRelation[i].second; }
  const std::vector<AreaWithRelation> &getInputsWithRelation() const { return inputsWithRelation; }
  size_t inputNum() const { return inputsWithRelation.size(); }
  // get the number of operators in the area
  size_t size() const { return ops.size(); }
  std::vector<AreaPtr> getUsers() const;
  std::vector<AreaWithRelation> usersWithRelation() const;
  size_t userNum() const { return hd->getUsers().size(); }
  AreaMode getMode() const { return mode; }
  // get the dominant op node
  PrimOpPtr dom() const { return isAlive() ? ops[0] : nullptr; }
  NodePattern pattern() const { return hd->getComputeType(); }
  const std::vector<PrimOpPtr> &getOps() const { return ops; }
  bool OutputJudge() const { return isOutput; }
  int64_t computeSize() const;
  bool hasSameComputeSize(const AreaPtr &other) const;
  // check whether the area is alive(true) or is fused(false)
  bool isAlive() const { return !ops.empty(); }
  std::string toString() const;
  void setOps(const std::vector<PrimOpPtr> &newOps) { ops = newOps; }
  void setMode(AreaMode newMode) { mode = newMode; }
  // fuse `inputArea` into `this` area. after that, the `inputArea` will be
  // discarded. the `inputArea` node should be in the input list of `this`
  // area.
  void fuseInput(const AreaPtr &inputArea);

 protected:
  // Make the inputs unique, and sync the inputs to NodeHandle
  void makeUniqueAndSyncInputs();
  // Relink the `inputArea`'s users to `this` area
  void updateUsersRelation(const AreaPtr &inputArea);

  std::shared_ptr<NodeHandle> hd;
  const size_t uniqueId;
  bool isOutput;
  std::vector<PrimOpPtr> ops;
  AreaMode mode{AreaMode::BASIC};
  // The `inputsWithRelation.first` stores the input area of `this` area.
  // The `hd->inputs` stores the NodeHandle of `this` area, to maintain the
  // user edges. They should always be in sync.
  std::vector<AreaWithRelation> inputsWithRelation;
};
}  // namespace mlir::spliter
#endif  // COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_AREA_H_
