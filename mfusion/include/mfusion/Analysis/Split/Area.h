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

#ifndef MFUSION_ANALYSIS_SPLIT_AREA_H
#define MFUSION_ANALYSIS_SPLIT_AREA_H

#include <memory>
#include <vector>
#include <utility>
#include <string>
#include <unordered_map>
#include "mfusion/Analysis/Split/Node.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace mfuse {
namespace split {

// EdgeRelation indicates the pattern of node's input edges.
// the INJECTIVE means the input is directly sent into the kernel,
// the BROADCAST means the input is implicit broadcasted.
enum class EdgeRelation : int { INJECTIVE = 0, BROADCAST = 1 };

// AreaMode indicates the finally mode of kernels.
// the BASIC means the node(s) of area will be inlined into the main graph
// the COMPOSITE means the node(s) of area will be kept as a GraphKernel node.
enum class AreaMode { BASIC, COMPOSITE };

// NodePattern indicates the compute type of operations
enum class NodePattern {
  VIRTUAL = 0,
  RESHAPE = 1,
  ELEMWISE = 2,
  BROADCAST = 3,
  REDUCE = 4,
  OPAQUE = 5,
};

class Area;
using AreaPtr = std::shared_ptr<Area>;
using AreaWithRelation = std::pair<AreaPtr, EdgeRelation>;

/// Area is used to maintain the operator set that was fused
class Area : public std::enable_shared_from_this<Area> {
  // NodeHandle is used to maintain the input and user edges of areas.
  // The handle's inputs should be other areas' handle.
  class NodeHandle : public Node {
   public:
    NodeHandle(Area *area, NodePattern pattern) : area_(area), pattern_(pattern) {}
    ~NodeHandle() = default;

    NodePattern pattern() const { return pattern_; }
    void setPattern(NodePattern pattern) { pattern_ = pattern; }

    AreaPtr area() const { return area_->shared_from_this(); }

   private:
    Area *const area_;
    NodePattern pattern_;
  };

 public:
  /// Constructor
  Area(size_t id, Operation *op, bool is_output, const std::unordered_map<Operation *, AreaPtr> &node_area_map);
  ~Area() = default;

  /// Get the ID of this area
  size_t id() const { return unique_id_; }

  /// Get the input area at the given index
  const AreaPtr &input(size_t i) const { return inputs_with_relation_[i].first; }

  /// Get all input areas
  std::vector<AreaPtr> inputs() const;

  /// Get the edge relation for the input at the given index
  EdgeRelation inputRelation(size_t i) const { return inputs_with_relation_[i].second; }

  /// Get all inputs with their relations
  const std::vector<AreaWithRelation> &inputsWithRelation() const { return inputs_with_relation_; }

  /// Get the outputs of this area
  std::vector<Operation *> &areaOutputs() { return area_outputs_; }

  /// Get the number of inputs
  size_t inputNum() const { return inputs_with_relation_.size(); }

  /// Get the number of operators in the area
  size_t size() const { return ops_.size(); }

  /// Get all user areas
  std::vector<AreaPtr> users() const;

  /// Get all users with their relations
  std::vector<AreaWithRelation> usersWithRelation() const;

  /// Get the number of users
  size_t userNum() const { return users_.size(); }

  /// Get the mode of this area
  AreaMode mode() const { return mode_; }

  /// Get the dominant operation in this area
  Operation *dom() const { return isAlive() ? ops_[0] : nullptr; }

  /// Get the pattern of this area
  NodePattern pattern() const { return hd_->pattern(); }

  /// Get all operations in this area
  const std::vector<Operation *> &ops() const { return ops_; }

  /// Check if this area is an output area
  bool isOutput() const { return is_output_; }

  /// Calculate the compute size of this area
  int64_t computeSize() const;

  /// Check if the compute size is equal to another area
  bool computeSizeEqual(const AreaPtr &other) const;

  /// Check whether the area is alive (not fused)
  bool isAlive() const { return !ops_.empty(); }

  /// Convert to string representation
  std::string toString() const;

  /// Set the operations in this area
  void setOps(const std::vector<Operation *> &ops) { ops_ = ops; }

  /// Set the mode of this area
  void setMode(AreaMode mode) { mode_ = mode; }

  /// Fuse an input area into this area
  void fuseInput(const AreaPtr &input_area);

 protected:
  /// Make the inputs unique and sync them
  void makeUniqueAndSyncInputs();

  /// Update user relations when fusing areas
  void updateUsersRelation(const AreaPtr &input_area);

  std::shared_ptr<NodeHandle> hd_;
  const size_t unique_id_;
  bool is_output_ = false;
  std::vector<Operation *> ops_;
  AreaMode mode_{AreaMode::BASIC};
  std::vector<AreaWithRelation> inputs_with_relation_;
  std::vector<AreaWithRelation> users_;
  std::vector<Operation *> area_outputs_;
};

}  // namespace split
}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_ANALYSIS_SPLIT_AREA_H
