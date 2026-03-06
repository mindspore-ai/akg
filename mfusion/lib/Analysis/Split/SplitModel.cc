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

#include "mfusion/Analysis/Split/SplitModel.h"

#include <algorithm>
#include <unordered_set>
#include "mfusion/Dialect/Mfuse/Mfuse.h"
#include "mfusion/Dialect/Mfuse/MfuseDialect.h"
#include "mfusion/Analysis/Split/FusePattern.h"
#include "mfusion/Analysis/Split/OpRegister.h"
#include "mfusion/Dialect/Dvm/Dvm.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "split"

namespace mlir {
namespace mfuse {
namespace split {

ReachTable::ReachTable(size_t size) : size_(size), reach_(size, std::vector<bool>(size, false)) {
  for (size_t i = 0; i < size_; ++i) {
    reach_[i][i] = true;
    alive_.insert(i);
  }
}

void ReachTable::link(size_t from, size_t to) {
  // If there's an edge <from, to>, the `from` can reach to `to`'s succeeding areas
  for (const size_t suc : alive_) {
    if (reachable(to, suc)) {
      reach_[from][suc] = true;
    }
  }
}

void ReachTable::fuseArea(size_t target, size_t other) {
  // If `suc` is the succeeding nodes of other_node,
  // link the target_node's previous nodes to `suc`
  for (const size_t suc : alive_) {
    if (reachable(other, suc) && !reachable(target, suc)) {
      for (const size_t pre : alive_) {
        if (reachable(pre, target)) {
          reach_[pre][suc] = true;
        }
      }
    }
  }

  // If `pre` is the previous nodes of other_node,
  // link `pre` to target_node's succeeding nodes
  for (const size_t pre : alive_) {
    if (reachable(pre, other) && !reachable(pre, target)) {
      for (const size_t suc : alive_) {
        if (reachable(target, suc)) {
          reach_[pre][suc] = true;
        }
      }
    }
  }

  // Discard other_node
  alive_.erase(other);
}

bool ReachTable::hasCircle(const AreaPtr &a, const AreaPtr &b) const {
  // a is the input of b
  if (reachable(a->id(), b->id())) {
    // use `inputsWithRelation` instead of `inputs` to avoid generating a new vector.
    for (auto &inp : b->inputsWithRelation()) {
      if (inp.first != a && reachable(a->id(), inp.first->id())) {
        return true;
      }
    }
  } else {
    // b is the input of a
    for (auto &inp : a->inputsWithRelation()) {
      if (inp.first != b && reachable(b->id(), inp.first->id())) {
        return true;
      }
    }
  }
  return false;
}

void SplitModel::alignShape(Block *block) const {
  // Check if operation is elemwise, broadcast, or reduce
  auto check_pattern = [](Operation *op) {
    // Check operation type to determine if it's elemwise, broadcast, or reduce
    std::string opName = op->getName().getStringRef().str();
    NodePattern compute_type = OpRegistry::Instance().GetPattern(opName);
    return compute_type == NodePattern::ELEMWISE || compute_type == NodePattern::BROADCAST ||
           compute_type == NodePattern::REDUCE;
  };

  // Align shapes for all operations
  for (auto &op : block->getOperations()) {
    if (!check_pattern(&op)) {
      // For non-elemwise/broadcast/reduce operations, ensure they have at least one dimension
      for (auto result : op.getResults()) {
        if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(result.getType())) {
          if (tensorType.getShape().empty()) {
            auto newType = mlir::RankedTensorType::get({}, tensorType.getElementType());
            result.setType(newType);
          }
        }
      }
      continue;
    }
    // For elemwise/broadcast/reduce operations, align shape with inputs
    size_t maxRank = 0;
    // Get max rank from results
    for (auto result : op.getResults()) {
      if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(result.getType())) {
        maxRank = std::max(maxRank, (size_t)tensorType.getShape().size());
      }
    }
    // Get max rank from inputs
    for (size_t i = 0; i < op.getNumOperands(); ++i) {
      auto operand = op.getOperand(i);
      if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(operand.getType())) {
        maxRank = std::max(maxRank, (size_t)tensorType.getShape().size());
      }
    }
    // Align result shapes
    for (auto result : op.getResults()) {
      if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(result.getType())) {
        size_t currentRank = tensorType.getShape().size();
        if (currentRank < maxRank) {
          // Prepend 1s to match max rank
          SmallVector<int64_t> newShape(maxRank, 1);
          auto oldShape = tensorType.getShape();
          std::copy(oldShape.begin(), oldShape.end(), newShape.begin() + (maxRank - currentRank));
          auto newType = mlir::RankedTensorType::get(newShape, tensorType.getElementType());
          result.setType(newType);
        }
      }
    }
  }
}

void SplitModel::initGraph(Block *block) {
  // Align shapes first
  alignShape(block);
  std::unordered_set<Operation *> outputs_set;
  Operation *terminator = block->getTerminator();
  if (!terminator) {
    llvm::errs() << "Block has no terminator!\n";
    return;
  }

  // Iterate over the terminator's operands (these are the block's outputs)
  for (auto operand : terminator->getOperands()) {
    if (auto *defOp = operand.getDefiningOp()) {
      outputs_set.insert(defOp);
    }
  }

  // Initialize areas for all operations
  for (auto &op : block->getOperations()) {
    // Skip terminator operations (like return)
    if (mlir::isa<mlir::mfuse::YieldOp>(&op)) {
      continue;
    }
    bool is_output = outputs_set.find(&op) != outputs_set.end();
    newArea(&op, is_output);
  }

  // Initialize reach table
  reach_table_ = std::make_shared<ReachTable>(areas_.size());

  // Link areas based on dependencies in reversed order
  for (auto iter = areas_.rbegin(); iter != areas_.rend(); ++iter) {
    auto *area = iter->get();
    auto users = area->users();
    for (auto &user : users) {
      reach_table_->link(area->id(), user->id());
    }
  }
}

AreaPtr SplitModel::newArea(Operation *op, bool is_output) {
  auto new_area = std::make_shared<Area>(cur_area_id_++, op, is_output, node_area_map_);
  areas_.emplace_back(new_area);
  node_area_map_[op] = new_area;
  setDefaultAreaMode(new_area);
  updateAreaOutput(new_area);
  return new_area;
}

void SplitModel::fuseAreas(const AreaPtr &dom, const std::vector<AreaPtr> &areas, FuseDirection direction) {
  if (areas.empty()) {
    return;
  }
  auto target = dom;
  if (direction == FuseDirection::BACKWARD) {
    for (auto a : areas) {
      // always use back node to fuse the front node.
      std::swap(target, a);
      target->fuseInput(a);
      reach_table_->fuseArea(target->id(), a->id());
    }
    for (auto &op : target->ops()) {
      node_area_map_[op] = target;
    }
  } else {
    for (auto a : areas) {
      for (auto &op : a->ops()) {
        node_area_map_[op] = target;
      }
      target->fuseInput(a);
      reach_table_->fuseArea(target->id(), a->id());
    }
  }
  if (target->pattern() > NodePattern::RESHAPE) {
    target->setMode(AreaMode::COMPOSITE);
  }
  updateAreaOutput(target);
}

void SplitModel::limitAreaSize(const AreaPtr &dom, std::vector<AreaPtr> *areas) const {
  const uint64_t kMaxAreaSize = 200;
  // Calculate current size
  uint64_t dom_size = dom->size();
  for (auto a = areas->begin(); a != areas->end(); ++a) {
    dom_size += (*a)->size();
  }
  // Check if size is within limit
  if (dom_size <= kMaxAreaSize) {
    return;
  }
  // Sort areas by size (smallest first)
  std::sort(areas->begin(), areas->end(), [](const AreaPtr &a, const AreaPtr &b) { return a->size() < b->size(); });
  // Find the first area that causes the total size to exceed the limit
  auto iter = std::find_if(areas->begin(), areas->end(), [cur_size = dom->size()](const AreaPtr &a) mutable {
    cur_size += a->size();
    return cur_size > kMaxAreaSize;
  });
  if (iter != areas->end()) {
    areas->erase(iter, areas->cend());
  }
}

void SplitModel::updateAreaOutput(const AreaPtr &area) const {
  // Get area outputs and clear them
  auto &area_outputs = area->areaOutputs();
  area_outputs.clear();

  // Iterate through all operations in the area
  for (auto &op : area->ops()) {
    // Check if this operation has users outside the area
    for (auto user : op->getUsers()) {
      // Find the area of the user operation
      auto iter = node_area_map_.find(user);
      if (iter == node_area_map_.end() || iter->second.get() != area.get()) {
        // Add this operation to area outputs
        area_outputs.push_back(op);
        break;  // No need to check other users
      }
    }
  }
}

bool SplitModel::runOnePattern(const FusePatternPtr &pattern) {
  // in one step, we only match the adjacent areas of the "area",
  // so if matched, we should handle the same area again in the next step
  bool fused = false;
  for (auto iter = areas_.begin(); iter != areas_.end();) {
    auto area = *iter;
    if (!area->isAlive()) {
      iter = areas_.erase(iter);
      continue;
    }
    if (pattern->run(area)) {
      LLVM_DEBUG(llvm::dbgs() << "Area " << area->ToString() << " matches " << pattern->ToString() << "\n");
      auto &fused_areas = const_cast<std::vector<AreaPtr> &>(pattern->fused_areas());
      limitAreaSize(area, &fused_areas);
      if (!fused_areas.empty()) {
        fuseAreas(area, fused_areas, pattern->direction());
        fused = true;
        continue;
      }
    }
    ++iter;
  }
  return fused;
}

void SplitModel::runFusePatterns() {
  // Run each pattern
  for (auto &[pattern, enable] : patterns_) {
    if (enable) {
      runOnePattern(pattern);
    }
  }

  // Remove fused areas
  for (auto iter = areas_.begin(); iter != areas_.end();) {
    if (!(*iter)->isAlive()) {
      iter = areas_.erase(iter);
    } else {
      ++iter;
    }
  }
}

// Get default area mode of the dominant node
// Function for ascend processors, GPU and CPU need override this function
AreaMode SplitModel::getDefaultAreaMode(Operation *node) const {
  if (node == nullptr) {
    return AreaMode::COMPOSITE;
  }
  static constexpr llvm::StringLiteral kReshapeOpName = "mfuse.reshape";
  static constexpr llvm::StringLiteral kAssignOpName = "mfuse.assign";
  static constexpr llvm::StringLiteral kTransposeOpName = "mfuse.permute";
  static constexpr llvm::StringLiteral kCastOpName = "mfuse.cast";
  llvm::StringRef nodeName = node->getName().getStringRef();
  if (nodeName == kReshapeOpName || nodeName == kAssignOpName) {
    return AreaMode::BASIC;
  }
  if (nodeName == kTransposeOpName || nodeName == kCastOpName) {
    return AreaMode::BASIC;
  }
  return AreaMode::COMPOSITE;
}

void SplitModel::addPattern(const FusePatternPtr &pn, bool enable) {
  LLVM_DEBUG(llvm::dbgs() << "Adding pattern, enable: " << (enable ? "true" : "false") << "\n");
  patterns_.emplace_back(pn, enable);
  patterns_.back().first->setCircleChecker(reach_table_);
}

void SplitModel::run(Block *block) {
  initGraph(block);
  initFusePatterns();
  runFusePatterns();
}

}  // namespace split
}  // namespace mfuse
}  // namespace mlir
