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
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/IR/MfuseDialect.h"
#include "mfusion/Analysis/Split/FusePattern.h"
#include "mfusion/Analysis/Split/OpRegister.h"
#include "mfusion/Dialect/Dvm/IR/Dvm.h"
#include "mfusion/Dialect/Mfuse/Support/SymbolAttrUtils.h"
#include "mfusion/Analysis/SymbolicShape/SymExprBuilder.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/raw_ostream.h"
#include "mfusion/Support/Logging.h"

namespace mlir {
namespace mfuse {
namespace split {

namespace {
DShape getOutputShape(Operation *op) {
  // Assume we only care about the first result for now
  if (op->getResultTypes().empty()) {
    return {};
  }
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!tensorType) {
    return {};
  }
  DShape shape(tensorType.getShape().begin(), tensorType.getShape().end());
  return shape;
}

std::vector<SymExpr> getOutputSymShape(Operation *op) {
  if (op->getResultTypes().empty()) {
    return {};
  }
  auto maybeSymExprs = SymbolAttrUtils::getSymbolicShapeExprs(op->getResult(0).getType());
  if (mlir::succeeded(maybeSymExprs)) {
    return std::vector<SymExpr>(maybeSymExprs->begin(), maybeSymExprs->end());
  }
  return {};
}
}  // namespace

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
  for (auto &node : nodes_) {
    auto op = node->op();
    if (!check_pattern(op)) {
      // For non-elemwise/broadcast/reduce operations, ensure they have at least one dimension
      if (node->shape.empty()) {
        node->shape.push_back(1LL);
      }
      continue;
    }
    auto cur_shape_size = node->shape.size();
    for (auto &inp : node->inputs()) {
      if (inp->shape.size() > cur_shape_size) {
        cur_shape_size = inp->shape.size();
      }
    }
    if (cur_shape_size > node->shape.size()) {
      auto num = cur_shape_size - node->shape.size();
      (void)node->shape.insert(node->shape.cbegin(), num, 1LL);
      if (!node->sym_shape.empty()) {
        mfusion::SymExprBuilder builder;
        auto oneExpr = builder.makeInteger(1);
        (void)node->sym_shape.insert(node->sym_shape.cbegin(), num, oneExpr);
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
  for (auto node : nodes_) {
    // Skip terminator operations (like return)
    if (mlir::isa<mlir::mfuse::YieldOp>(node->op())) {
      continue;
    }
    bool is_output = outputs_set.find(node->op()) != outputs_set.end();
    newArea(node, is_output);
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

AreaPtr SplitModel::newArea(Node *node, bool is_output) {
  auto new_area = std::make_shared<Area>(cur_area_id_++, node, is_output, node_area_map_);
  areas_.emplace_back(new_area);
  node_area_map_[node] = new_area;
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
    for (auto &node : target->nodes()) {
      node_area_map_[node] = target;
    }
  } else {
    for (auto a : areas) {
      for (auto &node : a->nodes()) {
        node_area_map_[node] = target;
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
  for (auto &node : area->nodes()) {
    // Check if this operation has users outside the area
    for (auto [user, _] : node->users()) {
      // Find the area of the user operation
      auto iter = node_area_map_.find(user);
      if (iter == node_area_map_.end() || iter->second.get() != area.get()) {
        // Add this operation to area outputs
        area_outputs.push_back(node);
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
      MLOG(DEBUG) << "Area " << area->toString() << " matches " << pattern->toString();
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
    if (!enable) {
      continue;
    }
    MLOG(DEBUG) << "Run pattern " << pattern->name();
    runOnePattern(pattern);
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
AreaMode SplitModel::getDefaultAreaMode(Node *node) const {
  if (node == nullptr || node->op() == nullptr) {
    return AreaMode::COMPOSITE;
  }
  static constexpr llvm::StringLiteral kReshapeOpName = "mfuse.reshape";
  static constexpr llvm::StringLiteral kAssignOpName = "mfuse.assign";
  static constexpr llvm::StringLiteral kTransposeOpName = "mfuse.permute";
  static constexpr llvm::StringLiteral kCastOpName = "mfuse.cast";
  llvm::StringRef nodeName = node->op()->getName().getStringRef();
  if (nodeName == kReshapeOpName || nodeName == kAssignOpName) {
    return AreaMode::BASIC;
  }
  if (nodeName == kTransposeOpName || nodeName == kCastOpName) {
    return AreaMode::BASIC;
  }
  return AreaMode::COMPOSITE;
}

void SplitModel::addPattern(const FusePatternPtr &pn, bool enable) {
  patterns_.emplace_back(pn, enable);
  patterns_.back().first->setCircleChecker(reach_table_);
}

void SplitModel::mapOperationsToNodes(Block *block) {
  std::unordered_map<Operation *, Node *> op_node_map;
  size_t node_id = 0;
  for (auto &op : block->getOperations()) {
    nodes_ptrs_.emplace_back(std::make_unique<Node>(&op, node_id++));
    auto node = nodes_ptrs_.back().get();
    node->shape = getOutputShape(&op);
    node->sym_shape = getOutputSymShape(&op);
    nodes_.emplace_back(node);
    op_node_map[&op] = node;
  }
  for (auto &op : block->getOperations()) {
    auto *node = op_node_map[&op];
    for (auto operand : op.getOperands()) {
      if (auto *defOp = operand.getDefiningOp()) {
        node->addInput(op_node_map[defOp]);
      }
    }
  }
}

void SplitModel::run(Block *block) {
  mapOperationsToNodes(block);
  initGraph(block);
  MLOG(DEBUG) << "== Initial areas ==";
  for (auto &area : areas_) {
    MLOG(DEBUG) << area->toString() << ": " << area->dom()->toString();
  }
  initFusePatterns();
  runFusePatterns();
}

}  // namespace split
}  // namespace mfuse
}  // namespace mlir
