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
#include "akg/Dialect/MindSpore/Spliter/Model.h"

#include <algorithm>
#include "akg/Dialect/MindSpore/Spliter/Utils.h"

#define DEBUG_TYPE "akg-mindspore-spliter-model"

namespace mlir::spliter {
ReachTable::ReachTable(size_t newSize) : size(newSize), reach(newSize, std::vector<bool>(newSize, false)) {
  for (size_t i = 0; i < newSize; ++i) {
    reach[i][i] = true;
    (void)alive.insert(i);
  }
}

void ReachTable::link(size_t from, size_t to) {
  // if there's an edge <from, to>, the `from` can reach to `to`'s succeeding
  // areas. so we connect `from` to all succeeding areas of `to`.
  for (const size_t suc : alive) {
    if (reachable(to, suc)) {
      reach[from][suc] = true;
    }
  }
}

void ReachTable::fuseArea(size_t target, size_t other) {
  // if `suc` is the succeeding nodes of other_node,
  // link the target_node's previous nodes to `suc`.
  for (const size_t suc : alive) {
    if (reachable(other, suc) && !reachable(target, suc)) {
      for (const size_t pre : alive) {
        if (reachable(pre, target)) {
          reach[pre][suc] = true;
        }
      }
    }
  }
  // if `pre` is the previous nodes of other_node,
  // link `pre` to target_node's succeeding nodes.
  for (const size_t pre : alive) {
    if (reachable(pre, other) && !reachable(pre, target)) {
      for (const size_t suc : alive) {
        if (reachable(target, suc)) {
          reach[pre][suc] = true;
        }
      }
    }
  }
  // discard other_node.
  (void)alive.erase(other);
}

bool ReachTable::hasCircle(const AreaPtr &a, const AreaPtr &b) const {
  // a is the input of b
  if (reachable(a->id(), b->id())) {
    // use `inputs_with_relation` instead of `inputs` to avoid generating a new
    // vector.
    for (auto &inp : b->getInputsWithRelation()) {
      if (inp.first != a && reachable(a->id(), inp.first->id())) {
        return true;
      }
    }
  } else {
    // b is the input of a
    for (auto &inp : a->getInputsWithRelation()) {
      if (inp.first != b && reachable(b->id(), inp.first->id())) {
        return true;
      }
    }
  }
  return false;
}

AreaPtr Model::newArea(const PrimOpPtr &op, bool isOutput) {
  auto newArea = std::make_shared<Area>(curAreaId++, op->as<PrimOp>(), isOutput, nodeAreaMap);
  (void)areas.emplace_back(newArea);
  nodeAreaMap[op] = newArea;
  setDefaultAreaMode(newArea);
  return newArea;
}

void Model::alignShape(const LiteGraphPtr &litegraph) const {
  for (auto &inp : litegraph->getInputs()) {
    if (inp->shape.empty()) {
      inp->shape.push_back(1LL);
    }
  }
  auto checkPattern = [](const NodePtr &op) {
    auto pn = op->as<PrimOp>()->getComputeType();
    return pn == NodePattern::ELEMWISE || pn == NodePattern::BROADCAST || pn == NodePattern::REDUCE;
  };
  for (auto &op : litegraph->getOps()) {
    if (!checkPattern(op)) {
      if (op->shape.empty()) {
        op->shape.push_back(1LL);
      }
      continue;
    }
    auto curShapeSize = op->shape.size();
    for (auto &inp : op->getInputs()) {
      if (inp->shape.size() > curShapeSize) {
        curShapeSize = inp->shape.size();
      }
    }
    if (curShapeSize > op->shape.size()) {
      auto num = curShapeSize - op->shape.size();
      (void)op->shape.insert(op->shape.cbegin(), num, 1LL);
    }
  }
}

void Model::initGraph(const LiteGraphPtr &litegraph) {
  alignShape(litegraph);
  auto &outputs = litegraph->getOutputs();
  HashSet<NodePtr> outputsSet(outputs.begin(), outputs.end());
  for (const auto &op : litegraph->getOps()) {
    if (op->nodeType() != NType::Primitive) {
      llvm::errs() << "Op " << op->getDebugName() << " should be a Primitive node, but got " << op->nodeType();
    }
    bool isOutput = (outputsSet.count(op) > 0);
    (void)newArea(op->as<PrimOp>(), isOutput);
  }

  // Initialize reach table in reversed topological order
  reachTable = std::make_shared<ReachTable>(litegraph->getOps().size());
  assert(reachTable != nullptr);
  for (auto iter = areas.rbegin(); iter != areas.rend(); ++iter) {
    auto users = (*iter)->getUsers();
    for (auto &user : users) {
      reachTable->link((*iter)->id(), user->id());
    }
  }
}

void Model::addPattern(const std::shared_ptr<FusePattern> &pn, bool enable) {
  (void)patterns.emplace_back(std::make_pair(pn, enable));
  patterns.back().first->setCircleChecker(reachTable);
}

void Model::limitAreaSize(const AreaPtr &dom, std::vector<AreaPtr> *areas, size_t maxSize) const {
  auto domSize = dom->size();
  for (auto a = areas->begin(); a != areas->end(); ++a) {
    domSize += (*a)->size();
  }
  if (domSize <= maxSize) {
    return;
  }
  // fuse the smaller area in priority
  std::sort(areas->begin(), areas->end(),
            [maxSize](const AreaPtr &a, const AreaPtr &b) { return a->size() < b->size(); });
  const auto iter =
    std::find_if(areas->begin(), areas->end(), [curSize = dom->size(), maxSize](const AreaPtr &a) mutable {
      curSize += a->size();
      return curSize > maxSize;
    });
  (void)areas->erase(iter, areas->cend());
}

void Model::fuseAreas(const AreaPtr &dom, const std::vector<AreaPtr> &areas, FuseDirection direction) {
  if (areas.empty()) {
    return;
  }
  auto target = dom;
  for (auto a : areas) {
    if (direction == FuseDirection::BACKWARD) {
      // always use back node to fuse the front node.
      std::swap(target, a);
    }
    for (auto &op : a->getOps()) {
      nodeAreaMap[op] = target;
    }
    target->fuseInput(a);
    reachTable->fuseArea(target->id(), a->id());
  }
  if (target->pattern() > NodePattern::RESHAPE) {
    target->setMode(AreaMode::COMPOSITE);
  }
}

bool Model::runOnePattern(const FusePatternPtr &pattern) {
  // in one step, we only match the adjacent areas of the "area",
  // so if matched, we should handle the same area again in the next step
  bool changed = false;
  for (auto iter = areas.begin(); iter != areas.end();) {
    auto area = *iter;
    if (!area->isAlive()) {
      iter = areas.erase(iter);
      continue;
    }
    if (pattern->run(area)) {
      LLVM_DEBUG(llvm::dbgs() << "Area " << area->toString() << " matches " << pattern->toString());
      limitAreaSize(area, &pattern->fusedAreas);
      if (!pattern->fusedAreas.empty()) {
        fuseAreas(area, pattern->fusedAreas, pattern->getDirection());
        changed = true;
        continue;
      }
    }
    ++iter;
  }
  return changed;
}

void Model::runFusePatterns() {
  // process one pattern for all areas before process next pattern.
  for (auto &[pattern, enable] : patterns) {
    if (!enable) {
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << "run pattern " << pattern->getName());
    (void)runOnePattern(pattern);
  }
  // remove the areas that is fused
  for (auto iter = areas.begin(); iter != areas.end();) {
    if (!(*iter)->isAlive()) {
      iter = areas.erase(iter);
    } else {
      ++iter;
    }
  }
}

void Model::run(const LiteGraphPtr &litegraph) {
  initGraph(litegraph);
  initFusePatterns();
  runFusePatterns();
}
}  // namespace mlir::spliter
