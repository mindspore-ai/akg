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
#include "akg/Dialect/MindSpore/Spliter/Area.h"

#include <algorithm>
#include <sstream>

namespace mlir::spliter {
namespace {
EdgeRelation getRelation(const PrimOpPtr &node, const NodePtr &input) {
  if (node->getComputeType() != NodePattern::ELEMWISE) {
    return EdgeRelation::INJECTIVE;
  }
  // naively set the edge relation to "broadcast" if the result shape is not
  // equal to the input shape.
  return (node->shape == input->shape) ? EdgeRelation::INJECTIVE : EdgeRelation::BROADCAST;
}

bool sameArea(const AreaWithRelation &a, const AreaWithRelation &b) { return a.first == b.first; }

bool areaWithRelationCmp(const AreaWithRelation &a, const AreaWithRelation &b) {
  // for same areas, put the area with greater EdgeRelation in front when
  // sorting. compare the areas with unique id, instead of Area pointer, to
  // avoid random result.
  return sameArea(a, b) ? (a.second > b.second) : (a.first->id() < b.first->id());
}
}  // namespace

Area::Area(size_t id, const PrimOpPtr &primOp, bool isOutput, const HashMap<NodePtr, AreaPtr> &nodeAreaMap)
    : hd(new NodeHandle(this, primOp)), uniqueId(id), isOutput(isOutput), ops(1, primOp) {
  // link inputs of the handle node
  for (auto &inp : primOp->getInputs()) {
    auto inputRelation = getRelation(primOp, inp);
    if (pattern() == NodePattern::ELEMWISE && inputRelation == EdgeRelation::BROADCAST) {
      hd->computeType = NodePattern::BROADCAST;
    }
    if (auto inpAreaIter = nodeAreaMap.find(inp); inpAreaIter != nodeAreaMap.end()) {
      (void)inputsWithRelation.emplace_back(std::make_pair(inpAreaIter->second, inputRelation));
    }
  }
  makeUniqueAndSyncInputs();
}

std::vector<AreaPtr> Area::getInputs() const {
  std::vector<AreaPtr> result;
  (void)std::transform(inputsWithRelation.begin(), inputsWithRelation.end(), std::back_inserter(result),
                       [](const AreaWithRelation &inp) { return inp.first; });
  return result;
}

std::vector<AreaPtr> Area::getUsers() const {
  std::vector<AreaPtr> result;
  (void)std::transform(hd->getUsers().begin(), hd->getUsers().end(), std::back_inserter(result), [](const auto &u) {
    Node *node = u.first;
    return node->as<NodeHandle>()->getArea();
  });
  return result;
}

std::vector<AreaWithRelation> Area::usersWithRelation() const {
  std::vector<AreaWithRelation> result;
  (void)std::transform(hd->getUsers().begin(), hd->getUsers().end(), std::back_inserter(result), [](const auto &u) {
    Node *node = u.first;
    auto area = node->as<NodeHandle>()->getArea();
    // the input edge of area is unique
    const auto relation = area->inputRelation(*(u.second.begin()));
    return std::make_pair(area, relation);
  });
  return result;
}

int64_t Area::computeSize() const {
  auto op = dom();
  assert(op != nullptr);
  return sizeToLong(op->tensorSize());
}

bool Area::hasSameComputeSize(const AreaPtr &other) const {
  auto op = dom();
  assert(op != nullptr);
  return op->hasSameTensorSize(other->dom());
}

std::string Area::toString() const {
  std::ostringstream oss;
  bool is_first = true;
  oss << "<";
  for (auto op : ops) {
    if (is_first) {
      is_first = false;
      oss << id() << ":";
    } else {
      oss << "-";
    }
    oss << op->getDebugName();
  }
  oss << ">";
  return oss.str();
}

void Area::makeUniqueAndSyncInputs() {
  // remove the repeated inputs, keep the area with greater EdgeRelation.
  std::sort(inputsWithRelation.begin(), inputsWithRelation.end(), areaWithRelationCmp);
  (void)inputsWithRelation.erase(std::unique(inputsWithRelation.begin(), inputsWithRelation.end(), sameArea),
                                 inputsWithRelation.cend());
  // sync the inputs to NodeHandle to maintain users
  this->hd->clearInputs();
  (void)std::for_each(inputsWithRelation.begin(), inputsWithRelation.end(),
                      [this](const AreaWithRelation &inp) { this->hd->addInput(inp.first->hd); });
}

void Area::updateUsersRelation(const AreaPtr &inputArea) {
  auto &userNodeWithIndex = inputArea->hd->getUsers();
  std::vector<AreaPtr> userAreas;
  for (auto &[user_hd, index] : userNodeWithIndex) {
    (void)userAreas.emplace_back(user_hd->as<NodeHandle>()->getArea());
    const auto idx = *(index.begin());
    userAreas.back()->inputsWithRelation[idx].first = this->shared_from_this();
  }
  // the inputs should be updated outside the above for-loop,
  // since the users cannot be updated while traversing.
  for (auto user : userAreas) {
    user->makeUniqueAndSyncInputs();
  }
}

void Area::fuseInput(const AreaPtr &inputArea) {
  auto iter = std::find_if(inputsWithRelation.begin(), inputsWithRelation.end(),
                           [&inputArea](const AreaWithRelation &a) { return a.first == inputArea; });
  if (iter == inputsWithRelation.end()) {
    llvm::errs() << "The area " << inputArea->toString() << " should be the input of area " << this->toString();
  }
  auto input_idx = longToSize(iter - inputsWithRelation.begin());

  if (inputArea->isOutput) {
    isOutput = true;
  }

  // Update ops, and discard the inputArea's ops.
  // The dominant node is ops[0], keep the dominant with greater pattern.
  if (pattern() < inputArea->pattern()) {
    ops.swap(inputArea->ops);
  }
  (void)ops.insert(ops.cend(), inputArea->ops.cbegin(), inputArea->ops.cend());

  // update area pattern
  hd->computeType = std::max(pattern(), inputArea->pattern());
  if ((pattern() == NodePattern::ELEMWISE) && (inputRelation(input_idx) == EdgeRelation::BROADCAST)) {
    hd->computeType = NodePattern::BROADCAST;
  }

  // update inputs and relations
  (void)inputsWithRelation.erase(iter);
  (void)inputsWithRelation.insert(inputsWithRelation.cend(), inputArea->inputsWithRelation.cbegin(),
                                  inputArea->inputsWithRelation.cend());
  makeUniqueAndSyncInputs();
  updateUsersRelation(inputArea);

  // clear the inputArea.
  inputArea->ops.clear();
  inputArea->inputsWithRelation.clear();
  inputArea->hd->clearInputs();
}
}  // namespace mlir::spliter
