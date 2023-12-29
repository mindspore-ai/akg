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
#include "akg/Dialect/MindSpore/Spliter/FusePattern.h"

#include <sstream>

namespace mlir::spliter {
bool FuseReshape::match(const AreaPtr &dom) {
  minArea = nullptr;
  // Reshape nodes have at most one user, which is graranteed by the pass
  // "shape_ops_splitter".
  for (auto &user : dom->getUsers()) {
    if (user->pattern() <= NodePattern::BROADCAST && !hasCircle(dom, user)) {
      keepMinimumArea(user, FuseDirection::BACKWARD);
    }
  }

  for (auto &inp : dom->getInputs()) {
    if (inp->pattern() <= NodePattern::BROADCAST && !hasCircle(inp, dom)) {
      keepMinimumArea(inp, FuseDirection::FORWARD);
    }
  }
  if (minArea == nullptr) {
    return false;
  }
  (void)fusedAreas.emplace_back(minArea);
  return true;
}

void FuseReshape::keepMinimumArea(const AreaPtr &a, FuseDirection dir) {
  if (minArea == nullptr || a->pattern() < minArea->pattern()) {
    minArea = a;
    direction = dir;
  }
}

bool FuseIsolateReshape::match(const AreaPtr &dom) {
  for (auto &user : dom->getUsers()) {
    if (user->getMode() == AreaMode::COMPOSITE && !hasCircle(dom, user)) {
      (void)fusedAreas.emplace_back(user);
      direction = FuseDirection::BACKWARD;
      return true;
    }
  }
  for (auto &inp : dom->getInputs()) {
    if (inp->getMode() == AreaMode::COMPOSITE && !hasCircle(inp, dom)) {
      (void)fusedAreas.emplace_back(inp);
      direction = FuseDirection::FORWARD;
      return true;
    }
  }
  return false;
}

bool FuseElemwiseBroadcastFwd::check(const AreaPtr &dom) {
  if (dom->pattern() != NodePattern::ELEMWISE && dom->pattern() != NodePattern::BROADCAST) {
    return false;
  }
  return fuseType == FuseType::kWidth || dom->inputNum() == 1;
}

bool FuseElemwiseBroadcastFwd::match(const AreaPtr &dom) {
  for (auto &[a, r] : dom->getInputsWithRelation()) {
    // depth match only support one to one pattern
    if (fuseType == FuseType::kDepth && a->userNum() != 1) {
      continue;
    }
    if (a->pattern() <= NodePattern::BROADCAST && r <= EdgeRelation::BROADCAST) {
      // it's unnecessary to check circle for depth match
      if (fuseType == FuseType::kWidth && hasCircle(a, dom)) {
        continue;
      }
      (void)fusedAreas.emplace_back(a);
    }
  }
  return !fusedAreas.empty();
}

bool FuseReduceFwd::check(const AreaPtr &dom) {
  if (dom->pattern() != NodePattern::REDUCE) {
    return false;
  }
  return fuseType == FuseType::kWidth || dom->inputNum() == 1;
}

bool FuseReduceFwd::match(const AreaPtr &dom) {
  for (auto &input : dom->getInputsWithRelation()) {
    auto a = input.first;
    if (fuseType == FuseType::kDepth && a->userNum() != 1) {
      continue;
    }
    if (a->size() > sizeLimit) {
      continue;
    }
    if (a->pattern() <= NodePattern::BROADCAST) {
      // it's unnecessary to check circle for depth match
      if (fuseType == FuseType::kWidth && hasCircle(a, dom)) {
        continue;
      }
      (void)fusedAreas.emplace_back(a);
    }
  }
  return !fusedAreas.empty();
}

bool FuseElemwiseBroadcastBwd::check(const AreaPtr &dom) {
  if (dom->pattern() != NodePattern::ELEMWISE && dom->pattern() != NodePattern::BROADCAST) {
    return false;
  }
  if (dom->OutputJudge()) {
    return false;
  }
  if (fuseType == FuseType::kDepth && dom->userNum() > 1) {
    return false;
  }
  return dom->size() <= sizeLimit;
}

bool FuseElemwiseBroadcastBwd::match(const AreaPtr &dom) {
  // this pattern is to fuse ALL users of dom area,
  // since the broadcast node should not be an output when it fuse nodes in
  // backward.
  for (auto &[a, r] : dom->usersWithRelation()) {
    if (fuseType == FuseType::kDepth && a->inputNum() != 1) {
      return false;
    }
    if (a->pattern() > NodePattern::REDUCE) {
      return false;
    }
    if (fuseType == FuseType::kWidth) {
      if (hasCircle(dom, a)) {
        continue;
      }
    }
    if (a->pattern() == NodePattern::REDUCE) {
      // elemwise + reduce
      if (dom->pattern() == NodePattern::ELEMWISE && r <= EdgeRelation::BROADCAST) {
        (void)fusedAreas.emplace_back(a);
      } else {
        return false;
      }
    } else {  // a->pattern() < NodePattern::REDUCE
      (void)fusedAreas.emplace_back(a);
    }
  }
  return fusedAreas.size() == dom->userNum();
}

std::string FusePattern::toString() const {
  std::ostringstream oss;
  if (direction == FuseDirection::FORWARD) {
    oss << "Forward {";
  } else {
    oss << "Backward {";
  }
  bool first = true;
  for (auto &area : fusedAreas) {
    if (first) {
      first = false;
    } else {
      oss << ",";
    }
    oss << area->toString();
  }
  oss << "}";
  return oss.str();
}

bool FuseVirtualNode::match(const AreaPtr &area) {
  fusedAreas = area->getInputs();
  return true;
}
}  // namespace mlir::spliter
