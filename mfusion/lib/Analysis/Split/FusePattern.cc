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
#include "mfusion/Analysis/Split/FusePattern.h"

#include "mfusion/Analysis/Split/Area.h"
#include "mfusion/Analysis/Split/SplitModel.h"

namespace mlir {
namespace mfuse {
namespace split {

void FusePattern::setCircleChecker(std::shared_ptr<ReachTable> checker) { circle_checker_ = checker; }

void FusePattern::reset() { fused_areas_.clear(); }

bool FusePattern::run(const AreaPtr &area) {
  reset();
  return check(area) && match(area);
}

bool FusePattern::hasCircle(const AreaPtr &area, const AreaPtr &fuse_area) {
  if (!circle_checker_) {
    llvm::report_fatal_error("Circle checker is not set");
  }
  return circle_checker_->hasCircle(area, fuse_area);
}

std::string FusePattern::toString() const {
  std::ostringstream oss;
  if (direction_ == FuseDirection::FORWARD) {
    oss << "Forward{";
  } else {
    oss << "Backward{";
  }
  bool first = true;
  for (auto &area : fused_areas_) {
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

// FuseReshape implementation
bool FuseReshape::match(const AreaPtr &area) {
  min_area_ = nullptr;

  for (auto &user : area->users()) {
    if (user->pattern() <= NodePattern::BROADCAST && !hasCircle(area, user)) {
      keepMinimumArea(user, FuseDirection::BACKWARD);
    }
  }

  for (auto &inp : area->inputs()) {
    if (inp->pattern() <= NodePattern::BROADCAST && !hasCircle(inp, area)) {
      keepMinimumArea(inp, FuseDirection::FORWARD);
    }
  }
  if (min_area_ == nullptr) {
    return false;
  }
  fused_areas_.push_back(min_area_);
  return true;
}

void FuseReshape::keepMinimumArea(const AreaPtr &a, FuseDirection dir) {
  if (min_area_ == nullptr || a->pattern() < min_area_->pattern()) {
    min_area_ = a;
    direction_ = dir;
  }
}

// FuseIsolateReshape implementation
bool FuseIsolateReshape::match(const AreaPtr &area) {
  for (auto &user : area->users()) {
    if (user->mode() == AreaMode::COMPOSITE && !hasCircle(area, user)) {
      fused_areas_.push_back(user);
      direction_ = FuseDirection::BACKWARD;
      return true;
    }
  }
  for (auto &inp : area->inputs()) {
    if (inp->mode() == AreaMode::COMPOSITE && !hasCircle(inp, area)) {
      fused_areas_.push_back(inp);
      direction_ = FuseDirection::FORWARD;
      return true;
    }
  }
  return false;
}

// FuseElemwiseBroadcastFwd implementation
bool FuseElemwiseBroadcastFwd::check(const AreaPtr &area) {
  if (area->pattern() != NodePattern::ELEMWISE && area->pattern() != NodePattern::BROADCAST) {
    return false;
  }
  return fuse_type_ == FuseType::kWidth || area->inputNum() == 1;
}

bool FuseElemwiseBroadcastFwd::match(const AreaPtr &area) {
  for (auto &[input, relation] : area->inputsWithRelation()) {
    if (fuse_type_ == FuseType::kDepth && input->userNum() != 1) {
      continue;
    }
    if (input->pattern() <= NodePattern::BROADCAST && relation == EdgeRelation::INJECTIVE) {
      if (fuse_type_ == FuseType::kWidth && hasCircle(input, area)) {
        continue;
      }
      if (input->computeSizeEqual(area)) {
        fused_areas_.push_back(input);
      }
    }
  }
  return !fused_areas_.empty();
}

// FuseDynElemwiseBroadcastFwd implementation
bool FuseDynElemwiseBroadcastFwd::check(const AreaPtr &area) {
  if (area->pattern() != NodePattern::ELEMWISE && area->pattern() != NodePattern::BROADCAST) {
    return false;
  }
  return fuse_type_ == FuseType::kWidth || area->inputNum() == 1;
}

bool FuseDynElemwiseBroadcastFwd::match(const AreaPtr &area) {
  for (auto &[input, relation] : area->inputsWithRelation()) {
    if (fuse_type_ == FuseType::kDepth && input->userNum() != 1) {
      continue;
    }
    if (input->pattern() <= NodePattern::BROADCAST && relation <= EdgeRelation::BROADCAST) {
      if (fuse_type_ == FuseType::kWidth && hasCircle(input, area)) {
        continue;
      }
      fused_areas_.push_back(input);
    }
  }
  return !fused_areas_.empty();
}

// FuseReduceFwd implementation
bool FuseReduceFwd::check(const AreaPtr &area) {
  if (area->pattern() != NodePattern::REDUCE) {
    return false;
  }
  return fuse_type_ == FuseType::kWidth || area->inputNum() == 1;
}

bool FuseReduceFwd::match(const AreaPtr &area) {
  for (auto &[input, relation] : area->inputsWithRelation()) {
    if (fuse_type_ == FuseType::kDepth && input->userNum() != 1) {
      continue;
    }
    if (input->size() > size_limit_) {
      continue;
    }
    if (input->pattern() <= NodePattern::ELEMWISE && relation == EdgeRelation::INJECTIVE) {
      if (fuse_type_ == FuseType::kWidth && hasCircle(input, area)) {
        continue;
      }
      fused_areas_.push_back(input);
    }
  }
  return !fused_areas_.empty();
}

// FuseDynReduceFwd implementation
bool FuseDynReduceFwd::check(const AreaPtr &area) {
  if (area->pattern() != NodePattern::REDUCE) {
    return false;
  }
  return fuse_type_ == FuseType::kWidth || area->inputNum() == 1;
}

bool FuseDynReduceFwd::match(const AreaPtr &area) {
  for (auto &[input, _] : area->inputsWithRelation()) {
    if (fuse_type_ == FuseType::kDepth && input->userNum() != 1) {
      continue;
    }
    if (input->size() > size_limit_) {
      continue;
    }
    if (input->pattern() <= NodePattern::ELEMWISE) {
      if (fuse_type_ == FuseType::kWidth && hasCircle(input, area)) {
        continue;
      }
      fused_areas_.push_back(input);
    }
  }
  return !fused_areas_.empty();
}

// FuseElemwiseBroadcastBwd implementation
bool FuseElemwiseBroadcastBwd::check(const AreaPtr &area) {
  if (area->pattern() != NodePattern::ELEMWISE && area->pattern() != NodePattern::BROADCAST) {
    return false;
  }
  if (area->isOutput()) {
    return false;
  }
  if (fuse_type_ == FuseType::kDepth && area->userNum() > 1) {
    return false;
  }
  return area->size() <= size_limit_;
}

bool FuseElemwiseBroadcastBwd::match(const AreaPtr &area) {
  // this pattern is to fuse ALL users of dom area,
  // since the broadcast node should not be an output when it fuse nodes in backward.
  for (auto &[a, r] : area->usersWithRelation()) {
    if (fuse_type_ == FuseType::kDepth && a->inputNum() != 1) {
      return false;
    }
    if (a->pattern() > NodePattern::REDUCE) {
      return false;
    }
    if (fuse_type_ == FuseType::kWidth) {
      if (!fused_areas_.empty() && !fused_areas_[0]->computeSizeEqual(a)) {
        return false;
      }
      if (hasCircle(area, a)) {
        continue;
      }
    }
    if (a->pattern() == NodePattern::REDUCE) {
      // elemwise + reduce
      if (area->pattern() == NodePattern::ELEMWISE && r == EdgeRelation::INJECTIVE) {
        fused_areas_.push_back(a);
      } else {
        return false;
      }
    } else {  // a->pattern() < NodePattern::REDUCE
      fused_areas_.push_back(a);
    }
  }
  return fused_areas_.size() == area->userNum();
}

// FuseVirtualNode implementation
bool FuseVirtualNode::match(const AreaPtr &area) {
  fused_areas_ = area->inputs();
  return true;
}

}  // namespace split
}  // namespace mfuse
}  // namespace mlir
