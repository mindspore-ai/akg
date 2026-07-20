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

#include "mfusion/Analysis/Split/FuseTagBarrier.h"

#include "mfusion/Analysis/FusionRegion/FusionRegionTag.h"
#include "mfusion/Dialect/Mfuse/Transforms/Outlining/FusionAttributes.h"

namespace mlir {
namespace mfuse {
namespace split {

FuseTagBarrier::FuseTagBarrier(std::string name, llvm::ArrayRef<llvm::StringRef> mergeAttrs, FuseDirection direction)
    : FusePattern(name, direction), merge_attrs_storage_(mergeAttrs.begin(), mergeAttrs.end()) {
  merge_attrs_.reserve(merge_attrs_storage_.size());
  for (const auto &attr : merge_attrs_storage_) {
    merge_attrs_.push_back(attr);
  }
}

bool FuseTagBarrier::areaHasMergeAttr(const AreaPtr &area) const {
  if (!area) {
    return false;
  }
  for (auto *node : area->nodes()) {
    if (!node || !node->op()) {
      continue;
    }
    Operation *op = node->op();
    for (llvm::StringRef attr : merge_attrs_) {
      if (op->hasAttr(attr)) {
        return true;
      }
    }
  }
  return false;
}

bool FuseTagBarrier::check(const AreaPtr &area) { return areaHasMergeAttr(area); }

bool FuseTagBarrier::match(const AreaPtr &area) {
  if (direction() == FuseDirection::FORWARD) {
    for (const auto &[inp, relation] : area->inputsWithRelation()) {
      if (hasCircle(area, inp) || relation > EdgeRelation::BROADCAST) {
        continue;
      }
      if (areaHasMergeAttr(inp)) {
        fused_areas_.push_back(inp);
      }
    }
  } else {
    for (const auto &[user, relation] : area->usersWithRelation()) {
      if (hasCircle(area, user) || relation > EdgeRelation::BROADCAST) {
        continue;
      }
      if (areaHasMergeAttr(user)) {
        fused_areas_.push_back(user);
      }
    }
  }
  return !fused_areas_.empty();
}

FuseTagBarrierByGroupId::FuseTagBarrierByGroupId(std::string name, FuseDirection direction)
    : FusePattern(std::move(name), direction) {}

void FuseTagBarrierByGroupId::collectAreaGroups(const AreaPtr &area, llvm::StringSet<> &groups) const {
  if (!area) {
    return;
  }
  for (auto *node : area->nodes()) {
    if (!node || !node->op()) {
      continue;
    }
    fusion_region::collectMergeGroupIds(node->op(), groups);
  }
}

bool FuseTagBarrierByGroupId::groupsIntersect(const llvm::StringSet<> &lhs, const llvm::StringSet<> &rhs) {
  for (auto it = lhs.begin(), end = lhs.end(); it != end; ++it) {
    if (rhs.contains(it->getKey())) {
      return true;
    }
  }
  return false;
}

bool FuseTagBarrierByGroupId::check(const AreaPtr &area) {
  llvm::StringSet<> groups;
  collectAreaGroups(area, groups);
  return !groups.empty();
}

bool FuseTagBarrierByGroupId::match(const AreaPtr &area) {
  llvm::StringSet<> selfGroups;
  collectAreaGroups(area, selfGroups);
  if (selfGroups.empty()) {
    return false;
  }

  if (direction() == FuseDirection::FORWARD) {
    for (const auto &[inp, relation] : area->inputsWithRelation()) {
      if (hasCircle(area, inp) || relation > EdgeRelation::BROADCAST) {
        continue;
      }
      llvm::StringSet<> peerGroups;
      collectAreaGroups(inp, peerGroups);
      if (groupsIntersect(selfGroups, peerGroups)) {
        fused_areas_.push_back(inp);
      }
    }
  } else {
    for (const auto &[user, relation] : area->usersWithRelation()) {
      if (hasCircle(area, user) || relation > EdgeRelation::BROADCAST) {
        continue;
      }
      llvm::StringSet<> peerGroups;
      collectAreaGroups(user, peerGroups);
      if (groupsIntersect(selfGroups, peerGroups)) {
        fused_areas_.push_back(user);
      }
    }
  }
  return !fused_areas_.empty();
}

FusePatternPtr createDvmFuseGroupTagBarrier(FuseDirection direction) {
  const char *suffix = direction == FuseDirection::FORWARD ? "fwd" : "bwd";
  return std::make_shared<FuseTagBarrierByGroupId>(std::string("tag_barrier_dvm_fuse_group_") + suffix, direction);
}

FusePatternPtr createLayerNormDvmTagBarrier(FuseDirection direction) {
  return createDvmFuseGroupTagBarrier(direction);
}

}  // namespace split
}  // namespace mfuse
}  // namespace mlir
