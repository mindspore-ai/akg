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

#include "mfusion/Analysis/FusionRegion/FusionRegionTag.h"

#include "mfusion/Dialect/Mfuse/Transforms/Outlining/FusionAttributes.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace mfuse {
namespace fusion_region {
namespace {

unsigned gNextGroupId = 0;

void setRoleAttr(Operation *op, FuseRole role) {
  OpBuilder b(op);
  llvm::StringRef roleStr =
      role == FuseRole::Member ? mfusion_attrs::kDvmFuseRoleMember : mfusion_attrs::kDvmFuseRoleAffinity;
  op->setAttr(mfusion_attrs::kDvmFuseRole, b.getStringAttr(roleStr));
}

void setGroupAndKind(Operation *op, llvm::StringRef groupId, llvm::StringRef kind, FuseRole role) {
  OpBuilder b(op);
  if (!kind.empty()) {
    op->setAttr(mfusion_attrs::kDvmFuseKind, b.getStringAttr(kind));
  }
  setRoleAttr(op, role);
  op->setAttr(mfusion_attrs::kDvmFuseGroup, b.getStringAttr(groupId));
}

}  // namespace

FuseRole getFuseRole(Operation *op) {
  if (!op) {
    return FuseRole::Member;
  }
  if (auto role = op->getAttrOfType<StringAttr>(mfusion_attrs::kDvmFuseRole)) {
    if (role.getValue() == mfusion_attrs::kDvmFuseRoleAffinity) {
      return FuseRole::Affinity;
    }
  }
  return FuseRole::Member;
}

bool hasRegionMember(Operation *op) {
  return op && op->hasAttr(mfusion_attrs::kDvmFuseGroup) && getFuseRole(op) == FuseRole::Member;
}

bool hasRegionAffinity(Operation *op) {
  return op && op->hasAttr(mfusion_attrs::kDvmFuseGroup) && getFuseRole(op) == FuseRole::Affinity;
}

bool isTagged(Operation *op) { return hasRegionMember(op) || hasRegionAffinity(op); }

bool shouldSkipSplitForMatcherFusedIsland(Operation *op) {
  if (!op) {
    return false;
  }
  if (auto kind = op->getAttrOfType<StringAttr>(mfusion_attrs::kDvmFuseKind)) {
    return kind.getValue() == kSafeSoftmaxFuseKind || kind.getValue() == kLayerNormFuseKind;
  }
  return false;
}

std::optional<llvm::StringRef> getMergeGroupId(Operation *op) {
  if (!op || getFuseRole(op) != FuseRole::Member) {
    return std::nullopt;
  }
  if (auto group = op->getAttrOfType<StringAttr>(mfusion_attrs::kDvmFuseGroup)) {
    return group.getValue();
  }
  return std::nullopt;
}

void collectMergeGroupIds(Operation *op, llvm::StringSet<> &groups) {
  if (auto group = getMergeGroupId(op)) {
    groups.insert(group->str());
  }
}

void collectAreaMergeGroupIds(ArrayRef<Operation *> ops, llvm::StringSet<> &groups) {
  for (Operation *op : ops) {
    collectMergeGroupIds(op, groups);
  }
}

std::string allocateGroupId(llvm::StringRef kind) {
  std::string id = kind.str();
  id.push_back('#');
  id.append(std::to_string(gNextGroupId++));
  return id;
}

void rollbackLastGroupId() {
  if (gNextGroupId > 0) {
    --gNextGroupId;
  }
}

void resetGroupIdAllocator() { gNextGroupId = 0; }

void tagMember(Operation *op, llvm::StringRef groupId, llvm::StringRef kind) {
  if (!op || hasRegionMember(op)) {
    return;
  }
  setGroupAndKind(op, groupId, kind, FuseRole::Member);
}

void tagAffinity(Operation *op, llvm::StringRef groupId, llvm::StringRef kind) {
  if (!op || hasRegionMember(op) || hasRegionAffinity(op)) {
    return;
  }
  setGroupAndKind(op, groupId, kind, FuseRole::Affinity);
}

void tagMembers(ArrayRef<Operation *> ops, llvm::StringRef groupId, llvm::StringRef kind) {
  for (Operation *op : ops) {
    tagMember(op, groupId, kind);
  }
}

void retagMember(Operation *op, llvm::StringRef groupId, llvm::StringRef kind) {
  if (!op) {
    return;
  }
  setGroupAndKind(op, groupId, kind, FuseRole::Member);
}

void retagAffinity(Operation *op, llvm::StringRef groupId, llvm::StringRef kind) {
  if (!op) {
    return;
  }
  setGroupAndKind(op, groupId, kind, FuseRole::Affinity);
}

}  // namespace fusion_region
}  // namespace mfuse
}  // namespace mlir
