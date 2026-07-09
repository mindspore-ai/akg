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

#ifndef MFUSION_ANALYSIS_FUSIONREGION_FUSIONREGIONTAG_H
#define MFUSION_ANALYSIS_FUSIONREGION_FUSIONREGIONTAG_H

#include <optional>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace mfuse {
namespace fusion_region {

enum class FuseRole { Member, Affinity };

/// Fuse kind string for LayerNorm; legacy unit attrs are written only for this kind.
inline constexpr llvm::StringLiteral kLayerNormFuseKind = "layer_norm";

/// Fuse kind for bool-mask select / where with broadcast condition.
inline constexpr llvm::StringLiteral kBroadcastCondSelectFuseKind = "broadcast_cond_select";

/// Fuse kind for decomposed softmax + broadcast masked select (_safe_softmax semantics).
inline constexpr llvm::StringLiteral kSafeSoftmaxFuseKind = "safe_softmax";

FuseRole getFuseRole(Operation *op);
bool hasRegionMember(Operation *op);
bool hasRegionAffinity(Operation *op);
bool isTagged(Operation *op);

/// True when a matcher-materialized mfuse.fused island (e.g. safe-softmax) carries
/// dvm_fuse_kind on its wrapper and must not be fragmented by the split cost model.
/// LayerNorm uses top-level multi-op tags + FuseTagBarrierByGroupId instead.
bool shouldSkipSplitForMatcherFusedIsland(Operation *op);

/// Returns group id if this op participates in split merge (member or legacy LN unit tag).
std::optional<llvm::StringRef> getMergeGroupId(Operation *op);

void collectMergeGroupIds(Operation *op, llvm::StringSet<> &groups);
void collectAreaMergeGroupIds(ArrayRef<Operation *> ops, llvm::StringSet<> &groups);

std::string allocateGroupId(llvm::StringRef kind);
/// Reset the per-process group counter so each tagging pass assigns ids from `{kind}#0`.
void resetGroupIdAllocator();

void tagMember(Operation *op, llvm::StringRef groupId, llvm::StringRef kind = {});
void tagAffinity(Operation *op, llvm::StringRef groupId, llvm::StringRef kind = {});
void tagMembers(ArrayRef<Operation *> ops, llvm::StringRef groupId, llvm::StringRef kind = {});
/// Overwrite existing fusion-region tags (e.g. upgrade broadcast_cond_select → safe_softmax).
void retagMember(Operation *op, llvm::StringRef groupId, llvm::StringRef kind = {});
void retagAffinity(Operation *op, llvm::StringRef groupId, llvm::StringRef kind = {});

inline bool isSingleUse(Value value) { return value.hasOneUse(); }

inline Operation *getSingleUserOp(Value value) {
  if (!value.hasOneUse()) {
    return nullptr;
  }
  return *value.getUsers().begin();
}

}  // namespace fusion_region
}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_ANALYSIS_FUSIONREGION_FUSIONREGIONTAG_H
