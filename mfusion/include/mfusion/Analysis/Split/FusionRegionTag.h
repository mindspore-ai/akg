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

#ifndef MFUSION_ANALYSIS_SPLIT_FUSIONREGIONTAG_H
#define MFUSION_ANALYSIS_SPLIT_FUSIONREGIONTAG_H

#include <optional>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/IR/Operation.h"
#include "mlir/Rewrite/PatternApplicator.h"

namespace mlir {
class RewritePatternSet;

namespace mfuse {
namespace fusion_region {

enum class FuseRole { Member, Affinity };

/// Fuse kind string for LayerNorm; legacy unit attrs are written only for this kind.
inline constexpr llvm::StringLiteral kLayerNormFuseKind = "layer_norm";

/// Pluggable matcher that tags decomposed subgraphs for DVM split (FuseTagBarrierByGroupId).
class FusionRegionMatcher {
 public:
  virtual ~FusionRegionMatcher() = default;

  virtual llvm::StringRef kind() const = 0;
  virtual void populatePatterns(RewritePatternSet &patterns) = 0;
};

FuseRole getFuseRole(Operation *op);
bool hasRegionMember(Operation *op);
bool hasRegionAffinity(Operation *op);
bool isTagged(Operation *op);

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

void registerAllFusionRegionMatchers(RewritePatternSet &patterns);

}  // namespace fusion_region
}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_ANALYSIS_SPLIT_FUSIONREGIONTAG_H
