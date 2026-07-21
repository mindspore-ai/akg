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

#include "mfusion/Analysis/DvmFusion/LayerNorm/LayerNormDvmMaterializer.h"

#include <cstdlib>

#include "mfusion/Analysis/DvmFusion/LayerNorm/LayerNormDvmPartitioner.h"
#include "mfusion/Analysis/DvmFusion/LayerNorm/LayerNormDvmUtils.h"
#include "mfusion/Analysis/FusionRegion/FusionRegionTag.h"
#include "mfusion/Dialect/Mfuse/Support/FusedOpUtils.h"
#include "mfusion/Dialect/Mfuse/Transforms/Outlining/FusionAttributes.h"
#include "mfusion/Support/Logging.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace mfuse {
namespace layernorm_dvm {
namespace {

bool lnDvmMaterializeDebugEnabled() {
  if (const char *env = std::getenv("MFUSION_DEBUG_LN_DVM")) {
    return env[0] != '\0' && env[0] != '0';
  }
  return false;
}

bool materializeIsland(PatternRewriter &rewriter, const FusedIsland &island, llvm::StringRef groupId) {
  if (island.ops.size() < kMinClusterSize) {
    if (lnDvmMaterializeDebugEnabled()) {
      llvm::errs() << "[LN-DVM] materialize failed: island too small (" << island.ops.size() << ")\n";
    }
    return false;
  }
  if (!allMemberOpsDvmSupported(island.ops)) {
    MLOG(DEBUG) << "LayerNorm DVM island failed DVM legality pre-check";
    if (lnDvmMaterializeDebugEnabled()) {
      llvm::errs() << "[LN-DVM] materialize failed: DVM legality pre-check\n";
    }
    return false;
  }

  ClusterBuildInfo buildInfo;
  if (!buildCluster(island.ops, buildInfo)) {
    MLOG(DEBUG) << "LayerNorm DVM island failed buildCluster";
    if (lnDvmMaterializeDebugEnabled()) {
      llvm::errs() << "[LN-DVM] materialize failed: buildCluster\n";
    }
    return false;
  }

  FusedOp fusedOp;
  if (!materializeFusedOpFromBuildInfo(rewriter, buildInfo, "dvm", fusedOp)) {
    MLOG(DEBUG) << "LayerNorm DVM island failed materializeFusedOpFromBuildInfo";
    if (lnDvmMaterializeDebugEnabled()) {
      llvm::errs() << "[LN-DVM] materialize failed: materializeFusedOpFromBuildInfo\n";
    }
    return false;
  }

  tagLayerNormDvmFusedOp(fusedOp, groupId);
  return true;
}

bool canMaterializeIsland(const FusedIsland &island) {
  if (island.ops.size() < kMinClusterSize || !allMemberOpsDvmSupported(island.ops)) {
    return false;
  }
  ClusterBuildInfo buildInfo;
  return buildCluster(island.ops, buildInfo);
}

}  // namespace

void tagLayerNormDvmFusedOp(FusedOp fusedOp, llvm::StringRef groupId) {
  fusion_region::retagMember(fusedOp.getOperation(), groupId, fusion_region::kLayerNormFuseKind);
}

bool isLayerNormDvmMaterialized(Operation *op) {
  if (!op) {
    return false;
  }
  if (auto fused = dyn_cast<FusedOp>(op)) {
    if (auto kind = fused->getAttrOfType<StringAttr>(mfusion_attrs::kDvmFuseKind)) {
      return kind.getValue() == fusion_region::kLayerNormFuseKind;
    }
  }
  return isOpInFusedIsland(op);
}

LogicalResult materializeLayerNormDvmPlan(PatternRewriter &rewriter, const LayerNormDvmPlan &plan) {
  if (plan.islands.empty()) {
    return failure();
  }

  // Validate the complete plan before mutating IR. All failures detectable by
  // legality and cluster construction therefore preserve the original graph.
  if (llvm::any_of(plan.islands, [](const FusedIsland &island) { return !canMaterializeIsland(island); })) {
    return failure();
  }

  bool materialized = llvm::all_of(plan.islands, [&](const FusedIsland &island) {
    return materializeIsland(rewriter, island, plan.groupId);
  });
  return success(materialized);
}

LogicalResult fuseLayerNormDvmForward(LayerNormDvmMatch &match, PatternRewriter &rewriter) {
  if (!match.matched || llvm::any_of(match.ops, [](Operation *op) { return isOpInFusedIsland(op); })) {
    return failure();
  }
  std::string groupId = fusion_region::allocateGroupId(fusion_region::kLayerNormFuseKind);
  LayerNormDvmPlan plan = partitionForward(match, groupId);
  if (failed(materializeLayerNormDvmPlan(rewriter, plan))) {
    fusion_region::rollbackLastGroupId();
    return failure();
  }
  return success();
}

LogicalResult fuseLayerNormDvmBackwardGradDiv(const LayerNormDvmBwdMatch &match, PatternRewriter &rewriter) {
  if (!match.matched || llvm::any_of(match.ops, [](Operation *op) { return isOpInFusedIsland(op); })) {
    return failure();
  }
  std::string groupId = fusion_region::allocateGroupId(fusion_region::kLayerNormFuseKind);
  LayerNormDvmPlan plan = partitionBackwardGradDiv(match, groupId);
  if (failed(materializeLayerNormDvmPlan(rewriter, plan))) {
    fusion_region::rollbackLastGroupId();
    return failure();
  }
  return success();
}

LogicalResult fuseLayerNormDvmBackwardVector(const LayerNormDvmBwdMatch &match, PatternRewriter &rewriter) {
  if (!match.matched || llvm::any_of(match.ops, [](Operation *op) { return isOpInFusedIsland(op); })) {
    return failure();
  }
  std::string groupId = fusion_region::allocateGroupId(fusion_region::kLayerNormFuseKind);
  LayerNormDvmPlan plan = partitionBackwardVector(match, groupId);
  if (plan.islands.empty() || plan.islands.front().ops.empty()) {
    fusion_region::rollbackLastGroupId();
    return failure();
  }
  if (failed(materializeLayerNormDvmPlan(rewriter, plan))) {
    fusion_region::rollbackLastGroupId();
    return failure();
  }
  return success();
}

}  // namespace layernorm_dvm
}  // namespace mfuse
}  // namespace mlir
