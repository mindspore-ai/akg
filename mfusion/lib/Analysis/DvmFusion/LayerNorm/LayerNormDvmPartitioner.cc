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

#include "mfusion/Analysis/DvmFusion/LayerNorm/LayerNormDvmPartitioner.h"

#include <algorithm>

#include "mfusion/Analysis/DvmFusion/LayerNorm/LayerNormDvmMatcher.h"
#include "mfusion/Analysis/DvmFusion/LayerNorm/LayerNormDvmUtils.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace mfuse {
namespace layernorm_dvm {
namespace {

void sortOpsInBlockOrder(llvm::SmallVectorImpl<Operation *> &ops) {
  std::sort(ops.begin(), ops.end(), [](Operation *lhs, Operation *rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
}

bool isForwardMeanPrologueOp(Operation *op, LayerNormDvmMatch &match) {
  if (!op) {
    return false;
  }
  if (match.reduceMean && op == match.reduceMean.getOperation()) {
    return true;
  }
  if (match.reduceSum && op == match.reduceSum.getOperation()) {
    return true;
  }
  if (match.meanDiv && op == match.meanDiv.getOperation()) {
    return true;
  }
  return false;
}

bool meanResultHasExternalUser(Operation *meanOp, LayerNormDvmMatch &match) {
  if (!meanOp || meanOp->getNumResults() == 0) {
    return false;
  }
  Value result = meanOp->getResult(0);
  for (auto &use : result.getUses()) {
    Operation *user = use.getOwner();
    if (user && !llvm::is_contained(match.ops, user)) {
      return true;
    }
  }
  return false;
}

bool shouldPeelForwardMeanOp(Operation *op, LayerNormDvmMatch &match) {
  if (!isForwardMeanPrologueOp(op, match)) {
    return false;
  }
  if (match.reduceMean && meanResultHasExternalUser(match.reduceMean.getOperation(), match)) {
    return true;
  }
  if (match.reduceSum && meanResultHasExternalUser(match.reduceSum.getOperation(), match)) {
    return true;
  }
  return match.meanDiv && meanResultHasExternalUser(match.meanDiv.getOperation(), match);
}

void peelForwardMeanPrologueOps(llvm::SmallVectorImpl<Operation *> &islandOps, LayerNormDvmMatch &match) {
  llvm::erase_if(islandOps, [&](Operation *op) { return shouldPeelForwardMeanOp(op, match); });
}

LayerNormDvmPlan makeSingleIslandPlan(llvm::ArrayRef<Operation *> ops, llvm::StringRef groupId) {
  LayerNormDvmPlan plan;
  plan.groupId = groupId.str();
  FusedIsland island;
  island.ops.assign(ops.begin(), ops.end());
  sortOpsInBlockOrder(island.ops);
  if (!island.ops.empty()) {
    plan.islands.push_back(std::move(island));
  }
  return plan;
}

}  // namespace

LayerNormDvmPlan partitionForward(LayerNormDvmMatch &match, llvm::StringRef groupId) {
  LayerNormDvmPlan plan = makeSingleIslandPlan(match.ops, groupId);
  if (!plan.islands.empty()) {
    peelForwardMeanPrologueOps(plan.islands.front().ops, match);
  }
  return plan;
}

LayerNormDvmPlan partitionBackwardGradDiv(const LayerNormDvmBwdMatch &match, llvm::StringRef groupId) {
  return makeSingleIslandPlan(match.ops, groupId);
}

LayerNormDvmPlan partitionBackwardVector(const LayerNormDvmBwdMatch &match, llvm::StringRef groupId) {
  LayerNormDvmPlan plan;
  plan.groupId = groupId.str();

  FusedIsland island;
  collectExclusiveBackwardOps(match.ops, island.ops);
  sortOpsInBlockOrder(island.ops);
  if (!island.ops.empty()) {
    plan.islands.push_back(std::move(island));
  }
  return plan;
}

}  // namespace layernorm_dvm
}  // namespace mfuse
}  // namespace mlir
