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

#include "mfusion/Dialect/Mfuse/Transforms/Cluster/BaseCluster.h"

#include <algorithm>
#include <utility>
#include <unordered_set>

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"

#include "mfusion/Analysis/Cluster/Graph.h"
#include "mfusion/Dialect/Mfuse/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Transforms/Cluster/Utils.h"
#include "mfusion/Dialect/Mfuse/Utils/FusedOpUtils.h"

#define DEBUG_TYPE "graph-kernel-cluster"

namespace mlir {
namespace mfuse {
//===----------------------------------------------------------------------===//
// BaseCluster Implementation
//===----------------------------------------------------------------------===//

void BaseCluster::init() { opList_ = getClusterableOpList(); }

bool BaseCluster::run(func::FuncOp funcOp) {
  init();
  bool changed = process(funcOp);
  clean();
  return changed;
}

bool BaseCluster::process(func::FuncOp funcOp) {
  Block &block = funcOp.getBody().front();
  graphMerge(&block, true);

  if (graph_->HasCircle()) {
    LLVM_DEBUG(llvm::dbgs() << "Graph has circle, trying again with conservative strategy\n");
    graphMerge(&block, false);
    if (graph_->HasCircle()) {
      LLVM_DEBUG(llvm::dbgs() << "Graph still has circle!\n");
    }
  }

  // Rebuild the IR with fused operations
  bool changed = false;
  auto clusters = graph_->CollectClusters();

  for (size_t i = 0; i < clusters.size(); ++i) {
    size_t nodeCount = clusters[i].size();
    if (nodeCount == 0 || nodeCount == 1) {
      continue;
    }
    createFusedOp(funcOp, clusters[i]);
    changed = true;
  }

  return changed;
}

void BaseCluster::graphMerge(Block *block, bool aggressiveCut) {
  llvm::DenseMap<Operation *, size_t> opIdxMap;
  graph_ = Graph::Build(block, &ops_, &opIdxMap, aggressiveCut);

  // Process nodes in reverse order (from outputs to inputs)
  for (int i = static_cast<int>(ops_.size()) - 1; i >= 0; --i) {
    // Skip if already part of a multi-node cluster
    if (graph_->GetSize(static_cast<size_t>(i)) > 1) {
      continue;
    }

    auto candidates = findCandidates(static_cast<size_t>(i));
    CircleChecker circleChecker(graph_.get());
    circleChecker.RemoveCircle(&candidates);

    if (candidates.size() <= 1) {
      continue;
    }

    // Merge candidates into one cluster
    graph_->Merge(candidates);
  }
}

std::vector<size_t> BaseCluster::findCandidates(size_t baseNodeId) {
  std::vector<size_t> candidates;
  Operation *baseOp = ops_[baseNodeId];
  Block *block = baseOp->getBlock();

  auto include = [this, &candidates, block](size_t clusterId) {
    Operation *op = this->ops_[clusterId];
    // Must be in the same block
    if (op->getBlock() != block) {
      return VisitResult::kExclude;
    }
    // Must be clusterable
    if (!isClusterableOp(op)) {
      return VisitResult::kExclude;
    }
    candidates.push_back(clusterId);
    // Do not search from already clustered node
    if (this->graph_->GetSize(clusterId) > 1) {
      return VisitResult::kNoFollow;
    }
    return VisitResult::kFollow;
  };

  graph_->Dfs(baseNodeId, include);
  std::reverse(candidates.begin(), candidates.end());
  return candidates;
}

void BaseCluster::createFusedOp(func::FuncOp funcOp, const std::vector<size_t> &nodeIds) {
  if (nodeIds.empty()) {
    return;
  }

  // Collect operations in cluster
  llvm::SmallVector<Operation *> clusterOps(nodeIds.size());
  std::transform(nodeIds.begin(), nodeIds.end(), clusterOps.begin(), [this](size_t id) { return ops_[id]; });

  // Collect constant operations that should be included in the cluster
  llvm::DenseSet<Operation *> constantsToCluster = collectConstantsToCluster(clusterOps);

  // Add constant operations to clusterOps
  std::copy(constantsToCluster.begin(), constantsToCluster.end(), std::back_inserter(clusterOps));

  // Sort operations after adding constants
  std::sort(clusterOps.begin(), clusterOps.end(), [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });

  llvm::DenseSet<Operation *> clusterOpSet(clusterOps.begin(), clusterOps.end());
  llvm::SetVector<Value> externalInputs = findExternalInputs(clusterOps, clusterOpSet);
  llvm::SetVector<Value> externalOutputs = findExternalOutputs(clusterOps, constantsToCluster, clusterOpSet);

  if (externalOutputs.empty()) {
    // No external outputs means this cluster's results are not used
    // This shouldn't normally happen, but handle it gracefully
    return;
  }

  // Create the fused operation.
  // The insertion point must satisfy two SSA dominance constraints:
  //   1. After all external input definitions (so they dominate the fused op's uses).
  //   2. Before all non-cluster users of external outputs (so fused results dominate users).
  // We start from the first cluster op and move forward to cover any external input
  // that is defined between cluster ops (e.g., a non-clusterable op whose result is
  // consumed by a later cluster op). This keeps the fused op as early as possible,
  // which avoids breaking dominance for external output users located between cluster ops.
  OpBuilder builder(funcOp.getContext());
  Operation *insertPoint = clusterOps.front();
  for (Value input : externalInputs) {
    Operation *defOp = input.getDefiningOp();
    if (defOp && defOp->getBlock() == insertPoint->getBlock() && insertPoint->isBeforeInBlock(defOp)) {
      insertPoint = defOp;
    }
  }
  builder.setInsertionPointAfter(insertPoint);

  // Collect result types
  llvm::SmallVector<Type> resultTypes(externalOutputs.size());
  std::transform(externalOutputs.begin(), externalOutputs.end(), resultTypes.begin(),
                 [](Value output) { return output.getType(); });

  // Create FusedOp
  auto fusedOp = builder.create<FusedOp>(clusterOps.front()->getLoc(), resultTypes, externalInputs.getArrayRef(),
                                         builder.getStringAttr(getFusionType()), /*kernel_name=*/nullptr);

  // Create the body block with arguments for each external input
  Block *body = new Block();
  fusedOp.getBody().push_back(body);

  llvm::SmallVector<Location> argLocs;
  argLocs.reserve(externalInputs.size());
  std::transform(externalInputs.begin(), externalInputs.end(), std::back_inserter(argLocs),
                 [](Value input) { return input.getLoc(); });
  body->addArguments(TypeRange(externalInputs.getArrayRef()), argLocs);

  // Create mapping from external inputs to block arguments
  IRMapping mapping;
  for (auto [input, arg] : llvm::zip(externalInputs, body->getArguments())) {
    mapping.map(input, arg);
  }

  // Clone operations into the fused body
  builder.setInsertionPointToStart(body);
  for (Operation *op : clusterOps) {
    builder.clone(*op, mapping);
  }

  // Create yield operation with outputs
  llvm::SmallVector<Value> yieldValues(externalOutputs.size());
  std::transform(externalOutputs.begin(), externalOutputs.end(), yieldValues.begin(),
                 [&mapping](Value output) { return mapping.lookup(output); });
  builder.create<YieldOp>(fusedOp.getLoc(), yieldValues);

  // Replace uses of cluster outputs with fused op results
  for (auto [oldOutput, newResult] : llvm::zip(externalOutputs, fusedOp.getResults())) {
    Value output = oldOutput;  // Create non-const copy for replaceAllUsesWith
    output.replaceAllUsesWith(newResult);
  }

  // Erase original operations in reverse order
  for (auto it = clusterOps.rbegin(); it != clusterOps.rend(); ++it) {
    if ((*it)->use_empty()) {
      (*it)->erase();
    }
  }
}

}  // namespace mfuse
}  // namespace mlir
