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

#include "llvm/ADT/SetVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"

#include "mfusion/Analysis/Cluster/Graph.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Support/FusedOpUtils.h"
#include "mfusion/Support/Logging.h"

#define DEBUG_TYPE "graph-kernel-cluster"

namespace mlir {
namespace mfuse {
namespace {
bool materializeFusedOpFromBuildInfo(func::FuncOp funcOp, const ClusterBuildInfo &buildInfo,
                                     llvm::StringRef fusionType) {
  // Materialize one already-validated cluster slice. Partition selection and
  // legality analysis happen before we get here.
  OpBuilder builder(funcOp.getContext());
  builder.setInsertionPointAfter(buildInfo.insertPoint);

  llvm::SmallVector<Type> resultTypes(buildInfo.externalOutputs.size());
  std::transform(buildInfo.externalOutputs.begin(), buildInfo.externalOutputs.end(), resultTypes.begin(),
                 [](Value output) { return output.getType(); });

  auto fusedOp =
    builder.create<FusedOp>(buildInfo.clusterOps.front()->getLoc(), resultTypes, buildInfo.externalInputs.getArrayRef(),
                            builder.getStringAttr(fusionType), /*kernel_name=*/nullptr);

  Block *body = new Block();
  fusedOp.getBody().push_back(body);

  llvm::SmallVector<Location> argLocs;
  argLocs.reserve(buildInfo.externalInputs.size());
  std::transform(buildInfo.externalInputs.begin(), buildInfo.externalInputs.end(), std::back_inserter(argLocs),
                 [](Value input) { return input.getLoc(); });
  body->addArguments(TypeRange(buildInfo.externalInputs.getArrayRef()), argLocs);

  IRMapping mapping;
  for (auto [input, arg] : llvm::zip(buildInfo.externalInputs, body->getArguments())) {
    mapping.map(input, arg);
  }

  builder.setInsertionPointToStart(body);
  for (Operation *op : buildInfo.clusterOps) {
    builder.clone(*op, mapping);
  }

  llvm::SmallVector<Value> yieldValues(buildInfo.externalOutputs.size());
  std::transform(buildInfo.externalOutputs.begin(), buildInfo.externalOutputs.end(), yieldValues.begin(),
                 [&mapping](Value output) { return mapping.lookup(output); });
  builder.create<YieldOp>(fusedOp.getLoc(), yieldValues);

  for (auto [oldOutput, newResult] : llvm::zip(buildInfo.externalOutputs, fusedOp.getResults())) {
    Value output = oldOutput;
    output.replaceAllUsesWith(newResult);
  }

  for (auto it = buildInfo.clusterOps.rbegin(); it != buildInfo.clusterOps.rend(); ++it) {
    if ((*it)->use_empty()) {
      (*it)->erase();
    }
  }

  return true;
}
}  // namespace

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
    MLOG(DEBUG) << "Graph has circle, trying again with conservative strategy";
    graphMerge(&block, false);
    if (graph_->HasCircle()) {
      MLOG(DEBUG) << "Graph still has circle!";
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
    changed |= createFusedOp(funcOp, clusters[i]);
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

bool BaseCluster::createFusedOp(func::FuncOp funcOp, const std::vector<size_t> &nodeIds) {
  if (nodeIds.empty()) {
    return false;
  }

  llvm::SmallVector<Operation *> baseClusterOps(nodeIds.size());
  std::transform(nodeIds.begin(), nodeIds.end(), baseClusterOps.begin(), [this](size_t id) { return ops_[id]; });
  std::sort(baseClusterOps.begin(), baseClusterOps.end(),
            [](Operation *lhs, Operation *rhs) { return lhs->isBeforeInBlock(rhs); });

  ClusterBuildInfo buildInfo;
  if (buildCluster(baseClusterOps, buildInfo)) {
    return materializeFusedOpFromBuildInfo(funcOp, buildInfo, getFusionType());
  }

  auto partitions = partitionClusterOps(baseClusterOps);
  bool created = false;
  for (const llvm::SmallVector<Operation *> &partition : partitions) {
    ClusterBuildInfo partitionBuildInfo;
    // Rebuild against the current IR state. Earlier partitions may already
    // have replaced results and erased ops, so cached slice details are unsafe.
    if (!buildCluster(partition, partitionBuildInfo)) {
      MLOG(ERROR) << "Failed to build cluster for partition, it is illegal to reach here after partitioning.";
      continue;
    }
    // Fuse partitions in program order so later slices can naturally consume
    // results rewritten by earlier slices.
    created |= materializeFusedOpFromBuildInfo(funcOp, partitionBuildInfo, getFusionType());
  }

  return created;
}

}  // namespace mfuse
}  // namespace mlir
