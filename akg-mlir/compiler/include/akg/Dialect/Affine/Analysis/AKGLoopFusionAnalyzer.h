/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_AKGLOOPFUSIONANALYZER_H_
#define COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_AKGLOOPFUSIONANALYZER_H_

#include <cstddef>
#include <functional>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "akg/Dialect/Affine/Analysis/AKGLoopFusionBuilder.h"
#include "akg/Dialect/Affine/Analysis/LoopFusionUtils.h"

namespace mlir {
namespace akg {

struct PairHash {
  std::size_t operator()(const std::pair<unsigned, unsigned> &p) const {
    std::size_t h1 = std::hash<unsigned>{}(p.first);
    std::size_t h2 = std::hash<unsigned>{}(p.second);
    return h1 ^ (h2 << 1);
  }
};

// Priority-Kahn scheduler for fusion plan edges.
// Ordering: topological > normal-before-subview > per-node non-subview-before-subview > soft defer.
// Soft-deferred edges are held until their prior emits; if Kahn drains with deferred edges
// remaining, one is force-emitted to guarantee every plan edge appears exactly once.
struct TopoScheduler {
  TopoScheduler(std::vector<FusionPlan> &edges, unsigned numNodes,
                const std::vector<std::pair<unsigned, unsigned>> &softDeferConstraints)
      : edges(edges), numNodes(numNodes), softDeferConstraints(softDeferConstraints) {}
  std::vector<FusionPlan> run();

 private:
  void buildSchedulingState();
  bool hasReadyNodes() const { return !normalQueue.empty() || !subviewQueue.empty(); }
  bool priorsSatisfied(size_t edgeIdx) const;
  void commitEdge(size_t edgeIdx);
  void tryEmit(size_t edgeIdx);
  void releaseDeferred();
  void processNode(unsigned node);

  std::vector<FusionPlan> &edges;
  unsigned numNodes;
  const std::vector<std::pair<unsigned, unsigned>> &softDeferConstraints;
  std::unordered_map<unsigned, std::vector<size_t>> adjacency;
  std::unordered_set<unsigned> normalNodes;
  std::vector<unsigned> inDegree;
  std::unordered_map<size_t, std::unordered_set<size_t>> pendingPriors;
  std::priority_queue<unsigned> normalQueue;
  std::priority_queue<unsigned> subviewQueue;
  std::vector<FusionPlan> result;
  std::unordered_set<size_t> emittedEdges;
  std::vector<size_t> pendingDeferred;
};

// Direct predecessor node ID and corresponding memref.
struct DirectPredecessor {
  DirectPredecessor(unsigned id, Value ref, unsigned depth = 0, DepType type = DepType::OTHER)
      : nodeId(id), memref(ref), loopDepth(depth), depType(type) {}
  unsigned nodeId;
  Value memref;
  unsigned loopDepth;
  DepType depType;
};

struct FusionAnalyzer {
 public:
  FusionAnalyzer(MemRefDependenceGraphForFusion &depGraph, func::FuncOp funcOp) : depGraph(depGraph), funcOp(funcOp) {}

  void plan();
  void applyAndFuse(const GroupPtr targetGroup, const GroupPtr sourceGroup);
  bool checkAndFixMultiOut(FusionPlan &fusePlan);

  // Debug
  void print(llvm::raw_ostream &os) const;
  void dump() const { print(llvm::errs()); }

  MemRefDependenceGraphForFusion &depGraph;
  func::FuncOp funcOp{nullptr};
  std::unordered_map<unsigned, GroupPtr> groups;
  std::vector<FusionPlan> fusionPlans;

 private:
  // Init & Topo Sort
  void initGroups();
  void topoSortInit();
  std::vector<FusionPlan> deduplicateAndClassifyEdges();
  bool checkSubviewFusion(unsigned predNodeId, unsigned targetNodeId);
  GroupPtr getFusionTargetGroup();
  bool finishPlan();

  // Plan Management
  std::vector<FusionPlan>::iterator findFusionPlanByGroup(unsigned fromGroupId, unsigned toGroupId);
  bool addFusionPlan(const FusionPlan &plan);
  bool updateFusionPlanByGroup(unsigned fromGroupId, unsigned oldToGroupId, unsigned newToGroupId,
                               unsigned newToBandId);

  // Reachability
  std::unordered_map<unsigned, unsigned> findReachableGroups(unsigned startGroupId);
  std::vector<unsigned> findLastNodesInPath(unsigned srcGroupId);
  bool connectLastNodesToTarget(unsigned srcGroupId, unsigned dstGroupId, bool *dstSideWon = nullptr);

  // Sets fusionType, depInfo, and loopTransform.
  void setFusionPlanOptions(FusionPlan &plan);
  bool hasEdgeInFusionPlans(unsigned depGroupId, unsigned fromGroupId);
  std::pair<GroupPtr, GroupPtr> determineFusionOrder(const GroupPtr oldGroup, const GroupPtr newGroup);

  bool findBackwardIntersection(const GroupPtr oldGroup, const GroupPtr newGroup, bool *isAncestry = nullptr);
  void setupDirectFusionPlan(FusionPlan &fusePlan, FusionPlan &oldPlan, const GroupPtr srcGroup,
                             const GroupPtr dstGroup);

  unsigned outgoingTarget(unsigned id);
  std::pair<unsigned, unsigned> findBridgePoint(unsigned sourceId, unsigned targetId);
  void propagateDeletedDep(unsigned existingTo, unsigned targetId, DependenceInfo &deletedDep);
  bool isConflictingPair(const std::pair<unsigned, unsigned> &a, const std::pair<unsigned, unsigned> &b);
  void bridgeChainToTarget(std::pair<unsigned, unsigned> hEdge);
  void resolveConflictingDefers(std::vector<std::pair<unsigned, unsigned>> &candidates);

  // Precomputation
  void precomputeDirectPredecessors();
  void dumpDirectPredecessors(unsigned nodeId, const std::vector<unsigned> &allPredecessorIds,
                              const std::unordered_set<size_t> &skipIdx,
                              const std::vector<DirectPredecessor> &directPreds);

  // Prefers store-load edges; falls back to load-load only when none exist.
  // Output vectors are sorted by groupId descending.
  void collectFusionSourceGroups(const GroupPtr &targetGroup, std::vector<unsigned> &warGroupIds,
                                 std::vector<unsigned> &rarGroupIds);
  void dumpCollectFusionSourceInfo(const GroupPtr &targetGroup, const char *dependenceType,
                                   const std::vector<unsigned> &sourceGroups);

  // Member Variables
  std::unordered_set<unsigned> finished;
  std::vector<unsigned> topoSortNodeIds;
  // Direct predecessor cache: nodeId -> {predecessor, memref, depth, depType}
  std::unordered_map<unsigned, std::vector<DirectPredecessor>> directPredecessorsCache;
  // Group dependency cache: (srcGroupId, dstGroupId) -> the most-restrictive DependenceInfo
  // (smallest loopDepth wins on update). Updates merge by comparing loopDepth.
  std::unordered_map<std::pair<unsigned, unsigned>, DependenceInfo, PairHash> groupDependenciesCache;
  // Soft defer constraints from Phase 2 skipped RAR edges.
  // Each pair (deferredFromGroupId, mustEmitFirstFromGroupId): edges from deferredFromGroupId
  // are held until all edges from mustEmitFirstFromGroupId have emitted.
  std::vector<std::pair<unsigned, unsigned>> softDeferConstraints;
};

}  // namespace akg
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_AKGLOOPFUSIONANALYZER_H_
