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

#ifndef AKG_DIALECT_AFFINE_ANALYSIS_LOOPFUSIONANALYZER_H_
#define AKG_DIALECT_AFFINE_ANALYSIS_LOOPFUSIONANALYZER_H_

#include <cstddef>
#include <functional>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "akg/Dialect/Affine/Analysis/LoopFusionBuilder.h"
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
// Ordering: topological > soft defer.
// Soft-deferred edges are held until their prior emits; if Kahn drains with deferred edges
// remaining, one is force-emitted to guarantee every plan edge appears exactly once.
struct TopoScheduler {
  TopoScheduler(std::vector<FusionPlan> &edges, unsigned numNodes,
                const std::vector<std::pair<unsigned, unsigned>> &softDeferConstraints)
      : edges(edges), numNodes(numNodes), softDeferConstraints(softDeferConstraints) {}
  std::vector<FusionPlan> run();

 private:
  void buildSchedulingState();
  bool priorsSatisfied(size_t edgeIdx) const;
  void commitEdge(size_t edgeIdx);
  void tryEmit(size_t edgeIdx);
  void releaseDeferred();
  void processNode(unsigned node);

  std::vector<FusionPlan> &edges;
  unsigned numNodes;
  const std::vector<std::pair<unsigned, unsigned>> &softDeferConstraints;
  std::unordered_map<unsigned, std::vector<size_t>> adjacency;
  std::vector<unsigned> inDegree;
  std::unordered_map<size_t, std::unordered_set<size_t>> pendingPriors;
  std::priority_queue<unsigned> readyQueue;
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
  void print(llvm::raw_ostream &os, bool detail = true) const;
  void dump(bool detail = true) const { print(llvm::errs(), detail); }

  MemRefDependenceGraphForFusion &depGraph;
  func::FuncOp funcOp{nullptr};
  std::unordered_map<unsigned, GroupPtr> groups;
  std::vector<FusionPlan> fusionPlans;

 private:
  // Init & Topo Sort
  void initGroups();
  void topoSortInit();
  std::vector<FusionPlan> deduplicateAndClassifyEdges();
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
  void stitchChainsByGroupId(unsigned srcGroupId, unsigned dstGroupId, std::pair<unsigned, unsigned> origCacheKey,
                             bool *dstSideWon = nullptr);

  // Sets fusionType, depInfo, and loopTransform.
  void setFusionPlanOptions(FusionPlan &plan);
  bool hasEdgeInFusionPlans(unsigned depGroupId, unsigned fromGroupId);

  bool findBackwardIntersection(const GroupPtr oldGroup, const GroupPtr newGroup, bool *isAncestry = nullptr);

  unsigned outgoingTarget(unsigned id);
  std::pair<unsigned, unsigned> findBridgePoint(unsigned sourceId, unsigned targetId);
  void propagateDeletedDep(unsigned existingTo, unsigned targetId, DependenceInfo &deletedDep);
  bool isConflictingPair(const std::pair<unsigned, unsigned> &a, const std::pair<unsigned, unsigned> &b);
  void bridgeChainToTarget(std::pair<unsigned, unsigned> hEdge);
  size_t findConflictingCandidate(const std::vector<std::pair<unsigned, unsigned>> &candidates,
                                  const std::vector<bool> &rewritten, size_t startIdx);
  void resolveConflictingDefers(std::vector<std::pair<unsigned, unsigned>> &candidates);

  // Precomputation
  void precomputeDirectPredecessors();
  void dumpDirectPredecessors(unsigned nodeId, const std::vector<unsigned> &allPredecessorIds,
                              const std::unordered_set<size_t> &skipIdx,
                              const std::vector<DirectPredecessor> &directPreds);

  // Prefers store-load edges; falls back to load-load only when none exist.
  // Output vectors are sorted by groupId descending.
  void collectFusionSourceGroups(const GroupPtr &targetGroup, std::vector<unsigned> &warGroupIds,
                                 std::vector<unsigned> &nonWarGroupIds);
  void dumpCollectFusionSourceInfo(const GroupPtr &targetGroup, const char *dependenceType,
                                   const std::vector<unsigned> &sourceGroups);

  void processWarEdges(std::unordered_map<unsigned, std::vector<unsigned>> &nonWarMap);
  void processNonWarEdges(const std::unordered_map<unsigned, std::vector<unsigned>> &nonWarMap,
                          std::vector<std::pair<unsigned, unsigned>> &deferCandidates);
  // Returns 2 if the cached (source,target) deps contain any Subview, 1 if any Input
  // (and no Subview), 0 if all Normal; missing/empty is treated as Input (1).
  int pairKindRank(unsigned source, unsigned target) const;
  // True when the (source,target) deps are treated as Input (Input present, no Subview).
  bool isInputOnlyPair(unsigned source, unsigned target) const;
  // Propagates (source,target) depInfo entries to other cache entries with the same
  // target but a different source (ancestry case).
  void propagateAncestryDeps(unsigned sourceGroupId, unsigned targetGroupId);

  // Post-processing: merge independent fusion chains whose accesses are disjoint.
  void mergeIndependentChains();

  // Member Variables
  std::unordered_set<unsigned> finished;
  std::vector<unsigned> topoSortNodeIds;
  // Direct predecessor cache: nodeId -> {predecessor, memref, depth, depType}
  std::unordered_map<unsigned, std::vector<DirectPredecessor>> directPredecessorsCache;
  // Group dependency cache: (srcGroupId, dstGroupId) -> all DependenceInfo entries.
  // Multiple dependencies between the same group pair are retained.
  std::unordered_map<std::pair<unsigned, unsigned>, std::vector<DependenceInfo>, PairHash> groupDependenciesCache;
  // Soft defer constraints from Phase 2 skipped RAR edges.
  // Each pair (deferredFromGroupId, mustEmitFirstFromGroupId): edges from deferredFromGroupId
  // are held until all edges from mustEmitFirstFromGroupId have emitted.
  std::vector<std::pair<unsigned, unsigned>> softDeferConstraints;
};

}  // namespace akg
}  // namespace mlir

#endif  // AKG_DIALECT_AFFINE_ANALYSIS_LOOPFUSIONANALYZER_H_
