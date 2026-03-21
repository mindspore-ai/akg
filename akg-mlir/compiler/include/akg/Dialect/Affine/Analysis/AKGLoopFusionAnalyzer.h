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
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "akg/Dialect/Affine/Analysis/AKGLoopFusionBuilder.h"
#include "akg/Dialect/Affine/Analysis/LoopFusionUtils.h"

namespace mlir {
namespace akg {

struct TupleHash {
  std::size_t operator()(const std::tuple<unsigned, unsigned, unsigned> &t) const {
    std::size_t h1 = std::hash<unsigned>{}(std::get<0>(t));
    std::size_t h2 = std::hash<unsigned>{}(std::get<1>(t));
    std::size_t h3 = std::hash<unsigned>{}(std::get<2>(t));
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};

struct PairHash {
  std::size_t operator()(const std::pair<unsigned, unsigned> &p) const {
    std::size_t h1 = std::hash<unsigned>{}(p.first);
    std::size_t h2 = std::hash<unsigned>{}(p.second);
    return h1 ^ (h2 << 1);
  }
};

// Intersection Point Handling
struct BackwardIntersectionResult {
  unsigned closestIntersectionId;
  unsigned minOldPathLen;
  unsigned minNewPathLen;
  std::unordered_map<unsigned, unsigned> oldReachable;
  std::unordered_map<unsigned, unsigned> newReachable;
};

// Structure to store direct predecessor node ID and corresponding memref.
// isRAR marks edges where both source and target ops are AffineLoadOp (read-read dependency).
// These edges are lower priority than store-load (producer-consumer) edges during fusion.
struct DirectPredecessor {
  DirectPredecessor(unsigned id, Value ref, unsigned depth = 0, bool readAfterRead = false)
      : nodeId(id), memref(ref), loopDepth(depth), isRAR(readAfterRead) {}
  unsigned nodeId;
  Value memref;
  unsigned loopDepth;
  bool isRAR;
};

struct FusionAnalyzer {
 public:
  FusionAnalyzer(MemRefDependenceGraphForFusion &depGraph, func::FuncOp funcOp) : depGraph(depGraph), funcOp(funcOp) {}

  void plan();
  void applyAndFuse(const GroupPtr targetGroup, const GroupPtr sourceGroup);
  bool checkAndFixMultiOut(FusionPlan &fusePlan);

  // Debug and print
  void print(llvm::raw_ostream &os) const;
  void dump() const { print(llvm::errs()); }

  MemRefDependenceGraphForFusion &depGraph;
  func::FuncOp funcOp{nullptr};
  std::unordered_map<unsigned, GroupPtr> groups;
  std::vector<FusionPlan> fusionPlans;

 private:
  // Initialization and Topological Sorting
  void initGroups();
  void topoSortInit();
  std::vector<FusionPlan> topoSortFusionPlans(unsigned numNodes);
  GroupPtr getFusionTargetGroup();
  bool finishPlan();

  // Fusion Plan Management
  std::vector<FusionPlan>::iterator findFusionPlanByGroup(unsigned fromGroupId, unsigned toGroupId);
  std::vector<FusionPlan>::iterator findFusionPlanByBand(unsigned fromBandId, unsigned toBandId);
  bool addFusionPlan(const FusionPlan &plan);
  bool updateFusionPlanByGroup(unsigned fromGroupId, unsigned oldToGroupId, unsigned newToGroupId,
                               unsigned newToNodeId);
  bool updateFusionPlanByBand(unsigned oldFromBandId, unsigned oldToBandId, unsigned newFromBandId,
                              unsigned newToBandId);
  bool removeFusionPlanByGroup(unsigned fromGroupId, unsigned toGroupId);
  bool removeFusionPlanByBand(unsigned fromBandId, unsigned toBandId);

  // Path and Reachability Analysis
  std::unordered_map<unsigned, unsigned> findReachableGroups(unsigned startGroupId);
  std::vector<unsigned> findLastNodesInPath(unsigned srcGroupId);
  bool connectLastNodesToTarget(unsigned srcGroupId, unsigned dstGroupId);

  // Fusion Type and Order Determination (sets fusionType, depInfo, and loopTransform)
  void setFusionPlanOptions(FusionPlan &plan);
  bool hasEdgeInFusionPlans(unsigned depGroupId, unsigned fromGroupId);
  std::pair<GroupPtr, GroupPtr> determineFusionOrder(const GroupPtr oldGroup, const GroupPtr newGroup);

  bool findBackwardIntersection(const GroupPtr oldGroup, const GroupPtr newGroup, BackwardIntersectionResult &result);
  GroupPtr handleBackwardIntersectionPoints(const GroupPtr oldGroup, const GroupPtr newGroup,
                                            const BackwardIntersectionResult &intersection);
  void redirectFusionPlanToTarget(unsigned intersectionId, const std::unordered_map<unsigned, unsigned> &reachable,
                                  unsigned pathLen, GroupPtr targetGroup);
  void setupDirectFusionPlan(FusionPlan &fusePlan, FusionPlan &oldPlan, const GroupPtr srcGroup,
                             const GroupPtr dstGroup);

  // Precomputation
  void precomputeDirectPredecessors();

  // Collects source groups for fusion with the target group.
  // Prefers store-load (producer-consumer) edges; falls back to load-load edges
  // only when no store-load edges exist for the target group.
  void collectFusionSourceGroups(const GroupPtr &targetGroup, std::unordered_set<unsigned> &storeLoadGroups,
                                 std::unordered_set<unsigned> &readAfterReadGroups);

  // Gets dependent operations and memrefs between source and target groups.
  // Prefers store-load dependencies; falls back to load-load when none exist.
  DependenceInfo getGroupDependencies(const GroupPtr targetGroup, const GroupPtr sourceGroup);

  // Member Variables
  std::unordered_set<unsigned> finished;
  std::vector<unsigned> topoSortNodeIds;
  std::unordered_map<std::tuple<unsigned, unsigned, unsigned>, unsigned, TupleHash> intersectionCache;
  // Cache for direct predecessors of each node with corresponding memrefs (computed once, used many times)
  std::unordered_map<unsigned, std::vector<DirectPredecessor>> directPredecessorsCache;
  // Cache for group dependencies (groupId pair -> DependenceInfo)
  std::unordered_map<std::pair<unsigned, unsigned>, DependenceInfo, PairHash> groupDependenciesCache;
};

}  // namespace akg
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_AKGLOOPFUSIONANALYZER_H_
