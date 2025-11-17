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

#include "akg/Dialect/Affine/Analysis/DependenceAnalysis.h"
#include "akg/Dialect/Affine/Analysis/AKGLoopFusionBuilder.h"
#include "akg/Utils/AnalysisCommon.hpp"

namespace mlir {
namespace akg {

// Hash function for tuple<unsigned, unsigned, unsigned>
struct TupleHash {
  std::size_t operator()(const std::tuple<unsigned, unsigned, unsigned>& t) const {
    std::size_t h1 = std::hash<unsigned>{}(std::get<0>(t));
    std::size_t h2 = std::hash<unsigned>{}(std::get<1>(t));
    std::size_t h3 = std::hash<unsigned>{}(std::get<2>(t));
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};

struct FuseEdge {
  FuseEdge(unsigned from, unsigned to) : from(from), to(to) {}
  unsigned from;
  unsigned to;
};

struct FusionPlan {
  FuseEdge fusedGroup{FuseEdge(0, 0)};
  FuseEdge fusedBand{FuseEdge(0, 0)};
  std::string fusionType{"H"};
};

// FusionAnalyzer analyzes and plans loop fusion operations.
// This class provides methods to analyze loop dependencies, create fusion plans,
// and orchestrate the fusion process for loop optimization.
struct FusionAnalyzer {
 public:
  // Constructs an analyzer with the given dependency graph and function operation.
  FusionAnalyzer(MemRefDependenceGraphForFusion depGraph, func::FuncOp funcOp) : depGraph(depGraph), funcOp(funcOp) {}

  // Infers loop transforms for fusion between target and source groups.
  // Returns a vector of loop transforms to be applied.
  std::vector<LoopTransform> inferLoopTransforms(const GroupPtr targetGroup, const GroupPtr sourceGroup);

  // Checks and fixes multi-output fusion plans.
  // Converts multi-output producer-consumer plans to sibling plans.
  // Returns true if fusePlan should be inserted into fusionPlans, false otherwise.
  bool checkAndFixMultiOut(FusionPlan &fusePlan);

  // Applies loop transforms and fuses source group into target group.
  // Records the fusion plan and updates group relationships.
  void applyAndFuse(std::vector<LoopTransform> loopTransforms, const GroupPtr targetGroup, const GroupPtr sourceGroup);

  // Main planning function that orchestrates the fusion process.
  // Analyzes groups, performs topological sorting, and creates fusion plans.
  void plan();

  // Reorders fusion plans based on topological sort.
  // Removes completed fusion plans and sorts the remaining ones.
  // If srcId and dstId are provided, removes plans with srcId->dstId and updates
  // all plans that reference dstId by replacing dstId with srcId.
  // Example: if fusion plan is 1->2, then plan 2->3 becomes 1->3.
  void reorderPlans(unsigned numNodes, unsigned srcId = 0, unsigned dstId = 0);

  MemRefDependenceGraphForFusion depGraph{nullptr};
  func::FuncOp funcOp{nullptr};
  std::unordered_map<unsigned, GroupPtr> groups;
  std::vector<FusionPlan> fusionPlans;

 private:
  // Gets directly connected predecessor nodes for a given node.
  // Filters out nodes that have indirect connections through other nodes.
  void getDirectlyPredecessors(unsigned id, std::vector<unsigned> &predecessorIds);

  // Checks if the fusion plan is complete.
  // Returns true if all nodes have been processed.
  bool finishPlan();

  // Initializes groups from the dependency graph.
  void initGroups();

  // Performs topological sorting of groups based on dependencies.
  void topoSort();

  // Gets the next target group for fusion.
  // Returns the group that should be the target of the next fusion operation.
  GroupPtr getFusionTargetGroup();

  // Determines the fusion order between two groups.
  // Returns a pair of (sourceGroup, destinationGroup) indicating the fusion order.
  std::pair<GroupPtr, GroupPtr> determineFusionOrder(
      const GroupPtr oldGroup, const GroupPtr newGroup);

  // Handles backward intersection points between two groups in the fusion plan.
  // Finds the closest backward intersection point and redirects fusion paths accordingly.
  // Returns the target group if backward intersection points were found and handled, nullptr otherwise.
  GroupPtr handleBackwardIntersectionPoints(const GroupPtr oldGroup, const GroupPtr newGroup);

  // Connects all last nodes in paths from srcGroupId to dstGroupId.
  bool connectLastNodesToTarget(unsigned srcGroupId, unsigned dstGroupId);

  // Redirects shorter path to longer path's starting group.
  // Modifies fusion plans to redirect paths ending at intersectionId to targetGroup.
  void redirectFusionPlanToTarget(unsigned intersectionId,
                                  const std::unordered_map<unsigned, unsigned> &reachable,
                                  unsigned pathLen, GroupPtr targetGroup);

  // Sets up direct fusion plan from srcGroup to dstGroup and links all source groups to destination group.
  // Updates fusePlan, oldPlan, and redirects all fusion plans pointing to srcGroup to dstGroup.
  void setupDirectFusionPlan(FusionPlan &fusePlan, FusionPlan &oldPlan,
                             const GroupPtr srcGroup, const GroupPtr dstGroup);

  std::unordered_set<unsigned> finished;
  std::vector<unsigned> topoSortNodeIds;
  // Cache table to track processed (oldGroupId, newGroupId, closestIntersectionId) -> targetGroupId combinations
  std::unordered_map<std::tuple<unsigned, unsigned, unsigned>, unsigned, TupleHash> intersectionCache;
};

}  // namespace akg
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_AKGLOOPFUSIONANALYZER_H_
