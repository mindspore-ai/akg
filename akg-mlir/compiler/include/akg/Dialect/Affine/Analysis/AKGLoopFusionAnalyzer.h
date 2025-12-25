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

struct TupleHash {
  std::size_t operator()(const std::tuple<unsigned, unsigned, unsigned> &t) const {
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

struct FusionAnalyzer {
 public:
  FusionAnalyzer(MemRefDependenceGraphForFusion depGraph, func::FuncOp funcOp) : depGraph(depGraph), funcOp(funcOp) {}

  void plan();
  void applyAndFuse(std::vector<LoopTransform> loopTransforms, const GroupPtr targetGroup, const GroupPtr sourceGroup);
  bool checkAndFixMultiOut(FusionPlan &fusePlan);
  std::vector<LoopTransform> inferLoopTransforms(const GroupPtr targetGroup, const GroupPtr sourceGroup);

  // Debug and print
  void print(llvm::raw_ostream &os) const;
  void dump() const { print(llvm::errs()); }

  MemRefDependenceGraphForFusion depGraph{nullptr};
  func::FuncOp funcOp{nullptr};
  std::unordered_map<unsigned, GroupPtr> groups;
  std::vector<FusionPlan> fusionPlans;

 private:
  // Initialization and Topological Sorting
  void initGroups();
  void topoSortInit();
  std::vector<FusionPlan> topoSortFusionPlans(unsigned numNodes);
  GroupPtr getFusionTargetGroup();
  void getDirectlyPredecessors(unsigned id, std::vector<unsigned> &predecessorIds);
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

  // Fusion Type and Order Determination
  std::string determineFusionType(unsigned fromGroupId, unsigned toGroupId);
  bool hasEdgeInFusionPlans(unsigned depGroupId, unsigned fromGroupId);
  std::pair<GroupPtr, GroupPtr> determineFusionOrder(const GroupPtr oldGroup, const GroupPtr newGroup);

  // Intersection Point Handling
  GroupPtr handleBackwardIntersectionPoints(const GroupPtr oldGroup, const GroupPtr newGroup);
  void redirectFusionPlanToTarget(unsigned intersectionId, const std::unordered_map<unsigned, unsigned> &reachable,
                                  unsigned pathLen, GroupPtr targetGroup);
  void setupDirectFusionPlan(FusionPlan &fusePlan, FusionPlan &oldPlan, const GroupPtr srcGroup,
                             const GroupPtr dstGroup);

  // Member Variables
  std::unordered_set<unsigned> finished;
  std::vector<unsigned> topoSortNodeIds;
  std::unordered_map<std::tuple<unsigned, unsigned, unsigned>, unsigned, TupleHash> intersectionCache;
};

}  // namespace akg
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_AKGLOOPFUSIONANALYZER_H_
