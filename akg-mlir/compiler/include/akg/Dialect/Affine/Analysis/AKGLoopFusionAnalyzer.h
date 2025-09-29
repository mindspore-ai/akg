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

#include "akg/Dialect/Affine/Analysis/DependenceAnalysis.h"
#include "akg/Dialect/Affine/Analysis/AKGLoopFusionBuilder.h"
#include "akg/Utils/AnalysisCommon.hpp"

#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace mlir {
namespace akg {

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
  void checkAndFixMultiOut(FusionPlan &fusePlan);

  // Applies loop transforms and fuses source group into target group.
  // Records the fusion plan and updates group relationships.
  void applyAndFuse(std::vector<LoopTransform> loopTransforms, const GroupPtr targetGroup, const GroupPtr sourceGroup);

  // Main planning function that orchestrates the fusion process.
  // Analyzes groups, performs topological sorting, and creates fusion plans.
  void plan();

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

  std::unordered_set<unsigned> finished;
  std::vector<unsigned> topoSortNodeIds;
};

}  // namespace akg
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_AKGLOOPFUSIONANALYZER_H_
