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

#include "akg/Dialect/Affine/Analysis/AKGLoopFusionAnalyzer.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>
#include <queue>
#include <set>
#include <utility>

#include "akg/Dialect/Affine/Analysis/AKGLoopFusionBuilder.h"
#include "akg/Utils/AKGGlobalVars.hpp"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace akg {

// Comparison function for sorting groups in topological order.
// For global output groups, prioritizes by group template type, otherwise sorts by group ID.
static bool GroupCmp(const GroupPtr &g1, const GroupPtr &g2) {
  if (g1->isGlobalOut && g2->isGlobalOut) {
    if (g1->groupTemplate != g2->groupTemplate) {
      return static_cast<int>(g1->groupTemplate) > static_cast<int>(g2->groupTemplate);
    }
  }
  return g1->groupId > g2->groupId;
}

void FusionAnalyzer::initGroups() { groups = depGraph.groups; }

// Performs topological sorting of groups for fusion analysis.
// Groups are sorted according to their priority and dependencies using GroupCmp,
// then all node IDs from sorted groups are collected into topoSortNodeIds.
void FusionAnalyzer::topoSortInit() {
  topoSortNodeIds.clear();
  std::vector<GroupPtr> allGroups;
  allGroups.reserve(groups.size());
  std::transform(groups.begin(), groups.end(), std::back_inserter(allGroups),
                 [](const auto &pair) { return pair.second; });
  std::sort(allGroups.begin(), allGroups.end(), GroupCmp);
  for (auto g : allGroups) {
    for (auto node : g->nodesId) {
      topoSortNodeIds.push_back(node);
    }
  }
}

// Topological sort function for fusion plans
// Returns a sorted list of fusion plans based on their dependencies
// Example: Given fusion plans with dependencies: A->B, B->D, A->C, C->D
// Input: fusionPlans=[(A,B), (B,D), (A,C), (C,D)], numNodes=4
// Process: Start with nodes having no incoming edges (A), then B,C, finally D
// Output: [(A,B), (A,C), (B,D), (C,D)] in topological order
std::vector<FusionPlan> FusionAnalyzer::topoSortFusionPlans(unsigned numNodes) {
  // Remove duplicate edges based on fusedBand.from and fusedBand.to
  std::set<std::pair<unsigned, unsigned>> seenEdges;
  std::vector<FusionPlan> uniqueDependencies;
  for (const auto &plan : fusionPlans) {
    if (plan.fusedBand.from == plan.fusedBand.to) {
      continue;  // Skip self-loops
    }
    auto edgeKey = std::make_pair(plan.fusedBand.from, plan.fusedBand.to);
    if (seenEdges.insert(edgeKey).second) {
      uniqueDependencies.push_back(plan);
    }
  }

  std::vector<unsigned> inDegree(numNodes + 1, 0);

  // Calculate in-degree for each node by counting incoming edges
  for (const auto &edge : uniqueDependencies) {
    unsigned to = edge.fusedBand.to;
    inDegree[to]++;
  }

  // Nodes with zero in-degree are generally output nodes
  std::queue<unsigned> zeroInDegreeNodes;
  std::vector<FusionPlan> sortedEdges;

  // Find all nodes with zero in-degree and add them to the queue
  for (unsigned i = 0; i < numNodes; i++) {
    if (inDegree[i] == 0) {
      zeroInDegreeNodes.push(i);
    }
  }

  // Process nodes in topological order
  while (!zeroInDegreeNodes.empty()) {
    // Get the next node with zero in-degree
    unsigned node = zeroInDegreeNodes.front();
    zeroInDegreeNodes.pop();

    // Process all outgoing edges from this node
    for (auto &edge : uniqueDependencies) {
      if (edge.fusedBand.from == node) {
        setFusionType(edge);
        inferLoopTransforms(edge);
        sortedEdges.push_back(edge);
        unsigned to = edge.fusedBand.to;
        inDegree[to]--;

        if (inDegree[to] == 0) {
          zeroInDegreeNodes.push(to);
        }
      }
    }
  }
  return sortedEdges;
}

// Gets the next target group for fusion.
// Returns the group that should be the target of the next fusion operation.
GroupPtr FusionAnalyzer::getFusionTargetGroup() {
  for (auto nodeId : topoSortNodeIds) {
    if (!finished.count(nodeId)) {
      auto g = depGraph.getGroupByNode(nodeId);
      if (g != nullptr) {
        return g;
      }
      finished.insert(nodeId);
    }
  }
  return nullptr;
}

// Checks if the fusion plan is complete.
// Returns true if all nodes have been processed.
bool FusionAnalyzer::finishPlan() { return finished.size() == depGraph.nodes.size(); }

std::vector<FusionPlan>::iterator FusionAnalyzer::findFusionPlanByGroup(unsigned fromGroupId, unsigned toGroupId) {
  return std::find_if(fusionPlans.begin(), fusionPlans.end(), [fromGroupId, toGroupId](const FusionPlan &plan) {
    return plan.fusedGroup.from == fromGroupId && plan.fusedGroup.to == toGroupId;
  });
}

bool FusionAnalyzer::addFusionPlan(const FusionPlan &plan) {
  if (findFusionPlanByGroup(plan.fusedGroup.from, plan.fusedGroup.to) != fusionPlans.end()) {
    return false;
  }
  fusionPlans.emplace_back(plan);
  return true;
}

bool FusionAnalyzer::updateFusionPlanByGroup(unsigned fromGroupId, unsigned oldToGroupId, unsigned newToGroupId,
                                             unsigned newToNodeId) {
  // Check if the new plan would create a duplicate
  if (findFusionPlanByGroup(fromGroupId, newToGroupId) != fusionPlans.end()) {
    return false;
  }

  // Find the existing plan
  auto it = findFusionPlanByGroup(fromGroupId, oldToGroupId);
  if (it == fusionPlans.end()) {
    return false;
  }

  // Update the plan
  it->fusedGroup.to = newToGroupId;
  it->fusedBand.to = newToNodeId;
  return true;
}

bool FusionAnalyzer::removeFusionPlanByGroup(unsigned fromGroupId, unsigned toGroupId) {
  auto it = findFusionPlanByGroup(fromGroupId, toGroupId);
  if (it == fusionPlans.end()) {
    return false;
  }
  fusionPlans.erase(it);
  return true;
}

std::vector<FusionPlan>::iterator FusionAnalyzer::findFusionPlanByBand(unsigned fromBandId, unsigned toBandId) {
  return std::find_if(fusionPlans.begin(), fusionPlans.end(), [fromBandId, toBandId](const FusionPlan &plan) {
    return plan.fusedBand.from == fromBandId && plan.fusedBand.to == toBandId;
  });
}

bool FusionAnalyzer::removeFusionPlanByBand(unsigned fromBandId, unsigned toBandId) {
  auto it = findFusionPlanByBand(fromBandId, toBandId);
  if (it == fusionPlans.end()) {
    return false;
  }
  fusionPlans.erase(it);
  return true;
}

bool FusionAnalyzer::updateFusionPlanByBand(unsigned oldFromBandId, unsigned oldToBandId, unsigned newFromBandId,
                                            unsigned newToBandId) {
  auto it = findFusionPlanByBand(oldFromBandId, oldToBandId);
  if (it == fusionPlans.end()) {
    return false;
  }
  it->fusedBand.from = newFromBandId;
  it->fusedBand.to = newToBandId;
  return true;
}

// Helper function to find all reachable groups from a starting group through fusion plans
// Returns a map from reachable group ID to the shortest path length
std::unordered_map<unsigned, unsigned> FusionAnalyzer::findReachableGroups(unsigned startGroupId) {
  std::unordered_map<unsigned, unsigned> reachable;  // groupId -> path length
  std::queue<std::pair<unsigned, unsigned>> queue;   // (groupId, pathLength)
  reachable[startGroupId] = 0;
  queue.push({startGroupId, 0});

  while (!queue.empty()) {
    auto [currentGroup, pathLen] = queue.front();
    queue.pop();

    for (const auto &plan : fusionPlans) {
      if (plan.fusedGroup.from == currentGroup) {
        unsigned nextGroup = plan.fusedGroup.to;
        // If we haven't visited this group or found a shorter path
        if (reachable.find(nextGroup) == reachable.end() || reachable[nextGroup] > pathLen + 1) {
          reachable[nextGroup] = pathLen + 1;
          queue.push({nextGroup, pathLen + 1});
        }
      }
    }
  }
  return reachable;
}

// Finds all terminal nodes (nodes with no outgoing edges) in paths starting from srcGroupId in fusePlans.
// These terminal nodes represent the last nodes in different paths from srcGroupId.
//
// Example:
// Assume fusePlans contains the following edges:
//   - 33 -> 38
//   - 38 -> 42
// If srcGroupId=33, then:
//   - Path: 33 -> 38 -> 42 (terminal node is 42)
//   - Return value: [42]
std::vector<unsigned> FusionAnalyzer::findLastNodesInPath(unsigned srcGroupId) {
  std::vector<unsigned> lastNodes;
  auto reachable = findReachableGroups(srcGroupId);
  if (reachable.empty()) {
    return lastNodes;
  }

  // Find all nodes that have no outgoing edges in the path
  // These are the last nodes in different paths from srcGroupId
  for (const auto &[groupId, pathLen] : reachable) {
    // Check if this node has outgoing edges in the path
    bool hasOutgoing =
      std::any_of(fusionPlans.begin(), fusionPlans.end(), [groupId, &reachable](const FusionPlan &plan) {
        return plan.fusedGroup.from == groupId && reachable.find(plan.fusedGroup.to) != reachable.end();
      });

    // If this node has no outgoing edges in the path, it's a last node
    if (!hasOutgoing) {
      lastNodes.push_back(groupId);
    }
  }

  return lastNodes;
}

// Finds all terminal nodes (nodes with no outgoing edges) in paths starting from srcGroupId in fusePlans,
// then creates a fusion plan from each terminal node to dstGroupId (if the plan doesn't exist).
//
// Example:
// Assume fusePlans contains the following edges:
//   - 33 -> 38
//   - 38 -> 42
// If srcGroupId=33, dstGroupId=60, then:
//   - Found terminal nodes: [42]
//   - Created new plan: 42 -> 60
//
// Return value:
// - true: Found real paths and successfully connected (terminal node is not srcGroupId itself)
// - false: No real paths found (only srcGroupId itself, no other nodes)
bool FusionAnalyzer::connectLastNodesToTarget(unsigned srcGroupId, unsigned dstGroupId) {
  // Find all last nodes in paths starting from srcGroupId
  std::vector<unsigned> lastNodes = findLastNodesInPath(srcGroupId);

  // Check if real paths were found (lastNodes contains nodes other than srcGroupId)
  bool hasRealPaths =
    std::any_of(lastNodes.begin(), lastNodes.end(), [srcGroupId](unsigned nodeId) { return nodeId != srcGroupId; });

  if (lastNodes.empty()) {
    // If no paths exist, connect srcGroupId directly to dstGroupId
    lastNodes.push_back(srcGroupId);
  }

  // For each last node, create a fusion plan to dstGroupId if it doesn't exist
  for (unsigned lastNodeId : lastNodes) {
    // Skip if lastNodeId is the same as dstGroupId
    if (lastNodeId == dstGroupId) {
      continue;
    }

    // Create new fusion plan if it doesn't exist
    auto lastNodeGroup = depGraph.getGroup(lastNodeId);
    auto dstGroup = depGraph.getGroup(dstGroupId);

    if (lastNodeGroup != nullptr && dstGroup != nullptr) {
      FusionPlan newPlan;
      newPlan.fusedGroup = FuseEdge(lastNodeId, dstGroupId);
      newPlan.fusedBand = FuseEdge(lastNodeGroup->rootId, dstGroup->rootId);
      addFusionPlan(newPlan);
    }
  }

  // Return true if real paths were found (not just srcGroupId itself)
  return hasRealPaths;
}

// Checks if depGroupId exists in fusionPlans and has an edge (forward) to fromGroupId.
bool FusionAnalyzer::hasEdgeInFusionPlans(unsigned depGroupId, unsigned fromGroupId) {
  bool existsInPlans = std::any_of(fusionPlans.begin(), fusionPlans.end(), [depGroupId](const FusionPlan &plan) {
    return plan.fusedGroup.from == depGroupId || plan.fusedGroup.to == depGroupId;
  });
  if (!existsInPlans) {
    return false;
  }

  auto reachable = findReachableGroups(depGroupId);
  return reachable.find(fromGroupId) != reachable.end();
}

// Determines fusion type: if the nodes that to depends on exist in fusionPlans and have edges (forward) to from,
// or if from itself is a node that to directly or indirectly depends on (in the dependency graph),
// then the type is H (provided the dependent node is not load), otherwise it is V
void FusionAnalyzer::setFusionType(FusionPlan &plan) {
  unsigned fromGroupId = plan.fusedGroup.from;
  unsigned toGroupId = plan.fusedGroup.to;
  auto fromGroup = depGraph.getGroup(fromGroupId);
  auto toGroup = depGraph.getGroup(toGroupId);

  // 1. If from itself is a node that to directly or indirectly depends on (in the dependency graph), return H
  if (depGraph.isDependencyInGraph(fromGroupId, toGroupId)) {
    plan.depInfo = getGroupDependencies(toGroup, fromGroup);
    plan.fusionType = "H";
    return;
  }

  std::vector<unsigned> toDepGroups = depGraph.getDependentGroups(toGroupId);
  // 2. Check if each node that to depends on exists in fusionPlans and has an edge (forward) to from
  // If all dependent nodes satisfy the condition, return H, otherwise return V
  if (toDepGroups.empty()) {
    plan.fusionType = "V";
    return;
  }

  for (auto depGroupId : toDepGroups) {
    // Skip from itself
    if (depGroupId == fromGroupId) {
      continue;
    }

    if (hasEdgeInFusionPlans(depGroupId, fromGroupId)) {
      plan.depInfo = getGroupDependencies(toGroup, fromGroup);
      plan.fusionType = "H";
      return;
    }
  }

  plan.fusionType = "V";
}

std::pair<GroupPtr, GroupPtr> FusionAnalyzer::determineFusionOrder(const GroupPtr oldGroup, const GroupPtr newGroup) {
  if (oldGroup->groupTemplate == OperatorTemplate::Reduce) {
    return std::make_pair(newGroup, oldGroup);
  }
  if (newGroup->groupTemplate == OperatorTemplate::Reduce) {
    return std::make_pair(oldGroup, newGroup);
  }

  bool oldHasChild = false;
  bool newHasChild = false;
  for (auto prevPlan : fusionPlans) {
    if (prevPlan.fusedGroup.from == oldGroup->groupId) {
      oldHasChild = true;
    }
    if (prevPlan.fusedGroup.from == newGroup->groupId) {
      newHasChild = true;
    }
  }

  if (oldHasChild && !newHasChild) {
    // Example: old: A->B, B->C, new: A->D, output: D->B
    return std::make_pair(newGroup, oldGroup);
  } else if (!oldHasChild && newHasChild) {
    // Example: old: A->B, new: A->C, C->D, output: B->C
    return std::make_pair(oldGroup, newGroup);
  }

  // Default: Avoid fusing child nodes into parent nodes
  return std::make_pair(newGroup, oldGroup);
}

// Redirects shorter path to longer path's starting group.
// Modifies fusion plans to redirect paths ending at intersectionId to targetGroup.
void FusionAnalyzer::redirectFusionPlanToTarget(unsigned intersectionId,
                                                const std::unordered_map<unsigned, unsigned> &reachable,
                                                unsigned pathLen, GroupPtr targetGroup) {
  auto targetGroupId = targetGroup->groupId;
  auto targetNodeId = targetGroup->rootId;
  for (auto it = fusionPlans.begin(); it != fusionPlans.end(); ++it) {
    if (it->fusedGroup.to != intersectionId) {
      continue;
    }

    auto fromIt = reachable.find(it->fusedGroup.from);
    if (fromIt != reachable.end() && fromIt->second == pathLen - 1) {
      // Update the plan using the encapsulated method
      updateFusionPlanByGroup(it->fusedGroup.from, intersectionId, targetGroupId, targetNodeId);
      return;
    }
  }
}

// Handles backward intersection points between two groups, finds the closest backward
// intersection point and redirects fusion paths.
//
// Description:
// When oldGroup and newGroup have backward intersection points in the fusion graph,
// finds the closest backward intersection point (shortest total path), and redirects
// the shorter path to the starting group of the longer path to optimize the fusion structure.
//
// Example:
// Assume: newGroup has groupId=19 with fusion plan 19->29
//         oldGroup has groupId=19 with fusion plan 19->34
//         fusionPlans contains: 29->39, 34->39
//
// Fusion result:
// 19->29, 29->34, 34->39
GroupPtr FusionAnalyzer::handleBackwardIntersectionPoints(const GroupPtr oldGroup, const GroupPtr newGroup) {
  // Find all reachable groups from oldGroup and newGroup
  auto oldReachable = findReachableGroups(oldGroup->groupId);
  auto newReachable = findReachableGroups(newGroup->groupId);

  // Find the backward intersection point with the shortest total distance (oldPathLen + newPathLen)
  unsigned closestIntersectionId = 0;
  unsigned minTotalDistance = std::numeric_limits<unsigned>::max();
  unsigned minOldPathLen = std::numeric_limits<unsigned>::max();
  unsigned minNewPathLen = std::numeric_limits<unsigned>::max();
  bool foundIntersection = false;

  for (const auto &[groupId, oldPathLen] : oldReachable) {
    if (newReachable.find(groupId) != newReachable.end()) {
      unsigned newPathLen = newReachable[groupId];
      unsigned totalDistance = oldPathLen + newPathLen;

      // Track the backward intersection point with the shortest total distance
      // If total distances are equal, prefer the one with shorter individual path
      if (totalDistance < minTotalDistance) {
        minTotalDistance = totalDistance;
        closestIntersectionId = groupId;
        minOldPathLen = oldPathLen;
        minNewPathLen = newPathLen;
        foundIntersection = true;
      }
    }
  }

  // If no backward intersection points, return nullptr
  if (!foundIntersection) {
    return nullptr;
  }

  // Check cache: normalize cache key to handle reversed (oldGroup, newGroup) order
  // Ensure first groupId is always smaller to make (A, B, C) and (B, A, C) equivalent
  unsigned firstGroupId = std::min(oldGroup->groupId, newGroup->groupId);
  unsigned secondGroupId = std::max(oldGroup->groupId, newGroup->groupId);
  auto cacheKey = std::make_tuple(firstGroupId, secondGroupId, closestIntersectionId);
  auto cacheIt = intersectionCache.find(cacheKey);
  if (cacheIt != intersectionCache.end()) {
    unsigned cachedTargetGroupId = cacheIt->second;
    auto cachedTargetGroup = depGraph.getGroup(cachedTargetGroupId);
    if (cachedTargetGroup != nullptr) {
      return cachedTargetGroup;
    }
  }

  // Determine which group has shorter path and redirect accordingly
  GroupPtr targetGroup;
  if (minOldPathLen < minNewPathLen) {
    // Old path is shorter, redirect it to newGroup
    if (minOldPathLen == 0) {
      targetGroup = newGroup;
    } else {
      redirectFusionPlanToTarget(closestIntersectionId, oldReachable, minOldPathLen, newGroup);
      targetGroup = oldGroup;
    }
  } else {
    // New path is shorter or equal, redirect it to oldGroup
    if (minNewPathLen == 0) {
      targetGroup = oldGroup;
    } else {
      redirectFusionPlanToTarget(closestIntersectionId, newReachable, minNewPathLen, oldGroup);
      targetGroup = newGroup;
    }
  }
  intersectionCache[cacheKey] = targetGroup->groupId;
  return targetGroup;
}

// Sets up direct fusion plan from srcGroup to dstGroup and links all source groups to destination group.
// Updates fusePlan, oldPlan, and redirects all fusion plans pointing to srcGroup to dstGroup.
void FusionAnalyzer::setupDirectFusionPlan(FusionPlan &fusePlan, FusionPlan &oldPlan, const GroupPtr srcGroup,
                                           const GroupPtr dstGroup) {
  // Set up direct fusion plan from srcGroup to dstGroup
  fusePlan.fusedGroup.from = srcGroup->groupId;
  fusePlan.fusedGroup.to = dstGroup->groupId;
  fusePlan.fusedBand.from = srcGroup->rootId;
  fusePlan.fusedBand.to = dstGroup->rootId;

  // Update oldPlan to point to srcGroup
  updateFusionPlanByGroup(oldPlan.fusedGroup.from, oldPlan.fusedGroup.to, srcGroup->groupId, srcGroup->rootId);

  // Link all source groups to destination group
  // Collect plans to update first to avoid iterator invalidation
  std::vector<std::pair<unsigned, unsigned>> plansToUpdate;
  for (const auto &plan : fusionPlans) {
    if (plan.fusedGroup.to == srcGroup->groupId) {
      plansToUpdate.emplace_back(plan.fusedGroup.from, plan.fusedGroup.to);
    }
  }

  // Update collected plans
  for (const auto &[fromId, oldToId] : plansToUpdate) {
    updateFusionPlanByGroup(fromId, oldToId, dstGroup->groupId, fusePlan.fusedBand.to);
  }
}

/*
Logic for adding to fusionPlans:
1. If fusionPlans contains the same from and to, return false directly
2. If fusionPlans does not contain the same from, add it to fusionPlans directly
3. If fusionPlans contains the same from, i.e., there exists A->B,
   and currently A->C, i.e., a node needs to fuse to multiple nodes, which is not allowed,
   therefore the fusion plan needs to be updated:
   1) B and C have backward intersection points: fuse the shorter path to the longer one based on distance to
intersection, update the shorter fusion path (handleBackwardIntersectionPoints), e.g.: fusionPlans already has A->B->E,
C->D->E, now adding A->C, then update fusion plan to A->B->C->D->E 2) B and C have no backward intersection points, fuse
the new fuseplan into the old fuseplan

After determining the fusion fusionPlan, we need to determine fusionType:
If the nodes that to depends on have a dependency relationship with from, or there exists an edge in fusionPlans, then
it is H, otherwise it is V
*/
bool FusionAnalyzer::checkAndFixMultiOut(FusionPlan &fusePlan) {
  // If fusionPlans contains the same from and to, return false directly
  if (findFusionPlanByGroup(fusePlan.fusedGroup.from, fusePlan.fusedGroup.to) != fusionPlans.end()) {
    return false;
  }

  for (auto it = fusionPlans.begin(); it != fusionPlans.end(); ++it) {
    auto &oldPlan = *it;
    bool multiOut = oldPlan.fusedGroup.from == fusePlan.fusedGroup.from;

    if (multiOut) {
      auto oldGroup = depGraph.getGroup(oldPlan.fusedGroup.to);
      auto newGroup = depGraph.getGroup(fusePlan.fusedGroup.to);

      if (oldGroup == nullptr || newGroup == nullptr) {
        continue;
      }

      // Check if backward intersection points exist
      auto targetGroup = handleBackwardIntersectionPoints(oldGroup, newGroup);
      if (targetGroup != nullptr) {
        // Fuse the shorter path to the longer one based on distance to intersection, update the shorter fusion path
        // Example: A->B->E, C->D->E, update to A->B->C->D->E
        updateFusionPlanByGroup(oldPlan.fusedGroup.from, oldPlan.fusedGroup.to, targetGroup->groupId,
                                targetGroup->rootId);
        return false;
      }

      // No backward intersection points, fuse the new fuseplan into the old fuseplan
      auto [srcGroup, dstGroup] = determineFusionOrder(oldGroup, newGroup);

      // Connect all last nodes in paths from srcGroup to dstGroup
      bool pathsConnected = connectLastNodesToTarget(srcGroup->groupId, dstGroup->groupId);
      if (pathsConnected) {
        updateFusionPlanByGroup(oldPlan.fusedGroup.from, oldPlan.fusedGroup.to, srcGroup->groupId, srcGroup->rootId);
        return false;
      }

      // No paths found, set up direct fusion plan
      setupDirectFusionPlan(fusePlan, oldPlan, srcGroup, dstGroup);
      return true;
    }
  }
  return true;
}

void FusionAnalyzer::inferLoopTransforms(FusionPlan &plan) {
  // Get source and target groups from the plan
  auto sourceGroup = groups[plan.fusedGroup.from];
  auto targetGroup = groups[plan.fusedGroup.to];

  if (sourceGroup == nullptr || targetGroup == nullptr) {
    return;
  }

  // Infer loop transforms based on group templates and set them directly in the plan
  if (targetGroup->groupTemplate == OperatorTemplate::Broadcast) {
    plan.loopTransform = LoopTransform::Replicate;
  } else if (targetGroup->groupTemplate == OperatorTemplate::Transpose) {
    if (sourceGroup->groupTemplate == OperatorTemplate::Reshape) {
      plan.loopTransform = LoopTransform::StripMine;
    }
  } else if (targetGroup->groupTemplate == OperatorTemplate::Reduce) {
    if (sourceGroup->groupTemplate == OperatorTemplate::ReduceInit) {
      plan.loopTransform = LoopTransform::ReplicateIf;
    }
  }
}

// Applies loop transforms and fuses source group into target group.
// Records the fusion plan and updates group relationships.
void FusionAnalyzer::applyAndFuse(const GroupPtr targetGroup, const GroupPtr sourceGroup) {
  sourceGroup->fusedGroupId.emplace_back(targetGroup->groupId);

  for (auto fuseTargetId : targetGroup->nodesId) {
    finished.insert(fuseTargetId);
  }

  // Get the for node IDs of sourceGroup and targetGroup
  auto srcNodeId = sourceGroup->rootId;
  auto dstNodeId = targetGroup->rootId;
  FusionPlan fusePlan;
  fusePlan.fusedGroup = FuseEdge(sourceGroup->groupId, targetGroup->groupId);
  fusePlan.fusedBand = FuseEdge(srcNodeId, dstNodeId);

  bool shouldInsert = checkAndFixMultiOut(fusePlan);

  if (shouldInsert) {
    addFusionPlan(fusePlan);
  }
}

// Gets directly connected predecessor nodes for a given node.
// Filters out nodes that have indirect connections through other nodes.
// Example: Given dependency graph A->B->C, A->D->C, B->D
// Querying node C's direct predecessors:
// - All predecessors: [A, B, D] (sorted descending: [D, B, A])
// - Check B: B->D exists, so B is indirect predecessor, skip
// - Check A: A->B exists, so A is indirect predecessor, skip
// - Result: only D is direct predecessor
void FusionAnalyzer::precomputeDirectPredecessors() {
  directPredecessorsCache.clear();

  // Precompute direct predecessors for all nodes
  for (const auto &nodePair : depGraph.nodes) {
    unsigned nodeId = nodePair.first;

    // Get all incoming edges with their memrefs directly from inEdges
    auto inEdgesIt = depGraph.inEdges.find(nodeId);
    if (inEdgesIt == depGraph.inEdges.end()) {
      continue;
    }

    // Build a map from predecessor node ID to all edges (memrefs and loopDepth) from that node
    std::unordered_map<unsigned, std::vector<std::pair<Value, unsigned>>> predToMemrefsAndDepth;
    for (const auto &edge : inEdgesIt->second) {
      predToMemrefsAndDepth[edge.id].push_back({edge.value, edge.loopDepth});
    }

    // Get all predecessor IDs
    std::vector<unsigned> allPredecessorIds;
    allPredecessorIds.reserve(predToMemrefsAndDepth.size());
    std::transform(predToMemrefsAndDepth.begin(), predToMemrefsAndDepth.end(), std::back_inserter(allPredecessorIds),
                   [](const auto &pair) { return pair.first; });

    // Sort in descending order to prioritize direct predecessors
    std::sort(allPredecessorIds.begin(), allPredecessorIds.end(), std::greater<int>());

    std::unordered_set<size_t> skipIdx;
    // i=0: direct predecessor
    for (size_t i = 1; i < allPredecessorIds.size(); ++i) {
      if (skipIdx.count(i)) {
        continue;
      }
      auto pid = allPredecessorIds[i];
      // Predecessor nodes are sorted in descending order, so j < i means checking earlier nodes
      for (size_t j = 0; j < i; ++j) {
        if (skipIdx.count(j)) {
          continue;
        }
        // Check if there exists an edge from pid to allPredecessorIds[j]
        // If edge exists, pid is not a direct predecessor but an indirect one, add to skipIdx
        if (depGraph.hasEdge(pid, allPredecessorIds[j], mlir::Value())) {
          skipIdx.insert(i);
          break;
        }
      }
    }

    // Build direct predecessors list with memrefs and loopDepth
    std::vector<DirectPredecessor> directPreds;
    for (size_t i = 0; i < allPredecessorIds.size(); ++i) {
      if (!skipIdx.count(i)) {
        auto predId = allPredecessorIds[i];
        // For each direct predecessor, store all memrefs and loopDepth that create the dependency
        // Typically there's one memref per edge, but we store all to be complete
        const auto &memrefsAndDepths = predToMemrefsAndDepth[predId];
        std::transform(memrefsAndDepths.begin(), memrefsAndDepths.end(), std::back_inserter(directPreds),
                       [predId](const auto &memrefAndDepth) {
                         return DirectPredecessor(predId, memrefAndDepth.first, memrefAndDepth.second);
                       });
      }
    }
    directPredecessorsCache[nodeId] = std::move(directPreds);
  }
}

// Gets dependent operations between source and target groups.
// Uses cached direct predecessors for efficient lookup.
// Only records the operations (targetNodeId and predId) that have direct dependencies.
DependenceInfo FusionAnalyzer::getGroupDependencies(const GroupPtr targetGroup, const GroupPtr sourceGroup) {
  // Check cache first
  auto cacheKey = std::make_pair(sourceGroup->groupId, targetGroup->groupId);
  auto cacheIt = groupDependenciesCache.find(cacheKey);
  if (cacheIt != groupDependenciesCache.end()) {
    return cacheIt->second;
  }

  DependenceInfo depInfo;
  // Iterate through target group nodes to find dependencies from source group
  for (auto targetNodeId : targetGroup->nodesId) {
    // Use cached direct predecessors for fast lookup
    auto it = directPredecessorsCache.find(targetNodeId);
    if (it == directPredecessorsCache.end()) {
      continue;
    }

    const auto &directPreds = it->second;
    for (const auto &directPred : directPreds) {
      auto predNodeId = directPred.nodeId;
      // Check if predecessor belongs to source group
      auto predGroup = depGraph.getGroupByNode(predNodeId);
      if (predGroup == nullptr || predGroup->groupId != sourceGroup->groupId) {
        continue;
      }

      // Found a dependency edge - record the main operations (targetNodeId and predId) and memref and loopDepth
      depInfo.sourceOps.push_back(predNodeId);
      depInfo.targetOps.push_back(targetNodeId);
      depInfo.loopDepth = std::min(depInfo.loopDepth, directPred.loopDepth);
      if (isa<MemRefType>(directPred.memref.getType())) {
        depInfo.memrefs.push_back(directPred.memref);
      }
    }
  }

  // Cache the result
  groupDependenciesCache[cacheKey] = depInfo;

  return depInfo;
}

void FusionAnalyzer::plan() {
  initGroups();
  topoSortInit();
  precomputeDirectPredecessors();

  while (!finishPlan()) {
    auto targetGroup = getFusionTargetGroup();
    if (targetGroup == nullptr) {
      break;
    }

    std::vector<unsigned> sourceGroupIds;
    for (auto fuseTargetId : targetGroup->nodesId) {
      auto it = directPredecessorsCache.find(fuseTargetId);
      if (it == directPredecessorsCache.end()) {
        continue;
      }

      const auto &directPreds = it->second;
      // Use a set to track unique group IDs we've already seen
      std::unordered_set<unsigned> seenGroupIds;
      for (const auto &directPred : directPreds) {
        auto id = directPred.nodeId;
        auto tmp = depGraph.getGroupByNode(id);
        if (tmp != nullptr && tmp->groupId != targetGroup->groupId) {
          if (seenGroupIds.insert(tmp->groupId).second) {
            sourceGroupIds.emplace_back(tmp->groupId);
          }
        }
      }
    }
    if (sourceGroupIds.empty()) {
      for (auto fuseTargetId : targetGroup->nodesId) {
        finished.insert(fuseTargetId);
      }
      continue;
    }
    for (auto id : sourceGroupIds) {
      auto sourceGroup = groups[id];
      applyAndFuse(targetGroup, sourceGroup);
    }
  }

  auto sortedPlan = topoSortFusionPlans(depGraph.nodes.size());
  fusionPlans = sortedPlan;
}

void FusionAnalyzer::print(llvm::raw_ostream &os) const {
  std::unordered_map<int, std::string> loopTransformToStr{
    {static_cast<int>(LoopTransform::Merge), "Merge"},
    {static_cast<int>(LoopTransform::Replicate), "Replicate"},
    {static_cast<int>(LoopTransform::ReplicateIf), "ReplicateIf"},
    {static_cast<int>(LoopTransform::Permute), "Permute"},
    {static_cast<int>(LoopTransform::StripMine), "StripMine"},
    {static_cast<int>(LoopTransform::Collapse), "Collapse"},
    {static_cast<int>(LoopTransform::BackTracking), "BackTracking"}};

  os << "\n===== FusionPlans =====\n";
  for (const auto &plan : fusionPlans) {
    os << "FusionPlan: Group [" << plan.fusedGroup.from << " -> " << plan.fusedGroup.to << "], "
       << "Band [" << plan.fusedBand.from << " -> " << plan.fusedBand.to << "], "
       << "FusionType: " << plan.fusionType << ", "
       << "LoopTransform: " << loopTransformToStr[static_cast<int>(plan.loopTransform)] << "\n";
  }
}

}  // namespace akg
}  // namespace mlir
