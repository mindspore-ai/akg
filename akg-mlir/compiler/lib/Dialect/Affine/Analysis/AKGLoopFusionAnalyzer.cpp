/**
 * Copyright 2035 Huawei Technologies Co., Ltd
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
#include <iterator>
#include <limits>
#include <queue>
#include <set>

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

using llvm::outs;
// Topological sort function for fusion plans
// Returns a sorted list of fusion plans based on their dependencies
// Example: Given fusion plans with dependencies: A->B, B->D, A->C, C->D
// Input: dependencies=[(A,B), (B,D), (A,C), (C,D)], numNodes=4
// Process: Start with nodes having no incoming edges (A), then B,C, finally D
// Output: [(A,B), (A,C), (B,D), (C,D)] in topological order
static std::vector<FusionPlan> topologicalSort(const std::vector<FusionPlan> &dependencies, unsigned numNodes) {
  // Remove duplicate edges based on fusedBand.from and fusedBand.to
  std::set<std::pair<unsigned, unsigned>> seenEdges;
  std::vector<FusionPlan> uniqueDependencies;
  for (const auto &plan : dependencies) {
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
    for (const auto &edge : uniqueDependencies) {
      if (edge.fusedBand.from == node) {
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

// Comparison function for sorting groups in topological order.
// For global output groups, prioritizes by group template type, otherwise sorts by group ID.
bool GroupCmp(const GroupPtr &g1, const GroupPtr &g2) {
  if (g1->isGlobalOut && g2->isGlobalOut) {
    if (g1->groupTemplate != g2->groupTemplate) {
      return static_cast<int>(g1->groupTemplate) > static_cast<int>(g2->groupTemplate);
    }
  }
  return g1->groupId > g2->groupId;
}

std::vector<LoopTransform> FusionAnalyzer::inferLoopTransforms(const GroupPtr targetGroup, const GroupPtr sourceGroup) {
  std::vector<LoopTransform> res;
  if (targetGroup->groupTemplate == OperatorTemplate::Broadcast) {
    if (sourceGroup->groupTemplate == OperatorTemplate::Elementwise) {
      res.emplace_back(LoopTransform::Replicate);
    }
  } else if (targetGroup->groupTemplate == OperatorTemplate::Transpose) {
    if (sourceGroup->groupTemplate == OperatorTemplate::Reshape) {
      res.emplace_back(LoopTransform::StripMine);
    }
  }
  return res;
}


// Helper function to find all reachable groups from a starting group through fusion plans
// Returns a map from reachable group ID to the shortest path length
static std::unordered_map<unsigned, unsigned> findReachableGroups(
    unsigned startGroupId, const std::vector<FusionPlan> &plans) {
  std::unordered_map<unsigned, unsigned> reachable;  // groupId -> path length
  std::queue<std::pair<unsigned, unsigned>> queue;  // (groupId, pathLength)
  reachable[startGroupId] = 0;
  queue.push({startGroupId, 0});

  while (!queue.empty()) {
    auto [currentGroup, pathLen] = queue.front();
    queue.pop();

    for (const auto &plan : plans) {
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

// Helper function to find all last nodes in paths starting from srcGroupId
// Returns a vector of groupIds of all last nodes in paths, or empty vector if no path exists
// Example: If fusionPlans contains 33->38, 38->42, 33->52, and srcGroupId=33,
//          the function returns [42, 52] (all last nodes in different paths)
static std::vector<unsigned> findLastNodesInPath(unsigned srcGroupId, const std::vector<FusionPlan> &plans) {
  std::vector<unsigned> lastNodes;
  auto reachable = findReachableGroups(srcGroupId, plans);
  if (reachable.empty()) {
    return lastNodes;
  }

  // Find all nodes that have no outgoing edges in the path
  // These are the last nodes in different paths from srcGroupId
  for (const auto &[groupId, pathLen] : reachable) {
    // Check if this node has outgoing edges in the path
    bool hasOutgoing = false;
    for (const auto &plan : plans) {
      if (plan.fusedGroup.from == groupId) {
        // Check if the target is also in the reachable set (part of the path)
        if (reachable.find(plan.fusedGroup.to) != reachable.end()) {
          hasOutgoing = true;
          break;
        }
      }
    }

    // If this node has no outgoing edges in the path, it's a last node
    if (!hasOutgoing) {
      lastNodes.push_back(groupId);
    }
  }

  return lastNodes;
}

// Connects all last nodes in paths from srcGroupId to dstGroupId.
// Creates fusion plans from each last node to dstGroupId if they don't already exist.
// Example: If srcGroupId=33, paths are 33->38->42 and 33->52, and dstGroupId=60,
//          creates plans 42->60 and 52->60
// Returns true if paths were found and connected (i.e., lastNodes exist and are not just srcGroupId),
// false otherwise
bool FusionAnalyzer::connectLastNodesToTarget(unsigned srcGroupId, unsigned dstGroupId) {
  // Find all last nodes in paths starting from srcGroupId
  std::vector<unsigned> lastNodes = findLastNodesInPath(srcGroupId, fusionPlans);

  // Check if real paths were found (lastNodes contains nodes other than srcGroupId)
  bool hasRealPaths = std::any_of(lastNodes.begin(), lastNodes.end(),
      [srcGroupId](unsigned nodeId) { return nodeId != srcGroupId; });

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

    // Check if a plan from lastNodeId to dstGroupId already exists
    bool planExists = std::any_of(fusionPlans.begin(), fusionPlans.end(),
        [lastNodeId, dstGroupId](const FusionPlan &plan) {
          return plan.fusedGroup.from == lastNodeId && plan.fusedGroup.to == dstGroupId;
        });

    // Create new fusion plan if it doesn't exist
    if (!planExists) {
      auto lastNodeGroup = depGraph.getGroup(lastNodeId);
      auto dstGroup = depGraph.getGroup(dstGroupId);

      if (lastNodeGroup != nullptr && dstGroup != nullptr) {
        FusionPlan newPlan;
        newPlan.fusedGroup = FuseEdge(lastNodeId, dstGroupId);
        newPlan.fusedBand = FuseEdge(
            depGraph.getNodeId(lastNodeGroup->getLeadingFor()),
            depGraph.getNodeId(dstGroup->getLeadingFor()));
        newPlan.fusionType = "V";

        llvm::outs() << "Creating new plan: group " << lastNodeId << " -> " << dstGroupId
                     << " (node " << newPlan.fusedBand.from << " -> " << newPlan.fusedBand.to << ")\n";
        fusionPlans.emplace_back(newPlan);
      }
    }
  }

  // Return true if real paths were found (not just srcGroupId itself)
  return hasRealPaths;
}

// Redirects shorter path to longer path's starting group.
// Modifies fusion plans to redirect paths ending at intersectionId to targetGroup.
void FusionAnalyzer::redirectFusionPlanToTarget(unsigned intersectionId,
                                                 const std::unordered_map<unsigned, unsigned> &reachable,
                                                 unsigned pathLen, GroupPtr targetGroup) {
  auto targetGroupId = targetGroup->groupId;
  auto targetNodeId = depGraph.getNodeId(targetGroup->getLeadingFor());
  for (auto it = fusionPlans.begin(); it != fusionPlans.end(); ++it) {
    if (it->fusedGroup.to != intersectionId) {
      continue;
    }

    auto fromIt = reachable.find(it->fusedGroup.from);
    if (fromIt != reachable.end() && fromIt->second == pathLen - 1) {
      it->fusedGroup.to = targetGroupId;
      it->fusedBand.to = targetNodeId;
      it->fusionType = "V";

      auto group = depGraph.getGroup(it->fusedGroup.from);
      unsigned fromNodeId = 0;
      if (group != nullptr) {
        fromNodeId = depGraph.getNodeId(group->getLeadingFor());
      }

      auto intersectionGroup = depGraph.getGroup(intersectionId);
      unsigned toNodeId = 0;
      if (intersectionGroup != nullptr) {
        toNodeId = depGraph.getNodeId(intersectionGroup->getLeadingFor());
      }
      llvm::outs() << "Modifying plan from nodeId " << fromNodeId
                    << " -> " << toNodeId
                    << " to nodeId " << fromNodeId << " -> " << targetNodeId << "\n";
      return;
    }
  }
}

// Sets up direct fusion plan from srcGroup to dstGroup and links all source groups to destination group.
// Updates fusePlan, oldPlan, and redirects all fusion plans pointing to srcGroup to dstGroup.
void FusionAnalyzer::setupDirectFusionPlan(FusionPlan &fusePlan, FusionPlan &oldPlan,
                                            const GroupPtr srcGroup, const GroupPtr dstGroup) {
  // Set up direct fusion plan from srcGroup to dstGroup
  fusePlan.fusedGroup.from = srcGroup->groupId;
  fusePlan.fusedGroup.to = dstGroup->groupId;
  fusePlan.fusedBand.from = depGraph.getNodeId(srcGroup->getLeadingFor());
  fusePlan.fusedBand.to = depGraph.getNodeId(dstGroup->getLeadingFor());

  // Update oldPlan to point to srcGroup
  oldPlan.fusedGroup.to = srcGroup->groupId;
  oldPlan.fusedBand.to = depGraph.getNodeId(srcGroup->getLeadingFor());

  // Link all source groups to destination group
  for (size_t i = 0; i < fusionPlans.size(); ++i) {
    if (fusionPlans[i].fusedGroup.to == srcGroup->groupId) {
      llvm::outs() << "Update fusePlan!!!\n";
      llvm::outs() << "Old plan node " << fusionPlans[i].fusedBand.from << " -> "
                   << fusionPlans[i].fusedBand.to << " to ";
      fusionPlans[i].fusedGroup.to = dstGroup->groupId;
      fusionPlans[i].fusedBand.to = fusePlan.fusedBand.to;
      llvm::outs() << "Updated plan node " << fusionPlans[i].fusedBand.from << " -> "
                   << fusionPlans[i].fusedBand.to << "\n";
    }
  }

  llvm::outs() << "New Plan node " << fusePlan.fusedBand.from << " to " << fusePlan.fusedBand.to << "\n";
  fusePlan.fusionType = "V";
}

std::pair<GroupPtr, GroupPtr> FusionAnalyzer::determineFusionOrder(
    const GroupPtr oldGroup, const GroupPtr newGroup) {
  std::string oldTemplateString = oldGroup->getGroupTemplateString();
  std::string newTemplateString = newGroup->getGroupTemplateString();
  llvm::outs() << "oldGroup " << oldGroup->groupId << " groupTemplate " << oldTemplateString
               << ", new group " << newGroup->groupId << " groupTemplate " << newTemplateString << "\n";

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
    llvm::outs() << "Only old group has child\n";
    return std::make_pair(newGroup, oldGroup);
  } else if (!oldHasChild && newHasChild) {
    // Example: old: A->B, new: A->C, C->D, output: B->C
    llvm::outs() << "Only new group has child\n";
    return std::make_pair(oldGroup, newGroup);
  }

  // Default: Avoid fusing child nodes into parent nodes
  return std::make_pair(newGroup, oldGroup);
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
  auto oldReachable = findReachableGroups(oldGroup->groupId, fusionPlans);
  auto newReachable = findReachableGroups(newGroup->groupId, fusionPlans);

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
      llvm::outs() << "Cache hit: returning cached targetGroup " << cachedTargetGroupId
                   << " for (oldGroupId=" << oldGroup->groupId
                   << ", newGroupId=" << newGroup->groupId
                   << ", closestIntersectionId=" << closestIntersectionId << ")\n";
      return cachedTargetGroup;
    }
  }


  // Process only the closest backward intersection point
  auto intersectionGroup = depGraph.getGroup(closestIntersectionId);
  unsigned intersectionNodeId = 0;
  if (intersectionGroup != nullptr) {
    intersectionNodeId = depGraph.getNodeId(intersectionGroup->getLeadingFor());
  }

  llvm::outs() << "Found closest backward intersection point: nodeId " << intersectionNodeId
               << ", oldGroup path length: " << minOldPathLen
               << ", newGroup path length: " << minNewPathLen
               << ", total distance: " << minTotalDistance << "\n";

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

bool FusionAnalyzer::checkAndFixMultiOut(FusionPlan &fusePlan) {
  // Use iterator-based loop to safely handle deletion
  for (auto it = fusionPlans.begin(); it != fusionPlans.end(); ++it) {
    auto &oldPlan = *it;
    // Check if a for loop is fused into different for loops
    bool multiOut = oldPlan.fusedGroup.from == fusePlan.fusedGroup.from;

    if (multiOut) {
      llvm::outs() << "Original Plan node " << fusePlan.fusedBand.from << " to " << fusePlan.fusedBand.to << "\n";
      llvm::outs() << "Old Plan node " << oldPlan.fusedBand.from << " to " << oldPlan.fusedBand.to << "\n";

      // Handle duplicate plan case
      if (oldPlan.fusedGroup.to == fusePlan.fusedGroup.to) {
        it = fusionPlans.erase(it);
        return true;  // Still need to insert fusePlan
      }

      llvm::outs() << "Convert multiout pc plan to sib plan\n";
      // Get group references
      auto oldGroup = depGraph.getGroup(oldPlan.fusedGroup.to);
      auto newGroup = depGraph.getGroup(fusePlan.fusedGroup.to);

      auto targetGroup = handleBackwardIntersectionPoints(oldGroup, newGroup);
      if (targetGroup != nullptr) {
        oldPlan.fusedGroup.to = targetGroup->groupId;
        oldPlan.fusedBand.to = depGraph.getNodeId(targetGroup->getLeadingFor());
        return false;  // fusePlan is not needed, don't insert it
      }

      // Handle non-dependency case
      auto [srcGroup, dstGroup] = determineFusionOrder(oldGroup, newGroup);

      // Connect all last nodes in paths from srcGroup to dstGroup
      // Example: If srcGroup=33, paths are 33->38->42 and 33->52, and dstGroup=60,
      //          creates plans 42->60 and 52->60
      llvm::outs() << "Connecting last nodes from srcGroup " << srcGroup->groupId
                   << " to dstGroup " << dstGroup->groupId << "\n";
      bool pathsConnected = connectLastNodesToTarget(srcGroup->groupId, dstGroup->groupId);

      // If paths were successfully connected, don't insert srcGroup->dstGroup directly
      if (pathsConnected) {
        oldPlan.fusedGroup.to = srcGroup->groupId;
        oldPlan.fusedBand.to = depGraph.getNodeId(srcGroup->getLeadingFor());
        llvm::outs() << "Paths connected successfully, skipping direct srcGroup->dstGroup plan\n";
        return false;  // Don't insert fusePlan
      }

      // No paths found, set up direct fusion plan
      setupDirectFusionPlan(fusePlan, oldPlan, srcGroup, dstGroup);
      return true;  // Need to insert fusePlan
    }
  }
  return true;  // No multi-out conflict, insert fusePlan
}

// Applies loop transforms and fuses source group into target group.
// Records the fusion plan and updates group relationships.
void FusionAnalyzer::applyAndFuse(std::vector<LoopTransform> loopTransforms, const GroupPtr targetGroup,
                                  const GroupPtr sourceGroup) {
  for (auto lt : loopTransforms) {
    sourceGroup->nodeTransformRecords[sourceGroup->groupId].emplace_back(lt);
  }

  sourceGroup->fusedGroupId.emplace_back(targetGroup->groupId);

  for (auto fuseTargetId : targetGroup->nodesId) {
    finished.insert(fuseTargetId);
  }
  if (false /*succ*/) {
    for (auto fuseTargetId : sourceGroup->nodesId) {
      finished.insert(fuseTargetId);
    }
  }
  llvm::outs() << "Pre-fusion group " << sourceGroup->groupId << " to " << targetGroup->groupId << "\n";

  // Get the for node IDs of sourceGroup and targetGroup
  auto srcNodeId = depGraph.getNodeId(sourceGroup->getLeadingFor());
  auto dstNodeId = depGraph.getNodeId(targetGroup->getLeadingFor());
  FusionPlan fusePlan;
  fusePlan.fusedGroup = FuseEdge(sourceGroup->groupId, targetGroup->groupId);
  fusePlan.fusedBand = FuseEdge(srcNodeId, dstNodeId);

  bool shouldInsert = checkAndFixMultiOut(fusePlan);

  if (shouldInsert) {
    llvm::outs() << "Add Plan node " << fusePlan.fusedBand.from << " to " << fusePlan.fusedBand.to << "\n";
    fusionPlans.emplace_back(fusePlan);
  } else {
    llvm::outs() << "Skip inserting fusePlan (handled by intersection points)\n";
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
void FusionAnalyzer::getDirectlyPredecessors(unsigned id, std::vector<unsigned> &predecessorIds) {
  std::vector<unsigned> allPredecessorIds;

  depGraph.getPredecessorNodes(id, allPredecessorIds);
  // Sort in descending order to prioritize direct predecessors.
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
        llvm::outs() << "Skip " << id << " cause it has edge to " << allPredecessorIds[j] << "\n";
        skipIdx.insert(i);
        break;
      }
    }
  }
  for (size_t i = 0; i < allPredecessorIds.size(); ++i) {
    if (skipIdx.count(i)) {
      continue;
    }
    predecessorIds.emplace_back(allPredecessorIds[i]);
  }
}

// Checks if the fusion plan is complete.
// Returns true if all nodes have been processed.
bool FusionAnalyzer::finishPlan() { return finished.size() == depGraph.nodes.size(); }

void FusionAnalyzer::plan() {
  initGroups();

  topoSort();

  while (!finishPlan()) {
    auto targetGroup = getFusionTargetGroup();
    if (targetGroup == nullptr) {
      llvm::outs() << "No target group, finish\n";
      break;
    }

    std::vector<unsigned> sourceGroupIds;
    for (auto fuseTargetId : targetGroup->nodesId) {
      std::vector<unsigned> predecessorIds;
      getDirectlyPredecessors(fuseTargetId, predecessorIds);

      for (size_t i = 0; i < predecessorIds.size(); ++i) {
        auto id = predecessorIds[i];
        auto tmp = depGraph.getGroupByNode(id);
        if (tmp != nullptr && tmp->groupId != targetGroup->groupId) {
          if (!std::count(sourceGroupIds.begin(), sourceGroupIds.end(), tmp->groupId)) {
            llvm::outs() << "Fuse source group " << tmp->groupId << " (depends on node " << id
                         << ") into target group " << targetGroup->groupId
                         << " (target node " << fuseTargetId << ")\n";
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
      auto loopTransforms = inferLoopTransforms(targetGroup, sourceGroup);
      applyAndFuse(loopTransforms, targetGroup, sourceGroup);
    }
  }
  auto sortedPlan = topologicalSort(fusionPlans, depGraph.nodes.size());
  for (auto p : sortedPlan) {
    llvm::outs() << "Sort plan: node " << p.fusedBand.from << " to " << p.fusedBand.to << "\n";
  }
  fusionPlans = sortedPlan;
}

void FusionAnalyzer::reorderPlans(unsigned numNodes, unsigned srcId, unsigned dstId) {
  // Update plans that reference the destination node
  for (auto &plan : fusionPlans) {
    if (plan.fusedBand.from == dstId) {
      llvm::outs() << "  Updating plan: " << plan.fusedBand.from << "->" << plan.fusedBand.to
                   << " to " << srcId << "->" << plan.fusedBand.to << "\n";
      plan.fusedBand.from = srcId;
    }
    if (plan.fusedBand.to == dstId) {
      llvm::outs() << "  Updating plan: " << plan.fusedBand.from << "->" << plan.fusedBand.to
                   << " to " << plan.fusedBand.from << "->" << srcId << "\n";
      plan.fusedBand.to = srcId;
    }
  }

  // Remove the completed fusion plan (srcId->dstId)
  fusionPlans.erase(
    std::remove_if(fusionPlans.begin(), fusionPlans.end(),
      [srcId, dstId](const FusionPlan &plan) {
        return plan.fusedBand.from == srcId && plan.fusedBand.to == dstId;
      }),
    fusionPlans.end());


  // Re-sort the remaining plans
  if (!fusionPlans.empty()) {
    auto sortedPlan = topologicalSort(fusionPlans, numNodes);
    llvm::outs() << "Reordered fusion plans after fusion:\n";
    for (auto p : sortedPlan) {
      llvm::outs() << "Sort plan: node " << p.fusedBand.from << " to " << p.fusedBand.to << "\n";
    }
    fusionPlans = sortedPlan;
  }
}


void FusionAnalyzer::initGroups() { groups = depGraph.groups; }

// Performs topological sorting of groups for fusion analysis.
// Groups are sorted according to their priority and dependencies using GroupCmp,
// then all node IDs from sorted groups are collected into topoSortNodeIds.
void FusionAnalyzer::topoSort() {
  topoSortNodeIds.clear();
  std::vector<GroupPtr> allGroups;
  allGroups.reserve(groups.size());
  std::transform(groups.begin(), groups.end(), std::back_inserter(allGroups),
                 [](const auto &pair) { return pair.second; });
  std::sort(allGroups.begin(), allGroups.end(), GroupCmp);
  llvm::outs() << "topoSort: \n";
  for (auto g : allGroups) {
    std::string groupTemplateString = g->getGroupTemplateString();
    llvm::outs() << "group " << g->groupId << " groupTemplate " << groupTemplateString << "\n";
    for (auto node : g->nodesId) {
      topoSortNodeIds.push_back(node);
    }
  }
  llvm::outs() << "topoSortNodeIds: ";
  for (auto it : topoSortNodeIds) {
    llvm::outs() << it << ", ";
  }
  llvm::outs() << "\n";
  return;
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

}  // namespace akg
}  // namespace mlir
