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

#include "akg/Dialect/Affine/Analysis/AffineAnalysis.h"
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

static DepType classifyDepType(const Node *predNode, const Node *targetNode) {
  bool predIsStore = predNode != nullptr && isa<affine::AffineWriteOpInterface>(predNode->op);
  bool predIsLoad = predNode != nullptr && isa<affine::AffineReadOpInterface>(predNode->op);
  bool targetIsStore = targetNode != nullptr && isa<affine::AffineWriteOpInterface>(targetNode->op);
  bool targetIsLoad = targetNode != nullptr && isa<affine::AffineReadOpInterface>(targetNode->op);
  if (predIsStore && targetIsLoad) return DepType::WAR;
  if (predIsLoad && targetIsStore) return DepType::RAW;
  if (predIsStore && targetIsStore) return DepType::WAW;
  if (predIsLoad && targetIsLoad) return DepType::RAR;
  return DepType::OTHER;
}

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

// Deduplicates fusion plans by (fusedBand.from, fusedBand.to), removes self-loops,
// and calls setFusionPlanOptions on each unique edge.
std::vector<FusionPlan> FusionAnalyzer::deduplicateAndClassifyEdges() {
  std::set<std::pair<unsigned, unsigned>> seenEdges;
  std::vector<FusionPlan> edges;
  for (const auto &plan : fusionPlans) {
    if (plan.fusedBand.from == plan.fusedBand.to) {
      continue;
    }
    if (seenEdges.emplace(plan.fusedBand.from, plan.fusedBand.to).second) {
      edges.push_back(plan);
    }
  }
  for (auto &edge : edges) {
    setFusionPlanOptions(edge);
  }
  return edges;
}

// Emits outgoing edges for a single node during Kahn's traversal.
// Non-subview edges are emitted before subview edges; within each category,
// edges are sorted by descending (fusedBand.to, fusedBand.from).
void FusionAnalyzer::emitEdgesForNode(const std::vector<FusionPlan> &edges, const std::vector<size_t> &edgeIndices,
                                      std::vector<unsigned> &inDegree, std::vector<FusionPlan> &result,
                                      const std::function<void(unsigned)> &enqueue) {
  auto edgeCmpDesc = [](const FusionPlan &a, const FusionPlan &b) {
    if (a.fusedBand.to != b.fusedBand.to) return a.fusedBand.to > b.fusedBand.to;
    return a.fusedBand.from > b.fusedBand.from;
  };

  std::vector<FusionPlan> nonSubviewEdges, subviewEdges;
  for (size_t idx : edgeIndices) {
    (edges[idx].isSubviewFusion ? subviewEdges : nonSubviewEdges).push_back(edges[idx]);
  }
  std::sort(nonSubviewEdges.begin(), nonSubviewEdges.end(), edgeCmpDesc);
  std::sort(subviewEdges.begin(), subviewEdges.end(), edgeCmpDesc);

  for (auto *group : {&nonSubviewEdges, &subviewEdges}) {
    for (auto &edge : *group) {
      result.push_back(edge);
      if (--inDegree[edge.fusedBand.to] == 0) {
        enqueue(edge.fusedBand.to);
      }
    }
  }
}

// Topological sort for fusion plans (Kahn's algorithm with priority scheduling).
//
// Sorting rules (in descending priority):
//   1. Topological order: a node's outgoing edges are emitted only after all its
//      incoming edges have been emitted (standard Kahn invariant).
//   2. Node scheduling priority: among zero-in-degree nodes, nodes that have at
//      least one non-subview outgoing edge (normalQueue) are scheduled before
//      nodes whose outgoing edges are ALL subview fusions (subviewQueue).
//      Within each queue, larger band IDs are scheduled first (max-heap).
//   3. Edge emission order per node: for each scheduled node, non-subview edges
//      are emitted before subview edges. Within each category, edges are sorted
//      by descending fusedBand.to (ties broken by descending fusedBand.from).
std::vector<FusionPlan> FusionAnalyzer::topoSortFusionPlans(unsigned numNodes) {
  std::vector<FusionPlan> edges = deduplicateAndClassifyEdges();

  // Identify nodes that have at least one non-subview outgoing edge.
  std::unordered_set<unsigned> normalNodes;
  for (const auto &edge : edges) {
    if (!edge.isSubviewFusion) {
      normalNodes.insert(edge.fusedBand.from);
    }
  }

  // Build adjacency list and compute in-degrees.
  std::unordered_map<unsigned, std::vector<size_t>> adjacency;
  std::vector<unsigned> inDegree(numNodes + 1, 0);
  for (size_t i = 0; i < edges.size(); ++i) {
    adjacency[edges[i].fusedBand.from].push_back(i);
    inDegree[edges[i].fusedBand.to]++;
  }

  // Two max-heaps: normalQueue (nodes with non-subview out-edges, higher priority)
  // and subviewQueue (subview-only nodes, lower priority).
  std::priority_queue<unsigned> normalQueue, subviewQueue;
  auto enqueue = [&](unsigned nodeId) {
    if (normalNodes.count(nodeId)) {
      normalQueue.push(nodeId);
    } else {
      subviewQueue.push(nodeId);
    }
  };

  for (unsigned i = 0; i < numNodes; ++i) {
    if (inDegree[i] == 0) {
      enqueue(i);
    }
  }

  // Kahn's loop — schedule zero-in-degree nodes and emit their outgoing edges.
  std::vector<FusionPlan> result;
  result.reserve(edges.size());

  while (!normalQueue.empty() || !subviewQueue.empty()) {
    unsigned node = !normalQueue.empty() ? normalQueue.top() : subviewQueue.top();
    (!normalQueue.empty() ? normalQueue : subviewQueue).pop();

    auto adjIt = adjacency.find(node);
    if (adjIt == adjacency.end()) continue;

    emitEdgesForNode(edges, adjIt->second, inDegree, result, enqueue);
  }

  return result;
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
                                             unsigned newToBandId) {
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
  it->fusedBand.to = newToBandId;
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

// Checks whether a fusion plan represents a subview fusion.
bool FusionAnalyzer::checkSubviewFusion(unsigned predNodeId, unsigned targetNodeId) {
  if (predNodeId == UINT_MAX || targetNodeId == UINT_MAX) return false;
  Node *predNode = depGraph.getNode(predNodeId);
  Node *tgtNode = depGraph.getNode(targetNodeId);
  if (!predNode || !tgtNode) return false;

  auto getMemrefBase = [](Operation *op) -> std::pair<Value, bool> {
    Value memref;
    bool hasSubView = false;
    if (auto readOp = dyn_cast<affine::AffineReadOpInterface>(op)) {
      memref = readOp.getMemRef();
    } else if (auto writeOp = dyn_cast<affine::AffineWriteOpInterface>(op)) {
      memref = writeOp.getMemRef();
    } else {
      return {memref, hasSubView};
    }

    Value base = affine::getSourceMemRef(memref, &hasSubView);
    return {base, hasSubView};
  };

  auto [predBase, predSV] = getMemrefBase(predNode->op);
  auto [tgtBase, tgtSV] = getMemrefBase(tgtNode->op);
  if (!predBase || !tgtBase) return false;

  return predBase == tgtBase && (predSV || tgtSV);
}

// Determines fusion type: if the nodes that to depends on exist in fusionPlans and have edges (forward) to from,
// or if from itself is a node that to directly or indirectly depends on (in the dependency graph),
// then the type is H (provided the dependent node is not load), otherwise it is V
void FusionAnalyzer::setFusionPlanOptions(FusionPlan &plan) {
  // Infer loop transforms based on group templates (merged from inferLoopTransforms)
  auto sourceGroup = groups[plan.fusedGroup.from];
  auto targetGroup = groups[plan.fusedGroup.to];
  if (sourceGroup != nullptr && targetGroup != nullptr) {
    if (targetGroup->groupTemplate == OperatorTemplate::Broadcast) {
      plan.loopTransform = LoopTransform::Replicate;
    } else if (targetGroup->groupTemplate == OperatorTemplate::Transpose) {
      if (sourceGroup->groupTemplate == OperatorTemplate::Reshape) {
        plan.loopTransform = LoopTransform::StripMine;
      }
    }
  }

  unsigned fromGroupId = plan.fusedGroup.from;
  unsigned toGroupId = plan.fusedGroup.to;

  auto cacheKey = std::make_pair(fromGroupId, toGroupId);
  auto cacheIt = groupDependenciesCache.find(cacheKey);
  if (cacheIt != groupDependenciesCache.end()) {
    plan.depInfo = cacheIt->second;
  }
  plan.isSubviewFusion = checkSubviewFusion(plan.depInfo.predNodeId, plan.depInfo.targetNodeId);
  plan.fusionType = "V";

  auto toDepGroups = depGraph.getDependentGroups(toGroupId);
  if (toDepGroups.count(fromGroupId)) {
    plan.fusionType = "H";
    return;
  }

  for (auto depGroupId : toDepGroups) {
    if (hasEdgeInFusionPlans(depGroupId, fromGroupId)) {
      auto depCacheKey = std::make_pair(depGroupId, toGroupId);
      auto depCacheIt = groupDependenciesCache.find(depCacheKey);
      if (depCacheIt != groupDependenciesCache.end()) {
        plan.depInfo = depCacheIt->second;
      }
      plan.fusionType = "H";
      return;
    }
  }
}

std::pair<GroupPtr, GroupPtr> FusionAnalyzer::determineFusionOrder(const GroupPtr oldGroup, const GroupPtr newGroup) {
  if (oldGroup->groupTemplate == OperatorTemplate::Reduction) {
    return std::make_pair(newGroup, oldGroup);
  }
  if (newGroup->groupTemplate == OperatorTemplate::Reduction) {
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
  if (oldGroup->groupId < newGroup->groupId) {
    return std::make_pair(oldGroup, newGroup);
  }
  return std::make_pair(newGroup, oldGroup);
}

// Redirects shorter path to longer path's starting group.
// Modifies fusion plans to redirect paths ending at intersectionId to targetGroup.
void FusionAnalyzer::redirectFusionPlanToTarget(unsigned intersectionId,
                                                const std::unordered_map<unsigned, unsigned> &reachable,
                                                unsigned pathLen, GroupPtr targetGroup) {
  auto targetGroupId = targetGroup->groupId;
  auto targetBandId = targetGroup->rootId;
  for (auto it = fusionPlans.begin(); it != fusionPlans.end(); ++it) {
    if (it->fusedGroup.to != intersectionId) {
      continue;
    }

    auto fromIt = reachable.find(it->fusedGroup.from);
    if (fromIt != reachable.end() && fromIt->second == pathLen - 1) {
      // Update the plan using the encapsulated method
      updateFusionPlanByGroup(it->fusedGroup.from, intersectionId, targetGroupId, targetBandId);
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
// Finds whether there exists a backward intersection point between oldGroup and newGroup.
// On success writes the intersection (shortest total path and reachable sets) into result and returns true.
bool FusionAnalyzer::findBackwardIntersection(const GroupPtr oldGroup, const GroupPtr newGroup,
                                              BackwardIntersectionResult &result) {
  auto oldReachable = findReachableGroups(oldGroup->groupId);
  auto newReachable = findReachableGroups(newGroup->groupId);

  unsigned closestIntersectionId = 0;
  unsigned minTotalDistance = std::numeric_limits<unsigned>::max();
  unsigned minOldPathLen = std::numeric_limits<unsigned>::max();
  unsigned minNewPathLen = std::numeric_limits<unsigned>::max();
  bool foundIntersection = false;

  for (const auto &[groupId, oldPathLen] : oldReachable) {
    auto newIt = newReachable.find(groupId);
    if (newIt != newReachable.end()) {
      unsigned newPathLen = newIt->second;
      unsigned totalDistance = oldPathLen + newPathLen;
      if (totalDistance < minTotalDistance) {
        minTotalDistance = totalDistance;
        closestIntersectionId = groupId;
        minOldPathLen = oldPathLen;
        minNewPathLen = newPathLen;
        foundIntersection = true;
      }
    }
  }

  if (!foundIntersection) {
    return false;
  }
  result.closestIntersectionId = closestIntersectionId;
  result.minOldPathLen = minOldPathLen;
  result.minNewPathLen = minNewPathLen;
  result.oldReachable = std::move(oldReachable);
  result.newReachable = std::move(newReachable);
  return true;
}

// Example:
// Assume: newGroup has groupId=19 with fusion plan 19->29
//         oldGroup has groupId=19 with fusion plan 19->34
//         fusionPlans contains: 29->39, 34->39
//
// Fusion result:
// 19->29, 29->34, 34->39
GroupPtr FusionAnalyzer::handleBackwardIntersectionPoints(const GroupPtr oldGroup, const GroupPtr newGroup,
                                                          const BackwardIntersectionResult &intersection) {
  unsigned closestIntersectionId = intersection.closestIntersectionId;
  unsigned minOldPathLen = intersection.minOldPathLen;
  unsigned minNewPathLen = intersection.minNewPathLen;
  const auto &oldReachable = intersection.oldReachable;
  const auto &newReachable = intersection.newReachable;

  // Check cache: normalize cache key to handle reversed (oldGroup, newGroup) order
  unsigned firstGroupId = std::min(oldGroup->groupId, newGroup->groupId);
  unsigned secondGroupId = std::max(oldGroup->groupId, newGroup->groupId);
  auto cacheKey = std::make_tuple(firstGroupId, secondGroupId, closestIntersectionId);
  auto cacheIt = intersectionCache.find(cacheKey);
  if (cacheIt != intersectionCache.end()) {
    auto cachedTargetGroup = depGraph.getGroup(cacheIt->second);
    if (cachedTargetGroup != nullptr) {
      return cachedTargetGroup;
    }
  }

  // Determine which group has shorter path and redirect accordingly
  GroupPtr targetGroup;
  if (minOldPathLen < minNewPathLen) {
    if (minOldPathLen == 0) {
      targetGroup = newGroup;
    } else {
      redirectFusionPlanToTarget(closestIntersectionId, oldReachable, minOldPathLen, newGroup);
      targetGroup = oldGroup;
    }
  } else {
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

  // Link all source groups to destination group, but skip oldPlan which was
  // intentionally updated to point to srcGroup in the step above.
  // Collect plans to update first to avoid iterator invalidation
  std::vector<std::pair<unsigned, unsigned>> plansToUpdate;
  for (const auto &plan : fusionPlans) {
    if (plan.fusedGroup.to == srcGroup->groupId &&
        !(plan.fusedGroup.from == oldPlan.fusedGroup.from && plan.fusedGroup.to == oldPlan.fusedGroup.to)) {
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
C->D->E, now adding A->C, then update fusion plan to A->B->C->D->E
   2) B and C have no backward intersection points, fuse the new fuseplan into the old fuseplan

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
      BackwardIntersectionResult intersection;
      bool foundIntersection = findBackwardIntersection(oldGroup, newGroup, intersection);
      if (foundIntersection) {
        GroupPtr targetGroup = handleBackwardIntersectionPoints(oldGroup, newGroup, intersection);
        if (targetGroup != nullptr) {
          // Fuse the shorter path to the longer one based on distance to intersection, update the shorter fusion path
          // Example: A->B->E, C->D->E, update to A->B->C->D->E
          updateFusionPlanByGroup(oldPlan.fusedGroup.from, oldPlan.fusedGroup.to, targetGroup->groupId,
                                  targetGroup->rootId);
          return false;
        }
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

// Applies loop transforms and fuses source group into target group.
// Records the fusion plan and updates group relationships.
void FusionAnalyzer::applyAndFuse(const GroupPtr targetGroup, const GroupPtr sourceGroup) {
  sourceGroup->fusedGroupId.emplace_back(targetGroup->groupId);

  FusionPlan fusePlan;
  fusePlan.fusedGroup = FuseEdge(sourceGroup->groupId, targetGroup->groupId);
  fusePlan.fusedBand = FuseEdge(sourceGroup->rootId, targetGroup->rootId);

  bool shouldInsert = checkAndFixMultiOut(fusePlan);
  if (shouldInsert) {
    addFusionPlan(fusePlan);
  }
}

void FusionAnalyzer::dumpDirectPredecessors(unsigned nodeId, const std::vector<unsigned> &allPredecessorIds,
                                            const std::unordered_set<size_t> &skipIdx,
                                            const std::vector<DirectPredecessor> &directPreds) {
  auto printFlags = mlir::OpPrintingFlags().skipRegions();
  auto *targetNode = depGraph.getNode(nodeId);

  llvm::dbgs() << "\n=== DirectPredecessors for nodeId: " << nodeId << " ===\n";
  llvm::dbgs() << "  Target op: ";
  if (targetNode) targetNode->op->print(llvm::dbgs(), printFlags);
  llvm::dbgs() << "\n";

  llvm::dbgs() << "  All predecessors (sorted desc, total=" << allPredecessorIds.size() << "):\n";
  for (size_t i = 0; i < allPredecessorIds.size(); ++i) {
    auto predId = allPredecessorIds[i];
    auto *predNode = depGraph.getNode(predId);
    bool skipped = skipIdx.count(i) > 0;
    llvm::dbgs() << "    [" << i << "] ID: " << predId << (skipped ? " (SKIPPED)" : " (DIRECT)") << ", OP: ";
    if (predNode) predNode->op->print(llvm::dbgs(), printFlags);
    llvm::dbgs() << "\n";
  }

  llvm::dbgs() << "  Direct predecessor edges (total=" << directPreds.size() << "):\n";
  for (const auto &pred : directPreds) {
    auto *predNode = depGraph.getNode(pred.nodeId);
    llvm::dbgs() << "    ID: " << pred.nodeId << ", MEMREF: " << pred.memref << ", DEPTH: " << pred.loopDepth
                 << ", DepType: " << depTypeToString(pred.depType) << ", OP: ";
    if (predNode) predNode->op->print(llvm::dbgs(), printFlags);
    llvm::dbgs() << "\n";
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
    for (size_t i = 1; i < allPredecessorIds.size(); ++i) {
      if (skipIdx.count(i)) {
        continue;
      }
      auto iCandidateID = allPredecessorIds[i];
      // Predecessor nodes are sorted in descending order, so j < i means checking earlier nodes
      for (size_t j = 0; j < i; ++j) {
        if (skipIdx.count(j)) {
          continue;
        }
        auto jCandidateID = allPredecessorIds[j];
        // Check if there exists an edge from iCandidateID to jCandidateID
        // If edge exists, iCandidateID is not a direct predecessor but an indirect one, add to skipIdx
        if (depGraph.hasEdge(iCandidateID, jCandidateID, mlir::Value())) {
          skipIdx.insert(i);
          break;
        }
      }
    }

    // Build direct predecessors list with memrefs and loopDepths
    auto *targetNode = depGraph.getNode(nodeId);
    std::vector<DirectPredecessor> directPreds;
    for (size_t i = 0; i < allPredecessorIds.size(); ++i) {
      if (skipIdx.count(i)) {
        continue;
      }
      auto predId = allPredecessorIds[i];
      auto *predNode = depGraph.getNode(predId);
      DepType depType = classifyDepType(predNode, targetNode);
      const auto &memrefsAndDepths = predToMemrefsAndDepth[predId];
      std::transform(memrefsAndDepths.begin(), memrefsAndDepths.end(), std::back_inserter(directPreds),
                     [predId, depType](const auto &memrefAndDepth) {
                       return DirectPredecessor(predId, memrefAndDepth.first, memrefAndDepth.second, depType);
                     });
    }

    directPredecessorsCache.emplace(nodeId, std::move(directPreds));
  }
}

void FusionAnalyzer::collectFusionSourceGroups(const GroupPtr &targetGroup, std::unordered_set<unsigned> &warGroupIds,
                                               std::unordered_set<unsigned> &rarGroupIds) {
  // Accumulate DependenceInfo per source group to pre-populate groupDependenciesCache.
  struct DepAccum {
    DependenceInfo warDepInfo;
    DependenceInfo rarDepInfo;
    bool haswarDep = false;
  };
  std::unordered_map<unsigned, DepAccum> depAccumMap;
  for (auto targetNodeId : targetGroup->nodesId) {
    auto it = directPredecessorsCache.find(targetNodeId);
    if (it == directPredecessorsCache.end()) {
      continue;
    }
    for (const auto &pred : it->second) {
      auto sourceGroup = depGraph.getGroupByNode(pred.nodeId);
      if (sourceGroup == nullptr || sourceGroup->groupId == targetGroup->groupId) {
        continue;
      }
      unsigned sourceGroupId = sourceGroup->groupId;
      auto &accum = depAccumMap[sourceGroupId];
      bool isRAR = (pred.depType == DepType::RAR);
      auto &depInfo = isRAR ? accum.rarDepInfo : accum.warDepInfo;
      depInfo.loopDepth = std::min(depInfo.loopDepth, pred.loopDepth);
      depInfo.memref = pred.memref;
      depInfo.predNodeId = pred.nodeId;
      depInfo.targetNodeId = targetNodeId;
      depInfo.depType = pred.depType;
      if (isRAR) {
        rarGroupIds.insert(sourceGroupId);
      } else {
        warGroupIds.insert(sourceGroupId);
        accum.haswarDep = true;
      }
    }
  }

  // Pre-populate groupDependenciesCache
  for (auto &[sourceGroupId, accum] : depAccumMap) {
    auto cacheKey = std::make_pair(sourceGroupId, targetGroup->groupId);
    if (groupDependenciesCache.find(cacheKey) == groupDependenciesCache.end()) {
      DependenceInfo depInfo = accum.haswarDep ? std::move(accum.warDepInfo) : std::move(accum.rarDepInfo);
      groupDependenciesCache[cacheKey] = std::move(depInfo);
    }
  }
}

void FusionAnalyzer::dumpCollectFusionSourceInfo(const GroupPtr &targetGroup, const char *dependenceType,
                                                 const std::unordered_set<unsigned> &sourceGroupIds) {
  if (sourceGroupIds.empty()) return;
  auto printFlags = mlir::OpPrintingFlags().skipRegions();

  llvm::dbgs() << "\n=== collectFusionSourceGroups for targetGroup: " << targetGroup->groupId << ", " << dependenceType
               << " ===\n";
  for (unsigned sourceGroupId : sourceGroupIds) {
    llvm::dbgs() << "  sourceGroupId=" << sourceGroupId << "\n";
    auto cacheKey = std::make_pair(sourceGroupId, targetGroup->groupId);
    auto cacheIt = groupDependenciesCache.find(cacheKey);
    if (cacheIt == groupDependenciesCache.end()) continue;
    const auto &depInfo = cacheIt->second;
    auto *targetNode = depGraph.getNode(depInfo.targetNodeId);
    auto *predNode = depGraph.getNode(depInfo.predNodeId);
    llvm::dbgs() << "    predNodeId: " << depInfo.predNodeId << " (OP: ";
    if (predNode) predNode->op->print(llvm::dbgs(), printFlags);
    llvm::dbgs() << ") -> targetNodeId: " << depInfo.targetNodeId << " (OP: ";
    if (targetNode) targetNode->op->print(llvm::dbgs(), printFlags);
    llvm::dbgs() << ")\n";
  }
}

void FusionAnalyzer::plan() {
  initGroups();
  topoSortInit();
  precomputeDirectPredecessors();

  std::unordered_map<unsigned, std::unordered_set<unsigned>> rarMap;
  while (!finishPlan()) {
    auto targetGroup = getFusionTargetGroup();
    if (targetGroup == nullptr) {
      break;
    }

    std::unordered_set<unsigned> warGroupIds;
    std::unordered_set<unsigned> rarGroupIds;
    collectFusionSourceGroups(targetGroup, warGroupIds, rarGroupIds);
    rarMap[targetGroup->groupId] = rarGroupIds;

    // Phase 1: Process store-load (producer-consumer) edges only.
    // RAR edges are deferred to Phase 2 to avoid connecting unrelated chains.
    for (unsigned sourceGroupId : warGroupIds) {
      auto sourceGroup = depGraph.getGroup(sourceGroupId);
      if (sourceGroup != nullptr) {
        applyAndFuse(targetGroup, sourceGroup);
      }
    }

    for (auto targetId : targetGroup->nodesId) {
      finished.insert(targetId);
    }
  }

  // Phase 2: Process RAR (load-load) edges with backward intersection checking.
  // Only connect RAR pairs that are not already reachable through Phase 1 plans.
  for (const auto &[targetGroupId, rarGroupIds] : rarMap) {
    auto targetGroup = depGraph.getGroup(targetGroupId);
    if (targetGroup == nullptr) {
      continue;
    }

    for (unsigned sourceGroupId : rarGroupIds) {
      auto sourceGroup = depGraph.getGroup(sourceGroupId);
      if (sourceGroup == nullptr) {
        continue;
      }

      BackwardIntersectionResult intersection;
      if (findBackwardIntersection(sourceGroup, targetGroup, intersection)) {
        continue;
      }
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
       << "LoopDepth: " << plan.depInfo.loopDepth << ", "
       << "LoopTransform: " << loopTransformToStr[static_cast<int>(plan.loopTransform)] << ", "
       << "IsSubviewFusion: " << (plan.isSubviewFusion ? "true" : "false") << ", "
       << "FusionNodeRecord: (" << plan.depInfo.predNodeId << " -> " << plan.depInfo.targetNodeId << ")\n";
  }
}

}  // namespace akg
}  // namespace mlir
