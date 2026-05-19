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

// adjacency:     fusedGroup.from -> outgoing edge indices (also indexes softDefer lookup).
// normalNodes:   groups that are the source of at least one non-subview edge.
// inDegree:      group -> incoming edge count, sized to cover any group ID.
// pendingPriors: edgeIdx -> set of edge indices that must emit before it.
void TopoScheduler::buildSchedulingState() {
  unsigned maxGroupId = 0;
  for (size_t i = 0; i < edges.size(); ++i) {
    const auto &edge = edges[i];
    adjacency[edge.fusedGroup.from].push_back(i);
    if (edge.depInfo.memrefKind != MemrefKind::Subview) normalNodes.insert(edge.fusedGroup.from);
    maxGroupId = std::max({maxGroupId, edge.fusedGroup.from, edge.fusedGroup.to});
  }
  inDegree.assign(std::max<size_t>(numNodes, maxGroupId) + 1, 0);
  for (const auto &edge : edges) {
    inDegree[edge.fusedGroup.to]++;
  }
  for (const auto &[deferredGroupId, mustFirstGroupId] : softDeferConstraints) {
    auto dIt = adjacency.find(deferredGroupId);
    auto mIt = adjacency.find(mustFirstGroupId);
    if (dIt == adjacency.end() || mIt == adjacency.end()) continue;
    for (size_t dIdx : dIt->second) {
      for (size_t mIdx : mIt->second) {
        if (dIdx == mIdx) continue;
        pendingPriors[dIdx].insert(mIdx);
      }
    }
  }
}

// A deferred edge is ready once every prior in pendingPriors is emitted.
bool TopoScheduler::priorsSatisfied(size_t edgeIdx) const {
  auto it = pendingPriors.find(edgeIdx);
  if (it == pendingPriors.end()) return true;
  for (size_t priorIdx : it->second) {
    if (!emittedEdges.count(priorIdx)) return false;
  }
  return true;
}

// Append the edge to result and cascade the in-degree decrement: if the
// target group hits zero, queue it for processing.
void TopoScheduler::commitEdge(size_t edgeIdx) {
  const auto &edge = edges[edgeIdx];
  result.push_back(edge);
  emittedEdges.insert(edgeIdx);
  unsigned toGroup = edge.fusedGroup.to;
  if (toGroup < inDegree.size() && inDegree[toGroup] > 0 && --inDegree[toGroup] == 0) {
    auto &q = normalNodes.count(toGroup) ? normalQueue : subviewQueue;
    q.push(toGroup);
  }
}

void TopoScheduler::tryEmit(size_t edgeIdx) {
  if (!priorsSatisfied(edgeIdx)) {
    pendingDeferred.push_back(edgeIdx);
    return;
  }
  commitEdge(edgeIdx);
}

// Scan deferred list to fixed-point: each newly-emitted edge can satisfy
// downstream priors, so retry repeatedly until nothing changes.
void TopoScheduler::releaseDeferred() {
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto it = pendingDeferred.begin(); it != pendingDeferred.end();) {
      if (priorsSatisfied(*it)) {
        commitEdge(*it);
        it = pendingDeferred.erase(it);
        changed = true;
      } else {
        ++it;
      }
    }
  }
}

// Per-node emission: split outgoing edges into non-subview / subview buckets,
// sort each by descending (to, from), and hand to tryEmit in order.
void TopoScheduler::processNode(unsigned node) {
  auto adjIt = adjacency.find(node);
  if (adjIt == adjacency.end()) return;
  std::vector<size_t> nonSubview, subview;
  for (size_t idx : adjIt->second) {
    (edges[idx].depInfo.memrefKind == MemrefKind::Subview ? subview : nonSubview).push_back(idx);
  }
  auto cmpDesc = [this](size_t a, size_t b) {
    if (edges[a].fusedGroup.to != edges[b].fusedGroup.to) {
      return edges[a].fusedGroup.to > edges[b].fusedGroup.to;
    }
    return edges[a].fusedGroup.from > edges[b].fusedGroup.from;
  };
  std::sort(nonSubview.begin(), nonSubview.end(), cmpDesc);
  std::sort(subview.begin(), subview.end(), cmpDesc);
  for (size_t idx : nonSubview) tryEmit(idx);
  for (size_t idx : subview) tryEmit(idx);
}

// 1. Build scheduling state (adjacency, in-degree, normalNodes, soft-defer priors).
// 2. Seed ready queues with zero-in-degree groups (normalQueue > subviewQueue; max-heap by group ID).
// 3. Main loop: Kahn propagation until queues drain, then releaseDeferred; if still stalled,
//    force-emit one deferred edge. Repeat until queues and deferred list are both empty.
//    e.g. chains 4→17→18→21 and 11→12→19→20→21, softDefer(4,11) ⇒ 18 before 20.
std::vector<FusionPlan> TopoScheduler::run() {
  buildSchedulingState();
  result.reserve(edges.size());
  for (unsigned i = 0; i < inDegree.size(); ++i) {
    if (inDegree[i] == 0) {
      auto &q = normalNodes.count(i) ? normalQueue : subviewQueue;
      q.push(i);
    }
  }
  while (hasReadyNodes() || !pendingDeferred.empty()) {
    while (hasReadyNodes()) {
      auto &q = !normalQueue.empty() ? normalQueue : subviewQueue;
      unsigned node = q.top();
      q.pop();
      processNode(node);
      releaseDeferred();
    }
    releaseDeferred();
    if (hasReadyNodes()) continue;
    if (pendingDeferred.empty()) continue;
    size_t idx = pendingDeferred.back();
    pendingDeferred.pop_back();
    commitEdge(idx);
  }
  return std::move(result);
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
  for (const auto &[groupId, _] : reachable) {
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

// Bridges two branches: the side with the smaller min-last-node id is wrapped
// inside the other (bridge edge: min-last → opposite anchor).
// Returns true if a real downstream path existed on either side.
// dstSideWon: true ⇒ dst wrapped inside src (bridge: min-last-of-dst → srcGroupId);
//             false ⇒ src wrapped inside dst (bridge: min-last-of-src → dstGroupId).
bool FusionAnalyzer::connectLastNodesToTarget(unsigned srcGroupId, unsigned dstGroupId, bool *dstSideWon) {
  auto lastNodesSrc = findLastNodesInPath(srcGroupId);
  auto lastNodesDst = findLastNodesInPath(dstGroupId);

  bool hasRealSrc =
    std::any_of(lastNodesSrc.begin(), lastNodesSrc.end(), [srcGroupId](unsigned id) { return id != srcGroupId; });
  bool hasRealDst =
    std::any_of(lastNodesDst.begin(), lastNodesDst.end(), [dstGroupId](unsigned id) { return id != dstGroupId; });
  bool hasRealPaths = hasRealSrc || hasRealDst;

  if (lastNodesSrc.empty()) lastNodesSrc.push_back(srcGroupId);
  if (lastNodesDst.empty()) lastNodesDst.push_back(dstGroupId);

  unsigned minLastSrc = *std::min_element(lastNodesSrc.begin(), lastNodesSrc.end());
  unsigned minLastDst = *std::min_element(lastNodesDst.begin(), lastNodesDst.end());

  unsigned bridgeFromId;
  unsigned bridgeToId;
  bool dstWon;
  if (minLastDst < minLastSrc) {
    bridgeFromId = minLastDst;
    bridgeToId = srcGroupId;
    dstWon = true;
  } else {
    bridgeFromId = minLastSrc;
    bridgeToId = dstGroupId;
    dstWon = false;
  }
  if (dstSideWon) *dstSideWon = dstWon;

  if (bridgeFromId != bridgeToId) {
    auto bridgeFromGroup = depGraph.getGroup(bridgeFromId);
    auto bridgeToGroup = depGraph.getGroup(bridgeToId);
    if (bridgeFromGroup != nullptr && bridgeToGroup != nullptr) {
      FusionPlan bridge;
      bridge.fusedGroup = FuseEdge(bridgeFromId, bridgeToId);
      bridge.fusedBand = FuseEdge(bridgeFromGroup->rootId, bridgeToGroup->rootId);
      auto oldCacheKey = std::make_pair(srcGroupId, dstGroupId);
      auto newCacheKey = std::make_pair(bridgeFromId, bridgeToId);
      if (groupDependenciesCache.find(oldCacheKey) != groupDependenciesCache.end() &&
          groupDependenciesCache.find(newCacheKey) == groupDependenciesCache.end()) {
        groupDependenciesCache[newCacheKey] = groupDependenciesCache[oldCacheKey];
      }
      addFusionPlan(bridge);
    }
  }

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

  auto updateDepInfoFromCache = [&](unsigned srcId, unsigned dstId) {
    auto key = std::make_pair(srcId, dstId);
    auto it = groupDependenciesCache.find(key);
    if (it != groupDependenciesCache.end()) {
      plan.depInfo = it->second;
    }
  };

  updateDepInfoFromCache(fromGroupId, toGroupId);
  // If pred.memref's chain already had a subview, the cached classification covers it;
  // otherwise fall back to the op-level base/subview check across both pred and target.
  if (plan.depInfo.memrefKind != MemrefKind::Subview &&
      checkSubviewFusion(plan.depInfo.predNodeId, plan.depInfo.targetNodeId)) {
    plan.depInfo.memrefKind = MemrefKind::Subview;
  }
  plan.fusionType = "V";

  auto toDepGroups = depGraph.getDependentGroups(toGroupId);
  if (toDepGroups.count(fromGroupId)) {
    plan.fusionType = "H";
    return;
  }

  for (auto depGroupId : toDepGroups) {
    if (hasEdgeInFusionPlans(depGroupId, fromGroupId)) {
      updateDepInfoFromCache(depGroupId, toGroupId);
      plan.fusionType = "H";
      return;
    }
  }
}

std::pair<GroupPtr, GroupPtr> FusionAnalyzer::determineFusionOrder(const GroupPtr oldGroup, const GroupPtr newGroup) {
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

// Returns true if oldGroup and newGroup are two distinct chains that converge
// at some downstream group.
bool FusionAnalyzer::findBackwardIntersection(const GroupPtr oldGroup, const GroupPtr newGroup, bool *isAncestry) {
  auto oldReachable = findReachableGroups(oldGroup->groupId);
  auto newReachable = findReachableGroups(newGroup->groupId);
  bool ancestry = oldReachable.count(newGroup->groupId) || newReachable.count(oldGroup->groupId);
  if (isAncestry) *isAncestry = ancestry;
  if (ancestry) return false;
  for (const auto &[groupId, _] : oldReachable) {
    if (newReachable.count(groupId)) return true;
  }
  return false;
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
}

/*
Logic for adding to fusionPlans:
1. If fusionPlans contains the same from and to, return false directly
2. If fusionPlans does not contain the same from, add it to fusionPlans directly
3. If fusionPlans contains the same from, i.e., there exists A->B,
   and currently A->C, i.e., a node needs to fuse to multiple nodes, which is not allowed,
   therefore the fusion plan needs to be updated:
   1) B and C have backward intersection: keep the smaller-groupId target edge, delete the
      larger-groupId one. Add softDefer(deferred=larger, mustFirst=smaller) so TopoScheduler
      emits the kept side first.
      e.g. 9->14 exists, adding 9->10; 14 and 10 backward-intersect ⇒ keep 9->10, drop 9->14,
      softDefer(14, 10).
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
      bool isAncestry = false;
      if (findBackwardIntersection(oldGroup, newGroup, &isAncestry)) {
        // Keep smaller-groupId target edge, delete larger-groupId one; softDefer(deferred=larger, mustFirst=smaller).
        // deferred = larger-groupId side, mustFirst = smaller-groupId side.
        if (newGroup->groupId < oldGroup->groupId) {
          // newGroup smaller — drop the existing A->oldGroup edge; caller will add A->newGroup.
          softDeferConstraints.emplace_back(oldGroup->groupId, newGroup->groupId);
          fusionPlans.erase(it);
          return true;
        } else {
          // oldGroup smaller (or equal) — keep the existing edge; reject the new plan.
          softDeferConstraints.emplace_back(newGroup->groupId, oldGroup->groupId);
          return false;
        }
      }

      if (isAncestry) {
        return false;
      }

      // No backward intersection points, fuse the new fuseplan into the old fuseplan
      auto [srcGroup, dstGroup] = determineFusionOrder(oldGroup, newGroup);
      auto oldCacheKey = std::make_pair(fusePlan.fusedGroup.from, fusePlan.fusedGroup.to);
      auto newCacheKey = std::make_pair(srcGroup->groupId, dstGroup->groupId);
      if (groupDependenciesCache.find(oldCacheKey) != groupDependenciesCache.end() &&
          groupDependenciesCache.find(newCacheKey) == groupDependenciesCache.end()) {
        groupDependenciesCache[newCacheKey] = groupDependenciesCache[oldCacheKey];
      }

      // Bridge: smaller min-last-node side is wrapped inside the other.
      // dstSideWon=true ⇒ dst wrapped inside src, redirect oldPlan's `to` to dstGroup; otherwise stays on srcGroup.
      bool dstSideWon = false;
      bool pathsConnected = connectLastNodesToTarget(srcGroup->groupId, dstGroup->groupId, &dstSideWon);
      if (pathsConnected) {
        unsigned newToId = dstSideWon ? dstGroup->groupId : srcGroup->groupId;
        unsigned newToBandId = dstSideWon ? dstGroup->rootId : srcGroup->rootId;
        updateFusionPlanByGroup(oldPlan.fusedGroup.from, oldPlan.fusedGroup.to, newToId, newToBandId);
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

// Conflict iff the two pairs share a convergence and their sources enter it
// via different branches (opposite emit orders → cycle).
bool FusionAnalyzer::isConflictingPair(const std::pair<unsigned, unsigned> &a, const std::pair<unsigned, unsigned> &b) {
  auto [aSrc, aTgt] = a;
  auto [bSrc, bTgt] = b;
  if (aSrc == bSrc || aTgt == bTgt) return false;

  auto firstConvergence = [this](unsigned x, unsigned y) -> unsigned {
    auto reachX = findReachableGroups(x);
    auto reachY = findReachableGroups(y);
    unsigned best = UINT_MAX;
    unsigned bestSum = UINT_MAX;
    for (const auto &[gid, lenX] : reachX) {
      auto it = reachY.find(gid);
      if (it == reachY.end()) continue;
      unsigned sum = lenX + it->second;
      if (sum < bestSum) {
        bestSum = sum;
        best = gid;
      }
    }
    return best;
  };

  unsigned ca = firstConvergence(aSrc, aTgt);
  unsigned cb = firstConvergence(bSrc, bTgt);
  if (ca == UINT_MAX || ca != cb) return false;

  // Walk g's linear fusion chain to find the immediate predecessor of conv,
  // i.e., the branch of conv that g enters through.
  auto sideOf = [this](unsigned g, unsigned conv) -> unsigned {
    unsigned cur = g;
    std::unordered_set<unsigned> seen{cur};
    while (true) {
      unsigned next = outgoingTarget(cur);
      if (next == conv) return cur;
      if (next == UINT_MAX || !seen.insert(next).second) return UINT_MAX;
      cur = next;
    }
  };

  unsigned sideA = sideOf(aSrc, ca);
  unsigned sideB = sideOf(bSrc, ca);
  return sideA != UINT_MAX && sideB != UINT_MAX && sideA != sideB;
}

unsigned FusionAnalyzer::outgoingTarget(unsigned id) {
  auto it =
    std::find_if(fusionPlans.begin(), fusionPlans.end(), [id](const FusionPlan &p) { return p.fusedGroup.from == id; });
  return (it == fusionPlans.end()) ? UINT_MAX : it->fusedGroup.to;
}

std::pair<unsigned, unsigned> FusionAnalyzer::findBridgePoint(unsigned sourceId, unsigned targetId) {
  unsigned cur = sourceId;
  std::unordered_set<unsigned> seen{cur};
  while (true) {
    unsigned next = outgoingTarget(cur);
    if (next == UINT_MAX || next == targetId || !seen.insert(next).second) {
      return {UINT_MAX, UINT_MAX};
    }
    if (next > targetId) return {cur, next};
    cur = next;
  }
}

void FusionAnalyzer::propagateDeletedDep(unsigned existingTo, unsigned targetId, DependenceInfo &deletedDep) {
  if (deletedDep.predNodeId == UINT_MAX) return;

  auto firstReachable = [&](unsigned start, const std::unordered_map<unsigned, unsigned> &reach) {
    unsigned cur = start;
    std::unordered_set<unsigned> seen;
    while (cur != UINT_MAX && seen.insert(cur).second) {
      if (reach.count(cur)) return cur;
      cur = outgoingTarget(cur);
    }
    return UINT_MAX;
  };
  auto findChainPredecessor = [&](unsigned start, unsigned end) {
    unsigned cur = start;
    std::unordered_set<unsigned> seen{cur};
    while (true) {
      unsigned next = outgoingTarget(cur);
      if (next == UINT_MAX) return UINT_MAX;
      if (next == end) return cur;
      if (!seen.insert(next).second) return UINT_MAX;
      cur = next;
    }
  };

  auto targetReach = findReachableGroups(targetId);
  unsigned convergenceId = firstReachable(existingTo, targetReach);
  if (convergenceId == UINT_MAX) return;

  unsigned recipientFrom = findChainPredecessor(targetId, convergenceId);
  if (recipientFrom == UINT_MAX) return;

  auto recipientKey = std::make_pair(recipientFrom, convergenceId);
  auto it = groupDependenciesCache.find(recipientKey);
  if (it == groupDependenciesCache.end() || deletedDep.loopDepth < it->second.loopDepth) {
    groupDependenciesCache[recipientKey] = std::move(deletedDep);
  }
}

// Chain-rewrite an RAR backward-intersection candidate (source, target):
// walk source's chain, find first edge X→Y where Y > target, redirect X→Y to X→target.
// Propagate deleted X→Y dep info to the recipient edge at the convergence point,
// and inherit (source, target) RAR cache for the new bridge edge.
//
// Example (13,16) : source chain 13→14→26→27→28, target chain 16→25→28.
//   (1) bridgeFromId=14, existingTo=26 (26>16)
//   (2) snapshot cache[(14,26)] as deletedDep, remove edge 14→26
//   (3) convergence Z=28 (first node in 26's chain reachable from 16),
//       recipient 25→28, overwrite cache[(25,28)] if deletedDep.loopDepth is smaller
//   (4) inherit cache[(13,16)] → cache[(14,16)]
//   (5) add bridge FusionPlan 14→16
void FusionAnalyzer::bridgeChainToTarget(std::pair<unsigned, unsigned> hEdge) {
  auto [sourceId, targetId] = hEdge;
  auto sourceGrp = depGraph.getGroup(sourceId);
  auto targetGrp = depGraph.getGroup(targetId);
  if (sourceGrp == nullptr || targetGrp == nullptr) return;

  // (1) Find bridge point: first edge X→Y along source's chain where Y > target.
  //     (13,16): 13→14 (14<16, continue) → 14→26 (26>16, bridge!) → bridgeFromId=14, existingTo=26
  auto [bridgeFromId, existingTo] = findBridgePoint(sourceId, targetId);
  if (bridgeFromId == UINT_MAX) return;
  auto bridgeFromGrp = depGraph.getGroup(bridgeFromId);
  if (bridgeFromGrp == nullptr) return;

  // (2) Snapshot X→Y's cached dep info, then remove the edge.
  //     (13,16): deletedDep = cache[(14,26)], remove edge 14→26
  DependenceInfo deletedDep;
  if (auto it = groupDependenciesCache.find(std::make_pair(bridgeFromId, existingTo));
      it != groupDependenciesCache.end()) {
    deletedDep = it->second;
  }
  fusionPlans.erase(std::remove_if(fusionPlans.begin(), fusionPlans.end(),
                                   [&](const FusionPlan &p) {
                                     return p.fusedGroup.from == bridgeFromId && p.fusedGroup.to == existingTo;
                                   }),
                    fusionPlans.end());

  // (3) Propagate deletedDep to the recipient edge at convergence Z.
  //     (13,16): overwrite cache[(25,28)] if deletedDep.loopDepth is smaller
  propagateDeletedDep(existingTo, targetId, deletedDep);

  // (4) Inherit (source, target) RAR cache for the new bridge edge if not already present.
  //     (13,16): cache[(13,16)] → cache[(14,16)]
  auto bridgeCacheKey = std::make_pair(bridgeFromId, targetId);
  if (auto srcIt = groupDependenciesCache.find(std::make_pair(sourceId, targetId));
      srcIt != groupDependenciesCache.end() &&
      groupDependenciesCache.find(bridgeCacheKey) == groupDependenciesCache.end()) {
    groupDependenciesCache[bridgeCacheKey] = srcIt->second;
  }

  // (5) Add the bridge FusionPlan bridgeFromId → target.
  //     (13,16): add FusionPlan 14→16
  FusionPlan bridge;
  bridge.fusedGroup = FuseEdge(bridgeFromId, targetId);
  bridge.fusedBand = FuseEdge(bridgeFromGrp->rootId, targetGrp->rootId);
  bridgeFromGrp->fusedGroupId.emplace_back(targetId);
  addFusionPlan(bridge);
}

// Resolve conflicting RAR backward-intersection candidates: repeatedly find a
// conflicting pair, rewrite the one with larger source groupId as an H fusion edge,
// and re-scan (each rewrite mutates fusionPlans, which may dissolve/expose conflicts).
// Survivors → softDeferConstraints.
void FusionAnalyzer::resolveConflictingDefers(std::vector<std::pair<unsigned, unsigned>> &candidates) {
  std::vector<bool> rewritten(candidates.size(), false);

  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < candidates.size() && !changed; ++i) {
      if (rewritten[i]) continue;
      for (size_t j = i + 1; j < candidates.size(); ++j) {
        if (rewritten[j]) continue;
        if (!isConflictingPair(candidates[i], candidates[j])) continue;
        size_t pickIdx = (candidates[i].first > candidates[j].first) ? i : j;
        bridgeChainToTarget(candidates[pickIdx]);
        rewritten[pickIdx] = true;
        changed = true;
        break;
      }
    }
  }

  for (size_t i = 0; i < candidates.size(); ++i) {
    if (!rewritten[i]) {
      softDeferConstraints.emplace_back(candidates[i].first, candidates[i].second);
    }
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
        auto *iNode = depGraph.getNode(iCandidateID);
        auto *jNode = depGraph.getNode(jCandidateID);
        if (depGraph.hasEdge(iCandidateID, jCandidateID, mlir::Value()) &&
            classifyDepType(iNode, jNode) == DepType::WAR) {
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

void FusionAnalyzer::collectFusionSourceGroups(const GroupPtr &targetGroup, std::vector<unsigned> &warGroupIds,
                                               std::vector<unsigned> &nonWarGroupIds) {
  // Accumulate one DependenceInfo per source group; on each new pred, replace the entry
  // when its loopDepth is smaller (most-restrictive constraint wins).
  std::unordered_map<unsigned, DependenceInfo> depAccumMap;
  std::unordered_set<unsigned> warGroupIdSet;
  std::unordered_set<unsigned> nonWarGroupIdSet;
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
      bool isNotWAR = (pred.depType != DepType::WAR);
      DependenceInfo entry;
      entry.loopDepth = pred.loopDepth;
      entry.memref = pred.memref;
      entry.predNodeId = pred.nodeId;
      entry.targetNodeId = targetNodeId;
      entry.depType = pred.depType;
      bool hasSubView = false;
      (void)affine::getSourceMemRef(pred.memref, &hasSubView);
      if (hasSubView) {
        entry.memrefKind = MemrefKind::Subview;
      } else if (!pred.memref.getDefiningOp<memref::AllocOp>()) {
        entry.memrefKind = MemrefKind::Input;
      } else {
        entry.memrefKind = MemrefKind::Normal;
      }
      auto &accum = depAccumMap[sourceGroupId];
      if (accum.predNodeId == UINT_MAX || entry.loopDepth < accum.loopDepth) {
        accum = std::move(entry);
      }
      if (isNotWAR) {
        nonWarGroupIdSet.insert(sourceGroupId);
      } else {
        warGroupIdSet.insert(sourceGroupId);
      }
    }
  }

  // Pre-populate groupDependenciesCache: smaller loopDepth wins on conflict.
  for (auto &[sourceGroupId, accum] : depAccumMap) {
    auto cacheKey = std::make_pair(sourceGroupId, targetGroup->groupId);
    auto it = groupDependenciesCache.find(cacheKey);
    if (it == groupDependenciesCache.end() || accum.loopDepth < it->second.loopDepth) {
      groupDependenciesCache[cacheKey] = std::move(accum);
    }
  }

  // Emit results sorted by groupId descending so downstream loops process
  // larger groupIds first without needing another sort pass.
  warGroupIds.assign(warGroupIdSet.begin(), warGroupIdSet.end());
  nonWarGroupIds.assign(nonWarGroupIdSet.begin(), nonWarGroupIdSet.end());
  std::sort(warGroupIds.begin(), warGroupIds.end(), std::greater<unsigned>());
  std::sort(nonWarGroupIds.begin(), nonWarGroupIds.end(), std::greater<unsigned>());
}

void FusionAnalyzer::dumpCollectFusionSourceInfo(const GroupPtr &targetGroup, const char *dependenceType,
                                                 const std::vector<unsigned> &sourceGroupIds) {
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

void FusionAnalyzer::processWarEdges(std::unordered_map<unsigned, std::vector<unsigned>> &nonWarMap) {
  while (!finishPlan()) {
    auto targetGroup = getFusionTargetGroup();
    if (targetGroup == nullptr) break;

    std::vector<unsigned> warGroupIds;
    std::vector<unsigned> nonWarGroupIds;
    collectFusionSourceGroups(targetGroup, warGroupIds, nonWarGroupIds);
    nonWarMap[targetGroup->groupId] = nonWarGroupIds;

    for (unsigned sourceGroupId : warGroupIds) {
      auto sourceGroup = depGraph.getGroup(sourceGroupId);
      if (sourceGroup != nullptr) applyAndFuse(targetGroup, sourceGroup);
    }

    for (auto targetId : targetGroup->nodesId) finished.insert(targetId);
  }
}

bool FusionAnalyzer::hasSubviewRarDep(unsigned targetGroupId,
                                      const std::unordered_map<unsigned, std::vector<unsigned>> &nonWarMap) const {
  auto it = nonWarMap.find(targetGroupId);
  if (it == nonWarMap.end()) return false;
  for (unsigned sourceGroupId : it->second) {
    auto cacheIt = groupDependenciesCache.find(std::make_pair(sourceGroupId, targetGroupId));
    if (cacheIt != groupDependenciesCache.end() && cacheIt->second.memrefKind == MemrefKind::Subview) return true;
  }
  return false;
}

void FusionAnalyzer::processNonWarEdges(const std::unordered_map<unsigned, std::vector<unsigned>> &nonWarMap,
                                        std::vector<std::pair<unsigned, unsigned>> &deferCandidates) {
  std::vector<std::pair<unsigned, unsigned>> sortedRarEdges;
  for (const auto &[targetGroupId, sourceIds] : nonWarMap) {
    std::transform(sourceIds.begin(), sourceIds.end(), std::back_inserter(sortedRarEdges),
                   [targetGroupId](unsigned sourceGroupId) { return std::make_pair(sourceGroupId, targetGroupId); });
  }

  auto isSubviewEdge = [&](unsigned source, unsigned target) {
    auto it = groupDependenciesCache.find(std::make_pair(source, target));
    return it != groupDependenciesCache.end() && it->second.memrefKind == MemrefKind::Subview;
  };

  std::sort(sortedRarEdges.begin(), sortedRarEdges.end(),
            [&](const std::pair<unsigned, unsigned> &a, const std::pair<unsigned, unsigned> &b) {
              bool aSubview = isSubviewEdge(a.first, a.second);
              bool bSubview = isSubviewEdge(b.first, b.second);
              if (aSubview != bSubview) return !aSubview;
              if (a.second != b.second) return a.second < b.second;
              return a.first < b.first;
            });

  for (const auto &[sourceGroupId, targetGroupId] : sortedRarEdges) {
    auto targetGroup = depGraph.getGroup(targetGroupId);
    if (!targetGroup) continue;
    auto sourceGroup = depGraph.getGroup(sourceGroupId);
    if (!sourceGroup) continue;

    bool isAncestry = false;
    if (findBackwardIntersection(sourceGroup, targetGroup, &isAncestry)) {
      auto cacheIt = groupDependenciesCache.find(std::make_pair(sourceGroupId, targetGroupId));
      if (cacheIt != groupDependenciesCache.end() && cacheIt->second.memrefKind == MemrefKind::Input) continue;
      deferCandidates.emplace_back(sourceGroupId, targetGroupId);
      continue;
    }
    if (!isAncestry) applyAndFuse(targetGroup, sourceGroup);
  }
}

void FusionAnalyzer::plan() {
  initGroups();
  topoSortInit();
  precomputeDirectPredecessors();

  // Phase 1: Process store-load (producer-consumer) edges only.
  std::unordered_map<unsigned, std::vector<unsigned>> nonWarMap;
  processWarEdges(nonWarMap);

  std::vector<std::pair<unsigned, unsigned>> deferCandidates;
  processNonWarEdges(nonWarMap, deferCandidates);

  // Phase 3: Identify conflicting defer pairs and rewrite.
  resolveConflictingDefers(deferCandidates);

  std::vector<FusionPlan> edges = deduplicateAndClassifyEdges();
  fusionPlans = TopoScheduler(edges, depGraph.nodes.size(), softDeferConstraints).run();
}

void FusionAnalyzer::print(llvm::raw_ostream &os, bool detail) const {
  os << "\n===== FusionPlans =====\n";
  for (const auto &plan : fusionPlans) {
    os << "FusionPlan: Group [" << plan.fusedGroup.from << " -> " << plan.fusedGroup.to << "], "
       << "Band [" << plan.fusedBand.from << " -> " << plan.fusedBand.to << "]";
    if (detail) {
      os << ", "
         << "FusionType: " << plan.fusionType << ", "
         << "LoopTransform: " << loopTransformToString(plan.loopTransform) << ", "
         << "LoopDepth: " << plan.depInfo.loopDepth << ", "
         << "MemrefKind: " << memrefKindToString(plan.depInfo.memrefKind) << ", "
         << "FusionNodeRecord: (" << plan.depInfo.predNodeId << " -> " << plan.depInfo.targetNodeId << ")";
    }
    os << "\n";
  }
}

}  // namespace akg
}  // namespace mlir
