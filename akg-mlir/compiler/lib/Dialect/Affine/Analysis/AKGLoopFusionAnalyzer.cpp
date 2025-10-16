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
#include "akg/Dialect/Affine/Analysis/AKGLoopFusionBuilder.h"
#include "akg/Utils/AKGGlobalVars.hpp"

#include <queue>

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

using namespace mlir;
using namespace llvm;
using namespace akg;
using namespace akgglobal;

namespace mlir {
namespace akg {
// Topological sort function for fusion plans
// Returns a sorted list of fusion plans based on their dependencies
static std::vector<FusionPlan> topologicalSort(const std::vector<FusionPlan> &dependencies, unsigned numNodes) {
  std::vector<unsigned> inDegree(numNodes + 1, 0);
  
  // Calculate in-degree for each node by counting incoming edges
  for (const auto &edge : dependencies) {
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
    for (const auto &edge : dependencies) {
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

void FusionAnalyzer::checkAndFixMultiOut(FusionPlan &fusePlan) {
  for (auto oldPlan : fusionPlans) {
    bool multiOut = oldPlan.fusedGroup.from == fusePlan.fusedGroup.from;
    if (multiOut) {
      llvm::outs() << "Convert multiout pc plan to sib plan\n";
      llvm::outs() << "Original " << fusePlan.fusedBand.from << " to " << fusePlan.fusedBand.to << "\n";

      auto orderGroups = [&]() -> std::pair<GroupPtr, GroupPtr> {
        auto oldGroup = depGraph.getGroup(oldPlan.fusedGroup.to);
        auto newGroup = depGraph.getGroup(fusePlan.fusedGroup.to);
        std::string oldTemplateString = oldGroup->getGroupTemplateString();
        std::string newTemplateString = newGroup->getGroupTemplateString();
        llvm::outs() << "oldGroup " << oldPlan.fusedGroup.to << " groupTemplate " << oldTemplateString
                     << ", new group " << fusePlan.fusedGroup.to << " groupTemplate " << newTemplateString << "\n";
        if (oldGroup->groupTemplate == OperatorTemplate::Reduce) {
          return std::make_pair(newGroup, oldGroup);
        }
        if (newGroup->groupTemplate == OperatorTemplate::Reduce) {
          return std::make_pair(oldGroup, newGroup);
        }
        for (auto prevPlan : fusionPlans) {
          if (prevPlan.fusedGroup.from == oldGroup->groupId) {
            llvm::outs() << "Old group has child\n";
            return std::make_pair(newGroup, oldGroup);
          }
          if (prevPlan.fusedGroup.from == newGroup->groupId) {
            llvm::outs() << "New group has child\n";
            return std::make_pair(oldGroup, newGroup);
          }
        }
        // default
        return std::make_pair(newGroup, oldGroup);
      };

      auto [srcGroup, dstGroup] = orderGroups();

      fusePlan.fusedGroup.from = srcGroup->groupId;
      fusePlan.fusedGroup.to = dstGroup->groupId;

      fusePlan.fusedBand.from = depGraph.getNodeId(srcGroup->getLeadingFor());
      fusePlan.fusedBand.to = depGraph.getNodeId(dstGroup->getLeadingFor());

      // link all source group to dst group
      for (size_t i = 0; i < fusionPlans.size(); ++i) {
        if (fusionPlans[i].fusedGroup.to == srcGroup->groupId) {
          llvm::outs() << "Update!!!\n";
          fusionPlans[i].fusedGroup.to = dstGroup->groupId;
          fusionPlans[i].fusedBand.to = fusePlan.fusedBand.to;
        }
      }

      llvm::outs() << "New " << fusePlan.fusedBand.from << " to " << fusePlan.fusedBand.to << "\n";
      fusePlan.fusionType = "V";
      break;
    }
  }
}

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
  llvm::outs() << "Fuse " << sourceGroup->groupId << " into " << targetGroup->groupId << "\n";
  // update fusionPlans
  auto srcNodeId = depGraph.getNodeId(sourceGroup->getLeadingFor());
  auto dstNodeId = depGraph.getNodeId(targetGroup->getLeadingFor());
  FusionPlan fusePlan;
  fusePlan.fusedGroup = FuseEdge(sourceGroup->groupId, targetGroup->groupId);
  fusePlan.fusedBand = FuseEdge(srcNodeId, dstNodeId);

  checkAndFixMultiOut(fusePlan);

  llvm::outs() << "Add Plan " << fusePlan.fusedBand.from << " to " << fusePlan.fusedBand.to << "\n";
  fusionPlans.emplace_back(fusePlan);
}

void FusionAnalyzer::getDirectlyPredecessors(unsigned id, std::vector<unsigned> &predecessorIds) {
  std::vector<unsigned> allPredecessorIds;

  depGraph.getPredecessorNodes(id, allPredecessorIds);
  std::sort(allPredecessorIds.begin(), allPredecessorIds.end(), std::greater<int>());

  std::unordered_set<size_t> skipIdx;
  for (size_t i = 1; i < allPredecessorIds.size(); ++i) {
    if (skipIdx.count(i)) {
      continue;
    }
    auto pid = allPredecessorIds[i];
    for (size_t j = 0; j < i; ++j) {
      if (skipIdx.count(j)) {
        continue;
      }
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
            llvm::outs() << "Insert source group " << tmp->groupId << " by node " << id << " , target is "
                         << fuseTargetId << "\n";
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
      llvm::outs() << "Do fusion for target  g1 = " << targetGroup->groupId << " g2 = " << sourceGroup->groupId << "\n";

      auto loopTransforms = inferLoopTransforms(targetGroup, sourceGroup);
      applyAndFuse(loopTransforms, targetGroup, sourceGroup);
    }
  }
  auto sortedPlan = topologicalSort(fusionPlans, depGraph.nodes.size());
  for (auto p : sortedPlan) {
    llvm::outs() << "Sort plan : " << p.fusedBand.from << " into " << p.fusedBand.to << "\n";
  }
  fusionPlans = sortedPlan;
}


void FusionAnalyzer::initGroups() { groups = depGraph.groups; }

void FusionAnalyzer::topoSort() {
  topoSortNodeIds.clear();
  std::vector<GroupPtr> allGroups;
  for (auto it : groups) {
    allGroups.push_back(it.second);
  }
  std::sort(allGroups.begin(), allGroups.end(), GroupCmp);
  for (auto g : allGroups) {
    std::string groupTemplateString = g->getGroupTemplateString();
    llvm::outs() << "group " << g->groupId << " groupTemplate " << groupTemplateString << "\n";
    for (auto node : g->nodesId) {
      topoSortNodeIds.push_back(node);
    }
  }
  llvm::outs() << "topoSort: ";
  for (auto it : topoSortNodeIds) {
    llvm::outs() << it << ", ";
  }
  llvm::outs() << "\n";
  return;
}

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
