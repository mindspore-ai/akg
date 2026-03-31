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

#include "akg/Dialect/Affine/Analysis/AKGLoopFusionBuilder.h"

#include <algorithm>
#include <queue>

#include "akg/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "akg/Utils/AKGGlobalVars.hpp"
#include "akg/Utils/AnalysisCommon.hpp"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
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

using llvm::DenseMap;
using llvm::DenseSet;
using llvm::raw_ostream;
using llvm::SetVector;
using llvm::SmallVector;
using mlir::kReductionInitAttr;
using mlir::kReductionTypeStr;

// Get group by group ID, returns nullptr if group doesn't exist
GroupPtr MemRefDependenceGraphForFusion::getGroup(unsigned groupId) {
  return groups.find(groupId) != groups.end() ? groups[groupId] : nullptr;
}

// Get group by node ID, returns nullptr if node is not in any group
GroupPtr MemRefDependenceGraphForFusion::getGroupByNode(unsigned nodeId) {
  return nodeToGroup.find(nodeId) != nodeToGroup.end() ? nodeToGroup[nodeId] : nullptr;
}

// Collect loop nest state from operations in the block
void LoopNestStateCollector::collect(Operation *opToWalk) {
  opToWalk->walk([&](Operation *op) {
    if (isa<affine::AffineForOp>(op)) {
      forOps.push_back(cast<affine::AffineForOp>(op));
    } else {
      otherInsts.push_back(op);
      if (op->getNumRegions() != 0 && !isa<affine::AffineIfOp>(op)) {
        hasNonAffineRegionOp = true;
      } else if (isa<affine::AffineReadOpInterface>(op)) {
        loadOpInsts.push_back(op);
      } else if (isa<affine::AffineWriteOpInterface>(op)) {
        storeOpInsts.push_back(op);
      }
    }
  });
}

bool MemRefDependenceGraphForFusion::init() {
  // Map from a memref to the set of ids of the nodes that have ops accessing
  // the memref.
  DenseMap<Value, SetVector<unsigned>> memrefAccesses;
  createInitNode(memrefAccesses);
  // Create Node Edges
  createEdges(memrefAccesses);
  // Precompute dependent groups for all groups
  precomputeDependentGroups();
  return true;
}

void MemRefDependenceGraphForFusion::createInitNode(DenseMap<Value, SetVector<unsigned>> &memrefAccesses) {
  block->walk([&](Operation *op) {
    if (auto forOp = dyn_cast<affine::AffineForOp>(op)) {
      if (isa<affine::AffineForOp>(op->getParentOp())) {
        return;
      }
      // Create graph node 'id' to represent top-level 'forOp' and record
      // all loads and store accesses it contains.
      LoopNestStateCollector collector;
      collector.collect(op);
      // Return false if a region holding op other than 'affine.for' and
      // 'affine.if' was found (not currently supported).
      // TODO(hjh): check this condition
      if (collector.hasNonAffineRegionOp) {
        return;
      }
      Node node(nextNodeId++, op);
      for (auto *opInst : collector.loadOpInsts) {
        node.loads.push_back(opInst);
        auto memref = cast<affine::AffineReadOpInterface>(opInst).getMemRef();
        memrefAccesses[memref].insert(node.id);
        // Also track the source memref if this is a subview or other aliasing operation
        Value sourceMemref = mlir::affine::getSourceMemRef(memref);
        if (sourceMemref != memref) {
          memrefAccesses[sourceMemref].insert(node.id);
        }
      }
      for (auto *opInst : collector.storeOpInsts) {
        node.stores.push_back(opInst);
        auto memref = cast<affine::AffineWriteOpInterface>(opInst).getMemRef();
        memrefAccesses[memref].insert(node.id);
        // Also track the source memref if this is a subview or other aliasing operation
        Value sourceMemref = mlir::affine::getSourceMemRef(memref);
        if (sourceMemref != memref) {
          memrefAccesses[sourceMemref].insert(node.id);
        }
      }
      nodes.insert({node.id, node});
      auto groupId = nextGroupId++;
      std::vector<unsigned> currNodeId;
      auto group = std::make_shared<Group>(groupId, node.id, forOp);
      for (auto innerOp : collector.otherInsts) {
        auto innerId = getNodeId(innerOp);
        if (innerId != -1) {
          currNodeId.emplace_back(innerId);
          nodeToGroup[innerId] = group;
        }
      }
      group->nodesId = currNodeId;
      group->groupTemplate = getGroupType(currNodeId);
      groups[groupId] = group;

      funcOperatorType = group->groupTemplate > funcOperatorType ? group->groupTemplate : funcOperatorType;
    } else if (isa<affine::AffineReadOpInterface>(op)) {
      // Create graph node for top-level load op.
      Node node(nextNodeId++, op);
      node.loads.push_back(op);
      auto memref = cast<affine::AffineReadOpInterface>(op).getMemRef();
      memrefAccesses[memref].insert(node.id);
      // Also track the source memref if this is a subview or other aliasing operation
      Value sourceMemref = mlir::affine::getSourceMemRef(memref);
      if (sourceMemref != memref) {
        memrefAccesses[sourceMemref].insert(node.id);
      }
      nodes.insert({node.id, node});
    } else if (isa<affine::AffineWriteOpInterface>(op)) {
      // Create graph node for top-level store op.
      Node node(nextNodeId++, op);
      node.stores.push_back(op);
      auto memref = cast<affine::AffineWriteOpInterface>(op).getMemRef();
      memrefAccesses[memref].insert(node.id);
      // Also track the source memref if this is a subview or other aliasing operation
      Value sourceMemref = mlir::affine::getSourceMemRef(memref);
      if (sourceMemref != memref) {
        memrefAccesses[sourceMemref].insert(node.id);
      }
      nodes.insert({node.id, node});
    } else if (isa<memref::SubViewOp, memref::MemorySpaceCastOp, memref::ExpandShapeOp, memref::CollapseShapeOp,
                   memref::ReshapeOp, memref::ReinterpretCastOp>(op)) {
      Node node(nextNodeId++, op);
      Value result = op->getResult(0);
      Value source = mlir::affine::getSourceMemRef(result);
      memrefAccesses[result].insert(node.id);
      if (source != result) {
        memrefAccesses[source].insert(node.id);
      }
      nodes.insert({node.id, node});
    } else if (op->getNumRegions() != 0 ||
               isa<memref::AllocOp, arith::ConstantOp, affine::AffineApplyOp, affine::AffineYieldOp, func::ReturnOp>(
                 op)) {
      // Return false if another region is found (not currently supported).
      return;
    } else {
      Node node(nextNodeId++, op);
      nodes.insert({node.id, node});
    }
  });
}

// Check if a node has a store to a memref that is either the given base memref itself,
// or an alias (subview/reshape) of it.
// This is needed because getStoreOpCount only checks exact memref matches,
// missing stores through subview/reshape aliases.
static bool isSameOrAliasedMemRef(Value accessMemref, Value baseMemref) {
  return mlir::affine::getSourceMemRef(accessMemref) == mlir::affine::getSourceMemRef(baseMemref);
}

static bool hasAliasedStoreToMemref(Node *node, Value baseMemref) {
  for (Operation *storeOp : node->stores) {
    if (auto writeOp = dyn_cast<affine::AffineWriteOpInterface>(storeOp)) {
      Value storeMemref = writeOp.getMemRef();
      if (isSameOrAliasedMemRef(storeMemref, baseMemref)) {
        return true;
      }
    }
  }
  return false;
}

static int getEnclosingForLoopNodeId(MemRefDependenceGraphForFusion &graph, const SetVector<unsigned> &nodeIds,
                                     unsigned dstId) {
  Operation *dstOp = graph.getNode(dstId)->op;
  for (unsigned nodeId : nodeIds) {
    Operation *candidate = graph.getNode(nodeId)->op;
    if (isa<affine::AffineForOp>(candidate) && candidate->isAncestor(dstOp)) {
      return static_cast<int>(nodeId);
    }
  }
  return -1;
}

void MemRefDependenceGraphForFusion::collectLoadNodeIdsAndNonForNodes(
  Value memref, const SetVector<unsigned> &nodeIds, SmallVector<unsigned> &loadNodeIds,
  SmallVector<std::pair<unsigned, bool>, 16> &nonForNodesWithStore) {
  for (unsigned nodeId : nodeIds) {
    Node *node = getNode(nodeId);
    if (auto loadOpInterface = dyn_cast<affine::AffineReadOpInterface>(node->op)) {
      if (isSameOrAliasedMemRef(loadOpInterface.getMemRef(), memref)) {
        loadNodeIds.push_back(nodeId);
      }
    } else if (isa<affine::AffineForOp>(node->op)) {
      for (Operation *loadOpInst : node->loads) {
        if (auto loadOpInterface = dyn_cast<affine::AffineReadOpInterface>(loadOpInst)) {
          if (isSameOrAliasedMemRef(loadOpInterface.getMemRef(), memref)) {
            int loadNodeId = getNodeId(loadOpInst);
            if (loadNodeId != -1) {
              if (std::find(loadNodeIds.begin(), loadNodeIds.end(), loadNodeId) == loadNodeIds.end()) {
                loadNodeIds.push_back(loadNodeId);
              }
            } else {
              Node loadNode(nextNodeId++, loadOpInst);
              loadNode.loads.push_back(loadOpInst);
              nodes.insert({loadNode.id, loadNode});
              loadNodeIds.push_back(loadNode.id);
            }
          }
        }
      }
    }
    if (!isa<affine::AffineForOp>(node->op)) {
      nonForNodesWithStore.push_back({nodeId, hasAliasedStoreToMemref(node, memref)});
    }
  }
}

void MemRefDependenceGraphForFusion::addAliasedStoreEdges(
  Value memref, const SetVector<unsigned> &nodeIds,
  const SmallVector<std::pair<unsigned, bool>, 16> &nonForNodesWithStore) {
  for (unsigned i = 0; i < nonForNodesWithStore.size(); ++i) {
    unsigned srcId = nonForNodesWithStore[i].first;
    bool srcHasStore = nonForNodesWithStore[i].second;
    for (unsigned j = i + 1; j < nonForNodesWithStore.size(); ++j) {
      unsigned dstId = nonForNodesWithStore[j].first;
      bool dstHasStore = nonForNodesWithStore[j].second;
      if ((srcHasStore || dstHasStore) && !hasEdge(srcId, dstId, memref)) {
        int forLoopNodeId = getEnclosingForLoopNodeId(*this, nodeIds, dstId);
        unsigned edgeLoopDepth = computeMemrefLoopDepth(forLoopNodeId, memref);
        addEdge(srcId, dstId, memref, edgeLoopDepth);
      }
    }
  }
}

void MemRefDependenceGraphForFusion::addMultipleLoadEdges(Value memref, const SetVector<unsigned> &nodeIds,
                                                          SmallVector<unsigned> &loadNodeIds) {
  if (loadNodeIds.size() < 2) {
    return;
  }
  std::sort(loadNodeIds.begin(), loadNodeIds.end());
  for (unsigned i = 0; i < loadNodeIds.size(); ++i) {
    for (unsigned j = i + 1; j < loadNodeIds.size(); ++j) {
      unsigned srcId = loadNodeIds[i];
      unsigned dstId = loadNodeIds[j];
      if (!hasEdge(srcId, dstId, memref)) {
        int forLoopNodeId = getEnclosingForLoopNodeId(*this, nodeIds, dstId);
        unsigned edgeLoopDepth = computeMemrefLoopDepth(forLoopNodeId, memref);
        addEdge(srcId, dstId, memref, edgeLoopDepth);
      }
    }
  }
}

bool MemRefDependenceGraphForFusion::createEdges(const DenseMap<Value, SetVector<unsigned>> &memrefAccesses) {
  if (!MemRefDependenceGraph::createEdges(memrefAccesses)) {
    return false;
  }
  for (auto &memrefAndList : memrefAccesses) {
    Value memref = memrefAndList.first;
    const SetVector<unsigned> &nodeIds = memrefAndList.second;
    if (nodeIds.size() < 2) {
      continue;
    }
    SmallVector<unsigned> loadNodeIds;
    SmallVector<std::pair<unsigned, bool>, 16> nonForNodesWithStore;
    collectLoadNodeIdsAndNonForNodes(memref, nodeIds, loadNodeIds, nonForNodesWithStore);
    addAliasedStoreEdges(memref, nodeIds, nonForNodesWithStore);
    addMultipleLoadEdges(memref, nodeIds, loadNodeIds);
  }
  return true;
}

void MemRefDependenceGraphForFusion::print(raw_ostream &os) const {
  os << "MemRefDependenceGraphForFusion!!!\n";
  MemRefDependenceGraph::print(os);
  for (auto it : groups) {
    auto g = it.second;
    std::string groupTemplateString = g->getGroupTemplateString();
    os << "Group " << g->groupId << " (GroupTemplate " << groupTemplateString << ") IsGlobalOut (" << g->isGlobalOut
       << ") root is " << g->rootId << " has " << g->nodesId.size() << " nodes inside: [";
    for (auto nid : g->nodesId) {
      os << nid << ", ";
    }
    os << "]\n";
  }
}

static llvm::ArrayRef<int64_t> getStaticShape(Value memrefVal, llvm::SmallVector<int64_t, 4> &shapeBuf) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(memrefVal.getType());
  if (!memrefType || !memrefType.hasStaticShape()) {
    return {};
  }
  shapeBuf.assign(memrefType.getShape().begin(), memrefType.getShape().end());
  return shapeBuf;
}

static int64_t getShapeProduct(llvm::ArrayRef<int64_t> shape) {
  if (shape.empty()) return -1;
  int64_t prod = 1;
  for (int64_t d : shape) {
    if (d < 0) return -1;
    prod *= d;
  }
  return prod;
}

static bool isSameShape(llvm::ArrayRef<int64_t> a, llvm::ArrayRef<int64_t> b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

static bool isPermutationShape(llvm::ArrayRef<int64_t> a, llvm::ArrayRef<int64_t> b) {
  if (a.size() != b.size()) return false;
  llvm::SmallVector<int64_t, 4> aSorted(a.begin(), a.end());
  llvm::SmallVector<int64_t, 4> bSorted(b.begin(), b.end());
  std::sort(aSorted.begin(), aSorted.end());
  std::sort(bSorted.begin(), bSorted.end());
  return isSameShape(aSorted, bSorted);
}

static bool isBroadcastLike(llvm::ArrayRef<int64_t> inShape, llvm::ArrayRef<int64_t> outShape) {
  if (inShape.size() != outShape.size()) return false;

  bool diffFound = false;
  for (size_t i = 0; i < inShape.size(); ++i) {
    auto inD = inShape[i];
    auto outD = outShape[i];
    if (inD == outD) continue;

    diffFound = true;
    if (inD != 1) {
      return false;
    }
  }
  return diffFound;
}

static bool isReshape(llvm::ArrayRef<int64_t> inShape, llvm::ArrayRef<int64_t> outShape) {
  if (inShape.size() == outShape.size()) return false;

  auto inProd = getShapeProduct(inShape);
  auto outProd = getShapeProduct(outShape);
  if (inProd > 0 && outProd > 0 && inProd == outProd) {
    return true;
  }
  return false;
}

static bool isSubsequence(llvm::ArrayRef<int64_t> big, llvm::ArrayRef<int64_t> small) {
  if (small.empty()) return false;

  size_t i = 0;
  size_t j = 0;

  while (i < big.size() && j < small.size()) {
    if (big[i] == small[j]) ++j;
    ++i;
  }

  return j == small.size();
}

static bool isSubsequencePart(llvm::ArrayRef<int64_t> a, llvm::ArrayRef<int64_t> b) {
  if (a.size() == b.size()) return false;

  return isSubsequence(a, b) || isSubsequence(b, a);
}

OperatorTemplate MemRefDependenceGraphForFusion::getGroupType(const std::vector<unsigned> &nodes) {
  SmallVector<affine::AffineLoadOp> loads;
  SmallVector<affine::AffineStoreOp> stores;
  for (auto nid : nodes) {
    auto op = getNode(nid)->op;
    if (op->hasAttr(kReductionTypeStr)) {
      return OperatorTemplate::Reduction;
    }

    if (isa<memref::ExpandShapeOp, memref::CollapseShapeOp>(op)) {
      // TODO(baiji): cannot identify reshape by op after FoldMemRefAliasOps pass
      return OperatorTemplate::Reshape;
    }
  }

  for (auto nid : nodes) {
    auto *op = getNode(nid)->op;
    if (auto store = dyn_cast<affine::AffineStoreOp>(op)) {
      stores.push_back(store);
    } else if (auto load = dyn_cast<affine::AffineLoadOp>(op)) {
      loads.push_back(load);
    }
  }

  if (stores.size() != 1) {
    return OperatorTemplate::Default;
  }

  if (loads.empty()) {
    return OperatorTemplate::Elementwise;
  }

  auto store = stores.front();
  Value outMemref = store.getMemRef();

  llvm::SmallVector<int64_t, 4> outShapeBuf;
  auto outShape = getStaticShape(outMemref, outShapeBuf);
  if (outShape.empty()) {
    return OperatorTemplate::Default;
  }

  bool hasElementwise = false;

  for (auto load : loads) {
    Value inMemref = load.getMemRef();
    llvm::SmallVector<int64_t, 4> inShapeBuf;
    auto inShape = getStaticShape(inMemref, inShapeBuf);
    if (inShape.empty()) {
      continue;
    }

    if (isSameShape(inShape, outShape)) {
      hasElementwise = true;
      continue;
    } else {
      hasElementwise = false;
    }

    if (isReshape(inShape, outShape)) {
      return OperatorTemplate::Reshape;
    }

    if (isPermutationShape(inShape, outShape)) {
      return OperatorTemplate::Transpose;
    }

    if (isBroadcastLike(inShape, outShape)) {
      return OperatorTemplate::Broadcast;
    }

    if (isSubsequencePart(inShape, outShape)) {
      return OperatorTemplate::Broadcast;
    }
  }

  if (hasElementwise) {
    return OperatorTemplate::Elementwise;
  }

  return OperatorTemplate::Default;
}

int MemRefDependenceGraphForFusion::getMemrefSourceOfNode(unsigned id) {
  Operation *op = getNode(id)->op;
  if (isa<affine::AffineLoadOp, affine::AffineStoreOp, memref::LoadOp, memref::StoreOp>(op)) {
    return static_cast<int>(id);
  }
  for (Edge edge : inEdges[id]) {
    auto backTraceId = getMemrefSourceOfNode(edge.id);
    if (backTraceId != -1) {
      return backTraceId;
    }
  }
  return -1;
}

// Gets all dependent groups (predecessor groups) of a given group.
std::vector<unsigned> MemRefDependenceGraphForFusion::getDependentGroups(unsigned groupId) {
  // Use precomputed cache if available
  auto it = dependentGroupsCache.find(groupId);
  if (it != dependentGroupsCache.end()) {
    return it->second;
  }

  // Fallback to computation if cache miss (should not happen after init)
  std::vector<unsigned> depGroups;
  auto group = getGroup(groupId);
  if (group == nullptr) {
    return depGroups;
  }

  for (auto nodeId : group->nodesId) {
    std::vector<unsigned> predecessorIds;
    bool isLoad = isa<affine::AffineLoadOp>(getNode(nodeId)->op);
    getPredecessorNodes(nodeId, predecessorIds);
    for (auto predId : predecessorIds) {
      if (isLoad && isa<affine::AffineLoadOp>(getNode(predId)->op)) {
        continue;
      }
      auto predGroup = getGroupByNode(predId);
      if (predGroup != nullptr && predGroup->groupId != groupId) {
        if (std::find(depGroups.begin(), depGroups.end(), predGroup->groupId) == depGroups.end()) {
          depGroups.push_back(predGroup->groupId);
        }
      }
    }
  }
  return depGroups;
}

// Precomputes dependent groups for all groups and caches them.
void MemRefDependenceGraphForFusion::precomputeDependentGroups() {
  dependentGroupsCache.clear();

  for (const auto &groupPair : groups) {
    unsigned groupId = groupPair.first;
    auto group = groupPair.second;

    // Precompute dependent groups
    std::vector<unsigned> depGroups;
    for (auto nodeId : group->nodesId) {
      std::vector<unsigned> predecessorIds;
      bool isLoad = isa<affine::AffineLoadOp>(getNode(nodeId)->op);
      getPredecessorNodes(nodeId, predecessorIds);
      for (auto predId : predecessorIds) {
        if (isLoad && isa<affine::AffineLoadOp>(getNode(predId)->op)) {
          continue;
        }
        auto predGroup = getGroupByNode(predId);
        if (predGroup != nullptr && predGroup->groupId != groupId) {
          if (std::find(depGroups.begin(), depGroups.end(), predGroup->groupId) == depGroups.end()) {
            depGroups.push_back(predGroup->groupId);
          }
        }
      }
    }
    dependentGroupsCache[groupId] = std::move(depGroups);
  }
}

// Checks if fromGroupId is a direct or indirect dependency of toGroupId in the dependency graph.
bool MemRefDependenceGraphForFusion::isDependencyInGraph(unsigned fromGroupId, unsigned toGroupId) {
  auto toGroup = getGroup(toGroupId);
  if (toGroup == nullptr) {
    return false;
  }
  // Get all predecessor nodes of to
  std::unordered_set<unsigned> visited;
  std::queue<unsigned> queue;

  // Initialize: add all predecessor nodes of to to the queue
  auto toDepGroups = getDependentGroups(toGroupId);
  for (auto depGroupId : toDepGroups) {
    if (visited.insert(depGroupId).second) {
      queue.push(depGroupId);
    }
  }

  // BFS traverse all predecessor groups
  while (!queue.empty()) {
    unsigned currentGroupId = queue.front();
    queue.pop();

    // If from is found, it means from is a dependency of to
    if (currentGroupId == fromGroupId) {
      return true;
    }

    // Continue searching for predecessors of the current group
    auto currentDepGroups = getDependentGroups(currentGroupId);
    for (auto depGroupId : currentDepGroups) {
      if (visited.insert(depGroupId).second) {
        queue.push(depGroupId);
      }
    }
  }

  return false;
}

unsigned FusionCodeGenHelper::getAliasId(unsigned srcId) {
  // Follow the alias chain to find the final destination
  std::unordered_set<unsigned> visited;
  unsigned currentId = srcId;

  while (nodeAlias.find(currentId) != nodeAlias.end()) {
    if (visited.count(currentId)) {
      // Circular alias detected, break to avoid infinite loop
      break;
    }
    visited.insert(currentId);
    currentId = nodeAlias[currentId];
  }

  return currentId;
}

// Collect all loops in a loop nest starting from the given loop
// This includes the loop itself and all nested loops, ordered from outermost to innermost
static void collectLoopNest(affine::AffineForOp rootLoop, SmallVector<affine::AffineForOp, 4> &loops) {
  loops.clear();

  // First, collect all parent loops (from outermost to the root loop's parent)
  SmallVector<affine::AffineForOp, 4> parentLoops;
  affine::getAffineForIVs(*rootLoop, &parentLoops);
  loops.append(parentLoops.begin(), parentLoops.end());

  // Then add the root loop itself
  loops.push_back(rootLoop);

  // Finally, recursively collect nested loops in depth-first order
  std::function<void(affine::AffineForOp)> collectNested = [&](affine::AffineForOp loop) {
    for (auto &op : *loop.getBody()) {
      if (auto nestedLoop = dyn_cast<affine::AffineForOp>(op)) {
        loops.push_back(nestedLoop);
        collectNested(nestedLoop);
      }
    }
  };
  collectNested(rootLoop);
}

void FusionLoopNestInfo::collect(affine::AffineForOp rootOp) {
  root = rootOp;
  loops.clear();
  collectLoopNest(rootOp, loops);
  loopDepth = static_cast<unsigned>(loops.size());

  SmallVector<affine::AffineForOp, 4> perfectBand;
  affine::getPerfectlyNestedLoops(perfectBand, rootOp);
  perfectDepth = static_cast<unsigned>(perfectBand.size());
  isPerfect = (perfectDepth == loopDepth);
}

static bool isDependentLoadOrStoreOp(Operation *candidate, DenseMap<Value, bool> &memFlags) {
  auto asLoad = dyn_cast<affine::AffineReadOpInterface>(candidate);
  if (asLoad) {
    Value memref = asLoad.getMemRef();
    auto it = memFlags.find(memref);
    return it != memFlags.end() && it->second;
  }

  auto asStore = dyn_cast<affine::AffineWriteOpInterface>(candidate);
  if (asStore) {
    Value memref = asStore.getMemRef();
    return memFlags.count(memref) != 0;
  }

  return false;
}

// Returns the earliest operation strictly between opA and opB that is
// data-dependent on opA.
static Operation *getFirstDependentOpInRange(affine::AffineForOp opA, affine::AffineForOp opB,
                                             DenseMap<Value, bool> &memFlags) {
  // Walk over each opX in the block in the open interval (opA, opB) and look
  // for a dependence from opA to opX, i.e. both touch the same memref and
  // at least one of them performs a write.
  Operation *firstDependent = nullptr;

  for (auto it = std::next(Block::iterator(opA)), e = Block::iterator(opB); it != e && !firstDependent; ++it) {
    Operation *opX = &*it;

    opX->walk([&](Operation *nested) {
      if (firstDependent) {
        return WalkResult::interrupt();
      }
      if (isDependentLoadOrStoreOp(nested, memFlags)) {
        firstDependent = opX;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }

  return firstDependent;
}

// Returns the last operation opX strictly between opA and opB such that there
// exists a data dependence from opX to opB.
static Operation *getLastDependentOpInRange(affine::AffineForOp opA, affine::AffineForOp opB,
                                            DenseMap<Value, bool> &memFlags) {
  // Traverse operations in (opA, opB) in reverse order and determine the last
  // opX that has a dependence to opB via:
  //  * conflicting memory access on the same memref with at least one store, or
  //  * SSA value produced by opX and eventually used in the loop nest of opB.
  Operation *lastDependent = nullptr;

  for (auto it = std::next(Block::reverse_iterator(opB)), e = Block::reverse_iterator(opA); it != e && !lastDependent;
       ++it) {
    Operation *opX = &*it;

    opX->walk([&](Operation *nested) {
      if (lastDependent) {
        return WalkResult::interrupt();
      }

      if (isa<affine::AffineReadOpInterface, affine::AffineWriteOpInterface>(nested)) {
        if (isDependentLoadOrStoreOp(nested, memFlags)) {
          lastDependent = opX;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }

      for (Value produced : nested->getResults()) {
        for (Operation *user : produced.getUsers()) {
          SmallVector<affine::AffineForOp, 4> surroundingLoops;
          // Check whether any loop in the loop nest enclosing 'user' is opB.
          affine::getAffineForIVs(*user, &surroundingLoops);
          if (llvm::is_contained(surroundingLoops, opB)) {
            lastDependent = opX;
            return WalkResult::interrupt();
          }
        }
      }

      return WalkResult::advance();
    });
  }

  return lastDependent;
}

// Attempts to compute an insertion position for the fused loop nest such that
// the original data dependences of the surrounding program are preserved.
// Returns true if a valid insertion position exists (including when firstFromSrc
// and lastToDst are both null, meaning no dependence conflict and inserting at
// dstLoop is safe); returns false when dependence directions conflict and no
// valid position exists.
static bool hasValidFusionPoint(affine::AffineForOp srcLoop, affine::AffineForOp dstLoop,
                                DenseMap<Value, bool> &srcMemFlags, DenseMap<Value, bool> &dstMemFlags) {
  Operation *firstFromSrc = getFirstDependentOpInRange(srcLoop, dstLoop, srcMemFlags);
  Operation *lastToDst = getLastDependentOpInRange(srcLoop, dstLoop, dstMemFlags);

  // Block layout abstraction:
  //
  //   ...
  //   srcLoop
  //   ...
  //   lastToDst   (lastToDst → dstLoop)
  //   ...
  //   firstFromSrc (srcLoop → firstFromSrc)
  //   ...
  //   dstLoop
  //
  // Legal insertion range lies strictly inside: (lastToDst, firstFromSrc).
  if (firstFromSrc && lastToDst && firstFromSrc->isBeforeInBlock(lastToDst)) {
    // There is no position that simultaneously respects both dependence
    // directions.
    return false;
  }

  // If there is no dependence from srcLoop to any operation in (srcLoop, dstLoop),
  // inserting at dstLoop is always safe.
  return true;
}

// Helper function to check if an operation is still valid (not removed or
// structurally broken).
// An operation is treated as invalid when:
// - It lacks a parent operation (detached from the IR), or
// - It has zero operands (AffineLoad/Store always expect at least a memref).
static bool isOperationValid(Operation *candidate) {
  if (!candidate) {
    return false;
  }

  if (!candidate->getParentOp()) {
    return false;
  }

  if (candidate->getNumOperands() == 0) {
    return false;
  }

  return true;
}

// Collects affine load/store operations from a loop that access any of 'memrefs'
// into 'memFlags' and 'loadAndStoreOps'. Skips invalid ops and non-affine accesses.
static void collectLoadAndStoreOpsFromOps(const llvm::DenseSet<Value> &memrefs, affine::AffineForOp loopOp,
                                          DenseMap<Value, bool> &memFlags,
                                          SmallVector<Operation *, 4> &loadAndStoreOps) {
  if (!loopOp) return;
  loopOp->walk([&](Operation *op) {
    if (!isOperationValid(op)) return;
    if (auto read = dyn_cast<affine::AffineReadOpInterface>(op)) {
      if (memrefs.contains(read.getMemRef())) {
        memFlags.insert({read.getMemRef(), false});
        loadAndStoreOps.push_back(op);
      }
    } else if (auto write = dyn_cast<affine::AffineWriteOpInterface>(op)) {
      if (memrefs.contains(write.getMemRef())) {
        memFlags.insert({write.getMemRef(), true});
        loadAndStoreOps.push_back(op);
      }
    }
  });
}

void FusionCodeGenHelper::buildStrategyOpsA(const affine::FusionStrategy &strategy, llvm::ArrayRef<Operation *> allOpsA,
                                            llvm::SmallVector<Operation *, 4> &strategyOpsA) {
  strategyOpsA.clear();

  switch (strategy.getStrategy()) {
    case affine::FusionStrategy::Generic: {
      strategyOpsA.append(allOpsA.begin(), allOpsA.end());
      break;
    }
    case affine::FusionStrategy::ProducerConsumer: {
      for (Operation *op : allOpsA) {
        if (isa<affine::AffineWriteOpInterface>(op)) {
          strategyOpsA.push_back(op);
        }
      }
      break;
    }
    case affine::FusionStrategy::Sibling: {
      Value siblingMem = strategy.getSiblingFusionMemRef();
      for (Operation *op : allOpsA) {
        auto load = dyn_cast<affine::AffineReadOpInterface>(op);
        if (load && load.getMemRef() == siblingMem) {
          strategyOpsA.push_back(op);
        }
      }
      break;
    }
  }
}

// Helper function to check if the outermost loops (up to depth) have matching bounds.
static bool checkLoopBoundsMatch(const SmallVector<affine::AffineForOp, 4> &loops1,
                                 const SmallVector<affine::AffineForOp, 4> &loops2, unsigned depth) {
  if (depth > loops1.size() || depth > loops2.size()) {
    return false;
  }
  for (unsigned i = 0; i < depth; ++i) {
    auto loop1 = loops1[i];
    auto loop2 = loops2[i];
    if (loop1.getLowerBoundMap() != loop2.getLowerBoundMap() || loop1.getUpperBoundMap() != loop2.getUpperBoundMap() ||
        loop1.getStep() != loop2.getStep()) {
      return false;
    }
  }
  return true;
}

unsigned FusionCodeGenHelper::findMaxLegalFusionDepth(
  const FusionLoopNestInfo &srcInfo, const FusionLoopNestInfo &dstInfo, const affine::FusionStrategy &strategy,
  llvm::SmallVector<affine::ComputationSliceState, 8> &depthSliceUnions, const FusionPlan &plan, bool srcDstReversed) {
  if (plan.depInfo.memrefs.empty()) {
    llvm::dbgs() << "No memrefs found for fusion\n";
    return 0;
  }

  affine::AffineForOp srcAffineForOp = srcInfo.root;
  affine::AffineForOp dstAffineForOp = dstInfo.root;

  DenseMap<Value, bool> srcMemFlags;
  SmallVector<Operation *, 4> srcAccesses;
  DenseMap<Value, bool> dstMemFlags;
  SmallVector<Operation *, 4> dstAccesses;

  if (!srcDstReversed) {
    collectLoadAndStoreOpsFromOps(plan.depInfo.memrefs, srcAffineForOp, srcMemFlags, srcAccesses);
    collectLoadAndStoreOpsFromOps(plan.depInfo.memrefs, dstAffineForOp, dstMemFlags, dstAccesses);
  } else {
    collectLoadAndStoreOpsFromOps(plan.depInfo.memrefs, dstAffineForOp, srcMemFlags, srcAccesses);
    collectLoadAndStoreOpsFromOps(plan.depInfo.memrefs, srcAffineForOp, dstMemFlags, dstAccesses);
  }

  bool isSrcBeforeDst = srcAffineForOp->isBeforeInBlock(dstAffineForOp);
  affine::AffineForOp loopA = isSrcBeforeDst ? srcAffineForOp : dstAffineForOp;
  affine::AffineForOp loopB = isSrcBeforeDst ? dstAffineForOp : srcAffineForOp;
  if (!hasValidFusionPoint(loopA, loopB, srcMemFlags, dstMemFlags)) {
    llvm::dbgs() << "Fusion would violate dependences in block\n";
    return 0;
  }

  SmallVector<Operation *, 4> strategyOpsA;
  buildStrategyOpsA(strategy, srcAccesses, strategyOpsA);
  if (strategyOpsA.empty()) {
    llvm::dbgs() << "No strategy ops found for fusion\n";
    return 0;
  }

  unsigned loopDepth = dstInfo.loopDepth;
  if (!srcInfo.isPerfect) {
    loopDepth = std::min(srcInfo.perfectDepth, dstInfo.perfectDepth);
  }

  // Use depth of insertion target (second param = dstAffineForOp).
  depthSliceUnions.clear();
  depthSliceUnions.resize(loopDepth);
  unsigned maxDepth = 0;
  for (unsigned depth = loopDepth; depth >= 1; --depth) {
    if (strategy.getStrategy() == affine::FusionStrategy::ProducerConsumer) {
      if (plan.depInfo.loopDepth < depth) {
        continue;
      }
    }

    auto &sliceState = depthSliceUnions[depth - 1];
    affine::SliceComputationResult res =
      computeSliceUnionAKG(strategyOpsA, dstAccesses, depth, 0, isSrcBeforeDst, &sliceState);

    if (res.value == affine::SliceComputationResult::Success) {
      maxDepth = depth;
      break;
    }
  }

  return maxDepth;
}

// Helper function to perform loop fusion at a specific depth
// This function handles the common logic of cloning operations from source to destination loops
static void performLoopFusion(SmallVector<affine::AffineForOp, 4> srcLoops,
                              SmallVector<affine::AffineForOp, 4> dstLoops, unsigned bestDstLoopDepth) {
  // Create IRMapping to map source loop IVs to destination loop IVs
  IRMapping mapper;
  for (unsigned i = 0; i < bestDstLoopDepth && i < srcLoops.size() && i < dstLoops.size(); ++i) {
    mapper.map(srcLoops[i].getInductionVar(), dstLoops[i].getInductionVar());
  }

  // Recursively clone operations from source loop body to destination loop body
  // This function handles nested loops by merging their operations
  std::function<void(affine::AffineForOp, affine::AffineForOp)> cloneLoopBody = [&](affine::AffineForOp srcLoop,
                                                                                    affine::AffineForOp dstLoop) {
    // Map the induction variables
    mapper.map(srcLoop.getInductionVar(), dstLoop.getInductionVar());

    // This ensures srcLoops operations are placed before dstLoops operations
    Block::iterator insertPoint = dstLoop.getBody()->begin();
    // Find first non-terminator operation (skip terminator if it exists)
    while (insertPoint != dstLoop.getBody()->end() && isa<affine::AffineYieldOp>(*insertPoint)) {
      ++insertPoint;
    }
    OpBuilder builder(dstLoop.getBody(), insertPoint);

    // Clone all operations from source loop body to destination loop body
    for (Operation &op : srcLoop.getBody()->getOperations()) {
      // Skip the terminator (affine.yield)
      if (isa<affine::AffineYieldOp>(op)) {
        continue;
      }

      builder.clone(op, mapper);
    }
  };

  // Get the destination loop at fusion depth
  affine::AffineForOp targetLoop = dstLoops[bestDstLoopDepth - 1];

  // Get the corresponding source loop
  affine::AffineForOp sourceLoop = srcLoops[bestDstLoopDepth - 1];

  // Clone operations from source loop body to destination loop body
  cloneLoopBody(sourceLoop, targetLoop);
}

void FusionCodeGenHelper::doIFuse(unsigned srcGroupId, unsigned dstGroupId, FusionLoopNestInfo &srcInfo,
                                  FusionLoopNestInfo &dstInfo) {
  auto &srcLoops = srcInfo.loops;
  auto &dstLoops = dstInfo.loops;
  affine::AffineForOp srcAffineForOp = srcInfo.root;

  // Helper lambda to clear erased node's loads and stores
  auto clearErasedNode = [this](unsigned nodeId) {
    if (auto *node = mdg.getNode(nodeId)) {
      node->loads.clear();
      node->stores.clear();
      node->op = nullptr;
    }
  };

  auto depth = std::min(srcInfo.perfectDepth, dstInfo.perfectDepth);
  while (depth != 0 && !checkLoopBoundsMatch(srcLoops, dstLoops, depth)) {
    depth--;
  }
  if (depth == 0) {
    llvm::errs() << "srcLoops and dstLoops have no same loop bounds\n";
    return;
  }
  performLoopFusion(srcLoops, dstLoops, depth);
  srcAffineForOp.erase();
  auto srcGroup = mdg.getGroup(srcGroupId);
  auto dstGroup = mdg.getGroup(dstGroupId);
  if (!srcGroup || !dstGroup) {
    llvm::errs() << "srcGroup is nullptr or dstGroup is nullptr";
    return;
  }
  auto srcId = srcGroup->rootId;
  auto dstId = dstGroup->rootId;
  auto srcGroupTemplate = srcGroup->groupTemplate;
  auto dstGroupTemplate = dstGroup->groupTemplate;
  srcGroup->groupTemplate = std::max(srcGroupTemplate, dstGroupTemplate);
  dstGroup->groupTemplate = std::max(srcGroupTemplate, dstGroupTemplate);
  nodeAlias[srcId] = dstId;
  clearErasedNode(srcId);
}

void FusionCodeGenHelper::doHFuse(unsigned srcGroupId, unsigned dstGroupId, affine::AffineForOp srcAffineForOp,
                                  affine::AffineForOp dstAffineForOp, const FusionPlan &plan) {
  FusionLoopNestInfo srcInfo, dstInfo;
  srcInfo.collect(srcAffineForOp);
  dstInfo.collect(dstAffineForOp);

  SmallVector<affine::ComputationSliceState, 8> depthSliceUnions;
  affine::FusionStrategy strategy(affine::FusionStrategy::ProducerConsumer);

  unsigned maxLegalFusionDepth = 0;
  // If src is perfectly nested or both are non-perfectly neste, fuse src into dst.
  if (srcInfo.isPerfect || !dstInfo.isPerfect) {
    maxLegalFusionDepth = findMaxLegalFusionDepth(srcInfo, dstInfo, strategy, depthSliceUnions, plan);
  } else {
    maxLegalFusionDepth = findMaxLegalFusionDepth(dstInfo, srcInfo, strategy, depthSliceUnions, plan, true);
  }

  if (maxLegalFusionDepth == 0) {
    // When fusing dst into src, try manual slice (e.g. insert at innermost
    // loop that has enough enclosing depth) before falling back to doIFuse.
    doIFuse(srcGroupId, dstGroupId, srcInfo, dstInfo);
    return;
  }

  unsigned bestDepth = maxLegalFusionDepth;
  assert(bestDepth > 0 && "Unexpected loop fusion depth");
  affine::ComputationSliceState &bestSlice = depthSliceUnions[bestDepth - 1];
  assert(!bestSlice.isEmpty() && "Missing slice union for depth");

  if (srcInfo.isPerfect || !dstInfo.isPerfect) {
    // Fuse src into dst: merge src into dst, then erase src.
    fuseLoops(srcAffineForOp, dstAffineForOp, bestSlice);
    eraseLoopAndCleanupNode(srcGroupId, dstGroupId, srcAffineForOp);
  } else {
    // Fuse dst into src: merge dst into src, then erase dst.
    fuseLoops(dstAffineForOp, srcAffineForOp, bestSlice);
    eraseLoopAndCleanupNode(dstGroupId, srcGroupId, dstAffineForOp);
  }
}

void FusionCodeGenHelper::doVFuse(unsigned srcGroupId, unsigned dstGroupId, affine::AffineForOp srcAffineForOp,
                                  affine::AffineForOp dstAffineForOp, const FusionPlan &plan) {
  FusionLoopNestInfo srcInfo, dstInfo;
  srcInfo.collect(srcAffineForOp);
  dstInfo.collect(dstAffineForOp);

  SmallVector<affine::ComputationSliceState, 8> depthSliceUnions;
  unsigned maxLegalFusionDepth = 0;
  for (auto memref : plan.depInfo.memrefs) {
    if (memref) {
      affine::FusionStrategy strategy(memref);
      maxLegalFusionDepth = findMaxLegalFusionDepth(srcInfo, dstInfo, strategy, depthSliceUnions, plan);
    } else {
      // No memref found, try generic strategy for structure-based fusion
      affine::FusionStrategy strategy(affine::FusionStrategy::Generic);
      maxLegalFusionDepth = findMaxLegalFusionDepth(srcInfo, dstInfo, strategy, depthSliceUnions, plan);
    }
  }

  // Skip if fusion is not feasible at any loop depths.
  if (maxLegalFusionDepth == 0) {
    doIFuse(srcGroupId, dstGroupId, srcInfo, dstInfo);
    return;
  }

  unsigned bestDstLoopDepth = maxLegalFusionDepth;
  assert(bestDstLoopDepth > 0 && "Unexpected loop fusion depth");
  affine::ComputationSliceState &bestSlice = depthSliceUnions[bestDstLoopDepth - 1];
  assert(!bestSlice.isEmpty() && "Fusion depth has no computed slice union");
  fuseLoops(srcAffineForOp, dstAffineForOp, bestSlice, true);
  eraseLoopAndCleanupNode(srcGroupId, dstGroupId, srcAffineForOp);
}

void FusionCodeGenHelper::eraseLoopAndCleanupNode(unsigned erasedGroupId, unsigned aliasTargetGroupId,
                                                  affine::AffineForOp loopToErase) {
  auto erasedGroup = mdg.getGroup(erasedGroupId);
  auto aliasTargetGroup = mdg.getGroup(aliasTargetGroupId);
  if (!erasedGroup || !aliasTargetGroup) {
    llvm::errs() << "erasedGroup is nullptr or aliasTargetGroup is nullptr";
    return;
  }
  auto erasedNodeId = erasedGroup->rootId;
  auto aliasTargetId = aliasTargetGroup->rootId;
  auto erasedNodeGroupTemplate = erasedGroup->groupTemplate;
  auto aliasTargetGroupTemplate = aliasTargetGroup->groupTemplate;
  erasedGroup->groupTemplate = std::max(erasedNodeGroupTemplate, aliasTargetGroupTemplate);
  aliasTargetGroup->groupTemplate = std::max(erasedNodeGroupTemplate, aliasTargetGroupTemplate);
  nodeAlias[erasedNodeId] = aliasTargetId;

  // Clear the loads and stores of the erased node to prevent
  // subsequent accesses to invalid operation pointers during fusion analysis
  if (auto *erasedNode = mdg.getNode(erasedNodeId)) {
    erasedNode->loads.clear();
    erasedNode->stores.clear();
    erasedNode->op = nullptr;
  }

  loopToErase.erase();
}

std::string Group::getGroupTemplateString() const {
  auto it = operatorTemplateMap.find(static_cast<int>(groupTemplate));
  if (it != operatorTemplateMap.end()) {
    return it->second;
  } else {
    return std::to_string(static_cast<int>(groupTemplate));
  }
}

void Group::print(raw_ostream &os) const {
  std::string indent = "  ";
  os << "[Group " << groupId << "]\n";
  std::string groupTemplateString = getGroupTemplateString();
  os << indent << ">> GroupTemplate: " << groupTemplateString << "\n";
  os << indent << ">> FusedGroups: [";
  for (auto gid : fusedGroupId) {
    os << gid << ", ";
  }
  os << "]\n";
  os << indent << ">> Nodes: [";
  for (auto nid : nodesId) {
    os << nid << ", ";
  }
  os << "]\n";
}

}  // namespace akg
}  // namespace mlir
