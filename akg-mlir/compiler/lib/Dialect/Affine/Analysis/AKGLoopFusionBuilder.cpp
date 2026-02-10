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

// Helper function to get the source memref from a value that might be a subview
// or other aliasing operation. This traces back through subview operations
// to find the underlying memref.
static Value getSourceMemRef(Value memrefVal) {
  Value current = memrefVal;
  while (true) {
    if (auto subview = current.getDefiningOp<memref::SubViewOp>()) {
      current = subview.getSource();
    } else if (auto reshape = current.getDefiningOp<memref::ReshapeOp>()) {
      current = reshape.getSource();
    } else if (auto expand = current.getDefiningOp<memref::ExpandShapeOp>()) {
      current = expand.getSrc();
    } else if (auto collapse = current.getDefiningOp<memref::CollapseShapeOp>()) {
      current = collapse.getSrc();
    } else if (auto cast = current.getDefiningOp<memref::ReinterpretCastOp>()) {
      current = cast.getSource();
    } else {
      break;
    }
  }
  return current;
}

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
  std::vector<unsigned> currNodeId;
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
        Value sourceMemref = getSourceMemRef(memref);
        if (sourceMemref != memref) {
          memrefAccesses[sourceMemref].insert(node.id);
        }
      }
      for (auto *opInst : collector.storeOpInsts) {
        node.stores.push_back(opInst);
        auto memref = cast<affine::AffineWriteOpInterface>(opInst).getMemRef();
        memrefAccesses[memref].insert(node.id);
        // Also track the source memref if this is a subview or other aliasing operation
        Value sourceMemref = getSourceMemRef(memref);
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
    } else if (auto loadOp = dyn_cast<affine::AffineReadOpInterface>(op)) {
      // Create graph node for top-level load op.
      Node node(nextNodeId++, op);
      node.loads.push_back(op);
      auto memref = cast<affine::AffineReadOpInterface>(op).getMemRef();
      memrefAccesses[memref].insert(node.id);
      // Also track the source memref if this is a subview or other aliasing operation
      Value sourceMemref = getSourceMemRef(memref);
      if (sourceMemref != memref) {
        memrefAccesses[sourceMemref].insert(node.id);
      }
      nodes.insert({node.id, node});
    } else if (auto storeOp = dyn_cast<affine::AffineWriteOpInterface>(op)) {
      // Create graph node for top-level store op.
      Node node(nextNodeId++, op);
      node.stores.push_back(op);
      auto memref = cast<affine::AffineWriteOpInterface>(op).getMemRef();
      memrefAccesses[memref].insert(node.id);
      // Also track the source memref if this is a subview or other aliasing operation
      Value sourceMemref = getSourceMemRef(memref);
      if (sourceMemref != memref) {
        memrefAccesses[sourceMemref].insert(node.id);
      }
      nodes.insert({node.id, node});
    } else if (op->getNumRegions() != 0 ||
               isa<memref::AllocOp, arith::ConstantOp, affine::AffineApplyOp, affine::AffineYieldOp, func::ReturnOp>(
                 op)) {
      // Return false if another region is found (not currently supported).
      return;
    } else {
      // Possibly an arith op
      // Create graph node for top-level op, which could have a memory write
      // side effect.
      Node node(nextNodeId++, op);
      nodes.insert({node.id, node});
    }
  });
}

// Check if a node has a store to a memref that is either the given base memref itself,
// or an alias (subview/reshape) of it.
// This is needed because getStoreOpCount only checks exact memref matches,
// missing stores through subview/reshape aliases.
static bool hasAliasedStoreToMemref(Node *node, Value baseMemref) {
  for (Operation *storeOp : node->stores) {
    if (auto writeOp = dyn_cast<affine::AffineWriteOpInterface>(storeOp)) {
      Value storeMemref = writeOp.getMemRef();
      if (storeMemref == baseMemref) {
        return true;
      }
      if (getSourceMemRef(storeMemref) == baseMemref) {
        return true;
      }
    }
  }
  return false;
}

void MemRefDependenceGraphForFusion::collectLoadNodeIdsAndNonForNodes(
  Value memref, const SetVector<unsigned> &nodeIds, SmallVector<unsigned> &loadNodeIds,
  SmallVector<std::pair<unsigned, bool>, 16> &nonForNodesWithStore) {
  for (unsigned nodeId : nodeIds) {
    Node *node = getNode(nodeId);
    if (auto loadOpInterface = dyn_cast<affine::AffineReadOpInterface>(node->op)) {
      if (loadOpInterface.getMemRef() == memref) {
        loadNodeIds.push_back(nodeId);
      }
    } else if (isa<affine::AffineForOp>(node->op)) {
      for (Operation *loadOpInst : node->loads) {
        if (auto loadOpInterface = dyn_cast<affine::AffineReadOpInterface>(loadOpInst)) {
          if (loadOpInterface.getMemRef() == memref) {
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
  Value memref, const SmallVector<std::pair<unsigned, bool>, 16> &nonForNodesWithStore) {
  for (unsigned i = 0; i < nonForNodesWithStore.size(); ++i) {
    unsigned srcId = nonForNodesWithStore[i].first;
    bool srcHasStore = nonForNodesWithStore[i].second;
    for (unsigned j = i + 1; j < nonForNodesWithStore.size(); ++j) {
      unsigned dstId = nonForNodesWithStore[j].first;
      bool dstHasStore = nonForNodesWithStore[j].second;
      if ((srcHasStore || dstHasStore) && !hasEdge(srcId, dstId, memref)) {
        addEdge(srcId, dstId, memref);
      }
    }
  }
}

void MemRefDependenceGraphForFusion::addMultipleLoadEdges(Value memref, SmallVector<unsigned> &loadNodeIds) {
  if (loadNodeIds.size() < 2) {
    return;
  }
  std::sort(loadNodeIds.begin(), loadNodeIds.end());
  for (unsigned i = 0; i < loadNodeIds.size(); ++i) {
    for (unsigned j = i + 1; j < loadNodeIds.size(); ++j) {
      unsigned srcId = loadNodeIds[i];
      unsigned dstId = loadNodeIds[j];
      if (!hasEdge(srcId, dstId, memref)) {
        addEdge(srcId, dstId, memref);
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
    addAliasedStoreEdges(memref, nonForNodesWithStore);
    addMultipleLoadEdges(memref, loadNodeIds);
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
  auto memrefType = memrefVal.getType().dyn_cast<mlir::MemRefType>();
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

static void classifyIOShape(llvm::ArrayRef<int64_t> inShape, llvm::ArrayRef<int64_t> outShape, bool &hasElementwise,
                            bool &hasBroadcast, bool &hasReshape, bool &hasTranspose) {
  if (inShape.empty() || outShape.empty()) return;

  if (isSameShape(inShape, outShape)) {
    hasElementwise = true;
    return;
  }

  if (inShape.size() == outShape.size()) {
    if (isPermutationShape(inShape, outShape)) {
      hasTranspose = true;
      return;
    }
    if (isBroadcastLike(inShape, outShape)) {
      hasBroadcast = true;
      return;
    }

    hasBroadcast = true;
    return;
  }

  auto inProd = getShapeProduct(inShape);
  auto outProd = getShapeProduct(outShape);
  if (inProd > 0 && outProd > 0 && inProd == outProd) {
    hasReshape = true;
  }
}

OperatorTemplate MemRefDependenceGraphForFusion::getGroupType(const std::vector<unsigned> &nodes) {
  SmallVector<affine::AffineLoadOp> loads;
  SmallVector<affine::AffineStoreOp> stores;
  for (auto nid : nodes) {
    auto op = getNode(nid)->op;
    if (op->hasAttr(kReductionTypeStr)) {
      return OperatorTemplate::Reduce;
    } else if (op->hasAttr(kReductionInitAttr)) {
      return OperatorTemplate::ReduceInit;
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
  bool hasBroadcast = false;
  bool hasReshape = false;
  bool hasTranspose = false;

  for (auto load : loads) {
    Value inMemref = load.getMemRef();
    llvm::SmallVector<int64_t, 4> inShapeBuf;
    auto inShape = getStaticShape(inMemref, inShapeBuf);
    if (inShape.empty()) {
      continue;
    }

    classifyIOShape(inShape, outShape, hasElementwise, hasBroadcast, hasReshape, hasTranspose);
  }

  if (hasTranspose) {
    return OperatorTemplate::Transpose;
  }
  if (hasReshape) {
    return OperatorTemplate::Reshape;
  }
  if (hasBroadcast) {
    return OperatorTemplate::Broadcast;
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

// Collect operations that are outside the target loop but inside its parent loop.
// Separates operations into those before the target loop and those after it.
// parentLoop: the parent loop containing both operations and targetLoop
// targetLoop: the target loop operation
// opsBefore: operations before the target loop
// opsAfter: operations after the target loop
static void collectOpsOutsideTargetLoop(affine::AffineForOp parentLoop, affine::AffineForOp targetLoop,
                                        SmallVector<Operation *, 8> &opsBefore, SmallVector<Operation *, 8> &opsAfter) {
  opsBefore.clear();
  opsAfter.clear();
  auto *body = parentLoop.getBody();

  bool foundTargetLoop = false;
  for (auto &op : body->getOperations()) {
    if (isa<affine::AffineYieldOp>(op)) continue;
    if (&op == targetLoop.getOperation()) {
      foundTargetLoop = true;
      continue;
    }
    // Collect based on position relative to target loop
    if (foundTargetLoop) {
      opsAfter.push_back(&op);
    } else {
      opsBefore.push_back(&op);
    }
  }
}

// Create AffineIfOp with condition: all intermediate loop IVs equal their lower bounds (first iteration)
// Condition: iv[j] == lb[j] for all j in [startLevel, endLevel)
static affine::AffineIfOp createFirstIterationIf(OpBuilder &builder, Location loc, ArrayRef<affine::AffineForOp> loops,
                                                 unsigned startLevel, unsigned endLevel) {
  SmallVector<AffineExpr> exprs;
  SmallVector<bool> eqFlags;
  SmallVector<Value> dimOperands;

  MLIRContext *context = builder.getContext();
  unsigned dimIdx = 0;

  for (unsigned j = startLevel; j < endLevel; ++j) {
    auto loop = loops[j];
    // Condition: iv == lb, expressed as iv - lb == 0
    // For constant lower bound: iv - lb == 0
    if (loop.hasConstantLowerBound()) {
      int64_t lb = loop.getConstantLowerBound();
      AffineExpr ivExpr = getAffineDimExpr(dimIdx++, context);
      exprs.push_back(ivExpr - lb);
      eqFlags.push_back(true);  // equality constraint
      dimOperands.push_back(loop.getInductionVar());
    }
  }

  if (exprs.empty()) {
    return nullptr;
  }

  IntegerSet ifCondSet = IntegerSet::get(dimOperands.size(), 0, exprs, eqFlags);
  return builder.create<affine::AffineIfOp>(loc, ifCondSet, dimOperands, false);
}

// Create AffineIfOp with condition: all intermediate loop IVs equal their upper bounds minus step (last iteration)
// Condition: iv[j] == ub[j] - step[j] for all j in [startLevel, endLevel)
static affine::AffineIfOp createLastIterationIf(OpBuilder &builder, Location loc, ArrayRef<affine::AffineForOp> loops,
                                                unsigned startLevel, unsigned endLevel) {
  SmallVector<AffineExpr> exprs;
  SmallVector<bool> eqFlags;
  SmallVector<Value> dimOperands;

  MLIRContext *context = builder.getContext();
  unsigned dimIdx = 0;

  for (unsigned j = startLevel; j < endLevel; ++j) {
    auto loop = loops[j];
    // Condition: iv == ub - step, expressed as iv - ub + step == 0
    // For constant upper bound: iv - ub + step == 0
    if (loop.hasConstantUpperBound()) {
      int64_t ub = loop.getConstantUpperBound();
      int64_t step = loop.getStepAsInt();
      AffineExpr ivExpr = getAffineDimExpr(dimIdx++, context);
      exprs.push_back(ivExpr - ub + step);
      eqFlags.push_back(true);  // equality constraint
      dimOperands.push_back(loop.getInductionVar());
    }
  }

  if (exprs.empty()) {
    return nullptr;
  }

  IntegerSet ifCondSet = IntegerSet::get(dimOperands.size(), 0, exprs, eqFlags);
  return builder.create<affine::AffineIfOp>(loc, ifCondSet, dimOperands, false);
}

// Sink all operations outside maxLegalFusionDepth loop into the innermost loop.
// This transforms imperfect loop nests into perfect ones by moving operations directly.
// Operations before inner loops are moved to the beginning of the innermost loop body.
// Operations after inner loops are moved to the end of the innermost loop body.
// If wrapWithIf is true, operations are wrapped with AffineIfOp to ensure they execute
// only on the first/last iteration of intermediate loops.
static void sinkOpsOutsideFusionDepth(SmallVector<affine::AffineForOp, 4> &loops, unsigned maxLegalFusionDepth,
                                      bool wrapWithIf = false) {
  if (maxLegalFusionDepth == 0 || maxLegalFusionDepth > loops.size()) {
    return;
  }

  affine::AffineForOp targetLoop = loops[maxLegalFusionDepth - 1];
  auto *targetBody = targetLoop.getBody();
  OpBuilder builder(targetLoop.getContext());

  // Process each level from maxLegalFusionDepth-2 down to 0
  // Level i contains operations between loops[i] and loops[i+1]
  for (int i = static_cast<int>(maxLegalFusionDepth) - 2; i >= 0; --i) {
    SmallVector<Operation *, 8> opsBefore;
    SmallVector<Operation *, 8> opsAfter;
    collectOpsOutsideTargetLoop(loops[i], loops[i + 1], opsBefore, opsAfter);

    if (opsBefore.empty() && opsAfter.empty()) {
      continue;
    }

    if (wrapWithIf) {
      Location loc = loops[i].getLoc();

      // Wrap opsBefore with if: execute only on first iteration of intermediate loops
      if (!opsBefore.empty()) {
        builder.setInsertionPointToStart(targetBody);
        auto ifOp = createFirstIterationIf(builder, loc, loops, i + 1, maxLegalFusionDepth);
        if (ifOp) {
          Block *thenBlock = ifOp.getThenBlock();
          for (auto *op : opsBefore) {
            op->moveBefore(thenBlock->getTerminator());
          }
        } else {
          // Fallback: move without if
          for (auto *op : opsBefore) {
            op->moveBefore(&targetBody->front());
          }
        }
      }

      // Wrap opsAfter with if: execute only on last iteration of intermediate loops
      if (!opsAfter.empty()) {
        builder.setInsertionPoint(targetBody->getTerminator());
        auto ifOp = createLastIterationIf(builder, loc, loops, i + 1, maxLegalFusionDepth);
        if (ifOp) {
          Block *thenBlock = ifOp.getThenBlock();
          for (auto *op : opsAfter) {
            op->moveBefore(thenBlock->getTerminator());
          }
        } else {
          // Fallback: move without if
          for (auto *op : opsAfter) {
            op->moveBefore(targetBody->getTerminator());
          }
        }
      }
    } else {
      // Move operations before inner loop to the beginning of target loop body
      for (auto *op : opsBefore) {
        op->moveBefore(&targetBody->front());
      }

      // Move operations after inner loop to the end of target loop body (before terminator)
      for (auto *op : opsAfter) {
        op->moveBefore(targetBody->getTerminator());
      }
    }
  }
}

// Check if two loops have the same structure (same nesting depth and same bounds)
static bool hasSameLoopStructure(SmallVector<affine::AffineForOp, 4> srcLoops,
                                 SmallVector<affine::AffineForOp, 4> dstLoops) {
  // Check if they have the same nesting depth
  if (srcLoops.size() != dstLoops.size()) {
    return false;
  }

  // Check if corresponding loops have the same bounds
  for (unsigned i = 0; i < srcLoops.size(); ++i) {
    auto srcLoop = srcLoops[i];
    auto dstLoop = dstLoops[i];

    // Check step
    if (srcLoop.getStep() != dstLoop.getStep()) {
      return false;
    }

    // Check lower bound maps
    auto srcLbMap = srcLoop.getLowerBoundMap();
    auto dstLbMap = dstLoop.getLowerBoundMap();
    if (srcLbMap != dstLbMap) {
      return false;
    }

    // Check upper bound maps
    auto srcUbMap = srcLoop.getUpperBoundMap();
    auto dstUbMap = dstLoop.getUpperBoundMap();
    if (srcUbMap != dstUbMap) {
      return false;
    }

    // Check operands (bounds may depend on different values, but maps should match)
    auto srcLbOperands = srcLoop.getLowerBoundOperands();
    auto dstLbOperands = dstLoop.getLowerBoundOperands();
    if (srcLbOperands.size() != dstLbOperands.size()) {
      return false;
    }

    auto srcUbOperands = srcLoop.getUpperBoundOperands();
    auto dstUbOperands = dstLoop.getUpperBoundOperands();
    if (srcUbOperands.size() != dstUbOperands.size()) {
      return false;
    }
  }

  return true;
}

// Overloaded version: Check if two single loops have the same structure
static bool hasSameLoopStructure(affine::AffineForOp srcLoop, affine::AffineForOp dstLoop) {
  SmallVector<affine::AffineForOp, 4> srcLoops;
  collectLoopNest(srcLoop, srcLoops);
  SmallVector<affine::AffineForOp, 4> dstLoops;
  collectLoopNest(dstLoop, dstLoops);
  return hasSameLoopStructure(srcLoops, dstLoops);
}

// Helper function to get the constant value from a fixed index
static std::optional<int64_t> getFixedConstantIndexValue(Operation *op, unsigned dim) {
  AffineMap map;
  if (auto loadOp = dyn_cast<affine::AffineReadOpInterface>(op)) {
    map = loadOp.getAffineMap();
  } else if (auto storeOp = dyn_cast<affine::AffineWriteOpInterface>(op)) {
    map = storeOp.getAffineMap();
  } else {
    return std::nullopt;
  }

  if (dim >= map.getNumResults()) return std::nullopt;

  auto expr = map.getResult(dim);
  if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
    return constExpr.getValue();
  }
  return std::nullopt;
}

// Helper function to check if bounds represent full loop bounds (constant bounds)
// Returns true if both lb and ub are constant expressions
static bool isFullLoopBounds(const AffineMap &lbMap, const AffineMap &ubMap) {
  if (!lbMap || !ubMap) return false;

  if (lbMap.getNumResults() != 1 || ubMap.getNumResults() != 1) return false;

  // Check if both are constant expressions
  auto lbExpr = lbMap.getResult(0);
  auto ubExpr = ubMap.getResult(0);

  return isa<AffineConstantExpr>(lbExpr) && isa<AffineConstantExpr>(ubExpr);
}

// Returns true if slice dimension sliceDim should be adjusted to d_i..d_i+1.
// Case 1: sliceDim >= memrefDim
// Case 2: sliceDim < memrefDim but all shared-memref accesses use the same fixed
// constant index at this dimension.
static bool shouldAdjustSliceDim(unsigned sliceDim, unsigned memrefDim, ArrayRef<Operation *> opsA,
                                 ArrayRef<Operation *> opsB, const affine::ComputationSliceState &sliceUnion) {
  if (sliceDim >= memrefDim) return true;
  bool hasShared = false;
  std::optional<int64_t> fixedVal;
  for (Operation *a : opsA) {
    affine::MemRefAccess srcAccess(a);
    for (Operation *b : opsB) {
      affine::MemRefAccess dstAccess(b);
      if (srcAccess.memref != dstAccess.memref) continue;
      hasShared = true;
      if (sliceDim >= srcAccess.getRank() || sliceDim >= dstAccess.getRank()) return false;
      auto srcFixed = getFixedConstantIndexValue(a, sliceDim);
      auto dstFixed = getFixedConstantIndexValue(b, sliceDim);
      if (!srcFixed || !dstFixed || *srcFixed != *dstFixed) return false;
      if (!fixedVal) {
        fixedVal = *srcFixed;
      } else if (*fixedVal != *srcFixed) {
        return false;
      }
    }
  }
  return hasShared;
}

// Rewrites slice bounds for dimension sliceDim to d_i..d_i+1 using dstIV.
static void applySingleIterationBounds(unsigned sliceDim, Value dstIV, MLIRContext *context,
                                       affine::ComputationSliceState *sliceUnion) {
  unsigned numMapDims = sliceUnion->lbOperands[sliceDim].size();
  unsigned numMapSymbols = 0;
  int dstIVPos = -1;
  for (unsigned i = 0; i < sliceUnion->lbOperands[sliceDim].size(); ++i) {
    if (sliceUnion->lbOperands[sliceDim][i] == dstIV) {
      dstIVPos = static_cast<int>(i);
      break;
    }
  }
  if (dstIVPos < 0) return;
  auto dimExpr = getAffineDimExpr(static_cast<unsigned>(dstIVPos), context);
  sliceUnion->lbs[sliceDim] = AffineMap::get(numMapDims, numMapSymbols, dimExpr);
  sliceUnion->ubs[sliceDim] = AffineMap::get(numMapDims, numMapSymbols, dimExpr + 1);
}

// Adjusts slice bounds when they are constant (e.g. 0 to 10).
// Two cases are adjusted so the fused loop runs one iteration per outer
// (e.g. last dimension 0..10 becomes d3..d3+1):
// 1) Dependency uses lower-rank memref (e.g. 3D) but loop nest has more levels
//    (e.g. 4 loops): adjust dimensions beyond memrefDim (sliceDim >= memrefDim).
// 2) Shared memref has same rank but one dimension is fixed constant
//    (e.g. alloc_10[..., 0]): adjust when all shared accesses use that constant.
void FusionCodeGenHelper::adjustSliceBounds(ArrayRef<Operation *> opsA, ArrayRef<Operation *> opsB, unsigned loopDepth,
                                            unsigned numCommonLoops, bool isBackwardSlice,
                                            affine::ComputationSliceState *sliceUnion) {
  SmallVector<affine::AffineForOp, 4> srcLoops, dstLoops;
  if (!opsA.empty()) affine::getAffineForIVs(*opsA[0], &srcLoops);
  if (!opsB.empty()) affine::getAffineForIVs(*opsB[0], &dstLoops);

  unsigned numSliceLoops = sliceUnion->ivs.size();
  if (numSliceLoops == 0 || srcLoops.empty() || dstLoops.empty()) return;

  unsigned memrefDim = numSliceLoops;
  for (Operation *a : opsA) {
    affine::MemRefAccess srcAccess(a);
    for (Operation *b : opsB) {
      affine::MemRefAccess dstAccess(b);
      if (srcAccess.memref != dstAccess.memref) continue;
      memrefDim = std::min(memrefDim, std::min(srcAccess.getRank(), dstAccess.getRank()));
    }
  }

  MLIRContext *context = opsA[0]->getContext();
  for (unsigned sliceDim = 0; sliceDim < numSliceLoops; ++sliceDim) {
    if (!isFullLoopBounds(sliceUnion->lbs[sliceDim], sliceUnion->ubs[sliceDim])) continue;
    if (sliceDim >= dstLoops.size() || sliceDim >= srcLoops.size()) continue;
    if (!shouldAdjustSliceDim(sliceDim, memrefDim, opsA, opsB, *sliceUnion)) continue;

    Value dstIV = dstLoops[sliceDim].getInductionVar();
    applySingleIterationBounds(sliceDim, dstIV, context, sliceUnion);
  }
}

static bool isDependentLoadOrStoreOp(Operation *op, DenseMap<Value, bool> &values) {
  if (auto loadOp = dyn_cast<affine::AffineReadOpInterface>(op)) {
    return values.count(loadOp.getMemRef()) > 0 && values[loadOp.getMemRef()];
  }
  if (auto storeOp = dyn_cast<affine::AffineWriteOpInterface>(op)) {
    return values.count(storeOp.getMemRef()) > 0;
  }
  return false;
}

// Returns the first operation in range (opA, opB) which has a data dependence on opA.
static Operation *getFirstDependentOpInRange(affine::AffineForOp opA, affine::AffineForOp opB,
                                             DenseMap<Value, bool> &values) {
  // For each opX in block in range (opA, opB), check if there is a data dependence from opA to opX (opA and opX access
  // the same memref and at least one of the accesses is a store).
  Operation *firstDepOp = nullptr;
  for (Block::iterator it = std::next(Block::iterator(opA)); it != Block::iterator(opB); ++it) {
    Operation *opX = &(*it);
    opX->walk([&](Operation *op) {
      if (!firstDepOp && isDependentLoadOrStoreOp(op, values)) {
        firstDepOp = opX;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (firstDepOp) {
      break;
    }
  }
  return firstDepOp;
}

// Returns the last operation opX in range (opA, opB), for which there exists a data dependence from opX to opB.
static Operation *getLastDependentOpInRange(affine::AffineForOp opA, affine::AffineForOp opB,
                                            DenseMap<Value, bool> &values) {
  // For each opX in block in range (opA, opB) in reverse order, check if there is a data dependence from opX to opB:
  // *) opX and opB access the same memref and at least one of the accesses is a store.
  // *) opX produces an SSA Value which is used by opB.
  Operation *lastDepOp = nullptr;
  for (Block::reverse_iterator it = std::next(Block::reverse_iterator(opB)); it != Block::reverse_iterator(opA); ++it) {
    Operation *opX = &(*it);
    opX->walk([&](Operation *op) {
      if (isa<affine::AffineReadOpInterface, affine::AffineWriteOpInterface>(op)) {
        if (isDependentLoadOrStoreOp(op, values)) {
          lastDepOp = opX;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      for (Value value : op->getResults()) {
        for (Operation *user : value.getUsers()) {
          SmallVector<affine::AffineForOp, 4> loops;
          // Check if any loop in loop nest surrounding 'user' is opB.
          getAffineForIVs(*user, &loops);
          if (llvm::is_contained(loops, opB)) {
            lastDepOp = opX;
            return WalkResult::interrupt();
          }
        }
      }
      return WalkResult::advance();
    });
    if (lastDepOp) {
      break;
    }
  }
  return lastDepOp;
}

// Attempts to find a location where, when inserting the fused loop, the original data dependencies of the program
// remain intact.
static Operation *getFusedLoopNestInsertionPoint(affine::AffineForOp forOpA, affine::AffineForOp forOpB,
                                                 DenseMap<Value, bool> &valuesOpA, DenseMap<Value, bool> &valuesOpB) {
  Operation *firstDepOpA = getFirstDependentOpInRange(forOpA, forOpB, valuesOpA);
  Operation *lastDepOpB = getLastDependentOpInRange(forOpA, forOpB, valuesOpB);
  // Block:
  //   ...
  //    opA
  //    ...
  //    lastDepOpB  (lastDepOpB → opB)
  //    ...
  //    firstDepOpA (opA → firstDepOpA)
  //    ...
  //    opB
  //
  // Valid insertion point range: (lastDepOpB, firstDepOpA)
  if (firstDepOpA) {
    if (lastDepOpB) {
      if (firstDepOpA->isBeforeInBlock(lastDepOpB) || firstDepOpA == lastDepOpB) {
        // No valid insertion point exists which preserves dependences.
        return nullptr;
      }
    }
    // Return insertion point in valid range closest to opB.
    return firstDepOpA;
  }
  // No dependences from opA to operation in range (opA, opB), return opB insertion point.
  return forOpB;
}

// Helper function to check if an operation is still valid (not erased or corrupted).
// An operation is considered invalid if:
// - It has no parent operation (detached from IR)
// - It has no operands (AffineLoad/Store requires at least a memref operand)
static bool isOperationValid(Operation *op) {
  if (!op) {
    return false;
  }
  // Check if operation is still attached to the IR (has a valid parent)
  if (!op->getParentOp()) {
    return false;
  }
  // AffineLoad/Store operations must have at least one operand (the memref)
  if (op->getNumOperands() == 0) {
    return false;
  }
  return true;
}

// Collects load and store operations from a single node.
// Populates values map and loadAndStoreOps vector with the collected operations.
static void collectLoadAndStoreOpsFromNode(Node *node, DenseMap<Value, bool> &values,
                                           SmallVector<Operation *, 4> &loadAndStoreOps) {
  if (!node) {
    return;
  }
  // Skip nodes that have been erased during fusion (op is nullptr)
  if (!node->op) {
    return;
  }
  // Add all loads from this node
  for (Operation *loadOp : node->loads) {
    // Skip invalid operations (may have been erased during previous fusion)
    if (!isOperationValid(loadOp)) {
      continue;
    }
    if (auto readOp = dyn_cast<affine::AffineReadOpInterface>(loadOp)) {
      Value memref = readOp.getMemRef();
      values.insert({memref, false});
      loadAndStoreOps.push_back(loadOp);
    }
  }
  // Add all stores from this node
  for (Operation *storeOp : node->stores) {
    // Skip invalid operations (may have been erased during previous fusion)
    if (!isOperationValid(storeOp)) {
      continue;
    }
    if (auto writeOp = dyn_cast<affine::AffineWriteOpInterface>(storeOp)) {
      Value memref = writeOp.getMemRef();
      values.insert({memref, true});
      loadAndStoreOps.push_back(storeOp);
    }
  }
}

void FusionCodeGenHelper::buildStrategyOpsA(const affine::FusionStrategy &strategy,
                                            llvm::ArrayRef<Operation *> loadAndStoreOpsA,
                                            llvm::SmallVector<Operation *, 4> &strategyOpsA) {
  switch (strategy.getStrategy()) {
    case affine::FusionStrategy::Generic:
      strategyOpsA.append(loadAndStoreOpsA.begin(), loadAndStoreOpsA.end());
      break;
    case affine::FusionStrategy::ProducerConsumer:
      for (Operation *op : loadAndStoreOpsA) {
        if (isa<affine::AffineWriteOpInterface>(op)) {
          strategyOpsA.push_back(op);
        }
      }
      break;
    case affine::FusionStrategy::Sibling:
      for (Operation *op : loadAndStoreOpsA) {
        auto load = dyn_cast<affine::AffineReadOpInterface>(op);
        if (load && load.getMemRef() == strategy.getSiblingFusionMemRef()) {
          strategyOpsA.push_back(op);
        }
      }
      break;
  }
}

unsigned FusionCodeGenHelper::tryFusionDepths(llvm::ArrayRef<Operation *> strategyOpsA,
                                              llvm::ArrayRef<Operation *> loadAndStoreOpsB, unsigned dstLoopDepth,
                                              unsigned numCommonLoops, bool isSrcForOpBeforeDstForOp,
                                              llvm::SmallVector<affine::ComputationSliceState, 8> &depthSliceUnions,
                                              const FusionPlan &plan,
                                              const llvm::SmallVector<affine::AffineForOp, 4> &dstLoops) {
  depthSliceUnions.clear();
  depthSliceUnions.resize(dstLoopDepth);
  for (unsigned i = dstLoopDepth; i >= 1; --i) {
    if (plan.loopTransform == LoopTransform::ReplicateIf && dstLoops[i - 1]->hasAttr(kReductionLoopAttr)) {
      continue;
    }
    affine::SliceComputationResult result = affine::computeSliceUnion(
      strategyOpsA, loadAndStoreOpsB, i, numCommonLoops, isSrcForOpBeforeDstForOp, &depthSliceUnions[i - 1]);
    if (result.value == affine::SliceComputationResult::Success) {
      return i;
    }
  }
  return 0;
}

unsigned FusionCodeGenHelper::findMaxLegalFusionDepth(
  affine::AffineForOp srcAffineForOp, affine::AffineForOp dstAffineForOp, unsigned dstLoopDepth,
  const affine::FusionStrategy &strategy, llvm::SmallVector<affine::ComputationSliceState, 8> &depthSliceUnions,
  const FusionPlan &plan, const llvm::SmallVector<affine::AffineForOp, 4> &dstLoops) {
  bool isSrcForOpBeforeDstForOp = srcAffineForOp->isBeforeInBlock(dstAffineForOp);
  auto forOpA = isSrcForOpBeforeDstForOp ? srcAffineForOp : dstAffineForOp;
  auto forOpB = isSrcForOpBeforeDstForOp ? dstAffineForOp : srcAffineForOp;
  const auto &sourceNodeIds = isSrcForOpBeforeDstForOp ? plan.depInfo.sourceOps : plan.depInfo.targetOps;
  const auto &targetNodeIds = isSrcForOpBeforeDstForOp ? plan.depInfo.targetOps : plan.depInfo.sourceOps;

  DenseMap<Value, bool> valuesOpA;
  SmallVector<Operation *, 4> loadAndStoreOpsA;
  for (unsigned nodeId : sourceNodeIds) {
    Node *node = mdg.getNode(nodeId);
    collectLoadAndStoreOpsFromNode(node, valuesOpA, loadAndStoreOpsA);
  }

  DenseMap<Value, bool> valuesOpB;
  SmallVector<Operation *, 4> loadAndStoreOpsB;
  for (unsigned nodeId : targetNodeIds) {
    Node *node = mdg.getNode(nodeId);
    collectLoadAndStoreOpsFromNode(node, valuesOpB, loadAndStoreOpsB);
  }

  if (!getFusedLoopNestInsertionPoint(forOpA, forOpB, valuesOpA, valuesOpB)) {
    llvm::dbgs() << "Fusion would violate dependences in block\n";
    return 0;
  }

  SmallVector<Operation *, 4> strategyOpsA;
  buildStrategyOpsA(strategy, loadAndStoreOpsA, strategyOpsA);

  unsigned numCommonLoops = affine::getNumCommonSurroundingLoops(*srcAffineForOp, *dstAffineForOp);
  unsigned maxLegalFusionDepth = tryFusionDepths(strategyOpsA, loadAndStoreOpsB, dstLoopDepth, numCommonLoops,
                                                 isSrcForOpBeforeDstForOp, depthSliceUnions, plan, dstLoops);

  if (maxLegalFusionDepth > 0) {
    adjustSliceBounds(strategyOpsA, loadAndStoreOpsB, maxLegalFusionDepth, numCommonLoops, isSrcForOpBeforeDstForOp,
                      &depthSliceUnions[maxLegalFusionDepth - 1]);
  }
  return maxLegalFusionDepth;
}

// Helper function to fuse loops into a perfect nest when src has deeper nesting than dst.
// This moves the src loop to dst's position, then clones dst's operations into src's innermost level.
// This is necessary to avoid domination issues - values defined between src and dst must be
// visible at the fusion point.
//
// Example:
// Before:
//   for i { for j { for k { src_ops } } }  // src (deeper nesting, 3 levels)
//   for i { for j { dst_ops } }            // dst (shallower nesting, 2 levels)
//
// After (with fusionDepth=2):
//   for i { for j { for k { src_ops; dst_ops } } }
//
// The key insight: we need to map ALL dst loop IVs to corresponding src loop IVs,
// and only clone the innermost dst loop's operations (not the nested loops themselves).
static void fuseToPerfectNest(affine::AffineForOp srcAffineForOp, affine::AffineForOp dstAffineForOp,
                              SmallVector<affine::AffineForOp, 4> &srcLoops, unsigned fusionDepth) {
  // Step 1: Move src loop to dst's position (right before dst)
  // This ensures all values defined between src and dst are visible
  srcAffineForOp->moveBefore(dstAffineForOp);

  // Step 2: Collect all dst loops to get all loop IVs
  SmallVector<affine::AffineForOp, 4> dstLoops;
  collectLoopNest(dstAffineForOp, dstLoops);

  // Step 3: Get the innermost loop of src (after moving)
  affine::AffineForOp innermostSrcLoop = srcLoops.back();

  // Step 4: Create IRMapping to map ALL dst loop IVs to corresponding src loop IVs
  IRMapping mapper;
  // Map each dst loop IV to the corresponding src loop IV
  // fusionDepth indicates how many outer loops match between src and dst
  for (unsigned i = 0; i < dstLoops.size() && i < fusionDepth; ++i) {
    mapper.map(dstLoops[i].getInductionVar(), srcLoops[i].getInductionVar());
  }

  // Step 5: Find the insertion point in the innermost src loop (before the yield)
  Block *innermostBody = innermostSrcLoop.getBody();
  Operation *terminator = innermostBody->getTerminator();
  OpBuilder builder(innermostBody, terminator->getIterator());

  // Step 6: Clone ONLY the innermost dst loop's operations (not nested loops)
  // This avoids creating duplicate loop structures
  affine::AffineForOp innermostDstLoop = dstLoops.back();
  for (Operation &op : innermostDstLoop.getBody()->getOperations()) {
    if (!isa<affine::AffineYieldOp>(op)) {
      builder.clone(op, mapper);
    }
  }
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

      // Handle nested loops - find corresponding loop in destination and merge operations
      if (auto nestedSrcFor = dyn_cast<affine::AffineForOp>(op)) {
        // Find corresponding nested loop in destination with same structure
        affine::AffineForOp correspondingDstLoop;
        for (auto &dstOp : dstLoop.getBody()->getOperations()) {
          if (auto dstFor = dyn_cast<affine::AffineForOp>(dstOp)) {
            // Check if loops have same structure (same bounds)
            if (hasSameLoopStructure(nestedSrcFor, dstFor)) {
              correspondingDstLoop = dstFor;
              break;
            }
          }
        }

        if (correspondingDstLoop) {
          // Found corresponding loop - merge operations from nested source loop
          // into the corresponding destination loop (recursive call)
          cloneLoopBody(nestedSrcFor, correspondingDstLoop);
        } else {
          // No corresponding loop found - clone the entire nested loop
          builder.clone(op, mapper);
        }
      } else {
        // Clone regular operations
        builder.clone(op, mapper);
      }
    }
  };

  // Get the destination loop at fusion depth
  affine::AffineForOp targetLoop = dstLoops[bestDstLoopDepth - 1];

  // Get the corresponding source loop
  affine::AffineForOp sourceLoop = srcLoops[bestDstLoopDepth - 1];

  // Clone operations from source loop body to destination loop body
  cloneLoopBody(sourceLoop, targetLoop);
}

void FusionCodeGenHelper::doIFuse(unsigned srcId, unsigned dstId, affine::AffineForOp srcAffineForOp,
                                  affine::AffineForOp dstAffineForOp, SmallVector<affine::AffineForOp, 4> &srcLoops,
                                  SmallVector<affine::AffineForOp, 4> &dstLoops) {
  // Helper lambda to check if the outermost loops (up to depth) have matching bounds
  auto checkLoopBoundsMatch = [](SmallVector<affine::AffineForOp, 4> &loops1,
                                 SmallVector<affine::AffineForOp, 4> &loops2, unsigned depth) -> bool {
    if (depth > loops1.size() || depth > loops2.size()) {
      return false;
    }
    for (unsigned i = 0; i < depth; ++i) {
      auto loop1 = loops1[i];
      auto loop2 = loops2[i];
      if (loop1.getLowerBoundMap() != loop2.getLowerBoundMap() ||
          loop1.getUpperBoundMap() != loop2.getUpperBoundMap() || loop1.getStep() != loop2.getStep()) {
        return false;
      }
    }
    return true;
  };

  // Helper lambda to clear erased node's loads and stores
  auto clearErasedNode = [this](unsigned nodeId) {
    if (auto *node = mdg.getNode(nodeId)) {
      node->loads.clear();
      node->stores.clear();
      node->op = nullptr;
    }
  };

  // Handle case where src has deeper nesting than dst
  // Check if src depth > dst depth and outer loops have matching bounds
  if (srcLoops.size() > dstLoops.size() && checkLoopBoundsMatch(srcLoops, dstLoops, dstLoops.size())) {
    // Fuse src (deeper) into dst's position, clone dst's ops into src's innermost loop
    unsigned fusionDepth = dstLoops.size();
    fuseToPerfectNest(srcAffineForOp, dstAffineForOp, srcLoops, fusionDepth);
    dstAffineForOp.erase();
    nodeAlias[dstId] = srcId;
    clearErasedNode(dstId);
    return;
  } else if (srcLoops.size() < dstLoops.size() && checkLoopBoundsMatch(srcLoops, dstLoops, srcLoops.size())) {
    // Handle case where dst has deeper nesting than src
    // Fuse dst (deeper) into src's position, clone src's ops into dst's innermost loop
    unsigned fusionDepth = srcLoops.size();
    fuseToPerfectNest(dstAffineForOp, srcAffineForOp, dstLoops, fusionDepth);
    srcAffineForOp.erase();
    nodeAlias[srcId] = dstId;
    clearErasedNode(srcId);
    return;
  } else if (hasSameLoopStructure(srcLoops, dstLoops)) {
    // Check if loops have the same structure - if so, fuse directly without slice state checks
    // For same-structure loops, use the full depth for fusion
    unsigned bestDstLoopDepth = dstLoops.size();
    assert(bestDstLoopDepth > 0 && "Unexpected loop fusion depth");

    // Perform fusion using the helper function
    performLoopFusion(srcLoops, dstLoops, bestDstLoopDepth);

    srcAffineForOp.erase();
    nodeAlias[srcId] = dstId;
    clearErasedNode(srcId);
    return;
  }
}

void FusionCodeGenHelper::doHFuse(unsigned srcId, unsigned dstId, affine::AffineForOp srcAffineForOp,
                                  affine::AffineForOp dstAffineForOp, const FusionPlan &plan) {
  // Collect loop nests upfront to avoid repeated calculations
  SmallVector<affine::AffineForOp, 4> srcLoops;
  collectLoopNest(srcAffineForOp, srcLoops);
  SmallVector<affine::AffineForOp, 4> dstLoops;
  collectLoopNest(dstAffineForOp, dstLoops);

  if (plan.loopTransform == LoopTransform::Replicate) {
    sinkOpsOutsideFusionDepth(srcLoops, srcLoops.size(), true);
    srcLoops.clear();
    collectLoopNest(srcAffineForOp, srcLoops);
  }

  unsigned dstLoopDepth = dstLoops.size();
  // Check the feasibility of fusing src loop nest into dst loop nest
  // at loop depths in range [1, dstLoopDepth].
  SmallVector<affine::ComputationSliceState, 8> depthSliceUnions;
  affine::FusionStrategy strategy(affine::FusionStrategy::ProducerConsumer);
  unsigned maxLegalFusionDepth =
    findMaxLegalFusionDepth(srcAffineForOp, dstAffineForOp, dstLoopDepth, strategy, depthSliceUnions, plan, dstLoops);

  if (maxLegalFusionDepth == 0) {
    // Pass precomputed loop nests to avoid recalculation in doIFuse
    doIFuse(srcId, dstId, srcAffineForOp, dstAffineForOp, srcLoops, dstLoops);
    return;
  }

  unsigned bestDstLoopDepth = maxLegalFusionDepth;
  assert(bestDstLoopDepth > 0 && "Unexpected loop fusion depth");

  affine::ComputationSliceState &bestSlice = depthSliceUnions[bestDstLoopDepth - 1];
  assert(!bestSlice.isEmpty() && "Missing slice union for depth");

  // Normal fusion - src and dst have compatible depths
  // fuseLoops merges src into dst, so we erase src
  fuseLoops(srcAffineForOp, dstAffineForOp, bestSlice);

  eraseLoopAndCleanupNode(srcId, dstId, srcAffineForOp);
}

void FusionCodeGenHelper::doVFuse(unsigned srcId, unsigned dstId, affine::AffineForOp srcAffineForOp,
                                  affine::AffineForOp dstAffineForOp, const FusionPlan &plan) {
  auto *dstNode = mdg.getNode(dstId);
  mlir::Value memref;
  // Skip if node has been erased during previous fusion
  if (dstNode && dstNode->op) {
    for (auto op : dstNode->loads) {
      // Skip invalid operations (may have been erased during previous fusion)
      if (!isOperationValid(op)) {
        continue;
      }
      auto loadOp = dyn_cast<affine::AffineReadOpInterface>(op);
      if (!loadOp) {
        continue;
      }
      if (loadOp.getMemRef().getDefiningOp()) {
        memref = loadOp.getMemRef();
      }
    }
  }

  // Collect loop nests upfront to avoid repeated calculations
  SmallVector<affine::AffineForOp, 4> srcLoops;
  collectLoopNest(srcAffineForOp, srcLoops);
  SmallVector<affine::AffineForOp, 4> dstLoops;
  collectLoopNest(dstAffineForOp, dstLoops);

  unsigned dstLoopDepth = dstLoops.size();
  SmallVector<affine::ComputationSliceState, 8> depthSliceUnions;
  unsigned maxLegalFusionDepth = 0;
  if (memref) {
    affine::FusionStrategy strategy(memref);
    maxLegalFusionDepth =
      findMaxLegalFusionDepth(srcAffineForOp, dstAffineForOp, dstLoopDepth, strategy, depthSliceUnions, plan, dstLoops);
  } else {
    // No memref found, try generic strategy for structure-based fusion
    affine::FusionStrategy strategy(affine::FusionStrategy::Generic);
    maxLegalFusionDepth =
      findMaxLegalFusionDepth(srcAffineForOp, dstAffineForOp, dstLoopDepth, strategy, depthSliceUnions, plan, dstLoops);
  }

  // Skip if fusion is not feasible at any loop depths.
  if (maxLegalFusionDepth == 0) {
    // Pass precomputed loop nests to avoid recalculation in doIFuse
    doIFuse(srcId, dstId, srcAffineForOp, dstAffineForOp, srcLoops, dstLoops);
    return;
  }

  unsigned bestDstLoopDepth = maxLegalFusionDepth;
  assert(bestDstLoopDepth > 0 && "Unexpected loop fusion depth");
  affine::ComputationSliceState &bestSlice = depthSliceUnions[bestDstLoopDepth - 1];
  assert(!bestSlice.isEmpty() && "Fusion depth has no computed slice union");
  fuseLoops(srcAffineForOp, dstAffineForOp, bestSlice, true);
  eraseLoopAndCleanupNode(srcId, dstId, srcAffineForOp);
}

void FusionCodeGenHelper::eraseLoopAndCleanupNode(unsigned erasedNodeId, unsigned aliasTargetId,
                                                  affine::AffineForOp loopToErase) {
  loopToErase.erase();
  nodeAlias[erasedNodeId] = aliasTargetId;

  // Clear the loads and stores of the erased node to prevent
  // subsequent accesses to invalid operation pointers during fusion analysis
  if (auto *erasedNode = mdg.getNode(erasedNodeId)) {
    erasedNode->loads.clear();
    erasedNode->stores.clear();
    erasedNode->op = nullptr;
  }
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
