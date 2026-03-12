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
    } else if (auto loadOp = dyn_cast<affine::AffineReadOpInterface>(op)) {
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
    } else if (auto storeOp = dyn_cast<affine::AffineWriteOpInterface>(op)) {
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
      if (mlir::affine::getSourceMemRef(storeMemref) == baseMemref) {
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
  if (small.empty())
    return false;

  size_t i = 0;
  size_t j = 0;

  while (i < big.size() && j < small.size()) {
    if (big[i] == small[j])
      ++j;
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
    } else if (op->hasAttr(kReductionInitAttr)) {
      return OperatorTemplate::ReductionInit;
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

static bool isDependentLoadOrStoreOp(Operation *candidate,
                                     DenseMap<Value, bool> &memFlags) {
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
static Operation *getFirstDependentOpInRange(affine::AffineForOp opA,
                                             affine::AffineForOp opB,
                                             DenseMap<Value, bool> &memFlags) {
  // Walk over each opX in the block in the open interval (opA, opB) and look
  // for a dependence from opA to opX, i.e. both touch the same memref and
  // at least one of them performs a write.
  Operation *firstDependent = nullptr;

  for (auto it = std::next(Block::iterator(opA)), e = Block::iterator(opB);
       it != e && !firstDependent; ++it) {
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
static Operation *getLastDependentOpInRange(affine::AffineForOp opA,
                                            affine::AffineForOp opB,
                                            DenseMap<Value, bool> &memFlags) {
  // Traverse operations in (opA, opB) in reverse order and determine the last
  // opX that has a dependence to opB via:
  //  * conflicting memory access on the same memref with at least one store, or
  //  * SSA value produced by opX and eventually used in the loop nest of opB.
  Operation *lastDependent = nullptr;

  for (auto it = std::next(Block::reverse_iterator(opB)),
            e  = Block::reverse_iterator(opA);
       it != e && !lastDependent; ++it) {
    Operation *opX = &*it;

    opX->walk([&](Operation *nested) {
      if (lastDependent) {
        return WalkResult::interrupt();
      }

      if (isa<affine::AffineReadOpInterface,
              affine::AffineWriteOpInterface>(nested)) {
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
          getAffineForIVs(*user, &surroundingLoops);
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
static Operation *getFusedLoopNestInsertionPoint(
    affine::AffineForOp srcLoop, affine::AffineForOp dstLoop,
    DenseMap<Value, bool> &srcMemFlags, DenseMap<Value, bool> &dstMemFlags) {

  Operation *firstFromSrc =
      getFirstDependentOpInRange(srcLoop, dstLoop, srcMemFlags);
  Operation *lastToDst =
      getLastDependentOpInRange(srcLoop, dstLoop, dstMemFlags);

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
  if (firstFromSrc) {
    if (lastToDst &&
        (firstFromSrc == lastToDst ||
         firstFromSrc->isBeforeInBlock(lastToDst))) {
      // There is no position that simultaneously respects both dependence
      // directions.
      return nullptr;
    }
    // Choose the valid insertion point that appears closest to dstLoop.
    return firstFromSrc;
  }

  // If there is no dependence from srcLoop to any operation in (srcLoop, dstLoop),
  // inserting at dstLoop is always safe.
  return dstLoop;
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

// Collects all load and store operations reachable from a single dependency
// graph node. The 'values' map and 'loadAndStoreOps' vector are populated with
// the encountered accesses.
static void collectLoadAndStoreOpsFromNode(
    Node *n, DenseMap<Value, bool> &values,
    SmallVector<Operation *, 4> &loadAndStoreOps) {

  if (!n || !n->op) {
    // Node is null or its root operation has already been removed by fusion.
    return;
  }

  // Record load operations attached to this node.
  for (Operation *loadOp : n->loads) {
    if (!isOperationValid(loadOp)) {
      continue;
    }
    if (auto read = dyn_cast<affine::AffineReadOpInterface>(loadOp)) {
      Value memref = read.getMemRef();
      values.insert({memref, /*isStore*/false});
      loadAndStoreOps.push_back(loadOp);
    }
  }

  // Record store operations attached to this node.
  for (Operation *storeOp : n->stores) {
    if (!isOperationValid(storeOp)) {
      continue;
    }
    if (auto write = dyn_cast<affine::AffineWriteOpInterface>(storeOp)) {
      Value memref = write.getMemRef();
      values.insert({memref, /*isStore*/true});
      loadAndStoreOps.push_back(storeOp);
    }
  }
}

void FusionCodeGenHelper::buildStrategyOpsA(
    const affine::FusionStrategy &strategy,
    llvm::ArrayRef<Operation *> allOpsA,
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

unsigned FusionCodeGenHelper::tryFusionDepths(
    llvm::ArrayRef<Operation *> strategyOpsA,
    llvm::ArrayRef<Operation *> loadAndStoreOpsB,
    unsigned dstLoopDepth, unsigned numCommonLoops,
    bool isSrcForOpBeforeDstForOp,
    llvm::SmallVector<affine::ComputationSliceState, 8> &depthSliceUnions,
    const FusionPlan &plan,
    const llvm::SmallVector<affine::AffineForOp, 4> &dstLoops) {

  depthSliceUnions.clear();
  depthSliceUnions.resize(dstLoopDepth);

  for (unsigned depth = dstLoopDepth; depth >= 1; --depth) {
    if (plan.loopTransform == LoopTransform::ReplicateIf &&
        dstLoops[depth - 1]->hasAttr(kReductionLoopAttr)) {
      continue;
    }

    auto &sliceState = depthSliceUnions[depth - 1];
    affine::SliceComputationResult res = computeSliceUnionAKG(
        strategyOpsA, loadAndStoreOpsB, depth, numCommonLoops,
        isSrcForOpBeforeDstForOp, &sliceState);

    if (res.value == affine::SliceComputationResult::Success) {
      return depth;
    }
  }

  return 0;
}

unsigned FusionCodeGenHelper::findMaxLegalFusionDepth(
    affine::AffineForOp srcAffineForOp, affine::AffineForOp dstAffineForOp,
    unsigned dstLoopDepth, const affine::FusionStrategy &strategy,
    llvm::SmallVector<affine::ComputationSliceState, 8> &depthSliceUnions,
    const FusionPlan &plan,
    const llvm::SmallVector<affine::AffineForOp, 4> &dstLoops) {

  bool srcBeforeDst = srcAffineForOp->isBeforeInBlock(dstAffineForOp);
  affine::AffineForOp loopA =
      srcBeforeDst ? srcAffineForOp : dstAffineForOp;
  affine::AffineForOp loopB =
      srcBeforeDst ? dstAffineForOp : srcAffineForOp;

  const auto &sourceNodeIds =
      srcBeforeDst ? plan.depInfo.sourceOps : plan.depInfo.targetOps;
  const auto &targetNodeIds =
      srcBeforeDst ? plan.depInfo.targetOps : plan.depInfo.sourceOps;

  DenseMap<Value, bool> srcMemFlags;
  SmallVector<Operation *, 4> srcAccesses;
  for (unsigned nodeId : sourceNodeIds) {
    Node *n = mdg.getNode(nodeId);
    collectLoadAndStoreOpsFromNode(n, srcMemFlags, srcAccesses);
  }

  DenseMap<Value, bool> dstMemFlags;
  SmallVector<Operation *, 4> dstAccesses;
  for (unsigned nodeId : targetNodeIds) {
    Node *n = mdg.getNode(nodeId);
    collectLoadAndStoreOpsFromNode(n, dstMemFlags, dstAccesses);
  }

  if (!getFusedLoopNestInsertionPoint(loopA, loopB, srcMemFlags, dstMemFlags)) {
    llvm::dbgs() << "Fusion would violate dependences in block\n";
    return 0;
  }

  SmallVector<Operation *, 4> strategyOpsA;
  buildStrategyOpsA(strategy, srcAccesses, strategyOpsA);

  unsigned commonLoops =
      affine::getNumCommonSurroundingLoops(*srcAffineForOp, *dstAffineForOp);

  unsigned maxDepth = tryFusionDepths(
      strategyOpsA, dstAccesses, dstLoopDepth, commonLoops, srcBeforeDst,
      depthSliceUnions, plan, dstLoops);

  if (maxDepth > 0) {
    adjustSliceBounds(strategyOpsA, dstAccesses, maxDepth, commonLoops,
                      srcBeforeDst,
                      &depthSliceUnions[maxDepth - 1]);
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

  auto depth = std::min(srcLoops.size(), dstLoops.size());
  while (depth != 0 && !checkLoopBoundsMatch(srcLoops, dstLoops, depth)) {
    depth--;
  }
  if (depth == 0) {
    llvm::errs() << "srcLoops and dstLoops have no same loop bounds\n";
    return;
  }
  performLoopFusion(srcLoops, dstLoops, depth);
  dstAffineForOp.dump();
  srcAffineForOp.erase();
  nodeAlias[srcId] = dstId;
  clearErasedNode(srcId);
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
