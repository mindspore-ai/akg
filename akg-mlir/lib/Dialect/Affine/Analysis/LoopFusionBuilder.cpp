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

#include "akg/Dialect/Affine/Analysis/LoopFusionBuilder.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <numeric>
#include <queue>

#include "akg/Analysis/SymbolicShapeAnalysis.h"
#include "akg/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "akg/Utils/GlobalVars.hpp"
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
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Value.h"
#include "akg/Utils/Constants.h"

namespace mlir {
namespace akg {

using llvm::DenseMap;
using llvm::DenseSet;
using llvm::raw_ostream;
using llvm::SetVector;
using llvm::SmallVector;
using mlir::kReductionInitAttr;
using mlir::kReductionTypeStr;

namespace {
constexpr int kInvalidNodeId = -1;
constexpr int kNoMismatchPos = -1;
constexpr int kMultipleMismatchPos = -2;
constexpr int64_t kInvalidShapeProduct = -1;
constexpr int kInvalidRank = -1;
constexpr int kUnmappedLevel = -1;

constexpr size_t kMinNodesForEdge = 2;
constexpr int64_t kBroadcastDimSize = 1;
constexpr size_t kElementwiseStoreCount = 1;
constexpr unsigned kExtraDimDepthDiff = 1;
constexpr unsigned kMaxSiblingForLoops = 1;

constexpr unsigned kGuardDimCount = 1;
constexpr unsigned kGuardSymbolCount = 0;
constexpr unsigned kGuardDimIndex = 0;
constexpr unsigned kMinFusionDepth = 1;
constexpr unsigned kDefaultNumCommonLoops = 0;
}  // namespace

// ===----------------------------------------------------------------------===//
// LoopNestStateCollector implementation
// ===----------------------------------------------------------------------===//

// Collect loop nest state from operations in the block
void LoopNestStateCollector::collect(Operation *opToWalk) {
  opToWalk->walk([this](Operation *op) {
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

// ===----------------------------------------------------------------------===//
// MemRefDependenceGraphForFusion implementation
// ===----------------------------------------------------------------------===//

// --- Static helpers for MemRefDependenceGraphForFusion ---

// Check if a node has a store to a memref that is either the given base memref itself,
// or an alias (reshape) of it.
// This is needed because getStoreOpCount only checks exact memref matches,
// missing stores through reshape aliases.
static bool isSameOrAliasedMemRef(Value accessMemref, Value baseMemref) {
  return affine::getSourceMemRef(accessMemref) == affine::getSourceMemRef(baseMemref);
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
  return kInvalidNodeId;
}

static llvm::ArrayRef<int64_t> getStaticShape(Value memrefVal,
                                              llvm::SmallVector<int64_t, kSmallVectorSizeFour> &shapeBuf) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(memrefVal.getType());
  if (!memrefType || !memrefType.hasStaticShape()) {
    return {};
  }
  shapeBuf.assign(memrefType.getShape().begin(), memrefType.getShape().end());
  return shapeBuf;
}

static int64_t getShapeProduct(llvm::ArrayRef<int64_t> shape) {
  if (shape.empty()) {
    return kInvalidShapeProduct;
  }
  int64_t prod = 1;
  for (int64_t d : shape) {
    if (d < 0) {
      return kInvalidShapeProduct;
    }
    prod *= d;
  }
  return prod;
}

static bool isSameShape(llvm::ArrayRef<int64_t> a, llvm::ArrayRef<int64_t> b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

static bool isPermutationShape(llvm::ArrayRef<int64_t> a, llvm::ArrayRef<int64_t> b) {
  if (a.size() != b.size()) {
    return false;
  }
  llvm::SmallVector<int64_t, kSmallVectorSizeFour> aSorted(a.begin(), a.end());
  llvm::SmallVector<int64_t, kSmallVectorSizeFour> bSorted(b.begin(), b.end());
  std::sort(aSorted.begin(), aSorted.end());
  std::sort(bSorted.begin(), bSorted.end());
  return isSameShape(aSorted, bSorted);
}

static bool isBroadcastLike(llvm::ArrayRef<int64_t> inShape, llvm::ArrayRef<int64_t> outShape) {
  if (inShape.size() != outShape.size()) {
    return false;
  }

  bool diffFound = false;
  for (size_t i = 0; i < inShape.size(); ++i) {
    auto inD = inShape[i];
    auto outD = outShape[i];
    if (inD == outD) {
      continue;
    }

    diffFound = true;
    if (inD != kBroadcastDimSize) {
      return false;
    }
  }
  return diffFound;
}

static bool isReshape(llvm::ArrayRef<int64_t> inShape, llvm::ArrayRef<int64_t> outShape) {
  if (inShape.size() == outShape.size()) {
    return false;
  }

  auto inProd = getShapeProduct(inShape);
  auto outProd = getShapeProduct(outShape);
  return inProd > 0 && outProd > 0 && inProd == outProd;
}

static bool isSubsequence(llvm::ArrayRef<int64_t> big, llvm::ArrayRef<int64_t> small) {
  if (small.empty()) {
    return false;
  }

  size_t i = 0;
  size_t j = 0;

  while (i < big.size() && j < small.size()) {
    if (big[i] == small[j]) {
      ++j;
    }
    ++i;
  }

  return j == small.size();
}

static bool isSubsequencePart(llvm::ArrayRef<int64_t> a, llvm::ArrayRef<int64_t> b) {
  if (a.size() == b.size()) {
    return false;
  }

  return isSubsequence(a, b) || isSubsequence(b, a);
}

// --- MemRefDependenceGraphForFusion member functions ---

// Get group by group ID, returns nullptr if group doesn't exist
GroupPtr MemRefDependenceGraphForFusion::getGroup(unsigned groupId) {
  return groups.find(groupId) != groups.end() ? groups[groupId] : nullptr;
}

// Get group by node ID, returns nullptr if node is not in any group
GroupPtr MemRefDependenceGraphForFusion::getGroupByNode(unsigned nodeId) {
  return nodeToGroup.find(nodeId) != nodeToGroup.end() ? nodeToGroup[nodeId] : nullptr;
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
  block->walk([this, &memrefAccesses](Operation *op) {
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
      // Check this condition
      if (collector.hasNonAffineRegionOp) {
        return;
      }
      Node node(nextNodeId++, op);
      for (auto *opInst : collector.loadOpInsts) {
        node.loads.push_back(opInst);
        auto memref = cast<affine::AffineReadOpInterface>(opInst).getMemRef();
        memrefAccesses[memref].insert(node.id);
        // Track source memreffor other alias operations as well.
        Value sourceMemref = affine::getSourceMemRef(memref);
        if (sourceMemref != memref) {
          memrefAccesses[sourceMemref].insert(node.id);
        }
      }
      for (auto *opInst : collector.storeOpInsts) {
        node.stores.push_back(opInst);
        auto memref = cast<affine::AffineWriteOpInterface>(opInst).getMemRef();
        memrefAccesses[memref].insert(node.id);
        // Track source memreffor other alias operations as well.
        Value sourceMemref = affine::getSourceMemRef(memref);
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
        if (innerId != kInvalidNodeId) {
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
      // Track source memreffor other alias operations as well.
      Value sourceMemref = affine::getSourceMemRef(memref);
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
      // Track source memreffor other alias operations as well.
      Value sourceMemref = affine::getSourceMemRef(memref);
      if (sourceMemref != memref) {
        memrefAccesses[sourceMemref].insert(node.id);
      }
      nodes.insert({node.id, node});
    } else if (isa<memref::SubViewOp, memref::MemorySpaceCastOp, memref::ExpandShapeOp, memref::CollapseShapeOp,
                   memref::ReshapeOp, memref::ReinterpretCastOp>(op)) {
      Node node(nextNodeId++, op);
      Value result = op->getResult(0);
      Value source = affine::getSourceMemRef(result);
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

void MemRefDependenceGraphForFusion::collectForNodeLoadIds(Operation *loadOpInst, Value memref,
                                                           SmallVector<unsigned> &loadNodeIds) {
  auto loadOpInterface = dyn_cast<affine::AffineReadOpInterface>(loadOpInst);
  if (!loadOpInterface) {
    return;
  }
  if (!isSameOrAliasedMemRef(loadOpInterface.getMemRef(), memref)) {
    return;
  }
  int loadNodeId = getNodeId(loadOpInst);
  if (loadNodeId != kInvalidNodeId) {
    if (std::find(loadNodeIds.begin(), loadNodeIds.end(), loadNodeId) == loadNodeIds.end()) {
      loadNodeIds.push_back(loadNodeId);
    }
    return;
  }
  Node loadNode(nextNodeId++, loadOpInst);
  loadNode.loads.push_back(loadOpInst);
  nodes.insert({loadNode.id, loadNode});
  loadNodeIds.push_back(loadNode.id);
}

void MemRefDependenceGraphForFusion::collectLoadNodeIdsAndNonForNodes(
  Value memref, const SetVector<unsigned> &nodeIds, SmallVector<unsigned> &loadNodeIds,
  SmallVector<std::pair<unsigned, bool>, kSmallVectorSizeSixteen> &nonForNodesWithStore,
  SmallVector<std::pair<unsigned, bool>, kSmallVectorSizeEight> &forNodesWithStore) {
  for (unsigned nodeId : nodeIds) {
    Node *node = getNode(nodeId);
    if (auto loadOpInterface = dyn_cast<affine::AffineReadOpInterface>(node->op)) {
      if (isSameOrAliasedMemRef(loadOpInterface.getMemRef(), memref)) {
        loadNodeIds.push_back(nodeId);
      }
    } else if (isa<affine::AffineForOp>(node->op)) {
      for (Operation *loadOpInst : node->loads) {
        collectForNodeLoadIds(loadOpInst, memref, loadNodeIds);
      }
      forNodesWithStore.push_back({nodeId, hasAliasedStoreToMemref(node, memref)});
    }
    if (!isa<affine::AffineForOp>(node->op)) {
      nonForNodesWithStore.push_back({nodeId, hasAliasedStoreToMemref(node, memref)});
    }
  }
}

void MemRefDependenceGraphForFusion::addAliasedStoreEdges(
  Value memref, const SetVector<unsigned> &nodeIds,
  const SmallVector<std::pair<unsigned, bool>, 16> &nonForNodesWithStore,
  const SmallVector<std::pair<unsigned, bool>, 8> &forNodesWithStore, bool hasLoadsForBaseMemref) {
  auto addStoreEdges = [&](const auto &nodes, bool requireBothStores) {
    for (unsigned i = 0; i < nodes.size(); ++i) {
      unsigned srcId = nodes[i].first;
      bool srcHasStore = nodes[i].second;
      for (unsigned j = i + 1; j < nodes.size(); ++j) {
        unsigned dstId = nodes[j].first;
        bool dstHasStore = nodes[j].second;
        bool hasStore = requireBothStores ? (srcHasStore && dstHasStore) : (srcHasStore || dstHasStore);
        if (hasStore && !hasEdge(srcId, dstId, memref)) {
          int forLoopNodeId = getEnclosingForLoopNodeId(*this, nodeIds, dstId);
          unsigned edgeLoopDepth = computeMemrefLoopDepth(forLoopNodeId, memref);
          addEdge(srcId, dstId, memref, edgeLoopDepth);
        }
      }
    }
  };

  addStoreEdges(nonForNodesWithStore, false);
  addStoreEdges(forNodesWithStore, true);
}

void MemRefDependenceGraphForFusion::addMultipleLoadEdges(Value memref, const SetVector<unsigned> &nodeIds,
                                                          SmallVector<unsigned> &loadNodeIds) {
  if (loadNodeIds.size() < kMinNodesForEdge) {
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
    if (nodeIds.size() < kMinNodesForEdge) {
      continue;
    }
    SmallVector<unsigned> loadNodeIds;
    SmallVector<std::pair<unsigned, bool>, kSmallVectorSizeSixteen> nonForNodesWithStore;
    SmallVector<std::pair<unsigned, bool>, kSmallVectorSizeEight> forNodesWithStore;
    collectLoadNodeIdsAndNonForNodes(memref, nodeIds, loadNodeIds, nonForNodesWithStore, forNodesWithStore);
    bool hasLoadsForBaseMemref = !loadNodeIds.empty();
    addAliasedStoreEdges(memref, nodeIds, nonForNodesWithStore, forNodesWithStore, hasLoadsForBaseMemref);
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
    os << "Group " << g->groupId << " (GroupTemplate " << groupTemplateString << ") IsGlobalOut ("
       << static_cast<int>(g->isGlobalOut) << ") root is " << g->rootId << " has " << g->nodesId.size()
       << " nodes inside: [";
    for (auto nid : g->nodesId) {
      os << nid << ", ";
    }
    os << "]\n";
  }
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
      // Cannot identify reshape by op after FoldMemRefAliasOps pass
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

  if (stores.size() != kElementwiseStoreCount) {
    return OperatorTemplate::Default;
  }

  if (loads.empty()) {
    return OperatorTemplate::Elementwise;
  }

  auto store = stores.front();
  Value outMemref = store.getMemRef();

  llvm::SmallVector<int64_t, kSmallVectorSizeFour> outShapeBuf;
  auto outShape = getStaticShape(outMemref, outShapeBuf);
  if (outShape.empty()) {
    return OperatorTemplate::Default;
  }

  bool hasElementwise = false;

  for (auto load : loads) {
    Value inMemref = load.getMemRef();
    llvm::SmallVector<int64_t, kSmallVectorSizeFour> inShapeBuf;
    auto inShape = getStaticShape(inMemref, inShapeBuf);
    if (inShape.empty()) {
      continue;
    }

    if (isSameShape(inShape, outShape)) {
      hasElementwise = true;
      continue;
    }
    hasElementwise = false;

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

// Gets all dependent groups (predecessor groups) of a given group.
void MemRefDependenceGraphForFusion::collectDepGroupsForNode(unsigned nodeId, unsigned groupId,
                                                             std::unordered_set<unsigned> &depGroups) {
  std::vector<unsigned> predecessorIds;
  bool isLoad = isa<affine::AffineLoadOp>(getNode(nodeId)->op);
  getPredecessorNodes(nodeId, predecessorIds);
  for (auto predId : predecessorIds) {
    if (isLoad && isa<affine::AffineLoadOp>(getNode(predId)->op)) {
      continue;
    }
    auto predGroup = getGroupByNode(predId);
    if (predGroup != nullptr && predGroup->groupId != groupId) {
      depGroups.insert(predGroup->groupId);
    }
  }
}

std::unordered_set<unsigned> MemRefDependenceGraphForFusion::getDependentGroups(unsigned groupId) {
  auto it = dependentGroupsCache.find(groupId);
  if (it != dependentGroupsCache.end()) {
    return it->second;
  }

  std::unordered_set<unsigned> depGroups;
  auto group = getGroup(groupId);
  if (group == nullptr) {
    return depGroups;
  }

  for (auto nodeId : group->nodesId) {
    collectDepGroupsForNode(nodeId, groupId, depGroups);
  }
  return depGroups;
}

void MemRefDependenceGraphForFusion::precomputeDependentGroups() {
  dependentGroupsCache.clear();

  for (const auto &groupPair : groups) {
    unsigned groupId = groupPair.first;
    auto group = groupPair.second;

    std::unordered_set<unsigned> depGroups;
    for (auto nodeId : group->nodesId) {
      collectDepGroupsForNode(nodeId, groupId, depGroups);
    }
    dependentGroupsCache[groupId] = std::move(depGroups);
  }
}

// ===----------------------------------------------------------------------===//
// FusionLoopNestInfo implementation
// ===----------------------------------------------------------------------===//

// Collect all loops in a loop nest starting from the given loop
// This includes the loop itself and all nested loops, ordered from outermost to innermost
static void collectLoopNest(affine::AffineForOp rootLoop,
                            SmallVector<affine::AffineForOp, kSmallVectorSizeFour> &loops) {
  loops.clear();

  // First, collect all parent loops (from outermost to the root loop's parent)
  SmallVector<affine::AffineForOp, kSmallVectorSizeFour> parentLoops;
  affine::getAffineForIVs(*rootLoop, &parentLoops);
  loops.append(parentLoops.begin(), parentLoops.end());

  // Then add the root loop itself
  loops.push_back(rootLoop);

  // Finally, recursively collect nested loops in depth-first order
  std::function<void(affine::AffineForOp)> collectNested = [&loops, &collectNested](affine::AffineForOp loop) {
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

  SmallVector<affine::AffineForOp, kSmallVectorSizeFour> perfectBand;
  affine::getPerfectlyNestedLoops(perfectBand, rootOp);
  perfectDepth = static_cast<unsigned>(perfectBand.size());
  isPerfect = (perfectDepth == loopDepth);
}

// ===----------------------------------------------------------------------===//
// FusionCodeGenHelper implementation
// ===----------------------------------------------------------------------===//

// --- Static helpers for FusionCodeGenHelper ---

static bool isDependentLoadOrStoreOp(Operation *candidate, DenseMap<Value, bool> &memFlags) {
  auto lookupMemFlag = [&memFlags](Value memref) -> DenseMap<Value, bool>::iterator {
    auto it = memFlags.find(memref);
    if (it != memFlags.end()) {
      return it;
    }
    Value sourceMemref = affine::getSourceMemRef(memref);
    return memFlags.find(sourceMemref);
  };

  auto asLoad = dyn_cast<affine::AffineReadOpInterface>(candidate);
  if (asLoad) {
    auto it = lookupMemFlag(asLoad.getMemRef());
    return it != memFlags.end() && it->second;
  }

  auto asStore = dyn_cast<affine::AffineWriteOpInterface>(candidate);
  if (asStore) {
    return lookupMemFlag(asStore.getMemRef()) != memFlags.end();
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

  for (auto it = std::next(Block::iterator(opA)), e = Block::iterator(opB); it != e && (firstDependent == nullptr);
       ++it) {
    Operation *opX = &*it;

    opX->walk([&firstDependent, &memFlags, &opX](Operation *nested) {
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

  for (auto it = std::next(Block::reverse_iterator(opB)), e = Block::reverse_iterator(opA);
       it != e && (lastDependent == nullptr); ++it) {
    Operation *opX = &*it;

    opX->walk([&lastDependent, &memFlags, &opB, &opX](Operation *nested) {
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
          SmallVector<affine::AffineForOp, kSmallVectorSizeFour> surroundingLoops;
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
  //   ...
  //   srcLoop
  //   ...
  //   lastToDst   (lastToDst → dstLoop)
  //   ...
  //   firstFromSrc (srcLoop → firstFromSrc)
  //   ...
  //   dstLoop
  // Legal insertion range lies strictly inside: (lastToDst, firstFromSrc).
  if ((firstFromSrc != nullptr) && (lastToDst != nullptr) && firstFromSrc->isBeforeInBlock(lastToDst)) {
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
  if (candidate == nullptr) {
    return false;
  }

  if (candidate->getParentOp() == nullptr) {
    return false;
  }

  if (candidate->getNumOperands() == 0) {
    return false;
  }

  return true;
}

// Collects affine load/store operations from a loop that access any of 'memrefs'
// into 'memFlags' and 'loadAndStoreOps'. Skips invalid ops and non-affine accesses.
static void collectLoadAndStoreOpsFromOps(Value depMemref, affine::AffineForOp loopOp, DenseMap<Value, bool> &memFlags,
                                          SmallVector<Operation *, kSmallVectorSizeFour> &loadAndStoreOps) {
  if (!loopOp || !depMemref) {
    return;
  }

  Value sourceMemref = affine::getSourceMemRef(depMemref);

  auto recordMemFlag = [&memFlags](Value memref, bool isStore) {
    auto updateFlag = [&memFlags, &isStore](Value key) {
      auto it = memFlags.find(key);
      if (it == memFlags.end()) {
        memFlags.insert({key, isStore});
      } else {
        it->second = it->second || isStore;
      }
    };

    updateFlag(memref);
    Value src = affine::getSourceMemRef(memref);
    if (src != memref) {
      updateFlag(src);
    }
  };

  loopOp->walk([&sourceMemref, &recordMemFlag, &loadAndStoreOps](Operation *op) {
    if (!isOperationValid(op)) {
      return;
    }
    if (auto read = dyn_cast<affine::AffineReadOpInterface>(op)) {
      Value memref = read.getMemRef();
      if (affine::getSourceMemRef(memref) == sourceMemref) {
        recordMemFlag(memref, false);
        loadAndStoreOps.push_back(op);
      }
    } else if (auto write = dyn_cast<affine::AffineWriteOpInterface>(op)) {
      Value memref = write.getMemRef();
      if (affine::getSourceMemRef(memref) == sourceMemref) {
        recordMemFlag(memref, true);
        loadAndStoreOps.push_back(op);
      }
    }
  });
}

// Collects load/store access ops from src and dst loops that touch depMemref.
// Returns false if depMemref is null.
static bool collectFusionAccesses(Value depMemref, affine::AffineForOp srcForOp, affine::AffineForOp dstForOp,
                                  FusionAccessInfo &info) {
  if (!depMemref) {
    llvm::dbgs() << "No memrefs found for fusion\n";
    return false;
  }
  collectLoadAndStoreOpsFromOps(depMemref, srcForOp, info.srcMemFlags, info.srcAccesses);
  collectLoadAndStoreOpsFromOps(depMemref, dstForOp, info.dstMemFlags, info.dstAccesses);
  return true;
}

// Fusion by induction variable requires the producer-written and consumer-read views to share the
// same logical shape. For dynamic shapes this correspondence is only available via SymShapeAttr
// (loop bounds have been rewritten to primes and no longer encode it). If the ranks differ or
// SymShapeAttr is missing/mismatched, we skip fusion.
static bool isDependenceAxisRelationUnresolved(const SmallVector<Operation *, kSmallVectorSizeFour> &srcAccesses,
                                               const SmallVector<Operation *, kSmallVectorSizeFour> &dstAccesses) {
  auto &symAnalysis = SymbolicShapeAnalysis::getInstance();
  auto memrefTypeOf = [](Operation *op) -> MemRefType {
    Value memref;
    if (auto read = dyn_cast<affine::AffineReadOpInterface>(op)) {
      memref = read.getMemRef();
    } else if (auto write = dyn_cast<affine::AffineWriteOpInterface>(op)) {
      memref = write.getMemRef();
    }
    return memref ? dyn_cast<MemRefType>(memref.getType()) : MemRefType();
  };

  for (Operation *srcOp : srcAccesses) {
    MemRefType srcType = memrefTypeOf(srcOp);
    if (!srcType) {
      continue;
    }
    auto srcSym = symAnalysis.getSymbolicShape(srcType);
    for (Operation *dstOp : dstAccesses) {
      MemRefType dstType = memrefTypeOf(dstOp);
      if (!dstType) {
        continue;
      }
      // Different rank: the two views cannot describe the same logical shape.
      if (srcType.getRank() != dstType.getRank()) {
        return true;
      }
      // Fully static and same rank: concrete bounds are validated by the regular path.
      if (srcType.hasStaticShape() && dstType.hasStaticShape()) {
        continue;
      }
      // Dynamic shape: the axis correspondence must be established via SymShapeAttr.
      auto dstSym = symAnalysis.getSymbolicShape(dstType);
      if (!srcSym.has_value() || !dstSym.has_value() || *srcSym != *dstSym) {
        return true;
      }
    }
  }
  return false;
}

// Helper function to check if the outermost loops (up to depth) have matching bounds.
static bool checkLoopBoundsMatch(const SmallVector<affine::AffineForOp, kSmallVectorSizeFour> &loops1,
                                 const SmallVector<affine::AffineForOp, kSmallVectorSizeFour> &loops2, unsigned depth) {
  if (depth > loops1.size() || depth > loops2.size()) {
    return false;
  }
  for (unsigned i = 0; i < depth; ++i) {
    if (!affine::loopBoundsMatch(loops1[i], loops2[i])) {
      return false;
    }
  }
  return true;
}

// Collect aligned axis chains: pair src's last for-child with dst's first for-child at each level for deeper merging.
static void collectAlignedAxisChains(affine::AffineForOp srcRoot, affine::AffineForOp dstRoot,
                                     SmallVector<affine::AffineForOp, kSmallVectorSizeFour> &srcChain,
                                     SmallVector<affine::AffineForOp, kSmallVectorSizeFour> &dstChain) {
  srcChain.clear();
  dstChain.clear();
  affine::AffineForOp s = srcRoot;
  affine::AffineForOp d = dstRoot;
  while (s && d) {
    srcChain.push_back(s);
    dstChain.push_back(d);

    affine::AffineForOp srcNext;
    for (auto &op : *s.getBody()) {
      if (auto f = dyn_cast<affine::AffineForOp>(op)) {
        srcNext = f;
      }
    }
    affine::AffineForOp dstNext;
    for (auto &op : *d.getBody()) {
      if (auto f = dyn_cast<affine::AffineForOp>(op)) {
        dstNext = f;
        break;
      }
    }
    s = srcNext;
    d = dstNext;
  }
}

// Clone src body ops (excluding yield and next-axis child) into dst at each
// aligned level, placing src ops before dst ops.
static void performLoopFusion(SmallVector<affine::AffineForOp, kSmallVectorSizeFour> srcChain,
                              SmallVector<affine::AffineForOp, kSmallVectorSizeFour> dstChain,
                              unsigned bestDstLoopDepth) {
  IRMapping mapper;
  for (unsigned i = 0; i < bestDstLoopDepth && i < srcChain.size() && i < dstChain.size(); ++i) {
    mapper.map(srcChain[i].getInductionVar(), dstChain[i].getInductionVar());
  }

  for (unsigned i = 0; i < bestDstLoopDepth; ++i) {
    affine::AffineForOp srcLoop = srcChain[i];
    affine::AffineForOp dstLoop = dstChain[i];
    Operation *srcAxisChildOp = (i + 1 < bestDstLoopDepth) ? srcChain[i + 1].getOperation() : nullptr;

    Block::iterator insertPoint = dstLoop.getBody()->begin();
    while (insertPoint != dstLoop.getBody()->end() && isa<affine::AffineYieldOp>(*insertPoint)) {
      ++insertPoint;
    }
    OpBuilder builder(dstLoop.getBody(), insertPoint);

    for (Operation &op : srcLoop.getBody()->getOperations()) {
      if (isa<affine::AffineYieldOp>(op)) {
        continue;
      }
      if ((srcAxisChildOp != nullptr) && &op == srcAxisChildOp) {
        continue;
      }

      builder.clone(op, mapper);
    }
  }
}

// Effective dep depth for ProducerConsumer fusion:
// (1) Data-driven: innermost loop whose IV appears in an access.
// (2) Structural extension: past (1), extend while loops[i] and partnerLoops[i] share lb/ub/step.
static unsigned computeEffectiveDepDepth(ArrayRef<affine::AffineForOp> loops, ArrayRef<Operation *> accesses,
                                         ArrayRef<affine::AffineForOp> partnerLoops) {
  // 0 = no IV reference found
  unsigned depth = 0;
  for (unsigned i = 0; i < loops.size(); ++i) {
    Value iv = const_cast<affine::AffineForOp &>(loops[i]).getInductionVar();
    for (Operation *op : accesses) {
      bool referenced = false;
      if (auto loadOp = dyn_cast<affine::AffineReadOpInterface>(op)) {
        referenced = llvm::is_contained(loadOp.getMapOperands(), iv);
      } else if (auto storeOp = dyn_cast<affine::AffineWriteOpInterface>(op)) {
        referenced = llvm::is_contained(storeOp.getMapOperands(), iv);
      }
      if (referenced) {
        // 1-indexed; keep overwriting -> innermost wins
        depth = i + 1;
        break;
      }
    }
  }
  unsigned maxExt = std::min(loops.size(), partnerLoops.size());
  while (depth < maxExt && affine::loopBoundsMatch(loops[depth], partnerLoops[depth])) {
    ++depth;
  }
  return depth;
}

// --- Imperfect loop fusion: src→dst stage injection ---
// Spine: root→anchor for-loop chain. src/dst spines pair level-by-level for IV remapping;
// off-spine siblings form per-level "stage" ops. src (guest) stages are cloned into dst (host).

// Returns spine from root to anchor, truncated to `depth`. Empty if anchor not inside root or chain too short.
static SmallVector<affine::AffineForOp, kSmallVectorSizeFour> buildSpine(Operation *anchor, affine::AffineForOp root,
                                                                         unsigned depth) {
  SmallVector<affine::AffineForOp, kSmallVectorSizeFour> spine;
  if ((anchor == nullptr) || !root) {
    return spine;
  }
  bool reachedRoot = false;
  for (Operation *cur = anchor->getParentOp(); cur != nullptr; cur = cur->getParentOp()) {
    if (auto f = dyn_cast<affine::AffineForOp>(cur)) {
      spine.push_back(f);
      if (f == root) {
        reachedRoot = true;
        break;
      }
    }
  }
  if (!reachedRoot) {
    return {};
  }
  std::reverse(spine.begin(), spine.end());
  if (spine.size() < depth) {
    return {};
  }
  spine.truncate(depth);
  return spine;
}

// Split parentLevel.body off-spine ops by src program-order position relative to spineChild.
// After fusion the spine child becomes the merged leaf, so preSpine ops (those that ran before
// spineChild in src) must clone BEFORE dst spine child, and postSpine ops (those that ran after)
// must clone AFTER it — otherwise off-spine consumers can land ahead of the leaf-merged producer.
static void collectStageOps(affine::AffineForOp parentLevel, affine::AffineForOp spineChild,
                            SmallVector<Operation *, kSmallVectorSizeEight> &preSpine,
                            SmallVector<Operation *, kSmallVectorSizeEight> &postSpine) {
  bool seen = false;
  for (Operation &op : parentLevel.getBody()->without_terminator()) {
    if (&op == spineChild.getOperation()) {
      seen = true;
      continue;
    }
    (seen ? postSpine : preSpine).push_back(&op);
  }
}

// Insertion point in dstLevel.body for cloning src stages.
// src's off-spine stages ran fully before dst in the original program, so any
// memref they touch must be settled before dst sees it. Place stages BEFORE
// the earliest dst op that has any memref-conflict with srcStages:
//   - dst write of M where M is in srcReads ∪ srcWrites (WAR/WAW)
//   - dst read of M where M is in srcWrites (RAW)
// If no dst op conflicts, default to just before spineChild — the latest
// safe position keeps cloned stages adjacent to the spine continuation.
static Block::iterator computeStageInsertPoint(affine::AffineForOp dstLevel, affine::AffineForOp dstSpineChild,
                                               ArrayRef<Operation *> srcStages) {
  Block *dstBody = dstLevel.getBody();
  auto childIt = Block::iterator(dstSpineChild);
  if (srcStages.empty() || dstBody->begin() == childIt) {
    return childIt;
  }

  DenseSet<Value> srcReads;
  DenseSet<Value> srcWrites;
  for (Operation *sop : srcStages) {
    sop->walk([&srcReads, &srcWrites](Operation *nested) {
      if (auto r = dyn_cast<affine::AffineReadOpInterface>(nested)) {
        srcReads.insert(affine::getSourceMemRef(r.getMemRef()));
      }
      if (auto w = dyn_cast<affine::AffineWriteOpInterface>(nested)) {
        srcWrites.insert(affine::getSourceMemRef(w.getMemRef()));
      }
    });
  }

  for (Block::iterator it = dstBody->begin(); it != childIt; ++it) {
    bool hit = false;
    it->walk([&srcReads, &srcWrites, &hit](Operation *nested) {
      if (auto w = dyn_cast<affine::AffineWriteOpInterface>(nested)) {
        Value m = affine::getSourceMemRef(w.getMemRef());
        if (srcReads.count(m) || srcWrites.count(m)) {
          hit = true;
          return WalkResult::interrupt();
        }
      } else if (auto r = dyn_cast<affine::AffineReadOpInterface>(nested)) {
        if (srcWrites.count(affine::getSourceMemRef(r.getMemRef()))) {
          hit = true;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (hit) {
      return it;
    }
  }
  return childIt;
}

// Insertion point in dstLevel.body for cloning POST-spine src stages (ops that ran AFTER the
// src spine child in src program order). Default: right after dst spine child. Walk forward
// and stop at the FIRST conflicting dst op so RAW/WAW/WAR with dst's own post-spine ops are
// honored — the conflict op (or earlier) gets ahead of cloned src.
static Block::iterator computePostStageInsertPoint(affine::AffineForOp dstLevel, affine::AffineForOp dstSpineChild,
                                                   ArrayRef<Operation *> srcStages) {
  Block *dstBody = dstLevel.getBody();
  Block::iterator afterChild = std::next(Block::iterator(dstSpineChild));
  if (srcStages.empty()) {
    return afterChild;
  }

  DenseSet<Value> srcReads;
  DenseSet<Value> srcWrites;
  for (Operation *sop : srcStages) {
    sop->walk([&srcReads, &srcWrites](Operation *nested) {
      if (auto r = dyn_cast<affine::AffineReadOpInterface>(nested)) {
        srcReads.insert(affine::getSourceMemRef(r.getMemRef()));
      }
      if (auto w = dyn_cast<affine::AffineWriteOpInterface>(nested)) {
        srcWrites.insert(affine::getSourceMemRef(w.getMemRef()));
      }
    });
  }

  for (Block::iterator it = afterChild; it != dstBody->end(); ++it) {
    if (it->hasTrait<OpTrait::IsTerminator>()) {
      break;
    }
    bool hit = false;
    it->walk([&srcReads, &srcWrites, &hit](Operation *nested) {
      if (auto w = dyn_cast<affine::AffineWriteOpInterface>(nested)) {
        Value m = affine::getSourceMemRef(w.getMemRef());
        if (srcReads.count(m) || srcWrites.count(m)) {
          hit = true;
          return WalkResult::interrupt();
        }
      } else if (auto r = dyn_cast<affine::AffineReadOpInterface>(nested)) {
        if (srcWrites.count(affine::getSourceMemRef(r.getMemRef()))) {
          hit = true;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (hit) {
      return it;
    }
  }
  return afterChild;
}

// For an affine load/store, return the single induction variable indexing each
// result dimension. dimIV[d] is null if dim d is not a bare single-dim expression.
static SmallVector<Value, kSmallVectorSizeFour> extractPerDimIV(Operation *accessOp) {
  AffineMap map;
  SmallVector<Value, kSmallVectorSizeFour> operands;
  if (auto r = dyn_cast<affine::AffineReadOpInterface>(accessOp)) {
    map = r.getAffineMap();
    operands = llvm::to_vector<kSmallVectorSizeFour>(r.getMapOperands());
  } else if (auto w = dyn_cast<affine::AffineWriteOpInterface>(accessOp)) {
    map = w.getAffineMap();
    operands = llvm::to_vector<kSmallVectorSizeFour>(w.getMapOperands());
  } else {
    return {};
  }
  SmallVector<Value, kSmallVectorSizeFour> dimIV(map.getNumResults());
  for (unsigned d = 0; d < map.getNumResults(); ++d) {
    if (auto dimE = dyn_cast<AffineDimExpr>(map.getResult(d))) {
      if (dimE.getPosition() < operands.size()) {
        dimIV[d] = operands[dimE.getPosition()];
      }
    }
  }
  return dimIV;
}

// Index of the loop in `spine` whose IV is `iv`, or -1 if not found.
static int spineLevelOf(Value iv, ArrayRef<affine::AffineForOp> spine) {
  for (unsigned i = 0; i < spine.size(); ++i) {
    affine::AffineForOp f = spine[i];
    if (f.getInductionVar() == iv) {
      return static_cast<int>(i);
    }
  }
  return kUnmappedLevel;
}

// Map each src spine level onto the dst spine level that indexes the SAME dep-memref
// dimension, derived from how srcAnchor (write) and dstAnchor (read) index it. When
// src and dst iterate the shared memref in a transposed axis order this is non-identity.
// Returns identity if a full level bijection cannot be established (caller then keeps
// the original positional behavior).
static SmallVector<unsigned, kSmallVectorSizeFour> computeAxisLevelMap(Operation *srcAnchor, Operation *dstAnchor,
                                                                       ArrayRef<affine::AffineForOp> srcSpine,
                                                                       ArrayRef<affine::AffineForOp> dstSpine) {
  auto depth = static_cast<unsigned>(srcSpine.size());
  SmallVector<unsigned, kSmallVectorSizeFour> identity;
  for (unsigned i = 0; i < depth; ++i) {
    identity.push_back(i);
  }
  if (dstSpine.size() != depth) {
    return identity;
  }

  SmallVector<Value, kSmallVectorSizeFour> srcDimIV = extractPerDimIV(srcAnchor);
  SmallVector<Value, kSmallVectorSizeFour> dstDimIV = extractPerDimIV(dstAnchor);
  if (srcDimIV.empty() || srcDimIV.size() != dstDimIV.size()) {
    return identity;
  }

  SmallVector<int, kSmallVectorSizeFour> m(depth, kUnmappedLevel);
  SmallVector<bool, kSmallVectorSizeFour> used(depth, false);
  for (unsigned d = 0; d < srcDimIV.size(); ++d) {
    if (!srcDimIV[d] || !dstDimIV[d]) {
      continue;
    }
    int sl = spineLevelOf(srcDimIV[d], srcSpine);
    int dl = spineLevelOf(dstDimIV[d], dstSpine);
    if (sl < 0 || dl < 0) {
      continue;
    }
    if (m[sl] != kUnmappedLevel && m[sl] != dl) {
      return identity;  // src level feeds two dst levels
    }
    if (used[dl] && m[sl] != dl) {
      return identity;  // dst level claimed by two src levels
    }
    m[sl] = dl;
    used[dl] = true;
  }
  SmallVector<unsigned, kSmallVectorSizeFour> result;
  for (unsigned i = 0; i < depth; ++i) {
    if (m[i] == kUnmappedLevel) {
      return identity;  // not a full bijection
    }
    result.push_back(static_cast<unsigned>(m[i]));
  }
  return result;
}

// --- Transposed part-wise fusion helpers ---
// A "part" is a group of src spine ops that must stay together at the same dst spine
// level because they are tightly coupled by intra-part SSA/memref dependencies (e.g.
// a reduction body and its surrounding init/update ops). Splitting a part across dst
// levels would leave SSA producers out of scope relative to their consumers.
using SpinePart = SmallVector<Operation *, kSmallVectorSizeEight>;

// Collect src spine ops grouped into parts by non-perfect-nest topology, in src
// program order. At each spine level, a run of consecutive non-for ops forms one
// part; a non-spine AffineForOp (not the spine child) forms its own part (cloned as
// a whole, body included). The spine child triggers recursion into the next level;
// the innermost spine body (leaf) is one part, but not necessarily the last (post-
// spine ops at outer levels may follow it in program order).
// `leafPartIdx` is set to the index of the leaf part within `parts`.
static void collectSpineParts(ArrayRef<affine::AffineForOp> srcSpine, unsigned level,
                              SmallVector<SpinePart, kSmallVectorSizeFour> &parts, unsigned &leafPartIdx) {
  unsigned depth = static_cast<unsigned>(srcSpine.size());
  if (level >= depth) {
    leafPartIdx = UINT_MAX;
    return;
  }
  affine::AffineForOp cur = srcSpine[level];
  affine::AffineForOp childLoop = (level + 1 < depth) ? srcSpine[level + 1] : affine::AffineForOp();
  Operation *child = childLoop ? childLoop.getOperation() : nullptr;
  bool isLeaf = (level + 1 == depth);

  // Leaf level: all body ops form one part.
  if (isLeaf) {
    SpinePart leafPart;
    auto bodyRange = cur.getBody()->without_terminator();
    leafPart.reserve(std::distance(bodyRange.begin(), bodyRange.end()));
    std::transform(bodyRange.begin(), bodyRange.end(), std::back_inserter(leafPart),
                   [](Operation &op) { return &op; });
    if (!leafPart.empty()) {
      leafPartIdx = static_cast<unsigned>(parts.size());
      parts.push_back(std::move(leafPart));
    } else {
      leafPartIdx = UINT_MAX;
    }
    return;
  }

  // Non-leaf level: split body ops around the spine child.
  SpinePart current;
  for (Operation &op : cur.getBody()->without_terminator()) {
    if (child && &op == child) {
      // Flush the pre-spine part.
      if (!current.empty()) {
        parts.push_back(std::move(current));
        current = SpinePart();
      }
      // Recurse into the spine child.
      unsigned childLeaf = UINT_MAX;
      collectSpineParts(srcSpine, level + 1, parts, childLeaf);
      if (childLeaf != UINT_MAX) {
        leafPartIdx = childLeaf;
      }
    } else if (isa<affine::AffineForOp>(op)) {
      // Non-spine nested for: flush current, then the for is its own part.
      if (!current.empty()) {
        parts.push_back(std::move(current));
        current = SpinePart();
      }
      SpinePart forPart;
      forPart.push_back(&op);
      parts.push_back(std::move(forPart));
    } else {
      current.push_back(&op);
    }
  }
  // Flush the post-spine part.
  if (!current.empty()) {
    parts.push_back(std::move(current));
  }
}

// Returns the src spine level (0..depth-1) of the given op by locating the affine.for
// whose body directly contains it. Returns kUnmappedLevel if not found.
static int opSpineLevel(Operation *op, ArrayRef<affine::AffineForOp> srcSpine) {
  if (!op) {
    return kUnmappedLevel;
  }
  Block *block = op->getBlock();
  for (unsigned i = 0; i < srcSpine.size(); ++i) {
    affine::AffineForOp forOp = srcSpine[i];
    if (forOp.getBody() == block) {
      return static_cast<int>(i);
    }
  }
  return kUnmappedLevel;
}

// Deepest dst spine level touched by any memref access in the part, via the
// shared-memref axis alignment in levelMap. For ops with no memref dependency
// (e.g. pure arith), falls back to the src spine level of the op itself, so the
// part stays at the topological level where it was emitted in src.
static unsigned requiredPartDstLevel(const SpinePart &part, ArrayRef<affine::AffineForOp> srcSpine,
                                     ArrayRef<unsigned> levelMap) {
  unsigned lvl = 0;
  bool any = false;
  for (Operation *op : part) {
    op->walk([&lvl, &srcSpine, &levelMap, &any](Operation *n) {
      for (Value v : n->getOperands()) {
        int sl = spineLevelOf(v, srcSpine);
        if (sl >= 0) {
          lvl = std::max(lvl, levelMap[sl]);
          any = true;
        }
      }
    });
  }
  // Fallback: use the src spine level of the part's first op, mapped to dst.
  if (!any) {
    int sl = opSpineLevel(part.front(), srcSpine);
    if (sl >= 0) {
      lvl = levelMap[sl];
    }
  }
  return lvl;
}

// Positional fusion (src spine level i aligns with dst level i): clone src stage ops
// into dst[i] at a dependence-safe position, then clone the leaf body at srcSlice.
static void fuseSpinePositional(ArrayRef<affine::AffineForOp> srcSpine, ArrayRef<affine::AffineForOp> dstSpine,
                                affine::ComputationSliceState &srcSlice, IRMapping &mapper, unsigned fusionDepth) {
  for (unsigned L = 0; L + 1 < fusionDepth; ++L) {
    affine::AffineForOp dstL = dstSpine[L];
    SmallVector<Operation *, kSmallVectorSizeEight> preSpine;
    SmallVector<Operation *, kSmallVectorSizeEight> postSpine;
    collectStageOps(srcSpine[L], srcSpine[L + 1], preSpine, postSpine);
    if (!preSpine.empty()) {
      Block::iterator insertPt = computeStageInsertPoint(dstSpine[L], dstSpine[L + 1], preSpine);
      OpBuilder b(dstL.getBody(), insertPt);
      for (Operation *op : preSpine) {
        b.clone(*op, mapper);
      }
    }
    if (!postSpine.empty()) {
      Block::iterator insertPt = computePostStageInsertPoint(dstSpine[L], dstSpine[L + 1], postSpine);
      OpBuilder b(dstL.getBody(), insertPt);
      for (Operation *op : postSpine) {
        b.clone(*op, mapper);
      }
    }
  }

  OpBuilder b(srcSlice.insertPoint->getBlock(), srcSlice.insertPoint);
  affine::AffineForOp leaf = srcSpine.back();
  for (Operation &op : leaf.getBody()->without_terminator()) {
    b.clone(op, mapper);
  }
}

// Returns true if any operand SSA value of `op` is defined by an operation in a
// LATER dst part (per partLevel/opToPart) than `useLvl`. Such a use would be
// out-of-scope after the part is cloned to `useLvl`.
static bool hasOutOfScopeOperands(Operation *op, const DenseMap<Operation *, unsigned> &opToPart,
                                  const SmallVector<unsigned, kSmallVectorSizeFour> &partLevel, unsigned useLvl,
                                  unsigned curPart) {
  bool outOfScope = false;
  op->walk([&](Operation *n) {
    for (Value v : n->getOperands()) {
      if (Operation *def = v.getDefiningOp()) {
        auto it = opToPart.find(def);
        if (it != opToPart.end() && it->second != curPart && partLevel[it->second] > useLvl) {
          outOfScope = true;
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });
  return outOfScope;
}

// Emits one non-leaf `part` at a dependence-safe insertion point in its target dst
// level, reusing/creating a per-region OpBuilder. Pre-leaf parts insert before the
// dst spine child; post-leaf parts insert after it. Region key separates pre/post.
static void emitNonLeafPart(const SpinePart &part, unsigned partIdx, unsigned leafPartIdx, unsigned fusionDepth,
                            ArrayRef<affine::AffineForOp> dstSpine,
                            const SmallVector<unsigned, kSmallVectorSizeFour> &partLevel, IRMapping &mapper,
                            DenseMap<unsigned, std::unique_ptr<OpBuilder>> &levelBuilder) {
  unsigned lvl = partLevel[partIdx];
  affine::AffineForOp dstLevel = dstSpine[lvl];
  bool isPre = (partIdx < leafPartIdx);
  Block *dstBody = dstLevel.getBody();
  Block::iterator insertPt;
  if (lvl + 1 < fusionDepth) {
    insertPt = isPre ? computeStageInsertPoint(dstLevel, dstSpine[lvl + 1], part)
                     : computePostStageInsertPoint(dstLevel, dstSpine[lvl + 1], part);
  } else {
    // Target is the leaf level but this is not the leaf part.
    insertPt = isPre ? dstBody->begin() : std::prev(dstBody->end());
  }
  unsigned regionKey = lvl * 2 + (isPre ? 0 : 1);
  auto it = levelBuilder.find(regionKey);
  if (it == levelBuilder.end()) {
    levelBuilder[regionKey] = std::make_unique<OpBuilder>(dstBody, insertPt);
  }
  OpBuilder &b = *levelBuilder[regionKey];
  for (Operation *op : part) {
    b.clone(*op, mapper);
  }
}

// Stage-wise fuse for two imperfectly nested loop nests (src→dst, dst is host).
// - Collects spine parts via collectSpineParts.
// - Assigns each part a dst spine level (leaf → fusionDepth-1; others by
//   requiredPartDstLevel).
// - Part-level scope check: a part's consumer must not reference a producer SSA value
//   that lives in a LATER dst level (would be out-of-scope). Returns false on violation.
// - Non-leaf parts: find a dependence-safe insertion point in the target dst level
//   (before any dst op that conflicts with the part's reads/writes), then clone the
//   whole part there. This honors WAR/WAW/RAW between src and dst's own ops.
// - The leaf part is emitted at srcSlice.insertPoint (innermost dst leaf).
static bool fuseSpineTransposed(ArrayRef<affine::AffineForOp> srcSpine, ArrayRef<affine::AffineForOp> dstSpine,
                                ArrayRef<unsigned> levelMap, IRMapping &mapper, unsigned fusionDepth,
                                affine::ComputationSliceState &srcSlice) {
  SmallVector<SpinePart, kSmallVectorSizeFour> parts;
  unsigned leafPartIdx = UINT_MAX;
  collectSpineParts(srcSpine, 0, parts, leafPartIdx);
  if (parts.empty() || leafPartIdx == UINT_MAX) {
    return false;
  }

  // Assign each part to a dst spine level. The leaf part → leaf level (fusionDepth-1);
  // other parts → deepest dst level touched by any memref access in the part (fallback
  // to the part's topological src spine level for pure-arith parts).
  unsigned leafLevel = fusionDepth - 1;
  SmallVector<unsigned, kSmallVectorSizeFour> partLevel(parts.size(), 0);
  for (unsigned p = 0; p < parts.size(); ++p) {
    partLevel[p] = (p == leafPartIdx) ? leafLevel : requiredPartDstLevel(parts[p], srcSpine, levelMap);
  }

  // Part-level out-of-scope check: for each part, none of its operand SSA defining ops
  // (that belong to an earlier part) may live at a deeper dst level than this part.
  DenseMap<Operation *, unsigned> opToPart;
  for (unsigned p = 0; p < parts.size(); ++p) {
    for (Operation *op : parts[p]) {
      opToPart[op] = p;
    }
  }
  for (unsigned p = 0; p < parts.size(); ++p) {
    unsigned useLvl = partLevel[p];
    if (std::any_of(parts[p].begin(), parts[p].end(), [&](Operation *op) {
          return hasOutOfScopeOperands(op, opToPart, partLevel, useLvl, p);
        })) {
      return false;
    }
  }

  // Emit non-leaf parts at dependence-safe insertion points in their target dst level.
  // Parts are processed in src program order (parts index order). To preserve both SSA
  // def-use and memref RAW dependencies between parts, parts falling to the SAME dst
  // level AND same pre/post region must be cloned in src program order: a per-level
  // OpBuilder is seeded at the first part's computed insertion point and reused for
  // subsequent parts in the same region, so a later part never lands ahead of an earlier
  // part that produces a value or writes a memref it reads.
  DenseMap<unsigned, std::unique_ptr<OpBuilder>> levelBuilder;  // region key → running builder
  for (unsigned p = 0; p < parts.size(); ++p) {
    if (p == leafPartIdx) {
      continue;
    }
    emitNonLeafPart(parts[p], p, leafPartIdx, fusionDepth, dstSpine, partLevel, mapper, levelBuilder);
  }

  // Emit the leaf part at srcSlice.insertPoint (innermost dst leaf).
  OpBuilder lb(srcSlice.insertPoint->getBlock(), srcSlice.insertPoint);
  for (Operation *op : parts[leafPartIdx]) {
    lb.clone(*op, mapper);
  }
  return true;
}

// Stage-wise fuse for two imperfectly nested loop nests (src→dst, dst is host).
// 1. Build src/dst spines from anchor ops.
// 2. Map srcSpine[i].iv → dstSpine[levelMap[i]].iv (levelMap aligns shared-memref axes).
// 3. Identity levelMap: clone src stages into dst[i] / leaf at srcSlice.insertPoint.
//    Transposed levelMap: clone each src op at the dst level matching its deepest axis.
// Returns true on success; false on structural failure (caller falls back to fuseLoops).
static bool fuseImperfectLoops(const FusionLoopNestInfo &srcInfo, const FusionLoopNestInfo &dstInfo,
                               affine::ComputationSliceState &srcSlice, const FusionAccessInfo &accessInfo) {
  auto fusionDepth = static_cast<unsigned>(srcSlice.ivs.size());
  if (fusionDepth == 0) {
    return false;
  }
  if (srcInfo.loopDepth < fusionDepth || dstInfo.loopDepth < fusionDepth) {
    return false;
  }
  if (accessInfo.srcAccesses.empty() || accessInfo.dstAccesses.empty()) {
    return false;
  }

  Operation *srcAnchor = accessInfo.srcAccesses.front();
  Operation *dstAnchor = accessInfo.dstAccesses.front();
  SmallVector<affine::AffineForOp, kSmallVectorSizeFour> srcSpine = buildSpine(srcAnchor, srcInfo.root, fusionDepth);
  SmallVector<affine::AffineForOp, kSmallVectorSizeFour> dstSpine = buildSpine(dstAnchor, dstInfo.root, fusionDepth);
  if (srcSpine.empty() || dstSpine.empty()) {
    return false;
  }

  // Align src/dst loop levels by the shared dep-memref axes rather than by nesting
  // position, so a transposed src store ([..,arg3,arg2]) keeps feeding the dimension
  // the dst expects after fusion.
  SmallVector<unsigned, kSmallVectorSizeFour> levelMap = computeAxisLevelMap(srcAnchor, dstAnchor, srcSpine, dstSpine);
  bool isIdentity = true;
  for (unsigned i = 0; i < fusionDepth; ++i) {
    if (levelMap[i] != i) {
      isIdentity = false;
      break;
    }
  }

  IRMapping mapper;
  for (unsigned i = 0; i < fusionDepth; ++i) {
    mapper.map(srcSpine[i].getInductionVar(), dstSpine[levelMap[i]].getInductionVar());
  }

  if (isIdentity) {
    fuseSpinePositional(srcSpine, dstSpine, srcSlice, mapper, fusionDepth);
    return true;
  }
  return fuseSpineTransposed(srcSpine, dstSpine, levelMap, mapper, fusionDepth, srcSlice);
}

// --- FusionCodeGenHelper member functions ---

unsigned FusionCodeGenHelper::getAliasId(unsigned srcId) {
  // Follow the alias chain to find the final destination
  std::unordered_set<unsigned> visited;
  unsigned currentId = srcId;

  while (nodeAlias.find(currentId) != nodeAlias.end()) {
    if (visited.count(currentId) != 0u) {
      // Circular alias detected, break to avoid infinite loop
      break;
    }
    visited.insert(currentId);
    currentId = nodeAlias[currentId];
  }

  return currentId;
}

void FusionCodeGenHelper::buildStrategyOpsA(const affine::FusionStrategy &strategy, llvm::ArrayRef<Operation *> allOpsA,
                                            llvm::SmallVector<Operation *, kSmallVectorSizeFour> &strategyOpsA) {
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

// Selects the most appropriate DependenceInfo from plan.depInfo for the fusion:
//  1) prefer the smallest loopDepth (shallowest dependence);
//  2) then prefer the memref that is not a block argument (defined by an op);
//  3) then prefer the memref that appears latest (last access) in srcAffineForOp,
//     so the fused slice is placed closest to the producer of the dependency.
// Example (see tools/case.log): depInfo entries hold alloc_19 and alloc_13; since
// alloc_13 is accessed after alloc_19 in srcAffineForOp, alloc_13's entry is chosen
// (assuming equal loopDepth and both are non-block-argument memrefs).
const DependenceInfo *FusionCodeGenHelper::selectDepInfo(const FusionPlan &plan,
                                                            affine::AffineForOp srcAffineForOp) const {
  if (plan.depInfo.empty()) {
    return nullptr;
  }
  if (plan.depInfo.size() == 1) {
    return &plan.depInfo.front();
  }

  // Walk srcAffineForOp's body and record the last operation index that accesses
  // each candidate memref. Default pre-order walk follows lexical source order.
  DenseMap<Value, size_t> lastAccessIdx;
  size_t opIdx = 0;
  srcAffineForOp.getBody()->walk([&](Operation *op) {
    Value memref;
    if (auto read = dyn_cast<affine::AffineReadOpInterface>(op)) {
      memref = read.getMemRef();
    } else if (auto write = dyn_cast<affine::AffineWriteOpInterface>(op)) {
      memref = write.getMemRef();
    }
    if (memref) {
      lastAccessIdx[memref] = opIdx;
    }
    ++opIdx;
  });

  // Pick the best depInfo by (loopDepth asc, non-block-arg, last-access idx desc).
  const DependenceInfo *best = nullptr;
  size_t bestIdx = 0;
  for (const auto &depInfo : plan.depInfo) {
    auto it = lastAccessIdx.find(depInfo.memref);
    if (it == lastAccessIdx.end()) {
      continue;
    }
    if (best == nullptr) {
      best = &depInfo;
      bestIdx = it->second;
      continue;
    }
    bool curIsBlockArg = isa<BlockArgument>(depInfo.memref);
    bool bestIsBlockArg = isa<BlockArgument>(best->memref);
    if (depInfo.loopDepth != best->loopDepth) {
      if (depInfo.loopDepth < best->loopDepth) {
        best = &depInfo;
        bestIdx = it->second;
      }
    } else if (curIsBlockArg != bestIsBlockArg) {
      if (bestIsBlockArg && !curIsBlockArg) {
        best = &depInfo;
        bestIdx = it->second;
      }
    } else if (it->second > bestIdx) {
      best = &depInfo;
      bestIdx = it->second;
    }
  }
  return best;
}

unsigned FusionCodeGenHelper::findMaxLegalFusionDepth(
  const FusionLoopNestInfo &srcInfo, const FusionLoopNestInfo &dstInfo, const affine::FusionStrategy &strategy,
  llvm::SmallVector<affine::ComputationSliceState, kSmallVectorSizeEight> &depthSliceUnions, const FusionPlan &plan,
  const DependenceInfo &depInfo, FusionAccessInfo &accessInfo, bool srcDstReversed) {
  affine::AffineForOp srcAffineForOp = srcInfo.root;
  affine::AffineForOp dstAffineForOp = dstInfo.root;
  auto &srcMemFlags = accessInfo.srcMemFlags;
  auto &srcAccesses = accessInfo.srcAccesses;
  auto &dstMemFlags = accessInfo.dstMemFlags;
  auto &dstAccesses = accessInfo.dstAccesses;

  bool isSrcBeforeDst = srcAffineForOp->isBeforeInBlock(dstAffineForOp);
  affine::AffineForOp loopA = isSrcBeforeDst ? srcAffineForOp : dstAffineForOp;
  affine::AffineForOp loopB = isSrcBeforeDst ? dstAffineForOp : srcAffineForOp;
  if (!hasValidFusionPoint(loopA, loopB, srcMemFlags, dstMemFlags)) {
    llvm::dbgs() << "Fusion would violate dependences in block\n";
    return 0;
  }

  SmallVector<Operation *, kSmallVectorSizeFour> strategyOpsA;
  buildStrategyOpsA(strategy, srcAccesses, strategyOpsA);
  if (strategyOpsA.empty()) {
    llvm::dbgs() << "No strategy ops found for fusion\n";
    return 0;
  }

  auto srcGroup = mdg.getGroup(plan.fusedGroup.from);
  auto dstGroup = mdg.getGroup(plan.fusedGroup.to);
  if (!srcGroup || !dstGroup) {
    llvm::dbgs() << "srcGroup or dstGroup is nullptr\n";
    return 0;
  }
  auto srcGroupTemplate = srcGroup->groupTemplate;
  auto dstGroupTemplate = dstGroup->groupTemplate;
  bool isReduction =
    (srcGroupTemplate == OperatorTemplate::Reduction) || (dstGroupTemplate == OperatorTemplate::Reduction);

  unsigned depDepth = depInfo.loopDepth;
  unsigned loopDepth = dstInfo.loopDepth;
  if (!isReduction) {
    auto &depAccesses = srcDstReversed ? srcAccesses : dstAccesses;
    depDepth = computeEffectiveDepDepth(dstInfo.loops, depAccesses, srcInfo.loops);
  }

  // Use depth of insertion target (second param = dstAffineForOp).
  depthSliceUnions.clear();
  depthSliceUnions.resize(loopDepth);
  unsigned maxDepth = 0;
  for (unsigned depth = loopDepth; depth >= kMinFusionDepth; --depth) {
    if (strategy.getStrategy() == affine::FusionStrategy::ProducerConsumer) {
      if (depDepth < depth) {
        continue;
      }
    }

    auto &sliceState = depthSliceUnions[depth - 1];
    // isBackwardSlice must match which loop fuseLoops() will clone:
    //  - non-reversed: fuseLoops clones srcAffineForOp, whose ops are in
    //    strategyOpsA (opsA) -> backward slice gives opsA IVs.
    //  - reversed: fuseLoops clones caller's dstAffineForOp, whose ops are in
    //    dstAccesses (opsB) -> forward slice gives opsB IVs.
    affine::SliceComputationResult res =
      computeSliceUnionAKG(strategyOpsA, dstAccesses, depth, kDefaultNumCommonLoops, !srcDstReversed, &sliceState);
    if (res.value == affine::SliceComputationResult::Success) {
      maxDepth = depth;
      break;
    }
  }

  return maxDepth;
}

void FusionCodeGenHelper::doIFuse(unsigned srcGroupId, unsigned dstGroupId, FusionLoopNestInfo &srcInfo,
                                  FusionLoopNestInfo &dstInfo, const FusionPlan &plan) {
  // Fusion depth = deepest common prefix of aligned axis chains with matching bounds.
  SmallVector<affine::AffineForOp, kSmallVectorSizeFour> srcChain;
  SmallVector<affine::AffineForOp, kSmallVectorSizeFour> dstChain;
  collectAlignedAxisChains(srcInfo.root, dstInfo.root, srcChain, dstChain);

  auto depth = std::min(srcChain.size(), dstChain.size());
  while (depth != 0 && !checkLoopBoundsMatch(srcChain, dstChain, depth)) {
    depth--;
  }

  if (depth == 0) {
    llvm::errs() << "srcLoops and dstLoops have no same loop bounds\n";
    return;
  }
  performLoopFusion(srcChain, dstChain, depth);
  srcInfo.root.erase();
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
  if (auto *node = mdg.getNode(srcId)) {
    node->loads.clear();
    node->stores.clear();
    node->op = nullptr;
  }
}

// Resolve keepSrcDst for doFuse. Returns std::nullopt when src/dst group is
// missing (caller should bail out), otherwise the resolved keepSrcDst value.
static std::optional<bool> resolveKeepSrcDst(MemRefDependenceGraphForFusion &mdg, unsigned srcGroupId,
                                             unsigned dstGroupId, const FusionLoopNestInfo &srcInfo,
                                             const FusionLoopNestInfo &dstInfo) {
  auto srcGroup = mdg.getGroup(srcGroupId);
  auto dstGroup = mdg.getGroup(dstGroupId);
  if (!srcGroup || !dstGroup) {
    llvm::dbgs() << "srcGroup or dstGroup is nullptr\n";
    return std::nullopt;
  }
  bool keepSrcDst = srcInfo.isPerfect || !dstInfo.isPerfect;
  auto srcGroupTemplate = srcGroup->groupTemplate;
  auto dstGroupTemplate = dstGroup->groupTemplate;
  if (srcGroupTemplate == OperatorTemplate::Broadcast && dstGroupTemplate == OperatorTemplate::Broadcast &&
      !srcInfo.isPerfect && dstInfo.isPerfect && dstInfo.loopDepth > srcInfo.loopDepth &&
      dstInfo.perfectDepth > srcInfo.perfectDepth) {
    keepSrcDst = true;
  }
  return keepSrcDst;
}

void FusionCodeGenHelper::doFuse(unsigned srcGroupId, unsigned dstGroupId, affine::AffineForOp srcAffineForOp,
                                 affine::AffineForOp dstAffineForOp, const FusionPlan &plan) {
  FusionLoopNestInfo srcInfo;
  FusionLoopNestInfo dstInfo;
  srcInfo.collect(srcAffineForOp);
  dstInfo.collect(dstAffineForOp);

  // Select the most appropriate DependenceInfo: the one whose memref appears latest
  // in srcAffineForOp, so the fused slice is placed closest to the dependency producer.
  const DependenceInfo *selectedDep = selectDepInfo(plan, srcAffineForOp);
  if (selectedDep == nullptr) {
    doIFuse(srcGroupId, dstGroupId, srcInfo, dstInfo, plan);
    return;
  }
  const DependenceInfo &depInfo = *selectedDep;

  FusionAccessInfo accessInfo;
  bool haveAccesses =
    depInfo.memref && collectFusionAccesses(depInfo.memref, srcAffineForOp, dstAffineForOp, accessInfo);
  if (haveAccesses && depInfo.memrefKind != MemrefKind::Subview &&
      isDependenceAxisRelationUnresolved(accessInfo.srcAccesses, accessInfo.dstAccesses)) {
    return;
  }

  if (plan.fusionType == "I" || !haveAccesses) {
    doIFuse(srcGroupId, dstGroupId, srcInfo, dstInfo, plan);
    return;
  }

  // H-fusion uses ProducerConsumer; V-fusion uses Sibling on the dep memref (or Generic if missing).
  affine::FusionStrategy strategy = plan.fusionType == "H"
                                      ? affine::FusionStrategy(affine::FusionStrategy::ProducerConsumer)
                                      : (depInfo.memref ? affine::FusionStrategy(depInfo.memref)
                                                        : affine::FusionStrategy(affine::FusionStrategy::Generic));

  SmallVector<affine::ComputationSliceState, kSmallVectorSizeEight> depthSliceUnions;
  unsigned maxLegalFusionDepth = 0;

  auto keepSrcDstOpt = resolveKeepSrcDst(mdg, srcGroupId, dstGroupId, srcInfo, dstInfo);
  if (!keepSrcDstOpt) {
    return;
  }
  bool keepSrcDst = *keepSrcDstOpt;

  if (keepSrcDst) {
    maxLegalFusionDepth =
      findMaxLegalFusionDepth(srcInfo, dstInfo, strategy, depthSliceUnions, plan, depInfo, accessInfo);
  } else {
    maxLegalFusionDepth =
      findMaxLegalFusionDepth(dstInfo, srcInfo, strategy, depthSliceUnions, plan, depInfo, accessInfo, true);
  }

  if (maxLegalFusionDepth == 0) {
    doIFuse(srcGroupId, dstGroupId, srcInfo, dstInfo, plan);
    return;
  }

  unsigned bestDepth = maxLegalFusionDepth;
  assert(bestDepth > 0 && "Unexpected loop fusion depth");
  affine::ComputationSliceState &bestSlice = depthSliceUnions[bestDepth - 1];
  assert(!bestSlice.isEmpty() && "Missing slice union for depth");

  applySliceFusion(srcAffineForOp, dstAffineForOp, srcInfo, dstInfo, keepSrcDst, bestSlice, accessInfo, srcGroupId,
                   dstGroupId);
}

void FusionCodeGenHelper::applySliceFusion(affine::AffineForOp srcAffineForOp, affine::AffineForOp dstAffineForOp,
                                           FusionLoopNestInfo &srcInfo, FusionLoopNestInfo &dstInfo, bool keepSrcDst,
                                           affine::ComputationSliceState &bestSlice, const FusionAccessInfo &accessInfo,
                                           unsigned srcGroupId, unsigned dstGroupId) {
  if (keepSrcDst) {
    bool fused = false;
    if (!srcInfo.isPerfect && !dstInfo.isPerfect) {
      fused = fuseImperfectLoops(srcInfo, dstInfo, bestSlice, accessInfo);
    }
    if (!fused) {
      fuseLoops(srcAffineForOp, dstAffineForOp, bestSlice);
    }
    eraseLoopAndCleanupNode(srcGroupId, dstGroupId, srcAffineForOp);
  } else {
    fuseLoops(dstAffineForOp, srcAffineForOp, bestSlice);
    eraseLoopAndCleanupNode(dstGroupId, srcGroupId, dstAffineForOp);
  }
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

  if (auto *erasedNode = mdg.getNode(erasedNodeId)) {
    erasedNode->loads.clear();
    erasedNode->stores.clear();
    erasedNode->op = nullptr;
  }

  loopToErase.erase();
}

// ===----------------------------------------------------------------------===//
// Group implementation
// ===----------------------------------------------------------------------===//

std::string Group::getGroupTemplateString() const {
  auto it = operatorTemplateMap.find(static_cast<int>(groupTemplate));
  if (it != operatorTemplateMap.end()) {
    return it->second;
  }
  return std::to_string(static_cast<int>(groupTemplate));
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
