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
// or an alias (subview/reshape) of it.
// This is needed because getStoreOpCount only checks exact memref matches,
// missing stores through subview/reshape aliases.
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
        // Also track the source memref if this is a subview or other aliasing operation
        Value sourceMemref = affine::getSourceMemRef(memref);
        if (sourceMemref != memref) {
          memrefAccesses[sourceMemref].insert(node.id);
        }
      }
      for (auto *opInst : collector.storeOpInsts) {
        node.stores.push_back(opInst);
        auto memref = cast<affine::AffineWriteOpInterface>(opInst).getMemRef();
        memrefAccesses[memref].insert(node.id);
        // Also track the source memref if this is a subview or other aliasing operation
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
      // Also track the source memref if this is a subview or other aliasing operation
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
      // Also track the source memref if this is a subview or other aliasing operation
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

// Returns true if both nodes have stores that write through SubViewOps to the
// same base buffer, and those subviews are provably non-overlapping.
// Conservative: returns false if overlap cannot be ruled out.
bool MemRefDependenceGraphForFusion::areNonOverlappingSubviewStores(unsigned nodeIdA, unsigned nodeIdB) {
  Node *nodeA = getNode(nodeIdA);
  Node *nodeB = getNode(nodeIdB);
  if ((nodeA == nullptr) || (nodeB == nullptr)) {
    return false;
  }

  DenseMap<Value, memref::SubViewOp> subviewsA;
  DenseMap<Value, memref::SubViewOp> subviewsB;

  auto collectSubviewStores = [](Node *node, DenseMap<Value, memref::SubViewOp> &result) {
    for (auto *storeOp : node->stores) {
      auto writeOp = dyn_cast<affine::AffineWriteOpInterface>(storeOp);
      if (!writeOp) {
        continue;
      }
      memref::SubViewOp sv;
      // hasSubView: unused output parameter (pass nullptr).
      Value base = affine::getSourceMemRef(writeOp.getMemRef(), nullptr, &sv);
      if (!sv) {
        continue;
      }
      result[base] = sv;
    }
  };

  collectSubviewStores(nodeA, subviewsA);
  collectSubviewStores(nodeB, subviewsB);

  bool foundSharedBase = false;
  for (auto &[base, svA] : subviewsA) {
    auto it = subviewsB.find(base);
    if (it == subviewsB.end()) {
      continue;
    }
    foundSharedBase = true;
    auto svB = it->second;

    auto offsetsA = svA.getStaticOffsets();
    auto sizesA = svA.getStaticSizes();
    auto offsetsB = svB.getStaticOffsets();
    auto sizesB = svB.getStaticSizes();
    if (offsetsA.size() != offsetsB.size()) {
      return false;
    }

    bool disjoint = false;
    for (unsigned d = 0; d < offsetsA.size(); ++d) {
      if (offsetsA[d] == ShapedType::kDynamic || sizesA[d] == ShapedType::kDynamic ||
          offsetsB[d] == ShapedType::kDynamic || sizesB[d] == ShapedType::kDynamic) {
        continue;
      }
      if (offsetsA[d] + sizesA[d] <= offsetsB[d] || offsetsB[d] + sizesB[d] <= offsetsA[d]) {
        disjoint = true;
        break;
      }
    }
    if (!disjoint) {
      return false;
    }
  }
  return foundSharedBase;
}

void MemRefDependenceGraphForFusion::addAliasedStoreEdges(
  Value memref, const SetVector<unsigned> &nodeIds,
  const SmallVector<std::pair<unsigned, bool>, kSmallVectorSizeSixteen> &nonForNodesWithStore,
  const SmallVector<std::pair<unsigned, bool>, kSmallVectorSizeEight> &forNodesWithStore, bool hasLoadsForBaseMemref) {
  for (unsigned i = 0; i < nonForNodesWithStore.size(); ++i) {
    unsigned srcId = nonForNodesWithStore[i].first;
    bool srcHasStore = nonForNodesWithStore[i].second;
    for (unsigned j = i + 1; j < nonForNodesWithStore.size(); ++j) {
      unsigned dstId = nonForNodesWithStore[j].first;
      bool dstHasStore = nonForNodesWithStore[j].second;
      if (!(srcHasStore || dstHasStore) || hasEdge(srcId, dstId, memref)) {
        continue;
      }
      if (srcHasStore && dstHasStore && areNonOverlappingSubviewStores(srcId, dstId) && hasLoadsForBaseMemref) {
        continue;
      }
      int forLoopNodeId = getEnclosingForLoopNodeId(*this, nodeIds, dstId);
      unsigned edgeLoopDepth = computeMemrefLoopDepth(forLoopNodeId, memref);
      addEdge(srcId, dstId, memref, edgeLoopDepth);
    }
  }

  for (unsigned i = 0; i < forNodesWithStore.size(); ++i) {
    unsigned srcId = forNodesWithStore[i].first;
    bool srcHasStore = forNodesWithStore[i].second;
    for (unsigned j = i + 1; j < forNodesWithStore.size(); ++j) {
      unsigned dstId = forNodesWithStore[j].first;
      bool dstHasStore = forNodesWithStore[j].second;
      if (!(srcHasStore && dstHasStore) || hasEdge(srcId, dstId, memref)) {
        continue;
      }
      if (areNonOverlappingSubviewStores(srcId, dstId) && hasLoadsForBaseMemref) {
        continue;
      }
      int forLoopNodeId = getEnclosingForLoopNodeId(*this, nodeIds, dstId);
      unsigned edgeLoopDepth = computeMemrefLoopDepth(forLoopNodeId, memref);
      addEdge(srcId, dstId, memref, edgeLoopDepth);
    }
  }
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
// SubviewFusionHelper implementation
// ===----------------------------------------------------------------------===//

IntegerSet FusionGuard::buildCondSet(MLIRContext *ctx) const {
  OpBuilder builder(ctx);
  AffineExpr d0 = builder.getAffineDimExpr(kGuardDimIndex);
  if (offset != 0) {
    AffineExpr lowerCond = d0 - builder.getAffineConstantExpr(offset);
    AffineExpr upperCond = builder.getAffineConstantExpr(offset + boundValue - 1) - d0;
    // eqFlags: both constraints are inequalities (>=).
    return IntegerSet::get(kGuardDimCount, kGuardSymbolCount, {lowerCond, upperCond}, {false, false});
  }
  AffineExpr condExpr = (kind == Kind::ExtraDimEqLB) ? builder.getAffineConstantExpr(boundValue) - d0
                                                     : builder.getAffineConstantExpr(boundValue - 1) - d0;
  // eqFlags: single inequality constraint (>=).
  return IntegerSet::get(kGuardDimCount, kGuardSymbolCount, {condExpr}, {false});
}

// Helper: check whether two loops have identical LB and step (UB may differ).
static bool loopLBStepMatch(affine::AffineForOp a, affine::AffineForOp b) {
  return a.getLowerBoundMap() == b.getLowerBoundMap() && a.getStep() == b.getStep();
}

// Build an identity dim map: [0, 1, ..., depth-1].
static SmallVector<int, kSmallVectorSizeFour> makeIdentityDimMap(unsigned depth) {
  SmallVector<int, kSmallVectorSizeFour> map;
  for (unsigned i = 0; i < depth; ++i) {
    map.push_back(i);
  }
  return map;
}

// Build a dim map that skips one position: [0,..,skipPos-1, skipPos+1,..,depth].
static SmallVector<int, kSmallVectorSizeFour> makeSkipOneDimMap(unsigned secondaryDepth, unsigned skipPos) {
  SmallVector<int, kSmallVectorSizeFour> map;
  for (unsigned i = 0; i < skipPos; ++i) {
    map.push_back(i);
  }
  for (unsigned i = skipPos; i < secondaryDepth; ++i) {
    map.push_back(i + 1);
  }
  return map;
}

// Compute the inverse permutation: if perm[i] = j, then inverse[j] = i.
static SmallVector<int, kSmallVectorSizeFour> computeInversePermutation(ArrayRef<int> perm) {
  SmallVector<int, kSmallVectorSizeFour> inverse(perm.size());
  for (unsigned i = 0; i < perm.size(); ++i) {
    inverse[perm[i]] = static_cast<int>(i);
  }
  return inverse;
}

bool SubviewFusionHelper::buildSubviewFusionPlan(FusionLoopNestInfo &srcInfo, FusionLoopNestInfo &dstInfo, int srcRank,
                                                 int dstRank) {
  // Prioritize loop depth for strategy selection: loop structure determines
  // how IVs are aligned. Memref rank is only used as a hint when depths differ.
  if (srcInfo.loopDepth == dstInfo.loopDepth) {
    if (buildSameRankPlan(srcInfo, dstInfo)) {
      return true;
    }
    // Identity matching failed — try permuted dimension mappings.
    return buildPermutedPlan(srcInfo, dstInfo);
  }

  // Different depth — use rank to decide primary side; fall back to loop depth.
  bool rankUseful = (srcRank >= 0 && dstRank >= 0 && srcRank != dstRank);
  int effectiveSrcRank = rankUseful ? srcRank : static_cast<int>(srcInfo.loopDepth);
  int effectiveDstRank = rankUseful ? dstRank : static_cast<int>(dstInfo.loopDepth);
  return buildExtraDimPlan(srcInfo, dstInfo, effectiveSrcRank, effectiveDstRank);
}

bool SubviewFusionHelper::buildExtraDimPlan(FusionLoopNestInfo &srcInfo, FusionLoopNestInfo &dstInfo, int srcRank,
                                            int dstRank) {
  // Primary = the side with higher memref rank (or loop depth as fallback).
  bool srcIsPrimary = (srcRank > dstRank);
  FusionLoopNestInfo *primary = srcIsPrimary ? &srcInfo : &dstInfo;
  FusionLoopNestInfo *secondary = srcIsPrimary ? &dstInfo : &srcInfo;

  // Primary must have exactly one more loop than secondary.
  if (primary->loopDepth != secondary->loopDepth + kExtraDimDepthDiff) {
    return false;
  }

  unsigned N = primary->loopDepth;
  unsigned M = secondary->loopDepth;

  for (unsigned p = 0; p < N; ++p) {
    bool matched = true;
    for (unsigned i = 0; i < p && matched; ++i) {
      matched = affine::loopBoundsMatch(primary->loops[i], secondary->loops[i]);
    }
    for (unsigned i = p; i < M && matched; ++i) {
      matched = affine::loopBoundsMatch(primary->loops[i + 1], secondary->loops[i]);
    }
    if (!matched) {
      continue;
    }

    auto extraLoop = primary->loops[p];
    if (!extraLoop.hasConstantLowerBound() || !extraLoop.hasConstantUpperBound()) {
      return false;
    }

    int64_t lb = extraLoop.getConstantLowerBound();
    int64_t ub = extraLoop.getConstantUpperBound();
    int64_t step = extraLoop.getStep().getSExtValue();

    // The cloned secondary body will be inserted into primary->loops[N-1]'s
    // body, with secondary IVs mapped onto primary IVs at positions
    // [0..p-1, p+1..N-1] (dimMap skips p).  For that mapping to be valid,
    // every loop on the post-skip chain must enclose loops[N-1] so its IV
    // is in scope inside the target body.  When primary is imperfect (e.g.,
    // sibling sub-loops collected by DFS), this is not automatic.
    affine::AffineForOp targetLoop = primary->loops[N - 1];
    bool chainEnclosesTarget = true;
    for (unsigned i = 0; i + 1 < N && chainEnclosesTarget; ++i) {
      if (i == p) {
        continue;
      }
      chainEnclosesTarget = primary->loops[i]->isProperAncestor(targetLoop.getOperation());
    }
    if (!chainEnclosesTarget) {
      continue;
    }

    // The guard "extraIV == lb" is only meaningful when extraLoop encloses
    // the cloned body.  When loops[p] is a sibling of the chain (does not
    // enclose loops[N-1]), the sibling structure itself isolates the paths
    // and using extraLoop's IV would reference an out-of-scope value.
    bool extraEnclosesTarget = (p == N - 1) || primary->loops[p]->isProperAncestor(targetLoop.getOperation());

    // Guard needed only when the extra dim iterates more than once AND it
    // actually encloses the target body.
    FusionGuard guard;
    if (extraEnclosesTarget && ub - lb > step) {
      guard = {FusionGuard::Kind::ExtraDimEqLB, p, lb};
    }

    plan.srcInfo = &srcInfo;
    plan.dstInfo = &dstInfo;
    plan.srcIsPrimary = srcIsPrimary;
    plan.dimMap = makeSkipOneDimMap(M, p);
    plan.guard = guard;
    return true;
  }
  return false;
}

bool SubviewFusionHelper::buildSameRankPlan(FusionLoopNestInfo &srcInfo, FusionLoopNestInfo &dstInfo) {
  unsigned depth = srcInfo.loopDepth;
  int mismatchPos = kNoMismatchPos;
  for (unsigned i = 0; i < depth; ++i) {
    if (affine::loopBoundsMatch(srcInfo.loops[i], dstInfo.loops[i])) {
      continue;
    }
    if (loopLBStepMatch(srcInfo.loops[i], dstInfo.loops[i])) {
      if (mismatchPos != kNoMismatchPos) {
        return false;  // multiple mismatches
      }
      mismatchPos = static_cast<int>(i);
    } else {
      return false;
    }
  }

  plan.srcInfo = &srcInfo;
  plan.dstInfo = &dstInfo;
  plan.dimMap = makeIdentityDimMap(depth);

  if (mismatchPos == kNoMismatchPos) {
    // All bounds match exactly — trivial fusion, no guard needed.
    plan.srcIsPrimary = true;
    return true;
  }

  auto srcLoop = srcInfo.loops[mismatchPos];
  auto dstLoop = dstInfo.loops[mismatchPos];
  if (!srcLoop.hasConstantUpperBound() || !dstLoop.hasConstantUpperBound()) {
    return false;
  }

  int64_t srcUB = srcLoop.getConstantUpperBound();
  int64_t dstUB = dstLoop.getConstantUpperBound();

  plan.srcIsPrimary = (srcUB >= dstUB);
  plan.guard = {FusionGuard::Kind::SmallerUB, static_cast<unsigned>(mismatchPos), std::min(srcUB, dstUB)};
  plan.guard.offset = detectGuardDimSubviewOffset(static_cast<unsigned>(mismatchPos));
  return true;
}

static int checkPermMismatch(FusionLoopNestInfo &srcInfo, FusionLoopNestInfo &dstInfo, ArrayRef<int> perm,
                             unsigned depth) {
  int mismatchPos = kNoMismatchPos;
  for (unsigned i = 0; i < depth; ++i) {
    if (affine::loopBoundsMatch(srcInfo.loops[perm[i]], dstInfo.loops[i])) {
      continue;
    }
    if (loopLBStepMatch(srcInfo.loops[perm[i]], dstInfo.loops[i])) {
      if (mismatchPos != kNoMismatchPos) {
        return kMultipleMismatchPos;
      }
      mismatchPos = static_cast<int>(i);
    } else {
      return kMultipleMismatchPos;
    }
  }
  return mismatchPos;
}

bool SubviewFusionHelper::buildPermutedPlan(FusionLoopNestInfo &srcInfo, FusionLoopNestInfo &dstInfo) {
  unsigned depth = srcInfo.loopDepth;

  SmallVector<int, kSmallVectorSizeFour> perm;
  for (unsigned i = 0; i < depth; ++i) {
    perm.push_back(static_cast<int>(i));
  }

  while (std::next_permutation(perm.begin(), perm.end())) {
    int mismatchPos = checkPermMismatch(srcInfo, dstInfo, perm, depth);
    if (mismatchPos == kMultipleMismatchPos) {
      continue;
    }

    plan.srcInfo = &srcInfo;
    plan.dstInfo = &dstInfo;

    if (mismatchPos == kNoMismatchPos) {
      plan.srcIsPrimary = true;
      plan.dimMap.assign(perm.begin(), perm.end());
      return true;
    }

    auto srcLoop = srcInfo.loops[perm[mismatchPos]];
    auto dstLoop = dstInfo.loops[mismatchPos];
    if (!srcLoop.hasConstantUpperBound() || !dstLoop.hasConstantUpperBound()) {
      continue;
    }

    int64_t srcUB = srcLoop.getConstantUpperBound();
    int64_t dstUB = dstLoop.getConstantUpperBound();

    unsigned secondaryGuardDim;
    if (srcUB >= dstUB) {
      plan.srcIsPrimary = true;
      plan.dimMap.assign(perm.begin(), perm.end());
      plan.guard = {FusionGuard::Kind::SmallerUB, static_cast<unsigned>(perm[mismatchPos]), std::min(srcUB, dstUB)};
      secondaryGuardDim = static_cast<unsigned>(mismatchPos);
    } else {
      plan.srcIsPrimary = false;
      plan.dimMap = computeInversePermutation(perm);
      plan.guard = {FusionGuard::Kind::SmallerUB, static_cast<unsigned>(mismatchPos), std::min(srcUB, dstUB)};
      secondaryGuardDim = static_cast<unsigned>(perm[mismatchPos]);
    }
    plan.guard.offset = detectGuardDimSubviewOffset(secondaryGuardDim);
    return true;
  }

  return false;
}

// Scan `currentLoop`'s body for sibling AffineForOps. Sets `nextLevelFor` to
// the main-chain inner for matching `expectedNext` (if any), counts all fors,
// and returns false (side-branch detected) when the count exceeds the limit.
static bool findNextLevelFor(affine::AffineForOp currentLoop, affine::AffineForOp expectedNext,
                             affine::AffineForOp &nextLevelFor, unsigned &forCount) {
  for (Operation &op : currentLoop.getBody()->without_terminator()) {
    auto forOp = dyn_cast<affine::AffineForOp>(&op);
    if (!forOp) {
      continue;
    }
    forCount++;
    if (forOp == expectedNext) {
      nextLevelFor = forOp;
    }
  }
  return forCount <= kMaxSiblingForLoops;
}

SmallVector<CloneStage> SubviewFusionHelper::collectCloneStages(const FusionLoopNestInfo &info) {
  SmallVector<CloneStage> stages;

  if (info.isPerfect) {
    // Single stage: all ops in the innermost body.
    auto innermost = info.loops[info.loopDepth - 1];
    auto body = innermost.getBody()->without_terminator();
    CloneStage stage;
    std::transform(body.begin(), body.end(), std::back_inserter(stage), [](Operation &op) { return &op; });
    stages.push_back(std::move(stage));
    return stages;
  }

  // Imperfect nest: one stage per level from perfectDepth-1 onward.
  for (unsigned level = info.perfectDepth - 1; level < info.loopDepth; ++level) {
    auto currentLoop = info.loops[level];

    // Identify the main-chain inner for (if any) and reject side branches.
    affine::AffineForOp nextLevelFor;
    if (level + 1 < info.loopDepth) {
      unsigned forCount = 0;
      if (!findNextLevelFor(currentLoop, info.loops[level + 1], nextLevelFor, forCount)) {
        return {};  // side-branch for — bail out
      }
    }

    CloneStage stage;
    for (Operation &op : currentLoop.getBody()->without_terminator()) {
      if (nextLevelFor && &op == nextLevelFor.getOperation()) {
        continue;
      }
      stage.push_back(&op);
    }
    if (!stage.empty()) {
      stages.push_back(std::move(stage));
    }
  }
  return stages;
}

void SubviewFusionHelper::emitCloneStages(const SmallVector<CloneStage> &stages) {
  auto targetLoop = plan.primaryInfo()->loops[plan.primaryInfo()->loopDepth - 1];
  Block *targetBody = targetLoop.getBody();

  // Secondary is src → insert at start; secondary is dst → insert at end.
  bool secondaryIsSrc = !plan.srcIsPrimary;
  Block::iterator insertPt = secondaryIsSrc ? targetBody->begin() : std::prev(targetBody->end());

  // Build IV mapping: secondary.loops[i].iv → primary.loops[dimMap[i]].iv
  IRMapping mapper;
  for (unsigned i = 0; i < plan.secondaryInfo()->loopDepth; ++i) {
    mapper.map(plan.secondaryInfo()->loops[i].getInductionVar(),
               plan.primaryInfo()->loops[plan.dimMap[i]].getInductionVar());
  }

  // When a subview offset is detected, the secondary loop's guarded-dim IV
  // must be shifted: secondaryIV = primaryIV - offset.  Create an affine.apply
  // and override the corresponding entry in the mapper.
  if (plan.guard.offset != 0) {
    auto primaryIV = plan.primaryInfo()->loops[plan.guard.dimPos].getInductionVar();
    OpBuilder b(targetBody, std::prev(targetBody->end()));
    auto shiftMap = AffineMap::get(kGuardDimCount, kGuardSymbolCount,
                                   b.getAffineDimExpr(kGuardDimIndex) - plan.guard.offset, targetLoop->getContext());
    auto applyOp = b.create<affine::AffineApplyOp>(targetLoop.getLoc(), shiftMap, ValueRange{primaryIV});
    for (unsigned i = 0; i < plan.secondaryInfo()->loopDepth; ++i) {
      if (plan.dimMap[i] == static_cast<int>(plan.guard.dimPos)) {
        mapper.map(plan.secondaryInfo()->loops[i].getInductionVar(), applyOp.getResult());
        break;
      }
    }
  }

  // Build guard condition once, apply to every stage.
  Value guardIV;
  IntegerSet condSet;
  if (plan.guard.isNeeded()) {
    guardIV = plan.primaryInfo()->loops[plan.guard.dimPos].getInductionVar();
    condSet = plan.guard.buildCondSet(targetLoop->getContext());
  }

  for (const auto &stage : stages) {
    OpBuilder builder(targetBody, insertPt);
    Operation *anchorOp = nullptr;

    if (guardIV) {
      // withElseRegion = false: no else block needed.
      auto ifOp = builder.create<affine::AffineIfOp>(targetLoop.getLoc(), condSet, ValueRange{guardIV},
                                                     false);
      anchorOp = ifOp;
      builder.setInsertionPointToStart(ifOp.getThenBlock());
    }

    for (auto *op : stage) {
      auto *cloned = builder.clone(*op, mapper);
      if (!guardIV) {
        anchorOp = cloned;
      }
    }

    if (anchorOp != nullptr) {
      insertPt = std::next(Block::iterator(anchorOp));
    }
  }
}

int64_t SubviewFusionHelper::detectGuardDimSubviewOffset(unsigned secondaryGuardDim) {
  // Collect base memrefs written by the primary loop.
  DenseSet<Value> primaryBases;
  plan.primaryInfo()->loops.back().walk([&primaryBases](Operation *op) {
    auto store = dyn_cast<affine::AffineWriteOpInterface>(op);
    if (!store) {
      return;
    }
    primaryBases.insert(affine::getSourceMemRef(store.getMemRef()));
  });

  // Walk secondary loop body for load/store ops whose memref chain passes
  // through a SubViewOp and whose base memref matches a primary write target.
  memref::SubViewOp foundSubview;
  plan.secondaryInfo()->loops.back().walk([&foundSubview, &primaryBases](Operation *op) {
    if (foundSubview) {
      return;
    }
    Value memref;
    if (auto load = dyn_cast<affine::AffineReadOpInterface>(op)) {
      memref = load.getMemRef();
    } else if (auto store = dyn_cast<affine::AffineWriteOpInterface>(op)) {
      memref = store.getMemRef();
    } else {
      return;
    }

    bool hasSubview = false;
    memref::SubViewOp sv;
    Value base = affine::getSourceMemRef(memref, &hasSubview, &sv);
    if (hasSubview && sv && primaryBases.contains(base)) {
      foundSubview = sv;
    }
  });

  if (!foundSubview) {
    return 0;
  }

  // Extract static offset at the dimension corresponding to the secondary guard dim.
  auto offsets = foundSubview.getStaticOffsets();
  if (secondaryGuardDim >= offsets.size()) {
    return 0;
  }
  int64_t offset = offsets[secondaryGuardDim];
  if (offset == ShapedType::kDynamic || offset == 0) {
    return 0;
  }

  return offset;
}

void SubviewFusionHelper::expandPrimaryLoopForOffset() {
  if (plan.guard.kind != FusionGuard::Kind::SmallerUB || plan.guard.offset == 0) {
    return;
  }

  int64_t effectiveSecondaryUB = plan.guard.offset + plan.guard.boundValue;
  auto primaryLoop = plan.primaryInfo()->loops[plan.guard.dimPos];
  if (!primaryLoop.hasConstantUpperBound()) {
    return;
  }
  int64_t originalPrimaryUB = primaryLoop.getConstantUpperBound();
  if (effectiveSecondaryUB <= originalPrimaryUB) {
    return;
  }

  // Expand the primary loop's upper bound to cover the secondary's range.
  primaryLoop.setConstantUpperBound(effectiveSecondaryUB);

  // Wrap existing ops in the innermost loop body with guard: IV < originalPrimaryUB.
  // This prevents the primary body from executing at iterations beyond its original range.
  auto innermostLoop = plan.primaryInfo()->loops.back();
  Block *body = innermostLoop.getBody();

  auto bodyOps = body->without_terminator();
  SmallVector<Operation *, kSmallVectorSizeSixteen> opsToGuard;
  std::transform(bodyOps.begin(), bodyOps.end(), std::back_inserter(opsToGuard), [](Operation &op) { return &op; });
  if (opsToGuard.empty()) {
    return;
  }

  Value guardIV = primaryLoop.getInductionVar();
  OpBuilder builder(body, body->begin());
  AffineExpr d0 = builder.getAffineDimExpr(kGuardDimIndex);
  AffineExpr condExpr = builder.getAffineConstantExpr(originalPrimaryUB - 1) - d0;
  // eqFlags: single inequality constraint (>=).
  IntegerSet condSet = IntegerSet::get(kGuardDimCount, kGuardSymbolCount, {condExpr}, {false});

  // withElseRegion = false: no else block needed.
  auto ifOp = builder.create<affine::AffineIfOp>(innermostLoop.getLoc(), condSet, ValueRange{guardIV},
                                                 false);
  Block *thenBlock = ifOp.getThenBlock();
  for (auto *op : opsToGuard) {
    op->moveBefore(thenBlock->getTerminator());
  }
}

std::optional<SubviewFusionPlan> SubviewFusionHelper::tryFuse(FusionLoopNestInfo &srcInfo, FusionLoopNestInfo &dstInfo,
                                                              int srcRank, int dstRank) {
  plan = {};  // reset for each fusion attempt
  if (!buildSubviewFusionPlan(srcInfo, dstInfo, srcRank, dstRank)) {
    return std::nullopt;
  }

  auto stages = collectCloneStages(*plan.secondaryInfo());
  if (stages.empty()) {
    return std::nullopt;
  }

  // When a subview offset makes the secondary's effective range exceed the
  // primary loop's upper bound, expand the loop and guard the primary body.
  expandPrimaryLoopForOffset();

  emitCloneStages(stages);
  return plan;
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

// Extracts the maximum memref rank across all access ops in the list.
// Returns -1 if no valid access is found.
static int getRankFromAccesses(const SmallVector<Operation *, kSmallVectorSizeFour> &accesses) {
  int maxRank = kInvalidRank;
  for (auto *op : accesses) {
    Value memref;
    if (auto read = dyn_cast<affine::AffineReadOpInterface>(op)) {
      memref = read.getMemRef();
    } else if (auto write = dyn_cast<affine::AffineWriteOpInterface>(op)) {
      memref = write.getMemRef();
    }
    if (memref) {
      maxRank = std::max(maxRank, static_cast<int>(cast<MemRefType>(memref.getType()).getRank()));
    }
  }
  return maxRank;
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

// Collect src spine ops to clone, in src program order: off-spine ops at each level,
// descending into the spine child at its program position. Spine for-ops and
// terminators are excluded.
static void collectSpineCloneOps(ArrayRef<affine::AffineForOp> srcSpine, unsigned level,
                                 SmallVector<Operation *, kSmallVectorSizeSixteen> &out) {
  affine::AffineForOp cur = srcSpine[level];
  affine::AffineForOp childLoop = (level + 1 < srcSpine.size()) ? srcSpine[level + 1] : affine::AffineForOp();
  Operation *child = childLoop ? childLoop.getOperation() : nullptr;
  for (Operation &op : cur.getBody()->without_terminator()) {
    if ((child != nullptr) && &op == child) {
      collectSpineCloneOps(srcSpine, level + 1, out);
      continue;
    }
    out.push_back(&op);
  }
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

// Deepest dst spine level the op touches, via the shared-memref axis alignment in levelMap.
static unsigned requiredDstLevel(Operation *op, ArrayRef<affine::AffineForOp> srcSpine, ArrayRef<unsigned> levelMap) {
  unsigned lvl = 0;
  op->walk([&lvl, &srcSpine, &levelMap](Operation *n) {
    for (Value v : n->getOperands()) {
      int sl = spineLevelOf(v, srcSpine);
      if (sl >= 0) {
        lvl = std::max(lvl, levelMap[sl]);
      }
    }
  });
  return lvl;
}

// Transposed-axis fusion: place each src op at the dst spine level matching the deepest
// dep-memref axis it touches, so every remapped IV stays in scope. src ran fully before
// dst, so cloning src ops before dst's own ops at each level is safe. Returns false if an
// SSA value would be consumed outside its producer's scope (needs loop restructuring).
static bool fuseSpineTransposed(ArrayRef<affine::AffineForOp> srcSpine, ArrayRef<affine::AffineForOp> dstSpine,
                                ArrayRef<unsigned> levelMap, IRMapping &mapper, unsigned fusionDepth) {
  SmallVector<Operation *, kSmallVectorSizeSixteen> cloneOps;
  collectSpineCloneOps(srcSpine, 0, cloneOps);

  DenseMap<Operation *, unsigned> assigned;
  for (Operation *op : cloneOps) {
    assigned[op] = requiredDstLevel(op, srcSpine, levelMap);
  }

  for (Operation *op : cloneOps) {
    unsigned useLvl = assigned[op];
    bool outOfScope = false;
    op->walk([&outOfScope, &assigned, &useLvl](Operation *n) {
      for (Value v : n->getOperands()) {
        if (Operation *def = v.getDefiningOp()) {
          if (assigned.count(def) && assigned[def] > useLvl) {
            outOfScope = true;
          }
        }
      }
    });
    if (outOfScope) {
      return false;
    }
  }

  SmallVector<std::unique_ptr<OpBuilder>, kSmallVectorSizeFour> levelBuilders;
  for (unsigned d = 0; d < fusionDepth; ++d) {
    affine::AffineForOp dstD = dstSpine[d];
    levelBuilders.push_back(std::make_unique<OpBuilder>(dstD.getBody(), dstD.getBody()->begin()));
  }

  for (Operation *op : cloneOps) {
    levelBuilders[assigned[op]]->clone(*op, mapper);
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
  return fuseSpineTransposed(srcSpine, dstSpine, levelMap, mapper, fusionDepth);
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

unsigned FusionCodeGenHelper::findMaxLegalFusionDepth(
  const FusionLoopNestInfo &srcInfo, const FusionLoopNestInfo &dstInfo, const affine::FusionStrategy &strategy,
  llvm::SmallVector<affine::ComputationSliceState, kSmallVectorSizeEight> &depthSliceUnions, const FusionPlan &plan,
  FusionAccessInfo &accessInfo, bool srcDstReversed) {
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

  unsigned depDepth = plan.depInfo.loopDepth;
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
  if (srcGroupTemplate == OperatorTemplate::Broadcast &&
      dstGroupTemplate == OperatorTemplate::Broadcast &&
      !srcInfo.isPerfect && dstInfo.isPerfect &&
      dstInfo.loopDepth > srcInfo.loopDepth &&
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

  FusionAccessInfo accessInfo;
  bool haveAccesses =
    plan.depInfo.memref && collectFusionAccesses(plan.depInfo.memref, srcAffineForOp, dstAffineForOp, accessInfo);
  if (haveAccesses && plan.depInfo.memrefKind != MemrefKind::Subview &&
      isDependenceAxisRelationUnresolved(accessInfo.srcAccesses, accessInfo.dstAccesses)) {
    return;
  }

  if (plan.fusionType == "I" || !haveAccesses) {
    doIFuse(srcGroupId, dstGroupId, srcInfo, dstInfo, plan);
    return;
  }

  if (trySubviewFuse(srcGroupId, dstGroupId, accessInfo, srcInfo, dstInfo, plan)) {
    return;
  }

  affine::FusionStrategy strategy = getFusionStrategy(plan);

  SmallVector<affine::ComputationSliceState, kSmallVectorSizeEight> depthSliceUnions;
  unsigned maxLegalFusionDepth = 0;

  auto keepSrcDstOpt = resolveKeepSrcDst(mdg, srcGroupId, dstGroupId, srcInfo, dstInfo);
  if (!keepSrcDstOpt) {
    return;
  }
  bool keepSrcDst = *keepSrcDstOpt;

  if (keepSrcDst) {
    maxLegalFusionDepth = findMaxLegalFusionDepth(srcInfo, dstInfo, strategy, depthSliceUnions, plan, accessInfo);
  } else {
    maxLegalFusionDepth = findMaxLegalFusionDepth(dstInfo, srcInfo, strategy, depthSliceUnions, plan, accessInfo, true);
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

bool FusionCodeGenHelper::trySubviewFuse(unsigned srcGroupId, unsigned dstGroupId, const FusionAccessInfo &accessInfo,
                                         FusionLoopNestInfo &srcInfo, FusionLoopNestInfo &dstInfo,
                                         const FusionPlan &plan) {
  if (plan.depInfo.memrefKind != MemrefKind::Subview) {
    return false;
  }
  int srcRank = getRankFromAccesses(accessInfo.srcAccesses);
  int dstRank = getRankFromAccesses(accessInfo.dstAccesses);
  if (auto fusionPlan = subviewHelper.tryFuse(srcInfo, dstInfo, srcRank, dstRank)) {
    unsigned erasedGid = fusionPlan->srcIsPrimary ? dstGroupId : srcGroupId;
    unsigned aliasGid = fusionPlan->srcIsPrimary ? srcGroupId : dstGroupId;
    eraseLoopAndCleanupNode(erasedGid, aliasGid, fusionPlan->secondaryInfo()->root);
    return true;
  }
  return false;
}

affine::FusionStrategy FusionCodeGenHelper::getFusionStrategy(const FusionPlan &plan) const {
  if (plan.fusionType == "H") {
    return affine::FusionStrategy(affine::FusionStrategy::ProducerConsumer);
  }
  if (plan.depInfo.memref) {
    return affine::FusionStrategy(plan.depInfo.memref);
  }
  return affine::FusionStrategy(affine::FusionStrategy::Generic);
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
