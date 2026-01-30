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
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
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
using llvm::raw_ostream;
using llvm::SetVector;
using llvm::SmallVector;
// Check if the operation is an elementwise operation
// Elementwise operations perform the same computation on each element independently
static bool isElementwiseOp(Operation *op) {
  return (
    isa<arith::AddFOp, arith::AddIOp, arith::TruncFOp, arith::TruncIOp, arith::ExtFOp, arith::CmpFOp, arith::CmpIOp,
        arith::MulFOp, arith::MulIOp, arith::SubFOp, arith::SubIOp, arith::AndIOp, arith::OrIOp, arith::NegFOp>(op));
}

// Check if the operation involves broadcasting between two operands
// Broadcasting occurs when operands have different shapes that need to be aligned
template <typename T>
static bool isBroadcastOp(T lhs, T rhs) {
  auto getShapeInfo = [&](T memrefOp) -> llvm::ArrayRef<int64_t> {
    llvm::ArrayRef<int64_t> shapeInfo;
    auto memref = memrefOp.getMemref();
    if (!memref) {
      return shapeInfo;
    }
    auto memrefType = memref.getType();
    if (!memrefType) {
      return shapeInfo;
    }
    // TODO(hjh): support dynamic shape by symbolic info
    shapeInfo = memrefType.getShape();
    return shapeInfo;
  };
  auto lhsShapeInfo = getShapeInfo(lhs);
  auto rhsShapeInfo = getShapeInfo(rhs);

  // Different number of dimensions indicates broadcasting
  if (lhsShapeInfo.size() != rhsShapeInfo.size()) {
    return true;
  }

  // Check if any corresponding dimensions have different sizes
  for (size_t i = 0; i < lhsShapeInfo.size(); ++i) {
    auto lhsDim = lhsShapeInfo[i];
    auto rhsDim = rhsShapeInfo[i];
    if (lhsDim != rhsDim) {
      return true;
    }
  }
  return false;
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
      }
      for (auto *opInst : collector.storeOpInsts) {
        node.stores.push_back(opInst);
        auto memref = cast<affine::AffineWriteOpInterface>(opInst).getMemRef();
        memrefAccesses[memref].insert(node.id);
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
      nodes.insert({node.id, node});
    } else if (auto storeOp = dyn_cast<affine::AffineWriteOpInterface>(op)) {
      // Create graph node for top-level store op.
      Node node(nextNodeId++, op);
      node.stores.push_back(op);
      auto memref = cast<affine::AffineWriteOpInterface>(op).getMemRef();
      memrefAccesses[memref].insert(node.id);
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

bool MemRefDependenceGraphForFusion::createEdges(const DenseMap<Value, SetVector<unsigned>> &memrefAccesses) {
  // First call the base class method to create edges for producer-consumer relationships
  if (!MemRefDependenceGraph::createEdges(memrefAccesses)) {
    return false;
  }

  // Add edges for multiple loads from the same memref
  // If multiple load statements in different loops load the same memref,
  // build edges between the AffineLoadOp nodes according to id order
  for (auto &memrefAndList : memrefAccesses) {
    Value memref = memrefAndList.first;
    const SetVector<unsigned> &nodeIds = memrefAndList.second;

    // Only process if there are multiple nodes accessing this memref
    if (nodeIds.size() < 2) {
      continue;
    }

    // Collect all AffineLoadOp nodes that load this memref
    SmallVector<unsigned> loadNodeIds;

    for (unsigned nodeId : nodeIds) {
      Node *node = getNode(nodeId);

      // If the node itself is an AffineLoadOp and loads this memref
      if (auto loadOpInterface = dyn_cast<affine::AffineReadOpInterface>(node->op)) {
        if (loadOpInterface.getMemRef() == memref) {
          loadNodeIds.push_back(nodeId);
        }
      } else if (isa<affine::AffineForOp>(node->op)) {
        // If the node is an AffineForOp, check its internal loads
        // Find all AffineLoadOp operations inside this loop that load this memref
        for (Operation *loadOpInst : node->loads) {
          if (auto loadOpInterface = dyn_cast<affine::AffineReadOpInterface>(loadOpInst)) {
            if (loadOpInterface.getMemRef() == memref) {
              // Check if this load operation has its own node
              int loadNodeId = getNodeId(loadOpInst);
              if (loadNodeId != -1) {
                // This load has its own node, add it
                if (std::find(loadNodeIds.begin(), loadNodeIds.end(), loadNodeId) == loadNodeIds.end()) {
                  loadNodeIds.push_back(loadNodeId);
                }
              } else {
                // This load doesn't have its own node (it's inside a loop)
                // Create a node for it so we can add edges between AffineLoadOp nodes
                Node loadNode(nextNodeId++, loadOpInst);
                loadNode.loads.push_back(loadOpInst);
                nodes.insert({loadNode.id, loadNode});
                loadNodeIds.push_back(loadNode.id);
              }
            }
          }
        }
      }
    }

    // If we found multiple load nodes, create edges between them
    if (loadNodeIds.size() >= 2) {
      // Sort by node id to ensure we process from smaller to larger
      std::sort(loadNodeIds.begin(), loadNodeIds.end());

      // Create edges from smaller id to larger id
      for (unsigned i = 0; i < loadNodeIds.size(); ++i) {
        for (unsigned j = i + 1; j < loadNodeIds.size(); ++j) {
          unsigned srcId = loadNodeIds[i];
          unsigned dstId = loadNodeIds[j];
          // Add edge from smaller id to larger id
          if (!hasEdge(srcId, dstId, memref)) {
            addEdge(srcId, dstId, memref);
          }
        }
      }
    }
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

static llvm::ArrayRef<int64_t> getStaticShape(Value memrefVal,
                                              llvm::SmallVector<int64_t, 4> &shapeBuf) {
  auto memrefType = memrefVal.getType().dyn_cast<mlir::MemRefType>();
  if (!memrefType || !memrefType.hasStaticShape()) {
    return {};
  }
  shapeBuf.assign(memrefType.getShape().begin(), memrefType.getShape().end());
  return shapeBuf;
}

static int64_t getShapeProduct(llvm::ArrayRef<int64_t> shape) {
  if (shape.empty())
    return -1;
  int64_t prod = 1;
  for (int64_t d : shape) {
    if (d < 0)
      return -1;
    prod *= d;
  }
  return prod;
}

static bool isSameShape(llvm::ArrayRef<int64_t> a, llvm::ArrayRef<int64_t> b) {
  if (a.size() != b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i])
      return false;
  }
  return true;
}

static bool isPermutationShape(llvm::ArrayRef<int64_t> a, llvm::ArrayRef<int64_t> b) {
  if (a.size() != b.size())
    return false;
  llvm::SmallVector<int64_t, 4> aSorted(a.begin(), a.end());
  llvm::SmallVector<int64_t, 4> bSorted(b.begin(), b.end());
  std::sort(aSorted.begin(), aSorted.end());
  std::sort(bSorted.begin(), bSorted.end());
  return isSameShape(aSorted, bSorted);
}

static bool isBroadcastLike(llvm::ArrayRef<int64_t> inShape,
                            llvm::ArrayRef<int64_t> outShape) {
  if (inShape.size() != outShape.size())
    return false;

  bool diffFound = false;
  for (size_t i = 0; i < inShape.size(); ++i) {
    auto inD  = inShape[i];
    auto outD = outShape[i];
    if (inD == outD)
      continue;

    diffFound = true;
    if (inD != 1) {
      return false;
    }
  }
  return diffFound;
}

static void classifyIOShape(llvm::ArrayRef<int64_t> inShape,
                            llvm::ArrayRef<int64_t> outShape,
                            bool &hasElementwise,
                            bool &hasBroadcast,
                            bool &hasReshape,
                            bool &hasTranspose) {
  if (inShape.empty() || outShape.empty())
    return;

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

  auto inProd  = getShapeProduct(inShape);
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
    if (op->hasAttr("reduction_type")) {
      return OperatorTemplate::Reduce;
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
  bool hasBroadcast   = false;
  bool hasReshape     = false;
  bool hasTranspose   = false;

  for (auto load : loads) {
    Value inMemref = load.getMemRef();
    llvm::SmallVector<int64_t, 4> inShapeBuf;
    auto inShape = getStaticShape(inMemref, inShapeBuf);
    if (inShape.empty()) {
      continue;
    }

    classifyIOShape(inShape, outShape,
                    hasElementwise,
                    hasBroadcast,
                    hasReshape,
                    hasTranspose);
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

bool MemRefDependenceGraphForFusion::elementwiseMatch(Operation *op) {
  int nodeId = getNodeId(op);
  if (nodeId == -1) {
    return true;
  }
  llvm::DenseSet<unsigned> dependentIds;
  getPredecessorNodes(nodeId, dependentIds);
  if (dependentIds.size() != 2) {
    return true;
  }
  std::vector<unsigned> oprandIds;
  for (auto id : dependentIds) {
    auto memrefSourceId = getMemrefSourceOfNode(id);
    if (memrefSourceId == -1 || !isa<affine::AffineLoadOp>(getNode(memrefSourceId)->op)) {
      // We regard const as elem for simplification cause they won't introduce any loop.
      return true;
    }
    oprandIds.emplace_back(memrefSourceId);
  }
  auto res = !isBroadcastOp(dyn_cast<affine::AffineLoadOp>(getNode(oprandIds[0])->op),
                            dyn_cast<affine::AffineLoadOp>(getNode(oprandIds[1])->op));
  return res;
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

// Create ComputationSliceState for loops with same structure but no dependencies
// This creates a slice state that represents the full loop bounds (no slicing)
// For loops without dependencies, the source loop should execute completely for each
// iteration of the destination loop at the fusion depth
static bool createSliceStateForSameStructure(affine::AffineForOp srcForOp, affine::AffineForOp dstForOp,
                                             unsigned loopDepth, affine::ComputationSliceState *sliceState) {
  // Get loop nests (including the loops themselves)
  SmallVector<affine::AffineForOp, 4> srcLoops;
  collectLoopNest(srcForOp, srcLoops);
  SmallVector<affine::AffineForOp, 4> dstLoops;
  collectLoopNest(dstForOp, dstLoops);

  if (loopDepth == 0 || loopDepth > dstLoops.size() || loopDepth > srcLoops.size()) {
    return false;
  }

  // Check if src precedes dst in the block
  bool isSrcBeforeDst = srcForOp->isBeforeInBlock(dstForOp);

  // Clear the slice state
  sliceState->clearBounds();
  sliceState->ivs.clear();

  // For fusion, we need to slice the source loop nest starting from loopDepth
  // The slice should represent the full iteration space of source loops
  unsigned numSliceLoops = srcLoops.size() - (loopDepth - 1);
  if (numSliceLoops == 0) {
    return false;
  }

  // Get the slice loop IVs (from loopDepth-1 to innermost of src)
  for (unsigned i = loopDepth - 1; i < srcLoops.size(); ++i) {
    sliceState->ivs.push_back(srcLoops[i].getInductionVar());
  }

  // Set up bounds - use the original source loop bounds
  unsigned numSliceLoopIVs = sliceState->ivs.size();
  sliceState->lbs.resize(numSliceLoopIVs);
  sliceState->ubs.resize(numSliceLoopIVs);

  // Collect bound operands from outer loops (up to loopDepth-1)
  // These will be used as symbols in the slice bounds
  SmallVector<Value, 4> outerBoundOperands;
  for (unsigned i = 0; i < loopDepth - 1; ++i) {
    auto lbOps = dstLoops[i].getLowerBoundOperands();
    auto ubOps = dstLoops[i].getUpperBoundOperands();
    outerBoundOperands.append(lbOps.begin(), lbOps.end());
    outerBoundOperands.append(ubOps.begin(), ubOps.end());
  }

  // Set bounds for each slice loop
  // For loops with same structure, we need to ensure that AffineMaps and operands match
  // The key issue is that AffineMap's symbol indices must match the operands order
  sliceState->lbOperands.resize(numSliceLoopIVs);
  sliceState->ubOperands.resize(numSliceLoopIVs);

  // Check if all source loops have the same bound operands
  // If they do, we can use source loop's maps and operands directly
  bool allOperandsMatch = true;
  auto firstSrcLoop = srcLoops[loopDepth - 1];
  auto firstLbOps = firstSrcLoop.getLowerBoundOperands();
  auto firstUbOps = firstSrcLoop.getUpperBoundOperands();

  for (unsigned i = 1; i < numSliceLoopIVs; ++i) {
    auto srcLoop = srcLoops[loopDepth - 1 + i];
    auto lbOps = srcLoop.getLowerBoundOperands();
    auto ubOps = srcLoop.getUpperBoundOperands();

    if (lbOps.size() != firstLbOps.size() || ubOps.size() != firstUbOps.size()) {
      allOperandsMatch = false;
      break;
    }
    for (unsigned j = 0; j < lbOps.size(); ++j) {
      if (lbOps[j] != firstLbOps[j]) {
        allOperandsMatch = false;
        break;
      }
    }
    if (!allOperandsMatch) break;
    for (unsigned j = 0; j < ubOps.size(); ++j) {
      if (ubOps[j] != firstUbOps[j]) {
        allOperandsMatch = false;
        break;
      }
    }
    if (!allOperandsMatch) break;
  }

  if (!allOperandsMatch) {
    // If operands don't match, we cannot safely create slice state
    // Return false to skip this fusion depth
    return false;
  }

  // Build unified operands list
  // For same-structure loops, we use source loop's maps and their original operands
  // The maps expect these specific operands in this specific order
  // We don't add outer loop IVs here because the maps may not expect them as symbols
  SmallVector<Value, 4> unifiedLbOperands;
  SmallVector<Value, 4> unifiedUbOperands;

  // Use source loop's bound operands directly (all loops have the same operands)
  // This ensures the AffineMap's symbol indices match the operands
  unifiedLbOperands.append(firstLbOps.begin(), firstLbOps.end());
  unifiedUbOperands.append(firstUbOps.begin(), firstUbOps.end());

  // Set bounds for each slice loop using source loop's maps
  // All lbOperands and ubOperands must be the same (as required by getAsConstraints)
  for (unsigned i = 0; i < numSliceLoopIVs; ++i) {
    auto srcLoop = srcLoops[loopDepth - 1 + i];

    // Use source loop's maps since slice IVs are from source loops
    // and maps must match their original operands to avoid symbol index mismatches
    sliceState->lbs[i] = srcLoop.getLowerBoundMap();
    sliceState->ubs[i] = srcLoop.getUpperBoundMap();

    // All operands must be the same for all loops (as per getAsConstraints requirement)
    // Use source loop's original operands to match the maps
    sliceState->lbOperands[i] = unifiedLbOperands;
    sliceState->ubOperands[i] = unifiedUbOperands;
  }

  // Set insertion point in destination loop
  if (isSrcBeforeDst) {
    // Forward slice: insert at the end of the destination loop at loopDepth-1
    sliceState->insertPoint = std::prev(dstLoops[loopDepth - 1].getBody()->end());
  } else {
    // Backward slice: insert at the beginning of the destination loop at loopDepth-1
    sliceState->insertPoint = dstLoops[loopDepth - 1].getBody()->begin();
  }

  return true;
}

static unsigned findMaxLegalFusionDepth(affine::AffineForOp srcAffineForOp, affine::AffineForOp dstAffineForOp,
                                        unsigned dstLoopDepthTest, const affine::FusionStrategy &strategy,
                                        llvm::SmallVector<affine::ComputationSliceState, 8> &depthSliceUnions) {
  depthSliceUnions.clear();
  depthSliceUnions.resize(dstLoopDepthTest);
  unsigned maxLegalFusionDepth = 0;

  // Try normal fusion with dependencies
  for (unsigned i = 1; i <= dstLoopDepthTest; ++i) {
    affine::FusionResult result = canFuseLoops(srcAffineForOp, dstAffineForOp,
                                               /*dstLoopDepth=*/i, &depthSliceUnions[i - 1], strategy);

    if (result.value == affine::FusionResult::Success) {
      maxLegalFusionDepth = i;
    }
  }

  return maxLegalFusionDepth;
}

void FusionCodeGenHelper::doVFuse(unsigned srcId, unsigned dstId, affine::AffineForOp srcAffineForOp,
                                  affine::AffineForOp dstAffineForOp) {
  auto *dstNode = mdg.getNode(dstId);
  mlir::Value memref;
  for (auto op : dstNode->loads) {
    auto loadOp = dyn_cast<affine::AffineReadOpInterface>(op);
    if (!loadOp) {
      continue;
    }
    if (loadOp.getMemRef().getDefiningOp()) {
      memref = loadOp.getMemRef();
    }
  }

  unsigned dstLoopDepthTest = 0;
  dstAffineForOp.walk([&](mlir::affine::AffineForOp op) { dstLoopDepthTest++; });

  SmallVector<affine::ComputationSliceState, 8> depthSliceUnions;
  unsigned maxLegalFusionDepth = 0;
  if (memref) {
    affine::FusionStrategy strategy(memref);
    maxLegalFusionDepth =
      findMaxLegalFusionDepth(srcAffineForOp, dstAffineForOp, dstLoopDepthTest, strategy, depthSliceUnions);
  } else {
    // No memref found, try generic strategy for structure-based fusion
    affine::FusionStrategy strategy(affine::FusionStrategy::Generic);
    maxLegalFusionDepth =
      findMaxLegalFusionDepth(srcAffineForOp, dstAffineForOp, dstLoopDepthTest, strategy, depthSliceUnions);
  }

  // Skip if fusion is not feasible at any loop depths.
  if (maxLegalFusionDepth == 0) {
    doIFuse(srcId, dstId, srcAffineForOp, dstAffineForOp);
    return;
  }

  unsigned bestDstLoopDepth = maxLegalFusionDepth;
  assert(bestDstLoopDepth > 0 && "Unexpected loop fusion depth");
  affine::ComputationSliceState &bestSlice = depthSliceUnions[bestDstLoopDepth - 1];
  assert(!bestSlice.isEmpty() && "Fusion depth has no computed slice union");
  fuseLoops(srcAffineForOp, dstAffineForOp, bestSlice, true);
  srcAffineForOp.erase();
  nodeAlias[srcId] = dstId;
}

void FusionCodeGenHelper::doHFuse(unsigned srcId, unsigned dstId, affine::AffineForOp srcAffineForOp,
                                  affine::AffineForOp dstAffineForOp) {
  unsigned dstLoopDepthTest = 0;
  dstAffineForOp.walk([&](affine::AffineForOp op) { dstLoopDepthTest++; });

  // Check the feasibility of fusing src loop nest into dst loop nest
  // at loop depths in range [1, dstLoopDepthTest].
  SmallVector<affine::ComputationSliceState, 8> depthSliceUnions;
  affine::FusionStrategy strategy(affine::FusionStrategy::ProducerConsumer);
  unsigned maxLegalFusionDepth =
    findMaxLegalFusionDepth(srcAffineForOp, dstAffineForOp, dstLoopDepthTest, strategy, depthSliceUnions);

  if (maxLegalFusionDepth == 0) {
    doIFuse(srcId, dstId, srcAffineForOp, dstAffineForOp);
    return;
  }

  unsigned bestDstLoopDepth = maxLegalFusionDepth;
  assert(bestDstLoopDepth > 0 && "Unexpected loop fusion depth");
  affine::ComputationSliceState &bestSlice = depthSliceUnions[bestDstLoopDepth - 1];
  assert(!bestSlice.isEmpty() && "Missing slice union for depth");
  fuseLoops(srcAffineForOp, dstAffineForOp, bestSlice);

  // srcNode is no longer valid after it is removed from mdg.
  srcAffineForOp.erase();
  nodeAlias[srcId] = dstId;
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
                                  affine::AffineForOp dstAffineForOp) {
  // Collect loop nests to determine fusion depth
  SmallVector<affine::AffineForOp, 4> srcLoops;
  collectLoopNest(srcAffineForOp, srcLoops);
  SmallVector<affine::AffineForOp, 4> dstLoops;
  collectLoopNest(dstAffineForOp, dstLoops);

  // Check if loops have the same structure - if so, fuse directly without slice state checks
  if (!hasSameLoopStructure(srcLoops, dstLoops)) {
    return;
  }

  // For same-structure loops, use the full depth for fusion
  unsigned bestDstLoopDepth = dstLoops.size();
  assert(bestDstLoopDepth > 0 && "Unexpected loop fusion depth");

  // Perform fusion using the helper function
  performLoopFusion(srcLoops, dstLoops, bestDstLoopDepth);

  srcAffineForOp.erase();
  nodeAlias[srcId] = dstId;
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
  os << indent << ">> LoopTransforms: [\n";
  for (auto it : nodeTransformRecords) {
    os << indent << indent << ">> Node " << it.first << ": [";
    for (LoopTransform lt : it.second) {
      int ltI = static_cast<int>(lt);
      auto it2 = loopTransformToStr.find(ltI);
      if (it2 != loopTransformToStr.end()) {
        os << it2->second << " -> ";
      } else {
        os << ltI << " -> ";
      }
    }
    os << "]\n";
  }
  os << indent << indent << "]\n";
}

}  // namespace akg
}  // namespace mlir
