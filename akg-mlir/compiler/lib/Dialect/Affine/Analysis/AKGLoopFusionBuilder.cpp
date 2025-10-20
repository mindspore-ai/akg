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

using namespace mlir;
using namespace llvm;
using namespace akg;
using namespace akgglobal;

namespace mlir {
namespace akg {
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
    // TODO: support dynamic shape by symbolic info
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

// Get all groups that contain any of the specified nodes
std::unordered_set<GroupPtr> MemRefDependenceGraphForFusion::getGroupsByNode(llvm::DenseSet<unsigned> nodeIds) {
  std::unordered_set<GroupPtr> allgroups;
  for (auto id : nodeIds) {
    auto group = getGroupByNode(id);
    if (group == nullptr) {
      continue;
    } else {
      allgroups.insert(group);
    }
  }
  return allgroups;
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
      // TODO: check this condition
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
               isa<memref::AllocOp, arith::ConstantOp, affine::AffineApplyOp, affine::AffineYieldOp, func::ReturnOp>(op)) {
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

void MemRefDependenceGraphForFusion::print(raw_ostream &os) const {
  os << "MemRefDependenceGraphForFusion!!!\n";
  MemRefDependenceGraph::print(os);
  for (auto it : groups) {
    auto g = it.second;
    std::string groupTemplateString = g->getGroupTemplateString();
    os << "Group " << g->groupId << " (GroupTemplate " << groupTemplateString << ") IsGlobalOut (" << g->isGlobalOut << ") root is "
       << g->rootId << " has " << g->nodesId.size() << " nodes inside: [";
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
    if (op->hasAttr("reduction_type")) {
      return OperatorTemplate::Reduce;
    }
    if (isElementwiseOp(op)) {
      if (elementwiseMatch(op)) {
        return OperatorTemplate::Elementwise;
      } else {
        return OperatorTemplate::Broadcast;
      }
    }
    if (isa<memref::ExpandShapeOp, memref::CollapseShapeOp>(op)) {
      // TODO(baiji): cannot identify reshape by op after FoldMemRefAliasOps pass
      return OperatorTemplate::Reshape;
    }
    if (auto load = dyn_cast<affine::AffineLoadOp>(op)) {
      loads.emplace_back(load);
    }
    if (auto store = dyn_cast<affine::AffineStoreOp>(op)) {
      stores.emplace_back(store);
    }
  }
  if (loads.size() == 1 && stores.size() == 1) {
    // TODO: check transpose
    return OperatorTemplate::Transpose;
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

// Check if a node represents a global memory reference
// Global memrefs are those that write to function arguments or have specific patterns
// Case1: Direct store to function argument (global output)
//   affine.store %2, %arg3[%arg4] : memref<15xf32>  --> return true
// Case2: Store to local alloc followed by load and store to global
//   affine.for ...
//     affine.store ..., %alloc_2[] : memref<f32>  --> return true
//   %1 = affine.load %alloc_2[] : memref<f32>
//   affine.store %1, %arg2[0] : memref<1xf32>
bool MemRefDependenceGraphForFusion::isGlobalMemref(unsigned id) {
  // Get the operation for this node
  auto op = getNode(id)->op;
  if (!op) {
    return false;
  }
  
  // Must be a store operation
  auto write = dyn_cast<affine::AffineStoreOp>(op);
  if (!write) {
    return false;
  }
  
  // Get the memref being written to
  auto memref = write.getMemref();
  if (!memref) {
    return false;
  }
  
  // Case1: Direct store to function argument (no defining operation)
  if (!memref.getDefiningOp()) {
    return true;
  }
  
  // Case2: Check for local alloc -> load -> global store pattern
  std::vector<unsigned> successorIds;
  getSuccessorNodes(id, successorIds);
  
  // 1. Must have exactly one successor
  if (successorIds.size() != 1) {
    return false;
  }
  
  // 2. Successor must be a load operation
  auto writeTo = getNode(successorIds.back())->op;
  if (!writeTo || !isa<affine::AffineLoadOp>(writeTo)) {
    return false;
  }
  
  // 3. Load must have exactly one successor
  std::vector<unsigned> grandchild;
  getSuccessorNodes(successorIds.back(), grandchild);
  if (grandchild.size() != 1) {
    return false;
  }

  // 4. Grandchild must be a store to global memory
  if (auto store = dyn_cast<affine::AffineStoreOp>(getNode(grandchild.back())->op)) {
    auto storemem = store.getMemref();
    if (!storemem.getDefiningOp()) {
      return true;
    }
  }
  
  return false;
}

unsigned FusionCodeGenHelper::getAliasId(unsigned srcId) {
  auto it = nodeAlias.find(srcId);
  if (it != nodeAlias.end()) {
    llvm::outs() << "Find alias id " << srcId << " -> " << it->second << "\n";
    return it->second;
  }
  return srcId;
}

void FusionCodeGenHelper::doVFuse(unsigned srcId, unsigned dstId, affine::AffineForOp sibAffineForOp,
                                  affine::AffineForOp dstAffineForOp, unsigned maxLegalFusionDepth, unsigned dstLoopDepthTest) {
  // dstId = getAliasId(dstId);
  llvm::outs() << "Perform V Fusion at " << srcId << " to " << dstId << "\n";
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
  if (memref) {
    SmallVector<affine::ComputationSliceState, 8> depthSliceUnions;
    depthSliceUnions.resize(dstLoopDepthTest);
    affine::FusionStrategy strategy(memref);
    for (unsigned i = 1; i <= dstLoopDepthTest; ++i) {
      affine::FusionResult result = canFuseLoops(sibAffineForOp, dstAffineForOp,
                                                 /*dstLoopDepth=*/i, &depthSliceUnions[i - 1], strategy);

      if (result.value == affine::FusionResult::Success) {
        maxLegalFusionDepth = i;
        llvm::outs() << "fuseWithSiblingNodes to maxLegalFusionDepth " << maxLegalFusionDepth << "\n";
      }
    }
    // Skip if fusion is not feasible at any loop depths.
    if (maxLegalFusionDepth == 0) {
      llvm::outs() << "fusion is not feasible at any loop depths.\n";
      return;
    }

    unsigned bestDstLoopDepth = maxLegalFusionDepth;
    assert(bestDstLoopDepth > 0 && "Unexpected loop fusion depth");
    assert(!depthSliceUnions[bestDstLoopDepth - 1].isEmpty() && "Fusion depth has no computed slice union");
    bool isInnermostInsertion = true;  //?
    fuseLoops(sibAffineForOp, dstAffineForOp, depthSliceUnions[bestDstLoopDepth - 1], isInnermostInsertion);
    sibAffineForOp.erase();
    nodeAlias[srcId] = dstId;
  }
}

void FusionCodeGenHelper::doHFuse(unsigned srcId, unsigned dstId, affine::AffineForOp srcAffineForOp,
                                  affine::AffineForOp dstAffineForOp) {
  // dstId = getAliasId(dstId);
  llvm::outs() << "Perform H Fusion at " << srcId << " to " << dstId << "\n";
  auto *srcNode = mdg.getNode(srcId);
  auto *dstNode = mdg.getNode(dstId);
  mlir::DenseSet<mlir::Value> producerConsumerMemrefs;
  affine::gatherProducerConsumerMemrefs(srcNode->stores, dstNode->loads, producerConsumerMemrefs);

  unsigned dstLoopDepthTest = 0;
  dstAffineForOp.walk([&](affine::AffineForOp op) { dstLoopDepthTest++; });

  // Check the feasibility of fusing src loop nest into dst loop nest
  // at loop depths in range [1, dstLoopDepthTest].
  unsigned maxLegalFusionDepth = 0;
  SmallVector<affine::ComputationSliceState, 8> depthSliceUnions;
  depthSliceUnions.resize(dstLoopDepthTest);
  affine::FusionStrategy strategy(affine::FusionStrategy::ProducerConsumer);
  for (unsigned i = 1; i <= dstLoopDepthTest; ++i) {
    affine::FusionResult result = canFuseLoops(srcAffineForOp, dstAffineForOp,
                                               /*dstLoopDepth=*/i, &depthSliceUnions[i - 1], strategy);
    if (result.value == affine::FusionResult::Success) maxLegalFusionDepth = i;
  }

  if (maxLegalFusionDepth == 0) {
    llvm::outs() << "Can't fuse: fusion is not legal at any depth\n";
    return;
  }
  llvm::outs() << "Final maxLegalFusionDepth: " << maxLegalFusionDepth << "\n";

  unsigned bestDstLoopDepth = maxLegalFusionDepth;
  // Retrieve producer stores from the src loop.
  SmallVector<Operation *, 2> producerStores;
  for (Operation *op : srcNode->stores)
    if (producerConsumerMemrefs.count(cast<affine::AffineWriteOpInterface>(op).getMemRef())) producerStores.push_back(op);

  // TODO: Suppport multiple producer stores in profitability
  // analysis. We limit profitability analysis to only scenarios with
  // a single producer store for now. Note that some multi-store
  // producer scenarios will still go through profitability analysis
  // if only one of the stores is involved the producer-consumer
  // relationship of the candidate loops.
  if (producerStores.empty()) {
    llvm::outs() << "TODO: remove this into analysis part.\n";
    doVFuse(srcId, dstId, srcAffineForOp, dstAffineForOp, maxLegalFusionDepth, dstLoopDepthTest);
    return;
  }
  assert(!producerStores.empty() && "Expected producer store");
  if (producerStores.size() > 1) {
    llvm::outs() << "Skipping profitability analysis. Not "
                    "supported for this case\n";
  }

  assert(bestDstLoopDepth > 0 && "Unexpected loop fusion depth");
  affine::ComputationSliceState &bestSlice = depthSliceUnions[bestDstLoopDepth - 1];
  if (bestSlice.isEmpty()) {
    return;
  }
  assert(!bestSlice.isEmpty() && "Missing slice union for depth");

  fuseLoops(srcAffineForOp, dstAffineForOp, bestSlice);

  // srcNode is no longer valid after it is removed from mdg.
  srcAffineForOp.erase();
  // mdg.removeNode(srcId);
  // srcNode = nullptr;
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
