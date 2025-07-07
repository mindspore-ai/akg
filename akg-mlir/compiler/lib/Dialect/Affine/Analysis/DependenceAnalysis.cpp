/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "akg/Dialect/Affine/Analysis/DependenceAnalysis.h"
#include "akg/Dialect/Affine/Analysis/AffineAnalysis.h"

#include "akg/Utils/AnalysisCommon.hpp"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"

#define DEBUG_TYPE "dependence-analysis"

namespace mlir {
namespace akg {

// Returns the load op count for 'memref'.
unsigned Node::getLoadOpCount(Value memref) const {
  unsigned loadOpCount = 0;
  for (Operation *loadOp : loads) {
    if (memref == cast<affine::AffineReadOpInterface>(loadOp).getMemRef()) {
      ++loadOpCount;
    }
  }
  return loadOpCount;
}

// Returns the store op count for 'memref'.
unsigned Node::getStoreOpCount(Value memref) const {
  unsigned storeOpCount = 0;
  for (Operation *storeOp : stores) {
    if (memref == cast<affine::AffineWriteOpInterface>(storeOp).getMemRef()) {
      ++storeOpCount;
    }
  }
  return storeOpCount;
}

// Returns the graph node for 'id'.
Node *MemRefDependenceGraph::getNode(unsigned id) {
  auto it = nodes.find(id);
  assert(it != nodes.end());
  return &it->second;
}

// Returns the graph id for 'op'.
int MemRefDependenceGraph::getNodeId(const Operation *op) {
  for (auto &idAndNode : nodes) {
    if (idAndNode.second.op == op) {
      return idAndNode.first;
    }
  }
  return -1;
}

// Update the op in the graph node.
// This command is used to delete the current op after it is cloned.
void MemRefDependenceGraph::updateNodeOp(const Operation *oldOp, Operation *newOp) {
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    if (it->second.op == oldOp) {
      Node *node = &it->second;
      node->op = newOp;
      return;
    }
  }
}

// Returns true iff there is an edge from node 'srcId' to node 'dstId' which
// is for 'value' if non-null, or for any value otherwise. Returns false
// otherwise.
bool MemRefDependenceGraph::hasEdge(unsigned srcId, unsigned dstId, Value value) {
  if (outEdges.count(srcId) == 0 || inEdges.count(dstId) == 0) {
    return false;
  }
  bool hasOutEdge = llvm::any_of(outEdges[srcId],
                                 [=](const Edge &edge) { return edge.id == dstId && (!value || edge.value == value); });
  bool hasInEdge =
    llvm::any_of(inEdges[dstId], [=](const Edge &edge) { return edge.id == srcId && (!value || edge.value == value); });
  return hasOutEdge && hasInEdge;
}

// Adds an edge from node 'srcId' to node 'dstId' for 'value'.
void MemRefDependenceGraph::addEdge(unsigned srcId, unsigned dstId, Value value) {
  if (!hasEdge(srcId, dstId, value)) {
    outEdges[srcId].push_back({dstId, value});
    inEdges[dstId].push_back({srcId, value});
    if (isa<MemRefType>(value.getType())) {
      memrefEdgeCount[value]++;
    }
  }
}

void MemRefDependenceGraph::print(raw_ostream &os) const {
  os << "\nMemRefDependenceGraph\n";
  os << "\nNodes:\n";
  for (const auto &idAndNode : nodes) {
    os << "Node: " << idAndNode.first << ": ";
    idAndNode.second.op->dump();
    auto it = inEdges.find(idAndNode.first);
    if (it != inEdges.end()) {
      for (const auto &e : it->second) {
        os << "  InEdge: " << e.id << " " << e.value << "\n";
      }
    }
    it = outEdges.find(idAndNode.first);
    if (it != outEdges.end()) {
      for (const auto &e : it->second) {
        os << "  OutEdge: " << e.id << " " << e.value << "\n";
      }
    }
  }
}

// Returns true if there is a path in the dependence graph from node 'srcId'
// to node 'dstId'. Returns false otherwise. `srcId`, `dstId`, and the
// operations that the edges connected are expected to be from the same block.
bool MemRefDependenceGraph::hasDependencePath(unsigned srcId, unsigned dstId) {
  // Worklist state is: <node-id, next-output-edge-index-to-visit>
  SmallVector<std::pair<unsigned, unsigned>, 4> worklist;
  worklist.push_back({srcId, 0});
  Operation *dstOp = getNode(dstId)->op;
  // Run DFS traversal to see if 'dstId' is reachable from 'srcId'.
  while (!worklist.empty()) {
    auto &idAndIndex = worklist.back();
    // Return true if we have reached 'dstId'.
    if (idAndIndex.first == dstId) {
      return true;
    }
    // Pop and continue if node has no out edges, or if all out edges have
    // already been visited.
    if (outEdges.count(idAndIndex.first) == 0 || idAndIndex.second == outEdges[idAndIndex.first].size()) {
      worklist.pop_back();
      continue;
    }
    // Get graph edge to traverse.
    Edge edge = outEdges[idAndIndex.first][idAndIndex.second];
    // Increment next output edge index for 'idAndIndex'.
    ++idAndIndex.second;
    // Add node at 'edge.id' to the worklist. We don't need to consider
    // nodes that are "after" dstId in the containing block; one can't have a
    // path to `dstId` from any of those nodes.
    bool afterDst = dstOp->isBeforeInBlock(getNode(edge.id)->op);
    if (!afterDst && edge.id != idAndIndex.first) {
      worklist.push_back({edge.id, 0});
    }
  }
  return false;
}

// When 'srcId' and 'dstId' are under two parallel for nodes, check whether
// the data structures accessed by 'srcId' and  'dstId' overlap.
// For example:
// ```
// affine.for %arg14 = 0 to 4953 {
//   affine.for %arg15 = 0 to 4952 {
//     %61 = affine.load %alloc[] : memref<f32>
//     affine.store %61, %alloc_8[%arg15] : memref<4954xf32>
//   }
//   affine.for %arg15 = 4952 to 4954 {
//     %61 = affine.load %alloc[] : memref<f32>
//     affine.store %61, %alloc_8[%arg15] : memref<4954xf32>
//   }
// }
// ```
bool MemRefDependenceGraph::hasMemrefAccessDependence(unsigned srcId, unsigned dstId) {
  Operation *srcOp = getNode(srcId)->op;
  Operation *dstOp = getNode(dstId)->op;
  unsigned numCommonLoops = affine::getNumCommonSurroundingLoops(*srcOp, *dstOp);
  affine::AKGMemRefAccess srcAccess(srcOp);
  affine::AKGMemRefAccess dstAccess(dstOp);
  for (unsigned d = 1; d <= numCommonLoops + 1; ++d) {
    affine::FlatAffineValueConstraints dependenceConstraints;
    // todo: Cache dependence analysis results, check cache here.
    affine::DependenceResult result =
      mlir::affine::checkMemrefAccessDependenceAKG(srcAccess, dstAccess, d, &dependenceConstraints, nullptr);
    if (result.value == affine::DependenceResult::HasDependence) {
      return true;
    }
  }
  return false;
}

/// Return all nodes which define SSA values used in node 'id'.
void MemRefDependenceGraph::getDirectlyDependentNodes(unsigned id, DenseSet<unsigned> &dependentNodes) {
  for (Edge edge : inEdges[id]) {
    dependentNodes.insert(edge.id);
  }
}

void MemRefDependenceGraph::createInitNode(DenseMap<Value, SetVector<unsigned>> &memrefAccesses) {
  block->walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<affine::AffineReadOpInterface>(op)) {
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
    } else if (op->getNumRegions() != 0 || isa<affine::AffineYieldOp, func::ReturnOp>(op)) {
      // Return false if another region is found (not currently supported).
      return;
    } else if (isa<CallOpInterface>(op)) {
      // Create graph node for top-level Call Op that takes any argument of
      // memref type. Call Op that returns one or more memref type results
      // is already taken care of, by the previous conditions.
      if (llvm::any_of(op->getOperandTypes(), [&](Type t) { return isa<MemRefType>(t); })) {
        Node node(nextNodeId++, op);
        nodes.insert({node.id, node});
      }
    } else {
      // Create graph node for top-level op, which could have a memory write
      // side effect.
      Node node(nextNodeId++, op);
      nodes.insert({node.id, node});
    }
  });
}

// Initializes the dependence graph based on operations in 'f'.
// Returns true on success, false otherwise.
// Initializes the data dependence graph by walking operations in `block`.
// todo: Add support for taking a Block arg to construct the
// dependence graph at a different depth.
bool MemRefDependenceGraph::init() {
  // Map from a memref to the set of ids of the nodes that have ops accessing
  // the memref.
  DenseMap<Value, SetVector<unsigned>> memrefAccesses;
  createInitNode(memrefAccesses);

  DenseMap<Operation *, unsigned> tempNodes;
  for (auto &idAndNode : nodes) {
    tempNodes.insert({idAndNode.second.op, idAndNode.first});
  }

  // Add dependence edges between nodes which produce SSA values and their
  // users. Load ops can be considered as the ones producing SSA values.
  for (auto &idAndNode : nodes) {
    const Node &node = idAndNode.second;
    Operation *opInst = node.op;
    for (Value value : opInst->getResults()) {
      for (Operation *user : value.getUsers()) {
        // Ignore users outside of the block.
        if (!block->findAncestorOpInBlock(*user) || tempNodes.count(user) == 0) {
          continue;
        }
        addEdge(node.id, tempNodes[user], value);
      }
    }
  }

  // Walk memref access lists and add graph edges between dependent nodes.
  for (auto &memrefAndList : memrefAccesses) {
    unsigned n = memrefAndList.second.size();
    for (unsigned i = 0; i < n; ++i) {
      unsigned srcId = memrefAndList.second[i];
      bool srcHasStore = getNode(srcId)->getStoreOpCount(memrefAndList.first) > 0;
      for (unsigned j = i + 1; j < n; ++j) {
        unsigned dstId = memrefAndList.second[j];
        bool dstHasStore = getNode(dstId)->getStoreOpCount(memrefAndList.first) > 0;
        if ((srcHasStore || dstHasStore) && hasMemrefAccessDependence(srcId, dstId)) {
          addEdge(srcId, dstId, memrefAndList.first);
        }
      }
    }
  }
  return true;
}

}  // namespace akg
}  // namespace mlir
