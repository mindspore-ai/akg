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

#ifndef COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_DEPENDENCEANALYSIS_H_
#define COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_DEPENDENCEANALYSIS_H_

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace akg {

// Node represents a node in the graph. A Node is either an entire loop nest
// rooted at the top level which contains loads/stores, or a top level
// load/store.
struct Node {
  // The unique identifier of this node in the graph.
  unsigned id;
  // The top-level statement which is (or contains) a load/store.
  Operation *op;
  // List of load operations.
  SmallVector<Operation *, 4> loads;
  // List of store op insts.
  SmallVector<Operation *, 4> stores;
  Node(unsigned id, Operation *op) : id(id), op(op) {}

  unsigned getLoadOpCount(Value memref) const;
  unsigned getStoreOpCount(Value memref) const;
};

// Edge represents a data dependence between nodes in the graph.
struct Edge {
  // The id of the node at the other end of the edge.
  // If this edge is stored in Edge = Node.inEdges[i], then
  // 'Node.inEdges[i].id' is the identifier of the source node of the edge.
  // If this edge is stored in Edge = Node.outEdges[i], then
  // 'Node.outEdges[i].id' is the identifier of the dest node of the edge.
  unsigned id;
  // The SSA value on which this edge represents a dependence.
  // If the value is a memref, then the dependence is between graph nodes
  // which contain accesses to the same memref 'value'. If the value is a
  // non-memref value, then the dependence is between a graph node which
  // defines an SSA value and another graph node which uses the SSA value
  // (e.g. a constant or load operation defining a value which is used inside
  // a loop nest).
  Value value;
};

// MemRefDependenceGraph is a graph data structure where graph nodes are
// top-level operations in a `Block` which contain load/store ops, and edges
// are memref dependences between the nodes.
// TODO: Add a more flexible dependence graph representation.
// TODO: Add a depth parameter to dependence graph construction.
struct MemRefDependenceGraph {
 public:
  // Map from node id to Node.
  DenseMap<unsigned, Node> nodes;
  // Map from node id to list of input edges.
  DenseMap<unsigned, SmallVector<Edge, 2>> inEdges;
  // Map from node id to list of output edges.
  DenseMap<unsigned, SmallVector<Edge, 2>> outEdges;
  // Map from memref to a count on the dependence edges associated with that
  // memref.
  DenseMap<Value, unsigned> memrefEdgeCount;
  // The next unique identifier to use for newly created graph nodes.
  unsigned nextNodeId = 0;
  // The block for which this graph is created to perform fusion.
  Block *block;

  explicit MemRefDependenceGraph(Block *block) : block(block) {}

  void createInitNode(DenseMap<Value, SetVector<unsigned>> &memrefAccesses);
  bool createEdges(const DenseMap<Value, SetVector<unsigned>> &memrefAccesses);
  bool init();
  // Returns the graph node for 'id'.
  Node *getNode(unsigned id);
  int getNodeId(const Operation *op);
  void updateNodeOp(const Operation *oldOp, Operation *newOp);

  bool hasEdge(unsigned srcId, unsigned dstId, Value value);
  // Adds an edge from node 'srcId' to node 'dstId' for 'value'.
  void addEdge(unsigned srcId, unsigned dstId, Value value);
  bool hasDependencePath(unsigned srcId, unsigned dstId);
  bool hasMemrefAccessDependence(unsigned srcId, unsigned dstId);

  void getPredecessorNodes(unsigned id, DenseSet<unsigned> &dependentNodes);
  void getPredecessorNodes(unsigned id, std::vector<unsigned> &dependentNodes);
  void getSuccessorNodes(unsigned id, DenseSet<unsigned> &dependentNodes);
  void getSuccessorNodes(unsigned id, std::vector<unsigned> &dependentNodes);

  void print(raw_ostream &os) const;
  void dump() const { print(llvm::errs()); }
};

}  // namespace akg
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_DEPENDENCEANALYSIS_H_
