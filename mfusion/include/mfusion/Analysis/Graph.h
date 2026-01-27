/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MFUSION_INCLUDE_ANALYSIS_GRAPH_H_
#define MFUSION_INCLUDE_ANALYSIS_GRAPH_H_

#include <functional>
#include <memory>
#include <set>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// Graph - Union-Find based cluster graph
//===----------------------------------------------------------------------===//

/// Represents the result of DFS traversal
enum class VisitResult { kFollow, kNoFollow, kExclude };

/// Graph class for managing operation clusters using Union-Find algorithm.
class Graph {
 public:
  using VisitFunc = std::function<VisitResult(size_t)>;

  /// Build a graph from a block's operations.
  /// @param block The block containing operations to build graph from.
  /// @param ops Output vector of operations in topological order.
  /// @param op_idx_map Output map from operation to index.
  /// @param aggressive_cut Whether to use aggressive cutting strategy.
  static std::unique_ptr<Graph> Build(Block *block, std::vector<Operation *> *ops,
                                      llvm::DenseMap<Operation *, size_t> *op_idx_map, bool aggressive_cut);

  Graph(const std::vector<Operation *> &ops, const llvm::DenseMap<Operation *, size_t> &op_idx_map,
        bool aggressive_cut);

  virtual ~Graph() = default;

  /// Find the representative of the cluster containing node_id.
  size_t Find(size_t node_id);

  /// Merge multiple nodes into one cluster.
  void Merge(const std::vector<size_t> &candidates);

  /// Collect all clusters, returning groups of node indices.
  std::vector<std::vector<size_t>> CollectClusters();

  /// Perform DFS traversal starting from node_id.
  void Dfs(size_t node_id, const VisitFunc &visitor);

  /// Get the size of a cluster.
  size_t GetSize(size_t cluster_id);

  /// Get the maximum node id in the cluster with cut strategy consideration.
  size_t GetMaxIdWithCutStrategy(size_t cluster_id);

  /// Get the maximum node id in the cluster.
  size_t GetMaxId(size_t cluster_id);

  /// Get the inputs of a cluster.
  const std::set<size_t> &GetInputs(size_t cluster_id);

  /// Check if the graph has any cycles.
  bool HasCircle();

  /// Get the number of nodes in the graph.
  size_t Size() const { return clusters_.size(); }

  /// Get operations vector.
  const std::vector<Operation *> &GetOperations() const { return ops_; }

 protected:
  /// Internal cluster structure.
  struct Cluster {
    size_t cluster_id_;
    size_t cluster_size_{1};
    size_t max_id_;
    std::set<size_t> inputs_;
    size_t seen_{0};

    Cluster(size_t node_id, Operation *op, const llvm::DenseMap<Operation *, size_t> &op_idx_map);
    void Merge(Cluster *other);
    void Clean() {
      inputs_.clear();
      cluster_size_ = 0;
    }
  };

  /// Binary Indexed Tree for max value queries.
  class BitMax {
   public:
    explicit BitMax(size_t n) : vec_(n + 1) {
      for (size_t i = 0; i < vec_.size(); ++i) {
        vec_[i] = i;
      }
    }

    void SetMax(size_t i, size_t val) {
      i++;
      while (i < vec_.size()) {
        vec_[i] = std::max(vec_[i], val);
        i += LowBit(i);
      }
    }

    size_t FindMax(size_t i) {
      i++;
      if (i >= vec_.size()) {
        i = vec_.size() - 1;
      }
      size_t result = 0;
      while (i > 0) {
        result = std::max(result, vec_[i]);
        i -= LowBit(i);
      }
      return result;
    }

   private:
    static size_t LowBit(size_t x) { return x & (-x); }
    std::vector<size_t> vec_;
  };

  void RefreshInputs(size_t i);
  void DepthFirstSearch(size_t cluster_id, const VisitFunc &visitor);

  std::vector<Operation *> ops_;
  std::vector<Cluster> clusters_;
  std::unique_ptr<BitMax> bitmax_;
  size_t seen_{0};
};

//===----------------------------------------------------------------------===//
// CircleChecker - Detects and removes cycles in cluster candidates
//===----------------------------------------------------------------------===//

/// CircleChecker detects and removes nodes that would form cycles if merged.
class CircleChecker {
 public:
  explicit CircleChecker(Graph *graph) : graph_(graph) {}
  virtual ~CircleChecker() = default;

  /// Remove candidates that would create cycles when merged.
  void RemoveCircle(std::vector<size_t> *candidates);

 protected:
  /// Check if merging basenode with other candidates would create a cycle.
  virtual bool CheckCircle(size_t basenode);

  /// Remove circle nodes and their dependencies from candidates.
  void RemoveCircleNodesFromCandidates();

  Graph *graph_;
  std::set<size_t> candidates_;
  std::vector<size_t> circle_nodes_;
  std::set<size_t> acyclic_nodes_;
};

}  // namespace mlir

#endif  // MFUSION_INCLUDE_ANALYSIS_GRAPH_H_
