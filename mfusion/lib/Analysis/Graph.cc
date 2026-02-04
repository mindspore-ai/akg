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

#include "mfusion/Analysis/Graph.h"

#include <algorithm>
#include <queue>

#include "llvm/Support/Debug.h"
#include "mlir/IR/OpDefinition.h"

#define DEBUG_TYPE "analysis-graph"

namespace mlir {

//===----------------------------------------------------------------------===//
// Graph::Cluster Implementation
//===----------------------------------------------------------------------===//

Graph::Cluster::Cluster(size_t node_id, Operation *op, const llvm::DenseMap<Operation *, size_t> &op_idx_map)
    : cluster_id_(node_id), max_id_(node_id) {
  // Collect inputs from the operation's operands
  for (auto operand : op->getOperands()) {
    if (auto *def_op = operand.getDefiningOp()) {
      auto it = op_idx_map.find(def_op);
      if (it != op_idx_map.end()) {
        inputs_.insert(it->second);
      }
    }
  }
}

void Graph::Cluster::Merge(Cluster *other) {
  other->cluster_id_ = cluster_id_;
  max_id_ = std::max(max_id_, other->max_id_);
  cluster_size_ += other->cluster_size_;
  inputs_.insert(other->inputs_.cbegin(), other->inputs_.cend());
  other->Clean();
}

//===----------------------------------------------------------------------===//
// Graph Implementation
//===----------------------------------------------------------------------===//

std::unique_ptr<Graph> Graph::Build(Block *block, std::vector<Operation *> *ops,
                                    llvm::DenseMap<Operation *, size_t> *op_idx_map, bool aggressive_cut) {
  std::vector<Operation *> local_ops;
  llvm::DenseMap<Operation *, size_t> local_map;

  // Collect operations in order (already in topological order in MLIR block)
  for (Operation &op : block->getOperations()) {
    // Skip terminator and non-Muse operations for clustering
    if (op.hasTrait<OpTrait::IsTerminator>()) {
      continue;
    }
    local_map[&op] = local_ops.size();
    local_ops.push_back(&op);
  }

  auto graph = std::unique_ptr<Graph>(new Graph(local_ops, local_map, aggressive_cut));

  if (ops != nullptr) {
    *ops = std::move(local_ops);
  }
  if (op_idx_map != nullptr) {
    *op_idx_map = std::move(local_map);
  }

  return graph;
}

Graph::Graph(const std::vector<Operation *> &ops, const llvm::DenseMap<Operation *, size_t> &op_idx_map,
             bool aggressive_cut)
    : ops_(ops) {
  clusters_.reserve(ops.size());
  for (size_t i = 0; i < ops.size(); ++i) {
    clusters_.emplace_back(i, ops[i], op_idx_map);
  }
  if (!aggressive_cut) {
    bitmax_ = std::make_unique<BitMax>(ops.size());
  }
}

size_t Graph::Find(size_t node_id) {
  size_t &pre_id = clusters_[node_id].cluster_id_;
  return (pre_id == clusters_[pre_id].cluster_id_) ? pre_id : (pre_id = Find(pre_id));
}

void Graph::Merge(const std::vector<size_t> &candidates) {
  size_t min_id = *std::min_element(candidates.begin(), candidates.end());
  for (auto id : candidates) {
    if (id == min_id) {
      continue;
    }
    clusters_[min_id].Merge(&clusters_[id]);
  }
  if (bitmax_ != nullptr) {
    bitmax_->SetMax(min_id, clusters_[min_id].max_id_);
  }
}

std::vector<std::vector<size_t>> Graph::CollectClusters() {
  std::vector<std::vector<size_t>> cluster_map(clusters_.size());
  for (size_t i = 0; i < clusters_.size(); ++i) {
    cluster_map[Find(i)].push_back(i);
  }
  return cluster_map;
}

void Graph::Dfs(size_t node_id, const VisitFunc &visitor) {
  ++seen_;
  DepthFirstSearch(Find(node_id), visitor);
}

size_t Graph::GetSize(size_t cluster_id) { return clusters_[Find(cluster_id)].cluster_size_; }

size_t Graph::GetMaxIdWithCutStrategy(size_t cluster_id) {
  return (bitmax_ == nullptr) ? GetMaxId(cluster_id) : bitmax_->FindMax(Find(cluster_id));
}

size_t Graph::GetMaxId(size_t cluster_id) { return clusters_[Find(cluster_id)].max_id_; }

const std::set<size_t> &Graph::GetInputs(size_t cluster_id) {
  cluster_id = Find(cluster_id);
  RefreshInputs(cluster_id);
  return clusters_[cluster_id].inputs_;
}

void Graph::RefreshInputs(size_t i) {
  auto &inputs = clusters_[i].inputs_;
  std::set<size_t> new_inputs;
  for (auto it = inputs.cbegin(); it != inputs.cend(); ++it) {
    size_t new_id = Find(*it);
    if (new_id != i) {
      new_inputs.insert(new_id);
    }
  }
  inputs = std::move(new_inputs);
}

void Graph::DepthFirstSearch(size_t cluster_id, const VisitFunc &visitor) {
  if (clusters_[cluster_id].seen_ >= seen_) {
    return;
  }
  clusters_[cluster_id].seen_ = seen_;

  auto result = visitor(cluster_id);
  if (result != VisitResult::kFollow) {
    return;
  }

  // Traverse inputs in descending order
  const auto &inputs = GetInputs(cluster_id);
  for (auto it = inputs.crbegin(); it != inputs.crend(); ++it) {
    DepthFirstSearch(*it, visitor);
  }
}

bool Graph::HasCircle() {
  std::vector<size_t> valid_clusters;
  for (size_t i = 0; i < clusters_.size(); ++i) {
    if (clusters_[i].cluster_id_ == i) {
      valid_clusters.push_back(i);
    }
  }

  std::vector<int> out_degree(clusters_.size(), 0);
  std::queue<size_t> que;

  for (auto &cluster_id : valid_clusters) {
    for (size_t i : GetInputs(cluster_id)) {
      out_degree[i]++;
    }
  }

  for (auto &cluster_id : valid_clusters) {
    if (out_degree[cluster_id] == 0) {
      que.push(cluster_id);
    }
  }

  size_t count = 0;
  while (!que.empty()) {
    size_t u = que.front();
    que.pop();
    count++;
    for (size_t i : GetInputs(u)) {
      if (--out_degree[i] == 0) {
        que.push(i);
      }
    }
  }

  return count != valid_clusters.size();
}

//===----------------------------------------------------------------------===//
// CircleChecker Implementation
//===----------------------------------------------------------------------===//

void CircleChecker::RemoveCircle(std::vector<size_t> *candidates) {
  if (candidates->size() <= 1) {
    return;
  }

  candidates_.clear();
  candidates_.insert(candidates->cbegin(), candidates->cend());

  for (auto iter = candidates->cbegin(); iter != candidates->cend(); ++iter) {
    if (candidates_.count(*iter) == 0) {
      continue;
    }
    circle_nodes_.clear();
    if (CheckCircle(*iter)) {
      RemoveCircleNodesFromCandidates();
    }
  }

  candidates->erase(std::remove_if(candidates->begin(), candidates->end(),
                                 [this](size_t c) { return this->candidates_.count(c) == 0; }),
                  candidates->end());
}

bool CircleChecker::CheckCircle(size_t basenode) {
  const auto &inputs = graph_->GetInputs(basenode);
  std::set<size_t> visited_circle_nodes;

  for (auto x : inputs) {
    if (candidates_.count(x) > 0) {
      continue;
    }

    bool has_circle = false;
    std::set<size_t> done;
    auto candidate_min = *candidates_.begin();

    auto vis_func = [this, &has_circle, &done, &visited_circle_nodes, &candidate_min](size_t cluster_id) {
      // Cut strategy: if max id is less than candidate min, no need to continue
      if (graph_->GetMaxIdWithCutStrategy(cluster_id) < candidate_min) {
        return VisitResult::kExclude;
      }
      if (done.count(cluster_id) > 0 || acyclic_nodes_.count(cluster_id) > 0 ||
          visited_circle_nodes.count(cluster_id) > 0) {
        return VisitResult::kExclude;
      }
      done.insert(cluster_id);
      if (candidates_.count(cluster_id) > 0) {
        has_circle = true;
        circle_nodes_.push_back(cluster_id);
        return VisitResult::kExclude;
      }
      return VisitResult::kFollow;
    };

    graph_->Dfs(x, vis_func);

    if (has_circle) {
      visited_circle_nodes.insert(done.cbegin(), done.cend());
    } else {
      acyclic_nodes_.insert(done.cbegin(), done.cend());
    }
  }

  return !circle_nodes_.empty();
}

void CircleChecker::RemoveCircleNodesFromCandidates() {
  auto remove_from_candidates = [this](size_t node_id) {
    if (candidates_.count(node_id) > 0) {
      candidates_.erase(node_id);
      return VisitResult::kFollow;
    }
    return VisitResult::kExclude;
  };

  for (auto node : circle_nodes_) {
    graph_->Dfs(node, remove_from_candidates);
  }
}

}  // namespace mlir
