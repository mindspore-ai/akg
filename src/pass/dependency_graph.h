/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef PASS_DEPENDENCY_GRAPH_H_
#define PASS_DEPENDENCY_GRAPH_H_

#include <vector>
#include <utility>
#include <set>
namespace akg {
namespace ir {
template <typename T>
class DependencyGraph {
 public:
  enum class DepType {
    kRAW,
    kWAR,
    kWAW,
    kNone,
    kDep,  // general dependency, don't distinguish type
  };

  explicit DependencyGraph(std::vector<T> &nodes, bool check_redundant_arcs = false)
      : nodes_(nodes),
        done_(nodes.size(), false),
        reaching_(nodes.size()),
        edges_(nodes.size()),
        versionID_(nodes.size(), 0),
        check_redundancy_(check_redundant_arcs) {}
  virtual ~DependencyGraph() {}

  virtual DepType GetDepType(const T *a, const T *b) = 0;
  virtual void AddDepRelation(T *a, T *b, DepType type) = 0;
  virtual bool IsBranchAway(const T *a, const T *b) = 0;

  void BuildGraph() {
    currentID_ = 0;

    // forward
    is_forward_ = true;
    for (uint32_t idx = 0; idx < nodes_.size(); idx++) {
      if (done_[idx]) {
        continue;
      }
      BuildGraphPass_(idx);
    }

    // clear states
    if (check_redundancy_) {
      CheckRedundantArcs();
    }
    std::fill(done_.begin(), done_.end(), false);
    for (uint32_t i = 0; i < nodes_.size(); ++i) {
      reaching_[i].clear();
      edges_[i].clear();
    }

    // backward
    is_forward_ = false;
    for (int32_t idx = static_cast<int32_t>(nodes_.size()) - 1; idx >= 0; idx--) {
      if (done_[idx]) {
        continue;
      }
      BuildGraphPass_(idx);
    }

    if (check_redundancy_) {
      CheckRedundantArcs();
    }
  }

  void CheckRedundantArcs() {
    // do DFS from each vertex.
    // a) For each vertex i ∈ G, start the DFS from each vertex j such that j is the direct descendant of i
    // b) For each vertex k reachable by the DFS from i, check whether the redundant edge (i, k) exists
    std::set<std::pair<int, int>> error_pairs;

    for (uint32_t i = 0; i < nodes_.size(); ++i) {
      for (auto j : edges_[i]) {
        currentID_++;
        DFSCheck_(static_cast<int>(i), j, error_pairs);
      }
    }

    if (error_pairs.size() != 0) {
      std::cerr << "Find redundant arcs" << std::endl;
      for (auto x : error_pairs) {
        std::cerr << x.first << " " << x.second << std::endl;
      }
    } else {
      std::cerr << "No redundant arcs" << std::endl;
    }
  }

 private:
  void BuildGraphPass_(uint32_t idx) {
    int begin, end, step;

    if (is_forward_) {
      begin = static_cast<int>(idx) + 1;
      end = static_cast<int>(nodes_.size());
      step = 1;
    } else {
      begin = static_cast<int>(idx) - 1;
      end = -1;
      step = -1;
    }

    for (int jdx = begin; jdx != end; jdx += step) {
      if (!IsReaching_(static_cast<int>(idx), jdx) && !IsBranchAway(&nodes_[idx], &nodes_[jdx])) {
        DepType type = GetDepType(&nodes_[idx], &nodes_[jdx]);
        if (type != DepType::kNone) {
          currentID_++;
          SetReaching_(static_cast<int>(idx), jdx);
          AddDepRelation(&nodes_[idx], &nodes_[jdx], type);
          edges_[idx].insert(jdx);
          if (!done_[jdx]) {
            BuildGraphPass_(jdx);
          }
        }
      }
    }
    done_[idx] = true;
  }

  void SetReaching_(int idx, int jdx) {
    if (versionID_[jdx] == currentID_) {
      return;
    }
    versionID_[jdx] = currentID_;

    reaching_[jdx].insert(reaching_[idx].begin(), reaching_[idx].end());
    reaching_[jdx].insert(idx);

    for (auto k : edges_[jdx]) {
      SetReaching_(idx, k);
    }
  }

  bool IsReaching_(int idx, int jdx) { return reaching_[jdx].count(idx) != 0; }

  void DFSCheck_(int i, int j, std::set<std::pair<int, int>> &error_pairs) {
    if (versionID_[j] == currentID_) {
      return;
    }
    versionID_[j] = currentID_;

    for (auto k : edges_[j]) {
      if (edges_[i].count(k) != 0) {
        error_pairs.insert(std::make_pair(i, j));
      }
      DFSCheck_(i, k, error_pairs);
    }
  }

  std::vector<T> &nodes_;
  bool is_forward_{false};
  uint32_t currentID_{0};

  // additional attributes to nodes
  std::vector<bool> done_;
  std::vector<std::set<int>> reaching_;
  std::vector<std::set<int>> edges_;
  std::vector<uint32_t> versionID_;

  // correctness checking
  bool check_redundancy_;
};
}  // namespace ir
}  // namespace akg

#endif  // PASS_DEPENDENCY_GRAPH_H_
