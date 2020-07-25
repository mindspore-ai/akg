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
#include "group.h"

#include <tvm/ir_visitor.h>
#include <tvm/operation.h>
#include <isl/constraint.h>

#include <climits>
#include <fstream>
#include <queue>
#include <cmath>

#include "poly/dump_log.h"

namespace akg {
namespace ir {
namespace poly {

isl::schedule GroupStatements::Run(isl::schedule sch_group) {
  int cluster_id = 0;
  pass_info_.has_grouped_ = false;
  auto fn = [&cluster_id, this](isl::schedule_node node) -> isl::schedule_node {
    if (node.isa<isl::schedule_node_sequence>() && node.n_children() > 1 &&
        !node.parent().isa<isl::schedule_node_domain>()) {
      isl::schedule_node_sequence seq_node = node.as<isl::schedule_node_sequence>();
      bool should_group = true;
      isl::union_set_list filter_list(node.ctx(), seq_node.n_children());

      for (unsigned int i = 0; i < seq_node.n_children(); i++) {
        isl::schedule_node child = seq_node.child(i);
        if (!child.isa<isl::schedule_node_filter>() || !child.child(0).isa<isl::schedule_node_leaf>()) {
          should_group = false;
          break;
        } else {
          isl::schedule_node_filter filnode = child.as<isl::schedule_node_filter>();
          filter_list = filter_list.add(filnode.get_filter());
        }
      }
      if (should_group) {
        pass_info_.has_grouped_ = true;
        isl::id gid = isl::id(node.ctx(), std::string("group") + std::to_string(cluster_id));
        pass_info_.group_filter_map_[gid] = filter_list;
        cluster_id++;
        isl_schedule_node *snode = isl_schedule_node_group(node.copy(), gid.release());
        node = isl::manage(snode);
      }
    }
    return node;
  };
  sch_group = sch_group.get_root().map_descendant_bottom_up(fn).get_schedule();
  if (pass_info_.has_grouped_) {
    ComputeDependenceList();
    GroupDependence(sch_group);
  }
  return sch_group;
}

void GroupStatements::GroupDependence(const isl::schedule &schedule) {
  isl::schedule_node rnode = schedule.get_root().child(0);
  isl_union_pw_multi_aff *contract = isl_schedule_node_get_subtree_contraction(rnode.get());
  pass_info_.group_upma_ = isl::manage(contract);
  isl::union_map gmap = isl::union_map::from(pass_info_.group_upma_);
  pass_info_.dependences_ = pass_info_.dependences_.apply_range(gmap).apply_domain(gmap);
}

void GroupStatements::ComputeDependenceList() {
  pass_info_.dependences_.foreach_map([&](const isl::map &m) -> void {
    if (m.domain().get_tuple_id() != m.range().get_tuple_id()) {
      isl::space domain_space_obj = m.domain().get_space();
      isl::local_space domain_space = isl::manage(isl_local_space_from_space(domain_space_obj.copy()));
      int dim = m.dim(isl_dim_in);
      int64_t weight = 1;
      for (int i = 0; i < dim; ++i) {
        isl::aff get_dim_in_domain = isl::aff::var_on_domain(domain_space, isl_dim_out, i);
        int max = static_cast<int>(m.domain().max_val(get_dim_in_domain).get_num_si());
        int min = static_cast<int>(m.domain().min_val(get_dim_in_domain).get_num_si());
        weight *= (max - min + 1);
      }
      Dependency dependency(m.domain().get_tuple_id(), m.range().get_tuple_id(), weight);
      pass_info_.dependency_list_.push_back(dependency);
    }
  });
}

void UnGroupStatements::IsContainsCircle(const std::vector<std::vector<int>> &graph, std::vector<int> &vis, int node,
                                         int size) {
  vis[node] = 1;
  for (int i = 0; i < size; ++i) {
    if (graph[node][i] != 0) {
      if (vis[node] == 1) {
        is_circle_ = true;
        break;
      } else if (vis[node] == -1) {
        continue;
      } else {
        IsContainsCircle(graph, vis, i, size);
      }
    }
  }
  vis[node] = -1;
}

void UnGroupStatements::DfsTopsort(std::vector<std::vector<int>> &graph, std::vector<int> &indegree,
                                   std::set<int> &zeros, int cnt, int size, int64_t current_value,
                                   int64_t current_max) {
  cnt_dfs_times_++;
  // constraint 1:  return when dfs reaches a limit times.
  if (cnt_dfs_times_ > DFS_TIMES_MAX) return;
  // constraint 2: return when current max is bigger than best result.
  if ((min_topsort_ != -1) && (current_max >= min_topsort_)) return;

  if (cnt == size) {
    min_topsort_ = current_max;
    std::vector<int> res(temp_res_);
    topsort_res_ = res;
  } else {
    for (auto it = zeros.begin(); it != zeros.end(); ++it) {
      std::set<int> zeros_copy(zeros);
      zeros_copy.erase(*it);
      temp_res_[cnt] = *it;
      std::vector<int> temp;

      for (int j = 0; j < size; ++j) {
        if (graph[*it][j] == 1) {
          graph[*it][j] = 0;
          indegree[j]--;
          if (indegree[j] == 0) {
            zeros_copy.insert(j);
          }
          temp.emplace_back(j);
        }
      }
      int64_t updated_value = current_value;
      if (cost_map_.find(temp_res_[cnt]) != cost_map_.end()) {
        updated_value += cost_map_.find(temp_res_[cnt])->second;
      }
      DfsTopsort(graph, indegree, zeros_copy, cnt + 1, size, updated_value, std::max(updated_value, current_max));
      for (int &itj : temp) {
        graph[*it][itj] = 1;
        indegree[itj]++;
      }
    }
  }
}

isl::union_set_list UnGroupStatements::DependenciesTopsort(const isl::union_set_list &filterlist) {
  if (pass_info_.dependency_list_.empty()) return filterlist;
  if (filterlist.size() == 0) return filterlist;

  // 1. build graph from dependency_list_ and filterlist
  int graph_size = filterlist.size();
  std::unordered_map<isl::id, int, isl::IslIdIslHash> filter_map;
  for (int i = 0; i < graph_size; ++i) {
    isl::union_set temp = filterlist.get_at(i);
    CHECK(temp.n_set() == 1u) << "number of union_set's children in filterlist should be 1";
    filter_map.insert(std::pair<isl::id, int>(temp.get_set_list().get_at(0).get_tuple_id(), i));
  }

  std::vector<std::vector<int>> graph(graph_size, std::vector<int>(graph_size, 0));
  std::vector<int> indegree(graph_size, 0);
  for (auto &i : pass_info_.dependency_list_) {
    isl::id from = i.GetStartNode();
    isl::id to = i.GetEndNode();
    if (filter_map.find(from) != filter_map.end() && filter_map.find(to) != filter_map.end()) {
      int x = filter_map.find(from)->second;
      int y = filter_map.find(to)->second;
      // we only use similar dependency once
      if (graph[x][y] == 0) {
        graph[x][y] = 1;
        indegree[y]++;
      }
      int64_t value;
      if (cost_map_.find(x) == cost_map_.end()) {
        value = i.GetEdgeWeight();
      } else {
        value = cost_map_.find(x)->second + i.GetEdgeWeight();
      }
      cost_map_.insert(std::pair<int, int64_t>(x, value));

      if (cost_map_.find(y) == cost_map_.end()) {
        value = -i.GetEdgeWeight();
      } else {
        value = cost_map_.find(y)->second - i.GetEdgeWeight();
      }
      cost_map_.insert(std::pair<int, int64_t>(y, value));
    }
  }
  // 2. judge if graph has a circle by using dfs
  std::vector<int> vis(graph_size, 0);
  is_circle_ = false;
  for (int i = 0; i < graph_size; ++i) {
    if (vis[i] == -1) {
      continue;
    }
    IsContainsCircle(graph, vis, i, graph_size);
    if (is_circle_) return filterlist;
  }
  // 3. compute all the Topsort list
  if (temp_res_.empty()) {
    temp_res_.insert(temp_res_.begin(), graph_size, 0);
  } else {
    temp_res_.assign(graph_size, 0);
  }
  std::set<int> zeros;
  for (int i = 0; i < graph_size; ++i) {
    if (indegree[i] == 0) {
      zeros.insert(i);
    }
  }
  // minTopsort == -1 means never found a result of dfs.
  min_topsort_ = -1;
  cnt_dfs_times_ = 0;
  DfsTopsort(graph, indegree, zeros, 0, graph_size, 0, 0);

  // 4. output the smallest filterlist
  isl::union_set_list reslist = isl::union_set_list(filterlist.ctx(), graph_size);
  for (int i = 0; i < graph_size; ++i) {
    isl::union_set temp = filterlist.get_at(topsort_res_[i]);
    reslist = reslist.add(temp);
  }
  return reslist;
}

isl::schedule_node UnGroupStatements::InsertMarknode(isl::schedule_node node, const isl::id &gid) {
  if (node.isa<isl::schedule_node_leaf>()) {
    return node.insert_mark(gid);
  } else {
    if (node.n_children() == 1) {
      node = InsertMarknode(node.child(0), gid);
      node = node.parent();
    }
    return node;
  }
}

isl::schedule UnGroupStatements::Run(isl::schedule schedule) {
  if (!pass_info_.has_grouped_) {
    return schedule;
  }
  bool find_filter = false;
  auto findAndMarkGroupFilter = [this, &find_filter](isl::schedule_node node) -> isl::schedule_node {
    if (node.isa<isl::schedule_node_filter>() && node.as<isl::schedule_node_filter>().n_children() == 1) {
      find_filter = true;
      auto filter_node = node.as<isl::schedule_node_filter>().first_child();
      isl::map_list schmap = filter_node.get_prefix_schedule_union_map().get_map_list();
      if (schmap.size() == 1) {
        isl::id gid = schmap.get_at(0).domain().get_tuple_id();
        if (pass_info_.group_filter_map_.find(gid) != pass_info_.group_filter_map_.end()) {
          node = InsertMarknode(node, gid);
        }
      }
    }
    if ((node.isa<isl::schedule_node_domain>()) && (!find_filter)) {
      find_filter = true;
      isl::union_set domain = node.as<isl::schedule_node_domain>().domain();
      isl::set_list setlist = domain.get_set_list();
      isl::id groupid;
      if (setlist.size() == 1) {
        groupid = setlist.get_at(0).get_tuple_id();
      }
      if (pass_info_.group_filter_map_.find(groupid) != pass_info_.group_filter_map_.end()) {
        while (node.has_children()) {
          if (node.n_children() > 1) {
            return node.root();
          } else {
            node = node.first_child();
          }
        }
        node = InsertMarknode(node, groupid);
        node = node.root();
      }
    }
    return node;
  };
  schedule = schedule.get_root().map_descendant_bottom_up(findAndMarkGroupFilter).get_schedule();

  schedule = schedule.pullback(pass_info_.group_upma_);

  auto ReplaceUngroupedFilterWithSequence = [this](isl::schedule_node node) -> isl::schedule_node {
    if (node.isa<isl::schedule_node_mark>()) {
      isl::schedule_node_mark marknode = node.as<isl::schedule_node_mark>();
      isl::id markid = marknode.get_id();
      isl::union_set_list filterlist = pass_info_.group_filter_map_[markid];
      isl::union_set_list resfilterlist = DependenciesTopsort(filterlist);
      if (pass_info_.group_filter_map_.find(markid) != pass_info_.group_filter_map_.end()) {
        node = node.del();
        node = node.insert_sequence(resfilterlist);
      }
    }
    return node;
  };
  schedule = schedule.get_root().map_descendant_bottom_up(ReplaceUngroupedFilterWithSequence).get_schedule();
  pass_info_.dependences_ = pass_info_.orig_dependences_;
  pass_info_.constraints_ = MakeScheduleConstraints(schedule, pass_info_);
  return schedule;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
