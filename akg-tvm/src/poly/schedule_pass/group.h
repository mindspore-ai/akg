/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#ifndef POLY_GROUP_H_
#define POLY_GROUP_H_

#include "poly/schedule_pass.h"

namespace akg {
namespace ir {
namespace poly {

/*
 * This class combines multiple statements into a group to accelerate the calculation process of Pluto algorithm
 */
class GroupStatements : public SchedulePass {
 public:
  GroupStatements(PassInfo &pass_info) : pass_info_(pass_info) { pass_name_ = __FUNCTION__; }
  ~GroupStatements() {}

  virtual isl::schedule Run(isl::schedule sch);

  void GroupDependence(const isl::schedule &schedule);

  void ComputeDependenceList();

 private:
  PassInfo &pass_info_;
};

/*
 * After compute schedule, this class will restore the group statements to the original statement sequence
 */
class UnGroupStatements : public SchedulePass {
 public:
  UnGroupStatements(PassInfo &pass_info) : pass_info_(pass_info) { pass_name_ = __FUNCTION__; }
  ~UnGroupStatements() {}

  virtual isl::schedule Run(isl::schedule sch);

 private:
  void IsContainsCircle(const std::vector<std::vector<int>> &graph, std::vector<int> &vis, int node, int size);

  void DfsTopsort(std::vector<std::vector<int>> &graph, std::vector<int> &indegree, std::set<int> &zeros, int cnt,
                  int size, int64_t current_value, int64_t current_max);
  isl::union_set_list DependenciesTopsort(const isl::union_set_list &filterlist);
  void BuildGraph(const isl::union_set_list &filterlist, std::vector<std::vector<int>> &graph,
                  std::vector<int> &indegree);

  isl::schedule_node InsertMarknode(isl::schedule_node node, const isl::id &gid);

  PassInfo &pass_info_;
  bool is_circle_ = false;

  // the maximum times of dfs Topsort
  const int DFS_TIMES_MAX = 1000000;

  // counter times of dfs Topsort for limiting a long-time dfs process
  int cnt_dfs_times_ = 0;

  // the min total cost for dfs Topsort
  int64_t min_topsort_ = -1;

  std::map<int, int64_t> cost_map_;

  std::vector<int> topsort_res_;

  std::vector<int> temp_res_;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_GROUP_H_
