/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "reset_coincidence_of_reduce.h"

namespace akg {
namespace ir {
namespace poly {
bool ResetCoincidenceOfReduce::IsStmtScheduleContainsReduceAxis(
  const isl::pw_aff &stmt, const std::unordered_set<std::string> &reduce_axis_list) {
  int num_dims = stmt.domain().n_dim();
  isl_space *domain_space = stmt.domain().get_space().get();
  for (int dim = 0; dim < num_dims; ++dim) {
    const char *axis_name = isl_space_get_dim_name(domain_space, isl_dim_out, dim);
    if (axis_name == nullptr) continue;
    if (reduce_axis_list.count(axis_name) == 0) continue;
    if (isl_pw_aff_involves_dims(stmt.get(), isl_dim_in, dim, 1)) return true;
  }
  return false;
}

bool ResetCoincidenceOfReduce::IsDimScheduleContainsReduceAxis(const isl::union_pw_aff &schedule) {
  auto reduce_stmts = scop_info_.analysis_result_.GetReduceTensorInfoMap();
  bool found_reduce_axis = false;
  auto stmt_list = schedule.get_pw_aff_list();
  stmt_list.foreach([&found_reduce_axis, &reduce_stmts, this](const isl::pw_aff &stmt) -> void {
    isl::id stmt_id = stmt.domain().get_tuple_id();
    if (reduce_stmts.count(stmt_id)) {
      std::unordered_set<std::string> reduce_axis_list;
      for (const auto &axis : reduce_stmts.at(stmt_id).axis_vec) {
        reduce_axis_list.insert(axis);
      }
      if (IsStmtScheduleContainsReduceAxis(stmt, reduce_axis_list)) {
        found_reduce_axis = true;
      }
    }
  });
  return found_reduce_axis;
}

isl::schedule ResetCoincidenceOfReduce::Run(isl::schedule curr_schedule) {
  const auto &new_schedule = curr_schedule;
  auto fn = [this](isl::schedule_node node) -> isl::schedule_node {
    if (auto band = node.as<isl::schedule_node_band>()) {
      int num_dims = static_cast<int>(band.n_member());
      for (int dim = 0; dim < num_dims; ++dim) {
        bool is_coincident = band.member_get_coincident(dim);
        if (!is_coincident) continue;
        auto dim_schedule = band.get_partial_schedule().get_union_pw_aff(dim);
        if (IsDimScheduleContainsReduceAxis(dim_schedule)) {
          LOG(INFO) << "reset coincidence of reduce axis on dim " << dim << " in partial schedule: " << dim_schedule;
          node = band.member_set_coincident(dim, false);
          band = node.as<isl::schedule_node_band>();
        }
      }
    }
    return node;
  };
  return new_schedule.get_root().map_descendant_bottom_up(fn).get_schedule();
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
