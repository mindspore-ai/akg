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

#include "reorder_invariant_set_schedule.h"

namespace akg {
namespace ir {
namespace poly {

isl::schedule ReorderInvariantSetSchedule::Run(isl::schedule sch) {
  if (!pass_info_.has_invariant_dependence_) {
    return sch;
  }
  isl::schedule_node root = sch.get_root();
  isl::schedule_node outer_band = GetOuterBand(root);
  if (outer_band.isa<isl::schedule_node_set>()) {
    std::vector<size_t> new_filters;
    std::vector<size_t> invariant_filters;
    std::vector<size_t> rest_filters;
    for (unsigned int i = 0; i < outer_band.n_children(); ++i) {
      isl::schedule_node node = outer_band.get_child(i);
      auto filter = node.as<isl::schedule_node_filter>();
      isl::union_set sets = filter.get_filter();
      unsigned int invariant_count = 0;
      sets.foreach_set([&invariant_count, this](const isl::set &s) -> void {
        if (s.n_dim() == 0 && this->pass_info_.invariant_state_.count(s.get_tuple_name()) > 0) {
          invariant_count++;
        }
      });

      if (invariant_count == sets.n_set()) {
        invariant_filters.push_back(i);
      } else {
        rest_filters.push_back(i);
      }
    }

    for (unsigned long &invariant_filter : invariant_filters) {
      new_filters.push_back(invariant_filter);
    }

    for (unsigned long &rest_filter : rest_filters) {
      new_filters.push_back(rest_filter);
    }

    std::unordered_map<size_t, size_t> old_to_new_map;
    for (size_t i = 0; i < new_filters.size(); ++i) {
      old_to_new_map.emplace(new_filters[i], i);
    }

    outer_band = ReorderFilters(outer_band, old_to_new_map);
  }
  return outer_band.get_schedule();
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
