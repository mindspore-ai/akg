/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "change_marknode_position.h"

namespace akg {
namespace ir {
namespace poly {

isl::schedule ChangeMarkNodePosition::Run(isl::schedule curr_schedule) {
  std::unordered_set<std::string> ids = with_stmts_ids_;
  if (ids.empty()) {
    return curr_schedule;
  }

  auto fn = [&ids](isl::schedule_node node) -> isl::schedule_node {
    if (!node.isa<isl::schedule_node_mark>()) {
      return node;
    }
    std::string mark_id = node.as<isl::schedule_node_mark>().get_id().get_name();
    if (mark_id != "realize_UB" || !node.child(0).isa<isl::schedule_node_band>()) {
      return node;
    }
    if (!node.child(0).child(0).isa<isl::schedule_node_sequence>()) {
      return node;
    }
    node = node.get_child(0).get_child(0);  // sequence
    bool delete_outer_mark = true;
    int n = node.n_children();
    for (int i = 0; i < n; i++) {
      isl::schedule_node_filter filter_node = node.child(i).as<isl::schedule_node_filter>();
      bool is_not_with_stmt = filter_node.get_filter().every_set(
        [&ids](const isl::set &s) -> bool { return (ids.count(s.get_tuple_name()) == 0); });
      if (is_not_with_stmt) {
        delete_outer_mark = false;
      } else {
        node = node.child(i).child(0);
        node = node.insert_mark(isl::id(node.ctx(), mark_id));
        node = node.parent().parent();
      }
    }
    node = node.parent().parent();
    if (delete_outer_mark) {
      node = node.del();
    }
    return node;
  };

  return curr_schedule.get_root().map_descendant_bottom_up(fn).get_schedule();
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
