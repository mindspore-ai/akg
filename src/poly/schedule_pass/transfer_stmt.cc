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
#include "transfer_stmt.h"

namespace akg {
namespace ir {
namespace poly {


isl::schedule TransferStmt::Run(isl::schedule curr_schedule) {
  if (scop_info_.analysis_result_.GetTransferStmt().is_empty()) {
    return curr_schedule;
  }
  pass_info_.transfer_stmt_ = scop_info_.analysis_result_.GetTransferStmt();
  isl::schedule_node root_ = curr_schedule.get_root();
  isl::schedule_node node = GetOuterBand(root_);
  if (node.isa<isl::schedule_node_sequence>() || node.isa<isl::schedule_node_set>()) {
    int n = static_cast<int>(node.n_children());
    for (int i = 0; i < n; ++i) {
      isl::schedule_node child = node.child(i);
      CHECK(child.isa<isl::schedule_node_filter>()) << "The child of set or sequence must filter!";
      isl::schedule_node_filter filter_node = child.as<isl::schedule_node_filter>();
      isl::union_set filter = filter_node.get_filter();
      if (!filter.intersect(pass_info_.transfer_stmt_).is_empty()) {
        filter = filter.subtract(pass_info_.transfer_stmt_);
        child = isl::manage(isl_schedule_node_filter_set_filter(child.copy(), filter.copy()));
        node = child.parent();
        return node.get_schedule();
      }
    }
  }
  return curr_schedule;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
