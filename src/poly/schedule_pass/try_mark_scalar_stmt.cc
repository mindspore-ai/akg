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

#include "try_mark_scalar_stmt.h"
#include "poly/schedule_pass.h"
#include "poly/schedule_tree_util.h"

namespace akg {
namespace ir {
namespace poly {
bool TryMarkScalarStmt::SubtreeHasPermutableBands(const isl::schedule_node &node) const {
  bool all_non_permutable = false;
  auto IsPermutable = [](const isl::schedule_node &node, bool check_coincident) -> bool {
    if (!node) return false;
    if (!node.isa<isl::schedule_node_band>()) return false;
    if (!node.as<isl::schedule_node_band>().get_permutable()) return false;
    if (node.as<isl::schedule_node_band>().n_member() < 1) return false;
    return !(check_coincident && !node.as<isl::schedule_node_band>().member_get_coincident(0));
  };
  all_non_permutable = node.every_descendant([this, &IsPermutable](const isl::schedule_node &node) -> bool {
    return !(IsPermutable(node, pass_info_.tile_check_coincident_));
  });
  return !all_non_permutable;
}

isl::schedule TryMarkScalarStmt::Run(isl::schedule curr_schedule) {
  const auto &curr_node = curr_schedule.get_root();
  // Return "root" if given an inappropriate node
  if (!curr_node.isa<isl::schedule_node_domain>() && !curr_node.isa<isl::schedule_node_filter>()) return curr_schedule;
  // Check whether each stmt is scalar
  auto domain = curr_node.isa<isl::schedule_node_domain>() ? curr_node.as<isl::schedule_node_domain>().get_domain()
                                                           : curr_node.as<isl::schedule_node_filter>().get_filter();
  if (!domain.every_set([](const isl::set &set) {
        auto dim = set.n_dim();
        return dim == 0;
      }))
    return curr_schedule;

  // Return if there exist any band nodes
  if (SubtreeHasPermutableBands(curr_node)) return curr_schedule;

  auto node = GetOuterBand(curr_node);
  // Mark to copy to UB
  if (node.isa<isl::schedule_node_leaf>() || (IsSequenceOrSet(node))) {
    node = InsertEmptyPermutableBand(node);
    auto tag = REALIZE_BUF;
    node = node.insert_mark(isl::id(node.ctx(), tag));
    return node.get_schedule();
  }

  // Return if none of the above
  return curr_schedule;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
