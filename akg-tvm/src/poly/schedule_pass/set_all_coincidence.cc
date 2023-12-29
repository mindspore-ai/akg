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
#include "set_all_coincidence.h"

namespace akg {
namespace ir {
namespace poly {

isl::schedule SetAllCoincidence::Run(isl::schedule curr_schedule) {
  const auto &new_schedule = curr_schedule;
  auto fn = [](isl::schedule_node node) -> isl::schedule_node {
    if (auto band = node.as<isl::schedule_node_band>()) {
      int num_dims = static_cast<int>(band.n_member());
      for (int dim = 0; dim < num_dims; ++dim) {
        bool is_coincident = band.member_get_coincident(dim);
        if (is_coincident) continue;
        node = band.member_set_coincident(dim, true);
        band = node.as<isl::schedule_node_band>();
      }
    }
    return node;
  };
  return new_schedule.get_root().map_descendant_bottom_up(fn).get_schedule();
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
