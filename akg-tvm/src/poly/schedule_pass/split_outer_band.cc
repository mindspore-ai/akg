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
#include "split_outer_band.h"

namespace akg {
namespace ir {
namespace poly {

isl::schedule SplitOuterBand::Run(isl::schedule curr_schedule) {
  isl::schedule_node node = curr_schedule.get_root();
  while (!node.isa<isl::schedule_node_band>()) {
    node = node.child(0);
  }
  isl::schedule_node_band band = node.as<isl::schedule_node_band>();
  unsigned i = 0;
  unsigned n = band.n_member();
  for (; i < n; ++i) {
    if (!band.member_get_coincident(i)) {
      break;
    }
  }
  if ((n <= 1) || (i == 0) || (i == n)) {
    return node.get_schedule();
  }
  node = band.split(i);
  return node.get_schedule();
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
