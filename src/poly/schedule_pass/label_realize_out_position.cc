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

#include "label_realize_out_position.h"
#include <climits>

namespace akg {
namespace ir {
namespace poly {

isl::schedule LabelRealizeOutPosition::Run(isl::schedule sch_label) {
  auto fn_ = [](isl::schedule_node node) -> isl::schedule_node {
    if (node.isa<isl::schedule_node_mark>()) {
      if (REALIZE_UB == node.as<isl::schedule_node_mark>().get_id().get_name() &&
          node.child(0).isa<isl::schedule_node_band>()) {
        auto band = node.child(0).as<isl::schedule_node_band>();

        unsigned pos = UINT_MAX;
        auto updatePos_ = [&pos](isl::schedule_node node) -> isl::schedule_node {
          if (node.isa<isl::schedule_node_filter>()) {
            node = node.get_child(0);
            if (node.isa<isl::schedule_node_band>()) {
              auto band = node.as<isl::schedule_node_band>();
              CHECK_LT(band.n_member(), UINT_MAX);
              for (unsigned i = 0; i < band.n_member(); ++i) {
                if (!band.member_get_coincident(i)) {
                  if (i < pos) pos = i;
                  break;
                }
              }
            }
            node = node.parent();
          }
          return node;
        };

        static_cast<void>(band.map_descendant_bottom_up(updatePos_));

        for (unsigned i = 0; i < band.n_member(); ++i) {
          if (!band.member_get_coincident(i)) {
            if (i < pos) pos = i;
            break;
          }
        }

        if (pos < band.n_member()) {
          node = node.del();
          node = node.as<isl::schedule_node_band>().split(pos);
          node = node.child(0);
          node = node.insert_mark(isl::id(node.ctx(), REALIZE_UB));
          node = node.insert_mark(isl::id(node.ctx(), ALLOC_REALIZE_OUT));
          node = node.parent();
        }
      }
    }
    return node;
  };
  return sch_label.get_root().map_descendant_bottom_up(fn_).get_schedule();
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
