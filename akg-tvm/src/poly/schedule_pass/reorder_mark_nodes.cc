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

#include "reorder_mark_nodes.h"

namespace akg {
namespace ir {
namespace poly {

isl::schedule ReorderMarkNodes::Run(isl::schedule schedule_mark) {
  auto fn = [](isl::schedule_node node) -> isl::schedule_node {
    if (node.isa<isl::schedule_node_mark>()) {
      // mark node cannot be inserted between sequence node and its filter children, so skip reorder
      if (node.get_child(0).as<isl::schedule_node_sequence>()) return node;

      std::string mark_id = node.as<isl::schedule_node_mark>().get_id().get_name();
      size_t pos = mark_id.find(REALIZE_);
      if (pos != std::string::npos) {
        node = node.del();
        node = node.get_child(0);
        node = node.insert_mark(isl::id(node.ctx(), mark_id));
        node = node.parent();
      }
    }
    return node;
  };
  return schedule_mark.get_root().map_descendant_bottom_up(fn).get_schedule();
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
