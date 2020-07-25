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

#include "insert_node_for_allocc.h"

namespace akg {
namespace ir {
namespace poly {

isl::schedule_node InsertNodeForAllocCImpl(isl::schedule_node node) {
  if (node.isa<isl::schedule_node_mark>()) {
    if (node.as<isl::schedule_node_mark>().get_id().get_name() == REALIZE_L1) {
      node = node.del();
      node =
        node.as<isl::schedule_node_band>().split(static_cast<int>(node.as<isl::schedule_node_band>().n_member()) - 1);
      node = node.child(0);
      node = node.insert_mark(isl::id(node.ctx(), REALIZE_L1));
      node = node.insert_mark(isl::id(node.ctx(), ALLOC_C));
      node = node.parent();
    }
  }
  return node;
}

isl::schedule InsertNodeForAllocC::Run(isl::schedule sched) {
  // add alloc_C
  sched = sched.get_root().map_descendant_bottom_up(InsertNodeForAllocCImpl).get_schedule();
  return sched;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
