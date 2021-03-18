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

#include "mark_fuse_op.h"

namespace akg {
namespace ir {
namespace poly {

isl::schedule MarkFuseOp::Run(isl::schedule schedule_mark) {
  auto fn = [](isl::schedule_node node) -> isl::schedule_node {
    if (node.isa<isl::schedule_node_mark>()) {
      std::string mark_id = node.as<isl::schedule_node_mark>().get_id().get_name();
      size_t pos = mark_id.find(UBL0);
      if (pos != std::string::npos) {
        std::string m = FUSE_VECTOR;
        node = node.insert_mark(isl::id(node.ctx(), m));
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
