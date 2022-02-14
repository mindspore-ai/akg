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

#include "reschedule.h"

#include <fstream>
#include "poly/dump_log.h"

namespace akg {
namespace ir {
namespace poly {

bool Reschedule::ValidateSchedule(const isl::schedule &sch) {
  auto sched = sch.get_map();
  auto psched = isl_union_map_lex_lt_union_map(sched.copy(), sched.copy());
  sched = isl::manage(psched);
  auto identity = pass_info_.dependences_.domain().identity();
  auto dependences = pass_info_.dependences_.subtract(identity);
  bool is_valid = dependences.is_subset(sched);
  return is_valid;
}

isl::schedule_node Reschedule::RecomputeScheduleTree(const isl::schedule_node &node) {
  auto ctx = node.ctx();
  auto origin_scc = isl_options_get_schedule_serialize_sccs(ctx.get());
  isl_stat status = isl_options_set_schedule_serialize_sccs(ctx.get(), 1);
  CHECK(status == isl_stat_ok);
  auto pnode = isl_schedule_node_schedule(node.copy(), pass_info_.constraints_.copy());
  auto new_node = isl::manage(pnode);
  status = isl_options_set_schedule_serialize_sccs(ctx.get(), origin_scc);
  CHECK(status == isl_stat_ok);
  return new_node;
}

/* Reschedule schedule tree with serialize sccs for "root".
 * Currently, two patterns of rescheduling are implemented.
 * pattern 1:
 *     Mark(realize_UB or realize_UBL0)
 *        |
 *        Tile band
 *           |
 *           Point band  <--- reschedule position
 * pattern 2:
 *     Mark(realize_L1)
 *        |
 *        Band
 *           |
 *           Mark(realize_UB or realize_UBL0)
 *              |
 *              Band  <--- reschedule position
 * Return the node of the schedule after rescheduling.
 */
isl::schedule_node Reschedule::RescheduleSchTree(const isl::schedule_node &root) {

  auto fn = [&](isl::schedule_node node) -> isl::schedule_node {
    if (node.isa<isl::schedule_node_mark>()) {
      auto tag = node.as<isl::schedule_node_mark>().get_id().get_name();
      if ((tag == "realize_UBL0") || (tag == "realize_UB")) {
        auto node_m = node.insert_mark(RESCHEDULE);
        node_m = node_m.get_child(0).get_child(0);
        if (node_m.isa<isl::schedule_node_band>()) {
          if ((node.parent().isa<isl::schedule_node_band>())
            && (node.parent().parent().isa<isl::schedule_node_mark>())
            && (node.parent().parent().as<isl::schedule_node_mark>().get_id().get_name() == "realize_L1")) {
            node_m = RecomputeScheduleTree(node_m);
          } else {
            node_m = node_m.get_child(0);
            node_m = RecomputeScheduleTree(node_m);
            node_m = node_m.parent();
          }
        }
        node = node_m.parent().parent();
      }
    }
    return node;
  };
  auto node = root.map_descendant_bottom_up(fn);
  if (ValidateSchedule(node.get_schedule())) {
    LOG(INFO) << "Schedule tree is valid, ^_^";
  } else {
    LOG(WARNING) << "Schedule tree is invalid, pls check the correctness！";
  }
  return node;
}

isl::schedule Reschedule::Run(isl::schedule sch) {
  auto root = sch.get_root();
  auto node = RescheduleSchTree(root);
  return node.get_schedule();
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
