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
#include "gtest/gtest.h"
#include "base/dump_helper.h"
#include "base/schedule_tree_helper.h"
#include "poly/pass_info.h"
#include "poly/schedule_pass/reorder_invariant_set_schedule.h"

namespace akg {

TEST(TestReorderInvariantSetSchedule, TestCase1) {
  isl::schedule input_sch;
  isl::schedule expect_ouput_sch;
  std::tie(input_sch, expect_ouput_sch) = ScheduleTreeHelper("reorder_invariant_set_sch_case").Prepare();

  ir::poly::PassInfo pass_info;
  pass_info.has_invariant_dependence_ = true;

  // construct pass_info.invariant_state_
  isl::schedule_node root = input_sch.get_root();
  isl::schedule_node outer_band = ir::poly::GetOuterBand(root);
  if (outer_band.as<isl::schedule_node_sequence>() || outer_band.as<isl::schedule_node_set>()) {
    for (unsigned int i = 0; i < outer_band.n_children(); ++i) {
      isl::schedule_node node = outer_band.get_child(i);
      auto filter = node.as<isl::schedule_node_filter>();
      isl::union_set sets = filter.filter();
      sets.foreach_set(
        [&pass_info](const isl::set &s) -> void { pass_info.invariant_state_.emplace(s.get_tuple_name(), 1); });
    }
  }

  input_sch = ir::poly::ReorderInvariantSetSchedule(pass_info).Run(input_sch);
  EXPECT_TRUE(SCH_EQUAL(input_sch, expect_ouput_sch));
}
}  // namespace akg
