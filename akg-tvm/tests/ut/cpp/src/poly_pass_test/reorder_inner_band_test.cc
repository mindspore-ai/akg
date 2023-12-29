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
#include "poly/schedule_pass/reorder_inner_band.h"

namespace akg {

TEST(TestReorderInnerBand, TestCase1) {
  isl::schedule input_schedule;
  isl::schedule expect_schedule;
  std::tie(input_schedule, expect_schedule) = ScheduleTreeHelper("reorder_inner_band_case").Prepare();

  ir::poly::CondVarsMap cond_vars;
  cond_vars[isl::id(input_schedule.ctx(), "S_0")].insert("c_i");
  cond_vars[isl::id(input_schedule.ctx(), "S_0")].insert("c_i0");
  auto output_schedule = ir::poly::ReorderInnerBand(cond_vars).Run(input_schedule);
  EXPECT_TRUE(SCH_EQUAL(output_schedule, expect_schedule));
}
}  // namespace akg
