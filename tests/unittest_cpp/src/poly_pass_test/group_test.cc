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
#include "poly/schedule_pass/group.h"
#include "poly/scop_info.h"

namespace akg {

TEST(TestGroup, TestCase1) {
  isl::schedule input_schedule;
  isl::schedule expect_schedule;
  std::tie(input_schedule, expect_schedule) = ScheduleTreeHelper("group_case").Prepare();

  std::string dependences_str =
"{ S_1[i0, i1, i2, k1] -> S_1[i0' = i0, i1' = i1, i2' = i2, k1' = 1 + k1] : 0 <= i0 <= 63 and 0 <= i1 <= 15 and 0 <= i2 <= 127 and 0 <= k1 <= 126;"
 " S_3[i0, i1, i2, k1] -> S_4[i0' = i0, i1' = i1, i2' = i2, k1' = k1] : 0 <= i0 <= 63 and 0 <= i1 <= 15 and 0 <= i2 <= 127 and 0 <= k1 <= 127;"
 " S_4[i0, i1, i2, k1] -> S_5[i0' = i0, i1' = i1, i2' = i2, k1' = k1] : 0 <= i0 <= 63 and 0 <= i1 <= 15 and 0 <= i2 <= 127 and 0 <= k1 <= 127;"
 " S_6[i0, i1, i2, k1] -> S_6[i0' = i0, i1' = i1, i2' = i2, k1' = 1 + k1] : 0 <= i0 <= 63 and 0 <= i1 <= 15 and 0 <= i2 <= 127 and 0 <= k1 <= 126;"
 " S_5[i0, i1, i2, k1] -> S_6[i0' = i0, i1' = i1, i2' = i2, k1' = k1] : 0 <= i0 <= 63 and 0 <= i1 <= 15 and 0 <= i2 <= 127 and 0 <= k1 <= 127;"
 " S_5[i0, i1, i2, k1] -> S_8[ax0 = i0, ax1 = i1, ax2 = i2, ax3 = k1] : 0 <= i0 <= 63 and 0 <= i1 <= 15 and 0 <= i2 <= 127 and 0 <= k1 <= 127;"
 " S_2[i0, i1, i2] -> S_6[i0' = i0, i1' = i1, i2' = i2, k1 = 0] : 0 <= i0 <= 63 and 0 <= i1 <= 15 and 0 <= i2 <= 127;"
 " S_6[i0, i1, i2, k1 = 127] -> S_7[ax0 = i0, ax1 = i1, ax2 = i2, ax3] : 0 <= i0 <= 63 and 0 <= i1 <= 15 and 0 <= i2 <= 127 and 0 <= ax3 <= 127;"
 " S_1[i0, i1, i2, k1 = 127] -> S_3[i0' = i0, i1' = i1, i2' = i2, k1'] : 0 <= i0 <= 63 and 0 <= i1 <= 15 and 0 <= i2 <= 127 and 0 <= k1' <= 127;"
 " S_0[i0, i1, i2] -> S_1[i0' = i0, i1' = i1, i2' = i2, k1 = 0] : 0 <= i0 <= 63 and 0 <= i1 <= 15 and 0 <= i2 <= 127;"
 " S_7[ax0, ax1, ax2, ax3] -> S_8[ax0' = ax0, ax1' = ax1, ax2' = ax2, ax3' = ax3] : 0 <= ax0 <= 63 and 0 <= ax1 <= 15 and 0 <= ax2 <= 127 and 0 <= ax3 <= 127 }";

  isl_union_map *dependence = isl_union_map_read_from_str(input_schedule.ctx().get(), dependences_str.c_str());
  CHECK(dependence);

  ir::poly::PassInfo pass_info;
  pass_info.dependences_ = isl::manage(dependence);

  input_schedule = ir::poly::GroupStatements(pass_info).Run(input_schedule);
  auto ret = input_schedule.plain_is_equal(expect_schedule);

  EXPECT_EQ(ret, true);
}
}  // namespace akg
