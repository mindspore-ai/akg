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
#include "poly/schedule_pass/sink_last_axis.h"

namespace akg {

TEST(TestSinkLastAxis, TestCase1) {
  isl::schedule input_sch;
  isl::schedule expect_ouput_sch;
  std::tie(input_sch, expect_ouput_sch) = ScheduleTreeHelper("sink_last_axis_case").Prepare();

  // Use union_map::to_str() to convert pass_info_.dependences_ to string type
  std::string dependences_str =
    "[NO, KO, MO] -> { S_0[b = 0, no, mo, mi, ni] -> S_1[b' = 0, no' = no, mo' = mo, mi' = mi, ni' = ni, ko = 0, ki = "
    "0] : KO > 0 and 0 <= no < NO and 0 <= mo < MO and 0 <= mi <= 15 and 0 <= ni <= 15; S_1[b = 0, no, mo, mi, ni, ko, "
    "ki] -> S_1[b' = 0, no' = no, mo' = mo, mi' = mi, ni' = ni, ko' = ko, ki' = 1 + ki] : 0 <= no < NO and 0 <= mo < "
    "MO and 0 <= mi <= 15 and 0 <= ni <= 15 and 0 <= ko < KO and 0 <= ki <= 14; S_1[b = 0, no, mo, mi, ni, ko, ki = "
    "15] -> S_1[b' = 0, no' = no, mo' = mo, mi' = mi, ni' = ni, ko' = 1 + ko, ki' = 0] : 0 <= no < NO and 0 <= mo < MO "
    "and 0 <= mi <= 15 and 0 <= ni <= 15 and 0 <= ko <= -2 + KO }";
  isl_union_map *read_tmp = isl_union_map_read_from_str(isl_ctx_alloc(), dependences_str.c_str());
  CHECK(read_tmp);
  ir::poly::PassInfo pass_info;
  pass_info.dependences_ = isl::manage(read_tmp);

  input_sch = ir::poly::SinkLastAxis(pass_info).Run(input_sch);
  EXPECT_TRUE(SCH_EQUAL(input_sch, expect_ouput_sch));
}
}  // namespace akg
