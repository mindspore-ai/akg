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
#include "poly/schedule_pass_gpu/mapping_outer_band.h"
#include "poly/scop_info.h"
#include "poly/pass_info.h"

namespace akg {

TEST(TestMappingOuterBand, TestCase1) {
  isl::schedule input_sch;
  isl::schedule expect_ouput_sch;
  std::tie(input_sch, expect_ouput_sch) = ScheduleTreeHelper("mapping_outer_band_case").Prepare();

  auto test_ctx = input_sch.ctx();
  ir::poly::ScopInfo scop_info(test_ctx);
  scop_info.user_config_.SetBlockConfig("256 256");
  scop_info.user_config_.SetThreadConfig("32 8");
  ir::poly::PassInfo pass_info;

  input_sch = ir::poly::MappingOuterBand(pass_info, scop_info).Run(input_sch);

  // The sch read from a file or str using ISL wil keep the following order {S1, S0},
  // while isl::schedule_node_band.tile will keep {S0,S1} order,
  // resulting in input_sch and expect_ouput_sch never being the same.
  // So here save the generated sch as string type and convert to isl::schedule.
  std::string generate_str = isl_schedule_to_str(input_sch.get());
  generate_str = ScheduleTreeHelper("mapping_outer_band_case").UndoPrettyPrintSchTree(generate_str);
  isl_schedule *exp_out_ss = isl_schedule_read_from_str(input_sch.ctx().get(), generate_str.c_str());
  CHECK(exp_out_ss != nullptr) << "fail to read string";
  input_sch = isl::manage(exp_out_ss);

  EXPECT_TRUE(SCH_EQUAL(input_sch, expect_ouput_sch));
}
}  // namespace akg
