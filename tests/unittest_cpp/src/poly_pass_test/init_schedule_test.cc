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
#include "poly/schedule_pass/init_schedule.h"
#include "poly/scop_info.h"

namespace akg {

TEST(TestInitSchedule, TestCase1) {
  isl::schedule input_schedule;
  isl::schedule expect_schedule;
  std::tie(input_schedule, expect_schedule) = ScheduleTreeHelper("init_schedule_case").Prepare();

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
  auto expect_dependence = isl::manage(dependence);
  ir::poly::PassInfo pass_info;
  ir::poly::ScopInfo scop_info = ir::poly::ScopInfo(input_schedule.ctx());

  // set reads
  std::string reads_str =
  "{ [S_5[i0, i1, i2, k1] -> __poly_ref_10[]] -> reduce_1_1[arg0 = i0, arg1 = i1, arg2 = i2, arg3 = k1] : 0 <= i0 <= 63 and 0 <= i1 <= 15 and 0 <= i2 <= 127 and 0 <= k1 <= 127;"
   " [S_4[i0, i1, i2, k1] -> __poly_ref_8[]] -> reduce_1_0[arg0 = i0, arg1 = i1, arg2 = i2, arg3 = k1] : 0 <= i0 <= 63 and 0 <= i1 <= 15 and 0 <= i2 <= 127 and 0 <= k1 <= 127;"
   " [S_6[i0, i1, i2, k1] -> __poly_ref_12[]] -> reduce_1[arg0 = i0, arg1 = i1, arg2 = i2, arg3 = 0] : 0 <= i0 <= 63 and 0 <= i1 <= 15 and 0 <= i2 <= 127 and 0 <= k1 <= 127;"
   " [S_7[ax0, ax1, ax2, ax3] -> __poly_ref_15[]] -> reduce_1[arg0 = ax0, arg1 = ax1, arg2 = ax2, arg3 = 0] : 0 <= ax0 <= 63 and 0 <= ax1 <= 15 and 0 <= ax2 <= 127 and 0 <= ax3 <= 127;"
   " [S_6[i0, i1, i2, k1] -> __poly_ref_13[]] -> reduce_1_2[arg0 = i0, arg1 = i1, arg2 = i2, arg3 = k1] : 0 <= i0 <= 63 and 0 <= i1 <= 15 and 0 <= i2 <= 127 and 0 <= k1 <= 127;"
   " [S_8[ax0, ax1, ax2, ax3] -> __poly_ref_17[]] -> reduce_1_2[arg0 = ax0, arg1 = ax1, arg2 = ax2, arg3 = ax3] : 0 <= ax0 <= 63 and 0 <= ax1 <= 15 and 0 <= ax2 <= 127 and 0 <= ax3 <= 127;"
   " [S_8[ax0, ax1, ax2, ax3] -> __poly_ref_18[]] -> T_divide_exp_1_broadcast_tensor_1_7[arg0 = ax0, arg1 = ax1, arg2 = ax2, arg3 = ax3] : 0 <= ax0 <= 63 and 0 <= ax1 <= 15 and 0 <= ax2 <= 127 and 0 <= ax3 <= 127;"
   " [S_3[i0, i1, i2, k1] -> __poly_ref_5[]] -> reduce_0[arg0 = i0, arg1 = i1, arg2 = i2, arg3 = 0] : 0 <= i0 <= 63 and 0 <= i1 <= 15 and 0 <= i2 <= 127 and 0 <= k1 <= 127;"
   " [S_1[i0, i1, i2, k1] -> __poly_ref_1[]] -> reduce_0[arg0 = i0, arg1 = i1, arg2 = i2, arg3 = 0] : 0 <= i0 <= 63 and 0 <= i1 <= 15 and 0 <= i2 <= 127 and 0 <= k1 <= 127;"
   " [S_4[i0, i1, i2, k1] -> __poly_ref_7[]] -> input_1[arg0 = i0, arg1 = i1, arg2 = i2, arg3 = k1] : 0 <= i0 <= 63 and 0 <= i1 <= 15 and 0 <= i2 <= 127 and 0 <= k1 <= 127;"
   " [S_1[i0, i1, i2, k1] -> __poly_ref_2[]] -> input_1[arg0 = i0, arg1 = i1, arg2 = i2, arg3 = k1] : 0 <= i0 <= 63 and 0 <= i1 <= 15 and 0 <= i2 <= 127 and 0 <= k1 <= 127 }";
  isl_union_map *p_reads = isl_union_map_read_from_str(input_schedule.ctx().get(), reads_str.c_str());
  auto reads = isl::manage(p_reads);
  scop_info.analysis_result_.RecordReads(reads);

  // set writes
  std::string writes_str =
  "{ [S_6[i0, i1, i2, k1] -> __poly_ref_14[]] -> reduce_1[arg0 = i0, arg1 = i1, arg2 = i2, arg3 = 0] : 0 <= i0 <= 63 and 0 <= i1 <= 15 and 0 <= i2 <= 127 and 0 <= k1 <= 127;"
  " [S_0[i0, i1, i2] -> __poly_ref_0[]] -> reduce_0[arg0 = i0, arg1 = i1, arg2 = i2, arg3 = 0] : 0 <= i0 <= 63 and 0 <= i1 <= 15 and 0 <= i2 <= 127;"
  " [S_1[i0, i1, i2, k1] -> __poly_ref_3[]] -> reduce_0[arg0 = i0, arg1 = i1, arg2 = i2, arg3 = 0] : 0 <= i0 <= 63 and 0 <= i1 <= 15 and 0 <= i2 <= 127 and 0 <= k1 <= 127;"
  " [S_2[i0, i1, i2] -> __poly_ref_4[]] -> reduce_1[arg0 = i0, arg1 = i1, arg2 = i2, arg3 = 0] : 0 <= i0 <= 63 and 0 <= i1 <= 15 and 0 <= i2 <= 127;"
  " [S_7[ax0, ax1, ax2, ax3] -> __poly_ref_16[]] -> T_divide_exp_1_broadcast_tensor_1_7[arg0 = ax0, arg1 = ax1, arg2 = ax2, arg3 = ax3] : 0 <= ax0 <= 63 and 0 <= ax1 <= 15 and 0 <= ax2 <= 127 and 0 <= ax3 <= 127;"
  " [S_3[i0, i1, i2, k1] -> __poly_ref_6[]] -> reduce_1_0[arg0 = i0, arg1 = i1, arg2 = i2, arg3 = k1] : 0 <= i0 <= 63 and 0 <= i1 <= 15 and 0 <= i2 <= 127 and 0 <= k1 <= 127;"
  " [S_5[i0, i1, i2, k1] -> __poly_ref_11[]] -> reduce_1_2[arg0 = i0, arg1 = i1, arg2 = i2, arg3 = k1] : 0 <= i0 <= 63 and 0 <= i1 <= 15 and 0 <= i2 <= 127 and 0 <= k1 <= 127;"
  " [S_8[ax0, ax1, ax2, ax3] -> __poly_ref_19[]] -> T_divide_exp_1_broadcast_tensor_1[arg0 = ax0, arg1 = ax1, arg2 = ax2, arg3 = ax3] : 0 <= ax0 <= 63 and 0 <= ax1 <= 15 and 0 <= ax2 <= 127 and 0 <= ax3 <= 127;"
  " [S_4[i0, i1, i2, k1] -> __poly_ref_9[]] -> reduce_1_1[arg0 = i0, arg1 = i1, arg2 = i2, arg3 = k1] : 0 <= i0 <= 63 and 0 <= i1 <= 15 and 0 <= i2 <= 127 and 0 <= k1 <= 127 }";
  isl_union_map *p_writes = isl_union_map_read_from_str(input_schedule.ctx().get(), writes_str.c_str());
  auto writes = isl::manage(p_writes);
  scop_info.analysis_result_.RecordWrites(writes);

  input_schedule = ir::poly::InitSchedule(pass_info, scop_info).Run(input_schedule);
  auto ret = pass_info.dependences_.is_equal(expect_dependence);
  std::cout << "ret: " << ret << std::endl;

  EXPECT_EQ(ret, true);
}
}  // namespace akg
