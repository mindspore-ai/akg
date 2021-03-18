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
#include "poly/stmt_parse.h"
#include "poly/schedule_pass/keep_outer_band_order.h"

namespace akg {

TEST(TestKeepOuterBandOrder, TestCase1) {
  isl::schedule input_sch;
  isl::schedule expect_ouput_sch;
  std::tie(input_sch, expect_ouput_sch) = ScheduleTreeHelper("keep_outer_band_order_case").Prepare();

  ir::poly::ScopInfo scop_info_(input_sch.ctx().get());
  ir::poly::StmtOpInfo stmt_op_Info;
  stmt_op_Info.isMMU = true;
  scop_info_.analysis_result_.RecordStmtOpInfo(isl::id(input_sch.ctx().get(), "S_0"), stmt_op_Info);
  input_sch = ir::poly::KeepOuterBandOrder(scop_info_).Run(input_sch);
  EXPECT_TRUE(SCH_EQUAL(input_sch, expect_ouput_sch));
}
}  // namespace akg
