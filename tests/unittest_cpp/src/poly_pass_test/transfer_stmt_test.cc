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
#include "poly/schedule_pass/transfer_stmt.h"

namespace akg {

TEST(TestTransferStmt, TestCase1) {
  isl::schedule input_sch;
  isl::schedule expect_ouput_sch;
  std::tie(input_sch, expect_ouput_sch) = ScheduleTreeHelper("transfer_stmt_case").Prepare();

  auto test_ctx = isl_ctx_alloc();
  ir::poly::ScopInfo scop_info(isl_ctx_alloc());
  ir::poly::PassInfo pass_info;

  // construct transfer_stmt : { S_0[kc1, h, w, kc0] }
  isl::schedule_node root = input_sch.get_root();
  isl::schedule_node node = ir::poly::GetOuterBand(root);
  if (node.isa<isl::schedule_node_sequence>() || node.isa<isl::schedule_node_set>()) {
    int n = static_cast<int>(node.n_children());
    for (int i = 0; i < n; ++i) {
      isl::schedule_node child = node.child(i);
      CHECK(child.isa<isl::schedule_node_filter>()) << "The child of set or sequence must filter!";
      isl::schedule_node_filter filter_node = child.as<isl::schedule_node_filter>();
      isl::union_set filter = filter_node.get_filter();
      if (filter.to_str() == "{ S_0[kc1, h, w, kc0] }") {
        scop_info.analysis_result_.RecordTransferStmt(filter);
        break;
      }
    }
  }

  input_sch = ir::poly::TransferStmt(scop_info, pass_info).Run(input_sch);
  EXPECT_TRUE(SCH_EQUAL(input_sch, expect_ouput_sch));
}
}  // namespace akg
