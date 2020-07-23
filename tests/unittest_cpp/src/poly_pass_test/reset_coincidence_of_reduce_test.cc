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
#include "poly/schedule_pass/reset_coincidence_of_reduce.h"

namespace akg {

TEST(TestResetCoincidence, TestCase1) {
  isl::schedule input_sch;
  isl::schedule expect_ouput_sch;
  std::tie(input_sch, expect_ouput_sch) = ScheduleTreeHelper("reset_coincidence_of_reduce_case").Prepare();

  // construct reduce_stmt : {"S_2" : ["k0", "k1"]}
  auto test_ctx = isl_ctx_alloc();
  ir::poly::ScopInfo scop_info(test_ctx);
  ir::poly::PassInfo pass_info;
  auto ConstructReduceStmtMap = [&scop_info](isl::schedule_node node) -> isl::schedule_node {
    if (auto band = node.as<isl::schedule_node_band>()) {
      int num_dims = static_cast<int>(band.n_member());
      for (int dim = 0; dim < num_dims; ++dim) {
        auto dim_schedule = band.get_partial_schedule().get_union_pw_aff(dim);
        auto stmt_list = dim_schedule.get_pw_aff_list();
        stmt_list.foreach([&scop_info](const isl::pw_aff &stmt) {
          isl::id stmt_id = stmt.domain().get_tuple_id();
          if (stmt_id.to_str() == "S_2" && isl_pw_aff_involves_dims(stmt.get(), isl_dim_in, 1, 1) &&
              scop_info.analysis_result_.GetReduceStmtMap().count(stmt_id) == 0) {
            std::vector<std::string> reduce_axis_list = {"k0", "k1"};
            scop_info.analysis_result_.RecordReduceStmt(stmt_id, reduce_axis_list);
            return;
          }
        });
      }
    }
    return node;
  };
  input_sch.get_root().map_descendant_bottom_up(ConstructReduceStmtMap);

  input_sch = ir::poly::ResetCoincidenceOfReduce(scop_info, pass_info).Run(input_sch);
  EXPECT_TRUE(SCH_EQUAL(input_sch, expect_ouput_sch));
}
}  // namespace akg
