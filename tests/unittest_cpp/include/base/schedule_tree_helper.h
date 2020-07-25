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

#ifndef UT_BASE_SCHEDULE_TREE_HELPER_H_
#define UT_BASE_SCHEDULE_TREE_HELPER_H_
#include <fstream>
#include "poly/poly_util.h"

namespace akg {
bool SCH_EQUAL(const isl::schedule &schedule1, const isl::schedule &schedule2);

class ScheduleTreeHelper {
 public:
  ScheduleTreeHelper(const std::string &file_name)
      : input_sch_file(GetInputScheduleTree(file_name)),
        expect_output_sch_file(GetExpectOutputScheduleTree(file_name)) {}
  ScheduleTreeHelper(std::string in, std::string out) : input_sch_file(in), expect_output_sch_file(out) {}
  ~ScheduleTreeHelper(){};

  std::tuple<isl::schedule, isl::schedule> Prepare();
  std::string GetPolyPassCasePath();
  std::string GetInputScheduleTree(const std::string &file_name);
  std::string GetExpectOutputScheduleTree(const std::string &file_name);

 private:
  static std::string UndoPrettyPrintSchTree(const std::string &schedule);

  const std::string input_sch_file;
  const std::string expect_output_sch_file;
};
}  // namespace akg
#endif  // UT_BASE_SCHEDULE_TREE_HELPER_H_
