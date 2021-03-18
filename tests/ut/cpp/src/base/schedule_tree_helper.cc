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
#define PATH_MAX 4096
#define MAX_PATH_DEPTH 20
#include <fstream>
#include <unistd.h>
#include "poly/poly_util.h"
#include "base/schedule_tree_helper.h"
#include "base/dump_helper.h"

namespace akg {
bool SCH_EQUAL(const isl::schedule &schedule1, const isl::schedule &schedule2) {
  return schedule1.plain_is_equal(schedule2);
}

std::tuple<isl::schedule, isl::schedule> ScheduleTreeHelper::Prepare() {
  std::ifstream input_sch_file_stream(input_sch_file);
  std::string input_str((std::istreambuf_iterator<char>(input_sch_file_stream)), std::istreambuf_iterator<char>());
  input_str = UndoPrettyPrintSchTree(input_str);
  isl_schedule *in_ss = isl_schedule_read_from_str(isl_ctx_alloc(), input_str.c_str());
  isl::schedule input_sch;
  if (in_ss != nullptr) {
    input_sch = isl::manage(in_ss);
  } else {
    LOG(WARNING) << "Failed to load file " << input_sch_file
                 << " to schedule tree, please check syntax of the input schedule.";
  }

  std::ifstream expect_output_sch_file_stream(expect_output_sch_file);
  std::string expect_output_str((std::istreambuf_iterator<char>(expect_output_sch_file_stream)),
                                std::istreambuf_iterator<char>());
  expect_output_str = UndoPrettyPrintSchTree(expect_output_str);
  isl_schedule *exp_out_ss = isl_schedule_read_from_str(input_sch.ctx().get(), expect_output_str.c_str());
  isl::schedule expect_output_sch;
  if (exp_out_ss != nullptr) {
    expect_output_sch = isl::manage(exp_out_ss);
  } else {
    LOG(WARNING) << "Failed to load file " << expect_output_sch_file
                 << " to schedule tree, please check syntax of the expect output schedule.";
  }

  return std::make_tuple(input_sch, expect_output_sch);
}

std::string ScheduleTreeHelper::UndoPrettyPrintSchTree(const std::string &schedule) {
  const char *src = schedule.c_str();
  std::stringstream dst;
  bool in_string = false;
  while (*src != '\0') {
    if (*src == '"') {
      in_string = !in_string;
      if (!in_string) {
        // end of string, find next non-empty char
        const char *next = src + 1;
        while (*next != '\0') {
          char c = *next;
          if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
            break;
          }
          ++next;
        }
        if (*next == '"') {
          // multiple consecutive strings, merge them and insert a white space
          dst << " ";
          src = next + 1;
          in_string = true;
          continue;
        }
      }
    }
    dst << *src++;
  }
  return dst.str();
}

std::string ScheduleTreeHelper::GetPolyPassCasePath() {
  std::string relative_path("/tests/unittest_cpp/src/poly_pass_case/");

  char cwd[PATH_MAX];
  char *ret = getcwd(cwd, sizeof(cwd));
  CHECK(ret != nullptr);

  char abspath[PATH_MAX];
  char *res = realpath(cwd, abspath);
  CHECK(res != nullptr);
  CHECK_EQ(0, access(abspath, F_OK));

  int path_depth_count = 0;
  std::string dirname(abspath);
  while (access((dirname + relative_path).c_str(), F_OK) != 0) {
    std::string parent_path = dirname;
    std::string::size_type pos = dirname.find_last_of("/");
    if (pos != std::string::npos) {
      parent_path = dirname.substr(0, pos);
    }

    if (parent_path == dirname) {
      LOG(WARNING) << "Failed to find " << relative_path << " file.";
      return "";
    }
    dirname = parent_path;
    ++path_depth_count;
    if (path_depth_count > MAX_PATH_DEPTH) {
      LOG(WARNING) << "Failed to find " << relative_path << " file.";
      return "";
    }
  }
  return dirname + relative_path;
}

std::string ScheduleTreeHelper::GetInputScheduleTree(const std::string &file_name) {
  return GetPolyPassCasePath() + file_name + "/input_case.txt";
}
std::string ScheduleTreeHelper::GetExpectOutputScheduleTree(const std::string &file_name) {
  return GetPolyPassCasePath() + file_name + "/expect_output_case.txt";
}

}  // namespace akg
