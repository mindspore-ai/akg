/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "codegen/util.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <libgen.h>

#include <fstream>
#include <iostream>

#include "pass/utils.h"

namespace akg {
void RecordCore(const int core, bool enable_file_log) {
  if (enable_file_log) {
    auto file_env = getenv(kPerformanceTestFile);
    if (file_env != nullptr) {
      std::string result_file = file_env;
      std::ofstream of(result_file, std::ios::app);
      CHECK(of.is_open()) << "Failed open " << result_file << " to record core.";

      of << core << "; ";
      of.close();
    }
  }
}

void RecordCore(const Stmt &stmt, bool enable_file_log) {
  int core_num = 1;
  const auto attr_stmt = stmt.as<AttrStmt>();
  if (attr_stmt && attr_stmt->attr_key == "thread_extent") {
    auto attr_value = attr_stmt->value;
    core_num = ir::GetInt32Const(attr_value);
  }
  if (enable_file_log) {
    auto file_env = getenv(kPerformanceTestFile);
    if (file_env != nullptr) {
      std::string result_file = file_env;
      std::ofstream of(result_file, std::ios::app);
      CHECK(of.is_open()) << "Failed open " << result_file << " to record core.";

      of << core_num << "; ";
      of.close();
    }
  }
}

void CreateDir(const std::string &file) {
  char *file_name = strdup(file.c_str());
  CHECK(file_name != nullptr);
  char *dir = file_name;
  if (strcmp(dir, ".") == 0) {
    LOG(WARNING) << "Cannot create root directory " << file;
    free(file_name);
    return;
  }
  struct stat info;
  if (stat(dir, &info) == 0) {
    if (!(info.st_mode & S_IFDIR)) {
      LOG(WARNING) << "Directory " << std::string(dir) << " already exists but it is not a directory";
    }
    free(file_name);
    return;
  }
  const int dir_mode = S_IRUSR | S_IWUSR | S_IXUSR;
  if (mkdir(dir, dir_mode) != 0) {
    char *dir_copy = strdup(dir);
    CHECK(dir_copy != nullptr);
    const char *parent_dir = dirname(dir_copy);
    CHECK(parent_dir != nullptr);
    CreateDir(parent_dir);
    free(dir_copy);
    if (mkdir(dir, dir_mode) != 0) {
      LOG(WARNING) << "Failed to create directory " << std::string(dir);
    }
  }
  free(file_name);
}

int AttrMap::GetIntAttr(const std::string &attr_name, int dft_value) {
  if (this->count(attr_name) == 0) {
    return dft_value;
  }
  const NodeRef &e = this->at(attr_name);
  return ir::GetInt32Const(Downcast<Expr>(e));
}
double AttrMap::GetFloatAttr(const std::string &attr_name, double dft_value) {
  if (this->count(attr_name) == 0) {
    return dft_value;
  }
  const NodeRef &e = this->at(attr_name);
  return ir::GetFloatConst(Downcast<Expr>(e));
}
bool AttrMap::GetBoolAttr(const std::string &attr_name, bool dft_value) {
  int result = GetIntAttr(attr_name, static_cast<int>(dft_value));
  CHECK(result == 0 || result == 1) << "Bool attribute " << attr_name << " must be 0 or 1, but found "
                                    << this->at(attr_name);
  return static_cast<bool>(result);
}

bool AttrMap::GetStringAttr(const std::string &attr_name, std::string *const attr_to_set) {
  if (this->count(attr_name) == 0) {
    return false;
  }
  const NodeRef &e = this->at(attr_name);
  if (auto val = e.as<StringImm>()) {
    *attr_to_set = val->value;
  } else {
    LOG(FATAL) << "Failed to parse attribute: " << attr_name << " = " << e << " as string";
  }
  return true;
}

std::string AttrMap::GetStringAttr(const std::string &attr_name, const std::string &dft_value) {
  std::string tmp;
  if (GetStringAttr(attr_name, &tmp)) {
    return tmp;
  }
  return dft_value;
}

void PassTimer::AddItem(const std::string &pass_name, int64_t elapsed_seconds) {
  auto iter = pass_time_.find(pass_name);
  if (iter != pass_time_.end()) {
    iter->second += elapsed_seconds;
  } else {
    pass_time_[pass_name] = elapsed_seconds;
  }
}

std::string PassTimer::ToString() const {
  std::stringstream buf;
  buf << "PassName - Time";
  if (pass_time_.empty()) {
    return buf.str();
  }

  std::vector<std::pair<std::string, int64_t>> timers;
  for (auto iter = pass_time_.begin(); iter != pass_time_.end(); ++iter) {
    timers.push_back(*iter);
  }
  sort(timers.begin(), timers.end(),
       [](const std::pair<std::string, int64_t> t1, const std::pair<std::string, int64_t> t2) {
         return t1.second > t2.second;
       });
  if (timers.size() > kMaxNumOfPassTimeToPrint) {
    timers.resize(kMaxNumOfPassTimeToPrint);
  }

  for (auto iter : timers) {
    buf << "\n" << iter.first << " - " << iter.second << " s";
  }
  return buf.str();
}

std::ostream &operator<<(std::ostream &os, const PassTimer &time) {
  os << time.ToString();
  return os;
}
}  // namespace akg
