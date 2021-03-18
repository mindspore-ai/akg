/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef COMMON_UTIL_H_
#define COMMON_UTIL_H_

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

#include <tvm.h>

namespace akg {
namespace common {
/// IsNumber check whether string is a number.
/// \param str input string
/// \return result of operation.
inline bool IsNumber(const std::string &str) {
  return !str.empty() && std::find_if(str.begin(), str.end(), [](char c) { return !std::isdigit(c); }) == str.end();
}

/// Split string to string list with specific character
/// \param str string needs to be splited
/// \param delimiter split character
/// \return vector of string, with substrings of input string
inline std::vector<std::string> Split(const std::string &str, const std::string delimiter) {
  std::vector<std::string> result;
  std::string curr_str = str;
  while (!curr_str.empty()) {
    size_t pos = curr_str.find(delimiter);
    if (pos == std::string::npos) {
      result.push_back(curr_str);
      break;
    }
    if (pos != 0) {
      result.push_back(curr_str.substr(0, pos));
    }
    curr_str = curr_str.substr(pos + delimiter.size());
  }
  return result;
}

inline std::string GetStringEnv(const char *env_name) {
  const char *ret = getenv(env_name);
  return ret != nullptr ? std::string(ret) : std::string();
}

inline int GetIntegerEnv(const char *env_name) {
  auto str_ret = GetStringEnv(env_name);
  return !str_ret.empty() ? static_cast<int>(std::strtol(str_ret.c_str(), nullptr, 10)) : 0;
}

inline air::Expr SplitCast(const air::Expr &input, const air::Type &target_type) {
  auto cast = input.as<Cast>();
  if (cast == nullptr) {
    return input;
  } else if (cast->type == target_type) {
    return cast->value;
  }
  return air::Expr();
}

}  // namespace common
}  // namespace akg
#endif  // COMMON_UTIL_H_
