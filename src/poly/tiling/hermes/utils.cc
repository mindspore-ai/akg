/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "poly/tiling/hermes/utils.h"

namespace akg {
namespace ir {
namespace poly {
std::string ParseString(air::Expr expression) {
  if (const auto *const strimm = expression.as<air::ir::StringImm>()) {
    return strimm->value;
  } else {
    LOG(FATAL) << "String cannot be parsed";
    return "";
  }
}

int ParseInt(air::Integer integer) {
  if (const auto *const intimm = integer.as<air::ir::IntImm>()) {
    return intimm->value;
  } else {
    LOG(FATAL) << "Int cannot be parsed";
    return -1;
  }
}

std::vector<int> ParseIntArray(air::Array<air::Integer> arr) {
  std::vector<int> vec;
  for (air::Integer i : arr) {
    vec.push_back(ParseInt(i));
  }
  return vec;
}

std::vector<std::string> ParseStringArray(air::Array<air::Expr> arr) {
  std::vector<std::string> vec;
  for (air::Expr s : arr) {
    vec.push_back(ParseString(s));
  }
  return vec;
}

std::string StripRename(std::string name) {
  size_t pos = name.rfind("_rename");
  if (pos == std::string::npos) {
    return name;
  }
  return name.substr(0, pos);
}

int Get2PowerBelow(int n) {
  int result = n;
  constexpr int twice = 2;
  for (int i = 1; i < n; i *= twice) {
    result = i;
  }
  return result;
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
