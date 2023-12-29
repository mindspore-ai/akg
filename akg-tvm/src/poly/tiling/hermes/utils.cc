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
#include <cmath>

#include "poly/tiling/hermes/utils.h"

namespace akg {
namespace ir {
namespace poly {
std::string ParseString(const air::Expr &expr) {
  if (const auto *const strimm = expr.as<air::ir::StringImm>()) {
    return strimm->value;
  }
  LOG(FATAL) << "String cannot be parsed";
  return "";
}

int ParseInt(const air::Integer &num) {
  if (const auto *const intimm = num.as<air::ir::IntImm>()) {
    return static_cast<int>(intimm->value);
  }
  LOG(FATAL) << "Int cannot be parsed";
  return -1;
}

std::vector<int> ParseIntArray(const air::Array<air::Integer> &arr) {
  std::vector<int> vec;
  for (air::Integer num : arr) {
    vec.push_back(ParseInt(num));
  }
  return vec;
}

std::vector<std::string> ParseStringArray(const air::Array<air::Expr> &arr) {
  std::vector<std::string> vec;
  for (air::Expr expr : arr) {
    vec.push_back(ParseString(expr));
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

int64_t Get2PowerLess(int64_t num) {
  int64_t result = num;
  for (int64_t i = 1; i < num; i *= kByTwoL) {
    result = i;
  }
  return result;
}

int64_t Get2PowerLessEq(int64_t num) {
  int64_t result = num;
  for (int64_t i = 1; i <= num; i *= kByTwoL) {
    result = i;
  }
  return result;
}

int64_t GetLowestPrimeFactorsProductBelow(int64_t num, int64_t ulimit) {
  int64_t factors_prod = 1;
  int64_t prod = 0;

  while (num % kByTwoL == 0) {
    prod = factors_prod * kByTwoL;
    if (prod > ulimit) {
      return factors_prod;
    }
    factors_prod = prod;
    num /= kByTwoL;
  }

  for (int64_t i = 3; i <= static_cast<int64_t>(std::round(std::sqrt(num))); i += kByTwoL) {
    while (num % i == 0) {
      prod = factors_prod * i;
      if (prod > ulimit) {
        return factors_prod;
      }
      factors_prod = prod;
      num /= i;
    }
  }

  if (num > kByTwoL) {
    prod = factors_prod * num;
    if (prod > ulimit) {
      return factors_prod;
    }
    factors_prod = prod;
  }

  return factors_prod;
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
