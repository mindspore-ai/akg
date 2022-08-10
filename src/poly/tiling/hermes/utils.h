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
#ifndef POLY_TILING_HERMES_UTILS_H_
#define POLY_TILING_HERMES_UTILS_H_

#include <tvm/ir.h>

#include <string>
#include <vector>

namespace akg {
namespace ir {
namespace poly {
std::string ParseString(const air::Expr &);
int ParseInt(const air::Integer &);
std::vector<int> ParseIntArray(air::Array<air::Integer>);
std::vector<std::string> ParseStringArray(air::Array<air::Expr> arr);
std::string StripRename(std::string name);
int64_t Get2PowerBelow(int64_t n);

const int kByTwo = 2;
const size_t kByTwoUL = 2;
const int64_t kByTwoL = 2;
}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_TILING_HERMES_UTILS_H_
