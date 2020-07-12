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
#ifndef COMPOSITE_UTIL_H_
#define COMPOSITE_UTIL_H_
#include <string>
#include <unordered_map>

#include "tvm.h"

namespace akg {
constexpr auto kMsDavinciKernelPath = "./kernel_meta/";
static std::unordered_map<std::string, air::Type> type_mapping = {
  {"float32", air::Float(32)}, {"float16", air::Float(16)}, {"int32", air::Int(32)}, {"bool", air::Bool()}};
}  // namespace akg

#endif  // COMPOSITE_UTIL_H_
