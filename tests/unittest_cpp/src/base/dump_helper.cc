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
#include <iostream>
#include <regex>
#include <sstream>
#include "base/dump_helper.h"

namespace akg {
const std::string UTRegxMatch::pattern_hex_ = "0[xX][0-9a-fA-F]+";

bool UTRegxMatch::RegxMatchHex(const std::string &str) {
  return std::regex_match(str, std::regex(pattern_hex_));
}

std::string UTDumpHelper::Dump(const ktvm::NodeRef &node) {
  std::stringstream ss;
  ss << node;
  return ss.str();
}

bool UTDumpHelper::RegxMatchPlaceholder(const std::string &str, const std::string &name) {
  std::string pattern = "placeholder\\(" + name + ", " + UTRegxMatch::pattern_hex_ + "\\)";
  return std::regex_match(str, std::regex(pattern));
}
}  // namespace akg
