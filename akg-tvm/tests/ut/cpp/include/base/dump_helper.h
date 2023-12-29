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
#ifndef UT_BASE_DUMP_HELPER_H_
#define UT_BASE_DUMP_HELPER_H_
#include <string>
#include <tvm/node/node.h>
#include <tvm/expr.h>
#include <poly/isl.h>

namespace akg {
class UTRegxMatch {
 public:
  UTRegxMatch() = default;
  ~UTRegxMatch() = default;

  // match pattern 0x...
  static bool RegxMatchHex(const std::string &str);

  static const std::string pattern_hex_;
};  // class UTRegxMatch

class UTDumpHelper {
 public:
  UTDumpHelper() = default;
  ~UTDumpHelper() = default;

  static std::string Dump(const air::NodeRef &node);
  static bool RegxMatchPlaceholder(const std::string &str, const std::string &name);
  static std::string DumpScheduleTree(const isl::schedule &sch);
};  // UTDumpHelper
}  // namespace akg
#endif  // UT_BASE_DUMP_HELPER_H_
