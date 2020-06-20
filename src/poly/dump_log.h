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
#ifndef POLY_DUMP_LOG_H_
#define POLY_DUMP_LOG_H_

#include <isl/cpp.h>
#include <tvm/node/node.h>
#include <string>
namespace akg {
namespace ir {
namespace poly {
std::string FormatMupaStr(const std::string &mupa_str, bool checkInString = false);
std::string FormatMupaStr(const isl::union_map &map);
std::string FormatMupaStr(const isl::union_set &set);
std::string FormatMupaStr(const isl::multi_aff &aff);
std::string FormatMupaStr(const isl::multi_pw_aff &mpa);
std::string FormatMupaStr(const isl::multi_union_pw_aff &mupa);
std::string FormatMupaStr(const isl::union_pw_aff &upa);

std::string FilePathCanonicalize(const std::string &file_name, bool is_log);
bool CreateFileIfNotExist(const std::string &file_name);
void CreateDirIfNotExist(const std::string &file_name);
std::string DumpSchTreeToString(const isl::schedule &sch);
void DumpSchTreeImpl(const std::string &file_name, const isl::schedule &sch);
void PrintHeader(std::ofstream &of, const std::string &str);
void DumpNode(std::ofstream &of, const ktvm::Node *node);

bool CompareSchTreeWithString(const std::string &compare_sch, const isl::schedule &sch);

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_DUMP_LOG_H_
