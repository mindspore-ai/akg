/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef COMPOSITE_UTILS_DUMP_H_
#define COMPOSITE_UTILS_DUMP_H_
#include <iomanip>
#include <fstream>
#include "tvm.h"
#include <codegen/util.h>
#include "composite/utils/util.h"
#include "composite/lower_tree/stitch_fusion.h"
#include "build_module.h"

namespace akg {
class DumpManager {
 public:
  DumpManager(const std::string &dir_path, bool active) : dir_path_(dir_path), active_(active) {}
  ~DumpManager() = default;

  void DumpStmt(const std::string &file_name, const Stmt &stmt) {
    if (!active_) {
      return;
    }
    ReadyCheck();
    auto file_path = GetIrFilePath(file_name);
    std::ofstream of(file_path);
    CHECK(of.is_open()) << "Failed to open " << file_path << " to dump ir.";
    of << stmt;
    of.close();
  }

  void DumpStmt(const std::string &file_name, const std::vector<Stmt> &stmts) {
    if (stmts.size() == 1) {
      DumpStmt(file_name, stmts[0]);
      return;
    }
    if (!active_) {
      return;
    }
    ReadyCheck();
    auto file_path = GetIrFilePath(file_name);
    std::ofstream of(file_path);
    CHECK(of.is_open()) << "Failed to open " << file_path << " to dump ir.";
    for (size_t i = 0; i < stmts.size(); ++i) {
      of << "---------[" << i << "]";
      of << "\n";
      of << stmts[i];
      of << "\n---------\n";
    }
    of.close();
  }

 private:
  void ReadyCheck() {
    if (!dumped_) {
      // Lazy create.
      CreateDir(dir_path_);
      dumped_ = true;
    }
  }
  std::string GetIrFilePath(const std::string &file_name) {
    std::stringstream sstr;
    sstr << dir_path_ << "/" << std::setw(2) << std::setfill('0') << (id_++) << "_" << file_name << ".cc";
    return sstr.str();
  }

  std::string dir_path_;
  bool dumped_{false};
  bool active_{false};
  int id_{0};
};
void DumpStr2File(const std::string &file_name, const std::string &str);
void DumpStmt2File(const std::string &file_name, const Stmt &stmt);
void DumpStitchInfo(const std::string &kernel_name, StitchAttrInfo &store_attr,
                    std::unordered_map<std::string, StitchBufferInfo> &stitch_buffer_map,
                    std::unordered_map<std::string, StitchBufferInfo> &buf_within_op_map,
                    std::vector<std::string> &allocate_revoke);
void DumpIRAttr(const std::string &kernel_name, const IrAttrInfo &attr, size_t index);
void DumpHeader(std::ofstream &of, const std::string &str);
void DumpBuildInfo(const BuildInfo &info);
}  // namespace akg

#endif  // COMPOSITE_UTILS_DUMP_H_
