/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef COMPOSITE_OPTIMIZE_H_
#define COMPOSITE_OPTIMIZE_H_
#include <string>
#include <vector>
#include <functional>
#include "composite/utils/util.h"
#include "composite/utils/dump.h"

namespace akg {

class TranslatePassMgr {
 public:
  using PassFunc = std::function<Stmt(const Stmt &, BuildInfo *)>;

  explicit TranslatePassMgr(BuildInfo *info, const std::string &dump_dir_suffix = "composite")
      : info_(info), dump_dir_suffix_(dump_dir_suffix) {}
  ~TranslatePassMgr() = default;

  void Register(const std::string &name, PassFunc pass) { passes_.emplace_back(std::make_pair(name, pass)); }

  Stmt Run(const Stmt &s) {
    auto file_name = !info_->opt.stitch
                       ? info_->kernel_name + "_" + dump_dir_suffix_
                       : "stitch_info/" + info_->kernel_name + "_stitch_" + std::to_string(info_->opt.stitch_ir_idx);
    auto enable_dump = info_->opt.enable_dump && getenv(GetDumpIRFlag().c_str()) != nullptr;
    auto dump_mng = DumpManager(file_name, enable_dump);
    auto stmt = s;
    dump_mng.DumpStmt("Origin", stmt);
    for (auto &pass : passes_) {
      stmt = pass.second(stmt, info_);
      dump_mng.DumpStmt(pass.first, stmt);
    }
    return stmt;
  }

 private:
  BuildInfo *info_;
  std::string dump_dir_suffix_;
  std::vector<std::pair<std::string, PassFunc>> passes_;
};

Stmt Optimize(Stmt &s, BuildInfo &info);

Stmt OptimizeForTBE(const Stmt &s, BuildInfo &info);
}  // namespace akg
#endif  // COMPOSITE_OPTIMIZE_H_
