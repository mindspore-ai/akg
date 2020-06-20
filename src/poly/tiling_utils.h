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
#ifndef POLY_TILING_UTILS_H_
#define POLY_TILING_UTILS_H_

#include <tvm/target_info.h>
#include <iostream>
#include <fstream>

namespace akg {
namespace ir {
namespace poly {
enum DavinciMemScope {
  MEM_SCOPE_GM = 0,
  MEM_SCOPE_UB,
  MEM_SCOPE_L1,
  MEM_SCOPE_L0A,
  MEM_SCOPE_L0B,
  MEM_SCOPE_L0C,
  MEM_SCOPE_BULK,
};
enum LogStage { ANA_SCHETREE, ANA_BUF_LIVE_EXTENT, ANA_TILING_SPACE, DO_TILING, DO_TUNING };

class DavinciInfo {
 public:
  ~DavinciInfo() {}
  static DavinciInfo &GetInstance() {
    static DavinciInfo hardware_info;
    return hardware_info;
  }

  int64_t GetMemoryLimitInScope(int scope_idx) {
    CHECK_LT(scope_idx, MEM_SCOPE_BULK);
    return davinci_mem_limit_[scope_idx];
  }

 private:
  DavinciInfo() { InitDavinciMemoryLimit(); }
  int64_t davinci_mem_limit_[MEM_SCOPE_BULK]{0};

  void InitDavinciMemoryLimit() {
    auto CollectLimit = [this](const std::string &scope, DavinciMemScope mem) {
      ktvm::MemoryInfo info = ktvm::GetMemoryInfo(scope);
      CHECK(info.defined());
      davinci_mem_limit_[mem] = info->max_num_bits / 8;
    };
    CollectLimit("local.UB", MEM_SCOPE_UB);
    CollectLimit("local.L1", MEM_SCOPE_L1);
    CollectLimit("local.L0A", MEM_SCOPE_L0A);
    CollectLimit("local.L0B", MEM_SCOPE_L0B);
    CollectLimit("local.L0C", MEM_SCOPE_L0C);
    davinci_mem_limit_[MEM_SCOPE_GM] = 0;
  }
};

class TileLogger {
 public:
  ~TileLogger() {}
  using LogFile = std::vector<std::string>;
  static TileLogger &GetInstance(std::string log_file_name) {
    static TileLogger tile_logger_(log_file_name);
    return tile_logger_;
  }
  void AppendLine(LogStage stage, const std::string &line);
  void AppendLog(LogStage stage, std::stringstream &ss);
  bool DumpLogFile();
  void LogFatalAndSaveLog(const std::string &fatal_log);
  std::string GetDumpDir();

 private:
  explicit TileLogger(std::string log_file_name) : log_file_name_(log_file_name) {}

  std::string log_file_name_;
  LogFile analyze_schedule_tree_stage_;
  LogFile analyze_buffer_live_extent_stage_;
  LogFile analyze_tiling_space_stage_;
  LogFile do_tiling_stage_;
  LogFile do_tuning_stage_;
};
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_TILING_UTILS_H_
