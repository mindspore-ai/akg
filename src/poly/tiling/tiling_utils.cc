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

#include "tiling_utils.h"

namespace akg {
namespace ir {
namespace poly {

void TileLogger::AppendLine(LogStage stage, const std::string &line) {
  if (stage == ANA_SCHETREE) {
    analyze_schedule_tree_stage_.emplace_back(line);
  } else if (stage == ANA_BUF_LIVE_EXTENT) {
    analyze_buffer_live_extent_stage_.emplace_back(line);
  } else if (stage == ANA_TILING_SPACE) {
    analyze_tiling_space_stage_.emplace_back(line);
  } else if (stage == DO_TILING) {
    do_tiling_stage_.emplace_back(line);
  } else if (stage == MICRO_TUNING) {
    micro_tuning_strage_.emplace_back(line);
  } else {
    do_tuning_stage_.emplace_back(line);
  }
}

void TileLogger::AppendLog(LogStage stage, std::stringstream &ss) {
  AppendLine(stage, ss.str());
  ss.str("");
}

bool TileLogger::DumpLogFile() {
  std::ofstream of;
  of.open(log_file_name_, std::ios::out);
  if (!of.is_open()) {
    return false;
  }
  of << " >>>>>>>>>> Analyze schedule tree stage <<<<<<<<<<<<" << std::endl;
  for (const auto &line : analyze_schedule_tree_stage_) {
    of << line << std::endl;
  }
  of << "=========================" << std::endl;
  of << " >>>>>>>>>> Analyze buffer live extent stage <<<<<<<<<<<<" << std::endl;
  for (const auto &line : analyze_buffer_live_extent_stage_) {
    of << line << std::endl;
  }
  of << "=========================" << std::endl;
  of << ">>>>>>>>>> Analyze tiling space stage <<<<<<<<<<<<" << std::endl;
  for (const auto &line : analyze_tiling_space_stage_) {
    of << line << std::endl;
  }
  of << "=========================" << std::endl;
  of << ">>>>>>>>>> Do tiling stage <<<<<<<<<<<<" << std::endl;
  for (const auto &line : do_tiling_stage_) {
    of << line << std::endl;
  }
  of << "=========================" << std::endl;
  of << ">>>>>>>>>> Do tuning stage <<<<<<<<<<<<" << std::endl;
  for (const auto &line : do_tuning_stage_) {
    of << line << std::endl;
  }
  of << "=========================" << std::endl;
  of << ">>>>>>>>>> Micro tuning stage <<<<<<<<<<<<" << std::endl;
  for (const auto &line : micro_tuning_strage_) {
    of << line << std::endl;
  }
  of << "=========================" << std::endl;
  of.close();
  return true;
}

void TileLogger::LogFatalAndSaveLog(const std::string &fatal_log) {
  if (!this->DumpLogFile()) LOG(WARNING) << "Write tiling log fail.";
  LOG(FATAL) << fatal_log;
}
std::string TileLogger::GetDumpDir() { return this->log_file_name_; }
}  // namespace poly
}  // namespace ir
}  // namespace akg
