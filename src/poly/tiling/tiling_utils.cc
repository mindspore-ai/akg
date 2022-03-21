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

int GetCoreNumConf() {
  int product_block = GetCoreValue("Core_num");
  int user_defined_block = g_attrs.GetInt(kEnableMulticore, -1);
  if (user_defined_block == -1) {
    // User is not defining core num, assume we can use maximal number.
    return product_block;
  } else if (user_defined_block > 1) {
    // Use core according to user and product.
    return std::min(product_block, user_defined_block);
  } else {
    // User disables multicore.
    return 1;
  }
}

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
    micro_tuning_stage_.emplace_back(line);
  } else if (stage == GPU_MAPPING) {
    gpu_mapping_stage_.emplace_back(line);
  } else if (stage == CPU_TILING) {
    cpu_tiling_stage_.emplace_back(line);
  } else {
    do_tuning_stage_.emplace_back(line);
  }
}

void TileLogger::AppendLog(LogStage stage, std::stringstream &ss) {
  AppendLine(stage, ss.str());
  ss.str("");
}

bool TileLogger::DumpLogFile() {
  if (!enable_dump_) {
    return true;
  }
  std::ofstream of;
  of.open(log_file_name_, std::ios::out);
  if (!of.is_open()) {
    return false;
  }
  of << " ============ Analyze schedule tree stage ============" << std::endl;
  for (const auto &line : analyze_schedule_tree_stage_) {
    of << line << std::endl;
  }
  of << " ============ Analyze buffer live extent stage ============" << std::endl;
  for (const auto &line : analyze_buffer_live_extent_stage_) {
    of << line << std::endl;
  }
  of << "============ Analyze tiling space stage ============" << std::endl;
  for (const auto &line : analyze_tiling_space_stage_) {
    of << line << std::endl;
  }
  of << "============ Do tiling stage ============" << std::endl;
  for (const auto &line : do_tiling_stage_) {
    of << line << std::endl;
  }
  of << "============ Do tuning stage ============" << std::endl;
  for (const auto &line : do_tuning_stage_) {
    of << line << std::endl;
  }
  of << "============ Micro tuning stage ============" << std::endl;
  for (const auto &line : micro_tuning_stage_) {
    of << line << std::endl;
  }
  of << "============ Gpu mapping stage ============" << std::endl;
  for (const auto &line : gpu_mapping_stage_) {
    of << line << std::endl;
  }
  of << "============ Cpu tiling stage ============" << std::endl;
  for (const auto &line : cpu_tiling_stage_) {
    of << line << std::endl;
  }
  of << "===========================================" << std::endl;
  of.close();
  ClearCache();
  return true;
}

void TileLogger::ClearCache() {
  analyze_schedule_tree_stage_.clear();
  analyze_buffer_live_extent_stage_.clear();
  analyze_tiling_space_stage_.clear();
  do_tiling_stage_.clear();
  do_tuning_stage_.clear();
  micro_tuning_stage_.clear();
  gpu_mapping_stage_.clear();
  cpu_tiling_stage_.clear();
}

void TileLogger::LogFatalAndSaveLog(const std::string &fatal_log) {
  if (!this->DumpLogFile()) LOG(WARNING) << "Write tiling log fail.";
  LOG(FATAL) << fatal_log;
}
std::string TileLogger::GetDumpDir() { return this->log_file_name_; }

/*
 * For a matmul operator C[b, m, n] += A[b, m, k] * B[b, n, k] where `b` indicates batch dim, `m` indicates row dim, `n`
 * indicates col dim and `k` indicates reduction dim, this function can extract `b, m, n, k` from all the loop vars
 * using in the matrices C, A and B. The input should be sorted in `CAB` order, i.e. var_names_list = [C_vars, A_vars,
 * B_vars], which equals [[b, m, n], [b, m, k], [b, n, k]] in the example above.
 */
std::unordered_map<std::string, std::string> ExtractLoopIndicesFromMatrices(std::vector<VarNames> var_names_list) {
  CHECK_EQ(var_names_list.size(), 3)
    << "Matmul should have exactly three matrices in C(output), A(lhs) and B(rhs) order.";
  VarNames mx_c = var_names_list[0];
  VarNames mx_a = var_names_list[1];
  VarNames mx_b = var_names_list[2];

  VarNames gemm_m, gemm_n, gemm_bk, gemm_b, gemm_k;
  std::unordered_set<std::string> stack;

  for (const auto &var : mx_a) {
    stack.insert(var);
  }

  // 1. N = B_vars - A_vars;
  //    [B, K] = A_vars & B_vars
  for (const auto &var : mx_b) {
    auto it = stack.find(var);
    if (it != stack.end()) {
      gemm_bk.emplace_back(var);
      stack.erase(it);
    } else {
      gemm_n.emplace_back(var);
    }
  }

  // 2. M = A_vars - B - K
  for (const auto &var : mx_a) {
    if (stack.find(var) != stack.end()) {
      gemm_m.emplace_back(var);
    }
  }

  // 3. B = C_vars - M - N
  for (const auto &var : gemm_n) {
    stack.insert(var);
  }
  for (const auto &var : mx_c) {
    auto it = stack.find(var);
    if (it != stack.end()) {
      stack.erase(it);
    } else {
      gemm_b.emplace_back(var);
    }
  }

  // 4. K = [B, K] - B
  for (const auto &var : gemm_bk) {
    bool found = false;
    for (const auto &b : gemm_b) {
      if (b == var) {
        found = true;
        break;
      }
    }
    if (!found) {
      gemm_k.emplace_back(var);
    }
  }

  CHECK_LE(gemm_m.size(), FormatM.size());
  CHECK_LE(gemm_n.size(), FormatN.size());
  CHECK_LE(gemm_k.size(), FormatK.size());
  CHECK_LE(gemm_b.size(), FormatB.size());

  std::unordered_map<std::string, std::string> cube_var_map;
  for (auto i = static_cast<int>(gemm_m.size()) - 1; i >= 0; --i) {
    cube_var_map[gemm_m[i]] = FormatM[static_cast<int>(gemm_m.size()) - 1 - i];
  }
  for (auto i = static_cast<int>(gemm_n.size()) - 1; i >= 0; --i) {
    cube_var_map[gemm_n[i]] = FormatN[static_cast<int>(gemm_n.size()) - 1 - i];
  }
  for (auto i = static_cast<int>(gemm_k.size()) - 1; i >= 0; --i) {
    cube_var_map[gemm_k[i]] = FormatK[static_cast<int>(gemm_k.size()) - 1 - i];
  }
  for (auto i = static_cast<int>(gemm_b.size()) - 1; i >= 0; --i) {
    cube_var_map[gemm_b[i]] = FormatB[static_cast<int>(gemm_b.size()) - 1 - i];
  }
  return cube_var_map;
}

std::unordered_map<std::string, std::string> ExtractLoopIndicesFromMatricesConv(std::vector<VarNames> var_names_list) {
  CHECK_EQ(var_names_list.size(), 3)
    << "Matmul should have exactly three matrices in C(output), A(lhs) and B(rhs) order.";
  VarNames mx_c = var_names_list[0];
  VarNames mx_a = var_names_list[1];
  VarNames mx_b = var_names_list[2];

  VarNames conv_m, conv_n, conv_k;
  std::unordered_set<std::string> stack;

  for (const auto &var : mx_a) {
    stack.insert(var);
  }

  // 1. N = B_vars - A_vars;
  //    [B, K] = A_vars & B_vars
  for (const auto &var : mx_b) {
    auto it = stack.find(var);
    if (it != stack.end()) {
      conv_k.emplace_back(var);
      stack.erase(it);
    } else {
      conv_n.emplace_back(var);
    }
  }

  // 2. M = A_vars - B - K
  for (const auto &var : mx_a) {
    if (stack.find(var) != stack.end()) {
      conv_m.emplace_back(var);
    }
  }

  CHECK_LE(conv_m.size(), ConvFormatM.size());
  CHECK_LE(conv_n.size(), ConvFormatN.size());
  CHECK_LE(conv_k.size(), ConvFormatK.size());

  std::unordered_map<std::string, std::string> cube_var_map;
  for (auto i = static_cast<int>(conv_m.size()) - 1; i >= 0; --i) {
    cube_var_map[conv_m[i]] = ConvFormatM[static_cast<int>(conv_m.size()) - 1 - i];
  }
  for (auto i = static_cast<int>(conv_n.size()) - 1; i >= 0; --i) {
    cube_var_map[conv_n[i]] = ConvFormatN[static_cast<int>(conv_n.size()) - 1 - i];
  }
  for (auto i = static_cast<int>(conv_k.size()) - 1; i >= 0; --i) {
    cube_var_map[conv_k[i]] = ConvFormatK[static_cast<int>(conv_k.size()) - 1 - i];
  }
  return cube_var_map;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
