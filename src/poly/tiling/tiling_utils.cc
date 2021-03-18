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
    micro_tuning_stage_.emplace_back(line);
  } else if (stage == GPU_MAPPING) {
    gpu_mapping_stage_.emplace_back(line);
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

VarNames VisitVarNames(const air::Expr &arg, VarNames var_names, bool add_num) {
  if (const auto var = arg.as<air::ir::Variable>()) {
    var_names.emplace_back(var->name_hint);
  } else if (const auto sub = arg.as<air::ir::Sub>()) {
    var_names = VisitVarNames(sub->a, var_names, add_num);
    var_names = VisitVarNames(sub->b, var_names, add_num);
  } else if (const auto add = arg.as<air::ir::Add>()) {
    var_names = VisitVarNames(add->a, var_names, add_num);
    var_names = VisitVarNames(add->b, var_names, add_num);
  } else if (const auto mul = arg.as<air::ir::Mul>()) {
    var_names = VisitVarNames(mul->a, var_names, add_num);
    var_names = VisitVarNames(mul->b, var_names, add_num);
  } else if (const auto div = arg.as<air::ir::Div>()) {
    var_names = VisitVarNames(div->a, var_names, add_num);
    var_names = VisitVarNames(div->b, var_names, add_num);
  } else if (const auto mod = arg.as<air::ir::Mod>()) {
    var_names = VisitVarNames(mod->a, var_names, add_num);
    var_names = VisitVarNames(mod->b, var_names, add_num);
  } else if (const auto int_imm = arg.as<air::ir::IntImm>()) {
    if (add_num) {
      var_names.emplace_back(std::to_string(int_imm->value));
    }
  } else if (const auto f_mod = arg.as<air::ir::FloorMod>()) {
    var_names = VisitVarNames(f_mod->a, var_names, add_num);
    var_names = VisitVarNames(f_mod->b, var_names, add_num);
  } else if (const auto f_div = arg.as<air::ir::FloorDiv>()) {
    var_names = VisitVarNames(f_div->a, var_names, add_num);
    var_names = VisitVarNames(f_div->b, var_names, add_num);
  }
  return var_names;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
