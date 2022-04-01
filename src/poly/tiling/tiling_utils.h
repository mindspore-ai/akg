/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include <iostream>
#include <fstream>

#include <tvm/target_info.h>
#include <tvm/ir.h>
#include <tvm/packed_func_ext.h>

#include "common/target_info.h"
#include "build_module.h"
#include "poly/dsa_utils.h"

namespace akg {
namespace ir {
namespace poly {

/* Device Info  */
enum TilingMemScope {
  // global
  MEM_SCOPE_GM = 0,
  // npu
  MEM_SCOPE_BUFFER,
  MEM_SCOPE_CACHE1,
  MEM_SCOPE_CACHE0_A,
  MEM_SCOPE_CACHE0_B,
  MEM_SCOPE_CACHE0_C,
  // gpu
  MEM_SCOPE_SHARED,
  MEM_SCOPE_LOCAL,
  // end
  MEM_SCOPE_BULK,
};

int GetCoreNumConf();

class NpuInfo {
 public:
  ~NpuInfo() {}
  static NpuInfo &GetInstance() {
    static NpuInfo hardware_info;
    return hardware_info;
  }

  int64_t GetMemoryLimitInScope(int scope_idx) {
    CHECK_LT(scope_idx, MEM_SCOPE_BULK);
    return npu_mem_limit_[scope_idx];
  }

 private:
  NpuInfo() { InitNpuMemoryLimit(); }
  int64_t npu_mem_limit_[MEM_SCOPE_BULK]{0};

  void InitNpuMemoryLimit() {
    auto CollectLimit = [this](const std::string &scope, TilingMemScope mem) {
      air::MemoryInfo info = air::GetMemoryInfo(scope);
      CHECK(info.defined());
      npu_mem_limit_[mem] = info->max_num_bits / 8;
    };
    CollectLimit(DOT_LOCAL_BUF, MEM_SCOPE_BUFFER);
    CollectLimit(DOT_LOCAL_C1, MEM_SCOPE_CACHE1);
    CollectLimit(DOT_LOCAL_C0A, MEM_SCOPE_CACHE0_A);
    CollectLimit(DOT_LOCAL_C0B, MEM_SCOPE_CACHE0_B);
    CollectLimit(DOT_LOCAL_C0C, MEM_SCOPE_CACHE0_C);
    npu_mem_limit_[MEM_SCOPE_GM] = 0;
  }
};

class GpuInfo {
 public:
  ~GpuInfo() {}
  static GpuInfo &GetInstance(const std::string &device_type) {
    static GpuInfo hardware_info(device_type);
    return hardware_info;
  }

  int64_t GetMemoryLimitInScope(int scope_idx) {
    CHECK_LT(scope_idx, MEM_SCOPE_BULK);
    return gpu_mem_limit_[scope_idx];
  }

  int GetNumSm() { return num_sm_; }

  int GetActiveBlocksPerSm() { return active_blocks_per_sm_; }

  int GetMinElemForIoBound() { return min_elem_for_io_bound_; }

 private:
  explicit GpuInfo(const std::string &device_type) {
    InitGpuMemoryLimit(device_type);
    InitGpuComputeCapability(device_type);
  }
  int64_t gpu_mem_limit_[MEM_SCOPE_BULK]{0};
  int num_sm_{80};
  int active_blocks_per_sm_{5};
  int min_elem_for_io_bound_{2};

  void InitGpuMemoryLimit(const std::string &device_type) {
    auto CollectLimit = [this, &device_type](const std::string &scope, TilingMemScope mem) {
      air::GpuMemoryInfo info = air::GetGpuMemoryInfo(scope, device_type);
      CHECK(info.defined());
      gpu_mem_limit_[mem] = info->max_bytes_per_block;
    };
    CollectLimit("shared", MEM_SCOPE_SHARED);
    CollectLimit("reg", MEM_SCOPE_LOCAL);
    gpu_mem_limit_[MEM_SCOPE_GM] = 0;
  }

  void InitGpuComputeCapability(const std::string &device_type) {
    std::string scope = "instance";
    air::GpuComputeInfo info = air::GetGpuComputeInfo(scope, device_type);
    num_sm_ = info->num_sm;
    active_blocks_per_sm_ = info->active_blocks_per_sm;
    min_elem_for_io_bound_ = info->min_elem_for_io_bound;
  }
};

/* Log utils */
enum LogStage {
  ANA_SCHETREE,
  ANA_BUF_LIVE_EXTENT,
  ANA_TILING_SPACE,
  DO_TILING,
  DO_TUNING,
  MICRO_TUNING,
  GPU_MAPPING,
  CPU_TILING
};

class TileLogger {
 public:
  explicit TileLogger(const std::string &log_file_name, bool enable_dump)
      : log_file_name_(log_file_name), enable_dump_(enable_dump) {}
  ~TileLogger() {}

  using LogFile = std::vector<std::string>;
  void AppendLine(LogStage stage, const std::string &line);
  void AppendLog(LogStage stage, std::stringstream &ss);
  bool DumpLogFile();
  void ClearCache();
  void LogFatalAndSaveLog(const std::string &fatal_log);
  std::string GetDumpDir();

 private:
  std::string log_file_name_;
  bool enable_dump_{true};
  LogFile analyze_schedule_tree_stage_;
  LogFile analyze_buffer_live_extent_stage_;
  LogFile analyze_tiling_space_stage_;
  LogFile do_tiling_stage_;
  LogFile do_tuning_stage_;
  LogFile micro_tuning_stage_;
  LogFile gpu_mapping_stage_;
  LogFile cpu_tiling_stage_;
};

/* Halide & Schedule tree analysis utils */
using Band = std::vector<const air::ir::For *>;
using VarNames = std::vector<std::string>;

std::unordered_map<std::string, std::string> ExtractLoopIndicesFromMatrices(std::vector<VarNames> var_names_list);
std::unordered_map<std::string, std::string> ExtractLoopIndicesFromMatricesConv(std::vector<VarNames> var_names_list);

/* Data format definition */
constexpr auto kDsaN = "N";
constexpr auto kDsaC = "C";
constexpr auto kDsaH = "H";
constexpr auto kDsaW = "W";
constexpr auto kDsaC1 = "C1";
constexpr auto kDsaC0 = "C0";
constexpr auto kDsaC1In = "C1_in";
constexpr auto kDsaC1Out = "C1_out";
constexpr auto kDsaC0In = "C0_in";
constexpr auto kDsaC0Out = "C0_out";
constexpr auto kDsaC1InOut = "C1_in_out";
constexpr auto kDsaHIn = "H_in";
constexpr auto kDsaWIn = "WIn";

constexpr auto kDsami = "mi";
constexpr auto kDsamo = "mo";
constexpr auto kDsani = "ni";
constexpr auto kDsano = "no";
constexpr auto kDsaki = "ki";
constexpr auto kDsako = "ko";
constexpr auto kDsabi = "bi";
constexpr auto kDsabo = "bo";
constexpr auto kDsawi = "wi";
constexpr auto kDsahi = "hi";
constexpr auto kDsaoc = "oc";
constexpr auto kDsaic = "ic";
constexpr auto kDsakh = "kh";
constexpr auto kDsakw = "kw";

const VarNames DsaNCHW = {kDsaN, kDsaC, kDsaH, kDsaW, kDsaC0};
const VarNames DsaNHWCC0 = {kDsaN, kDsaH, kDsaW, kDsaC, kDsaC0};
const VarNames DsaNC1HWC0 = {kDsaN, kDsaC1, kDsaH, kDsaW, kDsaC0};

const VarNames ForwardFilter = {kDsaC1In, kDsaC1Out, kDsaC0Out, kDsaC0In};   //  nZ, Cin = [kc1,kh,kw]
const VarNames BackpropFilter = {kDsaC1Out, kDsaC1In, kDsaC0In, kDsaC0Out};  //  backprop_input, Cout = [kc1,kh,kw]
const VarNames ForwardFeaturemap = {kDsaN, kDsaC1In, kDsaHIn, kDsaWIn, kDsaC0In};  // zZ, H_in = [H, Kh], W_in = [W, kw]
const VarNames BackpropFeaturemap = {kDsaN, kDsaC1Out, kDsaHIn, kDsaWIn,
                                     kDsaC0Out};  // zZ, H_in = [H, Kh], W_in = [W, kw]
const VarNames FilterOutput = {kDsaC1Out, kDsakh, kDsakw, kDsaC1In, kDsaC0In, kDsaC0Out};
const VarNames FilterInput = {kDsaN, kDsaC1Out, kDsaH, kDsaW, kDsaC0Out};

const VarNames FormatM = {kDsami, kDsamo};
const VarNames FormatN = {kDsani, kDsano};
const VarNames FormatK = {kDsaki, kDsako};
const VarNames FormatB = {kDsabi, kDsabo};

const VarNames ConvFormatM = {kDsawi, kDsahi, kDsami};
const VarNames ConvFormatN = {kDsaoc};
const VarNames ConvFormatK = {kDsaic, kDsakw, kDsakh};

constexpr auto NO_PRUNE = 0;
constexpr auto PRUNE_MEM_EXCEED = 1;
constexpr auto PRUNE_ALIGNED_MEM_EXCEED = 2;
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_TILING_UTILS_H_
