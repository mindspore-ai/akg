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
#ifndef CODEGEN_UTIL_H_
#define CODEGEN_UTIL_H_

#include <dlpack/dlpack.h>
#include <stdlib.h>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tvm.h"

namespace akg {
constexpr auto kPerformanceTestFile = "PERFORMANCE_TEST_FILE";
constexpr auto kToThreeAddressMinSplit = "to_three_address_min_split";
constexpr auto kHelpTiling = "help_tiling";
constexpr auto kEnableConvertIf = "enable_convert_if";
constexpr auto kEnableFixLoopExtent = "enable_fix_loop_extent";
constexpr auto kEnableIsolateLoop = "enable_isolate_loop";
constexpr auto kEnableIsolateMinMax = "enable_isolate_min_max";
constexpr auto kEnableDmaSink = "enable_dma_sink";
constexpr auto kCoarsenImg2Col = "coarsenImg2Col";
constexpr auto kEnableHoistInsn = "enable_hoist_insn";
constexpr auto kEnableInvariantHoist = "enable_invariant_hoist";
constexpr auto kEnablePostPolyLoopPartition = "enable_post_poly_loop_partition";
constexpr auto kEnablePreStorageWriteSimplify = "enable_pre_storage_write_simplify";
constexpr auto kLoopPartitionUnroll = "loop_partition_unroll";
constexpr auto kExtentToCond = "extent_to_cond";
constexpr auto kEnableMulticore = "enable_multicore";
constexpr auto kMergeOuterLoop = "merge_outer_loop_for_multicore";
constexpr auto kMultiCoreLoopMaxDepth = "multicore_loop_max_depth";
constexpr auto kMultiCoreScalarRerrange = "multicore_scalar_rearrange";
constexpr auto kMultiCoreLoopSwitchHoist = "multicore_loop_switch_hoist";
constexpr auto kRecordCore = "record_core";
constexpr auto kEnableBisectOptimize = "enable_bisect_optimize";
constexpr auto kEnableCoverProtectOptimize = "enable_cover_protect_optimize";
constexpr auto kEnableDoubleBuffer = "enable_double_buffer";
constexpr auto kEnableUnrollLoop = "enable_unroll_loop";
constexpr auto kAlgebraSimplify = "enable_algebra_simplify";
constexpr auto kPromoteCommonExpr = "promote_common_expr";
constexpr auto kPromoteConstExpr = "promote_const_expr";
constexpr auto kUseBcOpt = "enable_bk_optimize";
constexpr auto kDumpIrDir = "dump_ir_dir";
constexpr auto kDumpPassIr = "dump_pass_ir";
constexpr auto kDumpPolyDir = "dump_poly_dir";
constexpr auto kMaxsatFile = "maxsat_file";
constexpr auto kEnablePrePolyLoopPartition = "enable_pre_poly_loop_partition";
constexpr auto kEnableToThreeAddress = "enable_to_three_address";
constexpr auto kToThreeAddressCrossSimply = "to_three_address_cross_simplify";
constexpr auto kToThreeAddressReuse = "to_three_address_reuse";
constexpr auto kDisableCse = "disable_cse";
constexpr auto kDeadCodeElim = "dead_code_elim";
constexpr auto kDisableVn = "disable_vn";
constexpr auto kRewriteVarTensorIdx = "RewriteVarTensorIdx";
constexpr auto kDumpTuningLevel = "dump_tuning_level";
constexpr auto kKernelName = "kernel_name";
constexpr auto kAlways = "always";
constexpr auto kEleminateOutmostForCond = "eleminate_outmost_for_cond";
constexpr auto kDisableHalfToFloatSumOpt = "disable_half_to_float_sum_opt";
constexpr auto kAkgTargetHostName = "stackvm";
constexpr auto kEnableAutoInline = "enable_auto_inline";
constexpr auto kEnableFeatureLibrary = "enable_feature_library";
constexpr auto kEnableFeatureLibraryPrePoly = "enable_feature_library_pre_poly";
constexpr auto kEnableHoistCondWrite = "enable_hoist_cond_write";
constexpr double kUsPerSecond = 1e6;
constexpr size_t kMaxNumOfPassTimeToPrint = 5;
constexpr auto kIsDynamic = "is_dynamic";
constexpr auto kEnableConvAnalyzeAlign = "enable_conv_analyze_align";
constexpr auto kEnableHoistAllocate = "enable_hoist_allocate";
constexpr auto kEnableScalarAlign = "enable_scalar_align";
constexpr auto kEnableStrideKernelOp = "enable_stride_kernel_op";
constexpr auto kTileSizeIsVar = "pragma_tilesize_is_var";
constexpr auto kEnableSinkAllocate = "enable_sink_allocate";
constexpr auto kEnableRemoveBroadcastCopy = "enable_remove_broadcast_copy";
constexpr auto kEnableSubstituteDivVar = "enable_divide_var";
constexpr auto kEnableComputeInPlace = "enable_compute_in_place";
constexpr auto kEnableRewriteScalarCompute = "enable_rewrite_scalar_compute";
constexpr auto kMaxNumRetryPoly = "max_num_retry_poly";
constexpr auto kUBRatio = "ub_ratio";
constexpr auto kErrorInfo = "";
constexpr auto kErrorScope = "";
constexpr auto kAllocBits = "alloc_bits";

static std::unordered_map<std::string, int> help_tiling_level = {
  {"None", 0},
  {"General", 1},
  {"Candidates", 2},
  {"Tuning", 3},
};

void RecordCore(const int core, bool enable_file_log);
void RecordCore(const Stmt &stmt, bool enable_file_log);
void CreateDir(const std::string &file);

class AttrMap : public Map<std::string, NodeRef> {
 public:
  using Map<std::string, NodeRef>::operator=;

  bool GetBoolAttr(const std::string &attr_name, bool dft_value);
  int GetIntAttr(const std::string &attr_name, int dft_value);
  double GetFloatAttr(const std::string &attr_name, double dft_value);
  bool GetStringAttr(const std::string &attr_name, std::string *attr_to_set);
  std::string GetStringAttr(const std::string &attr_name, const std::string &dft_value);
};

class PassTimer {
 public:
  ~PassTimer() = default;

  void AddItem(const std::string &pass_name, int64_t elapsed_seconds);
  void Clear() { pass_time_.clear(); }
  std::string ToString() const;

  static PassTimer *GetInstance() {
    static PassTimer pass_timer;
    return &pass_timer;
  }

 private:
  PassTimer() { Clear(); }

  std::unordered_map<std::string, int64_t> pass_time_;
};

std::ostream &operator<<(std::ostream &os, const PassTimer &time);

std::string DumpC(const Stmt &stmt, const Array<Buffer> &extern_buffer);
}  // namespace akg

#endif  // CODEGEN_UTIL_H_
