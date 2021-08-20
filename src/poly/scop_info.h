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
#ifndef POLY_SCOP_INFO_H_
#define POLY_SCOP_INFO_H_

#include <tvm/ir.h>

#include "common/common_util.h"
#include "pass/convolution_model.h"
#include "poly/poly_util.h"
#include "poly/dynamic_shape.h"
#include "poly/tiling/custom_tiling.h"
#include "poly/dma_dataflow.h"
#include "poly/pass_info.h"
#include "poly/sync_manager.h"

namespace akg {
namespace ir {
namespace poly {

constexpr auto kReadSuffix = "read";
constexpr auto kWriteSuffix = "write";
constexpr auto kIterNamePrefix = "cc";
constexpr auto kGemmIterNamePrefix = "ee";

struct ParamInfo {
  std::string type_key;
  Expr key;
  Expr value;
};
struct TilingInfo {
  int tiling_flag;  // flag=1, tailing; flag=0, not tailing
  TileSizes dim_infos;
};
using Tiles = std::vector<TilingInfo>;

enum MappingType { NONE = -1, BLOCKS, THREADS, REPLACE_BLOCKS, REPLACE_THREADS };
struct MappingCfg {
 private:
  std::pair<std::string, int> x;
  std::pair<std::string, int> y;
  std::pair<std::string, int> z;
  std::vector<std::pair<std::string, int>> dim;

 public:
  MappingType type{NONE};
  size_t bound{0};
  size_t MaxDim() { return std::max(dim.size(), static_cast<size_t>(3)); }
  std::string GetPrefix(MappingType type) {
    CHECK_NE(type, MappingType::NONE);
    if (type == MappingType::BLOCKS || type == MappingType::REPLACE_BLOCKS) {
      return "b";
    } else {
      return "t";
    }
  }
  void BindFromStr(const std::string &cfg, const std::string &id_name = "", const bool enable_max_dim = true) {
    std::vector<std::string> res = common::Split(cfg, " ");
    if (enable_max_dim) {
      CHECK_LE(res.size(), MaxDim());
    }
    for (size_t i = 0; i < res.size(); ++i) {
      CHECK(!res[i].empty());
      auto size = static_cast<int>(std::strtol(res[i].c_str(), nullptr, 10));
      BindAt(i, size, id_name, enable_max_dim);
    }
  }
  void BindAt(size_t pos, int size, const std::string &id_name = "", const bool enable_max_dim = true) {
    if (enable_max_dim) {
      CHECK_LT(pos, MaxDim());
    }
    bound = std::max(bound, pos + 1);
    std::string id = "";
    if (!id_name.empty()) {
      id = REPLACE + id_name + "_";
    }
    id += GetPrefix(type) + std::to_string(pos);
    if (pos == 0) {
      x.first = id;
      x.second = size;
    } else if (pos == 1) {
      y.first = id;
      y.second = size;
    } else if (pos == 2) {
      z.first = id;
      z.second = size;
    }
    std::pair<std::string, int> dim_pos;
    dim_pos.first = id;
    dim_pos.second = size;
    dim.push_back(dim_pos);
  }
  std::pair<std::string, int> GetAt(std::string cfg_name) {
    std::pair<std::string, int> fixed_pos_cfg = {};
    if (cfg_name.find(T0) != std::string::npos || cfg_name.find(B0) != std::string::npos) {
      fixed_pos_cfg = GetX();
    } else if (cfg_name.find(T1) != std::string::npos || cfg_name.find(B1) != std::string::npos) {
      fixed_pos_cfg = GetY();
    } else if (cfg_name.find(T2) != std::string::npos || cfg_name.find(B2) != std::string::npos) {
      fixed_pos_cfg = GetZ();
    } else {
      LOG(FATAL) << "Mapping config can contain t0, t1, t2, b0, b1 and b2.";
    };
    return fixed_pos_cfg;
  }
  std::pair<std::string, int> GetAt(size_t pos) {
    if (pos == 0) {
      return GetX();
    } else if (pos == 1) {
      return GetY();
    } else if (pos == 2) {
      return GetZ();
    } else {
      CHECK_LT(pos, dim.size());
      auto res = dim[pos];
      res.second = res.second == 0 ? 1 : res.second;
      return res;
    }
  }
  std::pair<std::string, int> GetX() {
    auto res = x;
    res.second = res.second == 0 ? 1 : res.second;
    return res;
  }
  std::pair<std::string, int> GetY() {
    auto res = y;
    res.second = res.second == 0 ? 1 : res.second;
    return res;
  }
  std::pair<std::string, int> GetZ() {
    auto res = z;
    res.second = res.second == 0 ? 1 : res.second;
    return res;
  }
  void Reset() {
    bound = 0;
    x.second = 0;
    y.second = 0;
    z.second = 0;
    dim.clear();
  }
  void SwapConfig(size_t pos1, size_t pos2) {
    auto cfg1 = GetAt(pos1);
    auto cfg2 = GetAt(pos2);
    auto ReplaceSize = [this](size_t pos, int new_size) {
      if (pos == 0) {
        x.second = new_size;
      } else if (pos == 1) {
        y.second = new_size;
      } else {
        z.second = new_size;
      }
    };
    ReplaceSize(pos1, cfg2.second);
    ReplaceSize(pos2, cfg1.second);
  }
  void ModifySize(size_t pos, int size) {
    if (pos == 0) {
      x.second = size;
    } else if (pos == 1) {
      y.second = size;
    } else {
      z.second = size;
    }
  }
};

class TensorFootprintCluster;
struct BufferedFootPrintInfo {
  std::shared_ptr<TensorFootprintCluster> cluster;
  isl::union_map outer_schedule;
  isl::id cluster_id;
};
using Binds = Map<Tensor, Buffer>;
using TimeRecords = std::vector<std::string>;

class UserConfig {
 public:
  UserConfig() = default;
  ~UserConfig() = default;

  void SetTarget(const std::string target) {
    if (target == "aicore") {
      target_ = "cce";
    } else {
      target_ = target;
    }
  }

  void SetAttrs(const Map<std::string, NodeRef> &attrs) {
    if (attrs.empty()) return;
    ParseBoolAttr(attrs, "enable_restart", &enable_restart_);

    ParseDynamicShapeAttr(attrs, "dynamic_shape", &dynamic_shape_);
    ParseIntAttr(attrs, "dynamic_shape_bound", &dynamic_shape_bound_);
    ParseBoolAttr(attrs, "pragma_tilesize_is_var", &tile_size_is_var_);
    ParseBoolAttr(attrs, "pragma_outerband_need_split", &outer_band_need_split_);

    // Mind-trick pass
    ParseIntAttr(attrs, "constrain_schedule_verbosity", &constrain_schedule_verbosity_);
    ParseIntAttr(attrs, "enable_multicore", &enable_multicore_);
    ParseBoolAttr(attrs, "enable_mind_trick", &enable_mind_trick_);
    ParseBoolAttr(attrs, "enable_mind_trick_autogen", &enable_mind_trick_autogen_);
    ParseStringAttr(attrs, "mind_trick", &mind_trick_json_);
    ParseBoolAttr(attrs, "mind_trick_autogen_gpu_automap", &mind_trick_gpu_autogen_automap_);

    ParseStringAttr(attrs, "dim", &b_dim_);
    ParseStringAttr(attrs, "bind_elem_per_thread", &elem_per_thread_);
    ParseMappingCfgAttr(attrs, "bind_block", &block_cfg_);
    ParseMappingCfgAttr(attrs, "bind_thread", &thread_cfg_);
    ParseIntAttr(attrs, "max_elem_per_thread", &max_elem_per_thread_);

    ParseCustomTilingAttr(attrs, "custom_tiling", &custom_tiling_);
    ParseBoolAttr(attrs, "pragma_analyze_reuse_buffer", &pragma_analyze_reuse_buffer_);
    ParseBoolAttr(attrs, "pragma_speedup_tiling", &pragma_speedup_tiling_);
    ParseBoolAttr(attrs, "pragma_allow_tail_tiling", &pragma_allow_tail_tiling_);
    ParseBoolAttr(attrs, "pragma_analyze_multicore", &pragma_analyze_multicore_);
    ParseIntAttr(attrs, "prune_tuning_space_level", &prune_tuning_space_level_);
    ParseBoolAttr(attrs, "pragma_checkcoincident", &tile_check_coincident_);
    ParseIntAttr(attrs, "max_unroll_loop", &max_unroll_loop_);
    ParseBoolAttr(attrs, "unroll_shared", &unroll_shared_);

    ParseBoolAttr(attrs, "pragma_rmselfdep", &remove_self_dependence_);
    ParseBoolAttr(attrs, "pragma_force_rmselfdep", &force_remove_self_dependence_);
    ParseBoolAttr(attrs, "pragma_reschedule", &compute_reschedule_);
    ParseBoolAttr(attrs, "pragma_remove_invariant_dependence", &remove_invariant_dependence_);
    ParseBoolAttr(attrs, "pragma_disable_schedule_shift", &disable_schedule_shift_);
    ParseBoolAttr(attrs, "pragma_enable_schedule_max_constant", &enable_schedule_max_constant_);
    ParseBoolAttr(attrs, "pragma_disable_loop_reversal", &disable_loop_reversal_);
    ParseBoolAttr(attrs, "pragma_disable_loop_fusion", &disable_loop_fusion_);
    ParseBoolAttr(attrs, "pragma_reorder_schedule", &reorder_schedule_);
    ParseBoolAttr(attrs, "pragma_sink_last_axis", &sink_last_axis_);
    ParseBoolAttr(attrs, "pragma_keep_outer_band_order", &keep_outer_band_order_);
    ParseBoolAttr(attrs, "pragma_modshift", &mod_schedule_shift_);
    ParseBoolAttr(attrs, "pragma_disable_group", &disable_group_);
    ParseBoolAttr(attrs, "pragma_tile_inner_band", &tile_inner_band_);
    ParseBoolAttr(attrs, "pragma_set_all_coincident", &pragma_set_all_coincident_);

    ParseBoolAttr(attrs, "pragma_opt_for_dsa", &optimize_for_dsa_);
    ParseBoolAttr(attrs, "enable_feature_library", &enable_feature_library_);
    ParseBoolAttr(attrs, "enable_hoist_cond_write", &enable_hoist_cond_write_);

    ParseIntAttr(attrs, "kernel_h", &matB_dim_h_);
    ParseIntAttr(attrs, "kernel_w", &matB_dim_w_);
    ParseIntAttr(attrs, "bypassL1", &bypassL1_);
    ParseIntAttr(attrs, "isolated_idx", &isolated_idx_);
    ParseIntAttr(attrs, "conv_backprop_filter", &conv_back_prop_filter_);
    ParseBoolAttr(attrs, "pragma_conv_special_dma", &conv_special_dma_);
    ParseStringAttr(attrs, "kernel_name", &kernel_name_);
    ParseIntAttr(attrs, "pragma_is_conv", &pragma_is_conv_);
    ParseBoolAttr(attrs, "dynamic_shape_conv_full_parametric", &dynamic_shape_conv_full_parametric_);

    ParseIntAttr(attrs, "dump_tuning_level", &dump_tuning_level_);
    ParseBoolAttr(attrs, "dump_pass_ir", &dump_pass_ir_);
    ParseStringAttr(attrs, "dump_poly_dir", &dump_poly_dir_);

    ParseBoolAttr(attrs, "enable_atomic_add", &enable_atomic_add_);
    ParseBoolAttr(attrs, "use_new_space", &use_new_space_);

    if (GetTarget() == TARGET_CUDA) {
      ParseBoolAttr(attrs, "pragma_enable_tensor_core", &enable_tensor_core_);
      ParseBoolAttr(attrs, "pragma_enable_emit_core", &pragma_enable_emit_core_);
      ParseBoolAttr(attrs, "pragma_enable_conv_tensor_core", &enable_conv_tensor_core_);
      ParseBoolAttr(attrs, "pragma_enable_matmul", &enable_matmul_);
      ParseBoolAttr(attrs, "enable_tensor_core_use_poly", &enable_tensor_core_use_poly_);
      ParseBoolAttr(attrs, "enable_akg_reduce_lib", &enable_akg_reduce_lib_);
      ParseBoolAttr(attrs, "use_register_memory", &use_register_memory_);
      ParseBoolAttr(attrs, "use_shared_memory", &use_shared_memory_);
      ParseBoolAttr(attrs, "enable_bank_conflict_opt", &enable_bank_conflict_);
      ParseBoolAttr(attrs, "enable_one_dim_thread", &enable_one_dim_thread_);
      ParseBoolAttr(attrs, "shared_inversed_thread_map", &shared_inversed_thread_map_);
      ParseBoolAttr(attrs, "enable_stitch_fusion", &enable_stitch_fusion_);
      ParseIntAttr(attrs, "shared_vector_align", &shared_vector_align_);
      ParseIntAttr(attrs, "register_memory_depth", &register_depth_);
      ParseIntAttr(attrs, "shared_memory_depth", &shared_depth_);
      ParseStringAttr(attrs, "shared_memory_tensors", &shared_tensors_);
      ParseStringAttr(attrs, "reduce_lib_type", &reduce_lib_type_);
      ParseStringAttr(attrs, "local_memory_tensors", &local_tensors_);
      ParseVectorLoadTypeAttr(attrs, "vector_load_type", &vector_load_type_);
    }

    if (force_remove_self_dependence_) {
      LOG(WARNING) << "pragma_force_rmselfdep should be used with care. "
                   << "It removes all self dependence and cannot ensure that reduce axis do not use multicore.";
    }
  }

  std::string GetTarget() { return target_; }
  bool GetEnableRestart() { return enable_restart_; }
  void SetEnableRestart(bool enable_restart) { enable_restart_ = enable_restart; }
  // getter for dynamic shape config
  bool GetIsDynamic() const { return is_dynamic_; }
  std::vector<NodeRef> GetDynamicShape() { return dynamic_shape_; }
  int GetDynamicShapeBound() const { return dynamic_shape_bound_; }
  bool GetTileSizeIsVar() const { return tile_size_is_var_; }
  bool GetOuterBandNeedSplit() const { return outer_band_need_split_; }

  // getter for tiling config
  MappingCfg *GetBlockConfig() { return &block_cfg_; }
  MappingCfg *GetThreadConfig();
  std::unordered_map<std::string, MappingCfg *> GetReplaceConfig() { return replace_cfg_; }
  void FreeReplaceConfig() {
    std::unordered_map<std::string, MappingCfg *> empty_cfg;
    std::swap(this->replace_cfg_, empty_cfg);
  }
  void SetMaxElemPerThread(int max_elem_per_thread) { max_elem_per_thread_ = max_elem_per_thread; }
  int GetMaxElemPerThread() const { return max_elem_per_thread_; }
  void SetBlockConfig(const std::string &block_cfg) {
    this->block_cfg_.type = BLOCKS;
    this->block_cfg_.BindFromStr(block_cfg);
  }
  void SetThreadConfig(const std::string &thread_cfg);
  void RecordReplaceConfig(const std::string id, const std::string replace_cfg_str, const MappingType mapping_type,
                           const bool enable_max_dim = true) {
    MappingCfg *replace_cfg(new (std::nothrow) MappingCfg());
    CHECK(replace_cfg) << "memory alloc fail.";
    replace_cfg->type = mapping_type;
    replace_cfg->BindFromStr(replace_cfg_str, id, enable_max_dim);
    this->replace_cfg_[id] = replace_cfg;
  }
  void SetC0BlockSize(const std::vector<int> c0_block_size) { c0_block_size_ = c0_block_size; }
  std::vector<int> GetC0BlockSize() { return c0_block_size_; }
  std::vector<NodeRef> GetCustomTiling() { return custom_tiling_; }
  std::string GetBDim() const { return b_dim_; }
  std::string GetElemPerThread() const { return elem_per_thread_; }
  void SetDefaultDim(std::string b_dim) { b_dim_ = b_dim; }
  void SetPragmaSpeedUpTiling(bool pragma_speedup_tiling) { pragma_speedup_tiling_ = pragma_speedup_tiling; }
  bool GetPragmaSpeedUpTiling() const { return pragma_speedup_tiling_; }
  bool GetPragmaAnalyzeReuseBuffer() const { return pragma_analyze_reuse_buffer_; }
  bool GetPragmaAllowTailTiling() const { return pragma_allow_tail_tiling_; }
  bool GetPragmaAnalyzeMulticore() const { return pragma_analyze_multicore_; }
  int GetEnableMulticore() const { return enable_multicore_; }
  int GetPruneTuningSpaceLevel() const { return prune_tuning_space_level_; }
  bool GetTileCheckCoincident() const { return tile_check_coincident_; }
  void SetTileCheckCoincident(const bool tile_check_coincident) { tile_check_coincident_ = tile_check_coincident; }
  int GetMaxUnrollLoop() const { return max_unroll_loop_; }
  void SetUnroll(const int max_unroll_loop) { this->max_unroll_loop_ = max_unroll_loop; }
  bool GetUnrollShared() const { return unroll_shared_; }
  void SetUnrollShared(const bool unroll_shared) { this->unroll_shared_ = unroll_shared; }
  void SetDisableLoopFusion(const bool disable_loop_fusion) { this->disable_loop_fusion_ = disable_loop_fusion; }

  // getter/setter for schedule_pass config
  int GetConstrainScheduleVerbosity() const { return constrain_schedule_verbosity_; }
  void SetEnableMindTrick(bool status) { enable_mind_trick_ = status; }
  bool GetEnableMindTrick() const { return enable_mind_trick_; }
  void SetEnableMindTrickAutogen(bool status) { enable_mind_trick_autogen_ = status; }
  bool GetEnableMindTrickAutogen() const { return enable_mind_trick_autogen_; }
  std::string GetMindTrick() const { return mind_trick_json_; }
  std::string GetMindTrickStatus(void) const { return mind_trick_status_; }
  void SetMindTrickStatus(const std::string &status) { mind_trick_status_ = status; }
  void SetMindTrickWasUsed(bool used) { mind_trick_was_used_ = used; }
  bool GetMindTrickWasUsed() const { return mind_trick_was_used_; }
  void SetMindTrickGpuHasMapping(bool status) { mind_trick_gpu_has_mapping_ = status; }
  bool GetMindTrickGpuHasMapping(void) const { return mind_trick_gpu_has_mapping_; }
  void SetMindTrickGpuHasSwizzle(bool status) { mind_trick_gpu_has_swizzle_ = status; }
  bool GetMindTrickGpuHasSwizzle(void) const { return mind_trick_gpu_has_swizzle_; }
  void SetMindTrickGpuAutogenAutomap(bool status) { mind_trick_gpu_autogen_automap_ = status; }
  bool GetMindTrickGpuAutogenAutomap(void) const { return mind_trick_gpu_autogen_automap_; }

  // getter for schedule tree transform config
  bool GetRemoveSelfDependence() const { return remove_self_dependence_; }
  bool GetForceRemoveSelfDependence() const { return force_remove_self_dependence_; }
  bool GetRemoveInvariantDependence() const { return remove_invariant_dependence_; }
  bool GetComputeReschedule() const { return compute_reschedule_; }
  bool GetDisableScheduleShift() const { return disable_schedule_shift_; }
  bool GetEnableScheduleMaxConstant() const { return enable_schedule_max_constant_; }
  bool GetDisableLoopReversal() const { return disable_loop_reversal_; }
  bool GetDisableLoopFusion() const { return disable_loop_fusion_; }
  bool GetReorderSchedule() const { return reorder_schedule_; }
  bool GetSinkLastAxis() const { return sink_last_axis_; }
  bool GetKeepOuterBandOrder() const { return keep_outer_band_order_; }
  bool GetModScheduleShift() const { return mod_schedule_shift_; }
  bool GetDisableGroup() const { return disable_group_; }
  bool GetTileInnerBand() const { return tile_inner_band_; }
  bool GetPragmaSetAllCoincident() const { return pragma_set_all_coincident_; }
  bool GetConsiderCoincidence() const { return consider_conincidence_; }
  void SetConsiderCoincidence(bool consider_conincidence) { consider_conincidence_ = consider_conincidence; }
  void SetIsTuning(bool is_tuning) { is_tuning_ = is_tuning; }
  bool GetIsTuning() { return is_tuning_; }

  // getter for specialized optimization config
  bool GetOptimizeForNPU() const { return optimize_for_dsa_; }
  bool GetEnableFeatureLib() const { return enable_feature_library_; }
  bool GetEnableHoistCondWrite() const { return enable_hoist_cond_write_; }

  // getter for conv config
  int GetMatBDimH() const { return matB_dim_h_; }
  int GetMatBDimW() const { return matB_dim_w_; }
  int GetByPathC1() const { return bypassL1_; }
  int GetIsolatedIdx() const { return isolated_idx_; }
  std::string GetKernelName() { return kernel_name_; }
  int GetPragmaIsConv() const { return pragma_is_conv_; }
  int GetConvBackPropFilter() const { return conv_back_prop_filter_; }
  bool GetConvSpecialDma() const { return conv_special_dma_; }
  bool GetDynamicShapeConvFullParametric() const { return dynamic_shape_conv_full_parametric_; }
  Schedule GetScheduleInfo() const { return origin_sch_; }

  // getter for dump config
  int GetDumpTuningLevel() const { return dump_tuning_level_; }
  bool GetDumpPassIr() const { return dump_pass_ir_; }
  std::string GetDumpPolyDir() { return dump_poly_dir_; }

  // setter for conv config
  void SetMatBDimH(int matB_dim_h) { this->matB_dim_h_ = matB_dim_h; }
  void SetMatBDimW(int matB_dim_w) { this->matB_dim_w_ = matB_dim_w; }
  void SetByPassL1(int by_passL1) { this->bypassL1_ = by_passL1; }
  void SetIsolatedIdx(int isolated_idx) { this->isolated_idx_ = isolated_idx; }
  void SetDynamic(bool is_dynamic) { this->is_dynamic_ = is_dynamic; }

  void SetScheduleInfo(const Schedule &sch) { this->origin_sch_ = sch; }

  std::vector<Stmt> GetOuterLetStmts() { return outer_let_stmts_; }
  void SetOuterLetStmts(std::vector<Stmt> &outer_let_stmts) { outer_let_stmts_ = outer_let_stmts; }
  std::unordered_set<isl::id, isl::IslIdIslHash> GetRealizeFromInput() { return realize_from_input_; }
  void InsertRealizeFromInput(const isl::id &id) { realize_from_input_.insert(id); }
  void SetOriginBind(const Binds &binds_orig) { binds_orig_ = binds_orig; }
  Binds GetOriginBind() const { return binds_orig_; }
  void RecordRealizeTensors(const Tensor &t) { realize_tensors_.emplace(t); }
  std::unordered_set<Tensor> GetRealizeTensors() const { return realize_tensors_; }
  void SetBind(const Tensor &t, const Buffer &buf) { binds_.Set(t, buf); }
  void SetBind(const Binds &binds) { binds_ = binds; }
  Binds GetBind() const { return binds_; }
  Binds &GetBind() { return binds_; }
  Stmt GetBody() const { return body_; }
  void SetBody(const Stmt &body) { body_ = body; }
  std::unordered_map<std::string, Var> GetParams() const { return params_; }
  std::unordered_map<std::string, Expr> GetParamsRevMap() { return params_rev_map_; }
  std::map<int64_t, Expr> GetParamTilingMap() { return param_tiling_map_; }
  void SetParamTilingMap(const std::map<int64_t, Expr> &param_tiling_map) { param_tiling_map_ = param_tiling_map; }
  void RegisterParam(const Expr &expr);
  void CollectParams();
  std::string GetIterPrefix(bool is_spec_gemm = false) const {
    return is_spec_gemm ? kGemmIterNamePrefix : kIterNamePrefix;
  }
  int GetDataBytes(const std::string &name) const;

  // dump all info
  void DumpScopDataScheduleAttrs(std::ofstream &of);

  bool GetUseNewSpace() { return use_new_space_; }

  bool GetEnableAtomicAdd() { return enable_atomic_add_; }

  bool GetEnableAkgReduceLib() { return enable_akg_reduce_lib_; }
  void SetEnableAkgReduceLib(bool enable_akg_reduce_lib) { enable_akg_reduce_lib_ = enable_akg_reduce_lib; }

  // tensor core info
  bool GetEnableMatmul() { return enable_matmul_; }
  void SetEnableMatmul(bool enable_matmul) { enable_matmul_ = enable_matmul; }

  bool GetEnableTensorCore() {
    SetEnableTensorCore(enable_tensor_core_);
    return enable_tensor_core_;
  }
  void SetEnableTensorCore(bool enable_tensor_core) { enable_tensor_core_ = enable_matmul_ && enable_tensor_core; }

  bool GetEnableEmitCore() { return pragma_enable_emit_core_; }
  void SetEnableEmitCore(bool pragma_enable_emit_core) { pragma_enable_emit_core_ = pragma_enable_emit_core; }

  bool GetEnableTensorCoreUsePoly() {
    SetEnableTensorCoreUsePoly(enable_tensor_core_use_poly_);
    return enable_tensor_core_use_poly_;
  }
  void SetEnableTensorCoreUsePoly(bool enable_tensor_core_use_poly) {
    enable_tensor_core_use_poly_ = enable_tensor_core_ && enable_tensor_core_use_poly;
  }

  bool GetEnableConvTensorCore() { return enable_conv_tensor_core_; }
  void SetEnableConvTensorCore(bool enable_conv_tensor_core) { enable_conv_tensor_core_ = enable_conv_tensor_core; }

  bool GetEnableOneDimThread() { return enable_one_dim_thread_; }
  void SetEnableOneDimThread(bool enable_one_dim_thread) { enable_one_dim_thread_ = enable_one_dim_thread; }

  std::unordered_map<int, std::string> GetCustomInnerMapping() { return custom_inner_mapping_; }
  void RecordCustomInnerMapping(const int axis_dim, const std::string inner_mapping) {
    custom_inner_mapping_[axis_dim] = inner_mapping;
  }

  std::unordered_map<int, std::string> GetCustomOuterMapping() { return custom_outer_mapping_; }
  void RecordCustomOuterMapping(const int axis_dim, const std::string outer_mapping) {
    custom_outer_mapping_[axis_dim] = outer_mapping;
  }
  bool GetEnableVectorization() { return enable_vectorization_; }
  void SetEnableVectorization(bool enable_vectorization) { enable_vectorization_ = enable_vectorization; }

  bool UseRegisterMemory() { return use_register_memory_; }
  bool UseSharedMemory() { return use_shared_memory_; }
  void SetUseSharedMemory(bool use_shared_memory) { use_shared_memory_ = use_shared_memory; }
  void SetUseRegisterMemory(bool use_register_memory) { use_register_memory_ = use_register_memory; }
  int GetRegisterDepth() { return register_depth_; }
  int GetSharedDepth() { return shared_depth_; }
  void SetSharedTensors(std::string shared_tensors) { shared_tensors_ = shared_tensors; }
  std::string GetSharedTensors() { return shared_tensors_; }
  std::string GetReduceLibType() { return reduce_lib_type_; }
  std::string GetLocalTensors() { return local_tensors_; }
  void SetEnableBankConflict(bool enable_bank_conflict) { enable_bank_conflict_ = enable_bank_conflict; }
  bool GetEnableBankConflict() { return enable_bank_conflict_; }
  int GetVectorLoadType() { return vector_load_type_; }
  void SetVectorLoadType(int vector_load_type) { vector_load_type_ = vector_load_type; }
  void SetSharedInversedThreadMap(bool shared_inversed_thread_map) {
    shared_inversed_thread_map_ = shared_inversed_thread_map;
  }
  bool GetSharedInversedThreadMap() { return shared_inversed_thread_map_; }
  bool EnableStitchFusion() { return enable_stitch_fusion_; }
  void SetSharedVectorAlign(int shared_vector_align) { shared_vector_align_ = shared_vector_align; }
  int GetSharedVectorAlign() { return shared_vector_align_; }
  void SetTransposeOp(bool has_transpose) { has_transpose_ = has_transpose; }
  bool HasTranspose() { return has_transpose_; }

 private:
  // tools for parsing user config
  static void ParseIntAttr(const Map<std::string, NodeRef> &attrs, const std::string &attr_name, int *attr_to_set) {
    CHECK(attr_to_set != nullptr);
    if (attrs.count(attr_name) == 0) return;
    const NodeRef &e = attrs.at(attr_name);
    if (auto i = e.as<IntImm>()) {
      *attr_to_set = static_cast<int>(i->value);
    } else if (auto ui = e.as<UIntImm>()) {
      *attr_to_set = static_cast<int>(ui->value);
    } else {
      LOG(FATAL) << "Failed to parse attribute: " << attr_name << " = " << e << " as integer";
    }
  }

  static void ParseBoolAttr(const Map<std::string, NodeRef> &attrs, const std::string &attr_name, bool *attr_to_set) {
    const int invalid_value = -1;
    int attr = invalid_value;
    ParseIntAttr(attrs, attr_name, &attr);
    if (attr != invalid_value) {
      CHECK(attr == 0 || attr == 1) << "Bool attribute " << attr_name << " must be 0 or 1, but found "
                                    << attrs.at(attr_name);
      *attr_to_set = static_cast<bool>(attr);
    }
  }

  static void ParseStringAttr(const Map<std::string, NodeRef> &attrs, const std::string &attr_name,
                              std::string *attr_to_set) {
    CHECK(attr_to_set != nullptr);
    if (attrs.count(attr_name) == 0) return;
    const NodeRef &e = attrs.at(attr_name);
    if (auto val = e.as<StringImm>()) {
      *attr_to_set = val->value;
    } else {
      LOG(FATAL) << "Failed to parse attribute: " << attr_name << " = " << e << " as string";
    }
  }

  static void ParseMappingCfgAttr(const Map<std::string, NodeRef> &attrs, const std::string &attr_name,
                                  MappingCfg *attr_to_set) {
    std::string str_cfg = "";
    ParseStringAttr(attrs, attr_name, &str_cfg);
    attr_to_set->type = attr_name == "bind_block" ? BLOCKS : THREADS;
    attr_to_set->BindFromStr(str_cfg);
  }

  static void ParseCustomTilingAttr(const Map<std::string, NodeRef> &attrs, const std::string &attr_name,
                                    std::vector<NodeRef> *attr_to_set) {
    CHECK(attr_to_set != nullptr);
    if (attrs.count(attr_name) == 0) return;
    const NodeRef &e = attrs.at(attr_name);
    Array<NodeRef> array = air::runtime::Downcast<Array<NodeRef>>(e);
    for (auto d : array) {
      if (d.as<air::CustomTilingNode>()) {
        attr_to_set->emplace_back(d);
      } else {
        LOG(FATAL) << "Failed to parse attribute: " << attr_name << " = " << e << " as CustomTilingNode";
      }
    }
  }

  static void ParseDynamicShapeAttr(const Map<std::string, NodeRef> &attrs, const std::string &attr_name,
                                    std::vector<NodeRef> *attr_to_set) {
    CHECK(attr_to_set != nullptr);
    if (attrs.count(attr_name) == 0) return;
    const NodeRef &e = attrs.at(attr_name);
    Array<NodeRef> array = air::runtime::Downcast<Array<NodeRef>>(e);
    for (auto d : array) {
      if (d.as<air::DynamicShapeNode>()) {
        attr_to_set->emplace_back(d);
      } else {
        LOG(FATAL) << "Failed to parse attribute: " << attr_name << " = " << e << " as DynamicShapeNode";
      }
    }
  }

  static void ParseVectorLoadTypeAttr(const Map<std::string, NodeRef> &attrs, const std::string &attr_name,
                                      int *attr_to_set) {
    std::string str_cfg = "";
    ParseStringAttr(attrs, attr_name, &str_cfg);
    // Vectorization only supports float1/float2/float3/float4
    std::string support_type = "float";
    if (str_cfg.size() != support_type.size() + 1 || str_cfg.find(support_type) != 0) {
      return;
    }
    int type_num = std::stoi(str_cfg.substr(support_type.size()));
    if (type_num < 1 || type_num > 4) {
      return;
    }
    *attr_to_set = type_num * (Float(32).bits());
  }

 private:
  isl::ctx ctx_{isl_ctx_alloc()};
  std::string target_;
  std::vector<Stmt> outer_let_stmts_;
  std::unordered_set<isl::id, isl::IslIdIslHash> realize_from_input_;
  Binds binds_orig_;
  Binds binds_;
  std::unordered_set<Tensor> realize_tensors_;
  Stmt body_;
  std::unordered_map<std::string, Var> params_;
  std::unordered_map<std::string, Expr> params_rev_map_;
  std::map<int64_t, Expr> param_tiling_map_;
  bool enable_restart_{false};
  bool is_spec_gemm_{false};

  // dynamic shape config
  bool is_dynamic_{false};
  std::vector<NodeRef> dynamic_shape_;
  int dynamic_shape_bound_{0};
  bool tile_size_is_var_{false};
  bool outer_band_need_split_{false};

  bool enable_atomic_add_{false};
  bool use_new_space_{false};
  // tensor_core config
  bool enable_matmul_{false};
  bool enable_tensor_core_{false};
  bool pragma_enable_emit_core_{true};
  bool enable_tensor_core_use_poly_{false};
  // conv config
  bool enable_conv_tensor_core_{false};
  // lib config
  bool enable_akg_reduce_lib_{true};
  // memory config
  bool use_register_memory_{true};
  bool use_shared_memory_{true};
  // shared memory position in schedule tree
  int register_depth_{-1};
  int shared_depth_{-1};
  // shared memory tensor list
  std::string shared_tensors_;
  // reduce lib type, for now, there are two selection
  // one is named "origin"
  // one is named "paris"
  std::string reduce_lib_type_{"origin"};
  // local memory tensor list
  std::string local_tensors_;
  // vectorization
  int vector_load_type_{0};
  bool enable_one_dim_thread_{false};
  bool enable_vectorization_{false};

  // tiling config
  std::string b_dim_;
  MappingCfg block_cfg_;
  MappingCfg thread_cfg_;
  std::unordered_map<std::string, MappingCfg *> replace_cfg_;
  std::vector<int> c0_block_size_;
  int max_elem_per_thread_{1024};
  std::string elem_per_thread_;
  std::vector<NodeRef> custom_tiling_;
  bool pragma_analyze_reuse_buffer_{true};
  bool pragma_speedup_tiling_{false};
  bool pragma_allow_tail_tiling_{true};
  bool pragma_analyze_multicore_{true};
  int enable_multicore_{-1};
  int prune_tuning_space_level_{0};  // 0: no_prune; 1: prune mem-exceed; 2: prune aligned_mem-exceed
  bool tile_check_coincident_{true};
  int max_unroll_loop_{1};
  bool unroll_shared_{false};
  bool enable_bank_conflict_{false};
  bool shared_inversed_thread_map_{false};
  bool enable_stitch_fusion_{false};
  int shared_vector_align_{0};

  // mind_trick config
  int constrain_schedule_verbosity_{-1};
  bool enable_mind_trick_{true};
  bool enable_mind_trick_autogen_{false};
  std::string mind_trick_json_{""};
  std::string mind_trick_status_{"null"};
  bool mind_trick_was_used_{false};
  bool mind_trick_gpu_has_mapping_{false};
  bool mind_trick_gpu_has_swizzle_{false};
  bool mind_trick_gpu_autogen_automap_{true};

  // schedule tree transform config
  bool remove_self_dependence_{true};
  bool force_remove_self_dependence_{false};
  bool remove_invariant_dependence_{true};
  bool compute_reschedule_{false};
  bool disable_schedule_shift_{false};
  bool enable_schedule_max_constant_{false};
  bool disable_loop_reversal_{false};
  bool disable_loop_fusion_{false};
  bool reorder_schedule_{false};
  bool sink_last_axis_{true};
  bool keep_outer_band_order_{false};
  bool mod_schedule_shift_{false};
  bool disable_group_{false};
  bool tile_inner_band_{false};
  bool pragma_set_all_coincident_{false};
  bool consider_conincidence_{true};
  bool is_tuning_{false};

  // specialized optimization
  bool optimize_for_dsa_{false};
  bool enable_feature_library_{false};
  bool enable_hoist_cond_write_{true};

  // conv config
  int matB_dim_h_{-1};
  int matB_dim_w_{-1};
  int bypassL1_{0};
  int isolated_idx_{0};
  std::string kernel_name_;
  int pragma_is_conv_{0};
  int conv_back_prop_filter_{0};
  bool conv_special_dma_{false};
  bool dynamic_shape_conv_full_parametric_{false};

  // custom mapping config
  std::unordered_map<int, std::string> custom_inner_mapping_;
  std::unordered_map<int, std::string> custom_outer_mapping_;

  // dump config
  int dump_tuning_level_{0};
  bool dump_pass_ir_{false};
  std::string dump_poly_dir_;

  Schedule origin_sch_;

  bool has_transpose_{false};
};

struct OperatorDomainSpace {
  isl::space param_space;
  isl::multi_id tuple;
};

using AccessMap = std::unordered_map<const Node *, isl::id>;
using StatementMap = std::unordered_map<isl::id, const Node *, isl::IslIdIslHash>;
using OperatorDomainMap = std::unordered_map<isl::id, OperatorDomainSpace, isl::IslIdIslHash>;
using ReduceMap = std::unordered_map<const Provide *, Array<IterVar>>;
using CondVarsMap = std::unordered_map<isl::id, std::unordered_set<std::string>, isl::IslIdIslHash>;
using BufferBindVec = std::vector<std::pair<const NodeRef, const Expr>>;

using Mapping = std::unordered_map<isl::id, isl::union_pw_aff, isl::IslIdIslHash>;
using UpaNodeMapping = std::vector<std::pair<isl::schedule_node, Mapping>>;
struct AtomicInfo {
  std::string tensor_name;
  std::string tensor_type;
};

struct StatementUnionMappingInfo {
  std::vector<isl::id> stmt_vec;
  bool inject_mapping;
  bool biject_mapping;
};

using TensorScheduleRepo = std::unordered_map<std::string, StatementUnionMappingInfo>;

struct ReduceTensorInfo {
  isl::union_map stmt_map;
  const Node *stmt_node;
  std::string write_tensor_name;
  Expr init_value;
  Type write_dtype;
  std::vector<std::string> axis_vec;
};

using ReduceTensorInfoMap = std::unordered_map<isl::id, ReduceTensorInfo, isl::IslIdIslHash>;

struct Mma {
  int64_t m;
  int64_t n;
  int64_t k;
};

struct MmaConv {
  int64_t m;
  int64_t h;
  int64_t w;
  int64_t n;
  int64_t k;
};

class AnalysisResult {
 public:
  AnalysisResult() = default;
  ~AnalysisResult() = default;

  void RecordWrites(const isl::union_map &writes) { writes_ = writes; }
  void RecordReads(const isl::union_map &reads) { reads_ = reads; }

  void RecordBindCopyin(const isl::union_map &bind_copyin) { bind_copyin_ = bind_copyin; }
  void RecordCopyin(const isl::union_map &copyin) { copyin_ = copyin; }
  void RecordFakeCopyin(const isl::union_map &fake_copyin) { fake_copyin_ = fake_copyin; }
  void RecordTransferStmt(const isl::union_set &transfer_stmt) { transfer_stmt_ = transfer_stmt; }
  void RecordInnerBandDependency(const isl::union_map &inter_band_dependency) {
    inter_band_dependency_ = inter_band_dependency;
  }

  void RecordAccess(const Node *node, const isl::id &tensor_id) { accesses_.emplace(node, tensor_id); }

  void RecordStatement(const isl::id &tensor_id, const Node *node) { statements_.emplace(tensor_id, node); }
  void RecordStmtOpInfo(const isl::id &tensor_id, const StmtOpInfo &op_info) {
    stmt_op_Info_.emplace(tensor_id, op_info);
  }
  void RecordOperatorDomain(const isl::id &tensor_id, const OperatorDomainSpace &dom_space) {
    domains_.emplace(tensor_id, dom_space);
  }
  void RecordBufferBindVec(const std::pair<const NodeRef, const Expr> &buf_bind) { buf_bind_vec_.push_back(buf_bind); }
  void RecordUpdateTensor(const Tensor &tensor) { update_tensors_.push_back(tensor); }
  void RecordAttrStmt(const AttrStmt *attr_stmt) { attr_stmts_.push_back(attr_stmt); }
  void RecordAtomicTensors(const AtomicInfo &atomic_info) { atomic_tensors_.push_back(atomic_info); }
  void RecordAtomicMarkers(const std::string &marker_name) { atomic_markers_.insert(marker_name); }
  void RecordReduceOutTensors(const std::string &tensor_name) { reduce_out_tensors_.insert(tensor_name); }
  void RecordContextParams(const isl::set &context_params) { context_params_ = context_params; }
  void RecordMatrixMatmulMap(const std::string matrix_name, const std::string matrix_position) {
    matrix_matmul_map_.emplace(matrix_name, matrix_position);
  }
  void RecordCastTensors(const std::string tensor_name) { cast_tensors_.insert(tensor_name); }
  void RecordSharedTensorBitsMap(const std::string tensor_name, const int tensor_bits) {
    shared_tensor_bits_map_.emplace(tensor_name, tensor_bits);
  }
  std::unordered_map<std::string, int> GetSharedTensorBitsMap() const { return shared_tensor_bits_map_; }
  void RecordMatrixMatmulMajor(const std::string matrix_name, const std::string matrix_major) {
    matrix_matmul_major_[matrix_name] = matrix_major;
  }
  void SetMmaMode(Mma mma) { mma_ = mma; }
  std::unordered_map<std::string, std::string> GetMatrixMatmulMap() const { return matrix_matmul_map_; }
  std::unordered_map<std::string, std::string> GetMatrixMatmulMajor() const { return matrix_matmul_major_; }
  Mma GetMmaMode() const { return mma_; }
  std::unordered_set<std::string> GetCastTensors() const { return cast_tensors_; }
  isl::set GetContextParams() { return context_params_; }
  std::vector<AtomicInfo> GetAtomicTensors() { return atomic_tensors_; }
  std::unordered_set<std::string> GetAtomicMarkers() { return atomic_markers_; }
  std::unordered_set<std::string> GetReduceOutTensors() { return reduce_out_tensors_; }
  isl::union_map GetReads() const { return reads_; }
  isl::union_map &GetWrites() { return writes_; }
  isl::union_map GetWrites() const { return writes_; }
  isl::union_map &GetCopyin() { return copyin_; }
  isl::union_map GetCopyin() const { return copyin_; }
  isl::union_map GetBindCopyin() const { return bind_copyin_; }
  isl::union_map GetFakeCopyin() const { return fake_copyin_; }
  isl::union_set GetTransferStmt() const { return transfer_stmt_; }
  isl::union_map GetInnerBandDependency() const { return inter_band_dependency_; }

  AccessMap &GetAccessMap() { return accesses_; }
  StatementMap &GetStatementMap() { return statements_; }
  StatementMap GetStatementMap() const { return statements_; }
  StmtOpInfoMap &GetStmtOpInfoMap() { return stmt_op_Info_; }
  StmtOpInfoMap GetStmtOpInfoMap() const { return stmt_op_Info_; }
  OperatorDomainMap &GetOperatorDomainMap() { return domains_; }
  OperatorDomainMap GetOperatorDomainMap() const { return domains_; }
  CondVarsMap GetCondVarsMap();
  BufferBindVec GetBufferBindVec() const { return buf_bind_vec_; }
  std::vector<Tensor> GetUpdateTensor() const { return update_tensors_; }
  std::vector<const AttrStmt *> GetAttrStmt() const { return attr_stmts_; }
  std::map<std::string, std::vector<std::string>> &GetTensorNameFlows() { return tensor_name_flows_; }
  void SetTensorNameFlows(const std::map<std::string, std::vector<std::string>> &tensor_name_flows) {
    tensor_name_flows_ = tensor_name_flows;
  }
  std::map<std::string, MemFlow> GetTensorMemFlows() { return tensor_mem_flows_; }
  void SetTensorMemFlows(std::map<std::string, MemFlow> &tensor_mem_flows) { tensor_mem_flows_ = tensor_mem_flows; }
  std::unordered_set<std::string> GetConditionalWriteBufferFootprints() { return conditional_write_buffer_footprints_; }
  void InsertConditionalWriteBufferFootprints(const std::string &s) { conditional_write_buffer_footprints_.insert(s); }
  bool GetIsTiled() const { return is_tiled_; }
  void SetIsTiled(bool is_tiled) { is_tiled_ = is_tiled; }
  bool GetIsCustomMapping() const { return is_custom_mapping_; }
  void SetIsCustomMapping(bool is_custom_mapping) { is_custom_mapping_ = is_custom_mapping; }
  bool GetIsOuterBlockMapping() const { return is_outer_block_mapping_; }
  void SetIsOuterBlockMapping(bool is_outer_block_mapping) { is_outer_block_mapping_ = is_outer_block_mapping; }
  bool GetIsGpuDmaAnalysed() const { return is_gpu_dma_analysed_; }
  void SetIsGpuDmaAnalysed(bool is_gpu_dma_analysed) { is_gpu_dma_analysed_ = is_gpu_dma_analysed; }
  void SetScheduleMapBeforeTile(const isl::union_map &schedule_map_before_tile) {
    schedule_map_before_tile_ = schedule_map_before_tile;
  }
  void InitScheduleMapBeforeTile(const isl::ctx &ctx) { schedule_map_before_tile_ = isl::union_map::empty(ctx); }
  const isl::union_map &GetScheduleMapBeforeTile() { return schedule_map_before_tile_; }
  void SetTransformedSchedule(const isl::schedule &transformed_schedule) {
    transformed_schedule_ = transformed_schedule;
  }
  isl::union_set Domain() const { return transformed_schedule_.get_domain(); }

  TileSizes &GetTileSizes() { return tile_sizes_; }
  TileSizes GetTileSizes() const { return tile_sizes_; }
  void SetTileSizes(TileSizes tile_size) { tile_sizes_ = std::move(tile_size); }
  void InsertDimensionInfo(const DimensionInfo &dim_info) { tile_sizes_.emplace_back(dim_info); }

  std::deque<ParamInfo> GetTileConstraints() { return tiling_constraints_; }
  void SetTileConstraints(std::deque<ParamInfo> tiling_constraints) {
    tiling_constraints_ = std::move(tiling_constraints);
  }

  std::map<std::string, std::string> GetTensorOfTensorStmt() const { return tensor_of_tensor_stmt_; }
  void RecordTensorOfTensorStmt(const std::string &id_name, const std::string &op_type) {
    tensor_of_tensor_stmt_[id_name] = op_type;
  }

  bool GetTensorOfTensor() const { return is_tensor_of_tensor_; }
  void SetTensorOfTensor(const bool &is_tensor_of_tensor) { is_tensor_of_tensor_ = is_tensor_of_tensor; }

  int GetLastAxisInScheduleTree() const { return last_axis_in_schedule_tree_; }
  void SetLastAxisInScheduleTree(const int last_axis_in_schedule_tree) {
    last_axis_in_schedule_tree_ = last_axis_in_schedule_tree;
  }

  std::unordered_set<std::string> GetTensorsNotPromote() const { return tensors_not_promote_; }
  void RecordTensorsNotPromote(const std::string &tensor_name) { tensors_not_promote_.insert(tensor_name); }

  std::unordered_set<std::string> GetInnerTensor() const { return inner_tensor_; }
  void RecordInnerTensor(const std::string &tensor_name) { inner_tensor_.insert(tensor_name); }

  // dump all data
  void DumpScopDataBasics(std::ofstream &of);

  int CountBufferDefInfo(const isl::id &tensor_id) const;
  const std::vector<BufferDefInfo> &BufferDefInfos() const { return buffer_def_infos_; }
  const BufferDefInfo &GetBufferDefInfo(const isl::id &tensor_id, const bool is_dst_tensor_id = true) const;
  bool HasBufferDefInfo(const isl::id &tensor_id) const;
  const std::vector<std::pair<isl::union_set, BufferedFootPrintInfo>> &ActiveBufferFootprints() const {
    return active_buffer_footprints_;
  }
  void DumpBufferDefInfos(std::ostream &out);
  std::unordered_set<std::string> ExtractWithStmtId() const;

  // Akg Reduce utils
  void RecordReduce(const Provide *node, const Array<IterVar> &reduces) { reduces_.emplace(node, reduces); }
  ReduceMap &GetReduceMap() { return reduces_; }
  ReduceMap GetReduceMap() const { return reduces_; }

  void RecordReduceAxisForMatmul(const std::vector<const Variable *> &reduce_axis) {
    reduce_axis_ = std::move(reduce_axis);
  }
  std::vector<const Variable *> GetReduceAxisForMatmul() const { return reduce_axis_; }

  void RecordNotReduceAxisForMatmul(const std::vector<const Variable *> &not_reduce_axis) {
    not_reduce_axis_ = std::move(not_reduce_axis);
  }
  std::vector<const Variable *> GetNotReduceAxisForMatmul() const { return not_reduce_axis_; }

  void RecordBatchAxisNumForMatmul(const unsigned int &batch_axis_num) { batch_axis_num_ = std::move(batch_axis_num); }
  unsigned int GetBatchAxisNumForMatmul() const { return batch_axis_num_; }

  void RecordReduceDirection(const std::string reduce_direction) { reduce_direction_ = reduce_direction; }
  std::string GetReduceDirection() const { return reduce_direction_; }

  void RecordReduceInitIds(isl::id reduce_init_id) { reduce_init_ids_.push_back(reduce_init_id); }
  std::vector<isl::id> GetReduceInitIds() const { return reduce_init_ids_; }

  ReduceTensorInfoMap GetReduceTensorInfoMap() const { return reduce_tensor_info_; }
  void RecordReduceTensorInfoMap(const isl::id id, const ReduceTensorInfo &reduceinfo) {
    reduce_tensor_info_.emplace(id, reduceinfo);
  }
  void UpdateReduceTensorInfoMap(const isl::id id, const ReduceTensorInfo &reduceinfo) {
    reduce_tensor_info_[id] = reduceinfo;
  }

  bool IsPureReduceSum(const Add *add, const std::string &prov_func_name);
  isl::union_map GetReduceWriteStmt(const isl::schedule_node_band &band);
  std::string GetReduceOpType(isl::id reduce_stmt);
  bool IsReduceInitStmt(const isl::id id) const;

  bool GetEnabledAutoTiling() const { return enabled_auto_tiling_; }
  void SetEnableAutoTiling(bool enabled_auto_tiling) { enabled_auto_tiling_ = enabled_auto_tiling; }

  TensorScheduleRepo GetTensorScheduleRepo() const { return tensor_schedule_repo_; }
  void SetTensorScheduleRepo(const TensorScheduleRepo &repo) { tensor_schedule_repo_ = std::move(repo); }

  bool IsFakeCopyin(const isl::id &tensor_id);

 public:
  std::vector<std::pair<std::string, STMT_OP_TYPE>> stmt_type_;
  std::vector<std::pair<isl::union_set, BufferedFootPrintInfo>> active_buffer_footprints_;
  std::vector<BufferDefInfo> buffer_def_infos_;
  BufferDefInfo default_buffer_def_info_;

 private:
  ReduceMap reduces_;
  ReduceTensorInfoMap reduce_tensor_info_;
  std::string reduce_direction_;
  std::vector<isl::id> reduce_init_ids_;
  std::unordered_set<std::string> reduce_attrs_;
  std::unordered_set<std::string> not_reduce_attrs_;
  std::vector<const Variable *> reduce_axis_;
  std::vector<const Variable *> not_reduce_axis_;
  unsigned int batch_axis_num_;

  isl::union_map reads_;
  isl::union_map writes_;
  isl::union_map bind_copyin_;
  isl::union_map copyin_;
  isl::union_map fake_copyin_;
  isl::union_set transfer_stmt_;
  isl::union_map inter_band_dependency_;
  AccessMap accesses_;
  StatementMap statements_;
  StmtOpInfoMap stmt_op_Info_;
  OperatorDomainMap domains_;
  BufferBindVec buf_bind_vec_;
  std::vector<Tensor> update_tensors_;
  std::vector<const AttrStmt *> attr_stmts_;

  std::map<std::string, std::vector<std::string>> tensor_name_flows_;
  std::map<std::string, MemFlow> tensor_mem_flows_;
  std::unordered_set<std::string> conditional_write_buffer_footprints_;

  std::deque<ParamInfo> tiling_constraints_;
  TileSizes tile_sizes_;
  bool is_tiled_{false};
  bool is_gpu_dma_analysed_{false};
  isl::union_map schedule_map_before_tile_;  // before tiling, after ungroup.
  isl::schedule transformed_schedule_;
  isl::set context_params_;

  std::vector<AtomicInfo> atomic_tensors_;
  std::unordered_set<std::string> atomic_markers_;
  std::unordered_set<std::string> reduce_out_tensors_;
  std::unordered_set<std::string> cast_tensors_;
  bool enabled_auto_tiling_{false};
  std::unordered_map<std::string, std::string> matrix_matmul_map_;
  std::unordered_map<std::string, int> shared_tensor_bits_map_;
  TensorScheduleRepo tensor_schedule_repo_;
  std::unordered_map<std::string, std::string> matrix_matmul_major_;
  Mma mma_;

  // custom mapping
  bool is_custom_mapping_{false};
  bool is_outer_block_mapping_{false};

  // All axis of each tensor
  std::unordered_map<std::string, std::vector<std::string>> tensor_all_axis_;
  // tensor_of_tensor
  std::map<std::string, std::string> tensor_of_tensor_stmt_;
  std::unordered_set<std::string> tensors_not_promote_;
  std::unordered_set<std::string> inner_tensor_;
  int last_axis_in_schedule_tree_{-1};
  bool is_tensor_of_tensor_{false};
};

class CubeInfo {
 public:
  CubeInfo(UserConfig &user_config, AnalysisResult &analysis_result)
      : user_config_(user_config), analysis_result_(analysis_result){};
  ~CubeInfo();

  void SetAttrs(const Map<std::string, NodeRef> &attrs) {
    for (auto iter : attrs) {
      if (iter.first == ATTR_CONV_GMM_FEATURE || iter.first == ATTR_CONV_GMM_WEIGHT ||
          iter.first == ATTR_GEMM_DATA_TRANSPOSE || iter.first == ATTR_GEMM_WEIGHT_TRANSPOSE ||
          iter.first == ATTR_GEMM_DATA_TRANSPOSE_BLOCK || iter.first == ATTR_GEMM_WEIGHT_TRANSPOSE_BLOCK ||
          iter.first == ATTR_GEMM_DATA_TRANSPOSE_BLOCK_INNER || iter.first == ATTR_GEMM_WEIGHT_TRANSPOSE_BLOCK_INNER) {
        attr_info_.Set(iter.first, iter.second);
      }
    }
  }
  void SetConvAttrInfo(const Map<std::string, NodeRef> &attr_info) { attr_info_ = attr_info; }
  Map<std::string, NodeRef> GetConvAttrInfo() const { return attr_info_; }
  void SetSpecGemm(bool is_spec_gemm) { this->is_spec_gemm_ = is_spec_gemm; }
  bool IsSpecGemm() const { return is_spec_gemm_; }
  void CreateConvModel();
  std::vector<Stmt> GetOldC1Write() { return old_l1_write_; }
  int GetOutReduceInit() const { return out_reduce_init_; }
  TileSizes GetConvMNKDims() { return conv_mnk_dims_; }
  void SetConvMNKDims(const TileSizes &conv_mnk_dims) { conv_mnk_dims_ = conv_mnk_dims; }
  void OldC1WriteInsert(Stmt &s) { old_l1_write_.emplace_back(s); }
  std::vector<std::vector<Range>> GetRangeInfo() const { return range_info_; }
  void RecordRangeAt(size_t idx, const Range &range) {
    if (idx < range_info_.size()) {
      range_info_[idx].push_back(range);
    } else {
      std::vector<Range> tmp = {range};
      range_info_.emplace_back(tmp);
    }
  }
  void RecordRangeStrideFront(int stride) { range_stride_.push_front(stride); }
  void RecordRangeStrideBack(int stride) { range_stride_.push_back(stride); }
  std::deque<int> &GetRangeStride() { return range_stride_; }
  std::deque<int> GetRangeStride() const { return range_stride_; }
  // common tools for conv
  std::string ExtractStringFromAttrsAndInfo(const std::string &name) const;
  std::string ExtractStringFromAttrs(const std::string &name) const;
  int ExtractIntFromAttrs(const std::string &name) const;
  Expr ExtractExprFromAttrs(const std::string &name) const;
  // conv info getter
  bool IsConvBackpropFilter() const;
  bool IsConvBackpropInput() const;
  bool IsA(const std::string &name) const;
  bool IsB(const std::string &name) const;
  bool IsC(const std::string &name) const;
  bool IsCUB(const std::string &name) const;
  std::string GetAName() const;
  std::string GetBName() const;
  std::string GetCName() const;
  bool IsIm2col() const;
  bool IsLoadIm2colC1BUF() const;
  bool IsLoadIm2colC1BUFStmt(const std::string &stmtName) const;
  bool HasCube() const;
  bool IsConv() const;
  bool IsGemm() const;
  bool IsGemmDataTranspose() const;
  bool IsGemmDataTransposeBlock() const;
  bool IsGemmDataTransposeInnerBlock() const;
  bool IsGemmWeightTranspose() const;
  bool IsGemmWeightTransposeBlock() const;
  bool IsGemmWeightTransposeInnerBlock() const;
  void UpdateComputeAttrInfo();
  void FindComputeAttr(const std::vector<std::string> &op_keys);
  bool IsFilterCanByPass();
  bool IsConvHeadTail(const std::string &conv_output, const isl::id &stmtId, const StmtOpInfo &op_info,
                      const StmtIdHashMap &op_write_map);
  std::string ConvOutName();
  void ComputeByPassL1();
  void UpdateFractalIntInfoConvForward(int range_idx);
  void UpdateFractalIntInfo(int range_idx);
  void UpdateFractalIntLastInfo(std::vector<size_t> filter_fp_cluster_size);
  void UpdateSpecGemmFractalInfo(const BufferDefInfo &tensor_info);
  air::DataType MadCastType();
  int GetAttrValue(const std::string &key);
  bool InitRangeStrideVec();
  std::vector<int> GetIsolateVec(int range_idx);
  std::vector<Range> GetRange(int range_idx);
  void SetConvMNKInfo();
  std::unordered_map<std::string, Expr> GetConvInfoForTiling();
  void UpdateFractalIntInfoConvBackpropFilter(int range_idx);
  void UpdateFractalIntFirstInfoConvForward(std::vector<size_t> im2col_fp_cluster_size,
                                            std::vector<size_t> fractal_fp_cluster_size);
  void UpdateFractalIntFirstInfoConvBackpropFilter(std::vector<size_t> im2col_fp_cluster_size,
                                                   std::vector<size_t> fractal_fp_cluster_size);
  void UpdateFractalIntFirstInfo(bool is_conv_backprop_filter, const std::vector<size_t> &im2col_fp_cluster_size,
                                 const std::vector<size_t> &fractal_fp_cluster_size);

 public:
  std::map<std::string, Expr> fractal_int_info_;
  std::map<std::string, std::string> fractal_str_info_;

 private:
  UserConfig &user_config_;
  AnalysisResult &analysis_result_;
  Map<std::string, NodeRef> attr_info_;
  std::vector<std::vector<Range>> range_info_;
  std::deque<int> range_stride_;
  ConvolutionModel *model_{nullptr};
  std::vector<Stmt> old_l1_write_;
  int out_reduce_init_{0};
  TileSizes conv_mnk_dims_;
  bool is_spec_gemm_{false};
};

class ScopInfo {
 public:
  explicit ScopInfo(isl::ctx ctx)
      : ctx_(ctx), mmu_info_(CubeInfo(user_config_, analysis_result_)), sync_manager_(ctx) {}
  ~ScopInfo() = default;

  // dump tools
  int dump_schtree_count = 0;
  void DumpSchTree(const std::string &file_name, const isl::schedule &sch);
  bool DumpScopData(const std::string &file_name);
  void DumpScopDataAdvanced(std::ofstream &of);
  void DumpTransform(const std::string &file_name, PassInfo &pass_info);
  std::string AddDumpDir(const std::string &file_name);
  std::string CreateDumpDir(const std::string &file_name);
  void RecordTime(const std::string &time_log) { time_records_.emplace_back(time_log); }
  void ClearTimeRecords() { this->time_records_.clear(); };

  // tools for data info
  isl::ctx GetCtx() const { return ctx_; }
  bool MayWriteAfterRead(const std::string &name) const;
  bool IsElewiseVMStmt(const isl::id &id) const;
  void CreateDataFlowInfo();
  StmtIdHashMap StmtWriteMap();
  StmtIdHashMap StmtReadMap();
  StmtIdHashMap StmtCopyinMap();
  StmtIdHashMap StmtBindCopyinMap();
  bool IsCopyinTensor(const std::string &tensor_name);
  bool IsFunctionalCopyin(const std::string &tensor_name, const StmtIdHashMap &func_map);

  Tensor FindTensorInOrig(const isl::id &var);
  Tensor FindTensorInOrig(const std::string &str);
  Tensor FindTensor(const isl::id &var);
  Tensor FindTensor(const std::string &str);
  Type GetDtypeOf(const std::string &tensor_name) const;
  Type GetDtypeOf(const isl::id &var) const { return GetDtypeOf(var.get_name()); }
  Type GetDtypeOf(const isl::ast_expr &e) const;
  bool IsInBinds(const std::string &name) const;
  inline bool IsInBinds(const isl::id &id) const { return IsInBinds(id.get_name()); }
  Tensor FindTensorWithLargestShape(const isl::id &var);
  Tensor FindTensorWithLargestShape(const std::string &str);
  isl::id GetOriginTensorId(const std::string &name) const;
  isl::id GetOriginTensorId(const isl::id &id) const;
  bool IsWriteWholeBufferFootPrint(const isl::id &poly_ref_id) const;
  bool IsConditionalWriteTensor(const std::string &name,
                                const std::vector<std::pair<isl::id, isl::id>> &write_stmts) const;
  void CollectConditionalWritePromotions();
  void AddPartitionInfoToData(const std::vector<std::vector<int>> &partition_info);
  std::string GetIslReadName(const isl::id &cluster_id);
  std::string GetIslWriteName(const isl::id &cluster_id);
  static bool IsRead(const isl::id &id) { return IsEndsWith(id.get_name(), kReadSuffix); }
  static bool IsWrite(const isl::id &id) { return IsEndsWith(id.get_name(), kWriteSuffix); }
  static bool IsGMLWrite(const isl::id &id) { return id.get_name() == std::string("GMLwrite"); }
  static bool IsGMWrite(const isl::id &id) { return id.get_name() == std::string("GMwrite"); }
  static bool IsGMRead(const isl::id &id) { return id.get_name() == std::string("GMread"); }
  static bool IsSync(const isl::id &id) { return IsStartsWith(id.name(), SYNC_FLAG); }
  static bool IsRealize(const isl::id &id) { return IsStartsWith(id.get_name(), "REALIZE"); }
  static bool IsReduceInit(const isl::id &id) { return IsStartsWith(id.get_name(), "red_init"); }
  static bool IsReduceUpdate(const isl::id &id) { return IsStartsWith(id.get_name(), "red_update"); }
  static bool IsReduceInit(const std::string &name) { return IsStartsWith(name, "red_init"); }
  static bool IsReduceUpdate(const std::string &name) { return IsStartsWith(name, "red_update"); }

 public:
  isl::ctx ctx_;
  UserConfig user_config_;
  AnalysisResult analysis_result_;
  CubeInfo mmu_info_;
  TimeRecords time_records_;
  SyncManager sync_manager_;
  UpaNodeMapping upa_node_mapping_;
};

class PartitionSingle {
 private:
  static PartitionSingle *single_;
  static int m_times_;
  static int m_cut_m_;
  static std::map<std::string, Expr> m_fractal_int_info_;
  PartitionSingle(int times, int tile_start, int cut_m, const std::map<std::string, Expr> &fractal_int_info);
  ~PartitionSingle() = default;

 public:
  static PartitionSingle *CreateInstance(int times, int tile_start, int cut_m,
                                         const std::map<std::string, Expr> &fractal_int_info) {
    if (single_ == nullptr) {
      single_ = new PartitionSingle(times, tile_start, cut_m, fractal_int_info);
    }
    return single_;
  }
  static PartitionSingle *getInstance() { return single_; }
  static int getCutM() { return m_cut_m_; }
  static int getTimes() { return m_times_; }
  static std::map<std::string, Expr> getFractalInfo() { return m_fractal_int_info_; }

  static void free() {
    if (single_ != nullptr) {
      delete single_;
      single_ = nullptr;
    }
  }
};

std::string TensorMarkTag(MemType mem_type, MemFlow mem_flow);
struct NodeInfo {
  isl::pw_multi_aff iterator_map;
  isl::ast_build build;
};
using NodeInfoRepo = std::unordered_map<isl::id, NodeInfo, isl::IslIdIslHash>;

}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_SCOP_INFO_H_
