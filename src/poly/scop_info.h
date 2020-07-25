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
#ifndef POLY_SCOP_INFO_H_
#define POLY_SCOP_INFO_H_

#include <tvm/ir.h>

#include "poly/poly_util.h"
#include "poly/dynamic_shape.h"
#include "poly/tiling/custom_tiling.h"
#include "poly/dma_dataflow.h"
#include "pass/convolution_model.h"
#include "poly/pass_info.h"

namespace akg {
namespace ir {
namespace poly {

constexpr auto kReadSuffix = "read";
constexpr auto kWriteSuffix = "write";
constexpr auto kIterNamePrefix = "cc";
constexpr auto kGemmIterNamePrefix = "ee";

struct DimensionInfo {
  int64_t index;
  std::string axis;
  int64_t l1_tiling_size;
  int64_t l0_tiling_size;
  int64_t dim_seq;
  Expr l1_var;
  Expr l0_var;
  Expr pragma;
  bool is_inner{false};
};
struct ParamInfo {
  std::string type_key;
  Expr key;
  Expr value;
};
using TileSizes = std::vector<DimensionInfo>;
struct TilingInfo {
  int tiling_flag;  // flag=1, tailing; flag=0, not tailing
  TileSizes dim_infos;
};
using Tiles = std::vector<TilingInfo>;

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

  void SetAttrs(const Map<std::string, NodeRef> &attrs) {
    if (attrs.empty()) return;
    ParseDynamicShapeAttr(attrs, "dynamic_shape", &dynamic_shape_);
    ParseIntAttr(attrs, "dynamic_shape_bound", &dynamic_shape_bound_);
    ParseBoolAttr(attrs, "pragma_tilesize_is_var", &tile_size_is_var_);
    ParseBoolAttr(attrs, "pragma_outerband_need_split", &outer_band_need_split_);

    ParseStringAttr(attrs, "dim", &b_dim_);
    ParseCustomTilingAttr(attrs, "custom_tiling", &custom_tiling_);
    ParseBoolAttr(attrs, "pragma_analyze_reuse_buffer", &pragma_analyze_reuse_buffer_);
    ParseBoolAttr(attrs, "pragma_speedup_tiling", &pragma_speedup_tiling_);
    ParseBoolAttr(attrs, "pragma_allow_tail_tiling", &pragma_allow_tail_tiling_);
    ParseBoolAttr(attrs, "pragma_analyze_multicore", &pragma_analyze_multicore_);
    ParseBoolAttr(attrs, "pragma_checkcoincident", &tile_check_coincident_);

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

    ParseBoolAttr(attrs, "pragma_opt_for_davinci", &optimize_for_davinci_);
    ParseBoolAttr(attrs, "enable_feature_library", &enable_feature_library_);
    ParseBoolAttr(attrs, "enable_hoist_cond_write", &enable_hoist_cond_write_);
    ParseBoolAttr(attrs, "enable_mark_multi_core", &enable_mark_multi_core_);

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

    if (force_remove_self_dependence_) {
      LOG(WARNING) << "pragma_force_rmselfdep should be used with care. "
                   << "It removes all self dependence and cannot ensure that reduce axis do not use multicore.";
    }
  }

  // getter for dynamic shape config
  bool GetIsDynamic() const { return is_dynamic_; }
  std::vector<NodeRef> GetDynamicShape() { return dynamic_shape_; }
  int GetDynamicShapeBound() const { return dynamic_shape_bound_; }
  bool GetTileSizeIsVar() const { return tile_size_is_var_; }
  bool GetOuterBandNeedSplit() const { return outer_band_need_split_; }

  // getter for tiling config
  std::vector<NodeRef> GetCustomTiling() { return custom_tiling_; }
  std::string GetBDim() const { return b_dim_; }
  bool GetPragmaSpeedUpTiling() const { return pragma_speedup_tiling_; }
  bool GetPragmaAnalyzeReuseBuffer() const { return pragma_analyze_reuse_buffer_; }
  bool GetPragmaAllowTailTiling() const { return pragma_allow_tail_tiling_; }
  bool GetPragmaAnalyzeMulticore() const { return pragma_analyze_multicore_; }
  bool GetTileCheckCoincident() const { return tile_check_coincident_; }

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
  bool GetOptimizeForDavinci() const { return optimize_for_davinci_; }
  bool GetEnableFeatureLib() const { return enable_feature_library_; }
  bool GetEnableHoistCondWrite() const { return enable_hoist_cond_write_; }
  bool GetEnableMarkMultiCore() const { return enable_mark_multi_core_; }

  // getter for conv config
  int GetMatBDimH() const { return matB_dim_h_; }
  int GetMatBDimW() const { return matB_dim_w_; }
  int GetByPassL1() const { return bypassL1_; }
  int GetIsolatedIdx() const { return isolated_idx_; }
  std::string GetKernelName() { return kernel_name_; }
  int GetPragmaIsConv() const { return pragma_is_conv_; }
  int GetConvBackPropFilter() const { return conv_back_prop_filter_; }
  bool GetConvSpecialDma() const { return conv_special_dma_; }
  bool GetDynamicShapeConvFullParametric() const { return dynamic_shape_conv_full_parametric_; }

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

  std::vector<Stmt> GetOuterLetStmts() { return outer_let_stmts_; }
  void SetOuterLetStmts(std::vector<Stmt> &outer_let_stmts) { outer_let_stmts_ = outer_let_stmts; }
  std::unordered_set<isl::id, isl::IslIdIslHash> GetRealizeFromInput() { return realize_from_input_; }
  void InsertRealizeFromInput(const isl::id &id) { realize_from_input_.insert(id); }
  void SetOriginBind(const Binds &binds_orig) { binds_orig_ = binds_orig; }
  Binds GetOriginBind() const { return binds_orig_; }
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

  // dump all info
  void DumpScopDataScheduleAttrs(std::ofstream &of);

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

 private:
  std::vector<Stmt> outer_let_stmts_;
  std::unordered_set<isl::id, isl::IslIdIslHash> realize_from_input_;
  Binds binds_orig_;
  Binds binds_;
  Stmt body_;
  std::unordered_map<std::string, Var> params_;
  std::unordered_map<std::string, Expr> params_rev_map_;
  std::map<int64_t, Expr> param_tiling_map_;
  bool is_spec_gemm_{false};

  // dynamic shape config
  bool is_dynamic_{false};
  std::vector<NodeRef> dynamic_shape_;
  int dynamic_shape_bound_{0};
  bool tile_size_is_var_{false};
  bool outer_band_need_split_{false};

  // tiling config
  std::string b_dim_;
  std::vector<NodeRef> custom_tiling_;
  bool pragma_analyze_reuse_buffer_{true};
  bool pragma_speedup_tiling_{false};
  bool pragma_allow_tail_tiling_{true};
  bool pragma_analyze_multicore_{true};
  bool tile_check_coincident_{true};

  // schedule tree transform config
  bool remove_self_dependence_{true};
  bool force_remove_self_dependence_{false};
  bool remove_invariant_dependence_{false};
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
  bool optimize_for_davinci_{false};
  bool enable_feature_library_{false};
  bool enable_hoist_cond_write_{true};
  bool enable_mark_multi_core_{false};

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

  // dump config
  int dump_tuning_level_{0};
  bool dump_pass_ir_{false};
  std::string dump_poly_dir_;
};

struct OperatorDomainSpace {
  isl::space param_space;
  isl::multi_id tuple;
};

using ReduceStmtMap = std::unordered_map<isl::id, std::vector<std::string>, isl::IslIdIslHash>;
using AccessMap = std::unordered_map<const Node *, isl::id>;
using StatementMap = std::unordered_map<isl::id, const Node *, isl::IslIdIslHash>;
using OperatorDomainMap = std::unordered_map<isl::id, OperatorDomainSpace, isl::IslIdIslHash>;
using ReduceMap = std::unordered_map<const Provide *, Array<IterVar>>;
using CondVarsMap = std::unordered_map<isl::id, std::unordered_set<std::string>, isl::IslIdIslHash>;
using BufferBindVec = std::vector<std::pair<const NodeRef, const Expr>>;

class AnalysisResult {
 public:
  AnalysisResult() = default;
  ~AnalysisResult() = default;

  void RecordWrites(const isl::union_map &writes) { writes_ = writes; }
  void RecordReads(const isl::union_map &reads) { reads_ = reads; }
  void RecordCopyin(const isl::union_map &copyin) { copyin_ = copyin; }
  void RecordFakeCopyin(const isl::union_map &fake_copyin) { fake_copyin_ = fake_copyin; }
  void RecordTransferStmt(const isl::union_set &transfer_stmt) { transfer_stmt_ = transfer_stmt; }
  void RecordInnerBandDependency(const isl::union_map &inter_band_dependency) {
    inter_band_dependency_ = inter_band_dependency;
  }

  void RecordReduceStmt(const isl::id &stmt_id, const std::vector<std::string> &reduce_axis_list) {
    reduce_stmts_.emplace(stmt_id, reduce_axis_list);
  }
  void RecordAccess(const Node *node, const isl::id &tensor_id) { accesses_.emplace(node, tensor_id); }
  void RecordStatement(const isl::id &tensor_id, const Node *node) { statements_.emplace(tensor_id, node); }
  void RecordStmtOpInfo(const isl::id &tensor_id, const StmtOpInfo &op_info) {
    stmt_op_Info_.emplace(tensor_id, op_info);
  }
  void RecordOperatorDomain(const isl::id &tensor_id, const OperatorDomainSpace &dom_space) {
    domains_.emplace(tensor_id, dom_space);
  }
  void RecordReduce(const Provide *node, const Array<IterVar> &reduces) { reduces_.emplace(node, reduces); }
  void RecordBufferBindVec(const std::pair<const NodeRef, const Expr> &buf_bind) { buf_bind_vec_.push_back(buf_bind); }
  void RecordUpdateTensor(const Tensor &tensor) { update_tensors_.push_back(tensor); }
  void RecordAttrStmt(const AttrStmt *attr_stmt) { attr_stmts_.push_back(attr_stmt); }

  isl::union_map GetReads() const { return reads_; }
  isl::union_map &GetWrites() { return writes_; }
  isl::union_map GetWrites() const { return writes_; }
  isl::union_map &GetCopyin() { return copyin_; }
  isl::union_map GetCopyin() const { return copyin_; }
  isl::union_map GetFakeCopyin() const { return fake_copyin_; }
  isl::union_set GetTransferStmt() const { return transfer_stmt_; }
  isl::union_map GetInnerBandDependency() const { return inter_band_dependency_; }

  ReduceStmtMap &GetReduceStmtMap() { return reduce_stmts_; }
  AccessMap &GetAccessMap() { return accesses_; }
  StatementMap &GetStatementMap() { return statements_; }
  StatementMap GetStatementMap() const { return statements_; }
  StmtOpInfoMap &GetStmtOpInfoMap() { return stmt_op_Info_; }
  StmtOpInfoMap GetStmtOpInfoMap() const { return stmt_op_Info_; }
  OperatorDomainMap &GetOperatorDomainMap() { return domains_; }
  OperatorDomainMap GetOperatorDomainMap() const { return domains_; }
  ReduceMap &GetReduceMap() { return reduces_; }
  ReduceMap GetReduceMap() const { return reduces_; }
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
  void SetScheduleMapBeforeTile(const isl::union_map &schedule_map_before_tile) {
    schedule_map_before_tile_ = schedule_map_before_tile;
  }
  void InitScheduleMapBeforeTile(const isl::ctx &ctx) { schedule_map_before_tile_ = isl::union_map::empty(ctx); }
  const isl::union_map &GetScheduleMapBeforeTile() { return schedule_map_before_tile_; }
  void SetTranstormedSchedule(const isl::schedule &transformed_schedule) {
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

  // dump all data
  void DumpScopDataBasics(std::ofstream &of);

  int CountBufferDefInfo(const isl::id &tensor_id) const;
  const std::vector<BufferDefInfo> &BufferDefInfos() const { return buffer_def_infos_; }
  const BufferDefInfo &GetBufferDefInfo(const isl::id &tensor_id) const;
  bool HasBufferDefInfo(const isl::id &tensor_id) const;
  const std::vector<std::pair<isl::union_set, BufferedFootPrintInfo>> &ActiveBufferFootprints() const {
    return active_buffer_footprints_;
  }
  void DumpBufferDefInfos(std::ostream &out);
  std::unordered_set<std::string> ExtractWithStmtId() const;

 public:
  std::vector<std::pair<std::string, STMT_OP_TYPE>> stmt_type_;
  std::vector<std::pair<isl::union_set, BufferedFootPrintInfo>> active_buffer_footprints_;
  std::vector<BufferDefInfo> buffer_def_infos_;
  BufferDefInfo default_buffer_def_info_;

 private:
  isl::union_map reads_;
  isl::union_map writes_;
  isl::union_map copyin_;
  isl::union_map fake_copyin_;
  isl::union_set transfer_stmt_;
  isl::union_map inter_band_dependency_;
  ReduceStmtMap reduce_stmts_;
  AccessMap accesses_;
  StatementMap statements_;
  StmtOpInfoMap stmt_op_Info_;
  OperatorDomainMap domains_;
  ReduceMap reduces_;
  BufferBindVec buf_bind_vec_;
  std::vector<Tensor> update_tensors_;
  std::vector<const AttrStmt *> attr_stmts_;

  std::map<std::string, std::vector<std::string>> tensor_name_flows_;
  std::map<std::string, MemFlow> tensor_mem_flows_;
  std::unordered_set<std::string> conditional_write_buffer_footprints_;

  std::deque<ParamInfo> tiling_constraints_;
  TileSizes tile_sizes_;
  bool is_tiled_{false};
  isl::union_map schedule_map_before_tile_;  // before tiling, after ungroup.
  isl::schedule transformed_schedule_;
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
  std::vector<Stmt> GetOldL1Write() { return old_l1_write_; }
  int GetOutReduceInit() const { return out_reduce_init_; }
  TileSizes GetConvMNKDims() { return conv_mnk_dims_; }
  void SetConvMNKDims(const TileSizes &conv_mnk_dims) { conv_mnk_dims_ = conv_mnk_dims; }
  void OldL1WriteInsert(Stmt &s) { old_l1_write_.emplace_back(s); }
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
  bool IsLoad3dL1Ub() const;
  bool IsLoad3dL1UBStmt(const std::string &stmtName) const;
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
  explicit ScopInfo(isl::ctx ctx) : ctx_(ctx), cube_info_(CubeInfo(user_config_, analysis_result_)) {}
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
  bool IsCopyinTensor(const std::string &tensorName);

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
  static bool IsGMWrite(const isl::id &id) { return id.get_name() == std::string("GMwrite"); }

 public:
  isl::ctx ctx_;
  UserConfig user_config_;
  AnalysisResult analysis_result_;
  CubeInfo cube_info_;
  TimeRecords time_records_;
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
