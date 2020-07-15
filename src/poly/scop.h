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
#ifndef POLY_SCOP_H_
#define POLY_SCOP_H_

#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/operation.h>
#include <chrono>
#include <queue>

#include "poly/isl.h"
#include "poly/stmt_parse.h"
#include "poly/poly_util.h"
#include "poly/dma_dataflow.h"
#include "poly/custom_tiling.h"
#include "poly/dynamic_shape.h"
#include "pass/convolution_model.h"

// timer records
#define TIMER_START timer_start = std::chrono::high_resolution_clock::now()
#define TIMER_DURATION                                                                                                \
  (std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - timer_start) \
     .count()) *                                                                                                      \
    1000
#define TIMER_SHOW(NAME, SPEC_GEMM) \
  { LOG(INFO) << "[ Polyhedral exec time" << SPEC_GEMM << " ], " << NAME << " spent " << TIMER_DURATION << " ms"; }

// Prime numbers for prime-param replacement
#define PRIME_1 53
#define PRIME_2 59
#define PRIME_3 61

namespace akg {
namespace ir {
namespace poly {
class TensorFootprintCluster;

struct OperatorDomainSpace {
  isl::space param_space;
  isl::multi_id tuple;
};

using IteratorMap = std::unordered_map<isl::id, std::vector<std::string>, isl::IslIdIslHash>;
using StatementMap = std::unordered_map<isl::id, const Node *, isl::IslIdIslHash>;
using AccessMap = std::unordered_map<const Node *, isl::id>;
using ReduceMap = std::unordered_map<const Provide *, Array<IterVar>>;
using BufferBindVec = std::vector<std::pair<const NodeRef, const Expr>>;
using OperatorDomainMap = std::unordered_map<isl::id, OperatorDomainSpace, isl::IslIdIslHash>;
using PartialTileAccessespair = std::vector<isl::union_map>;
using ReduceStmtMap = std::unordered_map<isl::id, std::vector<std::string>, isl::IslIdIslHash>;
using CondVarsMap = std::unordered_map<isl::id, std::unordered_set<std::string>, isl::IslIdIslHash>;

struct NodeInfo {
  isl::pw_multi_aff iterator_map;
  isl::ast_build build;
};
using NodeInfoRepo = std::unordered_map<isl::id, NodeInfo, isl::IslIdIslHash>;

void GetAffOffsetAndNumVars(const isl::aff &aff, int &offset, int &num_vars);
bool IsAffVarPlusOffset(const isl::aff &aff);
bool IsAffNonZeroConst(const isl::aff &aff);

std::string TensorMarkTag(MemType memType, MemFlow memFlow);

Stmt OptimizeHalide(const Stmt &s, bool dynamic_shape = false);

class Scop {
 public:
  struct TilingInfo;
  using Binds = Map<Tensor, Buffer>;
  using Tiles = std::vector<Scop::TilingInfo>;
  struct ParamInfo {
    std::string type_key;
    Expr key;
    Expr value;
  };
  enum AtomicType { Equ = 0, Add };

  // transform, save group stmts
  std::unordered_map<isl::id, isl::union_set_list, isl::IslIdIslHash> group_filter_map_;
  // save halide IR let stmts
  std::vector<Stmt> outer_let_stmts_;
  // save halide IR realize
  std::unordered_set<isl::id, isl::IslIdIslHash> realize_from_input_;
  Stmt body_;
  Binds binds_;
  const Binds binds_orig_;

  // dynamic shape
  std::unordered_map<std::string, Var> params_;
  std::unordered_map<std::string, Expr> params_rev_map_;

  isl::ctx ctx_;
  bool is_spec_gemm_{false};
  bool is_tiled_{false};
  int conv_back_prop_filter_{0};
  int bypassL1_{0};
  int dump_tuning_level_{0};
  bool disable_group_{false};
  bool tile_inner_band_{false};
  bool pragma_set_all_coincident_{false};
  bool remove_self_dependence_{true};
  bool force_remove_self_dependence_{false};
  bool remove_invariant_dependence_{false};
  bool compute_reschedule_{false};
  bool disable_schedule_shift_{false};
  bool enable_schedule_max_constant_{false};
  bool disable_loop_reversal_{false};
  bool disable_loop_fusion_{false};
  bool mod_schedule_shift_{false};
  bool conv_special_dma_{false};
  bool tile_check_coincident_{true};
  bool reorder_schedule_{false};
  bool sink_last_axis_{true};
  bool keep_outer_band_order_{false};
  bool optimize_for_davinci_{false};
  bool enable_feature_library_{false};
  bool enable_hoist_cond_write_{true};
  bool enable_mark_multi_core_{false};
  bool is_dynamic_{false};
  int dump_pass_ir_{0};
  int depth_ = 0;
  int dynamic_shape_bound_{0};
  int tile_size_is_var_{0};
  int outer_band_need_split_{0};
  int pragma_is_conv_{0};

  std::string dump_poly_dir_;
  std::string kernel_name_;
  std::string iter_prefix_;
  isl::schedule schedule_;
  isl::union_map sch_;  // before tiling, after ungroup.

  std::vector<Stmt> old_l1_write_;

  NodeRef spaces_;
  int matB_dim_h_{-1};
  int matB_dim_w_{-1};

  /// Store related information for analysis
  struct Data {
    isl::union_map reads;
    isl::union_map copyin;
    isl::union_map writes;
    isl::union_map fake_copyin;
    isl::union_set transfer_stmt;
    isl::union_map inter_band_dependency;
    ReduceStmtMap reduce_stmts;
    AccessMap accesses;
    StatementMap statements;
    StmtOpInfoMap stmt_op_Info;
    IteratorMap iterators;
    OperatorDomainMap domains;
    ReduceMap reduces;
    BufferBindVec vecs;
    std::vector<Tensor> update_tensors;
    std::vector<const AttrStmt *> attrs;

    std::vector<std::vector<Range>> range_info;
    std::vector<int> range_stride;
  } data_;

  std::shared_ptr<TensorFootprintCluster> gemm_a_transpose_fp_cluster_;
  std::shared_ptr<TensorFootprintCluster> gemm_b_transpose_fp_cluster_;
  std::shared_ptr<TensorFootprintCluster> im2col_fp_cluster;

  // dimension info read from file,erery dimInfo
  // represents every row in the file.
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
  using TileSizes = std::vector<Scop::DimensionInfo>;

  std::vector<DimensionInfo> dim_infos_;

  std::map<int64_t, Expr> param_tiling_map_;

  std::map<std::string, Expr> fractal_int_info_;
  std::map<std::string, std::string> fractal_str_info_;

  struct TilingInfo {
    int tiling_flag;  // flag=1, tailing; flag=0, not tailing
    std::vector<DimensionInfo> dim_infos;
  };

  std::vector<DimensionInfo> conv_mnk_dims_;
  struct BufferedDecl {
    enum Kind { L1, L0, L0A, L0B, L0C, UB };

    isl::id tensor_id;
    std::vector<size_t> sizes;
    Type type;
    Kind kind;
    Tensor tensor;
  };

  std::map<std::string, std::vector<std::string>> tensor_name_flows_;
  std::map<std::string, MemFlow> tensor_mem_flows_;
  std::vector<BufferDefInfo> buffer_def_infos_;
  std::queue<isl::id> buffer_footprint_queue_;
  BufferDefInfo place_holder_;
  std::vector<std::pair<std::string, STMT_OP_TYPE>> stmt_type_;

  struct BufferedFootPrintInfo {
    std::shared_ptr<TensorFootprintCluster> cluster;
    isl::union_map outer_schedule;
    isl::id cluster_id;
  };

  std::deque<ParamInfo> tiling_constraints_;
  std::string b_dim_;

  std::unordered_map<isl::id, size_t, isl::IslIdIslHash> n_clusters_;

  std::unordered_map<isl::id, BufferedDecl, isl::IslIdIslHash> buffered_decls_;

  std::vector<std::pair<isl::union_set, BufferedFootPrintInfo>> active_buffer_footprints_;

  std::unordered_set<std::string> conditional_write_buffer_footprints_;

  Map<std::string, NodeRef> attr_info_;

  bool isolated_{false};
  int isolated_idx_{0};
  int out_reduce_init_{0};
  std::vector<NodeRef> custom_tiling_;
  std::vector<NodeRef> dynamic_shape_;
  bool dynamic_shape_conv_full_parametric_{false};
  bool pragma_analyze_reuse_buffer_{false};
  bool pragma_speedup_tiling_{false};
  bool pragma_allow_tail_tiling_{true};
  bool pragma_analyze_multicore_{true};

  ConvolutionModel *model_{nullptr};

  struct GemmVar {
    VarExpr var_batch_name{"b"};
    VarExpr var_no_name{"no"};
    VarExpr var_mo_name{"mo"};
    VarExpr var_mi_name{"mi"};
    VarExpr var_ni_name{"ni"};
    VarExpr var_ko_name{"ko"};
    VarExpr var_ki_name{"ki"};
  };

  // scop
  Scop(Stmt body, const Binds &binds, isl::ctx ctx, bool is_spec_gemm);
  ~Scop();
  static std::shared_ptr<Scop> make(const Stmt &body, const Binds &binds, isl::ctx ctx, bool is_spec_gemm) {
    return std::make_shared<Scop>(body, binds, ctx, is_spec_gemm);
  }

  // main
  isl::schedule GenIsl();
  void ComputeTransferCopyin(isl::union_map &fake_copyin);
  void TransferStmt(isl::schedule &t_sch);
  void ComputeByPassL1();
  void AddPartitionInfoToData(const std::vector<std::vector<int>> &partition_info);
  isl::schedule Transform(isl::schedule, bool coincident = true, bool tuning = false);
  Stmt GenHalide(const isl::schedule &);

  // transform
  isl::schedule GroupStatements(const isl::schedule &sch, bool &has_group);
  void InitDimensionInfo(const isl::schedule &);
  void MergeTilingInfo(Tiles &tiling_infos);
  std::unordered_map<std::string, Expr> GetConvInfoForTiling();
  std::vector<std::vector<int>> AddTileInfo(const std::vector<std::vector<int>> &partition_info);
  isl::schedule ChangeMarkNodePosition(const isl::schedule &);
  isl::schedule LabelRealizeOutPosition(const isl::schedule &) const;
  isl::schedule ReorderMarkNodes(const isl::schedule &) const;
  isl::schedule ReorderInnerBandLoops(const isl::schedule &schedule) const;
  bool InjectMulticoreToSchedule(isl::schedule_node &outer_band);
  bool SingleMulticoreBand(isl::schedule_node &outer_band);
  isl::schedule MarkOuterMost(const isl::schedule &);
  isl::schedule MarkFuseOp(const isl::schedule &) const;
  isl::schedule InsertNodeForAllocC(isl::schedule &sched);

  // tool
  isl::id_list CreateIteratorList(const isl::schedule &schedule_iter, const std::string &prefix);
  int ExtractIntFromAttrs(const std::string &name) const;
  Expr ExtractExprFromAttrs(const std::string &name) const;
  std::string ExtractStringFromAttrs(const std::string &name) const;
  std::unordered_set<std::string> ExtractWithStmtId() const;
  std::string ExtractStringFromAttrsAndInfo(const std::string &name) const;
  isl::pw_multi_aff RemoveConstOffsetFromBufferFootprint(const isl::pw_multi_aff &promotion);
  CondVarsMap ExtractCondVarsMap() const;

  // data info
  static bool IsRead(const isl::id &id) { return IsEndsWith(id.get_name(), kReadSuffix); }
  static bool IsWrite(const isl::id &id) { return IsEndsWith(id.get_name(), kWriteSuffix); }
  static bool IsGMWrite(const isl::id &id) { return id.get_name() == std::string("GMwrite"); }
  const isl::union_set Domain() const;
  AtomicType GetAtomicWrite(const isl::id &id) const;
  Type GetDtypeOf(const std::string &tensor_name) const;
  Type GetDtypeOf(const isl::id &var) const { return GetDtypeOf(var.get_name()); }
  Type GetDtypeOf(const isl::ast_expr &e) const;
  bool IsInBinds(const std::string &name) const;
  inline bool IsInBinds(const isl::id &id) const { return IsInBinds(id.get_name()); }
  void RecordReduceStmt(const isl::id &stmt_id, const std::vector<std::string> &reduce_axis_list);
  bool InitRangeStrideVec();
  bool MayWriteAfterRead(const std::string &name) const;
  bool IsElewiseVMStmt(const isl::id &id) const;
  void CreateDataFlowInfo();
  StmtIdHashMap StmtWriteMap();
  StmtIdHashMap StmtReadMap();
  StmtIdHashMap StmtCopyinMap();
  bool IsCopyinTensor(const std::string &tensorName);
  void AddTensorDataFlow(const std::vector<MemType> &mem_flow, const std::vector<std::string> &name_flow);
  void AddStateTensorsDataFlow();
  Tensor FindTensor(const isl::id &var);
  Tensor FindTensor(const std::string &str);
  Tensor FindTensorInOrig(const isl::id &var);
  Tensor FindTensorInOrig(const std::string &str);
  Tensor FindTensorWithLargestShape(const isl::id &var);
  Tensor FindTensorWithLargestShape(const std::string &str);
  void ParseIntAttr(const Map<std::string, NodeRef> &attrs, const std::string &attr_name, int *attr_to_set);
  void ParseBoolAttr(const Map<std::string, NodeRef> &attrs, const std::string &attr_name, bool *attr_to_set);
  void ParseStringAttr(const Map<std::string, NodeRef> &attrs, const std::string &attr_name, std::string *attr_to_set);
  void ParseCustomTilingAttr(const Map<std::string, NodeRef> &attrs, const std::string &attr_name,
                             std::vector<NodeRef> *attr_to_set);
  void ParseDynamicShapeAttr(const Map<std::string, NodeRef> &attrs, const std::string &attr_name,
                             std::vector<NodeRef> *attr_to_set);
  void SetAttrs(const Map<std::string, NodeRef> &attrs);
  std::string GetbDim() const { return b_dim_; }
  std::string GetcDim();
  isl::id GetOriginTensorId(const std::string &name) const;
  isl::id GetOriginTensorId(const isl::id &id) const;

  // conv info
  std::vector<int> GetIsolateVec(int range_idx);
  std::vector<Range> GetRange(int range_idx);
  std::string ConvOutName();
  air::DataType MadCastType();
  bool IsConvHeadTail(const std::string &conv_output, const isl::id &stmtId, const StmtOpInfo &op_info,
                      const StmtIdHashMap &op_write_map);
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
  bool IsConvBackpropInput() const;
  bool IsConvBackpropFilter() const;
  bool IsGemm() const;
  bool IsGemmDataTranspose() const;
  bool IsGemmDataTransposeBlock() const;
  bool IsGemmDataTransposeInnerBlock() const;
  bool IsGemmWeightTranspose() const;
  bool IsGemmWeightTransposeBlock() const;
  bool IsGemmWeightTransposeInnerBlock() const;
  void FindComputeAttr(const std::vector<std::string> &op_keys);
  void UpdateComputeAttrInfo();
  void CreateConvModel(bool is_dynamic);
  bool IsFilterCanByPass();
  void GetConvMNKInfo(std::vector<DimensionInfo> &dim_infos);

  // record buffer footprint
  bool UpdateBufferDefInfoSizes(const isl::id &tensor_id, const std::vector<size_t> &new_sizes);
  void AddOneBufferDefInfo(const isl::id &ancestorId, const std::vector<std::pair<isl::id, MemType>> &data_stream);
  void MakeBufferFootprintCluster(BufferDefInfo &tensor_info);
  void GatherBufferFootprintDefInfo(const isl::schedule_node &tree, BufferDefInfo &tensor_info);
  void GatherFractalDefInfo(const isl::schedule_node &tree, BufferDefInfo &tensor_info, std::vector<size_t> &sizes);
  void HoistIm2colBufferFootprintCluster(const isl::union_map &schedule, const isl::schedule_node &node, int index,
                                         BufferDefInfo &tensor_info);
  void MakeMultiBufferFootprint(const isl::union_map &schedule, const isl::schedule_node &node, int &index,
                                BufferDefInfo &tensor_info);
  void ReorderBufferedDefInfos();
  void RecordAllTensorBufferFootprintToExtension();
  void CollectBufferFootprintDefInfo(BufferDefInfo &tensor_info, const isl::union_map &schedule,
                                     const isl::schedule_node &node);
  isl::schedule HoistBufferFootprintAtMarkNode(const isl::schedule_node &root, const std::string &markTag,
                                               size_t index);
  isl::schedule_node HoistBufferFootprintAtMarkNode(const isl::schedule_node &tree, size_t index);
  isl::schedule_node HoistTensorClusterFootprint(isl::schedule_node tree, size_t index, const isl::union_map &schedule);
  std::shared_ptr<TensorFootprintCluster> GetFootPrintsCluster(const isl::id &tensor_id);
  const std::vector<BufferDefInfo> &BufferDefInfos() const { return buffer_def_infos_; }
  bool HasBufferDefInfo(const isl::id &tensor_id) const;
  const BufferDefInfo &GetBufferDefInfo(const isl::id &tensor_id) const;

  void UpdateFractalIntFirstInfoConvForward(std::vector<size_t> im2col_fp_cluster_size,
                                            std::vector<size_t> fractal_fp_cluster_size);
  void UpdateFractalIntFirstInfoConvBackpropFilter(std::vector<size_t> im2col_fp_cluster_size,
                                                   std::vector<size_t> fractal_fp_cluster_size);
  void UpdateFractalIntFirstInfo(bool is_conv_backprop_filter, const std::vector<size_t> &im2col_fp_cluster_size,
                                 const std::vector<size_t> &fractal_fp_cluster_size);
  void UpdateFractalIntLastInfo(std::vector<size_t> filter_fp_cluster_size);
  void SetFindBuffer(const isl::id &tensor_id, bool find_buffer);
  int CountBufferDefInfo(const isl::id &tensor_id) const;
  void AddGemmTransposeFpCluster(const isl::union_map &schedule);
  const std::vector<std::pair<isl::union_set, BufferedFootPrintInfo>> &ActiveBufferFootprints() const {
    return active_buffer_footprints_;
  }
  std::vector<std::pair<isl::union_set, Scop::BufferedFootPrintInfo>> CollectBufferedFootprints(
    const isl::union_set &active_points, const isl::id &tensor_id) const;
  std::vector<size_t> CollectBufferedFootprintsIndexes(const isl::union_set &active_points,
                                                       const isl::id &tensor_id) const;
  bool IsWriteWholeBufferFootPrint(const isl::id &poly_ref_id) const;
  bool IsConditionalWriteTensor(const std::string &name,
                                const std::vector<std::pair<isl::id, isl::id>> &write_stmts) const;
  void FindConditionalWritePromotions();

  // specgemm
  void UpdateSpecGemmFractalInfo(const BufferDefInfo &tensor_info);
  Binds BuildConvGemmBand();
  void BuildConvGemmFeatureBand(Scop::Binds &new_bind);
  void BuildConvGemmFilterBand(Scop::Binds &new_bind);
  void BuildConvGemmResultBand(Scop::Binds &new_bind);
  void UpdateFractalIntInfo(int range_idx);
  void UpdateFractalIntInfoConvForward(int range_idx);
  void UpdateFractalIntInfoConvBackpropFilter(int range_idx);
  Stmt ConstructPolyGemm(const Expr &cond = Expr());
  Stmt ConstructGemm(const Binds &gemm_bind, const Expr &cond = Expr());
  Stmt ConstructGemmReduceBody(const Binds &gemm_bind, const Expr &mad_init_cond, const GemmVar &gv);
  static Stmt ConstructFor(int init, Expr cond_exp, const VarExpr &iter, const Stmt &s);
  std::string ConstructGemmDimensionInfo();
  std::string AutoConstructGemmDimensionInfo();
  void CheckConvGemmParam();
  static int64_t AutoConvMNKTile(const std::string &param_name, int64_t param_size);
  bool CheckFeatureTensorShape(const Array<Expr> &shape);
  bool CheckFilterTensorShape(const Array<Expr> &shape);
  static Tensor FindBindTensor(const Binds &bind, const std::string &name);
  int GetAttrValue(const std::string &key);
  int GetMAxisSetDim();

  // dynamic
  void RegisterParam(const Expr &expr);
  void GetParams();
  isl::set CreateParamsSet() const;
  Stmt RestoreCombinedParams(Stmt stmt);
  void InsertRange(std::map<int64_t, Expr> &param_map, const std::pair<int64_t, Expr> &item);
  void InsertPairs(Stmt &stmt, std::map<int64_t, Expr> &param_map);
  void InsertPairsConvTileVar(Stmt &stmt, std::map<int64_t, Expr> &param_map);
  void InsertPairsSpecGemmTileVar(std::map<int64_t, Expr> &param_map);
  void InsertPairsSpecGemmOrConv(Stmt &stmt, std::map<int64_t, Expr> &param_map);
  void Full2PartialDynamic(std::unordered_map<std::string, Expr> &params_map,
                           const Map<std::string, NodeRef> &attr_info);
  Stmt ReplacePrimesWithParameters(Stmt stmt);
  Expr ReplacePragmaPrimeByVar(Expr prime);
  Stmt AddTilingStrategyApplet(Stmt stmt);

  // debug
  void DumpSchTree(const std::string &file_name, const isl::schedule &sch);
  bool DumpScopData(const std::string &file_name);
  void DumpScopDataBasics(std::ofstream &of);
  void DumpScopDataAdvanced(std::ofstream &of);
  void DumpScopDataScheduleAttrs(std::ofstream &of);
  std::string AddDumpDir(const std::string &file_name);
  std::string CreateDumpDir(const std::string &file_name);
  void DumpBufferDefInfos(std::ostream &out = LOG(INFO));
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

std::pair<std::vector<Scop::DimensionInfo>, std::deque<Scop::ParamInfo>> GenerateTiling(Scop *scop,
                                                                                        const isl::schedule &,
                                                                                        const std::vector<NodeRef> &,
                                                                                        const std::vector<NodeRef> &);
isl::union_map ShortSchedule(const isl::schedule_node &node);
isl::union_map LocalSchedule(const isl::schedule_node &node);
NodeRef GenerateTilingSpace(Scop *scop, const isl::schedule &, int dump_level,
                            const std::vector<NodeRef> &custom_tiling, const std::vector<NodeRef> &dynamic_shape);
Stmt OptimizeCce(const Stmt &s, bool dynamic_shape = false);
}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_SCOP_H_
