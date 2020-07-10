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
#ifndef POLY_TILING_ANALYZER_H_
#define POLY_TILING_ANALYZER_H_

#include <tvm/ir.h>
#include <tvm/packed_func_ext.h>
#include <vector>
#include <deque>
#include <memory>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "build_module.h"
#include "contrib/cce_parm/cceconf.h"
#include "common/util_cce.h"
#include "pass/expr_alg_simplify.h"
#include "pass/utils.h"
#include "poly/scop.h"
#include "poly/tiling_utils.h"

namespace akg {
namespace ir {
namespace poly {
// common integers
constexpr auto ALIGN_BYTES = 32;
constexpr auto CUBE_UNIT = 16;
constexpr auto MIN_TILE = 1;
constexpr auto EXCEED_MEM_CODE = -2;
constexpr auto BISEC_REDUCE_MEM_EXPANSION = 2;
constexpr auto DUMP_LEVEL_GENERAL = 1;
constexpr auto DUMP_LEVEL_CANDIDATE = 2;
constexpr auto DUMP_LEVEL_TUNING = 3;
constexpr auto DUMP_LINE_BREAK_NUM = 100;
constexpr auto GEN_PRIME_NUM = 32;
constexpr auto VECTORIZE_BYTE = 256;
constexpr auto MAX_REPEAT = 255;
constexpr auto MIN_CORE_GRANULARITY = 256;
constexpr auto DESIRE_CORE_GRANULARITY = 8192;  // 256 Bytes * 64 repeat

// Controlled by custom tiling.
constexpr auto ALLOCATION_PERCENTAGE = 0.5;  // reserved for double buffer in default

inline int64_t GetAlignBytes(const int64_t dtype) {
  CHECK_GE(dtype, 0) << "Data type should be positive.";
  if (dtype == 0) return ALIGN_BYTES;
  CHECK_LE(dtype, ALIGN_BYTES);
  return (ALIGN_BYTES + dtype - 1) / dtype;
}

inline int64_t GetMaxAlignBytes(std::unordered_map<std::string, int> dtypes) {
  int64_t min_byte = -1;
  for (auto it : dtypes) {
    if (min_byte == -1 || min_byte > it.second) min_byte = it.second;
  }
  return GetAlignBytes(min_byte);
}

inline Expr CastToExpr(const std::string &value) {
  for (uint i = 0; i < value.length(); ++i) {
    if (value[i] < '0' || value[i] > '9') {
      return Expr(Var(value));
    }
  }
  return Expr(static_cast<int>(std::strtol(value.c_str(), nullptr, 10)));
}

inline Expr CastInt64ToExpr(const int64_t value) { return ktvm::ir::IntImm::make(Int(32), value); }

inline Expr CastIntToExpr(const int value) { return ktvm::ir::IntImm::make(Int(32), value); }

enum TileOpType { VECTOR_OP, CONV_OP, GEMM_OP };

enum TileLevel { LEVEL0 = 0, LEVEL1 };

enum TileVarId { UNDEFINE = -1, VAR };

// Represent an attribute for marking special axes.
struct AttrInfo {
  std::string attr_key;
  std::string attr_value;
};

class TilingAnalyzer;

class TileAxis {
 public:
  TileAxis(TileAxis *p, int i, int da, bool mc, const std::pair<std::string, int> &ds, bool inner, TilingAnalyzer *ta);
  TileAxis(const Expr &l1_size, Expr l0_size, std::string at, TilingAnalyzer *ta, bool inner = false);
  ~TileAxis() {}
  struct Constraint {
    Expr tile_mod_{MIN_TILE};
    Expr tile_min_{MIN_TILE};
    Expr tile_extent_{MIN_TILE};
    std::vector<Expr> cand_factor{};  // list of available factor
  };

  TileAxis *parent{nullptr};
  int index{0};
  int dim_axis{0};
  bool mc_sup{false};
  std::unordered_map<std::string, int> data_size;
  int64_t range_min;
  Expr range_extent;
  Constraint l1_constraints;
  Constraint l0_constraints;
  std::vector<const For *> loops;
  bool forbid_iso;
  bool is_inner;
  bool is_pragma{false};
  std::vector<std::unique_ptr<TileAxis>> children;
  std::vector<std::pair<int64_t, Expr>> tree_ranges;
  int seq_index{0};
  int priority{-1};
  int dyn_shape_limit{-1};
  std::string axis_type_{""};  // record the type of special axis type
  std::vector<AttrInfo> attrs;
  inline Constraint GetConstConstraint(TileLevel level) const {
    Constraint cons = level == LEVEL1 ? this->l1_constraints : this->l0_constraints;
    const auto tile_min = cons.tile_min_.as<IntImm>();
    const auto tile_extent = cons.tile_extent_.as<IntImm>();
    const auto tile_mod = cons.tile_mod_.as<IntImm>();
    Expr const_min = tile_min == nullptr ? -1 : tile_min->value;
    Expr const_extent = tile_extent == nullptr ? -1 : tile_extent->value;
    Expr const_mod = tile_mod == nullptr ? -1 : tile_mod->value;
    std::vector<Expr> const_cand = {};
    for (auto cand : cons.cand_factor) {
      if (const auto imm = cand.as<IntImm>()) const_cand.emplace_back(Expr(imm->value));
    }
    Constraint ret;
    ret.tile_mod_ = const_mod;
    ret.tile_min_ = const_min;
    ret.tile_extent_ = const_extent;
    ret.cand_factor = const_cand;
    return ret;
  }
  inline int64_t GetConstExtent() {
    const auto const_extent = this->range_extent.as<IntImm>();
    if (const_extent == nullptr)
      return -1;
    else
      return const_extent->value;
  }
  void TileRestrainMod(const Expr &mod, TileLevel level);
  void TileRestrainToSingleValue(const Expr &value, TileLevel level);
  void TileRestrainEntire(TileLevel level);

  void LinkToLoop(const For *loop);
  void MarkWithAttr(const AttrInfo &attr);

  bool HasAttr(const std::string &attr_key) const;
  bool HasAttr(const AttrInfo &attr) const;
  bool HasAnyAttr(const std::unordered_set<std::string> &attr_keys) const;
  void RemoveAttr(const std::string &attr_key);
  void RemoveAttr(const AttrInfo &attr);
  std::vector<std::string> GetAttrValue(const std::string &attr_key) const;
  void InsertL1CandFactor(const Expr &f);
  void InsertL0CandFactor(const Expr &f);
  void DumpAxis(bool on_screen = false);

 private:
  TilingAnalyzer *analyzer_{nullptr};
};

class TilingAnalyzer {
 public:
  TilingAnalyzer(Scop *scop, const isl::schedule &sch)
      : scop_(scop),
        body_(scop->GenHalide(sch)),
        binds_(scop->binds_),
        sch_(sch),
        logger_(TileLogger::GetInstance(scop->AddDumpDir("tiling.log"))) {
    if (scop->IsGemm()) {
      op_type_ = GEMM_OP;
    } else if (scop->IsConv()) {
      op_type_ = CONV_OP;
    } else {
      op_type_ = VECTOR_OP;
    }
  }
  TilingAnalyzer(Scop *scop, const isl::schedule &sch, const std::vector<NodeRef> &ct, const std::vector<NodeRef> &ds)
      : scop_(scop),
        body_(scop->GenHalide(sch)),
        binds_(scop->binds_),
        sch_(sch),
        custom_tiling_(ct),
        dynamic_shape_(ds),
        logger_(TileLogger::GetInstance(scop->AddDumpDir("tiling.log"))) {
    if (scop->IsGemm()) {
      op_type_ = GEMM_OP;
    } else if (scop->IsConv()) {
      op_type_ = CONV_OP;
    } else {
      op_type_ = VECTOR_OP;
    }
  }
  ~TilingAnalyzer() = default;

  // represent a buffer
  struct BufferEntry {
    std::string name;
    DavinciMemScope scope;
    Expr shape;           // tensor size
    int64_t size;         // data type size
    int64_t align_size;   // determine the bytes used for alignment
    int64_t expand_size;  // buffer used for reduce or other special purpose will be expanded in future pass
    int alloc_seq;
    std::shared_ptr<std::vector<TileAxis *>> tile_axis;
  };
  // represent a stmt in ir
  struct StmtEntry {
    TileAxis *parent;
    int scope_pair_offset;
    BufferEntry *def;                         // buffer defined in this stmt (write to)
    std::unordered_set<BufferEntry *> ref;    // buffers referred in this stmt (read from)
    std::unordered_set<BufferEntry *> alloc;  // buffers that will be used in this stmt (take up memory space)
  };
  // represent a tilable outer band
  using Band = std::vector<const For *>;
  using VarNames = std::vector<std::string>;
  ktvm::arith::Analyzer arith_ana_;
  ExprSimplifier expr_ac_;
  bool Prepare();

  void ForEachAxisTopDown(const std::function<void(TileAxis *)> &fn, TileAxis *top = nullptr) const;

  TileAxis *RootAxis() const { return root_axis_.get(); }

  inline Stmt Halide() const { return body_; }

  std::vector<TileAxis *> GetAxesContainsAttr(std::string attr_key) const;
  std::vector<TileAxis *> GetAxesOfAttr(std::string attr_key) const;
  std::vector<TileAxis *> GetAxesOfAttr(AttrInfo attr_info) const;

  TileAxis *Axis(const For *loop) const {
    auto it = tile_axis_.find(loop);
    return it != tile_axis_.end() ? it->second : nullptr;
  }
  int GetDataType(const std::string &name) const;
  int GetNumOfAxisInBand(int band_idx) const;

  void DumpLinearSeq();
  void DumpBufferInfo();
  void DumpBufferUsageTimeable();
  static int64_t FindDivisibleTilingFactor(int64_t limit, int64_t range);
  VarNames VisitVarNames(const Expr &arg, VarNames var_names, bool add_num = true);

  TileOpType op_type_;
  Scop *scop_;
  Stmt body_;
  Scop::Binds &binds_;
  isl::schedule sch_;
  std::vector<NodeRef> custom_tiling_{};
  std::vector<NodeRef> dynamic_shape_{};
  TileLogger &logger_;

  std::vector<StmtEntry> linear_seq_{};
  // Axis space get from schedule tree.
  std::unordered_map<const For *, TileAxis *> tile_axis_;
  VarNames NHWCC0 = {"N", "H", "W", "C", "C0"};
  VarNames NCHW = {"N", "C", "H", "W", "C0"};
  VarNames NC1HWC0 = {"N", "C1", "H", "W", "C0"};

  VarNames FTMatrix = {"C1_in", "C1_out", "C0_out", "C0_in"};          //  nZ, Cin = [kc1,kh,kw]
  VarNames FTBACK_Matrix = {"C1_out", "C1_in", "C0_in", "C0_out"};     //  backprop_input, Cout = [kc1,kh,kw]
  VarNames FMMatrix = {"N", "C1_in", "H_in", "W_in", "C0_in"};         // zZ, H_in = [H, Kh], W_in = [W, kw]
  VarNames FMBACK_Matrix = {"N", "C1_out", "H_in", "W_in", "C0_out"};  // zZ, H_in = [H, Kh], W_in = [W, kw]
  VarNames FilterOutput_Matrix = {"C1_out", "kh", "kw", "C1_in", "C0_in", "C0_out"};
  VarNames FilterInput_Matrix = {"N", "C1_out", "H", "W", "C0_out"};
  bool is_dynamic_{false};
  std::unordered_map<TilingAnalyzer::BufferEntry *, std::pair<int, int>> buffer_usage_timetable_;
  std::unordered_map<std::string, std::shared_ptr<BufferEntry>> buf_info_;

 private:
  void AddTilingConstraints();
  std::unique_ptr<TileAxis> root_axis_;
};

class TileCandidate {
 public:
  explicit TileCandidate(TilingAnalyzer *analyzer) : analyzer_(analyzer) {
    for (const auto &attr : analyzer_->RootAxis()->attrs) {
      std::string ub_name = attr.attr_value + "_local_UB";
      if (attr.attr_key == "ELEMWISE")
        this->elem_align_buf.insert(ub_name);
      else if (attr.attr_key == "BROADCAST")
        this->broadcast_align_buf.insert(ub_name);
    }
  }
  ~TileCandidate() = default;
  using BufferEntry = TilingAnalyzer::BufferEntry;
  struct MemInferInfo {
    int64_t live_size[MEM_SCOPE_BULK]{0};
    int64_t actual_live_size[MEM_SCOPE_BULK]{0};
    int64_t max_live_size[MEM_SCOPE_BULK]{0};
    int64_t max_act_live_size[MEM_SCOPE_BULK]{0};
    std::unordered_map<const BufferEntry *, int64_t> live_buf{};
  };
  struct DynamicMemInfo {
    Expr live_size[MEM_SCOPE_BULK]{Expr(0)};
    Expr max_live_size[MEM_SCOPE_BULK]{Expr(0)};
    std::unordered_map<const TilingAnalyzer::BufferEntry *, Expr> live_buf{};
    std::unordered_map<std::string, Var> tile_var_map{};
  };
  struct CalAlignInfo {
    const int64_t tile;
    const int64_t divisor;
    const TileAxis *a;
    const BufferEntry *buf;
    bool is_elem;
    bool is_bcast;
  };
  struct TileVal {
    Expr tile_l1;
    Expr tile_l0;
  };
  struct BufSizeInfo {
    int64_t buf_size;
    int64_t act_buf_size;
    int64_t f_mul;
    bool is_elem;
    bool is_bcast;
  };
  std::unique_ptr<DynamicMemInfo> dynamic_mem_info_{nullptr};
  std::unordered_map<const TileAxis *, TileVal> tile_val_;

  void SetBatchAxis(const std::vector<TileAxis *> &axis);

  void InitTileAxis(TileLevel level);
  void UpdateFixTileAxis(TileLevel level);

  std::vector<TileAxis *> GetTileAxis() { return this->tile_axis_; }
  void ResetTileAxis() { this->tile_axis_.clear(); }
  void ResetTileVal() { this->tile_val_.clear(); }
  void UpdateConstTile(const TileAxis *a, int64_t l1_val, const int64_t l0_val = -1);
  void UpdateL1Tile(const TileAxis *a, const Expr &l1_val);
  void UpdateL0Tile(const TileAxis *a, const Expr &l0_val);
  void UpdateTile(const TileAxis *a, const Expr &l1_val, const Expr &l0_val = Expr());
  std::pair<Expr, Expr> GetTileVal(const TileAxis *a);
  std::pair<int64_t, int64_t> GetConstTileVal(const TileAxis *a);

  bool SpaceVerify(const TileAxis *axis, TileLevel level, int band);
  std::pair<int64_t, int64_t> MemInfer(DavinciMemScope type, int band);

  void InsertAxisBack(TileAxis *a) {
    this->tile_axis_.emplace_back(a);
    this->tile_val_.emplace(a, TileVal{a->l1_constraints.tile_extent_, a->l0_constraints.tile_extent_});
    is_update_ = false;
  }
  int TileAxisSize() const { return static_cast<int>(this->tile_axis_.size()); }
  void UpdateMemoryAfterBuffer(const BufferEntry *buf, MemInferInfo *mem_infer_info);
  bool GetActualBufSize(const BufferEntry *buf, BufSizeInfo *buf_size_info);
  void GetElemwiseActualBufSize(const BufferEntry *buf, BufSizeInfo *buf_size_info);

  int64_t CalActualTile(const CalAlignInfo *align_info);
  void SortByPriority() {
    auto priority_cmp = [](TileAxis *a, const TileAxis *b) {
      if (b->priority <= -1) return false;
      if (a->priority == -1) return true;
      return a->priority > b->priority;
    };
    std::sort(this->tile_axis_.begin(), this->tile_axis_.end(), priority_cmp);
  }
  static int GetCoreNumConf();
  int GetMinFactorToEnableMulticore(TileAxis *axis);
  int GetMaximalPendingBlocks(TileAxis *excluded_axis);
  int GetDmaCopySizeWithinAxis(TileAxis *axis);
  int GetMinFactorForMinDataGranularity(TileAxis *axis);

 private:
  void DoMemInfer();

  std::vector<TileAxis *> tile_axis_;
  TilingAnalyzer *analyzer_;
  bool is_update_{false};
  int tiling_band_{0};
  std::unordered_set<std::string> elem_align_buf;
  std::unordered_set<std::string> broadcast_align_buf;
  int64_t mem_infer_[MEM_SCOPE_BULK]{0};
  int64_t align_mem_infer_[MEM_SCOPE_BULK]{0};
};
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_TILING_ANALYZER_H_
