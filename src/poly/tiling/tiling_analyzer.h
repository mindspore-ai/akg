/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "pass/expr_alg_simplify.h"
#include "pass/utils.h"
#include "poly/scop_info.h"
#include "poly/tiling/tiling_utils.h"
#include "poly/poly_util.h"

namespace akg {
namespace ir {
namespace poly {
// common integers
constexpr auto ALIGN_BYTES = 32;
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
constexpr auto DESIRE_CORE_GRANULARITY = 8192;
constexpr auto MIN_EXEC_NUM_PER_THREAD = 4096;
constexpr auto BEST_PARALLEL_NUM = 192;
constexpr auto PARALLEL_DECREASE_VALUE = 8;
constexpr auto BEST_UNROLL_NUM = 256;
constexpr auto MIN_UNROLL_NUM = 8;
constexpr auto MATMUL_BEST_FACTOR = 128;
constexpr auto MATMUL_AXIS_M = 0;
constexpr auto MATMUL_AXIS_N = 1;
constexpr auto MATMUL_AXIS_K = 2;
constexpr auto TOT_BEST_NUM_PER_BLOCK = 1024;
constexpr auto OUTERMOST_AXIS = 0;
constexpr auto CPU_CSR_TILING_FACTOR = 1;
constexpr auto CPU_CSR_PARALLEL_CUTOFF = 4096;

// Controlled by custom tiling.
constexpr auto ALLOCATION_PERCENTAGE = 0.5;  // reserved for double buffer in default

inline int64_t GetAlignBytes(const int64_t dtype) {
  if (dtype > 0) {
    CHECK_LE(dtype, ALIGN_BYTES);
    return (ALIGN_BYTES + dtype - 1) / dtype;
  } else {
    return ALIGN_BYTES;
  }
}

inline int64_t GetMinBytes(std::unordered_map<std::string, std::vector<int>> dtypes) {
  int64_t min_byte = -1;
  for (const auto &it : dtypes) {
    if (it.second.empty()) {
      continue;
    }
    int min_elem = *min_element(it.second.begin(), it.second.end());
    if (min_byte == -1 || min_byte > min_elem) {
      min_byte = min_elem;
    }
  }
  return min_byte;
}

inline int64_t GetMaxAlignBytes(std::unordered_map<std::string, std::vector<int>> dtypes) {
  return GetAlignBytes(GetMinBytes(dtypes));
}

inline Expr CastToExpr(const std::string &value) {
  for (uint i = 0; i < value.length(); ++i) {
    if (value[i] < '0' || value[i] > '9') {
      return Expr(Var(value));
    }
  }
  return Expr(static_cast<int>(std::strtol(value.c_str(), nullptr, 10)));
}

inline Expr CastInt64ToExpr(const int64_t value) { return air::ir::IntImm::make(Int(32), value); }

inline Expr CastIntToExpr(const int value) { return air::ir::IntImm::make(Int(32), value); }

enum class TileOpType { VECTOR_OP, CONV_OP, GEMM_OP };

enum TileVarId { UNDEFINE = -1, VAR };

enum SpItemPerThread { FULL = -1, AUTO };

// Represent an attribute for marking special axes.
struct AttrInfo {
  std::string attr_key;
  std::string attr_value;
};

// valid attr_key used in AttrInfo
constexpr auto AT_VECTORIZED = "VECTORIZED";
constexpr auto AT_TOT = "TOT";
constexpr auto AT_ALIGN = "ALIGN";
constexpr auto AT_DMA = "DMA";
constexpr auto AT_DMA2 = "DMA2";
constexpr auto AT_DMA3 = "DMA3";
constexpr auto AT_OP_TYPE = "OP_TYPE";

constexpr auto AT_REDUCE_DST_LAST = "REDUCE_DST_LAST";
constexpr auto AT_REDUCE_SRC_LAST = "REDUCE_SRC_LAST";
constexpr auto AT_TRANSPOSE_INNERMOST_AXIS = "TRANSPOSE_INNERMOST_AXIS";
constexpr auto AT_BROADCAST_INNERMOST_AXIS = "BROADCAST_INNERMOST_AXIS";
constexpr auto AT_REDUCE_FLOW = "REDUCE_FLOW";
constexpr auto AT_REDUCE_AXIS = "REDUCE_AXIS";
constexpr auto AT_COUNT_AXIS = "COUNT_AXIS";
constexpr auto AT_POST_FUSION_REDUCE_TENSOR = "POST_FUSION_REDUCE_TENSOR";
constexpr auto AT_CONV = "CONV";
constexpr auto AT_GEMM = "GEMM";
constexpr auto AT_ATTRIBUTE = "ATTRIBUTE";
constexpr auto AT_SHIFT = "SHIFT";
constexpr auto AT_MODSHIFT = "MODSHIFT";
constexpr auto AT_DYNAMIC_SHIFT = "DYNAMIC_SHIFT";
constexpr auto AT_DYNAMIC_BOUND = "DYNAMIC_BOUND";
constexpr auto AT_MOD = "MOD";
constexpr auto AT_CAST = "CAST";
constexpr auto AT_MEM_RATIO = "MEM_RATIO";
constexpr auto AT_THREAD_MIN = "THREAD_MIN";
constexpr auto AT_THREAD_MAX = "THREAD_MAX";
constexpr auto AT_THREAD_MOD = "THREAD_MOD";
constexpr auto AT_BLOCK_MIN = "BLOCK_MIN";
constexpr auto AT_BLOCK_MAX = "BLOCK_MAX";
constexpr auto AT_BLOCK_MOD = "BLOCK_MOD";

class TilingAnalyzer;

class TileAxis {
 public:
  TileAxis(TileAxis *p, int i, int da, bool mc, const std::pair<std::string, int> &ds, bool inner, TilingAnalyzer *ta);
  TileAxis(const Expr &l1_size, const Expr &l0_size, const std::string &at, TilingAnalyzer *ta, bool inner = false);
  ~TileAxis() {}
  struct Constraint {
    Expr tile_mod_{MIN_TILE};
    Expr tile_min_{MIN_TILE};
    Expr tile_extent_{MIN_TILE};
    std::vector<Expr> cand_factor{};  // list of available factor
  };
  struct MappingConstraint {
    int64_t map_mod_{MIN_TILE};
    int64_t map_min_{MIN_TILE};
    int64_t map_extent_{0};           // 0 means it is not determined yet
    int64_t item_process_{MIN_TILE};  // for thread only, equals to tile / thread_num
    std::vector<int64_t> map_cand_{};
  };
  TileAxis *parent{nullptr};
  int index{0};
  int dim_axis{0};
  bool mc_sup{false};
  std::unordered_map<std::string, std::vector<int>> data_size;
  int64_t range_min;
  Expr range_extent;
  int64_t extent_val;
  Constraint c1_constraints;
  Constraint c0_constraints;
  MappingConstraint block_constraints;
  MappingConstraint thread_constraints;
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
  std::unordered_map<std::string, Var> var_names;  // used in inequality solver to record unique var address
  inline Constraint GetConstConstraint(TileLevel level) const {
    Constraint cons = level == CACHE1 ? this->c1_constraints : this->c0_constraints;
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
  void TileRestrainUpper(const Expr &value, TileLevel level);
  void TileRestrainLower(const Expr &value, TileLevel level);
  void TileRestrainToSingleValue(const Expr &value, TileLevel level);
  void TileRestrainEntire(TileLevel level);

  void LinkToLoop(const For *loop);
  void MarkWithAttr(const AttrInfo &attr);

  bool HasAttr(const std::string &attr_key, const bool partial_match = false) const;
  bool HasAttr(const AttrInfo &attr) const;
  bool HasAnyAttr(const std::unordered_set<std::string> &attr_keys, const bool partial_match = false) const;
  void RemoveAttr(const std::string &attr_key);
  void RemoveAttr(const AttrInfo &attr);
  std::vector<std::string> GetAttrValue(const std::string &attr_key) const;
  void InsertC1CandFactor(const Expr &f);
  void InsertC0CandFactor(const Expr &f);
  void DumpAxis(bool on_screen = false);

 private:
  TilingAnalyzer *analyzer_{nullptr};
};

class TilingAnalyzer {
 public:
  TilingAnalyzer(const isl::schedule &sch, ScopInfo &scop_info, Stmt body)
      : body_(body),
        binds_(scop_info.user_config_.GetBind()),
        sch_(sch),
        scop_info_(scop_info),
        is_retry_(!g_attrs.GetStr(kErrorInfo, "").empty()) {
    if (scop_info.mmu_info_.IsGemm()) {
      op_type_ = TileOpType::GEMM_OP;
    } else if (scop_info.mmu_info_.IsConv()) {
      op_type_ = TileOpType::CONV_OP;
    } else {
      op_type_ = TileOpType::VECTOR_OP;
    }
  }

  ~TilingAnalyzer() = default;

  // represent a buffer
  struct BufferEntry {
    std::string name;
    TilingMemScope scope;
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

  air::arith::Analyzer arith_ana_;
  ExprSimplifier expr_ac_;
  bool Prepare();

  void ForEachAxisTopDown(const std::function<void(TileAxis *)> &fn, TileAxis *top = nullptr) const;

  TileAxis *RootAxis() const { return root_axis_.get(); }

  inline Stmt Halide() const { return body_; }

  std::vector<TileAxis *> GetAxesContainsAttr(const std::string &attr_key) const;
  std::vector<TileAxis *> GetAxesOfAttr(const std::string &attr_key, int band_index = -1) const;
  std::vector<TileAxis *> GetAxesOfAttr(const AttrInfo &attr_info, int band_index = -1) const;

  TileAxis *Axis(const For *loop) const {
    auto it = tile_axis_.find(loop);
    return it != tile_axis_.end() ? it->second : nullptr;
  }
  int GetNumOfAxisInBand(int band_idx) const;

  std::vector<int> GetSortedBands() const;

  void DumpLinearSeq();
  void DumpBufferInfo();
  void DumpBufferUsageTimeable();
  static int64_t FindDivisibleTilingFactor(int64_t limit, int64_t range);
  TileLogger &GetTileLogger() const {
    CHECK(logger_);
    return *(logger_.get());
  }
  void UpdateBindingSpace(TileAxis::MappingConstraint constraint) { binding_spaces_.emplace_back(constraint); }
  Stmt body_;
  Binds &binds_;
  isl::schedule sch_;
  ScopInfo &scop_info_;
  std::unique_ptr<TileLogger> logger_;
  TileOpType op_type_;

  std::vector<StmtEntry> linear_seq_{};
  // Axis space get from schedule tree.
  std::unordered_map<const For *, TileAxis *> tile_axis_;

  std::unordered_map<TilingAnalyzer::BufferEntry *, std::pair<int, int>> buffer_usage_timetable_;
  std::unordered_map<std::string, std::shared_ptr<BufferEntry>> buf_info_;
  bool is_retry_{false};
  std::vector<TileAxis::MappingConstraint>
    binding_spaces_;  // [thread.x[min, max, mod], thread.y, thread.z, block.x, block.y, block.z]
 private:
  void AddTilingConstraints();
  void AddPostTilingConstraints();
  std::unique_ptr<TileAxis> root_axis_;
};

class TileCandidate {
 public:
  explicit TileCandidate(TilingAnalyzer *analyzer) : analyzer_(analyzer) {
    if (analyzer_->RootAxis() == nullptr || analyzer_->scop_info_.user_config_.GetTarget() != TARGET_CCE) {
      return;
    }
    for (const auto &attr : analyzer_->RootAxis()->attrs) {
      std::string buf_name = attr.attr_value + LOCAL_BUF;
      if (attr.attr_key == AT_ELEMWISE) {
        this->elem_align_buf_.insert(buf_name);
      } else if (attr.attr_key == AT_BROADCAST) {
        this->broadcast_align_buf_.insert(buf_name);
      }
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
    Expr tile_c1;
    Expr tile_c0;
  };
  struct BufSizeInfo {
    int64_t buf_size;
    int64_t act_buf_size;
    int64_t f_mul;
    bool is_elem;
    bool is_bcast;
  };

  struct BufferUsage {
    void Build(std::unordered_map<TilingAnalyzer::BufferEntry *, std::pair<int, int>> time_table) {
      max_time = static_cast<int>(time_table.size() - 1);
      for (auto cur_time = 0; cur_time <= max_time; ++cur_time) {
        for (auto it : time_table) {
          auto alloc_time = it.second.first;
          if (buf_alloc_time.find(alloc_time) == buf_alloc_time.end()) {
            buf_alloc_time[alloc_time] = {it.first};
          } else {
            buf_alloc_time[alloc_time].insert(it.first);
          }
          auto release_time = it.second.second + 1;
          if (buf_release_time.find(release_time) == buf_release_time.end()) {
            buf_release_time[release_time] = {it.first};
          } else {
            buf_release_time[release_time].insert(it.first);
          }
        }
      }
    }
    int max_time;
    std::unordered_map<int, std::unordered_set<TilingAnalyzer::BufferEntry *>> buf_alloc_time;
    std::unordered_map<int, std::unordered_set<TilingAnalyzer::BufferEntry *>> buf_release_time;
  };

  std::unique_ptr<BufferUsage> buffer_usage_{nullptr};
  std::unique_ptr<DynamicMemInfo> dynamic_mem_info_{nullptr};
  std::unordered_map<const TileAxis *, TileVal> tile_val_;

  void SetBatchAxis(const std::vector<TileAxis *> &axis);

  void InitTileAxis(TileLevel level);
  void UpdateFixTileAxis(TileLevel level);

  std::vector<TileAxis *> GetTileAxis() { return this->tile_axis_; }
  void ResetTileAxis() { 
    this->tile_axis_.clear(); 
  }
  void ResetTileVal() { 
    this->tile_val_.clear(); 
  }
  void UpdateConstTile(const TileAxis *a, int64_t c1_val, int64_t c0_val = -1);
  void UpdateC1Tile(const TileAxis *a, const Expr &c1_val);
  void UpdateC0Tile(const TileAxis *a, const Expr &c0_val);
  void UpdateTile(const TileAxis *a, const Expr &c1_val, const Expr &c0_val = Expr());
  std::pair<Expr, Expr> GetTileVal(const TileAxis *a);
  std::pair<int64_t, int64_t> GetConstTileVal(const TileAxis *a);

  bool SpaceVerify(const TileAxis *axis, TileLevel level, int band);
  std::pair<int64_t, int64_t> MemInfer(TilingMemScope type, int band);

  void InsertAxisBack(TileAxis *a) {
    this->tile_axis_.emplace_back(a);
    this->tile_val_.emplace(a, TileVal{a->c1_constraints.tile_extent_, a->c0_constraints.tile_extent_});
    is_update_ = false;
  }
  int TileAxisSize() const { return static_cast<int>(this->tile_axis_.size()); }
  void UpdateMemoryAfterBuffer(const BufferEntry *buf, MemInferInfo *mem_infer_info);
  bool GetActualBufSize(const BufferEntry *buf, BufSizeInfo *buf_size_info);
  void GetElemwiseActualBufSize(const BufferEntry *buf, BufSizeInfo *buf_size_info);

  int64_t CalActualTile(const CalAlignInfo *align_info);
  void SortByPriority() {
    if (analyzer_->scop_info_.user_config_.GetTarget() != TARGET_CCE) {
      return;
    }
    auto priority_cmp = [](TileAxis *a, const TileAxis *b) {
      if (b->priority <= -1) {
        return false;
      }
      if (a->priority == -1) {
        return true;
      }
      return a->priority > b->priority;
    };
    std::sort(this->tile_axis_.begin(), this->tile_axis_.end(), priority_cmp);
  }
  int GetMinFactorToEnableMulticore(TileAxis *axis);
  int GetMaximalPendingBlocks(TileAxis *excluded_axis);
  int GetDmaCopySizeWithinAxis(TileAxis *target_axis);
  int GetMinFactorForMinDataGranularity(TileAxis *axis);

 private:
  void DoMemInfer();

  std::vector<TileAxis *> tile_axis_;
  TilingAnalyzer *analyzer_;
  bool is_update_{false};
  int tiling_band_{0};
  std::unordered_set<std::string> elem_align_buf_;
  std::unordered_set<std::string> broadcast_align_buf_;
  int64_t mem_infer_[MEM_SCOPE_BULK]{0};
  int64_t align_mem_infer_[MEM_SCOPE_BULK]{0};
};
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_TILING_ANALYZER_H_
