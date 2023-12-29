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
#ifndef POLY_TILING_SOLVER_H_
#define POLY_TILING_SOLVER_H_

#include "poly/tiling/tiling_analyzer.h"
#include "poly/tiling/tiling_algorithm.h"
#include "poly/tiling/tiling_strategy_manager.h"

namespace akg {
namespace ir {
namespace poly {

constexpr auto BLOCK_TAG = "block";
constexpr auto THREAD_TAG = "thread";

class TilingSolver {
 public:
  explicit TilingSolver(TilingAnalyzer &analyzer) : analyzer_(analyzer), cand_(&analyzer) {}
  ~TilingSolver() {}
  void CollectMemoryLimit();
  void CollectTileAxisTopDown();
  double GetNewAllocRatioWhenFlattenFail(const std::string &error_info);
  double GetNewAllocRatioWhenRewriteFail(int64_t memory_bits);
  TileCandidate *Solve();
  TilingAnalyzer &analyzer_;
  TileCandidate cand_;
  int64_t mem_limit_[MEM_SCOPE_BULK]{0};
  int tiling_band_{0};
  double percentage_ = 0.5;
  double exceed_ratio_ = 1;  // allow memory allocation to exceed memory_size * percentage, may disable double buffer
  bool is_retry_ = false;
};

enum AllocPos { ANY_POS = -1, X_POS, Y_POS, Z_POS, NULL_POS };

class GpuSolver : TilingSolver {
 public:
  explicit GpuSolver(TilingAnalyzer &analyzer) : TilingSolver(analyzer) {}
  ~GpuSolver() {}

  struct Application {
    Application(TileAxis *appl) { applicant = appl; }
    Application(TileAxis *appl, std::vector<int64_t> s, std::vector<int> p) {
      applicant = appl;
      size = s;
      pos = p;
    }
    void Append(int64_t s, int p) {
      size.emplace_back(s);
      pos.emplace_back(p);
    }
    TileAxis *applicant;
    std::vector<int64_t> size;
    std::vector<int> pos;
  };

  struct MapResourceCenter {
    std::vector<int64_t> resource_limit;
    std::vector<int64_t> alloced_slot;
    std::vector<Application> waiting_list;
    std::unordered_map<TileAxis *, size_t> query;
    std::unordered_map<TileAxis *, size_t> alloced_record;
    const int64_t max_total_thread = 1024;

    void Receive(Application appl) {
      if (query.find(appl.applicant) == query.end()) {
        query[appl.applicant] = waiting_list.size();
        waiting_list.push_back(appl);
      } else {
        waiting_list[query[appl.applicant]].size = appl.size;
        waiting_list[query[appl.applicant]].pos = appl.pos;
      }
    }

    void Init(const std::string &custom_map = "") {
      if (!custom_map.empty()) {
        auto res = akg::common::Split(custom_map, " ");
        for (auto s : res) {
          if (s.empty()) {
            continue;
          }
          alloced_slot.emplace_back(StrToDecimalInt64(s));
        }
      }
      for (size_t i = alloced_slot.size(); i < resource_limit.size(); ++i) {
        alloced_slot.emplace_back(-1);
      }
    }

    bool CheckExceedLimit(const int64_t alloc_pos, const int64_t alloc_size, const std::string &alloc_type) {
      if (alloc_pos >= static_cast<int>(alloced_slot.size()) || alloc_pos >= static_cast<int>(resource_limit.size()) ||
          alloc_size > resource_limit[alloc_pos]) {
        return true;
      }

      if (alloc_type == BLOCK_TAG) {
        return false;
      }

      int64_t total_alloc_size = 1;
      for (int i = 0; i < static_cast<int>(alloced_slot.size()); ++i) {
        if (alloc_pos == i) {
          total_alloc_size *= alloc_size;
          continue;
        }

        if (alloced_slot[i] == -1) {
          continue;
        }

        total_alloc_size *= alloced_slot[i];
      }

      if (total_alloc_size > max_total_thread) {
        return true;
      }
      return false;
    }

    int Alloc(size_t index, const std::string &alloc_type, const bool is_reuse_same_band) {
      // check invalid
      if (index >= waiting_list.size()) {
        return -1;
      }

      if (alloced_slot.size() < resource_limit.size()) {
        Init();
      }

      auto appl = waiting_list[index];
      for (int i = 0; i < static_cast<int>(appl.size.size()); ++i) {
        auto alloc_size = appl.size[i];
        auto alloc_pos = appl.pos[i];
        if (alloc_pos == -1 && (i == static_cast<int>(appl.size.size()) - 1)) {
          for (size_t slot_idx = 0; slot_idx < alloced_slot.size(); ++slot_idx) {
            bool is_same_band = false;
            for (auto it : alloced_record) {
              if (it.second != slot_idx) {
                continue;
              }
              // we can reuse slots between different bands
              is_same_band |= (it.first->index == appl.applicant->index);
            }
            if (alloced_slot[slot_idx] == alloc_size && !is_same_band) {
              waiting_list[index].pos[i] = slot_idx;
              return i;
            }
          }
          // if it is the last hope for axis, try any pos
          while (alloc_pos < static_cast<int>(AllocPos::Z_POS)) {
            alloc_pos++;
            waiting_list[index].pos[i] = alloc_pos;
            if (Alloc(index, alloc_type, is_reuse_same_band) != -1) {
              return i;
            }
          }
          return -1;
        }
        if (alloc_type == BLOCK_TAG && alloc_size > resource_limit[alloc_pos]) {
          alloc_size = resource_limit[alloc_pos];
        }
        if (CheckExceedLimit(alloc_pos, alloc_size, alloc_type)) {
          continue;
        }

        if (alloced_slot[alloc_pos] == -1) {
          // not alloced yet
          alloced_slot[alloc_pos] = alloc_size;
          return i;
        }
        bool is_same_band = false;
        for (auto it : alloced_record) {
          if (static_cast<int>(it.second) == alloc_pos) {
            // we can reuse slots between different bands
            is_same_band |= (it.first->index == appl.applicant->index);
          }
        }
        if (is_same_band) {
          if (!is_reuse_same_band) {
            return -1;
          }

          auto reuse_alloc_size = alloced_slot[alloc_pos] * alloc_size;
          if (CheckExceedLimit(alloc_pos, reuse_alloc_size, alloc_type)) {
            continue;
          }
          alloced_slot[alloc_pos] = reuse_alloc_size;
          return i;
        }

        // Determine whether different bands are reused.
        if (alloced_slot[alloc_pos] >= alloc_size) {
          // e.g original alloc 32, current alloc 8, do not update.
          return i;
        } else {
          // update to larger size
          // e.g original alloc 8, current alloc 32, update to 32
          alloced_slot[alloc_pos] = alloc_size;
          return i;
        }
      }
      return -1;
    }

    std::string Show() {
      std::stringstream ss;
      for (auto &it : waiting_list) {
        ss << "Applicant " << it.applicant->index << "_" << it.applicant->dim_axis << "\n";
        ss << "  Size:[";
        for (size_t i = 0; i < it.size.size(); ++i) {
          ss << "(" << it.size[i] << "," << it.pos[i] << ")";
        }
        ss << "]\n";
      }
      return ss.str();
    }
  };

  TileCandidate *Solve();
  void SolveMapping();

 private:
  void DetermineTileFactor(TileAxis *axis, const TileLevel &level);
  void InnerThreadOuterBlock();
  AllocPos GetThreadAllocPos(TileAxis *axis);
  AllocPos GetBlockAllocPos(TileAxis *axis);
  MapResourceCenter thread_center_;
  MapResourceCenter block_center_;

  int64_t max_x_dim_block_ = pow(2, 31) - 1;
  int64_t max_y_z_dim_block_ = 65535;
  int64_t max_x_y_dim_thread_ = 1024;
  int64_t max_z_dim_thread_ = 64;
  void TotSpeedup();
  int CalculateBoxSize(const std::string &name);
};

class InequalitySolver : TilingSolver {
 public:
  explicit InequalitySolver(TilingAnalyzer &analyzer) : TilingSolver(analyzer) {}
  ~InequalitySolver() {}
  TileCandidate *Solve();
  std::deque<ParamInfo> param_info_{};
  Array<Expr> GetMemoryConstraints() { return memory_constraints_; }

 private:
  struct TilingMemInfo {
    Expr live_size[MEM_SCOPE_BULK]{Expr(0)};
    Expr max_live_size[MEM_SCOPE_BULK]{Expr(0)};
    std::unordered_map<const TilingAnalyzer::BufferEntry *, Expr> live_buf{};
    std::unordered_map<std::string, Var> tile_var_map{};
  };

  void InitTileAxis(const TileLevel &level);
  Expr SolveMemoryConstraint(const Array<Expr> &memory_constraints, const Var var);
  void DetermineTileFactor(TileAxis *axis, const TileLevel &level, const Array<Expr> &memory_constraints);
  Expr SolveTileResult(const Expr &to_tile, const Array<Expr> &memory_constraints, const TileAxis::Constraint &cons);
  void SolveTileRanges(Expr &shape_range, Expr &tile_min, Expr &tile_range, const TileLevel &level,
                       const TileAxis *axis, const Expr &l1_expr);
  void GoToStaticFactor(Expr &final_factor_expr, const Expr &mem_constraint, const Expr &tile_range,
                        const TileLevel &level, TileAxis *axis);

  Expr SolveByInferBound(const Array<Expr> &cons_on_var, const Var tiling_var);
  int64_t DetermineTileForStatic(TileAxis *axis, const Expr &mem_limit, const Expr &tile_range, const TileLevel &level);
  void GoWithCandidates(int64_t &final_factor, int64_t static_mem_constraint, const TileAxis::Constraint &cons);
  void GoWithConstraints(int64_t &final_factor, int64_t static_mem_constraint, int64_t static_shape,
                         const TileAxis::Constraint &cons, const TileAxis *axis, const TileLevel &level);

  int64_t PostprocessFinalFactor(int64_t final_factor, TileAxis *axis);

  Expr DetermineTileForDynamic(const TileAxis *axis, const Expr &mem_constraint, const Expr &to_tile,
                               const Expr &shape_range, const Expr &tile_range, const TileLevel &level);
  void AppendShapeLimitConstraint(const TileAxis *axis, const Expr &to_tile);

  void UpdateMemInfo();
  void UpdateMemInfoWithBufReuse();

  void CalculateMemoryInBuffer(const TilingAnalyzer::BufferEntry *buf, TilingMemInfo *mem_info);
  Expr EstimateAlignment(const TilingAnalyzer::BufferEntry *buf, const TileAxis *axis, const Expr &tile) const;

  void CollectMemoryConstraints();

  bool ContainVar(Expr expr, Var var);
  Expr GetSubstitutedExpr(const NodeRef &op);

  Array<Expr> memory_constraints_;
  Map<Var, Expr> defined_vars_{};
  bool tile_success_{true};
  std::unique_ptr<TilingMemInfo> tiling_mem_info_{nullptr};
  std::unordered_map<int, std::string> memory_map_ = {{1, BUF}, {2, C1},       {3, C0A},    {4, C0B},
                                                      {5, C0C}, {6, "SHARED"}, {7, "LOCAL"}};
};

class DynamicShapeSolver : TilingSolver {
 public:
  explicit DynamicShapeSolver(TilingAnalyzer &analyzer) : TilingSolver(analyzer), solver_(analyzer) {}
  ~DynamicShapeSolver() {}
  TileCandidate *Solve();
  std::deque<ParamInfo> GetParamInfo();

  void AppendTileConstraintInIR(TileCandidate *cand, TileLevel level);

 private:
  InequalitySolver solver_;
};

class TraverseSolver : TilingSolver {
 public:
  explicit TraverseSolver(TilingAnalyzer &analyzer) : TilingSolver(analyzer) {}
  ~TraverseSolver() {}
  TileCandidate *Solve();
  std::vector<TileAxis *> GetSpecTileAxis();

 private:
  struct TileInfo {
    TileInfo(TileAxis *a, TileLevel l, int b) : axis(a), level(l), band(b) {}
    TileAxis *axis;
    TileLevel level;
    int band;
    int64_t min_tile = 0;
    int64_t deviation = 0;
  };
  struct MemoryInferInfo {
    int64_t dev;
    int64_t val;
  };
  std::unique_ptr<MemoryInferInfo> cur_iso_info_;
  std::unique_ptr<MemoryInferInfo> cur_no_iso_info_;
  bool IsTilable(TileInfo *info);
  bool MemoryVerify(TileLevel level, int band, int64_t *deviation = nullptr);
  bool DoTiling(const TileInfo *info);
  bool GoWithCandidates(const TileInfo *info, const TileAxis::Constraint &cons, TileAxis *axis, int64_t deviation,
                        int dst);

  void InitMemoryInferInfo();
  void UpdateChosenValue(int64_t tail, int64_t deviation, int64_t tile_size, TileAxis *axis);
  void UpdateTile(const TileInfo *info, TileAxis *axis, int64_t tile_size);

  int64_t PostprocessFinalFactor(int64_t final_factor, TileAxis *axis);
  void AppendConvPragma();
  void AppendConvBackpropPragma();
  void SolveConvCache0();
  void RestrainConvBackInputTileK(TileAxis *k_axis) const;
  void CreateSpecgemmTileAxis(Expr mo, Expr no, Expr ko, bool cut_reduce);
  void CreateConvPragma(const Expr &co_cut, Expr tile_out_h, Expr tile_out_w, Expr kh_cut, Expr kw_cut, Expr ci_cut,
                        const Expr &batch_cut);
  TileAxis *GeneratePragmaAxes(const Expr &size, const std::string &type, bool is_pragma);
  std::vector<TileAxis *> spec_tile_axis_;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_TILING_SOLVER_H_
