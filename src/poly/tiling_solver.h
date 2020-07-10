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

#include "poly/tiling_analyzer.h"
#include "poly/tiling_algorithm.h"
#include "poly/tiling_strategy_manager.h"

namespace akg {
namespace ir {
namespace poly {

class TilingSolver {
 public:
  explicit TilingSolver(TilingAnalyzer &analyzer) : analyzer_(analyzer), cand_(&analyzer) {}
  ~TilingSolver() {}
  void CollectMemoryLimit();
  void CollectTileAxisTopDown();
  TileCandidate *Solve();

  TilingAnalyzer &analyzer_;
  TileCandidate cand_;
  int64_t mem_limit_[MEM_SCOPE_BULK]{0};
  int tiling_band_{0};
  double percentage_ = 0.5;
  double exceed_ratio_ = 1;  // allow memory allocation to exceed memory_size * percentage, may disable double buffer
};

class InequalitySolver : TilingSolver {
 public:
  explicit InequalitySolver(TilingAnalyzer &analyzer) : TilingSolver(analyzer) {}
  ~InequalitySolver() {}
  TileCandidate *Solve();
  std::deque<Scop::ParamInfo> param_info_{};

 private:
  struct TilingMemInfo {
    Expr live_size[MEM_SCOPE_BULK]{Expr(0)};
    Expr max_live_size[MEM_SCOPE_BULK]{Expr(0)};
    std::unordered_map<const TilingAnalyzer::BufferEntry *, Expr> live_buf{};
    std::unordered_map<std::string, Var> tile_var_map{};
  };

  void InitTileAxis(TileLevel level);
  Expr SolveMemoryConstraint(const Array<Expr> &memory_constraints, const Var var);
  void DetermineTileFactor(TileAxis *axis, TileLevel level, const Array<Expr> &memory_constraints);
  Expr SolveByInferBound(const Array<Expr> &cons_on_var, const Var tiling_var);
  int64_t DetermineTileForStatic(TileAxis *axis, const Expr &mem_limit, const Expr &tile_range, TileLevel level);
  Expr DetermineTileForDynamic(TileAxis *axis, const Expr &mem_constraint, const Expr &to_tile, const Expr &shape_range,
                               const Expr &tile_range, TileLevel level);
  void AppendShapeLimitConstraint(TileAxis *axis, Expr to_tile);

  void UpdateMemInfo();
  void UpdateMemInfoWithBufReuse();

  void CalculateMemoryInBuffer(const TilingAnalyzer::BufferEntry *buf, TilingMemInfo *mem_info);
  Expr EstimateAlignment(const TilingAnalyzer::BufferEntry *buf, TileAxis *axis, Expr tile) const;

  Array<Expr> CollectMemoryConstraints();

  bool ContainVar(Expr expr, Var var);
  Expr GetSubstitutedExpr(const NodeRef &op);

  Map<Var, Expr> defined_vars_{};
  bool tile_success_{true};
  std::unique_ptr<TilingMemInfo> tiling_mem_info_{nullptr};
};

class DynamicShapeSolver : TilingSolver {
 public:
  explicit DynamicShapeSolver(TilingAnalyzer &analyzer) : TilingSolver(analyzer), solver_(analyzer) {}
  ~DynamicShapeSolver() {}
  TileCandidate *Solve();
  std::deque<Scop::ParamInfo> GetParamInfo();

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
  bool IsTilable(TileInfo *info);
  bool MemoryVerify(TileLevel level, int band, int64_t *deviation = nullptr);
  bool DoTiling(const TileInfo *info);
  int64_t PostprocessFinalFactor(int64_t final_factor, TileAxis *axis);
  void AppendConvPragma();
  void AppendConvBackpropPragma();
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
