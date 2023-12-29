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

#ifndef POLY_SCHTREE_ANALYZER_H_
#define POLY_SCHTREE_ANALYZER_H_

#include <tvm/ir.h>
#include <deque>
#include <vector>
#include <unordered_map>
#include <utility>
#include <string>
#include <memory>
#include <algorithm>
#include <cmath>
#include <unordered_set>

#include "poly/scop_builder.h"
#include "poly/tiling/tiling_analyzer.h"

namespace akg {
namespace ir {
namespace poly {
class ScheduleTreeAnalyzer {
 public:
  ScheduleTreeAnalyzer(TilingAnalyzer *a, const isl::schedule &s);
  ~ScheduleTreeAnalyzer() = default;

  std::unique_ptr<TileAxis> root_{nullptr};

  std::unique_ptr<TileAxis> Build(const Stmt &stmt);

  void AnalyzeCubeInfo();

 private:
  TilingAnalyzer *analyzer_{nullptr};
  isl::schedule sch_;

  struct TilePos {
    bool is_outer;
    size_t var_pos;
    std::string stmt_name;
    std::string var_name;
    std::string actual_name;
    int64_t min_range;
    Expr max_range;
    bool mc_sup;
  };
  struct TileNode {
    bool is_outer;
    int index;          // Band id
    size_t axis;        // Tile seq
    int64_t range_min;  // Minimum value
    Expr range_max;     // Maximum value
    size_t var_pos;     // Corresponding position of tiled loop in stmt, map to loop
    bool mc_sup;        // Whether support multi-core
    const For *loop;    // Related loop in HalideIR
    std::pair<std::string, int> data_size;
    std::string fractal;
  };
  std::vector<TileNode> tile_nodes_;
  std::unordered_map<size_t, size_t> tile_size_in_band_;

  std::vector<isl::schedule_node_band> outer_bands_;
  std::unordered_map<std::string, std::vector<std::pair<int64_t, Expr>>> dim_range_;
  std::vector<Band> band_list_;
  std::vector<Band> tileable_band_;
  std::vector<Band> untileable_band_;
  std::unordered_map<const For *, std::vector<const Provide *>> provides_map_;
  std::unordered_map<const For *, std::vector<const IfThenElse *>> ifs_map_;
  std::unordered_map<const For *, std::vector<std::pair<int64_t, int64_t>>> loop_range_map_;
  std::unordered_map<const For *, std::vector<std::pair<int64_t, std::string>>> loop_dynamic_range_map_;
  std::vector<const For *> loop_seq_;
  std::unordered_map<const For *, std::pair<std::string, int>> loop_data_size_map_;
  std::unordered_map<size_t, std::vector<TilePos>> candidates_;
  std::unordered_map<std::string, std::string> mmu_var_map_;
  std::vector<const For *> defined_static_loop_;
  std::vector<const For *> defined_dynamic_loop_;

  bool AnalyzeScheduleTree();
  void GetDimRangeFromTree(const isl::schedule &sch);
  void GetCandidatesInSequence(size_t seq, const isl::pw_aff_list &pa_list, bool is_outer = true, bool mc_sup = false);
  void UpdateCandidates(std::vector<std::string> all_var_names, const std::string &var, const std::string &pa_name,
                        size_t seq, bool is_outer, bool mc_sup);

  bool GetPosShiftedTileRange(const std::string &vname, const std::string &actual_name,
                              std::pair<int, int> &old_ranges);
  bool GetNegShiftedTileRange(const std::string &vname, const std::string &actual_name,
                              std::pair<int, int> &old_ranges);
  void ConstructTreePattern(int band_id);

  void AnalyzeHalide(const Stmt &stmt);
  void AddLoopRangeFromBand();
  void AddLoopRangeFromIfs();
  std::vector<Expr> SeparateAndInCondition(const Expr &condition);
  void AddLoopRangeInConditions(std::vector<Expr> if_expr, const For *loop);
  void DecodeGreaterEqual(const GE *ge, const For *loop);
  void DecodeLessEqual(const LE *le, const For *loop);

  void AddLoopDataSize();
  bool MatchNodeWithLoop(std::unordered_set<const For *> &matched, TileNode &node, const For *loop);
  void UpdateMatchedNode(std::unordered_set<const For *> &matched, TileNode &node, const For *loop);

  bool MatchNodeWithDynamicLoop(std::unordered_set<const For *> &matched, TileNode &node, const For *loop);
  static int GetLayerIndex(const std::string &var_name);

  void TryMatchTileNodes();
  void TrySortTileNodes();

  void CreateTileAxes();

  const For *GetSameNameLoop(const For *loop);
  TileAxis *CreateStaticUndefinedLoop(const For *loop, TileAxis *last_axis);
  TileAxis *CreateDynamicUndefinedLoop(const For *loop, TileAxis *last_axis);

  void CreateAxisForUndefinedLoop(TileAxis *last_axis);

  void RecordTreeRanges(TileAxis *axis, const For *loop);

  VarNames GetConvVarInArg(Expr arg, bool add_num);
  void MatchConvFilterVarNames(const Call *call);
  void MatchConvVarNames(const Call *call);
  void MatchGemmVarNames(std::vector<const Call *> op_list);
  Band GetPreviousLoops(const For *loop);

  std::vector<const Call *> GetCallListInProvides(std::vector<const Provide *> pros, const std::string &target_name);
  void SortMatrixInCBAOrder(std::vector<const Call *> &op_list);
  void MatchCubeVarNames(std::vector<const Call *> op_list);
  void AnalyzeAxisType(const For *loop);
};
}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_SCHTREE_ANALYZER_H_
