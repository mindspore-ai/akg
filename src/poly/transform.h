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
#ifndef POLY_TRANSFORM_H_
#define POLY_TRANSFORM_H_

#pragma once
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>

#include "isl.h"
#include "scop.h"
#include "ir_pass.h"

#define MAX_STRIDE 65535

namespace akg {
namespace ir {
namespace poly {
enum Tile_type {
  L0 = 0,
  L1,
  UB,
  UBL1,
  UBL0,
  L1UBL1,
  Invalid,
};

isl::union_map DependenceAnalysis(const isl::union_map &sources, const isl::union_map &targets,
                                  const isl::union_map &kills, const isl::union_map &sch);

class Dependency {
 private:
  isl::id start_node_id_;
  isl::id end_node_id_;
  int64_t edge_weight_;

 public:
  Dependency(const isl::id start_node_id, const isl::id end_node_id, const int64_t edge_weight)
      : start_node_id_(start_node_id), end_node_id_(end_node_id), edge_weight_(edge_weight) {}
  ~Dependency() {}

  isl::id GetStartNode() { return start_node_id_; }
  isl::id GetEndNode() { return end_node_id_; }
  int64_t GetEdgeWeight() const { return edge_weight_; }
};

struct PointBandInfo {
  isl::multi_union_pw_aff mupa;
  size_t n_member{0};
  bool permutable{false};
  std::vector<bool> coincident;
};

void CheckAndRemoveUninitializedCopyin(isl::union_map &copy_in, const Scop::Binds &binds);
isl::schedule SplitOuterBand(const isl::schedule &curr_schedule);
bool IsStmtScheduleContainsReduceAxis(const isl::pw_aff &stmt, const std::unordered_set<std::string> &reduce_axis_list);
bool IsDimScheduleContainsReduceAxis(const isl::union_pw_aff &schedule, const ReduceStmtMap &reduce_stmts);
isl::schedule ResetCoincidenceOfReduceAxis(const isl::schedule &schedule, const ReduceStmtMap &reduce_stmts);
std::vector<bool> getIsolateVector(const isl::schedule_node_band &node);
bool InjectMulticoreToBand(isl::schedule_node &band_node);
isl::schedule_node &ObtainSequenceOrSetNodeAncestor(isl::schedule_node &node);
bool InjectMulticoreToChildrenBands(isl::schedule_node &sequence_node);

// class for hold transformations on schedule node
class Transform {
 public:
  Transform(const isl::schedule schedule, const Scop::Data &data, Scop &scop, bool has_group = false)
      : has_grouped_(has_group), schedule_(schedule), data_(data), scop_(scop) {}
  ~Transform() {}

  // set up Transform
  isl::schedule Initialize(bool coincidence = true);

  void InsertDependenceCompute(isl::union_map &dependences);
  void InsertScheduleCompute();

  isl::schedule Ungroup(isl::schedule schedule_, const isl::union_pw_multi_aff &group_upma);

  static bool IsL1OrUbMark(const isl::schedule_node &node);

  static bool IsL0OrUbL0Mark(const isl::schedule_node &node);

  static bool IsSequenceOrSet(const isl::schedule_node &node);

  isl::schedule_node RetrieveNodeList(isl::schedule_node node, const std::vector<isl::schedule_node> &node_list);

  isl::schedule_node RetrieveAstBuildOptions(isl::schedule_node node, const isl::union_set &options);

  // intra-tile reschedule
  isl::schedule_node RescheduleSchTree(const isl::schedule_node &root);
  isl::schedule_node RescheduleInnerBand(const isl::schedule_node &root);

  isl::schedule RescheduleSerializeSccs(const isl::union_set &active_domain, const bool need_dist);
  isl::schedule_node ReschedulePreserveFilterOrder(const isl::schedule_node &node, const isl::union_set &active_domain,
                                                   const bool need_dist);
  static PointBandInfo SavePointBand(const isl::schedule_node &node);
  static isl::schedule_node SetPointBandInfo(isl::schedule_node node, const PointBandInfo &point_band_info);
  static isl::schedule_node RestorePointBandInfo(isl::schedule_node node, const PointBandInfo &point_band_info);
  void IntraTileReschedule(isl::schedule &sched, bool tile_inner_band, bool is_spec_gemm);

  // mark scalar statements for memory promotion
  isl::schedule_node TryMarkScalarStmts(const isl::schedule_node &root);

  isl::union_map ComputeCopyIn();

  isl::schedule_node InsertMarknode(isl::schedule_node node, const isl::id &gid);

  isl::union_map ComputeFilterCopyin(const isl::schedule_node &node);
  isl::union_map ComputeFakeCopyin(const isl::schedule &schedule);

  // compute all dependences for current schedule_
  isl::union_map ComputeAllDependences();

  isl::schedule ComputeSchedule();
  isl::schedule SinkLastAxis(const isl::schedule &sch);
  isl::schedule SinkC0(const isl::schedule &sch);
  isl::schedule_node SinkC0Schedule(isl::schedule_node &node);
  isl::schedule KeepOuterBandOrder(const isl::schedule &sch);

  void ComputeDependenceList();
  isl::union_pw_multi_aff GroupDependence();
  void ValidateShiftedSchedule(const isl::schedule &original_schedule, const isl::union_pw_multi_aff &group_upma);
  bool ValidateReorderedSchedule(const isl::schedule &new_schedule);

  isl::schedule_node TryRestoreStmtOrder(const isl::schedule_node &node, const std::vector<isl::id> &filter_total_order,
                                         const std::vector<std::vector<isl::id>> &filter_partial_order);

  bool ReplaceScheduleTree(isl::schedule &schedule);
  void DumpTransform(const std::string &file_name);
  void DumpSchTree(const std::string &file_name, const isl::schedule &sch);

  void ShowDimInfo(const Scop::Tiles &tiles);

  // tiling according to TilingInfo
  isl::schedule TileOuterBand(const Scop::Tiles &tiles, const isl::schedule &sch);

  // apply schedule constraints
  isl::schedule_constraints MakeScheduleConstraints(bool coincidence = true,
                                                    const isl::union_set &restrictDomain = isl::union_set());

  static isl::schedule_node GetOuterBand(const isl::schedule_node &root);

  isl::schedule_node InsertEmptyPermutableBand(isl::schedule_node node);

  bool SubtreeHasPermutableBands(const isl::schedule_node &node) const;

  int IsCandidate(const isl::schedule_node &node);

  int IsOuterTilable(const isl::schedule_node &node);

  isl::schedule_node MarkOuterPermutable(isl::schedule_node node);

  isl::schedule_node MarkTileBand(isl::schedule_node node, size_t tile_type);

  // tiling for l0
  isl::schedule_node TileL0(isl::schedule_node node);
  isl::schedule_node TileUbL1(isl::schedule_node node);

  void PaddingIsolate(int &h_head, int &h_tail, int &w_head, int &w_tail);
  bool IsConv();
  bool NeedIsolate();

  void ComputeHInfo(int &h_base, bool &head, bool &tail, int &h_head, int &h_tail, int &win_h, int &win_cut_h);

  void ComputeWInfo(int &w_base, bool &head, bool &tail, int &w_head, int &w_tail, int &win_w, int &win_cut_w);

  static isl::map ComputeTileMap(const isl::schedule_node &original_node, const isl::schedule_node &tiled_node);

  std::pair<isl::set, isl::set> ComputeFullTile(const isl::schedule_node &original_node,
                                                const isl::schedule_node &tiled_node);

  static isl::schedule_node SetIsolateLoopType(isl::schedule_node node);

  isl::schedule_node IsolateTiles(const isl::schedule_node &original_node, isl::schedule_node tiled_node,
                                  size_t tile_type, const int *full_tile_min, const int *full_tile_max);
  isl::schedule_node TileBand(isl::schedule_node node, const isl::multi_val &sizes, size_t tile_type,
                              const int *full_tile_min, const int *full_tile_max, bool isolation);

  void IsolateLevelInfo(size_t &tile_type, isl::set &tiles, isl::set &all);

  isl::schedule SetAllCoincident(const isl::schedule &schedule);

  isl::multi_val MultiValFromIntList(const isl::space &space, int dim, const int *list);

  isl::multi_val ComputeBandTilesSizes(const isl::schedule_node &node, const int *tile_size);

  // tile band and collect mark
  isl::schedule_node TileBandAndCollectMark(isl::schedule_node node, const int *tile_size, int *full_tile_min,
                                            int *full_tile_max, size_t tile_type, bool isolate);

  void TileTypeL1(isl::schedule_node &node, int *full_tile_min, int *full_tile_max, size_t &tile_type, bool &isolate,
                  isl::multi_val &sizes);

  void TileTypeL0(isl::schedule_node &node, int *full_tile_min, int *full_tile_max, size_t &tile_type, bool &isolate,
                  isl::multi_val &sizes);

  // get isolated
  bool GetIsolated() const;

  std::vector<std::vector<int>> getPartitionInfo() { return partition_info_; }

  // data structure for recording tile band data
  struct TileBandData {
    // flag indicating whether L0 tiled
    bool l0_tiled;
    // mark node of the tile band, if any
    isl::schedule_node mark;
    // mark node of conv_gemm, if any
    isl::schedule_node gemm_mark;
    // members of tile band
    unsigned int n_member;
    // schedule mupa
    isl::multi_union_pw_aff mupa;
    // permutable
    bool permutable;
    // coincident
    std::vector<bool> coincident;
    // ast build options
    isl::union_set ast_build_options;
  };

  isl::schedule_node RetrieveTileBandData(isl::schedule_node node, struct TileBandData *tile_band_data);
  void CollectTileBandData(const isl::schedule_node &node, struct TileBandData *tile_band_data);
  isl::union_map RemoveSelfDependence();
  isl::union_map RemoveReduceOpSelfDependence(bool multiAxisOnly = true);
  isl::union_map RemoveInvariantDependence();
  isl::schedule ReorderInvariantSetSchedule(isl::schedule &sch);

  void IsContainsCircle(const std::vector<std::vector<int>> &graph, std::vector<int> &vis, int node, int size);
  void DfsTopsort(std::vector<std::vector<int>> &graph, std::vector<int> &indegree, std::set<int> &zeros, int cnt,
                  int size, int64_t current_value, int64_t current_max);
  isl::union_set_list DependenciesTopsort(const isl::union_set_list &filterlist);
  bool HasInvariantDependence() { return has_invariant_dependence_; }

 private:
  bool has_grouped_;
  // schedule node to be transformed
  isl::schedule schedule_;
  // reads and writes
  Scop::Data data_;
  // constraints
  isl::schedule_constraints constraints_;
  // dependences_ in schedule node
  isl::union_map dependences_;
  // std::vector<Scop::TilingInfo>
  Scop::TileSizes tile_sizes_;
  // mark info for ub
  std::vector<std::vector<int>> partition_info_;
  // flag of tile isolate
  bool isolated_ = false;
  // scop
  Scop &scop_;

  // for recording L1/UB tile band build options
  std::vector<isl::union_set> l1_build_options_;

  // for recording L0 tile band build options
  std::vector<isl::union_set> l0_build_options_;

  // for recording nodes along the path from root to L1/UB band
  std::vector<isl::schedule_node> node_list_0_;

  // for recording nodes along the path from L1/UB band to L0/UBL0 band
  std::vector<isl::schedule_node> node_list_1_;

  // for recording nodes along the path from L0/UBL0 band to point band
  std::vector<isl::schedule_node> node_list_2_;

  std::vector<int> temp_res_;

  std::vector<int> topsort_res_;

  std::map<int, int64_t> cost_map_;

  // the min total cost for dfs Topsort
  int64_t min_topsort_ = -1;

  // counter times of dfs Topsort for limiting a long-time dfs process
  int cnt_dfs_times_ = 0;

  // the maximum times of dfs Topsort
  const int DFS_TIMES_MAX = 1000000;

  bool is_circle_ = false;

  bool find_filter_ = false;

  std::vector<Dependency> dependency_list_;

  std::map<std::string, int> invariant_state_;

  bool has_invariant_dependence_ = false;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_TRANSFORM_H_
