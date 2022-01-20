/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef POLY_TILING_H_
#define POLY_TILING_H_

#include "poly/schedule_pass.h"

namespace akg {
namespace ir {
namespace poly {

constexpr auto KH_KW_DEPTH = 2;
constexpr auto DIM_SIZE = 4;
constexpr auto CUSTOM_DIM_SIZE = 6;
constexpr auto VECTORIZATION_NODE_DEPTH = 3;

/*
 * Tile the outer band accoding to TilingInfo. In this pass, we get the out-most band,
 * decide tile_size depending on the types of operators, and then start tiling.
 */
class TileOuterBand : public SchedulePass {
 public:
  TileOuterBand(PassInfo &pass_info, ScopInfo &scop_info) : pass_info_(pass_info), scop_info_(scop_info) {
    pass_name_ = __FUNCTION__;
  };
  ~TileOuterBand() {}

  enum class TileType {
    C0 = 0,
    C1,
    BUF,
    BUFC1,
    BUFC0,
    C1BUFC1,
    Invalid,
  };
  virtual isl::schedule Run(isl::schedule sch);
  // general function
  isl::schedule TileOuterBandHelper(const isl::schedule sch,
                                    const std::function<isl::schedule_node(isl::schedule_node)> &f);
  void InitDimensionInfo(const isl::schedule &);
  void MergeTilingInfo();
  void ShowDimInfo();
  std::string GetbDim() const { return scop_info_.user_config_.GetBDim(); }
  std::string GetcDim();
  isl::schedule_node ReverseTraverseChild(isl::schedule_node node,
                                          const std::function<isl::schedule_node(isl::schedule_node)> &f);
  int IsOuterTilable(const isl::schedule_node &node);
  int IsCandidate(const isl::schedule_node &node);
  bool IsPermutable(const isl::schedule_node &node, bool checkCoincident);
  bool SubtreeHasPermutableBands(const isl::schedule_node &node);
  isl::multi_val ComputeBandTilesSizes(const isl::schedule_node &node, const int *tile_size);
  bool BoolNot(bool b) { return !b; }

  // npu related functions
  isl::schedule RunNpu(isl::schedule sch);
  isl::schedule_node MarkOuterPermutableNpu(isl::schedule_node node);
  std::vector<std::vector<int>> AddTileInfo(const std::vector<std::vector<int>> &partition_info);
  isl::schedule_node MarkTileBand(isl::schedule_node node, TileType tile_type);
  isl::schedule_node TileBandAndCollectMark(isl::schedule_node node, const int *tile_size, int *full_tile_min,
                                            int *full_tile_max, TileType tile_type, bool isolate);
  isl::multi_val MultiValFromIntList(const isl::space &space, int dim, const int *list);
  void TileTypeC0(isl::schedule_node &node, int *full_tile_min, int *full_tile_max, TileType &tile_type, bool &isolate,
                  isl::multi_val &sizes);
  isl::schedule_node IsolateTiles(const isl::schedule_node &original_node, isl::schedule_node tiled_node,
                                  TileType tile_type, const int *full_tile_min, const int *full_tile_max,
                                  bool isolation);
  std::pair<isl::set, isl::set> ComputeFullTile(const isl::schedule_node &original_node,
                                                const isl::schedule_node &tiled_node);
  isl::map ComputeTileMap(const isl::schedule_node &original_node, const isl::schedule_node &tiled_node);
  void IsolateLevelInfo(TileType &tile_type, isl::set &tiles, isl::set &all);
  isl::schedule_node SetIsolateLoopType(isl::schedule_node node);
  void TileTypeC1(isl::schedule_node &node, int *full_tile_min, int *full_tile_max, TileType &tile_type, bool &isolate,
                  isl::multi_val &sizes);
  isl::schedule_node TileBufC1(isl::schedule_node node);
  isl::schedule_node TileC0(isl::schedule_node node);
  void PaddingIsolate(int &h_head, int &h_tail, int &w_head, int &w_tail);
  void ComputeHInfo(int &h_base, bool &head, bool &tail, int &h_head, int &h_tail, int &win_h, int &win_cut_h);
  void ComputeWInfo(int &w_base, bool &head, bool &tail, int &w_head, int &w_tail, int &win_w, int &win_cut_w);
  bool NeedIsolate();

  // cuda related functions
  isl::schedule RunCuda(isl::schedule sch);
  isl::schedule_node MarkOuterPermutableCuda(isl::schedule_node node);
  isl::schedule_node SetTileSizeAndTile(const isl::schedule_node &node, const std::string &tile_level,
                                        const int count_coincident = -1);
  isl::schedule_node InsertPromoteMarker(const isl::schedule_node node);
  void ResetWarpMappingConfig();
  isl::schedule_node TileMatmulOperatorForCuda(const isl::schedule_node &node);
  void CheckCustomMapping(const MappingStrategyFilterMap &custom_mapping_map);
  bool IsMatrixCPromoteToShared();

  // cpu related functions
  isl::schedule RunCpu(isl::schedule sch);
  isl::schedule_node MarkOuterPermutableCpu(isl::schedule_node node);
  
  isl::schedule_node TileCsrForCpu(const isl::schedule_node &orig_node);
  isl::schedule_node TileReduceXForCpu(const isl::schedule_node &orig_node);
  isl::schedule_node TileAllReduceForCpu(const isl::schedule_node &orig_node);
  isl::schedule_node TileGemmOperatorForCpu(const isl::schedule_node &orig_node);
  isl::schedule_node TileGemmBandNodeForCpu(const isl::schedule_node &orig_node);
  isl::schedule_node TileElementWiseForCpu(const isl::schedule_node &orig_node, const bool is_all_reduce = false);

  bool IsContainReduceStatement(const isl::schedule_node &orig_node);
  isl::multi_val GetVectorizationTileSize(const isl::schedule_node &orig_node);
  isl::schedule_node SplitReduceStatements(const isl::schedule_node &orig_node);
  isl::schedule_node IsolateTilesCpu(const isl::schedule_node &orig_node, const std::string &tile_level = "");
  isl::schedule_node InsertAllMarker(const isl::schedule_node &orig_node, const bool is_all_reduce);
  isl::schedule_node InsertMarkerForLoop(const isl::schedule_node &orig_node, const std::string &marker_name,
                                         const int insert_pos = 0);
  isl::schedule_node InsertParallelMarkerForGemm(const isl::schedule_node &orig_node, const std::string &marker_name);
  isl::schedule_node InsertMarkerForReduceY(const isl::schedule_node &orig_node, size_t start_depth);

 private:
  PassInfo &pass_info_;
  ScopInfo &scop_info_;
  Tiles tiles_;
  TileSizes tile_sizes_;
  TileSizes tile_sizes_all_;
  size_t cur_band_index_{0};
  std::vector<std::vector<int>> partition_info_;

  // cpu related parameters
  int vectorization_axis_pos_{-1};
  int start_pos_{0};
  isl::union_set reduce_statements_;
  Template template_type_{Template::DEFAULT};
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_TILING_H_
