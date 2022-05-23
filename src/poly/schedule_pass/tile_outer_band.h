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
#include "poly/isolate_tile_manager.h"

namespace akg {
namespace ir {
namespace poly {

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

  virtual isl::schedule Run(isl::schedule sch);
  // general function
  isl::schedule TileOuterBandHelper(const isl::schedule sch,
                                    const std::function<isl::schedule_node(isl::schedule_node)> &f);
  void InitDimensionInfo(const isl::schedule &);
  void InitCustomDimensionInfo(std::string dim);
  void MergeTilingInfo();
  void ShowDimInfo();
  std::string GetbDim() const { return scop_info_.user_config_.GetBDim(); }
  std::string GetcDim();
  isl::schedule_node ReverseTraverseChild(isl::schedule_node node,
                                          const std::function<isl::schedule_node(isl::schedule_node)> &f);
  int IsOuterTilable(const isl::schedule_node &node);
  int IsCandidate(const isl::schedule_node &node);
  bool IsPermutable(const isl::schedule_node &node);
  bool SubtreeHasPermutableBands(const isl::schedule_node &node);
  isl::multi_val ComputeBandTilesSizes(const isl::schedule_node &node, const int *tile_size);
  bool BoolNot(bool b) { return !b; }
  isl::schedule_node MarkOuterPermutable(isl::schedule_node node);

  // npu related functions
  TileType JudgeTileType(isl::schedule_node &node);
  isl::schedule_node MarkOuterPermutableNpu(isl::schedule_node node);
  std::vector<std::vector<int>> AddTileInfo(const std::vector<std::vector<int>> &partition_info);
  isl::schedule_node MarkTileBand(isl::schedule_node node, TileType tile_type);
  isl::schedule_node TileBandAndCollectMark(isl::schedule_node node, const int *tile_size, int *full_tile_min,
                                            int *full_tile_max, TileType tile_type, bool isolate);
  isl::multi_val MultiValFromIntList(const isl::space &space, int dim, const int *list);
  void TileTypeC0(isl::schedule_node &node, int *full_tile_min, int *full_tile_max, TileType &tile_type, bool &isolate,
                  isl::multi_val &sizes);
  void TileTypeC1(isl::schedule_node &node, int *full_tile_min, int *full_tile_max, TileType &tile_type, bool &isolate,
                  isl::multi_val &sizes);
  isl::schedule_node TileBufC1(isl::schedule_node node);
  isl::schedule_node TileC0(isl::schedule_node node);
  void PaddingIsolate(int &h_head, int &h_tail, int &w_head, int &w_tail);
  void ComputeHInfo(int &h_base, bool &head, bool &tail, int &h_head, int &h_tail, int &win_h, int &win_cut_h);
  void ComputeWInfo(int &w_base, bool &head, bool &tail, int &w_head, int &w_tail, int &win_w, int &win_cut_w);
  bool NeedIsolate();
  unsigned int GetMmuIndex();

  // cuda and cpu general function
  std::vector<int> GetTileSizeOfLevel(const int member_size, const int dim_size,
                                      const TileType tile_level = TileType::Invalid, const int count_coincident = -1,
                                      const std::vector<int> &warp_list = {});

  // cuda related functions
  isl::schedule_node MarkOuterPermutableCuda(isl::schedule_node node);

  isl::schedule_node TileMatmulOperatorForCuda(const isl::schedule_node &node);
  isl::schedule_node TileElementWiseForCuda(const isl::schedule_node &node);

  isl::schedule_node InsertPromoteMarker(const isl::schedule_node node);
  void ResetWarpMappingConfig();
  void CheckCustomMapping(const MappingStrategyFilterMap &custom_mapping_map);
  bool IsMatrixCPromoteToShared();

  isl::multi_val GetTileSizeOfLevelForCuda(const isl::schedule_node &node,
                                           const TileType tile_level = TileType::Invalid,
                                           const int count_coincident = -1);
  isl::multi_val GetMappedTileSize(const isl::schedule_node &orig_node, MappingCfg *mapping_cfg,
                                   const std::vector<int> &vectorization_tile_size = {});
  isl::schedule_node TileThreadAndBlockConfig(const isl::schedule_node &orig_node, const bool is_block_mapping = false);

  // cpu related functions
  isl::schedule_node MarkOuterPermutableCpu(isl::schedule_node node);
  isl::schedule_node TileAccordingToTileType(const isl::schedule_node &orig_node,
                                             const TileType tile_level = TileType::Invalid,
                                             const std::vector<int> &tile_size = {});
  std::vector<int> GetTileSizeForCpu(const isl::schedule_node &orig_node,
                                     const TileType tile_level = TileType::Invalid);

  isl::schedule_node TileCsrForCpu(const isl::schedule_node &orig_node);
  isl::schedule_node TileReduceXForCpu(const isl::schedule_node &orig_node);
  isl::schedule_node TileAllReduceForCpu(const isl::schedule_node &orig_node);
  isl::schedule_node TileGemmOperatorForCpu(const isl::schedule_node &orig_node);
  isl::schedule_node TileElementWiseForCpu(const isl::schedule_node &orig_node, const bool is_all_reduce = false);
  isl::schedule_node TileConvForCpu(const isl::schedule_node &orig_node);

  bool IsContainReduceStatement(const isl::schedule_node &orig_node);
  isl::schedule_node SplitReduceStatements(const isl::schedule_node &orig_node);
  isl::schedule_node InsertAllMarker(const isl::schedule_node &orig_node, const bool is_all_reduce);
  isl::schedule_node InsertMultiMarker(const isl::schedule_node &orig_node, const std::string &marker_name,
                                       const bool return_orig_pos = false, const int insert_marker_num = -1);
  isl::schedule_node InsertMarkerForReduceY(const isl::schedule_node &orig_node, size_t start_depth);
  void RecordCopyinForGemm(const isl::schedule_node_sequence &seq_node);

 private:
  PassInfo &pass_info_;
  ScopInfo &scop_info_;
  Tiles tiles_;
  TileSizes tile_sizes_;
  bool is_sequence_node_{false};
  size_t cur_band_index_{0};
  std::vector<std::vector<int>> partition_info_;
  std::unique_ptr<IsolateTileManager> isolate_tile_{nullptr};

  // cpu related parameters
  int vectorization_axis_pos_{-1};
  int start_pos_{0};
  isl::union_set reduce_statements_;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_TILING_H_
