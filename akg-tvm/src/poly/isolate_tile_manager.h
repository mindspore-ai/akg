/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef POLY_ISOLATE_TILE_MANAGER_H_
#define POLY_ISOLATE_TILE_MANAGER_H_

#include "poly/scop.h"
#include "isl.h"
#include <iostream>
#include <vector>
namespace akg {
namespace ir {
namespace poly {
class IsolateTileManager {
 public:
  IsolateTileManager(ScopInfo &scop_info, bool is_promotion = false)
      : scop_info_(scop_info), is_promotion_(is_promotion) {}
  ~IsolateTileManager() {}

  isl::schedule_node IsolateTilesForCce(const isl::schedule_node &orig_node, const isl::schedule_node &tiled_node,
                                        TileType tile_type, const int *full_tile_min, const int *full_tile_max,
                                        const bool isolation);
  isl::schedule_node IsolateTilesForCudaAndCpu(const isl::schedule_node &orig_node,
                                               const isl::multi_val &mapped_tile_size, const int start_pos,
                                               const int all_tile_size = 0);
  std::vector<std::vector<int>> partition_info_;

 private:
  std::vector<int> GetFullTileMax(const isl::multi_val &mapped_tile_size, const int start_pos,
                                  const int all_tile_size = 0);
  void IsolateLevelInfo(TileType &tile_type, isl::set &tiles, isl::set &all);
  isl::map ComputeTileMap();
  std::pair<isl::set, isl::set> ComputeFullTile();
  isl::schedule_node SetIsolateLoopType(const isl::schedule_node &orig_node);
  void ComputeUpperAndLowerBounds(isl::set &tiles, const int *full_tile_min, const int *full_tile_max);
  isl::schedule_node IsolateTiles(const isl::set &tiles);

  ScopInfo &scop_info_;
  bool is_promotion_{false};
  isl::schedule_node before_tile_node_;
  isl::schedule_node after_tile_node_;
};
}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_ISOLATE_TILE_MANAGER_H_