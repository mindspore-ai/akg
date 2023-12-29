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

#include "poly/schedule_tree_util.h"
#include "poly/schedule_pass.h"
#include "isolate_tile_manager.h"

namespace akg {
namespace ir {
namespace poly {
/* Isolate tiles on demand.
 */
isl::schedule_node IsolateTileManager::IsolateTiles(const isl::set &tiles) {
  CHECK(after_tile_node_.isa<isl::schedule_node_band>());
  isl::map map = tiles.unwrap();
  int in = static_cast<int>(map.dim(isl_dim_in));
  auto out = map.dim(isl_dim_out);

  int depth = after_tile_node_.get_schedule_depth();
  int dim = static_cast<int>(after_tile_node_.as<isl::schedule_node_band>().n_member());
  auto upos = static_cast<unsigned int>(depth - in);
  auto udim = static_cast<unsigned int>(dim);
  map = map.project_out(isl_dim_out, (upos + udim), out - (upos + udim));

  isl::space space = map.get_space().range();

  isl::multi_aff ma1 = isl::multi_aff::project_out_map(space, isl_dim_set, upos, udim);
  isl::multi_aff ma2 = isl::multi_aff::project_out_map(space, isl_dim_set, 0, upos);
  ma1 = ma1.range_product(ma2);

  map = map.apply_range(isl::map(ma1));
  map = map.uncurry();
  map = map.flatten_domain();

  isl::set set = map.wrap();
  const std::string isolate_name = "isolate";
  set = set.set_tuple_name(isolate_name);

  isl::union_set opt = after_tile_node_.as<isl::schedule_node_band>().get_ast_build_options();
  opt = opt.add_set(set);
  auto node = after_tile_node_.as<isl::schedule_node_band>().set_ast_build_options(opt);
  return SetIsolateLoopType(node);
}

/*
 * compute a set "tiles" for all full tiles
 */
void IsolateTileManager::ComputeUpperAndLowerBounds(isl::set &tiles, const int *full_tile_min,
                                                    const int *full_tile_max) {
  CHECK(after_tile_node_.isa<isl::schedule_node_band>());
  int dim = static_cast<int>(after_tile_node_.as<isl::schedule_node_band>().n_member());

  if (full_tile_min != nullptr) {
    unsigned int n_dim = tiles.n_dim();
    for (int i = 0; i < dim; ++i) {
      if (full_tile_min[i] == 0) {
        continue;
      }
      tiles = isl::manage(
        isl_set_lower_bound_si(tiles.copy(), isl_dim_set, (n_dim - (unsigned int)(dim - i)), full_tile_min[i]));
    }
  }
  if (full_tile_max != nullptr) {
    unsigned int n_dim = tiles.n_dim();
    for (int i = 0; i < dim; ++i) {
      if (MAX_STRIDE == full_tile_max[i]) {
        continue;
      }
      tiles = isl::manage(
        isl_set_upper_bound_si(tiles.copy(), isl_dim_set, (n_dim - (unsigned int)(dim - i)), full_tile_max[i]));
    }
  }
}

/*
 * Set the non-isolated loop type to the isolated part.
 */
isl::schedule_node IsolateTileManager::SetIsolateLoopType(const isl::schedule_node &orig_node) {
  if (!orig_node.isa<isl::schedule_node_band>()) {
    return orig_node;
  }

  auto node = orig_node;
  int n = static_cast<int>(orig_node.as<isl::schedule_node_band>().n_member());
  for (int i = 0; i < n; ++i) {
    auto type = isl_schedule_node_band_member_get_ast_loop_type(node.get(), i);
    if (type == isl_ast_loop_default) {
      node = node.as<isl::schedule_node_band>().member_set_isolate_ast_loop_default(i);
    }
    if (type == isl_ast_loop_atomic) {
      node = node.as<isl::schedule_node_band>().member_set_isolate_ast_loop_atomic(i);
    }
    if (type == isl_ast_loop_unroll) {
      node = node.as<isl::schedule_node_band>().member_set_isolate_ast_loop_unroll(i);
    }
    if (type == isl_ast_loop_separate) {
      node = node.as<isl::schedule_node_band>().member_set_isolate_ast_loop_separate(i);
    } else {
      return node;
    }
  }

  return node;
}

/*
 * Compute full tiles
 */
std::pair<isl::set, isl::set> IsolateTileManager::ComputeFullTile() {
  isl::map el2tile = ComputeTileMap();
  isl::map tile2el = el2tile.reverse();

  isl::union_map prefix = isl::union_map::empty(before_tile_node_.ctx());
  isl::union_set domain = isl::union_set::empty(before_tile_node_.ctx());
  if (is_promotion_) {
    prefix = ShortSchedule(before_tile_node_);
    isl::schedule_node parent = before_tile_node_;
    while (parent.has_parent() && !parent.isa<isl::schedule_node_extension>()) {
      parent = parent.parent();
    }
    if (parent.isa<isl::schedule_node_extension>()) {
      auto extension = parent.as<isl::schedule_node_extension>();
      domain = extension.get_extension().range();
    }
  } else {
    prefix = before_tile_node_.as<isl::schedule_node_band>().get_prefix_schedule_union_map();
    domain = before_tile_node_.as<isl::schedule_node_band>().get_domain();
  }

  isl::union_map before_schedule = before_tile_node_.as<isl::schedule_node_band>().get_partial_schedule_union_map();
  isl::multi_union_pw_aff mupa = isl::multi_union_pw_aff::from_union_map(before_schedule);

  isl::union_map schedule = isl::union_map::from(mupa);
  schedule = prefix.range_product(schedule);

  isl::set all_el = isl::set::from_union_set(domain.apply(schedule));
  all_el = all_el.coalesce();

  isl::set all = all_el.apply(el2tile);

  isl::set partial = all.apply(tile2el);
  partial = partial.subtract(all_el);
  partial = partial.apply(el2tile);

  return {all.subtract(partial), all};
}

/* Build tile map which maps the elements of the original band
 * to applied tile, with the form:
 *  [[outer] -> [orig]] -> [[outer] -> [tile]].
 */
isl::map IsolateTileManager::ComputeTileMap() {
  auto before_band_node = before_tile_node_.as<isl::schedule_node_band>();
  auto after_band_node = after_tile_node_.as<isl::schedule_node_band>();
  isl::union_map before_umap = isl::union_map::empty(before_band_node.ctx());
  isl::union_map after_umap = isl::union_map::empty(after_band_node.ctx());
  if (is_promotion_) {
    auto before_partial_schedule = GetCurrentPartialSchedule(before_band_node, is_promotion_);
    before_umap = isl::union_map::from(before_partial_schedule);

    auto after_partial_schedule = GetCurrentPartialSchedule(after_band_node, is_promotion_);
    after_umap = isl::union_map::from(after_partial_schedule);
  } else {
    before_umap = before_band_node.get_partial_schedule_union_map();
    after_umap = after_band_node.get_partial_schedule_union_map();
  }
  unsigned int depth = before_tile_node_.get_schedule_depth();

  isl::space space = before_umap.get_space().params().set_from_params();
  space = space.add_dims(isl_dim_set, depth);
  space = space.map_from_set();

  isl::multi_aff maff = isl::multi_aff::identity(space);
  after_umap = before_umap.reverse().apply_range(after_umap);
  isl::multi_union_pw_aff tiling = isl::multi_union_pw_aff::from_union_map(after_umap);

  isl::map el2tile = isl::map::from(isl::union_map::from(tiling));
  el2tile = isl::map::from(isl::union_map(isl::map::from(maff)).product(el2tile));

  return el2tile;
}

void IsolateTileManager::IsolateLevelInfo(TileType &tile_type, isl::set &tiles, isl::set &all) {
  // which level do we need isolate info?
  if (TileType::C1 == tile_type || TileType::BUF == tile_type) {
    partition_info_.clear();
    auto tiles_hull = tiles.simple_hull();
    auto tiles_lexmin = tiles_hull.lexmin().simple_hull();
    auto tiles_lexmax = tiles_hull.lexmax().simple_hull();
    auto all_lexmax = all.simple_hull().lexmax().simple_hull();
    for (int i = 0; i < static_cast<int>(tiles.n_dim()); ++i) {
      std::vector<int> part;
      partition_info_.push_back(part);
      partition_info_[i].push_back(0);

      int edge = static_cast<int>(tiles_lexmin.dim_max_val(i).get_num_si());
      if (edge > partition_info_[i].back()) {
        partition_info_[i].push_back(edge);
      }

      edge = static_cast<int>(tiles_lexmax.dim_max_val(i).get_num_si()) + 1;
      if (edge > partition_info_[i].back()) {
        partition_info_[i].push_back(edge);
      }

      edge = static_cast<int>(all_lexmax.dim_max_val(i).get_num_si()) + 1;
      if (edge > partition_info_[i].back()) {
        partition_info_[i].push_back(edge);
      }
    }
  }
}

std::vector<int> IsolateTileManager::GetFullTileMax(const isl::multi_val &mapped_tile_size, const int start_pos,
                                                    const int all_tile_size) {
  auto mapping_partial_schedule =
    GetCurrentPartialSchedule(before_tile_node_.as<isl::schedule_node_band>(), is_promotion_);
  if (!is_promotion_) {
    mapping_partial_schedule = mapping_partial_schedule.intersect_domain(before_tile_node_.domain());
  }
  auto upa_list = mapping_partial_schedule.get_union_pw_aff_list();

  const int n_member = before_tile_node_.as<isl::schedule_node_band>().n_member();
  int tile_number = static_cast<int>(mapped_tile_size.size());
  if (scop_info_.user_config_.GetTarget() == TARGET_CPU) {
    tile_number = all_tile_size;
  }
  CHECK(start_pos < tile_number) << "The starting position cannot be greater than or equal to the tiling size.";
  int dim_num = std::min(n_member, tile_number);

  std::vector<int> full_tile_max(n_member, MAX_STRIDE);
  for (int i = 0, j = start_pos; i < n_member; ++i, ++j) {
    if (i >= dim_num || j >= tile_number) {
      continue;
    }
    int tiling_size = upa_list.get_at(i).floor().max_val().get_num_si() + 1;
    int current_tiling_size = mapped_tile_size.val(i).get_num_si();
    if (MAX_STRIDE == tiling_size) {
      continue;
    }
    if (tiling_size > current_tiling_size) {
      full_tile_max[i] = tiling_size / current_tiling_size - 1;
    }
  }
  return full_tile_max;
}

isl::schedule_node IsolateTileManager::IsolateTilesForCudaAndCpu(const isl::schedule_node &orig_node,
                                                                 const isl::multi_val &mapped_tile_size,
                                                                 const int start_pos, const int all_tile_size) {
  auto tiled_node = TileBand(orig_node, mapped_tile_size);
  if (scop_info_.analysis_result_.GetCsr() && scop_info_.user_config_.GetTarget() == TARGET_CPU) {
    return tiled_node;
  }

  // If not tiled, return
  if (orig_node.is_equal(tiled_node)) {
    return tiled_node;
  }

  before_tile_node_ = orig_node;
  after_tile_node_ = tiled_node;
  auto full_tile_max = GetFullTileMax(mapped_tile_size, start_pos, all_tile_size);
  isl::set tiles, all;
  std::tie(tiles, all) = ComputeFullTile();
  ComputeUpperAndLowerBounds(tiles, nullptr, &full_tile_max[0]);
  return IsolateTiles(tiles).child(0);
}

isl::schedule_node IsolateTileManager::IsolateTilesForCce(const isl::schedule_node &orig_node,
                                                          const isl::schedule_node &tiled_node, TileType tile_type,
                                                          const int *full_tile_min, const int *full_tile_max,
                                                          const bool isolation) {
  if ((scop_info_.user_config_.GetIsDynamic()) && (!scop_info_.mmu_info_.IsSpecGemm())) {
    return tiled_node;
  } else if (scop_info_.user_config_.GetTileSizeIsVar() || (!isolation)) {
    return tiled_node;
  }

  // If not tiled, return
  if (orig_node.is_equal(tiled_node)) {
    return tiled_node;
  }

  before_tile_node_ = orig_node;
  after_tile_node_ = tiled_node;
  isl::set tiles, all;
  std::tie(tiles, all) = ComputeFullTile();
  ComputeUpperAndLowerBounds(tiles, full_tile_min, full_tile_max);
  IsolateLevelInfo(tile_type, tiles, all);
  return IsolateTiles(tiles);
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
