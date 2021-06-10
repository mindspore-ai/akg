/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "tile_outer_band.h"

#include "poly/scop.h"
#include "poly/schedule_pass/transfer_stmt.h"
#include "poly/schedule_pass/try_mark_scalar_stmt.h"
#include "poly/reduce_manager.h"

#include <cmath>

namespace akg {
namespace ir {
namespace poly {

class DimInfoMatcher : public IRVisitor {
 public:
  DimInfoMatcher() = default;
  ~DimInfoMatcher() override = default;

  std::string dim() { return dim_; }

  void Visit_(const AttrStmt *op) final {
    if (const auto Cop = op->node.as<ComputeOpNode>()) {
      for (auto iter : Cop->attrs) {
        if (dim_.empty() && iter.first == "dim") {
          if (auto dim = iter.second.as<StringImm>()) {
            dim_ = dim->value;
            break;
          }
        }
      }
    }
  }

 private:
  std::string dim_ = "";
};

std::string TileOuterBand::GetcDim() {
  auto matcher = DimInfoMatcher();
  matcher.Visit(scop_info_.user_config_.GetBody());
  return matcher.dim();
}

// Init set_dim info
void TileOuterBand::InitDimensionInfo(const isl::schedule &sch_init) {
  // get compute dim
  std::string dim = GetcDim();
  // get build dim
  if (dim.empty()) {
    dim = GetbDim();
  }

  // apply default tiling
  if (dim.empty()) {
    scop_info_.analysis_result_.SetEnableAutoTiling(true);
    auto tiling_res = GenerateTiling(sch_init, scop_info_, GenHalide(scop_info_, sch_init, true));
    scop_info_.analysis_result_.SetTileSizes(tiling_res.first);
    scop_info_.analysis_result_.SetTileConstraints(tiling_res.second);
    if (scop_info_.mmu_info_.IsConv()) scop_info_.mmu_info_.SetConvMNKInfo();
    return;
  }

  const std::string pattern = " ";
  std::vector<std::string> str = Split(dim, pattern);
  const int dim_info_entry_size = 4;
  CHECK(!str.empty() && !(str.size() % dim_info_entry_size)) << "Error: You need to set dim !";
  int sequence = 0;
  for (size_t i = 0; i < str.size(); i += dim_info_entry_size) {
    DimensionInfo dim_info;
    char *endptr = nullptr;
    const int radix = 10;
    dim_info.index = strtol(str[i].c_str(), &endptr, radix);
    if (endptr == nullptr || *endptr != '\0') LOG(FATAL) << "failed to convert string " << str[i] << " to number";
    const int max_dim_index = 16;
    CHECK(dim_info.index < max_dim_index) << "set_dim index must be less than " << max_dim_index << "!";
    dim_info.axis = str[i + 1];
    const int default_tiling_size = 65535;
    endptr = nullptr;
    int64_t str_2_number = strtol(str[i + 2].c_str(), &endptr, radix);
    if (endptr == nullptr || *endptr != '\0' || str_2_number <= 0) {
      dim_info.c1_tiling_size = default_tiling_size;
    } else {
      dim_info.c1_tiling_size = str_2_number;
    }
    endptr = nullptr;
    int64_t str_3_number = strtol(str[i + 3].c_str(), &endptr, radix);
    if (endptr == nullptr || *endptr != '\0' || str_3_number <= 0) {
      dim_info.c0_tiling_size = default_tiling_size;
    } else {
      dim_info.c0_tiling_size = str_3_number;
    }
    dim_info.dim_seq = sequence;
    sequence++;
    scop_info_.analysis_result_.InsertDimensionInfo(dim_info);
  }
}

void TileOuterBand::MergeTilingInfo() {
  int64_t tiles_num = 0;
  auto tile_sizes = scop_info_.analysis_result_.GetTileSizes();
  for (unsigned i = 0; i < tile_sizes.size(); ++i) {
    if (tiles_num <= tile_sizes[i].index) {
      tiles_num = tile_sizes[i].index + 1;
    }
  }
  tiles_.resize((size_t)tiles_num);

  for (unsigned i = 0; i < tile_sizes.size(); ++i) {
    tiles_[(unsigned int)tile_sizes[i].index].dim_infos.push_back(tile_sizes[i]);
  }
}

std::vector<std::vector<int>> TileOuterBand::AddTileInfo(const std::vector<std::vector<int>> &partition_info) {
  std::vector<std::vector<int>> info;
  PartitionSingle *single = PartitionSingle::getInstance();
  if (single == nullptr) {
    return partition_info;
  } else if (PartitionSingle::getTimes() < 2) {
    // first time gemm or m isolate main gemm
    return partition_info;
  }

  for (auto it : partition_info) {
    info.push_back(it);
  }
  return info;
}

isl::schedule TileOuterBand::Run(isl::schedule sch) {
  if (scop_info_.user_config_.GetTarget() == TARGET_CUDA) {
    return RunCuda(sch);
  } else {
    return RunNpu(sch);
  }
}

isl::schedule_node TileOuterBand::ReverseTraverseChild(isl::schedule_node node,
                                                       const std::function<isl::schedule_node(isl::schedule_node)> f) {
  if (node.isa<isl::schedule_node_band>()) {
    tile_sizes_ = tiles_[0].dim_infos;
    node = node.map_descendant_bottom_up(f);
  } else {
    // multiple outer bands, use same filter strategy as in auto tiling
    unsigned int band_idx = 0;
    for (auto i = 0; i < static_cast<int>(node.n_children()); ++i) {
      tile_sizes_ = band_idx < tiles_.size() ? tiles_[band_idx].dim_infos : tiles_[0].dim_infos;
      if (node.get_child(i).isa<isl::schedule_node_filter>()) {
        auto filter = node.get_child(i).as<isl::schedule_node_filter>();
        if (!filter.get_filter().is_empty() && filter.has_children() &&
            filter.get_child(0).isa<isl::schedule_node_band>()) {
          band_idx += 1;
        }
      }
      node = node.child(i).map_descendant_bottom_up(f);
      node = node.parent();
    }
  }
  return node;
}

isl::schedule TileOuterBand::RunCuda(const isl::schedule sch) {
  auto map_before_tile = sch.get_map();

  isl::schedule tiling_schedule = sch;
  InitDimensionInfo(tiling_schedule);
  MergeTilingInfo();

  isl::schedule_node root = sch.get_root();

  // 1. obtain the outermost tilable band
  isl::schedule_node node = GetOuterBand(root);

  scop_info_.analysis_result_.SetScheduleMapBeforeTile(sch.get_map());

  ShowDimInfo();

  // 2. Traverse the descendants of "node" (including the node itself)
  // in depth first postorder via the callback function.
  using std::placeholders::_1;
  const std::function<isl::schedule_node(isl::schedule_node)> f =
    std::bind(&TileOuterBand::MarkOuterPermutableCuda, this, _1);

  auto final_schedule = ReverseTraverseChild(node, f).get_schedule();
  if (final_schedule.get_map().is_equal(map_before_tile) &&
      (pass_info_.coincident_ || scop_info_.user_config_.GetConsiderCoincidence())) {
    restart_ = true;
  } else if (sch.plain_is_equal(final_schedule)) {
    pass_info_.tile_check_coincident_ = scop_info_.user_config_.GetTileCheckCoincident();
  }
  if (scop_info_.user_config_.GetIsTuning()) {
    // restore schedule before tiling as input of GenerateTilingSpace
    return sch;
  }
  pass_info_.tile_sizes_ = tile_sizes_;

  return final_schedule;
}

isl::schedule TileOuterBand::RunNpu(const isl::schedule sch) {
  auto map_before_tile = sch.get_map();
  // TransferStmt pass
  isl::schedule tiling_schedule = sch;
  if (!scop_info_.mmu_info_.IsSpecGemm()) {
    tiling_schedule = TransferStmt(scop_info_, pass_info_).Run(tiling_schedule);
  }
  scop_info_.analysis_result_.InitScheduleMapBeforeTile(scop_info_.GetCtx());
  if (!scop_info_.mmu_info_.IsSpecGemm() && (scop_info_.mmu_info_.IsConv() || scop_info_.mmu_info_.IsGemm())) {
    scop_info_.analysis_result_.SetScheduleMapBeforeTile(sch.get_map());
  }
  InitDimensionInfo(tiling_schedule);
  MergeTilingInfo();

  isl::schedule_node root = sch.get_root();

  // 1. obtain the outermost tilable band
  isl::schedule_node node = GetOuterBand(root);

  ShowDimInfo();

  // 2. Traverse the descendants of "node" (including the node itself)
  // in depth first postorder via the callback function.
  using std::placeholders::_1;
  const std::function<isl::schedule_node(isl::schedule_node)> f =
    std::bind(&TileOuterBand::MarkOuterPermutableNpu, this, _1);
  node = ReverseTraverseChild(node, f);

  scop_info_.AddPartitionInfoToData(AddTileInfo(partition_info_));
  scop_info_.analysis_result_.SetIsTiled(true);

  auto final_schedule = node.get_schedule();
  if (final_schedule.get_map().is_equal(map_before_tile) &&
      (pass_info_.coincident_ || scop_info_.user_config_.GetConsiderCoincidence())) {
    restart_ = true;
  } else if (sch.plain_is_equal(final_schedule)) {
    pass_info_.tile_check_coincident_ = scop_info_.user_config_.GetTileCheckCoincident();
    final_schedule = TryMarkScalarStmt(pass_info_).Run(final_schedule);
  }
  if (scop_info_.user_config_.GetIsTuning()) {
    // restore schedule before tiling as input of GenerateTilingSpace
    return sch;
  }
  return final_schedule;
}

void TileOuterBand::ShowDimInfo() {
  for (size_t i = 0; i < tiles_.size(); ++i) {
    LOG(INFO) << "No: " << i << ", tiling_flag: " << tiles_[i].tiling_flag;

    for (const auto &dim_info : tiles_[i].dim_infos) {
      std::stringstream ss;
      ss << "index: " << dim_info.index << ", axis: " << dim_info.axis << ", c1_size: " << dim_info.c1_tiling_size
         << ", c0_size: " << dim_info.c0_tiling_size << ", seq: " << dim_info.dim_seq
         << ", is inner: " << dim_info.is_inner;
      if (dim_info.c1_var.defined()) ss << ", c1_var: " << dim_info.c1_var;
      if (dim_info.c0_var.defined()) ss << ", c0_var: " << dim_info.c0_var;
      LOG(INFO) << ss.str();
    }
  }
}

bool TileOuterBand::IsPermutable(const isl::schedule_node &node, bool checkCoincident) {
  if (!node) return false;
  if (!node.isa<isl::schedule_node_band>()) return false;
  if (!node.as<isl::schedule_node_band>().get_permutable()) return false;
  if (node.as<isl::schedule_node_band>().n_member() < 1) return false;
  return !(checkCoincident && !node.as<isl::schedule_node_band>().member_get_coincident(0));
}

isl::schedule_node TileOuterBand::InsertEmptyPermutableBand(isl::schedule_node node) {
  isl::space space;
  isl::multi_union_pw_aff mupa;

  space = node.get_schedule().get_domain().get_space();

  space = space.set_from_params();
  mupa = isl::multi_union_pw_aff::zero(space);
  node = node.insert_partial_schedule(mupa);
  node = node.as<isl::schedule_node_band>().set_permutable(1);

  return node;
}

bool TileOuterBand::SubtreeHasPermutableBands(const isl::schedule_node &node) {
  bool all_non_permutable = false;
  all_non_permutable = node.every_descendant([&, this](const isl::schedule_node &node) -> bool {
    return BoolNot(IsPermutable(node, scop_info_.user_config_.GetTileCheckCoincident()));
  });

  return BoolNot(all_non_permutable);
}

int TileOuterBand::IsCandidate(const isl::schedule_node &node) {
  int permutable;

  if (node.isa<isl::schedule_node_leaf>()) return 1;
  permutable = static_cast<int>(IsPermutable(node, scop_info_.user_config_.GetTileCheckCoincident()));
  if (permutable) return permutable;
  if (node.isa<isl::schedule_node_filter>()) return 0;
  permutable = static_cast<int>(SubtreeHasPermutableBands(node));
  if (permutable < 0) return -1;
  return static_cast<int>(!permutable);
}

int TileOuterBand::IsOuterTilable(const isl::schedule_node &node) {
  int tilable;
  isl::schedule_node ancestor;

  tilable = IsCandidate(node);
  if (tilable < 0) return -1;
  if (!tilable) return 0;

  tilable = 0;
  ancestor = node;
  while (ancestor.has_parent()) {
    ancestor = ancestor.parent();

    tilable = IsCandidate(ancestor);
    if (tilable) break;
  }

  return static_cast<int>(BoolNot(static_cast<bool>(tilable)));
}

isl::schedule_node TileOuterBand::MarkTileBand(isl::schedule_node node, TileType tile_type) {
  std::string markTag;

  if (tile_type == TileType::C0) {
    markTag = REALIZE_C0;
    node = node.insert_mark(isl::id(node.ctx(), markTag));
#if SPEC_GEMM
    if (scop_info_.mmu_info_.IsConv()) {
      std::string mark_tag_gmm = CONV_GEMM;
      node = node.insert_mark(isl::id(node.ctx(), mark_tag_gmm));
    }
#endif
  }
  if (tile_type == TileType::C1) {
    markTag = REALIZE_C1;
    node = node.insert_mark(isl::id(node.ctx(), markTag));
  }
  if (tile_type == TileType::BUF) {
    markTag = REALIZE_BUF;
    node = node.insert_mark(isl::id(node.ctx(), markTag));
  }
  if (tile_type == TileType::BUFC0) {
    markTag = REALIZE_BUFC0;
    node = node.insert_mark(isl::id(node.ctx(), markTag));
  }
  if (tile_type == TileType::BUFC1) {
    markTag = REALIZE_BUFC1;
    node = node.insert_mark(isl::id(node.ctx(), markTag));
  }
  if (tile_type == TileType::C1BUFC1) {
    markTag = REALIZE_C1BUFC1;
    node = node.insert_mark(isl::id(node.ctx(), markTag));
  }

  return node;
}

isl::multi_val TileOuterBand::MultiValFromIntList(const isl::space &space, int dim, const int *list) {
  int i;
  isl::multi_val mv;

  isl::ctx ctx = space.ctx();
  mv = isl::multi_val::zero(space);
  for (i = 0; i < dim; ++i) {
    mv = mv.set_val(i, isl::val(ctx, list[i]));
  }

  return mv;
}

/* Build tile map which maps the elements of the original band
 * to applied tile, with the form:
 *  [[outer] -> [orig]] -> [[outer] -> [tile]].
 */
isl::map TileOuterBand::ComputeTileMap(const isl::schedule_node &original_node, const isl::schedule_node &tiled_node) {
  isl::union_map original_umap = original_node.as<isl::schedule_node_band>().get_partial_schedule_union_map();
  unsigned int depth = original_node.get_schedule_depth();

  isl::space space = original_umap.get_space().params().set_from_params();
  space = space.add_dims(isl_dim_set, depth);
  space = space.map_from_set();

  isl::multi_aff maff = isl::multi_aff::identity(space);
  isl::union_map tiled_umap = tiled_node.as<isl::schedule_node_band>().get_partial_schedule_union_map();
  tiled_umap = original_umap.reverse().apply_range(tiled_umap);
  isl::multi_union_pw_aff tiling = isl::multi_union_pw_aff::from_union_map(tiled_umap);

  isl::map el2tile = isl::map::from(isl::union_map::from(tiling));
  el2tile = isl::map::from(isl::union_map(isl::map::from(maff)).product(el2tile));

  return el2tile;
}

/*
 * Compute full tiles
 */
std::pair<isl::set, isl::set> TileOuterBand::ComputeFullTile(const isl::schedule_node &original_node,
                                                             const isl::schedule_node &tiled_node) {
  isl::map el2tile = ComputeTileMap(original_node, tiled_node);
  isl::map tile2el = el2tile.reverse();

  isl::union_map prefix = original_node.as<isl::schedule_node_band>().get_prefix_schedule_union_map();
  isl::union_set domain = original_node.as<isl::schedule_node_band>().get_domain();
  isl::union_map original_schedule = original_node.as<isl::schedule_node_band>().get_partial_schedule_union_map();
  isl::multi_union_pw_aff mupa = isl::multi_union_pw_aff::from_union_map(original_schedule);

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

void TileOuterBand::IsolateLevelInfo(TileType &tile_type, isl::set &tiles, isl::set &all) {
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
      if (edge > partition_info_[i].back()) partition_info_[i].push_back(edge);

      edge = static_cast<int>(tiles_lexmax.dim_max_val(i).get_num_si()) + 1;
      if (edge > partition_info_[i].back()) partition_info_[i].push_back(edge);

      edge = static_cast<int>(all_lexmax.dim_max_val(i).get_num_si()) + 1;
      if (edge > partition_info_[i].back()) partition_info_[i].push_back(edge);
    }
  }
}

/*
 * Set the non-isolated loop type to the isolated part.
 */
isl::schedule_node TileOuterBand::SetIsolateLoopType(isl::schedule_node node) {
  int i, n;

  if (!node.isa<isl::schedule_node_band>()) return node;

  n = static_cast<int>(node.as<isl::schedule_node_band>().n_member());
  for (i = 0; i < n; ++i) {
    enum isl_ast_loop_type type;

    type = isl_schedule_node_band_member_get_ast_loop_type(node.get(), i);
    if (type == isl_ast_loop_default) node = node.as<isl::schedule_node_band>().member_set_isolate_ast_loop_default(i);
    if (type == isl_ast_loop_atomic) node = node.as<isl::schedule_node_band>().member_set_isolate_ast_loop_atomic(i);
    if (type == isl_ast_loop_unroll) node = node.as<isl::schedule_node_band>().member_set_isolate_ast_loop_unroll(i);
    if (type == isl_ast_loop_separate)
      node = node.as<isl::schedule_node_band>().member_set_isolate_ast_loop_separate(i);
    else
      return node;
  }

  return node;
}

/* Isolate tiles on demand.
 */
isl::schedule_node TileOuterBand::IsolateTiles(const isl::schedule_node &original_node, isl::schedule_node tiled_node,
                                               TileType tile_type, const int *full_tile_min, const int *full_tile_max,
                                               bool isolation) {
  if ((scop_info_.user_config_.GetIsDynamic()) && (!scop_info_.mmu_info_.IsSpecGemm())) {
    return tiled_node;
  } else {
    if (scop_info_.user_config_.GetTileSizeIsVar() || (!isolation)) {
      return tiled_node;
    }
  }
  CHECK(tiled_node.isa<isl::schedule_node_band>());
  int in, depth, dim;
  isl::space space;
  isl::set tiles, all;
  isl::map map;
  isl::set set;
  isl::union_set opt;
  isl::multi_aff ma1, ma2;

  // If not tiled, return
  if (original_node.is_equal(tiled_node)) return tiled_node;

  depth = tiled_node.get_schedule_depth();
  dim = static_cast<int>(tiled_node.as<isl::schedule_node_band>().n_member());

  // compute a set "tiles" for all full tiles
  std::tie(tiles, all) = ComputeFullTile(original_node, tiled_node);
  if (nullptr != full_tile_min) {
    unsigned int n_dim = tiles.n_dim();
    for (int i = 0; i < dim; ++i) {
      if (0 == full_tile_min[i]) continue;
      tiles = isl::manage(
        isl_set_lower_bound_si(tiles.copy(), isl_dim_set, (n_dim - (unsigned int)(dim - i)), full_tile_min[i]));
    }
  }
  if (nullptr != full_tile_max) {
    unsigned int n_dim = tiles.n_dim();
    for (int i = 0; i < dim; ++i) {
      if (MAX_STRIDE == full_tile_max[i]) continue;
      tiles = isl::manage(
        isl_set_upper_bound_si(tiles.copy(), isl_dim_set, (n_dim - (unsigned int)(dim - i)), full_tile_max[i]));
    }
  }

  IsolateLevelInfo(tile_type, tiles, all);

  map = tiles.unwrap();
  in = static_cast<int>(map.dim(isl_dim_in));
  auto out = map.dim(isl_dim_out);

  auto upos = static_cast<unsigned int>(depth - in);
  auto udim = static_cast<unsigned int>(dim);
  map = map.project_out(isl_dim_out, (upos + udim), out - (upos + udim));

  space = map.get_space().range();

  ma1 = isl::multi_aff::project_out_map(space, isl_dim_set, upos, udim);
  ma2 = isl::multi_aff::project_out_map(space, isl_dim_set, 0, upos);
  ma1 = ma1.range_product(ma2);

  map = map.apply_range(isl::map(ma1));
  map = map.uncurry();
  map = map.flatten_domain();

  set = map.wrap();
  set = set.set_tuple_name("isolate");

  opt = tiled_node.as<isl::schedule_node_band>().get_ast_build_options();
  opt = opt.add_set(set);
  tiled_node = tiled_node.as<isl::schedule_node_band>().set_ast_build_options(opt);
  tiled_node = SetIsolateLoopType(tiled_node);

  return tiled_node;
}

isl::multi_val TileOuterBand::ComputeBandTilesSizes(const isl::schedule_node &node, const int *tile_size) {
  isl::space space;

  space = node.as<isl::schedule_node_band>().get_space();
  auto dim = static_cast<int>(node.as<isl::schedule_node_band>().n_member());
  return MultiValFromIntList(space, dim, tile_size);
}

void TileOuterBand::TileTypeC0(isl::schedule_node &node, int *full_tile_min, int *full_tile_max, TileType &tile_type,
                               bool &isolate, isl::multi_val &sizes) {
  isl::set_list domain_list = node.get_domain().get_set_list();
  isl::union_set filter_mmu = isl::union_set();
  isl::union_set filter_after_mmu = isl::union_set();

  unsigned int mmu_index = 0;
  for (; mmu_index < scop_info_.analysis_result_.stmt_type_.size() - 1; ++mmu_index) {
    if (scop_info_.analysis_result_.stmt_type_[mmu_index].second == STMT_OP_TYPE::MMU_CONV ||
        scop_info_.analysis_result_.stmt_type_[mmu_index].second == STMT_OP_TYPE::MMU_GEMM ||
        scop_info_.analysis_result_.stmt_type_[mmu_index].second == STMT_OP_TYPE::IM2COL_BUF) {
      break;
    }
  }
  std::vector<isl::union_set> filter_before_mmu;

  for (unsigned int set_index = 0; set_index < domain_list.size(); ++set_index) {
    isl::set set_i = domain_list.get_at(set_index);
    std::string name = set_i.get_tuple_name();
    CHECK(name.find('_') != std::string::npos) << "invalid name " << name;
    unsigned int index = WrappedStrtol(name.substr(name.find('_') + 1));
    set_i = isl::manage(isl_set_eliminate_dims(set_i.copy(), 0, isl_set_n_dim(set_i.get())));
    if (index + 1 < mmu_index) {
      filter_before_mmu.resize(mmu_index - 1);
      filter_before_mmu[index] = isl::union_set(set_i);
    }
    if (index + 1 == mmu_index || index == mmu_index) {
      filter_mmu = filter_mmu.is_null() ? isl::union_set(set_i) : filter_mmu.add_set(set_i);
    }
    if (index > mmu_index) {
      filter_after_mmu = filter_after_mmu.is_null() ? isl::union_set(set_i) : filter_after_mmu.add_set(set_i);
    }
  }

  isl::union_set_list filters =
    isl::union_set_list(node.ctx(), static_cast<int>(scop_info_.analysis_result_.stmt_type_.size() - 1));
  for (const auto &a : filter_before_mmu) {
    filters = a.is_null() ? filters : filters.add(a);
  }
  filters = filter_mmu.is_null() ? filters : filters.add(filter_mmu);
  filters = filter_after_mmu.is_null() ? filters : filters.add(filter_after_mmu);

  isl::schedule_node before_tile_node = node;

  if (scop_info_.mmu_info_.IsLoadIm2colC1BUF()) {
    node = TileBand(node, sizes);
    node = IsolateTiles(before_tile_node, node, TileType::BUF, full_tile_min, full_tile_max, isolate);
    node = MarkTileBand(node, TileType::BUF);
  } else if ((!filter_before_mmu.empty() || !filter_after_mmu.is_null()) && !filter_mmu.is_null()) {
    auto pos = 0;
    node = node.insert_sequence(filters);
    for (auto a : filter_before_mmu) {
      node = TileBand(node.child(pos).child(0), sizes);
      node = IsolateTiles(before_tile_node, node, tile_type, full_tile_min, full_tile_max, isolate);
      node = MarkTileBand(node, TileType::BUFC1);
      node = node.parent().parent();
      ++pos;
    }
    if (!filter_mmu.is_null()) {
      node = TileBand(node.child(pos).child(0), sizes);
      node = IsolateTiles(before_tile_node, node, tile_type, full_tile_min, full_tile_max, isolate);
      node = MarkTileBand(node, TileType::C0);
      node = node.parent().parent();
      ++pos;
    }
    if (!filter_after_mmu.is_null()) {
      node = TileBand(node.child(pos).child(0), sizes);
      node = IsolateTiles(before_tile_node, node, tile_type, full_tile_min, full_tile_max, isolate);
      node = MarkTileBand(node, TileType::BUFC0);
      node = node.parent().parent();
      ++pos;
    }
  } else {  // Don't insert a sequence node when there is only one filter child
    node = TileBand(node, sizes);
    node = IsolateTiles(before_tile_node, node, tile_type, full_tile_min, full_tile_max, isolate);
    node = MarkTileBand(node, tile_type);
  }
  node = node.parent().parent();
}

isl::schedule_node TileOuterBand::TileC0(isl::schedule_node node) {
  auto title_size = static_cast<unsigned int>(tile_sizes_.size());
  const unsigned int n_member = node.child(0).as<isl::schedule_node_band>().n_member();
  unsigned int dim_num = (n_member <= title_size) ? n_member : title_size;
  std::vector<int> ts(n_member, 0);
  std::vector<int> full_tile_max(n_member, 0);
  for (size_t j = 0; j < n_member; ++j) {
    ts[j] = MAX_STRIDE;
    full_tile_max[j] = MAX_STRIDE;
    if (j < dim_num) {
      ts[j] = static_cast<int>(tile_sizes_[j].c0_tiling_size);
      auto c1_tiling_size = static_cast<int>(tile_sizes_[j].c1_tiling_size);
      auto c0_tiling_size = static_cast<int>(tile_sizes_[j].c0_tiling_size);
      if (MAX_STRIDE == c1_tiling_size) continue;
      if (MAX_STRIDE == c0_tiling_size) continue;
      if ((c1_tiling_size > c0_tiling_size) && (0 != c0_tiling_size)) {
        full_tile_max[j] = c1_tiling_size / c0_tiling_size - 1;
      }
    }
  }
  node = TileBandAndCollectMark(node.child(0), &ts[0], nullptr, &full_tile_max[0], TileType::C0, true);
  return node;
}

bool TileOuterBand::NeedIsolate() { return scop_info_.mmu_info_.IsConv() || scop_info_.mmu_info_.IsLoadIm2colC1BUF(); }

void TileOuterBand::PaddingIsolate(int &h_head, int &h_tail, int &w_head, int &w_tail) {
  h_head = 0;
  h_tail = 0;
  w_head = 0;
  w_tail = 0;
  if (scop_info_.mmu_info_.GetConvAttrInfo().empty()) return;
  int pad_top = scop_info_.mmu_info_.GetAttrValue(ATTR_CONV_PAD_TOP);
  int pad_bottom = scop_info_.mmu_info_.GetAttrValue(ATTR_CONV_PAD_BOTTOM);
  int pad_left = scop_info_.mmu_info_.GetAttrValue(ATTR_CONV_PAD_LEFT);
  int pad_right = scop_info_.mmu_info_.GetAttrValue(ATTR_CONV_PAD_RIGHT);
  int h = scop_info_.mmu_info_.GetAttrValue(ATTR_CONV_FEATURE_H);
  int w = scop_info_.mmu_info_.GetAttrValue(ATTR_CONV_FEATURE_W);
  int kh = scop_info_.mmu_info_.GetAttrValue(ATTR_CONV_KERNEL_H);
  int kw = scop_info_.mmu_info_.GetAttrValue(ATTR_CONV_KERNEL_W);
  int stride_h = scop_info_.mmu_info_.GetAttrValue(ATTR_CONV_STRIDE_H);
  int stride_w = scop_info_.mmu_info_.GetAttrValue(ATTR_CONV_STRIDE_W);
  int dilation_h = scop_info_.mmu_info_.GetAttrValue(ATTR_CONV_DILATION_H);
  int dilation_w = scop_info_.mmu_info_.GetAttrValue(ATTR_CONV_DILATION_W);
  int h_cut = scop_info_.mmu_info_.GetAttrValue(ATTR_CONV_TILE_H);
  int w_cut = scop_info_.mmu_info_.GetAttrValue(ATTR_CONV_TILE_W);
  int d_kh = (kh - 1) * dilation_h + 1;
  CHECK_NE(stride_h, 0);
  int win_h = (h + pad_top + pad_bottom - d_kh) / stride_h + 1;
  int win_cut_h = (h_cut - d_kh) / stride_h + 1;
  if (win_cut_h > win_h) {
    if (!scop_info_.user_config_.GetIsDynamic() || win_h > 0) win_cut_h = win_h;
  }

  CHECK_NE(win_cut_h, 0);
  int h_base = (win_h + win_cut_h - 1) / win_cut_h;
  bool head = (pad_top > 0);
  bool tail = ((win_h - 1) * stride_h + d_kh > h + pad_top);

  ComputeHInfo(h_base, head, tail, h_head, h_tail, win_h, win_cut_h);

  int d_kw = (kw - 1) * dilation_w + 1;
  CHECK_NE(stride_w, 0);
  int win_w = (w + pad_left + pad_right - d_kw) / stride_w + 1;
  int win_cut_w = (w_cut - d_kw) / stride_w + 1;
  if (win_cut_w > win_w) {
    win_cut_w = win_w;
  }

  CHECK_NE(win_cut_w, 0);
  int w_base = (win_w + win_cut_w - 1) / win_cut_w;
  head = (pad_left > 0);
  tail = ((win_w - 1) * stride_w + d_kw > w + pad_right);

  ComputeWInfo(w_base, head, tail, w_head, w_tail, win_w, win_cut_w);
}

void TileOuterBand::ComputeWInfo(int &w_base, bool &head, bool &tail, int &w_head, int &w_tail, int &win_w,
                                 int &win_cut_w) {
  const int DIVIDED_PIECES_THREE = 3;
  const int DIVIDED_PIECES_TWO = 2;
  CHECK_NE(win_cut_w, 0);
  if (w_base >= DIVIDED_PIECES_THREE) {
    if (head) {
      w_head = 1;
      if (tail) {
        w_tail = w_base - DIVIDED_PIECES_TWO;
      } else {
        w_tail = win_w / win_cut_w - 1;
      }
    } else {
      w_head = 0;
      if (tail) {
        w_tail = w_base - DIVIDED_PIECES_TWO;
      } else {
        w_tail = win_w / win_cut_w - 1;
      }
    }
  } else if (w_base <= DIVIDED_PIECES_TWO) {
    if (!head && !tail && win_w / win_cut_w == DIVIDED_PIECES_TWO) {
      w_head = 0;
      w_tail = 1;
    } else if (head && !tail && win_w / win_cut_w == DIVIDED_PIECES_TWO) {
      w_head = 1;
      w_tail = 1;
    } else {
      w_head = 0;
      w_tail = 0;
    }
  }
}

void TileOuterBand::ComputeHInfo(int &h_base, bool &head, bool &tail, int &h_head, int &h_tail, int &win_h,
                                 int &win_cut_h) {
  const int DIVIDED_PIECES_THREE = 3;
  const int DIVIDED_PIECES_TWO = 2;
  CHECK_NE(win_cut_h, 0);
  if (h_base >= DIVIDED_PIECES_THREE) {
    if (head) {
      h_head = 1;
      if (tail) {
        h_tail = h_base - DIVIDED_PIECES_TWO;
      } else {
        h_tail = win_h / win_cut_h - 1;
      }
    } else {
      h_head = 0;
      if (tail) {
        h_tail = h_base - DIVIDED_PIECES_TWO;
      } else {
        h_tail = win_h / win_cut_h - 1;
      }
    }
  } else if (h_base <= DIVIDED_PIECES_TWO) {
    if (!head && !tail && win_h / win_cut_h == DIVIDED_PIECES_TWO) {
      h_head = 0;
      h_tail = 1;
    } else if (head && !tail && win_h / win_cut_h == DIVIDED_PIECES_TWO) {
      h_head = 1;
      h_tail = 1;
    } else {
      h_head = 0;
      h_tail = 0;
    }
  }
}

void TileOuterBand::TileTypeC1(isl::schedule_node &node, int *full_tile_min, int *full_tile_max, TileType &tile_type,
                               bool &isolate, isl::multi_val &sizes) {
  const unsigned int n_member = node.as<isl::schedule_node_band>().n_member();
  auto title_size = static_cast<unsigned int>(tile_sizes_.size());
  unsigned int dim_num = (n_member <= title_size) ? n_member : title_size;
  std::vector<int> full_tile_max_buf(n_member, 0);
  std::vector<int> full_tile_min_buf(n_member, 0);
  full_tile_max = &full_tile_max_buf[0];
  full_tile_min = &full_tile_min_buf[0];
  for (size_t j = 0; j < n_member; ++j) {
    full_tile_min[j] = 0;
    full_tile_max[j] = MAX_STRIDE;
    if (!scop_info_.user_config_.GetIsDynamic()) {
      if (NeedIsolate() && j < dim_num) {
        int h_head, h_tail, w_head, w_tail;
        PaddingIsolate(h_head, h_tail, w_head, w_tail);

        if (tile_sizes_[j].axis == "H") {
          full_tile_min[j] = h_head;
          full_tile_max[j] = h_tail;
        }

        if (tile_sizes_[j].axis == "W") {
          full_tile_min[j] = w_head;
          full_tile_max[j] = w_tail;
        }
      }
    }
  }
  isl::schedule_node before_tile_node = node;
  node = TileBand(node, sizes);
  node = IsolateTiles(before_tile_node, node, tile_type, full_tile_min, full_tile_max, isolate);
  node = MarkTileBand(node, tile_type);

  // C0 tiling
  node = TileC0(node.child(0));
}

isl::schedule_node TileOuterBand::TileBufC1(isl::schedule_node node) {
  const unsigned int n_member = node.child(0).as<isl::schedule_node_band>().n_member();
  unsigned int dim_num = (n_member <= static_cast<unsigned int>(tile_sizes_.size()))
                           ? n_member
                           : static_cast<unsigned int>(tile_sizes_.size());
  std::vector<int> ts(n_member, 0);
  std::vector<int> full_tile_max(n_member, 0);
  for (size_t j = 0; j < n_member; ++j) {
    ts[j] = MAX_STRIDE;
    full_tile_max[j] = MAX_STRIDE;
    if (j < dim_num) {
      ts[j] = static_cast<int>(tile_sizes_[j].c0_tiling_size);
      int c1_tiling_size = static_cast<int>(tile_sizes_[j].c1_tiling_size);
      int c0_tiling_size = static_cast<int>(tile_sizes_[j].c0_tiling_size);
      if (MAX_STRIDE == c1_tiling_size) continue;
      if (MAX_STRIDE == c0_tiling_size) continue;
      if ((c1_tiling_size > c0_tiling_size) && (0 != c0_tiling_size)) {
        full_tile_max[j] = c1_tiling_size / c0_tiling_size - 1;
      }
    }
  }
  node = TileBandAndCollectMark(node.child(0), &ts[0], nullptr, &full_tile_max[0], TileType::BUFC1, true);
  return node;
}

isl::schedule_node TileOuterBand::TileBandAndCollectMark(isl::schedule_node node, const int *tile_size,
                                                         int *full_tile_min, int *full_tile_max, TileType tile_type,
                                                         bool isolate) {
  isl::multi_val sizes = ComputeBandTilesSizes(node, tile_size);

  isl::schedule_node before_tile_node = node;
  if (tile_type == TileType::C1) {
    TileTypeC1(node, full_tile_min, full_tile_max, tile_type, isolate, sizes);
  } else if (tile_type == TileType::C0) {
    TileTypeC0(node, full_tile_min, full_tile_max, tile_type, isolate, sizes);
  } else if (tile_type == TileType::C1BUFC1) {
    node = TileBand(node, sizes);
    node = IsolateTiles(before_tile_node, node, tile_type, full_tile_min, full_tile_max, isolate);
    node = MarkTileBand(node, tile_type);
    node = TileBufC1(node.child(0));
  } else if (tile_type == TileType::BUFC1) {
    node = TileBand(node, sizes);
    node = IsolateTiles(before_tile_node, node, tile_type, full_tile_min, full_tile_max, isolate);
    node = MarkTileBand(node, tile_type);
    node = node.parent().parent();
  } else {
    node = TileBand(node, sizes);
    node = IsolateTiles(before_tile_node, node, tile_type, full_tile_min, full_tile_max, isolate);
    node = MarkTileBand(node, tile_type);
  }
  return node;
}

/***************************************************************************
 * steps:
 * 1. get tile size.
 * 2. tiling
 ***************************************************************************/
isl::schedule_node TileOuterBand::MarkOuterPermutableCuda(isl::schedule_node node) {
  // check tilable or not, and return the node if not
  if (IsOuterTilable(node) <= 0) return node;

  // make sure the node is a band node and has multiple members, insert empty band if not
  if (!node.isa<isl::schedule_node_band>() || (!node.as<isl::schedule_node_band>().member_get_coincident(0) &&
                                               scop_info_.user_config_.GetTileCheckCoincident()))
    node = InsertEmptyPermutableBand(node);

#if PRINT_SCHEDULE_INFO
  /// print band info
  isl::schedule_node_band outer_band = node.as<isl::schedule_node_band>();
  CHECK(!outer_band.is_null()) << " didn't find single outer_band \n" << pass_info_.schedule_;
  LOG(INFO) << "Please set dim based on loops band depth: " << outer_band.n_member() << " with "
            << outer_band.get_space();
  LOG(INFO) << "Domain info: " << outer_band;
#endif

  // get tile size
  node = SetTileSizeAndTile(node, TILE_WITH_C1);

  // tile matmul operator
  if (scop_info_.user_config_.GetEnableMatmul()) {
    node = MatmulTile(node);
  }
  return node;
}

isl::schedule_node TileOuterBand::MatmulTile(const isl::schedule_node &node) {
  auto tile_node = node;
  size_t start_depth = tile_node.get_tree_depth();

  isl::schedule_node_band band_node = tile_node.as<isl::schedule_node_band>();
  size_t count_coincident = 0;
  for (size_t i = 0; i < band_node.n_member(); ++i) {
    if (!band_node.member_get_coincident(i)) {
      break;
    }
    ++count_coincident;
  }

  // split the k axis
  tile_node = band_node.split(count_coincident);
  tile_node = InsertPromoteMarker(tile_node);

  if (scop_info_.user_config_.GetEnableTensorCoreUsePoly()) {
    auto replace_cfg_map = scop_info_.user_config_.GetReplaceConfig();
    if (replace_cfg_map.count(WARP_COMPUTE) == 0) {
      ResetWarpMappingConfig();
    }
    // The second tiling of tensor_core is to split the k-axis.
    tile_node = SetTileSizeAndTile(tile_node.child(0), TILE_WITH_C0_C1, count_coincident);
  }

  // The third tiling of tensor_core is to map to warp.
  tile_node = SetTileSizeAndTile(tile_node.child(0), TILE_WITH_WARP_C1, count_coincident);
  if (!scop_info_.user_config_.GetEnableTensorCoreUsePoly()) {
    tile_node = tile_node.child(0);
  }
  tile_node = tile_node.insert_mark(isl::id(tile_node.ctx(), PROMOTE_SHARED_TO_REGISTER_AB));
  // Locate the band to be mapped.
  tile_node = tile_node.child(0).insert_mark(MAP_TO_WARP).child(0);
  tile_node = tile_node.child(0).insert_mark(SKIP_MARKER).child(0);

  // The last tiling of tensor_core is to calculate the size of fragment.
  tile_node = SetTileSizeAndTile(tile_node, TILE_WITH_C0);
  tile_node = tile_node.child(0).insert_mark(SKIP_MARKER);

  if (scop_info_.user_config_.GetEnableConvTensorCore()) {
    int child_depth = KH_KW_DEPTH;
    while (tile_node.has_children() && child_depth != 0) {
      --child_depth;
      tile_node = tile_node.child(0);
    }
    if (tile_node.child(0).isa<isl::schedule_node_band>()) {
      tile_node = tile_node.insert_mark(KH_KW_MARKER);
    }
  }

  tile_node = tile_node.ancestor(tile_node.get_tree_depth() - start_depth);
  return tile_node;
}

void TileOuterBand::ResetWarpMappingConfig() {
  auto thread_cfg = scop_info_.user_config_.GetThreadConfig();
  CHECK(thread_cfg != nullptr) << "thread config is null";

  int total_warp = 1;
  for (size_t j = 0; j < thread_cfg->bound; ++j) {
    total_warp *= thread_cfg->GetAt(j).second;
  }
  total_warp = std::ceil(total_warp / WARP_SIZE);
  size_t warp_dim_x = std::sqrt(total_warp);
  size_t warp_dim_y = total_warp / warp_dim_x;
  std::string new_warp_cfg = std::to_string(warp_dim_x) + " " + std::to_string(warp_dim_y);
  scop_info_.user_config_.RecordReplaceConfig(WARP_COMPUTE, new_warp_cfg, MappingType::REPLACE_THREADS);
}

isl::schedule_node TileOuterBand::InsertPromoteMarker(const isl::schedule_node node) {
  isl::schedule_node tile_node = node.child(0);
  bool is_matrixc_promote_shared = IsMatrixCPromoteToShared();

  // Add different promotion marks in different positions.
  if (is_matrixc_promote_shared) {
    tile_node = tile_node.insert_mark(isl::id(tile_node.ctx(), PROMOTE_GLOBAL_TO_SHARED_C)).child(0);
    tile_node = tile_node.insert_mark(isl::id(tile_node.ctx(), PROMOTE_SHARED_TO_REGISTER_C)).child(0);
  } else {
    tile_node = tile_node.insert_mark(isl::id(tile_node.ctx(), PROMOTE_GLOBAL_TO_REGISTER_C)).child(0);
  }

  tile_node = tile_node.child(0).insert_mark(isl::id(tile_node.ctx(), PROMOTE_GLOBAL_TO_SHARED_AB));
  return tile_node;
}

bool TileOuterBand::IsMatrixCPromoteToShared() {
  std::string shared_tensors = scop_info_.user_config_.GetSharedTensors();
  if (shared_tensors.empty()) {
    return false;
  }

  shared_tensors += " ";
  auto pos = shared_tensors.find(" ");
  while (pos != std::string::npos) {
    std::string tensor = shared_tensors.substr(0, pos);
    auto matmul_map = scop_info_.analysis_result_.GetMatrixMatmulMap();
    if (matmul_map.count(tensor) && (matmul_map[tensor] == MATRIX_C || matmul_map[tensor] == MATRIX_ELSE)) {
      return true;
    }
    shared_tensors = shared_tensors.substr(pos + 1, shared_tensors.size());
    pos = shared_tensors.find(" ");
  }
  return false;
}

isl::schedule_node TileOuterBand::SplitMatmulStatement(const isl::schedule_node &node) {
  isl::schedule_node tile_node = node;
  auto all_reduce_map = scop_info_.analysis_result_.GetReduceTensorInfoMap();
  ReduceManager reduce_manager;

  auto band_node = tile_node.as<isl::schedule_node_band>();
  auto band_node_domain = band_node.get_partial_schedule().domain();
  StatementMap all_statements = scop_info_.analysis_result_.GetStatementMap();
  isl::union_map reduce_statement_map = isl::union_map::empty(node.ctx());
  isl::union_set reduce_statements = isl::union_set::empty(node.ctx());

  for (auto it = all_reduce_map.begin(); it != all_reduce_map.end(); ++it) {
    reduce_statement_map = reduce_statement_map.unite(it->second.stmt_map);
    auto this_reduce = reduce_manager.GetReduceStatements(band_node_domain, reduce_statement_map, all_statements);
    if (!this_reduce.is_empty()) {
      reduce_statements = reduce_statements.unite(this_reduce);
    }
  }

  if (!reduce_manager.SplitReduceStatements(tile_node, reduce_statements, pass_info_.dependences_, false)) {
    return node;
  }
  return tile_node;
}

isl::schedule_node TileOuterBand::SetTileSizeAndTile(const isl::schedule_node &node, const std::string &tile_level,
                                                     const int count_coincident) {
  const unsigned int n_member = node.as<isl::schedule_node_band>().n_member();
  auto title_size = static_cast<unsigned int>(tile_sizes_.size());
  unsigned int dim_num = (n_member <= title_size) ? n_member : title_size;
  std::vector<int> tile_size;
  auto replace_cfg_map = scop_info_.user_config_.GetReplaceConfig();
  if (tile_level == TILE_WITH_WARP_C1) {
    std::vector<int> warp_list;
    CHECK_NE(replace_cfg_map.count(WARP_COMPUTE), 0) << "Can't find warpconfig";
    auto warp_cfg = replace_cfg_map[WARP_COMPUTE];
    for (size_t i = 0, j = 0; i < n_member; ++i) {
      auto c1 = static_cast<int>(tile_sizes_[i].c1_tiling_size);
      auto c0 = static_cast<int>(tile_sizes_[i].c0_tiling_size);
      c1 = (static_cast<int>(i) < count_coincident) ? c1 : c0;
      if (c0 == scop_info_.analysis_result_.GetMmaMode().m && j < warp_cfg->bound) {
        c1 = std::max(c1 / warp_cfg->GetAt(j).second, c0);
        ++j;
      }
      warp_list.push_back(c1);
    }
    tile_size = GetTileSizeOfLevel(n_member, dim_num, tile_level, tile_sizes_, count_coincident, warp_list);
  } else {
    tile_size = GetTileSizeOfLevel(n_member, dim_num, tile_level, tile_sizes_, count_coincident);
  }
  isl::multi_val sizes = ComputeBandTilesSizes(node, &tile_size[0]);
  return TileBand(node, sizes);
}

/***************************************************************************
 * steps:
 * 1. get tile size.
 * 2. tiling
 ***************************************************************************/
isl::schedule_node TileOuterBand::MarkOuterPermutableNpu(isl::schedule_node node) {
  // check tilable or not, and return the node if not
  if (IsOuterTilable(node) <= 0) return node;

  // make sure the node is a band node and has multiple members, insert empty band if not
  if (!node.isa<isl::schedule_node_band>() || (!node.as<isl::schedule_node_band>().member_get_coincident(0) &&
                                               scop_info_.user_config_.GetTileCheckCoincident()))
    node = InsertEmptyPermutableBand(node);

#if PRINT_SCHEDULE_INFO
  /// print band info
  isl::schedule_node_band outer_band = node.as<isl::schedule_node_band>();
  CHECK(!outer_band.is_null()) << " didn't find single outer_band \n" << pass_info_.schedule_;
  LOG(INFO) << "Please set dim based on loops band depth: " << outer_band.n_member() << " with "
            << outer_band.get_space();
  LOG(INFO) << "Domain info: " << outer_band;
#endif

  const unsigned int n_member = node.as<isl::schedule_node_band>().n_member();
  auto title_size = static_cast<unsigned int>(tile_sizes_.size());
  unsigned int dim_num = (n_member <= title_size) ? n_member : title_size;
  if (dim_num == 0) {
    // direct scalar computation in GM is not allowed, need to promote to UB
    return MarkTileBand(node, TileType::BUF);
  }

  // get tile size
  std::vector<int> tile_size(n_member, 0);
  for (size_t j = 0; j < n_member; ++j) {
    tile_size[j] = MAX_STRIDE;
    // tile_size maybe bigger than dim_num
    if (j < dim_num) tile_size[j] = static_cast<int>(tile_sizes_[j].c1_tiling_size);
  }

  bool is_mmu = false;
  for (auto &info : scop_info_.analysis_result_.GetStmtOpInfoMap()) {
    if (info.second.isMMU) {
      is_mmu = true;
      break;
    }
  }

  bool is_before_mmu = false;
  bool is_in_mmu = false;
  unsigned int i = 0;
  for (; i < scop_info_.analysis_result_.stmt_type_.size() - 1; ++i) {
    if (scop_info_.analysis_result_.stmt_type_[i].second == STMT_OP_TYPE::MMU_CONV) {
      break;
    }
  }
  bool is_in_load_im2col = scop_info_.user_config_.GetIsDynamic() ? false : scop_info_.mmu_info_.IsLoadIm2colC1BUF();
  isl::set_list domain_list = node.get_domain().get_set_list();
  for (unsigned int set_index = 0; set_index < domain_list.size(); ++set_index) {
    isl::set set_i = domain_list.get_at(set_index);
    std::string name = set_i.get_tuple_name();
    if (name.find('_') == std::string::npos) {
      LOG(FATAL) << "Cannot find _ symbol";
    }
    unsigned int index = WrappedStrtol(name.substr(name.find('_') + 1));
    is_before_mmu = false;
    if ((index + 1 < i) && !scop_info_.mmu_info_.IsSpecGemm()) {
      is_before_mmu = true;
    }
    if (index + 1 == i) {
      is_in_mmu = true;
    }
    if (scop_info_.user_config_.GetIsDynamic()) {
      if (scop_info_.mmu_info_.IsLoadIm2colC1BUFStmt(set_i.get_tuple_name())) {
        is_in_load_im2col = true;
      }
    }
  }

  if (is_mmu && is_before_mmu && !is_in_mmu) {
    node = TileBandAndCollectMark(node, &tile_size[0], nullptr, nullptr, TileType::C1BUFC1, true);
  } else if (is_mmu || is_in_load_im2col) {
    node = TileBandAndCollectMark(node, &tile_size[0], nullptr, nullptr, TileType::C1, true);
  } else {
    node = TileBandAndCollectMark(node, &tile_size[0], nullptr, nullptr, TileType::BUF, true);
  }

  return node;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
