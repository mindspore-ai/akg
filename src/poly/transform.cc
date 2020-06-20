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
#include "poly/transform.h"

#include <tvm/ir_visitor.h>
#include <tvm/operation.h>
#include <isl/constraint.h>

#include <climits>
#include <fstream>
#include <queue>
#include <cmath>

#include "poly/reschedule.h"
#include "poly/dump_log.h"

namespace akg {
namespace ir {
namespace poly {
bool BoolNot(bool b) { return !b; }

bool IsPermutable(const isl::schedule_node &node, bool checkCoincident) {
  if (!node) return false;
  if (!node.isa<isl::schedule_node_band>()) return false;
  if (!node.as<isl::schedule_node_band>().get_permutable()) return false;
  if (node.as<isl::schedule_node_band>().n_member() < 1) return false;
  return !(checkCoincident && !node.as<isl::schedule_node_band>().member_get_coincident(0));
}

// Return whether "set" has dims
bool HasNoDims(const isl::set &set) {
  auto dim = set.n_dim();
  return dim == 0;
}

isl::union_map DependenceAnalysis(const isl::union_map &sources, const isl::union_map &targets,
                                  const isl::union_map &kills, const isl::union_map &sch) {
  auto access_info = isl::union_access_info(targets);
  access_info = access_info.set_kill(kills);
  access_info = access_info.set_may_source(sources);
  access_info = access_info.set_schedule_map(sch);
  auto union_flow = access_info.compute_flow();
  return union_flow.get_may_dependence();
}

bool Transform::IsSequenceOrSet(const isl::schedule_node &node) {
  if (node.isa<isl::schedule_node_sequence>()) return true;
  return node.isa<isl::schedule_node_set>();
}

isl::schedule_constraints Transform::MakeScheduleConstraints(bool coincidence, const isl::union_set &restrictDomain) {
  if (coincidence) {
    constraints_ = isl::schedule_constraints::on_domain(schedule_.get_domain())
                     .set_coincidence(dependences_)  // keep it, check for more cases
                     .set_validity(dependences_)
                     .set_proximity(dependences_);
  } else {
    constraints_ = isl::schedule_constraints::on_domain(schedule_.get_domain())
                     .set_validity(dependences_)
                     .set_proximity(dependences_);
  }

  if (restrictDomain) {
    constraints_ = constraints_.intersect_domain(restrictDomain);
  }

  return constraints_;
}

isl::schedule Transform::ComputeSchedule() {
  auto ctx = constraints_.ctx().get();
  int status = isl_options_set_schedule_unit_max_var_coefficient_sum(ctx, 1);
  CHECK(status == isl_stat_ok);
  if (scop_.compute_reschedule_ == 1) {
    status = isl_options_set_schedule_whole_component(ctx, 0);
    CHECK(status == isl_stat_ok);
  } else {
    status = isl_options_set_schedule_maximize_coincidence(ctx, 0);
    CHECK(status == isl_stat_ok);
    status = isl_options_set_schedule_whole_component(ctx, 1);
    CHECK(status == isl_stat_ok);
  }

  if (scop_.disable_schedule_shift_ == 1) {
    status = isl_options_set_schedule_max_constant_term(ctx, 0);
    CHECK(status == isl_stat_ok);
    status = isl_options_set_schedule_nonneg_var_coefficient(ctx, 1);
    CHECK(status == isl_stat_ok);
  }

  if (scop_.enable_schedule_max_constant_ == 1) {
    status = isl_options_set_schedule_max_constant_term(ctx, 0);
    CHECK(status == isl_stat_ok);
  }

  if (scop_.disable_loop_reversal_ == 1) {
    status = isl_options_set_schedule_nonneg_var_coefficient(ctx, 1);
    CHECK(status == isl_stat_ok);
  }

  if (scop_.disable_loop_fusion_ == 1) {
    status = isl_options_set_schedule_serialize_sccs(ctx, 1);
    CHECK(status == isl_stat_ok);
  }

  return constraints_.compute_schedule();
}

isl::union_map Transform::ComputeAllDependences() {
  auto reads = data_.reads.domain_factor_domain();
  auto writes = data_.writes.domain_factor_domain();
  auto sch = schedule_.get_map();

  // RAW
  auto flowDeps = DependenceAnalysis(writes, reads, writes, sch);

  // WAR and WAW
  auto falseDeps = DependenceAnalysis(writes.unite(reads), writes, writes, sch);

  return flowDeps.unite(falseDeps).coalesce();
}

isl::union_map Transform::ComputeCopyIn() {
  auto reads = data_.reads.domain_factor_domain();
  auto writes = data_.writes.domain_factor_domain();
  auto uai = isl::union_access_info(reads);
  uai = uai.set_kill(writes);
  uai = uai.set_may_source(writes);
  uai = uai.set_schedule(schedule_);
  auto flow = uai.compute_flow();
  auto mayNoSource = flow.get_may_no_source();
  data_.copyin = data_.reads.intersect_range(mayNoSource.range());
  return data_.copyin;
}

isl::schedule_node Transform::GetOuterBand(const isl::schedule_node &root) {
  auto outer_band = root;

  while (!outer_band.isa<isl::schedule_node_band>()) {
    auto n = outer_band.n_children();
    if (n == 1) {
      outer_band = outer_band.child(0);
      continue;
    } else {
      /*
       * return the node when encountered branching or a leaf
       * an empty band would be inserted elsewhere
       */
      return outer_band;
    }
  }

  return outer_band;
}

/* Compute copyin for each filter node, by intersecting the domains of reads
 * and writes of the entire scop.
 */
isl::union_map Transform::ComputeFilterCopyin(const isl::schedule_node &node) {
  CHECK(node.isa<isl::schedule_node_filter>()) << "The input should be a filter node!" << std::endl;

  auto filter = node.as<isl::schedule_node_filter>().get_filter();
  auto reads = data_.reads.domain_factor_domain().intersect_domain(filter);
  auto writes = data_.writes.domain_factor_domain().intersect_domain(filter);
  auto uai = isl::union_access_info(reads);
  uai = uai.set_kill(writes);
  uai = uai.set_may_source(writes);
  uai = uai.set_schedule(schedule_);
  auto flow = uai.compute_flow();
  auto mayNoSource = flow.get_may_no_source();
  auto copyin = data_.reads.intersect_range(mayNoSource.range());

  return copyin;
}

/* Compute copyin for each filter and return the union of such copyins.
 * In particular, return an empty result when the outermost band node
 * is not a sequence/set node.
 *
 * "result" is the union of "copyin" from each filter node, which in
 * turn is computed by ComputeFilterCopyin.
 */
isl::union_map Transform::ComputeFakeCopyin(const isl::schedule &schedule) {
  auto root = schedule.get_root();
  auto node = GetOuterBand(root);
  auto result = data_.fake_copyin;

  if (!IsSequenceOrSet(node)) return result;

  auto n = node.n_children();
  for (auto i = 0u; i < n; ++i) {
    auto child = node.child(i);
    auto copyin = ComputeFilterCopyin(child);
    result = result.unite(copyin);
  }

  return result;
}

/* Build tile map which maps the elements of the original band
 * to applied tile, with the form:
 *  [[outer] -> [orig]] -> [[outer] -> [tile]].
 */
isl::map Transform::ComputeTileMap(const isl::schedule_node &original_node, const isl::schedule_node &tiled_node) {
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
std::pair<isl::set, isl::set> Transform::ComputeFullTile(const isl::schedule_node &original_node,
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

/*
 * Set the non-isolated loop type to the isolated part.
 */
isl::schedule_node Transform::SetIsolateLoopType(isl::schedule_node node) {
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

void Transform::IsolateLevelInfo(size_t &tile_type, isl::set &tiles, isl::set &all) {
  // which level do we need isolate info?
  if (L1 == tile_type || UB == tile_type) {
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

/* Isolate tiles on demand.
 */
isl::schedule_node Transform::IsolateTiles(const isl::schedule_node &original_node, isl::schedule_node tiled_node,
                                           size_t tile_type, const int *full_tile_min, const int *full_tile_max) {
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

isl::schedule_node Transform::TileBand(isl::schedule_node node, const isl::multi_val &sizes, size_t tile_type,
                                       const int *full_tile_min, const int *full_tile_max, bool isolation) {
  isl::ctx ctx = node.ctx();
  int scale_tile;
  int shift_point;

  if (!node.isa<isl::schedule_node_band>()) {
    return node;
  }
  scale_tile = isl_options_get_tile_scale_tile_loops(ctx.get());
  isl_stat status = isl_options_set_tile_scale_tile_loops(ctx.get(), 0);
  CHECK(status == isl_stat_ok);
  shift_point = isl_options_get_tile_shift_point_loops(ctx.get());
  status = isl_options_set_tile_shift_point_loops(ctx.get(), 1);
  CHECK(status == isl_stat_ok);

  isl::schedule_node before_tile = node;
  node = node.as<isl::schedule_node_band>().tile(sizes);

  if (!scop_.is_dynamic_ || scop_.is_spec_gemm_) {
    if ((!scop_.tile_size_is_var_) && (isolation)) {
      node = IsolateTiles(before_tile, node, tile_type, full_tile_min, full_tile_max);
    }
  }

  status = isl_options_set_tile_scale_tile_loops(ctx.get(), scale_tile);
  CHECK(status == isl_stat_ok);
  status = isl_options_set_tile_shift_point_loops(ctx.get(), shift_point);
  CHECK(status == isl_stat_ok);
  return node;
}

bool Transform::GetIsolated() const { return isolated_; }

isl::schedule_node Transform::MarkTileBand(isl::schedule_node node, size_t tile_type) {
  std::string markTag;

  if (tile_type == L0) {
    markTag = REALIZE_L0;
    node = node.insert_mark(isl::id(node.ctx(), markTag));
#if SPEC_GEMM
    if (IsConv()) {
      std::string mark_tag_gmm = CONV_GEMM;
      node = node.insert_mark(isl::id(node.ctx(), mark_tag_gmm));
    }
#endif
  }
  if (tile_type == L1) {
    markTag = REALIZE_L1;
    node = node.insert_mark(isl::id(node.ctx(), markTag));
  }
  if (tile_type == UB) {
    markTag = REALIZE_UB;
    node = node.insert_mark(isl::id(node.ctx(), markTag));
  }
  if (tile_type == UBL0) {
    markTag = REALIZE_UBL0;
    node = node.insert_mark(isl::id(node.ctx(), markTag));
  }
  if (tile_type == UBL1) {
    markTag = REALIZE_UBL1;
    node = node.insert_mark(isl::id(node.ctx(), markTag));
  }
  if (tile_type == L1UBL1) {
    markTag = REALIZE_L1UBL1;
    node = node.insert_mark(isl::id(node.ctx(), markTag));
  }

  return node;
}

void Transform::ComputeHInfo(int &h_base, bool &head, bool &tail, int &h_head, int &h_tail, int &win_h,
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

void Transform::ComputeWInfo(int &w_base, bool &head, bool &tail, int &w_head, int &w_tail, int &win_w,
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

void Transform::PaddingIsolate(int &h_head, int &h_tail, int &w_head, int &w_tail) {
  h_head = 0;
  h_tail = 0;
  w_head = 0;
  w_tail = 0;
  if (scop_.attr_info_.empty()) return;
  int pad_top = scop_.GetAttrValue(ATTR_CONV_PAD_TOP);
  int pad_bottom = scop_.GetAttrValue(ATTR_CONV_PAD_BOTTOM);
  int pad_left = scop_.GetAttrValue(ATTR_CONV_PAD_LEFT);
  int pad_right = scop_.GetAttrValue(ATTR_CONV_PAD_RIGHT);
  int h = scop_.GetAttrValue(ATTR_CONV_FEATURE_H);
  int w = scop_.GetAttrValue(ATTR_CONV_FEATURE_W);
  int kh = scop_.GetAttrValue(ATTR_CONV_KERNEL_H);
  int kw = scop_.GetAttrValue(ATTR_CONV_KERNEL_W);
  int stride_h = scop_.GetAttrValue(ATTR_CONV_STRIDE_H);
  int stride_w = scop_.GetAttrValue(ATTR_CONV_STRIDE_W);
  int dilation_h = scop_.GetAttrValue(ATTR_CONV_DILATION_H);
  int dilation_w = scop_.GetAttrValue(ATTR_CONV_DILATION_W);
  int h_cut = scop_.GetAttrValue(ATTR_CONV_TILE_H);
  int w_cut = scop_.GetAttrValue(ATTR_CONV_TILE_W);
  int d_kh = (kh - 1) * dilation_h + 1;
  CHECK_NE(stride_h, 0);
  int win_h = (h + pad_top + pad_bottom - d_kh) / stride_h + 1;
  int win_cut_h = (h_cut - d_kh) / stride_h + 1;
  if (win_cut_h > win_h) {
    if (!scop_.is_dynamic_ || win_h > 0) win_cut_h = win_h;
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
bool Transform::NeedIsolate() { return IsConv() || scop_.IsLoad3dL1Ub(); }

bool Transform::IsConv() {
  for (auto &info : data_.stmt_op_Info) {
    if (info.second.isCube) {
      const Node *stmt_node = data_.statements.at(info.first);
      if (stmt_node->IsInstance<Provide>()) {
        auto provide = static_cast<const Provide *>(stmt_node);
        const auto cop = provide->func.as<ComputeOpNode>();
        if (cop != nullptr && cop->attrs.count("feature") != 0) {
          return true;
        }
      }
      break;
    }
  }
  return false;
}

isl::multi_val Transform::MultiValFromIntList(const isl::space &space, int dim, const int *list) {
  int i;
  isl::multi_val mv;

  isl::ctx ctx = space.ctx();
  mv = isl::multi_val::zero(space);
  for (i = 0; i < dim; ++i) {
    mv = mv.set_val(i, isl::val(ctx, list[i]));
  }

  return mv;
}

isl::multi_val Transform::ComputeBandTilesSizes(const isl::schedule_node &node, const int *tile_size) {
  isl::space space;

  space = node.as<isl::schedule_node_band>().get_space();
  auto dim = static_cast<int>(node.as<isl::schedule_node_band>().n_member());
  return MultiValFromIntList(space, dim, tile_size);
}

void Transform::TileTypeL1(isl::schedule_node &node, int *full_tile_min, int *full_tile_max, size_t &tile_type,
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
    if (!scop_.is_dynamic_) {
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
  node = TileBand(node, sizes, tile_type, full_tile_min, full_tile_max, isolate);
  node = MarkTileBand(node, tile_type);

  // L0 tiling
  node = TileL0(node.child(0));
}

void Transform::TileTypeL0(isl::schedule_node &node, int *full_tile_min, int *full_tile_max, size_t &tile_type,
                           bool &isolate, isl::multi_val &sizes) {
  isl::set_list domain_list = node.get_domain().get_set_list();
  isl::union_set filter_cube = isl::union_set();
  isl::union_set filter_after_cube = isl::union_set();

  unsigned int cube_index = 0;
  for (; cube_index < scop_.stmt_type_.size() - 1; ++cube_index) {
    if (scop_.stmt_type_[cube_index].second == STMT_OP_TYPE::CUBE_CONV ||
        scop_.stmt_type_[cube_index].second == STMT_OP_TYPE::CUBE_GEMM ||
        scop_.stmt_type_[cube_index].second == STMT_OP_TYPE::IM2COL_UB) {
      break;
    }
  }
  std::vector<isl::union_set> filter_before_cube;

  for (unsigned int set_index = 0; set_index < domain_list.size(); ++set_index) {
    isl::set set_i = domain_list.get_at(set_index);
    std::string name = set_i.get_tuple_name();
    CHECK(name.find('_') != std::string::npos) << "invalid name " << name;
    unsigned int index = WrappedStrtol(name.substr(name.find('_') + 1));
    set_i = isl::manage(isl_set_eliminate_dims(set_i.copy(), 0, isl_set_n_dim(set_i.get())));
    if (index + 1 < cube_index) {
      filter_before_cube.resize(cube_index - 1);
      filter_before_cube[index] = isl::union_set(set_i);
    }
    if (index + 1 == cube_index || index == cube_index) {
      filter_cube = filter_cube.is_null() ? isl::union_set(set_i) : filter_cube.add_set(set_i);
    }
    if (index > cube_index) {
      filter_after_cube = filter_after_cube.is_null() ? isl::union_set(set_i) : filter_after_cube.add_set(set_i);
    }
  }

  isl::union_set_list filters = isl::union_set_list(node.ctx(), static_cast<int>(scop_.stmt_type_.size() - 1));
  for (const auto &a : filter_before_cube) {
    filters = a.is_null() ? filters : filters.add(a);
  }
  filters = filter_cube.is_null() ? filters : filters.add(filter_cube);
  filters = filter_after_cube.is_null() ? filters : filters.add(filter_after_cube);

  if (scop_.IsLoad3dL1Ub()) {
    node = TileBand(node, sizes, UB, full_tile_min, full_tile_max, isolate);
    node = MarkTileBand(node, UB);
  } else if ((!filter_before_cube.empty() || !filter_after_cube.is_null()) && !filter_cube.is_null()) {
    auto pos = 0;
    node = node.insert_sequence(filters);
    for (auto a : filter_before_cube) {
      node = TileBand(node.child(pos).child(0), sizes, tile_type, full_tile_min, full_tile_max, isolate);
      node = MarkTileBand(node, UBL1);
      node = node.parent().parent();
      ++pos;
    }
    if (!filter_cube.is_null()) {
      node = TileBand(node.child(pos).child(0), sizes, tile_type, full_tile_min, full_tile_max, isolate);
      node = MarkTileBand(node, L0);
      node = node.parent().parent();
      ++pos;
    }
    if (!filter_after_cube.is_null()) {
      node = TileBand(node.child(pos).child(0), sizes, tile_type, full_tile_min, full_tile_max, isolate);
      node = MarkTileBand(node, UBL0);
      node = node.parent().parent();
      ++pos;
    }
  } else {  // Don't insert a sequence node when there is only one filter child
    node = TileBand(node, sizes, tile_type, full_tile_min, full_tile_max, isolate);
    node = MarkTileBand(node, tile_type);
  }
  node = node.parent().parent();
}

isl::schedule_node Transform::TileBandAndCollectMark(isl::schedule_node node, const int *tile_size, int *full_tile_min,
                                                     int *full_tile_max, size_t tile_type, bool isolate) {
  isl::multi_val sizes = ComputeBandTilesSizes(node, tile_size);

  if (tile_type == L1) {
    TileTypeL1(node, full_tile_min, full_tile_max, tile_type, isolate, sizes);
  } else if (tile_type == L0) {
    TileTypeL0(node, full_tile_min, full_tile_max, tile_type, isolate, sizes);
  } else if (tile_type == L1UBL1) {
    node = TileBand(node, sizes, tile_type, full_tile_min, full_tile_max, isolate);
    node = MarkTileBand(node, tile_type);
    node = TileUbL1(node.child(0));
  } else if (tile_type == UBL1) {
    node = TileBand(node, sizes, tile_type, full_tile_min, full_tile_max, isolate);
    node = MarkTileBand(node, tile_type);
    node = node.parent().parent();
  } else {
    node = TileBand(node, sizes, tile_type, full_tile_min, full_tile_max, isolate);
    node = MarkTileBand(node, tile_type);
  }
  return node;
}

isl::schedule_node Transform::TileUbL1(isl::schedule_node node) {
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
      ts[j] = static_cast<int>(tile_sizes_[j].l0_tiling_size);
      int l1_tiling_size = static_cast<int>(tile_sizes_[j].l1_tiling_size);
      int l0_tiling_size = static_cast<int>(tile_sizes_[j].l0_tiling_size);
      if (MAX_STRIDE == l1_tiling_size) continue;
      if (MAX_STRIDE == l0_tiling_size) continue;
      if ((l1_tiling_size > l0_tiling_size) && (0 != l0_tiling_size)) {
        full_tile_max[j] = l1_tiling_size / l0_tiling_size - 1;
      }
    }
  }
  node = TileBandAndCollectMark(node.child(0), &ts[0], nullptr, &full_tile_max[0], UBL1, true);
  return node;
}

isl::schedule_node Transform::TileL0(isl::schedule_node node) {
  auto title_size = static_cast<unsigned int>(tile_sizes_.size());
  const unsigned int n_member = node.child(0).as<isl::schedule_node_band>().n_member();
  unsigned int dim_num = (n_member <= title_size) ? n_member : title_size;
  std::vector<int> ts(n_member, 0);
  std::vector<int> full_tile_max(n_member, 0);
  for (size_t j = 0; j < n_member; ++j) {
    ts[j] = MAX_STRIDE;
    full_tile_max[j] = MAX_STRIDE;
    if (j < dim_num) {
      ts[j] = static_cast<int>(tile_sizes_[j].l0_tiling_size);
      auto l1_tiling_size = static_cast<int>(tile_sizes_[j].l1_tiling_size);
      auto l0_tiling_size = static_cast<int>(tile_sizes_[j].l0_tiling_size);
      if (MAX_STRIDE == l1_tiling_size) continue;
      if (MAX_STRIDE == l0_tiling_size) continue;
      if ((l1_tiling_size > l0_tiling_size) && (0 != l0_tiling_size)) {
        full_tile_max[j] = l1_tiling_size / l0_tiling_size - 1;
      }
    }
  }
  node = TileBandAndCollectMark(node.child(0), &ts[0], nullptr, &full_tile_max[0], L0, true);
  return node;
}

isl::schedule_node Transform::InsertEmptyPermutableBand(isl::schedule_node node) {
  isl::space space;
  isl::multi_union_pw_aff mupa;

  space = node.get_schedule().get_domain().get_space();

  space = space.set_from_params();
  mupa = isl::multi_union_pw_aff::zero(space);
  node = node.insert_partial_schedule(mupa);
  node = node.as<isl::schedule_node_band>().set_permutable(1);

  return node;
}

bool Transform::SubtreeHasPermutableBands(const isl::schedule_node &node) const {
  bool all_non_permutable = false;
  all_non_permutable = node.every_descendant([&, this](const isl::schedule_node &node) -> bool {
    return BoolNot(IsPermutable(node, scop_.tile_check_coincident_));
  });

  return BoolNot(all_non_permutable);
}

int Transform::IsCandidate(const isl::schedule_node &node) {
  int permutable;

  if (node.isa<isl::schedule_node_leaf>()) return 1;
  permutable = static_cast<int>(IsPermutable(node, scop_.tile_check_coincident_));
  if (permutable) return permutable;
  if (node.isa<isl::schedule_node_filter>()) return 0;
  permutable = static_cast<int>(SubtreeHasPermutableBands(node));
  if (permutable < 0) return -1;
  return static_cast<int>(!permutable);
}

int Transform::IsOuterTilable(const isl::schedule_node &node) {
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

/***************************************************************************
 * steps:
 * 1. get tile size.
 * 2. tiling
 ***************************************************************************/
isl::schedule_node Transform::MarkOuterPermutable(isl::schedule_node node) {
  // check tilable or not, and return the node if not
  if (IsOuterTilable(node) <= 0) return node;

  // make sure the node is a band node and has multiple members, insert empty band if not
  if (!node.isa<isl::schedule_node_band>() ||
      (!node.as<isl::schedule_node_band>().member_get_coincident(0) && scop_.tile_check_coincident_))
    node = InsertEmptyPermutableBand(node);

#if PRINT_SCHEDULE_INFO
  /// print band info
  isl::schedule_node_band outer_band = node.as<isl::schedule_node_band>();
  CHECK(!outer_band.is_null()) << " didn't find single outer_band \n" << schedule_;
  LOG(INFO) << "Please set dim based on loops band depth: " << outer_band.n_member() << " with "
            << outer_band.get_space();
  LOG(INFO) << "Domain info: " << outer_band;
#endif

  const unsigned int n_member = node.as<isl::schedule_node_band>().n_member();
  auto title_size = static_cast<unsigned int>(tile_sizes_.size());
  unsigned int dim_num = (n_member <= title_size) ? n_member : title_size;
  if (dim_num == 0) {
    // direct scalar computation in GM is not allowed, need to promote to UB
    return MarkTileBand(node, UB);
  }

  // get tile size
  std::vector<int> tile_size(n_member, 0);
  for (size_t j = 0; j < n_member; ++j) {
    tile_size[j] = MAX_STRIDE;
    // tile_size maybe bigger than dim_num
    if (j < dim_num) tile_size[j] = static_cast<int>(tile_sizes_[j].l1_tiling_size);
  }

  bool isCube = false;
  for (auto &info : data_.stmt_op_Info) {
    if (info.second.isCube) {
      isCube = true;
      break;
    }
  }

  bool is_before_cube = false;
  bool is_in_cube = false;
  unsigned int i = 0;
  for (; i < scop_.stmt_type_.size() - 1; ++i) {
    if (scop_.stmt_type_[i].second == STMT_OP_TYPE::CUBE_CONV) {
      break;
    }
  }
  bool is_in_load3d = scop_.is_dynamic_ ? false : scop_.IsLoad3dL1Ub();
  isl::set_list domain_list = node.get_domain().get_set_list();
  for (unsigned int set_index = 0; set_index < domain_list.size(); ++set_index) {
    isl::set set_i = domain_list.get_at(set_index);
    std::string name = set_i.get_tuple_name();
    if (name.find('_') == std::string::npos) {
      LOG(FATAL) << "Cannot find _ symbol";
    }
    unsigned int index = WrappedStrtol(name.substr(name.find('_') + 1));
    is_before_cube = false;
    if ((index + 1 < i) && !scop_.is_spec_gemm_) {
      is_before_cube = true;
    }
    if (index + 1 == i) {
      is_in_cube = true;
    }
    if (scop_.is_dynamic_) {
      if (scop_.IsLoad3dL1UBStmt(set_i.get_tuple_name())) {
        is_in_load3d = true;
      }
    }
  }

  if (isCube && is_before_cube && !is_in_cube) {
    node = TileBandAndCollectMark(node, &tile_size[0], nullptr, nullptr, L1UBL1, true);
  } else if (isCube || is_in_load3d) {
    node = TileBandAndCollectMark(node, &tile_size[0], nullptr, nullptr, L1, true);
  } else {
    node = TileBandAndCollectMark(node, &tile_size[0], nullptr, nullptr, UB, true);
  }

  return node;
}

void Transform::ShowDimInfo(const Scop::Tiles &tiles) {
  for (size_t i = 0; i < tiles.size(); ++i) {
    LOG(INFO) << "No: " << i << ", tiling_flag: " << tiles[i].tiling_flag;

    for (const auto &dim_info : tiles[i].dim_infos) {
      std::stringstream ss;
      ss << "index: " << dim_info.index << ", axis: " << dim_info.axis << ", l1_size: " << dim_info.l1_tiling_size
         << ", l0_size: " << dim_info.l0_tiling_size << ", seq: " << dim_info.dim_seq
         << ", is inner: " << dim_info.is_inner;
      if (dim_info.l1_var.defined()) ss << ", l1_var: " << dim_info.l1_var;
      if (dim_info.l0_var.defined()) ss << ", l0_var: " << dim_info.l0_var;
      LOG(INFO) << ss.str();
    }
  }
}

isl::schedule Transform::TileOuterBand(const Scop::Tiles &tiles, const isl::schedule &sch) {
  isl::schedule_node root = sch.get_root();

  // 1. obtain the outermost tilable band
  isl::schedule_node node = GetOuterBand(root);

  /* 2. Traverse the descendants of "node" (including the node itself)
   * in depth first postorder via the callback function.
   */
  ShowDimInfo(tiles);

  using std::placeholders::_1;
  const std::function<isl::schedule_node(isl::schedule_node)> f = std::bind(&Transform::MarkOuterPermutable, this, _1);

  if (node.isa<isl::schedule_node_band>()) {
    tile_sizes_ = tiles[0].dim_infos;
    node = node.map_descendant_bottom_up(f);
  } else {
    // multiple outer bands, use same filter strategy as in auto tiling
    unsigned int band_idx = 0;
    for (auto i = 0; i < static_cast<int>(node.n_children()); ++i) {
      tile_sizes_ = band_idx < tiles.size() ? tiles[band_idx].dim_infos : tiles[0].dim_infos;
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

  return node.get_schedule();
}

isl::schedule_node Transform::InsertMarknode(isl::schedule_node node, const isl::id &gid) {
  if (node.isa<isl::schedule_node_leaf>()) {
    return node.insert_mark(gid);
  } else {
    if (node.n_children() == 1) {
      node = InsertMarknode(node.child(0), gid);
      node = node.parent();
    }
    return node;
  }
}

void Transform::DumpTransform(const std::string &file_name) {
  std::ofstream of;
  of.open(file_name, std::ios::out);
  if (!of.is_open()) {
    return;
  }

  PrintHeader(of, "dependences");
  of << FormatMupaStr(dependences_.to_str()) << std::endl;

  PrintHeader(of, "constraints");
  isl_printer *p;
  char *s = nullptr;
  p = isl_printer_to_str(scop_.ctx_.get());
  CHECK(p != nullptr);
  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
  p = isl_printer_print_schedule_constraints(p, constraints_.get());
  s = isl_printer_get_str(p);
  if (s) {
    of << FormatMupaStr(s);
    free(s);
  }
  static_cast<void>(isl_printer_free(p));

  PrintHeader(of, "L1/UB tile band build options");
  for (const auto &option : l1_build_options_) {
    of << option << std::endl;
  }

  PrintHeader(of, "L0 tile band build options");
  for (const auto &option : l0_build_options_) {
    of << option << std::endl;
  }

  PrintHeader(of, "nodes from root to L1/UB band");
  for (const auto &node : node_list_0_) {
    of << node << std::endl;
  }

  PrintHeader(of, "nodes from L1/UB band to L0/UBL0 band");
  for (const auto &node : node_list_1_) {
    of << node << std::endl;
  }

  PrintHeader(of, "nodes from L0/UBL0 band to point band");
  for (const auto &node : node_list_2_) {
    of << node << std::endl;
  }

  PrintHeader(of, "partition_info_");
  for (const auto &row : partition_info_) {
    of << "[";
    for (auto col : row) {
      of << col << ",";
    }
    of << "]" << std::endl;
  }

  of.close();
}

void Transform::DumpSchTree(const std::string &file_name, const isl::schedule &sch) {
  if (scop_.dump_pass_ir_) {
    scop_.DumpSchTree(file_name, sch);

#if DUMP_TRANSFORM
#if DUMP_TRANSFORM_PER_PASS
    DumpTransform(file_name + "_transform.log");
#else
    DumpTransform(scop_.CreateDumpDir("transform.log"));
#endif
#endif
  }
}

/*
 * Merge multiple lines of strings into a single-line string
 */
static std::string UndoPrettyPrintSchTree(const std::string &schedule) {
  const char *src = schedule.c_str();
  std::stringstream dst;
  bool in_string = false;
  while (*src != '\0') {
    if (*src == '"') {
      in_string = !in_string;
      if (!in_string) {
        // end of string, find next non-empty char
        const char *next = src + 1;
        while (*next != '\0') {
          char c = *next;
          if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
            break;
          }
          ++next;
        }
        if (*next == '"') {
          // multiple consecutive strings, merge them and insert a white space
          dst << " ";
          src = next + 1;
          in_string = true;
          continue;
        }
      }
    }
    dst << *src++;
  }
  return dst.str();
}

bool LoadScheduleTreeFromFile(const std::string &filename, isl::schedule &schedule) {
  std::ifstream new_schedule_file_stream(filename);
  std::string schedule_to_replace_str((std::istreambuf_iterator<char>(new_schedule_file_stream)),
                                      std::istreambuf_iterator<char>());
  schedule_to_replace_str = UndoPrettyPrintSchTree(schedule_to_replace_str);
  isl_schedule *ss = isl_schedule_read_from_str(schedule.ctx().get(), schedule_to_replace_str.c_str());
  if (ss != nullptr) {
    schedule = isl::manage(ss);
    return true;
  } else {
    LOG(WARNING) << "Failed to load file " << filename << " to schedule tree, please check syntax of the new schedule.";
    return false;
  }
}

/*
 * Compare and replace schedule hook:
 * Enable users to replace a specific schedule for debugging purpose.
 * If the current schedule is identical to the schedule in OLD_SCHEDULE_FILE,
 * the schedule will be replaced with NEW_SCHEDULE_FILE.
 */
bool Transform::ReplaceScheduleTree(isl::schedule &schedule) {
  const std::string OLD_SCHEDULE_FILE = scop_.AddDumpDir("old_schedule.txt");
  const std::string NEW_SCHEDULE_FILE = scop_.AddDumpDir("new_schedule.txt");
  // check if two files exist
  char pathBuffOld[PATH_MAX + 1] = {0};
  char pathBuffNew[PATH_MAX + 1] = {0};
  bool should_compare_and_replace = false;
  if (realpath(OLD_SCHEDULE_FILE.c_str(), pathBuffOld) && realpath(NEW_SCHEDULE_FILE.c_str(), pathBuffNew)) {
    FILE *schedule_to_compare = fopen(pathBuffOld, "r");
    FILE *schedule_to_replace = fopen(pathBuffNew, "r");
    should_compare_and_replace = (schedule_to_compare != nullptr && schedule_to_replace != nullptr);
    if (schedule_to_compare != nullptr) {
      int status = fclose(schedule_to_compare);
      if (status != 0) LOG(WARNING) << "Failed to close old_schedule.txt";
    }
    if (schedule_to_replace != nullptr) {
      int status = fclose(schedule_to_replace);
      if (status != 0) LOG(WARNING) << "Failed to close new_schedule.txt";
    }
  }

  if (should_compare_and_replace) {
    std::ifstream old_schedule_file_stream(OLD_SCHEDULE_FILE);
    std::string schedule_to_compare_str((std::istreambuf_iterator<char>(old_schedule_file_stream)),
                                        std::istreambuf_iterator<char>());
    if (CompareSchTreeWithString(schedule_to_compare_str, schedule)) {
      LOG(INFO) << "Current schedule is same as " << OLD_SCHEDULE_FILE << ", replace it with new schedule "
                << NEW_SCHEDULE_FILE;
      CHECK(LoadScheduleTreeFromFile(NEW_SCHEDULE_FILE, schedule));
      return true;
    } else {
      LOG(INFO) << "Current schedule is different from " << OLD_SCHEDULE_FILE << ", not replacing.";
    }
  }
  return false;
}

isl::union_map ModDependences(const isl::union_map &dependences) {
  isl::union_map umap = isl::union_map::empty(dependences.ctx());
  dependences.foreach_map([&](const isl::map &m) -> void {
    isl::map mm = m;
    if (mm.get_tuple_id(isl_dim_in) != mm.get_tuple_id(isl_dim_out)) {
      isl_map *pmap = mm.copy();
      int n_in = isl_map_dim(pmap, isl_dim_in);
      for (int i = 0; i < n_in; ++i) {
        pmap = isl_map_plain_update_val_if_fixed(pmap, isl_dim_in, i);
      }
      mm = isl::manage(pmap);
    }
    umap = umap.unite(isl::union_map(mm));
  });
  return umap;
}

/* Reorder axes in the outer band to the same as input IR.
 */
isl::schedule Transform::KeepOuterBandOrder(const isl::schedule &sch) {
  auto outer_band_node = GetOuterBand(sch.get_root());
  if (!outer_band_node.isa<isl::schedule_node_band>()) {
    return sch;
  }
  auto outer_band = outer_band_node.as<isl::schedule_node_band>();
  if (!outer_band.get_permutable()) {
    return sch;
  }
  auto mupa = outer_band.get_partial_schedule();
  auto n_member = mupa.size();
  // rank outer band members according to input order
  std::multimap<size_t, size_t> axes_scores;  // map score to old axes
  for (auto i = 0u; i < n_member; ++i) {
    auto upa = mupa.get_at(i);
    size_t axis_score = 0;
    upa.get_pw_aff_list().foreach([&](const isl::pw_aff &pw_aff) {
      pw_aff.foreach_piece([&](const isl::set &set, const isl::aff &aff) {
        size_t n_dims = isl_aff_dim(aff.get(), isl_dim_in);
        CHECK_GE(n_dims, 0);
        // vector
        size_t min_dim_in_aff = 0;
        if (scop_.HasCube()) {
          // cube
          min_dim_in_aff = n_dims;
        }
        for (auto j = 0u; j < n_dims; ++j) {
          auto coef_val = isl_aff_get_coefficient_val(aff.get(), isl_dim_in, j);
          if (isl_val_get_num_si(coef_val) != 0) {
            min_dim_in_aff = j;
            break;
          }
          static_cast<void>(isl_val_free(coef_val));
        }
        axis_score += min_dim_in_aff;
      });
    });
    axes_scores.insert(std::make_pair(axis_score, i));
  }

  std::vector<size_t> axes_map;  // new axes to old axes map
  for (auto it : axes_scores) {
    axes_map.push_back(it.second);
  }

  // construct new outer band according to the axes map
  isl::union_pw_aff_list new_upal;
  for (auto i = 0u; i < n_member; ++i) {
    if (i == 0) {
      new_upal = isl::union_pw_aff_list(mupa.get_at(axes_map[i]));
    } else {
      new_upal = new_upal.add(mupa.get_at(axes_map[i]));
    }
  }

  auto new_mupa = isl::multi_union_pw_aff(mupa.get_space(), new_upal);

  // save permutable and coincident of old node
  bool permutable = outer_band.get_permutable();
  std::vector<bool> coincident;
  for (auto i = 0u; i < n_member; ++i) {
    coincident.push_back(outer_band.member_get_coincident(axes_map[i]));
  }
  if (!scop_.HasCube()) {
    coincident[0] = true;
  }

  // delete old node
  outer_band_node = outer_band_node.del();

  // insert new node
  outer_band_node = outer_band_node.insert_partial_schedule(new_mupa);
  outer_band_node = outer_band_node.as<isl::schedule_node_band>().set_permutable(permutable);
  for (auto i = 0u; i < n_member; ++i) {
    outer_band_node = outer_band_node.as<isl::schedule_node_band>().member_set_coincident(i, coincident[i]);
  }

  return outer_band_node.get_schedule();
}

void Transform::ComputeDependenceList() {
  dependences_.foreach_map([&](const isl::map &m) -> void {
    if (m.domain().get_tuple_id() != m.range().get_tuple_id()) {
      isl::space domain_space_obj = m.domain().get_space();
      isl::local_space domain_space = isl::manage(isl_local_space_from_space(domain_space_obj.copy()));
      int dim = m.dim(isl_dim_in);
      int64_t weight = 1;
      for (int i = 0; i < dim; ++i) {
        isl::aff get_dim_in_domain = isl::aff::var_on_domain(domain_space, isl_dim_out, i);
        int max = static_cast<int>(m.domain().max_val(get_dim_in_domain).get_num_si());
        int min = static_cast<int>(m.domain().min_val(get_dim_in_domain).get_num_si());
        weight *= (max - min + 1);
      }
      Dependency dependency(m.domain().get_tuple_id(), m.range().get_tuple_id(), weight);
      dependency_list_.push_back(dependency);
    }
  });
}

void Transform::ValidateShiftedSchedule(const isl::schedule &original_schedule,
                                        const isl::union_pw_multi_aff &group_upma) {
  if (scop_.sink_last_axis_ || scop_.mod_schedule_shift_) {
    isl::union_map new_dependence = ComputeAllDependences();
    if (!new_dependence.is_subset(dependences_)) {
      LOG(WARNING) << "After mod schedule, dependences changed, revert schedule";
      // restore the original schedule and redo ungroup
      if (has_grouped_) {
        schedule_ = Ungroup(original_schedule, group_upma);
      } else {
        schedule_ = original_schedule;
      }
    }
  }
}

isl::union_pw_multi_aff Transform::GroupDependence() {
  isl::schedule_node rnode = schedule_.get_root().child(0);
  isl_union_pw_multi_aff *contract = isl_schedule_node_get_subtree_contraction(rnode.get());
  isl::union_pw_multi_aff group_upma = isl::manage(contract);
  if (has_grouped_) {
    isl::union_map gmap = isl::union_map::from(group_upma);
    dependences_ = dependences_.apply_range(gmap).apply_domain(gmap);
  }
  return group_upma;
}

void Transform::InsertDependenceCompute(isl::union_map &dependences) {
  if (!scop_.is_spec_gemm_ && scop_.remove_self_dependence_) {
    dependences = RemoveReduceOpSelfDependence();
    DumpSchTree("02_removeSelfDependence" + std::string(scop_.is_spec_gemm_ ? "_specgemm" : ""), schedule_);
  }
  if (!scop_.is_spec_gemm_ && scop_.force_remove_self_dependence_) {
    dependences = RemoveSelfDependence();
    DumpSchTree("02_removeSelfDependence" + std::string(scop_.is_spec_gemm_ ? "_specgemm" : ""), schedule_);
  }

  if (scop_.remove_invariant_dependence_) {
    dependences = RemoveInvariantDependence();
  }
}

void Transform::InsertScheduleCompute() {
  if (has_invariant_dependence_) {
    schedule_ = ReorderInvariantSetSchedule(schedule_);
    DumpSchTree("04_reorderInvariantSetSchedule" + std::string(scop_.is_spec_gemm_ ? "_specgemm" : ""), schedule_);
  }

  if (scop_.reorder_schedule_) {
    schedule_ = SinkC0(schedule_);
    DumpSchTree("04_reorderSchedule" + std::string(scop_.is_spec_gemm_ ? "_specgemm" : ""), schedule_);
  }

  if (scop_.sink_last_axis_) {
    schedule_ = SinkLastAxis(schedule_);
    DumpSchTree("04_sinkLastAxis" + std::string(scop_.is_spec_gemm_ ? "_specgemm" : ""), schedule_);
  }

  if (scop_.keep_outer_band_order_) {
    schedule_ = KeepOuterBandOrder(schedule_);
    DumpSchTree("04_keepOuterBandOrder" + std::string(scop_.is_spec_gemm_ ? "_specgemm" : ""), schedule_);
  }
}

isl::schedule Transform::Initialize(bool coincidence) {
  // 1. compute all dependences: flow and false
  std::chrono::high_resolution_clock::time_point timer_start;
  TIMER_START;
  dependences_ = ComputeAllDependences();
  TIMER_SHOW("computeAllDependences", std::string(scop_.is_spec_gemm_ ? "_specgemm" : ""));

  auto orig_dependences = dependences_;
  ComputeDependenceList();

  InsertDependenceCompute(dependences_);

  auto group_upma = GroupDependence();
  DumpSchTree("02_initialize_entry" + std::string(scop_.is_spec_gemm_ ? "_specgemm" : ""), schedule_);
  // 2. set constraints for scheduler
  if (scop_.mod_schedule_shift_) {
    dependences_ = ModDependences(dependences_);
  }
  constraints_ = MakeScheduleConstraints(coincidence);

  // 3. scheduling
  TIMER_START;
#if USE_CACHED_SCHEDULE
  if (!LoadScheduleTreeFromFile(scop_.AddDumpDir("03_computeSchedule.cc"), schedule_)) {
    schedule_ = ComputeSchedule();
  }
#else
  schedule_ = ComputeSchedule();
#endif
  TIMER_SHOW("computeSchedule", std::string(scop_.is_spec_gemm_ ? "_specgemm" : ""));

#if ENABLE_REPLACE_SCHEDULE_HOOK
  if (ReplaceScheduleTree(schedule_)) {
    LOG(INFO) << "schedule tree is replaced";
  }
#endif

  DumpSchTree("03_computeSchedule" + std::string(scop_.is_spec_gemm_ ? "_specgemm" : ""), schedule_);

  isl::schedule original_schedule = schedule_;

  InsertScheduleCompute();

  if (has_grouped_) {
    schedule_ = Ungroup(schedule_, group_upma);
    dependences_ = orig_dependences;
    constraints_ = MakeScheduleConstraints(coincidence);
  }
  DumpSchTree("06_ungroup" + std::string(scop_.is_spec_gemm_ ? "_specgemm" : ""), schedule_);

  TIMER_START;
  ValidateShiftedSchedule(original_schedule, group_upma);
  TIMER_SHOW("validateShiftedSchedule", std::string(scop_.is_spec_gemm_ ? "_specgemm" : ""));
  return schedule_;
}

void Transform::IsContainsCircle(const std::vector<std::vector<int>> &graph, std::vector<int> &vis, int node,
                                 int size) {
  vis[node] = 1;
  for (int i = 0; i < size; ++i) {
    if (graph[node][i] != 0) {
      if (vis[node] == 1) {
        is_circle_ = true;
        break;
      } else if (vis[node] == -1) {
        continue;
      } else {
        IsContainsCircle(graph, vis, i, size);
      }
    }
  }
  vis[node] = -1;
}

void Transform::DfsTopsort(std::vector<std::vector<int>> &graph, std::vector<int> &indegree, std::set<int> &zeros,
                           int cnt, int size, int64_t current_value, int64_t current_max) {
  cnt_dfs_times_++;
  // constraint 1:  return when dfs reaches a limit times.
  if (cnt_dfs_times_ > DFS_TIMES_MAX) return;
  // constraint 2: return when current max is bigger than best result.
  if ((min_topsort_ != -1) && (current_max >= min_topsort_)) return;

  if (cnt == size) {
    min_topsort_ = current_max;
    std::vector<int> res(temp_res_);
    topsort_res_ = res;
  } else {
    for (auto it = zeros.begin(); it != zeros.end(); ++it) {
      std::set<int> zeros_copy(zeros);
      zeros_copy.erase(*it);
      temp_res_[cnt] = *it;
      std::vector<int> temp;

      for (int j = 0; j < size; ++j) {
        if (graph[*it][j] == 1) {
          graph[*it][j] = 0;
          indegree[j]--;
          if (indegree[j] == 0) {
            zeros_copy.insert(j);
          }
          temp.emplace_back(j);
        }
      }
      int64_t updated_value = current_value;
      if (cost_map_.find(temp_res_[cnt]) != cost_map_.end()) {
        updated_value += cost_map_.find(temp_res_[cnt])->second;
      }
      DfsTopsort(graph, indegree, zeros_copy, cnt + 1, size, updated_value, std::max(updated_value, current_max));
      for (int &itj : temp) {
        graph[*it][itj] = 1;
        indegree[itj]++;
      }
    }
  }
}

isl::union_set_list Transform::DependenciesTopsort(const isl::union_set_list &filterlist) {
  if (dependency_list_.empty()) return filterlist;
  if (filterlist.size() == 0) return filterlist;

  // 1. build graph from dependency_list_ and filterlist
  int graph_size = filterlist.size();
  std::unordered_map<isl::id, int, isl::IslIdIslHash> filter_map;
  for (int i = 0; i < graph_size; ++i) {
    isl::union_set temp = filterlist.get_at(i);
    CHECK(temp.n_set() == 1u) << "number of union_set's children in filterlist should be 1";
    filter_map.insert(std::pair<isl::id, int>(temp.get_set_list().get_at(0).get_tuple_id(), i));
  }

  std::vector<std::vector<int>> graph(graph_size, std::vector<int>(graph_size, 0));
  std::vector<int> indegree(graph_size, 0);
  for (auto &i : dependency_list_) {
    isl::id from = i.GetStartNode();
    isl::id to = i.GetEndNode();
    if (filter_map.find(from) != filter_map.end() && filter_map.find(to) != filter_map.end()) {
      int x = filter_map.find(from)->second;
      int y = filter_map.find(to)->second;
      // we only use similar dependency once
      if (graph[x][y] == 0) {
        graph[x][y] = 1;
        indegree[y]++;
      }
      int64_t value;
      if (cost_map_.find(x) == cost_map_.end()) {
        value = i.GetEdgeWeight();
      } else {
        value = cost_map_.find(x)->second + i.GetEdgeWeight();
      }
      cost_map_.insert(std::pair<int, int64_t>(x, value));

      if (cost_map_.find(y) == cost_map_.end()) {
        value = -i.GetEdgeWeight();
      } else {
        value = cost_map_.find(y)->second - i.GetEdgeWeight();
      }
      cost_map_.insert(std::pair<int, int64_t>(y, value));
    }
  }
  // 2. judge if graph has a circle by using dfs
  std::vector<int> vis(graph_size, 0);
  is_circle_ = false;
  for (int i = 0; i < graph_size; ++i) {
    if (vis[i] == -1) {
      continue;
    }
    IsContainsCircle(graph, vis, i, graph_size);
    if (is_circle_) return filterlist;
  }
  // 3. compute all the Topsort list
  if (temp_res_.empty()) {
    temp_res_.insert(temp_res_.begin(), graph_size, 0);
  } else {
    temp_res_.assign(graph_size, 0);
  }
  std::set<int> zeros;
  for (int i = 0; i < graph_size; ++i) {
    if (indegree[i] == 0) {
      zeros.insert(i);
    }
  }
  // minTopsort == -1 means never found a result of dfs.
  min_topsort_ = -1;
  cnt_dfs_times_ = 0;
  DfsTopsort(graph, indegree, zeros, 0, graph_size, 0, 0);

  // 4. output the smallest filterlist
  isl::union_set_list reslist = isl::union_set_list(filterlist.ctx(), static_cast<int>(scop_.stmt_type_.size() - 1));
  for (int i = 0; i < graph_size; ++i) {
    isl::union_set temp = filterlist.get_at(topsort_res_[i]);
    reslist = reslist.add(temp);
  }
  return reslist;
}

isl::schedule Transform::Ungroup(isl::schedule schedule, const isl::union_pw_multi_aff &group_upma) {
  find_filter_ = false;
  auto findAndMarkGroupFilter = [this](isl::schedule_node node) -> isl::schedule_node {
    if (node.isa<isl::schedule_node_filter>() && node.as<isl::schedule_node_filter>().n_children() == 1) {
      find_filter_ = true;
      auto filter_node = node.as<isl::schedule_node_filter>().first_child();
      isl::map_list schmap = filter_node.get_prefix_schedule_union_map().get_map_list();
      if (schmap.size() == 1) {
        isl::id gid = schmap.get_at(0).domain().get_tuple_id();
        if (scop_.group_filter_map_.find(gid) != scop_.group_filter_map_.end()) {
          node = InsertMarknode(node, gid);
        }
      }
    }
    if ((node.isa<isl::schedule_node_domain>()) && (!find_filter_)) {
      find_filter_ = true;
      isl::union_set domain = node.as<isl::schedule_node_domain>().domain();
      isl::set_list setlist = domain.get_set_list();
      isl::id groupid;
      if (setlist.size() == 1) {
        groupid = setlist.get_at(0).get_tuple_id();
      }
      if (scop_.group_filter_map_.find(groupid) != scop_.group_filter_map_.end()) {
        while (node.has_children()) {
          if (node.n_children() > 1) {
            return node.root();
          } else {
            node = node.first_child();
          }
        }
        node = InsertMarknode(node, groupid);
        node = node.root();
      }
    }
    return node;
  };
  schedule = schedule.get_root().map_descendant_bottom_up(findAndMarkGroupFilter).get_schedule();

  schedule = schedule.pullback(group_upma);

  auto replaceUngroupedFilterWithSequence = [this](isl::schedule_node node) -> isl::schedule_node {
    if (node.isa<isl::schedule_node_mark>()) {
      isl::schedule_node_mark marknode = node.as<isl::schedule_node_mark>();
      isl::id markid = marknode.get_id();
      isl::union_set_list filterlist = scop_.group_filter_map_[markid];
      isl::union_set_list resfilterlist = DependenciesTopsort(filterlist);
      if (scop_.group_filter_map_.find(markid) != scop_.group_filter_map_.end()) {
        node = node.del();
        node = node.insert_sequence(resfilterlist);
      }
    }
    return node;
  };
  schedule = schedule.get_root().map_descendant_bottom_up(replaceUngroupedFilterWithSequence).get_schedule();

  return schedule;
}

/* Mark each scalar statement with a "realize_UB" mark node. "root" should be
 * either a domain node or a filter node.
 *
 * First, check whether each statement in "root" is scalar. Each set of the
 * union set represented by "root" represents a statement. We determine a scalar
 * statement with "HasNoDims" function, checking whether a give "set" has dims.
 *
 * Next, check whether the subtree of "root" has permutable bands, and return
 * "root" if there are any permutable bands.
 *
 * Obtain the outermost permutable band, and this would go down to either a leaf
 * node or a sequence/set node.
 *
 * If it comes to a leaf node, "root" represents a single scalar statement. Insert
 * an empty band and mark this empty band with a "realize_UB" mark.
 *
 * If a sequence/set node is encountered, meaning "root" represents multiple
 * scalar statements. Mark each child recursively with a "realize_UB" mark.
 *
 * Return the original "root" in other cases.
 */
isl::schedule_node Transform::TryMarkScalarStmts(const isl::schedule_node &root) {
  // Return "root" if given an inappropriate node
  if (!root.isa<isl::schedule_node_domain>() && !root.isa<isl::schedule_node_filter>()) return root;
  // Check whether each stmt is scalar
  auto domain = root.isa<isl::schedule_node_domain>() ? root.as<isl::schedule_node_domain>().get_domain()
                                                      : root.as<isl::schedule_node_filter>().get_filter();
  if (!domain.every_set(&HasNoDims)) return root;

  // Return if there exist any band nodes
  if (SubtreeHasPermutableBands(root)) return root;

  auto node = GetOuterBand(root);
  // Mark to copy to UB
  if (node.isa<isl::schedule_node_leaf>() || (IsSequenceOrSet(node))) {
    node = InsertEmptyPermutableBand(node);
    auto tag = REALIZE_UB;
    node = node.insert_mark(isl::id(node.ctx(), tag));
    return node;
  }

  // Return if none of the above
  return root;
}

isl::schedule Transform::ReorderInvariantSetSchedule(isl::schedule &sch) {
  isl::schedule_node root = sch.get_root();
  isl::schedule_node outer_band = Transform::GetOuterBand(root);
  if (outer_band.isa<isl::schedule_node_set>()) {
    std::vector<size_t> new_filters;
    std::vector<size_t> invariant_filters;
    std::vector<size_t> rest_filters;
    for (unsigned int i = 0; i < outer_band.n_children(); ++i) {
      isl::schedule_node node = outer_band.get_child(i);
      auto filter = node.as<isl::schedule_node_filter>();
      isl::union_set sets = filter.get_filter();
      unsigned int invariant_count = 0;
      sets.foreach_set([&invariant_count, this](const isl::set &s) -> void {
        if (s.n_dim() == 0 && this->invariant_state_.count(s.get_tuple_name()) > 0) {
          invariant_count++;
        }
      });

      if (invariant_count == sets.n_set()) {
        invariant_filters.push_back(i);
      } else {
        rest_filters.push_back(i);
      }
    }

    for (unsigned long &invariant_filter : invariant_filters) {
      new_filters.push_back(invariant_filter);
    }

    for (unsigned long &rest_filter : rest_filters) {
      new_filters.push_back(rest_filter);
    }

    std::unordered_map<size_t, size_t> old_to_new_map;
    for (size_t i = 0; i < new_filters.size(); ++i) {
      old_to_new_map.emplace(new_filters[i], i);
    }

    outer_band = ReorderFilters(outer_band, old_to_new_map);
  }
  return outer_band.get_schedule();
}

isl::union_map Transform::RemoveInvariantDependence() {
  isl::schedule_node root = schedule_.get_root();
  isl::schedule_node outer_band = Transform::GetOuterBand(root);
  if (outer_band.as<isl::schedule_node_sequence>() || outer_band.as<isl::schedule_node_set>()) {
    for (unsigned int i = 0; i < outer_band.n_children(); ++i) {
      isl::schedule_node node = outer_band.get_child(i);
      auto filter = node.as<isl::schedule_node_filter>();
      isl::union_set sets = filter.filter();
      if (sets.n_set() == 1) {
        sets.foreach_set([this](const isl::set &s) -> void {
          if (s.n_dim() == 0) {
            // scalar single filter
            if (this->invariant_state_.count(s.get_tuple_name()) == 0) {
              this->invariant_state_.emplace(s.get_tuple_name(), 1);
            }
          }
        });
      }
    }
  }

  if (invariant_state_.empty()) {
    return dependences_;
  }

  isl::union_map preserved = isl::union_map::empty(dependences_.get_space());

  dependences_.foreach_map([&preserved, this](const isl::map &m) -> void {
    auto map_domain = m.domain();
    auto map_range = m.range();
    bool invariant_dependence = (this->invariant_state_.count(map_domain.get_tuple_name()) > 0) &&
                                (map_domain.n_dim() == 0) && (map_range.n_dim() > 0);

    if (invariant_dependence) {
      this->has_invariant_dependence_ = true;
    }

    if (!invariant_dependence) {
      preserved = preserved.add_map(m);
    }
  });
  return preserved;
}

void CheckAndRemoveUninitializedCopyin(isl::union_map &copy_in, const Scop::Binds &binds) {
  isl::union_set copyin_range = copy_in.range();
  auto ForeachSetFn = [&copy_in, &binds](const isl::set &set) {
    std::string tensor_name = set.get_tuple_name();
    bool defined = false;
    for (auto bind : binds) {
      if (bind.first->op->name == tensor_name) {
        defined = true;
      }
    }
    if (!defined) {
      LOG(WARNING) << "WARNING: detected access to uninitialized tensor " << tensor_name;
      LOG(WARNING) << "uninitialized accesses: " << set;
      copy_in = copy_in.subtract_range(set);
    }
  };
  copyin_range.foreach_set(ForeachSetFn);
}

isl::schedule SplitOuterBand(const isl::schedule &curr_schedule) {
  isl::schedule_node node = curr_schedule.get_root();
  while (!node.isa<isl::schedule_node_band>()) {
    node = node.child(0);
  }
  isl::schedule_node_band band = node.as<isl::schedule_node_band>();
  unsigned i = 0;
  unsigned n = band.n_member();
  for (; i < n; ++i) {
    if (!band.member_get_coincident(i)) {
      break;
    }
  }
  if ((n <= 1) || (i == 0) || (i == n)) {
    return node.get_schedule();
  }
  node = band.split(i);
  return node.get_schedule();
}

bool IsStmtScheduleContainsReduceAxis(const isl::pw_aff &stmt,
                                      const std::unordered_set<std::string> &reduce_axis_list) {
  int num_dims = stmt.domain().n_dim();
  isl_space *domain_space = stmt.domain().get_space().get();
  for (int dim = 0; dim < num_dims; ++dim) {
    const char *axis_name = isl_space_get_dim_name(domain_space, isl_dim_out, dim);
    if (axis_name == nullptr) continue;
    if (reduce_axis_list.count(axis_name) == 0) continue;
    if (isl_pw_aff_involves_dims(stmt.get(), isl_dim_in, dim, 1)) return true;
  }
  return false;
}

bool IsDimScheduleContainsReduceAxis(const isl::union_pw_aff &schedule, const ReduceStmtMap &reduce_stmts) {
  bool found_reduce_axis = false;
  auto stmt_list = schedule.get_pw_aff_list();
  stmt_list.foreach([&found_reduce_axis, &reduce_stmts](const isl::pw_aff &stmt) -> void {
    isl::id stmt_id = stmt.domain().get_tuple_id();
    if (reduce_stmts.count(stmt_id)) {
      std::unordered_set<std::string> reduce_axis_list;
      for (const auto &axis : reduce_stmts.at(stmt_id)) {
        reduce_axis_list.insert(axis);
      }
      if (IsStmtScheduleContainsReduceAxis(stmt, reduce_axis_list)) {
        found_reduce_axis = true;
      }
    }
  });
  return found_reduce_axis;
}

isl::schedule ResetCoincidenceOfReduceAxis(const isl::schedule &schedule, const ReduceStmtMap &reduce_stmts) {
  auto fn = [&reduce_stmts](isl::schedule_node node) -> isl::schedule_node {
    if (auto band = node.as<isl::schedule_node_band>()) {
      int num_dims = static_cast<int>(band.n_member());
      for (int dim = 0; dim < num_dims; ++dim) {
        bool is_coincident = band.member_get_coincident(dim);
        if (!is_coincident) continue;
        auto dim_schedule = band.get_partial_schedule().get_union_pw_aff(dim);
        if (IsDimScheduleContainsReduceAxis(dim_schedule, reduce_stmts)) {
          LOG(INFO) << "reset coincidence of reduce axis on dim " << dim << " in partial schedule: " << dim_schedule;
          node = band.member_set_coincident(dim, false);
          band = node.as<isl::schedule_node_band>();
        }
      }
    }
    return node;
  };
  return schedule.get_root().map_descendant_bottom_up(fn).get_schedule();
}

/*
 * Sometimes, coincident is set to `0` for some axes that can actually be parallelised in computed schedule tree.
 * Since we have no idea why these cases happen, we offer such transfrom to set all coincident to `1`.
 * Please be careful to do such transfrom since it may cause some incorrect result.
 */
isl::schedule Transform::SetAllCoincident(const isl::schedule &schedule) {
  auto fn = [](isl::schedule_node node) -> isl::schedule_node {
    if (auto band = node.as<isl::schedule_node_band>()) {
      int num_dims = static_cast<int>(band.n_member());
      for (int dim = 0; dim < num_dims; ++dim) {
        bool is_coincident = band.member_get_coincident(dim);
        if (is_coincident) continue;
        node = band.member_set_coincident(dim, true);
        band = node.as<isl::schedule_node_band>();
      }
    }
    return node;
  };
  return schedule.get_root().map_descendant_bottom_up(fn).get_schedule();
}

/*
 * "with" stmt aims to work around the irregular problem.
 * By default, the "realize_UB" mark is on the outer band. However, for tensor-of-tensor,
 * the intermediate tensor may be too large if realized in the outermost scope.
 * To narrow down the scope, we move "realize_UB" mark to the filter node.
 * If all filter nodes of the band are "with" stmts, we remove the outer "realize_UB" mark.
 */
isl::schedule Scop::ChangeMarkNodePosition(const isl::schedule &sch_mark) {
  std::unordered_set<std::string> ids = ExtractWithStmtId();
  if (ids.empty()) {
    return sch_mark;
  }

  auto fn = [&ids](isl::schedule_node node) -> isl::schedule_node {
    if (node.isa<isl::schedule_node_mark>()) {
      std::string mark_id = node.as<isl::schedule_node_mark>().get_id().get_name();
      if (mark_id == "realize_UB" && node.child(0).isa<isl::schedule_node_band>()) {
        if (node.child(0).child(0).isa<isl::schedule_node_sequence>()) {
          node = node.get_child(0).get_child(0);  // sequence
          bool delete_outer_mark = true;
          int n = node.n_children();
          for (int i = 0; i < n; i++) {
            CHECK(node.child(i).isa<isl::schedule_node_filter>());
            isl::schedule_node_filter filter_node = node.child(i).as<isl::schedule_node_filter>();
            bool is_not_with_stmt = filter_node.get_filter().every_set(
              [&ids](const isl::set &s) -> bool { return (ids.count(s.get_tuple_name()) == 0); });
            if (is_not_with_stmt) {
              delete_outer_mark = false;
            } else {
              node = node.child(i).child(0);
              node = node.insert_mark(isl::id(node.ctx(), mark_id));
              node = node.parent().parent();
            }
          }
          node = node.parent().parent();
          if (delete_outer_mark) {
            node = node.del();
          }
        }
      }
    }
    return node;
  };

  return sch_mark.get_root().map_descendant_bottom_up(fn).get_schedule();
}

isl::schedule Scop::LabelRealizeOutPosition(const isl::schedule &sch_label) const {
  auto fn_ = [](isl::schedule_node node) -> isl::schedule_node {
    if (node.isa<isl::schedule_node_mark>()) {
      if (REALIZE_UB == node.as<isl::schedule_node_mark>().get_id().get_name() &&
          node.child(0).isa<isl::schedule_node_band>()) {
        auto band = node.child(0).as<isl::schedule_node_band>();

        unsigned pos = UINT_MAX;
        auto updatePos_ = [&pos](isl::schedule_node node) -> isl::schedule_node {
          if (node.isa<isl::schedule_node_filter>()) {
            node = node.get_child(0);
            if (node.isa<isl::schedule_node_band>()) {
              auto band = node.as<isl::schedule_node_band>();
              CHECK_LT(band.n_member(), UINT_MAX);
              for (unsigned i = 0; i < band.n_member(); ++i) {
                if (!band.member_get_coincident(i)) {
                  if (i < pos) pos = i;
                  break;
                }
              }
            }
            node = node.parent();
          }
          return node;
        };

        static_cast<void>(band.map_descendant_bottom_up(updatePos_));

        for (unsigned i = 0; i < band.n_member(); ++i) {
          if (!band.member_get_coincident(i)) {
            if (i < pos) pos = i;
            break;
          }
        }

        if (pos < band.n_member()) {
          node = node.del();
          node = node.as<isl::schedule_node_band>().split(pos);
          node = node.child(0);
          node = node.insert_mark(isl::id(node.ctx(), REALIZE_UB));
          node = node.insert_mark(isl::id(node.ctx(), ALLOC_REALIZE_OUT));
          node = node.parent();
        }
      }
    }
    return node;
  };
  return sch_label.get_root().map_descendant_bottom_up(fn_).get_schedule();
}

std::vector<bool> getIsolateVector(const isl::schedule_node_band &node) {
  auto build_options = node.get_ast_build_options().get_set_list();
  std::vector<bool> isolate_vector(node.n_member(), true);
  for (auto idx = 0u; idx < build_options.size(); ++idx) {
    if (build_options.get_at(idx).get_tuple_name() == "isolate") {
      const isl::set &isolate_set = build_options.get_at(idx);
      for (int dim = 0; dim < static_cast<int>(node.n_member()); dim++) {
        isolate_vector[dim] = (isolate_set.simple_hull().dim_max_val(dim).get_num_si() > 0);
      }
      break;
    }
  }
  return isolate_vector;
}

bool InjectMulticoreToBand(isl::schedule_node &band_node) {
  auto node = band_node.as<isl::schedule_node_band>();
  if (node.is_null()) return false;
  if (node.n_member() < 1) return false;
  if (!node.get_permutable()) return false;

  auto isolate_vector = getIsolateVector(node);
  bool has_coincident = false;
  std::string mark = "multicore_coincident";
  for (int dim = 0; dim < static_cast<int>(node.n_member()); ++dim) {
    bool is_dim_coincident = isolate_vector[dim] && node.member_get_coincident(dim);
    has_coincident = has_coincident || is_dim_coincident;
    mark += "_" + std::to_string(is_dim_coincident);
  }
  if (has_coincident) {
    band_node = band_node.insert_mark(isl::id(band_node.ctx(), mark));
  }
  return has_coincident;
}

isl::schedule_node &ObtainSequenceOrSetNodeAncestor(isl::schedule_node &node) {
  do {
    node = node.parent();
  } while (!node.isa<isl::schedule_node_sequence>() && !node.isa<isl::schedule_node_set>());
  return node;
}

bool InjectMulticoreToChildrenBands(isl::schedule_node &sequence_node) {
  bool has_multicore = false;
  for (unsigned int filter = 0; filter < sequence_node.n_children(); ++filter) {
    auto filter_node = sequence_node.get_child(filter);
    auto band = Transform::GetOuterBand(filter_node);
    if (InjectMulticoreToBand(band)) {
      has_multicore = true;
      sequence_node = ObtainSequenceOrSetNodeAncestor(band);
    }
  }
  return has_multicore;
}

bool Scop::SingleMulticoreBand(isl::schedule_node &outer_band) {
  if (outer_band.as<isl::schedule_node_sequence>() || outer_band.as<isl::schedule_node_set>()) {
    int multi_core_band = 0;
    for (unsigned int i = 0; i < outer_band.n_children(); ++i) {
      isl::schedule_node node = outer_band.get_child(i);
      if (node.isa<isl::schedule_node_filter>()) {
        auto filter = node.as<isl::schedule_node_filter>();
        if (filter.has_children()) {
          auto node0 = filter.get_child(0);
          if (node0.isa<isl::schedule_node_band>() && node0.as<isl::schedule_node_band>().n_member() >= 1) {
            multi_core_band++;
          }
        }
      }
    }
    if (multi_core_band == 1) {
      return true;
    }
  }
  return false;
}

bool Scop::InjectMulticoreToSchedule(isl::schedule_node &outer_band) {
  if (outer_band.as<isl::schedule_node_band>()) {
    return InjectMulticoreToBand(outer_band);
  } else if (outer_band.as<isl::schedule_node_sequence>() || outer_band.as<isl::schedule_node_set>()) {
    if (SingleMulticoreBand(outer_band)) {
      for (unsigned int i = 0; i < outer_band.n_children(); ++i) {
        isl::schedule_node node = outer_band.get_child(i);
        if (node.isa<isl::schedule_node_filter>()) {
          auto filter = node.as<isl::schedule_node_filter>();
          if (filter.has_children() && filter.get_child(0).isa<isl::schedule_node_band>() &&
              filter.get_child(0).as<isl::schedule_node_band>().n_member() >= 1) {
            isl::schedule_node tmp = filter.get_child(0);
            bool injected = InjectMulticoreToBand(tmp);
            outer_band = ObtainSequenceOrSetNodeAncestor(tmp);
            return injected;
          }
        }
      }
    }
    bool is_bands_independent = data_.inter_band_dependency.is_empty();
    if (!is_bands_independent) {
      // Conv outer bands indeed have inter-band dependency, but it will be fixed in post_fusion,
      // so Conv can still use multicore. This is actually dangerous and may need double check.
      if (!this->IsConv()) return false;
    }
    return InjectMulticoreToChildrenBands(outer_band);
  }
  return false;
}

isl::schedule Scop::MarkOuterMost(const isl::schedule &schedule_mark) {
  isl::schedule_node root = schedule_mark.get_root();
  isl::schedule_node outer_band = Transform::GetOuterBand(root);
  bool has_multi_core = InjectMulticoreToSchedule(outer_band);
  if (has_multi_core) {
    return outer_band.get_schedule();
  } else {
    LOG(INFO) << "This operator is not capable of using multi-core. "
              << "Possible reasons are: "
              << "1) there is dependency between outer bands. "
              << "2) outer bands are not tiled (only tiles of outer band can use multicore).";
    return schedule_mark;
  }
}

isl::schedule Scop::MarkFuseOp(const isl::schedule &schedule_mark) const {
  auto fn = [](isl::schedule_node node) -> isl::schedule_node {
    if (node.isa<isl::schedule_node_mark>()) {
      std::string mark_id = node.as<isl::schedule_node_mark>().get_id().get_name();
      size_t pos = mark_id.find("UBL0");
      if (pos != std::string::npos) {
        std::string m = "fuse_vector";
        node = node.insert_mark(isl::id(node.ctx(), m));
        node = node.parent();
      }
    }
    return node;
  };
  return schedule_mark.get_root().map_descendant_bottom_up(fn).get_schedule();
}

isl::schedule Scop::ReorderMarkNodes(const isl::schedule &schedule_mark) const {
  auto fn = [](isl::schedule_node node) -> isl::schedule_node {
    if (node.isa<isl::schedule_node_mark>()) {
      // mark node cannot be inserted between sequence node and its filter children, so skip reorder
      if (node.get_child(0).as<isl::schedule_node_sequence>()) return node;

      std::string mark_id = node.as<isl::schedule_node_mark>().get_id().get_name();
      size_t pos = mark_id.find("realize_");
      if (pos != std::string::npos) {
        node = node.del();
        node = node.get_child(0);
        node = node.insert_mark(isl::id(node.ctx(), mark_id));
        node = node.parent();
      }
    }
    return node;
  };
  return schedule_mark.get_root().map_descendant_bottom_up(fn).get_schedule();
}

isl::schedule Scop::GroupStatements(const isl::schedule &sch_group, bool &has_group) {
  int cluster_id = 0;
  has_group = false;
  auto fn = [&cluster_id, &has_group, this](isl::schedule_node node) -> isl::schedule_node {
    if (node.isa<isl::schedule_node_sequence>() && node.n_children() > 1 &&
        !node.parent().isa<isl::schedule_node_domain>()) {
      isl::schedule_node_sequence seq_node = node.as<isl::schedule_node_sequence>();
      bool should_group = true;
      isl::union_set_list filter_list(node.ctx(), seq_node.n_children());

      for (unsigned int i = 0; i < seq_node.n_children(); i++) {
        isl::schedule_node child = seq_node.child(i);
        if (!child.isa<isl::schedule_node_filter>() || !child.child(0).isa<isl::schedule_node_leaf>()) {
          should_group = false;
          break;
        } else {
          isl::schedule_node_filter filnode = child.as<isl::schedule_node_filter>();
          filter_list = filter_list.add(filnode.get_filter());
        }
      }
      if (should_group) {
        has_group = true;
        isl::id gid = isl::id(node.ctx(), std::string("group") + std::to_string(cluster_id));
        group_filter_map_[gid] = filter_list;
        cluster_id++;
        isl_schedule_node *snode = isl_schedule_node_group(node.copy(), gid.release());
        node = isl::manage(snode);
      }
    }
    return node;
  };
  return sch_group.get_root().map_descendant_bottom_up(fn).get_schedule();
}

std::vector<std::string> ExtractDimNames(const isl::aff &aff) {
  std::vector<std::string> dim_names;
  int dims = isl_aff_dim(aff.get(), isl_dim_in);
  CHECK_GE(dims, 0);
  for (int i = 0; i < dims; ++i) {
    isl_val *coef_val = isl_aff_get_coefficient_val(aff.get(), isl_dim_in, i);
    int coef = isl_val_get_num_si(coef_val);
    static_cast<void>(isl_val_free(coef_val));
    if (coef != 0) {
      auto dim_name = std::string(isl_aff_get_dim_name(aff.get(), isl_dim_in, i));
      dim_names.push_back(dim_name);
    }
  }
  return dim_names;
}

isl::multi_union_pw_aff MergeTwoUPALs(const isl::multi_union_pw_aff &partial_schedule,
                                      const std::vector<isl::union_pw_aff> &dims_with_if,
                                      const std::vector<isl::union_pw_aff> &dims_without_if) {
  auto num_dims_with_if = dims_with_if.size();
  auto num_dims_without_if = dims_without_if.size();
  CHECK(partial_schedule.size() == num_dims_with_if + num_dims_without_if);
  auto new_schedule = partial_schedule;
  for (unsigned dim = 0; dim < num_dims_with_if; ++dim) {
    new_schedule = new_schedule.set_at(dim, dims_with_if[dim]);
  }
  for (unsigned dim = 0; dim < num_dims_without_if; ++dim) {
    new_schedule = new_schedule.set_at(dim + num_dims_with_if, dims_without_if[dim]);
  }
  return new_schedule;
}

void MergeTwoDimMaps(const std::vector<unsigned> &in1, const std::vector<unsigned> &in2, std::vector<unsigned> &out) {
  out.resize(in1.size() + in2.size());
  for (unsigned i = 0; i < in1.size(); ++i) {
    out[in1[i]] = i;
  }
  for (unsigned i = 0; i < in2.size(); ++i) {
    out[in2[i]] = i + in1.size();
  }
}

/*
 * Reorder the partial schedule such that isl::union_pw_aff with range var in cond_vars are
 * ordered before others.
 *
 * Example:
 * [{ S_0[j, k, l] -> [((k) mod 16)] },
 *  { S_0[j, k, l] -> [((l) mod 16)] },
 *  { S_0[j, k, l] -> [(0)] },
 *  { S_0[j, k, l] -> [((j) mod 32)] }]
 *
 * If "j" appears in the conditions, the partial schedule is transformed to:
 *
 * [{ S_0[j, k, l] -> [((j) mod 32)] },
 *  { S_0[j, k, l] -> [((k) mod 16)] },
 *  { S_0[j, k, l] -> [((l) mod 16)] },
 *  { S_0[j, k, l] -> [(0)] }]
 */
isl::multi_union_pw_aff ReorderLocalSchedule(const CondVarsMap &cond_vars,
                                             const isl::multi_union_pw_aff &partial_schedule,
                                             std::vector<unsigned> &dim_map, bool &need_update) {
  std::vector<isl::union_pw_aff> dims_with_if, dims_without_if;
  std::vector<unsigned> with_if_dim_map, without_if_dim_map;
  need_update = false;
  auto original_upal = partial_schedule.union_pw_aff_list();
  unsigned upal_size = original_upal.size();
  for (unsigned dim = 0; dim < upal_size; ++dim) {
    auto dim_schedule = original_upal.get_at(dim);
    bool found_dim_in_cond = false;
    dim_schedule.get_pw_aff_list().foreach([&cond_vars, &found_dim_in_cond](const isl::pw_aff &stmt_schedule) -> void {
      stmt_schedule.foreach_piece([&cond_vars, &found_dim_in_cond](const isl::set &set, const isl::aff &aff) -> void {
        isl::id stmt_id = set.get_tuple_id();
        if (cond_vars.count(stmt_id) == 0) return;
        const auto &cond_vars_in_stmt = cond_vars.at(stmt_id);
        auto dim_names = ExtractDimNames(aff);
        for (const auto &dim_name : dim_names) {
          if (cond_vars_in_stmt.count(dim_name)) found_dim_in_cond = true;
        }
      });
    });
    if (found_dim_in_cond) {
      with_if_dim_map.push_back(dim);
      dims_with_if.push_back(dim_schedule);
      need_update = true;
    } else {
      without_if_dim_map.push_back(dim);
      dims_without_if.push_back(dim_schedule);
    }
  }

  if (need_update) {
    MergeTwoDimMaps(with_if_dim_map, without_if_dim_map, dim_map);
    return MergeTwoUPALs(partial_schedule, dims_with_if, dims_without_if);
  } else {
    return partial_schedule;
  }
}

/*
 * isl::schedule_node_band does not provide an interface to update the partial schedule,
 * so we have to delete the band node, copy other attributes from the original node and
 * insert the new partial schedule.
 *
 * Note that the member coincident values needs to be updated according to the mapping
 * between original and new dims.
 */
isl::schedule_node setLocalSchedule(const isl::schedule_node_band &band,
                                    const isl::multi_union_pw_aff &partial_schedule,
                                    const std::vector<unsigned> &dim_map) {
  auto removed_band = band.del();
  auto new_band_obj = removed_band.insert_partial_schedule(partial_schedule);
  auto new_band = new_band_obj.copy();
  new_band = isl_schedule_node_band_set_permutable(new_band, band.get_permutable());
  auto ast_build_options = band.get_ast_build_options();
  new_band = isl_schedule_node_band_set_ast_build_options(new_band, ast_build_options.copy());
  unsigned n_member = band.n_member();
  CHECK(dim_map.size() == n_member);
  for (unsigned i = 0; i < n_member; ++i) {
    bool coincident = band.member_get_coincident(i);
    unsigned new_member = dim_map[i];
    new_band = isl_schedule_node_band_member_set_coincident(new_band, new_member, coincident);
  }
  return isl::manage(new_band);
}

void Transform::IntraTileReschedule(isl::schedule &sched, bool tile_inner_band, bool is_spec_gemm) {
  isl::schedule_node root = sched.get_root();
  if (tile_inner_band)
    sched = RescheduleInnerBand(root).get_schedule();
  else
    sched = RescheduleSchTree(root).get_schedule();
}

isl::schedule_node RewriteLeafBandNode(const CondVarsMap &cond_vars, const isl::schedule_node_band &band) {
  auto partial_schedule = band.get_partial_schedule();
  std::vector<unsigned> dim_map;
  bool need_update = false;
  auto new_partial_schedule = ReorderLocalSchedule(cond_vars, partial_schedule, dim_map, need_update);
  if (!need_update)
    return band;
  else
    return setLocalSchedule(band, new_partial_schedule, dim_map);
}

/*
 * Reorder the members of the leaf-band partial schedule (if it is permutable)
 * such that loop vars that appear in "if" conditions are the outer loops.
 * This aims to promote the "if" condition to the outermost loop, and maximize
 * the size of unconditional vectorized computation.
 */
isl::schedule Scop::ReorderInnerBandLoops(const isl::schedule &curr_schedule) const {
  isl::schedule_node root = curr_schedule.get_root();
  CondVarsMap cond_vars = ExtractCondVarsMap();
  root = root.map_descendant_bottom_up([&cond_vars](const isl::schedule_node &node) -> isl::schedule_node {
    bool is_leaf_band =
      node.as<isl::schedule_node_band>() && node.n_children() == 1 && node.first_child().as<isl::schedule_node_leaf>();
    if (!is_leaf_band) return node;

    auto band = node.as<isl::schedule_node_band>();
    if (!band.get_permutable()) return node;
    return RewriteLeafBandNode(cond_vars, band);
  });
  return root.get_schedule();
}

void Scop::ComputeTransferCopyin(isl::union_map &fake_copyin) {
  // compute fake copyin
  fake_copyin = fake_copyin.subtract(data_.copyin);
  data_.fake_copyin = fake_copyin;
  isl::union_map raw_writes = data_.writes.domain_factor_domain();
  isl::union_map raw_reads = data_.reads.domain_factor_domain();
  isl::union_map raw_copyin = data_.copyin.domain_factor_domain();
  isl::union_map reads = fake_copyin.domain_factor_domain();
  isl::union_map transfer_copyin = fake_copyin;
  while (!reads.is_empty()) {
    isl::union_map writes = raw_writes.intersect_range(reads.range());
    isl::union_map dependence = DependenceAnalysis(writes, reads, writes, this->sch_);
    isl::union_set stmt = dependence.domain().universe();
    data_.transfer_stmt = data_.transfer_stmt.unite(stmt);
    reads = raw_reads.intersect_domain(stmt);

    // compute transfer copyin
    isl::union_map target_acc = raw_writes.intersect_domain(stmt);
    isl::union_map relation = target_acc.reverse().apply_range(reads);
    transfer_copyin = transfer_copyin.apply_range(relation);
    isl::union_map copyin = transfer_copyin.intersect_range(raw_copyin.range().universe());
    data_.reads = data_.reads.unite(copyin);
    data_.copyin = data_.copyin.unite(copyin);
    transfer_copyin = transfer_copyin.subtract(copyin);
    reads = reads.subtract(raw_copyin);
    reads = reads.subtract(fake_copyin.domain_factor_domain());
  }
}

void Scop::TransferStmt(isl::schedule &t_sch) {
  isl::schedule_node root_ = t_sch.get_root();
  isl::schedule_node node = Transform::GetOuterBand(root_);
  if (node.isa<isl::schedule_node_sequence>() || node.isa<isl::schedule_node_set>()) {
    int n = static_cast<int>(node.n_children());
    for (int i = 0; i < n; ++i) {
      isl::schedule_node child = node.child(i);
      CHECK(child.isa<isl::schedule_node_filter>()) << "The child of set or sequence must filter!";
      isl::schedule_node_filter filter_node = child.as<isl::schedule_node_filter>();
      isl::union_set filter = filter_node.get_filter();
      if (!filter.intersect(data_.transfer_stmt).is_empty()) {
        filter = filter.subtract(data_.transfer_stmt);
        child = isl::manage(isl_schedule_node_filter_set_filter(child.copy(), filter.copy()));
        node = child.parent();
        t_sch = node.get_schedule();
      }
    }
  }
}

isl::schedule_node InsertNodeForAllocCImpl(isl::schedule_node node) {
  if (node.isa<isl::schedule_node_mark>()) {
    if (node.as<isl::schedule_node_mark>().get_id().get_name() == REALIZE_L1) {
      node = node.del();
      node =
        node.as<isl::schedule_node_band>().split(static_cast<int>(node.as<isl::schedule_node_band>().n_member()) - 1);
      node = node.child(0);
      node = node.insert_mark(isl::id(node.ctx(), REALIZE_L1));
      node = node.insert_mark(isl::id(node.ctx(), ALLOC_C));
      node = node.parent();
    }
  }
  return node;
}

isl::schedule Scop::InsertNodeForAllocC(isl::schedule &sched) {
  // add alloc_C
  if (is_spec_gemm_ || IsGemm() || IsConvBackpropFilter()) {
    sched = sched.get_root().map_descendant_bottom_up(InsertNodeForAllocCImpl).get_schedule();
    DumpSchTree("08_4_alloc_C" + std::string(is_spec_gemm_ ? "_specgemm" : ""), sched);
  }
  return sched;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
