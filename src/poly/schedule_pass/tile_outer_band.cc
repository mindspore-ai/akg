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
#include <cmath>
#include <iostream>
#include "tile_outer_band.h"
#include "build_module.h"
#include "poly/scop.h"
#include "poly/schedule_tree_util.h"
#include "poly/schedule_pass/transfer_stmt.h"
#include "poly/schedule_pass/try_mark_scalar_stmt.h"
#include "poly/schedule_pass_gpu/operator_mapping_strategy.h"
#include "poly/reduce_manager.h"

namespace akg {
namespace ir {
namespace poly {

constexpr int64_t DEFAULT_TILE_SIZE = 65535;

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

isl::schedule TileOuterBand::Run(isl::schedule sch) {
  scop_info_.analysis_result_.InitScheduleMapBeforeTile(scop_info_.GetCtx());
  if (scop_info_.user_config_.GetTarget() == TARGET_CCE) {
    if (!scop_info_.mmu_info_.IsSpecGemm() && (scop_info_.mmu_info_.IsConv() || scop_info_.mmu_info_.IsGemm())) {
      scop_info_.analysis_result_.SetScheduleMapBeforeTile(sch.get_map());
    }
  } else {
    scop_info_.analysis_result_.SetScheduleMapBeforeTile(sch.get_map());
  }
  isolate_tile_ = std::make_unique<IsolateTileManager>(scop_info_);

  auto final_schedule =
    TileOuterBandHelper(sch, std::bind(&TileOuterBand::MarkOuterPermutable, this, std::placeholders::_1));

  if (scop_info_.user_config_.GetTarget() == TARGET_CCE) {
    scop_info_.AddPartitionInfoToData(AddTileInfo(isolate_tile_->partition_info_));
    scop_info_.analysis_result_.SetIsTiled(true);
    if (sch.plain_is_equal(final_schedule)) {
      final_schedule = TryMarkScalarStmt(pass_info_).Run(final_schedule);
    }
  }

  auto map_before_tile = sch.get_map();
  if (final_schedule.get_map().is_equal(map_before_tile) && scop_info_.user_config_.GetConsiderCoincidence()) {
    if (scop_info_.user_config_.GetTarget() == TARGET_CCE) {
      scop_info_.analysis_result_.SetRestartPassName(RestartPassName::TILE_OUTER_BAND);
    } else {
      scop_info_.analysis_result_.SetRestartPassName(RestartPassName::EXIT);
    }
  }

  if (scop_info_.user_config_.GetIsTuning()) {
    // restore schedule before tiling as input of GenerateTilingSpace
    final_schedule = sch;
  }
  return final_schedule;
}

isl::schedule TileOuterBand::TileOuterBandHelper(const isl::schedule sch,
                                                 const std::function<isl::schedule_node(isl::schedule_node)> &f) {
  InitDimensionInfo(sch);
  MergeTilingInfo();
  ShowDimInfo();

  // 1. obtain the outermost tilable band
  isl::schedule_node node = GetOuterBand(sch.get_root());

  // 2. Traverse the descendants of "node" (including the node itself)
  // in depth first postorder via the callback function.
  node = ReverseTraverseChild(node, f);
  return node.get_schedule();
}

bool TileOuterBand::SubtreeHasPermutableBands(const isl::schedule_node &node) {
  bool all_non_permutable = false;
  all_non_permutable =
    node.every_descendant([&, this](const isl::schedule_node &node) -> bool { return BoolNot(IsPermutable(node)); });

  return BoolNot(all_non_permutable);
}

int TileOuterBand::IsCandidate(const isl::schedule_node &node) {
  if (node.isa<isl::schedule_node_leaf>()) {
    return 1;
  }

  int permutable = static_cast<int>(IsPermutable(node));
  if (permutable) {
    return permutable;
  }
  if (node.isa<isl::schedule_node_filter>()) {
    return 0;
  }

  permutable = static_cast<int>(SubtreeHasPermutableBands(node));
  if (permutable < 0) {
    return -1;
  }
  return static_cast<int>(!permutable);
}

int TileOuterBand::IsOuterTilable(const isl::schedule_node &node) {
  int tilable = IsCandidate(node);
  isl::schedule_node ancestor;

  if (tilable < 0) {
    return -1;
  } else if (tilable == 0) {
    return 0;
  }

  tilable = 0;
  ancestor = node;
  while (ancestor.has_parent()) {
    ancestor = ancestor.parent();
    tilable = IsCandidate(ancestor);
    if (tilable) break;
  }

  return static_cast<int>(BoolNot(static_cast<bool>(tilable)));
}

bool TileOuterBand::IsPermutable(const isl::schedule_node &node) {
  if (!node) return false;
  if (!node.isa<isl::schedule_node_band>()) return false;
  if (!node.as<isl::schedule_node_band>().get_permutable()) return false;
  if (node.as<isl::schedule_node_band>().n_member() < 1) return false;
  return true;
}

isl::schedule_node TileOuterBand::ReverseTraverseChild(isl::schedule_node node,
                                                       const std::function<isl::schedule_node(isl::schedule_node)> &f) {
  if (node.isa<isl::schedule_node_band>()) {
    if (!tiles_.empty()) {
      tile_sizes_ = tiles_[0].dim_infos;
    }
    node = node.map_descendant_bottom_up(f);
  } else {
    is_sequence_node_ = node.isa<isl::schedule_node_sequence>();
    // multiple outer bands, use same filter strategy as in auto tiling
    for (auto i = 0; i < static_cast<int>(node.n_children()); ++i) {
      if (!IsContainBandNode(node.child(i).child(0)) && scop_info_.user_config_.GetTarget() != TARGET_CCE) {
        continue;
      }
      tile_sizes_ = cur_band_index_ < tiles_.size() ? tiles_[cur_band_index_].dim_infos : tiles_[0].dim_infos;
      node = node.child(i).map_descendant_bottom_up(f);
      node = node.parent();
      cur_band_index_++;
    }
  }
  return node;
}

void TileOuterBand::ShowDimInfo() {
  for (size_t i = 0; i < tiles_.size(); ++i) {
    LOG(INFO) << "band No." << i << ", tiling_flag: " << tiles_[i].tiling_flag;

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

std::string TileOuterBand::GetcDim() {
  auto matcher = DimInfoMatcher();
  matcher.Visit(scop_info_.user_config_.GetBody());
  return matcher.dim();
}

void TileOuterBand::InitCustomDimensionInfo(std::string dim) {
  const int dim_size = 4;
  const int custom_dim_size = 6;
  int dim_info_entry_size = dim_size;
  bool is_custom_mapping = false;
  const std::vector<std::string> thread_block_list = {T0, T1, T2, B0, B1, B2};
  for (auto i : thread_block_list) {
    if (dim.find(i) != std::string::npos) {
      is_custom_mapping = true;
      dim_info_entry_size = custom_dim_size;
      break;
    }
  }

  std::vector<std::string> str = Split(dim, SPACE_PATTERN);
  CHECK(!str.empty() && !(str.size() % dim_info_entry_size)) << "Error: You need to set dim !";
  int sequence = 0;
  for (size_t i = 0; i < str.size(); i += dim_info_entry_size) {
    DimensionInfo dim_info;
    dim_info.c1_tiling_size = DEFAULT_TILE_SIZE;
    dim_info.c0_tiling_size = DEFAULT_TILE_SIZE;
    auto index = StrToDecimalInt64(str[i].c_str());
    CHECK(index >= 0) << "failed to convert string " << str[i] << " to number";
    dim_info.index = index;
    const int max_dim_index = 16;
    CHECK(dim_info.index < max_dim_index) << "set_dim index must be less than " << max_dim_index << "!";

    dim_info.axis = str[i + 1];

    int64_t str_2_number = StrToDecimalInt64(str[i + 2].c_str());
    if (str_2_number > 0) {
      dim_info.c1_tiling_size = str_2_number;
    }

    int64_t str_3_number = StrToDecimalInt64(str[i + 3].c_str());
    if (str_3_number > 0) {
      dim_info.c0_tiling_size = str_3_number;
    }

    dim_info.dim_seq = sequence;
    sequence++;
    scop_info_.analysis_result_.InsertDimensionInfo(dim_info);

    if (!is_custom_mapping) continue;
    const int custom_dim_size = 6;
    CHECK(str.size() >= custom_dim_size) << "The configuration length of custom mapping must not be less than "
                                         << custom_dim_size << "!";
    int filter_number = static_cast<int>(WrappedStrtol(str[i]));
    int axis_number = static_cast<int>(WrappedStrtol(str[i + 1]));
    std::string outer_mapping = str[i + 4];
    if (outer_mapping != "-") {
      scop_info_.user_config_.RecordOuterMappingStrategy(axis_number, outer_mapping, filter_number);
    }

    std::string inner_mapping = str[i + 5];
    if (inner_mapping != "-") {
      scop_info_.user_config_.RecordInnerMappingStrategy(axis_number, inner_mapping, filter_number);
    }
  }

  if (!is_custom_mapping) {
    return;
  }
  CheckCustomMapping(scop_info_.user_config_.GetInnerMappingStrategy());
  CheckCustomMapping(scop_info_.user_config_.GetOuterMappingStrategy());
}

// Init set_dim info
void TileOuterBand::InitDimensionInfo(const isl::schedule &sch_init) {
  // get compute dim
  std::string dim = GetcDim();
  // get build dim
  if (dim.empty()) {
    dim = GetbDim();
  }
  if (!dim.empty()) {
    InitCustomDimensionInfo(dim);
    return;
  }

  // apply auto tiling
  scop_info_.analysis_result_.SetEnableAutoTiling(true);
  auto tiling_res = GenerateTiling(sch_init, scop_info_, GenHalide(scop_info_, sch_init, true));
  scop_info_.analysis_result_.SetTileSizes(tiling_res.first);
  scop_info_.analysis_result_.SetTileConstraints(tiling_res.second);
  if (scop_info_.mmu_info_.IsConv()) {
    scop_info_.mmu_info_.SetConvMNKInfo();
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

isl::multi_val TileOuterBand::ComputeBandTilesSizes(const isl::schedule_node &node, const int *tile_size) {
  isl::space space;
  space = node.as<isl::schedule_node_band>().get_space();
  auto dim = static_cast<int>(node.as<isl::schedule_node_band>().n_member());
  return MultiValFromIntList(space, dim, tile_size);
}

isl::schedule_node TileOuterBand::MarkOuterPermutable(isl::schedule_node node) {
  if (scop_info_.user_config_.GetTarget() == TARGET_CUDA) {
    return MarkOuterPermutableCuda(node);
  } else if (scop_info_.user_config_.GetTarget() == TARGET_CPU) {
    return MarkOuterPermutableCpu(node);
  } else {
    return MarkOuterPermutableNpu(node);
  }
}

TileType TileOuterBand::JudgeTileType(isl::schedule_node &node) {
  unsigned int i = 0;
  for (; i < scop_info_.analysis_result_.stmt_type_.size() - 1; ++i) {
    if (scop_info_.analysis_result_.stmt_type_[i].second == STMT_OP_TYPE::MMU_CONV) {
      break;
    }
  }

  bool is_mmu = scop_info_.mmu_info_.HasCube();
  bool is_before_mmu = false;
  bool is_in_mmu = false;
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

  TileType ret;
  if (is_mmu && is_before_mmu && !is_in_mmu) {
    ret = TileType::C1BUFC1;
  } else if (is_mmu || is_in_load_im2col) {
    ret = TileType::C1;
  } else {
    ret = TileType::BUF;
  }

  return ret;
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
                                               scop_info_.user_config_.GetConsiderCoincidence())) {
    node = InsertEmptyPermutableBand(node);
  }

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

  auto tile_type = JudgeTileType(node);
  return TileBandAndCollectMark(node, &tile_size[0], nullptr, nullptr, tile_type, true);
}

void TileOuterBand::CheckCustomMapping(const MappingStrategyFilterMap &custom_mapping_map) {
  const std::unordered_set<std::string> thread_set = {T0, T1, T2};
  const std::unordered_set<std::string> block_set = {B0, B1, B2};

  size_t thread_prefix;
  size_t block_prefix;
  for (auto filter_custom_mapping : custom_mapping_map) {
    thread_prefix = 0;
    block_prefix = 0;
    for (auto axis_custom_mapping : filter_custom_mapping.second) {
      if (thread_set.find(axis_custom_mapping.second.mapping_idx) != thread_set.end()) {
        ++thread_prefix;
      } else if (block_set.find(axis_custom_mapping.second.mapping_idx) != block_set.end()) {
        ++block_prefix;
      } else {
        LOG(FATAL) << "The custom configuration must be t0, t1, t2, b0, b1 and b2.";
      }
    }

    if (thread_prefix != filter_custom_mapping.second.size() && block_prefix != filter_custom_mapping.second.size()) {
      LOG(FATAL) << "All of the inner configuration or the outer configuration must be threads or blocks.";
    }

    if (thread_prefix == filter_custom_mapping.second.size()) {
      scop_info_.analysis_result_.SetIsOuterBlockMapping(false);
    } else {
      scop_info_.analysis_result_.SetIsOuterBlockMapping(true);
    }
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

unsigned int TileOuterBand::GetMmuIndex() {
  unsigned int mmu_index = 0;
  for (; mmu_index < scop_info_.analysis_result_.stmt_type_.size() - 1; ++mmu_index) {
    if (scop_info_.analysis_result_.stmt_type_[mmu_index].second == STMT_OP_TYPE::MMU_CONV ||
        scop_info_.analysis_result_.stmt_type_[mmu_index].second == STMT_OP_TYPE::MMU_GEMM ||
        scop_info_.analysis_result_.stmt_type_[mmu_index].second == STMT_OP_TYPE::IM2COL_BUF) {
      break;
    }
  }
  return mmu_index;
}

void TileOuterBand::TileTypeC0(isl::schedule_node &node, int *full_tile_min, int *full_tile_max, TileType &tile_type,
                               bool &isolate, isl::multi_val &sizes) {
  isl::set_list domain_list = node.get_domain().get_set_list();
  isl::union_set filter_mmu = isl::union_set();
  isl::union_set filter_after_mmu = isl::union_set();
  std::vector<isl::union_set> filter_before_mmu;
  std::vector<TileType> tile_type_before_mmu;
  isl::union_set_list filters =
    isl::union_set_list(node.ctx(), static_cast<int>(scop_info_.analysis_result_.stmt_type_.size() - 1));

  auto mmu_index = GetMmuIndex();
  for (unsigned int set_index = 0; set_index < domain_list.size(); ++set_index) {
    isl::set set_i = domain_list.get_at(set_index);
    std::string name = set_i.get_tuple_name();
    CHECK(name.find('_') != std::string::npos) << "invalid name " << name;
    unsigned int index = WrappedStrtol(name.substr(name.find('_') + 1));
    set_i = isl::manage(isl_set_eliminate_dims(set_i.copy(), 0, isl_set_n_dim(set_i.get())));
    if (index + 1 < mmu_index) {
      filter_before_mmu.resize(mmu_index - 1);
      auto uset = isl::union_set(set_i);
      filter_before_mmu[index] = uset;
      filters = uset.is_null() ? filters : filters.add(uset);
      tile_type_before_mmu.resize(mmu_index - 1);
      if (!scop_info_.analysis_result_.GetFakeCopyin().is_empty()) {
        tile_type_before_mmu[index] = TileType::BUFC1;
      } else {
        tile_type_before_mmu[index] = TileType::BUFC0;
      }
    }
    if (index + 1 == mmu_index || index == mmu_index) {
      filter_mmu = filter_mmu.is_null() ? isl::union_set(set_i) : filter_mmu.add_set(set_i);
    }
    if (index > mmu_index) {
      filter_after_mmu = filter_after_mmu.is_null() ? isl::union_set(set_i) : filter_after_mmu.add_set(set_i);
    }
  }

  filters = filter_mmu.is_null() ? filters : filters.add(filter_mmu);
  filters = filter_after_mmu.is_null() ? filters : filters.add(filter_after_mmu);

  isl::schedule_node before_tile_node = node;
  bool condition = (!filter_before_mmu.empty() || !filter_after_mmu.is_null()) && !filter_mmu.is_null();
  if (scop_info_.mmu_info_.IsLoadIm2colC1BUF()) {
    node = TileBand(node, sizes);
    node =
      isolate_tile_->IsolateTilesForCce(before_tile_node, node, TileType::BUF, full_tile_min, full_tile_max, isolate);
    node = MarkTileBand(node, TileType::BUF);
  } else if (condition) {
    auto pos = 0;
    node = node.insert_sequence(filters);
    for (size_t index = 0; index < tile_type_before_mmu.size(); ++index) {
      node = TileBand(node.child(pos).child(0), sizes);
      node =
        isolate_tile_->IsolateTilesForCce(before_tile_node, node, tile_type, full_tile_min, full_tile_max, isolate);
      node = MarkTileBand(node, tile_type_before_mmu[index]);
      node = node.parent().parent();
      ++pos;
    }

    node = TileBand(node.child(pos).child(0), sizes);
    node = isolate_tile_->IsolateTilesForCce(before_tile_node, node, tile_type, full_tile_min, full_tile_max, isolate);
    node = MarkTileBand(node, TileType::C0);
    node = node.parent().parent();
    ++pos;

    if (!filter_after_mmu.is_null()) {
      node = TileBand(node.child(pos).child(0), sizes);
      node =
        isolate_tile_->IsolateTilesForCce(before_tile_node, node, tile_type, full_tile_min, full_tile_max, isolate);
      node = MarkTileBand(node, TileType::BUFC0);
      node = node.parent().parent();
      ++pos;
    }
  } else {  // Don't insert a sequence node when there is only one filter child
    node = TileBand(node, sizes);
    node = isolate_tile_->IsolateTilesForCce(before_tile_node, node, tile_type, full_tile_min, full_tile_max, isolate);
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
  if (stride_h == 0) {
    CHECK_NE(stride_h, 0);
  }
  int win_h = (h + pad_top + pad_bottom - d_kh) / stride_h + 1;
  CHECK_NE(stride_h, 0);
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
  if (stride_w == 0) {
    CHECK_NE(stride_w, 0);
  }
  int win_w = (w + pad_left + pad_right - d_kw) / stride_w + 1;
  CHECK_NE(stride_w, 0);
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
        if (win_cut_w == 0) {
          CHECK_NE(win_cut_w, 0);
        }
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
        if (win_cut_h == 0) {
          CHECK_NE(win_cut_h, 0);
        }
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
  node = isolate_tile_->IsolateTilesForCce(before_tile_node, node, tile_type, full_tile_min, full_tile_max, isolate);
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
    node = isolate_tile_->IsolateTilesForCce(before_tile_node, node, tile_type, full_tile_min, full_tile_max, isolate);
    node = MarkTileBand(node, tile_type);
    node = TileBufC1(node.child(0));
  } else if (tile_type == TileType::BUFC1) {
    node = TileBand(node, sizes);
    node = isolate_tile_->IsolateTilesForCce(before_tile_node, node, tile_type, full_tile_min, full_tile_max, isolate);
    node = MarkTileBand(node, tile_type);
    node = node.parent().parent();
  } else {
    node = TileBand(node, sizes);
    node = isolate_tile_->IsolateTilesForCce(before_tile_node, node, tile_type, full_tile_min, full_tile_max, isolate);
    node = MarkTileBand(node, tile_type);
  }
  return node;
}

std::vector<int> TileOuterBand::GetTileSizeOfLevel(const int member_size, const int dim_size, const TileType tile_level,
                                                   const int count_coincident, const std::vector<int> &warp_list) {
  std::vector<int> tile_size(member_size, 1);
  if (tile_level == TileType::VECTORIZATION) {
    // Get the size of the vectorized tiling.
    int vectorized_tile_size = GetVectorizationTileSize(scop_info_);
    tile_size[vectorization_axis_pos_] = vectorized_tile_size;
    return tile_size;
  }

  for (auto i = 0; i < member_size; ++i) {
    if (i >= dim_size) {
      tile_size[i] = MAX_STRIDE;
      continue;
    }
    // tile_size maybe bigger than dim_num
    if (tile_level == TileType::C0) {
      tile_size[i] = static_cast<int>(tile_sizes_[i].c0_tiling_size);
    } else if (tile_level == TileType::C1) {
      tile_size[i] = static_cast<int>(tile_sizes_[i].c1_tiling_size);
    } else if (tile_level == TileType::WARPC1) {
      tile_size[i] = warp_list[i];
    } else if (tile_level == TileType::LASTC1) {
      tile_size[i] = static_cast<int>(tile_sizes_[tile_sizes_.size() - 1 - i].c1_tiling_size);
    } else if (tile_level == TileType::LASTC0) {
      tile_size[i] = static_cast<int>(tile_sizes_[tile_sizes_.size() - 1 - i].c0_tiling_size);
    } else if (tile_level == TileType::C0C1) {
      // The tiling size of n and m is warp_number times of c0_tiling_size, which is equivalent to extracting the for
      // loop generated during mapping.This avoids the if condition and facilitates isl_emitter.
      tile_size[i] = (i < count_coincident) ? static_cast<int>(tile_sizes_[i].c1_tiling_size)
                                            : static_cast<int>(tile_sizes_[i].c0_tiling_size);
    } else {
      tile_size.clear();
      return tile_size;
    }
  }
  return tile_size;
}

/***************************************************************************
 * steps:
 * 1. get tile size.
 * 2. tiling
 ***************************************************************************/
isl::schedule_node TileOuterBand::MarkOuterPermutableCuda(isl::schedule_node node) {
  // check tilable or not, and return the node if not
  if (IsOuterTilable(node) <= 0) return node;

  // make sure the node is a band node, insert empty band if not
  if (!node.isa<isl::schedule_node_band>()) {
    node = InsertEmptyPermutableBand(node);
  }

  // tile matmul operator
  if (scop_info_.user_config_.GetEnableMatmul()) {
    return TileMatmulOperatorForCuda(node);
  } else {
    return TileElementWiseForCuda(node);
  }

  return node;
}

isl::schedule_node TileOuterBand::TileElementWiseForCuda(const isl::schedule_node &orig_node) {
  size_t start_depth = orig_node.get_tree_depth();
  // tile block config
  auto node = TileThreadAndBlockConfig(orig_node, true);

  // get tile size
  auto level_tile_size = GetTileSizeOfLevelForCuda(node, TileType::C1);
  bool enable_vectorization = scop_info_.user_config_.GetEnableVectorization() &&
                              scop_info_.analysis_result_.GetOuterBandNode(cur_band_index_)->enable_vectorization;
  if (enable_vectorization) {
    node = isolate_tile_->IsolateTilesForCudaAndCpu(node, level_tile_size, start_pos_);
  } else {
    node = TileBand(node, level_tile_size).child(0);
  }

  node = node.insert_mark(PROMOTE_GLOBAL_TO_SHARED).child(0);

  // tile thread config
  node = TileThreadAndBlockConfig(node);

  // vectorize for elementwise operator
  if (enable_vectorization) {
    level_tile_size = GetTileSizeOfLevelForCuda(node, TileType::C0);
    node = isolate_tile_->IsolateTilesForCudaAndCpu(node, level_tile_size, start_pos_);
    node = node.insert_mark(PROMOTE_GLOBAL_TO_REGISTER_VECTORIZED);
  }
  node = node.ancestor(node.get_tree_depth() - start_depth);
  return node;
}

isl::multi_val TileOuterBand::GetTileSizeOfLevelForCuda(const isl::schedule_node &node, const TileType tile_level,
                                                        const int count_coincident) {
  const unsigned int n_member = node.as<isl::schedule_node_band>().n_member();
  auto tile_member = static_cast<unsigned int>(tile_sizes_.size());
  unsigned int dim_num = (n_member <= tile_member) ? n_member : tile_member;
  std::vector<int> tile_size;
  auto replace_cfg_map = scop_info_.user_config_.GetReplaceConfig();
  if (tile_level == TileType::WARPC1) {
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
    tile_size = GetTileSizeOfLevel(n_member, dim_num, tile_level, count_coincident, warp_list);
  } else {
    tile_size = GetTileSizeOfLevel(n_member, dim_num, tile_level, count_coincident);
  }

  return ComputeBandTilesSizes(node, &tile_size[0]);
}

isl::multi_val TileOuterBand::GetMappedTileSize(const isl::schedule_node &orig_node, MappingCfg *mapping_cfg,
                                                const std::vector<int> &vectorization_tile_size) {
  CHECK(mapping_cfg) << "mapping config is null.";
  CHECK(orig_node.isa<isl::schedule_node_band>());

  MappingStrategyAxisMap required_mapping_strategy;
  bool is_thread_config =
    mapping_cfg->type == MappingType::THREADS || mapping_cfg->type == MappingType::REPLACE_THREADS;
  if (is_thread_config) {
    required_mapping_strategy = scop_info_.user_config_.GetInnerMappingStrategy(cur_band_index_);
  } else {
    required_mapping_strategy = scop_info_.user_config_.GetOuterMappingStrategy(cur_band_index_);
  }

  if (required_mapping_strategy.empty()) {
    OperatorMappingStrategy mapping_strategy(scop_info_, mapping_cfg, cur_band_index_, is_thread_config, false);
    mapping_strategy.SetRequiredMappingCfg(orig_node);
    required_mapping_strategy = mapping_strategy.required_mapping_strategy_;
  }

  bool is_replace_config =
    mapping_cfg->type == MappingType::REPLACE_THREADS || mapping_cfg->type == MappingType::REPLACE_BLOCKS;
  std::string replace_name = REPLACE;
  replace_name += COMPUTE;
  if (is_replace_config) {
    for (auto &mapping_strategy : required_mapping_strategy) {
      auto mapping_second = mapping_strategy.second;
      if (mapping_second.mapping_idx.find(replace_name) != std::string::npos) {
        continue;
      }
      mapping_second.mapping_idx = replace_name + UNDERSCORE_PATTERN + mapping_second.mapping_idx;
      mapping_strategy.second = mapping_second;
    }
  }

  auto mapping_partial_schedule = GetCurrentPartialSchedule(orig_node.as<isl::schedule_node_band>());
  mapping_partial_schedule = mapping_partial_schedule.intersect_domain(orig_node.domain());
  auto upa_list = mapping_partial_schedule.get_union_pw_aff_list();
  auto mapped_tile_size =
    CheckAndGetMapSize(orig_node, upa_list, required_mapping_strategy, mapping_cfg, vectorization_tile_size);

  return mapped_tile_size;
}

isl::schedule_node TileOuterBand::TileThreadAndBlockConfig(const isl::schedule_node &orig_node,
                                                           const bool is_block_mapping) {
  if (is_sequence_node_ && is_block_mapping) {
    return orig_node;
  }

  auto current_outer_bn = scop_info_.analysis_result_.GetOuterBandNode(cur_band_index_);

  std::vector<int> additional_tile_size;
  if (is_block_mapping || current_outer_bn->enable_vectorization) {
    const unsigned int n_member = orig_node.as<isl::schedule_node_band>().n_member();
    auto tile_member = static_cast<unsigned int>(tile_sizes_.size());
    unsigned int dim_num = (n_member <= tile_member) ? n_member : tile_member;
    auto tile_level = is_block_mapping ? TileType::C1 : TileType::C0;
    additional_tile_size = GetTileSizeOfLevel(n_member, dim_num, tile_level);
  }

  auto mapping_cfg =
    is_block_mapping ? scop_info_.user_config_.GetBlockConfig() : scop_info_.user_config_.GetThreadConfig();
  auto mapping_marker = is_block_mapping ? BLOCK_MARKER : THREAD_MARKER;

  auto mapped_tile_size = GetMappedTileSize(orig_node, mapping_cfg, additional_tile_size);
  if (mapped_tile_size.size() == 0 || mapped_tile_size.at(0).is_zero()) {
    return orig_node.insert_mark(mapping_marker).child(0);
  }

  auto node = orig_node;
  if (current_outer_bn->enable_vectorization) {
    node = isolate_tile_->IsolateTilesForCudaAndCpu(node, mapped_tile_size, start_pos_);
  } else {
    node = TileBand(orig_node, mapped_tile_size).child(0);
  }

  if (is_block_mapping) {
    current_outer_bn->is_block_tile = true;
  } else {
    current_outer_bn->is_thread_tile = true;
  }

  return node.insert_mark(mapping_marker).child(0);
}

isl::schedule_node TileOuterBand::TileMatmulOperatorForCuda(const isl::schedule_node &node) {
  auto tile_node = node;
  size_t start_depth = tile_node.get_tree_depth();

  tile_node = TileThreadAndBlockConfig(tile_node, true);
  tile_node = TileBand(tile_node, GetTileSizeOfLevelForCuda(tile_node, TileType::C1));

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
    tile_node = tile_node.child(0);
    tile_node = TileBand(tile_node, GetTileSizeOfLevelForCuda(tile_node, TileType::C0C1, count_coincident));
  }

  // The third tiling of tensor_core is to map to warp.
  tile_node = tile_node.child(0);
  tile_node = TileBand(tile_node, GetTileSizeOfLevelForCuda(tile_node, TileType::WARPC1, count_coincident));
  if (!scop_info_.user_config_.GetEnableTensorCoreUsePoly()) {
    tile_node = tile_node.child(0);
  }
  tile_node = tile_node.insert_mark(isl::id(tile_node.ctx(), PROMOTE_SHARED_TO_REGISTER_AB));
  // Locate the band to be mapped.
  tile_node = tile_node.child(0).insert_mark(WARP_MARKER);
  tile_node = tile_node.child(0).child(0);

  // The last tiling of tensor_core is to calculate the size of fragment.
  tile_node = TileBand(tile_node, GetTileSizeOfLevelForCuda(tile_node, TileType::C0));

  if (scop_info_.user_config_.GetEnableConvTensorCore()) {
    const int kh_kw_depth = 2;
    int child_depth = kh_kw_depth;
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
  } else {
    tile_node = tile_node.insert_mark(isl::id(tile_node.ctx(), PROMOTE_GLOBAL_TO_REGISTER_C)).child(0);
  }

  tile_node = tile_node.child(0).insert_mark(isl::id(tile_node.ctx(), PROMOTE_GLOBAL_TO_SHARED_AB));
  return tile_node;
}

bool TileOuterBand::IsMatrixCPromoteToShared() {
  std::unordered_set<std::string> shared_tensors = scop_info_.user_config_.GetSharedTensors();
  if (shared_tensors.empty()) {
    return false;
  }

  for (const auto &tensor : shared_tensors) {
    auto matmul_map = scop_info_.analysis_result_.GetMatrixMatmulMap();
    if (matmul_map.count(tensor) && (matmul_map[tensor] == MATRIX_C || matmul_map[tensor] == MATRIX_ELSE)) {
      return true;
    }
  }
  return false;
}

/***************************************************************************
 * steps:
 * 1. get tile size.
 * 2. tiling
 ***************************************************************************/
isl::schedule_node TileOuterBand::MarkOuterPermutableCpu(isl::schedule_node node) {
  // check tilable or not, and return the node if not
  if (IsOuterTilable(node) <= 0) return node;

  // make sure the node is a band node, insert empty band if not
  if (!node.isa<isl::schedule_node_band>()) {
    node = InsertEmptyPermutableBand(node);
  }

  auto current_outer_bn = scop_info_.analysis_result_.GetOuterBandNode(cur_band_index_);
  auto template_type = current_outer_bn->template_type;
  if (scop_info_.analysis_result_.GetCsr()) {
    return TileCsrForCpu(node);
  }

  if (template_type == Template::MATMUL && scop_info_.user_config_.GetEnableMatmul()) {
    return TileGemmOperatorForCpu(node);
  }

  if (template_type == Template::REDUCTION) {
    if (current_outer_bn->reduce_direction == ReduceDirection::X) {
      return TileReduceXForCpu(node);
    }

    if (current_outer_bn->reduce_direction == ReduceDirection::ALL) {
      return TileAllReduceForCpu(node);
    }
  }

  if (template_type == Template::CONV) {
    return TileConvForCpu(node);
  }

  return TileElementWiseForCpu(node);
}

isl::schedule_node TileOuterBand::TileConvForCpu(const isl::schedule_node &orig_node) {
  if (!orig_node.isa<isl::schedule_node_band>()) {
    return orig_node;
  }

  size_t start_depth = orig_node.get_tree_depth();
  auto node = orig_node;
  // Get reduce axis information that is not in the outermost band node.
  node = node.child(0);
  std::vector<isl::multi_union_pw_aff> all_mupa;
  while (node.isa<isl::schedule_node_band>()) {
    isl::multi_union_pw_aff current_mupa = node.as<isl::schedule_node_band>().get_partial_schedule();
    node = node.del();
    all_mupa.push_back(current_mupa);
  }
  node = node.parent();

  // Determines the number of reduction axes in the outermost band node.
  auto axes_names = scop_info_.analysis_result_.GetCpuConvolutionAxes();
  int reduce_axis_num = 0;
  if (axes_names.find(CONV_IC_IN) != std::string::npos || axes_names.find(CONV_KH) != std::string::npos ||
      axes_names.find(CONV_KW) != std::string::npos) {
    reduce_axis_num = 1;
  }

  // The current parallel strategy is only used on batch, oc_out and oh axes.
  auto parallel_num = static_cast<int>(node.as<isl::schedule_node_band>().n_member()) - reduce_axis_num;
  if (axes_names.find(CONV_OC_IN) != std::string::npos) {
    --parallel_num;
  }
  if (axes_names.find(CONV_OW) != std::string::npos) {
    --parallel_num;
  }
  node = TileAccordingToTileType(node, TileType::C1).parent();
  node = InsertMultiMarker(node, FOR_PARALLEL, false, parallel_num).child(0);
  node = node.insert_mark(PROMOTE_GLOBAL_TO_REGISTER_C).child(0);

  node = SplitReduceStatements(node).parent();
  size_t seq_start_depth = node.get_tree_depth();
  int seq_num = node.n_children();

  for (int i = 0; i < seq_num; ++i) {
    node = node.child(i).child(0);
    if (!node.isa<isl::schedule_node_band>()) {
      continue;
    }
    node = TileAccordingToTileType(node, TileType::C0);
    if (i == 1) {
      for (auto mupa : all_mupa) {
        node = node.insert_partial_schedule(mupa);
        node = node.as<isl::schedule_node_band>().set_permutable(1);
        node = node.child(0);
      }
      if (axes_names.find(CONV_OC_IN) != std::string::npos) {
        node = node.insert_mark(PROMOTE_GLOBAL_TO_REGISTER_AB).child(0);
      }
    }

    if (node.isa<isl::schedule_node_band>()) {
      if (axes_names.find(CONV_OC_IN) != std::string::npos) {
        auto band_node = node.as<isl::schedule_node_band>();
        vectorization_axis_pos_ = band_node.n_member() - 1 - reduce_axis_num;
        node = TileAccordingToTileType(node, TileType::VECTORIZATION);
        node = InsertMultiMarker(node, FOR_VECTORIZED, true);
        node = node.parent();
      }

      if (axes_names.find(CONV_OW) != std::string::npos) {
        node = InsertMultiMarker(node, FOR_UNROLLED);
      }
    }
    node = node.ancestor(node.get_tree_depth() - seq_start_depth);
  }

  return node.ancestor(node.get_tree_depth() - start_depth);
}

bool TileOuterBand::IsContainReduceStatement(const isl::schedule_node &orig_node) {
  if (!orig_node.isa<isl::schedule_node_filter>()) {
    return false;
  }

  auto filter = orig_node.as<isl::schedule_node_filter>().get_filter();
  return !filter.intersect(reduce_statements_).is_empty();
}

isl::schedule_node TileOuterBand::TileCsrForCpu(const isl::schedule_node &orig_node) {
  if (!orig_node.isa<isl::schedule_node_band>()) {
    return orig_node;
  }
  auto node = orig_node;
  size_t start_depth = node.get_tree_depth();

  // Tile outermost axis for parallel
  auto band_node = node.as<isl::schedule_node_band>();
  node = band_node.n_member() <= 1 ? band_node : band_node.split(band_node.n_member() - 1);
  node = TileAccordingToTileType(node, TileType::C1);
  node = InsertMarkerForLoop(node, FOR_PARALLEL);
  auto template_type = scop_info_.analysis_result_.GetOuterBandNode(cur_band_index_)->template_type;
  if (template_type == Template::REDUCTION) {
    node = SplitReduceStatements(node.child(0));
    while (node.has_children()) {
      node = node.child(0);
    }
    node = node.parent();
    vectorization_axis_pos_ = static_cast<int>(node.as<isl::schedule_node_band>().n_member()) - 1;
    node = TileAccordingToTileType(node, TileType::VECTORIZATION).child(0);
    node = node.insert_mark(FOR_VECTORIZED);
    node = node.parent().insert_mark(REDUCE_AREA_FLAG);
  }
  return node.ancestor(node.get_tree_depth() - start_depth);
}

isl::schedule_node TileOuterBand::TileGemmOperatorForCpu(const isl::schedule_node &orig_node) {
  size_t start_depth = orig_node.get_tree_depth();

  isl::schedule_node node = TileGemmBandNodeForCpu(orig_node);
  auto seq_node = SplitReduceStatements(node).parent();
  if (!seq_node.isa<isl::schedule_node_sequence>()) {
    return orig_node;
  }

  for (size_t i = 0; i < seq_node.n_children(); ++i) {
    node = seq_node.child(i);
    if (!node.isa<isl::schedule_node_filter>()) {
      continue;
    }
    bool is_gemm = IsContainReduceStatement(node);
    node = node.child(0);

    if (is_gemm) {
      node = node.insert_mark(TENSOR_C);
    } else {
      node = TileElementWiseForCpu(node);
    }
    seq_node = node.parent().parent();
  }

  return node.ancestor(node.get_tree_depth() - start_depth);
}

isl::schedule_node TileOuterBand::TileGemmBandNodeForCpu(const isl::schedule_node &orig_node) {
  if (!orig_node.isa<isl::schedule_node_band>()) {
    return orig_node;
  }

  auto node = orig_node;
  node = TileAccordingToTileType(node, TileType::C1);

  node = InsertMultiMarker(node.parent(), FOR_PARALLEL).child(0);
  node = node.insert_mark(PROMOTE_GLOBAL_TO_REGISTER).child(0);

  return node;
}

isl::schedule_node TileOuterBand::TileAllReduceForCpu(const isl::schedule_node &orig_node) {
  auto node = orig_node;
  size_t start_depth = node.get_tree_depth();

  node = InsertEmptyPermutableBand(node).child(0);
  auto seq_node = SplitReduceStatements(node).parent();
  if (!seq_node.isa<isl::schedule_node_sequence>()) {
    return orig_node;
  }

  for (size_t i = 0; i < seq_node.n_children(); ++i) {
    node = seq_node.child(i);
    if (!node.isa<isl::schedule_node_filter>()) {
      continue;
    }
    bool is_reduce = IsContainReduceStatement(node);
    node = node.child(0);
    node = TileElementWiseForCpu(node, is_reduce);
    seq_node = node.parent().parent();
  }

  return node.ancestor(node.get_tree_depth() - start_depth);
}

isl::schedule_node TileOuterBand::InsertMarkerForReduceY(const isl::schedule_node &orig_node, size_t start_depth) {
  auto node = orig_node;
  if (scop_info_.analysis_result_.GetOuterBandNode(cur_band_index_)->reduce_direction == ReduceDirection::Y) {
    if (node.as<isl::schedule_node_mark>()) {
      node = node.child(0);
    }
    auto band_node = node.as<isl::schedule_node_band>();
    node = band_node.split(band_node.n_member() - 1);
    node = node.child(0);
    node = InsertMarkerForLoop(node, FOR_PARALLEL);
    bool is_parallel = !GetMarkerName(node, FOR_PARALLEL).empty();
    if (is_parallel) {
      node = node.ancestor(node.get_tree_depth() - start_depth);
      node = node.insert_mark(REDUCE_AREA_FLAG);
      node = node.insert_mark(REDUCE_Y_FLAG);
    }
  }
  return node;
}

isl::schedule_node TileOuterBand::TileElementWiseForCpu(const isl::schedule_node &orig_node, const bool is_all_reduce) {
  if (!orig_node.isa<isl::schedule_node_band>()) {
    return orig_node;
  }
  vectorization_axis_pos_ = scop_info_.analysis_result_.GetOuterBandNode(cur_band_index_)->last_axis;
  if (vectorization_axis_pos_ == -1) {
    return orig_node;
  }

  auto node = orig_node;
  size_t start_depth = node.get_tree_depth();

  // first tiling: parallel
  node = TileAccordingToTileType(node, TileType::C1);

  // second tiling: unroll
  node = TileAccordingToTileType(node, TileType::C0);

  // sink last axis
  int n_member = static_cast<int>(node.as<isl::schedule_node_band>().n_member());
  node = AdjustAxisPosition(node, vectorization_axis_pos_, n_member - 1);
  auto band_node = node.as<isl::schedule_node_band>();
  node = band_node.split(band_node.n_member() - 1).child(0);

  // last tiling: vectorized
  vectorization_axis_pos_ = 0;
  node = TileAccordingToTileType(node, TileType::VECTORIZATION);
  node = InsertAllMarker(node, is_all_reduce);

  node = InsertMarkerForReduceY(node, start_depth);
  node = node.ancestor(node.get_tree_depth() - start_depth);
  return node;
}

isl::schedule_node TileOuterBand::TileReduceXForCpu(const isl::schedule_node &orig_node) {
  if (!orig_node.isa<isl::schedule_node_band>()) return orig_node;

  // split reduce axis
  auto band_node = orig_node.as<isl::schedule_node_band>();
  isl::schedule_node node = band_node.split(band_node.n_member() - 1);

  // tile non_reduce axis
  node = TileAccordingToTileType(node, TileType::C1);
  node = TileAccordingToTileType(node, TileType::C0);
  node = node.child(0);

  // tile reduce axis
  start_pos_ = static_cast<int>(band_node.n_member() - 1);
  node = TileAccordingToTileType(node, TileType::LASTC1);
  node = TileAccordingToTileType(node, TileType::LASTC0);

  // vetorized tile
  vectorization_axis_pos_ = 0;
  node = TileAccordingToTileType(node, TileType::VECTORIZATION);

  bool is_insert_mark = false;
  node = InsertMarkerForLoop(node, FOR_VECTORIZED).parent();
  is_insert_mark = !GetMarkerName(node.child(0), FOR_VECTORIZED).empty();
  node = InsertMarkerForLoop(node, FOR_UNROLLED).parent();
  node = node.parent();

  // split reduce statement
  node = SplitReduceStatements(node).child(0);
  node = is_insert_mark ? node.insert_mark(REDUCE_AREA_FLAG) : node;
  node = node.ancestor(node.get_tree_depth() - orig_node.get_tree_depth());
  node = InsertMarkerForLoop(node, FOR_PARALLEL);
  return node;
}

std::vector<int> TileOuterBand::GetTileSizeForCpu(const isl::schedule_node &orig_node, const TileType tile_level) {
  std::vector<int> tile_size;
  if (!orig_node.isa<isl::schedule_node_band>()) {
    return tile_size;
  }

  const int n_member = orig_node.as<isl::schedule_node_band>().n_member();
  const int tile_number = static_cast<int>(tile_sizes_.size());
  CHECK(start_pos_ < tile_number) << "The starting position cannot be greater than or equal to the tiling size.";
  int dim_num = std::min(n_member, tile_number);

  // Get the size of the parallel and unroll tiling.
  tile_size = GetTileSizeOfLevel(n_member, dim_num, tile_level);
  return tile_size;
}

isl::schedule_node TileOuterBand::TileAccordingToTileType(const isl::schedule_node &orig_node,
                                                          const TileType tile_level,
                                                          const std::vector<int> &tile_size) {
  std::vector<int> cur_tile_size;
  if (tile_level == TileType::Invalid) {
    cur_tile_size = tile_size;
  } else {
    cur_tile_size = GetTileSizeForCpu(orig_node, tile_level);
  }

  isl::multi_val mutial_val_tile_size = ComputeBandTilesSizes(orig_node, &cur_tile_size[0]);
  int all_tile_size = static_cast<int>(tile_sizes_.size());
  return isolate_tile_->IsolateTilesForCudaAndCpu(orig_node, mutial_val_tile_size, start_pos_, all_tile_size);
}

isl::schedule_node TileOuterBand::InsertAllMarker(const isl::schedule_node &orig_node, const bool is_all_reduce) {
  if (!orig_node.isa<isl::schedule_node_band>()) {
    return orig_node;
  }
  auto node = orig_node;
  bool is_insert_mark = false;
  // Insert vectoried marker on the last band.
  node = InsertMarkerForLoop(node, FOR_VECTORIZED).parent();
  is_insert_mark = !GetMarkerName(node.child(0), FOR_VECTORIZED).empty();

  // Insert the unroll marker on the axis corresponding to the vectorization.
  node = InsertMarkerForLoop(node, FOR_UNROLLED).parent();
  node = node.parent();

  if (is_all_reduce) {
    // For the reduce statement of the all_reduce operator, insert the reduce_area mark on the parallel and unroll
    // loop axis.
    node = is_insert_mark ? node.insert_mark(REDUCE_AREA_FLAG) : node;
  }

  node = node.parent();
  // Insert the parallel marker on the axis corresponding.
  if (is_all_reduce) {
    node = InsertMarkerForLoop(node, FOR_PARALLEL);
    is_insert_mark = !GetMarkerName(node, FOR_PARALLEL).empty();
    node = is_insert_mark ? node.insert_mark(REDUCE_AREA_FLAG) : node;
  } else {
    node = InsertMarkerForLoop(node, FOR_PARALLEL);
  }

  return node;
}

isl::schedule_node TileOuterBand::InsertMultiMarker(const isl::schedule_node &orig_node, const std::string &marker_name,
                                                    const bool return_orig_pos, const int insert_marker_num) {
  // Insert corresponding markers for all axes with shape greater than 1 in the current band node.
  if (!orig_node.isa<isl::schedule_node_band>()) {
    return orig_node;
  }

  auto band_node = orig_node.as<isl::schedule_node_band>();
  int band_member = static_cast<int>(band_node.n_member());
  auto partial_schedule = band_node.get_partial_schedule().intersect_domain(orig_node.get_domain());
  auto upa_list = partial_schedule.get_union_pw_aff_list();

  CHECK(insert_marker_num <= band_member)
    << "The number of parallel axes set must be less than the number of axes of the band node";

  auto node = orig_node;
  int cur_marker_num = 0;
  int child_depth = 0;
  // The size of the axis is judged from the back to the front, and the corresponding marker is inserted on the
  // corresponding axis.
  for (int i = band_member - 1; i >= 0; --i) {
    auto extent = upa_list.get_at(i).floor().max_val().get_num_si();
    // If the size of the current axis is 0 or the current axis does not need to insert a marker, then this axis is
    // skipped.
    if (extent < 1 || (i >= insert_marker_num && insert_marker_num != -1)) {
      ++cur_marker_num;
      continue;
    }

    if (!node.isa<isl::schedule_node_band>()) {
      break;
    }

    auto cur_band_node = node.as<isl::schedule_node_band>();
    band_member = static_cast<int>(cur_band_node.n_member());
    if ((band_member == 1) || (band_member - 1 == cur_marker_num)) {
      node = cur_band_node;
    } else {
      node = cur_band_node.split(band_member - 1 - cur_marker_num).child(0);
      ++child_depth;
    }
    node = node.insert_mark(marker_name);
    // If it is the 0th axis, it is fixed to the position of the marker node, otherwise it returns to the previous band
    // node.
    node = i != 0 ? node.parent() : node;
    cur_marker_num = 0;
    ++child_depth;
  }

  if (return_orig_pos) {
    return node;
  }

  // Return to the child nodes of the last mark node.
  while (child_depth != 0) {
    node = node.child(0);
    --child_depth;
  }
  return node;
}

isl::schedule_node TileOuterBand::SplitReduceStatements(const isl::schedule_node &orig_node) {
  isl::schedule_node tile_node = orig_node;
  auto all_reduce_map = scop_info_.analysis_result_.GetReduceTensorInfoMap();
  auto current_outer_bn = scop_info_.analysis_result_.GetOuterBandNode(cur_band_index_);
  bool need_split_reduce = current_outer_bn->template_type == Template::MATMUL;
  ReduceManager reduce_manager(pass_info_, scop_info_, cur_band_index_, need_split_reduce);
  reduce_statements_ = reduce_manager.GetCurrentNodeReduceStatements(tile_node, all_reduce_map, false);
  isl::union_map new_dependences = reduce_manager.GetCurrentDependence();

  if (!reduce_manager.SplitReduceStatements(tile_node, reduce_statements_, new_dependences)) {
    return orig_node;
  }
  return tile_node;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
