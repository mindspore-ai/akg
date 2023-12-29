/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "operator_mapping_strategy.h"
#include "poly/schedule_pass/reschedule.h"
#include "poly/schedule_tree_util.h"
#include "poly/sync_manager.h"
#include "poly/scop.h"

#include <numeric>

namespace akg {
namespace ir {
namespace poly {

isl::schedule_node OperatorMappingStrategy::GetTiledOuterBand(const isl::schedule_node &orig_node) {
  auto current_outer_bn = scop_info_.analysis_result_.GetOuterBandNode(band_index_);
  bool is_tiled = current_outer_bn->is_thread_tile;
  std::string marker_name = THREAD_MARKER;
  if (!is_thread_mapping_) {
    is_tiled = current_outer_bn->is_block_tile;
    marker_name = BLOCK_MARKER;
  }

  auto node = orig_node;
  if (orig_node.has_parent() && !GetMarkerName(node.parent(), marker_name).empty()) {
    node = node.parent();
  }
  if (orig_node.has_parent() && is_tiled) {
    node = node.parent();
  }
  return node;
}

isl::schedule_node OperatorMappingStrategy::InsertMapFilter(const isl::schedule_node &node) {
  if (mapping_sch_info_map_.size() == 0) {
    return node;
  }
  // extract unique domain
  auto map_domain = mapping_sch_info_map_.cbegin()->second.schedule_upa.domain();
  if (!is_promotion_mapping_) {
    for (const auto &kvp : mapping_sch_info_map_) {
      CHECK(map_domain.is_equal(kvp.second.schedule_upa.domain()));
    }
  }

  auto map_filter = map_domain.universe();
  for (const auto &kvp : mapping_sch_info_map_) {
    auto id = kvp.first;
    auto upa = kvp.second.schedule_upa;
    auto offset = kvp.second.offset;
    upa = upa.sub(isl::union_pw_aff::param_on_domain(map_domain.universe(), id));
    upa = upa.add(isl::union_pw_aff(upa.domain(), isl::val(upa.ctx(), (long)offset)));
    map_filter = map_filter.intersect(upa.zero_union_set());
  }

  // insert mapping filter
  isl::schedule_node map_filter_node = node;
  map_filter_node = is_insert_filter_ ? map_filter_node.insert_filter(map_filter) : map_filter_node;
  return map_filter_node;
}

isl::schedule_node OperatorMappingStrategy::AnalysisNodeAndInsertMapFilter(const isl::schedule_node &node,
                                                                           const isl::union_pw_aff_list &upa_list) {
  isl::union_set domain = node.get_schedule().get_domain();
  if (node.has_parent() && node.parent().has_parent() && node.parent().parent().isa<isl::schedule_node_filter>()) {
    domain = node.parent().parent().as<isl::schedule_node_filter>().get_filter();
  }

  std::unordered_set<std::string> current_mapping_cfg;
  for (size_t i = 0; i < upa_list.size(); ++i) {
    if (required_mapping_strategy_.count(static_cast<int>(i)) == 0) {
      continue;
    }
    auto mapping_i = required_mapping_strategy_[static_cast<int>(i)].mapping_idx;
    auto offset_i = required_mapping_strategy_[static_cast<int>(i)].offset;
    std::pair<std::string, int> cfg = mapping_cfg_->GetAt(mapping_i);
    current_mapping_cfg.emplace(cfg.first);

    auto upa = upa_list.get_at(i);
    CHECK_GT(cfg.second, 0);
    upa = upa.mod(isl::val(node.ctx(), cfg.second));
    auto id = isl::id(node.ctx(), cfg.first);
    MappingScheduleInfo mapping_schedule_info;
    mapping_schedule_info.schedule_upa = upa;
    mapping_schedule_info.offset = offset_i;
    mapping_sch_info_map_[id] = mapping_schedule_info;
    domain = upa.domain();
  }

  // Set other configurations to 0.
  if (is_set_config_zero_) {
    for (size_t i = 0; i < mapping_cfg_->bound; ++i) {
      CHECK(!domain.is_null());
      auto universe = domain.universe();
      // Remove the configuration that has been mapped.
      if (current_mapping_cfg.find(mapping_cfg_->GetAt(i).first) != current_mapping_cfg.end()) {
        continue;
      }
      std::pair<std::string, int> cfg = mapping_cfg_->GetAt(i);
      auto id = isl::id(node.ctx(), cfg.first);
      MappingScheduleInfo mapping_schedule_info;
      mapping_schedule_info.schedule_upa = isl::union_pw_aff(universe, isl::val::zero(domain.ctx()));
      mapping_sch_info_map_[id] = mapping_schedule_info;
    }
  }

  return InsertMapFilter(node);
}

isl::schedule_node OperatorMappingStrategy::MapDimToThreadsBlocks(const isl::schedule_node &orig_node,
                                                                  const bool is_insert_marker) {
  if (!orig_node.isa<isl::schedule_node_band>()) {
    return orig_node;
  }

  auto band_node = orig_node.as<isl::schedule_node_band>();
  auto mapping_partial_schedule = GetCurrentPartialSchedule(band_node, is_promotion_mapping_);
  auto upa_list = mapping_partial_schedule.get_union_pw_aff_list();

  bool is_tiled = false;
  auto node = orig_node;
  if (is_promotion_mapping_ || scop_info_.user_config_.GetMindTrickWasUsed()) {
    auto mapped_tile_size = CheckAndGetMapSize(node, upa_list, required_mapping_strategy_, mapping_cfg_);
    is_tiled = mapped_tile_size.size() > 0 && !mapped_tile_size.at(0).is_zero();
    node = is_tiled ? TileBand(node, mapped_tile_size).child(0) : node;
    // insert node with specific marker
    if (is_insert_marker) {
      std::string marker_name = is_thread_mapping_ ? THREAD_MARKER : BLOCK_MARKER;
      marker_name = is_promotion_mapping_ ? marker_name + SHARE_SUFFIX : marker_name;
      node = node.insert_mark(isl::id(node.ctx(), marker_name)).child(0);
    }

    node = AnalysisNodeAndInsertMapFilter(node, upa_list);
    node = node.parent();
    if (is_tiled) {
      node = node.parent();
    }
  } else {
    node = AnalysisNodeAndInsertMapFilter(node, upa_list);
  }

  return node;
}

// Map a thread/block configuration to multiple axis.
std::string OperatorMappingStrategy::SetOneConfigForMulAxis(const isl::schedule_node &node, const int orig_total_cfg,
                                                            const std::unordered_set<int> &excluded_axis_pos) {
  std::string new_cfg = "";
  if (!node.isa<isl::schedule_node_band>()) {
    return new_cfg;
  }

  auto band_node = node.as<isl::schedule_node_band>();
  auto mapping_partial_schedule = GetCurrentPartialSchedule(band_node, is_promotion_mapping_);
  if (!is_promotion_mapping_) {
    mapping_partial_schedule = mapping_partial_schedule.intersect_domain(node.domain());
  }
  auto upa_list = mapping_partial_schedule.get_union_pw_aff_list();
  upa_list = is_need_reverse_ ? upa_list.reverse() : upa_list;

  int config_size = 0;
  int total_cfg = orig_total_cfg;
  int mapping_dim = static_cast<int>(upa_list.size());
  for (int i = 0; i < mapping_dim; ++i) {
    if (!excluded_axis_pos.empty() && excluded_axis_pos.count(i) == 0) {
      continue;
    }
    auto extend = upa_list.get_at(i).floor().max_val().get_num_si() + 1;
    if (extend >= total_cfg || (i == mapping_dim - 1 && extend < total_cfg)) {
      new_cfg += (std::to_string(total_cfg) + " ");
      ++config_size;
      break;
    }

    total_cfg /= extend;
    new_cfg += (std::to_string(extend) + " ");
    ++config_size;
  }

  while (config_size < static_cast<int>(excluded_axis_pos.size())) {
    new_cfg += (std::to_string(1) + " ");
    ++config_size;
  }
  return new_cfg;
}

void OperatorMappingStrategy::SetRequiredMappingCfg(const isl::schedule_node &node, int start_pos, int end_pos) {
  required_mapping_strategy_.clear();
  if (!node.isa<isl::schedule_node_band>()) {
    return;
  }

  auto band_node = node.as<isl::schedule_node_band>();
  start_pos = std::max(0, start_pos);
  end_pos = std::min(static_cast<int>(band_node.n_member()) - 1, end_pos);

  // There is no need to consider the last axis of the calculation phase in the promotion phase.
  int last_axis = is_promotion_mapping_ ? -1 : scop_info_.analysis_result_.GetOuterBandNode(band_index_)->last_axis;
  int current_mapping_pos = 0;
  if (last_axis >= start_pos && last_axis <= end_pos) {
    required_mapping_strategy_[last_axis].mapping_idx = mapping_cfg_->GetAt(current_mapping_pos).first;
    ++current_mapping_pos;
  }

  for (int i = start_pos; i <= end_pos; ++i) {
    int current_axis_pos = i;
    if (is_need_reverse_) {
      current_axis_pos = end_pos - current_axis_pos;
    }

    if (!is_promotion_mapping_ && band_node.member_get_coincident(current_axis_pos) == 0) {
      continue;
    }

    if (current_mapping_pos >= static_cast<int>(mapping_cfg_->bound) || current_axis_pos == last_axis) {
      continue;
    }

    required_mapping_strategy_[current_axis_pos].mapping_idx = mapping_cfg_->GetAt(current_mapping_pos).first;
    ++current_mapping_pos;
  }
}

/*
 * Initialize repeated_mapping_cfg_axis_ and non_repeated_mapping_cfg_axis_.
 * E.g: axis_0 --> tx  axis_1 --> tx  axis_2 --> ty  axis_3 --> tz
 * repeated_mapping_cfg_axis_: tx --> {axis_0, axis_1}
 * non_repeated_mapping_cfg_axis_: ty --> axis_2  tz --> axis_3
 */
void OperatorMappingStrategy::InitRepeatedMappingConfig() {
  if (required_mapping_strategy_.empty()) {
    return;
  }

  std::unordered_map<std::string, std::unordered_set<int>> all_config_repeated_pos;
  for (auto cfg : required_mapping_strategy_) {
    std::unordered_set<int> one_config_repeated_axis;
    auto mapping_idx = cfg.second.mapping_idx;
    if (all_config_repeated_pos.find(mapping_idx) != all_config_repeated_pos.end()) {
      one_config_repeated_axis = all_config_repeated_pos[mapping_idx];
    }
    one_config_repeated_axis.emplace(cfg.first);
    all_config_repeated_pos[mapping_idx] = one_config_repeated_axis;
  }

  for (auto repeated_pos : all_config_repeated_pos) {
    if (repeated_pos.second.size() > 1) {
      repeated_mapping_cfg_axis_[repeated_pos.first] = repeated_pos.second;
    } else {
      non_repeated_mapping_cfg_axis_[repeated_pos.first] = repeated_pos.second;
    }
  }
}

// For the case where multiple axes are mapped to the same threadIdx/blockIdx, the corresponding thread/block is
// reallocated according to the size of the axis.
// E.g: required_mapping_strategy_: axis_0 --> tx  axis_1 --> tx
// required_mapping_strategy_: axis_0 --> replace_tx_ty  axis_1 --> replace_tx_tx
void OperatorMappingStrategy::SetRepeatedMappingStrategy(const std::string &mapping_str) {
  if (required_mapping_strategy_.empty()) {
    return;
  }

  auto mapping_size = required_mapping_strategy_.size();
  int new_number = is_need_reverse_ ? mapping_size - 1 : 0;
  auto tmp_required_mapping_strategy_ = required_mapping_strategy_;
  for (auto mapping_i : tmp_required_mapping_strategy_) {
    MappingStrategy mapping_strategy;
    mapping_strategy.offset = mapping_i.second.offset;
    mapping_strategy.mapping_idx = mapping_str + std::to_string(new_number);
    required_mapping_strategy_[mapping_i.first] = mapping_strategy;
    new_number = is_need_reverse_ ? new_number - 1 : new_number + 1;
  }
}

// Set the corresponding mapping configuration according to the strategy obtained by SetRepeatedMappingStrategy and the
// size of the axis of the current band.
MappingCfg *OperatorMappingStrategy::GetRepeatedReplaceMappingConfig(const isl::schedule_node &node,
                                                                     const std::string &replace_mapping_name) {
  if (required_mapping_strategy_.empty()) {
    return nullptr;
  }

  std::unordered_set<int> excluded_axis_pos;
  for (auto cfg : required_mapping_strategy_) {
    excluded_axis_pos.emplace(cfg.first);
  }

  int replace_mapping_size = mapping_cfg_->GetAt(replace_mapping_name).second;
  auto new_cfg_str = SetOneConfigForMulAxis(node, replace_mapping_size, excluded_axis_pos);
  auto mapping_type = MappingType::REPLACE_THREADS;
  if (mapping_cfg_->type == MappingType::BLOCKS) {
    mapping_type = MappingType::REPLACE_BLOCKS;
  }

  auto mapping_name = REPEATED_MAPPING + replace_mapping_name;
  scop_info_.user_config_.RecordReplaceConfig(mapping_name, new_cfg_str, mapping_type);
  return scop_info_.user_config_.GetReplaceConfig()[mapping_name];
}

// Determine the current mapping strategy based on whether multiple axes are mapped to the same threadIdx/blockIdx.
// E.g: repeated_mapping_cfg_axis_: tx --> {axis_0, axis_1}
// required_mapping_strategy_: axis_0 --> replace_tx_tx  axis_1 --> replace_tx_ty
void OperatorMappingStrategy::ReadjustRequireddMappingStrategy(const bool is_repeated_mapping,
                                                               const std::string &repeated_mapping_idx,
                                                               const std::string &mapping_str) {
  if (required_mapping_strategy_.empty()) {
    return;
  }

  auto tmp_required_mapping_strategy = required_mapping_strategy_;
  required_mapping_strategy_.clear();
  if (is_repeated_mapping) {
    for (auto axis : repeated_mapping_cfg_axis_[repeated_mapping_idx]) {
      MappingStrategy mapping_strategy;
      mapping_strategy.mapping_idx = repeated_mapping_idx;
      mapping_strategy.offset = tmp_required_mapping_strategy[axis].offset;
      required_mapping_strategy_[axis] = mapping_strategy;
    }
    SetRepeatedMappingStrategy(mapping_str);
  } else {
    for (auto all_axis : non_repeated_mapping_cfg_axis_) {
      for (auto axis : all_axis.second) {
        MappingStrategy mapping_strategy;
        mapping_strategy.mapping_idx = all_axis.first;
        mapping_strategy.offset = tmp_required_mapping_strategy[axis].offset;
        required_mapping_strategy_[axis] = mapping_strategy;
      }
    }
  }
}

isl::schedule_node OperatorMappingStrategy::MapThreadBlockHelper(const isl::schedule_node &orig_node) {
  if (required_mapping_strategy_.empty()) {
    SetRequiredMappingCfg(orig_node);
  }
  InitRepeatedMappingConfig();
  bool is_contains_repeated_mapping = !repeated_mapping_cfg_axis_.empty();

  auto node = orig_node;
  // Map one axis to one block/thread configuration.
  if (!non_repeated_mapping_cfg_axis_.empty()) {
    ReadjustRequireddMappingStrategy(false);
    if (is_contains_repeated_mapping) {
      is_insert_filter_ = false;
      is_set_config_zero_ = false;
    }
    node = MapDimToThreadsBlocks(node);
  }

  // Map multiple axes to one block/thread configuration.
  if (is_contains_repeated_mapping) {
    std::string mappng_str = mapping_cfg_->type == MappingType::BLOCKS ? BLOCK_STR : THREAD_STR;
    size_t i = 0;
    for (auto cfg : repeated_mapping_cfg_axis_) {
      ReadjustRequireddMappingStrategy(true, cfg.first, mappng_str);
      if (i == repeated_mapping_cfg_axis_.size() - 1) {
        is_insert_filter_ = true;
        is_set_config_zero_ = true;
      }
      mapping_cfg_ = GetRepeatedReplaceMappingConfig(node, cfg.first);
      CHECK(mapping_cfg_ != nullptr) << "mapping config is null";
      node = MapDimToThreadsBlocks(node);
      ++i;
    }
  }
  return node;
}

size_t OperatorMappingStrategy::MapThreadHelper(isl::schedule_node &thread_root) {
  if (mapping_cfg_->bound < 1 || !thread_root.isa<isl::schedule_node_band>()) {
    return 0;
  }

  // Determine max num dimension of threads that can be mapped.
  auto n_thread_map = CountConsecutiveCoincident(thread_root);
  if (n_thread_map < 1) {
    thread_root = GetTiledOuterBand(thread_root);
    return 0;
  }

  // Map band under thread_root from inner dim to outer dim.
  thread_root = MapThreadBlockHelper(thread_root);

  // Do unroll if needed.
  if (scop_info_.user_config_.GetMaxUnrollLoop() != 1) {
    thread_root = UnrollByMarkOptions(thread_root, scop_info_.user_config_.GetMaxUnrollLoop());
  }

  thread_root = GetTiledOuterBand(thread_root);

  auto upa_node = thread_root;
  if (GetMarkerName(upa_node, THREAD_MARKER).empty()) {
    upa_node = upa_node.child(0);
  }
  scop_info_.upa_node_mapping_.emplace_back(std::make_pair(upa_node, mapping_sch_info_map_));
  return mapping_cfg_->bound;
}

isl::schedule_node OperatorMappingStrategy::MapBlockHelper(const isl::schedule_node &orig_node) {
  auto band_node = orig_node.as<isl::schedule_node_band>();
  if (!band_node || !band_node.permutable()) {
    LOG(WARNING) << "No permutable outer band node to map block.";
    return orig_node;
  }

  auto node = MapThreadBlockHelper(orig_node);
  node = GetTiledOuterBand(node);

  auto upa_node = node;
  if (GetMarkerName(upa_node, BLOCK_MARKER).empty()) {
    upa_node = upa_node.child(0);
  }
  scop_info_.upa_node_mapping_.emplace_back(std::make_pair(upa_node, mapping_sch_info_map_));
  return node;
}

size_t ReduceMappingStrategy::MapThreadHelper(isl::schedule_node &thread_root) {
  if (mapping_cfg_->bound < 1 || !thread_root.isa<isl::schedule_node_band>()) {
    return 0;
  }

  // Determine max num dimension of threads that can be mapped.
  auto n_thread_map = CountConsecutiveCoincident(thread_root);
  std::string reduce_marker_name = GetReduceMarkerName(thread_root, n_thread_map);
  CHECK(!reduce_marker_name.empty());

  // When akg reduce lib is enabled, we can try to map other injective statements whose coincidence equals 0
  if (n_thread_map < mapping_cfg_->bound && scop_info_.analysis_result_.GetUseGpuReduceLib()) {
    n_thread_map = mapping_cfg_->bound;
  }

  if (n_thread_map < 1) {
    thread_root = GetTiledOuterBand(thread_root);
    return 0;
  }
  n_thread_map = GetFinalMappingThreadNumber(thread_root, n_thread_map);

  // Map band under thread_root from inner dim to outer dim.
  thread_root = MapThreadBlockHelper(thread_root);

  // thread marker position
  thread_root = thread_root.parent();
  // If the current band is split during the mapping process, split the reduce axis and non-reduce axis of
  // the outer band.
  bool is_tiled = scop_info_.analysis_result_.GetOuterBandNode(band_index_)->is_thread_tile;
  if (is_tiled) {
    thread_root = thread_root.parent();
    if (n_thread_map > 1) {
      isl::schedule_node_band band_node = thread_root.as<isl::schedule_node_band>();
      thread_root = band_node.split(n_thread_map - 1).child(0);
    }
  }

  thread_root = thread_root.insert_mark(reduce_marker_name);
  // Add the filter that initializes and calls the akg_reduce library for the reduce statement.
  // Return the location of the extension node.
  thread_root = InsertReduceExtension(thread_root);

  if (thread_root.has_parent() && thread_root.parent().isa<isl::schedule_node_band>()) {
    thread_root = thread_root.parent();
  }
  scop_info_.upa_node_mapping_.emplace_back(std::make_pair(thread_root, mapping_sch_info_map_));
  return mapping_cfg_->bound;
}

std::string ReduceMappingStrategy::GetReduceMarkerName(isl::schedule_node &orig_node, size_t &n_thread_map) {
  std::string reduce_marker_name = "";
  int parent_num = 0;

  auto node = orig_node;
  // thread marker position
  if (!node.has_parent() || GetMarkerName(node.parent(), THREAD_MARKER).empty()) {
    return reduce_marker_name;
  }
  node = node.parent();
  ++parent_num;

  // split band position
  bool is_tiled = scop_info_.analysis_result_.GetOuterBandNode(band_index_)->is_thread_tile;
  if (node.has_parent() && is_tiled) {
    node = node.parent();
    ++parent_num;
  }

  // reduce marker position
  if (!node.has_parent()) {
    return reduce_marker_name;
  }
  reduce_marker_name = GetMarkerName(node.parent(), REDUCE_MARKER);
  if (!reduce_marker_name.empty()) {
    node = node.parent().del();
    ++n_thread_map;
  }

  // return to original position
  while (parent_num != 0 && node.has_children()) {
    node = node.child(0);
    --parent_num;
  }
  orig_node = node;

  return reduce_marker_name;
}

/*
 * Adjust the mapping strategy after splitting the fixed position axis.
 * E.g: required_mapping_strategy_: axis_0 --> tx  axis_1 --> tx  axis_2 --> ty  axis_3 --> tz
 * split_pos = 2
 * required_mapping_strategy_: axis_0 --> ty  axis_1 --> tz
 */
void ReduceMappingStrategy::UpadateSplitMappingStatregy(const int split_pos) {
  if (required_mapping_strategy_.empty()) {
    return;
  }

  bool need_redistribute = false;
  auto tmp_required_mapping_strategy_ = required_mapping_strategy_;
  for (auto strategy : tmp_required_mapping_strategy_) {
    if (strategy.first < split_pos) {
      continue;
    }
    if (!need_redistribute) {
      required_mapping_strategy_.clear();
      need_redistribute = true;
    }
    MappingStrategy mapping_strategy;
    mapping_strategy.mapping_idx = strategy.second.mapping_idx;
    required_mapping_strategy_[strategy.first - split_pos] = mapping_strategy;
  }
}

// Split the mappable axis into a single band according to the dimensions of the mapping configuration.
size_t ReduceMappingStrategy::GetFinalMappingThreadNumber(isl::schedule_node &node, const size_t n_thread_map) {
  auto final_n_thread_map = n_thread_map;
  isl::schedule_node_band band_node = node.as<isl::schedule_node_band>();
  // Split band node according to mapping config and coincidence of band node.
  auto mapping_cfg_bound = mapping_cfg_->bound;
  if (final_n_thread_map > mapping_cfg_bound) {
    node = band_node.split(final_n_thread_map - mapping_cfg_bound);
    UpadateSplitMappingStatregy(final_n_thread_map - mapping_cfg_bound);
    node = node.child(0);
    final_n_thread_map = mapping_cfg_bound;
    band_node = node.as<isl::schedule_node_band>();
  }

  // Split to keep nodes with coincident equals to 1.
  if (final_n_thread_map < band_node.n_member() && !scop_info_.user_config_.EnableStitchFusion()) {
    node = band_node.split(final_n_thread_map);
    UpadateSplitMappingStatregy(final_n_thread_map);
  } else {
    final_n_thread_map = static_cast<size_t>(band_node.n_member());
  }
  return final_n_thread_map;
}

isl::schedule_node ReduceMappingStrategy::InsertReduceExtension(const isl::schedule_node &node) {
  std::string reduce_marker_name = GetMarkerName(node, REDUCE_MARKER);
  if (reduce_marker_name.empty()) {
    return node;
  }

  auto insert_node = node.child(0);
  reduce_marker_name.erase(0, strlen(REDUCE_MARKER));
  isl::id sync_id = isl::id(insert_node.ctx(), REDUCE_UPDATE + reduce_marker_name);
  isl::id reduction_id = isl::id(insert_node.ctx(), REDUCE_INIT + reduce_marker_name);

  insert_node = InsertExtensionNodeBeforeOrAfter(insert_node, reduction_id, true);
  insert_node = InsertExtensionNodeBeforeOrAfter(insert_node, sync_id, false);
  insert_node = insert_node.parent().parent();
  insert_node = insert_node.insert_mark(REDUCE_AREA_FLAG);
  insert_node = insert_node.parent().parent().del();

  return insert_node;
}

bool ReduceMappingStrategy::NeedAtomicAdd(const isl::schedule_node_band &band, size_t n_block_map) {
  if (!scop_info_.analysis_result_.GetUseGpuReduceLib()) {
    return false;
  }

  auto non_coin_start_idx = CountConsecutiveCoincident(band);
  auto reduce_direction = scop_info_.analysis_result_.GetOuterBandNode(band_index_)->reduce_direction;
  bool is_reduce_x = reduce_direction == ReduceDirection::X || reduce_direction == ReduceDirection::ALL;
  bool is_all_reduce = band.n_member() == 1 && is_reduce_x && non_coin_start_idx == 1;
  if (is_all_reduce) {
    non_coin_start_idx = 0;  // Compare block size of position 0 to enable atomic add for all reduce ops
  }
  if (n_block_map < non_coin_start_idx) {
    return false;
  }

  // In order to facilitate the gpu_emit, it is required that the reduce axis must be mapped to block x.
  if (mapping_cfg_->GetAt(0).second > 1) {
    return true;
  }
  return false;
}

void ReduceMappingStrategy::MarkAtomicAddTensor(const isl::schedule_node_band &band) {
  auto target_stmt = scop_info_.analysis_result_.GetReduceWriteStmt(band);
  auto tensor = target_stmt.range();
  std::unordered_set<isl::id, isl::IslIdIslHash> stmt_ids;
  target_stmt.foreach_map(
    [this, &stmt_ids](const isl::map m) { stmt_ids.insert(m.get_tuple_id(isl_dim_type::isl_dim_in)); });
  tensor.foreach_set([this, &stmt_ids](const isl::set &s) -> void {
    for (auto it : scop_info_.analysis_result_.GetReduceTensorInfoMap()) {
      if (it.second.stmt_node->IsInstance<Provide>()) {
        auto provide = static_cast<const Provide *>(it.second.stmt_node);
        if (stmt_ids.count(it.first) == 0 || provide->func->func_name() != s.get_tuple_name()) {
          continue;
        }
        auto type = scop_info_.analysis_result_.GetReduceOpType(it.first);
        scop_info_.analysis_result_.RecordAtomicTensors(AtomicInfo{s.get_tuple_name(), type});
      }
    }
  });
}

isl::schedule ConvMappingStrategy::MoveKernelHWBand(const isl::schedule &sch) {
  auto node = sch.root();
  isl::multi_union_pw_aff kh_mupa = isl::multi_union_pw_aff::zero(node.get_domain().get_space().set_from_params());
  isl::multi_union_pw_aff kw_mupa = kh_mupa;
  auto MapFromInner = [this, &kh_mupa, &kw_mupa](isl::schedule_node node) -> isl::schedule_node {
    if (!GetMarkerName(node, KH_KW_MARKER).empty()) {
      node = node.child(0);
      kh_mupa = node.as<isl::schedule_node_band>().get_partial_schedule();
      node = node.del();
      kw_mupa = node.as<isl::schedule_node_band>().get_partial_schedule();
      node = node.del();
      node = node.parent().del();
      return node;
    }
    if (!GetMarkerName(node, PROMOTE_GLOBAL_TO_SHARED_AB).empty()) {
      node = node.insert_mark(CONV_KHKW_OUTER).child(0);
      node = node.insert_partial_schedule(kw_mupa);
      node = node.as<isl::schedule_node_band>().set_permutable(1);
      node = node.insert_partial_schedule(kh_mupa);
      node = node.as<isl::schedule_node_band>().set_permutable(1);
      return node;
    }
    return node;
  };
  auto final_sch = sch.get_root().map_descendant_bottom_up(MapFromInner).get_schedule();
  return final_sch;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
