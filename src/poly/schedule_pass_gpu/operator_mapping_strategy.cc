/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

isl::union_pw_aff_list OperatorMappingStrategy::GetUpaList(const isl::schedule_node &node,
                                                           isl::multi_union_pw_aff &partial_schedule) {
  if (is_promotion_mapping_) {
    // we need to to get range of promoted band from extension node so that we can correctly fix stride
    auto parent = node;
    while (parent && parent.has_parent() && !parent.isa<isl::schedule_node_extension>()) {
      parent = parent.parent();
    }
    if (parent.isa<isl::schedule_node_extension>()) {
      auto extension = parent.as<isl::schedule_node_extension>();
      partial_schedule = partial_schedule.intersect_domain(extension.get_extension().range());
    }
  }

  auto upa_list = partial_schedule.get_union_pw_aff_list();
  return upa_list;
}

// append prefix to partial schedule for tiling
isl::union_pw_aff_list OperatorMappingStrategy::GetPrefixPartialSchedule(
  const isl::multi_union_pw_aff &partial_schedule, const isl::schedule_node &node) {
  auto add_prefix_schedule = partial_schedule;
  isl::union_pw_aff_list prefix_upa_list(node.ctx(), static_cast<int>(add_prefix_schedule.size()));
  if (!is_thread_mapping_) {
    auto domain = node.get_schedule().get_domain();
    for (int i = 0; i < static_cast<int>(add_prefix_schedule.size()); ++i) {
      auto range = add_prefix_schedule.get_at(i).intersect_domain(domain);
      prefix_upa_list = prefix_upa_list.add(range);
    }
  } else {
    size_t max_distance_to_filter = 2;
    size_t i = 0;
    auto filter = node;
    while (i < max_distance_to_filter && filter.has_parent()) {
      filter = filter.parent();
      if (filter.isa<isl::schedule_node_filter>()) {
        break;
      }
      ++i;
    }
    if (filter.isa<isl::schedule_node_filter>()) {
      add_prefix_schedule = add_prefix_schedule.intersect_domain(filter.as<isl::schedule_node_filter>().get_filter());
    }
    prefix_upa_list = add_prefix_schedule.get_union_pw_aff_list();
  }
  return prefix_upa_list;
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
  if (node.ancestor(2) && node.ancestor(2).isa<isl::schedule_node_filter>()) {
    domain = node.ancestor(2).as<isl::schedule_node_filter>().get_filter();
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

/*
 * When mapping size is smaller than the extent of corresponding axis, we may encounter several problems if the axis
 * is not tiled. Firstly, for case that extent is multiplies of mapping sizes, directly mapping the axis will
 * generate a for loop with stride equals to `extent / mapping size`, which is not firently to emit halide ir.
 * Secondly, for case that etxnet is not divisible by mapping size, we need to generate a for loop that has offset
 * with `min` in it to deal with the tail part. This type of for loop can be generated by tiling schedule tree.
 * Therefore, we must check map size and apply tile before mapping.
 */
isl::schedule_node OperatorMappingStrategy::CheckMapSizeAndApplyTile(const isl::schedule_node &mapping_root,
                                                                     const isl::union_pw_aff_list &aff_list) {
  if (required_mapping_strategy_.empty()) {
    return mapping_root;
  }
  bool need_tile = false;
  std::vector<int> mapping_sizes;

  for (size_t i = 0; i < aff_list.size(); ++i) {
    auto aff = aff_list.get_at(i).floor();
    auto extent = aff.max_val().get_num_si() + 1;
    auto map_size = extent;
    bool is_config = (required_mapping_strategy_.count(static_cast<int>(i)) != 0);
    if (is_config) {
      auto mapping_i = required_mapping_strategy_[static_cast<int>(i)].mapping_idx;
      map_size = mapping_cfg_->GetAt(mapping_i).second - required_mapping_strategy_[static_cast<int>(i)].offset;
    }
    if (is_thread_mapping_) {
      need_tile = need_tile || extent > map_size;
    } else {
      need_tile = need_tile || (extent > map_size && extent % map_size != 0);
    }
    mapping_sizes.emplace_back(map_size);
  }

  if (!need_tile) {
    return mapping_root;
  }

  isl::multi_val tile_size;
  auto ctx = mapping_root.ctx();
  auto space = mapping_root.as<isl::schedule_node_band>().get_space();
  tile_size = isl::multi_val::zero(space);

  for (int i = 0; i < static_cast<int>(mapping_sizes.size()); ++i) {
    tile_size = tile_size.set_val(i, isl::val(ctx, mapping_sizes[i]));
  }

  return TileBand(mapping_root, tile_size).child(0);
}

isl::schedule_node OperatorMappingStrategy::MapDimToThreadsBlocks(const isl::schedule_node &node) {
  isl::schedule_node_band band_node = node.as<isl::schedule_node_band>();
  auto partial_schedule = band_node.get_partial_schedule();
  auto upa_list = GetUpaList(node, partial_schedule);

  auto prefix_upa_list = GetPrefixPartialSchedule(partial_schedule, node);
  isl::schedule_node fix_node = CheckMapSizeAndApplyTile(node, prefix_upa_list);
  bool is_tiled = !fix_node.is_equal(node);

  // insert node with specific marker
  if (is_insert_filter_) {
    auto marker_name = is_thread_mapping_ ? THREAD_MARKER : BLOCK_MARKER;
    fix_node = fix_node.insert_mark(isl::id(fix_node.ctx(), marker_name)).child(0);
  }

  auto after_map_node = AnalysisNodeAndInsertMapFilter(fix_node, upa_list);
  after_map_node = is_insert_filter_ ? after_map_node.parent() : after_map_node;
  if (is_tiled) {
    after_map_node = after_map_node.parent();
  }

  return after_map_node;
}

// Map a thread/block configuration to multiple axis.
std::string OperatorMappingStrategy::SetOneConfigForMulAxis(const isl::schedule_node &node, const int orig_total_cfg,
                                                            const std::unordered_set<int> &excluded_axis_pos) {
  auto partial_schedule = node.as<isl::schedule_node_band>().get_partial_schedule();
  if (!is_promotion_mapping_) {
    partial_schedule = partial_schedule.intersect_domain(node.domain());
  }
  auto upa_list = GetUpaList(node, partial_schedule);
  upa_list = is_need_reverse_ ? upa_list.reverse() : upa_list;

  std::string new_cfg = "";
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

  int last_axis = scop_info_.analysis_result_.GetLastAxisOfBand();
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

  int start_node_depth = thread_root.get_tree_depth();
  // Determine max num dimension of threads that can be mapped.
  auto n_thread_map = CountConsecutiveCoincident(thread_root);
  if (n_thread_map < 1) {
    return 0;
  }
  auto orig_thread_root = thread_root;

  // Map band under thread_root from inner dim to outer dim.
  thread_root = MapThreadBlockHelper(thread_root);
  auto tile_node = GetMarkerName(thread_root, THREAD_MARKER).empty() ? thread_root.child(0) : thread_root;
  scop_info_.upa_node_mapping_.emplace_back(std::make_pair(tile_node, mapping_sch_info_map_));

  // Do unroll if needed.
  if (scop_info_.user_config_.GetMaxUnrollLoop() != 1) {
    isl::schedule_node unroll_node = thread_root.child(0);
    thread_root = UnrollByMarkOptions(unroll_node, scop_info_.user_config_.GetMaxUnrollLoop());
  }

  int end_node_depth = thread_root.get_tree_depth() - start_node_depth;
  thread_root = thread_root.ancestor(end_node_depth);
  return mapping_cfg_->bound;
}

isl::schedule_node OperatorMappingStrategy::MapBlockHelper(const isl::schedule_node &orig_node) {
  auto node = orig_node;
  auto band_node = node.as<isl::schedule_node_band>();
  if (!band_node || !band_node.permutable()) {
    LOG(WARNING) << "No permutable outer band node to map block.";
    return node;
  }

  int start_node_depth = node.get_tree_depth();
  node = MapThreadBlockHelper(node);
  node = GetMarkerName(node, BLOCK_MARKER).empty() ? node.child(0) : node;
  scop_info_.upa_node_mapping_.emplace_back(std::make_pair(node, mapping_sch_info_map_));

  int end_node_depth = node.get_tree_depth() - start_node_depth;
  node = node.ancestor(end_node_depth);
  return node;
}

size_t ReduceMappingStrategy::MapThreadHelper(isl::schedule_node &thread_root) {
  if (mapping_cfg_->bound < 1 || !thread_root.isa<isl::schedule_node_band>()) {
    return 0;
  }

  int start_node_depth = thread_root.get_tree_depth();
  // Determine max num dimension of threads that can be mapped.
  auto n_thread_map = CountConsecutiveCoincident(thread_root);

  std::string reduce_marker_name = "";
  if (thread_root.has_parent()) {
    reduce_marker_name = GetMarkerName(thread_root.parent(), REDUCE_MARKER);
    if (!reduce_marker_name.empty()) {
      thread_root = thread_root.parent().del();
      ++n_thread_map;
    }
  }

  // When akg reduce lib is enabled, we can try to map other injective statements whose coincidence equals 0
  if (n_thread_map < mapping_cfg_->bound && scop_info_.analysis_result_.GetUseGpuReduceLib()) {
    n_thread_map = mapping_cfg_->bound;
  }

  if (n_thread_map < 1) {
    return 0;
  }
  n_thread_map = GetFinalMappingThreadNumber(thread_root, n_thread_map);

  // Map band under thread_root from inner dim to outer dim.
  thread_root = MapThreadBlockHelper(thread_root);

  // If the current band is split during the mapping process, split the reduce axis and non-reduce axis of
  // the outer band.
  bool is_tiled = GetMarkerName(thread_root, THREAD_MARKER).empty();
  if (is_tiled && n_thread_map > 1) {
    isl::schedule_node_band band_node = thread_root.as<isl::schedule_node_band>();
    thread_root = band_node.split(n_thread_map - 1).child(0);
  }
  thread_root = thread_root.insert_mark(reduce_marker_name);
  thread_root = thread_root.child(0);
  // Add the filter that initializes and calls the akg_reduce library for the reduce statement.
  thread_root = InsertReduceExtension(thread_root);
  // The band corresponding to the reduce statement has a REDUCE_MARKER that needs to be deleted at the beginning.
  int end_node_depth = thread_root.get_tree_depth() - start_node_depth + 1;
  thread_root = thread_root.ancestor(end_node_depth);
  scop_info_.upa_node_mapping_.emplace_back(std::make_pair(thread_root, mapping_sch_info_map_));
  return mapping_cfg_->bound;
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
  isl::schedule_node insert_node = node;
  isl::schedule_node parent_node = node;
  isl::schedule_node ancestor_node = node;
  if (insert_node.has_parent()) {
    parent_node = parent_node.parent();
    if (parent_node.has_parent()) {
      ancestor_node = parent_node.parent();
    }
  }

  std::string reduce_marker_name = "";
  if (!GetMarkerName(parent_node, REDUCE_MARKER).empty()) {
    reduce_marker_name = GetMarkerName(parent_node, REDUCE_MARKER);
    insert_node = parent_node.del();
  }

  if (!GetMarkerName(ancestor_node, REDUCE_MARKER).empty()) {
    reduce_marker_name = GetMarkerName(ancestor_node, REDUCE_MARKER);
    insert_node = ancestor_node.del();
  }

  if (reduce_marker_name.empty()) {
    return node;
  }

  reduce_marker_name.erase(0, strlen(REDUCE_MARKER));
  isl::id sync_id = isl::id(insert_node.ctx(), REDUCE_UPDATE + reduce_marker_name);
  isl::id reduction_id = isl::id(insert_node.ctx(), REDUCE_INIT + reduce_marker_name);

  insert_node = InsertExtensionNodeBeforeOrAfter(insert_node, reduction_id, true);
  insert_node = InsertExtensionNodeBeforeOrAfter(insert_node, sync_id, false).parent();
  insert_node = insert_node.parent().insert_mark(REDUCE_AREA_FLAG);

  return insert_node;
}

bool ReduceMappingStrategy::NeedAtomicAdd(const isl::schedule_node_band &band, size_t n_block_map) {
  if (!scop_info_.analysis_result_.GetUseGpuReduceLib()) {
    return false;
  }

  auto non_coin_start_idx = CountConsecutiveCoincident(band);
  bool is_reduce_x = scop_info_.analysis_result_.GetReduceDirectionOfBand() == ReduceDirection::X ||
                     scop_info_.analysis_result_.GetReduceDirectionOfBand() == ReduceDirection::ALL;
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

size_t BatchMatmulMappingStrategy::MapThreadHelper(isl::schedule_node &thread_root) {
  if (mapping_cfg_->bound < 1 || !thread_root.isa<isl::schedule_node_band>()) {
    return 0;
  }

  int start_node_depth = thread_root.get_tree_depth();
  // Determine max num dimension of threads that can be mapped.
  auto n_thread_map = CountConsecutiveCoincident(thread_root);
  if (n_thread_map < 1) {
    return 0;
  }

  // Map band under thread_root from inner dim to outer dim.
  thread_root = MapThreadBlockHelper(thread_root);
  bool is_tiled = GetMarkerName(thread_root, THREAD_MARKER).empty();
  thread_root = is_tiled ? thread_root.child(0) : thread_root;
  thread_root = thread_root.del().insert_mark(isl::id(thread_root.ctx(), WARP_MARKER));

  int end_node_depth = thread_root.get_tree_depth() - start_node_depth;
  thread_root = thread_root.ancestor(end_node_depth);
  scop_info_.upa_node_mapping_.emplace_back(std::make_pair(thread_root, mapping_sch_info_map_));
  return mapping_cfg_->bound;
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
