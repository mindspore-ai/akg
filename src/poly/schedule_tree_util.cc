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
#include "poly/schedule_tree_util.h"
#include "poly/dma_inject.h"
#include "poly/schedule_pass.h"
#include "isl_schedule_node_private.h"

namespace akg {
namespace ir {
namespace poly {

isl::union_set CollectDomain(const isl::schedule_node &node) {
  int depth = node.get_tree_depth();
  isl::schedule_node tmp_node;
  isl::union_set domain = node.get_domain();
  for (int i = 0; i < depth; ++i) {
    tmp_node = node.ancestor(depth - i);
    if (auto filter_node = tmp_node.as<isl::schedule_node_filter>()) {
      domain = domain.intersect(filter_node.get_filter());
    }
    if (auto extension_node = tmp_node.as<isl::schedule_node_extension>()) {
      auto parent_schedule = ShortSchedule(tmp_node);
      auto extension = extension_node.get_extension();
      parent_schedule = parent_schedule.intersect_domain(domain);
      domain = domain.unite(parent_schedule.range().apply(extension));
    }
  }
  return domain;
}

isl::schedule_node MapDescendantTopDown(isl::schedule_node node,
                                        const std::function<isl::schedule_node(isl::schedule_node)> &fn) {
  unsigned int depth_ = node.get_tree_depth();
  do {
    do {
      node = fn(node);
    } while (node.has_children() && (node = node.first_child()));

    while (node.get_tree_depth() > depth_ && !node.has_next_sibling()) {
      node = node.parent();
    }

    if (node.get_tree_depth() > depth_) {
      node = node.next_sibling();
    }
  } while (node.get_tree_depth() > depth_);

  return node;
}

void GetVisitedStmts(const isl::schedule_node &root) {
  int n = root.n_children();
  if (n <= 0) return;

  isl::schedule_node node;
  if (root.isa<isl::schedule_node_sequence>()) {
    isl::union_set visited_stmts;
    for (int i = 0; i < n; ++i) {
      node = root.child(i);
      auto filter_node = node.as<isl::schedule_node_filter>();
      CHECK(filter_node) << "expected children of sequence to be filters";
      auto filter = filter_node.get_filter().universe();
      if (visited_stmts.get()) {
        CHECK(visited_stmts.intersect(filter).is_empty()) << "filters are expected to be disjoint as stmt level";
        visited_stmts = visited_stmts.unite(filter);
      } else {
        visited_stmts = filter;
      }
    }
  }

  for (int i = 0; i < n; ++i) {
    node = root.child(i);
    GetVisitedStmts(node);
  }
}

std::vector<isl::schedule_node> FilterNode(std::vector<isl::schedule_node> nodes, const std::vector<isl::id> &filters) {
  auto NeedAdd = [](const isl::space &space, const std::vector<isl::id> &filter) -> bool {
    for (const auto &item : filter) {
      if (!space.has_param(item)) {
        return false;
      }
    }
    return true;
  };

  std::vector<isl::schedule_node> res_nodes;
  for (auto node : nodes) {
    if (node.isa<isl::schedule_node_filter>()) {
      auto node_filter = node.as<isl::schedule_node_filter>();
      auto curr_uset = node_filter.filter();
      auto space = curr_uset.get_space();
      if (NeedAdd(space, filters)) {
        res_nodes.push_back(node);
      }
    }
  }
  return res_nodes;
}

isl::schedule_node GenerateEmptyBandInRoot(isl::schedule_node &root) {
  auto node = root;
  if (node.n_children() > 0 && node.child(0).isa<isl::schedule_node_context>()) {
    node = node.child(0).child(0);
  }

  // construct empty band
  isl::space space;
  isl::multi_union_pw_aff mupa;
  auto tmp_domain = node.get_schedule().get_domain();
  space = tmp_domain.get_space().set_from_params();
  auto mv = isl::multi_val::zero(space);
  mupa = isl::multi_union_pw_aff(tmp_domain, mv);

  node = node.insert_partial_schedule(mupa);
  return node;
}

int GetScheduleDepth(isl::schedule &root) {
  int depth = 0;
  auto root_node = root.get_root();
  auto fn = [&depth](const isl::schedule_node &node) -> isl::schedule_node {
    if (node.isa<isl::schedule_node_band>()) {
      auto schedule_depth = static_cast<int>(node.schedule_depth());
      schedule_depth = schedule_depth + static_cast<int>(node.as<isl::schedule_node_band>().n_member());
      depth = schedule_depth > depth ? schedule_depth : depth;
    }
    return node;
  };
  root_node = root_node.map_descendant_bottom_up(fn);
  return depth;
}

std::vector<isl::schedule_node> BandsContainingScheduleDepth(isl::schedule_node &root, size_t depth) {
  if (depth == 0) {
    return {GenerateEmptyBandInRoot(root)};
  }

  std::vector<isl::schedule_node> bands;
  CollectBandsOnTree(root, bands);
  std::function<bool(isl::schedule_node st)> contains_depth = [&depth](isl::schedule_node st) {
    auto depth_before = st.schedule_depth();
    auto band = st.as<isl::schedule_node_band>();
    auto depth_after = depth_before + band.n_member();
    return depth_before < depth && depth_after >= depth;
  };
  return FilterWithFunc(contains_depth, bands);
}

void CollectBandsOnTree(isl::schedule_node &root, std::vector<isl::schedule_node> &bands) {
  for (unsigned int i = 0; i < root.n_children(); ++i) {
    isl::schedule_node node = root.get_child(i);
    if (node.isa<isl::schedule_node_band>()) {
      bands.insert(bands.end(), node);
    }
    CollectBandsOnTree(node, bands);
  }
  return;
}

// whether the node is a "thread_marker".
// It means the band below this node is a thread-mapped band.
bool IsThreadMappedMark(const isl::schedule_node &node) {
  if (node.isa<isl::schedule_node_mark>() && node.n_children() > 0 &&
      node.as<isl::schedule_node_mark>().get_id().get_name().find(THREAD_MARKER) != std::string::npos) {
    return true;
  }
  return false;
}

// find all the ancestors to check whether any of them is a "thread_marker" node.
// NOTE: because of our schedule architecture, the "thread_marker" node is on top
// of thread-mapped band, like:
//  ----------
//  mark: "thread_marker"  <--
//  child:
//     filter : "..."
//     child:
//         schedule: "..." <--
bool IsAncestorMapToThread(const isl::schedule_node &curr_node) {
  bool has_thread_mark_node = false;
  auto FindThreadMarkNode = [&has_thread_mark_node](const isl::schedule_node node) {
    has_thread_mark_node |= IsThreadMappedMark(node);
    return node;
  };
  curr_node.foreach_ancestor_top_down(FindThreadMarkNode);
  return has_thread_mark_node;
}

isl::schedule_node BandSplitAtDepth(isl::schedule_node &band, size_t depth) {
  if (!band.isa<isl::schedule_node_band>()) {
    return band;
  }
  auto n_member = band.as<isl::schedule_node_band>().n_member();
  auto schedule_depth = band.schedule_depth();
  auto depth_after = schedule_depth + n_member;
  return depth_after == depth ? band : band.as<isl::schedule_node_band>().split(depth - schedule_depth);
}

std::vector<isl::schedule_node> BandsSplitAfterDepth(const std::vector<isl::schedule_node> &bands,
                                                     isl::schedule_node &root, size_t depth) {
  std::function<isl::schedule_node(isl::schedule_node)> split_at_depth = [&depth](isl::schedule_node st) {
    auto n_member = st.as<isl::schedule_node_band>().n_member();
    auto schedule_depth = st.schedule_depth();
    auto depth_after = schedule_depth + n_member;
    return depth_after == depth ? st : st.as<isl::schedule_node_band>().split(depth - schedule_depth);
  };
  return MapWithFunc(split_at_depth, bands);
}

isl::schedule InsertMarkerForThreadGroup(const isl::schedule &sch, const std::string &write_name,
                                         const std::string &marker_name) {
  auto GetPromotedWriteFilter = [write_name, marker_name](isl::schedule_node node) -> isl::schedule_node {
    if (!node.isa<isl::schedule_node_filter>()) {
      return node;
    }
    isl::union_set uset = node.as<isl::schedule_node_filter>().get_filter();
    bool is_gm_write = false;
    uset.foreach_set([&is_gm_write, write_name](isl::set s) {
      if (s.get_tuple_name() == write_name) {
        is_gm_write = true;
      }
    });
    if (is_gm_write && node.has_parent() && node.parent().isa<isl::schedule_node_sequence>()) {
      node = node.child(0).insert_mark(marker_name);
      node = node.parent();
    }
    return node;
  };
  auto final_sch = sch.get_root().map_descendant_bottom_up(GetPromotedWriteFilter).schedule();
  return final_sch;
}

isl::schedule_node AdjustConvScheduleTreeStructure(const isl::schedule_node &orig_node, const bool is_promotion) {
  auto node = orig_node;
  if (!node.isa<isl::schedule_node_band>()) {
    return node;
  }

  auto band_node = node.as<isl::schedule_node_band>();
  auto orig_number = band_node.n_member();
  if (orig_number <= 2) {
    return node;
  }

  // original node
  auto orig_partial_schedule = band_node.get_partial_schedule();
  bool orig_permutable = band_node.get_permutable();
  std::vector<bool> orig_coincident;
  for (int i = 0; i < static_cast<int>(orig_number); ++i) {
    orig_coincident.push_back(band_node.member_get_coincident(i));
  }

  isl::union_pw_aff_list new_partial_schedule(node.ctx(), orig_number);
  auto InsertPartialSchedule = [&new_partial_schedule](isl::schedule_node node) -> void {
    auto partial_schedule = node.as<isl::schedule_node_band>().get_partial_schedule();
    for (int i = 0; i < static_cast<int>(partial_schedule.size()); ++i) {
      new_partial_schedule = new_partial_schedule.add(partial_schedule.get_at(i));
    }
  };

  // split n axis
  node = node.as<isl::schedule_node_band>().split(1);
  auto n_node = node;
  node = node.del();

  // split h and w axis
  const int h_w_axis_size = 2;
  int real_h_w_axis_size = is_promotion ? static_cast<int>(orig_number) - h_w_axis_size : h_w_axis_size;
  node = node.as<isl::schedule_node_band>().split(real_h_w_axis_size);
  InsertPartialSchedule(node);
  node = node.del();
  InsertPartialSchedule(n_node);

  // split o and other axis
  InsertPartialSchedule(node);

  node = node.insert_partial_schedule(isl::multi_union_pw_aff(orig_partial_schedule.get_space(), new_partial_schedule));
  band_node = node.as<isl::schedule_node_band>();
  band_node = band_node.set_permutable(orig_permutable);
  for (int i = 0; i < static_cast<int>(orig_number); ++i) {
    band_node = band_node.member_set_coincident(i, orig_coincident[i]);
  }
  return band_node;
}

std::string GetMarkerName(const isl::schedule_node &node, std::string find_name) {
  std::string marker_name = "";
  if (node.isa<isl::schedule_node_mark>()) {
    marker_name = node.as<isl::schedule_node_mark>().get_id().get_name();
    if (marker_name.find(find_name) != std::string::npos) {
      return marker_name;
    }
    marker_name = "";
  }
  return marker_name;
}

isl::union_pw_aff_list GetUPAList(const isl::schedule_node &node, isl::multi_union_pw_aff &partial_schedule,
                                  const bool is_promotion, const bool need_reverse) {
  if (is_promotion) {
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

  auto upa_list = partial_schedule.get_union_pw_aff_list().reverse();

  if (need_reverse) {
    upa_list = upa_list.reverse();
  }
  return upa_list;
}

// append prefix to partial schedule for tiling
isl::union_pw_aff_list GetPrefixPartialSchedule(const isl::multi_union_pw_aff partial_schedule,
                                                const isl::schedule_node node, const bool need_reverse) {
  auto add_prefix_schedule = partial_schedule;
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
  auto prefix_upa_list = add_prefix_schedule.get_union_pw_aff_list().reverse();

  if (need_reverse) {
    prefix_upa_list = prefix_upa_list.reverse();
  }
  return prefix_upa_list;
}

isl::schedule_node MapInnerDimToThreads(const isl::schedule_node &node, MappingCfg *mapping_cfg, Mapping &mapping,
                                        const bool is_promotion, const bool need_reverse) {
  CHECK(mapping_cfg != nullptr) << "thread config is null";
  isl::schedule_node_band band_node = node.as<isl::schedule_node_band>();
  size_t n_thread_map = std::min(static_cast<size_t>(band_node.n_member()), mapping_cfg->bound);
  CHECK_LE(n_thread_map, mapping_cfg->MaxDim()) << "mapping to too many threads.";
  auto partial_schedule = band_node.get_partial_schedule();
  auto upa_list = GetUPAList(node, partial_schedule, is_promotion, need_reverse);

  auto prefix_upa_list = GetPrefixPartialSchedule(partial_schedule, node, need_reverse);
  isl::schedule_node fix_node = CheckMapSizeAndApplyTile(node, prefix_upa_list, mapping_cfg, need_reverse);
  bool is_tiled = !fix_node.is_equal(node);

  // drop un-mapped aff after tiling
  upa_list = upa_list.drop(n_thread_map, upa_list.size() - n_thread_map);

  // insert node with specific marker
  fix_node = fix_node.insert_mark(isl::id(fix_node.ctx(), THREAD_MARKER));
  fix_node = fix_node.child(0);

  auto after_map_node = AnalysisNodeAndInsertMapFilter(fix_node, is_promotion, upa_list, mapping_cfg, mapping);
  after_map_node = after_map_node.parent();
  if (is_tiled) {
    after_map_node = after_map_node.parent();
  }

  return after_map_node;
}

isl::schedule_node AnalysisNodeAndInsertMapFilter(const isl::schedule_node &node, const bool is_promotion,
                                                  isl::union_pw_aff_list upa_list, MappingCfg *mapping_cfg,
                                                  Mapping &mapping, std::unordered_map<size_t, size_t> map_idx_shift) {
  // create mapping filter
  CHECK(mapping_cfg != nullptr) << "threadconfig is null";

  isl::union_set domain = node.get_schedule().get_domain();
  if (node.ancestor(2) && node.ancestor(2).isa<isl::schedule_node_filter>()) {
    domain = node.ancestor(2).as<isl::schedule_node_filter>().get_filter();
  }

  size_t num_map = upa_list.size();
  for (size_t i = 0; i < num_map; ++i) {
    auto map_id = map_idx_shift.find(i) != map_idx_shift.end() ? map_idx_shift[i] : i;
    std::pair<std::string, int> cfg = mapping_cfg->GetAt(map_id);
    auto upa = upa_list.get_at(i);
    CHECK_GT(cfg.second, 0);
    upa = upa.mod(isl::val(node.ctx(), cfg.second));
    auto id = isl::id(node.ctx(), cfg.first);
    mapping[id] = upa;
    domain = upa.domain();
  }

  for (size_t i = num_map; i < mapping_cfg->bound; ++i) {
    CHECK(!domain.is_null());
    auto universe = domain.universe();
    auto map_id = map_idx_shift.find(i) != map_idx_shift.end() ? map_idx_shift[i] : i;
    std::pair<std::string, int> cfg = mapping_cfg->GetAt(map_id);
    auto id = isl::id(node.ctx(), cfg.first);
    mapping[id] = isl::union_pw_aff(universe, isl::val::zero(domain.ctx()));
  }

  return InsertMapFilter(node, is_promotion, mapping);
}

isl::schedule_node InsertMapFilter(const isl::schedule_node &node, const bool is_promotion, Mapping &mapping) {
  if (mapping.size() == 0) {
    return node;
  }
  // extract unique domain
  auto map_domain = mapping.cbegin()->second.domain();
  if (!is_promotion) {
    for (const auto &kvp : mapping) {
      CHECK(map_domain.is_equal(kvp.second.domain()));
    }
  }

  auto map_filter = map_domain.universe();
  for (const auto &kvp : mapping) {
    auto id = kvp.first;
    auto upa = kvp.second;
    upa = upa.sub(isl::union_pw_aff::param_on_domain(map_domain.universe(), id));
    map_filter = map_filter.intersect(upa.zero_union_set());
  }

  // insert mapping filter
  isl::schedule_node map_filter_node = node;
  map_filter_node = map_filter_node.insert_filter(map_filter);
  return map_filter_node;
}

isl::schedule_node InsertRequiredMappingFilter(const isl::schedule_node &node, isl::union_pw_aff_list upa_list,
                                               MappingCfg *mapping_cfg, Mapping &mapping,
                                               std::unordered_map<int, std::string> required_mapping,
                                               std::unordered_set<std::string> outer_mapping_cfg) {
  isl::union_set domain = node.get_schedule().get_domain();

  std::unordered_set<std::string> current_mapping_cfg;
  for (size_t i = 0; i < upa_list.size(); ++i) {
    if (required_mapping.count(static_cast<int>(i)) == 0) {
      continue;
    }
    auto mapping_i = required_mapping[static_cast<int>(i)];
    std::pair<std::string, int> cfg = mapping_cfg->GetAt(mapping_i);
    current_mapping_cfg.emplace(cfg.first);

    auto upa = upa_list.get_at(i);
    CHECK_GT(cfg.second, 0);
    upa = upa.mod(isl::val(node.ctx(), cfg.second));
    auto id = isl::id(node.ctx(), cfg.first);
    mapping[id] = upa;
    domain = upa.domain();
  }

  // Set other configurations to 0.
  if (!outer_mapping_cfg.empty()) {
    for (size_t i = 0; i < mapping_cfg->bound; ++i) {
      CHECK(!domain.is_null());
      auto universe = domain.universe();
      // Remove the configuration that has been mapped.
      if (current_mapping_cfg.find(mapping_cfg->GetAt(i).first) != current_mapping_cfg.end()) {
        continue;
      }
      // Remove the configuration in the outer mapping.
      if (outer_mapping_cfg.find(mapping_cfg->GetAt(i).first) != outer_mapping_cfg.end()) {
        continue;
      }
      std::pair<std::string, int> cfg = mapping_cfg->GetAt(i);
      auto id = isl::id(node.ctx(), cfg.first);
      mapping[id] = isl::union_pw_aff(universe, isl::val::zero(domain.ctx()));
    }
  }

  return InsertMapFilter(node, false, mapping);
}

/*
 * When mapping size is smaller than the extent of corresponding axis, we may encounter several problems if the axis
 * is not tiled. Firstly, for case that extent is multiplies of mapping sizes, directly mapping the axis will
 * generate a for loop with stride equals to `extent / mapping size`, which is not firently to emit halide ir.
 * Secondly, for case that etxnet is not divisible by mapping size, we need to generate a for loop that has offset
 * with `min` in it to deal with the tail part. This type of for loop can be generated by tiling schedule tree.
 * Therefore, we must check map size and apply tile before mapping.
 */
isl::schedule_node CheckMapSizeAndApplyTile(const isl::schedule_node &mapping_root,
                                            const isl::union_pw_aff_list &aff_list, MappingCfg *mapping_cfg,
                                            const bool need_reverse,
                                            std::unordered_map<int, std::string> required_mapping) {
  bool need_tile = false;
  std::vector<int> mapping_sizes;
  CHECK(mapping_cfg != nullptr) << "mapping config is null";
  size_t block_count = 0;
  int extent = 0;
  int map_size = 0;

  auto RecordMappingSizes = [&mapping_sizes, &need_tile, &map_size, &extent](const bool is_config,
                                                                             const bool is_thread) -> void {
    if (is_config) {
      if (is_thread) {
        need_tile = need_tile || extent > map_size;
      } else {
        need_tile = need_tile || (extent > map_size && extent % map_size != 0);
      }
    }
    mapping_sizes.emplace_back(map_size);
  };

  for (size_t i = 0; i < aff_list.size(); ++i) {
    auto aff = aff_list.get_at(i).floor();
    extent = aff.max_val().get_num_si() + 1;
    map_size = extent;
    // custom mapping
    if (!required_mapping.empty()) {
      bool is_config = (required_mapping.count(static_cast<int>(i)) != 0);
      if (is_config) {
        auto mapping_i = required_mapping[static_cast<int>(i)];
        map_size = mapping_cfg->GetAt(mapping_i).second;
      }
      RecordMappingSizes(is_config, false);
      continue;
    }

    if (mapping_cfg->type == MappingType::BLOCKS) {
      // block mapping
      bool is_config = (aff_list.size() - 1 - i < mapping_cfg->bound);
      if (is_config) {
        map_size = mapping_cfg->GetAt(block_count).second;
        ++block_count;
      }
      RecordMappingSizes(is_config, false);
    } else {
      // thread mapping
      bool is_config = (i < mapping_cfg->bound);
      if (is_config) {
        map_size = mapping_cfg->GetAt(i).second;
      }
      RecordMappingSizes(is_config, true);
    }
  }

  if (!need_tile) {
    return mapping_root;
  }

  isl::multi_val tile_size;
  auto ctx = mapping_root.ctx();
  auto space = mapping_root.as<isl::schedule_node_band>().get_space();
  tile_size = isl::multi_val::zero(space);

  auto len = static_cast<int>(mapping_sizes.size());
  for (auto i = len - 1; i >= 0; --i) {
    int pos = need_reverse ? i : len - 1 - i;
    tile_size = tile_size.set_val(pos, isl::val(ctx, mapping_sizes[i]));
  }

  return TileBand(mapping_root, tile_size).child(0);
}

bool IsEqualNode(const isl::schedule_node node1, const isl::schedule_node node2) {
  auto node_ptr1 = node1.get();
  auto node_ptr2 = node2.get();

  if (!node_ptr1 || !node_ptr2) {
    return false;
  }
  if (node_ptr1 == node_ptr2) {
    return true;
  }
  if (isl_schedule_node_get_type(node_ptr1) != isl_schedule_node_get_type(node_ptr2)) {
    return false;
  }
  if (node1.isa<isl::schedule_node_band>()) {
    isl::schedule_node_band band_node1 = node1.as<isl::schedule_node_band>();
    isl::schedule_node_band band_node2 = node2.as<isl::schedule_node_band>();

    if (band_node1.permutable() != band_node2.permutable()) {
      return false;
    }

    if (band_node1.n_member() != band_node2.n_member()) {
      return false;
    }

    size_t count = 0;
    while (count < band_node1.n_member()) {
      if (band_node1.member_get_coincident(static_cast<int>(count)) !=
          band_node2.member_get_coincident(static_cast<int>(count))) {
        return false;
      }
      ++count;
    }

    if (!band_node1.get_partial_schedule().plain_is_equal(band_node2.get_partial_schedule())) {
      return false;
    }
  } else if (node1.isa<isl::schedule_node_filter>()) {
    isl::schedule_node_filter filter_node1 = node1.as<isl::schedule_node_filter>();
    isl::schedule_node_filter filter_node2 = node2.as<isl::schedule_node_filter>();

    if (!filter_node1.filter().is_equal(filter_node2.filter())) {
      return false;
    }
    return IsEqualNode(filter_node1.child(0), filter_node2.child(0));
  }
  return true;
}

isl::multi_union_pw_aff MapDomainToThread(const isl::schedule_node &node, MappingCfg *mapping_cfg,
                                          const UpaNodeMapping &upa_node_mapping) {
  std::vector<isl::id> thread_ids;
  for (size_t i = 0; i < mapping_cfg->bound; ++i) {
    auto ti = mapping_cfg->GetAt(i);
    auto id = isl::id(node.ctx(), ti.first);
    thread_ids.emplace_back(id);
  }

  isl::space space = isl::space(node.ctx(), 0);
  isl::union_set empty_domain = isl::union_set::empty(space);
  space = space.add_named_tuple_id_ui(isl::id(node.ctx(), SYNC_BLOCK), thread_ids.size());
  auto domain_threads = isl::multi_union_pw_aff(empty_domain, isl::multi_val::zero(space));
  UpaNodeMapping tmp_upa_node_mapping = upa_node_mapping;

  auto CompareFromInner = [&domain_threads, &tmp_upa_node_mapping, thread_ids, node,
                           space](isl::schedule_node compare_node) -> isl::schedule_node {
    for (size_t i = 0; i < tmp_upa_node_mapping.size(); ++i) {
      auto upa_mapping = tmp_upa_node_mapping[i];
      auto upa_node = upa_mapping.first;
      auto tmp_node = upa_node;
      if (!tmp_node.is_null() && tmp_node.has_parent() && tmp_node.parent().isa<isl::schedule_node_filter>()) {
        tmp_node = tmp_node.parent();
      }

      if (IsEqualNode(tmp_node, compare_node)) {
        auto mapping = upa_mapping.second;
        auto upa_list = isl::union_pw_aff_list(node.ctx(), thread_ids.size());
        for (auto thread_id : thread_ids) {
          if (mapping.count(thread_id) == 0) {
            break;
          }
          upa_list = upa_list.add(mapping.at(thread_id));
        }
        if (upa_list.size() == thread_ids.size()) {
          auto domain_upa_node = CollectDomain(upa_node);
          auto domain_intersection = domain_upa_node.intersect(domain_threads.domain());
          CHECK(domain_intersection.is_empty())
            << "This domain has been mapped to threadID and show that there is an intersection.";

          auto upa_node_thread = isl::multi_union_pw_aff(space, upa_list);
          upa_node_thread = upa_node_thread.intersect_domain(domain_upa_node);
          domain_threads = domain_threads.union_add(upa_node_thread);
        }
        tmp_upa_node_mapping.erase(tmp_upa_node_mapping.begin() + i);
        return compare_node;
      }
    }
    return compare_node;
  };
  node.map_descendant_bottom_up(CompareFromInner);

  return domain_threads;
}

// this function map domain from all the mapping with specific marker: thread_marker or block_marker.
// we foreach the upa_node_mapping and check whether a mapping belongs to thread/block marker.
isl::multi_union_pw_aff MapDomainAllWithType(const isl::schedule_node &node, MappingCfg *mapping_cfg,
                                             const UpaNodeMapping &upa_node_mapping, const std::string &map_type) {
  CHECK((map_type == THREAD_MARKER || map_type == BLOCK_MARKER)) << "map_type should be THREAD_MARKER or BLCOK_MARKER.";
  std::vector<isl::id> ids;
  for (size_t i = 0; i < mapping_cfg->bound; ++i) {
    auto ti = mapping_cfg->GetAt(i);
    auto id = isl::id(node.ctx(), ti.first);
    ids.emplace_back(id);
  }

  isl::space space = isl::space(node.ctx(), 0);
  isl::union_set empty_domain = isl::union_set::empty(space);
  space = space.add_named_tuple_id_ui(isl::id(node.ctx(), map_type), ids.size());
  // domain_association: connect thread/block with domain
  auto domain_association = isl::multi_union_pw_aff(empty_domain, isl::multi_val::zero(space));

  for (auto upa_mapping : upa_node_mapping) {
    auto upa_node = upa_mapping.first;
    auto tmp_node = upa_node;
    CHECK(!tmp_node.is_null() && tmp_node.has_parent()) << "node from upa_node_mapping is invalid.";

    // check whether this node is a mark node with map_type.
    if (!tmp_node.isa<isl::schedule_node_mark>() ||
        (tmp_node.isa<isl::schedule_node_mark>() &&
         tmp_node.as<isl::schedule_node_mark>().get_id().get_name().find(map_type) == std::string::npos)) {
      continue;
    }

    auto mapping = upa_mapping.second;
    auto upa_list = isl::union_pw_aff_list(node.ctx(), ids.size());
    for (auto id : ids) {
      if (mapping.count(id) == 0) {
        break;
      }
      upa_list = upa_list.add(mapping.at(id));
    }
    if (upa_list.size() == ids.size()) {
      auto domain_upa_node = CollectDomain(upa_node);
      auto domain_intersection = domain_upa_node.intersect(domain_association.domain());
      CHECK(domain_intersection.is_empty())
        << "This domain has been mapped to threadID/blockID and show that there is an intersection.";

      auto upa_node_association = isl::multi_union_pw_aff(space, upa_list);
      upa_node_association = upa_node_association.intersect_domain(domain_upa_node);
      domain_association = domain_association.union_add(upa_node_association);
    }
  }

  auto domain_node = CollectDomain(node);
  bool sub_set = domain_node.is_subset(domain_association.domain());
  CHECK(sub_set) << "There are remaining domains that have not been mapped to threadID/blockID";

  return domain_association;
}

isl::map CreateMapIncreaseDim(isl::space space, unsigned dim) {
  isl::space map_space = space.map_from_set();
  isl::multi_aff identity = isl::multi_aff::identity(map_space);

  if (dim < 0 || dim >= identity.size()) {
    LOG(FATAL) << "In the space, " << dim << " should be in the range of [0, " << identity.size() << ")";
  }

  isl::aff aff = identity.get_aff(dim);
  identity = identity.set_aff(dim, aff + 1);
  return isl::map(identity);
}

bool IsSubsetForIncreaseDim(const isl::map access, size_t tensor_dim, size_t node_dim) {
  auto schedule_space = access.get_space().domain();
  auto tensor_space = access.get_space().range();
  auto element_next = CreateMapIncreaseDim(tensor_space, tensor_dim);

  auto schedule_next = CreateMapIncreaseDim(schedule_space, node_dim);
  auto access_by_adjacent_inner = schedule_next.apply_domain(access).apply_range(access);
  if (!access_by_adjacent_inner.is_subset(element_next)) {
    return false;
  }
  return true;
}

int GetLastAxis(const isl::schedule_node node, isl::union_map original_access,
                std::unordered_set<std::string> skip_tensors) {
  if (!node.isa<isl::schedule_node_band>()) {
    return -1;
  }
  // Get current node information.
  auto active_domains = CollectDomain(node);
  auto local_access = original_access.intersect_domain(active_domains);
  auto schedule = LocalSchedule(node);
  auto schedule_access = local_access.apply_domain(schedule);

  int node_depth = static_cast<int>(node.as<isl::schedule_node_band>().n_member());
  for (auto access : schedule_access.get_map_list()) {
    // Skip the related tensor in tensor of tensor.
    auto tensor_name = access.range().get_tuple_name();
    if (skip_tensors.count(tensor_name) != 0) {
      continue;
    }

    int tensor_dim = -1;
    for (int i = static_cast<int>(access.range().n_dim()) - 1; i >= 0; --i) {
      auto axis_i = access.range().dim_max(i);
      if (!axis_i.is_equal(isl::pw_aff(axis_i.domain(), isl::val(axis_i.ctx(), 0)))) {
        tensor_dim = i;
        break;
      }
    }

    if (tensor_dim < 0) {
      continue;
    }

    for (int i = node_depth - 1; i >= 0; --i) {
      if (!IsSubsetForIncreaseDim(access, tensor_dim, i)) {
        continue;
      }
      return i;
    }
  }
  return -1;
}

std::vector<isl::schedule_node> CollectFnNode(const std::function<bool(const isl::schedule_node &)> &fn,
                                              const isl::schedule_node &root) {
  std::vector<isl::schedule_node> res_nodes;
  auto GetFnNode = [&res_nodes, &fn](isl::schedule_node node) -> isl::schedule_node {
    if (fn(node)) {
      res_nodes.push_back(node);
    }
    return node;
  };
  root.map_descendant_bottom_up(GetFnNode);
  return res_nodes;
}

isl::val GetInstancesBound(isl::schedule_node &node, isl::union_map ancestors_schedule, isl::val unroll_val) {
  auto instances_bound = isl::val::zero(unroll_val.ctx());
  if (!node.has_children()) {
    instances_bound = isl::val::one(unroll_val.ctx());
  } else {
    // Combine the schedule of ancestors and expand own schedule.
    auto next_schedule = ancestors_schedule;
    if (auto band_node = node.as<isl::schedule_node_band>()) {
      if (band_node.n_member() > 0) {
        next_schedule = next_schedule.flat_range_product(band_node.get_partial_schedule_union_map());
      }
    } else if (auto filter_node = node.as<isl::schedule_node_filter>()) {
      next_schedule = next_schedule.intersect_domain(filter_node.get_filter());
    } else if (auto extension_node = node.as<isl::schedule_node_extension>()) {
      next_schedule =
        next_schedule.unite(extension_node.get_extension().reverse().intersect_range(next_schedule.range()));
    }

    for (size_t i = 0; i < node.n_children(); ++i) {
      auto child = node.child(i);
      instances_bound = instances_bound.add(GetInstancesBound(child, next_schedule, unroll_val));
      node = child.parent();
    }
  }

  // Calculate the total bound of instances executed by this band node.
  if (auto band_node = node.as<isl::schedule_node_band>()) {
    if (instances_bound.gt(unroll_val)) {
      return isl::val::infty(unroll_val.ctx());
    }

    isl::multi_union_pw_aff partial_schedule = band_node.get_partial_schedule();
    isl::union_pw_aff_list upa_list = partial_schedule.get_union_pw_aff_list();
    isl::space space = partial_schedule.get_space().params();

    for (size_t i = 0; i < band_node.n_member(); ++i) {
      isl::union_pw_aff upa = partial_schedule.get_at(i);
      auto tmp_scheduel = ancestors_schedule;
      if (i != band_node.n_member() - 1) {
        upa_list = upa_list.drop(i, 1);
        isl::space unnamed_space = space.add_unnamed_tuple_ui(upa_list.size());
        auto unname_upa = isl::multi_union_pw_aff(unnamed_space, upa_list);
        tmp_scheduel = tmp_scheduel.flat_range_product(isl::union_map::from(unname_upa));
      }
      // For fixed values of ancestors_schedule, calculate a bound on the range of values attained by upa.
      auto union_map = isl::union_map::from(isl::multi_union_pw_aff(upa)).apply_domain(tmp_scheduel);
      isl::val upa_bound = isl::val::zero(upa.ctx());
      if (!union_map.is_empty()) {
        union_map = union_map.range_product(union_map).range().unwrap().project_out_all_params();

        isl::set delta = isl::map::from(union_map).deltas();
        isl::basic_set hull = delta.simple_hull();
        isl::val stride = isl::set(hull).get_stride(0);
        hull = isl::set(hull).polyhedral_hull();

        upa_bound = hull.dim_max_val(0);
        upa_bound = upa_bound.div(stride).add(isl::val::one(upa.ctx()));
      }
      instances_bound = instances_bound.mul(upa_bound);
      if (instances_bound.gt(unroll_val)) {
        return isl::val::infty(unroll_val.ctx());
      }
      band_node = band_node.member_set_ast_loop_unroll(i);
      node = band_node;
    }
  }
  return instances_bound;
}

isl::schedule_node UnrollByMarkOptions(isl::schedule_node &node, uint64_t unroll) {
  if (unroll <= 1) {
    return node;
  }

  int depth = node.get_tree_depth();
  isl::schedule_node tmp_node;
  isl::union_set domain = node.get_schedule().get_domain();

  // In the mapping, locate above the mark to get the corresponding domain.
  auto child_node = node;
  if (node.isa<isl::schedule_node_mark>() && node.has_children()) {
    child_node = node.child(0);
  }
  for (int i = 0; i < depth; ++i) {
    tmp_node = child_node.ancestor(depth - i);

    if (tmp_node.isa<isl::schedule_node_mark>()) {
      std::string mark_id = tmp_node.as<isl::schedule_node_mark>().get_id().get_name();
      if (mark_id.find(THREAD_MARKER) != std::string::npos) {
        if (tmp_node.has_children()) {
          if (auto filter_node = tmp_node.child(0).as<isl::schedule_node_filter>()) {
            domain = domain.intersect(filter_node.get_filter());
          }
        }
      }
    }

    if (auto extension_node = tmp_node.as<isl::schedule_node_extension>()) {
      auto parent_schedule = ShortSchedule(tmp_node);
      auto extension = extension_node.get_extension();
      parent_schedule = parent_schedule.intersect_domain(domain);
      domain = domain.unite(parent_schedule.range().apply(extension));
    }
  }

  auto unroll_val = isl::val(node.ctx(), unroll);
  auto ancestors_schedule = ShortSchedule(node);

  ancestors_schedule = ancestors_schedule.intersect_domain(domain);
  GetInstancesBound(node, ancestors_schedule, unroll_val);
  return node;
}

isl::map GetExtensionSpace(const isl::schedule_node &node, const isl::id &id) {
  auto prefix = ShortScheduleMupaImpl(node.root(), node.root(), node.parent());
  auto schedule_space = prefix.get_space();
  auto space = schedule_space.params().add_named_tuple_id_ui(id, 0);
  auto extension_space = isl::map::universe(schedule_space.map_from_domain_and_range(space));
  return extension_space;
}

isl::schedule_node InsertExtensionNodeBeforeOrAfter(const isl::schedule_node &node, const isl::id &id, bool before) {
  auto space = GetExtensionSpace(node, id);
  isl::schedule_node graft = isl::schedule_node::from_extension(space);
  auto extension_node = node;
  if (before) {
    extension_node = extension_node.graft_before(graft);
  } else {
    extension_node = extension_node.graft_after(graft);
  }
  return extension_node;
}

isl::union_set GetBlockMappingFilterInfo(const isl::schedule_node node, MappingCfg *block_cfg,
                                         std::unordered_map<std::string, MappingCfg *> replace_cfg) {
  isl::union_set mapping;
  for (auto it : replace_cfg) {
    auto cfg = it.second;
    if (cfg->type == MappingType::REPLACE_BLOCKS) {
      if (mapping.is_null()) {
        mapping = GatherMappingsTo(node, cfg);
      } else {
        mapping = mapping.intersect(GatherMappingsTo(node, cfg));
      }
    }
  }
  if (mapping.is_null()) {
    mapping = GatherMappingsTo(node, block_cfg);
  }
  return mapping;
}

isl::union_set GatherMappingsTo(const isl::schedule_node &root, MappingCfg *cfg) {
  auto domain_node = root.as<isl::schedule_node_domain>();
  auto domain = domain_node.domain();
  auto sch = root.get_schedule();
  auto mapping_filters = CollectNode<isl::schedule_node_filter>(sch);

  std::vector<isl::id> filters;
  for (size_t idx = 0; idx < cfg->bound; ++idx) {
    auto value = cfg->GetAt(idx);
    auto id = isl::id(root.ctx(), value.first);
    filters.push_back(id);
  }
  mapping_filters = FilterNode(mapping_filters, filters);

  auto mapping = isl::union_set::empty(domain.ctx());
  for (auto item : mapping_filters) {
    if (item.isa<isl::schedule_node_filter>()) {
      auto filter = item.as<isl::schedule_node_filter>();
      if (filter.has_parent() && !filter.parent().isa<isl::schedule_node_mark>()) {
        continue;
      }

      isl::union_set uset = filter.get_filter();
      std::vector<isl::set> vset;
      uset.foreach_set([&vset](isl::set s) { vset.push_back(s); });
      if (!vset.empty()) {
        auto filter_name = vset[0].get_tuple_name();
        if (filter_name == READ_ID_NAME || filter_name == WRITE_ID_NAME) {
          continue;
        }
      }

      mapping = mapping.unite(filter.filter());
    }
  }
  return mapping;
}

/* Check that whether the mapping relation between instance statement
 * and outer schedule points and tensor elements pair is reusable. */
bool ReuseTensorCluster(const TensorFootprintCluster &cluster, const isl::multi_union_pw_aff &outer_pw_aff) {
  /* compute the mapping relation between statement instance and outer schedule space and tensor elements pair */
  /* Here we use the property of bijective to decide whether promote this tensor to shared.
   * For element wise operator, S -> tensor_schedule is bijective.
   * It should not be promoted to shared/local memory.
   * For reduced operator, S -> tensor_schedule is not bijective.
   * It should be promoted to shared/local memory.
   * For stencil operator in sciencetific computing, S -> tensor_schedule is not bijective.
   * It should be promoted to shared/local memory.
   * *******************************************************************************************/
  isl::union_map state_schedule_mapping = ScheduleTensorMapping(outer_pw_aff, cluster.OrigianlAccessRelations());
  return !state_schedule_mapping.is_injective();
}

isl::schedule_node CollectMarkNodeOnPromotion(const isl::schedule_node &root, const std::string &mark) {
  isl::schedule_node hoist_node;
  root.foreach_descendant_top_down([&hoist_node, &mark](const isl::schedule_node &node) -> bool {
    if (auto mark_node = node.as<isl::schedule_node_mark>()) {
      // ignore nested mark nodes
      if (mark_node.get_id().get_name() == mark) {
        hoist_node = mark_node;
        return false;
      }
    }
    return true;
  });
  return hoist_node;
}

std::unordered_map<std::string, std::string> GetMatmulTensorsName(ScopInfo &scop_info) {
  std::unordered_map<std::string, std::string> tensors;
  if (scop_info.user_config_.GetEnableMatmul()) {
    std::unordered_map<std::string, std::string> matmul_map = scop_info.analysis_result_.GetMatrixMatmulMap();
    for (auto i : matmul_map) {
      if (i.second == MATRIX_C) {
        tensors.emplace(MATRIX_C, i.first);
      } else if (i.second == MATRIX_A) {
        tensors.emplace(MATRIX_A, i.first);
      } else if (i.second == MATRIX_B) {
        tensors.emplace(MATRIX_B, i.first);
      } else if (i.second == MATRIX_ELSE) {
        tensors.emplace(MATRIX_ELSE, i.first);
      }
    }
  }
  return tensors;
}

bool IsTensorAB(const std::string &item, ScopInfo &scop_info) {
  auto tensors = GetMatmulTensorsName(scop_info);
  size_t pos = 0;
  std::string item_tensor_name = item;
  if ((pos = item_tensor_name.find(LOCAL_SUFFIX)) != std::string::npos ||
      (pos = item_tensor_name.find(SHARE_SUFFIX)) != std::string::npos) {
    item_tensor_name = item_tensor_name.erase(pos, item_tensor_name.size() - pos);
  }
  if (item_tensor_name != tensors[MATRIX_A] && item_tensor_name != tensors[MATRIX_B]) {
    return false;
  }
  return true;
}

// Map a thread/block configuration to multiple axis.
std::string SetOneConfigForMulAxis(const isl::schedule_node node, const bool is_promotion, const int orig_total_cfg,
                                   const std::set<int> &axis_pos) {
  auto partial_schedule = node.as<isl::schedule_node_band>().get_partial_schedule();
  auto upa_list = GetUPAList(node, partial_schedule, is_promotion, false);

  std::string new_cfg = "";
  int total_cfg = orig_total_cfg;
  int mapping_dim = static_cast<int>(upa_list.size());
  int config_size = 0;
  for (int i = 0; i < mapping_dim; ++i) {
    if (!axis_pos.empty() && axis_pos.count(mapping_dim - 1 - i) == 0) {
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

  while (config_size < static_cast<int>(axis_pos.size())) {
    new_cfg += (std::to_string(1) + " ");
    ++config_size;
  }
  return new_cfg;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
