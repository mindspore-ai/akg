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
#include "poly/schedule_tree_util.h"
#include "poly/dma_inject.h"
#include "poly/schedule_pass.h"
#include "isl_schedule_node_private.h"
#include <numeric>

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

// Insert the relevant marker on the boosted band node.
isl::schedule_node InsertMarkerForPromotedNode(
  const isl::schedule_node &orig_node, const std::unordered_map<std::string, PromoteMarkerInfo> &filter_marker_map) {
  auto GetPromotedFilter = [filter_marker_map](isl::schedule_node node) -> isl::schedule_node {
    if (!node.isa<isl::schedule_node_filter>()) {
      return node;
    }
    isl::union_set uset = node.as<isl::schedule_node_filter>().get_filter();
    std::string filter_name = "";
    uset.foreach_set([&filter_name, filter_marker_map](isl::set s) {
      std::string set_name = s.get_tuple_name();
      if (filter_marker_map.count(set_name) != 0) {
        filter_name = set_name;
      }
    });
    if (filter_name.empty()) {
      return node;
    }
    auto child_node = node.child(0);
    if (!child_node.isa<isl::schedule_node_band>()) {
      return node;
    }

    auto band_node = child_node.as<isl::schedule_node_band>();
    int n_member = band_node.n_member();
    if (n_member == 0) {
      return node;
    }

    size_t start_depth = node.get_tree_depth();
    PromoteMarkerInfo marker_info = filter_marker_map.at(filter_name);
    int aixs_pos = marker_info.axis_pos;
    auto marker_names = marker_info.markers;

    // aixs_pos: Indicates that the marker is inserted before the i-th axis, starting from 1.
    CHECK(std::abs(aixs_pos) <= n_member && aixs_pos != 0)
      << "The position of the inserted axis: " << std::abs(aixs_pos)
      << " cannot be greater than the total number of axes of the current band node: " << n_member << ".";
    int current_aixs_pos = aixs_pos - 1;
    if (aixs_pos < 0) {
      current_aixs_pos = n_member + aixs_pos;
    }
    current_aixs_pos -= (static_cast<int>(marker_names.size()) - 1);

    node = (current_aixs_pos > 0) ? band_node.split(current_aixs_pos).child(0) : child_node;
    for (auto marker_name : marker_names) {
      auto cur_node = node.as<isl::schedule_node_band>();
      int band_number = cur_node.n_member();
      if (band_number > 1) {
        node = cur_node.split(band_number - 1).child(0);
      }
      node = node.insert_mark(marker_name).parent();

      if (band_number == 1) {
        break;
      }
    }
    node = node.ancestor(node.get_tree_depth() - start_depth);
    return node;
  };
  return orig_node.map_descendant_bottom_up(GetPromotedFilter);
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
          upa_list = upa_list.add(mapping.at(thread_id).schedule_upa);
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
      upa_list = upa_list.add(mapping.at(id).schedule_upa);
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
  auto prefix = ShortScheduleMupa(node.root(), node.parent());
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

std::unordered_set<std::string> GetNonRepeatedIdx(const MappingStrategyAxisMap &mapping_strategy) {
  std::unordered_set<std::string> non_repeated_idx;
  for (auto strategy : mapping_strategy) {
    auto mapping_idx = strategy.second.mapping_idx;
    if (non_repeated_idx.find(mapping_idx) != non_repeated_idx.end()) {
      non_repeated_idx.erase(mapping_idx);
    } else {
      non_repeated_idx.emplace(mapping_idx);
    }
  }
  return non_repeated_idx;
}

isl::union_set GetMappingFilterInfo(const isl::schedule_node node, MappingCfg *mapping_cfg,
                                    const std::unordered_map<std::string, MappingCfg *> &replace_cfg,
                                    const std::unordered_set<std::string> &non_repeated_idx) {
  isl::union_set mapping_filter = GatherMappingsTo(node, mapping_cfg, non_repeated_idx);
  for (auto it : replace_cfg) {
    auto cfg = it.second;
    auto type = mapping_cfg->type == MappingType::BLOCKS ? MappingType::REPLACE_BLOCKS : MappingType::REPLACE_THREADS;
    if (cfg->type != type) {
      continue;
    }

    auto tmp_mapping_filter = GatherMappingsTo(node, cfg);
    if (mapping_filter.is_empty()) {
      mapping_filter = tmp_mapping_filter;
    }

    if (!tmp_mapping_filter.is_empty()) {
      mapping_filter = mapping_filter.intersect(tmp_mapping_filter);
    }
  }

  return mapping_filter;
}

isl::union_set GatherMappingsTo(const isl::schedule_node &root, MappingCfg *mapping_cfg,
                                const std::unordered_set<std::string> &non_repeated_idx) {
  auto domain_node = root.as<isl::schedule_node_domain>();
  auto domain = domain_node.domain();
  auto sch = root.get_schedule();
  auto mapping_filters = CollectNode<isl::schedule_node_filter>(sch);

  std::vector<isl::id> filters;
  if (!non_repeated_idx.empty()) {
    for (auto idx : non_repeated_idx) {
      auto id = isl::id(root.ctx(), idx);
      filters.push_back(id);
    }
  } else {
    for (size_t idx = 0; idx < mapping_cfg->bound; ++idx) {
      auto value = mapping_cfg->GetAt(idx);
      auto id = isl::id(root.ctx(), value.first);
      filters.push_back(id);
    }
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

std::vector<isl::schedule_node> CollectMarkNode(const isl::schedule_node &tree, const std::string &mark_tag) {
  std::vector<isl::schedule_node> mark_nodes;
  tree.foreach_descendant_top_down([&mark_nodes, &mark_tag](const isl::schedule_node &node) -> bool {
    if (auto mark_node = node.as<isl::schedule_node_mark>()) {
      // ignore nested mark nodes
      if (mark_node.get_id().get_name() == mark_tag) {
        mark_nodes.push_back(node);
        return false;
      }
    }
    return true;
  });
  return mark_nodes;
}

std::unordered_map<std::string, std::string> GetMatmulTensorsName(ScopInfo &scop_info) {
  std::unordered_map<std::string, std::string> tensors;
  if (!scop_info.user_config_.GetEnableMatmul() && !scop_info.user_config_.GetEnableConv2dDirect()) {
    return tensors;
  }
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
  return tensors;
}

std::string GetTensorMark(const std::string &item, ScopInfo &scop_info) {
  auto tensors = GetMatmulTensorsName(scop_info);
  size_t pos = 0;
  std::string item_tensor_name = item;
  if ((pos = item_tensor_name.find(LOCAL_SUFFIX)) != std::string::npos ||
      (pos = item_tensor_name.find(SHARE_SUFFIX)) != std::string::npos) {
    item_tensor_name = item_tensor_name.erase(pos, item_tensor_name.size() - pos);
  }

  std::string tensor_mark = "";
  if (item_tensor_name == tensors[MATRIX_A]) {
    tensor_mark = TENSOR_A;
  } else if (item_tensor_name == tensors[MATRIX_B]) {
    tensor_mark = TENSOR_B;
  } else if (item_tensor_name == tensors[MATRIX_C]) {
    tensor_mark = TENSOR_C;
  }
  return tensor_mark;
}

isl::schedule_node AdjustAxisPosition(const isl::schedule_node &orig_node, const int orig_pos, const int new_pos) {
  if (!orig_node.isa<isl::schedule_node_band>()) {
    return orig_node;
  }
  auto band_node = orig_node.as<isl::schedule_node_band>();
  int band_node_member = static_cast<int>(band_node.n_member());
  if (orig_pos >= band_node_member || orig_pos < 0 || new_pos >= band_node_member || new_pos < 0 ||
      orig_pos == new_pos) {
    return orig_node;
  }

  int permutable = band_node.get_permutable();
  if (permutable != 1) {
    return orig_node;
  }

  auto partial_schedule = band_node.get_partial_schedule();
  isl::union_pw_aff_list new_upa = isl::union_pw_aff_list();
  isl::union_pw_aff orig_upa = partial_schedule.get_union_pw_aff(orig_pos);
  bool orig_coincident = band_node.member_get_coincident(orig_pos);
  std::vector<bool> coincident;
  int counter = 0;
  // make new union pw aff list
  for (int i = 0; i < band_node_member; ++i) {
    if (orig_pos == i) {
      continue;
    }

    if (counter == new_pos) {
      new_upa = new_upa.is_null() ? isl::union_pw_aff_list(orig_upa) : new_upa.add(orig_upa);
      coincident.push_back(orig_coincident);
      ++counter;
    }

    isl::union_pw_aff upa = partial_schedule.get_union_pw_aff(i);
    new_upa = new_upa.is_null() ? isl::union_pw_aff_list(upa) : new_upa.add(upa);
    coincident.push_back(band_node.member_get_coincident(i));
    ++counter;
  }
  if (counter != band_node_member) {
    isl::union_pw_aff upa = partial_schedule.get_union_pw_aff(orig_pos);
    new_upa = new_upa.add(upa);
    coincident.push_back(band_node.member_get_coincident(orig_pos));
  }

  // make multi_union_pw_aff
  isl::multi_union_pw_aff mupa = isl::multi_union_pw_aff(partial_schedule.get_space(), new_upa);

  // delete old node
  auto node = orig_node;
  node = node.del();
  // insert new node
  node = node.insert_partial_schedule(mupa);
  node = node.as<isl::schedule_node_band>().set_permutable(permutable);
  for (int i = 0; i < band_node_member; ++i) {
    node = node.as<isl::schedule_node_band>().member_set_coincident(i, coincident[i]);
  }
  return node;
}

isl::schedule_node InsertEmptyPermutableBand(const isl::schedule_node &orig_node) {
  auto node = orig_node;
  isl::space space;
  isl::multi_union_pw_aff mupa;

  space = node.get_schedule().get_domain().get_space();

  space = space.set_from_params();
  mupa = isl::multi_union_pw_aff::zero(space);
  node = node.insert_partial_schedule(mupa);
  node = node.as<isl::schedule_node_band>().set_permutable(1);

  return node;
}

int GetVectorizationTileSize(ScopInfo &scop_info) {
  auto vectorized_loop_size = scop_info.analysis_result_.GetVectorizedLoopSize();
  if (vectorized_loop_size != 0) {
    return vectorized_loop_size;
  }

  int vectorized_length = scop_info.user_config_.GetVectorLength();
  if (vectorized_length == 0) {
    if (scop_info.user_config_.GetTarget() == TARGET_CPU) {
      std::string feature = scop_info.user_config_.GetFeature();
      if (feature.empty()) {
        LOG(WARNING) << "The acquired feature is empty and will be set to the default value: sse.";
        feature = SSE_INSTRUCTION_SET;
      }
      auto it = CpuInstructionSetBits.find(feature);
      CHECK(it != CpuInstructionSetBits.end())
        << "The instruction set supported by the cpu only includes sse, avx, avx2, avx512 and neon.";

      vectorized_length = it->second;
    } else {
      vectorized_length = VECTORIZED_128_BIT;
    }
  }

  scop_info.analysis_result_.SetVectorizedLength(vectorized_length);
  vectorized_length = (vectorized_length + ONE_BYTE_TO_BIT - 1) / ONE_BYTE_TO_BIT;
  CHECK(vectorized_length != 0);
  auto reads_access = scop_info.analysis_result_.GetReads().domain_factor_domain();
  auto write_access = scop_info.analysis_result_.GetWrites().domain_factor_domain();
  auto original_access = reads_access.unite(write_access);
  isl::map_list access_list = original_access.get_map_list();
  for (int i = 0; i < static_cast<int>(access_list.size()); ++i) {
    auto access = access_list.at(i);
    auto id = access.get_tuple_id(isl_dim_out).to_str();
    Type type = scop_info.GetDtypeOf(id);
    CHECK_NE(type.bytes(), 0);
    auto tmp_bytes = vectorized_length / type.bytes();
    vectorized_loop_size = (i == 0) ? tmp_bytes : std::min(tmp_bytes, vectorized_loop_size);
  }
  CHECK(vectorized_loop_size != 0);
  scop_info.analysis_result_.SetVectorizedLoopSize(vectorized_loop_size);
  return vectorized_loop_size;
}

/*
 * When mapping size is smaller than the extent of corresponding axis, we may encounter several problems if the axis
 * is not tiled. Firstly, for case that extent is multiplies of mapping sizes, directly mapping the axis will
 * generate a for loop with stride equals to `extent / mapping size`, which is not firently to emit halide ir.
 * Secondly, for case that etxnet is not divisible by mapping size, we need to generate a for loop that has offset
 * with `min` in it to deal with the tail part. This type of for loop can be generated by tiling schedule tree.
 * Therefore, we must check map size and apply tile before mapping.
 */
isl::multi_val CheckAndGetMapSize(const isl::schedule_node &mapping_root, const isl::union_pw_aff_list &aff_list,
                                  MappingStrategyAxisMap &required_mapping_strategy, MappingCfg *mapping_cfg,
                                  const std::vector<int> &additional_tile_size) {
  isl::multi_val mapped_tile_size = isl::multi_val::zero(mapping_root.as<isl::schedule_node_band>().get_space());
  if (required_mapping_strategy.empty()) {
    return mapped_tile_size;
  }

  CHECK(additional_tile_size.size() == 0 || additional_tile_size.size() == aff_list.size());
  std::unordered_map<std::string, int> mapping_cfg_map;
  for (size_t i = 0; i < mapping_cfg->bound; ++i) {
    auto mapping_i = mapping_cfg->GetAt(i);
    mapping_cfg_map[mapping_i.first] = mapping_i.second;
  }

  bool need_tile = false;
  std::vector<int> mapping_sizes;
  std::unordered_set<std::string> non_repeated_idx = GetNonRepeatedIdx(required_mapping_strategy);
  int aff_size = static_cast<int>(aff_list.size());
  for (int i = aff_size - 1; i >= 0; --i) {
    auto aff = aff_list.get_at(i).floor();
    int extent = static_cast<int>(aff.max_val().get_num_si()) + 1;
    int map_size = extent;

    if (required_mapping_strategy.count(static_cast<int>(i)) != 0) {
      std::string mapping_idx = required_mapping_strategy[static_cast<int>(i)].mapping_idx;
      if (static_cast<int>(additional_tile_size.size()) > i) {
        mapping_cfg_map[mapping_idx] *= additional_tile_size[i];
      }
      auto current_mapping_size = mapping_cfg_map[mapping_idx] - required_mapping_strategy[static_cast<int>(i)].offset;
      if (non_repeated_idx.find(mapping_idx) != non_repeated_idx.end()) {
        map_size = current_mapping_size;
      } else {
        map_size = std::min(map_size, current_mapping_size);
        CHECK(map_size != 0);
        mapping_cfg_map[mapping_idx] = mapping_cfg_map[mapping_idx] / map_size;
      }
    } else {
      map_size = 1;
      if (static_cast<int>(additional_tile_size.size()) > i) {
        map_size *= additional_tile_size[i];
      }
    }

    CHECK(map_size != 0);
    mapping_sizes.emplace_back(map_size);

    if (mapping_cfg->type == MappingType::THREADS || mapping_cfg->type == MappingType::REPLACE_THREADS) {
      need_tile = need_tile || extent > map_size;
    } else {
      need_tile = need_tile || (extent > map_size && extent % map_size != 0);
    }
  }

  if (!need_tile) {
    return mapped_tile_size;
  }

  for (int i = 0; i < aff_size; ++i) {
    int pos = aff_size - 1 - i;
    mapped_tile_size = mapped_tile_size.set_val(i, isl::val(mapping_root.ctx(), mapping_sizes[pos]));
  }

  return mapped_tile_size;
}

// get mapping partial schedule
isl::multi_union_pw_aff GetCurrentPartialSchedule(const isl::schedule_node_band &node, const bool is_promotion) {
  auto partial_schedule = node.get_partial_schedule();
  if (is_promotion) {
    // we need to to get range of promoted band from extension node so that we can correctly fix stride
    isl::schedule_node parent = node;
    while (parent.has_parent() && !parent.isa<isl::schedule_node_extension>()) {
      parent = parent.parent();
    }
    if (parent.isa<isl::schedule_node_extension>()) {
      auto extension = parent.as<isl::schedule_node_extension>();
      partial_schedule = partial_schedule.intersect_domain(extension.get_extension().range());
    }
  }

  return partial_schedule;
}

isl::schedule_node GetMarkerNode(const isl::schedule_node &orig_node, const std::string &marker_name) {
  auto node = orig_node;
  orig_node.foreach_descendant_top_down([&node, marker_name](const isl::schedule_node &each_node) -> bool {
    if (!GetMarkerName(each_node, marker_name).empty()) {
      node = each_node;
      return false;
    }
    return true;
  });
  return node;
}

isl::schedule_node DeleUselessMarker(const isl::schedule_node &orig_node,
                                     const std::unordered_set<std::string> &mark_names) {
  auto DeleteMarker = [mark_names](isl::schedule_node node) -> isl::schedule_node {
    if (!node.isa<isl::schedule_node_mark>()) {
      return node;
    }

    auto marker_node = node.as<isl::schedule_node_mark>();
    std::string marker_str = marker_node.get_id().get_name();
    if (mark_names.find(marker_str) != mark_names.end()) {
      return node.del();
    }

    return node;
  };
  return orig_node.map_descendant_bottom_up(DeleteMarker);
}

isl::schedule_node ReplaceMarker(const isl::schedule_node &orig_node, const std::string &orig_name,
                                 const std::string &replaced_name) {
  auto DeleteMarker = [orig_name, replaced_name](isl::schedule_node node) -> isl::schedule_node {
    if (!node.isa<isl::schedule_node_mark>()) {
      return node;
    }

    auto marker_node = node.as<isl::schedule_node_mark>();
    std::string marker_str = marker_node.get_id().get_name();
    if (marker_str == orig_name) {
      node = node.del();
      return node.insert_mark(replaced_name);
    }

    return node;
  };
  return orig_node.map_descendant_bottom_up(DeleteMarker);
}

// Reconstruct the schedule_tree, and insert the following child nodes in turn by recursive method.
// Currently only set, sequence, band, filter, context, mark, leaf nodes are supported.
isl::schedule_node ReConstructScheduleTree(const isl::schedule_node &cur_node, const isl::schedule_node &orig_node,
                                           const isl::schedule_node &exit_node) {
  if (!exit_node.is_null() && orig_node.is_equal(exit_node)) {
    return cur_node;
  }

  auto node = cur_node;
  if (orig_node.isa<isl::schedule_node_set>() || orig_node.isa<isl::schedule_node_sequence>()) {
    return ReConstructSetOrSequenceNode(node, orig_node, exit_node);
  } else if (orig_node.isa<isl::schedule_node_band>()) {
    auto band = orig_node.as<isl::schedule_node_band>();
    node = ReConstructBandNode(node, band);
  } else if (orig_node.isa<isl::schedule_node_filter>()) {
    auto filter = orig_node.as<isl::schedule_node_filter>();
    node = node.insert_filter(filter.filter());
  } else if (orig_node.isa<isl::schedule_node_context>()) {
    auto context = orig_node.as<isl::schedule_node_context>();
    node = node.insert_context(context.context());
  } else if (orig_node.isa<isl::schedule_node_mark>()) {
    auto mark = orig_node.as<isl::schedule_node_mark>();
    node = node.insert_mark(mark.get_id().get_name());
  } else if (orig_node.isa<isl::schedule_node_leaf>()) {
    return node;
  } else {
    LOG(FATAL) << "Currently only set, sequence, band, filter, context, mark, leaf nodes are supported.";
  }
  return ReConstructChildScheduleTree(node, orig_node, exit_node);
}

// Insert the following child nodes in turn by recursive method.
isl::schedule_node ReConstructChildScheduleTree(const isl::schedule_node &cur_node, const isl::schedule_node &orig_node,
                                                const isl::schedule_node &exit_node) {
  if (orig_node.n_children() == 0) {
    return cur_node;
  }
  return ReConstructScheduleTree(cur_node.child(0), orig_node.child(0), exit_node).parent();
}

isl::schedule_node ReConstructSetOrSequenceNode(const isl::schedule_node &cur_node, const isl::schedule_node &orig_node,
                                                const isl::schedule_node &exit_node, const std::vector<size_t> &pos) {
  std::vector<size_t> new_pos(orig_node.n_children());
  if (pos.size() == 0) {
    std::iota(std::begin(new_pos), std::end(new_pos), 0);
  } else {
    new_pos = pos;
  }

  auto filters = isl::union_set_list(cur_node.ctx(), orig_node.n_children());
  for (size_t i = 0; i < new_pos.size(); ++i) {
    auto child_node = orig_node.child(new_pos[i]);
    if (child_node.isa<isl::schedule_node_filter>()) {
      auto filter = child_node.as<isl::schedule_node_filter>();
      filters = filters.add(filter.get_filter());
    }
  }

  auto node = cur_node;
  if (orig_node.isa<isl::schedule_node_sequence>()) {
    node = node.insert_sequence(filters);
  } else {
    node = node.insert_set(filters);
  }

  for (size_t i = 0; i < new_pos.size(); ++i) {
    node = ReConstructChildScheduleTree(node.child(i), orig_node.child(new_pos[i]), exit_node).parent();
  }
  return node;
}

isl::schedule_node ReConstructBandNode(const isl::schedule_node &cur_node, const isl::schedule_node &orig_node) {
  auto orig_band = orig_node.as<isl::schedule_node_band>();
  auto node = cur_node.insert_partial_schedule(orig_band.get_partial_schedule());

  auto band_node = node.as<isl::schedule_node_band>();
  band_node = band_node.set_permutable(orig_band.permutable());
  for (size_t i = 0; i < orig_band.n_member(); ++i) {
    band_node = band_node.member_set_coincident(i, orig_band.member_get_coincident(i));
  }
  return band_node;
}

// Determine whether the current node contains a band node.
bool IsContainBandNode(const isl::schedule_node &orig_node) {
  bool is_include_band_node = false;
  orig_node.foreach_descendant_top_down([&is_include_band_node](const isl::schedule_node &node) -> bool {
    if (node.isa<isl::schedule_node_band>()) {
      is_include_band_node = true;
      return false;
    }
    return true;
  });
  return is_include_band_node;
}

isl::schedule_node InsertMarkerForLoop(const isl::schedule_node &orig_node, const std::string &marker_name,
                                       const bool is_promotion, const int insert_pos) {
  // Insert the corresponding marker at a fixed position of a band node. If the size of the axis is 1, the label is not
  // inserted.
  if (!orig_node.isa<isl::schedule_node_band>() || insert_pos < 0) {
    return orig_node;
  }

  auto band_node = orig_node.as<isl::schedule_node_band>();
  int band_member = static_cast<int>(band_node.n_member());
  CHECK(insert_pos < band_member) << "The split position cannot be greater than the number of axis.";
  auto partial_schedule = GetCurrentPartialSchedule(band_node, is_promotion);
  if (!is_promotion) {
    partial_schedule = partial_schedule.intersect_domain(orig_node.get_domain());
  }
  auto upa_list = partial_schedule.get_union_pw_aff_list();
  auto extent = upa_list.get_at(insert_pos).floor().max_val().get_num_si();
  if (extent < 1) {
    return orig_node;
  }

  auto node = orig_node;
  if (insert_pos != 0) {
    node = band_node.split(insert_pos).child(0);
  }
  return node.insert_mark(marker_name);
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
