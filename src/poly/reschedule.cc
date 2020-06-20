/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "poly/reschedule.h"

#include <tvm/ir_visitor.h>
#include <tvm/operation.h>
#include <isl/constraint.h>

#include <climits>
#include <fstream>
#include <queue>
#include <cmath>

#include "poly/dump_log.h"

namespace akg {
namespace ir {
namespace poly {

bool Transform::IsL1OrUbMark(const isl::schedule_node &node) {
  if (node.isa<isl::schedule_node_mark>()) {
    auto tag = node.as<isl::schedule_node_mark>().get_id().get_name();
    if (tag == REALIZE_L1 || tag == REALIZE_UB) return true;
  }
  return false;
}

bool Transform::IsL0OrUbL0Mark(const isl::schedule_node &node) {
  if (node.isa<isl::schedule_node_mark>()) {
    auto tag = node.as<isl::schedule_node_mark>().get_id().get_name();
    if (tag == REALIZE_L0 || tag == REALIZE_UBL0) return true;
  }
  return false;
}

/* Collect tile band data.
 *
 * The input node may either be an L1/UB tile band or an L0 tile band.
 *
 * First check whether "node" is a band node, return if not. Then set "l0_tiled"
 * if "node" is marked by "realize_L0" or "realize_UBL0". Save the ast build
 * options to "l0_build_options_" since we need to retrieve it after building
 * the whole schedule.
 */
void Transform::CollectTileBandData(const isl::schedule_node &node, struct TileBandData *tile_band_data) {
  CHECK(node.isa<isl::schedule_node_band>()) << "has to be a band node" << std::endl;

  tile_band_data->l0_tiled = false;
  tile_band_data->mark = node.parent();
  tile_band_data->ast_build_options = node.as<isl::schedule_node_band>().get_ast_build_options();

  if (tile_band_data->mark.isa<isl::schedule_node_mark>()) {
    auto marktag = tile_band_data->mark.as<isl::schedule_node_mark>().get_id().get_name();
    if (marktag == REALIZE_L0 || marktag == REALIZE_UBL0) {
      tile_band_data->l0_tiled = true;
      l0_build_options_.push_back(tile_band_data->ast_build_options);
    } else if (marktag == REALIZE_L1 || marktag == REALIZE_UB) {
      l1_build_options_.push_back(tile_band_data->ast_build_options);
    }
    tile_band_data->gemm_mark = node.parent().parent();
  }

  tile_band_data->n_member = node.as<isl::schedule_node_band>().n_member();
  tile_band_data->mupa = node.as<isl::schedule_node_band>().get_partial_schedule();
  tile_band_data->permutable = node.as<isl::schedule_node_band>().get_permutable();
  for (int i = 0; i < static_cast<int>(tile_band_data->n_member); ++i)
    tile_band_data->coincident.push_back(node.as<isl::schedule_node_band>().member_get_coincident(i));
}

/* Retrieve tile band data for "node". In particular, the ast build
 * options could be retrieved directly when "node" is an L1/UB tile
 * band, since the schedule tree is not anchored.
 */
isl::schedule_node Transform::RetrieveTileBandData(isl::schedule_node node, struct TileBandData *tile_band_data) {
  node = node.insert_partial_schedule(tile_band_data->mupa);
  CHECK(node.isa<isl::schedule_node_band>()) << "node has to be a band node" << std::endl;
  node = node.as<isl::schedule_node_band>().set_permutable(static_cast<int>(tile_band_data->permutable));
  for (int i = 0; i < static_cast<int>(tile_band_data->n_member); ++i)
    node = node.as<isl::schedule_node_band>().member_set_coincident(i, static_cast<int>(tile_band_data->coincident[i]));

  if (tile_band_data->mark.isa<isl::schedule_node_mark>()) {
    auto marktag = tile_band_data->mark.as<isl::schedule_node_mark>().get_id().get_name();
    node = node.insert_mark(tile_band_data->mark.as<isl::schedule_node_mark>().get_id());
    if (marktag == REALIZE_L0) {
      if (tile_band_data->gemm_mark.isa<isl::schedule_node_mark>()) {
        auto gemmtag = tile_band_data->gemm_mark.as<isl::schedule_node_mark>().get_id().get_name();
        if (gemmtag == CONV_GEMM) {
          node = node.insert_mark(tile_band_data->gemm_mark.as<isl::schedule_node_mark>().get_id());
        }
      }
    }
  }

  return node;
}

isl::schedule_node Transform::RetrieveNodeList(isl::schedule_node node,
                                               const std::vector<isl::schedule_node> &node_list) {
  auto n = static_cast<unsigned int>(node_list.size());
  if (!n) return node;

  for (unsigned int i = n; i >= 1; i--) {
    auto candidate = node_list[i - 1];
    if (candidate.isa<isl::schedule_node_band>()) {
      if (isl_schedule_node_is_subtree_anchored(node.get())) {
        LOG(INFO) << "subtree of the schedule node depends on outer band, cannot insert partial schedule";
        continue;
      }
      auto mupa = candidate.as<isl::schedule_node_band>().get_partial_schedule();
      auto permutable = candidate.as<isl::schedule_node_band>().get_permutable();
      auto num = candidate.as<isl::schedule_node_band>().n_member();
      std::vector<bool> coincident;
      for (int j = 0; j < static_cast<int>(num); ++j) {
        coincident.push_back(candidate.as<isl::schedule_node_band>().member_get_coincident(j));
      }

      node = node.insert_partial_schedule(mupa);
      node = node.as<isl::schedule_node_band>().set_permutable(static_cast<int>(permutable));
      for (int j = 0; j < static_cast<int>(num); ++j) {
        node = node.as<isl::schedule_node_band>().member_set_coincident(j, static_cast<int>(coincident[j]));
      }
      coincident.clear();
    } else if (candidate.isa<isl::schedule_node_mark>()) {
      auto id = candidate.as<isl::schedule_node_mark>().get_id();
      node = node.insert_mark(id);
    } else if (candidate.isa<isl::schedule_node_context>()) {
      auto context = candidate.as<isl::schedule_node_context>().get_context();
      node = node.insert_context(context);
    } else if (candidate.isa<isl::schedule_node_guard>()) {
      auto guard = candidate.as<isl::schedule_node_guard>().get_guard();
      node = node.insert_guard(guard);
    } else {
      LOG(WARNING) << "invalid node in node_list!!! " << candidate;
    }
  }

  return node;
}

isl::schedule_node Transform::RetrieveAstBuildOptions(isl::schedule_node node, const isl::union_set &options) {
  node = Transform::GetOuterBand(node);
  if (node.isa<isl::schedule_node_band>()) {
    node = node.as<isl::schedule_node_band>().set_ast_build_options(options);
    return node;
  }
  return node;
}

// Get the order of statement IDs in the leaf filter nodes of a sequence node.
std::vector<isl::id> GetStmtOrderInSequenceNode(const isl::schedule_node &node) {
  std::vector<isl::id> filter_order;
  if (!node.isa<isl::schedule_node_sequence>()) return filter_order;

  auto sequence = node.as<isl::schedule_node_sequence>();
  // check whether all children of the sequence node are point bands (i.e. leaf filter nodes)
  for (int pos = 0; pos < static_cast<int>(sequence.n_children()); ++pos) {
    if (!node.child(pos).isa<isl::schedule_node_filter>()) return filter_order;
    auto filter = node.child(pos).as<isl::schedule_node_filter>();
    // check if filter node is a point band
    if (filter.n_children() >= 2) return filter_order;
    if (filter.n_children() == 1 && filter.first_child().has_children()) return filter_order;

    filter.get_filter().foreach_set([&](const isl::set &set) -> void {
      auto stmt_id = set.get_tuple_id();
      filter_order.push_back(stmt_id);
    });
  }
  return filter_order;
}

// Get the order of statement IDs in the leaf filter nodes of a schedule tree.
std::vector<isl::id> GetStmtTotalOrdering(const isl::schedule_node &node) {
  std::vector<isl::id> stmt_order;
  node.foreach_descendant_top_down([&](const isl::schedule_node &node) -> bool {
    auto filter_order = GetStmtOrderInSequenceNode(node);
    for (const auto &it : filter_order) stmt_order.push_back(it);
    return true;
  });
  return stmt_order;
}

/* Get the order of statement IDs in point bands of each leaf sequence node of a schedule tree.
 * The result represents a vector of sequence nodes, and each sequence node has a vector of statement IDs.
 */
std::vector<std::vector<isl::id>> GetStmtPartialOrdering(const isl::schedule_node &node) {
  std::vector<std::vector<isl::id>> sequence_nodes;
  node.foreach_descendant_top_down([&](const isl::schedule_node &node) -> bool {
    auto filter_order = GetStmtOrderInSequenceNode(node);
    if (!filter_order.empty()) sequence_nodes.push_back(filter_order);
    return true;
  });
  return sequence_nodes;
}

/* Reassign values of the map as a permutation of the keys of the map.
 * The permutation ordering is determined by the ordering of the values (allow duplicates).
 * Example:
 * Input:  1 -> 0, 5 -> 1, 7 -> 1, 3 -> 2, 9 -> 3, 8 -> 4
 * Output: 1 -> 1, 5 -> 3, 7 -> 5, 3 -> 7, 9 -> 8, 8 -> 9
 */
void ConstructNewOrder(std::unordered_map<size_t, size_t> &map) {
  std::set<size_t> key_order;
  std::multimap<size_t, size_t> reverse_map;
  for (const auto &it : map) {
    key_order.insert(it.first);
    reverse_map.insert(std::make_pair(it.second, it.first));
  }
  std::unordered_map<size_t, size_t> new_order;
  auto key_order_it = key_order.begin();
  for (const auto &it : reverse_map) {
    size_t new_key = it.second;
    CHECK(key_order_it != key_order.end());
    size_t old_key = *key_order_it++;
    new_order[new_key] = old_key;
  }
  map = new_order;
}

/* Reorder filters of a sequence/set node.
 * node: must be a sequence or set node.
 * old_to_new_map: map from original child position to new child position.
 * The caller should make sure that there are no duplicate values.
 */
isl::schedule_node ReorderFilters(const isl::schedule_node &node,
                                  const std::unordered_map<size_t, size_t> &old_to_new_map) {
  auto n_children = node.n_children();
  isl_schedule_tree *old_tree = isl_schedule_node_get_tree(node.get());
  CHECK(old_tree != nullptr);
  isl_schedule_tree *new_tree = isl_schedule_node_get_tree(node.get());
  CHECK(new_tree != nullptr);
  for (auto &it : old_to_new_map) {
    auto old_pos = it.first;
    auto new_pos = it.second;
    CHECK(old_pos < n_children);
    CHECK(new_pos < n_children);
    isl_schedule_tree *old_child = isl_schedule_tree_get_child(old_tree, old_pos);
    CHECK(old_child != nullptr);
    new_tree = isl_schedule_tree_replace_child(new_tree, new_pos, old_child);
    CHECK(new_tree != nullptr);
  }
  static_cast<void>(isl_schedule_tree_free(old_tree));
  isl_schedule_node *new_node = isl_schedule_node_graft_tree(node.copy(), new_tree);
  CHECK(new_node != nullptr);
  return isl::manage(new_node);
}

// Restore the order of filter nodes.
isl::schedule_node RestoreOrderOfFilters(const isl::schedule_node &node, const std::vector<isl::id> &order) {
  std::unordered_map<isl::id, size_t, isl::IslIdIslHash> id_to_order_map;
  for (auto i = 0u; i < order.size(); ++i) {
    id_to_order_map[order[i]] = i;
  }
  // map from original child position to new child position
  std::unordered_map<size_t, size_t> node_order_map;
  for (int i = 0; i < static_cast<int>(node.n_children()); ++i) {
    if (!node.get_child(i).isa<isl::schedule_node_filter>()) return node;
    auto filter_node = node.get_child(i).as<isl::schedule_node_filter>();
    filter_node.get_filter().foreach_set([&](const isl::set &set) -> void {
      auto it = id_to_order_map.find(set.get_tuple_id());
      if (it == id_to_order_map.end()) return;
      size_t order = it->second;
      if (node_order_map.count(i) == 0) {
        node_order_map[i] = order;
      } else {
        node_order_map[i] = std::min(node_order_map[i], order);
      }
    });
  }

  ConstructNewOrder(node_order_map);
  return ReorderFilters(node, node_order_map);
}

/* Restore the order of filter nodes after reschedule.
 * "orders" represents a vector of ordering groups, and each group has a vector of statement IDs.
 * Ordering of filter nodes within each group should be restored.
 */
isl::schedule_node RestoreOrderOfSequenceNodes(isl::schedule_node node,
                                               const std::vector<std::vector<isl::id>> &orders) {
  for (const auto &order : orders) {
    node = RestoreOrderOfFilters(node, order);
  }
  return node;
}

bool Transform::ValidateReorderedSchedule(const isl::schedule &new_schedule) {
  auto backup_schedule = schedule_;
  schedule_ = new_schedule;
  isl::union_map new_dependence = ComputeAllDependences();
  bool is_valid = new_dependence.is_subset(dependences_);
  schedule_ = backup_schedule;
  return is_valid;
}

isl::schedule_node Transform::TryRestoreStmtOrder(const isl::schedule_node &node,
                                                  const std::vector<isl::id> &filter_total_order,
                                                  const std::vector<std::vector<isl::id>> &filter_partial_order) {
  if (filter_total_order.empty()) return node;
  if (filter_partial_order.empty()) return node;

  auto reordered_node = RestoreOrderOfFilters(node, filter_total_order);
  if (ValidateReorderedSchedule(reordered_node.get_schedule())) {
    LOG(INFO) << "reschedule: restored total order of point bands in sequence nodes.";
    return reordered_node;
  } else {
    reordered_node = RestoreOrderOfSequenceNodes(node, filter_partial_order);
    if (ValidateReorderedSchedule(reordered_node.get_schedule())) {
      LOG(INFO) << "reschedule: restored partial order of point bands in sequence nodes.";
      return reordered_node;
    }
  }
  LOG(INFO) << "reschedule: dependences changed, do not restore order of point bands.";
  return node;
}

// Loop distribution by serializing sccs
isl::schedule Transform::RescheduleSerializeSccs(const isl::union_set &active_domain, const bool need_dist) {
  auto ctx = constraints_.ctx();
  auto wasSerializingSccs = isl_options_get_schedule_serialize_sccs(ctx.get());
  isl_stat status = isl_options_set_schedule_serialize_sccs(ctx.get(), static_cast<int>(need_dist));
  CHECK(status == isl_stat_ok);
  auto constraints = constraints_.intersect_domain(active_domain);
  auto new_schedule = constraints.compute_schedule();
  status = isl_options_set_schedule_serialize_sccs(ctx.get(), wasSerializingSccs);
  CHECK(status == isl_stat_ok);
  return new_schedule;
}

// Save ordering of filter children, and restore the ordering after reschedule
isl::schedule_node Transform::ReschedulePreserveFilterOrder(const isl::schedule_node &node,
                                                            const isl::union_set &active_domain, const bool need_dist) {
  auto filter_total_order = GetStmtTotalOrdering(node);
  auto filter_partial_order = GetStmtPartialOrdering(node);

  auto new_schedule = RescheduleSerializeSccs(active_domain, need_dist);
  auto new_node = GetOuterBand(new_schedule.get_root());
  // Retrieve point band if a sequence/set node is introduced
  if (IsSequenceOrSet(new_node)) {
    return TryRestoreStmtOrder(new_node, filter_total_order, filter_partial_order);
  } else {
    return new_node;
  }
}

// Save partial schedule, permutable and coincident attrs of a band.
PointBandInfo Transform::SavePointBand(const isl::schedule_node &node) {
  PointBandInfo point_band_info;
  CHECK(node.isa<isl::schedule_node_band>());
  auto band = node.as<isl::schedule_node_band>();
  point_band_info.mupa = band.get_partial_schedule();
  point_band_info.permutable = band.get_permutable();
  point_band_info.n_member = band.n_member();
  for (int k = 0; k < static_cast<int>(point_band_info.n_member); ++k) {
    point_band_info.coincident.push_back(band.member_get_coincident(k));
  }
  return point_band_info;
}

/* Restore saved partial schedule, permutable and coincident attrs of a band.
 * Input must be a band node.
 */
isl::schedule_node Transform::SetPointBandInfo(isl::schedule_node node, const PointBandInfo &point_band_info) {
  node = node.del();
  node = node.insert_partial_schedule(point_band_info.mupa);
  auto n = node.as<isl::schedule_node_band>().n_member();
  node = node.as<isl::schedule_node_band>().set_permutable(static_cast<int>(point_band_info.permutable));
  for (unsigned int j = 0; j < point_band_info.n_member && j < n; ++j) {
    node = node.as<isl::schedule_node_band>().member_set_coincident(static_cast<int>(j),
                                                                    static_cast<int>(point_band_info.coincident[j]));
  }
  return node;
}

/* Restore saved partial schedule, permutable and coincident attrs of each band in the node.
 * Input may be a sequence, set or band node.
 */
isl::schedule_node Transform::RestorePointBandInfo(isl::schedule_node node, const PointBandInfo &point_band_info) {
  // Retrieve point band if a sequence/set node is introduced
  if (IsSequenceOrSet(node)) {
    // Update point band for each scc filter
    for (auto i = 0u; i < node.n_children(); ++i) {
      node = node.get_child(i);
      node = GetOuterBand(node);
      if (node.isa<isl::schedule_node_leaf>()) {
        while (!node.isa<isl::schedule_node_filter>()) node = node.parent();
        node = node.parent();
        continue;
      }
      node = SetPointBandInfo(node, point_band_info);
      node = node.parent().parent();
    }
  } else {
    node = SetPointBandInfo(node, point_band_info);
  }
  return node;
}

/* Reschedule point band with minimal fusion strategy for "root".
 * "root" should be either a domain or filter node.
 *
 * "need_dist" is used to indicate whether reschedule is needed.
 * In particular, only operators bypassing L0, i.e., those vector
 * operators, should be rescheduled. Operators like convolution,
 * multiplication, etc. should not be rescheduled.
 *
 * Rescheduling starts by checking the input node type, followed
 * by a computation of the active domain for the given "root".
 * In particular, the active domain of a filter node should be
 * a subset of the whole schedule.
 *
 * First, try to obtain the outermost band node. It may either
 * be a sequence/set node or a tile band node. If "node" refers
 * to a sequence/set node, reschedule each filter node individually
 * and construct a new schedule via a sequence/set node. If "node"
 * moves to a tile band, record L1/UB tile band and its mark node.
 * They should be retrieved to the generated schedule after
 * rescheduling, together with its permutable, coincident, options,
 * etc.
 *
 * When traversing from "root" to outermost band node, there may
 * be some additional nodes that should be reserved in "node_list".
 * Such nodes may be a band (in the case of node split during
 * tiling), context, guard, mark (not L1/UB/L0/UBL0 mark) node. All
 * these nodes should be retrieved after rescheduling.
 *
 * Then move down to the child node of L1/UB tile band. As
 * convolution operator group and vector operator group branch
 * into different buffers, the child node may be a sequence/set
 * node.
 *
 * Reschedule each filter node of a given sequence/set node and
 * construct a schedule with a sequence/set node by combining
 * all schedules of the filter nodes. For the ops in the conv
 * group, we reschedule it by maximizing fusion; for those ops
 * in vector group, each operator should be distributed. Such
 * groups are differentiated by checking the target local buffer,
 * i.e., L0 (for conv group) or UBL0 (for vector group).
 *
 * L0 tile band may be reached by moving down from either a L1/UB
 * tile band or a filter node. Record L1/UB tile band and its mark
 * node. They should also be retrieved to the generated schedule
 * after rescheduling, together with its permutable, coincident,
 * options, etc.
 *
 * L0 tile may not happen when the input ops are not convolution-like
 * ops. In such cases, one may reach point band directly from L1
 * tile band.
 *
 * Point band may be reached by moving down from L0 tile band.
 * Record point band information for later reclaiming. Again,
 * permutable, coincident, options, etc. should all be recovered.
 *
 * Try to reschedule the point band by serializing all sccs in
 * the active domain when "need_dist" is true, with schedule
 * constraints updated by intersecting with the active domain. The
 * scheduling options should be first recorded and then recovered for
 * the consistency along tile bands.
 *
 * Retrieve the original point band by intersecting each filter
 * of the generated schedule. The original L0 tile band may also
 * be retrieved after updating the introduced sequence/set node.
 * Also, the L0 tile mark node should also be recovered if any.
 *
 * The L1/UB tile band and its mark node should be added to the
 * generated schedule. "L1_tile_mupa" and "L1_mark" would be used
 * to record L0 tile information when given a filter node.
 *
 * The saved "node_list" may be retrieved to the new schedule tree
 * if any.
 *
 * Finally, the L1/UB AST build options may be introduced to the
 * generated schedule, since one or more nodes in "node_list" may
 * govern the L1/UB tile bands and/or L0/UBL0 tile bands. One may
 * come across with anchored subtrees if the options were introduced
 * before retrieving nodes in "node_list".
 *
 * Return the root of the schedule after rescheduling.
 */
isl::schedule_node Transform::RescheduleSchTree(const isl::schedule_node &root) {
  bool need_dist = true;
  // Return "root" if given an inappropriate node
  if (!root.isa<isl::schedule_node_domain>() && !root.isa<isl::schedule_node_filter>()) return root;

  // Compute the active domain
  auto active_domain = root.isa<isl::schedule_node_domain>() ? root.as<isl::schedule_node_domain>().get_domain()
                                                             : root.as<isl::schedule_node_filter>().get_filter();

  // Save L1/UB band and mark node
  auto node = GetOuterBand(root);
  // Save all nodes along the path from root to L1/UB
  if (!IsL1OrUbMark(node.parent()) && !IsL0OrUbL0Mark(node.parent())) {
    node = root.get_child(0);
    while (!IsL1OrUbMark(node) && !IsL0OrUbL0Mark(node) && !IsSequenceOrSet(node) &&
           !node.isa<isl::schedule_node_leaf>()) {
      node_list_0_.push_back(node);
      node = node.get_child(0);
    }
    if (IsL1OrUbMark(node) || IsL0OrUbL0Mark(node)) node = node.get_child(0);
  }

  // Construct the schedule recursively
  // when encountered a sequence/set node
  if (IsSequenceOrSet(node)) {
    // "schedule" is used to combine the schedules of all filters
    isl::schedule schedule;
    for (auto i = 0u; i < node.n_children(); ++i) {
      auto child = node.get_child(i);
      child = RescheduleSchTree(child);
      if (!child.isa<isl::schedule_node_domain>()) return root;
      if (i == 0) {
        schedule = child.get_schedule();
      } else {
        if (node.isa<isl::schedule_node_sequence>()) {
          schedule = schedule.sequence(child.get_schedule());
        } else {
          schedule = schedule.set(child.get_schedule());
        }
      }
    }
    node = GetOuterBand(schedule.get_root());
    // insert the original L1/UB band and its mark
    node = RetrieveNodeList(node, node_list_0_);

    // retrieve ast build options for each filter
    // The ast build options of L0/UBL0 have to be retrieved
    // after building the whole schedule tree, since it may
    // introduce an anchored subtree they were retrieved
    // before constructing schedule by sequence/set.
    node = GetOuterBand(node);
    if (IsSequenceOrSet(node)) {
      for (unsigned int i = 0; i < static_cast<unsigned int>(node.n_children()) && i < l1_build_options_.size(); ++i) {
        node = GetOuterBand(node.get_child(static_cast<int>(i)));
        if (node.as<isl::schedule_node_band>()) {
          node = node.as<isl::schedule_node_band>().set_ast_build_options(l1_build_options_[i]);
        }
        node = node.parent();
        while (!node.isa<isl::schedule_node_filter>()) node = node.parent();
        node = node.parent();
      }
    }

    return node.get_schedule().get_root();
  }

  auto scalar_filter = [](isl::schedule_node node) {
    if (!node.isa<isl::schedule_node_filter>()) {
      return false;
    }

    auto filter = node.as<isl::schedule_node_filter>();
    isl::union_set sets = filter.get_filter();
    bool scalar = true;
    sets.foreach_set([&scalar](const isl::set s) -> void {
      if (s.n_dim() > 0) {
        scalar = false;
      }
    });
    return scalar;
  };

  if (node.isa<isl::schedule_node_leaf>() && scalar_filter(node.parent())) {
    std::vector<isl::schedule_node> node_list_temp;
    auto temp_node = node.parent();
    while (!temp_node.is_equal(root)) {
      node_list_temp.push_back(temp_node);
      temp_node = temp_node.parent();
    }
    std::reverse(node_list_temp.begin(), node_list_temp.end());
    node = ReschedulePreserveFilterOrder(node, active_domain, need_dist);
    node = RetrieveNodeList(node, node_list_temp);
    return node.get_schedule().get_root();
  }

  if (!node.isa<isl::schedule_node_band>()) return root;

  struct TileBandData L1_Tile_Data;
  CollectTileBandData(node, &L1_Tile_Data);

  if (root.isa<isl::schedule_node_filter>()) {
    if (IsL0OrUbL0Mark(L1_Tile_Data.mark)) {
      auto L1tag = L1_Tile_Data.mark.as<isl::schedule_node_mark>().get_id().get_name();
      if (L1tag == REALIZE_L0) {
        need_dist = false;
      }
    }
  }

  // Move down to the child of L1/UB band and save all nodes along
  node = node.get_child(0);
  while (!node.isa<isl::schedule_node_band>() && !node.isa<isl::schedule_node_leaf>() && !IsL0OrUbL0Mark(node) &&
         !IsSequenceOrSet(node)) {
    node_list_1_.push_back(node);
    node = node.get_child(0);
  }
  if (IsL0OrUbL0Mark(node)) node = node.get_child(0);
  // Construct the schedule recursively
  // when encountered a sequence/set node
  if (IsSequenceOrSet(node)) {
    // "schedule" is used to combine the schedules of all filters
    isl::schedule schedule;
    for (auto i = 0u; i < node.n_children(); ++i) {
      auto child = node.get_child(i);
      child = RescheduleSchTree(child);
      if (!child.isa<isl::schedule_node_domain>()) return root;
      if (i == 0) {
        schedule = child.get_schedule();
      } else {
        if (node.isa<isl::schedule_node_sequence>()) {
          schedule = schedule.sequence(child.get_schedule());
        } else {
          schedule = schedule.set(child.get_schedule());
        }
      }
    }
    node = GetOuterBand(schedule.get_root());

    // retrieve all nodes from L1/UB to L0/UBL0
    node = RetrieveNodeList(node, node_list_1_);

    // insert the original L1/UB band and its mark
    node = RetrieveTileBandData(node, &L1_Tile_Data);

    // set ast build options
    node = RetrieveAstBuildOptions(node, l1_build_options_[0]);

    // retrieve ast build options for each filter
    // The ast build options of L0/UBL0 have to be retrieved
    // after building the whole schedule tree, since it may
    // introduce an anchored subtree they were retrieved
    // before constructing schedule by sequence/set.
    node = GetOuterBand(node).get_child(0);
    if (IsSequenceOrSet(node)) {
      for (unsigned int i = 0; i < node.n_children() && i < static_cast<unsigned int>(l0_build_options_.size()); ++i) {
        node = GetOuterBand(node.get_child(i));
        node =
          node.as<isl::schedule_node_band>().set_ast_build_options(l0_build_options_[static_cast<unsigned int>(i)]);
        node = node.parent();
        while (!node.isa<isl::schedule_node_filter>()) node = node.parent();
        node = node.parent();
      }
    }

    return node.get_schedule().get_root();
  }

  if (node.isa<isl::schedule_node_leaf>()) {
    std::vector<isl::schedule_node> node_list_temp;
    auto temp_node = node.parent();
    while (!temp_node.is_equal(root)) {
      node_list_temp.push_back(temp_node);
      temp_node = temp_node.parent();
    }
    std::reverse(node_list_temp.begin(), node_list_temp.end());
    node = ReschedulePreserveFilterOrder(node, active_domain, need_dist);
    node = RetrieveNodeList(node, node_list_temp);
    return node.get_schedule().get_root();
  }

  if (!node.isa<isl::schedule_node_band>()) return root;

  // Save L0 band and mark node, if any
  // "l0_tiled" is used to check L0 tiled or not
  struct TileBandData L0_Tile_Data;
  CollectTileBandData(node, &L0_Tile_Data);

  // Move down to point band if L0 tiled
  if (L0_Tile_Data.l0_tiled) {
    auto L0tag = L0_Tile_Data.mark.as<isl::schedule_node_mark>().get_id().get_name();
    if (L0tag == REALIZE_L0) {
      return root;
    }
    // Move down to the child of L0/UBL0 band
    // and save all nodes along
    node = node.get_child(0);
    while (!node.isa<isl::schedule_node_band>() && !IsSequenceOrSet(node) && !node.isa<isl::schedule_node_leaf>()) {
      node_list_2_.push_back(node);
      node = node.get_child(0);
    }
    if (!node.isa<isl::schedule_node_band>()) {
      if (IsSequenceOrSet(node)) {
        LOG(WARNING) << "reschedule of sequence/set node under L0/UBL0 is still ongoing!";
      }
      return root;
    }
  }

  // Save point band
  auto point_band_info = SavePointBand(node);

  // core operation of reschedule
  node = ReschedulePreserveFilterOrder(node, active_domain, need_dist);

  node = RestorePointBandInfo(node, point_band_info);

  // Retrieve L0 tile band and mark node if L0 tiled
  if (L0_Tile_Data.l0_tiled) {
    node = RetrieveNodeList(node, node_list_2_);
    node = RetrieveTileBandData(node, &L0_Tile_Data);
  }

  // retrieve all nodes from L1/UB to L0/UBL0
  node = RetrieveNodeList(node, node_list_1_);

  // Retrieve L1/UB tile band and its mark
  node = RetrieveTileBandData(node, &L1_Tile_Data);

  // Retrieve all saved nodes along the path to L1/UB band, if any
  node = RetrieveNodeList(node, node_list_0_);

  // Reset ast build options
  while (!IsL1OrUbMark(node) && !IsL0OrUbL0Mark(node) && !IsSequenceOrSet(node) &&
         !node.isa<isl::schedule_node_leaf>()) {
    node = node.get_child(0);
  }
  if (IsL1OrUbMark(node)) node = RetrieveAstBuildOptions(node, l1_build_options_[0]);
  if (IsSequenceOrSet(node)) {
    for (unsigned int i = 0; i < static_cast<unsigned int>(node.n_children()) && i < l1_build_options_.size(); ++i) {
      node = node.get_child(static_cast<int>(i));
      node = RetrieveAstBuildOptions(node, l1_build_options_[i]);
      while (!node.isa<isl::schedule_node_filter>()) node = node.parent();
      node = node.parent();
    }
  }
  return node.get_schedule().get_root();
}

static isl::schedule_node IslScheduleNodeReplaceChild(const isl::schedule_node &old_node, int pos,
                                                      const isl::schedule_node &child_node) {
  auto tree = isl_schedule_node_get_tree(old_node.get());
  CHECK(tree != nullptr);
  auto new_subtree = isl_schedule_node_get_tree(child_node.get());
  CHECK(new_subtree != nullptr);
  auto new_tree = isl_schedule_tree_replace_child(tree, pos, new_subtree);
  CHECK(new_tree != nullptr);
  auto new_node = isl_schedule_node_graft_tree(old_node.copy(), new_tree);
  CHECK(new_node != nullptr);
  return isl::manage(new_node);
}

/* Reschedule the subtree of each mark node for loop distribution.
 *
 * Transform::Reschedule assumes the mark nodes are the outer bands.
 * This function do not have the assumption, so it supports tiled inner bands.
 *
 * Assume mark nodes are not nested, so this is only suitable for vector ops.
 */
isl::schedule_node Transform::RescheduleInnerBand(const isl::schedule_node &root) {
  return root.map_descendant_bottom_up([this](const isl::schedule_node &node) -> isl::schedule_node {
    if (!IsL1OrUbMark(node) && !IsL0OrUbL0Mark(node)) return node;

    CHECK_EQ(node.n_children(), 1) << "mark node must have one child";
    auto outer_band = node.first_child();
    CHECK(outer_band.isa<isl::schedule_node_band>()) << "the child of mark node must be a band node";
    auto inner_band = outer_band.first_child();
    CHECK(inner_band.isa<isl::schedule_node_band>()) << "the mark node must be tiled to outer and inner bands";

    auto active_domain = inner_band.as<isl::schedule_node_band>().get_domain();
    auto need_dist = true;
    auto point_band_info = SavePointBand(inner_band);

    auto new_schedule = ReschedulePreserveFilterOrder(inner_band, active_domain, need_dist);

    auto new_inner_band = RestorePointBandInfo(GetOuterBand(new_schedule), point_band_info);
    auto new_outer_band = IslScheduleNodeReplaceChild(outer_band, 0, new_inner_band);
    return new_outer_band.parent();
  });
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
