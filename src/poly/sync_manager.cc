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

#include "sync_manager.h"
#include "poly_util.h"
#include "scop_info.h"

namespace akg {
namespace ir {
namespace poly {

isl::id SyncManager::MakeUniqueId(SyncLevel level) {
  if (level == SyncLevel::WARP) {
    // For now, tvm codegen does not support warp sync.
    // So, used shared sync instead of warp sync.
    return GetSyncId();
  } else {
    return GetSyncId();
  }
}

isl::id SyncManager::GetSyncId() const {
  static size_t count = 0;
  auto sync_id = std::string(SYNC_PREFIX) + std::to_string(count++);
  return isl::id(ctx_, sync_id);
}

isl::id SyncManager::GetWarpSyncId() const {
  static size_t count = 0;
  auto sync_id = std::string(WARP_SYNC_PREFIX) + std::to_string(count++);
  return isl::id(ctx_, sync_id);
}

isl::schedule_node SyncManager::InsertExtensionNode(const isl::schedule_node &node, SyncLevel level, bool after) {
  auto space = GetExtensionSpace(node, level);
  isl::schedule_node graft = isl::schedule_node::from_extension(space);
  auto extension_node = node;
  if (after) {
    extension_node = extension_node.graft_after(graft);
  } else {
    extension_node = extension_node.graft_before(graft);
  }
  return extension_node.ancestor(extension_distance_from_original_pos_);
}

isl::map SyncManager::GetExtensionSpace(const isl::schedule_node &node, SyncLevel level) {
  auto sync_id = MakeUniqueId(level);
  auto prefix = ShortScheduleMupa(node.root(), node.parent());
  auto schedule_space = prefix.get_space();
  auto space = schedule_space.params().add_named_tuple_id_ui(sync_id, 0);
  auto extension_space = isl::map::universe(schedule_space.map_from_domain_and_range(space));
  return extension_space;
}

isl::schedule_node SyncManager::InsertPromotionSync(const isl::schedule_node &tree) {
  if (!tree.has_parent() || !tree.parent().has_parent()) {
    return tree;
  }
  auto seq_node = tree.parent().parent();
  if (!seq_node.isa<isl::schedule_node_sequence>()) {
    return tree;
  }

  bool find_read_or_write = false;
  for (int i = seq_node.n_children() - 1; i >= 0; --i) {
    auto filter_node = seq_node.child(i).as<isl::schedule_node_filter>();
    CHECK(filter_node) << "Expected filters below sequence";
    // Transform isl::union_set to a vector of isl::set
    isl::union_set uset = filter_node.get_filter();
    std::vector<isl::set> vset;
    uset.foreach_set([&vset](isl::set s) { vset.push_back(s); });
    if (!vset.empty() && ((vset[0].get_tuple_name() == READ_ID_NAME) || (vset[0].get_tuple_name() == WRITE_ID_NAME))) {
      find_read_or_write = true;
      break;
    }
  }
  if (!find_read_or_write) {
    return tree;
  }

  std::string cur_filter_name = "";
  std::string next_filter_name = "";
  for (int i = seq_node.n_children() - 1; i >= 0; --i) {
    auto filter_node = seq_node.child(i).as<isl::schedule_node_filter>();
    isl::union_set uset = filter_node.get_filter();
    std::vector<isl::set> vset;
    uset.foreach_set([&vset](isl::set s) { vset.push_back(s); });
    // Get current filter name
    if (!vset.empty()) {
      cur_filter_name = vset[0].get_tuple_name();
    }
    // Do not insert sync after the filter node
    if (cur_filter_name == next_filter_name) {
      continue;
    }
    next_filter_name = cur_filter_name;

    if ((cur_filter_name == READ_ID_NAME && next_filter_name == WRITE_ID_NAME) ||
        (cur_filter_name == WRITE_ID_NAME && next_filter_name == READ_ID_NAME)) {
      continue;
    }

    // Insert sync after the filter node
    seq_node = InsertExtensionNode(filter_node.child(0), SyncLevel::BLOCK, true).child(0);
  }

  return seq_node;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
