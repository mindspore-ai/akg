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
#include "poly/schedule_tree_util.h"

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

bool SyncManager::IsRepeatSync(const isl::schedule_node orig_node) {
  // Determine whether there are repeated sync.
  auto node = orig_node;
  auto is_repeat_sync = false;
  while (node.has_children()) {
    node = node.child(node.n_children() - 1);
  }

  if (node.has_parent() && node.parent().isa<isl::schedule_node_filter>()) {
    auto filter = node.parent().as<isl::schedule_node_filter>().get_filter();
    filter.foreach_set([&is_repeat_sync](isl::set s) {
      if (s.get_tuple_name().find_first_of(SYNC_PREFIX) == 0) {
        is_repeat_sync = true;
        return;
      }
    });
  }
  return is_repeat_sync;
}

std::string SyncManager::GetCurrentFilterName(const isl::schedule_node orig_node) {
  if (!orig_node.isa<isl::schedule_node_filter>()) {
    return "";
  }

  auto filter_node = orig_node.as<isl::schedule_node_filter>();
  CHECK(filter_node) << "Expected filters below sequence";

  // Transform isl::union_set to a vector of isl::set
  isl::union_set uset = filter_node.get_filter();
  std::vector<isl::set> vset;
  uset.foreach_set([&vset](isl::set s) { vset.push_back(s); });

  // Get current filter name
  std::string cur_filter_name = "";
  if (!vset.empty()) {
    cur_filter_name = vset[0].get_tuple_name();
  }

  return cur_filter_name;
}

isl::schedule SyncManager::InsertPromotionSync(const isl::schedule &sch) {
  auto InsertSyncForSequence = [this](isl::schedule_node node) -> isl::schedule_node {
    if (!node.isa<isl::schedule_node_sequence>()) {
      return node;
    }

    if (!node.has_parent() || !node.parent().isa<isl::schedule_node_extension>()) {
      return node;
    }

    if (node.n_children() <= 1) {
      return node;
    }

    std::unordered_set<std::string> shared_promotion_set = {READ_ID_NAME, WRITE_ID_NAME};
    bool find_read_or_write = false;
    for (int i = node.n_children() - 1; i >= 0; --i) {
      auto filter_node = node.child(i).as<isl::schedule_node_filter>();
      CHECK(filter_node) << "Expected filters below sequence";
      std::string cur_filter_name = GetCurrentFilterName(filter_node);
      if (!cur_filter_name.empty() && shared_promotion_set.find(cur_filter_name) != shared_promotion_set.end()) {
        find_read_or_write = true;
        break;
      }
    }

    if (!find_read_or_write) {
      return node;
    }

    std::string cur_filter_name = "";
    std::string next_filter_name = "";
    for (int i = node.n_children() - 1; i >= 0; --i) {
      auto filter_node = node.child(i).as<isl::schedule_node_filter>();
      cur_filter_name = GetCurrentFilterName(filter_node);

      // When the current filter and the next filter are the same, do not insert synchronization.
      if (cur_filter_name == next_filter_name) {
        continue;
      }

      // When the current filter and the next filter are shared_read and shared_write at the same time, do not insert
      // synchronization.
      if (shared_promotion_set.find(cur_filter_name) != shared_promotion_set.end() &&
          shared_promotion_set.find(next_filter_name) != shared_promotion_set.end()) {
        continue;
      }

      bool is_continue = false;
      // When the current filter and the next filter have nothing to do with shared_read and shared_write, do not insert
      // synchronizatio
      if (shared_promotion_set.find(cur_filter_name) == shared_promotion_set.end() &&
          shared_promotion_set.find(next_filter_name) == shared_promotion_set.end()) {
        is_continue = true;
        // When the first filter is related to shared_read and shared_write, insert synchronization
        if (i == static_cast<int>(node.n_children() - 1) &&
            shared_promotion_set.find(GetCurrentFilterName(node.child(0))) != shared_promotion_set.end()) {
          is_continue = false;
        }
      }
      if (is_continue) {
        continue;
      }

      next_filter_name = cur_filter_name;

      if (IsRepeatSync(filter_node)) {
        continue;
      }

      // Insert sync after the filter node
      auto band_node = filter_node.child(0);
      auto sync_id = MakeUniqueId(SyncLevel::BLOCK);
      node = InsertExtensionNodeBeforeOrAfter(band_node, sync_id, false);
      node = node.ancestor(2);
    }
    return node;
  };
  auto final_sch = sch.get_root().map_descendant_bottom_up(InsertSyncForSequence).get_schedule();
  return final_sch;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
