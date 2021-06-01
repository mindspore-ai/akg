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

#include "realize_manager.h"
#include "poly/poly_util.h"

#include <set>
#include <queue>

namespace akg {
namespace ir {
namespace poly {

isl::id RealizeManager::GetRealizeId(const isl::schedule_node &node, std::string tensor_name) const {
  auto realize_id = std::string(REALIZE_PREFIX) + tensor_name;
  return isl::id(node.ctx(), realize_id);
}

isl::schedule_node RealizeManager::InsertExtensionNodeBefore(const isl::schedule_node &node,
                                                             const std::string tensor_name) {
  auto space = GetExtensionSpace(node, tensor_name);
  isl::schedule_node graft = isl::schedule_node::from_extension(space);
  auto extension_node = node;
  extension_node = extension_node.graft_before(graft);
  return extension_node;
}

isl::map RealizeManager::GetExtensionSpace(const isl::schedule_node &node, const std::string tensor_name) {
  auto realize_id = GetRealizeId(node, tensor_name);
  auto prefix = ShortScheduleMupa(node.root(), node.parent());
  auto schedule_space = prefix.get_space();
  auto space = schedule_space.params().add_named_tuple_id_ui(realize_id, 0);
  auto extension_space = isl::map::universe(schedule_space.map_from_domain_and_range(space));
  return extension_space;
}

isl::schedule_node RealizeManager::BreadthFirstTopDown(const isl::schedule_node &root, bool &end) {
  std::queue<isl::schedule_node> bfs_queue;
  bfs_queue.push(root);
  std::unordered_set<std::string> promotion_read_set = {READ_ID_NAME, SHARED_READ_ID_NAME, GML_READ_ID_NAME};
  std::unordered_set<std::string> promotion_write_set = {WRITE_ID_NAME, SHARED_WRITE_ID_NAME, GML_WRITE_ID_NAME};

  isl::schedule_node top;
  while (!bfs_queue.empty()) {
    // Pop
    top = bfs_queue.front();
    bfs_queue.pop();
    // Push top children into the queue
    for (size_t i = 0; i < top.n_children(); ++i) {
      bfs_queue.push(top.child(i));
    }
    // Process top node
    if (!top.isa<isl::schedule_node_filter>()) {
      continue;
    }
    auto filter_node = top.as<isl::schedule_node_filter>();
    std::string filter_name = GetFilterName(filter_node);
    if (promotion_read_set.find(filter_name) == promotion_read_set.end() &&
        promotion_write_set.find(filter_name) == promotion_write_set.end()) {
      continue;
    }
    std::string tensor_name = GetTensorName(filter_node);
    if (names_set_.count(tensor_name)) {
      continue;
    }
    // Insert realize node for read node
    if (promotion_read_set.find(filter_name) != promotion_read_set.end()) {
      top = InsertExtensionNodeBefore(top.child(0), tensor_name).parent();
      names_set_.insert(tensor_name);
      break;
    }
    // Insert realize node for write node
    if (promotion_write_set.find(filter_name) != promotion_write_set.end()) {
      size_t i = 0;
      auto top_parent = top.parent();
      for (; i < top_parent.n_children(); ++i) {
        auto tmp_name = GetFilterName(top_parent.child(i).as<isl::schedule_node_filter>());
        if (promotion_read_set.find(tmp_name) == promotion_read_set.end() &&
            promotion_write_set.find(tmp_name) == promotion_write_set.end()) {
          break;
        }
      }
      top = InsertExtensionNodeBefore(top_parent.child(i).child(0), tensor_name).parent();
      names_set_.insert(tensor_name);
      break;
    }
  }
  if (bfs_queue.empty()) {
    end = true;
  }
  auto res_root = top.root();
  return res_root;
}

std::string RealizeManager::GetFilterName(const isl::schedule_node_filter &filter_node) {
  std::string filter_name = "";
  if (filter_node) {
    isl::union_set uset = filter_node.get_filter();
    std::vector<isl::set> vset;
    uset.foreach_set([&vset](isl::set s) { vset.push_back(s); });
    if (!vset.empty()) {
      filter_name = vset[0].get_tuple_name();
    }
  }
  return filter_name;
}

std::string RealizeManager::GetTensorName(const isl::schedule_node_filter &filter_node) {
  std::string tensor_name = "";
  if (filter_node) {
    isl::union_set uset = filter_node.get_filter();
    std::vector<isl::set> vset;
    uset.foreach_set([&vset](isl::set s) { vset.push_back(s); });
    if (!vset.empty()) {
      tensor_name = vset[0].unwrap().get_tuple_id(isl_dim_out).get_name();
    }
  }
  return tensor_name;
}

isl::schedule_node RealizeManager::InsertRealize(const isl::schedule_node &root) {
  if (!root.isa<isl::schedule_node_domain>()) {
    LOG(FATAL) << "Root node should be domain: " << root;
    return root;
  }
  auto res_root = root;
  bool end = false;
  while (!end) {
    res_root = BreadthFirstTopDown(res_root, end);
  }

  return res_root;
}

isl::schedule RealizeManager::Run(isl::schedule sch) {
  sch = scop_info_.sync_manager_.InsertPromotionSync(sch);
  auto root = sch.get_root();
  auto res_root = InsertRealize(root);
  names_set_.clear();
  return res_root.get_schedule();
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
