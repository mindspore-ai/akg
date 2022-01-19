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

#include "realize_manager.h"
#include "poly/poly_util.h"
#include "poly/schedule_tree_util.h"

#include <set>
#include <queue>

namespace akg {
namespace ir {
namespace poly {
isl::id RealizeManager::GetRealizeId(const isl::schedule_node &node, const std::string &tensor_name) {
  auto realize_id = std::string(REALIZE_PREFIX) + tensor_name;
  return isl::id(node.ctx(), realize_id);
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
      auto child_node = top.child(0);
      auto realize_id = GetRealizeId(child_node, tensor_name);
      top = InsertExtensionNodeBeforeOrAfter(child_node, realize_id, true).parent();
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
      auto child_node = top_parent.child(i).child(0);
      auto realize_id = GetRealizeId(child_node, tensor_name);
      top = InsertExtensionNodeBeforeOrAfter(child_node, realize_id, true).parent();
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

std::string RealizeManager::GetCurrentFilterTenaosrName(const isl::schedule_node &node) {
  auto filter_node = node.as<isl::schedule_node_filter>();
  CHECK(filter_node) << "Expected filters below sequence";
  // Transform isl::union_set to a vector of isl::set
  isl::union_set uset = filter_node.get_filter();
  std::string cur_filter_name = "";
  uset.foreach_set([this, &cur_filter_name](const isl::set &s) -> void {
    std::string node_tensor_name = s.to_str();
    size_t pos = 0;
    if ((pos = node_tensor_name.find(PROMOTION_INFIX)) != std::string::npos ||
        (pos = node_tensor_name.find(LOCAL_SUFFIX)) != std::string::npos) {
      node_tensor_name = node_tensor_name.erase(pos, node_tensor_name.size() - pos);
      if ((pos = node_tensor_name.find_last_of(" ")) != std::string::npos) {
        node_tensor_name = node_tensor_name.erase(0, pos + 1);
      }
    }
    cur_filter_name = node_tensor_name;
  });
  return cur_filter_name;
}

isl::schedule RealizeManager::InsertPromotionMajor(const isl::schedule &sch) {
  if (!scop_info_.user_config_.GetEnableMatmul()) {
    return sch;
  }

  auto InsertMajorForSequence = [this](isl::schedule_node node) -> isl::schedule_node {
    if (!node.isa<isl::schedule_node_sequence>()) {
      return node;
    }

    if (!node.has_parent() || !node.parent().isa<isl::schedule_node_extension>()) {
      return node;
    }

    if (node.n_children() < 1) {
      return node;
    }

    for (size_t i = 0; i < node.n_children(); ++i) {
      auto filter_node = node.child(i).as<isl::schedule_node_filter>();
      if (!IsReadOrWriteTensor(filter_node, READ_ID_NAME, READ_ID_NAME)) {
        continue;
      }
      std::string cur_filter_name = GetCurrentFilterTenaosrName(filter_node);

      std::unordered_map<std::string, std::string> matmul_map = scop_info_.analysis_result_.GetMatrixMatmulMap();
      std::unordered_map<std::string, std::string> matmul_major = scop_info_.analysis_result_.GetMatrixMatmulMajor();
      if (matmul_map.find(cur_filter_name) == matmul_map.end() ||
          matmul_major.find(cur_filter_name) == matmul_major.end()) {
        continue;
      }

      auto matrix_matmul_map = matmul_map[cur_filter_name];
      auto matrix_matmul_major = matmul_major[cur_filter_name];

      auto insert_node = filter_node.child(0);
      insert_node = insert_node.insert_mark(matrix_matmul_major + "_" + matrix_matmul_map);
      node = insert_node.ancestor(2);
    }
    return node;
  };
  auto final_sch = sch.get_root().map_descendant_bottom_up(InsertMajorForSequence).get_schedule();
  return final_sch;
}

isl::schedule RealizeManager::Run(isl::schedule sch) {
  if (scop_info_.user_config_.GetTarget() == TARGET_CUDA) {
    sch = scop_info_.sync_manager_.InsertPromotionSync(sch);
  } else if (scop_info_.user_config_.GetTarget() == TARGET_CPU) {
    sch = InsertPromotionMajor(sch);
  }
  auto root = sch.get_root();
  auto res_root = InsertRealize(root);
  names_set_.clear();
  return res_root.get_schedule();
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
