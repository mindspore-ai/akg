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

#include "operator_shared_strategy.h"
#include "poly/schedule_tree_util.h"
#include "poly/scop.h"
#include "poly/dma_inject.h"
#include "poly/poly_util.h"
#include <vector>
#include <numeric>

namespace akg {
namespace ir {
namespace poly {

std::set<std::string> OperatorSharedStrategy::GetInitPromotedTensor() {
  auto read_map = scop_info_.StmtReadMap();
  auto write_map = scop_info_.StmtWriteMap();
  std::set<std::string> id_sets;
  std::set<std::string> read_sets;
  std::set<std::string> write_sets;
  for (auto item : read_map) {
    for (auto item_id : item.second) {
      if (read_sets.count(item_id.get_name()) == 0) {
        read_sets.insert(item_id.get_name());
      }
    }
  }
  for (auto item : write_map) {
    for (auto item_id : item.second) {
      if (write_sets.count(item_id.get_name()) == 0) {
        write_sets.insert(item_id.get_name());
      }
    }
  }
  /*********************************************************
   * manage only read tensors to share memory
   * for read and write tensor, should be managed to local memory
   ********************************************************/
  std::set_difference(read_sets.begin(), read_sets.end(), write_sets.begin(), write_sets.end(),
                      std::inserter(id_sets, id_sets.begin()));
  return id_sets;
}

void OperatorSharedStrategy::RecordPromotedTensorInfo(const isl::schedule_node &node, const isl::union_map &outer_sch,
                                                      const std::set<std::string> &id_sets) {
  std::vector<isl::id> tensor_list;
  for (auto item : id_sets) {
    tensor_list.push_back(isl::id(scop_info_.ctx_, item));
  }
  isl::union_map reads = scop_info_.analysis_result_.GetReads();
  isl::union_map writes = scop_info_.analysis_result_.GetWrites();
  isl::union_map copyin = scop_info_.analysis_result_.GetCopyin();
  isl::union_map fake_copyin = scop_info_.analysis_result_.GetFakeCopyin();

  for (const auto &item : tensor_list) {
    isl::id dst_tensor_id = GpuDstId(GpuMemType::SHARED, item);
    std::vector<size_t> buffer_sizes;
    std::vector<std::pair<isl::id, MemType>> data_stream;
    data_stream.push_back(std::make_pair(item, MemType::DDR));
    data_stream.push_back(std::make_pair(item, MemType::SHARED_));
    BufferDefInfo promoted_info = BufferDefInfo{item,
                                                dst_tensor_id,
                                                item,
                                                MemType::DDR,
                                                "",
                                                false,
                                                false,
                                                data_stream,
                                                Tensor(),
                                                Handle(),
                                                buffer_sizes,
                                                nullptr,
                                                isl::union_map::empty(isl::space(scop_info_.ctx_, 0))};
    promoted_info.footprints_cluster =
      TensorFootprintCluster::HoistBufferFootprintCluster(outer_sch, item, reads, copyin, writes, fake_copyin);
    if (promoted_info.footprints_cluster != nullptr) {
      promoted_info.footprint_cluster_map.emplace_back(std::make_pair(node, promoted_info.footprints_cluster));
      scop_info_.analysis_result_.buffer_def_infos_.push_back(promoted_info);
    }
  }
}

void OperatorSharedStrategy::RecordCustomPromotedTensors(std::set<std::string> &id_sets) {
  if (scop_info_.user_config_.GetSharedTensors().empty()) {
    return;
  }
  std::vector<std::string> configed_tensors = Split(scop_info_.user_config_.GetSharedTensors(), " ");
  for (const auto &item : configed_tensors) {
    if (id_sets.count(item) == 0) {
      id_sets.emplace(item);
    }
  }
}

void OperatorSharedStrategy::CreateClusterList(const isl::schedule_node &node, const isl::union_map &outer_sch) {
  std::set<std::string> id_sets = GetInitPromotedTensor();
  RecordCustomPromotedTensors(id_sets);
  RecordPromotedTensorInfo(node, outer_sch, id_sets);
}

void ReduceSharedStrategy::CreateClusterList(const isl::schedule_node &node, const isl::union_map &outer_sch) {
  std::set<std::string> id_sets = AnalysisReduceTensors();
  RecordCustomPromotedTensors(id_sets);
  RecordPromotedTensorInfo(node, outer_sch, id_sets);
}

std::set<std::string> ReduceSharedStrategy::AnalysisReduceTensors() {
  std::set<std::string> id_sets;
  /*************************************************
   * In order to enable cuda atomic operator, add
   * these tensors for shared memory promotion list
   *************************************************/
  auto atomic_tensors = scop_info_.analysis_result_.GetAtomicTensors();
  if (!atomic_tensors.empty()) {
    for (const auto &item : atomic_tensors) {
      if (id_sets.count(item.tensor_name) == 0) {
        id_sets.emplace(item.tensor_name);
      }
    }
  }

  /***********************************************
   * For the condition that it is without cuda
   * atomic usage, but with reduce operation.
   * Also need to add these tensors for shared memory
   * promotion list.
   *********************************************/
  auto reduce_out_tensors = scop_info_.analysis_result_.GetReduceTensorInfoMap();
  for (const auto &item : reduce_out_tensors) {
    if (id_sets.count(item.second.write_tensor_name) == 0) {
      id_sets.emplace(item.second.write_tensor_name);
    }
  }

  return id_sets;
}

void BatchMatmulSharedStrategy::CreateClusterList(const isl::schedule_node &node, const isl::union_map &outer_sch,
                                                  const bool hoist_tensor_c) {
  std::set<std::string> id_sets = GetInitPromotedTensor();
  RecordCustomPromotedTensors(id_sets);

  auto tensors = GetMatmulTensorsName(scop_info_);
  if (id_sets.count(tensors[MATRIX_A]) == 0) {
    id_sets.emplace(tensors[MATRIX_A]);
  }
  if (id_sets.count(tensors[MATRIX_B]) == 0) {
    id_sets.emplace(tensors[MATRIX_B]);
  }

  auto it = id_sets.begin();
  while (it != id_sets.end()) {
    if (!hoist_tensor_c) {
      if (!IsTensorAB(*it, scop_info_)) {
        it = id_sets.erase(it);
        continue;
      }
    } else {
      if (IsTensorAB(*it, scop_info_)) {
        it = id_sets.erase(it);
        continue;
      }
    }
    ++it;
  }

  RecordPromotedTensorInfo(node, outer_sch, id_sets);
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
