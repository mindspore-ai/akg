/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "create_cluster.h"
#include "poly/schedule_tree_util.h"
#include "poly/scop.h"
#include "poly/dma_inject.h"
#include "poly/poly_util.h"
#include <vector>

namespace akg {
namespace ir {
namespace poly {
std::set<std::string> CreateCluster::GetAllPromotedTensor() {
  std::set<std::string> all_tensors;
  auto RecordPromotedTensor = [&all_tensors](StmtIdHashMap tensor_map) -> void {
    for (const auto &item : tensor_map) {
      for (const auto &item_id : item.second) {
        all_tensors.emplace(item_id.get_name());
      }
    }
  };

  auto read_map = scop_info_.StmtReadMap();
  auto write_map = scop_info_.StmtWriteMap();
  RecordPromotedTensor(read_map);
  RecordPromotedTensor(write_map);
  return all_tensors;
}

std::set<std::string> CreateCluster::GetTempPromotedTensor(std::set<std::string> all_tensors) {
  auto origin_binds = scop_info_.user_config_.GetOriginBind();
  std::set<std::string> orig_tensors;

  for (const auto &item : origin_binds) {
    if (!item.first.defined()) continue;
    auto id = isl::id(scop_info_.ctx_, item.first->op->name);
    orig_tensors.insert(id.get_name());
  }
  std::set<std::string> temp_tensors;
  std::set_difference(all_tensors.begin(), all_tensors.end(), orig_tensors.begin(), orig_tensors.end(),
                      std::inserter(temp_tensors, temp_tensors.begin()));
  return temp_tensors;
}

void CreateCluster::RecordInitPromotedTensorType(const std::unordered_set<std::string> &configed_tensors) {
  std::set<std::string> all_tensors = GetAllPromotedTensor();
  std::set<std::string> temp_tensors = GetTempPromotedTensor(all_tensors);
  std::unordered_set<std::string> not_promoted_tensors = scop_info_.analysis_result_.GetTensorsNotPromote();

  // According to the current judgment, initialize the promoted type of all tensor.
  for (auto tensor : all_tensors) {
    auto id = isl::id(scop_info_.ctx_, tensor);
    if (configed_tensors.find(tensor) != configed_tensors.end()) {
      all_tensors_[id] = PromotedTensorType::CUSTOM;
    } else if (not_promoted_tensors.find(tensor) != not_promoted_tensors.end()) {
      all_tensors_[id] = PromotedTensorType::NONE;
    } else if (temp_tensors.find(tensor) != temp_tensors.end()) {
      all_tensors_[id] = PromotedTensorType::TEMP;
    } else {
      all_tensors_[id] = PromotedTensorType::OTHERS;
    }
  }
}

std::vector<std::pair<isl::id, PromotedTensorType>> CreateCluster::SortPromotedTensorInfo(
  const PromotedTensor &all_tensors) {
  // Sort the tensor according to the promoted type
  auto Compute = [](std::pair<isl::id, PromotedTensorType> a, std::pair<isl::id, PromotedTensorType> b) -> bool {
    if (a.second == b.second) {
      return a.first.get_name() < b.first.get_name();
    }
    return a.second > b.second;
  };

  std::vector<std::pair<isl::id, PromotedTensorType>> tensor_list;
  for (auto it = all_tensors.begin(); it != all_tensors.end(); it++) {
    // If the current operator does not need to be promoted, it does not need to be sorted.
    if (it->second == PromotedTensorType::NONE) {
      continue;
    }
    tensor_list.push_back(std::pair<isl::id, PromotedTensorType>(it->first, it->second));
  }

  std::sort(tensor_list.begin(), tensor_list.end(), Compute);
  return tensor_list;
}

// Record the final tensor that needs to be promoted.
void CreateCluster::RecordPromotedTensorInfo(const isl::schedule_node &orig_node, const std::string &mark_name,
                                             const PromotedTensor &all_tensors) {
  auto all_tensors_list = SortPromotedTensorInfo(all_tensors);
  if (all_tensors.size() <= 0) {
    return;
  }

  isl::union_map reads = scop_info_.analysis_result_.GetReads();
  isl::union_map writes = scop_info_.analysis_result_.GetWrites();
  isl::union_map copyin = scop_info_.analysis_result_.GetCopyin();
  isl::union_map fake_copyin = scop_info_.analysis_result_.GetFakeCopyin();

  std::vector<isl::schedule_node> nodes = CollectMarkNode(orig_node, mark_name);

  for (const auto &node : nodes) {
    auto tree = node.parent();
    auto partial_sched = GetPartialSchedule(tree);

    for (const auto &tensor : all_tensors_list) {
      auto promoted_id = tensor.first;
      BufferDefInfo promoted_info = GetPromotedInfo(promoted_id, mark_name);

      promoted_info.footprints_cluster = TensorFootprintCluster::HoistBufferFootprintCluster(
        partial_sched, promoted_id, reads, copyin, writes, fake_copyin);
      if (promoted_info.footprints_cluster == nullptr ||
          !CheckPromotion(tree, orig_node, *promoted_info.footprints_cluster, tensor)) {
        continue;
      }

      promoted_info.footprint_cluster_map.emplace_back(std::make_pair(tree, promoted_info.footprints_cluster));
      scop_info_.analysis_result_.buffer_def_infos_.push_back(promoted_info);
    }
  }
}

void CreateCluster::RecordGemmTensors() {
  auto tensors = GetMatmulTensorsName(scop_info_);
  auto RecordPromotedTensor = [this, &tensors](const std::string &matrix_name) -> void {
    if (tensors.count(matrix_name) == 0) {
      return;
    }
    auto id = isl::id(scop_info_.ctx_, tensors[matrix_name]);
    if (all_tensors_.count(id) == 0 || all_tensors_[id] < PromotedTensorType::SPECIAL) {
      all_tensors_[id] = PromotedTensorType::SPECIAL;
    }
  };

  RecordPromotedTensor(MATRIX_A);
  RecordPromotedTensor(MATRIX_B);
  RecordPromotedTensor(MATRIX_C);
}

PromotedTensor CreateCluster::GetCurrentMarkerTensorsForGemm(const std::unordered_set<std::string> &tensor_set) {
  PromotedTensor current_tensors;
  for (auto &tensor : all_tensors_) {
    auto id_name = tensor.first.get_name();
    auto tensor_mark = GetTensorMark(id_name, scop_info_);
    if (tensor_set.count(tensor_mark) != 0) {
      current_tensors.insert(tensor);
    }
  }

  return current_tensors;
}

/*********************************************
 * Shared Create Cluster
 *********************************************/
bool SharedCreateCluster::CoalescingAccessWay(const isl::schedule_node &node, const isl::schedule_node &root,
                                              const TensorFootprintCluster &cluster) {
  isl::union_map original = cluster.OrigianlAccessRelations();
  size_t tensor_dim = cluster.foot_print_.GetBoxDim();
  std::vector<isl::schedule_node> thread_marker = CollectFnNode(IsThreadMappedMark, root);
  for (auto item : thread_marker) {
    if (!(item.isa<isl::schedule_node_mark>()) && !(item.has_children()) &&
        !(item.child(0).isa<isl::schedule_node_filter>())) {
      continue;
    }
    isl::schedule_node thread_filter = item.child(0);
    if (!thread_filter.has_children()) {
      continue;
    }
    isl::schedule_node thread_band = thread_filter.child(0);
    if (!thread_band.has_children()) {
      continue;
    }
    isl::schedule_node inner_band = thread_band.child(0);
    size_t num_mapped_thread = inner_band.schedule_depth() - thread_band.schedule_depth();
    if (num_mapped_thread == 0) {
      continue;
    }
    size_t inner_depth = inner_band.schedule_depth();
    auto active_domains = CollectDomain(thread_band);
    auto local_access = original.intersect_domain(active_domains);
    auto schedule = ShortSchedule(inner_band);
    auto schedule_access = local_access.apply_domain(schedule);
    for (auto access : schedule_access.get_map_list()) {
      if (!IsSubsetForIncreaseDim(access, tensor_dim - 1, inner_depth - 1)) {
        return true;
      }
    }
  }
  return false;
}

// Determine whether the current tensor needs to be promoted.
bool SharedCreateCluster::CheckPromotion(const isl::schedule_node &current_node, const isl::schedule_node &node,
                                         const TensorFootprintCluster &cluster,
                                         const std::pair<isl::id, PromotedTensorType> &tensor_info) {
  if (tensor_info.second > PromotedTensorType::TEMP) {
    return true;
  }

  auto coalesced_access = scop_info_.analysis_result_.GetOuterBandNode(band_index_)->coalesced_access_tensors;
  auto tensor_name = tensor_info.first.get_name();
  if (!CoalescingAccessWay(current_node, node, cluster) &&
      coalesced_access.find(tensor_name) == coalesced_access.end()) {
    return false;
  }
  return true;
}

isl::union_map SharedCreateCluster::GetPartialSchedule(const isl::schedule_node &node) {
  auto root_node = node.root();
  CHECK(!IsAncestorMapToThread(node)) << "shared memory promotion cannot below thread_marker.";
  auto block_cfg = scop_info_.user_config_.GetBlockConfig();
  CHECK(block_cfg != nullptr) << "block config is null";
  auto replace_cfg = scop_info_.user_config_.GetReplaceConfig();
  MappingStrategyAxisMap mapping_strategy = scop_info_.user_config_.GetOuterMappingStrategy(band_index_);
  std::unordered_set<std::string> non_repeated_idx = GetNonRepeatedIdx(mapping_strategy);
  auto mapping_filter_info = GetMappingFilterInfo(root_node, block_cfg, replace_cfg, non_repeated_idx);

  auto partial_sched = LocalSchedule(node);
  if (!mapping_filter_info.is_empty()) {
    partial_sched = partial_sched.intersect_domain(mapping_filter_info);
  }
  return partial_sched;
}

BufferDefInfo SharedCreateCluster::GetPromotedInfo(const isl::id &promoted_id, const std::string &mark_name) {
  GpuMemType gpu_mem_type = GpuMemType::SHARED;
  MemType mem_type = MemType::SHARED_;

  isl::id dst_tensor_id = GetGpuIndexDstId(gpu_mem_type, promoted_id);
  if (scop_info_.IsCopyinTensor(promoted_id.get_name()) && band_index_ != 0) {
    dst_tensor_id = GetGpuIndexDstId(gpu_mem_type, promoted_id, band_index_);
  }
  std::vector<size_t> buffer_sizes;
  std::vector<std::pair<isl::id, MemType>> data_stream;
  data_stream.push_back(std::make_pair(promoted_id, MemType::DDR));
  data_stream.push_back(std::make_pair(promoted_id, mem_type));
  BufferDefInfo promoted_info = BufferDefInfo{promoted_id,
                                              dst_tensor_id,
                                              promoted_id,
                                              MemType::DDR,
                                              mark_name,
                                              false,
                                              false,
                                              data_stream,
                                              Tensor(),
                                              Handle(),
                                              buffer_sizes,
                                              nullptr,
                                              isl::union_map::empty(isl::space(scop_info_.ctx_, 0))};

  return promoted_info;
}

void SharedCreateCluster::CreateClusterListForGemm(const isl::schedule_node &node,
                                                   const std::unordered_set<std::string> &mark_names) {
  auto configed_tensors = scop_info_.user_config_.GetSharedTensors();
  // Initialize the promoted types of all tensors.
  RecordInitPromotedTensorType(configed_tensors);
  // Modify promoted type of tensor A/B/C for gemm operator.
  RecordGemmTensors();

  std::unordered_set<std::string> tensor_set;
  for (const auto &mark_name : mark_names) {
    bool hoist_tensor_c = mark_name == PROMOTE_GLOBAL_TO_SHARED_C;
    tensor_set.clear();
    if (hoist_tensor_c) {
      tensor_set.emplace(TENSOR_C);
    } else {
      tensor_set.emplace(TENSOR_A);
      tensor_set.emplace(TENSOR_B);
    }
    // Promote the specific tensor at the corresponding marker position.
    PromotedTensor current_tensors = GetCurrentMarkerTensorsForGemm(tensor_set);
    RecordPromotedTensorInfo(node, mark_name, current_tensors);
  }
}

void SharedCreateCluster::CreateClusterListForElementWise(const isl::schedule_node &node,
                                                          const std::unordered_set<std::string> &mark_names) {
  auto configed_tensors = scop_info_.user_config_.GetSharedTensors();
  // Initialize the promoted types of all tensors.
  RecordInitPromotedTensorType(configed_tensors);
  for (const auto &mark_name : mark_names) {
    RecordPromotedTensorInfo(node, mark_name, all_tensors_);
  }
}

void SharedCreateCluster::CreateClusterListForPartialElementWise(const isl::schedule_node &node,
                                                                 const std::unordered_set<std::string> &mark_names) {
  auto configed_tensors = scop_info_.user_config_.GetSharedTensors();
  // Initialize the promoted types of all tensors.
  RecordInitPromotedTensorType(configed_tensors);
  // Remove write tensors from promoted tensors
  auto write_map = scop_info_.StmtWriteMap();
  for (auto item : write_map) {
    for (auto item_id : item.second) {
      if (all_tensors_.count(item_id)) {
        all_tensors_.erase(item_id);
      }
    }
  }
  for (const auto &mark_name : mark_names) {
    RecordPromotedTensorInfo(node, mark_name, all_tensors_);
  }
}

void SharedCreateCluster::CreateClusterListForReduce(const isl::schedule_node &node,
                                                     const std::unordered_set<std::string> &mark_names) {
  auto configed_tensors = scop_info_.user_config_.GetSharedTensors();
  // Initialize the promoted types of all tensors.
  RecordInitPromotedTensorType(configed_tensors);
  // Modify promoted type of the returned tensor for reduce operator.
  RecordReduceTensors();

  for (const auto &mark_name : mark_names) {
    RecordPromotedTensorInfo(node, mark_name, all_tensors_);
  }
}

void SharedCreateCluster::RecordReduceTensors() {
  // In order to enable cuda atomic operator, add these tensors for shared memory promotion list
  auto atomic_tensors = scop_info_.analysis_result_.GetAtomicTensors();
  if (!atomic_tensors.empty()) {
    for (const auto &item : atomic_tensors) {
      auto id = isl::id(scop_info_.ctx_, item.tensor_name);
      if (all_tensors_.count(id) == 0 || all_tensors_[id] < PromotedTensorType::SPECIAL) {
        all_tensors_[id] = PromotedTensorType::SPECIAL;
      }
    }
  }

  // For the condition that it is without cuda atomic usage, but with reduce operation.
  // Also need to add these tensors for shared memory promotion list.
  auto reduce_out_tensors = scop_info_.analysis_result_.GetReduceTensorInfoMap();
  for (const auto &item : reduce_out_tensors) {
    auto id = isl::id(scop_info_.ctx_, item.second.write_tensor_name);
    if (all_tensors_.count(id) == 0 || all_tensors_[id] < PromotedTensorType::SPECIAL) {
      all_tensors_[id] = PromotedTensorType::SPECIAL;
    }
  }

  // For the reduce operator, only the return tensor and the temp tensor can be promoted. For ordinary tensor, it will
  // cause an error in the reduce interface after promotion.
  for (auto &tensor : all_tensors_) {
    if (tensor.second == PromotedTensorType::OTHERS) {
      all_tensors_[tensor.first] = PromotedTensorType::NONE;
    }
  }
}

/*********************************************
 * Register Create Cluster
 *********************************************/
isl::union_map RegisterCreateCluster::GetPartialSchedule(const isl::schedule_node &node) {
  auto root_node = node.root();
  auto block_cfg = scop_info_.user_config_.GetBlockConfig();
  CHECK(block_cfg != nullptr) << "block config is null";
  auto replace_cfg = scop_info_.user_config_.GetReplaceConfig();
  MappingStrategyAxisMap mapping_strategy = scop_info_.user_config_.GetOuterMappingStrategy(band_index_);
  std::unordered_set<std::string> non_repeated_idx = GetNonRepeatedIdx(mapping_strategy);
  auto block_mapping = GetMappingFilterInfo(root_node, block_cfg, replace_cfg, non_repeated_idx);

  auto thread_cfg = scop_info_.user_config_.GetThreadConfig();
  CHECK(thread_cfg != nullptr) << "thread config is null";
  auto thread_mapping = isl::union_set::empty(block_mapping.ctx());
  mapping_strategy = scop_info_.user_config_.GetInnerMappingStrategy(band_index_);
  non_repeated_idx = GetNonRepeatedIdx(mapping_strategy);
  thread_mapping = GetMappingFilterInfo(root_node, thread_cfg, replace_cfg, non_repeated_idx);

  auto partial_sched = LocalSchedule(node);
  if (!thread_mapping.is_empty() && !block_mapping.is_empty()) {
    auto mapping = block_mapping.intersect(thread_mapping);
    partial_sched = partial_sched.intersect_domain(mapping);
  } else if (!thread_mapping.is_empty()) {
    partial_sched = partial_sched.intersect_domain(thread_mapping);
  } else if (!block_mapping.is_empty()) {
    partial_sched = partial_sched.intersect_domain(block_mapping);
  }
  return partial_sched;
}

// Check if the given "group" can be promoted to registers for the given mapping to thread identifiers and within the
// given outer schedule.
bool RegisterCreateCluster::IsResueThread(const TensorFootprintCluster &cluster,
                                          const isl::schedule_node &current_node) {
  isl::schedule_node root_node = current_node.get_schedule().get_root();
  auto thread_cfg = scop_info_.user_config_.GetThreadConfig();
  CHECK(thread_cfg != nullptr) << "thread config is null";
  auto block_cfg = scop_info_.user_config_.GetBlockConfig();
  CHECK(block_cfg != nullptr) << "block config is null";

  auto thread_schedule = MapDomainAllWithType(root_node, thread_cfg, scop_info_.upa_node_mapping_, THREAD_MARKER);
  auto block_schedule = MapDomainAllWithType(root_node, block_cfg, scop_info_.upa_node_mapping_, BLOCK_MARKER);
  auto tmp_node = current_node;
  if (current_node.isa<isl::schedule_node_band>()) {
    tmp_node = current_node.child(0);
  }

  auto partial_sched_mupa = ShortScheduleMupa(root_node, tmp_node);
  partial_sched_mupa = partial_sched_mupa.flat_range_product(block_schedule).flat_range_product(thread_schedule);

  // compute the mapping relation between single thread and outer schedule space and tensor elements pair
  isl::union_map state_schedule_mapping = ScheduleTensorMapping(partial_sched_mupa, cluster.OrigianlAccessRelations());
  isl::union_map thread_schedule_mapping = state_schedule_mapping.apply_domain(isl::union_map::from(thread_schedule));
  // check that whether the mapping relation between single thread and outer schedule points and group elements pair
  // is injective.
  return thread_schedule_mapping.is_injective();
}

bool RegisterCreateCluster::IsSatisfyVectorization(const TensorFootprintCluster &cluster, const isl::id &cluster_id) {
  auto vectorized_loop_size = scop_info_.analysis_result_.GetVectorizedLoopSize();
  if (vectorized_loop_size == 0) {
    return false;
  }

  // check promoted shape
  auto box_sizes = cluster.GetFixedBoxSizes();
  auto local_size = 1;
  for (auto i : box_sizes) {
    local_size = local_size * i;
  }
  if (local_size != vectorized_loop_size || scop_info_.GetDtypeOf(cluster_id).bytes() == 1) {
    return false;
  }

  auto tensor_shape = scop_info_.FindTensor(cluster_id)->shape;
  CHECK(tensor_shape[tensor_shape.size() - 1].as<IntImm>());
  auto shape_vale = tensor_shape[tensor_shape.size() - 1].as<IntImm>()->value;
  if (shape_vale < vectorized_loop_size || (tensor_shape.size() > 1 && shape_vale % vectorized_loop_size != 0)) {
    return false;
  }

  return true;
}

// Determine whether the current tensor needs to be promoted.
bool RegisterCreateCluster::CheckPromotion(const isl::schedule_node &current_node, const isl::schedule_node &node,
                                           const TensorFootprintCluster &cluster,
                                           const std::pair<isl::id, PromotedTensorType> &tensor_info) {
  bool is_enable_vectorization = scop_info_.analysis_result_.GetOuterBandNode(band_index_)->enable_vectorization;
  if (is_enable_vectorization && hoist_vectorized_tensor_) {
    if (!IsSatisfyVectorization(cluster, tensor_info.first)) {
      return false;
    } else {
      need_start_ = false;
      return true;
    }
  }

  if (tensor_info.second > PromotedTensorType::OTHERS) {
    return true;
  }

  if (!IsResueThread(cluster, current_node)) {
    return false;
  }

  return true;
}

BufferDefInfo RegisterCreateCluster::GetPromotedInfo(const isl::id &promoted_id, const std::string &mark_name) {
  isl::id dst_tensor_id = GetGpuIndexDstId(GpuMemType::LOCAL, promoted_id);
  if (scop_info_.IsCopyinTensor(promoted_id.get_name()) && band_index_ != 0) {
    dst_tensor_id = GetGpuIndexDstId(GpuMemType::LOCAL, promoted_id, band_index_);
  }

  std::vector<size_t> buffer_sizes;
  std::vector<std::pair<isl::id, MemType>> data_stream;
  MemType memtype;
  isl::id tmp_item;
  if (!shared_tensor_.count(promoted_id.get_name() + SHARE_SUFFIX)) {
    tmp_item = promoted_id;
    data_stream.push_back(std::make_pair(promoted_id, MemType::DDR));
    data_stream.push_back(std::make_pair(promoted_id, MemType::LOCAL_));
    memtype = MemType::DDR;
  } else {
    tmp_item = isl::id(scop_info_.ctx_, promoted_id.get_name() + SHARE_SUFFIX);
    data_stream.push_back(std::make_pair(promoted_id, MemType::SHARED_));
    data_stream.push_back(std::make_pair(promoted_id, MemType::LOCAL_));
    memtype = MemType::SHARED_;
  }
  BufferDefInfo promoted_info = BufferDefInfo{tmp_item,
                                              dst_tensor_id,
                                              tmp_item,
                                              memtype,
                                              mark_name,
                                              false,
                                              false,
                                              data_stream,
                                              Tensor(),
                                              Handle(),
                                              buffer_sizes,
                                              nullptr,
                                              isl::union_map::empty(isl::space(scop_info_.ctx_, 0))};

  return promoted_info;
}

// Operators that have been promoted to the shared memory do not need to be promoted to the register memory in
// general. Except for gemm operators.
void RegisterCreateCluster::RecordSharedPromotedTensors(const bool is_gemm) {
  for (auto buffer : scop_info_.analysis_result_.active_buffer_footprints_) {
    shared_tensor_.insert(buffer.second.cluster_id.get_name());
  }

  if (is_gemm) {
    return;
  }

  std::string shared_suffix = SHARE_SUFFIX;
  for (const auto &item : shared_tensor_) {
    auto id = isl::id(scop_info_.ctx_, item.substr(0, item.length() - shared_suffix.size()));
    if (all_tensors_.count(id) == 0 || all_tensors_[id] < PromotedTensorType::NONE) {
      all_tensors_[id] = PromotedTensorType::NONE;
    }
  }
}

void RegisterCreateCluster::CreateClusterListForGemm(const isl::schedule_node &node,
                                                     const std::unordered_set<std::string> &mark_names) {
  auto configed_tensors = scop_info_.user_config_.GetRegisterTensors();
  // Initialize the promoted types of all tensors.
  RecordInitPromotedTensorType(configed_tensors);
  // Statistics shared_tensor_ information.
  RecordSharedPromotedTensors(true);
  // Modify promoted type of tensor A/B/C for gemm operator.
  RecordGemmTensors();

  std::unordered_set<std::string> tensor_set;
  for (const auto &mark_name : mark_names) {
    bool hoist_tensor_c = ((mark_name == PROMOTE_GLOBAL_TO_REGISTER_C) || (mark_name == PROMOTE_SHARED_TO_REGISTER_C));
    // Promote the specific tensor at the corresponding marker position.
    tensor_set.clear();
    if (hoist_tensor_c) {
      tensor_set.emplace(TENSOR_C);
    } else {
      tensor_set.emplace(TENSOR_A);
      tensor_set.emplace(TENSOR_B);
    }
    PromotedTensor current_tensors = GetCurrentMarkerTensorsForGemm(tensor_set);
    RecordPromotedTensorInfo(node, mark_name, current_tensors);
  }
}

void RegisterCreateCluster::CreateClusterListForElementWise(const isl::schedule_node &node,
                                                            const std::unordered_set<std::string> &mark_names) {
  auto configed_tensors = scop_info_.user_config_.GetRegisterTensors();
  // Initialize the promoted types of all tensors.
  RecordInitPromotedTensorType(configed_tensors);
  // Delete the tensor that has been promoted on shared memory.
  RecordSharedPromotedTensors();

  for (const auto &mark_name : mark_names) {
    hoist_vectorized_tensor_ = mark_name == PROMOTE_GLOBAL_TO_REGISTER_VECTORIZED;
    // Promote the specific tensor at the corresponding marker position.
    PromotedTensor current_tensors = GetCurrentMarkerTensorsForElementWise();
    RecordPromotedTensorInfo(node, mark_name, current_tensors);
  }
}

PromotedTensor RegisterCreateCluster::GetCurrentMarkerTensorsForElementWise() {
  bool is_enable_vectorization = scop_info_.analysis_result_.GetOuterBandNode(band_index_)->enable_vectorization;
  PromotedTensor current_tensors;
  for (auto tensor : all_tensors_) {
    if (tensor.second > PromotedTensorType::OTHERS && !hoist_vectorized_tensor_) {
      current_tensors.insert(tensor);
    } else if (tensor.second == PromotedTensorType::OTHERS && hoist_vectorized_tensor_ && is_enable_vectorization) {
      // Add the tensor that needs to be vectorized.
      current_tensors[tensor.first] = PromotedTensorType::SPECIAL;
    }
  }

  return current_tensors;
}

/*********************************************
 * Cpu Create Cluster
 *********************************************/
BufferDefInfo CpuCreateCluster::GetPromotedInfo(const isl::id &promoted_id, const std::string &mark_name) {
  GpuMemType gpu_mem_type = GpuMemType::LOCAL;
  MemType mem_type = MemType::LOCAL_;

  isl::id dst_tensor_id = GetGpuIndexDstId(gpu_mem_type, promoted_id);
  if (scop_info_.IsCopyinTensor(promoted_id.get_name()) && band_index_ != 0) {
    dst_tensor_id = GetGpuIndexDstId(gpu_mem_type, promoted_id, band_index_);
  }
  std::vector<size_t> buffer_sizes;
  std::vector<std::pair<isl::id, MemType>> data_stream;
  data_stream.push_back(std::make_pair(promoted_id, MemType::DDR));
  data_stream.push_back(std::make_pair(promoted_id, mem_type));
  BufferDefInfo promoted_info = BufferDefInfo{promoted_id,
                                              dst_tensor_id,
                                              promoted_id,
                                              MemType::DDR,
                                              mark_name,
                                              false,
                                              false,
                                              data_stream,
                                              Tensor(),
                                              Handle(),
                                              buffer_sizes,
                                              nullptr,
                                              isl::union_map::empty(isl::space(scop_info_.ctx_, 0))};

  return promoted_info;
}

isl::union_map CpuCreateCluster::GetPartialSchedule(const isl::schedule_node &node) { return LocalSchedule(node); }

// Determine whether the current tensor needs to be promoted.
bool CpuCreateCluster::CheckPromotion(const isl::schedule_node &current_node, const isl::schedule_node &node,
                                      const TensorFootprintCluster &cluster,
                                      const std::pair<isl::id, PromotedTensorType> &tensor_info) {
  auto template_type = scop_info_.analysis_result_.GetOuterBandNode(band_index_)->template_type;
  return template_type == Template::MATMUL || template_type == Template::CONV;
}

void CpuCreateCluster::CreateClusterListForGemm(const isl::schedule_node &node,
                                                const std::unordered_set<std::string> &mark_names) {
  auto configed_tensors = scop_info_.user_config_.GetRegisterTensors();
  // Initialize the promoted types of all tensors.
  RecordInitPromotedTensorType(configed_tensors);
  std::unordered_set<std::string> tensor_set;

  for (auto mark_name : mark_names) {
    // Promote the specific tensor at the corresponding marker position.
    tensor_set.clear();
    tensor_set.emplace(TENSOR_A);
    tensor_set.emplace(TENSOR_B);
    PromotedTensor current_tensors = GetCurrentMarkerTensorsForGemm(tensor_set);
    RecordPromotedTensorInfo(node, mark_name, current_tensors);
  }
}

void CpuCreateCluster::CreateClusterListForConv(const isl::schedule_node &node,
                                                const std::unordered_set<std::string> &mark_names) {
  auto configed_tensors = scop_info_.user_config_.GetRegisterTensors();
  // Initialize the promoted types of all tensors.
  RecordInitPromotedTensorType(configed_tensors);
  std::unordered_set<std::string> tensor_set;

  for (auto mark_name : mark_names) {
    bool hoist_tensor_c = mark_name == PROMOTE_GLOBAL_TO_REGISTER_C;
    tensor_set.clear();
    if (hoist_tensor_c) {
      tensor_set.emplace(TENSOR_C);
    } else {
      tensor_set.emplace(TENSOR_B);
    }
    // Promote the specific tensor at the corresponding marker position.
    PromotedTensor current_tensors = GetCurrentMarkerTensorsForGemm(tensor_set);
    RecordPromotedTensorInfo(node, mark_name, current_tensors);
  }
}
}  // namespace poly
}  // namespace ir
}  // namespace akg