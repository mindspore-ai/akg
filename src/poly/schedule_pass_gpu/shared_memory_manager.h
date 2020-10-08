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

#ifndef SHARED_MEMORY_MANAGER_H_
#define SHARED_MEMORY_MANAGER_H_

#include "poly/schedule_pass.h"

namespace akg {
namespace ir {
namespace poly {

using TensorClusters = std::pair<isl::id, std::vector<std::shared_ptr<TensorFootprintCluster>>>;

/*
 * Manager shared memory in GPU.
 */
class SharedMemoryManager : public SchedulePass {
 public:
  explicit SharedMemoryManager(ScopInfo &scop_info) : scop_info_(scop_info) {
    pass_name_ = __FUNCTION__;
    // use 48KB in current GPU
    share_memory_size_ = 49152;
    tensor_core_share_memory_size_ = 61440;
    if (!scop_info.user_config_.GetSharedTensors().empty()) {
      configed_tensors_ = Split(scop_info.user_config_.GetSharedTensors(), " ");
    }
    unroll_copies_ = false;
  };
  ~SharedMemoryManager() {}

  virtual isl::schedule Run(isl::schedule sch);

  isl::schedule_node HoistSharedMemoryOnDepth(const isl::schedule_node &root, size_t &remain_memory, size_t depth);

  isl::union_set GatherMappingsTo(MappingCfg *cfg);

  isl::schedule_node MapCopiesToThreads(isl::schedule_node &root, bool unroll);

  MappingCfg *GetCurrentConfig(isl::schedule_node &node);

  isl::schedule_node ManageToShareBelow(const isl::schedule &root_sch, isl::schedule_node &node,
                                        size_t &remaining_memory);

  void CreateClusterList(const isl::schedule_node &node, const isl::union_map &outer_sch);

  void GatherBufferFootprintDefInfo(const isl::schedule_node &node, BufferDefInfo &tensor_info);

  isl::schedule_node HoistClusters(const isl::schedule_node &root, const isl::schedule_node &node,
                                   size_t &remaining_memory);

  isl::schedule_node HoistToBlockThreadMemory(isl::schedule_node &tree, GpuMemType type, const isl::id &tensor_id,
                                              TensorFootprintCluster &cluster, bool force_last_extension_odd);

  bool ReuseTensorCluster(const TensorFootprintCluster &cluster, const isl::multi_union_pw_aff &outer_pw_aff);

  bool CoalescingAccessWay(const isl::schedule_node &root, const isl::schedule_node &node,
                           const TensorFootprintCluster &cluster);

  void UpdateDepth(const isl::schedule_node &root);

  std::vector<size_t> OptimizeSharedDimension(std::vector<size_t> sizes);
  std::vector<size_t> OptimizeBankConflict(std::vector<size_t> sizes);
  std::vector<size_t> OptimizeVectorAlign(std::vector<size_t> sizes);
  bool UnderThreadMarker(size_t depth);

  std::string InAtomicTensors(isl::schedule_node &node);
  bool InAtomicTensors(std::string name);
  bool InReduceTensors(std::string name);

  std::string AtomicMarker(std::string type);

  std::set<std::string> AnalysisReduceTensors();

  size_t Bytes(const isl::id tensor_id);
  isl::schedule_node CollectMarkNode(isl::schedule_node root, const std::string mark);

  isl::schedule_node HoistSharedMemoryOnMark(const isl::schedule_node &root, size_t &remain_memory, size_t depth);

 private:
  ScopInfo &scop_info_;
  isl::schedule schedule_;
  size_t share_memory_size_;
  size_t tensor_core_share_memory_size_;
  int depth_{1};
  bool use_config_{false};
  std::vector<std::string> configed_tensors_;
  bool unroll_copies_;
  bool bank_conflict_{false};
  bool hoist_tensor_c_ = true;
  std::string tensor_c_;
  std::string tensor_a_;
  std::string tensor_b_;
  bool shared_inversed_thread_map_{false};
  int shared_vector_align_{0};
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif
