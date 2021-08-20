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
#include "common/common_util.h"

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
    if (!scop_info.user_config_.GetSharedTensors().empty()) {
      configed_tensors_ = Split(scop_info.user_config_.GetSharedTensors(), " ");
    }
  };
  ~SharedMemoryManager() {}

  virtual isl::schedule Run(isl::schedule sch);
  isl::schedule RunReduce(isl::schedule_node &root);
  isl::schedule RunElemwise(isl::schedule_node &root);
  isl::schedule RunMatmul(isl::schedule_node &root);

  isl::schedule_node HoistSharedMemoryOnDepth(const isl::schedule_node &root);

  isl::schedule_node MapCopiesToThreads(isl::schedule_node &root, bool unroll);

  MappingCfg *GetCurrentConfig(isl::schedule_node &node);

  isl::schedule_node ManageToShareBelow(const isl::schedule &root_sch, isl::schedule_node &node);

  void GatherBufferFootprintDefInfo(const isl::schedule_node &node, BufferDefInfo &tensor_info);

  isl::schedule_node HoistClusters(const isl::schedule_node &root, const isl::schedule_node &node);

  isl::schedule_node HoistToBlockThreadMemory(isl::schedule_node &tree, GpuMemType type, const isl::id &tensor_id,
                                              TensorFootprintCluster &cluster, bool force_last_extension_odd);

  bool CoalescingAccessWay(const isl::schedule_node &root, const isl::schedule_node &node,
                           const TensorFootprintCluster &cluster);

  void OptimizeSharedDimension(std::vector<size_t> &sizes, Type type);
  void OptimizeBankConflict(std::vector<size_t> &sizes, Type type);
  void OptimizeVectorAlign(std::vector<size_t> &sizes);

  std::string InAtomicTensors(isl::schedule_node &node);
  bool InAtomicTensors(const std::string &name);
  bool InReduceTensors(const std::string &name);

  std::string AtomicMarker(const std::string &type);

  size_t Bytes(const isl::id tensor_id);

  isl::schedule_node HoistSharedMemoryOnMark(const isl::schedule_node &root);

  void PrepareInfoForPromotion(const isl::schedule_node &root);
  void SharedPromotion(std::vector<BufferDefInfo> &bd, isl::schedule_node &res_node,
                       const isl::schedule_node &root_node, const isl::schedule_node &node,
                       const isl::multi_union_pw_aff &partial_sched_mupa);

 private:
  ScopInfo &scop_info_;
  isl::schedule schedule_;
  size_t depth_{1};
  bool use_config_{false};
  std::vector<std::string> configed_tensors_;
  bool bank_conflict_{false};
  bool hoist_tensor_c_{false};
  bool shared_inversed_thread_map_{false};
  int shared_vector_align_{0};
  bool is_reduce_{false};
  bool is_matmul_{false};
  bool unroll_shared_{false};
  size_t remain_memory_{common::SHARED_MEMORY_SIZE};
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif
