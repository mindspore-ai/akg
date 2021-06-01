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

#ifndef REGISTER_MEMORY_MANAGER_H_
#define REGISTER_MEMORY_MANAGER_H_

#include "poly/schedule_pass.h"
#include "poly/schedule_tree_util.h"

namespace akg {
namespace ir {
namespace poly {

constexpr auto MAX_REGISTER_PER_THREAD_BLOCK = 65536;
constexpr auto BYTES_PER_REGISTER = 4;
constexpr auto REGISTER_ALLOC_RATIO = 1.0;  // percentage of local memory that allocated to tensors

/*
 * Manager shared memory in GPU.
 */
class RegisterMemoryManager : public SchedulePass {
 public:
  explicit RegisterMemoryManager(PassInfo &pass_info, ScopInfo &scop_info)
      : pass_info_(pass_info), scop_info_(scop_info) {
    pass_name_ = __FUNCTION__;
    if (!scop_info.user_config_.GetLocalTensors().empty()) {
      configed_tensors_ = Split(scop_info.user_config_.GetLocalTensors(), " ");
    }
    if (scop_info_.user_config_.GetEnableMatmul()) {
      local_tensor_c_ = GetMatmulTensorsName(scop_info)[MATRIX_C];
    }
  };
  ~RegisterMemoryManager() {}

  virtual isl::schedule Run(isl::schedule sch);

  isl::schedule HoistRegisterMemoryOnDepth(isl::schedule_node &node, size_t depth);

  void CreateTensorCluster(const isl::schedule_node &node, const isl::union_map &outer_sch);

  void GatherBufferFootprintDefInfo(const isl::schedule_node &node, BufferDefInfo &tensor_info);

  bool IsPromote(const TensorFootprintCluster &fp_cluster, const isl::multi_union_pw_aff &partial_sched_mupa,
                 const isl::multi_union_pw_aff &thread_schedule);

  bool UnrolledLoop(const TensorFootprintCluster &fp_cluster);

  isl::schedule HoistRegisterMemory(isl::schedule_node root, size_t depth);

  void IsOutofMemory(std::vector<BufferDefInfo> promoted_infos);

  size_t UpdateDepth(const isl::schedule_node &root);

  isl::schedule_node GetRegisterPromotedNode(isl::schedule_node &root);
  isl::schedule HoistRegisterMemoryOnMark(isl::schedule_node root);

  isl::schedule_node TileTensorAccordingInterfaceValue(isl::schedule_node &root);
  isl::multi_val GetRealTileSizeVal(const isl::schedule_node &node, const std::string &matrix_name,
                                    const std::string &matrix_major);
  std::string GetPromotedWriteName();

  void GetActualPromotedSharedTensors();

  bool IsReadOrWriteBand(isl::schedule_node node);

 private:
  PassInfo &pass_info_;
  ScopInfo &scop_info_;
  isl::schedule schedule_;
  std::vector<std::string> configed_tensors_;
  bool memory_exceeding_{false};
  bool hoist_compute_local_tensor_{true};
  bool hoist_tensor_all_{false};
  std::string local_tensor_c_;
  std::string shared_tensors_;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif