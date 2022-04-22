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

#ifndef CPU_MEMORY_MANAGER_H_
#define CPU_MEMORY_MANAGER_H_

#include "poly/schedule_pass.h"
#include "common/common_util.h"

namespace akg {
namespace ir {
namespace poly {

/*
 * Manager memory in CPU.
 */
class CpuMemoryManager : public SchedulePass {
 public:
  explicit CpuMemoryManager(ScopInfo &scop_info) : scop_info_(scop_info) { pass_name_ = __FUNCTION__; };
  ~CpuMemoryManager() {}

  virtual isl::schedule Run(isl::schedule sch);

 private:
  void GatherBufferFootprintDefInfo(const isl::schedule_node &node, BufferDefInfo &tensor_info);

  isl::schedule_node HoistClusters(const isl::schedule_node &node);

  isl::schedule_node HoistMemory(isl::schedule_node &tree, GpuMemType type, const isl::id &tensor_id,
                                 const isl::id &dst_tensor_id, TensorFootprintCluster &cluster,
                                 bool force_last_extension_odd);

  isl::schedule HoistCpuMemory();

  void CreateClusterForOperator(const isl::schedule_node &orig_node);
  isl::schedule_node InsertMarkerForVectorization(const isl::schedule_node &orig_node);
  isl::schedule_node HoistCpuMemoryOnMark(const isl::schedule_node &orig_node);

  ScopInfo &scop_info_;
  int band_index_{0};
  OuterBandNode *current_outer_bn_{nullptr};
  isl::schedule schedule_;
  std::unordered_set<std::string> mark_names_;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif
