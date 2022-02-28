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
  };
  ~RegisterMemoryManager() {}

  virtual isl::schedule Run(isl::schedule sch);

 private:
  isl::schedule HoistRegisterMemory();
  isl::schedule_node HoistRegisterMemoryOnMark(const isl::schedule_node &orig_node);
  isl::union_map GetPartialSchedule(const isl::schedule_node &node);
  isl::schedule_node HoistClusters(const isl::schedule_node &node);

  void CreateClusterForOperator(const isl::schedule_node &node);

  void GatherBufferFootprintDefInfo(const isl::schedule_node &node, BufferDefInfo &tensor_info);

  bool UnrolledLoop(const TensorFootprintCluster &fp_cluster);

  isl::schedule_node GetRegisterPromotedNode(isl::schedule_node &root);

  isl::schedule_node AdjustConvScheduleTreeStructure(const isl::schedule_node &orig_node);
  isl::schedule_node TileTensorAccordingInterfaceValue(const isl::schedule_node &orig_node);
  isl::multi_val GetRealTileSizeVal(const isl::schedule_node &node, const std::string &matrix_name,
                                    const std::string &matrix_major);
  void SetPromotedWriteNameForGemm(std::string &local_tensor_c);
  isl::schedule_node InsertMarkerForEmit(const isl::schedule_node &orig_node);

  PassInfo &pass_info_;
  ScopInfo &scop_info_;
  isl::schedule schedule_;
  std::string write_name_;

  int band_index_{0};
  OuterBandNode *current_outer_bn_{nullptr};
  std::unordered_set<std::string> mark_names_;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif