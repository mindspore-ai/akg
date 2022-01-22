/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef POLY_MEMORY_MANAGER_H_
#define POLY_MEMORY_MANAGER_H_

#include <queue>
#include "poly/pass_info.h"
#include "poly/scop_info.h"
#include "poly/schedule_pass.h"

namespace akg {
namespace ir {
namespace poly {
class MemoryManager : public SchedulePass {
 public:
  explicit MemoryManager(ScopInfo &scop_info) : scop_info_(scop_info) { pass_name_ = __FUNCTION__; }
  ~MemoryManager() {}

  virtual isl::schedule Run(isl::schedule sch);

 private:
  isl::schedule HoistBufferFootprintAtMarkNode(const isl::schedule_node &root, const std::string &markTag,
                                               size_t index);
  isl::schedule_node HoistBufferFootprintAtMarkNode(const isl::schedule_node &tree, size_t index);
  isl::schedule_node HoistTensorClusterFootprint(isl::schedule_node tree, size_t index, const isl::union_map &schedule);
  std::vector<std::pair<isl::union_set, BufferedFootPrintInfo>> CollectBufferedFootprints(
    const isl::union_set &active_points, const isl::id &tensor_id) const;
  std::vector<size_t> CollectBufferedFootprintsIndexes(const isl::union_set &active_points,
                                                       const isl::id &tensor_id) const;
  std::shared_ptr<TensorFootprintCluster> GetFootPrintsCluster(const isl::id &tensor_id);
  void SetFindBuffer(const isl::id &tensor_id, bool find_buffer);

  void AddStateTensorsDataFlow();
  void AddTensorDataFlow(const std::vector<MemType> &mem_flow, const std::vector<std::string> &name_flow,
                         std::string mark_tag_specific = "");

  // record buffer footprint
  void AddOneBufferDefInfo(const isl::id &ancestorId, const std::vector<std::pair<isl::id, MemType>> &data_stream);
  void MakeBufferFootprintCluster(BufferDefInfo &tensor_info);
  void GatherBufferFootprintDefInfo(const isl::schedule_node &tree, BufferDefInfo &tensor_info);
  void GatherFractalDefInfo(const isl::schedule_node &tree, BufferDefInfo &tensor_info, std::vector<size_t> &sizes);
  void HoistIm2colBufferFootprintCluster(const isl::union_map &schedule, const isl::schedule_node &node, int index,
                                         BufferDefInfo &tensor_info);
  void MakeMultiBufferFootprint(const isl::union_map &schedule, const isl::schedule_node &node, int &index,
                                BufferDefInfo &tensor_info);
  void ReorderBufferedDefInfos();
  void CollectBufferFootprintDefInfo(BufferDefInfo &tensor_info, const isl::union_map &schedule,
                                     const isl::schedule_node &node);

  void AddGemmTransposeFpCluster(const isl::union_map &schedule);

 private:
  // PassInfo &pass_info_;
  ScopInfo &scop_info_;
  std::queue<isl::id> buffer_footprint_queue_;

  std::shared_ptr<TensorFootprintCluster> gemm_a_transpose_fp_cluster_;
  std::shared_ptr<TensorFootprintCluster> gemm_b_transpose_fp_cluster_;
  std::shared_ptr<TensorFootprintCluster> im2col_fp_cluster;

  isl::schedule schedule_;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif