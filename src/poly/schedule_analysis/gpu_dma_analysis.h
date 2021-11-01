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
#ifndef GPU_DMA_ANALYSIS_H
#define GPU_DMA_ANALYSIS_H

#include "poly/schedule_pass.h"

namespace akg {
namespace ir {
namespace poly {

using TensorSets = std::unordered_set<isl::id, isl::IslIdIslHash>;

class GpuDmaAnalysis {
 public:
  explicit GpuDmaAnalysis(const isl::schedule &sch, ScopInfo &scop_info) : sch_(sch), scop_info_(scop_info) {
    if (!scop_info.user_config_.GetSharedTensors().empty()) {
      configed_share_tensors_ = Split(scop_info.user_config_.GetSharedTensors(), " ");
    }
  };

  ~GpuDmaAnalysis() {}
  void Run();

  void Analysis();
  std::vector<isl::id> SharedTensorAnalysis();
  TensorSets AllTensors();
  std::vector<isl::id> LocalTensorAnalysis(const TensorSets &all_tensors, const std::vector<isl::id> &shared_tensors);
  void SetTensorFlow(const std::vector<isl::id> &shares, const std::vector<isl::id> &locals);

  void RemoveInjectiveTensorFromMemFlows(isl::schedule schedule);
  void ResetMemFlows(isl::schedule_node root, isl::schedule_node node);
  isl::schedule_node GetTiledNode(isl::schedule schedule, isl::schedule_node node);

 private:
  const isl::schedule &sch_;
  ScopInfo &scop_info_;
  std::vector<std::string> configed_share_tensors_;
  const int MAX_STRIDE = 65535;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // GPU_DMA_ANALYSIS_H