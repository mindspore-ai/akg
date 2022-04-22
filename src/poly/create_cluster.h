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

#ifndef POLY_CREATE_CLUSTER_H_
#define POLY_CREATE_CLUSTER_H_

#include "poly/schedule_pass.h"

namespace akg {
namespace ir {
namespace poly {
// tensor priority: custom > none > special > temp > others
// OTHERS:Whether the tensor is promoted or not has no effect on functionality, only on performance
// TEMP: the temp tensor must be promoted
// SPECIAL: reduce, gemm, conv, etc. operators must be promoted
// NONE: the tensor does not need to be promoted
// CUSTOM: the custom tensor must be promoted
enum class PromotedTensorType { OTHERS = 0, TEMP, SPECIAL, NONE, CUSTOM };
using PromotedTensor = std::unordered_map<isl::id, PromotedTensorType, isl::IslIdIslHash>;

class CreateCluster {
 public:
  explicit CreateCluster(ScopInfo &scop_info, int band_index) : scop_info_(scop_info), band_index_(band_index) {}
  ~CreateCluster() {}

 protected:
  // Record the tensor that needs to be promoted.
  std::set<std::string> GetAllPromotedTensor();
  std::set<std::string> GetTempPromotedTensor(std::set<std::string> all_tensors);
  void RecordInitPromotedTensorType(const std::unordered_set<std::string> &configed_tensors);

  // Sort all tensors by their priority
  std::vector<std::pair<isl::id, PromotedTensorType>> SortPromotedTensorInfo(const PromotedTensor &all_tensors);

  // Record the final tensor that needs to be promoted.
  void RecordPromotedTensorInfo(const isl::schedule_node &orig_node, const std::string &mark_name,
                                const PromotedTensor &all_tensors);

  // Common functions required by shared, register in gpu and cpu.
  virtual isl::union_map GetPartialSchedule(const isl::schedule_node &node) = 0;
  virtual BufferDefInfo GetPromotedInfo(const isl::id &promoted_id, const std::string &mark_name) = 0;
  virtual bool CheckPromotion(const isl::schedule_node &current_node, const isl::schedule_node &node,
                              const TensorFootprintCluster &cluster,
                              const std::pair<isl::id, PromotedTensorType> &tensor_info) = 0;

  // gemm operator
  void RecordGemmTensors();
  PromotedTensor GetCurrentMarkerTensors(const bool hoist_tensor_c);

  ScopInfo &scop_info_;
  PromotedTensor all_tensors_;
  int band_index_;
};

class SharedCreateCluster : public CreateCluster {
 public:
  explicit SharedCreateCluster(ScopInfo &scop_info, int band_index) : CreateCluster(scop_info, band_index) {}
  ~SharedCreateCluster() {}

  // Promoted tensors needed to create different types of operators.
  void CreateClusterListForGemm(const isl::schedule_node &orig_node, const std::unordered_set<std::string> &mark_names);
  void CreateClusterListForReduce(const isl::schedule_node &orig_node,
                                  const std::unordered_set<std::string> &mark_names);
  void CreateClusterListForElementWise(const isl::schedule_node &orig_node,
                                       const std::unordered_set<std::string> &mark_names);

 private:
  bool CoalescingAccessWay(const isl::schedule_node &node, const isl::schedule_node &root,
                           const TensorFootprintCluster &cluster);

  // Common functions required by shared, register in gpu and cpu.
  bool CheckPromotion(const isl::schedule_node &current_node, const isl::schedule_node &node,
                      const TensorFootprintCluster &cluster,
                      const std::pair<isl::id, PromotedTensorType> &tensor_info) override;
  isl::union_map GetPartialSchedule(const isl::schedule_node &node) override;
  BufferDefInfo GetPromotedInfo(const isl::id &promoted_id, const std::string &mark_name) override;

  void RecordReduceTensors();
};

class RegisterCreateCluster : public CreateCluster {
 public:
  explicit RegisterCreateCluster(ScopInfo &scop_info, int band_index) : CreateCluster(scop_info, band_index) {}
  ~RegisterCreateCluster() {}

  // Promoted tensors needed to create different types of operators.
  void CreateClusterListForGemm(const isl::schedule_node &orig_node, const std::unordered_set<std::string> &mark_names);
  void CreateClusterListForElementWise(const isl::schedule_node &orig_node,
                                       const std::unordered_set<std::string> &mark_names);

  bool need_start_{true};
  isl::union_map GetPartialSchedule(const isl::schedule_node &node) override;

 private:
  void RecordSharedPromotedTensors(const bool is_gemm = false);
  bool IsResueThread(const TensorFootprintCluster &cluster, const isl::schedule_node &current_node);
  bool IsSatisfyVectorization(const TensorFootprintCluster &cluster, const isl::id &cluster_id);

  // Common functions required by shared, register in gpu and cpu.
  bool CheckPromotion(const isl::schedule_node &current_node, const isl::schedule_node &node,
                      const TensorFootprintCluster &cluster,
                      const std::pair<isl::id, PromotedTensorType> &tensor_info) override;
  BufferDefInfo GetPromotedInfo(const isl::id &promoted_id, const std::string &mark_name) override;

  void RecordVectorizedPromotedTensors();

  std::set<std::string> shared_tensor_;
};

class CpuCreateCluster : public CreateCluster {
 public:
  explicit CpuCreateCluster(ScopInfo &scop_info, int band_index) : CreateCluster(scop_info, band_index) {}
  ~CpuCreateCluster() {}
  // Promoted tensors needed to create different types of operators.
  void CreateClusterListForGemm(const isl::schedule_node &orig_node, const std::unordered_set<std::string> &mark_names);

 private:
  // Common functions required by shared, register in gpu and cpu.
  isl::union_map GetPartialSchedule(const isl::schedule_node &node) override;
  BufferDefInfo GetPromotedInfo(const isl::id &promoted_id, const std::string &mark_name) override;
  bool CheckPromotion(const isl::schedule_node &current_node, const isl::schedule_node &node,
                      const TensorFootprintCluster &cluster,
                      const std::pair<isl::id, PromotedTensorType> &tensor_info) override;
};
}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_CREATE_CLUSTER_H_