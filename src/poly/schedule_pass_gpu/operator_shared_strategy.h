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

#ifndef POLY_OPRATOR_SHARED_STRATEGY_H_
#define POLY_OPRATOR_SHARED_STRATEGY_H_

#include "poly/schedule_pass.h"

namespace akg {
namespace ir {
namespace poly {

class OperatorSharedStrategy {
 public:
  explicit OperatorSharedStrategy(ScopInfo &scop_info, std::unordered_set<std::string> &mark_names, int filter_pos)
      : scop_info_(scop_info), mark_names_(mark_names), band_index_(filter_pos) {}
  ~OperatorSharedStrategy() {}

  std::set<std::string> GetInitPromotedTensor();
  void RecordPromotedTensorInfo(const isl::schedule_node &orig_node, const std::set<std::string> &id_sets,
                                const std::string &mark_names);
  void CreateClusterList(const isl::schedule_node &node);
  void RecordCustomPromotedTensors(std::set<std::string> &id_sets);
  void DeleteNotPromotedTensors(std::set<std::string> &id_sets);

 protected:
  ScopInfo &scop_info_;
  std::unordered_set<std::string> mark_names_;
  int band_index_;
  bool is_local_{false};
};

class ReduceSharedStrategy : public OperatorSharedStrategy {
 public:
  explicit ReduceSharedStrategy(ScopInfo &scop_info, std::unordered_set<std::string> &mark_names, int filter_pos)
      : OperatorSharedStrategy(scop_info, mark_names, filter_pos) {}
  ~ReduceSharedStrategy() {}

  void CreateClusterList(const isl::schedule_node &node);
  std::set<std::string> AnalysisReduceTensors();
};

class BatchMatmulSharedStrategy : public OperatorSharedStrategy {
 public:
  explicit BatchMatmulSharedStrategy(ScopInfo &scop_info, std::unordered_set<std::string> &mark_names, int filter_pos)
      : OperatorSharedStrategy(scop_info, mark_names, filter_pos) {}
  ~BatchMatmulSharedStrategy() {}

  void CreateClusterList(const isl::schedule_node &node);
};

class CpuMemoryStrategy : public OperatorSharedStrategy {
 public:
  explicit CpuMemoryStrategy(ScopInfo &scop_info, std::unordered_set<std::string> &mark_names, int filter_pos)
      : OperatorSharedStrategy(scop_info, mark_names, filter_pos) {
    is_local_ = true;
  }
  ~CpuMemoryStrategy() {}
  std::set<std::string> GetInitPromotedTensor();
  void CreateClusterList(const isl::schedule_node &node);
};
}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_OPRATOR_SHARED_STRATEGY_H_