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
  explicit OperatorSharedStrategy(ScopInfo &scop_info) : scop_info_(scop_info) {}
  ~OperatorSharedStrategy() {}

  std::set<std::string> GetInitPromotedTensor();
  void RecordPromotedTensorInfo(const isl::schedule_node &node, const isl::union_map &outer_sch,
                                const std::set<std::string> &id_sets);
  void CreateClusterList(const isl::schedule_node &node, const isl::union_map &outer_sch);
  void RecordCustomPromotedTensors(std::set<std::string> &id_sets);
  void DeleteNotPromotedTensors(std::set<std::string> &id_sets);

 protected:
  ScopInfo &scop_info_;
};

class ReduceSharedStrategy : public OperatorSharedStrategy {
 public:
  explicit ReduceSharedStrategy(ScopInfo &scop_info) : OperatorSharedStrategy(scop_info) {}
  ~ReduceSharedStrategy() {}

  void CreateClusterList(const isl::schedule_node &node, const isl::union_map &outer_sch);
  std::set<std::string> AnalysisReduceTensors();
};

class BatchMatmulSharedStrategy : public OperatorSharedStrategy {
 public:
  explicit BatchMatmulSharedStrategy(ScopInfo &scop_info) : OperatorSharedStrategy(scop_info) {}
  ~BatchMatmulSharedStrategy() {}

  void CreateClusterList(const isl::schedule_node &node, const isl::union_map &outer_sch, const bool hoist_tensor_c);
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_OPRATOR_SHARED_STRATEGY_H_