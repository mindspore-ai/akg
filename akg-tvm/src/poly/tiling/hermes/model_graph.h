/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef POLY_TILING_HERMES_MODEL_GRAPH_H_
#define POLY_TILING_HERMES_MODEL_GRAPH_H_

#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "poly/poly_util.h"
#include "poly/tiling/hermes/axis.h"
#include "poly/tiling/hermes/init_graph.h"
#include "poly/tiling/hermes/node.h"

namespace akg {
namespace ir {
namespace poly {
class ModelGraph : public InitGraph {
 public:
  explicit ModelGraph(InitGraph &init_graph);
  ModelGraph(const InitGraph &, const std::vector<std::shared_ptr<Node>> &);
  ModelGraph() = default;

  std::tuple<int64_t, int> GetMinShapeAndDataCoef(const Axis &axis) const;
  static void InsertToNameDimRangeSet(const std::string &axis_name, const int sch_dim, const int64_t &range);

  std::vector<std::shared_ptr<Node>> critical_nodes_;
  Op::OpCategory dominant_category_;
  bool is_activated_double_buffer_{false};

  static std::vector<Axis> global_axis_vec_;
  static std::set<std::tuple<std::string, int, int64_t>> name_dim_range_set_;

 private:
  static void CompleteNodesGeneratedByReduce(InitGraph &init_graph);
  static bool IsInVector(const std::string &name, const std::vector<std::shared_ptr<Node>> &node_vec);
  static ReduceDirection GetReduceDirection(const std::shared_ptr<Node> &reduce_node);
  static std::shared_ptr<Node> SetReduceSrcDstNodes(const std::shared_ptr<Node> &reduce_node, const std::string &suffix,
                                                    Op::OpType op_type, int64_t shape_size);
  static std::vector<std::shared_ptr<Node>> GetCriticalNodes(const InitGraph &init_graph);

  static const int64_t kExtraMemoryCoeffRequiredByAllReduce = 16;
  static const int64_t kExtraMemoryCoeffRequiredByReduceDst = 8;
  static const int64_t kExtraMemoryCoeffRequiredByReduceSrc = 64;
  static const int64_t kMinShapeSize = 1;
  inline static const std::string kSrcTmpSuffix = "_src_tmp";
  inline static const std::string kDstTmpSuffix = "_dst_tmp";
};
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_TILING_HERMES_MODEL_GRAPH_H_
