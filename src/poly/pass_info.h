/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef POLY_PASS_INFO_H_
#define POLY_PASS_INFO_H_

#include <tvm/expr.h>

#include <vector>
#include <map>
#include <unordered_map>
#include "isl.h"

namespace akg {
namespace ir {
namespace poly {

struct DimensionInfo {
  int64_t index;
  std::string axis;
  int64_t c1_tiling_size;
  int64_t c0_tiling_size;
  int64_t dim_seq;
  air::Expr c1_var;
  air::Expr c0_var;
  air::Expr pragma;
  bool is_inner{false};
};
using TileSizes = std::vector<DimensionInfo>;

class Dependency {
 private:
  isl::id start_node_id_;
  isl::id end_node_id_;
  int64_t edge_weight_;

 public:
  Dependency(const isl::id start_node_id, const isl::id end_node_id, const int64_t edge_weight)
      : start_node_id_(start_node_id), end_node_id_(end_node_id), edge_weight_(edge_weight) {}
  ~Dependency() {}

  isl::id GetStartNode() { return start_node_id_; }
  isl::id GetEndNode() { return end_node_id_; }
  int64_t GetEdgeWeight() const { return edge_weight_; }
};

// pass info on schedule transform
class PassInfo {
 public:
  PassInfo() {}
  ~PassInfo() {}

  bool has_grouped_{false};

  std::unordered_map<isl::id, isl::union_set_list, isl::IslIdIslHash> group_filter_map_;

  std::vector<Dependency> dependency_list_;

  isl::union_pw_multi_aff group_upma_;

  isl::schedule_constraints constraints_;

  isl::union_map dependences_;
  isl::union_map force_dependences_;

  isl::union_map orig_dependences_;
  isl::union_set transfer_stmt_;
  std::map<std::string, int> invariant_state_;
  bool has_invariant_dependence_{false};
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_PASS_INFO_H_
