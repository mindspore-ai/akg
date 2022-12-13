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
#include "poly/tiling/hermes/stmt_info.h"

namespace akg {
namespace ir {
namespace poly {
std::unordered_map<std::string, std::string> StmtInfo::node_to_stmt_map_;
std::unordered_map<std::string, std::vector<StmtInfo::StmtAxes>> StmtInfo::stmt_name_dim_range_map_;

void StmtInfo::SetNodeStmtMap(const AnalysisResult &analysis_result) {
  node_to_stmt_map_.clear();

  auto writes_umap = analysis_result.GetWrites();
  writes_umap.foreach_map([&](const isl::map &map) -> void {
    std::string node_name = GetNodeName(map);
    std::string stmt_name = GetStmtName(map);
    node_to_stmt_map_.insert(std::make_pair(node_name, stmt_name));
  });

  auto reads_umap = analysis_result.GetReads();
  reads_umap.foreach_map([&](const isl::map &map) -> void {
    std::string node_name = GetNodeName(map);
    if (node_to_stmt_map_.find(node_name) == node_to_stmt_map_.end()) {
      std::string stmt_name = GetStmtName(map);
      node_to_stmt_map_.insert(std::make_pair(node_name, stmt_name));
    }
  });
}

std::string StmtInfo::GetNodeName(const isl::map &map) {
  auto tuple_id_out = map.get_tuple_id(isl_dim_out);
  return tuple_id_out.to_str();
}

std::string StmtInfo::GetStmtName(const isl::map &map) {
  auto tuple_id_in = map.domain_factor_domain().get_tuple_id(isl_dim_in);
  return tuple_id_in.to_str();
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
