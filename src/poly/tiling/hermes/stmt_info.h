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
#ifndef POLY_TILING_HERMES_STMT_INFO_H_
#define POLY_TILING_HERMES_STMT_INFO_H_

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "isl/cpp.h"
#include "poly/scop_info.h"

namespace akg {
namespace ir {
namespace poly {
class StmtInfo {
 public:
  struct StmtAxes {
    std::string name;
    int dim{0};
    int64_t range{0};

    StmtAxes(std::string axis_name, int axis_dim, int64_t axis_range)
        : name(std::move(axis_name)), dim(axis_dim), range(axis_range) {}
  };

  StmtInfo() = default;

  static void SetNodeStmtMap(const AnalysisResult &analysis_result);

  static std::unordered_map<std::string, std::string> node_to_stmt_map_;
  static std::unordered_map<std::string, std::vector<StmtAxes>> stmt_name_dim_range_map_;

 private:
  static std::string GetNodeName(const isl::map &map);
  static std::string GetStmtName(const isl::map &map);
};
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_TILING_HERMES_STMT_INFO_H_
