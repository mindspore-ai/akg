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
#ifndef POLY_TILING_HERMES_AXIS_H_
#define POLY_TILING_HERMES_AXIS_H_

#include <map>
#include <set>
#include <string>
#include <vector>

#include "isl/cpp.h"

namespace akg {
namespace ir {
namespace poly {
class Axis {
 public:
  enum AxisLabel {
    kMatMulAxisBatch,
    kMatMulAxisN,
    kMatMulAxisM,
    kMatMulAxis16,
    kMatMulAxisK,
    kMultiCore,
    kVectorization
  };

  Axis() = default;

  static void SetGlobalAxisName();

  std::string name_;
  std::string gemm_axis_;
  int index_{0};
  int dim_axis_{0};
  int64_t range_{0};
  int64_t c0_tiling_{1};
  int64_t c1_tiling_{1};
  std::set<AxisLabel> type_;

  bool is_inner_{false};
  bool is_innermost_{false};
  bool is_reduce_axis_{false};
  bool is_reduce_src_last_{false};
};

class ScheduleAxisInfo {
 public:
  ScheduleAxisInfo() = default;

  static void GetAxisInfo(const isl::schedule &sch);

 private:
  struct AxisRange {
    int64_t lb{0};
    int64_t ub{0};
    int64_t stride{0};
  };

  struct AxisDim {
    std::string name;
    int statement_dim{0};
    int schedule_dim{0};
    int offset{0};
  };

  struct AxisInfo {
    std::string name;
    std::map<std::string, AxisDim> mapping;
    std::vector<AxisRange> ranges;
  };

  static void GetAxisRanges(__isl_keep isl_schedule *tree, std::map<std::string, AxisInfo> *result);
  static void GetAxisRanges(__isl_keep isl_union_set *full_domain, __isl_keep isl_union_map *schedule,
                            std::map<std::string, AxisInfo> *result);
  static void GetAxisMapping(__isl_keep isl_schedule *tree, std::map<std::string, AxisInfo> *result);
  static void GetAxisMapping(__isl_keep isl_union_map *umap, std::map<std::string, AxisInfo> *result);
  static void GetAxisMapping(__isl_keep isl_map *map, std::map<std::string, AxisInfo> *result);
  static void GetAxisMapping(__isl_keep isl_basic_map *bmap, std::map<std::string, AxisInfo> *result);
  static std::vector<std::string> GetTupleNames(__isl_keep isl_union_set *uset);
};
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_TILING_HERMES_AXIS_H_
