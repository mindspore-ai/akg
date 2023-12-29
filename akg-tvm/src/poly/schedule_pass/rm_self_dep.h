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
#ifndef POLY_RM_SELF_DEP_H_
#define POLY_RM_SELF_DEP_H_

#include <vector>
#include <string>

namespace akg {
namespace ir {
namespace poly {

struct AxisInfo {
  std::string iter{""};
  int domain_min{0};
  int domain_max{0};
  int range_min{0};
  int range_max{0};
};

/*
 * S_x[i0, i1] -> S_y[i1, i2]
 *
 * outer_axis: i1
 * reduce_axis: i2
 * broadcast_axis: i0
 */
struct ReduceInfo {
  std::vector<AxisInfo> outer_axis;
  std::vector<AxisInfo> reduce_axis;
  std::vector<AxisInfo> broadcast_axis;
};

struct ReferenceAxisInfo {
  bool is_axis_type_known{false};
  bool is_reduce_axis{false};
  bool min_defined{false};
  int min{0};
  bool max_defined{false};
  int max{0};
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_RM_SELF_DEP_H_
