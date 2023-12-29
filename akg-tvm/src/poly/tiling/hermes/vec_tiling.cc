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
#include <algorithm>
#include <memory>
#include <vector>

#include "poly/tiling/hermes/tiling_mem.h"
#include "poly/tiling/hermes/utils.h"
#include "poly/tiling/hermes/vec_tiling.h"

namespace akg {
namespace ir {
namespace poly {
int64_t GetVecAxis(const Axis &axis, const ModelGraph &model_graph, Hardware hardware) {
  int64_t axis_result = 1;

  int64_t min_shape = axis.range_;
  int data_coef = 0;
  std::tie(std::ignore, data_coef) = model_graph.GetMinShapeAndDataCoef(axis);

  int64_t max_alloc_buffer = 0;
  int64_t upper_bound_buffer = 0;
  std::tie(max_alloc_buffer, upper_bound_buffer) =
    GetMaxAllocAndUpperBoundBuffer(hardware.mem_VC_size_, hardware.mem_VC_align_, axis, model_graph.critical_nodes_);

  axis_result = std::min(max_alloc_buffer, min_shape);
  if ((axis_result != min_shape) && (axis_result & (axis_result - 1)) != 0) {
    axis_result = Get2PowerLess(axis_result);
  }

  axis_result = std::min(min_shape, axis_result);

  int last_dim_axis = GetLastDimAxis();
  if (last_dim_axis - axis.dim_axis_ > 1) {
    if ((upper_bound_buffer != min_shape) && (upper_bound_buffer & (upper_bound_buffer - 1)) != 0) {
      upper_bound_buffer = Get2PowerLess(upper_bound_buffer);
    }
    if (axis.is_innermost_) {
      auto avg_tiling = static_cast<int64_t>(std::pow(static_cast<float>(max_alloc_buffer),
                                                      1.0F / static_cast<float>((last_dim_axis - axis.dim_axis_ + 1))));
      avg_tiling = (avg_tiling & (avg_tiling - 1)) != 0 ? Get2PowerLess(avg_tiling) : avg_tiling;
      avg_tiling = std::max(avg_tiling,
                            upper_bound_buffer);  // no need to reduce the tiling if there is enough space to fully tile
      axis_result = std::min(axis_result, avg_tiling);
    } else {
      axis_result = std::min(upper_bound_buffer, axis_result);
    }
  }

  if (axis.dim_axis_ == last_dim_axis) {
    size_t global_axis_size = 0;
    size_t second_last_global_axis = 0;
    for (size_t idx_global_axis = ModelGraph::global_axis_vec_.size(); idx_global_axis > 0; --idx_global_axis) {
      if (!ModelGraph::global_axis_vec_[idx_global_axis - 1].is_inner_) {
        global_axis_size++;
        if (global_axis_size == kSecondAxis) {
          second_last_global_axis = idx_global_axis - 1;
        }
      }
    }

    auto penultimate_axis = ModelGraph::global_axis_vec_[second_last_global_axis];
    bool is_penult_label_vec = penultimate_axis.type_.count(Axis::AxisLabel::kVectorization) != 0;
    if (is_penult_label_vec &&
        (axis_result < penultimate_axis.c0_tiling_ ||
         (!penultimate_axis.is_innermost_ &&
          (penultimate_axis.c0_tiling_ * axis.range_ > axis_result * penultimate_axis.range_) &&
          penultimate_axis.c0_tiling_ > static_cast<int64_t>(hardware.vblocknum_))) &&
        axis_result < min_shape && axis_result < axis.range_) {
      ModelGraph::global_axis_vec_[second_last_global_axis].c0_tiling_ = Get2PowerLess(penultimate_axis.c0_tiling_);
      axis_result = GetVecAxis(axis, model_graph, hardware);
    }
  }

  return axis_result;
}

int64_t GetMixTypeAxis(const Axis &axis, const ModelGraph &model_graph, Hardware hardware) {
  int64_t axis_result = 1;

  std::vector<std::shared_ptr<Node>> nodes = model_graph.nodes_;

  int64_t min_shape = axis.range_;
  int data_coef = 0;
  std::tie(std::ignore, data_coef) = model_graph.GetMinShapeAndDataCoef(axis);

  int64_t multi_core_size = min_shape;
  if (hardware.num_core_ > 0) {
    multi_core_size /= static_cast<int64_t>(hardware.num_core_);
  }
  int64_t vec_cal_size = GetVecAxis(axis, model_graph, hardware);
  int64_t vec_granularity = 0;
  if (data_coef > 0) {
    vec_granularity = static_cast<int64_t>(hardware.vblocknum_ * hardware.vblocksize_) / data_coef;
  }
  if (vec_granularity == 0) {
    vec_granularity = 1;
  }

  int global_axis_nb = 0;
  for (auto const &global_axis : ModelGraph::global_axis_vec_) {
    if (global_axis.index_ == 0) {
      global_axis_nb++;
    }
  }

  if (global_axis_nb <= 1) {
    if (vec_cal_size > 0) {
      if (multi_core_size < vec_granularity * static_cast<int64_t>(hardware.vblocknum_)) {
        hardware.num_core_ = static_cast<size_t>(min_shape / vec_cal_size);
        axis_result = vec_cal_size;
      } else {
        axis_result = std::min(multi_core_size, vec_cal_size);
      }
    }
  } else {
    if (PrioAxis(axis, model_graph)) {
      if (multi_core_size > vec_granularity) {
        if (multi_core_size % vec_granularity == 0) {
          axis_result = multi_core_size;
        } else {
          axis_result = vec_granularity;
          while (multi_core_size > axis_result + vec_granularity) {
            axis_result += vec_granularity;
          }
        }
      } else {
        multi_core_size = vec_granularity;
        hardware.num_core_ = static_cast<size_t>(min_shape / vec_granularity);
        axis_result = std::min(multi_core_size, vec_cal_size);
      }
    } else {
      if (axis.name_ != ModelGraph::global_axis_vec_[0].name_ &&
          ModelGraph::global_axis_vec_[0].range_ / ModelGraph::global_axis_vec_[0].c0_tiling_ >=
            static_cast<int64_t>(hardware.num_core_)) {
        axis_result = vec_cal_size;
      } else {
        multi_core_size = static_cast<int64_t>(static_cast<int>(hardware.vblocksize_) / data_coef);
        axis_result = std::min(multi_core_size, vec_cal_size);
      }
    }
  }

  if (axis_result != min_shape && (axis_result & (axis_result - 1)) != 0) {
    axis_result = Get2PowerLess(axis_result);
  }

  return axis_result;
}

bool PrioAxis(const Axis &axis, const ModelGraph &model_graph) {
  bool prio = false;
  if (axis.range_ == ModelGraph::global_axis_vec_[0].range_) {
    prio = false;
  }
  for (auto const &node : model_graph.nodes_) {
    if (node->op_.op_type_ == Op::OpType::ReduceY && node->HasAxis(axis) && node->axis_of_node_.size() == 1) {
      prio = true;
      break;
    }
  }
  return prio;
}

void ExtendVecAxisTile(Axis &axis, const ModelGraph &model_graph, Hardware hardware) {
  int64_t available_tiling = axis.range_ / axis.c0_tiling_;
  if (available_tiling <= 1) {
    return;
  }

  int64_t axis_result = 1;

  int64_t min_shape = axis.range_;
  int data_coef = 0;
  std::tie(std::ignore, data_coef) = model_graph.GetMinShapeAndDataCoef(axis);

  int64_t curr_max_alloc_buffer = 0;
  int64_t upper_bound_buffer = 0;
  std::tie(curr_max_alloc_buffer, upper_bound_buffer) =
    GetMaxAllocAndUpperBoundBuffer(hardware.mem_VC_size_, hardware.mem_VC_align_, axis, model_graph.critical_nodes_);

  if (model_graph.dominant_category_ == Op::OpCategory::Injective) {
    int64_t prime_factors_prod = GetLowestPrimeFactorsProductBelow(axis.range_, curr_max_alloc_buffer);
    if (prime_factors_prod > axis.c0_tiling_) {
      axis.c0_tiling_ = prime_factors_prod;
      return;
    }
  }

  axis_result = std::min(curr_max_alloc_buffer, min_shape);
  if ((axis_result != min_shape) && (axis_result & (axis_result - 1)) != 0) {
    axis_result = Get2PowerLess(axis_result);
  }

  if (axis_result == 0) {
    return;
  }

  int64_t remaining_tiling = curr_max_alloc_buffer / axis_result;
  if (remaining_tiling < 1) {
    return;
  }

  if (hardware.num_core_ == 0) {
    LOG(WARNING) << "Number of cores is 0!";
    return;
  }

  size_t num_mc_axis = axis.type_.count(Axis::AxisLabel::kMultiCore);

  for (int64_t i = remaining_tiling; i > 0; --i) {
    if (available_tiling % i == 0) {
      int64_t axis_range = axis.range_;
      int64_t axis_tile = axis.c0_tiling_;
      if (available_tiling > static_cast<int64_t>(hardware.num_core_) &&
          ((num_mc_axis != 0 && available_tiling / static_cast<int64_t>(hardware.num_core_) < i) ||
           axis_range % (i * axis_tile) != 0 || (i * axis_tile) % static_cast<int64_t>(hardware.num_core_) != 0)) {
        continue;
      }
      int64_t old_tiling = axis.c0_tiling_;
      axis.c0_tiling_ *= i;
      int64_t max_alloc_buffer = 0;
      std::tie(max_alloc_buffer, std::ignore) = GetMaxAllocAndUpperBoundBuffer(
        hardware.mem_VC_size_, hardware.mem_VC_align_, axis, model_graph.critical_nodes_);
      auto max_percentage =
        static_cast<int64_t>(std::round(static_cast<float>(max_alloc_buffer) * kMaxAllowedAllocPercentage));
      if (max_percentage < axis_result || axis.c0_tiling_ < old_tiling) {
        axis.c0_tiling_ = old_tiling;
      } else {
        remaining_tiling /= i;
        break;
      }
    }
  }
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
