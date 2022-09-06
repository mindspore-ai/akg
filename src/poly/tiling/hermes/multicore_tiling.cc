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

#include "poly/tiling/tiling_utils.h"
#include "poly/tiling/hermes/multicore_tiling.h"
#include "poly/tiling/hermes/tiling_mem.h"
#include "poly/tiling/hermes/utils.h"

namespace akg {
namespace ir {
namespace poly {
int64_t GetMcAxis(const Axis &axis, const ModelGraph &model_graph, Hardware hardware) {
  // Adapt the min shape that this axis mapped
  int64_t min_shape = axis.range_;
  int data_coef = 0;
  std::tie(std::ignore, data_coef) = model_graph.GetMinShapeAndDataCoef(axis);

  bool reduce_Y = false;
  bool input_fp16 = false;
  for (auto const &node : model_graph.nodes_) {
    if (node->op_.op_type_ == Op::OpType::ReduceY) {
      reduce_Y = true;
    }
    if (!input_fp16 && node->op_.IsInput() &&
        node->output_tensors_[0]->GetDataTypeCoef() == node->output_tensors_[0]->kTwoBytesPerVal) {
      input_fp16 = true;
    }
  }

  if (reduce_Y) {
    return GetMcReduceYAxisSize(hardware, min_shape, input_fp16, model_graph.outputs_.size());
  }

  return GetMcAxisSize(hardware, min_shape, data_coef);
}

int64_t GetMcReduceYAxisSize(Hardware hardware, int64_t min_shape, bool input_fp16, size_t model_graph_out_size) {
  // Initial Size for multicore (at least 1 per core)
  int64_t multicore_axis_size = 1;
  if (hardware.num_core_ > 0 && min_shape % static_cast<int64_t>(hardware.num_core_) == 0) {
    multicore_axis_size = min_shape / static_cast<int64_t>(hardware.num_core_);
  }

  if (model_graph_out_size > 1) {
    multicore_axis_size = Get2PowerBelow(multicore_axis_size);
  }

  if (input_fp16) {
    if (multicore_axis_size % kFloat16LoadGranularity != 0 && min_shape >= kFloat16LoadGranularity) {
      multicore_axis_size = kFloat16LoadGranularity;
    }
  } else {
    if (multicore_axis_size % kFloat32LoadGranularity != 0 && min_shape >= kFloat32LoadGranularity) {
      multicore_axis_size = kFloat32LoadGranularity;
    }
  }

  if (min_shape < kReduceMinShapeThreshold) {
    multicore_axis_size = min_shape;
  }

  return multicore_axis_size;
}

int64_t GetMcAxisSize(Hardware hardware, int64_t min_shape, int data_coef) {
  // Initial Size for multicore (at least 1 per core)
  int64_t multicore_axis_size = 1;

  auto num_core_vblock_per_datatype =
    static_cast<int64_t>(static_cast<int>(hardware.num_core_ * hardware.vblocksize_) / data_coef);
  if (data_coef > 0 && min_shape >= num_core_vblock_per_datatype) {
    if (min_shape > kSmallAxisSize) {
      multicore_axis_size = static_cast<int64_t>(static_cast<int>(hardware.vblocksize_) / SafeDivisor(data_coef));
    }
  }

  return multicore_axis_size;
}

std::tuple<float, float> GetTileAndMulticoreAxisSizes(const Axis &current_axis, const Axis &critical_node_axis,
                                                      const std::vector<Axis> &global_axis_vec_, float curr_tile_size,
                                                      float curr_tile_multicore_axis_size) {
  float tile_size = curr_tile_size;
  float tile_multicore_axis_size = curr_tile_multicore_axis_size;
  for (auto const &axis : global_axis_vec_) {
    if (critical_node_axis.dim_axis_ == axis.dim_axis_) {
      tile_size *= static_cast<float>(axis.c0_tiling_);
      if (axis.dim_axis_ > current_axis.dim_axis_) {
        tile_multicore_axis_size *= static_cast<float>(axis.range_);
      } else {
        tile_multicore_axis_size *= static_cast<float>(axis.c0_tiling_);
      }
    }
  }
  return std::make_tuple(tile_size, tile_multicore_axis_size);
}

void ExtendMulticoreAxisTile(const Axis &axis, const ModelGraph &model_graph, Hardware hardware) {
  int64_t axis_result = 1;

  int64_t min_shape = axis.range_;
  int data_coef = 0;
  std::tie(std::ignore, data_coef) = model_graph.GetMinShapeAndDataCoef(axis);

  int64_t curr_max_alloc_buffer = 0;
  int64_t upper_bound_buffer = 0;
  std::tie(curr_max_alloc_buffer, upper_bound_buffer) =
    GetMaxAllocAndUpperBoundBuffer(hardware.mem_VC_size_, hardware.mem_VC_align_, axis, model_graph.critical_nodes_);

  axis_result = std::min(curr_max_alloc_buffer, min_shape);
  if ((axis_result != min_shape) && (axis_result & (axis_result - 1)) != 0) {
    axis_result = Get2PowerBelow(axis_result);
  }

  if (axis_result == 0) {
    return;
  }

  int64_t remaining_tiling = curr_max_alloc_buffer / axis_result;
  if (remaining_tiling <= 1) {
    return;
  }

  if (hardware.num_core_ == 0) {
    LOG(WARNING) << "Number of cores is 0!";
    return;
  }

  for (auto iter_axis = ModelGraph::global_axis_vec_.rbegin() + 1; iter_axis != ModelGraph::global_axis_vec_.rend();
       ++iter_axis) {
    int64_t available_tiling = iter_axis->range_ / iter_axis->c0_tiling_;
    if (available_tiling > 1) {
      size_t num_mc_axis = iter_axis->type_.count(Axis::AxisLabel::kMultiCore);
      size_t num_vec_axis = iter_axis->type_.count(Axis::AxisLabel::kVectorization);

      for (int64_t i = remaining_tiling; i > 0; --i) {
        if (available_tiling % i == 0) {
          int64_t axis_range = iter_axis->range_;
          int64_t axis_tile = iter_axis->c0_tiling_;
          if (available_tiling > static_cast<int64_t>(hardware.num_core_) &&
              ((num_mc_axis != 0 && available_tiling / static_cast<int64_t>(hardware.num_core_) < i) ||
               axis_range % (i * axis_tile) != 0 || (i * axis_tile) % static_cast<int64_t>(hardware.num_core_) != 0)) {
            continue;
          }
          int64_t old_tiling = iter_axis->c0_tiling_;
          iter_axis->c0_tiling_ *= i;
          int64_t max_alloc_buffer = 0;
          std::tie(max_alloc_buffer, std::ignore) = GetMaxAllocAndUpperBoundBuffer(
            hardware.mem_VC_size_, hardware.mem_VC_align_, axis, model_graph.critical_nodes_);
          auto max_percentage =
            static_cast<int64_t>(std::round(static_cast<float>(max_alloc_buffer) * kMaxAllowedAllocPercentage));
          if (max_percentage < axis_result) {
            iter_axis->c0_tiling_ = old_tiling;
          } else {
            remaining_tiling /= i;
            break;
          }
        }
      }

      if (num_mc_axis != 0 && num_vec_axis == 0) {
        iter_axis->c0_tiling_ = GetTileFromRemainingVecGranularity(*iter_axis, data_coef, axis_result);
      }
    }
  }
}

int64_t GetTileFromRemainingVecGranularity(const Axis &global_axis, int data_coef, int64_t axis_result) {
  int64_t remaining_vec_granularity = 0;
  if (data_coef > 0 && axis_result > 0) {
    auto vec_granularity_per_data_coef = static_cast<int64_t>(kVectorizationGranularity / data_coef);
    remaining_vec_granularity = vec_granularity_per_data_coef / axis_result;
  }
  if ((remaining_vec_granularity & (remaining_vec_granularity - 1)) != 0) {
    remaining_vec_granularity = Get2PowerBelow(remaining_vec_granularity);
  }
  if (remaining_vec_granularity < 1) {
    remaining_vec_granularity = 1;
  }
  if (global_axis.range_ / global_axis.c0_tiling_ / static_cast<int64_t>(kNumCore) / remaining_vec_granularity > 1) {
    return global_axis.c0_tiling_ * remaining_vec_granularity;
  }
  return global_axis.c0_tiling_;
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
