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

  int axis_count = 0;
  for (auto &g_axis : ModelGraph::global_axis_vec_) {
    if (!g_axis.is_inner_) {
      ++axis_count;
    }
  }
  if (axis_count > 1 && model_graph.dominant_category_ == Op::OpCategory::Transpose) {
    float total_range = 0;
    for (auto const &g_axis : ModelGraph::global_axis_vec_) {
      if (!axis.is_inner_) {
        total_range += static_cast<float>(g_axis.range_);
      }
    }
    auto mc_axis_size_factor = static_cast<float>(axis.range_) / total_range;
    if (mc_axis_size_factor > kMaxAllowedAllocPercentage) {
      return 1;
    }
  }
  return GetMcAxisSize(hardware, min_shape, data_coef);
}

int64_t GetMcReduceYAxisSize(Hardware hardware, int64_t min_shape, bool input_fp16, size_t model_graph_out_size) {
  // Initial Size for multicore (at least 1 per core)
  int64_t multicore_axis_size = 1;
  if (min_shape % SafeDivisor(static_cast<int64_t>(hardware.num_core_)) == 0) {
    multicore_axis_size = min_shape / SafeDivisor(static_cast<int64_t>(hardware.num_core_));
  }

  if (model_graph_out_size > 1) {
    multicore_axis_size = Get2PowerLess(multicore_axis_size);
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

  if (min_shape < kMinShapeThreshold) {
    multicore_axis_size = min_shape;
  }

  return multicore_axis_size;
}

int64_t GetMcAxisSize(Hardware hardware, int64_t min_shape, int data_coef) {
  // Initial Size for multicore (at least 1 per core)
  int64_t multicore_axis_size = 1;

  auto num_core_vblock_per_datatype =
    static_cast<int64_t>(static_cast<int>(hardware.num_core_ * hardware.vblocksize_) / SafeDivisor(data_coef));
  if (min_shape >= num_core_vblock_per_datatype && min_shape > kSmallAxisSize) {
    multicore_axis_size = static_cast<int64_t>(static_cast<int>(hardware.vblocksize_) / SafeDivisor(data_coef));
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

void ExtendMulticoreAxisTile(Axis &axis, const ModelGraph &model_graph, Hardware hardware) {
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
    axis_result = Get2PowerLess(axis_result);
  }

  if (axis_result == 0) {
    return;
  }

  if (model_graph.dominant_category_ == Op::OpCategory::Injective) {
    int64_t prime_factors_prod = GetLowestPrimeFactorsProductBelow(axis.range_, curr_max_alloc_buffer);
    if (prime_factors_prod > axis.c0_tiling_) {
      axis.c0_tiling_ = prime_factors_prod;
      return;
    }
  }

  int axis_count = 0;
  for (auto &g_axis : ModelGraph::global_axis_vec_) {
    if (!g_axis.is_inner_) {
      ++axis_count;
    }
  }
  // Avoid communication overhead for single tilable and small sized axis.
  if (axis_count > 1 && axis.range_ > kMinShapeThreshold &&
      model_graph.dominant_category_ == Op::OpCategory::Transpose) {
    // Increase MC axis in transpose only if multiple of number of cores
    if (axis.range_ % static_cast<int64_t>(kNumCore) == 0) {
      int64_t mc_tile = axis.range_ / static_cast<int64_t>(kNumCore);
      while (mc_tile > curr_max_alloc_buffer) {
        mc_tile /= kByTwoL;
        if (axis.range_ % SafeDivisor(static_cast<int64_t>(mc_tile)) != 0) {
          return;
        }
      }
      axis.c0_tiling_ = mc_tile;
    }
    return;
  }

  int64_t tile_from_vec = GetTileFromRemainingVecGranularity(axis, data_coef, axis_result);
  if (tile_from_vec > axis.c0_tiling_) {
    axis.c0_tiling_ = tile_from_vec;
  } else if (axis_count == 1 || model_graph.dominant_category_ != Op::OpCategory::Transpose) {
    axis.c0_tiling_ = GetTileFromRemainingBuffer(axis, data_coef, axis_result, curr_max_alloc_buffer);
  }
}

int64_t GetTileFromRemainingVecGranularity(const Axis &axis, int data_coef, int64_t axis_result) {
  int64_t axis_results_per_core = axis_result / static_cast<int64_t>(kNumCore);
  if (axis_results_per_core < 1) {
    // Avoid communication overhead on small sized axis.
    return axis_result > axis.c0_tiling_ ? axis_result : axis.c0_tiling_;
  }

  auto vec_granularity_per_data_coef = static_cast<int64_t>(kVectorizationGranularity / SafeDivisor(data_coef));
  int64_t multiplicity = axis_results_per_core / SafeDivisor(vec_granularity_per_data_coef);
  int64_t tile_from_vec_granularity = Get2PowerLessEq(multiplicity * vec_granularity_per_data_coef);
  return tile_from_vec_granularity > axis.c0_tiling_ ? tile_from_vec_granularity : axis.c0_tiling_;
}

int64_t GetTileFromRemainingBuffer(const Axis &axis, int data_coef, int64_t axis_result, int64_t max_alloc_buffer) {
  int64_t tile_from_buf = 0;
  int64_t axis_results_per_core = axis_result / static_cast<int64_t>(kNumCore);
  if (axis_results_per_core < 1) {
    tile_from_buf = max_alloc_buffer / SafeDivisor(axis_result) / SafeDivisor(data_coef);
  } else {
    tile_from_buf = max_alloc_buffer / SafeDivisor(axis_results_per_core) / SafeDivisor(data_coef);
  }
  if (tile_from_buf < 1 || axis_result < axis.c0_tiling_) {
    return axis.c0_tiling_;
  }
  return axis_result;
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
