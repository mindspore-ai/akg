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
#include <cmath>

#include "poly/tiling/hermes/utils.h"
#include "poly/tiling/hermes/vec_tiling.h"

namespace akg {
namespace ir {
namespace poly {
int GetVecAxis(const Axis &axis, const ModelGraph &model_graph, Hardware hardware) {
  int axis_result = 1;

  int min_shape = axis.range_;
  int data_coef = 0;
  std::tie(std::ignore, data_coef) = model_graph.GetMinShapeAndDataCoef(axis);

  int max_alloc_buffer = 0;
  int upper_bound_buffer = 0;
  std::tie(max_alloc_buffer, upper_bound_buffer) =
    GetMaxAllocAndUpperBoundBuffer(hardware.mem_VC_size_, hardware.mem_VC_align_, axis, model_graph.critical_nodes_);

  axis_result = std::min(max_alloc_buffer, min_shape);
  if ((axis_result != min_shape) && (axis_result & (axis_result - 1)) != 0) {
    axis_result = Get2PowerBelow(axis_result);
  }

  size_t last_dim_axis = GetLastDimAxis();
  if (axis.dim_axis_ != last_dim_axis) {
    axis_result = std::min(min_shape, axis_result);
  }

  if (last_dim_axis - axis.dim_axis_ > 1) {
    if ((upper_bound_buffer != min_shape) && (upper_bound_buffer & (upper_bound_buffer - 1)) != 0) {
      upper_bound_buffer = Get2PowerBelow(upper_bound_buffer);
    }
    if (axis.is_innermost_) {
      int avg_tiling =
        static_cast<int>(std::pow(max_alloc_buffer, 1.0 / static_cast<double>((last_dim_axis - axis.dim_axis_ + 1))));
      avg_tiling = (avg_tiling & (avg_tiling - 1)) != 0 ? Get2PowerBelow(avg_tiling) : avg_tiling;
      avg_tiling = std::max(avg_tiling,
                            upper_bound_buffer);  // no need to reduce the tiling if there is enough space to fully tile
      axis_result = std::min(axis_result, avg_tiling);
    } else {
      axis_result = std::min(upper_bound_buffer, axis_result);
    }
  }

  if (axis.dim_axis_ == last_dim_axis) {
    ExtendMulticoreAxisTile(max_alloc_buffer, hardware.num_core_, axis_result, data_coef, hardware.mem_VC_size_,
                            hardware.mem_VC_align_, axis, model_graph.critical_nodes_);

    int global_axis_size = 0;
    int second_last_global_axis = 0;
    for (size_t idx_global_axis = ModelGraph::global_axis_vec_.size(); idx_global_axis > 0; --idx_global_axis) {
      if (!ModelGraph::global_axis_vec_[idx_global_axis - 1].is_inner_) {
        global_axis_size++;
        if (global_axis_size == kSecondAxis) {
          second_last_global_axis = static_cast<int>(idx_global_axis) - 1;
        }
      }
    }

    auto penultimate_axis = ModelGraph::global_axis_vec_[second_last_global_axis];
    bool is_penult_label_vec = penultimate_axis.type_.count(Axis::AxisLabel::kVectorization) != 0;
    if (is_penult_label_vec &&
        (axis_result < penultimate_axis.tile_ ||
         (!penultimate_axis.is_innermost_ &&
          (penultimate_axis.tile_ * axis.range_ > axis_result * penultimate_axis.range_) &&
          penultimate_axis.tile_ > hardware.vblocknum_)) &&
        axis_result < min_shape && static_cast<unsigned int>(axis_result) < axis.range_) {
      ModelGraph::global_axis_vec_[second_last_global_axis].tile_ = Get2PowerBelow(penultimate_axis.tile_);
      axis_result = GetVecAxis(axis, model_graph, hardware);
    }
  }

  return axis_result;
}

size_t GetLastDimAxis() {
  size_t last_dim_axis = ModelGraph::global_axis_vec_.back().dim_axis_;
  for (int idx_global_axis = static_cast<int>(ModelGraph::global_axis_vec_.size()) - 1; idx_global_axis > 0;
       idx_global_axis--) {
    if (ModelGraph::global_axis_vec_[idx_global_axis].is_inner_) {
      last_dim_axis = ModelGraph::global_axis_vec_[idx_global_axis - 1].dim_axis_;
    } else {
      break;
    }
  }
  return last_dim_axis;
}

void ExtendMulticoreAxisTile(int curr_max_alloc_buffer, int num_core, int axis_result, int data_coef, int mem_VC_size,
                             int mem_VC_align, const Axis &axis,
                             const std::vector<std::shared_ptr<Node>> &critical_nodes) {
  if (axis_result == 0) {
    return;
  }

  int remaining_tiling = curr_max_alloc_buffer / axis_result;
  if (remaining_tiling <= 1) {
    return;
  }

  if (num_core == 0) {
    LOG(WARNING) << "Number of cores is 0!";
    return;
  }

  for (auto iter_axis = ModelGraph::global_axis_vec_.rbegin() + 1; iter_axis != ModelGraph::global_axis_vec_.rend();
       ++iter_axis) {
    int available_tiling = static_cast<int>(iter_axis->range_) / iter_axis->tile_;
    if (available_tiling > 1) {
      size_t num_mc_axis = iter_axis->type_.count(Axis::AxisLabel::kMultiCore);
      size_t num_vec_axis = iter_axis->type_.count(Axis::AxisLabel::kVectorization);

      for (int i = remaining_tiling; i > 0; --i) {
        if (available_tiling % i == 0) {
          size_t axis_range = iter_axis->range_;
          int axis_tile = iter_axis->tile_;
          if (available_tiling > num_core &&
              ((num_mc_axis != 0 && available_tiling / num_core < i) ||
               axis_range % static_cast<size_t>(i * axis_tile) != 0 || (i * axis_tile) % num_core != 0)) {
            continue;
          }
          int old_tiling = iter_axis->tile_;
          iter_axis->tile_ *= i;
          int max_alloc_buffer = 0;
          std::tie(max_alloc_buffer, std::ignore) =
            GetMaxAllocAndUpperBoundBuffer(mem_VC_size, mem_VC_align, axis, critical_nodes);
          if (max_alloc_buffer * kMaxAllowedAllocPercentage < axis_result) {
            iter_axis->tile_ = old_tiling;
          } else {
            remaining_tiling /= i;
            break;
          }
        }
      }

      if (num_mc_axis != 0 && num_vec_axis == 0) {
        iter_axis->tile_ = GetTileFromRemainingVecGranularity(*iter_axis, data_coef, axis_result);
      }
    }
  }
}

int GetTileFromRemainingVecGranularity(const Axis &global_axis, int data_coef, int axis_result) {
  int remaining_vec_granularity = 0;
  if (data_coef > 0 && axis_result > 0) {
    remaining_vec_granularity = kVectorizationGranularity / data_coef / axis_result;
  }
  if ((remaining_vec_granularity & (remaining_vec_granularity - 1)) != 0) {
    remaining_vec_granularity = Get2PowerBelow(remaining_vec_granularity);
  }
  if (remaining_vec_granularity < 1) {
    remaining_vec_granularity = 1;
  }
  if (global_axis.range_ / global_axis.tile_ / kNumCore / remaining_vec_granularity > 1) {
    return global_axis.tile_ * remaining_vec_granularity;
  }
  return global_axis.tile_;
}

std::tuple<int, int> GetMaxAllocAndUpperBoundBuffer(int mem_VC_size, int mem_VC_align, const Axis &axis,
                                                    const std::vector<std::shared_ptr<Node>> &critical_nodes) {
  int available_mem_VC_size = mem_VC_size;
  int min_available_mem_VC_size = mem_VC_size;
  float buffer_coef = 0;
  float max_buf_coef = 0;

  for (auto const &c_node : critical_nodes) {
    if (c_node->op_.op_type_ == Op::OpType::AllReduce) {
      available_mem_VC_size -= kByTwo * kExtraMemoryCoeffForReduceDst - kExtraMemoryCoeffForAllReduce;
      continue;
    }
    float tile_size = 1;
    float tile_multicore_axis_size = 1;
    auto c_node_out_datatype_coeff = static_cast<float>(c_node->transformed_output_shape_[0].GetDataTypeCoef());
    int min_to_align = static_cast<int>(static_cast<float>(mem_VC_align) / c_node_out_datatype_coeff);
    if (c_node->HasAxis(axis)) {
      size_t align_miss_factor = 1;
      for (auto const &c_axis : c_node->axis_of_node_) {
        align_miss_factor =
          GetNewAlignMissFactor(align_miss_factor, c_axis, ModelGraph::global_axis_vec_, min_to_align);
        if (c_axis.dim_axis_ != axis.dim_axis_) {
          std::tie(tile_size, tile_multicore_axis_size) = GetTileAndMulticoreAxisSizes(
            axis, c_axis, ModelGraph::global_axis_vec_, tile_size, tile_multicore_axis_size);
        }
      }
      if (c_node->op_.op_type_ == Op::OpType::ReduceSRC) {
        tile_size /= kExtraMemoryCoeffForReduceSrc;
        tile_multicore_axis_size /= kExtraMemoryCoeffForReduceSrc;
      }
      buffer_coef += tile_size * c_node_out_datatype_coeff * static_cast<float>(align_miss_factor);
      max_buf_coef += tile_multicore_axis_size * c_node_out_datatype_coeff * static_cast<float>(align_miss_factor);
    } else {
      for (auto const &c_axis : c_node->axis_of_node_) {
        std::tie(tile_size, tile_multicore_axis_size) =
          GetTileAndMulticoreAxisSizes(axis, c_axis, ModelGraph::global_axis_vec_, tile_size, tile_multicore_axis_size);
      }
      if (c_node->op_.op_type_ == Op::OpType::ReduceDST) {
        tile_size *= kExtraMemoryCoeffForReduceDst;
        tile_multicore_axis_size *= kExtraMemoryCoeffForReduceDst;
      }
      min_available_mem_VC_size -= static_cast<int>(std::round(tile_multicore_axis_size * c_node_out_datatype_coeff));
      available_mem_VC_size -= static_cast<int>(std::round(tile_size * c_node_out_datatype_coeff));
    }
  }

  if (buffer_coef <= 0) {
    buffer_coef = 1;
    max_buf_coef = 1;
  }
  int max_alloc_buffer = std::max(1, static_cast<int>(static_cast<float>(available_mem_VC_size) / buffer_coef));
  int upper_bound_buffer = std::max(1, static_cast<int>(static_cast<float>(min_available_mem_VC_size) / max_buf_coef));
  return std::make_tuple(max_alloc_buffer, upper_bound_buffer);
}

size_t GetNewAlignMissFactor(size_t curr_align_miss_factor, const Axis &critical_node_axis,
                             const std::vector<Axis> &global_axis_vec, int min_to_align) {
  size_t align_miss_factor = curr_align_miss_factor;
  for (auto const &axis : global_axis_vec) {
    if (critical_node_axis.dim_axis_ == axis.dim_axis_ && critical_node_axis.range_ == axis.range_ && axis.is_inner_) {
      align_miss_factor = std::max(align_miss_factor, min_to_align / critical_node_axis.range_);
      LOG(DEBUG) << "align_miss_factor: " << align_miss_factor << ". range: " << critical_node_axis.range_;
    }
  }

  return align_miss_factor;
}

std::tuple<float, float> GetTileAndMulticoreAxisSizes(const Axis &current_axis, const Axis &critical_node_axis,
                                                      const std::vector<Axis> &global_axis_vec_, float curr_tile_size,
                                                      float curr_tile_multicore_axis_size) {
  float tile_size = curr_tile_size;
  float tile_multicore_axis_size = curr_tile_multicore_axis_size;
  for (auto const &axis : global_axis_vec_) {
    if (critical_node_axis.dim_axis_ == axis.dim_axis_) {
      tile_size *= static_cast<float>(axis.tile_);
      if (axis.dim_axis_ > current_axis.dim_axis_) {
        tile_multicore_axis_size *= static_cast<float>(axis.range_);
      } else {
        tile_multicore_axis_size *= static_cast<float>(axis.tile_);
      }
    }
  }
  return std::make_tuple(tile_size, tile_multicore_axis_size);
}

int GetMixTypeAxis(const Axis &axis, const ModelGraph &model_graph, Hardware hardware) {
  int axis_result = 1;

  std::vector<std::shared_ptr<Node>> nodes = model_graph.nodes_;

  int min_shape = axis.range_;
  int data_coef = 0;
  std::tie(std::ignore, data_coef) = model_graph.GetMinShapeAndDataCoef(axis);

  int multi_core_size = min_shape;
  if (hardware.num_core_ > 0) {
    multi_core_size /= hardware.num_core_;
  }
  int vec_cal_size = GetVecAxis(axis, model_graph, hardware);
  int vec_granularity = 0;
  if (data_coef > 0) {
    vec_granularity = hardware.vblocknum_ * hardware.vblocksize_ / data_coef;
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
      if (multi_core_size < vec_granularity * hardware.vblocknum_) {
        hardware.num_core_ = min_shape / vec_cal_size;
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
        hardware.num_core_ = min_shape / vec_granularity;
        axis_result = std::min(multi_core_size, vec_cal_size);
      }
    } else {
      if (axis.name_ != ModelGraph::global_axis_vec_[0].name_ &&
          (int)ModelGraph::global_axis_vec_[0].range_ / ModelGraph::global_axis_vec_[0].tile_ >= hardware.num_core_) {
        axis_result = vec_cal_size;
      } else {
        multi_core_size = hardware.vblocksize_ / data_coef;
        axis_result = std::min(multi_core_size, vec_cal_size);
      }
    }
  }

  if (axis_result != min_shape && (axis_result & (axis_result - 1)) != 0) {
    axis_result = Get2PowerBelow(axis_result);
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
}  // namespace poly
}  // namespace ir
}  // namespace akg
