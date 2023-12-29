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

#include "poly/tiling/hermes/model_graph.h"
#include "poly/tiling/hermes/multicore_tiling.h"
#include "poly/tiling/hermes/tiling_mem.h"
#include "poly/tiling/hermes/utils.h"

namespace akg {
namespace ir {
namespace poly {
std::tuple<int64_t, int64_t> GetMaxAllocAndUpperBoundBuffer(size_t mem_VC_size, size_t mem_VC_align, const Axis &axis,
                                                            const std::vector<std::shared_ptr<Node>> &critical_nodes) {
  auto available_mem_VC_size = static_cast<float>(mem_VC_size);
  auto min_available_mem_VC_size = static_cast<float>(mem_VC_size);
  float buffer_coef = 0;
  float max_buf_coef = 0;

  for (auto const &c_node : critical_nodes) {
    if (c_node->op_.op_type_ == Op::OpType::AllReduce) {
      available_mem_VC_size -=
        static_cast<float>(kByTwo * kExtraMemoryCoeffForReduceDst - kExtraMemoryCoeffForAllReduce);
      continue;
    }
    float tile_size = 1;
    float tile_multicore_axis_size = 1;
    auto c_node_out_datatype_coeff = static_cast<float>(c_node->transformed_output_shape_[0].GetDataTypeCoef());
    auto min_to_align = static_cast<int64_t>(static_cast<float>(mem_VC_align) / c_node_out_datatype_coeff);
    if (c_node->HasAxis(axis)) {
      int64_t align_miss_factor = 1;
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
        tile_size *= static_cast<float>(kExtraMemoryCoeffForReduceDst);
        tile_multicore_axis_size *= static_cast<float>(kExtraMemoryCoeffForReduceDst);
      }
      min_available_mem_VC_size -= std::round(tile_multicore_axis_size * c_node_out_datatype_coeff);
      available_mem_VC_size -= std::round(tile_size * c_node_out_datatype_coeff);
    }
  }

  if (buffer_coef <= 0) {
    buffer_coef = 1;
    max_buf_coef = 1;
  }
  auto max_alloc_buffer = static_cast<int64_t>(std::max(1.0F, available_mem_VC_size / buffer_coef));
  auto upper_bound_buffer = static_cast<int64_t>(std::max(1.0F, min_available_mem_VC_size / max_buf_coef));
  return std::make_tuple(max_alloc_buffer, upper_bound_buffer);
}

int64_t GetNewAlignMissFactor(int64_t curr_align_miss_factor, const Axis &critical_node_axis,
                              const std::vector<Axis> &global_axis_vec, int64_t min_to_align) {
  int64_t align_miss_factor = curr_align_miss_factor;
  for (auto const &axis : global_axis_vec) {
    if (critical_node_axis.dim_axis_ == axis.dim_axis_ && critical_node_axis.range_ == axis.range_ && axis.is_inner_) {
      align_miss_factor = std::max(align_miss_factor, min_to_align / critical_node_axis.range_);
      LOG(DEBUG) << "align_miss_factor: " << align_miss_factor << ". range: " << critical_node_axis.range_;
    }
  }

  return align_miss_factor;
}

int GetLastDimAxis() {
  int last_dim_axis = ModelGraph::global_axis_vec_.back().dim_axis_;
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
}  // namespace poly
}  // namespace ir
}  // namespace akg
