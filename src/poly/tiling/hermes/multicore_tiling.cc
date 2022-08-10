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
#include <memory>

#include "poly/tiling/tiling_utils.h"
#include "poly/tiling/hermes/multicore_tiling.h"
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
}  // namespace poly
}  // namespace ir
}  // namespace akg
