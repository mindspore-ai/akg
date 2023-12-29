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

#include "poly/tiling/hermes/cube_tiling.h"
#include "poly/tiling/hermes/classify_axis.h"
#include "poly/tiling/hermes/multicore_tiling.h"
#include "poly/tiling/hermes/tiling_algo.h"
#include "poly/tiling/hermes/tiling_mem.h"
#include "poly/tiling/hermes/vec_tiling.h"

namespace akg {
namespace ir {
namespace poly {
void GetTilingSize(ModelGraph &model_graph, Hardware hardware) {
  // classify Axis for op nodes
  for (auto const &node : model_graph.nodes_) {
    // should only work for op
    if (node->op_.IsInput()) {
      continue;
    }
    ClassifyAxis(*node);
  }

  // extra inner (16) axis treatment
  for (auto &axis : ModelGraph::global_axis_vec_) {
    if (axis.is_inner_) {
      axis.c0_tiling_ = axis.range_;
    }
  }

  // compute tile size
  for (auto &axis : ModelGraph::global_axis_vec_) {
    if (!axis.is_inner_) {
      axis.c0_tiling_ = GetAxis(axis, model_graph, hardware);
    }
  }

  // extend axis
  for (auto &axis : ModelGraph::global_axis_vec_) {
    if (!axis.is_inner_ && axis.type_.count(Axis::AxisLabel::kMatMulAxisBatch) == 0 &&
        axis.type_.count(Axis::AxisLabel::kMatMulAxisN) == 0 && axis.type_.count(Axis::AxisLabel::kMatMulAxisM) == 0 &&
        axis.type_.count(Axis::AxisLabel::kMatMulAxis16) == 0 && axis.type_.count(Axis::AxisLabel::kMatMulAxisK) == 0) {
      if (axis.type_.find(Axis::AxisLabel::kVectorization) != axis.type_.end()) {
        ExtendVecAxisTile(axis, model_graph, hardware);
      } else if (axis.type_.find(Axis::AxisLabel::kMultiCore) != axis.type_.end()) {
        ExtendMulticoreAxisTile(axis, model_graph, hardware);
      }
    }
  }
}

int64_t GetAxis(Axis &axis, const ModelGraph &model_graph, Hardware hardware) {
  if (axis.type_.find(Axis::AxisLabel::kMatMulAxisBatch) != axis.type_.end()) {
    // AKG limitation : the batch-axis can only tile by 1.
    return 1;
  }
  if (axis.type_.find(Axis::AxisLabel::kMatMulAxisN) != axis.type_.end()) {
    return GetMatmulAxisNTiling(axis, model_graph.nodes_, hardware.mem_VC_size_);
  }
  if (axis.type_.find(Axis::AxisLabel::kMatMulAxisM) != axis.type_.end()) {
    return GetMatmulAxisMTiling(axis, model_graph, hardware);
  }
  if (axis.type_.find(Axis::AxisLabel::kMatMulAxis16) != axis.type_.end()) {
    // AKG limitation : the 16-axis can only tile by 16.
    return kCubeDefaultTiling;
  }
  if (axis.type_.find(Axis::AxisLabel::kMatMulAxisK) != axis.type_.end()) {
    return GetMatmulAxisKTiling(axis, model_graph, hardware);
  }
  if (axis.type_.find(Axis::AxisLabel::kMultiCore) != axis.type_.end() &&
      axis.type_.count(Axis::AxisLabel::kVectorization) == 0) {
    return GetMcAxis(axis, model_graph, hardware);
  }
  if (axis.type_.find(Axis::AxisLabel::kVectorization) != axis.type_.end() &&
      axis.type_.count(Axis::AxisLabel::kMultiCore) == 0) {
    return GetVecAxis(axis, model_graph, hardware);
  }
  return GetMixTypeAxis(axis, model_graph, hardware);
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
