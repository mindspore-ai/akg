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
#ifndef POLY_TILING_HERMES_MULTICORE_TILING_H_
#define POLY_TILING_HERMES_MULTICORE_TILING_H_

#include <tuple>
#include <vector>

#include "poly/tiling/hermes/axis.h"
#include "poly/tiling/hermes/hardware.h"
#include "poly/tiling/hermes/model_graph.h"
#include "poly/tiling/hermes/tensor.h"

namespace akg {
namespace ir {
namespace poly {
int64_t GetMcAxis(const Axis &axis, const ModelGraph &model_graph, Hardware hardware);
int64_t GetMcReduceYAxisSize(Hardware hardware, int64_t min_shape, bool input_fp16, size_t model_graph_out_size);
int64_t GetMcAxisSize(Hardware hardware, int64_t min_shape, int data_coef);
std::tuple<float, float> GetTileAndMulticoreAxisSizes(const Axis &current_axis, const Axis &critical_node_axis,
                                                      const std::vector<Axis> &global_axis_vec_, float curr_tile_size,
                                                      float curr_tile_multicore_axis_size);
void ExtendMulticoreAxisTile(Axis &axis, const ModelGraph &model_graph, Hardware hardware);
int64_t GetTileFromRemainingBuffer(const Axis &axis, int data_coef, int64_t axis_result, int64_t max_alloc_buffer);
int64_t GetTileFromRemainingVecGranularity(const Axis &axis, int data_coef, int64_t axis_result);

const int64_t kSmallAxisSize = 300;
const int64_t kFloat16LoadGranularity = 128;
const int64_t kFloat32LoadGranularity = 64;
const int64_t kMinShapeThreshold = 64;
const int kVectorizationGranularity = 128;
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_TILING_HERMES_MULTICORE_TILING_H_
