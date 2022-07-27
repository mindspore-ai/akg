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

#include <vector>

#include "poly/tiling/hermes/axis.h"
#include "poly/tiling/hermes/hardware.h"
#include "poly/tiling/hermes/model_graph.h"
#include "poly/tiling/hermes/tensor.h"

namespace akg {
namespace ir {
namespace poly {
int GetMcAxis(const Axis &axis, const ModelGraph &model_graph, Hardware hardware);
int GetMcReduceYAxisSize(Hardware hardware, int min_shape, bool input_fp16, int model_graph_out_size);
int GetMcAxisSize(Hardware hardware, int min_shape, int data_coef);

const int kSmallAxisSize = 300;
const int kFloat16LoadGranularity = 128;
const int kFloat32LoadGranularity = 64;
const int kReduceMinShapeThreshold = 64;
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_TILING_HERMES_MULTICORE_TILING_H_
