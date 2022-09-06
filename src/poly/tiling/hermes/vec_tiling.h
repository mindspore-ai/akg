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
#ifndef POLY_TILING_HERMES_VEC_TILING_H_
#define POLY_TILING_HERMES_VEC_TILING_H_

#include "poly/tiling/hermes/axis.h"
#include "poly/tiling/hermes/hardware.h"
#include "poly/tiling/hermes/model_graph.h"

namespace akg {
namespace ir {
namespace poly {
int64_t GetVecAxis(const Axis &axis, const ModelGraph &model_graph, Hardware hardware);
int64_t GetMixTypeAxis(const Axis &axis, const ModelGraph &model_graph, Hardware hardware);
bool PrioAxis(const Axis &axis, const ModelGraph &model_graph);

const int kBlocksNumForVectorization = 8;
const int kSecondAxis = 2;
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_TILING_HERMES_VEC_TILING_H_
