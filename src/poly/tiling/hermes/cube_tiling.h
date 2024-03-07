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
#ifndef POLY_TILING_HERMES_CUBE_TILING_H_
#define POLY_TILING_HERMES_CUBE_TILING_H_

#include <memory>
#include <vector>

#include "poly/tiling/hermes/axis.h"
#include "poly/tiling/hermes/hardware.h"
#include "poly/tiling/hermes/model_graph.h"
#include "poly/tiling/hermes/node.h"

namespace akg {
namespace ir {
namespace poly {
int64_t GetMatmulAxisNTiling(const Axis &axis, const std::vector<std::shared_ptr<Node>> &nodes, size_t mem_VC_size);
int64_t GetMatmulAxisMTiling(const Axis &axis, const ModelGraph &model_graph, Hardware hardware);
int64_t GetMatmulAxisKTiling(Axis &axis, const ModelGraph &model_graph, Hardware hardware);
size_t GetMulticoreFromAxis(size_t remaining_core, int64_t axis_size);
size_t GetAvailableOutPutCacheSize(const std::vector<std::shared_ptr<Node>> &nodes, size_t mem_VC_size);

const size_t kC1DataSizePerBatch = 512;
const size_t kC0DataSizePerBatch = 512;
const size_t kVCDataSizePerBatch = 1024;
const size_t kMinNumOfUsedCores = 32;
const size_t kMinNumOfUsedCores910B = 24;
const int64_t kLargeTensorSize = 219360;
const int64_t kCubeDefaultTiling = 16;
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_TILING_HERMES_CUBE_TILING_H_
