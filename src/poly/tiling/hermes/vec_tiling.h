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

#include <memory>
#include <tuple>
#include <vector>

#include "poly/tiling/hermes/axis.h"
#include "poly/tiling/hermes/hardware.h"
#include "poly/tiling/hermes/model_graph.h"
#include "poly/tiling/hermes/node.h"

namespace akg {
namespace ir {
namespace poly {
int GetVecAxis(const Axis &axis, const ModelGraph &model_graph, Hardware hardware);
int GetMixTypeAxis(const Axis &axis, const ModelGraph &model_graph, Hardware hardware);
int GetTileFromRemainingVecGranularity(const Axis &global_axis, int data_coef, int axis_result);
size_t GetLastDimAxis();
size_t GetNewAlignMissFactor(size_t align_miss_factor, const Axis &critical_node_axis,
                             const std::vector<Axis> &global_axis_vec, int min_to_align);
void ExtendMulticoreAxisTile(int curr_max_alloc_buffer, int num_core, int axis_result, int data_coef, int mem_VC_size,
                             int mem_VC_align, const Axis &axis,
                             const std::vector<std::shared_ptr<Node>> &critical_node);
std::tuple<int, int> GetMaxAllocAndUpperBoundBuffer(int mem_VC_size, int mem_VC_align_, const Axis &axis,
                                                    const std::vector<std::shared_ptr<Node>> &critical_node);
std::tuple<float, float> GetTileAndMulticoreAxisSizes(const Axis &current_axis, const Axis &critical_node_axis,
                                                      const std::vector<Axis> &global_axis_vec, float tile_size,
                                                      float tile_multicore_axis_size);
bool PrioAxis(const Axis &axis, const ModelGraph &model_graph);

const int kBlocksNumForVectorization = 8;
const int kExtraMemoryCoeffForAllReduce = 16;
const int kExtraMemoryCoeffForReduceDst = 8;
const int kExtraMemoryCoeffForReduceSrc = 64;
const int kVectorizationGranularity = 128;
const int kSecondAxis = 2;
const float kMaxAllowedAllocPercentage = 0.9;
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_TILING_HERMES_VEC_TILING_H_
