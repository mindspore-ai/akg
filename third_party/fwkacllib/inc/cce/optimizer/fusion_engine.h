/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef FUSION_ENGINE_HPP_
#define FUSION_ENGINE_HPP_

#include "cce/cce.h"
#include "graph/compute_graph.h"
#include "proto/task.pb.h"

#include <map>
#include <vector>

using namespace domi;
using namespace std;

namespace fusion {
enum {
  FUSION_STATUS_SUCCESS = 0,
  FUSION_STATUS_FAIL = 1,
};

typedef struct {
  uint64_t weightSize;
  uint64_t memorySize;
  uint8_t *dataMemBase;
  uint8_t *weightMemBase;
  uint32_t l2Enable;      // 1 //1 - enable l2 buffer allocation, 0 - disable l2 buffer allocation
  uint32_t fusionEnable;  // 1    // 1 - enable buffer fusion, 0 - disable buffer fusion
} ModelRes;

static const std::string SCOPE_ID_ATTR = "fusion_scope";
static const std::string L2FUSION_DYNAMIC_CONVERGE_OP = "l2fusion_dynamic_converge_op";
static const std::string L2FUSION_DYNAMIC_SPLIT_NUM = "l2fusion_dynamic_split_num";
static const std::string FUSION_VIRTUAL_OP = "fusion_virtual_op";
static const std::string FUSION_MULTI_BATCH_STRIDE = "fusion_multi_bathc_stride";

#define TVM_TYPE 1

typedef std::map<int64_t, std::vector<ge::NodePtr>> kScopeNodeMap_t;
typedef std::pair<int64_t, std::vector<ge::NodePtr>> kScopeNodePair_t;

uint32_t BufferFusion(ge::ComputeGraphPtr origGraph, ge::ComputeGraphPtr fusionGraph, bool enable_l2dynamic = true);
uint32_t BufferFusionTrain(ge::ComputeGraphPtr origGraph, ge::ComputeGraphPtr fusionGraph);
uint32_t GraphFusion(ge::ComputeGraphPtr origGraph, ge::ComputeGraphPtr fusionGraph);
uint32_t FusionTaskBuild(cce::ccHandle_t ccHandle, ge::ComputeGraphPtr fusionGraph, ge::Buffer &buffer,
                         ModelRes &modelRes, std::vector<TaskDef> &task_def_list_);
void FusionTaskBuildComplete(std::vector<cce::ccHandle_t> cchandleList);
uint32_t GraphFusionTrain(ge::ComputeGraphPtr origGraph, ge::ComputeGraphPtr fusionGraph);
}  // namespace fusion

#endif  // FUSION_ENGINE_HPP_
