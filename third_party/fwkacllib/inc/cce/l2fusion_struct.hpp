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

#ifndef L2FUSION_STRUCT_HPP_
#define L2FUSION_STRUCT_HPP_

#include <map>
#include <string>
#include "runtime/kernel.h"

#define L2_DYNAMIC_SPLIT_NUM

using namespace std;

namespace fusion {

typedef struct tagL2Data {
  uint32_t l2Index;
  uint64_t l2Addr;
  uint64_t l2PageNum;
} L2Data_t;

typedef std::map<uint64_t, L2Data_t> L2DataMap_t;    // the key is ddr addr
typedef std::pair<uint64_t, L2Data_t> L2DataPair_t;  // the key is ddr addr

typedef struct TagTaskL2Info {
  string nodeName;
  rtL2Ctrl_t l2ctrl;

  L2DataMap_t input;
  L2DataMap_t output;
  uint32_t isUsed;
} TaskL2Info_t;

typedef std::map<uint32_t, TaskL2Info_t> TaskL2InfoMap_t;    // the key is nodeId
typedef std::pair<uint32_t, TaskL2Info_t> TaskL2InfoPair_t;  // the key is nodeId

typedef std::map<string, TaskL2Info_t> TaskL2InfoFEMap_t;    // the key is nodeName
typedef std::pair<string, TaskL2Info_t> TaskL2InfoFEPair_t;  // the key is nodeName

}  // namespace fusion

#endif  // L2FUSION_STRUCT_HPP_
