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

#ifndef TASKDOWN_API_H_
#define TASKDOWN_API_H_

#include <map>
#include <vector>
#include "cce/cce.h"
#include "l2fusion_struct.hpp"
#include "taskdown_common.hpp"

namespace cce {

#define CC_FUSION_OP_MAX 32

typedef struct tagOpAddrsInfo {
  void *addrPos;
  uintptr_t addrData;
} ccOpAddrsInfo;

#ifdef __cplusplus
extern "C" {
#endif

ccStatus_t ccUpdateKernelArgs(ccOpContext &opContext, uint64_t dataBaseAddr, uint64_t weightBaseAddr,
                              uint64_t variableBaseAddr, void *argsAddr, uint64_t argsSize, void *l2ctrlAddr);

#ifdef __cplusplus
}
#endif

ccStatus_t ccGetKernelArgsAddrs(ccOpContext &opContext, void *argsAddr, uint64_t argsSize, void *l2ctrlAddr,
                                std::vector<ccOpAddrsInfo> &opAddrsInfo);

ccStatus_t ccSetKernelArgs(std::vector<ccOpAddrsInfo> &dateInfo);

ccStatus_t ccGetKernelTypeByOpId(uint32_t opId, ccKernelType &kernelType);

}  // namespace cce
#endif  // TASKDOWN_API_H_
