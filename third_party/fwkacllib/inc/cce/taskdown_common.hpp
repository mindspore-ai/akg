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

#ifndef TASKDOWN_COMMON_H_
#define TASKDOWN_COMMON_H_

#include <map>
#include "cce/cce_def.hpp"
#include "common/attr_list.hpp"
#include "l2fusion_struct.hpp"

namespace cce {

#define CC_FUSION_OP_MAX 32

typedef enum tagccKernelType {
  CCE_AI_CORE = 0, /* cce aicore */
  CCE_AI_CPU = 1,  /* cce aicpu */
  TE = 2,          /* te operator*/
  CUSTOMIZED = 3,  /* customized operator */
  TE_AI_CORE = 4,  /* te aicore operator*/
  TE_AI_CPU = 5,   /* te aicpu operator */
  AI_CPU = 6,      /* aicpu */
  CUST_AI_CPU = 7, /* custom aicpu*/
  INVALID = 8,     /* unknown kernel type */
} ccKernelType;

typedef struct tagOpContext {
  ccKernelType kernelType;
  uint32_t opId;
  uint32_t kernelFuncId;
  uint32_t opIndex;
  uint32_t opCount;
  uint32_t opIndex2[CC_FUSION_OP_MAX];
  bool isFlowtable;
  uint16_t *argsOffset;
  uint32_t argsCount;
  uint64_t genDataBaseAddr;
  uint64_t genDataBaseSize;
  uint64_t genWeightBaseAddr;
  uint64_t genWeightBaseSize;
  uint64_t genVariableBaseAddr;
  uint64_t genVariableBaseSize;
  uint64_t l2ctrlSize;
} ccOpContext;

typedef struct tagOpReadCount {
  bool isEnable;
  std::map<uint64_t, uint32_t> tensorRc;
} ccOpReadCount;

typedef enum tagTaskDownKernelIdMode {
  CC_TASKDOWN_RESERVED = 0,
  CC_TASKDOWN_ROIPOOLING,
  CC_TASKDOWN_ROIPOOLING_PERF,
  CC_TASKDOWN_ROIALIGN,
  CC_TASKDOWN_ROIALIGN_PERF,
  CC_TASKDOWN_FC,
  CC_TASKDOWN_FC_COMPRESS,
  CC_TASKDOWN_SOFTMAX_LOWEST,
  CC_TASKDOWN_ROIALIGN_FP16,
  CC_TASKDOWN_RESIZE_NEAREST_NEIGHBOR,
  CC_TASKDOWN_RESIZE_NEAREST_NEIGHBOR_COMMON,
} ccTaskDownKernelIdMode_t;

ccStatus_t GetStream(ccHandle_t handle, rtStream_t *streamId);

ccStatus_t ccClearOpMap(ccHandle_t handle);

ccStatus_t ccSetKernelOpMap(ccHandle_t handle);

ccStatus_t ccSetKernelContext(ccHandle_t handle, uint32_t opId, AttrList &attrList, bool isFlowtable,
                              ccKernelType kernelType, void *pgraph);

ccStatus_t ccGetKernelContext(rtStream_t streamId, ccOpContext &opContext);

ccStatus_t ccGetKernelTypeByOpId(uint32_t opId, ccKernelType &kernelType);

ccStatus_t ccSetStreamL2Map(ccHandle_t handle, fusion::TaskL2InfoMap_t &l2AllocRes);

ccStatus_t ccGetStreamL2Map(rtStream_t streamId, uint32_t opIndex, fusion::TaskL2Info_t *&l2Data);

ccStatus_t ccSetOpIndex(ccHandle_t handle, uint32_t opIndex);

ccStatus_t ccGetOpIndex(ccHandle_t handle, uint32_t &opIndex);

ccStatus_t ccGetOpIndexByStream(rtStream_t streamId, uint32_t &opIndex);

ccStatus_t ccClearStreamL2Map(ccHandle_t handle);

ccStatus_t ccGetKernelReadCount(rtStream_t streamId, ccOpReadCount &rc);

}  // namespace cce
#endif  // TASKDOWN_COMMON_H_
