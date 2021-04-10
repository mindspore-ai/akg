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

#ifndef AICPU_ENGINE_STRUCT_H__
#define AICPU_ENGINE_STRUCT_H__

#include "fwk_adpt_struct.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
    The different framwork we adapted for.
*/
typedef enum {
  FMK_KERNEL_TYPE_TF = 0,
  FMK_KERNEL_TYPE_CF = 10,
  FMK_KERNEL_TYPE_PT = 20,
  FMK_KERNEL_TYPE_RESERVED
} FwkkernelType_t;

#pragma pack(push, 1)
typedef struct {
  uint32_t fwkKernelType;  // FwkkernelType_t
  union {
    ::aicpu::FWKAdapter::FWKOperateParam fwk_kernel;
  } fwkKernelBase;
} STR_FWK_OP_KERNEL;
#pragma pack(pop)

#pragma pack(push, 1)
struct SessionInfo {
  uint64_t sessionId;
  uint64_t kernelId;
  bool sessFlag;
};
#pragma pack(pop)

#ifdef __cplusplus
}
#endif
#endif  // AICPU_ENGINE_STRUCT_H__
