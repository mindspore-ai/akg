/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
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

#ifndef AICPU_EVENT_STRUCT_H
#define AICPU_EVENT_STRUCT_H


#include <cstdint>
#include "ascend_hal.h"

namespace aicpu {

struct HwtsCceKernel {
    uint64_t  kernelName;  // The cce op kernel  function name in kernel so which want to be called
    uint64_t  kernelSo;    // The so file which contains cce op kernel function
    uint64_t  paramBase;   // The param tranmit to cce op kernel
    uint64_t  l2VaddrBase; // The param tranmit to cce op kernel
    uint32_t  blockId;     // The param tranmit to cce op kernel
    uint32_t  blockNum;    // The param tranmit to cce op kernel
    uint32_t  l2Size;      // The page num used in l2 buffer
    uint32_t  l2InMain;    // The param tranmit to cce op kernel
};

struct HwtsFwkKernel {
    uint64_t kernel; // a pointer point to STR_FWK_OP_KERNEL
    uint32_t size;
};

struct HwtsTsKernel {
    uint32_t kernelType;
    union {
        HwtsCceKernel cceKernel;
        HwtsFwkKernel fwkKernel;
#ifndef CFG_SOC_PLATFORM_CLOUD
        struct hwts_ts_kernel hwtsKernel;
#endif
    } kernelBase;
};

enum KernelType {
    KERNEL_TYPE_CCE = 0,
    KERNEL_TYPE_FWK = 1,
    KERNEL_TYPE_AICPU = 2,
    KERNEL_TYPE_AICPU_CUSTOM = 4,
    KERNEL_TYPE_HWTS = 10,
    KERNEL_TYPE_RESERVED
};

}  // namespace

#endif
