/**
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2019-2021. All rights reserved.
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

#ifndef INC_TDT_STATUS_H
#define INC_TDT_STATUS_H
#include "common/type_def.h"
namespace tsd {
#ifdef __cplusplus
    using TSD_StatusT = uint32_t;
#else
    typedef uint32_t TSD_StatusT;
#endif
    // success code
    constexpr TSD_StatusT TSD_OK = 0U;
    // 区分TSD不对外的枚举，对外枚举ID从100开始
    enum ErroCodeExt : TSD_StatusT {
        TSD_SUBPROCESS_NUM_EXCEED_THE_LIMIT = 100U,
        TSD_SUBPROCESS_BINARY_FILE_DAMAGED = 101U,
        TSD_DEVICE_DISCONNECTED = 102U,
        TSD_VERIFY_OPP_FAIL = 103U,
        TSD_ADD_AICPUSD_TO_CGROUP_FAILED = 104U,
        TSD_OPEN_NOT_SUPPORT_FOR_MDC = 200U,
    };
}
#endif  // INC_TDT_STATUS_H
