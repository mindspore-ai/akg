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

#ifndef AICPU_ENGINE_H__
#define AICPU_ENGINE_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  AE_STATUS_SUCCESS = 0,
  AE_STATUS_BAD_PARAM = 1,
  AE_STATUS_OPEN_SO_FAILED = 2,
  AE_STATUS_GET_KERNEL_NAME_FAILED = 3,
  AE_STATUS_INNER_ERROR = 4,
  AE_STATUS_KERNEL_API_INNER_ERROR = 5,
  AE_STATUS_END_OF_SEQUENCE = 6,
  AE_STATUS_DUMP_FAILED = 7,
  AE_STATUS_RESERVED
} aeStatus_t;

/**
 * @ingroup aicpu engine
 * @brief aeCallInterface:
 *          a interface to call a function in a op kernfel lib
 * @param [in] addr     void *,  should be STR_KERNEL * format
 * @return aeStatus_t
 */
aeStatus_t aeCallInterface(void *addr);

/**
 * @ingroup aicpu engine
 * @brief aeBatchLoadKernelSo:
 *          a interface to load kernel so
 * @param [in] loadSoNum  load so number
 * @param [in] soPaths    load so paths
 * @param [in] soNames    load so names
 * @return aeStatus_t
 */
aeStatus_t aeBatchLoadKernelSo(const uint32_t loadSoNum, const char *soPaths[], const char *soNames[]);

#ifdef __cplusplus
}
#endif

#endif  // AICPU_ENGINE_H__
