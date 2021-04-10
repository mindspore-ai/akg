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

#ifndef __CCE_RUNTIME_DVFSPROFILE_H__
#define __CCE_RUNTIME_DVFSPROFILE_H__

#include "base.h"

#if defined(__cplusplus) && !defined(COMPILE_OMG_PACKAGE)
extern "C" {
#endif

typedef enum dvfsProfileMode {
  DVFS_PROFILE_PERFORMANCE_PRIORITY,
  DVFS_PROFILE_BALANCE_PRIORITY,
  DVFS_PROFILE_POWER_PRIORITY,
  DVFS_PROFILE_PRIORITY_MAX
} DvfsProfileMode;

/**
 * @ingroup dvrt_dvfsprofile
 * @brief Set the performance mode of the device
 * @param [in] mode   dvfsProfileMode
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtSetDvfsProfile(DvfsProfileMode mode);

/**
 * @ingroup dvrt_dvfsprofile
 * @brief Set the performance mode of the device
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for invalid value
 */
RTS_API rtError_t rtUnsetDvfsProfile();

/**
 * @ingroup dvrt_dvfsprofile
 * @brief Get the current performance mode of the device
 * @param [in|out] pmode   dvfsProfileMode type pointer
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtGetDvfsProfile(DvfsProfileMode *pmode);

#if defined(__cplusplus) && !defined(COMPILE_OMG_PACKAGE)
}
#endif

#endif  // __CCE_RUNTIME_PROFILE_H__
