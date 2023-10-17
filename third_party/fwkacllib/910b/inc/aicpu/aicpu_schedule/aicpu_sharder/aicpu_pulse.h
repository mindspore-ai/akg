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

#ifndef AICPU_PULSE_H
#define AICPU_PULSE_H

#include <cstdint>
#include "aicpu/common/type_def.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*PulseNotifyFunc)();

/**
 * aicpu pulse notify.
 * timer will call this method per second.
 */
void AicpuPulseNotify();

/**
 * Register kernel pulse notify func.
 * @param name name of kernel lib, must end with '\0' and unique.
 * @param func pulse notify function.
 * @return 0:success, other:failed.
 */
int32_t RegisterPulseNotifyFunc(const char_t * const name, const PulseNotifyFunc func);

/**
 * aicpu pulse notify.
 * call once when aicpu work end.
 */
void ClearPulseNotifyFunc();

#ifdef __cplusplus
}
#endif

#endif // AICPU_PULSE_H_
