/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef SRC_RUNTIME_ASCEND_RUNTIME_ERROR_CODES_H_
#define SRC_RUNTIME_ASCEND_RUNTIME_ERROR_CODES_H_

#include <string>
#include "external/runtime/rt_error_codes.h"
namespace air {
namespace runtime {
std::string GetErrorMsg(uint32_t rt_error_code);
}  // namespace runtime
}  // namespace air
#endif  // MINDSPORE_CCSRC_UTILS_RUNTIME_ERROR_CODES_H_
