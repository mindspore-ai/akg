/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_UTILS_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_UTILS_H_

#include "common/kernel.h"
#include <string>
#include <vector>

namespace mindspore {
namespace kernel {
bool GetSeqLenFromGraphAndCheckUpadate(
    const std::string &kernel_name,
    const std::vector<std::string> &tensor_name_list,
    std::vector<int32_t> *seq_len);
bool ConvertSeqLenToVectorAndCheckUpadate(
    KernelTensor *const actual_seq_length_ptr, std::vector<int32_t> *seq_len);
} // namespace kernel
} // namespace mindspore

#endif // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_UTILS_H_
