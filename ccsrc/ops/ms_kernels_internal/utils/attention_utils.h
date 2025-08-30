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

#ifndef __MS_CUSTOM_OPS_CCSRC_OPS_MS_KERNELS_INTERNAL_UTILS_ATTENTION_UTILS_H__
#define __MS_CUSTOM_OPS_CCSRC_OPS_MS_KERNELS_INTERNAL_UTILS_ATTENTION_UTILS_H__

#include <cstdint>
#include <string>
#include <vector>
#include "mindspore/ccsrc/include/runtime/hardware_abstract/kernel_base/kernel_tensor.h"

namespace ms_custom_ops {
inline bool GetSeqLenAndCheckUpdate(mindspore::kernel::KernelTensor *tensor, std::vector<int32_t> *seq_len) {
  auto new_value = tensor->GetValueWithCheck<std::vector<int32_t>>();
  bool is_need_update = false;
  if (seq_len->size() != new_value.size()) {
    is_need_update = true;
  } else {
    for (size_t i = 0; i < new_value.size(); i++) {
      if ((*seq_len)[i] != new_value[i]) {
        is_need_update = true;
        break;
      }
    }
  }
  if (is_need_update) {
    seq_len->clear();
    for (size_t i = 0; i < new_value.size(); i++) {
      seq_len->emplace_back(new_value[i]);
    }
  }

  return is_need_update;
}
}  // namespace ms_custom_ops

#endif  // __MS_CUSTOM_OPS_CCSRC_OPS_MS_KERNELS_INTERNAL_UTILS_ATTENTION_UTILS_H__
