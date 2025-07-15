
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

#include "internal_kernel_utils.h"
#include <string>

#include "utils/llm_manager.h"

namespace mindspore {
namespace kernel {
inline void SplitStringToNum(const std::string &str, char delim,
                             std::vector<int32_t> *output_list) {
  std::stringstream ss(str);
  std::string item;
  while (std::getline(ss, item, delim)) {
    if (!item.empty() && std::all_of(item.begin(), item.end(), ::isdigit)) {
      (void)output_list->emplace_back(std::stoi(item));
    }
  }
}

bool GetSeqLenFromGraphAndCheckUpadate(
    const std::string &kernel_name,
    const std::vector<std::string> &tensor_name_list,
    std::vector<int32_t> *seq_len) {
  auto &llm_manager = LLMManager::GetInstance();
  for (auto &tensor_name : tensor_name_list) {
    auto seq_length_tensor = llm_manager.get_graph_input(tensor_name);
    if (seq_length_tensor != nullptr) {
      // then use graph_input tensor value to set seq_len if saved
      auto seq_length_values =
          static_cast<int32_t *>(seq_length_tensor->data());
      auto seq_length_values_num =
          seq_length_tensor->nbytes() / sizeof(int32_t);

      bool is_need_update = false;
      if (seq_len->size() != seq_length_values_num) {
        is_need_update = true;
      } else {
        for (size_t i = 0; i < seq_length_values_num; i++) {
          if ((*seq_len)[i] != seq_length_values[i]) {
            is_need_update = true;
            break;
          }
        }
      }
      if (is_need_update) {
        seq_len->clear();
        for (size_t i = 0; i < seq_length_values_num; i++) {
          (*seq_len).emplace_back(seq_length_values[i]);
        }
      }
      MS_LOG(INFO) << "For op '" << kernel_name
                   << "', set param seq_len with graph_input '" << tensor_name
                   << "' as " << (*seq_len);
      return is_need_update;
    }
  }
  MS_LOG(INFO)
      << "For op '" << kernel_name
      << "', if custom op disabled, param seq_len must be set, but none of '"
      << tensor_name_list << "' is found in graph_input";
  if (seq_len->empty()) {
    return false;
  }
  seq_len->clear();
  return true;
}

bool ConvertSeqLenToVectorAndCheckUpadate(
    KernelTensor *const actual_seq_length_ptr, std::vector<int32_t> *seq_len) {
  MS_EXCEPTION_IF_NULL(actual_seq_length_ptr);
  std::vector<int32_t> actual_seq_lengths_vector;
  if (actual_seq_length_ptr->type_id() != kMetaTypeNone) {
    TypeId actual_seq_lengths_dtype_id = actual_seq_length_ptr->dtype_id();
    if (actual_seq_lengths_dtype_id == kNumberTypeInt64) {
      std::vector<int64_t> actual_seq_lengths_vector_64 =
          actual_seq_length_ptr->GetValueWithCheck<std::vector<int64_t>>();
      actual_seq_lengths_vector.assign(actual_seq_lengths_vector_64.begin(),
                                       actual_seq_lengths_vector_64.end());
    } else if (actual_seq_lengths_dtype_id == kNumberTypeInt32) {
      actual_seq_lengths_vector =
          actual_seq_length_ptr->GetValueWithCheck<std::vector<int32_t>>();
    } else {
      MS_LOG(EXCEPTION)
          << "actual_seq_lengths data type must be Int32 or Int64, but got "
          << TypeIdToString(actual_seq_lengths_dtype_id);
    }
  }

  if (actual_seq_lengths_vector == *seq_len) {
    return false;
  }
  seq_len->clear();
  for (const auto &item : actual_seq_lengths_vector) {
    (*seq_len).emplace_back(item);
  }
  return true;
}

} // namespace kernel
} // namespace mindspore
