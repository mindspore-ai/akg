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

#ifndef __MS_CUSTOM_OPS_CCSRC_UTILS_UTILS_H__
#define __MS_CUSTOM_OPS_CCSRC_UTILS_UTILS_H__

#include <cstdint>
#include <string>
#include <vector>
#include "mindspore/ops/kernel/include/common/kernel_tensor.h"

namespace ms_custom_ops {
// Helper function to convert optional tensor to tensor or empty tensor
inline ms::Tensor GetTensorOrEmpty(const std::optional<ms::Tensor> &opt_tensor) {
  return opt_tensor.has_value() ? opt_tensor.value() : ms::Tensor();
}

inline void *GetHostDataPtr(const ms::Tensor &tensor) {
  auto tensor_ptr = tensor.tensor();
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  auto &tensor_data = tensor_ptr->data();
  return tensor_data.data();
}

template <typename T, mindspore::TypeId DATA_TYPE>
T *GetRawPtr(const ms::Tensor &tensor, const std::string &op_name, const std::string &tensor_name) {
  if (tensor.data_type() != DATA_TYPE) {
    MS_LOG(EXCEPTION) << "For " << op_name << ", the data_type of " << tensor_name << " must be " << DATA_TYPE
                      << ", but got: " << tensor.data_type();
  }

  auto ptr = GetHostDataPtr(tensor);
  if (ptr == nullptr) {
    MS_LOG(EXCEPTION) << "For " << op_name << ", the data ptr of " << tensor_name << " can not be nullptr.";
  }
  return reinterpret_cast<T *>(ptr);
}

template <typename T, mindspore::TypeId DATA_TYPE>
inline std::vector<T> GetVectorFromTensor(const ms::Tensor &tensor, const std::string &op_name,
                                          const std::string &tensor_name) {
  auto vptr = GetRawPtr<T, DATA_TYPE>(tensor, op_name, tensor_name);
  return std::vector<T>(vptr, vptr + tensor.numel());
}

template <typename T>
T GetValueFromTensor(const ms::Tensor &tensor, const std::string &op_name, const std::string &tensor_name) {
  if constexpr (std::is_same_v<T, std::vector<int32_t>>) {
    return GetVectorFromTensor<int32_t, mindspore::kNumberTypeInt32>(tensor, op_name, tensor_name);
  }

  if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
    return GetVectorFromTensor<int32_t, mindspore::kNumberTypeInt64>(tensor, op_name, tensor_name);
  }

  MS_LOG(EXCEPTION) << "Not implemented. op_name: " << op_name << ", tensor_name: " << tensor_name
                    << ", type: " << typeid(T).name();
}
}  // namespace ms_custom_ops

#endif  // __MS_CUSTOM_OPS_CCSRC_UTILS_UTILS_H__
