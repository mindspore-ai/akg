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

#include "internal_pyboost_runner.h"

using namespace ms_custom_ops;
namespace ms::pynative {
class ReshapeAndCacheRunner : public InternalPyboostRunner {
public:
  using InternalPyboostRunner::InternalPyboostRunner;

  void SetHeadNum(const int32_t &head_num) { this->head_num_ = head_num; }

protected:
  internal::InternalOpPtr
  CreateKernel(const internal::InputsImmutableInfoList &inputs,
               const internal::OutputsImmutableInfoList &outputs) {
    internal::ReshapeAndCacheParam param;
    param.head_num = this->head_num_;
    return internal::CreateReshapeAndCacheOp(
        inputs, outputs, param, internal::kInternalReshapeAndCacheOpName);
  }

  void LaunchKernel() {
    tensor::TensorPtrList inputs;
    inputs.reserve(5);

    for (const auto &input : this->inputs()) {
      inputs.push_back(input.is_defined() ? input.tensor() : nullptr);
    }

    tensor::TensorPtrList outputs;
    TransInternalShapes(inputs, outputs);
    LAUNCH_INTERNAL(_op_name_, this->_device_context_, this->stream_id(),
                    inputs, outputs);
  }

private:
  int32_t head_num_{0};
};
MS_KERNELS_INTERNAL_FACTORY_REG(ReshapeAndCache,
                                internal::kInternalReshapeAndCacheOpName);
} // namespace ms::pynative

namespace ms_custom_ops {
// Helper function to convert optional tensor to tensor or empty tensor
ms::Tensor GetTensorOrEmpty(const std::optional<ms::Tensor> &opt_tensor) {
  return opt_tensor.has_value() ? opt_tensor.value() : ms::Tensor();
}

// infer shape and type func
ms::Tensor GenResultTensor(const ms::Tensor &key) {
  return ms::Tensor(key.data_type(), key.shape());
}

ms::Tensor npu_reshape_and_cache(const ms::Tensor &key,
                                 const std::optional<ms::Tensor> &value,
                                 const std::optional<ms::Tensor> &key_cache,
                                 const std::optional<ms::Tensor> &value_cache,
                                 const std::optional<ms::Tensor> &slot_mapping,
                                 std::optional<int64_t> head_num) {
  auto result = GenResultTensor(key);
  auto op_name = "ReshapeAndCache";
  auto runner = std::make_shared<ms::pynative::ReshapeAndCacheRunner>(op_name);

  // Set head_num if provided
  if (head_num.has_value()) {
    runner->SetHeadNum(static_cast<int32_t>(head_num.value()));
  }

  // Convert ms::Tensor to TensorPtr for hash calculation
  auto key_tensor_ptr = key.tensor();
  auto get_tensor_ptr = [](const std::optional<ms::Tensor> &opt_tensor) {
    auto tensor = GetTensorOrEmpty(opt_tensor);
    return tensor.is_defined() ? tensor.tensor() : nullptr;
  };

  auto value_tensor_ptr = get_tensor_ptr(value);
  auto key_cache_tensor_ptr = get_tensor_ptr(key_cache);
  auto value_cache_tensor_ptr = get_tensor_ptr(value_cache);
  auto slot_mapping_tensor_ptr = get_tensor_ptr(slot_mapping);

  // Setup the runner with all parameters (including hash calculation)
  runner->Setup(op_name, key_tensor_ptr, value_tensor_ptr, key_cache_tensor_ptr,
                value_cache_tensor_ptr, slot_mapping_tensor_ptr, head_num);

  // Run the operation
  runner->Run({key, GetTensorOrEmpty(value), GetTensorOrEmpty(key_cache),
               GetTensorOrEmpty(value_cache), GetTensorOrEmpty(slot_mapping)},
              {result});
  return result;
}
} // namespace ms_custom_ops

auto pyboost_reshape_and_cache(const ms::Tensor &key,
                               const std::optional<ms::Tensor> &value,
                               const std::optional<ms::Tensor> &key_cache,
                               const std::optional<ms::Tensor> &value_cache,
                               const std::optional<ms::Tensor> &slot_mapping,
                               std::optional<int64_t> head_num) {
  return ms::pynative::PyboostRunner::Call<1>(
      ms_custom_ops::npu_reshape_and_cache, key, value, key_cache, value_cache,
      slot_mapping, head_num);
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("reshape_and_cache", &pyboost_reshape_and_cache, "Reshape And Cache",
        pybind11::arg("key"), pybind11::arg("value") = std::nullopt,
        pybind11::arg("key_cache") = std::nullopt,
        pybind11::arg("value_cache") = std::nullopt,
        pybind11::arg("slot_mapping") = std::nullopt,
        pybind11::arg("head_num") = std::nullopt);
}
