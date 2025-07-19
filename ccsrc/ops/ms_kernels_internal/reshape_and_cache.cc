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

// =============================================================================
// GRAPH MODE IMPLEMENTATION
// =============================================================================

#include "internal_kernel_mod.h"
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_convert.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ms_extension/api.h"
#include "ops/base_operator.h"
#include "ops/ops_func_impl/op_func_impl.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "runtime/device/kernel_runtime.h"
#include "utils/check_convert_utils.h"
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace mindspore {
namespace ops {
class OPS_API CustomReshapeAndCacheOpFuncImpl : public OpFuncImpl {
public:
  ShapeArray InferShape(const PrimitivePtr &primitive,
                        const InferInfoPtrList &input_infos) const override {
    return {input_infos[0]->GetShape()};
  }
  std::vector<TypeId>
  InferType(const PrimitivePtr &primitive,
            const InferInfoPtrList &input_infos) const override {
    return {input_infos[0]->GetType()};
  }

  bool GeneralInferRegistered() const override { return true; }
};
} // namespace ops
} // namespace mindspore

namespace ms_custom_ops {
constexpr size_t kInputKeyIndex = 0;
constexpr size_t kInputValueIndex = 1;
constexpr size_t kInputKeyCacheIndex = 2;
constexpr size_t kInputValueCacheIndex = 3;
constexpr size_t kInputSlotMappingIndex = 4;
constexpr size_t kInputHeadNumIndex = 5;
constexpr size_t kOutputIndex = 0;
class CustomReshapeAndCache : public InternalKernelMod {
public:
  CustomReshapeAndCache() : InternalKernelMod() {}
  ~CustomReshapeAndCache() = default;

  void InitKernelInputsOutputsIndex() override {
    kernel_inputs_index_ = {kInputKeyIndex, kInputValueIndex, kInputKeyCacheIndex,
                            kInputValueCacheIndex, kInputSlotMappingIndex};
    kernel_outputs_index_ = {kOutputIndex};
  }

protected:
  internal::InternalOpPtr
  CreateKernel(const internal::InputsImmutableInfoList &inputs,
               const internal::OutputsImmutableInfoList &outputs,
               const std::vector<KernelTensor *> &ms_inputs,
               const std::vector<KernelTensor *> &ms_outputs) override {
    internal::ReshapeAndCacheParam param;
    auto head_num = ms_inputs.at(internal::kIndex5);
    if (head_num->dtype_id() == TypeId::kNumberTypeInt64) {
      param.head_num =
          static_cast<int32_t>(head_num->GetValue<int64_t>().value());
    } else {
      MS_LOG(EXCEPTION)
          << "ReshapeAndCache [head_num]'s dtype wrong, expect int64, but got: "
          << head_num->dtype_id();
    }
    return internal::CreateReshapeAndCacheOp(
        inputs, outputs, param, internal::kInternalReshapeAndCacheOpName);
  }
};
} // namespace ms_custom_ops

MS_CUSTOM_OPS_REGISTER(reshape_and_cache, CustomReshapeAndCacheOpFuncImpl,
                       CustomReshapeAndCache);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

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
               const internal::OutputsImmutableInfoList &outputs) override {
    internal::ReshapeAndCacheParam param;
    param.head_num = this->head_num_;
    return internal::CreateReshapeAndCacheOp(
        inputs, outputs, param, internal::kInternalReshapeAndCacheOpName);
  }

private:
  int32_t head_num_{0};
};
MS_KERNELS_INTERNAL_NAME_REG(ReshapeAndCache,
                             internal::kInternalReshapeAndCacheOpName);
} // namespace ms::pynative

namespace ms_custom_ops {
// Helper function to convert optional tensor to tensor or empty tensor
ms::Tensor GetTensorOrEmpty(const std::optional<ms::Tensor> &opt_tensor) {
  return opt_tensor.has_value() ? opt_tensor.value() : ms::Tensor();
}

// infer shape and type func
// ms::Tensor GenResultTensor(const ms::Tensor &key) {
//   return ms::Tensor(key.data_type(), key.shape());
// }

void npu_reshape_and_cache(const ms::Tensor &key,
                           const std::optional<ms::Tensor> &value,
                           const std::optional<ms::Tensor> &key_cache,
                           const std::optional<ms::Tensor> &value_cache,
                           const std::optional<ms::Tensor> &slot_mapping,
                           std::optional<int64_t> head_num) {
  auto op_name = "ReshapeAndCache";
  auto runner = std::make_shared<ms::pynative::ReshapeAndCacheRunner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);

  // Set head_num if provided
  if (head_num.has_value()) {
    runner->SetHeadNum(static_cast<int32_t>(head_num.value()));
  }

  // Setup the runner with all parameters (including hash calculation)
  runner->Setup(op_name, key, value, key_cache, value_cache, slot_mapping,
                head_num);

  // if you need infer shape and type, you can use this
  // auto result = GenResultTensor(key);
  std::vector<ms::Tensor> inputs = {
      key, GetTensorOrEmpty(value), GetTensorOrEmpty(key_cache),
      GetTensorOrEmpty(value_cache), GetTensorOrEmpty(slot_mapping)};
  std::vector<ms::Tensor> outputs = {};
  runner->GetOrCreateKernel(inputs, outputs);
  runner->Run(inputs, outputs);
  return;
}
} // namespace ms_custom_ops

auto pyboost_reshape_and_cache(const ms::Tensor &key,
                               const std::optional<ms::Tensor> &value,
                               const std::optional<ms::Tensor> &key_cache,
                               const std::optional<ms::Tensor> &value_cache,
                               const std::optional<ms::Tensor> &slot_mapping,
                               std::optional<int64_t> head_num) {
  return ms::pynative::PyboostRunner::Call<0>(
      ms_custom_ops::npu_reshape_and_cache, key, value, key_cache, value_cache,
      slot_mapping, head_num);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("reshape_and_cache", &pyboost_reshape_and_cache, "Reshape And Cache",
        pybind11::arg("key"), pybind11::arg("value") = std::nullopt,
        pybind11::arg("key_cache") = std::nullopt,
        pybind11::arg("value_cache") = std::nullopt,
        pybind11::arg("slot_mapping") = std::nullopt,
        pybind11::arg("head_num") = std::nullopt);
}
