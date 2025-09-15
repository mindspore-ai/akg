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

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "ccsrc/base/ms_kernels_internal/graphmode/internal_kernel_mod.h"
#include "ccsrc/utils/utils.h"

// =============================================================================
// COMMON FUNCTION
// =============================================================================

namespace ms_custom_ops {
enum class CacheMode : int32_t {
  ND = 0,
  NZ = 1,
};

enum class InputIndex : size_t {
  kInputKeyIndex = 0,
  kInputValueIndex = 1,
  kInputKeyCacheIndex = 2,
  kInputValueCacheIndex = 3,
  kInputSlotMappingIndex = 4,
  kInputCacheModeIndex = 5,
  kInputHeadNumIndex = 6,
};

enum class OutputIndex : size_t { kOutputIndex = 0 };

inline internal::InternalOpPtr CreateReshapeAndCacheOpWithFormat(const internal::InputsImmutableInfoList &inputs,
                                                                 const internal::OutputsImmutableInfoList &outputs,
                                                                 const internal::ReshapeAndCacheParam &param,
                                                                 int32_t cache_mode) {
  if (cache_mode == static_cast<int32_t>(CacheMode::NZ)) {
    auto inputs_clone = inputs;
    inputs_clone[static_cast<size_t>(InputIndex::kInputKeyCacheIndex)].SetFormat(internal::kFormatFRACTAL_NZ);
    inputs_clone[static_cast<size_t>(InputIndex::kInputValueCacheIndex)].SetFormat(internal::kFormatFRACTAL_NZ);
    return internal::CreateAsdReshapeAndCacheOp(inputs_clone, outputs, param,
                                                internal::kInternalAsdReshapeAndCacheOpName);
  }
  return internal::CreateAsdReshapeAndCacheOp(inputs, outputs, param, internal::kInternalAsdReshapeAndCacheOpName);
}

// =============================================================================
// GRAPH MODE IMPLEMENTATION
// =============================================================================

class OPS_API CustomReshapeAndCacheOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    return {input_infos[static_cast<size_t>(InputIndex::kInputKeyIndex)]->GetShape()};
  }
  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    return {input_infos[static_cast<size_t>(InputIndex::kInputKeyIndex)]->GetType()};
  }
  bool GeneralInferRegistered() const override { return true; }
};

class CustomReshapeAndCache : public InternalKernelMod {
 public:
  CustomReshapeAndCache() : InternalKernelMod(), skip_execution_(false) {}
  ~CustomReshapeAndCache() = default;

  void InitKernelInputsOutputsIndex() override {
    kernel_inputs_index_ = {
      static_cast<size_t>(InputIndex::kInputKeyIndex), static_cast<size_t>(InputIndex::kInputValueIndex),
      static_cast<size_t>(InputIndex::kInputKeyCacheIndex), static_cast<size_t>(InputIndex::kInputValueCacheIndex),
      static_cast<size_t>(InputIndex::kInputSlotMappingIndex)};
    kernel_outputs_index_ = {static_cast<size_t>(OutputIndex::kOutputIndex)};
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    // Check if any input has shape containing 0
    for (const auto &input : inputs) {
      if (input == nullptr) continue;
      auto shape = input->GetShapeVector();
      for (const auto &dim : shape) {
        if (dim == 0) {
          MS_LOG(INFO) << "ReshapeAndCache: Skipping execution due to zero "
                          "dimension in input shape: "
                       << shape;
          skip_execution_ = true;
          return KernelMod::Resize(inputs, outputs);  // Skip execution
        }
      }
    }

    skip_execution_ = false;
    // Call base class implementation
    return InternalKernelMod::Resize(inputs, outputs);
  }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    // Skip execution if flag is set
    if (skip_execution_) {
      return true;  // Skip execution, return success
    }

    // Call base class implementation
    return InternalKernelMod::Launch(inputs, workspace, outputs, stream_ptr);
  }

 protected:
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs,
                                       const std::vector<KernelTensor *> &ms_inputs,
                                       const std::vector<KernelTensor *> &ms_outputs) override {
    internal::ReshapeAndCacheParam param;
    auto head_num = ms_inputs.at(static_cast<size_t>(InputIndex::kInputHeadNumIndex));
    if (head_num->dtype_id() == TypeId::kNumberTypeInt64) {
      param.head_num = static_cast<int32_t>(head_num->GetValue<int64_t>().value());
    } else {
      MS_LOG(EXCEPTION) << "ReshapeAndCache [head_num]'s dtype wrong, expect int64, but got: " << head_num->dtype_id();
    }
    auto cache_mode = ms_inputs.at(static_cast<size_t>(InputIndex::kInputCacheModeIndex));
    int32_t cache_node_val = 0;
    if (cache_mode->dtype_id() == TypeId::kNumberTypeInt64) {
      cache_node_val = static_cast<int32_t>(cache_mode->GetValue<int64_t>().value());
    } else {
      MS_LOG(EXCEPTION) << "ReshapeAndCache [cache_mode]'s dtype wrong, expect int64, but got: "
                        << cache_mode->dtype_id();
    }

    return CreateReshapeAndCacheOpWithFormat(inputs, outputs, param, cache_node_val);
  }

 private:
  bool skip_execution_;  // Flag to skip execution when shape contains 0
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(reshape_and_cache, ms_custom_ops::CustomReshapeAndCacheOpFuncImpl,
                  ms_custom_ops::CustomReshapeAndCache);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

#include "internal_pyboost_runner.h"

namespace ms_custom_ops {
class ReshapeAndCacheRunner : public InternalPyboostRunner {
 public:
  using InternalPyboostRunner::InternalPyboostRunner;

  void SetHeadNum(const int32_t &head_num) { this->head_num_ = head_num; }
  void SetCacheMode(const int32_t &cache_mode) { this->cache_mode_ = cache_mode; }

 protected:
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs) override {
    internal::ReshapeAndCacheParam param;
    param.head_num = this->head_num_;

    return CreateReshapeAndCacheOpWithFormat(inputs, outputs, param, this->cache_mode_);
  }

 private:
  int32_t head_num_{0};
  int32_t cache_mode_{0};
};

void npu_reshape_and_cache(const ms::Tensor &key, const std::optional<ms::Tensor> &value,
                           const std::optional<ms::Tensor> &key_cache, const std::optional<ms::Tensor> &value_cache,
                           const std::optional<ms::Tensor> &slot_mapping, const int64_t cache_mode,
                           const int64_t head_num) {
  auto op_name = "ReshapeAndCache";
  auto runner = std::make_shared<ms_custom_ops::ReshapeAndCacheRunner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);
  runner->SetCacheMode(static_cast<int32_t>(cache_mode));
  runner->SetHeadNum(static_cast<int32_t>(head_num));

  // Setup the runner with all parameters (including hash calculation)
  runner->Setup(op_name, key, value, key_cache, value_cache, slot_mapping, cache_mode, head_num);

  // if you need infer shape and type, you need create output tensors.
  std::vector<ms::Tensor> inputs = {key, GetTensorOrEmpty(value), GetTensorOrEmpty(key_cache),
                                    GetTensorOrEmpty(value_cache), GetTensorOrEmpty(slot_mapping)};
  std::vector<ms::Tensor> outputs = {};
  runner->GetOrCreateKernel(inputs, outputs);
  runner->Run(inputs, outputs);
  return;
}
}  // namespace ms_custom_ops

auto pyboost_reshape_and_cache(const ms::Tensor &key, const std::optional<ms::Tensor> &value,
                               const std::optional<ms::Tensor> &key_cache, const std::optional<ms::Tensor> &value_cache,
                               const std::optional<ms::Tensor> &slot_mapping, const int64_t cache_mode,
                               const int64_t head_num) {
  return ms::pynative::PyboostRunner::Call<0>(ms_custom_ops::npu_reshape_and_cache, key, value, key_cache, value_cache,
                                              slot_mapping, cache_mode, head_num);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("reshape_and_cache", &pyboost_reshape_and_cache, "Reshape And Cache", pybind11::arg("key"),
        pybind11::arg("value") = std::nullopt, pybind11::arg("key_cache") = std::nullopt,
        pybind11::arg("value_cache") = std::nullopt, pybind11::arg("slot_mapping") = std::nullopt,
        pybind11::arg("cache_mode"), pybind11::arg("head_num"));
}
