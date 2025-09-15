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
#include "ccsrc/utils/utils.h"
#include "ccsrc/base/ms_kernels_internal/graphmode/internal_kernel_mod.h"

// =============================================================================
// COMMON FUNCTION
// =============================================================================

namespace ms_custom_ops {
enum class TransdataType : int32_t {
  FRACTAL_NZ_TO_ND = 0,
  ND_TO_FRACTAL_NZ = 1,
};

enum class InputIndex : size_t {
  kInputIndex = 0,
  kTransdataTypeIndex = 1,
};

enum class OutputIndex : size_t { kOutputIndex = 0 };

inline internal::InternalOpPtr CreateTransDataOpWithParam(const internal::InputsImmutableInfoList &inputs,
                                                         const internal::OutputsImmutableInfoList &outputs,
                                                         int32_t transdata_type) {
  internal::TransDataParam param;
  
  // Map transdata_type to internal enum and set appropriate input format
  auto inputs_clone = inputs;
  auto outputs_clone = outputs;

  if (transdata_type == static_cast<int32_t>(TransdataType::FRACTAL_NZ_TO_ND)) {
    param.transdataType = internal::TransDataParam::FRACTAL_NZ_TO_ND;
    // For FRACTAL_NZ_TO_ND: input should be FRACTAL_NZ format
    inputs_clone[0].SetFormat(internal::kFormatFRACTAL_NZ);
    outputs_clone[0].SetFormat(internal::kFormatND);
  } else if (transdata_type == static_cast<int32_t>(TransdataType::ND_TO_FRACTAL_NZ)) {
    param.transdataType = internal::TransDataParam::ND_TO_FRACTAL_NZ;
    // For ND_TO_FRACTAL_NZ: input should be ND format
    inputs_clone[0].SetFormat(internal::kFormatND);
    outputs_clone[0].SetFormat(internal::kFormatFRACTAL_NZ);
  } else {
    MS_LOG(EXCEPTION) << "TransData: Invalid transdata_type " << transdata_type
                      << ", valid values are: 0 (FRACTAL_NZ_TO_ND), 1 (ND_TO_FRACTAL_NZ)";
  }

  // Note: outCrops are handled internally by the ms_kernels_internal layer
  // Users do not need to specify outCrops - they are auto-calculated
  param.specialTransdata = internal::TransDataParam::NORMAL;
  
  return internal::CreateTransDataOp(inputs_clone, outputs_clone, param, internal::kInternalTransDataOpName);
}

// =============================================================================
// GRAPH MODE IMPLEMENTATION
// =============================================================================

class OPS_API CustomTransDataOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    // For TransData, output shape depends on the conversion type
    // For now, return the same shape as input (this might need refinement based on actual format conversion)
    return {input_infos[static_cast<size_t>(InputIndex::kInputIndex)]->GetShape()};
  }
  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    return {input_infos[static_cast<size_t>(InputIndex::kInputIndex)]->GetType()};
  }
  bool GeneralInferRegistered() const override { return true; }
};

class CustomTransData : public InternalKernelMod {
 public:
  CustomTransData() : InternalKernelMod(), skip_execution_(false) {}
  ~CustomTransData() = default;

  void InitKernelInputsOutputsIndex() override {
    kernel_inputs_index_ = {static_cast<size_t>(InputIndex::kInputIndex)};
    kernel_outputs_index_ = {static_cast<size_t>(OutputIndex::kOutputIndex)};
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    // Check if any input has shape containing 0
    for (const auto &input : inputs) {
      if (input == nullptr) continue;
      auto shape = input->GetShapeVector();
      for (const auto &dim : shape) {
        if (dim == 0) {
          MS_LOG(INFO) << "TransData: Skipping execution due to zero dimension in input shape: " << shape;
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
    auto transdata_type = ms_inputs.at(static_cast<size_t>(InputIndex::kTransdataTypeIndex));
    int32_t transdata_type_val = 0;
    if (transdata_type->dtype_id() == TypeId::kNumberTypeInt64) {
      transdata_type_val = static_cast<int32_t>(transdata_type->GetValue<int64_t>().value());
    } else {
      MS_LOG(EXCEPTION) << "TransData [transdata_type]'s dtype wrong, expect int64, but got: "
                        << transdata_type->dtype_id();
    }
    
    return CreateTransDataOpWithParam(inputs, outputs, transdata_type_val);
  }

 private:
  bool skip_execution_;  // Flag to skip execution when shape contains 0
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(trans_data, ms_custom_ops::CustomTransDataOpFuncImpl, ms_custom_ops::CustomTransData);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

#include "internal_pyboost_runner.h"

namespace ms_custom_ops {
class TransDataRunner : public InternalPyboostRunner {
 public:
  using InternalPyboostRunner::InternalPyboostRunner;

  void SetTransdataType(const int32_t &transdata_type) { this->transdata_type_ = transdata_type; }

 protected:
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs) override {
    return CreateTransDataOpWithParam(inputs, outputs, this->transdata_type_);
  }

 private:
  int32_t transdata_type_{0};
};

ms::Tensor npu_trans_data(const ms::Tensor &input, std::optional<int64_t> transdata_type) {
  auto op_name = "TransData";
  auto runner = std::make_shared<ms_custom_ops::TransDataRunner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);

  if (transdata_type.has_value()) {
    runner->SetTransdataType(static_cast<int32_t>(transdata_type.value()));
  }

  // Setup the runner with all parameters (including hash calculation)
  runner->Setup(op_name, input, transdata_type);

  // Create output tensor with same shape and type as input
  // Note: The actual output shape may be different due to format conversion
  // but the kernel will handle the correct output allocation
  auto output = ms::Tensor(input.data_type(), input.shape());

  // Create input and output tensors
  std::vector<ms::Tensor> inputs = {input};
  std::vector<ms::Tensor> outputs = {output};
  runner->GetOrCreateKernel(inputs, outputs);
  runner->Run(inputs, outputs);
  return output;
}
}  // namespace ms_custom_ops

auto pyboost_trans_data(const ms::Tensor &input, std::optional<int64_t> transdata_type) {
  return ms::pynative::PyboostRunner::Call<1>(ms_custom_ops::npu_trans_data, input, transdata_type);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("trans_data", &pyboost_trans_data, "Trans Data", pybind11::arg("input"),
        pybind11::arg("transdata_type") = std::nullopt);
}