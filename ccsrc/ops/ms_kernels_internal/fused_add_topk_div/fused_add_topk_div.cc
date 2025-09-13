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

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "ccsrc/base/ms_kernels_internal/graphmode/internal_kernel_mod.h"
#include "ccsrc/utils/utils.h"
#include "mindspore/core/include/mindapi/ir/tensor.h"
#include "mindspore/ops/kernel/ascend/acl_ir/acl_convert.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "mindspore/ccsrc/ms_extension/api.h"
#include "mindspore/core/include/ops/base_operator.h"
#include "mindspore/core/include/ops/ops_func_impl/op_func_impl.h"
#include "mindspore/core/include/ops/ops_func_impl/simple_infer.h"
#include "mindspore/core/include/utils/check_convert_utils.h"
#include "mindspore/ccsrc/include/backend/common/ms_device_shape_transfer.h"

namespace ms_custom_ops {
enum FusedAddTopKDivInputIndex : size_t {
  kFusedAddTopKDivXIndex = 0,
  kFusedAddTopKDivAddNumIndex,
  kFusedAddTopKDivGroupNumIndex,
  kFusedAddTopKDivGroupTopKIndex,
  kFusedAddTopKDivNIndex,
  kFusedAddTopKDivKIndex,
  kFusedAddTopKDivActivateTypeIndex,
  kFusedAddTopKDivIsNormIndex,
  kFusedAddTopKDivScaleIndex,
  kFusedAddTopKDivMappingNumIndex,
  kFusedAddTopKDivMappingTableIndex,
  kFusedAddTopKDivEnableExpertMappingIndex,
  kFusedAddTopKDivInputsNum,
};

enum FusedAddTopKDivOutputIndex : size_t {
  kFusedAddTopKDivOutPutWeightIndex = 0,
  kFusedAddTopKDivOutputIndicesIndex,
  kFusedAddTopKDivOutputNums,
};

static const size_t DIM0_INDEX = 0;

class OPS_API FusedAddTopKDivOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto op_name = primitive->name();
    auto x_shape = input_infos[kFusedAddTopKDivXIndex]->GetShape();
    if (MS_UNLIKELY(input_infos[kFusedAddTopKDivXIndex]->IsDynamicRank()) ||
        MS_UNLIKELY(input_infos[kFusedAddTopKDivAddNumIndex]->IsDynamicRank())) {
      auto out_shape = {abstract::Shape::kShapeRankAny};
      return {out_shape};
    }

    auto k = input_infos[kFusedAddTopKDivKIndex]->GetScalarValueWithCheck<int64_t>();
    auto a = x_shape[DIM0_INDEX];

    ShapeVector out_shape{a, k};
    return {out_shape, out_shape};
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    return {TypeId::kNumberTypeFloat32, TypeId::kNumberTypeInt32};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class CustomFusedAddTopkDiv : public InternalKernelMod {
 public:
  CustomFusedAddTopkDiv() : InternalKernelMod() {}
  ~CustomFusedAddTopkDiv() = default;

  void InitKernelInputsOutputsIndex() override {
    kernel_inputs_index_ = {
      static_cast<size_t>(FusedAddTopKDivInputIndex::kFusedAddTopKDivXIndex),
      static_cast<size_t>(FusedAddTopKDivInputIndex::kFusedAddTopKDivAddNumIndex),
      static_cast<size_t>(FusedAddTopKDivInputIndex::kFusedAddTopKDivMappingNumIndex),
      static_cast<size_t>(FusedAddTopKDivInputIndex::kFusedAddTopKDivMappingTableIndex),
    };
    kernel_outputs_index_ = {
      static_cast<size_t>(FusedAddTopKDivOutputIndex::kFusedAddTopKDivOutPutWeightIndex),
      static_cast<size_t>(FusedAddTopKDivOutputIndex::kFusedAddTopKDivOutputIndicesIndex),
    };
  }

 protected:
  bool GetValidIntType(const TypeId &type_id) {
    return (type_id == TypeId::kNumberTypeInt64) || (type_id == TypeId::kNumberTypeInt32);
  }

  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs,
                                       const std::vector<KernelTensor *> &ms_inputs,
                                       const std::vector<KernelTensor *> &ms_outputs) override {
    internal::FusedAddTopkDivParam param;
    auto group_num = ms_inputs.at(FusedAddTopKDivInputIndex::kFusedAddTopKDivGroupNumIndex);
    auto group_topk = ms_inputs.at(FusedAddTopKDivInputIndex::kFusedAddTopKDivGroupTopKIndex);
    auto n = ms_inputs.at(FusedAddTopKDivInputIndex::kFusedAddTopKDivNIndex);
    auto k = ms_inputs.at(FusedAddTopKDivInputIndex::kFusedAddTopKDivKIndex);
    auto activate_type = ms_inputs.at(FusedAddTopKDivInputIndex::kFusedAddTopKDivActivateTypeIndex);
    auto is_norm = ms_inputs.at(FusedAddTopKDivInputIndex::kFusedAddTopKDivIsNormIndex);
    auto scale = ms_inputs.at(FusedAddTopKDivInputIndex::kFusedAddTopKDivScaleIndex);
    auto enableExpertMapping = ms_inputs.at(FusedAddTopKDivInputIndex::kFusedAddTopKDivEnableExpertMappingIndex);

    if (GetValidIntType(group_num->dtype_id()) && GetValidIntType(group_topk->dtype_id()) &&
        GetValidIntType(n->dtype_id()) && GetValidIntType(k->dtype_id()) &&
        GetValidIntType(activate_type->dtype_id()) && (is_norm->dtype_id() == TypeId::kNumberTypeBool) &&
        (scale->dtype_id() == TypeId::kNumberTypeFloat32) &&
        (enableExpertMapping->dtype_id() == TypeId::kNumberTypeBool)) {
      param.group_num = static_cast<int32_t>(group_num->GetValueWithCheck<int64_t>());
      param.group_topk = static_cast<int32_t>(group_topk->GetValueWithCheck<int64_t>());
      param.n = static_cast<int32_t>(n->GetValueWithCheck<int64_t>());
      param.k = static_cast<int32_t>(k->GetValueWithCheck<int64_t>());
      param.activate_type = static_cast<int32_t>(activate_type->GetValueWithCheck<int64_t>());
      param.is_norm = is_norm->GetValueWithCheck<bool>();
      param.scale = scale->GetValueWithCheck<float>();
      param.enableExpertMapping = enableExpertMapping->GetValueWithCheck<bool>();
    } else {
      MS_LOG(EXCEPTION) << "FusedAddTopKDiv [group_num, group_topk, n, k, activate_type, is_norm, scale]'s dtype wrong";
    }
    return internal::CreateFusedAddTopkDivOp(inputs, outputs, param, internal::kInternalFusedAddTopkDivOpName);
  }
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(fused_add_topk_div, ms_custom_ops::FusedAddTopKDivOpFuncImpl, ms_custom_ops::CustomFusedAddTopkDiv);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

#include "internal_pyboost_runner.h"

using namespace ms_custom_ops;
namespace ms::pynative {
class FusedAddTopkDivRunner : public InternalPyboostRunner {
 public:
  using InternalPyboostRunner::InternalPyboostRunner;

  void SetParams(const int32_t &group_num, const int32_t &group_topk, const int32_t &n, const int32_t &k,
                 const int32_t &activate_type, const bool &is_norm, const float &scale,
                 const bool &enable_expert_mapping) {
    param_.group_num = group_num;
    param_.group_topk = group_topk;
    param_.n = n;
    param_.k = k;
    param_.activate_type = activate_type;
    param_.is_norm = is_norm;
    param_.scale = scale;
    param_.enableExpertMapping = enable_expert_mapping;
  }

 protected:
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs) override {
    return internal::CreateFusedAddTopkDivOp(inputs, outputs, param_, internal::kInternalFusedAddTopkDivOpName);
  }

 private:
  internal::FusedAddTopkDivParam param_;
};
}  // namespace ms::pynative

namespace ms_custom_ops {
std::vector<ms::Tensor> npu_fused_add_topk_div(const ms::Tensor &x, const ms::Tensor &add_num, int64_t group_num,
                                               int64_t group_topk, int64_t n, int64_t k, int64_t activate_type,
                                               bool is_norm, float scale, const std::optional<ms::Tensor> &mapping_num,
                                               const std::optional<ms::Tensor> &mapping_table,
                                               bool enable_expert_mapping) {
  auto op_name = "FusedAddTopkDiv";
  auto runner = std::make_shared<ms::pynative::FusedAddTopkDivRunner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);

  // SetParams
  runner->SetParams(static_cast<int32_t>(group_num), static_cast<int32_t>(group_topk), static_cast<int32_t>(n),
                    static_cast<int32_t>(k), static_cast<int32_t>(activate_type), is_norm, scale,
                    enable_expert_mapping);

  // Setup the runner with all parameters (including hash calculation)
  runner->Setup(op_name, x, add_num, group_num, group_topk, n, k, activate_type, is_norm, scale, mapping_num,
                mapping_table, enable_expert_mapping);
  auto x_shape = x.shape();
  auto a = x_shape[DIM0_INDEX];
  ShapeVector out_shape{a, static_cast<int32_t>(k)};

  std::vector<ms::Tensor> inputs = {x, add_num, GetTensorOrEmpty(mapping_num), GetTensorOrEmpty(mapping_table)};
  std::vector<ms::Tensor> outputs = {ms::Tensor(TypeId::kNumberTypeFloat32, out_shape),
                                     ms::Tensor(TypeId::kNumberTypeInt32, out_shape)};
  runner->GetOrCreateKernel(inputs, outputs);
  runner->Run(inputs, outputs);
  return outputs;
}
}  // namespace ms_custom_ops

auto pyboost_fused_add_topk_div(const ms::Tensor &x, const ms::Tensor &add_num, int64_t group_num, int64_t group_topk,
                                int64_t n, int64_t k, int64_t activate_type, bool is_norm, float scale,
                                const std::optional<ms::Tensor> &mapping_num,
                                const std::optional<ms::Tensor> &mapping_table, bool enable_expert_mapping) {
  return ms::pynative::PyboostRunner::Call<FusedAddTopKDivOutputIndex::kFusedAddTopKDivOutputNums>(
    ms_custom_ops::npu_fused_add_topk_div, x, add_num, group_num, group_topk, n, k, activate_type, is_norm, scale,
    mapping_num, mapping_table, enable_expert_mapping);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("fused_add_topk_div", &pyboost_fused_add_topk_div, "FusedAddTopkDiv", pybind11::arg("x"),
        pybind11::arg("add_num"), pybind11::arg("group_num"), pybind11::arg("group_topk"), pybind11::arg("n"),
        pybind11::arg("k"), pybind11::arg("activate_type") = 0, pybind11::arg("is_norm") = true,
        pybind11::arg("scale") = 2.5, pybind11::arg("mapping_num") = std::nullopt,
        pybind11::arg("mapping_table") = std::nullopt, pybind11::arg("enable_expert_mapping") = false);
}
