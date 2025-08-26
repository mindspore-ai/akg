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

namespace ms_custom_ops {
enum class MoeGatingGroupTopKInputIndex : size_t {
  kMoeGatingGroupTopKXIndex = 0,
  kMoeGatingGroupTopKBiasOptionalIndex,
  kMoeGatingGroupTopKKIndex,
  kMoeGatingGroupTopKKGroupIndex,
  kMoeGatingGroupTopKGroupCountIndex,
  kMoeGatingGroupTopKGroupSelectModeIndex,
  kMoeGatingGroupTopKRenormIndex,
  kMoeGatingGroupTopKNormTypeIndex,
  kMoeGatingGroupTopKOutFlagIndex,
  kMoeGatingGroupTopKRoutedScalingFactorIndex,
  kMoeGatingGroupTopKEpsIndex,
  kMoeGatingGroupTopKInputsNum
};
enum class MoeGatingGroupTopKOutputIndex : size_t {
  kMoeGatingGroupTopKYOutIndex = 0,
  MoeGatingGroupTopKExpertIdxOutIndex,
  MoeGatingGroupTopKNormOutOptionalIndex,
  MoeGatingGroupTopKOutsNum,
};
constexpr uint32_t MOE_GATING_TOPK_DIM = 2;
class OPS_API CustomMoeGatingGroupTopKOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto &x = input_infos[static_cast<size_t>(MoeGatingGroupTopKInputIndex::kMoeGatingGroupTopKXIndex)];
    auto x_shape = x->GetShape();
    if (x_shape.size() != MOE_GATING_TOPK_DIM) {
      MS_LOG(EXCEPTION) << "For MoeGatingGroupTopK, input 'X' must be 2D, but got:" << x_shape.size();
    }
    // input x dynamic rank
    if (x->IsDynamicRank()) {
      auto out_shape = ShapeVector{abstract::Shape::kShapeRankAny};
      return {out_shape, out_shape, out_shape};
    }
    auto k_scalar = input_infos[static_cast<size_t>(MoeGatingGroupTopKInputIndex::kMoeGatingGroupTopKKIndex)]
                      ->GetScalarValueWithCheck<int64_t>();
    auto out_shape_vec = x_shape;
    out_shape_vec[MOE_GATING_TOPK_DIM - 1] = k_scalar;

    return {out_shape_vec, out_shape_vec, x->GetShape()};
  }
  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto x_dtype = input_infos[static_cast<size_t>(MoeGatingGroupTopKInputIndex::kMoeGatingGroupTopKXIndex)]->GetType();
    return {x_dtype, TypeId::kNumberTypeInt32, TypeId::kNumberTypeFloat32};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class CustomMoeGatingGroupTopK : public InternalKernelMod {
 public:
  CustomMoeGatingGroupTopK() : InternalKernelMod() {}
  ~CustomMoeGatingGroupTopK() = default;

  void InitKernelInputsOutputsIndex() override {
    kernel_inputs_index_ = {
      static_cast<size_t>(MoeGatingGroupTopKInputIndex::kMoeGatingGroupTopKXIndex),
      static_cast<size_t>(MoeGatingGroupTopKInputIndex::kMoeGatingGroupTopKBiasOptionalIndex),
    };
    kernel_outputs_index_ = {
      static_cast<size_t>(MoeGatingGroupTopKOutputIndex::kMoeGatingGroupTopKYOutIndex),
      static_cast<size_t>(MoeGatingGroupTopKOutputIndex::MoeGatingGroupTopKExpertIdxOutIndex),
      static_cast<size_t>(MoeGatingGroupTopKOutputIndex::MoeGatingGroupTopKNormOutOptionalIndex)};
  }

 protected:
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs,
                                       const std::vector<KernelTensor *> &ms_inputs,
                                       const std::vector<KernelTensor *> &ms_outputs) override {
    internal::MoeGatingGroupTopKParam param;
    auto k = ms_inputs.at(kIndex2);
    auto k_group = ms_inputs.at(kIndex3);
    auto group_count = ms_inputs.at(kIndex4);
    auto group_select_mode = ms_inputs.at(kIndex5);
    auto renorm = ms_inputs.at(kIndex6);
    auto norm_type = ms_inputs.at(kIndex7);
    auto out_flag = ms_inputs.at(kIndex8);
    auto routed_scaling_factor = ms_inputs.at(kIndex9);
    auto eps = ms_inputs.at(kIndex10);

    if (k->dtype_id() == TypeId::kNumberTypeInt64 && k_group->dtype_id() == TypeId::kNumberTypeInt64 &&
        group_count->dtype_id() == TypeId::kNumberTypeInt64 &&
        group_select_mode->dtype_id() == TypeId::kNumberTypeInt64 && renorm->dtype_id() == TypeId::kNumberTypeInt64 &&
        norm_type->dtype_id() == TypeId::kNumberTypeInt64 && out_flag->dtype_id() == TypeId::kNumberTypeBool &&
        routed_scaling_factor->dtype_id() == TypeId::kNumberTypeFloat32 &&
        eps->dtype_id() == TypeId::kNumberTypeFloat32) {
      param.k = static_cast<int32_t>(k->GetValue<int64_t>().value());
      param.k_group = static_cast<int32_t>(k_group->GetValue<int64_t>().value());
      param.group_count = static_cast<int32_t>(group_count->GetValue<int64_t>().value());
      param.group_select_mode = static_cast<int32_t>(group_select_mode->GetValue<int64_t>().value());
      param.renorm = static_cast<int32_t>(renorm->GetValue<int64_t>().value());
      param.norm_type = static_cast<int32_t>(norm_type->GetValue<int64_t>().value());
      param.out_flag = out_flag->GetValue<bool>().value();
      param.routed_scaling_factor = routed_scaling_factor->GetValue<float>().value();
      param.eps = eps->GetValue<float>().value();
    } else {
      MS_LOG(EXCEPTION)
        << "MoeGatingGroupTopK inputs[k, k_group, group_count, group_select_mode, renorm, norm_type, "
           "out_flag, routed_scaling_factor, eps]'s dtype should be [kNumberTypeInt64, kNumberTypeInt64, "
           "kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeBool, "
           "kNumberTypeFloat32, kNumberTypeFloat32], but got ["
        << TypeIdToString(k->dtype_id()) << ", " << TypeIdToString(k_group->dtype_id()) << ", "
        << TypeIdToString(group_count->dtype_id()) << ", " << TypeIdToString(group_select_mode->dtype_id()) << ", "
        << TypeIdToString(renorm->dtype_id()) << ", " << TypeIdToString(norm_type->dtype_id()) << ", "
        << TypeIdToString(out_flag->dtype_id()) << ", " << TypeIdToString(routed_scaling_factor->dtype_id()) << ", "
        << TypeIdToString(eps->dtype_id()) << "]";
    }
    return internal::CreateMoeGatingGroupTopKOp(inputs, outputs, param, internal::kInternalMoeGatingGroupTopKOpName);
  }
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(moe_gating_group_topk, ms_custom_ops::CustomMoeGatingGroupTopKOpFuncImpl,
                  ms_custom_ops::CustomMoeGatingGroupTopK);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

#include "internal_pyboost_runner.h"

using namespace ms_custom_ops;
namespace ms::pynative {
class MoeGatingGroupTopKRunner : public InternalPyboostRunner {
 public:
  using InternalPyboostRunner::InternalPyboostRunner;

  void SetParams(const int32_t &k, const int32_t &k_group, const int32_t &group_count, const int32_t &group_select_mode,
                 const int32_t &renorm, const int32_t &norm_type, const bool &out_flag,
                 const float &routed_scaling_factor, const float &eps) {
    param_.k = k;
    param_.k_group = k_group;
    param_.group_count = group_count;
    param_.group_select_mode = group_select_mode;
    param_.renorm = renorm;
    param_.norm_type = norm_type;
    param_.out_flag = out_flag;
    param_.routed_scaling_factor = routed_scaling_factor;
    param_.eps = eps;
  }

 protected:
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs) override {
    return internal::CreateMoeGatingGroupTopKOp(inputs, outputs, param_, internal::kInternalMoeGatingGroupTopKOpName);
  }

 private:
  internal::MoeGatingGroupTopKParam param_;
};
}  // namespace ms::pynative

namespace ms_custom_ops {
std::vector<ms::Tensor> npu_moe_gating_group_topk(
  const ms::Tensor &x, const std::optional<ms::Tensor> &bias, std::optional<int64_t> k, std::optional<int64_t> k_group,
  std::optional<int64_t> group_count, std::optional<int64_t> group_select_mode, std::optional<int64_t> renorm,
  std::optional<int64_t> norm_type, std::optional<bool> out_flag, std::optional<float> routed_scaling_factor,
  std::optional<float> eps) {
  auto op_name = "MoeGatingGroupTopK";
  auto runner = std::make_shared<ms::pynative::MoeGatingGroupTopKRunner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);

  // Set params
  runner->SetParams(static_cast<int32_t>(k.value()), static_cast<int32_t>(k_group.value()),
                    static_cast<int32_t>(group_count.value()), static_cast<int32_t>(group_select_mode.value()),
                    static_cast<int32_t>(renorm.value()), static_cast<int32_t>(norm_type.value()),
                    static_cast<bool>(out_flag.value()), static_cast<float>(routed_scaling_factor.value()),
                    static_cast<float>(eps.value()));

  // Setup the runner with all parameters (including hash calculation)
  runner->Setup(op_name, x, bias, k, k_group, group_count, group_select_mode, renorm, norm_type, out_flag,
                routed_scaling_factor, eps);
  auto x_shape = x.shape();
  x_shape[1] = static_cast<int32_t>(k.value());
  // if you need infer shape and type, you can use this
  auto bias_tensor = bias.has_value() ? bias.value() : ms::Tensor();
  std::vector<ms::Tensor> inputs = {x, bias_tensor};
  std::vector<ms::Tensor> outputs = {ms::Tensor(x.data_type(), x_shape), ms::Tensor(TypeId::kNumberTypeInt32, x_shape),
                                     ms::Tensor(TypeId::kNumberTypeFloat32, x.shape())};
  runner->GetOrCreateKernel(inputs, outputs);
  runner->Run(inputs, outputs);
  return outputs;
}
}  // namespace ms_custom_ops

auto pyboost_moe_gating_group_topk(const ms::Tensor &x, const std::optional<ms::Tensor> &bias, std::optional<int64_t> k,
                                   std::optional<int64_t> k_group, std::optional<int64_t> group_count,
                                   std::optional<int64_t> group_select_mode, std::optional<int64_t> renorm,
                                   std::optional<int64_t> norm_type, std::optional<bool> out_flag,
                                   std::optional<float> routed_scaling_factor, std::optional<float> eps) {
  return ms::pynative::PyboostRunner::Call<3>(ms_custom_ops::npu_moe_gating_group_topk, x, bias, k, k_group,
                                              group_count, group_select_mode, renorm, norm_type, out_flag,
                                              routed_scaling_factor, eps);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("moe_gating_group_topk", &pyboost_moe_gating_group_topk, "MoeGatingGroupTopK", pybind11::arg("x"),
        pybind11::arg("bias") = std::nullopt, pybind11::arg("k"),
        pybind11::arg("k_group"), pybind11::arg("group_count"),
        pybind11::arg("group_select_mode"), pybind11::arg("renorm"),
        pybind11::arg("norm_type"), pybind11::arg("out_flag"),
        pybind11::arg("routed_scaling_factor"), pybind11::arg("eps"));
}
