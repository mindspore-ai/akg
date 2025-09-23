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

namespace ms_custom_ops {
enum class ApplyRotaryPosEmbQueryInputIndex : size_t {
  kApplyRotaryPosEmbQueryIndex = 0,
  kApplyRotaryPosEmbKeyIndex,
  kApplyRotaryPosEmbCosIndex,
  kApplyRotaryPosEmbSinIndex,
  kApplyRotaryPosEmbPositionIdsIndex,
  kApplyRotaryPosEmbCosFormatIndex,
  kApplyRotaryPosEmbInputsNum,
};
enum class ApplyRotaryPosEmbQueryOutputIndex : size_t {
  kApplyRotaryPosEmbQuerOutputIndex = 0,
  kApplyRotaryPosEmbKeyOutputIndex,
  kFApplyRotaryPosEmbOutputsNum,
};
class OPS_API CustomApplyRotaryPosEmbOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    return {
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbQueryInputIndex::kApplyRotaryPosEmbQueryIndex)]->GetShape(),
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbQueryInputIndex::kApplyRotaryPosEmbKeyIndex)]->GetShape()};
  }
  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    return {input_infos[static_cast<size_t>(ApplyRotaryPosEmbQueryInputIndex::kApplyRotaryPosEmbQueryIndex)]->GetType(),
            input_infos[static_cast<size_t>(ApplyRotaryPosEmbQueryInputIndex::kApplyRotaryPosEmbKeyIndex)]->GetType()};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class CustomApplyRotaryPosEmb : public InternalKernelMod {
 public:
  CustomApplyRotaryPosEmb() : InternalKernelMod() {}
  ~CustomApplyRotaryPosEmb() = default;

  void InitKernelInputsOutputsIndex() override {
    kernel_inputs_index_ = {static_cast<size_t>(ApplyRotaryPosEmbQueryInputIndex::kApplyRotaryPosEmbQueryIndex),
                            static_cast<size_t>(ApplyRotaryPosEmbQueryInputIndex::kApplyRotaryPosEmbKeyIndex),
                            static_cast<size_t>(ApplyRotaryPosEmbQueryInputIndex::kApplyRotaryPosEmbCosIndex),
                            static_cast<size_t>(ApplyRotaryPosEmbQueryInputIndex::kApplyRotaryPosEmbSinIndex),
                            static_cast<size_t>(ApplyRotaryPosEmbQueryInputIndex::kApplyRotaryPosEmbPositionIdsIndex),
                            static_cast<size_t>(ApplyRotaryPosEmbQueryInputIndex::kApplyRotaryPosEmbCosFormatIndex)};
    kernel_outputs_index_ = {static_cast<size_t>(ApplyRotaryPosEmbQueryOutputIndex::kApplyRotaryPosEmbQuerOutputIndex),
                             static_cast<size_t>(ApplyRotaryPosEmbQueryOutputIndex::kApplyRotaryPosEmbKeyOutputIndex)};
  }

 protected:
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs,
                                       const std::vector<KernelTensor *> &ms_inputs,
                                       const std::vector<KernelTensor *> &ms_outputs) override {
    internal::ApplyRotaryPosEmbParam param;
    auto cos_format =
      ms_inputs.at(static_cast<size_t>(ApplyRotaryPosEmbQueryInputIndex::kApplyRotaryPosEmbCosFormatIndex));
    if (cos_format->dtype_id() == TypeId::kNumberTypeInt64) {
      param.cos_format = static_cast<int32_t>(cos_format->GetValue<int64_t>().value());
    } else {
      MS_LOG(EXCEPTION) << "ApplyRotaryPosEmb [cos_format]'s dtype wrong, expect int64, but got: "
                        << cos_format->dtype_id();
    }
    return internal::CreateApplyRotaryPosEmbOp(inputs, outputs, param, internal::kInternalApplyRotaryPosEmbOpName);
  }
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(apply_rotary_pos_emb, ms_custom_ops::CustomApplyRotaryPosEmbOpFuncImpl,
                  ms_custom_ops::CustomApplyRotaryPosEmb);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

#include "internal_pyboost_runner.h"

using namespace ms_custom_ops;
namespace ms::pynative {
class ApplyRotaryPosEmbRunner : public InternalPyboostRunner {
 public:
  using InternalPyboostRunner::InternalPyboostRunner;

  void SetCosFormat(const int32_t &cos_format) { this->cos_format_ = cos_format; }

 protected:
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs) override {
    internal::ApplyRotaryPosEmbParam param;
    param.cos_format = this->cos_format_;
    return internal::CreateApplyRotaryPosEmbOp(inputs, outputs, param, internal::kInternalApplyRotaryPosEmbOpName);
  }

 private:
  int32_t cos_format_{0};
};
}  // namespace ms::pynative

namespace ms_custom_ops {
std::vector<ms::Tensor> npu_apply_rotary_pos_emb(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &cos,
                                                 const ms::Tensor &sin, const ms::Tensor &position_ids,
                                                 std::optional<int64_t> cos_format) {
  auto op_name = "ApplyRotaryPosEmb";
  auto runner = std::make_shared<ms::pynative::ApplyRotaryPosEmbRunner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);

  // Set cos_format if provided
  if (cos_format.has_value()) {
    runner->SetCosFormat(static_cast<int32_t>(cos_format.value()));
  }

  // Setup the runner with all parameters (including hash calculation)
  runner->Setup(op_name, query, key, cos, sin, position_ids, cos_format);

  // if you need infer shape and type, you can use this
  std::vector<ms::Tensor> inputs = {query, key, cos, sin, position_ids};
  std::vector<ms::Tensor> outputs = {ms::Tensor(query.data_type(), query.shape()),
                                     ms::Tensor(key.data_type(), key.shape())};
  runner->GetOrCreateKernel(inputs, outputs);
  runner->Run(inputs, outputs);
  return outputs;
}
}  // namespace ms_custom_ops

auto pyboost_apply_rotary_pos_emb(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &cos,
                                  const ms::Tensor &sin, const ms::Tensor &position_ids,
                                  std::optional<int64_t> cos_format) {
  return ms::pynative::PyboostRunner::Call<2>(ms_custom_ops::npu_apply_rotary_pos_emb, query, key, cos, sin,
                                              position_ids, cos_format);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("apply_rotary_pos_emb", &pyboost_apply_rotary_pos_emb, "ApplyRotaryPosEmb", pybind11::arg("query"),
        pybind11::arg("key"), pybind11::arg("cos"), pybind11::arg("sin"), pybind11::arg("position_ids"),
        pybind11::arg("cos_format") = std::nullopt);
}
