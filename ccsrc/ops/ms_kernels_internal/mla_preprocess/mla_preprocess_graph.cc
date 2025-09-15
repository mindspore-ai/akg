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

#include "ccsrc/base/ms_kernels_internal/graphmode/internal_kernel_mod.h"
#include "ccsrc/ops/ms_kernels_internal/mla_preprocess/mla_preprocess_common.h"

namespace ms_custom_ops {
class OPS_API CustomMlaPreprocessOpFuncImpl : public OpFuncImpl {
public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto input1_shape_ptr = input_infos[kMlaPreprocessInput1Index]->GetShape();
    auto key_cache_shape_ptr = input_infos[kMlaPreprocessKeyCacheIndex]->GetShape();
    auto wuk_ptr = input_infos[kMlaPreprocessWukIndex]->GetShape();

    auto cache_mode = input_infos[kMlaPreprocessParamCacheModeIndex]->GetScalarValueWithCheck<int64_t>();
    auto head_dim = key_cache_shape_ptr[3];
    auto n = input1_shape_ptr[0];
    auto head_num = wuk_ptr[0];

    if (cache_mode != kMlaPreCacheModeQK) {
      return {{n, head_num, 512}, {0}, {n, head_num, 64}, {0}};
    }
    return {{n, head_num, head_dim}, {0}, {}, {}};
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto input1_type = input_infos[kMlaPreprocessInput1Index]->GetType();
    auto offset1_type = input_infos[kMlaPreprocessQuantOffset1Index]->GetType();
    auto cache_mode = input_infos[kMlaPreprocessParamCacheModeIndex]->GetScalarValueWithCheck<int64_t>();
    if (cache_mode == kMlaPreCacheModeQKSplitQuant) {
      return {offset1_type, offset1_type, input1_type, input1_type};
    }
    return {input1_type, input1_type, input1_type, input1_type};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class CustomMlaPreprocess : public InternalKernelMod {
public:
  CustomMlaPreprocess() : InternalKernelMod() {}
  ~CustomMlaPreprocess() = default;
  void InitKernelInputsOutputsIndex() override {
    kernel_inputs_index_ = {kMlaPreprocessInput1Index, kMlaPreprocessGamma1Index, kMlaPreprocessBeta1Index,
                            kMlaPreprocessQuantScale1Index, kMlaPreprocessQuantOffset1Index, kMlaPreprocessWdqkvIndex,
                            kMlaPreprocessBias1Index, kMlaPreprocessGamma2Index, kMlaPreprocessBeta2Index,
                            kMlaPreprocessQuantScale2Index, kMlaPreprocessQuantOffset2Index, kMlaPreprocessGamma3Index,
                            kMlaPreprocessSin1Index, kMlaPreprocessCos1Index, kMlaPreprocessSin2Index,
                            kMlaPreprocessCos2Index, kMlaPreprocessKeyCacheIndex, kMlaPreprocessSlotMappingIndex,
                            kMlaPreprocessWuqIndex, kMlaPreprocessBias2Index, kMlaPreprocessWukIndex,
                            kMlaPreprocessDeScale1Index, kMlaPreprocessDeScale2Index, kMlaPreprocessCtkvScaleIndex,
                            kMlaPreprocessQnopeScaleIndex, kMlaPreprocessKropeCacheIndex};
    kernel_outputs_index_ = {kMlaPreprocessOutputQueryOutIndex, kMlaPreprocessOutputKeyOutIndex,
                             kMlaPreprocessOutputQropeIndex, kMlaPreprocessOutputKropeIndex};
  }
protected:
  internal::InternalOpPtr
  CreateKernel(const internal::InputsImmutableInfoList &inputs,
               const internal::OutputsImmutableInfoList &outputs,
               const std::vector<KernelTensor *> &ms_inputs,
               const std::vector<KernelTensor *> &ms_outputs) override {
    internal::MlaPreprocessParam param;
    auto cache_mode = ms_inputs.at(kMlaPreprocessParamCacheModeIndex);
    if (cache_mode->dtype_id() == TypeId::kNumberTypeInt64) {
      param.n = 0;
      param.head_num = 0;
      param.cache_mode = static_cast<int32_t>(cache_mode->GetValue<int64_t>().value());
    } else {
        MS_LOG(EXCEPTION) << "MlaPreprocess cache_mode should be a int value.";
    }
    return CreateMlaPreprocessOpWithFormat(inputs, outputs, param);
  }
};
} // namespace ms_custom_ops
REG_GRAPH_MODE_OP(mla_preprocess, ms_custom_ops::CustomMlaPreprocessOpFuncImpl,
                  ms_custom_ops::CustomMlaPreprocess);
