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
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

#include "ccsrc/base/ms_kernels_internal/pyboost/internal_pyboost_runner.h"
#include "ccsrc/ops/ms_kernels_internal/mla_preprocess/mla_preprocess_common.h"
#include "ccsrc/utils/utils.h"

namespace ms_custom_ops {
class MlaPreprocessLoadRunner : public InternalPyboostRunner {
public:
  using InternalPyboostRunner::InternalPyboostRunner;
  void SetParamCacheMode(const int32_t &cache_mode) { this->cache_mode_ = cache_mode; }
  internal::MlaPreprocessParam param_;
protected:
  internal::InternalOpPtr
  CreateKernel(const internal::InputsImmutableInfoList &inputs,
               const internal::OutputsImmutableInfoList &outputs) override {
    return CreateMlaPreprocessOpWithFormat(inputs, outputs, param_);
  }

private:
  int32_t n_{0};
  int32_t head_num_{0};
  int32_t cache_mode_{0};
};

std::vector<ms::Tensor> npu_mla_preprocess(const ms::Tensor &input1,
                                           const ms::Tensor &gamma1,
                                           const ms::Tensor &beta1,
                                           const ms::Tensor &quant_scale1,
                                           const ms::Tensor &quant_offset1,
                                           const ms::Tensor &wdqkv,
                                           const ms::Tensor &bias1,
                                           const ms::Tensor &gamma2,
                                           const ms::Tensor &beta2,
                                           const ms::Tensor &quant_scale2,
                                           const ms::Tensor &quant_offset2,
                                           const ms::Tensor &gamma3,
                                           const ms::Tensor &sin1,
                                           const ms::Tensor &cos1,
                                           const ms::Tensor &sin2,
                                           const ms::Tensor &cos2,
                                           const ms::Tensor &key_cache,
                                           const ms::Tensor &slot_mapping,
                                           const ms::Tensor &wuq,
                                           const ms::Tensor &bias2,
                                           const ms::Tensor &wuk,
                                           const ms::Tensor &de_scale1,
                                           const ms::Tensor &de_scale2,
                                           const ms::Tensor &ctkv_scale,
                                           const ms::Tensor &qnope_scale,
                                           const ms::Tensor &krope_cache,
                                           const int64_t param_cache_mode) {
  auto op_name = "MlaPreprocess";
  auto runner = std::make_shared<MlaPreprocessLoadRunner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);

  // Set head_num if provided
  runner->SetParamCacheMode(static_cast<int32_t>(param_cache_mode));
  runner->param_.n = 0;
  runner->param_.head_num = 0;
  runner->param_.cache_mode = param_cache_mode;

  // Setup the runner with all parameters (including hash calculation)
  runner->Setup(op_name, input1, gamma1, beta1, quant_scale1, quant_offset1, wdqkv, bias1, gamma2, beta2, quant_scale2,
                quant_offset2, gamma3, sin1, cos1, sin2, cos2, key_cache, slot_mapping, wuq, bias2, wuk, de_scale1,
                de_scale2, ctkv_scale, qnope_scale, krope_cache, param_cache_mode);
  std::vector<ms::Tensor> inputs = {input1, gamma1, beta1, quant_scale1, quant_offset1, wdqkv, bias1, gamma2, beta2,
                                    quant_scale2, quant_offset2, gamma3, sin1, cos1, sin2, cos2, key_cache,
                                    slot_mapping, wuq, bias2, wuk, de_scale1, de_scale2, ctkv_scale, qnope_scale,
                                    krope_cache};
  auto head_dim = key_cache.shape()[3];
  auto n = input1.shape()[0];
  auto head_num = wuk.shape()[0];
  ShapeVector q_out_shape{n, head_num, head_dim};
  ShapeVector key_out_shape{0};
  ShapeVector qrope_out_shape{};
  ShapeVector krope_out_shape{};
  if (param_cache_mode != kMlaPreCacheModeQK) {
    q_out_shape = {n, head_num, 512};
    key_out_shape = {0};
    qrope_out_shape = {n, head_num, 64};
    krope_out_shape = {0};
  }

  auto q_out = ms::Tensor(input1.data_type(), q_out_shape);
  auto key_out = ms::Tensor(input1.data_type(), key_out_shape);
  auto qrope_out = ms::Tensor(input1.data_type(), qrope_out_shape);
  auto krope_out = ms::Tensor(input1.data_type(), krope_out_shape);
  if (param_cache_mode == kMlaPreCacheModeQKSplitQuant) {
    q_out = ms::Tensor(quant_offset1.data_type(), q_out_shape);
    key_out = ms::Tensor(quant_offset1.data_type(), key_out_shape);
  }
  
  std::vector<ms::Tensor> outputs = {q_out, key_out, qrope_out, krope_out};
  runner->GetOrCreateKernel(inputs, outputs);
  runner->Run(inputs, outputs);
  return outputs;
}

auto pyboost_mla_preprocess(const ms::Tensor &input1,
                            const ms::Tensor &gamma1,
                            const ms::Tensor &beta1,
                            const ms::Tensor &quant_scale1,
                            const ms::Tensor &quant_offset1,
                            const ms::Tensor &wdqkv,
                            const ms::Tensor &bias1,
                            const ms::Tensor &gamma2,
                            const ms::Tensor &beta2,
                            const ms::Tensor &quant_scale2,
                            const ms::Tensor &quant_offset2,
                            const ms::Tensor &gamma3,
                            const ms::Tensor &sin1,
                            const ms::Tensor &cos1,
                            const ms::Tensor &sin2,
                            const ms::Tensor &cos2,
                            const ms::Tensor &key_cache,
                            const ms::Tensor &slot_mapping,
                            const ms::Tensor &wuq,
                            const ms::Tensor &bias2,
                            const ms::Tensor &wuk,
                            const ms::Tensor &de_scale1,
                            const ms::Tensor &de_scale2,
                            const ms::Tensor &ctkv_scale,
                            const ms::Tensor &qnope_scale,
                            const ms::Tensor &krope_cache,
                            const int64_t param_cache_mode) {
  return ms::pynative::PyboostRunner::Call<4>(
      npu_mla_preprocess, input1, gamma1, beta1, quant_scale1, quant_offset1, wdqkv, bias1, gamma2, beta2,
      quant_scale2, quant_offset2, gamma3, sin1, cos1, sin2, cos2, key_cache, slot_mapping, wuq, bias2, wuk, de_scale1,
      de_scale2, ctkv_scale, qnope_scale, krope_cache, param_cache_mode);
}
} // namespace ms_custom_ops

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("mla_preprocess", &ms_custom_ops::pyboost_mla_preprocess, "MlaPreprocess",
    pybind11::arg("input1"), pybind11::arg("gamma1"), pybind11::arg("beta1"), pybind11::arg("quant_scale1"),
    pybind11::arg("quant_offset1"), pybind11::arg("wdqkv"), pybind11::arg("bias1"), pybind11::arg("gamma2"),
    pybind11::arg("beta2"), pybind11::arg("quant_scale2"), pybind11::arg("quant_offset2"), pybind11::arg("gamma3"),
    pybind11::arg("sin1"), pybind11::arg("cos1"), pybind11::arg("sin2"), pybind11::arg("cos2"),
    pybind11::arg("key_cache"), pybind11::arg("slot_mapping"), pybind11::arg("wuq"), pybind11::arg("bias2"),
    pybind11::arg("wuk"), pybind11::arg("de_scale1"), pybind11::arg("de_scale2"), pybind11::arg("ctkv_scale"),
    pybind11::arg("qnope_scale"), pybind11::arg("krope_cache"), pybind11::arg("param_cache_mode"));
}
