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
#include "ccsrc/ops/ms_kernels_internal/mla/mla_common.h"
#include "ccsrc/base/ms_kernels_internal/pyboost/internal_pyboost_runner.h"
#include "ccsrc/utils/utils.h"
#include "mindspore/ccsrc/ms_extension/api.h"
#include "lib/plugin/ascend/ms_kernels_internal/internal_kernel/include/internal.h"

namespace ms_custom_ops {
class MlaRunner : public InternalPyboostRunner {
 public:
  MlaRunner(const std::string &op_name) : InternalPyboostRunner(op_name) {}
  ~MlaRunner() = default;

  void UpdateParam(int32_t head_size, float tor, int32_t kv_head, mindspore::internal::MLAParam::MaskType mask_type,
                   int32_t is_ring, const std::vector<int32_t> &q_seq_len, const std::vector<int32_t> &kv_seq_len) {
    param_.type = mindspore::internal::MLAParam::kSplitCache;
    param_.head_size = head_size;
    param_.tor = tor;
    param_.kv_head = kv_head;
    param_.mask_type = mask_type;
    param_.is_ring = is_ring;
    param_.q_seq_len = q_seq_len;
    param_.kv_seq_len = kv_seq_len;
  }

 protected:
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs) override {
    return mindspore::internal::CreateMLAOp(inputs, outputs, param_, internal::kInternalMLAOpName);
  }

 private:
  mindspore::internal::MLAParam param_;
};

std::vector<ms::Tensor> mla_atb(const ms::Tensor &q_nope, const ms::Tensor &q_rope, const ms::Tensor &ctkv,
                                const ms::Tensor &k_rope, const ms::Tensor &block_tables,
                                const std::optional<ms::Tensor> &attn_mask,
                                const std::optional<ms::Tensor> &deq_scale_qk,
                                const std::optional<ms::Tensor> &deq_scale_pv,
                                const std::optional<ms::Tensor> &q_seq_lens,
                                const std::optional<ms::Tensor> &context_lens, int64_t head_num, double scale_value,
                                int64_t kv_head_num, int64_t mask_mode, int64_t is_ring) {
  static auto op_name = "Mla";
  auto runner = std::make_shared<MlaRunner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);

  if (!q_seq_lens.has_value() || !context_lens.has_value()) {
    MS_LOG(EXCEPTION) << "For " << op_name
                      << ", the q_seq_lens and context_lens can not be None, but got q_seq_lens.has_value(): "
                      << q_seq_lens.has_value() << ", context_lens.has_value(): " << context_lens.has_value();
  }

  auto q_seq_lens_value = GetValueFromTensor<std::vector<int32_t>>(q_seq_lens.value(), op_name, "q_seq_lens");
  auto context_lens_value = GetValueFromTensor<std::vector<int32_t>>(context_lens.value(), op_name, "context_lens");
  runner->UpdateParam(static_cast<int32_t>(head_num), static_cast<float>(scale_value),
                      static_cast<int32_t>(kv_head_num),
                      static_cast<mindspore::internal::MLAParam::MaskType>(mask_mode), static_cast<int32_t>(is_ring),
                      q_seq_lens_value, context_lens_value);

  // Setup the runner with all parameters (including hash calculation)
  runner->Setup(op_name, q_nope, q_rope, ctkv, k_rope, block_tables, attn_mask, deq_scale_qk, deq_scale_pv, q_seq_lens,
                context_lens, head_num, scale_value, kv_head_num, mask_mode, is_ring);

  auto attn_out = ms::Tensor(q_nope.data_type(), q_nope.shape());
  auto lse_out = ms::Tensor(q_nope.data_type(), {0});

  std::vector<ms::Tensor> inputs = {q_nope,
                                    q_rope,
                                    ctkv,
                                    k_rope,
                                    block_tables,
                                    GetTensorOrEmpty(attn_mask),
                                    GetTensorOrEmpty(deq_scale_qk),
                                    GetTensorOrEmpty(deq_scale_pv)};
  std::vector<ms::Tensor> outputs = {attn_out, lse_out};
  runner->GetOrCreateKernel(inputs, outputs);
  runner->Run(inputs, outputs);
  return outputs;
}

auto pyboost_mla(const ms::Tensor &q_nope, const ms::Tensor &q_rope, const ms::Tensor &ctkv, const ms::Tensor &k_rope,
                 const ms::Tensor &block_tables, const std::optional<ms::Tensor> &attn_mask,
                 const std::optional<ms::Tensor> &deq_scale_qk, const std::optional<ms::Tensor> &deq_scale_pv,
                 const std::optional<ms::Tensor> &q_seq_lens, const std::optional<ms::Tensor> &context_lens,
                 int64_t head_num, double scale_value, int64_t kv_head_num, int64_t mask_mode, int64_t is_ring) {
  return ms::pynative::PyboostRunner::Call<2>(mla_atb, q_nope, q_rope, ctkv, k_rope, block_tables, attn_mask,
                                              deq_scale_qk, deq_scale_pv, q_seq_lens, context_lens, head_num,
                                              scale_value, kv_head_num, mask_mode, is_ring);
}
}  // namespace ms_custom_ops

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("mla", &ms_custom_ops::pyboost_mla, "Multi-head Latent Attention", pybind11::arg("q_nope"),
        pybind11::arg("q_rope"), pybind11::arg("ctkv"), pybind11::arg("k_rope"), pybind11::arg("block_tables"),
        pybind11::arg("attn_mask") = std::nullopt, pybind11::arg("deq_scale_qk") = std::nullopt,
        pybind11::arg("deq_scale_pv") = std::nullopt, pybind11::arg("q_seq_lens") = std::nullopt,
        pybind11::arg("context_lens") = std::nullopt, pybind11::arg("head_num") = 32,
        pybind11::arg("scale_value") = 0.0, pybind11::arg("kv_head_num") = 1, pybind11::arg("mask_mode") = 0,
        pybind11::arg("is_ring") = 0);
}
