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

#include "ring_mla_runner.h"

using namespace ms_custom_ops;
namespace ms::pynative {

namespace {

inline bool GetSeqLenFromInputTensor(const ms::Tensor &input_tensor, std::vector<int32_t> *seq_len) {
  if (seq_len == nullptr) {
    MS_LOG(EXCEPTION) << "For GetSeqLenFromInputTensor, the seq_len ptr is nullptr.";
  }
  auto input_tensor_ptr = input_tensor.tensor();
  auto input_tensor_value = static_cast<int32_t *>(input_tensor_ptr->data_c());
  if (input_tensor_value == nullptr) {
    MS_LOG(EXCEPTION) << "For GetSeqLenFromInputTensor, the input_tensor_value is nullptr.";
  }
  auto input_tensor_value_num = input_tensor.numel();
  seq_len->clear();
  for (size_t i = 0; i < input_tensor_value_num; ++i) {
    seq_len->emplace_back(input_tensor_value[i]);
  }
  return true;
}

}  // namespace

void RingMLARunner::SetSeqLen(const std::optional<ms::Tensor> &q_seq_lens,
                              const std::optional<ms::Tensor> &context_lens) {
  if (!q_seq_lens.has_value() || !context_lens.has_value()) {
    MS_LOG(EXCEPTION) << "For RingMLARunner, the q_seq_lens and context_lens must not be None.";
    return;
  }
  (void)GetSeqLenFromInputTensor(q_seq_lens.value(), &param_.qSeqLen);
  (void)GetSeqLenFromInputTensor(context_lens.value(), &param_.kvSeqLen);
}

void RingMLARunner::SetRingMLAParam(int64_t head_num, float scale_value,
                                    int64_t kv_head_num, int64_t mask_type, int64_t calc_type) {
  param_.headNum = static_cast<int32_t>(head_num);
  param_.qkScale = scale_value;
  param_.kvHeadNum = static_cast<int32_t>(kv_head_num);
  param_.maskType = static_cast<internal::RingMLAParam::MaskType>(mask_type);
  param_.calcType = static_cast<internal::RingMLAParam::CalcType>(calc_type);
}

bool RingMLARunner::UpdateParam() {
  if (created_flag_) {
    created_flag_ = false;
    return true;
  }
  if (internal_op_ == nullptr) {
    MS_LOG(ERROR) << "RingMLARunner UpdateParam failed, internal_op_ is nullptr.";
    return false;
  }
  auto ret = internal_op_->UpdateParam(&param_);
  if (ret != internal::kInternalOk) {
    MS_LOG(ERROR) << "RingMLARunner UpdateParam failed.";
    return false;
  }
  return true;
}

internal::InternalOpPtr RingMLARunner::CreateKernel(
    const internal::InputsImmutableInfoList &inputs,
    const internal::OutputsImmutableInfoList &outputs) {
  created_flag_ = true;
  return internal::CreateRingMLAOp(inputs, outputs, param_, internal::kInternalRingMLAOpName);
}

MS_KERNELS_INTERNAL_NAME_REG(RingMLA, internal::kInternalRingMLAOpName);

}  // namespace ms::pynative

namespace ms_custom_ops {

namespace {

ms::Tensor ToTensorOrEmpty(const std::optional<ms::Tensor> &opt_tensor) {
  return opt_tensor.has_value() ? opt_tensor.value() : ms::Tensor();
}

ms::Tensor GenAttnOutTensor(const ms::Tensor &query) {
  return ms::Tensor(query.data_type(), query.shape());
}

ms::Tensor GenLseOutTensor(const ms::Tensor &query, const std::optional<ms::Tensor> &lse_prev,
                           const int64_t &calc_type) {
  using CalcType = internal::RingMLAParam::CalcType;
  bool is_ring = static_cast<CalcType>(calc_type) == CalcType::CALC_TYPE_DEFAULT;
  if (is_ring && lse_prev.has_value()) {
    return ms::Tensor(lse_prev.value().data_type(), lse_prev.value().shape());
  }

  constexpr size_t QKV_N_TOKENS_IDX = 0;
  constexpr size_t QKV_HEAD_NUM_IDX = 1;
  constexpr size_t LSE_N_TOKENS_IDX = 1;
  constexpr size_t LSE_HEAD_NUM_IDX = 0;
  constexpr size_t LSE_SHAPE_RANK = 2;  // [headNum, qNTokens]

  auto query_shape = query.shape();
  auto lse_out_shape = query_shape;
  lse_out_shape[LSE_N_TOKENS_IDX] = query_shape[QKV_N_TOKENS_IDX];
  lse_out_shape[LSE_HEAD_NUM_IDX] = query_shape[QKV_HEAD_NUM_IDX];
  lse_out_shape.resize(LSE_SHAPE_RANK);
  return ms::Tensor(TypeId::kNumberTypeFloat32, lse_out_shape);
}

}  // namespace

std::vector<ms::Tensor> npu_ring_mla(
    const ms::Tensor &query, const ms::Tensor &query_rope, const ms::Tensor &key,
    const ms::Tensor &key_rope, const ms::Tensor &value, const std::optional<ms::Tensor> &mask,
    const std::optional<ms::Tensor> &alibi_coeff, const std::optional<ms::Tensor> &deq_scale_qk,
    const std::optional<ms::Tensor> &deq_offset_qk, const std::optional<ms::Tensor> &deq_scale_pv,
    const std::optional<ms::Tensor> &deq_offset_pv, const std::optional<ms::Tensor> &quant_p,
    const std::optional<ms::Tensor> &log_n, const std::optional<ms::Tensor> &o_prev,
    const std::optional<ms::Tensor> &lse_prev, const std::optional<ms::Tensor> &q_seq_lens,
    const std::optional<ms::Tensor> &context_lens, const int64_t &head_num, const float &scale_value,
    const int64_t &kv_head_num, const int64_t &mask_type, const int64_t &calc_type) {
  const std::string op_name = "RingMLA";
  auto runner = std::make_shared<ms::pynative::RingMLARunner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);

  runner->SetRingMLAParam(head_num, scale_value, kv_head_num, mask_type, calc_type);
  runner->SetSeqLen(q_seq_lens, context_lens);

  // Setup the runner with all parameters (including hash calculation)
  runner->Setup(op_name, query, query_rope, key, key_rope, value, mask, alibi_coeff, deq_scale_qk, deq_offset_qk,
                deq_scale_pv, deq_offset_pv, quant_p, log_n, o_prev, lse_prev, q_seq_lens, context_lens,
                head_num, scale_value, kv_head_num, mask_type, calc_type);

  auto attn_out = GenAttnOutTensor(query);
  auto lse_out = GenLseOutTensor(query, lse_prev, calc_type);

  std::vector<ms::Tensor> inputs = {
      query, query_rope, key, key_rope, value,
      ToTensorOrEmpty(mask), ToTensorOrEmpty(alibi_coeff),
      ToTensorOrEmpty(deq_scale_qk), ToTensorOrEmpty(deq_offset_qk),
      ToTensorOrEmpty(deq_scale_pv), ToTensorOrEmpty(deq_offset_pv),
      ToTensorOrEmpty(quant_p), ToTensorOrEmpty(log_n),
      ToTensorOrEmpty(o_prev), ToTensorOrEmpty(lse_prev)
  };
  std::vector<ms::Tensor> outputs = {attn_out, lse_out};
  runner->GetOrCreateKernel(inputs, outputs);
  runner->Run(inputs, outputs);
  return outputs;
}

}  // namespace ms_custom_ops

auto pyboost_ring_mla(const ms::Tensor &query, const ms::Tensor &query_rope, const ms::Tensor &key,
                      const ms::Tensor &key_rope, const ms::Tensor &value, const std::optional<ms::Tensor> &mask,
                      const std::optional<ms::Tensor> &alibi_coeff, const std::optional<ms::Tensor> &deq_scale_qk,
                      const std::optional<ms::Tensor> &deq_offset_qk, const std::optional<ms::Tensor> &deq_scale_pv,
                      const std::optional<ms::Tensor> &deq_offset_pv, const std::optional<ms::Tensor> &quant_p,
                      const std::optional<ms::Tensor> &log_n, const std::optional<ms::Tensor> &o_prev,
                      const std::optional<ms::Tensor> &lse_prev, const ms::Tensor &q_seq_lens,
                      const ms::Tensor &context_lens, const int64_t &head_num, const float &scale_value,
                      const int64_t &kv_head_num, const int64_t &mask_type, const int64_t &calc_type) {
  return ms::pynative::PyboostRunner::Call<2>(
      ms_custom_ops::npu_ring_mla, query, query_rope, key, key_rope, value, mask, alibi_coeff, deq_scale_qk,
      deq_offset_qk, deq_scale_pv, deq_offset_pv, quant_p, log_n, o_prev, lse_prev, q_seq_lens, context_lens,
      head_num, scale_value, kv_head_num, mask_type, calc_type);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("ring_mla", &pyboost_ring_mla, "Ring MLA",
        pybind11::arg("query"),
        pybind11::arg("query_rope"),
        pybind11::arg("key"),
        pybind11::arg("key_rope"),
        pybind11::arg("value"),
        pybind11::arg("mask") = std::nullopt,
        pybind11::arg("alibi_coeff") = std::nullopt,
        pybind11::arg("deq_scale_qk") = std::nullopt,
        pybind11::arg("deq_offset_qk") = std::nullopt,
        pybind11::arg("deq_scale_pv") = std::nullopt,
        pybind11::arg("deq_offset_pv") = std::nullopt,
        pybind11::arg("quant_p") = std::nullopt,
        pybind11::arg("log_n") = std::nullopt,
        pybind11::arg("o_prev") = std::nullopt,
        pybind11::arg("lse_prev") = std::nullopt,
        pybind11::arg("q_seq_lens"),
        pybind11::arg("context_lens"),
        pybind11::arg("head_num"),
        pybind11::arg("scale_value"),
        pybind11::arg("kv_head_num"),
        pybind11::arg("mask_type"),
        pybind11::arg("calc_type"));
}
