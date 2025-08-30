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
#include "ccsrc/ops/ms_kernels_internal/paged_cache_load/paged_cache_load_common.h"
#include "ccsrc/utils/utils.h"

namespace ms_custom_ops {
class PagedCacheLoadRunner : public InternalPyboostRunner {
public:
  using InternalPyboostRunner::InternalPyboostRunner;
  void SetKvCacheCfg(const int32_t &kv_cache_cfg) { this->kv_cache_cfg_ = kv_cache_cfg; }
  void SetIsSeqLensCumsumType(const bool &is_seq_lens_cumsum_type) {
    this->is_seq_lens_cumsum_type_ = is_seq_lens_cumsum_type;
  }
  void SetHasSeqStarts(const bool &has_seq_starts) { this->has_seq_starts_ = has_seq_starts; }
  internal::PagedCacheLoadParam param_;
protected:
  internal::InternalOpPtr
  CreateKernel(const internal::InputsImmutableInfoList &inputs,
               const internal::OutputsImmutableInfoList &outputs) override {
    return CreatePagedCacheLoadOpWithFormat(inputs, outputs, param_);
  }

private:
  int32_t kv_cache_cfg_{0};
  bool is_seq_lens_cumsum_type_{false};
  bool has_seq_starts_{false};
};

std::vector<ms::Tensor> npu_paged_cache_load(const ms::Tensor &key_cache,
                                             const ms::Tensor &value_cache,
                                             const ms::Tensor &block_table,
                                             const ms::Tensor &seq_lens,
                                             const ms::Tensor &key,
                                             const ms::Tensor &value,
                                             const std::optional<ms::Tensor> &seq_starts,
                                             std::optional<int64_t> kv_cache_cfg,
                                             std::optional<bool> is_seq_lens_cumsum_type,
                                             std::optional<bool> has_seq_starts) {
  auto op_name = "PagedCacheLoad";
  auto runner = std::make_shared<PagedCacheLoadRunner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);

  // Set head_num if provided
  if (kv_cache_cfg.has_value()) {
    runner->SetKvCacheCfg(static_cast<int32_t>(kv_cache_cfg.value()));
  }
  if (is_seq_lens_cumsum_type.has_value()) {
    runner->SetIsSeqLensCumsumType(is_seq_lens_cumsum_type.value());
  }
  if (has_seq_starts.has_value()) {
    runner->SetHasSeqStarts(has_seq_starts.value());
  }
  runner->param_.kv_cache_cfg_type = static_cast<int32_t>(kv_cache_cfg.value());
  runner->param_.is_seq_lens_cumsum_type = is_seq_lens_cumsum_type.value();
  runner->param_.has_seq_starts = has_seq_starts.value();

  // Setup the runner with all parameters (including hash calculation)
  runner->Setup(op_name, key_cache, value_cache, block_table, seq_lens, key, value, seq_starts, kv_cache_cfg,
                is_seq_lens_cumsum_type, has_seq_starts);
  std::vector<ms::Tensor> inputs = {key_cache, value_cache, block_table, seq_lens, key, value,
                                    GetTensorOrEmpty(seq_starts)};
  std::vector<ms::Tensor> outputs = {key, value};
  runner->GetOrCreateKernel(inputs, outputs);
  runner->Run(inputs, outputs);
  return outputs;
}

auto pyboost_paged_cache_load(const ms::Tensor &key_cache,
                              const ms::Tensor &value_cache,
                              const ms::Tensor &block_table,
                              const ms::Tensor &seq_lens,
                              const ms::Tensor &key,
                              const ms::Tensor &value,
                              const std::optional<ms::Tensor> &seq_starts,
                              std::optional<int64_t> kv_cache_cfg,
                              std::optional<bool> is_seq_lens_cumsum_type,
                              std::optional<bool> has_seq_starts) {
  return ms::pynative::PyboostRunner::Call<2>(
      npu_paged_cache_load, key_cache, value_cache, block_table, seq_lens, key, value, seq_starts,
      kv_cache_cfg, is_seq_lens_cumsum_type, has_seq_starts);
}
} // namespace ms_custom_ops

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("paged_cache_load", &ms_custom_ops::pyboost_paged_cache_load, "Paged Cache Load",
    pybind11::arg("key_cache"), pybind11::arg("value_cache"),
    pybind11::arg("block_table"), pybind11::arg("seq_lens"),
    pybind11::arg("key"), pybind11::arg("value"),
    pybind11::arg("seq_starts") = std::nullopt,
    pybind11::arg("kv_cache_cfg") = std::nullopt,
    pybind11::arg("is_seq_lens_cumsum_type") = std::nullopt,
    pybind11::arg("has_seq_starts") = std::nullopt);
}
