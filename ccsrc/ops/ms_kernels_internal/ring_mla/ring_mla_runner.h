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

#ifndef CCSRC_OPS_MS_KERNELS_INTERNAL_RING_MLA_RING_MLA_RUNNER_H_
#define CCSRC_OPS_MS_KERNELS_INTERNAL_RING_MLA_RING_MLA_RUNNER_H_

#include <map>
#include <string>
#include <utility>
#include <vector>
#include <optional>

#include "internal_kernel_mod.h"
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_convert.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ms_extension/api.h"
#include "ops/base_operator.h"
#include "ops/ops_func_impl/op_func_impl.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "runtime/device/kernel_runtime.h"
#include "utils/check_convert_utils.h"
#include "internal_pyboost_runner.h"

using namespace ms_custom_ops;
namespace ms::pynative {

class RingMLARunner : public InternalPyboostRunner {
 public:
  using InternalPyboostRunner::InternalPyboostRunner;
  void SetSeqLen(const std::optional<ms::Tensor> &q_seq_lens,
                 const std::optional<ms::Tensor> &context_lens);
  void SetRingMLAParam(int64_t head_num, float scale_value,
                       int64_t kv_head_num, int64_t mask_type, int64_t calc_type);

 protected:
  bool UpdateParam() override;
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs) override;

 private:
  bool created_flag_{false};
  internal::RingMLAParam param_;
};

}  // namespace ms::pynative

#endif  // CCSRC_OPS_MS_KERNELS_INTERNAL_RING_MLA_RING_MLA_RUNNER_H_
