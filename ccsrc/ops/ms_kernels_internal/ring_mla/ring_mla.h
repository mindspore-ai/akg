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

#ifndef CCSRC_OPS_MS_KERNELS_INTERNAL_RING_MLA_RING_MLA_H_
#define CCSRC_OPS_MS_KERNELS_INTERNAL_RING_MLA_RING_MLA_H_

#include <map>
#include <string>
#include <utility>
#include <vector>
#include "internal_kernel_mod.h"
#include "mindspore/core/include/mindapi/ir/tensor.h"
#include "mindspore/ops/kernel/ascend/acl_ir/acl_convert.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "mindspore/ccsrc/ms_extension/api.h"
#include "mindspore/core/include/ops/base_operator.h"
#include "mindspore/core/include/ops/ops_func_impl/op_func_impl.h"
#include "mindspore/core/include/ops/ops_func_impl/simple_infer.h"
#include "mindspore/core/include/utils/check_convert_utils.h"

namespace {
// shape rank
constexpr auto QKV_SHAPE_RANK = 3; // [sum(seqlen), headNum, headSize]
constexpr auto LSE_SHAPE_RANK = 2; // [headNum, qNTokens]
// query, key, value dim index
constexpr auto QKV_N_TOKENS_IDX = 0;
constexpr auto QKV_HEAD_NUM_IDX = 1;
constexpr auto QKV_HEAD_DIM_IDX = 2;
constexpr auto QK_SPLIT1_HEAD_DIM = 128;
constexpr auto QK_SPLIT2_HEAD_DIM = 64;
// lse dim index
constexpr auto LSE_N_TOKENS_IDX = 1;
constexpr auto LSE_HEAD_NUM_IDX = 0;
// seqlen, mask index
constexpr auto SEQLEN_BATCH_IDX = 0;

enum RingMLAInputIndex : int {
  kQueryIdx = 0,
  kQueryRopeIdx,
  kKeyIdx,
  kKeyRopeIdx,
  kValueIdx,
  kMaskIdx,
  kAlibiCoeffIdx,
  kDeqQKIdx,
  kOffsetQKIdx,
  kDeqPVIdx,
  kOffsetPVIdx,
  kQuantPIdx,
  kLogNIdx,
  kOPrevIdx,
  kLsePrevIdx,
  kQSeqLenIdx,
  kKVSeqLenIdx,
  kHeadNumIdx,
  kQkScaleIdx,
  kKvHeadNumIdx,
  kMaskTypeIdx,
  kCalcTypeIdx,
  kRingMLAInputNums
};

enum RingMLAOutputIndex : int {
  kAttentionOutIdx = 0,
  kSoftmaxLseOutIdx,
  kRingMLAOutputNums
};
}  // namespace

namespace ms_custom_ops {

class OPS_API CustomRingMLAOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive,
                        const InferInfoPtrList &input_infos) const override;
  std::vector<TypeId> InferType(const PrimitivePtr &primitive,
                                const InferInfoPtrList &input_infos) const override;
  bool GeneralInferRegistered() const override { return true; }
  std::set<int64_t> GetValueDependArgIndices() const override {
    return {kQSeqLenIdx, kKVSeqLenIdx, kHeadNumIdx, kQkScaleIdx, kKvHeadNumIdx, kMaskTypeIdx, kCalcTypeIdx};
  };

 protected:
  void CheckInputShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const;

 private:
  mutable bool is_input_softmax_lse_{false};
};

class CustomRingMLA : public InternalKernelMod {
 public:
  CustomRingMLA() = default;
  ~CustomRingMLA() override = default;
  void InitKernelInputsOutputsIndex() override;

 protected:
  internal::InternalOpPtr CreateKernel(
      const internal::InputsImmutableInfoList &inputs,
      const internal::OutputsImmutableInfoList &outputs,
      const std::vector<KernelTensor *> &ms_inputs,
      const std::vector<KernelTensor *> &ms_outputs) override;
  bool UpdateParam(const std::vector<KernelTensor *> &inputs,
                   const std::vector<KernelTensor *> &outputs) override;
  uint64_t GenerateTilingKey(const std::vector<KernelTensor *> &inputs) override;

 private:
  bool RingMLAParamCheck(const internal::RingMLAParam &op_param);
  bool created_flag_{false};
  internal::RingMLAParam param_;
};

}  // namespace ms_custom_ops

#endif  // CCSRC_OPS_MS_KERNELS_INTERNAL_RING_MLA_RING_MLA_H_
