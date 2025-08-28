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
#include "ccsrc/ops/ms_kernels_internal/utils/attention_utils.h"
#include "ccsrc/base/ms_kernels_internal/graphmode/internal_kernel_mod.h"
#include "mindspore/core/include/ir/tensor.h"
#include "mindspore/ops/kernel/ascend/acl_ir/acl_convert.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "mindspore/ccsrc/ms_extension/api.h"
#include "mindspore/core/include/ops/base_operator.h"
#include "mindspore/core/include/ops/ops_func_impl/op_func_impl.h"
#include "mindspore/core/include/ops/ops_func_impl/simple_infer.h"
#include "mindspore/ccsrc/runtime/device/kernel_runtime.h"
#include "mindspore/core/include/utils/check_convert_utils.h"

namespace ms_custom_ops {
static constexpr auto kMLAQshapeRank = 3;
static constexpr auto kMLAKVshapeRank = 4;
static constexpr auto kMLABlockSizeDim = 1;
static constexpr auto kMLABlockTablesRank = 2;
static constexpr auto kMLAMaskRank = 2;
static constexpr auto kMLADeqScaleRank = 1;
static constexpr auto kMLAMaskFreeLastDim = 128;
static constexpr auto kMLAQKVnopeHiddenSize = 512;
static constexpr auto kMLAQKropeHiddenSize = 64;
static constexpr auto kMLAQheadMax = 128;
static constexpr auto kMLABlockSizeheadMax = 128;

#define ALIGN_16(v) (((v) & (16 - 1)) == 0)

static void CheckParam(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) {
  auto kv_heads = input_infos[kMlaInputNumKVHeadIndex]->GetScalarValue<int64_t>();
  if (kv_heads.has_value()) {
    MS_CHECK_VALUE(kv_heads.value() == 1, CheckAndConvertUtils::FormatCommMsg(
                                            "For MLA The kv_head_num must be 1 , but got : ", kv_heads.value()));
  }

  auto q_heads = input_infos[kMlaInputNumHeadIndex]->GetScalarValue<int64_t>();
  if (q_heads.has_value()) {
    MS_CHECK_VALUE(q_heads.value() <= kMLAQheadMax,
                   CheckAndConvertUtils::FormatCommMsg("For MLA The head_num must be <= ", kMLAQheadMax,
                                                       ", but got : ", q_heads.value()));
    MS_CHECK_VALUE(ALIGN_16(q_heads.value()),
                   CheckAndConvertUtils::FormatCommMsg("For MLA The head_num must be the multiple of 16, but got : ",
                                                       q_heads.value()));

    if (q_heads.value() == kMLAQheadMax) {
      auto q_nope_type = input_infos[kMlaInputQnopeIndex]->GetType();
      if (q_nope_type == kNumberTypeInt8) {
        MS_LOG(EXCEPTION) << "For MLA int8 is not support when head_num=128.";
      }
    }
  }
}

static void CheckShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) {
  auto q_nope_shape = input_infos[kMlaInputQnopeIndex]->GetShape();
  auto q_rope_shape = input_infos[kMlaInputQropeIndex]->GetShape();
  auto ctkv_shape = input_infos[kMlaInputKvCacheIndex]->GetShape();
  auto k_rope_shape = input_infos[kMlaInputKropeIndex]->GetShape();
  auto block_tables_shape = input_infos[kMlaInputBlockTablesIndex]->GetShape();
  auto q_len_shape = input_infos[kMlaInputQueryLensIndex]->GetShape();
  auto context_len_shape = input_infos[kMlaInputContextLensIndex]->GetShape();

  if (!input_infos[kMlaInputQnopeIndex]->IsDynamic()) {
    MS_CHECK_VALUE(q_nope_shape.size() == kMLAQshapeRank,
                   CheckAndConvertUtils::FormatCommMsg("For MLA The rank of q_nope must be ", kMLAQshapeRank,
                                                       ", but got shape: ", q_nope_shape));
    MS_CHECK_VALUE(q_nope_shape[q_nope_shape.size() - 1] == kMLAQKVnopeHiddenSize,
                   CheckAndConvertUtils::FormatCommMsg("For MLA The last dim of q_nope must be ", kMLAQKVnopeHiddenSize,
                                                       ", but got shape: ", q_nope_shape));
  }

  if (!input_infos[kMlaInputQropeIndex]->IsDynamic()) {
    MS_CHECK_VALUE(q_rope_shape.size() == kMLAQshapeRank,
                   CheckAndConvertUtils::FormatCommMsg("For MLA The rank of q_rope must be ", kMLAQshapeRank,
                                                       ", but got shape: ", q_rope_shape));
    MS_CHECK_VALUE(q_rope_shape[q_rope_shape.size() - 1] == kMLAQKropeHiddenSize,
                   CheckAndConvertUtils::FormatCommMsg("For MLA The last dim of q_rope must be ", kMLAQKropeHiddenSize,
                                                       ", but got shape: ", q_rope_shape));
  }

  if (!input_infos[kMlaInputKvCacheIndex]->IsDynamic()) {
    MS_CHECK_VALUE(ctkv_shape.size() == kMLAKVshapeRank,
                   CheckAndConvertUtils::FormatCommMsg("For MLA The rank of ctkv must be ", kMLAKVshapeRank,
                                                       ", but got shape: ", ctkv_shape));
    MS_CHECK_VALUE(ctkv_shape[ctkv_shape.size() - 1] == kMLAQKVnopeHiddenSize,
                   CheckAndConvertUtils::FormatCommMsg("For MLA The last dim of ctkv must be ", kMLAQKVnopeHiddenSize,
                                                       ", but got shape: ", ctkv_shape));
    MS_CHECK_VALUE(ALIGN_16(ctkv_shape[kMLABlockSizeDim]),
                   CheckAndConvertUtils::FormatCommMsg("For MLA The block_size must be the multiple of 16 , but got: ",
                                                       ctkv_shape[kMLABlockSizeDim]));

    auto q_heads = input_infos[kMlaInputNumHeadIndex]->GetScalarValue<int64_t>();
    if (q_heads.has_value()) {
      if (q_heads.value() == kMLAQheadMax) {
        if (ctkv_shape[kMLABlockSizeDim] != kMLAQheadMax) {
          MS_LOG(EXCEPTION) << "For MLA the block_size must be 128 when "
                               "head_num is 128, but got block_size: "
                            << ctkv_shape[kMLABlockSizeDim];
        }
      }
    }
  }

  if (!input_infos[kMlaInputKropeIndex]->IsDynamic()) {
    MS_CHECK_VALUE(k_rope_shape.size() == kMLAKVshapeRank,
                   CheckAndConvertUtils::FormatCommMsg("For MLA The rank of k_rope must be ", kMLAKVshapeRank,
                                                       ", but got shape: ", k_rope_shape));
    MS_CHECK_VALUE(k_rope_shape[k_rope_shape.size() - 1] == kMLAQKropeHiddenSize,
                   CheckAndConvertUtils::FormatCommMsg("For MLA The last dim of k_rope must be ", kMLAQKropeHiddenSize,
                                                       ", but got shape: ", k_rope_shape));
  }

  if (!input_infos[kMlaInputBlockTablesIndex]->IsDynamic()) {
    MS_CHECK_VALUE(block_tables_shape.size() == kMLABlockTablesRank,
                   CheckAndConvertUtils::FormatCommMsg("For MLA The rank of block_tables must be ", kMLABlockTablesRank,
                                                       ", but got shape: ", block_tables_shape));
  }

  if (!input_infos[kMlaInputAttnMaskIndex]->IsNone() && !input_infos[kMlaInputAttnMaskIndex]->IsDynamic()) {
    auto mask_shape = input_infos[kMlaInputAttnMaskIndex]->GetShape();
    auto mask_type_value = input_infos[kMlaInputMaskTypeIndex]->GetScalarValue<int64_t>();
    if (!mask_type_value.has_value()) {
      MS_EXCEPTION(ValueError) << "For MLA mask_type must be constant but got variable.";
    }

    auto mask_type = mask_type_value.value();
    if (mask_type == kMaskSpec || mask_type == kMaskFree) {
      MS_CHECK_VALUE(mask_shape.size() == kMLAMaskRank,
                     CheckAndConvertUtils::FormatCommMsg("For MLA The rank of mask must be ", kMLAMaskRank,
                                                         ", but got shape: ", mask_shape));
    }

    if (mask_type == kMaskFree) {
      MS_CHECK_VALUE(mask_shape[mask_shape.size() - 1] == kMLAMaskFreeLastDim,
                     CheckAndConvertUtils::FormatCommMsg("For MLA The last dim of mask must be ", kMLAMaskFreeLastDim,
                                                         ", when mask_type is MASK_FREE but got shape: ", mask_shape));
    }
  }

  if (!input_infos[kMlaInputDeqScaleQkIndex]->IsNone()) {
    auto deq_scale_qk_shape = input_infos[kMlaInputDeqScaleQkIndex]->GetShape();
    MS_CHECK_VALUE(deq_scale_qk_shape.size() == kMLADeqScaleRank,
                   CheckAndConvertUtils::FormatCommMsg("For MLA The rank of deq_scale_qk must be ", kMLADeqScaleRank,
                                                       ", but got shape: ", deq_scale_qk_shape));
  }

  if (!input_infos[kMlaInputDeqScalePvIndex]->IsNone()) {
    auto deq_scale_pv_shape = input_infos[kMlaInputDeqScalePvIndex]->GetShape();

    MS_CHECK_VALUE(deq_scale_pv_shape.size() == kMLADeqScaleRank,
                   CheckAndConvertUtils::FormatCommMsg("For MLA The rank of deq_scale_pv must be ", kMLADeqScaleRank,
                                                       ", but got shape: ", deq_scale_pv_shape));
  }

  MS_CHECK_VALUE(q_len_shape.size() == kMLADeqScaleRank,
                 CheckAndConvertUtils::FormatCommMsg("For MLA The rank of q_seq_lens must be ", kMLADeqScaleRank,
                                                     ", but got shape: ", q_len_shape));
  MS_CHECK_VALUE(context_len_shape.size() == kMLADeqScaleRank,
                 CheckAndConvertUtils::FormatCommMsg("For MLA The rank of context_lengths must be ", kMLADeqScaleRank,
                                                     ", but got shape: ", context_len_shape));
  if (!input_infos[kMlaInputQueryLensIndex]->IsDynamic() && !input_infos[kMlaInputContextLensIndex]->IsDynamic()) {
    MS_CHECK_VALUE(context_len_shape[0] == q_len_shape[0],
                   CheckAndConvertUtils::FormatCommMsg("For MLA The shape of context_lengths and q_seq_lens "
                                                       "must be same but got context_len_shape: ",
                                                       context_len_shape, ", q_len_shape: ", q_len_shape));
  }
}

class OPS_API MlaFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto &q_nope_info = input_infos[kMlaInputQnopeIndex];
    auto q_nope_shape = q_nope_info->GetShape();
    auto is_ring_value = input_infos[kMlaInputIsRingIndex]->GetScalarValue<int64_t>();
    if (!is_ring_value.has_value()) {
      MS_EXCEPTION(ValueError) << "For MLA, the ring must be a constant, but got a variable.";
    }

    auto is_ring = is_ring_value.value();
    if (is_ring != 0) {
      MS_EXCEPTION(ValueError) << "For MLA, ir_ring must be 0 now, but got: " << is_ring;
    }

    CheckShape(primitive, input_infos);
    CheckParam(primitive, input_infos);

    return {q_nope_shape, {0}};
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto q_nope_type = input_infos[kMlaInputQnopeIndex]->GetType();
    auto q_rope_type = input_infos[kMlaInputQropeIndex]->GetType();

    return {q_rope_type, q_nope_type};
  }

  bool GeneralInferRegistered() const override { return true; }

  std::set<int64_t> GetValueDependArgIndices() const override {
    return {kMlaInputQueryLensIndex, kMlaInputContextLensIndex, kMlaInputNumHeadIndex,     kMlaInputScaleValueIndex,
            kMlaInputNumKVHeadIndex, kMlaInputMaskTypeIndex,    kMlaInputInputFormatIndex, kMlaInputIsRingIndex};
  };
};

class Mla : public InternalKernelMod {
 public:
  Mla() : InternalKernelMod() {}
  ~Mla() = default;

 protected:
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs,
                                       const std::vector<KernelTensor *> &ms_inputs,
                                       const std::vector<KernelTensor *> &ms_outputs) override {
    param_.type = internal::MLAParam::kSplitCache;
    param_.head_size = static_cast<int32_t>(ms_inputs[kMlaInputNumHeadIndex]->GetValueWithCheck<int64_t>());
    param_.tor = ms_inputs[kMlaInputScaleValueIndex]->GetValueWithCheck<float>();
    param_.kv_head = static_cast<int32_t>(ms_inputs[kMlaInputNumKVHeadIndex]->GetValueWithCheck<int64_t>());
    param_.mask_type =
      static_cast<internal::MLAParam::MaskType>(ms_inputs[kMlaInputMaskTypeIndex]->GetValueWithCheck<int64_t>());
    param_.is_ring = static_cast<int32_t>(ms_inputs[kMlaInputIsRingIndex]->GetValueWithCheck<int64_t>());

    param_.q_seq_len = ms_inputs[kMlaInputQueryLensIndex]->GetValueWithCheck<std::vector<int32_t>>();
    param_.kv_seq_len = ms_inputs[kMlaInputContextLensIndex]->GetValueWithCheck<std::vector<int32_t>>();

    auto input_format = static_cast<MlaInputFormat>(ms_inputs[kMlaInputInputFormatIndex]->GetValueWithCheck<int64_t>());
    created_flag_ = true;
    if (input_format == kKVFormatNZ) {
      auto inputs_new = inputs;
      inputs_new[kMlaInputKvCacheIndex].SetFormat(internal::kFormatFRACTAL_NZ);
      inputs_new[kMlaInputKropeIndex].SetFormat(internal::kFormatFRACTAL_NZ);
      return internal::CreateMLAOp(inputs_new, outputs, param_, internal::kInternalMLAOpName);
    }

    return internal::CreateMLAOp(inputs, outputs, param_, internal::kInternalMLAOpName);
  }

  bool UpdateParam(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    if (created_flag_) {
      // the q_seq_len and batch_valid_length are inited in CreateKernel, so
      // there is no need to load them again
      created_flag_ = false;
      return true;
    }

    auto q_need_recreate = GetSeqLenAndCheckUpdate(inputs[kMlaInputQueryLensIndex], &param_.q_seq_len);
    auto kv_need_recreate = GetSeqLenAndCheckUpdate(inputs[kMlaInputContextLensIndex], &param_.kv_seq_len);
    if (q_need_recreate || kv_need_recreate) {
      auto ret = internal_op_->UpdateParam(&param_);
      if (ret != internal::kInternalOk) {
        MS_LOG(ERROR) << "InternalMla UpdateParam failed, kernel_name: " << kernel_name_;
        return false;
      }
      return true;
    }

    return true;
  }

  uint64_t GenerateTilingKey(const std::vector<KernelTensor *> &inputs) override {
    // User defined CacheKey, the inputs should include all the factors which
    // will affect tiling result.
    return InternalTilingCache::GenerateKey(kernel_name_, inputs, param_.q_seq_len, param_.kv_seq_len);
  }

  void InitKernelInputsOutputsIndex() override {
    kernel_inputs_index_ = {kMlaInputQnopeIndex,      kMlaInputQropeIndex,       kMlaInputKvCacheIndex,
                            kMlaInputKropeIndex,      kMlaInputBlockTablesIndex, kMlaInputAttnMaskIndex,
                            kMlaInputDeqScaleQkIndex, kMlaInputDeqScalePvIndex};
    kernel_outputs_index_ = {0, 1};
  }

 private:
  bool created_flag_{false};
  internal::MLAParam param_;
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(mla, ms_custom_ops::MlaFuncImpl, ms_custom_ops::Mla);
