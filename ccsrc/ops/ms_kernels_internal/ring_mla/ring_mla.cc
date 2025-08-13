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

#include "ring_mla.h"

namespace ms_custom_ops {

void CustomRingMLAOpFuncImpl::CheckInputShape(const PrimitivePtr &primitive,
                                              const InferInfoPtrList &input_infos) const {
  // Helper lambda for shape check
  auto check_shape_rank = [](const std::vector<int64_t> &shape, size_t expected_rank, const std::string &name) {
    MS_CHECK_VALUE(shape.size() == expected_rank,
                   CheckAndConvertUtils::FormatCommMsg("For RingMLA The rank of " + name + " must be ", expected_rank,
                                                       ", but got shape: ", shape));
  };

  auto check_head_dim = [](const std::vector<int64_t> &shape, int64_t expected, const std::string &name) {
    MS_CHECK_VALUE(shape.back() == expected,
                   CheckAndConvertUtils::FormatCommMsg("For RingMLA The headDim of " + name + " must be ", expected,
                                                       ", but got shape: ", shape));
  };

  // query
  if (!input_infos[kQueryIdx]->IsDynamic()) {
    const auto &query_shape = input_infos[kQueryIdx]->GetShape();
    check_shape_rank(query_shape, QKV_SHAPE_RANK, "query");
    check_head_dim(query_shape, QK_SPLIT1_HEAD_DIM, "query");
  }

  // query_rope
  if (!input_infos[kQueryRopeIdx]->IsDynamic()) {
    const auto &query_rope_shape = input_infos[kQueryRopeIdx]->GetShape();
    check_shape_rank(query_rope_shape, QKV_SHAPE_RANK, "query_rope");
    check_head_dim(query_rope_shape, QK_SPLIT2_HEAD_DIM, "query_rope");
  }

  // key
  if (!input_infos[kKeyIdx]->IsDynamic()) {
    const auto &key_shape = input_infos[kKeyIdx]->GetShape();
    check_shape_rank(key_shape, QKV_SHAPE_RANK, "key");
    check_head_dim(key_shape, QK_SPLIT1_HEAD_DIM, "key");
  }

  // key_rope
  if (!input_infos[kKeyRopeIdx]->IsDynamic()) {
    const auto &key_rope_shape = input_infos[kKeyRopeIdx]->GetShape();
    check_shape_rank(key_rope_shape, QKV_SHAPE_RANK, "key_rope");
    check_head_dim(key_rope_shape, QK_SPLIT2_HEAD_DIM, "key_rope");
  }

  // value
  if (!input_infos[kValueIdx]->IsDynamic()) {
    const auto &value_shape = input_infos[kValueIdx]->GetShape();
    check_shape_rank(value_shape, QKV_SHAPE_RANK, "value");
    check_head_dim(value_shape, QK_SPLIT1_HEAD_DIM, "value");
  }

  if (is_input_softmax_lse_) {
    if (!input_infos[kOPrevIdx]->IsDynamic()) {
      const auto &prev_out_shape = input_infos[kOPrevIdx]->GetShape();
      check_shape_rank(prev_out_shape, QKV_SHAPE_RANK, "prev_out");
      check_head_dim(prev_out_shape, QK_SPLIT1_HEAD_DIM, "prev_out");
    }

    if (!input_infos[kLsePrevIdx]->IsDynamic()) {
      const auto &prev_lse_shape = input_infos[kLsePrevIdx]->GetShape();
      check_shape_rank(prev_lse_shape, LSE_SHAPE_RANK, "prev_lse");
    }
  }
}

ShapeArray CustomRingMLAOpFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  auto calc_type = static_cast<internal::RingMLAParam::CalcType>(
      input_infos[kCalcTypeIdx]->GetScalarValueWithCheck<int64_t>());
  is_input_softmax_lse_ = (calc_type == internal::RingMLAParam::CalcType::CALC_TYPE_DEFAULT);
  (void)CheckInputShape(primitive, input_infos);
  const auto &query_shape = input_infos[kQueryIdx]->GetShape();
  const auto &value_shape = input_infos[kValueIdx]->GetShape();
  ShapeVector attn_out_shape = query_shape;
  attn_out_shape[QKV_HEAD_DIM_IDX] = value_shape[QKV_HEAD_DIM_IDX];

  ShapeVector lse_out_shape;
  if (is_input_softmax_lse_) {
    lse_out_shape = input_infos[kLsePrevIdx]->GetShape();
    return {attn_out_shape, lse_out_shape};
  }
  lse_out_shape = query_shape;
  lse_out_shape[LSE_N_TOKENS_IDX] = query_shape[QKV_N_TOKENS_IDX];
  lse_out_shape[LSE_HEAD_NUM_IDX] = query_shape[QKV_HEAD_NUM_IDX];
  lse_out_shape.resize(LSE_SHAPE_RANK);
  return {attn_out_shape, lse_out_shape};
}

std::vector<TypeId> CustomRingMLAOpFuncImpl::InferType(const PrimitivePtr &primitive,
                                                       const InferInfoPtrList &input_infos) const {
  auto query_type = input_infos[kQueryIdx]->GetType();
  return {query_type, TypeId::kNumberTypeFloat32};
}

bool CustomRingMLA::RingMLAParamCheck(const internal::RingMLAParam &op_param) {
  if (op_param.calcType != internal::RingMLAParam::CalcType::CALC_TYPE_DEFAULT &&
      op_param.calcType != internal::RingMLAParam::CalcType::CALC_TYPE_FISRT_RING) {
    MS_LOG(ERROR) << "Ring MLA expects calcType to be one of CALC_TYPE_DEFAULT, CALC_TYPE_FISRT_RING. "
                  << "But got param.calcType = " << op_param.calcType;
    return false;
  }
  if (op_param.headNum <= 0) {
    MS_LOG(ERROR) << "Ring MLA expects headNum to be greater than zero, But got param.headNum = " << op_param.headNum;
    return false;
  }
  if (op_param.kvHeadNum < 0) {
    MS_LOG(ERROR) << "Ring MLA expects kvHeadNum to be no less than zero, "
                  << "But got param.kvHeadNum = " << op_param.kvHeadNum;
    return false;
  }
  if (op_param.kvHeadNum > 0 && op_param.headNum % op_param.kvHeadNum != 0) {
    MS_LOG(ERROR) << "Ring MLA expects headNum to be divisible by kvHeadNum, "
                  << "But got param.headNum = " << op_param.headNum
                  << ", param.kvHeadNum = " << op_param.kvHeadNum;
    return false;
  }
  if (op_param.headNum < op_param.kvHeadNum) {
    MS_LOG(ERROR) << "Ring MLA expects headNum >= kvHeadNum, "
                  << "But got param.headNum = " << op_param.headNum
                  << ", param.kvHeadNum = " << op_param.kvHeadNum;
    return false;
  }
  if (op_param.maskType != internal::RingMLAParam::MaskType::NO_MASK &&
      op_param.maskType != internal::RingMLAParam::MaskType::MASK_TYPE_TRIU) {
    MS_LOG(ERROR) << "Ring MLA expects maskType as one of NO_MASK, MASK_TYPE_TRIU, "
                  << "But got param.maskType = " << op_param.maskType;
    return false;
  }
  if (op_param.inputLayout != internal::RingMLAParam::InputLayout::TYPE_BSND) {
    MS_LOG(ERROR) << "Ring MLA only supports inputLayout as TYPE_BSND, "
                  << "But got param.inputLayout = " << op_param.inputLayout;
    return false;
  }
  if (op_param.kernelType != internal::RingMLAParam::KernelType::KERNELTYPE_HIGH_PRECISION) {
    MS_LOG(ERROR) << "Ring MLA only supports kernelType as KERNELTYPE_HIGH_PRECISION, "
                  << "But got param.kernelType = " << op_param.kernelType;
    return false;
  }
  return true;
}

// Helper to extract a vector<int32_t> from a KernelTensor, supporting int32 and int64
static void ExtractSeqLenVector(KernelTensor *const seq_len_tensor, std::vector<int32_t> *out_vec) {
  MS_EXCEPTION_IF_NULL(seq_len_tensor);
  out_vec->clear();
  TypeId dtype = seq_len_tensor->dtype_id();
  if (dtype == kNumberTypeInt64) {
    const auto &vec64 = seq_len_tensor->GetValueWithCheck<std::vector<int64_t>>();
    out_vec->assign(vec64.begin(), vec64.end());
  } else if (dtype == kNumberTypeInt32) {
    *out_vec = seq_len_tensor->GetValueWithCheck<std::vector<int32_t>>();
  } else {
    MS_LOG(EXCEPTION) << "actual_seq_lengths data type must be Int32 or Int64, but got "
                      << TypeIdToString(dtype);
  }
}

// Returns true if the new sequence length vector is different from the old one
static bool NeedUpdateSeqLen(const std::vector<int32_t> &old_seq_len, const std::vector<int32_t> &new_seq_len) {
  if (old_seq_len.size() != new_seq_len.size()) {
    return true;
  }
  for (size_t i = 0; i < new_seq_len.size(); ++i) {
    if (old_seq_len[i] != new_seq_len[i]) {
      return true;
    }
  }
  return false;
}

// Updates seq_len from the input tensor if needed, returns true if update is needed
static bool GetSeqLenFromInputAndCheckUpdate(const std::string &kernel_name, const std::string &tensor_name,
                                             KernelTensor *const seq_len_tensor, std::vector<int32_t> *seq_len) {
  MS_EXCEPTION_IF_NULL(seq_len_tensor);

  // If the tensor is not None, extract and compare
  if (seq_len_tensor->type_id() != kMetaTypeNone) {
    std::vector<int32_t> new_seq_len;
    ExtractSeqLenVector(seq_len_tensor, &new_seq_len);

    bool need_update = NeedUpdateSeqLen(*seq_len, new_seq_len);
    if (need_update) {
      *seq_len = std::move(new_seq_len);
    }

    MS_LOG(INFO) << "For op '" << kernel_name << "', set param seq_len with tensor_input '" << tensor_name << "' as "
                 << (*seq_len);
    return need_update;
  }

  // If tensor is None, handle accordingly
  MS_LOG(INFO) << "For op '" << kernel_name << "', param seq_len must be set, but none of '"
               << tensor_name << "' is found in tensor_input";
  if (seq_len->empty()) {
    // No previous value, nothing to update
    return false;
  }
  // Previous value exists, but now input is None: clear and signal update
  seq_len->clear();
  return true;
}

internal::InternalOpPtr CustomRingMLA::CreateKernel(const internal::InputsImmutableInfoList &inputs_ii,
                                                    const internal::OutputsImmutableInfoList &outputs_ii,
                                                    const std::vector<KernelTensor *> &ms_inputs,
                                                    const std::vector<KernelTensor *> &ms_outputs) {
  // Extract and set all required parameters from ms_inputs
  param_.headNum = static_cast<int32_t>(ms_inputs[kHeadNumIdx]->GetValueWithCheck<int64_t>());
  param_.qkScale = ms_inputs[kQkScaleIdx]->GetValueWithCheck<float>();
  param_.kvHeadNum = static_cast<int32_t>(ms_inputs[kKvHeadNumIdx]->GetValueWithCheck<int64_t>());
  param_.maskType = static_cast<internal::RingMLAParam::MaskType>(
      ms_inputs[kMaskTypeIdx]->GetValueWithCheck<int64_t>());
  param_.calcType = static_cast<internal::RingMLAParam::CalcType>(
      ms_inputs[kCalcTypeIdx]->GetValueWithCheck<int64_t>());

  // Update sequence lengths from input tensors
  (void)GetSeqLenFromInputAndCheckUpdate(kernel_name_, "q_seq_lens", ms_inputs[kQSeqLenIdx], &param_.qSeqLen);
  (void)GetSeqLenFromInputAndCheckUpdate(kernel_name_, "batch_valid_length",
                                         ms_inputs[kKVSeqLenIdx], &param_.kvSeqLen);

  MS_CHECK_VALUE(RingMLAParamCheck(param_),
                 CheckAndConvertUtils::FormatCommMsg("For RingMLA The param is invalid, please check the input "
                                                     "parameters, kernel_name: ", kernel_name_));

  created_flag_ = true;
  return internal::CreateRingMLAOp(inputs_ii, outputs_ii, param_, internal::kInternalRingMLAOpName);
}

bool CustomRingMLA::UpdateParam(const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) {
  if (created_flag_) {
    // Sequence lengths already initialized in CreateKernel, skip update
    created_flag_ = false;
    return true;
  }

  // Check if either q_seq_len or kv_seq_len needs update
  bool q_need_update = GetSeqLenFromInputAndCheckUpdate(kernel_name_, "q_seq_lens",
                                                        inputs[kQSeqLenIdx], &param_.qSeqLen);
  bool kv_need_update = GetSeqLenFromInputAndCheckUpdate(kernel_name_, "batch_valid_length",
                                                         inputs[kKVSeqLenIdx], &param_.kvSeqLen);

  if (q_need_update || kv_need_update) {
    auto ret = internal_op_->UpdateParam(&param_);
    if (ret != internal::kInternalOk) {
      MS_LOG(ERROR) << "CustomRingMLA UpdateParam failed, kernel_name: " << kernel_name_;
      return false;
    }
    return true;
  }

  return true;
}

uint64_t CustomRingMLA::GenerateTilingKey(const std::vector<KernelTensor *> &inputs) {
  // User defined CacheKey, the inputs should include all the factors which will affect tiling result.
  return InternalTilingCache::GenerateKey(kernel_name_, inputs, param_.qSeqLen, param_.kvSeqLen);
}

void CustomRingMLA::InitKernelInputsOutputsIndex() {
  kernel_inputs_index_ = {kQueryIdx, kQueryRopeIdx, kKeyIdx, kKeyRopeIdx, kValueIdx, kMaskIdx, kAlibiCoeffIdx,
                          kDeqQKIdx, kOffsetQKIdx, kDeqPVIdx, kOffsetPVIdx, kQuantPIdx, kLogNIdx,
                          kOPrevIdx, kLsePrevIdx};
  kernel_outputs_index_ = {kAttentionOutIdx, kSoftmaxLseOutIdx};
}

}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(ring_mla, ms_custom_ops::CustomRingMLAOpFuncImpl, ms_custom_ops::CustomRingMLA);
