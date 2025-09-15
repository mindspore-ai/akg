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
#include <vector>
#include "ascendc_kernel_mod.h"
#include "ccsrc/utils/utils.h"
#include "mindspore/ccsrc/ms_extension/api.h"
#include "ccsrc/base/ascendc/graphmode/ascendc_kernel_mod.h"
#include "mindspore/core/include/utils/core_op_utils.h"
#include "mindspore/core/include/utils/convert_utils_base.h"
#include "mindspore/core/include/utils/check_convert_utils.h"

namespace ms_custom_ops {
enum ApplyRotaryPosEmbExtInputIndex : size_t {
  kApplyRotaryPosEmbExtQueryIndex = 0,
  kApplyRotaryPosEmbExtKeyIndex,
  kApplyRotaryPosEmbExtCosIndex,
  kApplyRotaryPosEmbExtSinIndex,
  kApplyRotaryPosEmbExtLayoutIndex,
  kApplyRotaryPosEmbExtRotaryModeIndex,
  kApplyRotaryPosEmbExtInputsNum,
};

enum ApplyRotaryPosEmbExtEnum : size_t {
  kApplyRotaryPosEmbExtShapeSize = 4,
};

enum ApplyRotaryPosEmbExtLayoutMode : size_t {
  LAYOUT_INVALID = 0,
  LAYOUT_BSND_BSH = 1,
  LAYOUT_BNSD = 2,
  LAYOUT_SBND = 3,
};

static std::set<std::string> apply_rotary_pos_emb_ext_rotary_mode_set = {
  "half",
  "quarter",
  "interleave",
};

static std::set<std::string> apply_rotary_pos_emb_layout_mode_set = {
  "BSND",
  "BSH",
  "BNSD",
  "SBND",
};

static size_t GetRopeLayout(std::string layout_str) {
  if (layout_str == "BSH" || layout_str == "BSND") {
    return static_cast<size_t>(ApplyRotaryPosEmbExtLayoutMode::LAYOUT_BSND_BSH);
  } else if (layout_str == "BNSD") {
    return static_cast<size_t>(ApplyRotaryPosEmbExtLayoutMode::LAYOUT_BNSD);
  } else if (layout_str == "SBND") {
    return static_cast<size_t>(ApplyRotaryPosEmbExtLayoutMode::LAYOUT_SBND);
  }
  return static_cast<size_t>(ApplyRotaryPosEmbExtLayoutMode::LAYOUT_INVALID);
}

ShapeArray ApplyRotaryPosEmbExtMakeShape(const ShapeVector query_shape, const ShapeVector key_shape,
                                         const ShapeVector cos_shape, const ShapeVector sin_shape) {
  MS_CHECK_VALUE(query_shape.size() == kApplyRotaryPosEmbExtShapeSize,
                 "For ApplyRotaryPosEmbExt, Query must be a 4D tensor, but got shape " + ShapeVectorToStr(query_shape));
  MS_CHECK_VALUE(key_shape.size() == kApplyRotaryPosEmbExtShapeSize,
                 "For ApplyRotaryPosEmbExt, key must be a 4D tensor, but got shape " + ShapeVectorToStr(key_shape));
  MS_CHECK_VALUE(cos_shape.size() == kApplyRotaryPosEmbExtShapeSize,
                 "For ApplyRotaryPosEmbExt, cos must be a 4D tensor, but got shape " + ShapeVectorToStr(cos_shape));
  MS_CHECK_VALUE(sin_shape.size() == kApplyRotaryPosEmbExtShapeSize,
                 "For ApplyRotaryPosEmbExt, sin must be a 4D tensor, but got shape " + ShapeVectorToStr(sin_shape));
  return {query_shape, key_shape};
}

class OPS_API ApplyRotaryPosEmbExtCustomOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    if (input_infos[kApplyRotaryPosEmbExtQueryIndex]->IsDynamicRank() ||
        input_infos[kApplyRotaryPosEmbExtKeyIndex]->IsDynamicRank() ||
        input_infos[kApplyRotaryPosEmbExtCosIndex]->IsDynamicRank() ||
        input_infos[kApplyRotaryPosEmbExtSinIndex]->IsDynamicRank()) {
      return {input_infos[kApplyRotaryPosEmbExtQueryIndex]->GetShape(),
              input_infos[kApplyRotaryPosEmbExtKeyIndex]->GetShape()};
    }

    return ApplyRotaryPosEmbExtMakeShape(
      input_infos[kApplyRotaryPosEmbExtQueryIndex]->GetShape(), input_infos[kApplyRotaryPosEmbExtKeyIndex]->GetShape(),
      input_infos[kApplyRotaryPosEmbExtCosIndex]->GetShape(), input_infos[kApplyRotaryPosEmbExtSinIndex]->GetShape());
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    auto query_dtype = input_infos[kApplyRotaryPosEmbExtQueryIndex]->GetType();
    auto key_dtype = input_infos[kApplyRotaryPosEmbExtKeyIndex]->GetType();
    auto cos_dtype = input_infos[kApplyRotaryPosEmbExtCosIndex]->GetType();
    auto sin_dtype = input_infos[kApplyRotaryPosEmbExtSinIndex]->GetType();
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    const auto &soc_version = ms_context->ascend_soc_version();

    if (soc_version == kAscendVersion910_93 || soc_version == kAscendVersion910b) {
      const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeBFloat16, kNumberTypeFloat32};
      CheckAndConvertUtils::CheckTypeIdValid("query", query_dtype, valid_types, op_name);
      CheckAndConvertUtils::CheckTypeIdValid("key", key_dtype, valid_types, op_name);
      CheckAndConvertUtils::CheckTypeIdValid("cos", cos_dtype, valid_types, op_name);
      CheckAndConvertUtils::CheckTypeIdValid("sin", sin_dtype, valid_types, op_name);
    } else if (soc_version == kAscendVersion310p) {
      const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeFloat32};
      CheckAndConvertUtils::CheckTypeIdValid("query", query_dtype, valid_types, op_name);
      CheckAndConvertUtils::CheckTypeIdValid("key", key_dtype, valid_types, op_name);
      CheckAndConvertUtils::CheckTypeIdValid("cos", cos_dtype, valid_types, op_name);
      CheckAndConvertUtils::CheckTypeIdValid("sin", sin_dtype, valid_types, op_name);
    } else {
      MS_LOG(EXCEPTION) << "'ApplyRotaryPosEmbExt' only support [" << kAscendVersion910b << ", " << kAscendVersion910_93
                        << ", " << kAscendVersion310p << "], but got " << soc_version;
    }
    return {query_dtype, key_dtype};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class ApplyRotaryPosEmbExtCustomAscend : public AclnnCustomKernelMod {
 public:
  ApplyRotaryPosEmbExtCustomAscend() : AclnnCustomKernelMod("aclnnApplyRotaryPosEmbV2") {}
  ~ApplyRotaryPosEmbExtCustomAscend() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(stream_ptr);
    RunOp(stream_ptr, workspace, inputs[kApplyRotaryPosEmbExtQueryIndex], inputs[kApplyRotaryPosEmbExtKeyIndex],
          inputs[kApplyRotaryPosEmbExtCosIndex], inputs[kApplyRotaryPosEmbExtSinIndex], layout_, rotary_mode_);
    return true;
  }

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    auto layout_str = inputs[kApplyRotaryPosEmbExtLayoutIndex]->GetValueWithCheck<std::string>();
    layout_ = GetRopeLayout(layout_str);
    rotary_mode_ = inputs[kApplyRotaryPosEmbExtRotaryModeIndex]->GetValueWithCheck<std::string>();
    GetWorkspaceForResize(inputs[kApplyRotaryPosEmbExtQueryIndex], inputs[kApplyRotaryPosEmbExtKeyIndex],
                          inputs[kApplyRotaryPosEmbExtCosIndex], inputs[kApplyRotaryPosEmbExtSinIndex], layout_,
                          rotary_mode_);
    return;
  }

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE();
  size_t layout_ = ApplyRotaryPosEmbExtLayoutMode::LAYOUT_INVALID;
  std::string rotary_mode_ = "half";
  static constexpr int64_t bsnd_layout_ = 1;
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(apply_rotary_pos_emb_ext, ms_custom_ops::ApplyRotaryPosEmbExtCustomOpFuncImpl,
                  ms_custom_ops::ApplyRotaryPosEmbExtCustomAscend);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

namespace ms_custom_ops {
using namespace mindspore;
using namespace mindspore::device::ascend;
constexpr size_t kApplyRotaryPosEmbExtOutputNum = 2;

std::vector<ms::Tensor> apply_rotary_pos_emb_ext_custom(const ms::Tensor &query, const ms::Tensor &key,
                                                        const ms::Tensor &cos, const ms::Tensor &sin,
                                                        const std::string layout_str, const std::string rotary_mode) {
  (void)ApplyRotaryPosEmbExtMakeShape(query.shape(), key.shape(), cos.shape(), sin.shape());
  auto layout_mode = GetRopeLayout(layout_str);
  auto outputs = {ms::Tensor(query.data_type(), query.shape()), ms::Tensor(key.data_type(), key.shape())};
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("aclnnApplyRotaryPosEmbV2");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnApplyRotaryPosEmbV2, query, key, cos, sin, layout_mode, rotary_mode));
  // only set tensor.
  runner->Run({query, key, cos, sin}, outputs);
  return outputs;
}
}  // namespace ms_custom_ops

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("apply_rotary_pos_emb_ext",
        PYBOOST_CALLER(ms_custom_ops::kApplyRotaryPosEmbExtOutputNum, ms_custom_ops::apply_rotary_pos_emb_ext_custom));
}