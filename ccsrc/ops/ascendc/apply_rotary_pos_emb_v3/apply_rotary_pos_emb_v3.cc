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
#include "ccsrc/base/ascendc/graphmode/ascendc_kernel_mod.h"
#include "ccsrc/utils/utils.h"
#include "mindspore/ccsrc/include/backend/common/ms_device_shape_transfer.h"
#include "mindspore/core/include/ir/tensor.h"
#include "mindspore/ops/kernel/ascend/acl_ir/acl_convert.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "mindspore/ccsrc/ms_extension/api.h"
#include "mindspore/core/include/ops/base_operator.h"
#include "mindspore/core/include/ops/ops_func_impl/op_func_impl.h"
#include "mindspore/core/include/ops/ops_func_impl/simple_infer.h"
#include "mindspore/core/include/utils/check_convert_utils.h"

namespace ms_custom_ops {
enum class ApplyRotaryPosEmbV3InputIndex : size_t {
  kApplyRotaryPosEmbV3QueryIndex = 0,
  kApplyRotaryPosEmbV3KeyIndex,
  kApplyRotaryPosEmbV3CosIndex,
  kApplyRotaryPosEmbV3SinIndex,
  kApplyRotaryPosEmbV3LayoutIndex,
  kApplyRotaryPosEmbV3RotaryModeIndex,
  kApplyRotaryPosEmbV3InputsNum,
};

static void ApplyRotaryPosEmbV3CheckInputsShape(const std::string &op_name, const std::vector<int64_t> &query_shape,
                                                const std::vector<int64_t> &key_shape,
                                                const std::vector<int64_t> &cos_shape,
                                                const std::vector<int64_t> &sin_shape) {
  if (query_shape.size() != kDim3 || key_shape.size() != kDim3 || cos_shape.size() != kDim2 ||
      sin_shape.size() != kDim2) {
    MS_LOG(EXCEPTION) << op_name << ", the dim of inputs should be query.dim=key.dim=3, "
                      << "cos.dim=sin.dim=2, but got query.dim=" << query_shape.size()
                      << ", key.dim=" << key_shape.size() << ", cos.dim=" << cos_shape.size()
                      << ", sin.dim=" << sin_shape.size();
  }
  MS_CHECK_VALUE(query_shape[kIndex2] == key_shape[kIndex2] && query_shape[kIndex0] == key_shape[kIndex0],
                 CheckAndConvertUtils::FormatCommMsg(
                   op_name, ", query.dim0 should be equal key.dim0, query.dim2 should be equal key.dim2,",
                   " but got query.shape=", query_shape, ", key.shape=", key_shape));
  MS_CHECK_VALUE(
    cos_shape == sin_shape,
    CheckAndConvertUtils::FormatCommMsg(
      op_name, ", cos.shape should be equals sin.shape, but got cos.shape=", cos_shape, ", sin.shape=", sin_shape));
  MS_CHECK_VALUE(
    query_shape[kIndex2] >= 2 * cos_shape[kIndex1],
    CheckAndConvertUtils::FormatCommMsg(
      op_name, ", the head_dim of query and key should be greater than or equal to twice head_dim of cos or sin,",
      " but got query.shape=", query_shape, ", cos.shape=", cos_shape));
  MS_CHECK_VALUE(query_shape[kIndex0] == cos_shape[kIndex0],
                 CheckAndConvertUtils::FormatCommMsg(
                   op_name, ", query/key's dim0 should be equal cos/sin's dim0, but got query's shape is ", query_shape,
                   ", cos's shape is ", cos_shape));
}
static void ApplyRotaryPosEmbV3CheckInputsType(const std::string &op_name, const TypeId &query_dtype,
                                               const TypeId &key_dtype, const TypeId &cos_dtype,
                                               const TypeId &sin_dtype) {
  const std::unordered_set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeFloat32};
  std::unordered_set<TypeId> input_types = {query_dtype, key_dtype, cos_dtype, sin_dtype};
  if (input_types.size() > 1) {
    MS_LOG(EXCEPTION) << op_name << ", the dtype of 'query, key, cos, sin' should be same, but got '"
                      << TypeIdToString(query_dtype) << ", " << TypeIdToString(key_dtype) << ", "
                      << TypeIdToString(cos_dtype) << ", " << TypeIdToString(sin_dtype) << "'";
  }
  if (valid_types.find(query_dtype) == valid_types.end()) {
    MS_LOG(EXCEPTION) << op_name << ", the dtype of 'query, key, cos, sin' should be "
                      << TypeIdToString(kNumberTypeFloat16) << " or " << TypeIdToString(kNumberTypeFloat32)
                      << ", but got '" << TypeIdToString(query_dtype) << ", " << TypeIdToString(key_dtype) << ", "
                      << TypeIdToString(cos_dtype) << ", " << TypeIdToString(sin_dtype) << "'";
  }
}
class OPS_API ApplyRotaryPosEmbV3OpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    if (input_infos[static_cast<size_t>(ApplyRotaryPosEmbV3InputIndex::kApplyRotaryPosEmbV3QueryIndex)]
          ->IsDynamicRank() ||
        input_infos[static_cast<size_t>(ApplyRotaryPosEmbV3InputIndex::kApplyRotaryPosEmbV3KeyIndex)]
          ->IsDynamicRank() ||
        input_infos[static_cast<size_t>(ApplyRotaryPosEmbV3InputIndex::kApplyRotaryPosEmbV3CosIndex)]
          ->IsDynamicRank() ||
        input_infos[static_cast<size_t>(ApplyRotaryPosEmbV3InputIndex::kApplyRotaryPosEmbV3SinIndex)]
          ->IsDynamicRank()) {
      return {
        input_infos[static_cast<size_t>(ApplyRotaryPosEmbV3InputIndex::kApplyRotaryPosEmbV3QueryIndex)]->GetShape(),
        input_infos[static_cast<size_t>(ApplyRotaryPosEmbV3InputIndex::kApplyRotaryPosEmbV3KeyIndex)]->GetShape()};
    }
    auto op_name = primitive->name();
    auto query_shape =
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbV3InputIndex::kApplyRotaryPosEmbV3QueryIndex)]->GetShape();
    auto key_shape =
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbV3InputIndex::kApplyRotaryPosEmbV3KeyIndex)]->GetShape();
    auto cos_shape =
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbV3InputIndex::kApplyRotaryPosEmbV3CosIndex)]->GetShape();
    auto sin_shape =
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbV3InputIndex::kApplyRotaryPosEmbV3SinIndex)]->GetShape();
    auto rotary_mode =
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbV3InputIndex::kApplyRotaryPosEmbV3RotaryModeIndex)]
        ->GetScalarValueWithCheck<string>();
    auto layout = input_infos[static_cast<size_t>(ApplyRotaryPosEmbV3InputIndex::kApplyRotaryPosEmbV3LayoutIndex)]
                    ->GetScalarValueWithCheck<string>();
    MS_CHECK_VALUE(layout == "BSH",
                   CheckAndConvertUtils::FormatCommMsg(op_name, " layout should be 'BSH', but got ", layout));
    MS_CHECK_VALUE(
      rotary_mode == "interleave",
      CheckAndConvertUtils::FormatCommMsg(op_name, " rotary_mode should be 'interleave', but got ", rotary_mode));
    ApplyRotaryPosEmbV3CheckInputsShape(op_name, query_shape, key_shape, cos_shape, sin_shape);
    return {query_shape, key_shape};
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto op_name = primitive->name();
    auto query_dtype =
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbV3InputIndex::kApplyRotaryPosEmbV3QueryIndex)]->GetType();
    auto key_dtype =
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbV3InputIndex::kApplyRotaryPosEmbV3KeyIndex)]->GetType();
    auto cos_dtype =
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbV3InputIndex::kApplyRotaryPosEmbV3CosIndex)]->GetType();
    auto sin_dtype =
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbV3InputIndex::kApplyRotaryPosEmbV3SinIndex)]->GetType();
    ApplyRotaryPosEmbV3CheckInputsType(op_name, query_dtype, key_dtype, cos_dtype, sin_dtype);
    return {query_dtype, key_dtype};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class ApplyRotaryPosEmbV3Ascend : public AclnnCustomKernelMod {
 public:
  ApplyRotaryPosEmbV3Ascend() : AclnnCustomKernelMod(std::move("aclnnApplyRotaryPosEmbV3")) {}
  ~ApplyRotaryPosEmbV3Ascend() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(stream_ptr);
    RunOp(
      stream_ptr, workspace, inputs[static_cast<size_t>(ApplyRotaryPosEmbV3InputIndex::kApplyRotaryPosEmbV3QueryIndex)],
      inputs[static_cast<size_t>(ApplyRotaryPosEmbV3InputIndex::kApplyRotaryPosEmbV3KeyIndex)],
      inputs[static_cast<size_t>(ApplyRotaryPosEmbV3InputIndex::kApplyRotaryPosEmbV3CosIndex)],
      inputs[static_cast<size_t>(ApplyRotaryPosEmbV3InputIndex::kApplyRotaryPosEmbV3SinIndex)], layout_, rotary_mode_);
    return true;
  }
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    layout_ = inputs[static_cast<size_t>(ApplyRotaryPosEmbV3InputIndex::kApplyRotaryPosEmbV3LayoutIndex)]
                ->GetValueWithCheck<std::string>();
    rotary_mode_ = inputs[static_cast<size_t>(ApplyRotaryPosEmbV3InputIndex::kApplyRotaryPosEmbV3RotaryModeIndex)]
                     ->GetValueWithCheck<std::string>();
    GetWorkspaceForResize(inputs[static_cast<size_t>(ApplyRotaryPosEmbV3InputIndex::kApplyRotaryPosEmbV3QueryIndex)],
                          inputs[static_cast<size_t>(ApplyRotaryPosEmbV3InputIndex::kApplyRotaryPosEmbV3KeyIndex)],
                          inputs[static_cast<size_t>(ApplyRotaryPosEmbV3InputIndex::kApplyRotaryPosEmbV3CosIndex)],
                          inputs[static_cast<size_t>(ApplyRotaryPosEmbV3InputIndex::kApplyRotaryPosEmbV3SinIndex)],
                          layout_, rotary_mode_);
    return;
  }

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE();
  std::string layout_ = "BSH";
  std::string rotary_mode_ = "interleave";
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(apply_rotary_pos_emb_v3, ms_custom_ops::ApplyRotaryPosEmbV3OpFuncImpl,
                  ms_custom_ops::ApplyRotaryPosEmbV3Ascend);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

namespace ms_custom_ops {
using namespace mindspore;
using namespace mindspore::device::ascend;
constexpr size_t kApplyRotaryPosEmbV3OutputNum = 2;

std::vector<ms::Tensor> apply_rotary_pos_emb_v3_custom(const ms::Tensor &query, const ms::Tensor &key,
                                                       const ms::Tensor &cos, const ms::Tensor &sin,
                                                       const std::string layout_str, const std::string rotary_mode) {
  std::string op_name = "apply_rotary_pos_emb_v3";
  // 此处op_name是给人看的, 跟算子命名没有直接关联
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>(op_name);
  // 输入shape检查
  ApplyRotaryPosEmbV3CheckInputsShape(op_name, query.shape(), key.shape(), cos.shape(), sin.shape());
  // 输入dtype检查
  ApplyRotaryPosEmbV3CheckInputsType(op_name, query.data_type(), key.data_type(), cos.data_type(), sin.data_type());
  // 此处"aclnnApplyRotaryPosEmbV3", 是算字库函数表中名字前面加上aclnn
  // 可通过 nm -D ./build/xxx/xxx/ms_custom_ops.xxx.so | grep "ApplyRotaryPosEmbV3"来确认
  // 如果是复写算子(inplace), 不必添加输出参数
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnApplyRotaryPosEmbV3, query, key, cos, sin, layout_str, rotary_mode));
  // 如果是复写算子(inplace), 输出参数为空
  runner->Run({query, key, cos, sin}, {});
  // 如果是复写算子(inplace), 将复写的input参数作为output返回
  return {query, key};
}
}  // namespace ms_custom_ops

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("apply_rotary_pos_emb_v3",
        PYBOOST_CALLER(ms_custom_ops::kApplyRotaryPosEmbV3OutputNum, ms_custom_ops::apply_rotary_pos_emb_v3_custom));
}