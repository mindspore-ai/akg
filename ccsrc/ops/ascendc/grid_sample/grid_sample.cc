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
enum class GridSampleInputIndex : size_t {
  kGridSampleInputIndex = 0,
  kGridSampleGridIndex,
  kGridSampleModeIndex,
  kGridSamplePaddingModeIndex,
  kGridSampleAlignCornersIndex,
  kGridSampleInputsNum,
};

static void GridSampleCheckInputsShape(const std::string &op_name, const std::vector<int64_t> &input_shape,
                                                const std::vector<int64_t> &grid_shape) {
  if (input_shape.size() != kDim4 || grid_shape.size() != kDim4) {
    MS_LOG(EXCEPTION) << op_name << ", the dim of inputs should be input.dim=grid.dim=4, "
                      << "but got input.dim=" << input_shape.size()
                      << ", grid.dim=" << grid_shape.size();
  }
  MS_CHECK_VALUE(input_shape[kIndex0] == grid_shape[kIndex0],
                 CheckAndConvertUtils::FormatCommMsg(
                   op_name, ", input.dim0 should be equal grid.dim0,",
                   " but got input.shape=", input_shape, ", grid.shape=", grid_shape));
  MS_CHECK_VALUE(
    grid_shape[kIndex3] == 2,
    CheckAndConvertUtils::FormatCommMsg(
      op_name, ", grid.shape should be equals (N, H_OUT, W_OUT, 2), but got grid.shape=", grid_shape));
}

static void GridSampleCheckInputsType(const std::string &op_name, const TypeId &input_dtype,
                                               const TypeId &grid_dtype) {
  if (input_dtype != kNumberTypeFloat32) {
    MS_LOG(EXCEPTION) << op_name << ", the dtype of 'input' should be " << TypeIdToString(kNumberTypeFloat32)
                      << ", but got input.dtype=" << TypeIdToString(input_dtype);
  }
  if (grid_dtype != kNumberTypeFloat32) {
    MS_LOG(EXCEPTION) << op_name << ", the dtype of 'grid' should be " << TypeIdToString(kNumberTypeFloat32)
                      << ", but got grid.dtype=" << TypeIdToString(grid_dtype);
  }
}

class OPS_API GridSampleOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    if (input_infos[static_cast<size_t>(GridSampleInputIndex::kGridSampleInputIndex)]
          ->IsDynamicRank() ||
        input_infos[static_cast<size_t>(GridSampleInputIndex::kGridSampleGridIndex)]
          ->IsDynamicRank()) {
      return {input_infos[static_cast<size_t>(GridSampleInputIndex::kGridSampleInputIndex)]->GetShape()};
      // DON'T SUPPORT DYNAMIC RANK INPUT OR GRID
      // MS_LOG(EXCEPTION) << "GridSample don't support dynamic rank input or grid";
      // return {};
    }
    auto op_name = primitive->name();
    auto input_shape =
      input_infos[static_cast<size_t>(GridSampleInputIndex::kGridSampleInputIndex)]->GetShape();
    auto grid_shape =
      input_infos[static_cast<size_t>(GridSampleInputIndex::kGridSampleGridIndex)]->GetShape();
    auto mode =
      input_infos[static_cast<size_t>(GridSampleInputIndex::kGridSampleModeIndex)]
        ->GetScalarValueWithCheck<string>();
    auto padding_mode = input_infos[static_cast<size_t>(GridSampleInputIndex::kGridSamplePaddingModeIndex)]
                          ->GetScalarValueWithCheck<string>();
    auto align_corners = input_infos[static_cast<size_t>(GridSampleInputIndex::kGridSampleAlignCornersIndex)]
      ->GetScalarValueWithCheck<bool>();
    MS_CHECK_VALUE(mode == "bilinear",
                   CheckAndConvertUtils::FormatCommMsg(op_name, " mode only support 'bilinear', but got ", mode));
    MS_CHECK_VALUE(
      padding_mode == "border",
      CheckAndConvertUtils::FormatCommMsg(op_name, " padding_mode only support 'border', but got ", padding_mode));
    MS_CHECK_VALUE(align_corners == false,
                   CheckAndConvertUtils::FormatCommMsg(op_name, " align_corners only support false, but got ", align_corners));
    GridSampleCheckInputsShape(op_name, input_shape, grid_shape);
    auto output_shape = grid_shape;
    output_shape[kIndex3] = input_shape[kIndex3];
    return {output_shape};
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto op_name = primitive->name();
    auto input_dtype =
      input_infos[static_cast<size_t>(GridSampleInputIndex::kGridSampleInputIndex)]->GetType();
    auto grid_dtype =
      input_infos[static_cast<size_t>(GridSampleInputIndex::kGridSampleGridIndex)]->GetType();
    GridSampleCheckInputsType(op_name, input_dtype, grid_dtype);
    return {input_dtype};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class GridSample : public AclnnCustomKernelMod {
 public:
  GridSample() : AclnnCustomKernelMod(std::move("aclnnGridSample")) {}
  ~GridSample() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(stream_ptr);
    RunOp(
      stream_ptr, workspace, inputs[static_cast<size_t>(GridSampleInputIndex::kGridSampleInputIndex)],
      inputs[static_cast<size_t>(GridSampleInputIndex::kGridSampleGridIndex)],
      outputs[0]);
    return true;
  }
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    GetWorkspaceForResize(inputs[static_cast<size_t>(GridSampleInputIndex::kGridSampleInputIndex)],
                          inputs[static_cast<size_t>(GridSampleInputIndex::kGridSampleGridIndex)],
                          outputs[0]);
    return;
  }

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE();
  std::string mode_ = "bilinear";
  std::string padding_mode_ = "border";
  bool align_corners_ = false;
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(grid_sample, ms_custom_ops::GridSampleOpFuncImpl,
                  ms_custom_ops::GridSample);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

namespace ms_custom_ops {
using namespace mindspore;
using namespace mindspore::device::ascend;
constexpr size_t kGridSampleOutputNum = 1;

std::vector<ms::Tensor> grid_sample_custom(const ms::Tensor &input, const ms::Tensor &grid,
                                           const std::string mode, const std::string padding_mode, const bool align_corners) {
  std::string op_name = "grid_sample";
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>(op_name);
  // check if support mode, padding_mode, align_corners
  if (mode != "bilinear" || padding_mode != "border" || align_corners != false) {
    MS_LOG(EXCEPTION) << op_name << ", the mode should be 'bilinear', the padding_mode should be 'border'," 
      << "the align_corners should be false, but got mode=" << mode << ", padding_mode=" << padding_mode
      << ", align_corners=" << align_corners;
  }
  auto input_shape = input.shape();
  auto grid_shape = grid.shape();
  auto output_shape = grid_shape;
  output_shape[kIndex3] = input_shape[kIndex3];
  GridSampleCheckInputsShape(op_name, input.shape(), grid.shape());
  GridSampleCheckInputsType(op_name, input.data_type(), grid.data_type());
  auto out = ms::Tensor(input.data_type(), output_shape);
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGridSample, input, grid, out));
  runner->Run({input, grid}, {out});
  return {out};
}
}  // namespace ms_custom_ops

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("grid_sample",
        PYBOOST_CALLER(ms_custom_ops::kGridSampleOutputNum, ms_custom_ops::grid_sample_custom));
}