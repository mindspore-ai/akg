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
#include "mindspore/ccsrc/ms_extension/api.h"
#include "mindspore/ccsrc/include/backend/common/ms_device_shape_transfer.h"

namespace ms_custom_ops {
constexpr size_t kQbmmMatSize = 2;
constexpr size_t kQbmmInputX1 = 0;
constexpr size_t kQbmmInputX2 = 1;
constexpr size_t kQbmmInputScale = 2;
constexpr size_t kQbmmInputOffset = 3;
constexpr size_t kQbmmInputBias = 4;
constexpr size_t kQbmmInputPertokenScaleOptional = 5;
constexpr size_t kQbmmInputTransposeX1 = 6;
constexpr size_t kQbmmInputTransposeX2 = 7;
constexpr size_t kQbmmInputX2Format = 8;
constexpr size_t kQbmmInputDtype = 9;
constexpr size_t kQbmmOutputY = 0;

ShapeVector BatchMatMulMakeShape(const ShapeVector x1_shape, const ShapeVector x2_shape, bool transpose_x1,
                                 bool transpose_x2, size_t offset) {
  if (x1_shape.size() < kQbmmMatSize || x2_shape.size() < kQbmmMatSize) {
    MS_LOG(EXCEPTION) << "For 'QuantBatchMatmul', the dimension of 'x1' and 'x2' should be at least 2, but got "
                      << x1_shape.size() << " and " << x2_shape.size();
  }
  ShapeVector out_shape;
  ShapeVector long_shape = x1_shape.size() > x2_shape.size() ? x1_shape : x2_shape;
  ShapeVector short_shape = x1_shape.size() > x2_shape.size() ? x2_shape : x1_shape;
  size_t size_diff = long_shape.size() - short_shape.size();
  for (size_t i = 0; i < long_shape.size() - offset; i++) {
    if (long_shape[i] < 0) {
      out_shape.push_back(abstract::Shape::kShapeDimAny);
    } else if (i >= size_diff) {
      out_shape.push_back(long_shape[i] > short_shape[i - size_diff] ? long_shape[i] : short_shape[i - size_diff]);
    } else {
      out_shape.push_back(long_shape[i]);
    }
  }
  size_t x1_offset = x1_shape.size() - offset;
  size_t x2_offset = x2_shape.size() - offset;
  out_shape.push_back(x1_shape[x1_offset + (transpose_x1 ? 1 : 0)]);
  out_shape.push_back(x2_shape[x2_offset + (transpose_x2 ? 0 : 1)]);
  return out_shape;
}

class OPS_API QuantBatchMatmulCustomOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto x1_shape = input_infos[kQbmmInputX1]->GetShape();
    auto x2_shape = input_infos[kQbmmInputX2]->GetShape();
    if (IsDynamicRank(x1_shape) || IsDynamicRank(x2_shape)) {
      return {ShapeVector({abstract::Shape::kShapeRankAny})};
    }
    bool transpose_x1 = input_infos[kQbmmInputTransposeX1]->GetScalarValueWithCheck<bool>();
    bool transpose_x2 = input_infos[kQbmmInputTransposeX2]->GetScalarValueWithCheck<bool>();
    ShapeVector out_shape = BatchMatMulMakeShape(x1_shape, x2_shape, transpose_x1, transpose_x2, kQbmmMatSize);
    return {out_shape};
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    TypeId output_type = TypeId::kNumberTypeFloat16;
    if (!input_infos[kQbmmInputDtype]->IsNone()) {
      auto dtype_ptr = input_infos[kQbmmInputDtype]->GetScalarValueWithCheck<int64_t>();
      output_type = static_cast<TypeId>(dtype_ptr);
    }
    return {output_type};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class QuantBatchMatmulCustomAscend : public AclnnCustomKernelMod {
 public:
  QuantBatchMatmulCustomAscend() : AclnnCustomKernelMod("aclnnQuantMatmulV4") {}
  ~QuantBatchMatmulCustomAscend() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(stream_ptr);
    x2_tensor_->set_device_ptr(inputs[kQbmmInputX2]->device_ptr());
    RunOp(stream_ptr, workspace, inputs[kQbmmInputX1], x2_tensor_.get(), inputs[kQbmmInputScale],
          inputs[kQbmmInputOffset], inputs[kQbmmInputPertokenScaleOptional], inputs[kQbmmInputBias], transpose_x1_,
          transpose_x2_, outputs[kQbmmOutputY]);
    return true;
  }

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    transpose_x1_ = inputs[kQbmmInputTransposeX1]->GetValueWithCheck<bool>();
    transpose_x2_ = inputs[kQbmmInputTransposeX2]->GetValueWithCheck<bool>();
    x2_tensor_ = inputs[kQbmmInputX2]->CloneKernelTensor();
    std::string x2_format = inputs[kQbmmInputX2Format]->GetValueWithCheck<std::string>();
    if (x2_format != "ND" && x2_format != "FRACTAL_NZ") {
      MS_LOG(EXCEPTION) << "For quant_batch_matmul, the 'x2_format' is only support ['ND', 'FRACTAL_NZ'], but got "
                        << x2_format;
    }
    if (x2_format == "FRACTAL_NZ") {
      x2_tensor_->set_format(mindspore::Format::FRACTAL_NZ);
      if (x2_tensor_->tensor_storage_info() != nullptr) {
        MS_LOG(EXCEPTION) << "For quant_batch_matmul, FRACTAL_NZ is not support when storage_info is not nullptr";
      }

      auto nd_shape = x2_tensor_->GetShapeVector();
      auto nz_shape =
        mindspore::trans::DeviceShapeTransfer().GetDeviceShapeByFormat(nd_shape, x2_format, x2_tensor_->dtype_id());

      constexpr int64_t kStrideBase = 1;
      constexpr int kStrideOffset = 2;
      auto strides = nd_shape;
      if (!strides.empty()) {
        strides.erase(strides.begin());
      }
      strides.push_back(kStrideBase);
      for (int i = static_cast<int>(strides.size()) - kStrideOffset; i >= 0; i--) {
        strides[i] = strides[i] * strides[i + 1];
      }
      auto storage_info = std::make_shared<TensorStorageInfo>(nd_shape, strides, nz_shape, strides, true);
      x2_tensor_->set_tensor_storage_info(storage_info);
    }
    GetWorkspaceForResize(inputs[kQbmmInputX1], x2_tensor_.get(), inputs[kQbmmInputScale], inputs[kQbmmInputOffset],
                          inputs[kQbmmInputPertokenScaleOptional], inputs[kQbmmInputBias], transpose_x1_, transpose_x2_,
                          outputs[kQbmmOutputY]);
  }

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE();
  bool transpose_x1_{false};
  bool transpose_x2_{false};
  KernelTensorPtr x2_tensor_;
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(quant_batch_matmul, ms_custom_ops::QuantBatchMatmulCustomOpFuncImpl,
                  ms_custom_ops::QuantBatchMatmulCustomAscend);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

namespace ms_custom_ops {
using namespace mindspore;
using namespace mindspore::device::ascend;
constexpr size_t kQuantBatchMatmulOutputNum = 1;

ms::Tensor quant_batch_matmul_custom(const ms::Tensor &x1, const ms::Tensor &x2, const ms::Tensor &scale,
                                     const std::optional<ms::Tensor> &offset, const std::optional<ms::Tensor> &bias,
                                     const std::optional<ms::Tensor> &pertoken_scale, bool transpose_x1,
                                     bool transpose_x2, const std::string x2_format, const int64_t output_dtype) {
  auto x1_shape = x1.shape();
  auto x2_shape = x2.shape();
  auto output_shape = BatchMatMulMakeShape(x1.shape(), x2.shape(), transpose_x1, transpose_x2, kQbmmMatSize);
  if (x2_format != "ND") {
    MS_LOG(EXCEPTION) << "For 'quant_batch_matmul', x2 is only support 'ND' format in pynative mode, but got "
                      << x2_format;
  }
  TypeId out_dtype = static_cast<TypeId>(output_dtype);
  auto out = ms::Tensor(out_dtype, output_shape);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("QuantMatmulV4");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnQuantMatmulV4, x1, x2, scale, offset, pertoken_scale, bias, transpose_x1,
                                          transpose_x2, out));
  runner->Run({x1, x2, scale, GetTensorOrEmpty(offset), GetTensorOrEmpty(pertoken_scale), GetTensorOrEmpty(bias)},
              {out});
  return out;
}
}  // namespace ms_custom_ops

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("quant_batch_matmul",
        PYBOOST_CALLER(ms_custom_ops::kQuantBatchMatmulOutputNum, ms_custom_ops::quant_batch_matmul_custom));
}
