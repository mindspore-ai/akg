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

#include "ascendc_kernel_mod.h"
#include "ms_extension/api.h"
#include <map>
#include <string>
#include <vector>

namespace mindspore {
namespace ops {
class OPS_API AddRmsNormCustomOpFuncImpl : public OpFuncImpl {
public:
  ShapeArray InferShape(const PrimitivePtr &primitive,
                        const InferInfoPtrList &input_infos) const override {
    auto &x1 = input_infos[kInputIndex0];
    auto &x2 = input_infos[kInputIndex1];
    auto &gamma = input_infos[kInputIndex2];
    const auto &x1_shape = x1->GetShape();
    const auto &x2_shape = x2->GetShape();
    const auto &gamma_shape = gamma->GetShape();
    auto gamma_rank = gamma_shape.size();

    if (x1->IsDynamicRank() && x2->IsDynamicRank() && gamma->IsDynamicRank()) {
      auto out_shape = ShapeVector{abstract::Shape::kShapeRankAny};
      return {out_shape, out_shape, out_shape};
    }

    if (!(x1->IsDynamic() || x2->IsDynamic())) {
      if (x1_shape != x2_shape) {
        MS_EXCEPTION(ValueError) << "For AddRmsNorm, shape of x1: " << x1_shape
                                 << " are not consistent with the shape x2: " << x2_shape << " .";
      }
    }
    auto out_shape = x1_shape;
    auto out_rank = out_shape.size();
    auto rstd_shape = out_shape;
    if (gamma->IsDynamicRank()) {
      if (!IsDynamicRank(out_shape)) {
        rstd_shape = ShapeVector(out_rank, abstract::TensorShape::kShapeDimAny);
      } else {
        rstd_shape = ShapeVector{abstract::TensorShape::kShapeRankAny};
      }
    } else if (!IsDynamicRank(out_shape)) {
      if (gamma_rank > out_rank) {
        MS_LOG(EXCEPTION) << "For AddRmsNorm, The [gamma] rank can not be bigger than the rank of "
                             "other two inputs. but got gamma_rank: "
                          << gamma_rank << ", out_rank: " << out_rank;
      }
      for (auto dim = out_rank - gamma_rank; dim < out_rank; dim++) {
        int64_t x_dim = out_shape[dim];
        int64_t gamma_dim = gamma_shape[dim - out_rank + gamma_rank];
        if (x_dim != gamma_dim && (x_dim != abstract::TensorShape::kShapeDimAny &&
                                   gamma_dim != abstract::TensorShape::kShapeDimAny)) {
          MS_LOG(EXCEPTION) << "For AddRmsNorm, Each dimension of [gamma] must be aligned to the "
                               "corresponding dimension of other two inputs. But got: gamma_dim: "
                            << gamma_dim << ", x_dim: " << x_dim;
        }
        rstd_shape[dim] = 1;
      }
    }
    return {out_shape, rstd_shape, out_shape};
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive,
                                const InferInfoPtrList &input_infos) const override {
    auto x_dtype = input_infos[0]->GetType();
    return {x_dtype, TypeId::kNumberTypeFloat, x_dtype};
  }

  bool GeneralInferRegistered() const override { return true; }
};
} // namespace ops
} // namespace mindspore

namespace ms_custom_ops {
class AddRmsNormCustomAscend : public AscendCKernelMod {
public:
  AddRmsNormCustomAscend() : AscendCKernelMod(std::move("aclnnAddRmsNormCustom")) {}
  ~AddRmsNormCustomAscend() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs,
              const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(stream_ptr);
    epsilon_ = static_cast<double>(inputs[3]->GetValueWithCheck<float>());
    RunOp(stream_ptr, workspace, inputs[0], inputs[1], inputs[2], epsilon_, outputs[0], outputs[1],
          outputs[2]);
    return true;
  }

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    GetWorkspaceForResize(inputs[0], inputs[1], inputs[2], epsilon_, outputs[0], outputs[1],
                          outputs[2]);
  }

private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE();
  double epsilon_{1e-6f}; // Default epsilon value, can be overridden by input tensor
};
} // namespace ms_custom_ops

REG_GRAPH_MODE_OP(add_rms_norm, ms_custom_ops::AddRmsNormCustomOpFuncImpl,
                  ms_custom_ops::AddRmsNormCustomAscend);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

#include "ascendc_pyboost_runner.h"

namespace ms_custom_ops {
using namespace mindspore;
using namespace mindspore::device::ascend;

std::vector<ms::Tensor> custom_add_rms_norm(const ms::Tensor &x1, const ms::Tensor &x2,
                                            const ms::Tensor &gamma, float epsilon) {
  auto x1_shape = x1.shape();
  auto gamma_shape = gamma.shape();
  auto rstd_shape = x1_shape;
  size_t x1_rank = x1_shape.size();
  size_t gamma_rank = gamma_shape.size();
  for (size_t i = x1_rank - gamma_rank; i < x1_rank; ++i) {
    rstd_shape[i] = 1;
  }

  auto out_y = ms::Tensor(x1.data_type(), x1_shape);
  auto out_rstd = ms::Tensor(TypeId::kNumberTypeFloat32, rstd_shape);
  auto out_x = ms::Tensor(x1.data_type(), x1_shape);
  auto runner = std::make_shared<ms::pynative::AscendCOpRunner>("AddRmsNorm");
  runner->SetLaunchFunc(
      LAUNCH_ASCENDC_FUNC(aclnnAddRmsNormCustom, x1, x2, gamma, epsilon, out_y, out_rstd, out_x));
  runner->Run({x1, x2, gamma}, {out_y, out_rstd, out_x});
  return {out_y, out_rstd, out_x};
}

auto pyboost_add_rms_norm(const ms::Tensor &x1, const ms::Tensor &x2, const ms::Tensor &gamma,
                          float epsilon) {
  return ms::pynative::PyboostRunner::Call<3>(custom_add_rms_norm, x1, x2, gamma, epsilon);
}
} // namespace ms_custom_ops

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("add_rms_norm", &ms_custom_ops::pyboost_add_rms_norm, "add_rms_norm", pybind11::arg("x1"),
        pybind11::arg("x2"), pybind11::arg("gamma"), pybind11::arg("epsilon") = 1e-6f);
}
