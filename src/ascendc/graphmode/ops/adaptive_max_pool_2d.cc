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

#include "ascendc_kernel_mod.h"
#include <vector>
#include <map>
#include <string>
#include "ms_extension/api.h"

namespace mindspore {
namespace ops {
class OPS_API AdaptiveMaxPool2dCustomOpFuncImpl : public OpFuncImpl {
public:
  ShapeArray InferShape(const PrimitivePtr &primitive,
                        const InferInfoPtrList &input_infos) const override {
    auto out_shape = input_infos[0]->GetShape();
    auto output_size_opt = input_infos[1]->GetArrayValue<int64_t>();
    size_t len = out_shape.size();
    out_shape[len - 2] = output_size_opt.value()[1];
    out_shape[len - 1] = output_size_opt.value()[0];
    return {out_shape, out_shape};
  }
  std::vector<TypeId> InferType(const PrimitivePtr &primitive,
                                const InferInfoPtrList &input_infos) const override {
    return {input_infos[0]->GetType(), kNumberTypeInt64};
  }

  bool GeneralInferRegistered() const override { return true; }
};

AdaptiveMaxPool2dCustomOpFuncImpl gAdaptiveMaxPool2dFuncImpl;

OpDef gAdaptiveMaxPool2dCustom = {
  /*.name_=*/"Custom_adaptive_max_pool2d",
  /*.args_=*/ {

        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"output_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"indices", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
  },
  /*.signatures_ =*/ {
  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"output_size", 1},
  },
  /*.func_impl_=*/gAdaptiveMaxPool2dFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
  /*.is_graph_view_ =*/false,
};
REGISTER_PRIMITIVE_OP_DEF(Custom_adaptive_max_pool2d, &gAdaptiveMaxPool2dCustom);

} // namespace ops
} // namespace mindspore

namespace ms_custom_ops {
class AdaptiveMaxPool2dCustomAscend : public AscendCKernelMod {
public:
  AdaptiveMaxPool2dCustomAscend() : AscendCKernelMod(std::move("aclnnAdaptiveMaxPool2d")) {}
  ~AdaptiveMaxPool2dCustomAscend() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                      const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(stream_ptr);
    RunOp(stream_ptr, workspace, inputs[0], output_size_, outputs[0], outputs[1]);
    return true;
  }

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    output_size_ = inputs[1]->GetValueWithCheck<std::vector<int64_t>>();
    GetWorkspaceForResize(inputs[0], output_size_, outputs[0], outputs[1]);
  }
private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE();
  std::vector<int64_t> output_size_;

};

MS_ASCENDC_KERNEL_FACTORY_REG(Custom_adaptive_max_pool2d, AdaptiveMaxPool2dCustomAscend);
} // namespace ms_custom_ops
