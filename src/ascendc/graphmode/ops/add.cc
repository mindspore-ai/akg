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
class OPS_API AddCustomOpFuncImpl : public OpFuncImpl {
public:
  ShapeArray InferShape(const PrimitivePtr &primitive,
                        const InferInfoPtrList &input_infos) const override {
    auto out_shape = input_infos[0]->GetShape();
    return {out_shape};
  }
  std::vector<TypeId> InferType(const PrimitivePtr &primitive,
                                const InferInfoPtrList &input_infos) const override {
    return {input_infos[0]->GetType()};
  }

  bool GeneralInferRegistered() const override { return true; }
};

AddCustomOpFuncImpl gAddFuncImpl;
OpFuncImpl &gCustom_AddFuncImpl = gAddFuncImpl;
} // namespace ops
} // namespace mindspore

namespace ms_custom_ops {
class AddCustomAscend : public AscendCKernelMod {
public:
  AddCustomAscend() : AscendCKernelMod(std::move("aclnnAddCustom")) {}
  ~AddCustomAscend() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                      const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(stream_ptr);
    RunOp(stream_ptr, workspace, inputs[0], inputs[1], outputs[0]);
    return true;
  }

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    GetWorkspaceForResize(inputs[0], inputs[1], outputs[0]);
  }
private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE();

};

MS_ASCENDC_KERNEL_FACTORY_REG(Custom_Add, AddCustomAscend);
} // namespace ms_custom_ops
