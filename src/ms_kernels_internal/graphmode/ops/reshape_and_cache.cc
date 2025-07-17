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

#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "utils/check_convert_utils.h"
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"

#include "ops/ops_func_impl/op_func_impl.h"

#include "kernel/ascend/acl_ir/acl_convert.h"
#include "ops/base_operator.h"

#include "ms_extension/api.h"

#include "internal_kernel_mod.h"

namespace mindspore {
namespace ops {
class OPS_API CustomReshapeAndCacheOpFuncImpl : public OpFuncImpl {
public:
  ShapeArray InferShape(const PrimitivePtr &primitive,
                        const InferInfoPtrList &input_infos) const override {
    return {input_infos[0]->GetShape()};
  }
  std::vector<TypeId>
  InferType(const PrimitivePtr &primitive,
            const InferInfoPtrList &input_infos) const override {
    return {input_infos[0]->GetType()};
  }

  bool GeneralInferRegistered() const override { return true; }
};

// TODO
CustomReshapeAndCacheOpFuncImpl gCustomReshapeAndCacheFuncImpl;
OpFuncImpl &gCustom_reshape_and_cacheFuncImpl = gCustomReshapeAndCacheFuncImpl;
} // namespace ops
} // namespace mindspore

namespace ms_custom_ops {
class CustomReshapeAndCache : public InternalKernelMod {
public:
  CustomReshapeAndCache() : InternalKernelMod() {}
  ~CustomReshapeAndCache() = default;

protected:
  internal::InternalOpPtr
  CreateKernel(const internal::InputsImmutableInfoList &inputs,
               const internal::OutputsImmutableInfoList &outputs,
               const std::vector<KernelTensor *> &ms_inputs,
               const std::vector<KernelTensor *> &ms_outputs) override {
    internal::ReshapeAndCacheParam param;
    auto head_num = ms_inputs.at(internal::kIndex5);
    if (head_num->dtype_id() == TypeId::kNumberTypeInt64) {
      param.head_num =
          static_cast<int32_t>(head_num->GetValue<int64_t>().value());
    } else {
      MS_LOG(EXCEPTION)
          << "ReshapeAndCache [head_num]'s dtype wrong, expect int64, but got: "
          << head_num->dtype_id();
    }
    return internal::CreateReshapeAndCacheOp(
        inputs, outputs, param, internal::kInternalReshapeAndCacheOpName);
  }
};

MS_CUSTOM_INTERNAL_KERNEL_FACTORY_REG(Custom_reshape_and_cache,
                                      internal::kInternalReshapeAndCacheOpName,
                                      CustomReshapeAndCache);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(Custom_reshape_and_cache, INPUT_NUM_5,
                                     INDEX_0, INDEX_1, INDEX_2, INDEX_3,
                                     INDEX_4);
} // namespace ms_custom_ops
