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
#include <utility>
#include <vector>
#include "internal_kernel_mod.h"
#include "mindspore/core/include/mindapi/ir/tensor.h"
#include "mindspore/ops/kernel/ascend/acl_ir/acl_convert.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "mindspore/ccsrc/ms_extension/api.h"
#include "mindspore/core/include/ops/base_operator.h"
#include "mindspore/core/include/ops/ops_func_impl/op_func_impl.h"

namespace ms_custom_ops {
constexpr size_t kTypeIndex = 1;

bool CheckTypeValid(TypeId input_type, TypeId output_type) {
  static const std::set<TypeId> valid_type = {kNumberTypeInt8, kNumberTypeInt4};
  if (!valid_type.count(input_type) || !valid_type.count(output_type)) {
    MS_LOG(EXCEPTION) << "For 'type_cast'"
                      << ", the input and output dtype must be [int8, int4], but got input: "
                      << TypeIdToString(input_type)
                      << ", output type: " << TypeIdToString(output_type);
  }
}

class OPS_API CustomTypeCastOpFuncImpl : public OpFuncImpl {
public:
  ShapeArray InferShape(const PrimitivePtr &primitive,
                        const InferInfoPtrList &input_infos) const override {
    return {input_infos[0]->GetShape()};
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive,
                                const InferInfoPtrList &input_infos) const override {
    auto input_type = input_infos[0]->GetType();
    auto type_ptr = input_infos[kTypeIndex]->GetScalarValueWithCheck<int64_t>();
    auto output_type = static_cast<TypeId>(type_ptr);
    CheckTypeValid(input_type, output_type);
    return {output_type};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class CustomTypeCast : public InternalKernelMod {
public:
  CustomTypeCast() : InternalKernelMod() {}
  ~CustomTypeCast() = default;

  void InitKernelInputsOutputsIndex() override {
    kernel_inputs_index_ = {0};
    kernel_outputs_index_ = {0};
  }

  int Resize(const std::vector<KernelTensor *> &inputs,
             const std::vector<KernelTensor *> &outputs) override {
    auto ret = KernelMod::Resize(inputs, outputs);
    if (ret != KRET_OK) {
      MS_LOG(ERROR) << "Kernel " << kernel_name_ << " Resize failed";
      return ret;
    }
    return KRET_OK;
  }

  bool Launch(const std::vector<KernelTensor *> &inputs,
              const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    size_t copy_size = inputs[0]->size();
    auto ret = CALL_ASCEND_API(aclrtMemcpyAsync, outputs[0]->device_ptr(), copy_size,
                               inputs[0]->device_ptr(), copy_size, ACL_MEMCPY_DEVICE_TO_DEVICE,
                               stream_ptr);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "For 'TypeCast', Memcpy failed, ret=" << ret;
    }
    return true;
  }
};
} // namespace ms_custom_ops

REG_GRAPH_MODE_OP(type_cast, ms_custom_ops::CustomTypeCastOpFuncImpl,
                  ms_custom_ops::CustomTypeCast);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

#include "internal_pyboost_runner.h"

using namespace ms_custom_ops;
namespace ms::pynative {
using namespace mindspore;

class TypeCastRunner : public InternalPyboostRunner {
public:
  using InternalPyboostRunner::InternalPyboostRunner;

  void LaunchKernel() override {
    auto op_name = this->op_name();
    auto inputs = this->inputs();
    auto outputs = this->outputs();
    size_t copy_size = inputs[0].numel();
    auto ret = CALL_ASCEND_API(aclrtMemcpyAsync, outputs[0].GetDataPtr(), copy_size,
                               inputs[0].GetDataPtr(), copy_size, ACL_MEMCPY_DEVICE_TO_DEVICE,
                               this->stream());
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "For 'TypeCast', Memcpy failed, ret=" << ret;
    }
    MS_LOG(DEBUG) << "Launch InternalKernel " << op_name << " end";
  }

protected:
  size_t CalcWorkspace() override { return 0; }
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs) override {
    return nullptr;
  }

  void _PrepareDeviceAddress() override {
    PyboostRunner::_PrepareDeviceAddress();
    auto output_device_address =
        std::dynamic_pointer_cast<device::DeviceAddress>(_outputs_[0].tensor()->device_address());
    output_device_address->set_format(_inputs_[0].format());
  }
};
} // namespace ms::pynative

namespace ms_custom_ops {
constexpr size_t kTypeCastOutputNum = 1;

ms::Tensor npu_type_cast(const ms::Tensor &x, int64_t output_dtype) {
  auto op_name = "TypeCast";
  auto runner = std::make_shared<ms::pynative::TypeCastRunner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);
  auto type = static_cast<TypeId>(output_dtype);
  auto output = ms::Tensor(type, x.shape());
  CheckTypeValid(x.data_type(), output.data_type());
  runner->Run({x}, {output});
  return output;
}
} // namespace ms_custom_ops

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("type_cast",
        PYBOOST_CALLER(ms_custom_ops::kTypeCastOutputNum, ms_custom_ops::npu_type_cast));
}
