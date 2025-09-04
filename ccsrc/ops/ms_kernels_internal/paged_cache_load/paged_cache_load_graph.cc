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

#include "ccsrc/base/ms_kernels_internal/graphmode/internal_kernel_mod.h"
#include "ccsrc/ops/ms_kernels_internal/paged_cache_load/paged_cache_load_common.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "mindspore/core/include/ops/ops_func_impl/op_func_impl.h"

namespace ms_custom_ops {
class OPS_API CustomPagedCacheLoadOpFuncImpl : public OpFuncImpl {
public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    return {input_infos[kPCLInputKeyIndex]->GetShape(), input_infos[kPCLInputValueIndex]->GetShape()};
  }
  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    return {{input_infos[kPCLInputKeyIndex]->GetType(), input_infos[kPCLInputValueIndex]->GetType()}};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class CustomPagedCacheLoad : public InternalKernelMod {
public:
  CustomPagedCacheLoad() : InternalKernelMod(), skip_execution_(false) {}
  ~CustomPagedCacheLoad() = default;

  void InitKernelInputsOutputsIndex() override {
    kernel_inputs_index_ = {kPCLInputKeyCacheIndex, kPCLInputValueCacheIndex, kPCLInputBlockTableIndex, 
                            kPCLInputSeqLensIndex, kPCLInputKeyIndex, kPCLInputValueIndex, kPCLInputSeqStartsIndex};
    kernel_outputs_index_ = {kPCLOutputKeyOutIndex, kPCLOutputValueOutIndex};
  }

  int Resize(const std::vector<KernelTensor *> &inputs,
             const std::vector<KernelTensor *> &outputs) override {
    // Check if any input has shape containing 0
    for (const auto &input : inputs) {
      if (input == nullptr)
        continue;
      auto shape = input->GetShapeVector();
      for (const auto &dim : shape) {
        if (dim == 0) {
          MS_LOG(INFO) << "paged_cache_load: Skipping execution due to zero "
                          "dimension in input shape: "
                       << shape;
          skip_execution_ = true;
          return KernelMod::Resize(inputs, outputs); // Skip execution
        }
      }
    }

    skip_execution_ = false;
    // Call base class implementation
    return InternalKernelMod::Resize(inputs, outputs);
  }

  bool Launch(const std::vector<KernelTensor *> &inputs,
              const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs,
              void *stream_ptr) override {
    // Skip execution if flag is set
    if (skip_execution_) {
      return true; // Skip execution, return success
    }

    // Call base class implementation
    return InternalKernelMod::Launch(inputs, workspace, outputs, stream_ptr);
  }

protected:
  internal::InternalOpPtr
  CreateKernel(const internal::InputsImmutableInfoList &inputs,
               const internal::OutputsImmutableInfoList &outputs,
               const std::vector<KernelTensor *> &ms_inputs,
               const std::vector<KernelTensor *> &ms_outputs) override {
    internal::PagedCacheLoadParam param;
    auto kv_cache_cfg_type = ms_inputs.at(kPCLInputParamKvCacheCfgIndex);
    auto is_seq_lens_cumsum_type = ms_inputs.at(kPCLInputParamIsSeqLensCumsumTypeIndex);
    auto has_seq_starts = ms_inputs.at(kPCLInputParamHasSeqStartsIndex);
    param.kv_cache_cfg_type = kv_cache_cfg_type->GetValue<int64_t>().value();
    param.is_seq_lens_cumsum_type = is_seq_lens_cumsum_type->GetValue<bool>().value();
    param.has_seq_starts = has_seq_starts->GetValue<bool>().value();
    return CreatePagedCacheLoadOpWithFormat(inputs, outputs, param);
  }

private:
  bool skip_execution_; // Flag to skip execution when shape contains 0
};
} // namespace ms_custom_ops
REG_GRAPH_MODE_OP(paged_cache_load, ms_custom_ops::CustomPagedCacheLoadOpFuncImpl,
                  ms_custom_ops::CustomPagedCacheLoad);
