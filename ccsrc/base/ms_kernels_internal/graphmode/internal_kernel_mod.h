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
#ifndef MS_CUSTOM_OPS_INTERNAL_KERNEL_MOD_H_
#define MS_CUSTOM_OPS_INTERNAL_KERNEL_MOD_H_

#include <memory>
#include <string>
#include <vector>

#include "common/kernel.h"
#include "include/internal.h"
#include "tiling_mem_mgr.h"

#include "debug/profiler/profiling.h"
#include "internal_helper.h"
#include "internal_kernel_in_out_map.h"
#include "internal_spinlock.h"
#include "internal_tiling_cache.h"
#include "module.h"

namespace ms_custom_ops {
class InternalKernelMod : public KernelMod {
public:
  InternalKernelMod() {
    ascend_profiler_ = profiler::Profiler::GetInstance(kAscendDevice);
    MS_EXCEPTION_IF_NULL(ascend_profiler_);
  }

  virtual ~InternalKernelMod() = default;

  bool Init(const std::vector<KernelTensor *> &inputs,
            const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs,
             const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs,
              const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs,
              void *stream_ptr) override;

  std::vector<KernelAttr> GetOpSupport() override {
    MS_LOG(EXCEPTION) << "This interface is not support in internal kernel.";
  }

  void set_fullname(const std::string &fullname) override {
    fullname_ = fullname;
  }

protected:
  virtual bool IsNeedRecreate(const std::vector<KernelTensor *> &inputs,
                              const std::vector<KernelTensor *> &outputs);
  virtual bool UpdateParam(const std::vector<KernelTensor *> &inputs,
                           const std::vector<KernelTensor *> &outputs) {
    return true;
  }
  virtual internal::InternalOpPtr
  CreateKernel(const internal::InputsImmutableInfoList &inputs,
               const internal::OutputsImmutableInfoList &outputs,
               const std::vector<KernelTensor *> &ms_inputs,
               const std::vector<KernelTensor *> &ms_outputs) {
    return nullptr;
  }

  virtual uint64_t GenerateTilingKey(const std::vector<KernelTensor *> &inputs);

  internal::InternalOpPtr internal_op_{nullptr};
  std::vector<size_t> internal_to_ms_input_indices_mapper_;
  std::vector<size_t> internal_to_ms_output_indices_mapper_;
  internal::ShapeInfoList internal_inputs_shape_;
  internal::ShapeInfoList internal_outputs_shape_;
  internal::InputsAddrList internal_inputs_addr_;
  internal::OutputsAddrList internal_outputs_addr_;
  internal::WsAddrList internal_wss_addr_;

private:
  std::shared_ptr<profiler::Profiler> ascend_profiler_{nullptr};
  void GetOrGenerateTiling(const std::vector<KernelTensor *> &inputs,
                           const std::vector<KernelTensor *> &outputs);
  inline void UpdateAddr(const std::vector<KernelTensor *> &inputs,
                         const std::vector<KernelTensor *> &outputs,
                         const std::vector<KernelTensor *> &workspace);
  void GetInternalKernel(const std::vector<KernelTensor *> &inputs,
                         const std::vector<KernelTensor *> &outputs);

  MemoryType host_tiling_mem_type_{kMemoryUndefined};
  MemoryType device_tiling_mem_type_{kMemoryUndefined};
  uint64_t last_key_{0};
  TilingCacheItemPtr last_item_{nullptr};
  TilingCacheItemPtr not_cached_item_{nullptr};
  std::vector<size_t> recreate_cared_indices_;
  std::vector<size_t> nz_output_indices_;
  std::string fullname_;
  static SimpleSpinLock lock_;
};

using InternalKernelModPtr = std::shared_ptr<InternalKernelMod>;
using InternalKernelModPtrList = std::vector<InternalKernelModPtr>;

#define MS_CUSTOM_INTERNAL_KERNEL_NAME_REG(PRIM_NAME_STR, INTERNAL_NAME_VAR)   \
  static const InternalNameRegistrar                                           \
      g_##PRIM_NAME_STR##_ms_to_internal_mapper("Custom_" #PRIM_NAME_STR,      \
                                                INTERNAL_NAME_VAR);

} // namespace ms_custom_ops
#endif // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_MOD_H_
