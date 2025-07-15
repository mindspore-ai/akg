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

#ifndef MS_CUSTOM_OPS_INTERNAL_OP_PYBOOST_RUNNER_H_
#define MS_CUSTOM_OPS_INTERNAL_OP_PYBOOST_RUNNER_H_

#include "ms_extension/all.h"
#include <functional>
#include <optional>
#include <set>
#include <unordered_map>

#include "include/internal.h"
#include "internal_helper.h"
#include "internal_pyboost_utils.h"
#include "internal_spinlock.h"
#include "internal_tiling_cache.h"

namespace ms::pynative {
using namespace mindspore;
using namespace ms_custom_ops;

class InternalPyboostRunner : public PyboostRunner {
public:
  using PyboostRunner::PyboostRunner;

  void SetOpKey(const uint64_t &op_key);

  void SetTilingKey(const uint64_t &tiling_key);

  // Generic setup method for configuring the runner with parameters and
  // calculating hash keys
  template <typename... Args>
  void Setup(const std::string &op_name, const Args &...args) {
    // Calculate hash keys
    auto op_key = CalcInternalOpApiHash(op_name, args...);
    auto tiling_key = CalcInternalOpTilingHash(op_name, args...);

    // Set the calculated keys
    SetOpKey(op_key);
    SetTilingKey(tiling_key);
  }

  void GetOrCreateKernel(const device::DeviceContext *device_context,
                         const tensor::TensorPtrList &inputs,
                         const tensor::TensorPtrList &outputs);

  static void UpdateAddr(std::vector<internal::RawDeviceAddr> *addrlist,
                         const tensor::TensorPtrList &tensorlist) {
    addrlist->resize(tensorlist.size());
    for (size_t i = 0; i < tensorlist.size(); i++) {
      if (tensorlist[i] == nullptr) {
        addrlist->at(i) = nullptr;
      } else {
        addrlist->at(i) = tensorlist[i]->device_address()->GetMutablePtr();
      }
    }
  }

  static void MallocWorkspace(const device::DeviceContext *device_context,
                              size_t stream_id,
                              const internal::InternalOpPtr &internal_op,
                              internal::WsAddrList *internal_wss_addr) {
    auto workspace_size_list = internal_op->GetWorkspaceSize();
    internal_wss_addr->resize(workspace_size_list.size());
    for (size_t i = 0; i < workspace_size_list.size(); i++) {
      auto work_ptr = std::make_shared<kernel::pyboost::MemBlock>(
          device_context, workspace_size_list[i], stream_id);
      internal_wss_addr->at(i) = work_ptr->ptr_;
    }
  }

  static void FreeWorkspace(const device::DeviceContext *device_context,
                            internal::WsAddrList *internal_wss_addr) {
    for (size_t i = 0; i < internal_wss_addr->size(); i++) {
      device_context->device_res_manager_->FreeMemory(internal_wss_addr->at(i));
      internal_wss_addr->at(i) = nullptr;
    }
  }

protected:
  void _DispatchLaunchTask() override { LaunchKernel(); }

protected:
  bool IsInternalDtypeSupport(const tensor::TensorPtrList *ms_inputs,
                              const tensor::TensorPtrList *ms_outputs);
  virtual uint64_t GetOrGenerateOpKey(const uint64_t &op_key) const {
    return op_key;
  }
  virtual uint64_t GetOrGenerateOpTilingKey(const uint64_t &tiling_key) const {
    return tiling_key;
  }
  virtual bool UpdateParam() { return true; }
  TilingCacheItemPtr
  GetOrGenerateTiling(const device::DeviceContext *device_context,
                      const uint64_t &tiling_key);
  virtual internal::InternalOpPtr
  CreateKernel(const internal::InputsImmutableInfoList &inputs,
               const internal::OutputsImmutableInfoList &outputs) = 0;
  void TransInternalShapes(internal::ShapeInfoList *shapelist,
                           const tensor::TensorPtrList &tensorlist,
                           bool is_input = false);
  void TransInternalShapes(const tensor::TensorPtrList &inputs,
                           const tensor::TensorPtrList &outputs);

  uint64_t op_key_{0};
  uint64_t tiling_key_{0};
  internal::InternalOpPtr internal_op_{nullptr};
  inline static std::unordered_map<uint64_t, internal::InternalOpPtr> hash_map_;
  internal::DtypeInfoList internal_inputs_dtype_;
  internal::DtypeInfoList internal_outputs_dtype_;
  internal::ShapeInfoList internal_inputs_shape_;
  internal::ShapeInfoList internal_outputs_shape_;
  internal::InputsImmutableInfoList inputs_ii_;
  internal::OutputsImmutableInfoList outputs_ii_;
  TilingCacheItemPtr tiling_info_{nullptr};

private:
  void UpdateArgImmutableInfo(internal::ArgImmutableInfo *arginfo,
                              const tensor::TensorPtr &tensor,
                              internal::DataType dtype);
  void UpdateArgImmutableInfo(std::vector<internal::ArgImmutableInfo> *arginfos,
                              const tensor::TensorPtrList &tensorlist,
                              bool is_input = false);
  SimpleSpinLock lock_;
};

#define MS_KERNELS_INTERNAL_FACTORY_REG(PRIM_NAME_STR, INTERNAL_NAME_VAR)      \
  static const InternalNameRegistrar                                           \
      g_##PRIM_NAME_STR##_ms_to_internal_mapper(#PRIM_NAME_STR,                \
                                                INTERNAL_NAME_VAR);

#define LAUNCH_INTERNAL_KERNEL(device_ctx, stream_id, internal_op, tiling_ptr, \
                               inputs_addr, outputs_addr, internal_wss_addr,   \
                               kernel_name)                                    \
  do {                                                                         \
    runtime::OpExecutor::DispatchLaunchTask([device_ctx, stream_id,            \
                                             internal_op, tiling_ptr,          \
                                             inputs_addr, outputs_addr,        \
                                             internal_wss_addr,                \
                                             kernel_name]() {                  \
      MS_LOG(DEBUG) << "Launch InternalKernel " << kernel_name << " start";    \
      device_ctx->device_res_manager_->BindDeviceToCurrentThread(false);       \
      internal_op->SetTilingInfo(tiling_ptr->tiling_info_);                    \
      auto stream_ptr = device_ctx->device_res_manager_->GetStream(stream_id); \
      auto &internal_wss_addr_ =                                               \
          const_cast<internal::WsAddrList &>(internal_wss_addr);               \
      internal::InternalStatus status =                                        \
          internal_op->Launch(inputs_addr, outputs_addr, internal_wss_addr_,   \
                              stream_ptr, kernel_name);                        \
      InternalTilingCache::GetInstance().Unbind(tiling_ptr);                   \
      if (status != internal::InternalStatus::kInternalOk) {                   \
        MS_LOG(EXCEPTION) << "Launch InternalKernel failed, kernel_name: "     \
                          << kernel_name;                                      \
      }                                                                        \
      MS_LOG(DEBUG) << "Launch InternalKernel " << kernel_name << " end";      \
    });                                                                        \
  } while (false)

#define LAUNCH_INTERNAL(kernel_name_, device_context_, stream_id_, inputs,     \
                        outputs)                                               \
  do {                                                                         \
    const std::string kernel_name = kernel_name_;                              \
    auto device_ctx = device_context_;                                         \
    auto stream_id = stream_id_;                                               \
    GetOrCreateKernel(device_ctx, inputs, outputs);                            \
    internal::InternalOpPtr internal_op = internal_op_;                        \
    TilingCacheItemPtr tiling_ptr = tiling_info_;                              \
    pyboost::PyBoostUtils::MallocInternalOpInputs(device_ctx, inputs);         \
    pyboost::PyBoostUtils::MallocOpOutputs(device_ctx, outputs);               \
    internal::InputsAddrList inputs_addr;                                      \
    internal::OutputsAddrList outputs_addr;                                    \
    InternalPyboostRunner::UpdateAddr(&inputs_addr, inputs);                   \
    InternalPyboostRunner::UpdateAddr(&outputs_addr, outputs);                 \
    internal::WsAddrList internal_wss_addr;                                    \
    InternalPyboostRunner::MallocWorkspace(device_ctx, stream_id, internal_op, \
                                           &internal_wss_addr);                \
    LAUNCH_INTERNAL_KERNEL(device_ctx, stream_id, internal_op, tiling_ptr,     \
                           inputs_addr, outputs_addr, internal_wss_addr,       \
                           kernel_name);                                       \
  } while (false)

} // namespace ms::pynative
#endif // MS_CUSTOM_OPS_INTERNAL_OP_PYBOOST_RUNNER_H_
