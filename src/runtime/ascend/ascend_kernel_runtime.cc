/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include <dmlc/common.h>
#include "ascend_kernel_runtime.h"
#include "runtime/rt.h"
#include "ascend_memory_manager.h"
#include "kernel.h"
#include "tvm.h"
#include <tvm/runtime/registry.h>
#include "runtime_error_codes.h"
#include <climits>

using std::vector;

namespace air {
namespace runtime {
static thread_local rtContext_t thread_local_rt_context{nullptr};

void AscendKernelRuntime::SetContext() {
  if (rt_context_ == nullptr) {
    return;
  }
  if (thread_local_rt_context == rt_context_) {
    return;
  }
  auto ret = rtCtxSetCurrent(rt_context_);
  thread_local_rt_context = rt_context_;
  if (ret != RT_ERROR_NONE) {
    LOG(FATAL) << "Call rtCtxSetCurrent, ret[" << GetErrorMsg(ret) << "]";
  }
}

void AscendKernelRuntime::SetCurrentContext() {
  if (rt_context_ == nullptr) {
    return;
  }
  auto ret = rtCtxSetCurrent(rt_context_);
  if (ret != RT_ERROR_NONE) {
    LOG(FATAL) << "Call rtCtxSetCurrent, ret[" << GetErrorMsg(ret) << "]";
  }
}

void AscendKernelRuntime::ReleaseDeviceRes() {
  LOG(INFO) << "Ascend finalize start";
  if (!initialized_) {
    return;
  }
  SetCurrentContext();
  if (mem_manager_ != nullptr) {
    mem_manager_->FreeDeviceMemory();
  }
  (void)ResetDevice(device_id_);
  LOG(INFO) << "Ascend finalize end";
}

bool AscendKernelRuntime::Init() {
  if (initialized_) {
    SetCurrentContext();
    return true;
  }

  bool ret = InitDevice();
  if (!ret) {
    return ret;
  }

  mem_manager_ = std::make_shared<AscendMemoryManager>();
  CHECK_NOTNULL(mem_manager_);
  mem_manager_->MallocDeviceMemory();

  initialized_ = true;
  return ret;
}

void AscendKernelRuntime::CreateContext() {
  if (rt_context_ == nullptr) {
    auto ret = rtCtxCreate(&rt_context_, 0, device_id_);
    if (ret != RT_ERROR_NONE) {
      LOG(FATAL) << "Call rtCtxCreate, ret[" << static_cast<int>(ret) << "]";
    }
  }
  SetCurrentContext();
}

bool AscendKernelRuntime::InitDevice() {
  LOG(INFO) << "InitDevice: " << device_id_;
  int device_count = 0;
  auto ret = rtGetDeviceCount(&device_count);
  if (ret != RT_ERROR_NONE) {
    LOG(FATAL) << "Call rtGetDeviceCount, ret[" << static_cast<int>(ret) << "]";
  }

  ret = rtSetDevice(device_id_);
  if (ret != RT_ERROR_NONE) {
    LOG(FATAL) << "Call rtSetDevice, ret[" << static_cast<int>(ret) << "]";
  }

  // Context will be created by rtSetDevice
  ret = rtCtxGetCurrent(&rt_context_);
  if (ret != RT_ERROR_NONE || rt_context_ == nullptr) {
    LOG(FATAL) << "Call rtCtxGetCurrent failed, ret[" << GetErrorMsg(ret) << "]";
    return false;
  }

  ret = rtStreamCreateWithFlags(&stream_, 0, RT_STREAM_HUGE);
  if (ret != RT_ERROR_NONE) {
    LOG(FATAL) << "Call rtStreamCreate, ret[" << GetErrorMsg(ret) << "]";
  }
  return true;
}

bool AscendKernelRuntime::ResetDevice(uint32_t device_id) {
  SetCurrentContext();
  int32_t ret;
  if (stream_ != nullptr) {
    ret = rtStreamDestroy(stream_);
    if (ret != RT_ERROR_NONE) {
      LOG(FATAL) << "Call rtStreamDestroy, ret[" << GetErrorMsg(ret) << "]";
    }
    stream_ = nullptr;
  }
  ret = rtDeviceReset(device_id);
  if (ret != RT_ERROR_NONE) {
    LOG(FATAL) << "Call rtDeviceReset, ret[" << GetErrorMsg(ret) << "]";
  }
  // set to nullptr as its not created, only bounded to existing context
  rt_context_ = nullptr;
  LOG(INFO) << "ResetDevice: " << device_id;
  return true;
}

inline unsigned int UlongToUint(uint64_t u) {
  if (u > static_cast<uint64_t>((std::numeric_limits<unsigned int>::max)())) {
    LOG(FATAL) << "The size_t value(" << u << ") exceeds the maximum value of unsigned int.";
  }
  return static_cast<unsigned int>(u);
}

bool AscendKernelRuntime::Run(const std::string &kernel_name, const std::vector<TensorDevicePtr> &input_tensors,
                              const std::vector<int64_t> &input_shape_args) {
  uint32_t blockdim = 1;  // default blockdim equal to 1.
  auto kernel_pack_ptr = GetKernelPack(kernel_name);
  auto func_stub = GetFuncStub(*kernel_pack_ptr, &blockdim);
  if (func_stub == 0) {
    LOG(FATAL) << "GenFuncStub failed.";
    return false;
  }
  // pack all addresses into a vector.
  std::vector<void *> runtimeargs;
  for (auto tensor : input_tensors) {
    runtimeargs.push_back(tensor->GetDeviceAddress());
  }
  for (const auto &shape_arg : input_shape_args) {
    runtimeargs.push_back(reinterpret_cast<void *>(shape_arg));
  }
  rtL2Ctrl_t *l2ctrl = nullptr;
  const void *stubFunc = reinterpret_cast<void *>(func_stub);
  auto argsSize = static_cast<uint32_t>(UlongToUint(sizeof(void *)) * runtimeargs.size());
  if (input_shape_args.size() > 0 && blockdim == INT_MAX) {
    blockdim = input_shape_args[input_shape_args.size() - 1];
  }
  auto ret = rtKernelLaunch(stubFunc, blockdim, runtimeargs.data(), argsSize, l2ctrl, stream());
  SyncStream();
  if (ret != RT_ERROR_NONE) {
    LOG(FATAL) << "Call runtime rtKernelLaunch error, ret[" << GetErrorMsg(ret) << "]";
    return false;
  }
  return true;
}

bool AscendKernelRuntime::SyncDeviceToHost(size_t size, void *device_ptr, void *host_ptr) {
  CHECK_NOTNULL(host_ptr);
  LOG(INFO) << "SyncDeviceToHost: " << size << " bytes from " << device_ptr << "(device) to " << host_ptr << "(host)";
  SyncMemory(host_ptr, device_ptr, size, RT_MEMCPY_DEVICE_TO_HOST);
  return true;
}

bool AscendKernelRuntime::SyncHostToDevice(size_t size, const void *host_ptr, void *device_ptr) {
  CHECK_NOTNULL(host_ptr);
  LOG(INFO) << "SyncHostToDevice: " << size << " bytes from " << host_ptr << "(host) to " << device_ptr << "(device)";
  SyncMemory(device_ptr, host_ptr, size, RT_MEMCPY_HOST_TO_DEVICE);
  return true;
}

void AscendKernelRuntime::SyncMemory(void *dst, const void *src, uint64_t size, rtMemcpyKind_t kind) {
  SetContext();
  // Only apply asynchronous copy in Pynative && RT_MEMCPY_HOST_TO_DEVICE mode
  if (kind != RT_MEMCPY_HOST_TO_DEVICE) {
    auto ret_rt_memcpy = rtMemcpy(dst, size, src, size, kind);
    if (ret_rt_memcpy != RT_ERROR_NONE) {
      LOG(FATAL) << "rtMemcpy failed, ret[" << ret_rt_memcpy << "]";
    }
  } else {
    auto ret = MemcpyAsync(dst, src, size, static_cast<int32_t>(RT_MEMCPY_HOST_TO_DEVICE_EX));
    if (!ret) {
      LOG(FATAL) << "MemcpyAsync failed, ret[" << GetErrorMsg(ret) << "]";
    }
  }
}

bool AscendKernelRuntime::MemcpyAsync(void *dst, const void *src, uint64_t size, int32_t kind) {
  SetCurrentContext();
  if (stream_ == nullptr) {
    LOG(FATAL) << "MemcpyAsync failed. stream_ is nullptr";
    return false;
  }

  auto copy_kind = static_cast<rtMemcpyKind_t>(kind);
  if (copy_kind != RT_MEMCPY_HOST_TO_DEVICE_EX) {
    LOG(FATAL) << "Memory copy async not support cache host buffer in kind: " << kind;
  }
  auto ret = rtMemcpyAsync(dst, size, src, size, static_cast<rtMemcpyKind_t>(kind), stream_);
  if (ret != RT_ERROR_NONE) {
    LOG(FATAL) << "Call runtime rtMemcpyAsync error, ret[" << GetErrorMsg(ret) << "]";
    return false;
  }
  return true;
}

bool AscendKernelRuntime::SyncStream() {
  SetCurrentContext();
  if (stream_ == nullptr) {
    LOG(FATAL) << "SyncStream failed. stream_ is nullptr";
    return false;
  }
  auto ret = rtStreamSynchronize(stream_);
  if (ret != RT_ERROR_NONE) {  // o for switch stream
    LOG(FATAL) << "Call runtime rtStreamSynchronize error, ret[" << GetErrorMsg(ret) << "]";
    return false;
  }
  return true;
}

void AscendKernelRuntime::RunOpAssignMemory(const std::vector<TensorDevicePtr> &tensors) {
  for (auto tensor : tensors) {
    auto mem_size = tensor->GetDataSize();
    auto device_addr = mem_manager_->MallocMemFromMemPool(mem_size);
    tensor->SetDeviceAddress(device_addr);
  }
}

void AscendKernelRuntime::RunOpImpl(const std::string &kernel_name, const std::vector<TensorDevicePtr> &input_tensors,
                                    const std::vector<int64_t> &input_shape_args) {
  // InitResource
  if (!Init()) {
    LOG(FATAL) << "Kernel runtime init error.";
  }
  // malloc mem
  RunOpAssignMemory(input_tensors);
  // load input data to device
  for (const auto &tensor : input_tensors) {
    SyncHostToDevice(tensor->GetDataSize(), tensor->GetHostAddress(), tensor->GetDeviceAddress());
  }
  // run op
  if (!Run(kernel_name, input_tensors, input_shape_args)) {
    LOG(FATAL) << "Kernel runtime run error.";
  }
  // get output
  for (const auto &tensor : input_tensors) {
    if (tensor->IsOutput()) {
      SyncDeviceToHost(tensor->GetDataSize(), tensor->GetDeviceAddress(), tensor->GetHostAddress());
    }
  }
  // FreeResource
  for (const auto &tensor : input_tensors) {
    tensor->SetDeviceAddress(nullptr);
  }
  ReleaseDeviceRes();
}

TVM_REGISTER_GLOBAL("ascend_run").set_body([](TVMArgs args, TVMRetValue *ret) {
  auto kernel_name = args[0].operator std::string();
  auto device_id = static_cast<uint32_t>(args[1].operator int());
  auto input_tensors = std::vector<TensorDevicePtr>();
  auto input_shape_args = std::vector<int64_t>();
  for (auto i = 2; i < args.size();) {
    if (args[i].type_code() == kDLInt) {
      auto shape_args = args[i++].operator int64_t();
      input_shape_args.push_back(shape_args);
    } else {
      auto data_ptr = args[i++].operator void *();
      auto nbytes = static_cast<size_t>(args[i++].operator uint64_t());
      auto is_output = args[i++].operator bool();
      input_tensors.push_back(std::make_shared<TensorDevice>(data_ptr, nbytes, is_output));
    }
  }
  auto kernel_runtime_ptr = std::make_shared<AscendKernelRuntime>();
  kernel_runtime_ptr->set_device_id(device_id);
  kernel_runtime_ptr->RunOpImpl(kernel_name, input_tensors, input_shape_args);
});

}  // namespace runtime
}  // namespace air
