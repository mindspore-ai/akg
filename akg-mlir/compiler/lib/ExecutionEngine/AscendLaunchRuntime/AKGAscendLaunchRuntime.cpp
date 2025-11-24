/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "akg/ExecutionEngine/AscendLaunchRuntime/AKGAscendLaunchRuntime.h"
#include <climits>
#include <iostream>
#include <limits>
#include "akg/ExecutionEngine/AscendLaunchRuntime/AscendMemoryManager.h"
#include "akg/ExecutionEngine/AscendLaunchRuntime/RuntimeErrorCodes.h"

using std::vector;

namespace mlir {
namespace runtime {
constexpr auto kBinFileSuffix = ".so";
constexpr auto kDoBinFileSuffix = "";
constexpr auto kAiCoreStr = "AiCore";
constexpr auto kMIXStr = "MIX";

static thread_local aclrtContext thread_local_rt_context{nullptr};

AscendKernelRuntime::AscendKernelRuntime(uint32_t device_id) { set_device_id(device_id); }

void AscendKernelRuntime::SetContext() {
  if (rt_context_ == nullptr) {
    return;
  }
  if (thread_local_rt_context == rt_context_) {
    return;
  }
  auto ret = aclrtSetCurrentContext(rt_context_);
  thread_local_rt_context = rt_context_;
  if (ret != ACL_SUCCESS) {
    LOG(FATAL) << "Call aclrtSetCurrentContext, ret[" << GetErrorMsg(ret) << "]";
  }
}

void AscendKernelRuntime::SetCurrentContext() {
  if (rt_context_ == nullptr) {
    return;
  }
  auto ret = aclrtSetCurrentContext(rt_context_);
  if (ret != ACL_SUCCESS) {
    LOG(FATAL) << "Call aclrtSetCurrentContext, ret[" << GetErrorMsg(ret) << "]";
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
    auto ret = aclrtCreateContext(&rt_context_, device_id_);
    if (ret != ACL_SUCCESS) {
      LOG(FATAL) << "Call aclrtCreateContext, ret[" << static_cast<int>(ret) << "]";
    }
  }
  SetCurrentContext();
}

bool AscendKernelRuntime::InitDevice() {
  LOG(INFO) << "InitDevice: " << device_id_;
  uint32_t device_count = 0;
  auto ret = aclrtGetDeviceCount(&device_count);
  if (ret != ACL_SUCCESS) {
    LOG(FATAL) << "Call aclrtGetDeviceCount, ret[" << static_cast<int>(ret) << "]";
  }

  ret = aclrtSetDevice(device_id_);
  if (ret != ACL_SUCCESS) {
    LOG(FATAL) << "Call aclrtSetDevice, ret[" << static_cast<int>(ret) << "]";
  }

  // Context will be created by aclrtSetDevice
  ret = aclrtGetCurrentContext(&rt_context_);
  if (ret != ACL_SUCCESS || rt_context_ == nullptr) {
    LOG(FATAL) << "Call aclrtGetCurrentContext failed, ret[" << GetErrorMsg(ret) << "]";
    return false;
  }

  ret = aclrtCreateStreamWithConfig(&stream_, 0, RT_STREAM_DEFAULT);
  if (ret != ACL_SUCCESS) {
    LOG(FATAL) << "Call aclrtCreateStreamWithConfig, ret[" << GetErrorMsg(ret) << "]";
  }
  return true;
}

AscendKernelRuntime::~AscendKernelRuntime() {
  ReleaseDeviceRes();
  UnLoadKernelFunc();
}

bool AscendKernelRuntime::ResetDevice(uint32_t device_id) {
  SetCurrentContext();
  int32_t ret;
  if (stream_ != nullptr) {
    ret = aclrtDestroyStream(stream_);
    if (ret != ACL_SUCCESS) {
      LOG(FATAL) << "Call aclrtDestroyStream, ret[" << GetErrorMsg(ret) << "]";
    }
    stream_ = nullptr;
  }
  ret = aclrtResetDevice(device_id);
  if (ret != ACL_SUCCESS) {
    LOG(FATAL) << "Call aclrtResetDevice, ret[" << GetErrorMsg(ret) << "]";
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

void *AscendKernelRuntime::GetKernelFunc(const std::string &path, const std::string &kernel_name,
                                         const std::string &func_name) {
  // const auto *f = Registry::Get("get_kernel_meta_path");
  // CHECK(f != nullptr) << "Function get_kernel_meta_path is not registered";
  std::string file_str;
  auto dir_path = path;
  (void)file_str.append(dir_path).append("/lib" + kernel_name).append(kBinFileSuffix);
  char *file_c_str = (char *)file_str.c_str();

  void *handle = dlopen(file_c_str, RTLD_LAZY | RTLD_LOCAL);
  CHECK(handle != nullptr) << "dlopen failed, file: " << file_c_str;

  std::string func_str = func_name + kDoBinFileSuffix;
  char *func_c_str = (char *)func_str.c_str();
  void *func = dlsym(handle, func_c_str);
  CHECK(func != nullptr) << "dlsym failed, symbol: " << func_str;
  cce_handle_ = handle;
  return func;
}

bool AscendKernelRuntime::UnLoadKernelFunc() {
  if (cce_handle_ != nullptr) {
    if (dlclose(cce_handle_) != 0) {
      return false;
    }
  }
  cce_handle_ = nullptr;
  return true;
}

bool AscendKernelRuntime::Run(const std::string &path, const std::string &kernel_name,
                              const bool is_dynamic, const std::vector<TensorDevicePtr> &input_tensors,
                              const std::vector<std::vector<int64_t>> &input_shape_args,
                              int64_t tiling_key, int64_t tiling_struct_size) {
  uint32_t blockdim = 40;  // default blockdim equal to 1.
  int64_t offset = 0;
  std::string func_name = kernel_name;
  std::vector<void *> runtimeargs;
  
  if (is_dynamic) {
    size_t input_size = input_tensors.size();
    if (tiling_struct_size > 0)
      input_size -= 1;
    for (size_t idx = 0; idx < input_size; idx++) {
      auto tensor = input_tensors[idx];
      auto shape = input_shape_args[idx];
      runtimeargs.push_back(tensor->GetDeviceAddress());
      runtimeargs.push_back(tensor->GetDeviceAddress());
      runtimeargs.push_back(reinterpret_cast<void *>(offset));

      int64_t size = 1;
      for (auto dim : shape) {
        runtimeargs.push_back(reinterpret_cast<void *>(dim));
        size *= dim;
      }
      for (auto& dim : shape) {
        int64_t stride = size / dim;
        runtimeargs.push_back(reinterpret_cast<void *>(stride));
        size = stride;
      }
    }
    if (tiling_struct_size > 0) {
      auto tensor = input_tensors[input_size];
      runtimeargs.push_back(reinterpret_cast<void*>(&tiling_key));
      runtimeargs.push_back(tensor->GetDeviceAddress());
      runtimeargs.push_back(tensor->GetDeviceAddress());
      runtimeargs.push_back(reinterpret_cast<void*>(offset));
      runtimeargs.push_back(reinterpret_cast<void*>(tiling_struct_size));
      runtimeargs.push_back(reinterpret_cast<void*>(1));
    }
  } else {
    for (auto tensor : input_tensors)
      runtimeargs.push_back(tensor->GetDeviceAddress());
  }

  typedef void (*CallFunc)(uint32_t, void *, void *, void **);
  // kernel_name is for .so, func_name is for host func name.
  auto func_ptr = reinterpret_cast<CallFunc>(GetKernelFunc(path, kernel_name, func_name));
  func_ptr(blockdim, nullptr, stream(), runtimeargs.data());
  SyncStream();
  return true;
}

bool AscendKernelRuntime::SyncDeviceToHost(size_t size, void *device_ptr, void *host_ptr) {
  CHECK_NOTNULL(host_ptr);
  LOG(INFO) << "SyncDeviceToHost: " << size << " bytes from " << device_ptr << "(device) to " << host_ptr << "(host)";
  SyncMemory(host_ptr, device_ptr, size, ACL_MEMCPY_DEVICE_TO_HOST);
  return true;
}

bool AscendKernelRuntime::SyncHostToDevice(size_t size, const void *host_ptr, void *device_ptr) {
  CHECK_NOTNULL(host_ptr);
  LOG(INFO) << "SyncHostToDevice: " << size << " bytes from " << host_ptr << "(host) to " << device_ptr << "(device)";
  SyncMemory(device_ptr, host_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE);
  return true;
}

void AscendKernelRuntime::SyncMemory(void *dst, const void *src, uint64_t size, aclrtMemcpyKind kind) {
  SetContext();
  // Only apply asynchronous copy in Pynative && RT_MEMCPY_HOST_TO_DEVICE mode
  if (kind != ACL_MEMCPY_HOST_TO_DEVICE) {
    auto ret_rt_memcpy = aclrtMemcpy(dst, size, src, size, kind);
    if (ret_rt_memcpy != ACL_SUCCESS) {
      LOG(FATAL) << "aclrtMemcpy failed, ret[" << ret_rt_memcpy << "]";
    }
  } else {
    auto ret = MemcpyAsync(dst, src, size, static_cast<int32_t>(ACL_MEMCPY_HOST_TO_DEVICE));
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

  auto copy_kind = static_cast<aclrtMemcpyKind>(kind);
  if (copy_kind != ACL_MEMCPY_HOST_TO_DEVICE) {
    LOG(FATAL) << "Memory copy async not support cache host buffer in kind: " << kind;
  }
  auto ret = aclrtMemcpyAsync(dst, size, src, size, static_cast<aclrtMemcpyKind>(kind), stream_);
  if (ret != ACL_SUCCESS) {
    LOG(FATAL) << "Call runtime aclrtMemcpyAsync error, ret[" << GetErrorMsg(ret) << "]";
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
  auto ret = aclrtSynchronizeStream(stream_);
  if (ret != ACL_SUCCESS) {  // o for switch stream
    LOG(FATAL) << "Call runtime aclrtSynchronizeStream error, ret[" << GetErrorMsg(ret) << "]";
    return false;
  }
  return true;
}

void AscendKernelRuntime::InitDeviceMemory(const std::vector<TensorDevicePtr> &tensors) {
  for (auto tensor : tensors) {
    if(tensor->IsHostTensor()){
      auto mem_size = tensor->GetDataSize();
      auto device_addr = mem_manager_->MallocMemFromMemPool(mem_size);
      tensor->SetDeviceAddress(device_addr);
    }
  }
}

void AscendKernelRuntime::RunOpImpl(const std::string &path, const std::string &kernel_name,
                                    const bool is_dynamic, const std::vector<TensorDevicePtr> &input_tensors,
                                    const std::vector<std::vector<int64_t>> &input_shape_args,
                                    int64_t tiling_key, int64_t tiling_struct_size) {
  // InitResource
  if (!Init()) {
    LOG(FATAL) << "Kernel runtime init error.";
  }
  // malloc mem
  InitDeviceMemory(input_tensors);
  // load input data to device
  for (const auto &tensor : input_tensors) {
    if(tensor->IsHostTensor())
      SyncHostToDevice(tensor->GetDataSize(), tensor->GetHostAddress(), tensor->GetDeviceAddress());
  }
  // run op
  if (!Run(path, kernel_name, is_dynamic, input_tensors, input_shape_args, tiling_key, tiling_struct_size)) {
    LOG(FATAL) << "Kernel runtime run error.";
  }
  // get output
  for (const auto &tensor : input_tensors) {
    if (tensor->IsOutput() && tensor->IsHostTensor()) {
      SyncDeviceToHost(tensor->GetDataSize(), tensor->GetDeviceAddress(), tensor->GetHostAddress());
    }
  }
  // FreeResource
  for (const auto &tensor : input_tensors) {
    tensor->SetDeviceAddress(nullptr);
  }
}

}  // namespace runtime
}  // namespace mlir
