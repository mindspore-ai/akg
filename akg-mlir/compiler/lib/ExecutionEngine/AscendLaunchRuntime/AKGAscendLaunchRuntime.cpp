/**
 * Copyright 2024-2026 Huawei Technologies Co., Ltd
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
#include <cstdint>
#include <limits>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <string>
#include <fstream>
#include <memory>
#include <mutex>
#include <vector>
#include <unordered_map>
#include "akg/ExecutionEngine/AscendLaunchRuntime/AscendMemoryManager.h"
#include "akg/ExecutionEngine/AscendLaunchRuntime/CceWrapper.h"
#include "akg/ExecutionEngine/AscendLaunchRuntime/RuntimeErrorCodes.h"
#include <nlohmann/json.hpp>

using std::vector;

namespace mlir {
namespace runtime {

constexpr auto kDynamicSuffix = ".so";
constexpr auto kStaticSuffix = ".o";
constexpr auto kAiCoreStr = "AiCore";
constexpr auto kMIXStr = "MIX";
static thread_local aclrtContext thread_local_rt_context{nullptr};

uint64_t kDevRegStub = 0xbadbeefULL;
std::mutex kDevRegStubMutex;

/// Read a device binary file into memory. Returns empty vector if open/read fails.
std::vector<char> ReadDeviceBinaryFile(const std::string &bin_path) {
  std::ifstream ifs(bin_path, std::ios::binary);
  if (!ifs) {
    return {};
  }
  ifs.seekg(0, std::ios::end);
  const auto end = ifs.tellg();
  if (end <= 0) {
    return {};
  }
  ifs.seekg(0, std::ios::beg);
  std::vector<char> buffer(static_cast<size_t>(end));
  if (!ifs.read(buffer.data(), end)) {
    return {};
  }
  return buffer;
}

/// Register an in-memory device binary via rtDevBinaryRegister and rtFunctionRegister (CANN runtime).
/// \p data must remain valid for the duration of this call.
/// \return Host stub address suitable for rtKernelLaunch, or 0 on failure.
uintptr_t RegisterDeviceKernel(const std::string &func_name, const void *data, size_t length,
                               uint32_t magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC, uint32_t version = 0) {
  if (data == nullptr || length == 0) {
    return 0;
  }

  rtDevBinary_t binary{};
  binary.data = data;
  binary.length = static_cast<uint64_t>(length);
  binary.magic = magic;
  binary.version = version;

  void *bin_handle = nullptr;
  auto ret = rtDevBinaryRegister(&binary, &bin_handle);
  if (ret != RT_ERROR_NONE) {
    LOG(FATAL) << "Call rtDevBinaryRegister, ret[" << ret << "]";
    return 0;
  }

  uintptr_t stub_addr = 0;
  {
    std::lock_guard<std::mutex> lock(kDevRegStubMutex);
    stub_addr = kDevRegStub += 1;
  }

  ret = rtFunctionRegister(bin_handle, reinterpret_cast<void *>(stub_addr), func_name.c_str(),
                           reinterpret_cast<const void *>(func_name.c_str()), 0);
  if (ret != RT_ERROR_NONE) {
    LOG(FATAL) << "Call rtFunctionRegister, ret[" << ret << "]";
    return 0;
  }
  return stub_addr;
}

uintptr_t GetKernelFunction(const std::string &func_name, const std::string &bin_path) {
  std::vector<char> bin = ReadDeviceBinaryFile(bin_path);
  auto func = RegisterDeviceKernel(func_name, bin.data(), bin.size());
  return func;
}

namespace {
// key: so_path::symbol_name, value: (handle, func_ptr)
struct KernelFuncCache {
  std::mutex mutex;
  std::unordered_map<std::string, std::pair<void *, void *>> cache;

  void *Get(const std::string &path, const std::string &kernel_name, const std::string &func_name,
            bool is_dynamic = false) {
    std::string file_str =
      is_dynamic ? path + "/lib" + kernel_name + kDynamicSuffix : path + "/" + kernel_name + kStaticSuffix;
    std::string func_str = func_name;
    std::string key = file_str + "::" + func_str;

    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
      return it->second.second;
    }

    void *handle = nullptr;
    void *func = nullptr;
    if (is_dynamic) {
      handle = dlopen(file_str.c_str(), RTLD_LAZY | RTLD_LOCAL);
      CHECK(handle != nullptr) << "dlopen failed, file: " << file_str << ", Error:" << dlerror();

      func = dlsym(handle, func_str.c_str());
      CHECK(func != nullptr) << "dlsym failed, symbol: " << func_str;
    } else {
      func = reinterpret_cast<void *>(GetKernelFunction(kernel_name, file_str));
    }

    cache[key] = {handle, func};
    return func;
  }
};

KernelFuncCache &GetKernelFuncCache() {
  static KernelFuncCache cache;
  return cache;
}

struct BlockDimCache {
  std::mutex mutex;
  std::unordered_map<std::string, uint32_t> cache;

  uint32_t Get(const std::string &path, const std::string &kernel_name) {
    std::string key = path + "/" + kernel_name + ".json";

    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
      return it->second;
    }

    std::ifstream json_stream(key);
    nlohmann::json json_data;
    json_stream >> json_data;
    uint32_t blockdim = json_data["blockDim"].get<uint32_t>();
    cache[key] = blockdim;
    return blockdim;
  }
};

BlockDimCache &GetBlockDimCache() {
  static BlockDimCache cache;
  return cache;
}

struct RuntimeCache {
  std::mutex mutex;
  std::unordered_map<std::string, std::unique_ptr<AscendKernelRuntime>> cache;

  AscendKernelRuntime *GetOrCreate(uint32_t device_id, bool use_mem_pool, void *external_stream) {
    std::string key = std::to_string(device_id) + "_" + std::to_string(use_mem_pool) + "_" +
                      std::to_string(reinterpret_cast<uintptr_t>(external_stream));
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
      return it->second.get();
    }
    auto rt = std::make_unique<AscendKernelRuntime>(device_id, use_mem_pool, external_stream);
    AscendKernelRuntime *ptr = rt.get();
    cache[key] = std::move(rt);
    return ptr;
  }
};

RuntimeCache &GetRuntimeCache() {
  static RuntimeCache cache;
  return cache;
}

}  // namespace

AscendKernelRuntime *AscendKernelRuntime::GetOrCreateRuntime(uint32_t device_id, bool use_mem_pool,
                                                             void *external_stream) {
  return GetRuntimeCache().GetOrCreate(device_id, use_mem_pool, external_stream);
}

AscendKernelRuntime::AscendKernelRuntime(uint32_t device_id, bool use_mem_pool, void *external_stream) {
  set_device_id(device_id);
  use_mem_pool_ = use_mem_pool;
  if (external_stream != nullptr) {
    if (reinterpret_cast<uintptr_t>(external_stream) == static_cast<uintptr_t>(-1)) {
      stream_ = nullptr;
    } else {
      stream_ = external_stream;
    }
    owns_stream_ = false;
  }
}

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
  DLOG(INFO) << "Ascend finalize start";
  if (!initialized_) {
    return;
  }
  if (!owns_stream_) {
    initialized_ = false;
    DLOG(INFO) << "Ascend finalize end (external stream / PTA, skipped device reset)";
    return;
  }
  if (rt_context_ != nullptr) {
    auto ret = aclrtSetCurrentContext(rt_context_);
    if (ret != ACL_SUCCESS) {
      DLOG(WARNING) << "aclrtSetCurrentContext failed at shutdown, ret[" << GetErrorMsg(ret)
                    << "], skip ReleaseDeviceRes";
      return;
    }
  }
  if (mem_manager_ != nullptr) {
    mem_manager_->FreeDeviceMemory();
  }
  (void)ResetDevice(device_id_);
  DLOG(INFO) << "Ascend finalize end";
}

bool AscendKernelRuntime::Init() {
  if (initialized_) {
    if (owns_stream_) {
      SetCurrentContext();
    }
    return true;
  }

  bool ret = InitDevice();
  if (!ret) {
    return ret;
  }
  if (use_mem_pool_) {
    mem_manager_ = std::make_shared<AscendMemoryManager>();
    CHECK_NOTNULL(mem_manager_);
    mem_manager_->MallocDeviceMemory();
  }

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
  DLOG(INFO) << "InitDevice: " << device_id_;

  if (!owns_stream_) {
    auto ret = aclrtGetCurrentContext(&rt_context_);
    if (ret != ACL_SUCCESS || rt_context_ == nullptr) {
      DLOG(WARNING) << "External stream mode: aclrtGetCurrentContext failed, ret["
                    << GetErrorMsg(ret) << "], context will be nullptr";
      rt_context_ = nullptr;
    }
    return true;
  }

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

  if (stream_ == nullptr) {
    ret = aclrtCreateStreamWithConfig(&stream_, 0, RT_STREAM_DEFAULT);
    if (ret != ACL_SUCCESS) {
      LOG(FATAL) << "Call aclrtCreateStreamWithConfig, ret[" << GetErrorMsg(ret) << "]";
    }
    owns_stream_ = true;
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
  if (stream_ != nullptr && owns_stream_) {
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
  DLOG(INFO) << "ResetDevice: " << device_id;
  return true;
}

inline unsigned int UlongToUint(uint64_t u) {
  if (u > static_cast<uint64_t>((std::numeric_limits<unsigned int>::max)())) {
    LOG(FATAL) << "The size_t value(" << u << ") exceeds the maximum value of unsigned int.";
  }
  return static_cast<unsigned int>(u);
}

void *AscendKernelRuntime::GetKernelFunc(const std::string &path, const std::string &kernel_name,
                                         const std::string &func_name, bool is_dynamic) {
  void *func = GetKernelFuncCache().Get(path, kernel_name, func_name, is_dynamic);
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

bool AscendKernelRuntime::Run(const std::string &path, const std::string &kernel_name, const bool is_dynamic,
                              const std::vector<BaseDevicePtr> &input_tensors,
                              const std::vector<std::vector<int64_t>> &input_shape_args, int64_t tiling_key,
                              int64_t tiling_struct_size) {
  uint32_t blockdim = GetBlockDimCache().Get(path, kernel_name);
  std::string func_name = kernel_name;
  std::vector<void *> runtimeargs;

  if (is_dynamic) {
    int64_t offset = 0;
    size_t input_size = input_tensors.size();
    if (tiling_struct_size > 0) input_size -= 1;
    for (size_t idx = 0; idx < input_size; idx++) {
      auto base = input_tensors[idx];
      auto shape = input_shape_args[idx];

      auto tensor = mlir::runtime::AsTensorDevice(base);
      if (!tensor) {
        runtimeargs.push_back(mlir::runtime::GetScalarValuePtr(base));
      } else {
        runtimeargs.push_back(tensor->GetDeviceAddress());
        runtimeargs.push_back(tensor->GetDeviceAddress());
        runtimeargs.push_back(reinterpret_cast<void *>(offset));

        int64_t size = 1;
        for (auto dim : shape) {
          runtimeargs.push_back(reinterpret_cast<void *>(dim));
          size *= dim;
        }
        for (auto &dim : shape) {
          int64_t stride = size / dim;
          runtimeargs.push_back(reinterpret_cast<void *>(stride));
          size = stride;
        }
      }
    }

    if (tiling_struct_size > 0) {
      auto tensor = mlir::runtime::AsTensorDevice(input_tensors[input_size]);
      runtimeargs.push_back(reinterpret_cast<void *>(&tiling_key));
      runtimeargs.push_back(tensor->GetDeviceAddress());
      runtimeargs.push_back(tensor->GetDeviceAddress());
      runtimeargs.push_back(reinterpret_cast<void *>(offset));
      runtimeargs.push_back(reinterpret_cast<void *>(tiling_struct_size));
      runtimeargs.push_back(reinterpret_cast<void *>(1));
    }

    typedef void (*CallFunc)(uint32_t, void *, void *, void **);
    // kernel_name is for .so, func_name is for host func name.
    auto func_ptr = reinterpret_cast<CallFunc>(GetKernelFunc(path, kernel_name, func_name, is_dynamic));
    func_ptr(blockdim, nullptr, stream(), runtimeargs.data());
  } else {
    for (const auto &base : input_tensors) {
      auto tensor = mlir::runtime::AsTensorDevice(base);
      runtimeargs.push_back(tensor->GetDeviceAddress());
    }
    auto func = GetKernelFunc(path, kernel_name, func_name, is_dynamic);
    rtError_t ret =
      rtKernelLaunch(func, blockdim, runtimeargs.data(), runtimeargs.size() * sizeof(void *), NULL, stream());
    if (ret != RT_ERROR_NONE) {
      LOG(FATAL) << "Call rtKernelLaunch, ret[" << GetErrorMsg(ret) << "]";
      return false;
    }
  }

  if (owns_stream_) {
    SyncStream();
  }
  return true;
}

bool AscendKernelRuntime::SyncDeviceToHost(size_t size, void *device_ptr, void *host_ptr) {
  CHECK_NOTNULL(host_ptr);
  DLOG(INFO) << "SyncDeviceToHost: " << size << " bytes from " << device_ptr << "(device) to " << host_ptr << "(host)";
  SyncMemory(host_ptr, device_ptr, size, ACL_MEMCPY_DEVICE_TO_HOST);
  return true;
}

bool AscendKernelRuntime::SyncHostToDevice(size_t size, const void *host_ptr, void *device_ptr) {
  CHECK_NOTNULL(host_ptr);
  DLOG(INFO) << "SyncHostToDevice: " << size << " bytes from " << host_ptr << "(host) to " << device_ptr << "(device)";
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

void AscendKernelRuntime::InitDeviceMemory(const std::vector<BaseDevicePtr> &tensors) {
  if (!use_mem_pool_) return;
  for (auto base : tensors) {
    auto tensor = mlir::runtime::AsTensorDevice(base);
    if (!tensor) continue;
    if (tensor->IsHostTensor()) {
      auto mem_size = tensor->GetDataSize();
      auto device_addr = mem_manager_->MallocMemFromMemPool(mem_size);
      tensor->SetDeviceAddress(device_addr);
    }
  }
}

void AscendKernelRuntime::RunOpImpl(const std::string &path, const std::string &kernel_name, const bool is_dynamic,
                                    const std::vector<BaseDevicePtr> &input_tensors,
                                    const std::vector<std::vector<int64_t>> &input_shape_args, int64_t tiling_key,
                                    int64_t tiling_struct_size) {
  // InitResource
  if (!Init()) {
    LOG(FATAL) << "Kernel runtime init error.";
  }
  // malloc mem
  InitDeviceMemory(input_tensors);
  // load input data to device
  for (const auto &base : input_tensors) {
    auto tensor = mlir::runtime::AsTensorDevice(base);
    if (!tensor) continue;
    if (tensor->IsHostTensor()) {
      SyncHostToDevice(tensor->GetDataSize(), tensor->GetHostAddress(), tensor->GetDeviceAddress());
    }
  }
  // run op
  if (!Run(path, kernel_name, is_dynamic, input_tensors, input_shape_args, tiling_key, tiling_struct_size)) {
    LOG(FATAL) << "Kernel runtime run error.";
  }
  // get output
  for (const auto &base : input_tensors) {
    auto tensor = mlir::runtime::AsTensorDevice(base);
    if (!tensor) continue;
    if (tensor->IsOutput() && tensor->IsHostTensor()) {
      SyncDeviceToHost(tensor->GetDataSize(), tensor->GetDeviceAddress(), tensor->GetHostAddress());
    }
  }
  // FreeResource
  for (const auto &base : input_tensors) {
    auto tensor = mlir::runtime::AsTensorDevice(base);
    if (!tensor) continue;
    tensor->SetDeviceAddress(nullptr);
  }
}
}  // namespace runtime
}  // namespace mlir
