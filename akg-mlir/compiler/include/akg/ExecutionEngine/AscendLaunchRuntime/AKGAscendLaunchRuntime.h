/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#ifndef COMPILER_INCLUDE_AKG_EXECUTIONENGINE_AKGASCENDLAUNCHRUNTIME_AKGASCENDLAUNCHRUNTIME_H_
#define COMPILER_INCLUDE_AKG_EXECUTIONENGINE_AKGASCENDLAUNCHRUNTIME_AKGASCENDLAUNCHRUNTIME_H_
#include <dirent.h>
#include <dlfcn.h>

#include <stdint.h>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "akg/ExecutionEngine/AscendLaunchRuntime/AscendMemoryManager.h"
#include "akg/ExecutionEngine/AscendLaunchRuntime/CceAcl.h"
#include "akg/ExecutionEngine/AscendLaunchRuntime/TensorDevice.h"
#include "akg/ExecutionEngine/AscendLaunchRuntime/logger.h"

#ifdef akg_ascend_launch_runtime_EXPORTS
// We are building this library
#define AKG_ASCEND_LAUNCH_RUNTIME_DEFINE_FUNCTIONS
#endif  // akg_ascend_launch_runtime_EXPORTS

namespace mlir {
namespace runtime {
class AscendKernelRuntime {
 public:
  explicit AscendKernelRuntime(uint32_t device_id, bool use_mem_pool, void *external_stream = nullptr);
  ~AscendKernelRuntime();
  bool Init();
  void SetContext();
  void CreateContext();
  void ReleaseDeviceRes();
  bool Run(const std::string &path, const std::string &kernel_name, const bool is_dynamic,
           const std::vector<BaseDevicePtr> &input_tensors, const std::vector<std::vector<int64_t>> &input_shape_args,
           int64_t tiling_key, int64_t tiling_struct_size);
  bool SyncStream();
  bool MemcpyAsync(void *dst, const void *src, uint64_t size, int32_t kind);
  void InitDeviceMemory(const std::vector<BaseDevicePtr> &tensors);
  bool SyncDeviceToHost(size_t size, void *device_ptr, void *host_ptr);
  bool SyncHostToDevice(size_t size, const void *host_ptr, void *device_ptr);
  void RunOpImpl(const std::string &path, const std::string &kernel_name, const bool is_dynamic,
                 const std::vector<BaseDevicePtr> &input_tensors,
                 const std::vector<std::vector<int64_t>> &input_shape_args, int64_t tiling_key,
                 int64_t tiling_struct_size);

  void set_device_id(uint32_t device_id) { device_id_ = device_id; }
  uint32_t device_id() { return device_id_; }
  void *stream() { return stream_; }

  static AscendKernelRuntime *GetOrCreateRuntime(uint32_t device_id, bool use_mem_pool, void *external_stream);

 private:
  bool InitDevice();
  bool ResetDevice(uint32_t device_id);
  void SetCurrentContext();
  void SyncMemory(void *dst, const void *src, uint64_t size, aclrtMemcpyKind kind);
  void *GetKernelFunc(const std::string &path, const std::string &kernel_name, const std::string &func_name);
  bool UnLoadKernelFunc();

  aclrtContext rt_context_{nullptr};
  bool initialized_{false};
  uint32_t device_id_{0};
  void *stream_{nullptr};
  bool owns_stream_{true};  // false when using external_stream from PTA
  std::shared_ptr<AscendMemoryManager> mem_manager_{nullptr};
  void *cce_handle_{nullptr};
  bool use_mem_pool_{true};
};

}  // namespace runtime
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_EXECUTIONENGINE_AKGASCENDLAUNCHRUNTIME_AKGASCENDLAUNCHRUNTIME_H_
