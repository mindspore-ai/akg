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
#ifndef SRC_RUNTIME_ASCEND_ASCEND_KERNEL_RUNTIME_H_
#define SRC_RUNTIME_ASCEND_ASCEND_KERNEL_RUNTIME_H_
#include <dirent.h>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include "ascend_memory_manager.h"
#include "tensor_device.h"
#include "runtime/context.h"
#include "runtime/mem.h"

namespace air {
namespace runtime {
class AscendKernelRuntime {
 public:
  AscendKernelRuntime() = default;
  ~AscendKernelRuntime() = default;
  bool Init();
  void SetContext();
  void CreateContext();
  void ReleaseDeviceRes();
  bool Run(const std::string &kernel_name, const std::vector<TensorDevicePtr> &input_tensors);
  bool SyncStream();
  bool MemcpyAsync(void *dst, const void *src, uint64_t size, int32_t kind);
  void RunOpAssignMemory(const std::vector<TensorDevicePtr> &tensors);
  bool SyncDeviceToHost(size_t size, void *device_ptr, void *host_ptr);
  bool SyncHostToDevice(size_t size, const void *host_ptr, void *device_ptr);
  void RunOpImpl(const std::string &kernel_name, const std::vector<TensorDevicePtr> &input_tensors);

  void set_device_id(uint32_t device_id) { device_id_ = device_id; }
  uint32_t device_id() { return device_id_; }
  void *stream() { return stream_; }

 private:
  bool InitDevice();
  bool ResetDevice(uint32_t device_id);
  void SetCurrentContext();
  void SyncMemory(void *dst, const void *src, uint64_t size, rtMemcpyKind_t kind);

  rtContext_t rt_context_{nullptr};
  bool initialized_{false};
  uint32_t device_id_{0};
  void *stream_{nullptr};
  std::shared_ptr<AscendMemoryManager> mem_manager_{nullptr};
};
}  // namespace runtime
}  // namespace air
#endif  // SRC_RUNTIME_ASCEND_ASCEND_KERNEL_RUNTIME_H_
