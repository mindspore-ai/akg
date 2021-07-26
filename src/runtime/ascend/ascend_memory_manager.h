/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef SRC_RUNTIME_ASCEND_ASCEND_MEMORY_MANAGER_H_
#define SRC_RUNTIME_ASCEND_ASCEND_MEMORY_MANAGER_H_
#include <memory>
#include <utility>
#include <vector>

namespace air {
namespace runtime {
const uint64_t kMemAlignSize = 512;
using DeviceMemPtr = void(*);

class AscendMemoryManager {
 public:
  AscendMemoryManager() = default;
  ~AscendMemoryManager() = default;

  void MallocDeviceMemory();
  void FreeDeviceMemory();
  void *MallocMemFromMemPool(size_t size);

 private:
  static size_t GetCommonAlignSize(size_t input_size);
  void PoolAllocDeviceMem(size_t size, DeviceMemPtr *addr);
  uint8_t *device_mem_base_{nullptr};
  uint64_t device_mem_offset_{0};
  uint64_t device_mem_size_{0};
};
}  // namespace runtime
}  // namespace air
#endif  // SRC_RUNTIME_ASCEND_ASCEND_MEMORY_MANAGER_H_
