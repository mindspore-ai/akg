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

#include "akg/ExecutionEngine/AscendLaunchRuntime/AscendMemoryManager.h"
#include <thread>
#include "akg/ExecutionEngine/AscendLaunchRuntime/CceAcl.h"
#include "akg/ExecutionEngine/AscendLaunchRuntime/RuntimeErrorCodes.h"

namespace mlir {
namespace runtime {
namespace {
constexpr uint64_t kAscendInitDeviceMemGB = 30;
constexpr uint64_t kMemSizeGB = 30;
constexpr uint64_t kAscendDeviceMemSize = (kAscendInitDeviceMemGB << kMemSizeGB);

uint64_t GetDeviceMemSize() {
  size_t free = 0;
  size_t total = 0;
  aclError ret = aclrtGetMemInfo(ACL_HBM_MEM, &free, &total);
  if (ret != ACL_SUCCESS) {
    LOG(FATAL) << "Get Device HBM memory size failed, ret = " << ret << ", total =  " << total;
  }
  if (total != 0) {
    return total;
  }
  ret = aclrtGetMemInfo(ACL_DDR_MEM, &free, &total);
  if (ret != ACL_SUCCESS) {
    LOG(FATAL) << "Get Device DDR memory size failed, ret = " << ret << ", total =  " << total;
  }
  return total;
}

uint64_t GetDefaultDeviceMemSize() {
  auto total = GetDeviceMemSize();
  auto ret = total * 15 / 16;  // reserved memory is 1/16 of total
  LOG(INFO) << "The Device total memory size is " << total << ", allocate " << ret << " for backend.";
  return ret;
}
}  // namespace

void AscendMemoryManager::MallocDeviceMemory() {
  device_mem_size_ = GetDefaultDeviceMemSize();
  aclError ret;
  auto max_retry = 3;
  for (auto i = 0; i < max_retry; ++i) {
    ret = aclrtMalloc(reinterpret_cast<void **>(&device_mem_base_), device_mem_size_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret == ACL_ERROR_RT_MEMORY_ALLOCATION) {
      LOG(WARNING) << "Device may be occupied, sleep 1s and retry again!";
      device_mem_base_ = nullptr;
      std::this_thread::sleep_for(std::chrono::microseconds(1000000));
    } else {
      break;
    }
  }

  if (ret != ACL_SUCCESS) {
    LOG(FATAL) << "aclrtMalloc mem size[" << device_mem_size_ << "] fail, ret[" << ret << "]";
  } else {
    device_mem_offset_ = device_mem_size_;
    LOG(INFO) << "Call aclrtMalloc to allocate device memory Success, size : " << device_mem_size_
              << " bytes , address : " << reinterpret_cast<void *>(device_mem_base_);
  }
}

void AscendMemoryManager::FreeDeviceMemory() {
  if (device_mem_base_ != nullptr) {
    auto ret = aclrtFree(device_mem_base_);
    if (ret != ACL_SUCCESS) {
      LOG(FATAL) << "aclrtFree mem size[" << device_mem_size_ << "] fail, ret[" << ret << "]";
    }
    device_mem_base_ = nullptr;
  }
}

constexpr size_t kAlignBytes = 32;

size_t AscendMemoryManager::GetCommonAlignSize(size_t input_size) {
  return (input_size + kMemAlignSize + kAlignBytes - 1) / kMemAlignSize * kMemAlignSize;
}

void AscendMemoryManager::PoolAllocDeviceMem(size_t size, DeviceMemPtr *addr) {
  LOG(INFO) << "Malloc Memory: Pool, total[" << device_mem_size_ << "] memory pool["
            << device_mem_size_ - device_mem_offset_ << "])" << " malloc [" << size << "]";
  if (size == 0) {
    LOG(FATAL) << "Failed to alloc memory pool resource, the size is zero!";
  }

  if (device_mem_offset_ - size < 0) {
    LOG(FATAL) << "size: " << size << "exceed the device memory offset: " << device_mem_offset_;
  }
  device_mem_offset_ -= size;
  *addr = device_mem_base_ + device_mem_offset_;
  if (*addr == nullptr) {
    LOG(FATAL) << "Alloc device memory pool address is nullptr, failed to alloc memory pool resource!";
  }
  return;
}

void *AscendMemoryManager::MallocMemFromMemPool(size_t size) {
  DeviceMemPtr addr;
  auto align_size = GetCommonAlignSize(size);
  PoolAllocDeviceMem(align_size, &addr);
  return addr;
}

}  // namespace runtime
}  // namespace mlir
