/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef PROFILER_GPU_PROFILING_H_
#define PROFILER_GPU_PROFILING_H_
#include <cuda.h>
#include <cupti.h>
#include <cstdio>
#include <unordered_map>
#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <algorithm>
#include <utility>

namespace akg {
enum class CUPTIApiType { kCallback = 0, kActivity = 1 };
enum class ActivityType {
  kKernel = 0,
  kMemcpyH2D = 1,
  kMemcpyD2H = 2,
  kMemcpyH2A = 3,
  kMemcpyA2H = 4,
  kMemcpyA2D = 5,
  kMemcpyD2A = 6,
  kMemcpyD2D = 7,
  kMemcpyP2P = 8,
  kMemcpyH2H = 9,
  kMemset = 10,
  kMemcpyUnknown = 11
};

struct MemcpyInfo {
  size_t bytes;
  unsigned char src_kind;
  unsigned char dst_kind;
};

struct KernelInfo {
  uint64_t registers_per_thread;
  uint64_t static_shared_memory;
  uint64_t dynamic_shared_memory;
  uint64_t block_x;
  uint64_t block_y;
  uint64_t block_z;
  uint64_t grid_x;
  uint64_t grid_y;
  uint64_t grid_z;
};

struct Event {
  std::string kernel_name;
  std::string kernel_type;
  CUPTIApiType api_type;
  ActivityType activity_type;
  uint64_t start_time_stamp;
  uint64_t end_time_stamp;
  std::string op_name;
  uint32_t device_id;
  uint32_t correlation_id;
  uint32_t thread_id;
  uint32_t context_id;
  CUpti_CallbackId cb_id;
  union {
    MemcpyInfo memcpy_info;
    KernelInfo kernel_info;
  };
};

const float kTimeUnit = 1000;

std::unordered_map<uint32_t, std::string> op_name_map;
std::vector<Event> events;
std::mutex event_mutex;

std::vector<CUpti_ActivityKind> activities_enable;

uint64_t cupti_callback_events_count = 0l;
uint64_t cupti_callback_events_drop_count = 0l;
const uint64_t max_cupti_callback_events = 2 * 1024 * 10000;

uint64_t cupti_activity_events_count = 0l;
uint64_t cupti_activity_events_drop_count = 0l;
const uint64_t max_cupti_activity_events = 2 * 1024 * 10000;

CUpti_SubscriberHandle subscriber = nullptr;

}  // namespace akg

#endif  // PROFILER_GPU_PROFILING_H_
