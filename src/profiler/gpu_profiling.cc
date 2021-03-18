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

#include <cxxabi.h>
#include <cmath>
#include <chrono>
#include "gpu_profiling.h"
#include "cupti_interface.h"

namespace akg {
#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define CHECK_CUPTI_RET_WITH_ERROR(expression, message)                   \
  if (expression != CUPTI_SUCCESS) {                                      \
    const char *errstr;                                                   \
    CuptiGetResultString(expression, &errstr);                            \
    LOG(WARNING) << "CUPTI Error:" << errstr << " function:" << message; \
  }

#define CHECK_CUPTI_RET_WITH_EXCEPT(expression, message)                      \
  if (expression != CUPTI_SUCCESS) {                                          \
    const char *errstr;                                                       \
    CuptiGetResultString(expression, &errstr);                                \
    LOG(ERROR) << "CUPTI Error:" << errstr << " function:" << message; \
  }

#define PROFILER_ERROR_IF_NULLPTR(ptr)                           \
  do {                                                           \
    if ((ptr) == nullptr) {                                      \
      LOG(WARNING) << ": The pointer[" << #ptr << "] is null."; \
      return;                                                    \
    }                                                            \
  } while (0)

int32_t GetThreadID() {
  uint32_t thread_id = 0;
  thread_id = static_cast<uint32_t>(pthread_self());
  return thread_id;
}

uint64_t GetCUPTITimeStamp() {
  uint64_t time_stamp = 0l;
  CHECK_CUPTI_RET_WITH_ERROR(CuptiGetTimestamp(&time_stamp), "CuptiGetTimestamp");
  return time_stamp;
}

uint64_t GetHostTimeStamp() {
  auto cur_sys_clock = std::chrono::system_clock::now();
  uint64_t cur_time_stamp =
    std::chrono::duration_cast<std::chrono::nanoseconds>(cur_sys_clock.time_since_epoch()).count();
  return cur_time_stamp;
}

std::string GetKernelFunc(const char *name) {
  char *demangledName = abi::__cxa_demangle(name, nullptr, nullptr, nullptr);
  if (demangledName != nullptr) {
    return demangledName;
  } else {
    return name;
  }
}
void EventHandleProcess(CUpti_CallbackId cbid, const CUpti_CallbackData *cbdata,
                                     const std::string &typestring, uint64_t startTimestamp, uint64_t endTimestamp);
void CUPTICallBackFunc(void *user_data, CUpti_CallbackDomain domain, CUpti_CallbackId cb_id,
                       const CUpti_CallbackData *cb_data) {
  if (domain != CUPTI_CB_DOMAIN_DRIVER_API) {
    return;
  }
  PROFILER_ERROR_IF_NULLPTR(cb_data);
  if (cb_data->context == nullptr) {
    return;
  }

  uint64_t start_timestamp;
  uint64_t end_timestamp;

  if (cb_data->callbackSite == CUPTI_API_ENTER) {
    *cb_data->correlationData = GetCUPTITimeStamp();

  } else if (cb_data->callbackSite == CUPTI_API_EXIT) {
    start_timestamp = *cb_data->correlationData;
    end_timestamp = GetCUPTITimeStamp();

    switch (cb_id) {
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel:
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice:
        EventHandleProcess(cb_id, cb_data, "cuLaunchKernel", start_timestamp, end_timestamp);
        break;
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoH_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoD_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoA_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoA_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2D_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DUnaligned_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy3D_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoA_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoHAsync_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DAsync_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoAAsync_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeer:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeerAsync:
        EventHandleProcess(cb_id, cb_data, "cuMemcpy", start_timestamp, end_timestamp);
        break;
      case CUPTI_DRIVER_TRACE_CBID_cuMemAlloc:
      case CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2:
        EventHandleProcess(cb_id, cb_data, "cuMemAlloc", start_timestamp, end_timestamp);
        break;
      case CUPTI_DRIVER_TRACE_CBID_cuEventCreate:
      case CUPTI_DRIVER_TRACE_CBID_cuEventDestroy_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuEventRecord:
      case CUPTI_DRIVER_TRACE_CBID_cuEventSynchronize:
      case CUPTI_DRIVER_TRACE_CBID_cuEventElapsedTime:
      // In some cases, the callback of cuctxsetcurrent is only exist
      // without entry, so this callback is ignored
      case CUPTI_DRIVER_TRACE_CBID_cuCtxSetCurrent:
        break;
      default:
        EventHandleProcess(cb_id, cb_data, "others_api", start_timestamp, end_timestamp);
        break;
    }
  }
}

void FixOpNameByCorrelationId(Event *event) {
  PROFILER_ERROR_IF_NULLPTR(event);
  if (event->api_type != CUPTIApiType::kActivity) {
    return;
  }
  auto iter = op_name_map.find(event->correlation_id);
  if (iter != op_name_map.end()) {
    event->op_name = std::move(iter->second);
  }
}

void AddEvent(Event &&event) {
  // protect callback concurrency for driver api and activity
  std::unique_lock<std::mutex> lock(event_mutex);
  switch (event.api_type) {
    case CUPTIApiType::kCallback: {
      if (cupti_callback_events_count < max_cupti_callback_events) {
        events.emplace_back(std::move(event));
        cupti_callback_events_count++;
      } else {
        cupti_callback_events_drop_count++;
      }
      break;
    }
    case CUPTIApiType::kActivity: {
      if (cupti_activity_events_count < max_cupti_activity_events) {
        events.emplace_back(std::move(event));
        cupti_activity_events_count++;
      } else {
        cupti_activity_events_drop_count++;
      }
      break;
    }
    default:
      break;
  }
}

void EventLog(const Event &event) {
  LOG(INFO) << "GPUProfiler"
                << ",\"kernel_name:" << event.kernel_name << "\",kernel_type:" << event.kernel_type
                << ",api_type:" << static_cast<int>(event.api_type) << ",start_time_stamp:" << event.start_time_stamp
                << ",end_time_stamp:" << event.end_time_stamp << ",cost:,"
                << (event.end_time_stamp - event.start_time_stamp) / kTimeUnit << ",op_name:" << event.op_name
                << ",device_id:" << event.device_id << ",correlation_id:" << event.correlation_id
                << ",thread_id:" << event.thread_id << ",context_id:" << event.context_id << ",cb_id:" << event.cb_id;
}

float OpsParser() {
  LOG(INFO) << "Count the number of events size:" << events.size()
               << " callback api:" << cupti_callback_events_count << " activity:" << cupti_activity_events_count;

  if (cupti_activity_events_drop_count > 0 || cupti_callback_events_drop_count > 0) {
    LOG(WARNING)
      << "The total number of events exceeded the profiler's processing capacity, Some events were discarded."
      << " callback api events:" << cupti_activity_events_drop_count
      << " activity api events:" << cupti_callback_events_drop_count;
  }

  if (events.size() == 0) {
    return 0;
  }
  int repeat_times = 0;
  std::unordered_map<std::string, uint64_t> op_time;
  for (Event &event : events) {
    if (event.op_name.empty()) {
      FixOpNameByCorrelationId(&event);
    }
    if (event.kernel_type == "cuLaunchKernel" && static_cast<int>(event.api_type) == 1) {
      ++repeat_times;
      if (!op_time.count(event.kernel_name)) {
        op_time[event.kernel_name] = event.end_time_stamp - event.start_time_stamp;
      } else {
        op_time[event.kernel_name] += event.end_time_stamp - event.start_time_stamp;
      }
    }
    EventLog(event);

    if (event.op_name.empty() || event.cb_id == CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize) {
      continue;
    }
  }
  uint64_t profiling_time = 0l;
  for (const auto& op_single_time : op_time) {
    profiling_time += op_single_time.second;
  }
  return profiling_time / (repeat_times * 10) * op_time.size();
}

void EventHandleProcess(CUpti_CallbackId cbid, const CUpti_CallbackData *cbdata,
                                     const std::string &typestring, uint64_t startTimestamp, uint64_t endTimestamp) {
  Event event;
  uint32_t device_id = -1;
  CuptiGetDeviceId(cbdata->context, &device_id);
  event.kernel_name = cbdata->symbolName ? GetKernelFunc(cbdata->symbolName) : cbdata->functionName;
  event.kernel_type = typestring;
  event.api_type = CUPTIApiType::kCallback;
  event.start_time_stamp = startTimestamp;
  event.end_time_stamp = endTimestamp;
  event.op_name = cbdata->functionName;
  event.device_id = device_id;
  event.correlation_id = cbdata->correlationId;
  event.thread_id = GetThreadID();
  event.context_id = cbdata->contextUid;
  event.cb_id = cbid;
  op_name_map[event.correlation_id] = event.op_name;
  AddEvent(std::move(event));
}

void StopCUPTI() {
  if (subscriber != nullptr) {
    CHECK_CUPTI_RET_WITH_ERROR(CuptiUnsubscribe(subscriber), "CuptiUnsubscribe");
    CHECK_CUPTI_RET_WITH_ERROR(CuptiActivityFlushAll(0), "CuptiActivityFlushAll");
    for (std::vector<CUpti_ActivityKind>::iterator it = activities_enable.begin(); it != activities_enable.end();
         ++it) {
      CHECK_CUPTI_RET_WITH_ERROR(CuptiActivityDisable(*it), "CuptiActivityDisable");
    }
    subscriber = nullptr;
  }
}

void CUPTIAPI ActivityAllocBuffer(uint8_t **buffer, size_t *size, size_t *maxNumRecords);

void CUPTIAPI ActivityProcessBuffer(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize);

void GPUProfilerInit() {
  LOG(INFO) << "Initialize GPU Profiling";
  if (subscriber != nullptr) {
    StopCUPTI();
    LOG(ERROR)
      << "Repeated initialization, Please check whether you have created the Profiler object multiple times";
  }
  CHECK_CUPTI_RET_WITH_EXCEPT(CuptiSubscribe(&subscriber, (CUpti_CallbackFunc)CUPTICallBackFunc, nullptr),
                              "CuptiSubscribe");
  CHECK_CUPTI_RET_WITH_EXCEPT(CuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API), "CuptiEnableDomain");

  activities_enable.emplace_back(CUPTI_ACTIVITY_KIND_MEMCPY);
  activities_enable.emplace_back(CUPTI_ACTIVITY_KIND_MEMCPY2);
  activities_enable.emplace_back(CUPTI_ACTIVITY_KIND_KERNEL);

  for (std::vector<CUpti_ActivityKind>::iterator it = activities_enable.begin(); it != activities_enable.end();
       ++it) {
    CHECK_CUPTI_RET_WITH_EXCEPT(CuptiActivityEnable(*it), "CuptiActivityEnable");
  }

  CHECK_CUPTI_RET_WITH_EXCEPT(CuptiActivityRegisterCallbacks(ActivityAllocBuffer, ActivityProcessBuffer),
                              "CuptiActivityRegisterCallbacks");

  auto gpu_start_time = GetCUPTITimeStamp();
  auto host_start_time = GetHostTimeStamp();
  LOG(INFO) << "GPU start time(ns):" << gpu_start_time << " Host start time(ns):" << host_start_time;
}

void ClearInst() {
  op_name_map.clear();
  events.clear();
  activities_enable.clear();
  cupti_callback_events_count = 0l;
  cupti_callback_events_drop_count = 0l;
  cupti_activity_events_count = 0l;
  cupti_activity_events_drop_count = 0l;
}

float GPUProfilerStop() {
  LOG(INFO) << "Stop GPU Profiling";
  StopCUPTI();
  auto ret = OpsParser();
  ClearInst();
  return ret;
}

void CUPTIAPI ActivityAllocBuffer(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
  int stat = posix_memalign(reinterpret_cast<void **>(buffer), ALIGN_SIZE, BUF_SIZE);
  if (stat) {
    LOG(WARNING) << "Out of memory, activity buffer alloc failed.";
    return;
  }
  LOG(INFO) << "Alloc activity buffer, buffer size: " << BUF_SIZE;
  *size = BUF_SIZE;
  *maxNumRecords = 0;
}

void HandleActivityMemcpyRecord(Event *profilingData, CUpti_Activity *record) {
  CUpti_ActivityMemcpy *memcpy = reinterpret_cast<CUpti_ActivityMemcpy *>(record);
  switch (memcpy->copyKind) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      profilingData->activity_type = ActivityType::kMemcpyH2D;
      profilingData->kernel_name = "MemcpyH2D";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      profilingData->activity_type = ActivityType::kMemcpyD2H;
      profilingData->kernel_name = "MemcpyD2H";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
      profilingData->activity_type = ActivityType::kMemcpyH2A;
      profilingData->kernel_name = "MemcpyH2A";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
      profilingData->activity_type = ActivityType::kMemcpyA2H;
      profilingData->kernel_name = "MemcpyA2H";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
      profilingData->activity_type = ActivityType::kMemcpyA2D;
      profilingData->kernel_name = "MemcpyA2D";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
      profilingData->activity_type = ActivityType::kMemcpyD2A;
      profilingData->kernel_name = "MemcpyD2A";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
      profilingData->activity_type = ActivityType::kMemcpyD2D;
      profilingData->kernel_name = "MemcpyD2D";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
      profilingData->activity_type = ActivityType::kMemcpyH2H;
      profilingData->kernel_name = "MemcpyH2H";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
      profilingData->activity_type = ActivityType::kMemcpyP2P;
      profilingData->kernel_name = "MemcpyP2P";
      break;
    default:
      profilingData->activity_type = ActivityType::kMemcpyUnknown;
      profilingData->kernel_name = "MemcpyUnknown";
      break;
  }
  profilingData->kernel_type = "cuMemcpy";
  profilingData->api_type = CUPTIApiType::kActivity;
  profilingData->start_time_stamp = memcpy->start;
  profilingData->end_time_stamp = memcpy->end;
  profilingData->device_id = memcpy->deviceId;
  profilingData->context_id = memcpy->contextId;
  profilingData->correlation_id = memcpy->correlationId;
  profilingData->memcpy_info.bytes = memcpy->bytes;
  profilingData->memcpy_info.src_kind = memcpy->srcKind;
  profilingData->memcpy_info.dst_kind = memcpy->dstKind;
}

void HandleActivityMemcpy2Record(Event *profilingData, CUpti_Activity *record) {
  CUpti_ActivityMemcpy2 *memcpyP2P = reinterpret_cast<CUpti_ActivityMemcpy2 *>(record);
  profilingData->activity_type = ActivityType::kMemcpyP2P;
  profilingData->kernel_name = "MemcpyP2P";
  profilingData->kernel_type = "cuMemcpy";
  profilingData->api_type = CUPTIApiType::kActivity;
  profilingData->start_time_stamp = memcpyP2P->start;
  profilingData->end_time_stamp = memcpyP2P->end;
  profilingData->device_id = memcpyP2P->deviceId;
  profilingData->context_id = memcpyP2P->contextId;
  profilingData->correlation_id = memcpyP2P->correlationId;
  profilingData->memcpy_info.bytes = memcpyP2P->bytes;
  profilingData->memcpy_info.src_kind = memcpyP2P->srcKind;
  profilingData->memcpy_info.dst_kind = memcpyP2P->dstKind;
}

void HandleActivityMemsetRecord(Event *profilingData, CUpti_Activity *record) {
  CUpti_ActivityMemset *memset = reinterpret_cast<CUpti_ActivityMemset *>(record);
  profilingData->activity_type = ActivityType::kMemset;
  profilingData->kernel_name = "MemorySet";
  profilingData->api_type = CUPTIApiType::kActivity;
  profilingData->start_time_stamp = memset->start;
  profilingData->end_time_stamp = memset->end;
  profilingData->device_id = memset->deviceId;
  profilingData->context_id = memset->contextId;
  profilingData->correlation_id = memset->correlationId;
  profilingData->memcpy_info.bytes = memset->bytes;
}

void HandleActivityKernelRecord(Event *profilingData, CUpti_Activity *record) {
  CUpti_ActivityKernel4 *kernel = reinterpret_cast<CUpti_ActivityKernel4 *>(record);
  profilingData->activity_type = ActivityType::kKernel;
  profilingData->api_type = CUPTIApiType::kActivity;
  profilingData->kernel_name = GetKernelFunc(kernel->name);
  profilingData->kernel_type = "cuLaunchKernel";
  profilingData->start_time_stamp = kernel->start;
  profilingData->end_time_stamp = kernel->end;
  profilingData->device_id = kernel->deviceId;
  profilingData->context_id = kernel->contextId;
  profilingData->correlation_id = kernel->correlationId;
  profilingData->kernel_info.registers_per_thread = kernel->registersPerThread;
  profilingData->kernel_info.static_shared_memory = kernel->staticSharedMemory;
  profilingData->kernel_info.dynamic_shared_memory = kernel->dynamicSharedMemory;
  profilingData->kernel_info.block_x = kernel->blockX;
  profilingData->kernel_info.block_y = kernel->blockY;
  profilingData->kernel_info.block_z = kernel->blockZ;
  profilingData->kernel_info.grid_x = kernel->gridX;
  profilingData->kernel_info.grid_y = kernel->gridY;
  profilingData->kernel_info.grid_z = kernel->gridZ;
}

void HandleActivityRecord(CUpti_Activity *record) {
  PROFILER_ERROR_IF_NULLPTR(record);
  Event profilingData;
  profilingData.cb_id = 0;
  switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_MEMCPY: {
      HandleActivityMemcpyRecord(&profilingData, record);
      break;
    }
    case CUPTI_ACTIVITY_KIND_MEMCPY2: {
      HandleActivityMemcpy2Record(&profilingData, record);
      break;
    }
    case CUPTI_ACTIVITY_KIND_MEMSET: {
      HandleActivityMemsetRecord(&profilingData, record);
      break;
    }
    case CUPTI_ACTIVITY_KIND_KERNEL:
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
      HandleActivityKernelRecord(&profilingData, record);
      break;
    }
    default:
      LOG(WARNING) << "Unknown activity type!";
      return;
  }

  AddEvent(std::move(profilingData));
}

void CUPTIAPI ProcessBuffer(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size,
                                         size_t validSize) {
  CUptiResult status;
  CUpti_Activity *record = NULL;
  LOG(INFO) << "Process activity buffer, valid size:" << validSize << ",Stream ID:" << streamId;
  if (validSize > 0) {
    do {
      status = CuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        HandleActivityRecord(record);
      } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
        break;
      } else {
        CHECK_CUPTI_RET_WITH_ERROR(status, "CuptiActivityGetNextRecord");
      }
    } while (1);

    // report any records dropped from the queue
    size_t dropped;
    CHECK_CUPTI_RET_WITH_ERROR(CuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped),
                               "CuptiActivityGetNumDroppedRecords");
    if (dropped != 0) {
      LOG(INFO) << "Dropped " << (unsigned int)dropped << " activity records\n";
    }
  }

  free(buffer);
}

void CUPTIAPI ActivityProcessBuffer(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
  PROFILER_ERROR_IF_NULLPTR(buffer);
  ProcessBuffer(ctx, streamId, buffer, size, validSize);
}

TVM_REGISTER_GLOBAL("GPUProfilerInit").set_body_typed(GPUProfilerInit);
TVM_REGISTER_GLOBAL("GPUProfilerStop").set_body_typed(GPUProfilerStop);

}  // namespace akg
