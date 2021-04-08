/*!
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  \file cce_device_api.cc
 *  \brief cce specific API
 */

/*!
 * 2019.12.30 - Add file cce_device_api.cc.
 */

#include <tvm/runtime/device_api.h>

#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <runtime/rt.h>
#include <tvm/runtime/registry.h>
#include "runtime/cce/cce_common.h"
#include "prof_mgr_core.h"

namespace air {
namespace runtime {
class CceDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(TVMContext ctx) final { CCE_CALL(rtSetDevice(ctx.device_id)); }

  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final {
    switch (kind) {
      case kExist: {
        *rv = 1;
        break;
      }
      case kMaxThreadsPerBlock: {
        LOG(FATAL) << "Cce runtime unsupport kMaxThreadsPerBlock attribute";
      }
      case kWarpSize: {
        LOG(FATAL) << "Cce runtime unsupport kWarpSize attribute";
      }
      case kMaxSharedMemoryPerBlock: {
        LOG(FATAL) << "Cce runtime unsupport kMaxSharedMemoryPerBlock attribute";
      }
      case kComputeVersion: {
        LOG(FATAL) << "Cce runtime unsupport kComputeVersion attribute";
      }
      case kDeviceName: {
        LOG(FATAL) << "Cce runtime unsupport kDeviceName attribute";
      }
      case kMaxClockRate: {
        LOG(FATAL) << "Cce runtime unsupport kMaxClockRate attribute";
      }
      case kMultiProcessorCount: {
        LOG(FATAL) << "Cce runtime unsupport kMultiProcessorCount attribute";
      }
      case kMaxThreadDimensions: {
        LOG(FATAL) << "Cce runtime unsupport kMaxThreadDimensions attribute";
      }
      case kGcnArch: {
        LOG(FATAL) << "Cce runtime unsupport kGcnArch attribute";
      }
    }
  }

  void* AllocDataSpace(TVMContext ctx, size_t size, size_t alignment, TVMType type_hint) final {
    void* ptr = nullptr;

    // alignment check here
    CCE_CALL(rtSetDevice(ctx.device_id));
    CCE_CALL(rtMalloc(&ptr, size + 32, RT_MEMORY_HBM));

    return ptr;
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
    if (ptr != nullptr) {
      CCE_CALL(rtSetDevice(ctx.device_id));
      CCE_CALL(rtFree(ptr));
    }
  }

  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t num_bytes,
                      TVMContext ctx_from, TVMContext ctx_to, TVMType type_hint, TVMStreamHandle stream) final {
    LOG(INFO) << " from " << from << " to " << to << " ctx_from " << ctx_from;
    auto cce_stream = static_cast<rtStream_t>(stream);
    from = static_cast<const char*>(from) + from_offset;
    to = static_cast<char*>(to) + to_offset;

    if (ctx_from.device_type == kDLCce && ctx_to.device_type == kDLCce) {
      CCE_CALL(rtSetDevice(ctx_from.device_id));
      if (ctx_from.device_id == ctx_to.device_id) {
        CceCopy(from, to, num_bytes, RT_MEMCPY_DEVICE_TO_DEVICE, cce_stream);
      } else {
        LOG(FATAL) << "expect the same device id copy between Cce";
      }
    } else if (ctx_from.device_type == kDLCce && ctx_to.device_type == kDLCPU) {
      CCE_CALL(rtSetDevice(ctx_from.device_id));
      CceCopy(from, to, num_bytes, RT_MEMCPY_DEVICE_TO_HOST, cce_stream);
    } else if (ctx_from.device_type == kDLCPU && ctx_to.device_type == kDLCce) {
      CCE_CALL(rtSetDevice(ctx_to.device_id));
      CceCopy(from, to, num_bytes, RT_MEMCPY_HOST_TO_DEVICE, cce_stream);
    } else {
      LOG(FATAL) << "expect copy from/to Cce or between Cce";
    }
  }

  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {
    auto cce_stream = static_cast<rtStream_t>(stream);

    CCE_CALL(rtSetDevice(ctx.device_id));
    CCE_CALL(rtStreamSynchronize(cce_stream));
  }

  void SetStream(TVMContext ctx, TVMStreamHandle stream) final {
    CceThreadEntry::ThreadLocal()->stream = static_cast<rtStream_t>(stream);
  }

  void* AllocWorkspace(TVMContext ctx, size_t size, TVMType type_hint = {}) final {
    return CceThreadEntry::ThreadLocal()->pool.AllocWorkspace(ctx, size);
  }

  void FreeWorkspace(TVMContext ctx, void* data) final { CceThreadEntry::ThreadLocal()->pool.FreeWorkspace(ctx, data); }

  static const std::shared_ptr<CceDeviceAPI>& Global() {
    static std::shared_ptr<CceDeviceAPI> inst = std::make_shared<CceDeviceAPI>();
    return inst;
  }

 private:
  static void CceCopy(const void* from, void* to, size_t num_bytes, rtMemcpyKind_t kind, rtStream_t stream) {
    if (stream != RT_STREAM_DEFAULT) {
#ifdef USE_CCE_RT
      CCE_CALL(rtMemcpyAsync(to, num_bytes + 1, const_cast<void*>(from), num_bytes, kind, stream));
#else
      CCE_CALL(rtMemcpyAsync(to, const_cast<void*>(from), num_bytes, kind, stream));
#endif
    } else {
#ifdef USE_CCE_RT
#ifdef USE_KC_AIR
      CCE_CALL(rtMemcpy(to, num_bytes, from, num_bytes, kind));
#else

      // because cce runtime cannot support large size memcpy when des/src is alloc by general
      // malloc(not page-pinned), so we copy with large size by small blocks.
      size_t block_size = 1024 * 1024 * 1024;
      for (size_t i = 0; i < num_bytes / block_size; i++) {
        CCE_CALL(rtMemcpy(to, block_size, const_cast<void*>(from), block_size, kind));
        from = reinterpret_cast<char*>(const_cast<void*>(from)) + block_size;
        to = reinterpret_cast<char*>(to) + block_size;
      }
      size_t remain = num_bytes % block_size;
      if (remain > 0) {
        CCE_CALL(rtMemcpy(to, remain, const_cast<void*>(from), remain, kind));
      }
#endif
#else
      CCE_CALL(rtMemcpy(to, const_cast<void*>(from), num_bytes, kind));
#endif
    }
  }
};

typedef dmlc::ThreadLocalStore<CceThreadEntry> CceThreadStore;

CceThreadEntry::CceThreadEntry() : pool(kDLCce, CceDeviceAPI::Global()) {}

CceThreadEntry* CceThreadEntry::ThreadLocal() { return CceThreadStore::Get(); }

TVM_REGISTER_GLOBAL("device_api.cce").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = CceDeviceAPI::Global().get();
  *rv = static_cast<void*>(ptr);
});
}  // namespace runtime
}  // namespace air
