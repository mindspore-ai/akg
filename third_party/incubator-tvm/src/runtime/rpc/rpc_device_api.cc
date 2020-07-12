/*
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
 * \file rpc_device_api.cc
 */

/*
 * 2019.12.30 - Add nullptr check and some LOG print.
 */

#include <dmlc/logging.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/device_api.h>
#include "rpc_session.h"

namespace air {
namespace runtime {

class RPCDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(TVMContext ctx) final {
    GetSess(ctx)->CallRemote(
        RPCCode::kDevSetDevice, ctx);
  }
  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final {
    CHECK(rv != nullptr);
    *rv = GetSess(ctx)->CallRemote(
        RPCCode::kDevGetAttr, ctx, static_cast<int>(kind));
  }
  void* AllocDataSpace(TVMContext ctx,
                       size_t nbytes,
                       size_t alignment,
                       TVMType type_hint) final {
    auto sess = GetSess(ctx);
    void *data = sess->CallRemote(
            RPCCode::kDevAllocData, ctx, nbytes, alignment, type_hint);
    if (data == nullptr) {
      LOG(FATAL) << "remote data space allocation failed: ctx " << ctx << ", alloc " << nbytes << " bytes";
    }
    RemoteSpace* space = new RemoteSpace();
    space->data = data;
    space->sess = std::move(sess);
    return space;
  }
  void FreeDataSpace(TVMContext ctx, void* ptr) final {
    if (ptr != nullptr) {
        RemoteSpace* space = static_cast<RemoteSpace*>(ptr);
        try {
            if (space->data != nullptr) {
                GetSess(ctx)->CallRemote(
                    RPCCode::kDevFreeData, ctx, space->data);
            }
        } catch (const dmlc::Error &e) {
            // fault tolerance to remote close.
        }
        delete space;
    }
    else {
      LOG(INFO) << "Warning: try to FreeDataSpace ctx " << ctx << " with NULL ptr";
    }
  }
  void CopyDataFromTo(const void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t size,
                      TVMContext ctx_from,
                      TVMContext ctx_to,
                      TVMType type_hint,
                      TVMStreamHandle stream) final {
    LOG(INFO)<<"start from "<<ctx_from<<" to "<<ctx_to<<" size "<<size;
    CHECK(from != nullptr);
    CHECK(to != nullptr);
    auto start_time = std::chrono::system_clock::now();
    int from_dev_type = ctx_from.device_type;
    int to_dev_type = ctx_to.device_type;
    if (from_dev_type > kRPCSessMask &&
        to_dev_type > kRPCSessMask) {
      CHECK(ctx_from.device_type == ctx_to.device_type)
          << "Cannot copy across two different remote session";
      GetSess(ctx_from)->CallRemote(
          RPCCode::kCopyAmongRemote,
          static_cast<const RemoteSpace*>(from)->data, from_offset,
          static_cast<const RemoteSpace*>(to)->data, to_offset,
          size,  ctx_from, ctx_to, type_hint, stream);
    } else if (from_dev_type > kRPCSessMask &&
               to_dev_type == kDLCPU) {
      GetSess(ctx_from)->CopyFromRemote(
          static_cast<const RemoteSpace*>(from)->data, from_offset,
          to, to_offset, size, ctx_from, type_hint);
    } else if (from_dev_type == kDLCPU &&
               to_dev_type > kRPCSessMask) {
      GetSess(ctx_to)->CopyToRemote(
          (void*)from, from_offset,  // NOLINT(*)
          static_cast<const RemoteSpace*>(to)->data, to_offset,
          size, ctx_to, type_hint);
    } else {
      LOG(FATAL) << "expect copy from/to remote or between remote";
    }
    auto end_time = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    LOG(INFO)<<"end from "<<ctx_from<<" to "<<ctx_to<<" size "<<size<<" take "<<elapsed_seconds.count()<<" seconds";
  }
  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {
    LOG(INFO)<<"StreamSync start";
    GetSess(ctx)->CallRemote(
        RPCCode::kDevStreamSync, ctx, stream);
    LOG(INFO)<<"StreamSync done";
  }

 private:
  std::shared_ptr<RPCSession> GetSess(TVMContext ctx) {
    int dev_type = ctx.device_type;
    CHECK_GE(dev_type, kRPCSessMask);
    int tbl_index = dev_type / kRPCSessMask -  1;
    return RPCSession::Get(tbl_index);
  }
};

TVM_REGISTER_GLOBAL("device_api.rpc")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    static RPCDeviceAPI inst;
    DeviceAPI* ptr = &inst;
    *rv = static_cast<void*>(ptr);
  });
}  // namespace runtime
}  // namespace air
