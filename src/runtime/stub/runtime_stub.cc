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

#ifdef USE_CCE_RT_STUB

#include <cce/dnn.h>
#include <dmlc/logging.h>

#include <iostream>

#include "prof_mgr_core.h"
#define EVENT_LENTH 10

#define FUNC_ENTRY LOG(INFO) << "Run in func " << __FUNCTION__;

void *ProfMgrStartUp(const ProfMgrCfg *cfg) {
  return reinterpret_cast<void *>(0xffffff);
}

int ProfMgrStop(void *handle) {
  return 0;
}

rtError_t rtEventCreate(rtEvent_t *event) {
  *event = new (std::nothrow) int[EVENT_LENTH];
  if (*event == nullptr) {
    return RT_ERROR_MEMORY_ALLOCATION;
  }
  return RT_ERROR_NONE;
}

rtError_t rtMalloc(void **devPtr, uint64_t size, rtMemType_t type) {
  FUNC_ENTRY
  CHECK_GT(size, 0);
  *devPtr = new (std::nothrow) uint8_t[size];
  if (*devPtr == nullptr) {
    return RT_ERROR_MEMORY_ALLOCATION;
  }
  return RT_ERROR_NONE;
}

rtError_t rtFree(void *devPtr) {
  FUNC_ENTRY
  delete[] reinterpret_cast<uint8_t *>(devPtr);
  return RT_ERROR_NONE;
}

rtError_t rtStreamCreate(rtStream_t *stream, int32_t priority) {
  *stream = new (std::nothrow) uint32_t;
  if (*stream == nullptr) {
    return RT_ERROR_MEMORY_ALLOCATION;
  }
  return RT_ERROR_NONE;
}

rtError_t rtStreamDestroy(rtStream_t stream) {
  delete reinterpret_cast<uint32_t *>(stream);
  return RT_ERROR_NONE;
}

rtError_t rtSetDevice(int32_t device) {
  FUNC_ENTRY
  return RT_ERROR_NONE;
}

rtError_t rtStreamSynchronize(rtStream_t stream) {
  FUNC_ENTRY
  return RT_ERROR_NONE;
}

rtError_t rtMemcpy(void *dst, uint64_t destMax, const void *src, uint64_t count, rtMemcpyKind_t kind) {
  FUNC_ENTRY
  return RT_ERROR_NONE;
}

rtError_t rtMemcpyAsync(void *dst, uint64_t destMax, const void *src, uint64_t count, rtMemcpyKind_t kind,
                        rtStream_t stream) {
  FUNC_ENTRY
  return RT_ERROR_NONE;
}

rtError_t rtStreamWaitEvent(rtStream_t stream, rtEvent_t event) { return RT_ERROR_NONE; }

rtError_t rtGetDeviceCount(int32_t *count) {
  *count = 1;
  return RT_ERROR_NONE;
}

rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **handle) {
  FUNC_ENTRY
  return RT_ERROR_NONE;
}

rtError_t rtDevBinaryUnRegister(void *handle) {
  FUNC_ENTRY
  return RT_ERROR_NONE;
}

rtError_t rtFunctionRegister(void *binHandle, const void *stubFunc, const char *stubName, const void *devFunc,
                             uint32_t funcMode) {
  FUNC_ENTRY
  return RT_ERROR_NONE;
}

rtError_t rtKernelLaunch(const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize, rtL2Ctrl_t *l2ctrl,
                         rtStream_t stream) {
  FUNC_ENTRY
  return RT_ERROR_NONE;
}

rtError_t rtGetDevice(int32_t *device) {
  FUNC_ENTRY
  *device = 0;
  return RT_ERROR_NONE;
}

#endif
