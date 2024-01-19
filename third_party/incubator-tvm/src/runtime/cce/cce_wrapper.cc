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
 *  \file cce_wrapper.cc
 *  \brief cce symbols wrapper
 */

/*!
 * 2023.4.21 - Add file cce_wrapper.cc.
 * 2024.1.24 - Change rt*** to aclrt***.
 */

#include "cce_wrapper.h"
#include <dlfcn.h>
#include <mutex>

namespace air {
namespace runtime {

std::shared_ptr<CceWrapper> CceWrapper::cce_wrapper_singleton_ = nullptr;

CceWrapper* CceWrapper::GetInstance() {
  static std::once_flag cce_symbol_once;
  std::call_once(cce_symbol_once, []() { cce_wrapper_singleton_.reset(new CceWrapper()); });
  return cce_wrapper_singleton_.get();
}

CceWrapper::CceWrapper() { LoadLibraries(); }

CceWrapper::~CceWrapper() {
  if (cce_wrapper_singleton_.get() == nullptr) {
    return;
  }
  cce_wrapper_singleton_->UnLoadLibraries();
}

bool CceWrapper::UnLoadLibraries() {
  if (ascendcl_handle_ != nullptr) {
    if (dlclose(ascendcl_handle_) != 0) {
      return false;
    }
  }
  ascendcl_handle_ = nullptr;

  if (runtime_handle_ != nullptr) {
    if (dlclose(runtime_handle_) != 0) {
      return false;
    }
  }
  runtime_handle_ = nullptr;
  return true;
}

bool CceWrapper::LoadLibraries() {
  LoadAscendCL();
  LoadRuntime();
  return true;
}

bool CceWrapper::LoadAscendCL() {
  std::string library_path = "libascendcl.so";
  void *handle_ptr = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle_ptr == nullptr) {
    LOG(ERROR) << "load library " << library_path << " failed!";
    return false;
  }
  ascendcl_handle_ = handle_ptr;

  // aclrt
  LOAD_FUNCTION_PTR(aclrtSetDevice);
  LOAD_FUNCTION_PTR(aclrtCreateContext);
  LOAD_FUNCTION_PTR(aclrtCreateStream);
  LOAD_FUNCTION_PTR(aclrtMallocHost);
  LOAD_FUNCTION_PTR(aclrtMalloc);
  LOAD_FUNCTION_PTR(aclrtMemcpy);
  LOAD_FUNCTION_PTR(aclrtSynchronizeStream);
  LOAD_FUNCTION_PTR(aclrtFree);
  LOAD_FUNCTION_PTR(aclrtFreeHost);
  LOAD_FUNCTION_PTR(aclrtDestroyStream);
  LOAD_FUNCTION_PTR(aclrtDestroyContext);
  LOAD_FUNCTION_PTR(aclrtResetDevice);
  LOAD_FUNCTION_PTR(aclrtSetCurrentContext);
  LOAD_FUNCTION_PTR(aclrtGetDeviceCount);
  LOAD_FUNCTION_PTR(aclrtGetCurrentContext);
  LOAD_FUNCTION_PTR(aclrtCreateStreamWithConfig);
  LOAD_FUNCTION_PTR(aclrtMemcpyAsync);
  LOAD_FUNCTION_PTR(aclrtGetMemInfo);
  LOAD_FUNCTION_PTR(aclrtGetDevice);

  return true;
}

bool CceWrapper::LoadRuntime() {
  std::string library_path = "libruntime.so";
  void *handle_ptr = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle_ptr == nullptr) {
    LOG(ERROR) << "load library " << library_path << " failed!";
    return false;
  }
  runtime_handle_ = handle_ptr;

  // rt
  LOAD_FUNCTION_PTR(rtGetTaskIdAndStreamID);

  return true;
}

}  // namespace runtime
}  // namespace air

aclError aclrtSetCurrentContext(aclrtContext context) {
  auto func = air::runtime::CceWrapper::GetInstance()->aclrtSetCurrentContext;
  CHECK_NOTNULL(func);
  return func(context);
}

aclError aclrtGetDeviceCount(uint32_t *count) {
  auto func = air::runtime::CceWrapper::GetInstance()->aclrtGetDeviceCount;
  CHECK_NOTNULL(func);
  return func(count);
}

aclError aclrtGetCurrentContext(aclrtContext *context) {
  auto func = air::runtime::CceWrapper::GetInstance()->aclrtGetCurrentContext;
  CHECK_NOTNULL(func);
  return func(context);
}

aclError aclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t priority, uint32_t flag) {
  auto func = air::runtime::CceWrapper::GetInstance()->aclrtCreateStreamWithConfig;
  CHECK_NOTNULL(func);
  return func(stream, priority, flag);
}

aclError aclrtMemcpyAsync(void *dst, size_t destMax, const void *src, size_t count,
                 aclrtMemcpyKind kind, aclrtStream stream) {
  auto func = air::runtime::CceWrapper::GetInstance()->aclrtMemcpyAsync;
  CHECK_NOTNULL(func);
  return func(dst, destMax,  src, count, kind, stream);
}

aclError aclrtGetMemInfo(aclrtMemAttr attr, size_t *free, size_t *total) {
  auto func = air::runtime::CceWrapper::GetInstance()->aclrtGetMemInfo;
  CHECK_NOTNULL(func);
  return func(attr, free, total);
}

aclError aclrtSetDevice(int32_t deviceId) {
  auto func = air::runtime::CceWrapper::GetInstance()->aclrtSetDevice;
  CHECK_NOTNULL(func);
  return func(deviceId);
}

aclError aclrtCreateContext(aclrtContext *context, int32_t deviceId) {
  auto func = air::runtime::CceWrapper::GetInstance()->aclrtCreateContext;
  CHECK_NOTNULL(func);
  return func(context, deviceId);
}


aclError aclrtCreateStream(aclrtStream *stream) {
  auto func = air::runtime::CceWrapper::GetInstance()->aclrtCreateStream;
  CHECK_NOTNULL(func);
  return func(stream);
}

aclError aclrtMallocHost(void **hostPtr, size_t size) {
  auto func = air::runtime::CceWrapper::GetInstance()->aclrtMallocHost;
  CHECK_NOTNULL(func);
  return func(hostPtr, size);
}

aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy) {
  auto func = air::runtime::CceWrapper::GetInstance()->aclrtMalloc;
  CHECK_NOTNULL(func);
  return func(devPtr, size, policy);
}

aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind) {
  auto func = air::runtime::CceWrapper::GetInstance()->aclrtMemcpy;
  CHECK_NOTNULL(func);
  return func(dst, destMax, src, count, kind);
}

aclError aclrtSynchronizeStream(aclrtStream stream) {
  auto func = air::runtime::CceWrapper::GetInstance()->aclrtSynchronizeStream;
  CHECK_NOTNULL(func);
  return func(stream);
}

aclError aclrtFree(void *devPtr) {
  auto func = air::runtime::CceWrapper::GetInstance()->aclrtFree;
  CHECK_NOTNULL(func);
  return func(devPtr);
}

aclError aclrtFreeHost(void *hostPtr) {
  auto func = air::runtime::CceWrapper::GetInstance()->aclrtFreeHost;
  CHECK_NOTNULL(func);
  return func(hostPtr);
}

aclError aclrtDestroyStream(aclrtStream stream) {
  auto func = air::runtime::CceWrapper::GetInstance()->aclrtDestroyStream;
  CHECK_NOTNULL(func);
  return func(stream);
}

aclError aclrtDestroyContext(aclrtContext context) {
  auto func = air::runtime::CceWrapper::GetInstance()->aclrtDestroyContext;
  CHECK_NOTNULL(func);
  return func(context);
}

aclError aclrtResetDevice(int32_t deviceId) {
  auto func = air::runtime::CceWrapper::GetInstance()->aclrtResetDevice;
  CHECK_NOTNULL(func);
  return func(deviceId);
}

aclError aclrtGetDevice(int32_t *deviceId) {
  auto func = air::runtime::CceWrapper::GetInstance()->aclrtGetDevice;
  CHECK_NOTNULL(func);
  return func(deviceId);
}

rtError_t rtGetTaskIdAndStreamID(uint32_t *taskId, uint32_t *streamId) {
  auto func = air::runtime::CceWrapper::GetInstance()->rtGetTaskIdAndStreamID;
  CHECK_NOTNULL(func);
  return func(taskId, streamId);
}