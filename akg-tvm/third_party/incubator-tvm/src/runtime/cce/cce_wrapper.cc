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
  if (cce_handle_ != nullptr) {
    if (dlclose(cce_handle_) != 0) {
      return false;
    }
  }
  cce_handle_ = nullptr;
  return true;
}

bool CceWrapper::LoadLibraries() {
  std::string library_path = "libruntime.so";
  void *handle_ptr = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle_ptr == nullptr) {
    LOG(ERROR) << "load library " << library_path << " failed!";
    return false;
  }
  cce_handle_ = handle_ptr;

  LOAD_FUNCTION_PTR(rtGetDevice);
  LOAD_FUNCTION_PTR(rtGetDeviceCount);
  LOAD_FUNCTION_PTR(rtSetDevice);
  LOAD_FUNCTION_PTR(rtDeviceReset);
  LOAD_FUNCTION_PTR(rtCtxCreate);
  LOAD_FUNCTION_PTR(rtCtxDestroy);
  LOAD_FUNCTION_PTR(rtCtxGetCurrent);
  LOAD_FUNCTION_PTR(rtCtxSetCurrent);
  LOAD_FUNCTION_PTR(rtCtxSynchronize);
  LOAD_FUNCTION_PTR(rtMemGetInfoEx);
  LOAD_FUNCTION_PTR(rtEventCreate);
  LOAD_FUNCTION_PTR(rtStreamCreate);
  LOAD_FUNCTION_PTR(rtStreamCreateWithFlags);
  LOAD_FUNCTION_PTR(rtStreamDestroy);
  LOAD_FUNCTION_PTR(rtStreamSynchronize);
  LOAD_FUNCTION_PTR(rtStreamWaitEvent);
  LOAD_FUNCTION_PTR(rtMalloc);
  LOAD_FUNCTION_PTR(rtFree);
  LOAD_FUNCTION_PTR(rtMemcpy);
  LOAD_FUNCTION_PTR(rtMemcpyAsync);
  LOAD_FUNCTION_PTR(rtDevBinaryRegister);
  LOAD_FUNCTION_PTR(rtDevBinaryUnRegister);
  LOAD_FUNCTION_PTR(rtFunctionRegister);
  LOAD_FUNCTION_PTR(rtKernelLaunch);

  return true;
}

}  // namespace runtime
}  // namespace air


rtError_t rtGetDevice(int32_t *device) {
  auto func = air::runtime::CceWrapper::GetInstance()->rtGetDevice;
  CHECK_NOTNULL(func);
  return func(device);
}

rtError_t rtGetDeviceCount(int32_t *count) {
  auto func = air::runtime::CceWrapper::GetInstance()->rtGetDeviceCount;
  CHECK_NOTNULL(func);
  return func(count);
}

rtError_t rtSetDevice(int32_t device) {
  auto func = air::runtime::CceWrapper::GetInstance()->rtSetDevice;
  CHECK_NOTNULL(func);
  return func(device);
}

rtError_t rtDeviceReset(int32_t device) {
  auto func = air::runtime::CceWrapper::GetInstance()->rtDeviceReset;
  CHECK_NOTNULL(func);
  return func(device);
}

rtError_t rtCtxCreate(rtContext_t *ctx, uint32_t flags, int32_t dev) {
  auto func = air::runtime::CceWrapper::GetInstance()->rtCtxCreate;
  CHECK_NOTNULL(func);
  return func(ctx, flags, dev);
}

rtError_t rtCtxDestroy(rtContext_t ctx) {
  auto func = air::runtime::CceWrapper::GetInstance()->rtCtxDestroy;
  CHECK_NOTNULL(func);
  return func(ctx);
}

rtError_t rtCtxGetCurrent(rtContext_t *ctx) {
  auto func = air::runtime::CceWrapper::GetInstance()->rtCtxGetCurrent;
  CHECK_NOTNULL(func);
  return func(ctx);
}

rtError_t rtCtxSetCurrent(rtContext_t ctx) {
  auto func = air::runtime::CceWrapper::GetInstance()->rtCtxSetCurrent;
  CHECK_NOTNULL(func);
  return func(ctx);
}

rtError_t rtCtxSynchronize(void) {
  auto func = air::runtime::CceWrapper::GetInstance()->rtCtxSynchronize;
  CHECK_NOTNULL(func);
  return func();
}

rtError_t rtMemGetInfoEx(rtMemInfoType_t info_type, size_t *free_size, size_t *total_size) {
  auto func = air::runtime::CceWrapper::GetInstance()->rtMemGetInfoEx;
  CHECK_NOTNULL(func);
  return func(info_type, free_size, total_size);
}

rtError_t rtEventCreate(rtEvent_t *event) {
  auto func = air::runtime::CceWrapper::GetInstance()->rtEventCreate;
  CHECK_NOTNULL(func);
  return func(event);
}

rtError_t rtStreamCreate(rtStream_t *stream, int32_t priority) {
  auto func = air::runtime::CceWrapper::GetInstance()->rtStreamCreate;
  CHECK_NOTNULL(func);
  return func(stream, priority);
}

rtError_t rtStreamCreateWithFlags(rtStream_t *stm, int32_t priority, uint32_t flags) {
  auto func = air::runtime::CceWrapper::GetInstance()->rtStreamCreateWithFlags;
  CHECK_NOTNULL(func);
  return func(stm, priority, flags);
}

rtError_t rtStreamDestroy(rtStream_t stream) {
  auto func = air::runtime::CceWrapper::GetInstance()->rtStreamDestroy;
  CHECK_NOTNULL(func);
  return func(stream);
}

rtError_t rtStreamSynchronize(rtStream_t stream) {
  auto func = air::runtime::CceWrapper::GetInstance()->rtStreamSynchronize;
  CHECK_NOTNULL(func);
  return func(stream);
}

rtError_t rtStreamWaitEvent(rtStream_t stream, rtEvent_t event) {
  auto func = air::runtime::CceWrapper::GetInstance()->rtStreamWaitEvent;
  CHECK_NOTNULL(func);
  return func(stream, event);
}

rtError_t rtMalloc(void **dev_ptr, uint64_t size, rtMemType_t type, uint16_t moduleId) {
  auto func = air::runtime::CceWrapper::GetInstance()->rtMalloc;
  CHECK_NOTNULL(func);
  return func(dev_ptr, size, type, moduleId);
}

rtError_t rtFree(void *dev_ptr) {
  auto func = air::runtime::CceWrapper::GetInstance()->rtFree;
  CHECK_NOTNULL(func);
  return func(dev_ptr);
}

rtError_t rtMemcpy(void *dst, uint64_t dest_max, const void *src, uint64_t count, rtMemcpyKind_t kind) {
  auto func = air::runtime::CceWrapper::GetInstance()->rtMemcpy;
  CHECK_NOTNULL(func);
  return func(dst, dest_max, src, count, kind);
}

rtError_t rtMemcpyAsync(void *dst, uint64_t dest_max, const void *src, uint64_t count,
                        rtMemcpyKind_t kind, rtStream_t stream) {
  auto func = air::runtime::CceWrapper::GetInstance()->rtMemcpyAsync;
  CHECK_NOTNULL(func);
  return func(dst, dest_max, src, count, kind, stream);
}

rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **handle) {
  auto func = air::runtime::CceWrapper::GetInstance()->rtDevBinaryRegister;
  CHECK_NOTNULL(func);
  return func(bin, handle);
}

rtError_t rtDevBinaryUnRegister(void *handle) {
  auto func = air::runtime::CceWrapper::GetInstance()->rtDevBinaryUnRegister;
  CHECK_NOTNULL(func);
  return func(handle);
}

rtError_t rtFunctionRegister(void *handle, const void *stub_func, const char *stub_name,
                             const void *dev_func, uint32_t func_mode) {
  auto func = air::runtime::CceWrapper::GetInstance()->rtFunctionRegister;
  CHECK_NOTNULL(func);
  return func(handle, stub_func, stub_name, dev_func, func_mode);
}

rtError_t rtKernelLaunch(const void *stub_func, uint32_t block_dim, void *args, uint32_t args_size,
                         rtL2Ctrl_t *l2ctrl, rtStream_t stream) {
  auto func = air::runtime::CceWrapper::GetInstance()->rtKernelLaunch;
  CHECK_NOTNULL(func);
  return func(stub_func, block_dim, args, args_size, l2ctrl, stream);
}
