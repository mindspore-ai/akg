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
 *  \file cce_wrapper.h
 *  \brief cce symbols wrapper
 */

/*!
 * 2023.4.21 - Add file cce_wrapper.h.
 */

#ifndef TVM_RUNTIME_CCE_CCE_WRAPPER_H_
#define TVM_RUNTIME_CCE_CCE_WRAPPER_H_

#include "../symbols_wrapper.h"
#include <runtime/rt.h>

namespace air {
namespace runtime {


class CceWrapper : public SymbolsWrapper {
 public:
  static CceWrapper *GetInstance();
  CceWrapper(const CceWrapper &) = delete;
  CceWrapper &operator=(const CceWrapper &) = delete;
  ~CceWrapper();
  bool LoadLibraries();
  bool UnLoadLibraries();
 private:
  CceWrapper();
  static std::shared_ptr<CceWrapper> cce_wrapper_singleton_;
  void *cce_handle_{nullptr};

 public:

  using rtGetDeviceFunc = rtError_t (*)(int32_t *);
  using rtGetDeviceCountFunc = rtError_t (*)(int32_t *);
  using rtSetDeviceFunc = rtError_t (*)(int32_t );
  using rtDeviceResetFunc = rtError_t (*)(int32_t);
  using rtCtxCreateFunc = rtError_t (*)(rtContext_t *, uint32_t, int32_t);
  using rtCtxDestroyFunc = rtError_t (*)(rtContext_t);
  using rtCtxGetCurrentFunc = rtError_t (*)(rtContext_t *);
  using rtCtxSetCurrentFunc = rtError_t (*)(rtContext_t);
  using rtCtxSynchronizeFunc = rtError_t (*)(void);
  using rtMemGetInfoExFunc = rtError_t (*)(rtMemInfoType_t, size_t *, size_t *);
  using rtEventCreateFunc = rtError_t (*)(rtEvent_t *);
  using rtStreamCreateFunc = rtError_t (*)(rtStream_t *, int32_t);
  using rtStreamCreateWithFlagsFunc = rtError_t (*)(rtStream_t *, int32_t, uint32_t);
  using rtStreamDestroyFunc = rtError_t (*)(rtStream_t);
  using rtStreamSynchronizeFunc = rtError_t (*)(rtStream_t);
  using rtStreamWaitEventFunc = rtError_t (*)(rtStream_t, rtEvent_t);
  using rtMallocFunc = rtError_t (*)(void **, uint64_t, rtMemType_t);
  using rtFreeFunc = rtError_t (*)(void *);
  using rtMemcpyFunc = rtError_t (*)(void *, uint64_t, const void *, uint64_t, rtMemcpyKind_t);
  using rtMemcpyAsyncFunc = rtError_t (*)(void *, uint64_t, const void *, uint64_t, rtMemcpyKind_t, rtStream_t);
  using rtDevBinaryRegisterFunc = rtError_t (*)(const rtDevBinary_t *, void **);
  using rtDevBinaryUnRegisterFunc = rtError_t (*)(void *);
  using rtFunctionRegisterFunc = rtError_t (*)(void *, const void *, const char *, const void *, uint32_t);
  using rtKernelLaunchFunc = rtError_t (*)(const void *, uint32_t, void *, uint32_t, rtL2Ctrl_t *, rtStream_t);

  DEFINE_FUNC_PTR(rtGetDevice);
  DEFINE_FUNC_PTR(rtGetDeviceCount);
  DEFINE_FUNC_PTR(rtSetDevice);
  DEFINE_FUNC_PTR(rtDeviceReset);
  DEFINE_FUNC_PTR(rtCtxCreate);
  DEFINE_FUNC_PTR(rtCtxDestroy);
  DEFINE_FUNC_PTR(rtCtxGetCurrent);
  DEFINE_FUNC_PTR(rtCtxSetCurrent);
  DEFINE_FUNC_PTR(rtCtxSynchronize);
  DEFINE_FUNC_PTR(rtMemGetInfoEx);
  DEFINE_FUNC_PTR(rtEventCreate);
  DEFINE_FUNC_PTR(rtStreamCreate);
  DEFINE_FUNC_PTR(rtStreamCreateWithFlags);
  DEFINE_FUNC_PTR(rtStreamDestroy);
  DEFINE_FUNC_PTR(rtStreamSynchronize);
  DEFINE_FUNC_PTR(rtStreamWaitEvent);
  DEFINE_FUNC_PTR(rtMalloc);
  DEFINE_FUNC_PTR(rtFree);
  DEFINE_FUNC_PTR(rtMemcpy);
  DEFINE_FUNC_PTR(rtMemcpyAsync);
  DEFINE_FUNC_PTR(rtDevBinaryRegister);
  DEFINE_FUNC_PTR(rtDevBinaryUnRegister);
  DEFINE_FUNC_PTR(rtFunctionRegister);
  DEFINE_FUNC_PTR(rtKernelLaunch);
};

}  // namespace runtime
}  // namespace air

#endif  // TVM_RUNTIME_CCE_CCE_WRAPPER_H_
