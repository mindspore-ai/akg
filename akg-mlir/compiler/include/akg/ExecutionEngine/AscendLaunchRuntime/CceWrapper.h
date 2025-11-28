/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

/*!
 *  \file cce_wrapper.h
 *  \brief cce symbols wrapper
 */

/*!
 * 2023.4.21 - Add file cce_wrapper.h.
 * 2024.1.24 - Change rt*** to aclrt***.
 */

#ifndef COMPILER_INCLUDE_AKG_EXECUTIONENGINE_AKGASCENDLAUNCHRUNTIME_CCEWRAPPER_H_
#define COMPILER_INCLUDE_AKG_EXECUTIONENGINE_AKGASCENDLAUNCHRUNTIME_CCEWRAPPER_H_

#include <memory>
#include "akg/ExecutionEngine/AscendLaunchRuntime/CceAcl.h"
#include "akg/ExecutionEngine/AscendLaunchRuntime/SymbolsWrapper.h"
#include "akg/ExecutionEngine/AscendLaunchRuntime/logger.h"

namespace mlir {
namespace runtime {

class CceWrapper : public SymbolsWrapper {
 public:
  static CceWrapper *GetInstance();
  CceWrapper(const CceWrapper &) = delete;
  CceWrapper &operator=(const CceWrapper &) = delete;
  ~CceWrapper();
  bool LoadLibraries() override;
  bool UnLoadLibraries() override;

 private:
  CceWrapper();
  bool LoadAscendCL();
  bool LoadRuntime();
  static std::shared_ptr<CceWrapper> cce_wrapper_singleton_;
  void *ascendcl_handle_{nullptr};
  void *runtime_handle_{nullptr};

 public:
  using aclrtSetCurrentContextFunc = aclError (*)(aclrtContext);
  using aclrtGetDeviceCountFunc = aclError (*)(uint32_t *);
  using aclrtGetCurrentContextFunc = aclError (*)(aclrtContext *);
  using aclrtCreateStreamWithConfigFunc = aclError (*)(aclrtStream *, uint32_t, uint32_t);
  using aclrtMemcpyAsyncFunc = aclError (*)(void *, size_t, const void *, size_t, aclrtMemcpyKind, aclrtStream);
  using aclrtGetMemInfoFunc = aclError (*)(aclrtMemAttr, size_t *, size_t *);
  using aclrtSetDeviceFunc = aclError (*)(int32_t);
  using aclrtCreateContextFunc = aclError (*)(aclrtContext *, int32_t);
  using aclrtCreateStreamFunc = aclError (*)(aclrtStream *);
  using aclrtMallocHostFunc = aclError (*)(void **, size_t);
  using aclrtMallocFunc = aclError (*)(void **, size_t, aclrtMemMallocPolicy);
  using aclrtMemcpyFunc = aclError (*)(void *, size_t, const void *, size_t, aclrtMemcpyKind);
  using aclrtSynchronizeStreamFunc = aclError (*)(aclrtStream);
  using aclrtFreeFunc = aclError (*)(void *);
  using aclrtFreeHostFunc = aclError (*)(void *);
  using aclrtDestroyStreamFunc = aclError (*)(aclrtStream);
  using aclrtDestroyContextFunc = aclError (*)(aclrtContext);
  using aclrtResetDeviceFunc = aclError (*)(int32_t);
  using aclrtGetDeviceFunc = aclError (*)(int32_t *);
  using aclprofInitFunc = aclError (*)(const char *, size_t);
  using aclprofStartFunc = aclError (*)(const aclprofConfig *);
  using aclprofStopFunc = aclError (*)(const aclprofConfig *);
  using aclprofFinalizeFunc = aclError (*)();
  using aclprofCreateConfigFunc = aclprofConfig *(*)(uint32_t *, uint32_t, aclprofAicoreMetrics,
                                                     const aclprofAicoreEvents *, uint64_t);
  using aclprofDestroyConfigFunc = aclError (*)(const aclprofConfig *);

  using rtGetC2cCtrlAddrFunc = rtError_t (*)(uint64_t *, uint32_t *);
  using rtConfigureCallFunc = rtError_t (*)(uint32_t, rtSmDesc_t *, rtStream_t);
  using rtDevBinaryRegisterFunc = rtError_t (*)(const rtDevBinary_t *, void **);
  using rtDevBinaryUnRegisterFunc = rtError_t (*)(void *);
  using rtFunctionRegisterFunc = rtError_t (*)(void *, const void *, const char_t *, const void *, uint32_t);
  using rtKernelLaunchFunc = rtError_t (*)(const void *, uint32_t, void *, uint32_t, rtSmDesc_t *, rtStream_t);
  using rtLaunchFunc = rtError_t (*)(const void *);
  using rtSetupArgumentFunc = rtError_t (*)(const void *, uint32_t, uint32_t);

  // kernel launch
  DEFINE_FUNC_PTR(aclrtSetCurrentContext);
  DEFINE_FUNC_PTR(aclrtGetDeviceCount);
  DEFINE_FUNC_PTR(aclrtGetCurrentContext);
  DEFINE_FUNC_PTR(aclrtCreateStreamWithConfig);
  DEFINE_FUNC_PTR(aclrtMemcpyAsync);
  DEFINE_FUNC_PTR(aclrtGetMemInfo);
  DEFINE_FUNC_PTR(aclrtSetDevice);
  DEFINE_FUNC_PTR(aclrtCreateContext);
  DEFINE_FUNC_PTR(aclrtCreateStream);
  DEFINE_FUNC_PTR(aclrtMallocHost);
  DEFINE_FUNC_PTR(aclrtMalloc);
  DEFINE_FUNC_PTR(aclrtMemcpy);
  DEFINE_FUNC_PTR(aclrtSynchronizeStream);
  DEFINE_FUNC_PTR(aclrtFree);
  DEFINE_FUNC_PTR(aclrtFreeHost);
  DEFINE_FUNC_PTR(aclrtDestroyStream);
  DEFINE_FUNC_PTR(aclrtDestroyContext);
  DEFINE_FUNC_PTR(aclrtResetDevice);
  DEFINE_FUNC_PTR(aclrtGetDevice);

  // profiling
  DEFINE_FUNC_PTR(aclprofInit);
  DEFINE_FUNC_PTR(aclprofStart);
  DEFINE_FUNC_PTR(aclprofStop);
  DEFINE_FUNC_PTR(aclprofFinalize);
  DEFINE_FUNC_PTR(aclprofCreateConfig);
  DEFINE_FUNC_PTR(aclprofDestroyConfig);

  DEFINE_FUNC_PTR(rtGetC2cCtrlAddr);
  DEFINE_FUNC_PTR(rtConfigureCall);
  DEFINE_FUNC_PTR(rtDevBinaryRegister);
  DEFINE_FUNC_PTR(rtDevBinaryUnRegister);
  DEFINE_FUNC_PTR(rtFunctionRegister);
  DEFINE_FUNC_PTR(rtKernelLaunch);
  DEFINE_FUNC_PTR(rtLaunch);
  DEFINE_FUNC_PTR(rtSetupArgument);
};

}  // namespace runtime
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_EXECUTIONENGINE_AKGASCENDLAUNCHRUNTIME_CCEWRAPPER_H_
