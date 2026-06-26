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

#ifndef AKG_EXECUTIONENGINE_ASCENDLAUNCHRUNTIME_CCEWRAPPER_H_
#define AKG_EXECUTIONENGINE_ASCENDLAUNCHRUNTIME_CCEWRAPPER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
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
  bool IsMsprofAvailable() const;

 private:
  CceWrapper();
  bool LoadAscendCL();
  bool LoadRuntime();
  bool LoadMsprof();
  void *FindMsprofSymbol(const char *symbol);
  static std::shared_ptr<CceWrapper> cce_wrapper_singleton_;
  void *ascendcl_handle_{nullptr};
  void *runtime_handle_{nullptr};
  std::vector<void *> msprof_handles_;

 public:
  using aclrtSetCurrentContextFunc = int (*)(aclrtContext);
  using aclrtGetDeviceCountFunc = int (*)(uint32_t *);
  using aclrtGetCurrentContextFunc = int (*)(aclrtContext *);
  using aclrtCreateStreamWithConfigFunc = int (*)(aclrtStream *, uint32_t, uint32_t);
  using aclrtMemcpyAsyncFunc = int (*)(void *, size_t, const void *, size_t, aclrtMemcpyKind, aclrtStream);
  using aclrtGetMemInfoFunc = int (*)(aclrtMemAttr, size_t *, size_t *);
  using aclrtSetDeviceFunc = int (*)(int32_t);
  using aclrtCreateContextFunc = int (*)(aclrtContext *, int32_t);
  using aclrtCreateStreamFunc = int (*)(aclrtStream *);
  using aclrtMallocHostFunc = int (*)(void **, size_t);
  using aclrtMallocFunc = int (*)(void **, size_t, aclrtMemMallocPolicy);
  using aclrtMemcpyFunc = int (*)(void *, size_t, const void *, size_t, aclrtMemcpyKind);
  using aclrtSynchronizeStreamFunc = int (*)(aclrtStream);
  using aclrtFreeFunc = int (*)(void *);
  using aclrtFreeHostFunc = int (*)(void *);
  using aclrtDestroyStreamFunc = int (*)(aclrtStream);
  using aclrtDestroyContextFunc = int (*)(aclrtContext);
  using aclrtResetDeviceFunc = int (*)(int32_t);
  using aclrtGetDeviceFunc = int (*)(int32_t *);
  using aclprofInitFunc = int (*)(const char *, size_t);
  using aclprofStartFunc = int (*)(const aclprofConfig *);
  using aclprofStopFunc = int (*)(const aclprofConfig *);
  using aclprofFinalizeFunc = int (*)();
  using aclprofCreateConfigFunc = aclprofConfig *(*)(uint32_t *, uint32_t, aclprofAicoreMetrics,
                                                     const aclprofAicoreEvents *, uint64_t);
  using aclprofDestroyConfigFunc = int (*)(const aclprofConfig *);

  using rtGetC2cCtrlAddrFunc = int (*)(uint64_t *, uint32_t *);
  using rtConfigureCallFunc = int (*)(uint32_t, rtSmDesc_t *, rtStream_t);
  using rtDevBinaryRegisterFunc = int (*)(const rtDevBinary_t *, void **);
  using rtDevBinaryUnRegisterFunc = int (*)(void *);
  using rtFunctionRegisterFunc = int (*)(void *, const void *, const char *, const void *, uint32_t);
  using rtKernelLaunchFunc = int (*)(const void *, uint32_t, void *, uint32_t, rtSmDesc_t *, rtStream_t);
  using rtLaunchFunc = int (*)(const void *);
  using rtSetupArgumentFunc = int (*)(const void *, uint32_t, uint32_t);
  using MsprofSysCycleTimeFunc = uint64_t (*)();
  using MsprofGetHashIdFunc = uint64_t (*)(const char *, size_t);
  using MsprofReportApiFunc = int32_t (*)(uint32_t, const void *);
  using MsprofReportCompactInfoFunc = int32_t (*)(uint32_t, const void *, uint32_t);

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

  // msprof report
  DEFINE_FUNC_PTR(MsprofSysCycleTime);
  DEFINE_FUNC_PTR(MsprofGetHashId);
  DEFINE_FUNC_PTR(MsprofReportApi);
  DEFINE_FUNC_PTR(MsprofReportCompactInfo);
};

}  // namespace runtime
}  // namespace mlir

#endif  // AKG_EXECUTIONENGINE_ASCENDLAUNCHRUNTIME_CCEWRAPPER_H_
