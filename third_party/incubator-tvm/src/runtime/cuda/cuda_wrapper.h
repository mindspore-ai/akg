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
 *  \file cuda_wrapper.h
 *  \brief cuda symbols wrapper
 */

/*!
 * 2023.4.21 - Add file cuda_wrapper.h.
 */

#ifndef TVM_RUNTIME_CUDA_CUDA_WRAPPER_H_
#define TVM_RUNTIME_CUDA_CUDA_WRAPPER_H_

#include "../symbols_wrapper.h"
#include <cuda.h>
#include <nvrtc.h>
#include <cuda_runtime.h>

namespace air {
namespace runtime {


class CudaWrapper : public SymbolsWrapper {
 public:
  static CudaWrapper *GetInstance();
  CudaWrapper(const CudaWrapper &) = delete;
  CudaWrapper &operator=(const CudaWrapper &) = delete;
  ~CudaWrapper();
  bool LoadLibraries();
  bool UnLoadLibraries();
 private:
  CudaWrapper();
  static std::shared_ptr<CudaWrapper> cuda_wrapper_singleton_;
  bool LoadCuda();
  bool LoadCudart();
  bool LoadNvrtc();
  void *cuda_handle_{nullptr};
  void *nvrtc_handle_{nullptr};
  void *cudart_handle_{nullptr};

 public:
  using cudaGetDeviceFunc = cudaError_t (*)(int *);
  using cudaSetDeviceFunc = cudaError_t (*)(int);
  using cudaDeviceGetAttributeFunc = cudaError_t (*)(int *, enum cudaDeviceAttr, int);
  using cudaStreamCreateFunc = cudaError_t (*)(cudaStream_t *);
  using cudaStreamDestroyFunc = cudaError_t (*)(cudaStream_t);
  using cudaStreamSynchronizeFunc = cudaError_t (*)(cudaStream_t);
  using cudaMallocFunc = cudaError_t (*)(void **, size_t);
  using cudaFreeFunc = cudaError_t (*)(void *);
  using cudaMemcpyFunc = cudaError_t (*)(void *, const void *, size_t, enum cudaMemcpyKind);
  using cudaMemcpyAsyncFunc = cudaError_t (*)(void *, const void *, size_t, enum cudaMemcpyKind, cudaStream_t);
  using cudaMemcpyPeerAsyncFunc = cudaError_t (*)(void *, int, const void *, int, size_t, cudaStream_t);
  using cudaEventCreateFunc = cudaError_t (*)(cudaEvent_t *);
  using cudaEventDestroyFunc = cudaError_t (*)(cudaEvent_t);
  using cudaEventRecordFunc = cudaError_t (*)(cudaEvent_t, cudaStream_t);
  using cudaStreamWaitEventFunc = cudaError_t (*)(cudaStream_t, cudaEvent_t, unsigned int);
  using cudaLaunchKernelFunc = cudaError_t (*)(const void *, dim3, dim3, void **, size_t, cudaStream_t);
  using cudaGetErrorStringFunc = const char* (*)(cudaError_t);

  using cuDeviceGetNameFunc = CUresult (*)(char *, int, CUdevice);
  using cuModuleLoadDataFunc = CUresult (*)(CUmodule *, const void *);
  using cuModuleLoadDataExFunc = CUresult (*)(CUmodule *, const void *, unsigned int, CUjit_option *, void **);
  using cuModuleUnloadFunc = CUresult (*)(CUmodule);
  using cuModuleGetFunctionFunc = CUresult (*)(CUfunction *, CUmodule, const char *);
  using cuMemsetD32_v2Func = CUresult (*)(CUdeviceptr, unsigned int, size_t);
  using cuModuleGetGlobal_v2Func = CUresult (*)(CUdeviceptr*, size_t*, CUmodule, const char*);
  using cuLaunchKernelFunc = CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
                                          unsigned int, unsigned int, unsigned int, CUstream, void **, void **);
  using cuGetErrorNameFunc = CUresult (*)(CUresult, const char **);

  using nvrtcCreateProgramFunc = nvrtcResult (*)(nvrtcProgram *, const char *, const char *, int, const char * const *,
                                                 const char * const *);
  using nvrtcDestroyProgramFunc = nvrtcResult (*)(nvrtcProgram *);
  using nvrtcCompileProgramFunc = nvrtcResult (*)(nvrtcProgram, int, const char * const *);
  using nvrtcGetProgramLogSizeFunc = nvrtcResult (*)(nvrtcProgram, size_t *);
  using nvrtcGetProgramLogFunc = nvrtcResult (*)(nvrtcProgram, char *);
  using nvrtcGetPTXSizeFunc = nvrtcResult (*)(nvrtcProgram, size_t *);
  using nvrtcGetPTXFunc = nvrtcResult (*)(nvrtcProgram, char *);
  using nvrtcGetErrorStringFunc = const char * (*)(nvrtcResult);

  DEFINE_FUNC_PTR(cudaGetDevice);
  DEFINE_FUNC_PTR(cudaSetDevice);
  DEFINE_FUNC_PTR(cudaDeviceGetAttribute);
  DEFINE_FUNC_PTR(cudaStreamCreate);
  DEFINE_FUNC_PTR(cudaStreamDestroy);
  DEFINE_FUNC_PTR(cudaStreamSynchronize);
  DEFINE_FUNC_PTR(cudaMalloc);
  DEFINE_FUNC_PTR(cudaFree);
  DEFINE_FUNC_PTR(cudaMemcpy);
  DEFINE_FUNC_PTR(cudaMemcpyAsync);
  DEFINE_FUNC_PTR(cudaMemcpyPeerAsync);
  DEFINE_FUNC_PTR(cudaEventCreate);
  DEFINE_FUNC_PTR(cudaEventDestroy);
  DEFINE_FUNC_PTR(cudaEventRecord);
  DEFINE_FUNC_PTR(cudaStreamWaitEvent);
  DEFINE_FUNC_PTR(cudaLaunchKernel);
  DEFINE_FUNC_PTR(cudaGetErrorString);

  DEFINE_FUNC_PTR(cuDeviceGetName);
  DEFINE_FUNC_PTR(cuModuleLoadData);
  DEFINE_FUNC_PTR(cuModuleLoadDataEx);
  DEFINE_FUNC_PTR(cuModuleUnload);
  DEFINE_FUNC_PTR(cuModuleGetFunction);
  DEFINE_FUNC_PTR(cuMemsetD32_v2);
  DEFINE_FUNC_PTR(cuModuleGetGlobal_v2);
  DEFINE_FUNC_PTR(cuLaunchKernel);
  DEFINE_FUNC_PTR(cuGetErrorName);

  DEFINE_FUNC_PTR(nvrtcCreateProgram);
  DEFINE_FUNC_PTR(nvrtcDestroyProgram);
  DEFINE_FUNC_PTR(nvrtcCompileProgram);
  DEFINE_FUNC_PTR(nvrtcGetProgramLogSize);
  DEFINE_FUNC_PTR(nvrtcGetProgramLog);
  DEFINE_FUNC_PTR(nvrtcGetPTXSize);
  DEFINE_FUNC_PTR(nvrtcGetPTX);
  DEFINE_FUNC_PTR(nvrtcGetErrorString);
};

}  // namespace runtime
}  // namespace air

#endif  // TVM_RUNTIME_CUDA_CUDA_WRAPPER_H_
