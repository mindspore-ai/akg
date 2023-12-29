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
 *  \file cuda_wrapper.cc
 *  \brief cuda symbols wrapper
 */

/*!
 * 2023.4.21 - Add file cuda_wrapper.cc.
 */

#include "cuda_wrapper.h"
#include <dlfcn.h>
#include <mutex>

namespace air {
namespace runtime {

std::shared_ptr<CudaWrapper> CudaWrapper::cuda_wrapper_singleton_ = nullptr;

CudaWrapper* CudaWrapper::GetInstance() {
  static std::once_flag cuda_symbol_once;
  std::call_once(cuda_symbol_once, []() { cuda_wrapper_singleton_.reset(new CudaWrapper()); });
  return cuda_wrapper_singleton_.get();
}

CudaWrapper::CudaWrapper() { LoadLibraries(); }

CudaWrapper::~CudaWrapper() {
  if (cuda_wrapper_singleton_.get() == nullptr) {
    return;
  }
  cuda_wrapper_singleton_->UnLoadLibraries();
}

bool CudaWrapper::UnLoadLibraries() {
  if (cuda_handle_ != nullptr) {
    if (dlclose(cuda_handle_) != 0) {
      return false;
    }
  }
  if (nvrtc_handle_ != nullptr) {
    if (dlclose(nvrtc_handle_) != 0) {
      return false;
    }
  }
  if (cudart_handle_ != nullptr) {
    if (dlclose(cudart_handle_) != 0) {
      return false;
    }
  }
  cuda_handle_ = nullptr;
  nvrtc_handle_ = nullptr;
  cudart_handle_ = nullptr;
  return true;
}

bool CudaWrapper::LoadCudart() {
  std::string library_path = "libcudart.so";
  void *handle_ptr = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle_ptr == nullptr) {
    LOG(ERROR) << "load library " << library_path << " failed!";
    return false;
  }
  cudart_handle_ = handle_ptr;

  LOAD_FUNCTION_PTR(cudaGetDevice);
  LOAD_FUNCTION_PTR(cudaSetDevice);
  LOAD_FUNCTION_PTR(cudaDeviceGetAttribute);
  LOAD_FUNCTION_PTR(cudaStreamCreate);
  LOAD_FUNCTION_PTR(cudaStreamDestroy);
  LOAD_FUNCTION_PTR(cudaStreamSynchronize);
  LOAD_FUNCTION_PTR(cudaMalloc);
  LOAD_FUNCTION_PTR(cudaFree);
  LOAD_FUNCTION_PTR(cudaMemcpy);
  LOAD_FUNCTION_PTR(cudaMemcpyAsync);
  LOAD_FUNCTION_PTR(cudaMemcpyPeerAsync);
  LOAD_FUNCTION_PTR(cudaEventCreate);
  LOAD_FUNCTION_PTR(cudaEventDestroy);
  LOAD_FUNCTION_PTR(cudaEventRecord);
  LOAD_FUNCTION_PTR(cudaStreamWaitEvent);
  LOAD_FUNCTION_PTR(cudaLaunchKernel);
  LOAD_FUNCTION_PTR(cudaGetErrorString);
  return true;
}

bool CudaWrapper::LoadCuda() {
  std::string library_path = "libcuda.so";
  void *handle_ptr = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle_ptr == nullptr) {
    LOG(ERROR) << "load library " << library_path << " failed!";
    return false;
  }
  cudart_handle_ = handle_ptr;
  LOAD_FUNCTION_PTR(cuDeviceGetName);
  LOAD_FUNCTION_PTR(cuModuleLoadData);
  LOAD_FUNCTION_PTR(cuModuleLoadDataEx);
  LOAD_FUNCTION_PTR(cuModuleUnload);
  LOAD_FUNCTION_PTR(cuModuleGetFunction);
  LOAD_FUNCTION_PTR(cuMemsetD32_v2);
  LOAD_FUNCTION_PTR(cuModuleGetGlobal_v2);
  LOAD_FUNCTION_PTR(cuLaunchKernel);
  LOAD_FUNCTION_PTR(cuGetErrorName);
  return true;
}

bool CudaWrapper::LoadNvrtc() {
  std::string library_path = "libnvrtc.so";
  void *handle_ptr = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle_ptr == nullptr) {
    LOG(ERROR) << "load library " << library_path << " failed!";
    return false;
  }
  nvrtc_handle_ = handle_ptr;
  LOAD_FUNCTION_PTR(nvrtcCreateProgram);
  LOAD_FUNCTION_PTR(nvrtcDestroyProgram);
  LOAD_FUNCTION_PTR(nvrtcCompileProgram);
  LOAD_FUNCTION_PTR(nvrtcGetProgramLogSize);
  LOAD_FUNCTION_PTR(nvrtcGetProgramLog);
  LOAD_FUNCTION_PTR(nvrtcGetPTXSize);
  LOAD_FUNCTION_PTR(nvrtcGetPTX);
  LOAD_FUNCTION_PTR(nvrtcGetErrorString);
  return true;
}

bool CudaWrapper::LoadLibraries() {
  LoadCuda();
  LoadCudart();
  LoadNvrtc();
  return true;
}

}  // namespace runtime
}  // namespace air


__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDevice(int *device) {
  auto func = air::runtime::CudaWrapper::GetInstance()->cudaGetDevice;
  CHECK_NOTNULL(func);
  return func(device);
}

__host__ cudaError_t CUDARTAPI cudaSetDevice(int device) {
  auto func = air::runtime::CudaWrapper::GetInstance()->cudaSetDevice;
  CHECK_NOTNULL(func);
  return func(device);
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetAttribute(int *val, enum cudaDeviceAttr attr, int dev) {
  auto func = air::runtime::CudaWrapper::GetInstance()->cudaDeviceGetAttribute;
  CHECK_NOTNULL(func);
  return func(val, attr, dev);
}

__host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *stream) {
  auto func = air::runtime::CudaWrapper::GetInstance()->cudaStreamCreate;
  CHECK_NOTNULL(func);
  return func(stream);
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) {
  auto func = air::runtime::CudaWrapper::GetInstance()->cudaStreamDestroy;
  CHECK_NOTNULL(func);
  return func(stream);
}

__host__ cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
  auto func = air::runtime::CudaWrapper::GetInstance()->cudaStreamSynchronize;
  CHECK_NOTNULL(func);
  return func(stream);
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMalloc(void **dev, size_t size) {
  auto func = air::runtime::CudaWrapper::GetInstance()->cudaMalloc;
  CHECK_NOTNULL(func);
  return func(dev, size);
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFree(void *dev) {
  auto func = air::runtime::CudaWrapper::GetInstance()->cudaFree;
  CHECK_NOTNULL(func);
  return func(dev);
}

__host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
  auto func = air::runtime::CudaWrapper::GetInstance()->cudaMemcpy;
  CHECK_NOTNULL(func);
  return func(dst, src, count, kind);
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count,
                                                                  enum cudaMemcpyKind kind, cudaStream_t stream) {
  auto func = air::runtime::CudaWrapper::GetInstance()->cudaMemcpyAsync;
  CHECK_NOTNULL(func);
  return func(dst, src, count, kind, stream);
}

__host__ cudaError_t CUDARTAPI cudaMemcpyPeerAsync(void *dst, int dst_dev, const void *src, int src_dev,
                                                   size_t count, cudaStream_t stream) {
  auto func = air::runtime::CudaWrapper::GetInstance()->cudaMemcpyPeerAsync;
  CHECK_NOTNULL(func);
  return func(dst, dst_dev, src, src_dev, count, stream);
}

__host__ cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event) {
  auto func = air::runtime::CudaWrapper::GetInstance()->cudaEventCreate;
  CHECK_NOTNULL(func);
  return func(event);
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
  auto func = air::runtime::CudaWrapper::GetInstance()->cudaEventDestroy;
  CHECK_NOTNULL(func);
  return func(event);
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  auto func = air::runtime::CudaWrapper::GetInstance()->cudaEventRecord;
  CHECK_NOTNULL(func);
  return func(event, stream);
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                                                      unsigned int flags) {
  auto func = air::runtime::CudaWrapper::GetInstance()->cudaStreamWaitEvent;
  CHECK_NOTNULL(func);
  return func(stream, event, flags);
}

__host__ cudaError_t CUDARTAPI cudaLaunchKernel(const void *function, dim3 grid, dim3 block, void **args,
                                                size_t shared_mem, cudaStream_t stream) {
  auto func = air::runtime::CudaWrapper::GetInstance()->cudaLaunchKernel;
  CHECK_NOTNULL(func);
  return func(function, grid, block, args, shared_mem, stream);
}

__host__ __cudart_builtin__ const char* CUDARTAPI cudaGetErrorString(cudaError_t error) {
  auto func = air::runtime::CudaWrapper::GetInstance()->cudaGetErrorString;
  CHECK_NOTNULL(func);
  return func(error);
}

CUresult CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev) {
  auto func = air::runtime::CudaWrapper::GetInstance()->cuDeviceGetName;
  CHECK_NOTNULL(func);
  return func(name, len, dev);
}

CUresult CUDAAPI cuModuleLoadData(CUmodule *module, const void *image) {
  auto func = air::runtime::CudaWrapper::GetInstance()->cuModuleLoadData;
  CHECK_NOTNULL(func);
  return func(module, image);
}

CUresult CUDAAPI cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int num,
                                    CUjit_option *options, void **values) {
  auto func = air::runtime::CudaWrapper::GetInstance()->cuModuleLoadDataEx;
  CHECK_NOTNULL(func);
  return func(module, image, num, options, values);
}

CUresult CUDAAPI cuModuleUnload(CUmodule hmod) {
  auto func = air::runtime::CudaWrapper::GetInstance()->cuModuleUnload;
  CHECK_NOTNULL(func);
  return func(hmod);
}

CUresult CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
  auto func = air::runtime::CudaWrapper::GetInstance()->cuModuleGetFunction;
  CHECK_NOTNULL(func);
  return func(hfunc, hmod, name);
}

CUresult CUDAAPI cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
  auto func = air::runtime::CudaWrapper::GetInstance()->cuMemsetD32_v2;
  CHECK_NOTNULL(func);
  return func(dstDevice, ui, N);
}

CUresult cuModuleGetGlobal_v2(CUdeviceptr *ptr, size_t *offset, CUmodule mod, const char* name) {
  auto func = air::runtime::CudaWrapper::GetInstance()->cuModuleGetGlobal_v2;
  CHECK_NOTNULL(func);
  return func(ptr, offset, mod, name);
}

CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
                                unsigned int block_x, unsigned int block_y, unsigned int block_z,
                                unsigned int shared_size, CUstream stream, void **params, void **extra) {
  auto func = air::runtime::CudaWrapper::GetInstance()->cuLaunchKernel;
  CHECK_NOTNULL(func);
  return func(f, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_size, stream, params, extra);
}

CUresult CUDAAPI cuGetErrorName(CUresult error, const char **str) {
  auto func = air::runtime::CudaWrapper::GetInstance()->cuGetErrorName;
  CHECK_NOTNULL(func);
  return func(error, str);
}

nvrtcResult nvrtcCreateProgram(nvrtcProgram *prog,
                               const char *src,
                               const char *name,
                               int num_headers,
                               const char * const *headers,
                               const char * const *include_names) {
  auto func = air::runtime::CudaWrapper::GetInstance()->nvrtcCreateProgram;
  CHECK_NOTNULL(func);
  return func(prog, src, name, num_headers, headers, include_names);
}

nvrtcResult nvrtcDestroyProgram(nvrtcProgram *prog) {
  auto func = air::runtime::CudaWrapper::GetInstance()->nvrtcDestroyProgram;
  CHECK_NOTNULL(func);
  return func(prog);
}

nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int num_options, const char * const *options) {
  auto func = air::runtime::CudaWrapper::GetInstance()->nvrtcCompileProgram;
  CHECK_NOTNULL(func);
  return func(prog, num_options, options);
}

nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t *log_size) {
  auto func = air::runtime::CudaWrapper::GetInstance()->nvrtcGetProgramLogSize;
  CHECK_NOTNULL(func);
  return func(prog, log_size);
}

nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char *log) {
  auto func = air::runtime::CudaWrapper::GetInstance()->nvrtcGetProgramLog;
  CHECK_NOTNULL(func);
  return func(prog, log);
}

nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t *ptx_size) {
  auto func = air::runtime::CudaWrapper::GetInstance()->nvrtcGetPTXSize;
  CHECK_NOTNULL(func);
  return func(prog, ptx_size);
}

nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char *ptx) {
  auto func = air::runtime::CudaWrapper::GetInstance()->nvrtcGetPTX;
  CHECK_NOTNULL(func);
  return func(prog, ptx);
}

const char *nvrtcGetErrorString(nvrtcResult result) {
  auto func = air::runtime::CudaWrapper::GetInstance()->nvrtcGetErrorString;
  CHECK_NOTNULL(func);
  return func(result);
}
