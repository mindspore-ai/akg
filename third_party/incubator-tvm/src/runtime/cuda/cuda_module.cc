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
 * \file cuda_module.cc
 * 2020.09.19 - Modify operator() for kc_air.
 * 2020.09.22 - Separate the implementation of KC and GPU.
 * 2021.06.08 - While compiling the cuda, limit the register num for per thread to avoid out of
 * memory problem.
 * 2023.04.21 - Load cuda symbols.
 */
#include "cuda_module.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <tvm/runtime/registry.h>

#include <array>
#include <cmath>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../file_util.h"
#include "../meta_data.h"
#include "../pack_args.h"
#include "../thread_storage_scope.h"
#include "cuda_common.h"

namespace air {
namespace runtime {
bool LoadCudaLibrary();
// Module to support thread-safe multi-GPU execution.
// cuModule is a per-GPU module
// The runtime will contain a per-device module table
// The modules will be lazily loaded
class CUDAModuleNode : public runtime::ModuleNode {
 public:
  explicit CUDAModuleNode(std::string data, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap,
                          std::string cuda_source)
      : data_(data), fmt_(fmt), fmap_(fmap), cuda_source_(cuda_source) {
    std::fill(module_.begin(), module_.end(), nullptr);
    std::fill(func_.begin(), func_.end(), nullptr);
    CudaWrapper::GetInstance();
  }
  // destructor
  ~CUDAModuleNode() {
    for (size_t i = 0; i < module_.size(); ++i) {
      if (module_[i] != nullptr) {
        CUDA_CALL(cudaSetDevice(static_cast<int>(i)));
        CUDA_DRIVER_CALL(cuModuleUnload(module_[i]));
      }
    }
  }

  const char* type_key() const final { return "cuda"; }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final;

  void SaveToFile(const std::string& file_name, const std::string& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    std::string meta_file = GetMetaFilePath(file_name);
    if (fmt == "cu") {
      CHECK_NE(cuda_source_.length(), 0);
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, cuda_source_);
    } else {
      CHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, data_);
    }
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(fmt_);
    stream->Write(fmap_);
    stream->Write(data_);
  }

  std::string GetSource(const std::string& format) final {
    if (format == fmt_) return data_;
    if (cuda_source_.length() != 0) {
      return cuda_source_;
    } else {
      if (fmt_ == "ptx") return data_;
      return "";
    }
  }

  // get a CUfunction from primary context in device_id
  CUfunction GetFunc(int device_id, const std::string& func_name, ThreadWorkLoad wl) {
    std::lock_guard<std::mutex> lock(mutex_);
    // must recheck under the lock scope
    if (module_[device_id] == nullptr) {
      // See detail: https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.htmlt
      CUjit_option options[1];
      options[0] = CU_JIT_MAX_REGISTERS;
      void* values[1];
      int total_threads = wl.block_dim(0) * wl.block_dim(1) * wl.block_dim(2);
      int total_warps = std::ceil(float(total_threads) / float(WARP_SIZE));
      int limit_warps = (total_warps + WARP_ALLOC_GRAN - 1) / WARP_ALLOC_GRAN * WARP_ALLOC_GRAN;
      int total_register_unit_nums = MAX_REGISTER_PER_THREAD_BLOCK / REGISTER_UNIT_IN_WARP;
      int register_unit_nums_per_warp = total_register_unit_nums / limit_warps;
      long register_nums = (register_unit_nums_per_warp * REGISTER_UNIT_IN_WARP) / WARP_SIZE;

      values[0] = (void*)register_nums;
      CUDA_DRIVER_CALL(
          cuModuleLoadDataEx(&(module_[device_id]), data_.c_str(), 1, options, values));
    }
    CUresult result = CUDA_SUCCESS;
    CUfunction func = nullptr;
    if (func_[device_id] == nullptr) {
#ifdef USE_KC_AIR
      result = cuModuleGetFunction(&func_[device_id], module_[device_id], func_name.c_str());
#else
      result = cuModuleGetFunction(&func, module_[device_id], func_name.c_str());
#endif
    }
    if (result != CUDA_SUCCESS) {
      const char* msg;
      cuGetErrorName(result, &msg);
      LOG(FATAL) << "CUDAError: cuModuleGetFunction " << func_name << " failed with error: " << msg;
    }
#ifdef USE_KC_AIR
    return func_[device_id];
#else
    return func;
#endif
  }
  // get a global var from primary context in device_id
  CUdeviceptr GetGlobal(int device_id, const std::string& global_name, size_t expect_nbytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    // must recheck under the lock scope
    if (module_[device_id] == nullptr) {
      CUDA_DRIVER_CALL(cuModuleLoadData(&(module_[device_id]), data_.c_str()));
    }
    CUdeviceptr global;
    size_t nbytes;

    CUresult result = cuModuleGetGlobal(&global, &nbytes, module_[device_id], global_name.c_str());
    CHECK_EQ(nbytes, expect_nbytes);
    if (result != CUDA_SUCCESS) {
      const char* msg;
      cuGetErrorName(result, &msg);
      LOG(FATAL) << "CUDAError: cuModuleGetGlobal " << global_name << " failed with error: " << msg;
    }
    return global;
  }

 private:
  // the binary data
  std::string data_;
  // The format
  std::string fmt_;
  // function information table.
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // The cuda source.
  std::string cuda_source_;
  // the internal modules per GPU, to be lazily initialized.
  std::array<CUmodule, kMaxNumGPUs> module_;
  // internal mutex when updating the module
  std::mutex mutex_;
  std::array<CUfunction, kMaxNumGPUs> func_;
  const int MAX_REGISTER_PER_THREAD_BLOCK = 65536;
  const int REGISTER_UNIT_IN_WARP = 256;
  const int WARP_SIZE = 32;
  const int WARP_ALLOC_GRAN = 4;
};

// a wrapped function class to get packed func.
class CUDAWrappedFunc {
 public:
  // initialize the CUDA function.
  void Init(CUDAModuleNode* m, ObjectPtr<Object> sptr, const std::string& func_name,
            size_t num_void_args, std::vector<size_t> arg_size,
            const std::vector<std::string>& thread_axis_tags) {
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    num_void_args_ = num_void_args;
    arg_size_ = arg_size;
    std::fill(fcache_.begin(), fcache_.end(), nullptr);
    thread_axis_cfg_.Init(num_void_args, thread_axis_tags);
  }
  // invoke the function with void arguments
  void operator()(TVMArgs args, TVMRetValue* rv, void** void_args) const {
    int device_id;
    CUDA_CALL(cudaGetDevice(&device_id));
    ThreadWorkLoad wl = thread_axis_cfg_.Extract(args);
    if (fcache_[device_id] == nullptr) {
      fcache_[device_id] = m_->GetFunc(device_id, func_name_, wl);
    }
    CUstream strm = static_cast<CUstream>(CUDAThreadEntry::ThreadLocal()->stream);
    CUresult result;

#ifdef USE_KC_AIR
    size_t raw_size = num_void_args_;
    void** raw_args = new (std::nothrow) void*[raw_size];
    if (*raw_args == nullptr) {
      LOG(FATAL) << "Memory alloc fail.";
    }
    size_t args_size = 0;
    for (size_t i = 0; i < raw_size; ++i) {
      args_size += arg_size_[i];
      void** ptr = reinterpret_cast<void**>(void_args[i]);
      raw_args[i] = *ptr;
    }
    result = cuLaunchKernel(fcache_[device_id], wl.grid_dim(0), wl.grid_dim(1), wl.grid_dim(2),
                            wl.block_dim(0), wl.block_dim(1), wl.block_dim(2),
                            (static_cast<uint32_t>(args_size) / sizeof(void*)), strm, raw_args, 0);
    if (raw_args != NULL) {
      free(raw_args);
      raw_args = NULL;
    }
#else
    result =
        cuLaunchKernel(fcache_[device_id], wl.grid_dim(0), wl.grid_dim(1), wl.grid_dim(2),
                       wl.block_dim(0), wl.block_dim(1), wl.block_dim(2), 0, strm, void_args, 0);
#endif

    if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) {
      const char* msg;
      cuGetErrorName(result, &msg);
      std::ostringstream os;
      os << "CUDALaunch Error: " << msg << "\n"
         << " grid=(" << wl.grid_dim(0) << "," << wl.grid_dim(1) << "," << wl.grid_dim(2) << "), "
         << " block=(" << wl.block_dim(0) << "," << wl.block_dim(1) << "," << wl.block_dim(2)
         << ")\n";
      std::string cuda = m_->GetSource("");
      if (cuda.length() != 0) {
        os << "// func_name=" << func_name_ << "\n"
           << "// CUDA Source\n"
           << "// -----------\n"
           << cuda;
      }
      LOG(FATAL) << os.str();
    }
  }

 private:
  // internal module
  CUDAModuleNode* m_;
  // the resource holder
  ObjectPtr<Object> sptr_;
  // The name of the function.
  std::string func_name_;

  std::vector<size_t> arg_size_;
  size_t num_void_args_;
  // Device function cache per device.
  // mark as mutable, to enable lazy initialization
  mutable std::array<CUfunction, kMaxNumGPUs> fcache_;
  // thread axis configuration
  ThreadAxisConfig thread_axis_cfg_;
};

class CUDAPrepGlobalBarrier {
 public:
  CUDAPrepGlobalBarrier(CUDAModuleNode* m, ObjectPtr<Object> sptr) : m_(m), sptr_(sptr) {
    std::fill(pcache_.begin(), pcache_.end(), 0);
  }

  void operator()(const TVMArgs& args, TVMRetValue* rv) const {
    int device_id;
    CUDA_CALL(cudaGetDevice(&device_id));
    if (pcache_[device_id] == 0) {
      pcache_[device_id] =
          m_->GetGlobal(device_id, runtime::symbol::tvm_global_barrier_state, sizeof(unsigned));
    }
    CUDA_DRIVER_CALL(cuMemsetD32(pcache_[device_id], 0, 1));
  }

 private:
  // internal module
  CUDAModuleNode* m_;
  // the resource holder
  ObjectPtr<Object> sptr_;
  // mark as mutable, to enable lazy initialization
  mutable std::array<CUdeviceptr, kMaxNumGPUs> pcache_;
};

PackedFunc CUDAModuleNode::GetFunction(const std::string& name,
                                       const ObjectPtr<Object>& sptr_to_self) {
  CHECK_EQ(sptr_to_self.get(), this);
  CHECK_NE(name, symbol::tvm_module_main) << "Device function do not have main";
  if (name == symbol::tvm_prepare_global_barrier) {
    return PackedFunc(CUDAPrepGlobalBarrier(this, sptr_to_self));
  }
  auto it = fmap_.find(name);
  if (it == fmap_.end()) return PackedFunc();
  const FunctionInfo& info = it->second;
  CUDAWrappedFunc f;
  std::vector<size_t> arg_size(info.arg_types.size());
  for (int i = 0; i < static_cast<int>(info.arg_types.size()); ++i) {
    TVMType t = info.arg_types[i];
    CHECK_EQ(t.lanes, 1U);
    uint32_t bits = t.bits;
    CHECK_EQ(bits % 8, 0U);
    arg_size[i] = bits / 8;
  }
  f.Init(this, sptr_to_self, name, info.arg_types.size(), arg_size, info.thread_axis_tags);
  return PackFuncVoidAddr(f, info.arg_types);
}

Module CUDAModuleCreate(std::string data, std::string fmt,
                        std::unordered_map<std::string, FunctionInfo> fmap,
                        std::string cuda_source) {
  auto n = make_object<CUDAModuleNode>(data, fmt, fmap, cuda_source);
  return Module(n);
}

// Load module from module.
Module CUDAModuleLoadFile(const std::string& file_name, const std::string& format) {
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  return CUDAModuleCreate(data, fmt, fmap, std::string());
}

Module CUDAModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&data);
  return CUDAModuleCreate(data, fmt, fmap, std::string());
}

TVM_REGISTER_GLOBAL("module.loadfile_cubin").set_body_typed(CUDAModuleLoadFile);

TVM_REGISTER_GLOBAL("module.loadfile_ptx").set_body_typed(CUDAModuleLoadFile);

TVM_REGISTER_GLOBAL("module.loadbinary_cuda").set_body_typed(CUDAModuleLoadBinary);
}  // namespace runtime
}  // namespace air
