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
 * \file cce_module.cc
 */

/*!
 * 2019.12.30 - Add file cce_module.cc.
 */

#include "runtime/cce/cce_module.h"

#include <runtime/file_util.h>
#include <runtime/pack_args.h>
#include <runtime/rt.h>
#include <runtime/thread_storage_scope.h>
#include <tvm/runtime/registry.h>

#include <mutex>

#include "prof_mgr_core.h"
#include "runtime/cce/cce_common.h"
#include "codegen/util.h"
#include <climits>

namespace air {
namespace runtime {
// Module to support thread-safe multi-cce execution.
// cceModule is a per-cce module
// The runtime will contain a per-device module table
// The modules will be lazily loaded
class CceModuleNode : public air::runtime::ModuleNode {
 public:
  // constructor
  CceModuleNode(const std::string& data, const std::string& fmt,
                const std::unordered_map<std::string, FunctionInfo>& fmap,
                const std::string& cce_source)
      : data_(data), fmt_(fmt), fmap_(fmap), cce_source_(cce_source) {
    std::fill(module_.begin(), module_.end(), nullptr);
    std::fill(stub_.begin(), stub_.end(), std::unordered_map<std::string, void*>());
  }
  // destructor
  ~CceModuleNode() override {
    for (int i = 0; i < static_cast<int>(module_.size()); ++i) {
      if (module_[i] != nullptr) {
        try {
          CCE_CALL(rtSetDevice(i));
          static_cast<void>(rtDevBinaryUnRegister(module_[i]));
        } catch (...) {
        }
      }
    }
  }

  const char* type_key() const final { return "cce"; }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final;

  void SaveToFile(const std::string& file_name, const std::string& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    std::string meta_file = GetMetaFilePath(file_name);
    if (fmt == "cce") {
      CHECK_NE(cce_source_.length(), 0);
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, cce_source_);
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
    if (format == fmt_) {
      return data_;
    }

    if (cce_source_.length() != 0) {
      return cce_source_;
    } else {
      return "";
    }
  }

  // get a funcStub from primary context in device_id
  void* GetFuncStub(int device_id, const std::string& func_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    // must recheck under the lock scope
    if (module_[device_id] == nullptr) {
      rtDevBinary_t devBin;
      devBin.magic = RT_DEV_BINARY_MAGIC_ELF;
      devBin.version = 1;
      devBin.length = data_.size();
      devBin.data = data_.c_str();
      static_cast<void>(rtDevBinaryRegister(&devBin, &module_[device_id]));
    }

    void* func_stub = nullptr;
    auto search = stub_[device_id].find(func_name);
    if (search != stub_[device_id].end()) {
      func_stub = search->second;
    } else {
      kernel_stub_gen_++;
      func_stub = kernel_stub_gen_;
      static_cast<void>(rtFunctionRegister(module_[device_id], func_stub, func_name.c_str(),
                                           func_name.c_str(), 0));
      stub_[device_id][func_name] = func_stub;
    }

    return func_stub;
  }

 private:
  // The binary data
  std::string data_;
  // The format
  std::string fmt_;
  // function information table.
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // The cce source
  std::string cce_source_;
  // The internal modules per device, to be lazy initialized
  std::array<void*, kMaxNumDevices> module_;
  std::array<std::unordered_map<std::string, void*>, kMaxNumDevices> stub_;
  // internal mutex when updating the module
  std::mutex mutex_;
  // global increate to make stub unique
  static int* kernel_stub_gen_;
};

int* CceModuleNode::kernel_stub_gen_ = nullptr;

// a wrapped function class to get packed func.
class CceWrappedFunc {
 public:
  CceWrappedFunc() = default;
  ~CceWrappedFunc() = default;
  // initailize the cce function.
  void Init(CceModuleNode* m, ObjectPtr<Object> sptr, const std::string& func_name,
            std::vector<size_t> arg_size, const std::vector<std::string>& thread_axis_tags) {
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    arg_size_ = arg_size;
    std::fill(fcache_.begin(), fcache_.end(), nullptr);
    thread_axis_cfg_.Init(arg_size.size(), thread_axis_tags);
  }

  // invoke the function with void arguments
  void operator()(const TVMArgs args, TVMRetValue* rv, void** void_args, int64_t* shape_args,
		  size_t shape_arg_size) const {
    int device_id;
    CCE_CALL(rtGetDevice(&device_id));

    if (fcache_[device_id] == nullptr) {
      fcache_[device_id] = m_->GetFuncStub(device_id, func_name_);
    }

    ThreadWorkLoad wl = thread_axis_cfg_.Extract(args);
    int blockDim = static_cast<int>(wl.grid_dim(0));
    rtL2Ctrl_t* l2ctrl = nullptr;
    auto strm = static_cast<rtStream_t>(CceThreadEntry::ThreadLocal()->stream);

    size_t raw_size = arg_size_.size() - shape_arg_size;
    void** raw_args;
#ifdef USE_KC_AIR
    raw_args = new void*[raw_size];
#else
    raw_args = new void*[arg_size_.size()];
#endif
    size_t args_size = 0;
    for (size_t i = 0; i < raw_size; ++i) {
      args_size += arg_size_[i];
      void** ptr = reinterpret_cast<void**>(void_args[i]);
      raw_args[i] = *ptr;
    }

#ifdef USE_CCE_PROFILING
    std::ostringstream buffer;
    buffer << "{\"startCfg\":[{\"deviceID\":\"" << device_id << "\",\"jobID\":\"JOBTUNE";
    buffer << "\",\"features\":[{\"name\":\"task_trace\"}]}]}";
    std::string cfg = buffer.str();
    LOG(INFO) << "The profiling trace config: " << cfg;
    ProfMgrCfg profCfg = {cfg};
    LOG(INFO) << "Start profiling";
    CceThreadEntry::ThreadLocal()->profcfghandle = ProfMgrStartUp(&profCfg);
    if (CceThreadEntry::ThreadLocal()->profcfghandle == nullptr) {
      LOG(INFO) << "Start profiling failed";
    } else {
      LOG(INFO) << "Start profiling succ";
    }
#endif
    rtError_t result;

    if (shape_arg_size == 0) {
      result = rtKernelLaunch(fcache_[device_id], blockDim,
                              raw_args,  // void_args,
                              static_cast<uint32_t>(args_size), l2ctrl, strm);
    } else {
      result = RT_ERROR_NONE;
      if (blockDim == INT_MAX) {
        blockDim = shape_args[shape_arg_size - 1];
      }
#ifdef USE_KC_AIR
      result = rtKernelLaunchShapes(fcache_[device_id], blockDim,
                                    raw_args, // void_args,
                                    static_cast<uint32_t>(args_size), shape_args, shape_arg_size,
                                    l2ctrl, strm);
#else
      for (size_t ssize = raw_size; ssize < arg_size_.size(); ++ssize) {
        void* tempshape = reinterpret_cast<void*> (shape_args[ssize - raw_size]);
        raw_args[ssize] = tempshape;
        args_size += 8;
      }
      result = rtKernelLaunch(fcache_[device_id], blockDim, raw_args, static_cast<uint32_t>(args_size), l2ctrl, strm);
#endif
      akg::RecordCore(blockDim, true);
    }
    delete[] raw_args;
    if (result != RT_ERROR_NONE) {
      const char* msg{nullptr};
      std::ostringstream os;

      msg = CceGetErrorString(result);
      os << "cceLaunch Error: " << msg << "\n"
         << "blockDim=(" << blockDim << ")\n"
         << "func_name=" << func_name_ << "\n";

      LOG(FATAL) << os.str();
    }
  }

 private:
  // internal module
  CceModuleNode* m_{nullptr};
  // The resource holder
  ObjectPtr<Object> sptr_;
  // The name of the function
  std::string func_name_;
  // convert code for void argument
  std::vector<size_t> arg_size_;
  // Device function per device.
  // mark as mutable, to enable lazy initialization
  mutable std::array<void*, kMaxNumDevices> fcache_;
  // thread axis configuration
  ThreadAxisConfig thread_axis_cfg_;
};

PackedFunc CceModuleNode::GetFunction(const std::string& name,
                                      const ObjectPtr<Object>& sptr_to_self) {
  CHECK_EQ(sptr_to_self.get(), this);
  CHECK_NE(name, symbol::tvm_module_main) << "Device function do not have main";

  auto it = fmap_.find(name);
  if (it == fmap_.end()) {
    return PackedFunc();
  }

  const FunctionInfo& info = it->second;
  CceWrappedFunc f;
  std::vector<size_t> arg_size(info.arg_types.size());
  for (int i = 0; i < static_cast<int>(info.arg_types.size()); ++i) {
    TVMType t = info.arg_types[i];
    CHECK_EQ(t.lanes, 1U);
    uint32_t bits = t.bits;
    CHECK_EQ(bits % 8, 0U);
    arg_size[i] = bits / 8;
  }

  f.Init(this, sptr_to_self, name, arg_size, info.thread_axis_tags);

  return PackFuncVoidAddrCCE(f, info.arg_types);
}

Module CceModuleCreate(std::string data, std::string fmt,
                       std::unordered_map<std::string, FunctionInfo> fmap, std::string cce_source) {
  auto n = make_object<CceModuleNode>(data, fmt, fmap, cce_source);
  return Module(n);
}

Module CceModuleLoadFile(const std::string& file_name, const std::string& format) {
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  return CceModuleCreate(data, fmt, fmap, std::string());
}

TVM_REGISTER_GLOBAL("module.loadfile_cce").set_body([](const TVMArgs args, TVMRetValue* rv) {
  *rv = CceModuleLoadFile(args[0], args[1]);
});

Module CceModuleLoadBinary(void* strm) {
  auto stream = static_cast<dmlc::Stream*>(strm);
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&data);
  return CceModuleCreate(data, fmt, fmap, std::string());
}

TVM_REGISTER_GLOBAL("module.loadbinary_cce").set_body([](const TVMArgs args, TVMRetValue* rv) {
  *rv = CceModuleLoadBinary(args[0]);
});
}  // namespace runtime
}  // namespace air
