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
 * 2023.4.21 - load cce symbols.
 * 2024.1.24 - Change rt*** to aclrt***.
 */

#include "runtime/cce/cce_module.h"

#include <runtime/file_util.h>
#include <runtime/pack_args.h>
#include <runtime/thread_storage_scope.h>
#include <tvm/runtime/registry.h>

#include <mutex>

#include "runtime/cce/cce_common.h"
#include "runtime/cce/cce_acl.h"
#include "codegen/util.h"
#include <climits>
#include <dlfcn.h>

#ifdef USE_CCE_PROFILING
#include "profile_mgr.h"
#endif

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
    CceWrapper::GetInstance();
  }
  // destructor
  ~CceModuleNode() override {
    UnLoadKernelFunc();
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

  void *GetKernelFunc(const std::string &func_name) {
    const auto *f = Registry::Get("get_kernel_meta_path");
    CHECK(f != nullptr) << "Function get_kernel_meta_path is not registered";
    std::string file_str = (*f)().operator std::string();
    (void)file_str.append(func_name).append(".o");
    char *file_c_str = (char *)file_str.c_str();

    void *handle = dlopen(file_c_str, RTLD_LAZY | RTLD_LOCAL);
    CHECK(handle != nullptr) << "dlopen failed, file: " << file_c_str;

    std::string func_str = func_name + "_do";
    char *func_c_str = (char *)func_str.c_str();
    void *func = dlsym(handle, func_c_str);
    CHECK(func != nullptr) << "dlsym failed, symbol: " << func_str;
    return func;
  }

  bool UnLoadKernelFunc() {
    if (cce_handle_ != nullptr) {
      if (dlclose(cce_handle_) != 0) {
        return false;
      }
    }
    cce_handle_ = nullptr;
    return true;
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
  void *cce_handle_{nullptr};
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
    int32_t device_id;
    CCE_CALL(aclrtGetDevice(&device_id));

    ThreadWorkLoad wl = thread_axis_cfg_.Extract(args);
    int blockDim = static_cast<int>(wl.grid_dim(0));
    auto strm = static_cast<aclrtStream>(CceThreadEntry::ThreadLocal()->stream);

    size_t raw_size = arg_size_.size() - shape_arg_size;
    void** raw_args = new void*[arg_size_.size()];
    size_t args_size = 0;
    for (size_t i = 0; i < raw_size; ++i) {
      args_size += arg_size_[i];
      void** ptr = reinterpret_cast<void**>(void_args[i]);
      raw_args[i] = *ptr;
    }

    aclError result;
    typedef void (*CallFunc)(uint32_t, void*, void*, void**);
    auto func_ptr = reinterpret_cast<CallFunc>(m_->GetKernelFunc(func_name_));
    if (shape_arg_size == 0) {
      func_ptr(blockDim, nullptr, strm, raw_args);
    } else {
      result = ACL_SUCCESS;
      if (blockDim == INT_MAX) {
        blockDim = shape_args[shape_arg_size - 1];
      }

      for (size_t ssize = raw_size; ssize < arg_size_.size(); ++ssize) {
        void* tempshape = reinterpret_cast<void*> (shape_args[ssize - raw_size]);
        raw_args[ssize] = tempshape;
        args_size += 8;
      }

      func_ptr(blockDim, nullptr, strm, raw_args);
      akg::RecordCore(blockDim, true);
    }

#ifdef USE_CCE_PROFILING    
    uint32_t stream_id;
    uint32_t task_id;
    auto rt_ret = rtGetTaskIdAndStreamID(&task_id, &stream_id);
    if (rt_ret != ACL_SUCCESS) {
      LOG(FATAL) << "Profiling get task_id stream_id failed";
    }
    auto label = std::to_string(stream_id) + "_" + std::to_string(task_id);
    ProfileMgr::GetInstance().SetKernelLabel(label);
#endif

    delete[] raw_args;
    if (result != ACL_SUCCESS) {
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
