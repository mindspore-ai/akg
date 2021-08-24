/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <dmlc/common.h>
#include "runtime/rt.h"
#include "kernel.h"

namespace air {
namespace runtime {
constexpr auto kCceKernelMeta = "./kernel_meta/";
constexpr auto kJsonSuffix = ".json";
static uintptr_t kernel_stub_gen_ = 0;

inline size_t LongToSize(int64_t u) {
  if (u < 0) {
    LOG(FATAL) << "The int64_t value(" << u << ") is less than 0.";
  }
  return static_cast<size_t>(u);
}

inline int64_t SizeToLong(size_t u) {
  if (u > static_cast<size_t>((std::numeric_limits<int64_t>::max)())) {
    LOG(FATAL) << "The size_t value(" << u << ") exceeds the maximum value of int64_t.";
  }
  return static_cast<int64_t>(u);
}

bool KernelPack::ReadFromJsonFileHelper(std::ifstream &kernelbin) {
  size_t binsize = LongToSize(kernelbin.seekg(0, std::ios::end).tellg());
  // free old data
  if (kernel_ != nullptr) {
    delete[] kernel_;
    kernel_ = nullptr;
  }

  void *ptr = static_cast<void *>(new (std::nothrow) uint8_t[sizeof(KernelPack) + binsize]);
  if (ptr != nullptr) {
    kernel_ = static_cast<FlexArray *>(ptr);
  }
  if (kernel_ == nullptr) {
    LOG(FATAL) << "memory malloc failed.";
    kernelbin.close();
    return false;
  }
  memset(kernel_, 0, sizeof(KernelPack) + binsize);
  kernel_->len = binsize;
  (void)kernelbin.seekg(0, std::ios::beg);
  (void)kernelbin.read(kernel_->contents, SizeToLong(kernel_->len));
  return true;
}

void KernelPack::ParseKernelJson(const picojson::value::object &js) {
  kernel_json_info_.bin_file_name = js.at("binFileName").get<std::string>();
  kernel_json_info_.bin_file_suffix = js.at("binFileSuffix").get<std::string>();
  kernel_json_info_.block_dim = static_cast<uint32_t>(js.at("blockDim").get<int64_t>());
  kernel_json_info_.kernel_name = js.at("kernelName").get<std::string>();
  kernel_json_info_.magic = js.at("magic").get<std::string>();
  if (js.count("opParaSize")) {
    kernel_json_info_.op_para_size = static_cast<uint32_t>(js.at("opParaSize").get<int64_t>());
  }
  if (js.find("parameters") != js.end()) {
    if (!js.at("parameters").is<picojson::array>()) {
      LOG(DEBUG) << "Format error!,parameters should be array.";
    }
    picojson::array sizes = js.at("parameters").get<picojson::array>();
    for (auto size : sizes) {
      kernel_json_info_.parameters.push_back(size.get<int64_t>());
    }
  }
  kernel_json_info_.sha256 = js.at("sha256").get<std::string>();
}

bool KernelPack::LoadKernelMeta(const std::string &json_f) {
  if (json_f.length() <= strlen(kJsonSuffix)) {
    LOG(FATAL) << "please check json path.";
    return false;
  }
  std::ifstream kernel_json(json_f);
  if (!kernel_json.is_open()) {
    LOG(INFO) << "Open json file: " << json_f << " error, please check kernel_meta.";
    return false;
  }
  picojson::value js;
  std::string err = picojson::parse(js, kernel_json);
  CHECK(err.empty()) << "json parse error, error message: " << err;
  kernel_json.close();
  ParseKernelJson(js.get<picojson::object>());

  auto bin_file_suffix = ".o";
  std::string bin_f = json_f.substr(0, json_f.length() - 5) + bin_file_suffix;

  std::ifstream kernelbin(bin_f, std::ios::binary);
  if (!kernelbin.is_open()) {
    LOG(FATAL) << "read kernel binary file error, please check kernelmeta.";
    return false;
  }

  if (!ReadFromJsonFileHelper(kernelbin)) {
    return false;
  }
  return true;
}

KernelJsonInfo KernelPack::kernel_json_info() const { return kernel_json_info_; }

uintptr_t GetFuncStub(const KernelPack &kernel_pack, uint32_t *block_dim) {
  auto kernel = kernel_pack.GetKernel();
  if (kernel == nullptr) {
    LOG(FATAL) << "Invalid kernel pack, json or kernel is nullptr.";
  }
  auto kernel_contents = kernel->contents;
  if (kernel_contents == nullptr) {
    LOG(FATAL) << "Invalid kernel context, json or kernel is nullptr.";
  }
  auto kernel_json_info = kernel_pack.kernel_json_info();

  *block_dim = kernel_json_info.block_dim;
  std::string func_name = kernel_json_info.kernel_name;
  std::string magic = kernel_json_info.magic;

  // BinaryRegister
  void *module = nullptr;
  rtDevBinary_t devBin;
  devBin.magic = RT_DEV_BINARY_MAGIC_ELF;
  devBin.version = 0;
  devBin.length = kernel->len;
  devBin.data = kernel->contents;
  static_cast<void>(rtDevBinaryRegister(&devBin, &module));

  // to diff different funcs.
  uintptr_t func_stub = ++kernel_stub_gen_;
  static_cast<void>(
    rtFunctionRegister(module, reinterpret_cast<void *>(func_stub), func_name.c_str(), func_name.c_str(), 0));

  return func_stub;
}

KernelPackPtr GetKernelPack(const std::string &kernel_name) {
  std::string cce_json = kCceKernelMeta;
  (void)cce_json.append(kernel_name).append(kJsonSuffix);
  KernelPackPtr ret = std::make_shared<KernelPack>();
  if (!ret->LoadKernelMeta(cce_json)) {
    LOG(INFO) << "Read cache json and bin file failed[" << cce_json << "]";
    return nullptr;
  }
  return ret;
}
}  // namespace runtime
}  // namespace air
