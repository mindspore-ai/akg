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

#include <fstream>
#include "picojson.h"
#include <string.h>
#include <memory>
#include <vector>
#include <string>

#ifndef SRC_RUNTIME_ASCEND_KERNEL_H_
#define SRC_RUNTIME_ASCEND_KERNEL_H_

namespace air {
namespace runtime {

struct FlexArray {
  size_t len;
  char contents[];
};

struct KernelJsonInfo {
  std::string bin_file_name;
  std::string bin_file_suffix;
  uint32_t block_dim;
  std::string kernel_name;
  std::string magic;
  std::vector<size_t> parameters;
  std::string sha256;
  std::vector<size_t> workspaces;
  uint32_t op_para_size;
  KernelJsonInfo() : block_dim(0), op_para_size(0) {}
};

class KernelPack {
 public:
  KernelPack() : json_(nullptr), kernel_(nullptr) {}
  KernelPack(const KernelPack &) = default;
  KernelJsonInfo kernel_json_info() const;
  bool LoadKernelMeta(const std::string &json_f);
  const FlexArray *GetJson() const { return json_; }
  const FlexArray *GetKernel() const { return kernel_; }
  ~KernelPack() {
    if (json_) {
      delete[] json_;
      json_ = nullptr;
    }
    if (kernel_) {
      delete[] kernel_;
      kernel_ = nullptr;
    }
  }

 private:
  bool ReadFromJsonFileHelper(std::ifstream &kernelbin);
  void ParseKernelJson(const picojson::value::object &js);
  KernelJsonInfo kernel_json_info_;
  FlexArray *json_;
  FlexArray *kernel_;
};

using KernelPackPtr = std::shared_ptr<KernelPack>;

bool GetFuncStub(const KernelPack &kernel_pack, uint32_t *block_dim, std::string *func_name);
KernelPackPtr GetKernelPack(const std::string &kernel_name);

}  // namespace runtime
}  // namespace air

#endif  // SRC_RUNTIME_ASCEND_KERNEL_H_
