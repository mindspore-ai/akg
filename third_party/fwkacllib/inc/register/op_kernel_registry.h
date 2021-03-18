/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef INC_REGISTER_OP_KERNEL_REGISTRY_H_
#define INC_REGISTER_OP_KERNEL_REGISTRY_H_
#include <memory>
#include <string>
#include "register/register_types.h"
#include "register.h"

namespace ge {
class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY OpKernelRegistry {
 public:
  using CreateFn = HostCpuOp* (*)();
  ~OpKernelRegistry();

  static OpKernelRegistry& GetInstance() {
    static OpKernelRegistry instance;
    return instance;
  }

  bool IsRegistered(const std::string &op_type);

  void RegisterHostCpuOp(const std::string &op_type, CreateFn create_fn);

  std::unique_ptr<HostCpuOp> CreateHostCpuOp(const std::string &op_type);

 private:
  OpKernelRegistry();
  class OpKernelRegistryImpl;
  std::unique_ptr<OpKernelRegistryImpl> impl_;
};
} // namespace ge

#endif // INC_REGISTER_OP_KERNEL_REGISTRY_H_
