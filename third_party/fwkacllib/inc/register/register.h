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

#ifndef INC_REGISTER_REGISTRY_H_
#define INC_REGISTER_REGISTRY_H_

#include "external/register/register.h"

namespace ge {
class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY HostCpuOp {
 public:
  HostCpuOp() = default;
  virtual ~HostCpuOp() = default;

  virtual graphStatus Compute(Operator &op,
                              const std::map<std::string, const Tensor> &inputs,
                              std::map<std::string, Tensor> &outputs) = 0;
};

class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY HostCpuOpRegistrar {
 public:
  HostCpuOpRegistrar(const char *op_type, HostCpuOp *(*create_fn)());
  ~HostCpuOpRegistrar() = default;
};

#define REGISTER_HOST_CPU_OP_BUILDER(name, op) \
    REGISTER_HOST_CPU_OP_BUILDER_UNIQ_HELPER(__COUNTER__, name, op)

#define REGISTER_HOST_CPU_OP_BUILDER_UNIQ_HELPER(ctr, name, op) \
    REGISTER_HOST_CPU_OP_BUILDER_UNIQ(ctr, name, op)

#define REGISTER_HOST_CPU_OP_BUILDER_UNIQ(ctr, name, op)              \
  static ::ge::HostCpuOpRegistrar register_host_cpu_op##ctr           \
      __attribute__((unused)) =                                       \
          ::ge::HostCpuOpRegistrar(name, []()->::ge::HostCpuOp* {   \
            return new (std::nothrow) op();                           \
          })
} // namespace ge

#endif //INC_REGISTER_REGISTRY_H_
