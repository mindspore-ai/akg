/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <tvm/api_registry.h>
#include <tvm/expr_operator.h>
#include <tvm/target_info.h>

#include "codegen/pass_mgr.h"
#include "common/target_info.h"
#include "common/common_util.h"

namespace akg {
namespace ir {
using air::runtime::TVMArgs;
using air::runtime::TVMRetValue;

TVM_REGISTER_API("gpu.info.mem.shared").set_body([](const TVMArgs args, TVMRetValue *ret) {
  std::string device_type = akg::common::GetStringEnv("AKG_DEVICE_TYPE");
  device_type = device_type.empty() ? "v100" : device_type;
  int default_mem = -1;
  int max_mem = -1;
  if (device_type == "v100") {
    default_mem = 48 * 1024;
    max_mem = 96 * 1024;
  }
  CHECK_NE(default_mem, -1) << "Invalid query for memory on " << device_type;

  int conf_mem = akg::common::GetIntegerEnv("AKG_SHARED_MEM");
  CHECK_LE(conf_mem, max_mem) << "Invalid config for memory on " << device_type << ": max " << max_mem << " vs "
                              << conf_mem;

  auto node = air::make_node<air::GpuMemoryInfoNode>();
  node->max_bytes_per_block = conf_mem == 0 ? default_mem : conf_mem;
  *ret = air::GpuMemoryInfo(node);
});

TVM_REGISTER_API("gpu.info.mem.reg").set_body([](const TVMArgs args, TVMRetValue *ret) {
  std::string device_type = akg::common::GetStringEnv("AKG_DEVICE_TYPE");
  device_type = device_type.empty() ? "v100" : device_type;
  int default_mem = -1;
  if (device_type == "v100") {
    default_mem = 64 * 1024;
  }
  CHECK_NE(default_mem, -1) << "Invalid query for memory on " << device_type;

  auto node = air::make_node<air::GpuMemoryInfoNode>();
  node->max_bytes_per_block = default_mem;
  *ret = air::GpuMemoryInfo(node);
});

}  // namespace ir
}  // namespace akg
