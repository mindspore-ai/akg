/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
  std::string device_type = akg::common::GetStringEnv("GPU_DEVICE_TYPE");
  if (device_type.empty() && args.size() == 1) {
    auto expr = Expr(args[0]);
    if (expr.as<StringImm>()) {
      device_type = std::string(expr.as<StringImm>()->value);
    }
  }
  if (device_type.empty()) {
    device_type = "v100";
  }
  int default_mem = -1;
  int max_mem = -1;
  if (device_type == "v100") {
    default_mem = 48 * 1024;
    max_mem = 96 * 1024;
  } else if (device_type == "a100") {
    default_mem = 64 * 1024;
    max_mem = 128 * 1024;
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
  std::string device_type = akg::common::GetStringEnv("GPU_DEVICE_TYPE");
  if (device_type.empty() && args.size() == 1) {
    auto expr = Expr(args[0]);
    if (expr.as<StringImm>()) {
      device_type = std::string(expr.as<StringImm>()->value);
    }
  }
  if (device_type.empty()) {
    device_type = "v100";
  }
  int default_mem = -1;
  if (device_type == "v100") {
    default_mem = 64 * 1024;
  } else if (device_type == "a100") {
    default_mem = 64 * 1024;
  }
  CHECK_NE(default_mem, -1) << "Invalid query for memory on " << device_type;

  auto node = air::make_node<air::GpuMemoryInfoNode>();
  node->max_bytes_per_block = default_mem;
  *ret = air::GpuMemoryInfo(node);
});

TVM_REGISTER_API("gpu.info.compute.instance").set_body([](const TVMArgs args, TVMRetValue *ret) {
  std::string device_type = akg::common::GetStringEnv("GPU_DEVICE_TYPE");
  if (device_type.empty() && args.size() == 1) {
    auto expr = Expr(args[0]);
    if (expr.as<StringImm>()) {
      device_type = std::string(expr.as<StringImm>()->value);
    }
  }
  if (device_type.empty()) {
    device_type = "v100";
  }
  int abps = akg::common::GetIntegerEnv("abps");
  int io = akg::common::GetIntegerEnv("io");

  device_type = device_type.empty() ? "v100" : device_type;
  int num_sm = -1;
  int active_blocks_per_sm = -1;
  int min_elem_for_io_bound = -1;
  if (device_type == "v100") {
    num_sm = 80;
    active_blocks_per_sm = 5;
    min_elem_for_io_bound = 2;
  } else if (device_type == "a100") {
    num_sm = 108;
    active_blocks_per_sm = abps == 0 ? 10 : abps;
    min_elem_for_io_bound = io == 0 ? 2 : io;
  }
  CHECK_NE(num_sm, -1) << "Invalid query for compute ability on " << device_type;
  CHECK_NE(active_blocks_per_sm, -1) << "Invalid query for compute ability on " << device_type;
  CHECK_NE(min_elem_for_io_bound, -1) << "Invalid query for compute ability on " << device_type;

  auto node = air::make_node<air::GpuComputeInfoNode>();
  node->num_sm = num_sm;
  node->active_blocks_per_sm = active_blocks_per_sm;
  node->min_elem_for_io_bound = min_elem_for_io_bound;
  *ret = air::GpuComputeInfo(node);
});

}  // namespace ir
}  // namespace akg
