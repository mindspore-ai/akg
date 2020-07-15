/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "contrib/cce_parm/cceconf.h"

namespace akg {
namespace ir {
using air::runtime::TVMArgs;
using air::runtime::TVMRetValue;

TVM_REGISTER_API("tvm.info.mem.local.L1").set_body([](const TVMArgs args, TVMRetValue *ret) {
  cceconf::CceConf *conf = cceconf::CceConf::getInstance();
  CHECK(conf);

  auto node = air::make_node<air::MemoryInfoNode>();
  node->unit_bits = 2 * 16 * 16 * 8;
  node->max_simd_bits = 2 * 16 * 16 * 8;
  node->max_num_bits = conf->getBufferValue("L1_Buffer") * 8;
  node->head_address = air::make_const(air::Int(32), 0);
  *ret = air::MemoryInfo(node);
});

TVM_REGISTER_API("tvm.info.mem.local.UB").set_body([](const TVMArgs args, TVMRetValue *ret) {
  cceconf::CceConf *conf = cceconf::CceConf::getInstance();
  CHECK(conf);

  auto node = air::make_node<air::MemoryInfoNode>();
  node->unit_bits = 32 * 8;
  node->max_simd_bits = 32 * 8;
  node->max_num_bits = conf->getBufferValue("Unified_Buffer") * 8;
  node->head_address = air::make_const(air::Int(32), 0);
  *ret = air::MemoryInfo(node);
});

TVM_REGISTER_API("tvm.info.mem.local.L0A").set_body([](const TVMArgs args, TVMRetValue *ret) {
  cceconf::CceConf *conf = cceconf::CceConf::getInstance();
  CHECK(conf);

  auto node = air::make_node<air::MemoryInfoNode>();
  node->unit_bits = 2 * 16 * 16 * 8;
  node->max_simd_bits = 2 * 16 * 16 * 8;
  node->max_num_bits = conf->getBufferValue("L0A_Buffer") * 8;
  node->head_address = air::make_const(air::Int(32), 0);
  *ret = air::MemoryInfo(node);
});

TVM_REGISTER_API("tvm.info.mem.local.L0B").set_body([](const TVMArgs args, TVMRetValue *ret) {
  cceconf::CceConf *conf = cceconf::CceConf::getInstance();
  CHECK(conf);

  auto node = air::make_node<air::MemoryInfoNode>();
  node->unit_bits = 2 * 16 * 16 * 8;
  node->max_simd_bits = 2 * 16 * 16 * 8;
  node->max_num_bits = conf->getBufferValue("L0B_Buffer") * 8;
  node->head_address = air::make_const(air::Int(32), 0);
  *ret = air::MemoryInfo(node);
});

TVM_REGISTER_API("tvm.info.mem.local.L0C").set_body([](const TVMArgs args, TVMRetValue *ret) {
  cceconf::CceConf *conf = cceconf::CceConf::getInstance();
  CHECK(conf);

  auto node = air::make_node<air::MemoryInfoNode>();
  node->unit_bits = 2 * 16 * 16 * 8;
  node->max_simd_bits = 2 * 16 * 16 * 8;
  node->max_num_bits = conf->getBufferValue("L0C_Buffer") * 8;
  node->head_address = air::make_const(air::Int(32), 0);
  *ret = air::MemoryInfo(node);
});

TVM_REGISTER_API("tvm.info.mem.local.REG").set_body([](const TVMArgs args, TVMRetValue *ret) {
  auto node = air::make_node<air::MemoryInfoNode>();
  node->unit_bits = 16;
  node->max_simd_bits = 64;
  node->max_num_bits = 64 * 3200;
  node->head_address = air::make_const(air::Int(32), 0);
  *ret = air::MemoryInfo(node);
});

TVM_REGISTER_API("tvm.info.mem.local_aicpu").set_body([](const TVMArgs args, TVMRetValue *ret) {
  auto node = air::make_node<air::MemoryInfoNode>();
  node->unit_bits = 16;
  node->max_simd_bits = 64;
  node->max_num_bits = 16 * 1024 * 1024;
  node->head_address = air::make_const(air::Int(32), 0);
  *ret = air::MemoryInfo(node);
});

TVM_REGISTER_API("tvm.info.mem.L1_tmp").set_body([](const TVMArgs args, TVMRetValue *ret) {
  auto node = air::make_node<air::MemoryInfoNode>();
  node->unit_bits = 2 * 16 * 16 * 8;
  node->max_simd_bits = 2 * 16 * 16 * 8;
  node->max_num_bits = 1024 * 1024 * 1024;
  node->head_address = air::make_const(air::Int(32), 0);
  *ret = air::MemoryInfo(node);
});
}  // namespace ir
}  // namespace akg
