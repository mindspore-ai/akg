/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "contrib/cce_parm/cceconf.h"
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

#include <tvm/base.h>
#include <tvm/api_registry.h>

using std::map;
using std::string;

namespace akg {
namespace cceconf {
using air::runtime::TVMArgs;
using air::runtime::TVMRetValue;

CceConf CceConf::instance = CceConf();

CceConf *CceConf::getInstance() { return &instance; }

CceConf::CceConf() {
  // init the debug_switch_
  debug_switch_ = false;

  // init the cloud params
  map<string, int> cloud_buffer_map = {
    {"L0A_Buffer", 64 * 1024},      {"L0B_Buffer", 64 * 1024},       {"L0C_Buffer", 256 * 1024},
    {"L1_Buffer", 1 * 1024 * 1024}, {"L2_Buffer", 32 * 1024 * 1024}, {"Unified_Buffer", 256 * 1024},
  };

  map<string, string> cloud_compiler_map = {
    {"Compiler_arch", "dav-c100"},
    // Using lowercase true to turn on this feature,
    // when aicpu has deployed os. It will be converted
    // to boolean in python side
    {"Compiler_aicpu_support_os", "true"},
  };

  map<string, string> cloud_intrinsic_map = {
    {"Intrinsic_vrec", "float16,float32"},
    {"Intrinsic_vadd", "float16,float32,int32"},
    {"Intrinsic_vadds", "float16,float32"},
    {"Intrinsic_vsub", "float16,float32,int32"},
    {"Intrinsic_vmul", "float16,float32,int32"},
    {"Intrinsic_vmax", "float16,float32,int32"},
    {"Intrinsic_vmin", "float16,float32,int32"},
    {"Intrinsic_vlog", "float16,float32"},
    {"Intrinsic_vexp", "float16,float32"},
    {"Intrinsic_vmuls", "float16,float32"},
    {"Intrinsic_vabs", "float16,float32"},
    {"Intrinsic_vcmax", "float16"},
    {"Intrinsic_vcgmax", "float16"},
    {"Intrinsic_vcmin", "float16"},
    {"Intrinsic_vcgmin", "float16"},
    {"Intrinsic_vcadd", "float16,float32"},
    {"Intrinsic_vcgadd", "float16"},
    {"Intrinsic_vcmp", "float16,float32"},
    {"Intrinsic_vconv",
     "f322f16,f162f32,f322f16o,f162s8,f162s8a,f162s8f,f162s8c,f162s8z,f162u8,f162u8a,f162u8f,f162u8c,f162u8z,deq,\
s322f32,f162s32r,f162s32a,f162s32f,f162s32c,f162s32z,f322s32r,f322s32a,f322s32f,f322s32c,f322s32z,u82f16,s82f16"},
    {"Intrinsic_mmad", "u8,s8,f162f16,f162f32,f16u2,u8s8"},
    {"Intrinsic_vor", "int16,uint16"},
    {"Intrinsic_vand", "int16,uint16"},
    {"Intrinsic_vaxpy", "float16,float32"},
    {"Intrinsic_vnot", "int16,uint16"},
    {"Intrinsic_vsqrt", "float16,float32"},
    {"Intrinsic_vrelu", "float16"},
    {"Intrinsic_vmla", "float16,float32"},
    {"Intrinsic_vmadd", "float16,float32"},
    {"Intrinsic_vmaddrelu", "float16,float32"},
    {"Intrinsic_round", "float16,float32"},
    {"Intrinsic_floor", "float16,float32"},
    {"Intrinsic_ceil", "float16,float32"},
    {"Intrinsic_trunc", "float16,float32"},
  };
  map<string, int> cloud_core_map = {
    {"Core_num", 32},
  };
  CceParam cloudParam{"cloud", cloud_buffer_map, cloud_compiler_map, cloud_intrinsic_map, cloud_core_map};

  kvs_["cloud"] = cloudParam;

  // init the mini params
  map<string, int> mini_buffer_map = {
    {"L0A_Buffer", 64 * 1024},      {"L0B_Buffer", 64 * 1024},      {"L0C_Buffer", 256 * 1024},
    {"L1_Buffer", 1 * 1024 * 1024}, {"L2_Buffer", 8 * 1024 * 1024}, {"Unified_Buffer", 248 * 1024},
  };
  map<string, string> mini_compiler_map = {
    {"Compiler_arch", "dav-m100"},
    // Using lowercase true to turn on this feature,
    // when aicpu has deployed os. It will be converted
    // to boolean in python side
    {"Compiler_aicpu_support_os", "true"},
  };

  map<string, string> mini_intrinsic_map = {
    {"Intrinsic_vrec", "float16,float32"},
    {"Intrinsic_vadd", "float16,float32,int32"},
    {"Intrinsic_vadds", "float16,float32"},
    {"Intrinsic_vsub", "float16,float32,int32"},
    {"Intrinsic_vmul", "float16,float32,int32"},
    {"Intrinsic_vmax", "float16,float32,int32"},
    {"Intrinsic_vmin", "float16,float32,int32"},
    {"Intrinsic_vlog", "float16"},
    {"Intrinsic_vexp", "float16"},
    {"Intrinsic_vmuls", "float16,float32"},
    {"Intrinsic_vabs", "float16,float32"},
    {"Intrinsic_vcmax", "float16"},
    {"Intrinsic_vcgmax", "float16"},
    {"Intrinsic_vcmin", "float16"},
    {"Intrinsic_vcgmin", "float16"},
    {"Intrinsic_vcadd", "float16,float32"},
    {"Intrinsic_vcgadd", "float16"},
    {"Intrinsic_vcmp", "float16"},
    {"Intrinsic_vconv", "f322f16,f162f32,f162s8,f162u8,deq,f162s32f,f162s32c,f162s32r,u82f16,s82f16"},
    {"Intrinsic_mmad", "u8,s8,f162f16,f162f32,f16u2,u8s8"},
    {"Intrinsic_vor", "int16,uint16"},
    {"Intrinsic_vand", "int16,uint16"},
    {"Intrinsic_vaxpy", "float16,float32"},
    {"Intrinsic_vnot", "int16,uint16"},
    {"Intrinsic_vrelu", "float16"},
    {"Intrinsic_vmla", "float16,float32"},
    {"Intrinsic_vmadd", "float16,float32"},
    {"Intrinsic_vmaddrelu", "float16,float32"},
    {"Intrinsic_round", "float16"},
    {"Intrinsic_floor", "float16"},
    {"Intrinsic_ceil", "float16"},
    {"Intrinsic_trunc", "float16"},
  };
  map<string, int> mini_core_map = {
    {"Core_num", 2},
  };
  CceParam miniParam{"mini", mini_buffer_map, mini_compiler_map, mini_intrinsic_map, mini_core_map};

  kvs_["mini"] = miniParam;

  // init the lite-phoenix params
  map<string, int> phoenix_buffer_map = {
    {"L0A_Buffer", 32 * 1024}, {"L0B_Buffer", 32 * 1024},      {"L0C_Buffer", 128 * 1024},
    {"L1_Buffer", 512 * 1024}, {"L2_Buffer", 1 * 1024 * 1024}, {"Unified_Buffer", 120 * 1024},
  };
  map<string, string> lite_compiler_map = {
    {"Compiler_arch", "dav-l100"},
    // Aicpu has no os support now for lite and
    // `false` will be converted to boolean in python side
    {"Compiler_aicpu_support_os", "false"},
  };

  map<string, string> lite_intrinsic_map = {
    {"Intrinsic_vrec", "float16"},
    {"Intrinsic_vadd", "float16,int32"},
    {"Intrinsic_vadds", "float16"},
    {"Intrinsic_vsub", "float16,int32"},
    {"Intrinsic_vmul", "float16,int32"},
    {"Intrinsic_vmax", "float16,int32"},
    {"Intrinsic_vmin", "float16,int32"},
    {"Intrinsic_vlog", "float16"},
    {"Intrinsic_vexp", "float16"},
    {"Intrinsic_vmuls", "float16"},
    {"Intrinsic_vabs", "float16"},
    {"Intrinsic_vcmax", "float16"},
    {"Intrinsic_vcgmax", "float16"},
    {"Intrinsic_vcmin", "float16"},
    {"Intrinsic_vcgmin", "float16"},
    {"Intrinsic_vcadd", "float16"},
    {"Intrinsic_vcgadd", "float16"},
    {"Intrinsic_vcmp", "float16"},
    {"Intrinsic_vconv",
     "f162s8,f162u8,deq,f162s32f,f162s32c,f162s32r,u82f16,s82f16,f162s32a,f162s32z,f162s8a,f162s8f,f162s8c,f162s8z,\
f162u8a,f162u8f,f162u8c,f162u8z"},
    {"Intrinsic_mmad", "u8,s8,b8u2,f16u2,u8s8,f162f16"},
    {"Intrinsic_vor", "int16,uint16"},
    {"Intrinsic_vand", "int16,uint16"},
    {"Intrinsic_vaxpy", "float16"},
    {"Intrinsic_vnot", "int16,uint16"},
    {"Intrinsic_vrelu", "float16"},
    {"Intrinsic_vmla", "float16"},
    {"Intrinsic_vmadd", "float16"},
    {"Intrinsic_vmaddrelu", "float16"},
    {"Intrinsic_round", "float16"},
    {"Intrinsic_floor", "float16"},
    {"Intrinsic_ceil", "float16"},
    {"Intrinsic_trunc", "float16"},
  };
  map<string, int> phoenix_core_map = {
    {"Core_num", 1},
  };
  CceParam phoenixParam{"phoenix", phoenix_buffer_map, lite_compiler_map, lite_intrinsic_map, phoenix_core_map};

  kvs_["phoenix"] = phoenixParam;

  // init the lite-orlando params
  map<string, int> orlando_buffer_map = {
    {"L0A_Buffer", 32 * 1024}, {"L0B_Buffer", 32 * 1024}, {"L0C_Buffer", 128 * 1024},
    {"L1_Buffer", 512 * 1024}, {"L2_Buffer", 512 * 1024}, {"Unified_Buffer", 120 * 1024},
  };
  map<string, int> orlando_core_map = {
    {"Core_num", 1},
  };
  CceParam orlandoParam{"orlando", orlando_buffer_map, lite_compiler_map, lite_intrinsic_map, orlando_core_map};

  kvs_["orlando"] = orlandoParam;
}

string CceConf::getProductName(const string &section) const {
  if (section == "1.6") {
    return "cloud";
  } else if (section == "1.1") {
    return "mini";
  } else if (section == "3.5") {
    return "phoenix";
  } else if (section == "3.3") {
    return "orlando";
  } else {
    return section;
  }
}

int CceConf::getBufferValue(const string &section, const string &key) const {
  string name = getProductName(section);

  auto it = kvs_.find(name);
  if (it != kvs_.end()) {
    CceParam params = it->second;
    return params.getBufferValue(key);
  }
  return 0;
}

int CceConf::getBufferValue(const string &key) const { return getBufferValue(section_, key); }

string CceConf::getCompilerValue(const string &section, const string &key) const {
  string name = getProductName(section);

  auto it = kvs_.find(name);
  if (it != kvs_.end()) {
    CceParam params = it->second;
    return params.getCompilerValue(key);
  }
  return "";
}

std::string CceConf::getCompilerValue(const string &key) const { return getCompilerValue(section_, key); }

string CceConf::getIntrinsicValue(const string &section, const string &key) const {
  string name = getProductName(section);

  auto it = kvs_.find(name);
  if (it != kvs_.end()) {
    CceParam params = it->second;
    return params.getIntrinsicValue(key);
  }
  return "";
}

int CceConf::getCoreValue(const string &section, const string &key) const {
  string name = getProductName(section);

  auto it = kvs_.find(name);
  if (it != kvs_.end()) {
    CceParam params = it->second;
    return params.getCoreValue(key);
  }
  return 0;
}

int CceConf::getCoreValue(const string &key) const { return getCoreValue(section_, key); }

std::string CceConf::getSection() const { return this->section_; }

void CceConf::setDebugSwitch(bool sw) { this->debug_switch_ = sw; }

bool CceConf::getDebugSwitch() const { return this->debug_switch_; }

string CceConf::getProductName() const { return getProductName(section_); }

void CceConf::release() noexcept { kvs_.clear(); }

TVM_REGISTER_API("cce.product_conf_buffer").set_body([](const TVMArgs args, TVMRetValue *rv) {
  CceConf *conf = CceConf::getInstance();

  *rv = conf->getBufferValue(args[0], args[1]);
});

TVM_REGISTER_API("cce.product_conf_compiler").set_body([](const TVMArgs args, TVMRetValue *rv) {
  CceConf *conf = CceConf::getInstance();

  *rv = conf->getCompilerValue(args[0], args[1]);
});

TVM_REGISTER_API("cce.product_conf_intrinsic").set_body([](const TVMArgs args, TVMRetValue *rv) {
  CceConf *conf = CceConf::getInstance();

  *rv = conf->getIntrinsicValue(args[0], args[1]);
});

TVM_REGISTER_API("cce.product_conf_core").set_body([](const TVMArgs args, TVMRetValue *rv) {
  CceConf *conf = CceConf::getInstance();

  *rv = conf->getCoreValue(args[0], args[1]);
});

TVM_REGISTER_API("cce.status_check").set_body([](const TVMArgs args, TVMRetValue *rv) {
  CceConf *conf = CceConf::getInstance();
  conf->setDebugSwitch(args[0]);
});

TVM_REGISTER_API("cce.set_product_section").set_body([](const TVMArgs args, TVMRetValue *rv) {
  CceConf *conf = CceConf::getInstance();
  const std::string section = args[0];
  conf->setSection(section);
});
}  // namespace cceconf
}  // namespace akg
