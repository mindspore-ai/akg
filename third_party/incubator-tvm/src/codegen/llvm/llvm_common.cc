/*
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
 * \file llvm_common.cc
 */

/*
 * 2021.11.01
 *   Adapt LLVM 12 interface support
 * 2023.08.05
 *   Adapt LLVM 15 interface support
 */

#ifdef TVM_LLVM_VERSION

#include "llvm_common.h"

#include <tvm/base.h>

#include <atomic>
#include <memory>
#include <mutex>

namespace air {
namespace codegen {

struct LLVMEnv {
  std::mutex mu;
  std::atomic<bool> all_initialized{false};

  static LLVMEnv* Global() {
    static LLVMEnv inst;
    return &inst;
  }
};

void InitializeLLVM() {
  LLVMEnv* e = LLVMEnv::Global();
  if (!e->all_initialized.load(std::memory_order::memory_order_acquire)) {
    std::lock_guard<std::mutex> lock(e->mu);
    if (!e->all_initialized.load(std::memory_order::memory_order_acquire)) {
      llvm::InitializeAllTargetInfos();
      llvm::InitializeAllTargets();
      llvm::InitializeAllTargetMCs();
      llvm::InitializeAllAsmParsers();
      llvm::InitializeAllAsmPrinters();
      e->all_initialized.store(true, std::memory_order::memory_order_release);
    }
  }
}

void ParseLLVMTargetOptions(const std::string& target_str, std::string* triple, std::string* mcpu,
                            std::string* mattr, llvm::TargetOptions* options) {
  // setup target triple
  size_t start = 0;
  if (target_str.length() >= 4 && target_str.substr(0, 4) == "llvm") {
    start = 4;
  }
  // simple parser
  triple->resize(0);
  mcpu->resize(0);
  mattr->resize(0);

  bool soft_float_abi = false;
  std::string key, value;
  std::istringstream is(target_str.substr(start, target_str.length() - start));

  while (is >> key) {
    if (key == "--system-lib" || key == "-system-lib") {
      continue;
    }
    size_t pos = key.find('=');
    if (pos != std::string::npos) {
      CHECK_GE(key.length(), pos + 1) << "invalid argument " << key;
      value = key.substr(pos + 1, key.length() - 1);
      key = key.substr(0, pos);
    } else {
      CHECK(is >> value) << "Unspecified value for option " << key;
    }
    if (key == "-target" || key == "-mtriple") {
      *triple = value;
    } else if (key == "-mcpu") {
      *mcpu = value;
    } else if (key == "-mattr") {
      *mattr = value;
    } else if (key == "-mfloat-abi") {
      if (value == "hard") {
#if TVM_LLVM_VERSION < 60
        LOG(FATAL) << "-mfloat-abi hard is only supported for LLVM > 6.0";
#endif
        soft_float_abi = false;
      } else if (value == "soft") {
        soft_float_abi = true;
      } else {
        LOG(FATAL) << "invalid -mfloat-abi option " << value;
      }
    } else if (key == "-device" || key == "-libs" || key == "-model") {
      // pass
    } else {
      LOG(FATAL) << "unknown option " << key;
    }
  }

  if (triple->length() == 0 || *triple == "default") {
    *triple = llvm::sys::getDefaultTargetTriple();
  }
  // set target option
  llvm::TargetOptions& opt = *options;
  opt = llvm::TargetOptions();
#if TVM_LLVM_VERSION < 50
  opt.LessPreciseFPMADOption = true;
#endif
  opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  opt.UnsafeFPMath = false;
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  if (soft_float_abi) {
    opt.FloatABIType = llvm::FloatABI::Soft;
  } else {
    opt.FloatABIType = llvm::FloatABI::Hard;
  }
}

std::unique_ptr<llvm::TargetMachine> GetLLVMTargetMachine(const std::string& target_str,
                                                          bool allow_null) {
  std::string target_triple, mcpu, mattr;
  llvm::TargetOptions opt;

  ParseLLVMTargetOptions(target_str, &target_triple, &mcpu, &mattr, &opt);

  if (target_triple.length() == 0 || target_triple == "default") {
    target_triple = llvm::sys::getDefaultTargetTriple();
  }
  if (mcpu.length() == 0) {
    mcpu = "generic";
  }

  std::string err;
  const llvm::Target* target = llvm::TargetRegistry::lookupTarget(target_triple, err);
  if (target == nullptr) {
    CHECK(allow_null) << err << " target_triple=" << target_triple;
    return nullptr;
  }
  llvm::TargetMachine* tm =
      target->createTargetMachine(target_triple, mcpu, mattr, opt, llvm::Reloc::PIC_);
  return std::unique_ptr<llvm::TargetMachine>(tm);
}

}  // namespace codegen
}  // namespace air
#endif  // TVM_LLVM_VERSION
