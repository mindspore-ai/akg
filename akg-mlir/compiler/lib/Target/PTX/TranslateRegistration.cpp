/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "akg/Target/PTX/Passes.h"
#include "akg/Target/PTX/ToPTX.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir {

void registerToPTXTranslation() {
  static llvm::cl::OptionCategory PTXCodeGenCat("PTX Codegen", "PTX codegen options");

  static llvm::cl::opt<std::string> kernelName("kernel-name", llvm::cl::desc("kernel name"), llvm::cl::init("unknown"),
                                               llvm::cl::cat(PTXCodeGenCat));

  static llvm::cl::opt<std::string> arch("arch", llvm::cl::desc("architecture"), llvm::cl::init("sm_70"),
                                         llvm::cl::cat(PTXCodeGenCat));

  TranslateFromMLIRRegistration reg(
    "gen-ptx", "generate ptx code",
    [](ModuleOp module, raw_ostream &output) { return mlir::translateToPTX(module, output, kernelName, arch); },
    [](DialectRegistry &registry) {
      registerAllDialects(registry);
      registerLLVMDialectTranslation(registry);
      registerNVVMDialectTranslation(registry);
    });
}

}  // namespace mlir

