// Copyright 2023 Huawei Technologies Co., Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// ===----------------------------------------------------------------------===//
// Some code comes from TranslateRegistration.cpp in LLVM project
// Original license:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "akg/Dialect/MindSpore/IR/MindSporeOps.h"
#include "akg/Target/MindsporeDialect/ToMindsporeDialect.h"
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

void registerToMindsporeDialectTranslation() {
  static llvm::cl::opt<std::string> output_name("o-mlir", llvm::cl::desc("output mlir name"), llvm::cl::init(""));
  TranslateToMLIRRegistration reg(
    "json-to-mindspore", "convert Mindspore json file to mlir",
    [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
      return mlir::translateToMindsporeDialect(sourceMgr, context, output_name);
    },
    [](DialectRegistry &) {});
}

}  // namespace mlir
