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

#include "akg/Conversion/Passes.h"
#include "akg/Dialect/Affine/Passes.h"
#include "akg/Dialect/Fusion/IR/Fusion.h"
#include "akg/Dialect/GPU/Passes.h"
#include "akg/Dialect/LLVMIR/Passes.h"
#include "akg/Dialect/Linalg/IR/LinalgExtOps.h"
#include "akg/Dialect/Linalg/Passes.h"
#include "akg/Dialect/MindSpore/IR/MindSporeOps.h"
#include "akg/Dialect/MindSpore/Passes.h"
// #include "bishengir/Dialect/HACC/IR/HACC.h"
#include "akg/Dialect/SCF/Passes.h"
#include "akg/Pipelines/InitAllPipelines.h"
#include "akg/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  registerMindSporePasses();
  registerAKGAffinePasses();
  registerMindSporePasses();
  registerAKGLinalgPasses();
  registerAKGTransformsPasses();
  registerAKGIRConversionPasses();
  registerAKGLLVMIRPasses();
  registerAKGSCFPasses();
  registerAKGGPUPasses();

  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<mlir::linalgExt::LinalgExtDialect>();
  registry.insert<mlir::fusion::FusionDialect>();
  registry.insert<mlir::mindspore::MindSporeDialect>();
  // registry.insert<mlir::hacc::HACCDialect>();
  registerLLVMDialectTranslation(registry);

  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerConversionPasses();
  registerAllPiplines();
  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "AKG-MLIR pass driver\n", registry));
}
