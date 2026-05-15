/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

// MLIR infrastructure includes
#include "mlir/Support/FileUtilities.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"

// MLIR core components
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"

// LLVM utilities and support
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"

// External dialect registrations
#include "torch-mlir/InitAll.h"

// MFusion project specific includes
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Dvm/IR/DvmDialect.h"
#include "mfusion/Conversion/Passes.h"
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h"

namespace {
// Version information for mfusion-opt tool
constexpr int kMFusionOptMajorVersion = 1;
constexpr int kMFusionOptMinorVersion = 0;

// Helper function to initialize the registry
void initializeDialectRegistry(mlir::DialectRegistry &registry) {
  mlir::registerAllDialects(registry);
  mlir::torch::registerAllDialects(registry);
  mlir::torch::registerAllExtensions(registry);
  registry.insert<mlir::mfuse::MfuseDialect>();
  registry.insert<mlir::dvm::DvmDialect>();
}
}  // namespace

int main(int argc, char **argv) {
  // Initialize all standard MLIR passes
  mlir::registerAllPasses();

  // Register Torch-MLIR passes
  mlir::torch::registerAllPasses();

  // Register custom conversion passes for MFusion
  mlir::registerMFusionConversionPasses();

  // Register MFUSE transforms passes
  mlir::registerMfuseTransformsPasses();

  // Setup dialect registry with all required dialects
  mlir::DialectRegistry registry;
  initializeDialectRegistry(registry);

  std::string toolName = "MFusion optimizer driver v" + std::to_string(kMFusionOptMajorVersion) + "." +
                         std::to_string(kMFusionOptMinorVersion) + "\n";
  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, toolName.c_str(), registry));
}
