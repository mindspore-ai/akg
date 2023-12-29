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
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

using namespace llvm;
using namespace mlir;

extern "C" void LLVMInitializeNVPTXTarget();
extern "C" void LLVMInitializeNVPTXTargetInfo();
extern "C" void LLVMInitializeNVPTXTargetMC();
extern "C" void LLVMInitializeNVPTXAsmPrinter();

const char *triple = "nvptx64-nvidia-cuda";
const char *feature = "+ptx64";

static void getLibDevice(std::string &libdevice) {
  std::string path = "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc";
  auto cudaHome = llvm::sys::Process::GetEnv("CUDA_HOME");
  if (cudaHome) {
    path = *cudaHome + std::string("/nvvm/libdevice/libdevice.10.bc");
  }
  if (llvm::sys::fs::exists(path)) {
    libdevice = path;
  } else {
    llvm::errs() << "failed to find the libdevice file, please check the path.\n";
  }
}

static void InitPtxTarget() {
  (void)llvm::InitializeNativeTarget();
  (void)llvm::InitializeNativeTargetAsmPrinter();
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
}

LogicalResult mlir::translateToPTX(Operation *op, raw_ostream &os, const std::string &kernelName,
                                   const std::string &arch) {
  auto moduleOp = dyn_cast<ModuleOp>(op);
  if (!moduleOp) {
    return mlir::failure();
  }
  InitPtxTarget();

  std::string libdevice;
  getLibDevice(libdevice);
  std::string ptxStr;
  auto optLevel = 3;
  std::unique_ptr<mlir::Pass> pass = createSerializeToPTXPass(optLevel, libdevice, triple, arch, feature, ptxStr);

  auto ctx = moduleOp.getContext();
  mlir::PassManager pm(ctx);
  applyPassManagerCLOptions(pm);
  auto &kernelPm = pm.nest<mlir::gpu::GPUModuleOp>();
  kernelPm.addPass(std::move(pass));

  if (failed(pm.run(moduleOp))) {
    return mlir::failure();
  }

  std::string errorMessage;
  std::string ptxFilename = kernelName + ".ptx";
  auto ptxFile = mlir::openOutputFile(ptxFilename, &errorMessage);
  if (!ptxFile) {
    llvm::errs() << errorMessage << "\n";
    return mlir::failure();
  }
  ptxFile->os() << ptxStr;
  ptxFile->keep();
  return mlir::success();
}
