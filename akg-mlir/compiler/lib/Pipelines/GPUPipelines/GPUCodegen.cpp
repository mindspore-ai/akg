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

#include "akg/Pipelines/GPUPipelines/GPUCodegen.h"

#include "akg/Conversion/Passes.h"
#include "akg/Dialect/Affine/Passes.h"
#include "akg/Dialect/GPU/Passes.h"
#include "akg/Dialect/LLVMIR/Passes.h"
#include "akg/Transforms/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace mlir {

void createGpuCodegenPipeline(OpPassManager &pm, const GPUCodegenPipelineOptions &options) {
  // Add Passes.
  pm.addPass(createMathExtLowerPass());
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertVectorToGPUPass());
  pm.addPass(createConvertSCFToCFPass());

  pm.addPass(cf::createConvertControlFlowToLLVMPass());
  pm.addPass(createLowerGpuOpsToNVVMOpsPass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createConvertLinalgToLLVMPass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createMemRefToLLVMConversionPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());

  mlir::MLIRContext tmp_context;
  LowerToLLVMOptions llvmOptions(&tmp_context);
  llvmOptions.useBarePtrCallConv = true;
  pm.addPass(createConvertFuncToLLVMPass(llvmOptions));
  pm.addPass(createLowerGpuOpsToNVVMOpsPass());
  pm.addPass(createStripDebugInfoPass());
  pm.addPass(createLowerGpuOpsToNVVMOpsPass());
}
}  // namespace mlir

