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

#include "akg/Pipelines/CPUOpt.h"

#include <nlohmann/json.hpp>
#include <string>
#include "akg/Conversion/Passes.h"
#include "akg/Dialect/Affine/Passes.h"
#include "akg/Dialect/CPU/Passes.h"
#include "akg/Dialect/GPU/Passes.h"
#include "akg/Dialect/LLVMIR/Passes.h"
#include "akg/Dialect/Linalg/Passes.h"
#include "akg/Dialect/MindSpore/Passes.h"
#include "akg/Transforms/Passes.h"
#include "akg/Utils/AKGGlobalVars.hpp"

#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace akgglobal;

namespace {
void jsonToLinalg(OpPassManager &pm, const CpuOptPipelineOptions &options) {
  // Conversion: MindSpore Dialect -> Linalg
  OpPassManager &nestedFunctionPM = pm.nest<func::FuncOp>();
  if (options.dynamicShape) {
    nestedFunctionPM.addPass(mlir::createInferSymbolicShapesPass());
  }
  nestedFunctionPM.addPass(mlir::createRemoveRedundantReducePass());
  bool IgnoreImplicitBroadcast = true;
  nestedFunctionPM.addPass(mlir::createMakeDynamicBroadcastablePass(IgnoreImplicitBroadcast));
  nestedFunctionPM.addPass(mlir::createMindSporeToTosaPass());
  nestedFunctionPM.addPass(tosa::createTosaMakeBroadcastablePass());
  nestedFunctionPM.addPass(createCanonicalizerPass());
  nestedFunctionPM.addPass(createAKGOperatorIdentifyPass());

  if (!options.dynamicShape) {
    nestedFunctionPM.addPass(createEliminateReshapePass());
    nestedFunctionPM.addPass(createFoldDimensionPass());
    nestedFunctionPM.addPass(createCanonicalizerPass());
  }

  nestedFunctionPM.addPass(mlir::createMindSporeToLinalgPass());
  nestedFunctionPM.addPass(mlir::createMindSporeFinalizingLowerPass());

  // Conversion TOSA -> Linalg
  if (options.cpuFast) {
    nestedFunctionPM.addPass(createTosaMultiReduceToLinalgPass());
  }
  nestedFunctionPM.addPass(tosa::createTosaToLinalgNamed());
  nestedFunctionPM.addPass(tosa::createTosaToLinalg());
  if (options.cpuFast) {
    nestedFunctionPM.addPass(createMatchAndMarkReductionOpsPass());
  }
  nestedFunctionPM.addPass(createLinalgSimplifyPass());
  nestedFunctionPM.addPass(createSymbolicRemovalPass());
  if (!options.cpuFast || options.dynamicShape) {
    nestedFunctionPM.addPass(createLinalgElementwiseOpFusionPass());
    nestedFunctionPM.addPass(createLinalgElementwiseFusionExtPass());
  }
  nestedFunctionPM.addPass(tosa::createTosaToArith());
}

void commonBufferization(OpPassManager &pm) {
  // Bufferization and Tensor Transform
  OpPassManager &nestedFunctionPM = pm.nest<func::FuncOp>();
  nestedFunctionPM.addPass(createLinalgCopyBufferizePass());
  nestedFunctionPM.addPass(createLinalgExtBufferizePass());
  pm.addPass(bufferization::createEmptyTensorToAllocTensorPass());
  pm.addPass(func::createFuncBufferizePass());
  pm.addPass(bufferization::createBufferResultsToOutParamsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void affinePreprocess(OpPassManager &pm, const CpuOptPipelineOptions &options) {
  pm.addPass(createCSEPass());
  bool promoteSingleIter = true;
  pm.addPass(affine::createAffineLoopNormalizePass(promoteSingleIter));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCopyElisionPass());
  pm.addPass(createSimplifyShapePass());
  pm.addPass(createUnifyShapePass());
  pm.addPass(createCopyRemovalPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
}

void affineOptimize(OpPassManager &pm, const CpuOptPipelineOptions &options) {
  if (options.dynamicShape) {
    pm.addPass(createStoreLoadElimPass());
    pm.addPass(createMergeFusionOpPass(options.target));
  } else {
    OpPassManager &nestedFunctionPM = pm.nest<func::FuncOp>();
    nestedFunctionPM.addPass(createMergeFusionOpPass(options.target));
    nestedFunctionPM.addPass(createStoreLoadElimPass());
  }

  OpPassManager &nestedFunctionPM1 = pm.nest<func::FuncOp>();
  nestedFunctionPM1.addPass(createAKGLoopTilingPass(options.target, options.feature, true));
  nestedFunctionPM1.addPass(createRemoveRedundantLoopsPass());
  nestedFunctionPM1.addPass(createCanonicalizerPass());
  nestedFunctionPM1.addPass(createAffineIteratorConversionPass());
  nestedFunctionPM1.addPass(createExtractIfOpPass(options.target));
  if (!options.dynamicShape) {
    // elim store-load pairs before vectorization and after `merge-fusion + extract-if`
    nestedFunctionPM1.addPass(createStoreLoadElimPass());
  }
  nestedFunctionPM1.addPass(createRemoveRedundantLoopsPass());
  nestedFunctionPM1.addPass(createAKGLoopParallelizePass(options.enableParallel));
  nestedFunctionPM1.addPass(createVectorTransferLowerPass());

  if (options.dynamicShape) {
    pm.addPass(createFixDynamicIndexingPass());
  }
}

void convertToLLVM(OpPassManager &pm, const CpuOptPipelineOptions &options) {
  pm.addPass(createLowerAffinePass());
  pm.addPass(createAKGFuncOutliningPass(options.outliningPlatform == "MindSpore", options.cpuOutlining));
  if (!options.cpuOutlining) {
    pm.addPass(createConvertSCFToOpenMPPass());
  }
  pm.addPass(createCanonicalizerPass());
  //   MemrefToLLVM must be put before SCFToCF
  // pm.addPass(createMemRefToLLVMConversionPass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createCanonicalizerPass());
  // Convert All dialects to LLVM and packing parameters
  pm.addPass(createConvertMathToLibmPass());
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertMathToLLVMPass());
  // pm.addPass(createConvertLinalgToLLVMPass());
  mlir::MLIRContext tmp_context;
  ConvertFuncToLLVMPassOptions llvmOptions;
  llvmOptions.useBarePtrCallConv = true;
  pm.addPass(createConvertFuncToLLVMPass(llvmOptions));
  llvmOptions.useBarePtrCallConv = false;
  // ArithToLLVM must be put before CFToLLVM
  pm.addPass(createConvertFuncToLLVMPass(llvmOptions));
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(LLVM::createParameterPackingPass(options.outliningPlatform == "MindSpore"));
  pm.addPass(createAKGParallelLaunchPass(options.outliningPlatform == "MindSpore", options.cpuOutlining));
  if (!options.cpuOutlining) {
    pm.addPass(createPureOpenMPToLLVMPass());
  }

  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addPass(createCanonicalizerPass());
}

void createCpuOptPipelineImpl(OpPassManager &pm, const CpuOptPipelineOptions &options) {
  jsonToLinalg(pm, options);
  commonBufferization(pm);

  if (options.dynamicShape) {
    pm.addPass(memref::createFoldMemRefAliasOpsPass());
  }

  if (options.cpuFast) {
    OpPassManager &nestedFunctionPM = pm.nest<func::FuncOp>();
    nestedFunctionPM.addPass(createConvertLinalgToAffineLoopsPass());
    nestedFunctionPM.addPass(mlir::createLinalgExtLowerPass());
    affinePreprocess(pm, options);
    affineOptimize(pm, options);
  } else {
    OpPassManager &nestedFunctionPM1 = pm.nest<func::FuncOp>();
    nestedFunctionPM1.addPass(mlir::createLinalgExtLowerPass());
    pm.addPass(createConvertLinalgToLoopsPass());
    pm.addPass(createConvertShapeToStandardPass());
    pm.addPass(createCanonicalizerPass());
  }

  convertToLLVM(pm, options);
}
}  // namespace

namespace mlir {
void createCpuOptPipeline(OpPassManager &pm, const CpuOptPipelineOptions &options) {
  if (options.dynamicShape) {
    assert(options.cpuFast);
  }
  createCpuOptPipelineImpl(pm, options);
}
}  // namespace mlir
