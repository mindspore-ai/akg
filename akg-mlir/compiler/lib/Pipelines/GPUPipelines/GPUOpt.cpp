/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#include "akg/Pipelines/GPUPipelines/GPUOpt.h"
#include "akg/Conversion/Passes.h"
#include "akg/Dialect/Affine/Passes.h"
#include "akg/Dialect/GPU/Passes.h"
#include "akg/Dialect/LLVMIR/Passes.h"
#include "akg/Dialect/Linalg/Passes.h"
#include "akg/Dialect/MindSpore/Passes.h"
#include "akg/Dialect/SCF/Passes.h"
#include "akg/Transforms/Passes.h"
#include "akg/Utils/AnalysisCommon.hpp"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace mlir {

void createGpuOptPipeline(OpPassManager &pm, const GPUPipelineOptions &options) {
  if (!options.globalConfigFile.empty()) {
    pm.addPass(createLoadGlobalConfigPass(options.globalConfigFile));
  }
  OpPassManager &nestedFunctionPM = pm.nest<func::FuncOp>();
  // Conversion: MindSpore Dialect -> Linalg
  // Conversion: Tosa Dialect -> Linalg
  nestedFunctionPM.addPass(mlir::createRemoveRedundantReducePass());
  nestedFunctionPM.addPass(mlir::createMoveDownReductionOpsPass());
  nestedFunctionPM.addPass(createMindSporeToTosaPass());
  nestedFunctionPM.addPass(tosa::createTosaMakeBroadcastablePass());
  nestedFunctionPM.addPass(createCanonicalizerPass());
  nestedFunctionPM.addPass(createAKGOperatorIdentifyPass());
  nestedFunctionPM.addPass(createEliminateReshapePass());
  nestedFunctionPM.addPass(createFoldDimensionPass());
  nestedFunctionPM.addPass(createMindSporeToLinalgPass());
  nestedFunctionPM.addPass(createMindSporeFinalizingLowerPass());
  nestedFunctionPM.addPass(tosa::createTosaToLinalgNamed());
  nestedFunctionPM.addPass(createCanonicalizerPass());
  nestedFunctionPM.addPass(tosa::createTosaLayerwiseConstantFoldPass());
  nestedFunctionPM.addPass(createTosaMultiReduceToLinalgPass());
  nestedFunctionPM.addPass(tosa::createTosaToLinalg());
  nestedFunctionPM.addPass(tosa::createTosaToTensor());
  nestedFunctionPM.addPass(tosa::createTosaToArith());

  // Bufferization opt passes
  bool keepFakeOuts = true;
  nestedFunctionPM.addPass(createLinalgCopyBufferizePass(keepFakeOuts));
  nestedFunctionPM.addPass(createLinalgElementwiseOpFusionPass());

  pm.addPass(bufferization::createEmptyTensorToAllocTensorPass());
  pm.addPass(bufferization::createOneShotBufferizePass());
  pm.addPass(func::createFuncBufferizePass());
  pm.addPass(bufferization::createBufferResultsToOutParamsPass());
  OpPassManager &nestedFunctionPM2 = pm.nest<func::FuncOp>();
  nestedFunctionPM2.addPass(createMatchAndMarkReductionOpsPass());

  // Affine opt passes
  nestedFunctionPM2.addPass(createConvertLinalgToAffineLoopsPass());
  nestedFunctionPM2.addPass(affine::createAffineLoopNormalizePass());

  if (options.gpuFast) {
    // Affine opt passes
    pm.addPass(createMergeFusionOpPass(kTargetCuda));
    pm.addPass(createStoreLoadElimPass());
    OpPassManager &nestedFunctionPM4 = pm.nest<func::FuncOp>();
    nestedFunctionPM4.addPass(affine::createAffineLoopNormalizePass());
    nestedFunctionPM4.addPass(createAKGLoopTilingPass(kTargetCuda, true));
    nestedFunctionPM4.addPass(createMatchAndMarkReductionOpsPass("affine"));
    nestedFunctionPM4.addPass(affine::createAffineLoopNormalizePass());
    nestedFunctionPM4.addPass(createCanonicalizerPass());
    nestedFunctionPM4.addPass(createAffineLoopReorderPass());
    nestedFunctionPM4.addPass(createStoreAxisInfoPass());
    nestedFunctionPM4.addPass(createAffineHandleBoundaryIfExtract());
    nestedFunctionPM4.addPass(createWorkaroundFixReduceInitializationPass());
    nestedFunctionPM4.addPass(createAffineMemoryPromotionPass(kTargetCuda));
    nestedFunctionPM4.addPass(createGenerateSingleAffineParallelPass());

    nestedFunctionPM4.addPass(createLoadAxisInfoPass());
    nestedFunctionPM4.addPass(createVectorTransferLowerPass());
    nestedFunctionPM4.addPass(affine::createAffineLoopNormalizePass());
    nestedFunctionPM4.addPass(createCanonicalizerPass());
    nestedFunctionPM4.addPass(createStoreAxisInfoPass());
    nestedFunctionPM4.addPass(affine::createAffineParallelizePass());
    nestedFunctionPM4.addPass(createForceConvertAffineForToAffineParallelPass("gpu-reduction"));
    nestedFunctionPM4.addPass(createLoadAxisInfoPass());
  }

  // SCF MAPPING
  pm.addPass(createStoreAxisInfoPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createLoadAxisInfoPass());
  OpPassManager &nestedFunctionPM5 = pm.nest<func::FuncOp>();
  nestedFunctionPM5.addPass(createConvertLinalgToParallelLoopsPass());
  nestedFunctionPM5.addPass(createAKGGPUMapping());
  nestedFunctionPM5.addPass(createRewriteReduceInMultiLevelMemoryPass());
  pm.addPass(createParallelLoopToGpuPass());
  pm.addPass(createStoreLoadElimPass());
  pm.addPass(createGpuLauchSinkIndexComputationsPass());
  OpPassManager &nestedFunctionPM6 = pm.nest<func::FuncOp>();
  nestedFunctionPM6.addPass(createGpuUseAllReduceWithAtomicReturnPass());
  pm.addPass(createGpuKernelOutliningExt());
  pm.addPass(createAffineHandleBoundaryIfRestore());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createPromoteTempBufferPass());
  pm.addPass(createCopyAttributesToGpuPass());
  pm.addPass(createDumpShapeInfoPass(options.jsonFileName));
}
}  // namespace mlir
