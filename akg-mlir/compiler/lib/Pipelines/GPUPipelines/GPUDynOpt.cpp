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

#include "akg/Pipelines/GPUPipelines/GPUDynOpt.h"
#include "akg/Utils/AnalysisCommon.hpp"
#include "akg/Conversion/Passes.h"
#include "akg/Dialect/Affine/Passes.h"
#include "akg/Dialect/SCF/Passes.h"
#include "akg/Dialect/GPU/Passes.h"
#include "akg/Dialect/LLVMIR/Passes.h"
#include "akg/Dialect/Linalg/Passes.h"
#include "akg/Dialect/Tosa/Passes.h"
#include "akg/Dialect/MindSpore/Passes.h"
#include "akg/Transforms/Passes.h"
#include "akg/Utils/AKGGlobalVars.hpp"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
namespace mlir {

void createGpuDynOptPipeline(OpPassManager &pm, const GPUDynPipelineOptions &options) {
  if (options.stage == 1) {
    akgglobal::ShapeAlignTool::getInstance().reset();
    pm.addPass(createInferSymbolicShapesPass());
    // Add Passes.
    OpPassManager &nestedFunctionPM = pm.nest<func::FuncOp>();
    bool IgnoreImplicitBroadcast = true;
    nestedFunctionPM.addPass(mlir::createMakeDynamicBroadcastablePass(IgnoreImplicitBroadcast));
    // Conversion: MindSpore Dialect -> Linalg
    // Conversion: Tosa Dialect -> Linalg
    nestedFunctionPM.addPass(createMindSporeToTosaPass());
    nestedFunctionPM.addPass(createAKGOperatorIdentifyPass());
    nestedFunctionPM.addPass(createEliminateReshapePass());
    nestedFunctionPM.addPass(createFoldDimensionPass());
    nestedFunctionPM.addPass(createMindSporeToLinalgPass());
    nestedFunctionPM.addPass(createMindSporeFinalizingLowerPass());
    nestedFunctionPM.addPass(tosa::createTosaMakeBroadcastablePass());
    nestedFunctionPM.addPass(createTosaMultiReduceToLinalgPass());
    nestedFunctionPM.addPass(tosa::createTosaToLinalgNamed());
    nestedFunctionPM.addPass(tosa::createTosaToLinalg());
    nestedFunctionPM.addPass(createMatchAndMarkReductionOpsPass());
    nestedFunctionPM.addPass(createLinalgSimplifyPass());
    nestedFunctionPM.addPass(createSymbolicRemovalPass());
    nestedFunctionPM.addPass(createLinalgElementwiseOpFusionPass());
    nestedFunctionPM.addPass(createLinalgElementwiseFusionExtPass());
    nestedFunctionPM.addPass(tosa::createTosaToArith());

    // Bufferization opt passes
    bool keepFakeOuts = true;
    nestedFunctionPM.addPass(createLinalgCopyBufferizePass(keepFakeOuts));
    nestedFunctionPM.addPass(createLinalgBufferizePass());

    pm.addPass(arith::createArithBufferizePass());
    pm.addPass(bufferization::createEmptyTensorToAllocTensorPass());
    OpPassManager &nestedFunctionPM1 = pm.nest<func::FuncOp>();
    nestedFunctionPM1.addPass(createTensorBufferizePass());
    pm.addPass(func::createFuncBufferizePass());
    pm.addPass(bufferization::createBufferResultsToOutParamsPass());
    OpPassManager &nestedFunctionPM2 = pm.nest<func::FuncOp>();

    // Affine opt passes
    nestedFunctionPM2.addPass(createConvertLinalgToAffineLoopsPass());
    nestedFunctionPM2.addPass(createAffineLoopNormalizePass());

    // Polytops opt prepare passes
    pm.addPass(createCSEPass());
    bool promoteSingleIter = true;
    pm.addPass(createAffineLoopNormalizePass(promoteSingleIter));
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCopyElisionPass());
    pm.addPass(createSimplifyShapePass());
    pm.addPass(createUnifyShapePass());
    pm.addPass(createCopyRemovalPass());
    pm.addPass(createCanonicalizerPass());
  } else if (options.stage == 2) {
    // Affine opt passes
    pm.addPass(createStoreLoadElimPass());
    pm.addPass(createCopyRemovalPass());
    pm.addPass(createReplaceUnknownDimsToOutputDimPass());
    pm.addPass(createFixDynamicIndexingPass());
    pm.addPass(createMergeFusionOpPass(kTargetCuda));
    OpPassManager &nestedFunctionPM4 = pm.nest<func::FuncOp>();
    nestedFunctionPM4.addPass(createAffineLoopNormalizePass());
    nestedFunctionPM4.addPass(createAKGLoopTilingPass(kTargetCuda, true, options.tilingMode));
    nestedFunctionPM4.addPass(createMatchAndMarkReductionOpsPass("affine"));
    nestedFunctionPM4.addPass(createAffineHandleBoundaryIfExtract());
    nestedFunctionPM4.addPass(createAffineLoopNormalizePass());
    nestedFunctionPM4.addPass(createCanonicalizerPass());
    nestedFunctionPM4.addPass(createAffineLoopReorderPass());
    nestedFunctionPM4.addPass(createAKGVectorizePass(kTargetCuda, ""));
    nestedFunctionPM4.addPass(createVectorTransferLowerPass());
    nestedFunctionPM4.addPass(createStoreAxisInfoPass());
    nestedFunctionPM4.addPass(createAffineMemoryPromotionPass(kTargetCuda));

    bool isDynamicShape = true;
    nestedFunctionPM4.addPass(createGenerateSingleAffineParallelPass(isDynamicShape));
    nestedFunctionPM4.addPass(createAffineLoopNormalizePass());
    nestedFunctionPM4.addPass(createCanonicalizerPass());
    nestedFunctionPM4.addPass(createStoreAxisInfoPass());
    nestedFunctionPM4.addPass(createAffineParallelizePass());
    nestedFunctionPM4.addPass(createLoadAxisInfoPass());
    nestedFunctionPM4.addPass(createForceConvertAffineForToAffineParallelPass("gpu-reduction"));
    nestedFunctionPM4.addPass(createLoadAxisInfoPass());
    // SCF MAPPING
    pm.addPass(createStoreAxisInfoPass());
    pm.addPass(createLowerAffinePass());
    pm.addPass(createLoadAxisInfoPass());
    OpPassManager &nestedFunctionPM5 = pm.nest<func::FuncOp>();
    nestedFunctionPM5.addPass(createConvertLinalgToParallelLoopsPass());
    nestedFunctionPM5.addPass(createAKGGPUMapping());
    nestedFunctionPM5.addPass(createPrimeNumReplaceForDynamicShapePass("replace"));

    nestedFunctionPM5.addPass(createRewriteReduceInMultiLevelMemoryPass());
    pm.addPass(createParallelLoopToGpuPass());
    pm.addPass(createPrimeNumReplaceForDynamicShapePass("restore"));

    pm.addPass(createStoreLoadElimPass());
    pm.addPass(createConvertVectorToGPUPass());
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
}
}  // namespace mlir

