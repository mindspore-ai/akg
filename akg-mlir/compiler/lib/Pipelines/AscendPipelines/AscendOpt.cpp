/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "akg/Pipelines/AscendPipelines/AscendOpt.h"

#include <cstdlib>
#include <nlohmann/json.hpp>
#include "akg/Conversion/Passes.h"
#include "akg/Dialect/Affine/Passes.h"
#include "akg/Dialect/LLVMIR/Passes.h"
#include "akg/Dialect/Linalg/Passes.h"
#include "akg/Dialect/MindSpore/Passes.h"
#include "akg/Transforms/Passes.h"
#include "akg/Utils/AKGGlobalVars.hpp"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
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

using mlir::OpPassManager;

namespace {
void createAscendOptPipelineImpl(OpPassManager &pm, const mlir::AscendOptPipelineOptions &options) {
  pm.addPass(mlir::createAKGOperatorIdentifyPass());
  pm.addPass(mlir::createMindsporeMakeBroadcastablePass());
  pm.addPass(mlir::createEliminateDimensionPass());
  pm.addPass(mlir::createLegalizeTypePass());
  pm.addPass(mlir::createFoldDimensionPass());
  pm.addPass(mlir::createMindSporeToLinalgNamedPass());
  pm.addPass(mlir::createMindSporeToTosaPass());
  OpPassManager &nestedFunctionPM = pm.nest<mlir::func::FuncOp>();
  nestedFunctionPM.addPass(mlir::tosa::createTosaToLinalg());

  if (options.enableAKGLoopFusion) {
    bool keepFakeOuts = true;
    pm.addPass(mlir::createLinalgCopyBufferizePass(keepFakeOuts));
    pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());

    mlir::bufferization::OneShotBufferizationOptions bufferizationOpts;
    bufferizationOpts.bufferizeFunctionBoundaries = true;
    bufferizationOpts.setFunctionBoundaryTypeConversion(mlir::bufferization::LayoutMapOption::IdentityLayoutMap);
    bufferizationOpts.allowReturnAllocsFromLoops = true;
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufferizationOpts));

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createMemrefCopyToLoopsPass());

    OpPassManager &nestedFusionPM = pm.nest<mlir::func::FuncOp>();
    nestedFusionPM.addPass(mlir::createConvertLinalgToAffineLoopsPass());

    // pre-process
    nestedFusionPM.addPass(mlir::createCSEPass());
    nestedFusionPM.addPass(mlir::affine::createAffineReductionAnnotationPass());
    bool promoteSingleIter = true;
    nestedFusionPM.addPass(mlir::affine::createAffineLoopNormalizePass(promoteSingleIter));
    nestedFusionPM.addPass(mlir::createCanonicalizerPass());
    nestedFusionPM.addPass(mlir::createCopyElisionPass());
    nestedFusionPM.addPass(mlir::createUnifyShapePass());
    nestedFusionPM.addPass(mlir::createCopyRemovalPass());
    nestedFusionPM.addPass(mlir::createCSEPass());
    nestedFusionPM.addPass(mlir::createCanonicalizerPass());

    // fusion
    nestedFusionPM.addPass(mlir::createAKGLoopFusionPass());
    nestedFusionPM.addPass(mlir::createCanonicalizerPass());

    // tiling
    // nestedFusionPM.addPass(mlir::createMergeFusionOpPass(options.target));
    nestedFusionPM.addPass(mlir::createStoreLoadElimPass());
    nestedFusionPM.addPass(mlir::createAKGLoopTilingPass(options.target, true));
    nestedFusionPM.addPass(mlir::createRemoveRedundantLoopsPass());
    nestedFusionPM.addPass(mlir::createCanonicalizerPass());

    // vector
    // nestedFusionPM.addPass(mlir::createAffineIteratorConversionPass());
    // nestedFusionPM.addPass(mlir::createExtractIfOpPass(options.target));
    nestedFusionPM.addPass(mlir::affine::createAffineForVectPass());

    // parallel
    // nestedFusionPM.addPass(mlir::createRemoveRedundantLoopsPass());
    // nestedFusionPM.addPass(mlir::createAKGLoopParallelizePass(options.enableParallel));

    nestedFusionPM.addPass(mlir::affine::createVectorTransferTensorizePass());
    if (const char *v = std::getenv("TILINGFUNC"); v && std::string(v) == "1") {
      nestedFusionPM.addPass(mlir::affine::createTilingFuncPass());
    }
  }
}
}  // namespace

namespace mlir {
void createAscendOptPipeline(OpPassManager &pm, const AscendOptPipelineOptions &options) {
  createAscendOptPipelineImpl(pm, options);
}
}  // namespace mlir
