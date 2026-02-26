/**
 * Copyright 2024-2026 Huawei Technologies Co., Ltd
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
#include "llvm/ADT/SmallVector.h"
#include <nlohmann/json.hpp>
#include "akg/Conversion/Passes.h"
#include "akg/Dialect/Affine/Passes.h"
#include "akg/Dialect/SCF/Passes.h"
#include "akg/Dialect/Tensor/Passes.h"
#include "akg/Dialect/LLVMIR/Passes.h"
#include "akg/Dialect/Linalg/Passes.h"
#include "akg/Dialect/MindSpore/Passes.h"
#include "akg/Transforms/Passes.h"
#include "akg/Utils/AKGGlobalVars.hpp"
#include "bishengir/Dialect/Annotation/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HACC/Transforms/Passes.h"
#include "bishengir/Dialect/SCF/Transforms/Passes.h"
#include "bishengir/Dialect/Scope/Transforms/Passes.h"
#include "bishengir/Conversion/ArithToAffine/ArithToAffine.h"
#include "bishengir/Conversion/HIVMToStandard/HIVMToStandard.h"
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

void canonicalizationPipeline(OpPassManager &pm) {
  pm.addPass(mlir::createArithToAffineConversionPass());
  pm.nest<func::FuncOp>().addPass(mlir::scf::createCanonicalizeIterArgPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createSCFForLoopCanonicalizationPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createHIVMOptSinglePointPass());
  pm.addPass(mlir::createCanonicalizerPass());
  // pm.nest<func::FuncOp>().addPass(memref::createDeadStoreEliminationPass());
}

void createHIVMPipeline(OpPassManager &pm, const mlir::AscendOptPipelineOptions &options) {
  pm.addPass(mlir::hivm::createInferFuncCoreTypePass());
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createInitEntryKernelPass());

  pm.nest<func::FuncOp>().addPass(mlir::hivm::createLiftZeroRankPass());
  pm.nest<func::FuncOp>().addPass(mlir::scf::createMapForToForallPass());
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createHIVMMapForallToBlocksPass());
  // Op decompose, need mark buffer size for newly allocated buffer.
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createHIVMDecomposeOpPass());
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createSyncBlockHoistingPass());
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createBindSyncBlockLockArgPass());
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createInsertInferSyncBlockLockNumAndInitFuncPass());
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createSyncBlockLockLoweringPass());
  // Convert non-contiguous reshape to hivm.copy
  // Call this before infer mem scope. Otherwise, there might be UB allocs in AIC function.
  pm.addPass(mlir::hivm::createNonContiguousReshapeToCopyPass());
  pm.addPass(mlir::hivm::createInferHIVMMemScopePass());
  // Decompose copy_ub_to_ub after inferHIVMMemScope
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createHIVMDecomposeOpPass());
  HIVMAggregatedDecomposeOpOptions decomposeOption;
  // Currently no Ops decompose in this phase
  decomposeOption.decomposePhase = bishengir::DecomposePhase::BEFORE_HIVM_STRIDE_ALIGNMENT;
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createHIVMAggregatedDecomposeOpPass(decomposeOption));

  // Transform uncontinuous access to deinterleave op
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createHIVMRecognizeDeinterleaveOpPass());
  decomposeOption.decomposePhase = bishengir::DecomposePhase::AFTER_RECOGNIZE_DEINTERLEAVE;
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createHIVMAggregatedDecomposeOpPass(decomposeOption));
  decomposeOption.decomposePhase = bishengir::DecomposePhase::AFTER_RECOGNIZE_BROADCAST;
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createHIVMAggregatedDecomposeOpPass(decomposeOption));

  // align alloc size for special hivm op
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createAlignAllocSizePass());
  if (options.enableAutoStorageAlign) {
    pm.nest<func::FuncOp>().addPass(mlir::hivm::createMarkStrideAlignPass());
  }
  // pm.nest<func::FuncOp>().addPass(mlir::memref::createFoldAllocReshapePass());
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createEnableStrideAlignPass());

  // Decompose {vconcat} after stride alignment
  decomposeOption.decomposePhase = bishengir::DecomposePhase::AFTER_HIVM_STRIDE_ALIGNMENT;
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createHIVMAggregatedDecomposeOpPass(decomposeOption));

  // convert copyOp to nd2nzOp
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createInferHIVMDataLayoutPass());
  decomposeOption.decomposePhase = bishengir::DecomposePhase::AFTER_INFER_HIVM_DATA_LAYOUT;
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createHIVMAggregatedDecomposeOpPass(decomposeOption));

  pm.addPass(mlir::createCanonicalizerPass());
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createAutoInferBufferSizePass());
  pm.addPass(mlir::createArithToAffineConversionPass());
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createConstantizeBufferSizePass());
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createSetBufferSizePass());
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createFlattenOpsPass());
  decomposeOption.decomposePhase = bishengir::DecomposePhase::AFTER_HIVM_FLATTEN_OPS;
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createHIVMAggregatedDecomposeOpPass(decomposeOption));
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createReduceRankSubviewPass());
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createLiftLowestStridePass());
  decomposeOption.decomposePhase = bishengir::DecomposePhase::AFTER_LIFT_LOWEST_STRIDE;
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createHIVMAggregatedDecomposeOpPass(decomposeOption));
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createAllocExtraBufferPass());
  // Infer memory scope for newly allocated extra buffer
  pm.addPass(mlir::hivm::createInferHIVMMemScopePass());
  canonicalizationPipeline(pm);
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createInlineLoadCopyPass());

  MarkMultiBufferOptions multiBufferOptions;
  multiBufferOptions.enableAuto = options.enableAutoMultiBuffer;
  // Limit auto multi buffer only work for local buffer at this stage
  multiBufferOptions.limitAutoMultiBufferOnlyForLocalBuffer = true;
  multiBufferOptions.limitAutoMultiBufferOfLocalBuffer = MultiBufferStrategy::CUBE_NO_L0C;
  multiBufferOptions.limitMixAutoMultiBufferBuffer = MultiBufferStrategy::ONLY_CUBE;
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createMarkMultiBufferPass(multiBufferOptions));
  PlanMemoryOptions planMemoryOption;
  planMemoryOption.enableMemoryDisplay = options.enableMemoryDisplay;
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createPlanMemoryPass(planMemoryOption));

  // Lower hivm ops to loops
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createHIVMLowerToLoopsPass());
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createHIVMDecomposeOpPass());
  // Normal sync (inject-sync, graph-sync-solver) passes.
  bool enableHIVMUnitFlagSync = false;
  bool enableHIVMAssumeAliveLoops = false;
  if (options.enableGraphSyncSolver && !options.enableInjectBarrierAllSync && !options.enableAutoInjectSync) {
    GraphSyncSolverOptions gssOptions;
    gssOptions.enableUnitFlag = enableHIVMUnitFlagSync;
    pm.nest<func::FuncOp>().addPass(mlir::hivm::createGraphSyncSolverPass(gssOptions));
  } else if (options.enableAutoInjectSync) {
    InjectSyncOptions syncOptions;
    syncOptions.enableUnitFlag = enableHIVMUnitFlagSync;
    syncOptions.assumeAliveLoops = enableHIVMAssumeAliveLoops;
    if (options.enableInjectBarrierAllSync) {
      syncOptions.syncMode = mlir::hivm::SyncMode::BARRIERALL;
    }
    pm.nest<func::FuncOp>().addPass(mlir::hivm::createInjectSyncPass(syncOptions));
  }
  // pm.addPass(mlir::createMemrefExtLoweringPass());
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createAddFFTSToSyncBlockSetOpPass());
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createEnableMultiBufferPass());
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createLiftLowestStridePass());
  // Optimizations that relies on scope should be done after this point. Inline
  // all `scope.scope` ops.
  pm.addPass(mlir::scope::createInlineScopePass(InlineScopeOptions{/*forceInline=*/true}));
  pm.addPass(mlir::hivm::createEnableHIVMCCompatiblePrintPass());
  pm.addPass(mlir::annotation::createAnnotationLoweringPass());
  // pm.nest<func::FuncOp>().addPass(mlir::hivm::createInsertInitAndFinishForDebugPass());
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createMarkDisableLoadPass());
  pm.addPass(mlir::hivm::createMarkSyncBlockLockWithSubblockPass());
  pm.addPass(mlir::hivm::createInsertFreeLockVarBeforeReturnPass());
  pm.addPass(mlir::createConvertHIVMToStandardPass());
}

void createAscendOptPipelineImpl(OpPassManager &pm, const mlir::AscendOptPipelineOptions &options) {
  pm.addPass(mlir::createAKGOperatorIdentifyPass());
  pm.addPass(mlir::createHoistTensorSlicePass());
  pm.addPass(mlir::createMindsporeMakeBroadcastablePass());
  pm.addPass(mlir::createEliminateDimensionPass());
  pm.addPass(mlir::createLegalizeTypePass());
  // pm.addPass(mlir::createFoldDimensionPass());
  if (options.enableLoopFusion) {
    pm.addPass(mlir::createMindSporeToLinalgNamedPass(options.dynamicShape, !options.enableLoopFusion));
    pm.addPass(mlir::createCopyReturnedFuncArgsPass());
    pm.addPass(mlir::createLinalgGeneralizeNamedOpsPass());
    pm.addPass(mlir::createMindSporeToTosaPass());
    pm.addPass(mlir::createMindSporeToLinalgPass());
    pm.addPass(mlir::createCloneTensorEmptyPass());
  } else {
    pm.addPass(mlir::createMindSporeToLinalgNamedPass(options.dynamicShape));
    pm.addPass(mlir::createMindSporeToTosaPass());
  }
  OpPassManager &nestedFunctionPM = pm.nest<mlir::func::FuncOp>();
  nestedFunctionPM.addPass(mlir::tosa::createTosaToLinalg());
  // Erase unused linalg.generic operands/results before bufferization.
  nestedFunctionPM.addPass(mlir::createEraseUnusedOperandsAndResultsPass());

  if (options.enableLoopFusion) {
    pm.addPass(mlir::createDecomposeTensorPass());
    pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());

    mlir::bufferization::OneShotBufferizationOptions bufferizationOpts;
    bufferizationOpts.allowReturnAllocsFromLoops = true;
    bufferizationOpts.bufferizeFunctionBoundaries = true;
    bufferizationOpts.setFunctionBoundaryTypeConversion(mlir::bufferization::LayoutMapOption::IdentityLayoutMap);
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufferizationOpts));
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createMemrefCopyToLoopsPass());
    pm.addPass(mlir::createShapeNormalizationPass());

    OpPassManager &nestedFusionPM = pm.nest<mlir::func::FuncOp>();
    nestedFusionPM.addPass(mlir::createConvertLinalgToAffineLoopsPass());

    // pre-process
    nestedFusionPM.addPass(mlir::createCSEPass());
    bool promoteSingleIter = true;
    nestedFusionPM.addPass(mlir::affine::createAffineLoopNormalizePass(promoteSingleIter));
    nestedFusionPM.addPass(mlir::createCanonicalizerPass());
    nestedFusionPM.addPass(mlir::createCopyElisionPass());
    nestedFusionPM.addPass(mlir::createUnifyShapePass());
    nestedFusionPM.addPass(mlir::createCopyRemovalPass());
    nestedFusionPM.addPass(mlir::createCSEPass());
    nestedFusionPM.addPass(mlir::createCanonicalizerPass());

    // fusion
    nestedFusionPM.addPass(mlir::createRemoveRedundantLoopsPass());
    nestedFusionPM.addPass(mlir::createCanonicalizerPass());
    nestedFusionPM.addPass(mlir::affine::createAffineReductionAnnotationPass());
    nestedFusionPM.addPass(mlir::createHoistLoopIndependentOpsPass());
    nestedFusionPM.addPass(mlir::createAKGLoopFusionPass());
    nestedFusionPM.addPass(mlir::affine::createAffineLoopInvariantCodeMotionPass());
    nestedFusionPM.addPass(mlir::createCanonicalizerPass());
    nestedFusionPM.addPass(mlir::createSubviewAllocElimPass());
    nestedFusionPM.addPass(mlir::createReductionSiblingRecomputePass());
    nestedFusionPM.addPass(mlir::createAffineIteratorConversionPass());
    nestedFusionPM.addPass(mlir::createStoreLoadElimPass());
    nestedFusionPM.addPass(mlir::createLegalizeTypeForAscendPass());
    nestedFusionPM.addPass(mlir::createNormalizePass());
    nestedFusionPM.addPass(mlir::createConvertAffineToSCFPass());

    pm.addPass(mlir::createSymbolicRemovalPass());
    pm.addPass(mlir::createAddOutParameterPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createLegalizeBoolPass());
    // tiling
    pm.addPass(mlir::createNPUAutoTilingPass());
    pm.addPass(mlir::createAllocBufferShrinkPass());
    // vector
    pm.addPass(mlir::scf::createNPUVectorVectorizePass());
    pm.addPass(mlir::createArithToHIVMConversionPass());
    pm.addPass(mlir::createCanonicalizerPass());
  }

  pm.addPass(mlir::hacc::createAppendDeviceSpecPass());

  // hivm optimize pipeline
  createHIVMPipeline(pm, options);
}
}  // namespace

namespace mlir {
void createAscendOptPipeline(OpPassManager &pm, const AscendOptPipelineOptions &options) {
  createAscendOptPipelineImpl(pm, options);
}
}  // namespace mlir
