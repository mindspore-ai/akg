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
#include "akg/Dialect/NPUVector/Passes.h"
#include "akg/Dialect/Vector/Passes.h"
#include "akg/Dialect/Tensor/Passes.h"
#include "akg/Dialect/LLVMIR/Passes.h"
#include "akg/Dialect/Linalg/Passes.h"
#include "akg/Dialect/MindSpore/Passes.h"
#include "akg/Transforms/Passes.h"
#include "akg/Utils/GlobalVars.hpp"
#include "akg/Utils/AnalysisForNpu.hpp"
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

namespace mlir {
namespace {

void canonicalizationPipeline(OpPassManager &pm) {
  pm.addPass(createArithToAffineConversionPass());
  pm.nest<func::FuncOp>().addPass(scf::createCanonicalizeIterArgPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createSCFForLoopCanonicalizationPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.nest<func::FuncOp>().addPass(hivm::createHIVMOptSinglePointPass());
  pm.addPass(createCanonicalizerPass());
  // pm.nest<func::FuncOp>().addPass(memref::createDeadStoreEliminationPass());
}

void createHIVMPipeline(OpPassManager &pm, const AscendOptPipelineOptions &options) {
  pm.addPass(hivm::createInferFuncCoreTypePass());
  pm.nest<func::FuncOp>().addPass(hivm::createInitEntryKernelPass());

  pm.nest<func::FuncOp>().addPass(hivm::createLiftZeroRankPass());
  pm.nest<func::FuncOp>().addPass(scf::createMapForToForallPass());
  pm.nest<func::FuncOp>().addPass(hivm::createHIVMMapForallToBlocksPass());
  // Op decompose, need mark buffer size for newly allocated buffer.
  pm.nest<func::FuncOp>().addPass(hivm::createHIVMDecomposeOpPass());
  pm.nest<func::FuncOp>().addPass(hivm::createSyncBlockHoistingPass());
  pm.nest<func::FuncOp>().addPass(hivm::createBindSyncBlockLockArgPass());
  pm.nest<func::FuncOp>().addPass(hivm::createInsertInferSyncBlockLockNumAndInitFuncPass());
  pm.nest<func::FuncOp>().addPass(hivm::createSyncBlockLockLoweringPass());
  // Convert non-contiguous reshape to hivm.copy
  // Call this before infer mem scope. Otherwise, there might be UB allocs in AIC function.
  pm.addPass(hivm::createNonContiguousReshapeToCopyPass());
  pm.addPass(hivm::createInferHIVMMemScopePass());
  // Decompose copy_ub_to_ub after inferHIVMMemScope
  pm.nest<func::FuncOp>().addPass(hivm::createHIVMDecomposeOpPass());
  HIVMAggregatedDecomposeOpOptions decomposeOption;
  // Currently no Ops decompose in this phase
  decomposeOption.decomposePhase = bishengir::DecomposePhase::BEFORE_HIVM_STRIDE_ALIGNMENT;
  pm.nest<func::FuncOp>().addPass(hivm::createHIVMAggregatedDecomposeOpPass(decomposeOption));

  // Transform uncontinuous access to deinterleave op
  pm.nest<func::FuncOp>().addPass(hivm::createHIVMRecognizeDeinterleaveOpPass());
  decomposeOption.decomposePhase = bishengir::DecomposePhase::AFTER_RECOGNIZE_DEINTERLEAVE;
  pm.nest<func::FuncOp>().addPass(hivm::createHIVMAggregatedDecomposeOpPass(decomposeOption));
  decomposeOption.decomposePhase = bishengir::DecomposePhase::AFTER_RECOGNIZE_BROADCAST;
  pm.nest<func::FuncOp>().addPass(hivm::createHIVMAggregatedDecomposeOpPass(decomposeOption));

  // align alloc size for special hivm op
  pm.nest<func::FuncOp>().addPass(hivm::createAlignAllocSizePass());
  if (options.enableAutoStorageAlign) {
    pm.nest<func::FuncOp>().addPass(hivm::createMarkStrideAlignPass());
  }
  // pm.nest<func::FuncOp>().addPass(memref::createFoldAllocReshapePass());
  pm.nest<func::FuncOp>().addPass(hivm::createEnableStrideAlignPass());

  // Decompose {vconcat} after stride alignment
  decomposeOption.decomposePhase = bishengir::DecomposePhase::AFTER_HIVM_STRIDE_ALIGNMENT;
  pm.nest<func::FuncOp>().addPass(hivm::createHIVMAggregatedDecomposeOpPass(decomposeOption));

  // convert copyOp to nd2nzOp
  pm.nest<func::FuncOp>().addPass(hivm::createInferHIVMDataLayoutPass());
  decomposeOption.decomposePhase = bishengir::DecomposePhase::AFTER_INFER_HIVM_DATA_LAYOUT;
  pm.nest<func::FuncOp>().addPass(hivm::createHIVMAggregatedDecomposeOpPass(decomposeOption));

  pm.addPass(createCanonicalizerPass());
  pm.nest<func::FuncOp>().addPass(hivm::createAutoInferBufferSizePass());
  pm.addPass(createArithToAffineConversionPass());
  pm.nest<func::FuncOp>().addPass(hivm::createConstantizeBufferSizePass());
  pm.nest<func::FuncOp>().addPass(hivm::createSetBufferSizePass());
  pm.nest<func::FuncOp>().addPass(hivm::createFlattenOpsPass());
  decomposeOption.decomposePhase = bishengir::DecomposePhase::AFTER_HIVM_FLATTEN_OPS;
  pm.nest<func::FuncOp>().addPass(hivm::createHIVMAggregatedDecomposeOpPass(decomposeOption));
  pm.nest<func::FuncOp>().addPass(hivm::createReduceRankSubviewPass());
  pm.nest<func::FuncOp>().addPass(hivm::createLiftLowestStridePass());
  decomposeOption.decomposePhase = bishengir::DecomposePhase::AFTER_LIFT_LOWEST_STRIDE;
  pm.nest<func::FuncOp>().addPass(hivm::createHIVMAggregatedDecomposeOpPass(decomposeOption));
  pm.nest<func::FuncOp>().addPass(hivm::createAllocExtraBufferPass());
  // Infer memory scope for newly allocated extra buffer
  pm.addPass(hivm::createInferHIVMMemScopePass());
  canonicalizationPipeline(pm);
  pm.nest<func::FuncOp>().addPass(hivm::createInlineLoadCopyPass());

  MarkMultiBufferOptions multiBufferOptions;
  multiBufferOptions.enableAuto = options.enableAutoMultiBuffer;
  // Limit auto multi buffer only work for local buffer at this stage
  multiBufferOptions.limitAutoMultiBufferOnlyForLocalBuffer = true;
  multiBufferOptions.limitAutoMultiBufferOfLocalBuffer = MultiBufferStrategy::CUBE_NO_L0C;
  multiBufferOptions.limitMixAutoMultiBufferBuffer = MultiBufferStrategy::ONLY_CUBE;
  pm.nest<func::FuncOp>().addPass(hivm::createMarkMultiBufferPass(multiBufferOptions));
  PlanMemoryOptions planMemoryOption;
  planMemoryOption.enableMemoryDisplay = options.enableMemoryDisplay;
  pm.nest<func::FuncOp>().addPass(hivm::createPlanMemoryPass(planMemoryOption));

  // Lower hivm ops to loops
  pm.nest<func::FuncOp>().addPass(hivm::createHIVMLowerToLoopsPass());
  pm.nest<func::FuncOp>().addPass(hivm::createHIVMDecomposeOpPass());
  // Normal sync (inject-sync, graph-sync-solver) passes.
  bool enableHIVMUnitFlagSync = false;
  bool enableHIVMAssumeAliveLoops = false;
  if (options.enableGraphSyncSolver && !options.enableInjectBarrierAllSync && !options.enableAutoInjectSync) {
    GraphSyncSolverOptions gssOptions;
    gssOptions.enableUnitFlag = enableHIVMUnitFlagSync;
    pm.nest<func::FuncOp>().addPass(hivm::createGraphSyncSolverPass(gssOptions));
  } else if (options.enableAutoInjectSync) {
    InjectSyncOptions syncOptions;
    syncOptions.enableUnitFlag = enableHIVMUnitFlagSync;
    syncOptions.assumeAliveLoops = enableHIVMAssumeAliveLoops;
    if (options.enableInjectBarrierAllSync) {
      syncOptions.syncMode = hivm::SyncMode::BARRIERALL;
    }
    pm.nest<func::FuncOp>().addPass(hivm::createInjectSyncPass(syncOptions));
  }
  // pm.addPass(createMemrefExtLoweringPass());
  pm.nest<func::FuncOp>().addPass(hivm::createAddFFTSToSyncBlockSetOpPass());
  pm.nest<func::FuncOp>().addPass(hivm::createEnableMultiBufferPass());
  pm.nest<func::FuncOp>().addPass(hivm::createLiftLowestStridePass());
  // Optimizations that relies on scope should be done after this point. Inline
  // all `scope.scope` ops.
  pm.addPass(scope::createInlineScopePass(InlineScopeOptions{/*forceInline=*/true}));
  pm.addPass(hivm::createEnableHIVMCCompatiblePrintPass());
  pm.addPass(annotation::createAnnotationLoweringPass());
  // pm.nest<func::FuncOp>().addPass(hivm::createInsertInitAndFinishForDebugPass());
  pm.nest<func::FuncOp>().addPass(hivm::createMarkDisableLoadPass());
  pm.addPass(hivm::createMarkSyncBlockLockWithSubblockPass());
  pm.addPass(hivm::createInsertFreeLockVarBeforeReturnPass());
  pm.addPass(createConvertHIVMToStandardPass());
}

void createAscendOptPipelineImpl(OpPassManager &pm, const AscendOptPipelineOptions &options) {
  pm.addPass(createAKGOperatorIdentifyPass());
  pm.addPass(createHoistTensorSlicePass());
  pm.addPass(createMindsporeMakeBroadcastablePass());
  pm.addPass(createEliminateDimensionPass());
  pm.addPass(createLegalizeTypePass());
  // pm.addPass(createFoldDimensionPass());
  if (options.enableLoopFusion) {
    pm.addPass(createMindSporeToLinalgNamedPass(options.dynamicShape, !options.enableLoopFusion));
    pm.addPass(createCopyReturnedFuncArgsPass());
    pm.addPass(createLinalgGeneralizeNamedOpsPass());
    pm.addPass(createMindSporeToTosaPass());
    pm.addPass(createMindSporeToLinalgPass());
    pm.addPass(createCloneTensorEmptyPass());
  } else {
    pm.addPass(createMindSporeToLinalgNamedPass(options.dynamicShape));
    pm.addPass(createMindSporeToTosaPass());
  }
  OpPassManager &nestedFunctionPM = pm.nest<func::FuncOp>();
  nestedFunctionPM.addPass(tosa::createTosaToLinalg());
  // Erase unused linalg.generic operands/results before bufferization.
  nestedFunctionPM.addPass(createEraseUnusedOperandsAndResultsPass());

  if (options.enableLoopFusion) {
    pm.addPass(createDecomposeTensorPass());
    pm.addPass(bufferization::createEmptyTensorToAllocTensorPass());

    bufferization::OneShotBufferizationOptions bufferizationOpts;
    bufferizationOpts.allowReturnAllocsFromLoops = true;
    bufferizationOpts.bufferizeFunctionBoundaries = true;
    bufferizationOpts.setFunctionBoundaryTypeConversion(bufferization::LayoutMapOption::IdentityLayoutMap);
    pm.addPass(bufferization::createOneShotBufferizePass(bufferizationOpts));
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createMemrefCopyToLoopsPass());
    pm.addPass(createShapeNormalizationPass());

    OpPassManager &nestedFusionPM = pm.nest<func::FuncOp>();
    nestedFusionPM.addPass(createConvertLinalgToAffineLoopsPass());

    // pre-process
    nestedFusionPM.addPass(createCSEPass());
    bool promoteSingleIter = true;
    nestedFusionPM.addPass(affine::createAffineLoopNormalizePass(promoteSingleIter));
    nestedFusionPM.addPass(createCanonicalizerPass());
    nestedFusionPM.addPass(createCopyElisionPass());
    nestedFusionPM.addPass(createUnifyShapePass());
    nestedFusionPM.addPass(createCopyRemovalPass());
    nestedFusionPM.addPass(createCSEPass());
    nestedFusionPM.addPass(createCanonicalizerPass());

    // fusion
    nestedFusionPM.addPass(createRemoveRedundantLoopsPass());
    nestedFusionPM.addPass(createCanonicalizerPass());
    nestedFusionPM.addPass(affine::createAffineReductionAnnotationPass());
    nestedFusionPM.addPass(createHoistLoopIndependentOpsPass());
    nestedFusionPM.addPass(createAKGLoopFusionPass());
    nestedFusionPM.addPass(affine::createAffineLoopInvariantCodeMotionPass());
    nestedFusionPM.addPass(createCanonicalizerPass());
    nestedFusionPM.addPass(createSubviewAllocElimPass());
    nestedFusionPM.addPass(createReductionSiblingRecomputePass());
    nestedFusionPM.addPass(createAffineIteratorConversionPass());
    nestedFusionPM.addPass(createStoreLoadElimPass());
    nestedFusionPM.addPass(createLegalizeTypeForAscendPass());
    nestedFusionPM.addPass(createNormalizePass());
    nestedFusionPM.addPass(createConvertAffineToSCFPass());

    pm.addPass(createSymbolicRemovalPass());
    pm.addPass(createAddOutParameterPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createLegalizeBoolPass());
    // tiling
    pm.addPass(createNPUAutoTilingPass(options.arch));
    pm.addPass(createAllocBufferShrinkPass());
    // vector
    pm.addPass(createSCFForLoopCanonicalizationPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(scf::createNPUVectorVectorizePass());
    pm.addPass(npuvector::createElimScfIterArgsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(npuvector::createEliminateNPUVectorRedundantOpsPass());
    pm.addPass(createCSEPass());
    if (akg::NpuInfo::getInstance(options.arch).isRegBasedArch()) {
      pm.addPass(npuvector::createOutlineVectorFunctionPass());
      pm.addPass(createNPUVectorToVectorPass());
      pm.addPass(vector::createVectorLegalizeTypePass());
    }
    pm.addPass(createArithToHIVMConversionPass());
    pm.addPass(createCanonicalizerPass());
  }

  pm.addPass(hacc::createAppendDeviceSpecPass());

  // hivm optimize pipeline
  if (options.enableHIVMCompile) {
    createHIVMPipeline(pm, options);
  } else {
    pm.addPass(hivm::createInferFuncCoreTypePass());
  }
}
}  // namespace

void createAscendOptPipeline(OpPassManager &pm, const AscendOptPipelineOptions &options) {
  createAscendOptPipelineImpl(pm, options);
}
}  // namespace mlir
