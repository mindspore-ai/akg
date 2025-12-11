/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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

#ifndef AKG_MLIR_DIALECT_AFFINE_PASSES
#define AKG_MLIR_DIALECT_AFFINE_PASSES

include "mlir/Pass/PassBase.td"

def AffineLoopReorder : Pass<"affine-loop-reorder", "func::FuncOp"> {
  let summary = "Reorder loops with specific order";
  let constructor = "mlir::createAffineLoopReorderPass()";
  let options = [
    ListOption<"newOrder", "new-order", "int",
            "Reorder loops with specific new order">,
  ];
}

def AKGLoopFusion : Pass<"akg-loop-fusion", "func::FuncOp"> {
  let summary = "Fuse compatible loops for optimization";
  let description = [{
    AKGLoopFusion pass identifies and fuses compatible loops to improve performance.
    This pass analyzes loop dependencies and bounds to determine safe fusion opportunities.
  }];
  let constructor = "mlir::createAKGLoopFusionPass()";
}

def AffineHandleBoundaryIfExtract : Pass<"affine-handle-boundary-if-extract", "func::FuncOp"> {
  let summary = "Extract the boundary-if in nest loops cases";
  let constructor = "mlir::createAffineHandleBoundaryIfExtract()";
}

def AffineHandleBoundaryIfRestore : Pass<"affine-handle-boundary-if-restore", "mlir::ModuleOp"> {
  let summary = "Restore the boundary-if in nest loops cases";
  let constructor = "mlir::createAffineHandleBoundaryIfRestore()";
}

def ForceConvertAffineForToAffineParallel : Pass<"force-convert-affine-for-to-affine-parallel", "func::FuncOp"> {
  let summary = "Force convert affine.for to affine.parallel in affine dialect. Currently used for gpu reduction ops";
  let constructor = "mlir::createForceConvertAffineForToAffineParallelPass()";
  let options = [
    Option<"matchOpType", "match-op-type", "std::string", /*default=*/"",
           "Forcely convert affine.for to affine.parallel according to op type."
           "Currently only support reduction ops and work on cuda backend.">,
  ];
  let dependentDialects = [
    "arith::ArithDialect"
  ];
}

def AKGAffineLoopTiling : Pass<"akg-affine-loop-tile", "func::FuncOp"> {
  let summary = "AKG Tile affine loop nests";
  let constructor = "mlir::createAKGLoopTilingPass()";
  let options = [
    Option<"cacheSizeInKiB", "cache-size", "uint64_t", /*default=*/"512",
           "Set size of cache to tile for in KiB (default: 512)">,
    Option<"inequalityConvertToIf", "inequality-convert-to-if", "bool", /*default=*/"false",
           "Change the upper and lower bounds of the for loop to the equation,"
           "and insert an if statement into the innermost for loop. (default: false)">,
    Option<"separate", "separate", "bool", /*default=*/"false",
           "Separate full and partial tiles: use if-else to separate (default: false)">,
    Option<"separateNoIf", "separate-no-if", "bool", /*default=*/"false",
           "Separate full and partial tiles: replace if-else with the index upper and lower bounds of the loop (default: false)">,
    Option<"useAutoTiling", "use-auto-tiling", "bool", /*default=*/"false",
           "Whether to use auto tiling (default: false)">,
    Option<"tileSize", "tile-size", "unsigned", /*default=*/"1u",
           "Use this tile size for all loops">,
    ListOption<"tileSizes", "tile-sizes", "unsigned",
               "List of tile sizes for each perfect nest "
               "(overridden by -tile-size)">,
    ListOption<"multiTileSizes", "multi-tile-sizes", "unsigned",
               "List of tile sizes for multi-level tiling">,
  ];
}

def UnifyShape : Pass<"unify-shape", "mlir::ModuleOp"> {
  let summary = "Remove (if possible) expand/collapse_shape and update the usage of the corresponding variable";
  let description = [{
    Remove expand/collapse_shape and update the usage of the corresponding variable.
    Only correct the access function of the variables, not modifying the iterator.
  }];
  let constructor = "mlir::createUnifyShapePass()";
  let options = [
    Option<"keepArgsShape", "keep-argument-shape", "bool",
           "false", "Keep initial arguments shape (default=false)">,
    Option<"allowNonPolyhedralAccess", "allow-non-polyhedral-access", "bool",
           "false", "Keep generation of non polyhedral access when remove collapse/expand_shape, if not allow some may remain (default=false)">,
  ];
}

def SimplifyShape : Pass<"simplify-shape", "mlir::ModuleOp"> {
  let summary = "Remove dimension 1 from shapes";
  let description = [{
    Remove dimension 1 from shapes
  }];
  let constructor = "mlir::createSimplifyShapePass()";
  let options = [
    Option<"keepArgsShape", "keep-argument-shape", "bool",
           "false", "Keep initial arguments shape (default=false)">,
  ];
}

def AffineIteratorConversion : Pass<"affine-load-removal", "func::FuncOp"> {
  let summary = "yield affine loops";
  let constructor = "mlir::createAffineIteratorConversionPass()";
}

def AffineTailBlockTiling : Pass<"affine-tail-block-tiling", "func::FuncOp"> {
  let summary = "affine tail block tiling";
  let constructor = "mlir::createAffineTailBlockTilingPass()";
}

def AKGAffineLoopUnroll : Pass<"akg-affine-loop-unroll", "func::FuncOp"> {
  let summary = "AKG Unroll affine loops";
  let constructor = "mlir::createAKGLoopUnrollPass()";
}

def AKGAffineLoopParallelize : Pass<"akg-affine-loop-parallelize", "func::FuncOp"> {
  let summary = "AKG Parallelize affine loops";
  let constructor = "mlir::createAKGLoopParallelizePass()";
  let options = [
    Option<"maxNested", "max-nested", "unsigned", /*default=*/"1u",
           "Maximum number of nested parallel loops to produce. "
           "Defaults to 1.">,
    Option<"parallelReductions", "parallel-reductions", "bool",
           /*default=*/"false",
           "Whether to parallelize reduction loops. Defaults to be false.">
  ];
}

def MergeFusionOp : Pass<"merge-fusion-op", "func::FuncOp"> {
  let summary = "Move all operators between two for loops up or down to the innermost for loop";
  let options = [
    Option<"mergeSpecificOp", "merge-specific-op", "bool", "false",
    "Merge specific op: contains the outermost ops of the for loop and the reduce initialization statement. (default=false)">,
  ];
  let constructor = "mlir::createMergeFusionOpPass()";
}

def AffineMemoryPromotion : Pass<"affine-memory-promotion", "func::FuncOp"> {
  let summary = "AKG memory promotion on affine";
  let constructor = "mlir::createAffineMemoryPromotionPass()";
  let dependentDialects = ["memref::MemRefDialect"];
  let options = [
    Option<"fastMemoryCapacity", "fast-mem-capacity", "uint64_t",
           /*default=*/"std::numeric_limits<uint64_t>::max()",
           "Set fast memory space capacity in KiB (default: unlimited)">,
    Option<"fastMemorySpace", "fast-mem-space", "unsigned",
           /*default=*/"1",
           "Fast memory space identifier for copy generation (default: 1)">,
    Option<"generateDma", "generate-dma", "bool",
           /*default=*/"true", "Generate DMA instead of point-wise copy">,
    Option<"minDmaTransferSize", "min-dma-transfer", "int",
           /*default=*/"1024",
           "Minimum DMA transfer size supported by the target in bytes">,
    Option<"slowMemorySpace", "slow-mem-space", "unsigned",
           /*default=*/"0",
           "Slow memory space identifier for copy generation (default: 0)">,
    Option<"skipNonUnitStrideLoops", "skip-non-unit-stride-loops", "bool",
           /*default=*/"false", "Testing purposes: avoid non-unit stride loop "
                                "choice depths for copy placement">,
    Option<"tagMemorySpace", "tag-mem-space", "unsigned",
           /*default=*/"0",
           "Tag memory space identifier for copy generation (default: 0)">,
  ];
}

def GenerateSingleAffineParallel : Pass<"generate-single-affine-parallel", "func::FuncOp"> {
  let summary = "Generate a single affine parallel op that warps the body of func if there isn't any for inside func.";
  let constructor = "mlir::createGenerateSingleAffineParallelPass()";
  let options = [
    Option<"forceGen", "force-gen", "unsigned",
           /*default=*/"0",
           "`0` means generate a fake for on top only when there is no loop; while `1` means generate no matter what">,
  ];
  let dependentDialects = [
    "mlir::affine::AffineDialect"
  ];
}

def ExtractIfOp : Pass<"extract-if", "func::FuncOp"> {
  let summary = "Extract the affine::AffineIfOp inside the loop and move the statements inside the ThenBlock out of the loop.";
  let options = [
    Option<"extractAllOp", "extract-all-op", "bool", "false",
    "CPU backend: the reduce initialization statement does not need to be extracted. (default=true)">,
  ];
  let constructor = "mlir::createExtractIfOpPass()";
}

def FixDynamicIndexing : Pass<"fix-dynamic-indexing", "func::FuncOp"> {
  let summary = "Fix the dynamic indexing in affine.";
  let constructor = "mlir::createFixDynamicIndexingPass()";
  let dependentDialects = [
    "mlir::affine::AffineDialect"
  ];
}

def ReplaceUnknownDimsToOutputDim : Pass<"replace-unknown-dims-to-output-dim", "func::FuncOp"> {
  let summary = "Replace unknown dims to output dim.";
  let constructor = "mlir::createReplaceUnknownDimsToOutputDimPass()";
  let dependentDialects = [
    "mlir::affine::AffineDialect"
  ];
}

def RemoveRedundantLoops : Pass<"remove-redundant-loops", "func::FuncOp"> {
  let summary = "Remove redundant for loops.";
  let constructor = "mlir::createRemoveRedundantLoopsPass()";
}

def WorkaroundFixReduceInitialization : Pass<"workaround-fix-reduce-initialization", "func::FuncOp"> {
  let summary = "Workaround to fix the location of the initialization of the reduce variable.";
  let description = [{
    Workaround to move the initialization of the reduce variable, near the usage and the reduction of it.
    The initialization must be inside a 'if'.
    This pass is present because else we will have an issue during the memory-promotion pass that fail
    when the initialization is not near the usage.
    When fix memory-promotion, this pass may disappear. But it may require more work to do that this pass.
  }];
  let constructor = "mlir::createWorkaroundFixReduceInitializationPass()";
}

def AffineReductionAnnotation : Pass<"affine-reduction-annotation", "func::FuncOp"> {
  let summary = "Annotate reduction operations with reduction axes and direction in affine dialect";
  let constructor = "mlir::affine::createAffineReductionAnnotationPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "mlir::affine::AffineDialect"
  ];
}

def AKGAffineDataCopyGeneration : Pass<"akg-affine-data-copy-generate", "func::FuncOp"> {
  let summary = "Generate explicit copying for affine memory operations";
  let constructor = "mlir::affine::createAKGAffineDataCopyGenerationPass()";
  let dependentDialects = ["memref::MemRefDialect"];
  let options = [
    Option<"fastMemoryCapacity", "fast-mem-capacity", "uint64_t",
           /*default=*/"std::numeric_limits<uint64_t>::max()",
           "Set fast memory space capacity in KiB (default: unlimited)">,
    Option<"fastMemorySpace", "fast-mem-space", "unsigned",
           /*default=*/"1",
           "Fast memory space identifier for copy generation (default: 1)">,
    Option<"generateDma", "generate-dma", "bool",
           /*default=*/"true", "Generate DMA instead of point-wise copy">,
    Option<"minDmaTransferSize", "min-dma-transfer", "int",
           /*default=*/"1024",
           "Minimum DMA transfer size supported by the target in bytes">,
    Option<"slowMemorySpace", "slow-mem-space", "unsigned",
           /*default=*/"0",
           "Slow memory space identifier for copy generation (default: 0)">,
    Option<"skipNonUnitStrideLoops", "skip-non-unit-stride-loops", "bool",
           /*default=*/"false", "Testing purposes: avoid non-unit stride loop "
                                "choice depths for copy placement">,
    Option<"tagMemorySpace", "tag-mem-space", "unsigned",
           /*default=*/"0",
           "Tag memory space identifier for copy generation (default: 0)">,
  ];
}

def VectorTransferTensorize : Pass<"vector-transfer-tensorize", "mlir::func::FuncOp"> {
  let summary = "Promote vector.transfer_{read|write} into pure tensor/memref ops";
  let constructor = "mlir::affine::createVectorTransferTensorizePass()";
}

def AffineForVectPass : Pass<"affine-for-vectorize", "mlir::func::FuncOp"> {
  let summary = "Vectorize innermost affine.for loops (1-D, configurable VF)";
  let description = [{
    Vectorizes only leaf (innermost) affine.for loops using AKG's super-vectorize
    functionality. For dynamic trip count or ub==1, uses --default-vf (default 512).
    Supports iter_args reductions; skips broadcast-write stores.
  }];
  let constructor = "mlir::affine::createAffineForVectPass()";
  let options = [
    Option<"defaultVF", "default-vf", "int64_t", /*default=*/"512",
           "Vector factor used when ub is dynamic or equal to 1">
  ];
}

def TensorizeLiveOuts : Pass<"tensorize-live-outs", "mlir::func::FuncOp"> {
  let summary = "Thread tensor.insert_slice live-outs through affine.for via iter_args/yields and hoist to parents";
  let description = [{
    This pass discovers tensor.insert_slice updates to the same logical base tensor
    inside affine.for loops and threads the base as an explicit loop-carried value:

      - Inside a loop: chains of tensor.insert_slice on the same base are rewired
        to start from the loop's iter_arg, with the final value yielded by affine.yield.
      - After the loop: all uses of the original base are redirected to the yielded result.
      - Across nesting: live-out values used outside the immediate loop are progressively
        propagated to parent loops, merging sibling loops' requirements at each level.

    The result is explicit SSA of loop-carried tensor state, enabling cleaner canonicalization
    and downstream bufferization, and removing hidden dependencies on implicit loop-scoped bases.
  }];
  let constructor = "mlir::affine::createTensorizeLiveOutsPass()";
}

def TilingFunc : Pass<"tiling-func", "mlir::ModuleOp"> {
  let summary = "This pass applies tiling to operations on tensors, dynamically processing tiles";
  let constructor = "mlir::affine::createTilingFuncPass()";
}


def InsertLoadAndStore : Pass<"insert-load-and-store", "func::FuncOp"> {
  let summary = "Insert hfusion.load and hfusion.store operations";
  let description = [{
    This pass inserts hfusion.load and hfusion.store
  }];
  let constructor = "mlir::createInsertLoadAndStorePass()";
  let dependentDialects = [
    "tensor::TensorDialect",
    "annotation::AnnotationDialect",
    "hfusion::HFusionDialect",
  ];
}

def AddOutParameter : Pass<"add-out-parameter", "mlir::ModuleOp"> {
  let summary = "This pass applies to adding out parameters";
  let constructor = "mlir::affine::createAddOutParameterPass()";
}

def MemrefTilingFunc : Pass<"memref-tiling-func", "mlir::ModuleOp"> {
  let summary = "This pass applies tiling to operations on tensors, dynamically processing tiles";
  let constructor = "mlir::affine::createMemrefTilingFuncPass()";
}

def BF16ToF32 : Pass<"bf16-to-f32", "func::FuncOp"> {
  let summary = "Convert bf16 operands to float32 in affine operations";
  let description = [{
    This pass converts bf16 operands to float32 type in affine operations.
    It inserts type conversion operations (arith::ExtFOp) to extend bf16 values
    to float32 before they are used in affine operations.
  }];
  let constructor = "mlir::createBF16ToF32Pass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "affine::AffineDialect"
  ];
}

#endif // AKG_MLIR_DIALECT_AFFINE_PASSES
