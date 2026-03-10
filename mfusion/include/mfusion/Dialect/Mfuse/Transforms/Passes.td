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

#ifndef MFUSION_DIALECT_MFUSE_TRANSFORMS_PASSES
#define MFUSION_DIALECT_MFUSE_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Include Fusion Passes
//===----------------------------------------------------------------------===//

include "mfusion/Dialect/Mfuse/Transforms/Fusion/FusionPasses.td"

//===----------------------------------------------------------------------===//
// Decompose pass
//===----------------------------------------------------------------------===//

def Decompose : Pass<"decompose", "::mlir::ModuleOp"> {
  let summary = "Decompose operations in MFUSE dialect";

  let description = [{
    This pass decomposes operations in MFUSE dialect into simpler operations.
    The pass uses OpRewritePattern-based decomposition that automatically
    matches and decomposes some operations.
  }];

  let options = [
    Option<"patternType", "pattern-type", "std::string", "\"ALL\"",
           "Type of decompose patterns to use (NONE, ALL, BEFORE_MANUAL_FUSION, AFTER_MANUAL_FUSION)" >,
    ListOption<"opList", "op-list", "std::string", 
               "List of specific operations to decompose (empty means all)" >
  ];

  let constructor = "mlir::createDecomposePass()";

  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// GraphKernel Cluster Passes
//===----------------------------------------------------------------------===//

def AKGCluster : Pass<"mfuse-akg-cluster", "::mlir::func::FuncOp"> {
  let summary = "Cluster Mfuse operations checked by AKG backend into mfuse.fused";

  let description = [{
    This pass clusters operations checked by AKG backend into fused
    subgraphs represented by mfuse.fused operations. It uses a Union-Find
    algorithm to merge compatible operations while ensuring no cycles
    are introduced in the dependency graph.

    The pass performs the following steps:
    1. Build a dependency graph from the function's operations
    2. Find candidate operations that can be clustered together
    3. Remove candidates that would introduce cycles
    4. Merge remaining candidates into clusters
    5. Create mfuse.fused operations for each cluster
  }];

  let constructor = "mlir::createAKGClusterPass()";
  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

def DVMCluster : Pass<"mfuse-dvm-cluster", "::mlir::func::FuncOp"> {
  let summary = "Cluster Mfuse operations checked by DVM backend into mfuse.fused";

  let description = [{
    This pass clusters operations checked by DVM backend into fused
    subgraphs represented by mfuse.fused operations. It uses a Union-Find
    algorithm to merge compatible operations while ensuring no cycles
    are introduced in the dependency graph.

    The pass performs the following steps:
    1. Build a dependency graph from the function's operations
    2. Find candidate operations that can be clustered together
    3. Remove candidates that would introduce cycles
    4. Merge remaining candidates into clusters
    5. Create mfuse.fused operations for each cluster
  }];

  let constructor = "mlir::createDVMClusterPass()";

  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Recompose pass
//===----------------------------------------------------------------------===//

def Recompose : Pass<"recompose", "::mlir::ModuleOp"> {
  let summary = "Recompose (lower) operations in MFUSE dialect to aclnn operations";

  let description = [{
    This pass recomposes (lowers) operations in MFUSE dialect to aclnn operations.
    The pass uses OpRewritePattern-based recomposition that automatically
    matches and lowers Mfuse operations to aclnn operations.
  }];

  let constructor = "mlir::createRecomposePass()";

  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect",
    "::mlir::arith::ArithDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Outlining Passes
//===----------------------------------------------------------------------===//

def OutlineMfuseFusedSubgraphs : Pass<"outline-mfuse-fused-subgraphs", "::mlir::ModuleOp"> {
  let summary = "Outline Mfuse fused subgraphs into functions";
  let description = [{
    Identifies mfuse.fused operations, outlines them into new functions, and replaces
    the original ops with func.call. The outlined functions are tagged with the
    attribute 'mfusion.outlined'.
  }];
  let constructor = "mlir::createOutlineMfuseFusedSubgraphsPass()";
  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect",
    "::mlir::arith::ArithDialect"
  ];
}

def CopyFusedSubgraphs : Pass<"copy-fused-subgraphs", "::mlir::ModuleOp"> {
  let summary = "Clone outlined fusion functions";
  let description = [{
    Clones each outlined fusion function and stores the clone's name in the
    original function's `mfusion.copied_subgraph` attribute. The cloned function
    serves as the shape inference function for the DVM kernel.
  }];
  let constructor = "mlir::createCopyFusedSubgraphsPass()";
  let dependentDialects = [
    "::mlir::func::FuncDialect"
  ];
}

def ConvertDvmSubgraphToMfuseDvmCall : Pass<"convert-dvm-subgraph-to-mfuse-dvm-call", "::mlir::ModuleOp"> {
  let summary = "Replace DVM subgraph calls with mfuse.dvm_call";
  let description = [{
    Replaces func.call sites that target outlined DVM subgraphs with mfuse.dvm_call,
    embeds a self-contained DVM MLIR string, and removes the DVM subgraph function.
  }];
  let constructor = "mlir::createConvertDvmSubgraphToMfuseDvmCallPass()";
  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect",
    "::mlir::dvm::DvmDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Split pass
//===----------------------------------------------------------------------===//

def Split : Pass<"split", "::mlir::func::FuncOp"> {
  let summary = "Split fuse operations";
  let description = [{
    This pass splits fuse ops.
    It groups operations into areas and applies split patterns to optimize the execution.
  }];
  let options = [
    Option<"kernelGenerator", "kernel-generator", "std::string", "\"DVM\"",
           "Kernel generator type for split (DVM or AKG)">
  ];
  let constructor = "mlir::createSplitPass()";
  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

#endif // MFUSION_DIALECT_MFUSE_TRANSFORMS_PASSES
