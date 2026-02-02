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

#ifndef MFUSION_DIALECT_MUSE_TRANSFORMS_PASSES
#define MFUSION_DIALECT_MUSE_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Include Fusion Passes
//===----------------------------------------------------------------------===//

include "mfusion/Dialect/Muse/Transforms/Fusion/FusionPasses.td"

//===----------------------------------------------------------------------===//
// Decompose pass
//===----------------------------------------------------------------------===//

def Decompose : Pass<"decompose", "::mlir::ModuleOp"> {
  let summary = "Decompose operations in MUSE dialect";

  let description = [{
    This pass decomposes operations in MUSE dialect into simpler operations.
    The pass uses OpRewritePattern-based decomposition that automatically
    matches and decomposes some operations.
  }];

  let constructor = "mlir::createDecomposePass()";

  let dependentDialects = [
    "::mlir::muse::MuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// GraphKernel Cluster Passes
//===----------------------------------------------------------------------===//

def DVMCluster : Pass<"muse-dvm-cluster", "::mlir::func::FuncOp"> {
  let summary = "Cluster Muse operations checked by DVM backend into muse.fused";

  let description = [{
    This pass clusters operations checked by DVM backend into fused
    subgraphs represented by muse.fused operations. It uses a Union-Find
    algorithm to merge compatible operations while ensuring no cycles
    are introduced in the dependency graph.

    The pass performs the following steps:
    1. Build a dependency graph from the function's operations
    2. Find candidate operations that can be clustered together
    3. Remove candidates that would introduce cycles
    4. Merge remaining candidates into clusters
    5. Create muse.fused operations for each cluster
  }];

  let constructor = "mlir::createDVMClusterPass()";

  let dependentDialects = [
    "::mlir::muse::MuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// MatMul conversion passes
//===----------------------------------------------------------------------===//

def ConvertAclnnMatmulToMatmul : Pass<"convert-aclnn-matmul-to-matmul", "::mlir::ModuleOp"> {
  let summary = "Convert aclnn.mm / aclnn.matmul / aclnn.batch_matmul to muse.matmul or muse.matmul_with_bias";
  let description = [{
    Converts muse.aclnn.mm, muse.aclnn.matmul, and muse.aclnn.batch_matmul to muse.matmul
    (trans_x1=false, trans_x2=false). Fuses aclnn matmul-like + aclnn.add(result, bias, alpha=1)
    into muse.matmul_with_bias, so that downstream MatMul fusion passes can operate on a
    unified matmul representation.
  }];
  let constructor = "mlir::createConvertAclnnMatmulToMatmulPass()";
  let dependentDialects = [
    "::mlir::muse::MuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Recompose pass
//===----------------------------------------------------------------------===//

def Recompose : Pass<"recompose", "::mlir::ModuleOp"> {
  let summary = "Recompose (lower) operations in MUSE dialect to aclnn operations";

  let description = [{
    This pass recomposes (lowers) operations in MUSE dialect to aclnn operations.
    The pass uses OpRewritePattern-based recomposition that automatically
    matches and lowers Muse operations to aclnn operations.
  }];

  let constructor = "mlir::createRecomposePass()";

  let dependentDialects = [
    "::mlir::muse::MuseDialect",
    "::mlir::func::FuncDialect",
    "::mlir::arith::ArithDialect"
  ];
}

#endif // MFUSION_DIALECT_MUSE_TRANSFORMS_PASSES
