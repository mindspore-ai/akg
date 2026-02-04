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
// FuseAddRmsNorm pass
//===----------------------------------------------------------------------===//

def FuseAddRmsNorm : Pass<"fuse-addrmsnorm", "::mlir::ModuleOp"> {
  let summary = "Fuse Add and RmsNorm into AddRmsNorm";
  let description = [{
    Fuses a pattern of RmsNorm(Add(x1, x2), gamma, eps) into a single AddRmsNorm operator.
    Checks that the Add operation has a scaling factor of 1.0.
  }];
  let constructor = "mlir::createFuseAddRmsNormPass()";
  let dependentDialects = [
    "::mlir::muse::MuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// GeluFusion pass
//===----------------------------------------------------------------------===//

def FuseGelu : Pass<"fuse-gelu", "::mlir::ModuleOp"> {
  let summary = "Fuse Pow, Mul, Add, Tanh into Gelu";
  let description = [{
    Fuses the GELU approximation pattern 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    into a single muse.gelu operator.
  }];
  let constructor = "mlir::createFuseGeluPass()";
    let dependentDialects = [
      "::mlir::muse::MuseDialect",
      "::mlir::func::FuncDialect"
  ];
}

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

#endif // MFUSION_DIALECT_MUSE_TRANSFORMS_PASSES
