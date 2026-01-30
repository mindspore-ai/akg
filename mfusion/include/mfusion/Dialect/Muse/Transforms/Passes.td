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

//===----------------------------------------------------------------------===//
// MatMul fusion passes
//===----------------------------------------------------------------------===//

def FuseMatMulCast : Pass<"fuse-matmul-cast", "::mlir::ModuleOp"> {
  let summary = "Fuse MatMul/Mm (f16) and Cast (f32) into MatMul with f32 result";
  let description = [{
    Fuses MatMul or Mm with float16 inputs followed by Cast to float32
    into a single MatMul/Mm with float32 result type.
  }];
  let constructor = "mlir::createFuseMatMulCastPass()";
  let dependentDialects = [
    "::mlir::muse::MuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

def FuseMatmulUnsqueezeSqueeze : Pass<"fuse-matmul-unsqueeze-squeeze", "::mlir::ModuleOp"> {
  let summary = "Handle MatMul/Mm with 1D inputs via reshape (unsqueeze/squeeze)";
  let description = [{
    For MatMul/Mm with 1D inputs (1D x N-D, N-D x 1D, 1D x 1D), inserts
    muse.reshape to emulate unsqueeze before inputs and squeeze after output.
  }];
  let constructor = "mlir::createFuseMatmulUnsqueezeSqueezePass()";
  let dependentDialects = [
    "::mlir::muse::MuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

def FuseMatmulTransposeWeight : Pass<"fuse-matmul-transpose-weight", "::mlir::ModuleOp"> {
  let summary = "Insert permute for unaligned MatMul inputs; set trans_x1/trans_x2";
  let description = [{
    For MatMulOp, if an input's inner axis (last dim * element size) is not
    a multiple of 512 bytes, inserts muse.permute and sets trans_x1/trans_x2 accordingly.
    Does not re-match on outputs of its own transformations.
  }];
  let constructor = "mlir::createFuseMatmulTransposeWeightPass()";
  let dependentDialects = [
    "::mlir::muse::MuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

def FuseMatmulReshapeBiasAdd : Pass<"fuse-matmul-reshape-bias-add", "::mlir::ModuleOp"> {
  let summary = "Fuse MatMul -> Reshape -> Add(bias) into MatMul(with bias) -> Reshape";
  let description = [{
    Fuses MatMul -> Reshape -> Add(bias) into MatMul(with bias) -> Reshape.
    Reshape must not split MatMul output's last axis; bias must broadcast to
    Reshape output and size must match MatMul output's last dimension.
  }];
  let constructor = "mlir::createFuseMatmulReshapeBiasAddPass()";
  let dependentDialects = [
    "::mlir::muse::MuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

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
