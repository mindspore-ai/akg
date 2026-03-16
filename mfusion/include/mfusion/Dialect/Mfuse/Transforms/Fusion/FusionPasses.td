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

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// FuseAddRmsNorm pass
//===----------------------------------------------------------------------===//

def FuseAddRmsNorm : Pass<"fuse-addrmsnorm", "::mlir::ModuleOp"> {
  let summary = "Fuse Add and RmsNorm into AddRmsNorm";
  let description = [{
    Fuses a pattern of RmsNorm(Add(x1, x2), gamma, eps) into a single AddRmsNorm operator.
    Checks that the Add operation has a scaling factor of 1.0.
  }];
  let constructor = "mlir::mfuse::createFuseAddRmsNormPass()";
  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// FuseGelu pass
//===----------------------------------------------------------------------===//

def FuseGelu : Pass<"fuse-gelu", "::mlir::ModuleOp"> {
  let summary = "Fuse Pow, Mul, Add, Tanh into Gelu";
  let description = [{
    Fuses the GELU approximation pattern 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    into a single mfuse.gelu operator.
  }];
  let constructor = "mlir::mfuse::createFuseGeluPass()";
  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// MatMul fusion passes
//===----------------------------------------------------------------------===//

def FuseBiasaddConv : Pass<"fuse-biasadd-conv", "::mlir::ModuleOp"> {
  let summary = "Fuse Conv2D (no bias) and Add (bias) into Conv2DWithBias";
  let description = [{
    Fuses Conv2D without bias followed by BiasAdd/Add into a single
    Conv2DWithBias. Does not fuse when Conv2D already has bias or shapes are dynamic.
  }];
  let constructor = "mlir::mfuse::createFuseBiasaddConvPass()";
  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

def FuseConv2DCast : Pass<"fuse-conv2d-cast", "::mlir::ModuleOp"> {
  let summary = "Fuse Conv2D (f16) and Cast (f32) into Conv2D with f32 result";
  let description = [{
    Fuses Conv2D with float16 output followed by Cast to float32 into a single
    Conv2D with float32 result. Does not fuse when Conv2D/Cast is dynamic or
    any Conv2D input is from StridedRead. Conv must have exactly one user (the Cast)
    to fuse; only the Cast op is replaced (Conv is not erased).
  }];
  let constructor = "mlir::mfuse::createFuseConv2DCastPass()";
  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

def FuseMatMulBiasAdd : Pass<"fuse-matmul-bias-add", "::mlir::ModuleOp"> {
  let summary = "Fuse MatMul/BatchMatmul and Add(bias) into MatmulWithBias";
  let description = [{
    Fuses MatMul or BatchMatmul (no bias) followed by Add(bias) into a single
    MatmulWithBias. Constraints: one of add's inputs has rank 1 (bias); bias
    size equals matmul output last dimension; for 2D MatMul output rank is 2;
    no broadcast. Note: if BiasAdd/Add was fp16, fused bias is added in fp32
    inside matmul (precision may improve); disable pass to avoid this change.
  }];
  let constructor = "mlir::mfuse::createFuseMatMulBiasAddPass()";
  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

def FuseMatMulCast : Pass<"fuse-matmul-cast", "::mlir::ModuleOp"> {
  let summary = "Fuse MatMul/Mm (f16) and Cast (f32) into MatMul with f32 result";
  let description = [{
    Fuses MatMul or Mm with float16 inputs followed by Cast to float32
    into a single MatMul/Mm with float32 result type.
    MatMul/MatMulWithBias must have exactly one user (the Cast) to fuse.
  }];
  let constructor = "mlir::mfuse::createFuseMatMulCastPass()";
  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

def FuseMatmulUnsqueezeSqueeze : Pass<"fuse-matmul-unsqueeze-squeeze", "::mlir::ModuleOp"> {
  let summary = "Handle MatMul/Mm with 1D inputs via reshape (unsqueeze/squeeze)";
  let description = [{
    For MatMul/Mm with 1D inputs (1D x N-D, N-D x 1D, 1D x 1D), inserts
    mfuse.reshape to emulate unsqueeze before inputs and squeeze after output.
  }];
  let constructor = "mlir::mfuse::createFuseMatmulUnsqueezeSqueezePass()";
  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

def FuseMatmulTransposeWeight : Pass<"fuse-matmul-transpose-weight", "::mlir::ModuleOp"> {
  let summary = "Insert permute for unaligned MatMul inputs; set trans_x1/trans_x2";
  let description = [{
    For MatMulOp, if an input's inner axis (last dim * element size) is not
    a multiple of 512 bytes, inserts mfuse.permute and sets trans_x1/trans_x2 accordingly.
    Does not re-match on outputs of its own transformations.
  }];
  let constructor = "mlir::mfuse::createFuseMatmulTransposeWeightPass()";
  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
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
  let constructor = "mlir::mfuse::createFuseMatmulReshapeBiasAddPass()";
  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect"
    ];
}

//===----------------------------------------------------------------------===
// FuseBatchMatMul pass (transpose elimination + BatchMatMul 2D -> MatMul)
//===----------------------------------------------------------------------===

def FuseBatchMatMul : Pass<"fuse-batch-matmul", "::mlir::ModuleOp"> {
  let summary = "Fuse BatchMatMul: eliminate Permute into trans flag; BatchMatMul 2D -> MatMul";
  let description = [{
    Mode 1: When MatMul/BatchMatMul has one or both inputs from Permute that only
    swaps the last two dimensions, use permute input and flip trans flag to eliminate
    the permute. Matmul input dtypes: float16, float32, bfloat16 only.
    Mode 2: When BatchMatMul both inputs are 2D, convert BatchMatMul to MatMul.
  }];
  let constructor = "mlir::mfuse::createFuseBatchMatMulPass()";
  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===
// FuseBatchMatMulToMul pass
//===----------------------------------------------------------------------===

def FuseBatchMatMulToMul : Pass<"fuse-batchmatmul-to-mul", "::mlir::ModuleOp"> {
  let summary = "Fuse BatchMatMul/MatMul with k=1 to Mul";
  let description = [{
    Fuses BatchMatMul/MatMul/MatMulv2 operators with k=1 dimension to Mul operators
    to improve performance. Inserts reshape operators when adj/transpose is true.
    Only applies to static shapes without bias.
    Supports Float32, Float16, and BFloat16 data types.
  }];
  let constructor = "mlir::mfuse::createFuseBatchMatMulToMulPass()";
  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

def FuseMatMulReshape : Pass<"fuse-matmul-reshape", "::mlir::ModuleOp"> {
  let summary = "Fuse MatMul/MatMulv2 with N=1 to add Reshape to second input";
  let description = [{
    Fuses MatMul/MatMulv2 operators with N=1 dimension to add Reshape node to the second input.
    Only applies to static shapes.
    Supports BFLOAT16, FLOAT16, and FLOAT32 data types.
    x1 input shape: [M, K], x2 input shape (without transpose): [K, 1].
    For BFLOAT16 or FLOAT16 data types, K < 27392 for the fusion to apply.
  }];
  let constructor = "mlir::mfuse::createFuseMatMulReshapePass()";
  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// FuseRmsNorm pass: fuse decomposed RmsNorm chain to aclnn.rms_norm
//===----------------------------------------------------------------------===//

def FuseRmsNorm : Pass<"fuse-rms-norm", "::mlir::ModuleOp"> {
  let summary = "Fuse decomposed RmsNorm (pow+mean+add+rsqrt+mul+mul) into aclnn.rms_norm";
  let description = [{
    Matches the pattern: pow(x,2) -> mean.dim -> add.Scalar(eps) -> rsqrt -> mul(x, rsqrt) -> mul(gamma),
    and replaces it with a single mfuse.aclnn.rms_norm(x, gamma, epsilon).
    Supports torch.aten for pow/mean/add and mfuse for rsqrt/mul.
    Also supports decomposed-only chain (add+rsqrt+mul+mul) without pow/mean.
  }];
  let constructor = "mlir::mfuse::createFuseRmsNormPass()";
  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// FuseSwiGlu fusion passes
//===----------------------------------------------------------------------===//

def FuseSwiGlu : Pass<"fuse-swi-glu", "::mlir::ModuleOp"> {
  let summary = "Fuse SplitWithSize, Silu and Mul into SwiGlu (with or without Reshape)";
  let description = [{
    Fuses patterns of SwiGlu operations into a single SwiGlu operator:
    1. Mul(SplitWithSize(input, ...)[0], Silu(SplitWithSize(input, ...)[1]))
    2. Mul(Reshape(SplitWithSize(Reshape(input))[0]), Silu(Reshape(SplitWithSize(Reshape(input))[1])))
  }];
  let constructor = "mlir::mfuse::createFuseSwiGluPass()";
  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect"
  ];
}


//===----------------------------------------------------------------------===//
// MfuseFusion unified pass
//===----------------------------------------------------------------------===//

def MfuseFusion : Pass<"mfuse-fusion", "::mlir::ModuleOp"> {
  let summary = "Unified fusion pass containing all fusion patterns";
  let description = [{
    Unified fusion pass that runs all fusion patterns.
  }];
  let constructor = "mlir::mfuse::createMfuseFusionPass()";
  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect"
  ];
}
