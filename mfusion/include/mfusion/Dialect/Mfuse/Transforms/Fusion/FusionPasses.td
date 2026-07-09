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
// CanonicalizeBinaryScalarOperands pass
//===----------------------------------------------------------------------===//

def CanonicalizeBinaryScalarOperands : Pass<"mfuse-canonicalize-binary-scalar-operands", "::mlir::ModuleOp"> {
  let summary = "Recover NumToTensor scalar operands feeding binary ops and canonicalize commutative scalar side";
  let description = [{
    Recovers mfuse scalar constants from constant -> num_to_tensor -> binary
    patterns before FuseNumToTensor can materialize them as full tensors. For
    commutative binary ops, scalar lhs is canonicalized to rhs. Non-commutative
    ops preserve operand side.
  }];
  let constructor = "mlir::mfuse::createCanonicalizeBinaryScalarOperandsPass()";
  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
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
  let constructor = "mlir::mfuse::createFuseAddRmsNormPass()";
  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// FuseLayerNorm pass
//===----------------------------------------------------------------------===//

def FuseLayerNorm : Pass<"fuse-layernorm", "::mlir::ModuleOp"> {
  let summary = "Fuse var_mean, add, rsqrt, sub, mul, mul, add into LayerNorm";
  let description = [{
    Fuses a pattern of var_mean -> add(eps) -> rsqrt -> sub -> mul -> mul -> add
    into a single mfuse.aclnn.layer_norm operator.
    Pattern: LayerNorm(x, gamma, beta, eps) = ((x - mean) * rsqrt(var + eps)) * gamma + beta
  }];
  let constructor = "mlir::mfuse::createFuseLayerNormPass()";
  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// FuseLayerNormDvm pass
//===----------------------------------------------------------------------===//

def FuseLayerNormDvm : Pass<"fuse-layer-norm-dvm", "::mlir::ModuleOp"> {
  let summary = "Tag BERT LayerNorm subgraphs for DVM split/codegen";
  let description = [{
    Tag decomposed LayerNorm subgraphs via the generic fusion-region API
    (mfusion.dvm_fuse_group / mfusion.dvm_fuse_role member|affinity). Legacy unit attrs
    mfusion.layer_norm_dvm are also set for compatibility. Split uses
    FuseTagBarrierByGroupId to merge adjacent areas sharing the same group id (member
    only; affinity nodes are not merge hubs).
  }];
  let constructor = "mlir::mfuse::createFuseLayerNormDvmPass()";
  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Softmax fusion passes
//===----------------------------------------------------------------------===//

def FuseSafeSoftmaxDvm : Pass<"fuse-safe-softmax-dvm", "::mlir::ModuleOp"> {
  let summary = "Fuse safe-softmax regions into mfuse.fused for DVM";
  let description = [{
    No-op unless the module contains safe-softmax candidate regions (broadcast
    mfuse.select with a zero-filled branch and mfuse.softmax / torch.aten._softmax
    or an already decomposed softmax chain). When active, sets
    mfusion.safe_softmax_pipeline_active and fuses each matched region into a
    single mfuse.fused DVM island (decomposing mfuse.softmax inside the pattern).
    Non-safe softmax (e.g. BERT standalone or pre-softmax mask) is left unchanged.
  }];
  let constructor = "mlir::mfuse::createFuseSafeSoftmaxDvmPass()";
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
// FuseLogicalNotCompare pass
//===----------------------------------------------------------------------===//

def FuseLogicalNotCompare : Pass<"fuse-logical-not-compare", "::mlir::ModuleOp"> {
  let summary = "Fold logical_not(compare(x, y)) into the opposite comparison";
  let description = [{
    Rewrites logical_not over comparison ops into the opposite comparison:
    eq<->ne always, and gt<->le / ge<->lt only when operands are provably
    NaN-free. In practice this includes integer/index operands and floating-point
    constants that can be shown not to contain NaN. External floating-point
    inputs are left unchanged unless their defining ops prove they are NaN-free,
    so IEEE 754 NaN semantics are preserved.
  }];
  let constructor = "mlir::mfuse::createFuseLogicalNotComparePass()";
  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// MatMul fusion passes
//===----------------------------------------------------------------------===//

def FuseBiasaddConv : Pass<"fuse-biasadd-conv", "::mlir::ModuleOp"> {
  let summary = "Fuse aclnn.conv2d (no bias) and Add (bias) into aclnn.conv2d_with_bias";
  let description = [{
    Fuses aclnn.conv2d without bias followed by BiasAdd/Add into a single
    aclnn.conv2d_with_bias. Does not fuse when conv already has bias or shapes are dynamic.
  }];
  let constructor = "mlir::mfuse::createFuseBiasaddConvPass()";
  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

def FuseConv2DCast : Pass<"fuse-conv2d-cast", "::mlir::ModuleOp"> {
  let summary = "Fuse aclnn.conv2d (f16) and Cast (f32) into aclnn.conv2d with f32 result";
  let description = [{
    Fuses aclnn.conv2d with float16 output followed by Cast to float32 into a single
    aclnn.conv2d with float32 result. Does not fuse when conv/Cast is dynamic or
    any conv input is from StridedRead. Conv must have exactly one user (the Cast)
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

//===----------------------------------------------------------------------===
// FuseNumToTensor pass
//===----------------------------------------------------------------------===

def FuseNumToTensor : Pass<"fuse-num-to-tensor", "::mlir::ModuleOp"> {
  let summary = "Fuse NumToTensor (Cast) operations into a full op";
  let description = [{
    Fuses a pattern of Cast(NumToTensor(constant))  or single NumToTensor(constant) into a full operation.
    This eliminates redundant operations by creating a full op with the desired type and shape on npu device.
  }];
  let constructor = "mlir::mfuse::createFuseNumToTensorPass()";
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

def FuseBatchNorm : Pass<"fuse-batch-norm", "::mlir::ModuleOp"> {
  let summary = "Fuse decomposed eval batch norm into mfuse.aclnn.batch_norm";
  let description = [{
    The pass flag `fuse-batch-norm` matches **eval-mode** BN subgraphs written as
    decomposed `torch.aten.batch_norm`-like chains. The fused mfuse BN op is always
    `mfuse.aclnn.batch_norm` (eval defaults on attributes).

    Recognizes the mfuse decomposition (sub, inv_std via sqrt/reciprocal, optional mul-by-1,
    channel reshapes, mul, add) and replaces it with `mfuse.aclnn.batch_norm`.

    By default, mean and running_var must be provably computed from `x` as E(x) / Var(x)
    over NCHW spatial dims. Enable `allow-external-batch-norm-stats` to also accept
    inference-style running_mean / running_var from block arguments, constants, or
    reshape(constants) without requiring that online subgraph.
  }];
  let options = [
    Option<"allowExternalBatchNormStats", "allow-external-batch-norm-stats", "bool", "false",
           "When true, accept external running_mean/running_var (args, constants, or "
           "reshape(constants)). Default false requires explicit E(x)/Var(x) validation.">
  ];
  let constructor = "mlir::mfuse::createFuseBatchNormPass()";
  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

def FuseConvBatchNorm : Pass<"fuse-conv-batchnorm", "::mlir::ModuleOp"> {
  let summary = "Fold aclnn.conv2d[/with_bias] + aclnn.batch_norm into aclnn.conv2d_with_bias";
  let description = [{
    Fuses `mfuse.aclnn.conv2d` or `mfuse.aclnn.conv2d_with_bias` followed by
    `mfuse.aclnn.batch_norm` (inference, constant tensors) by folding BN into
    convolution weight and bias constants.

    Constraints (aligned with ConvBatchnormFusionPass intent):
    - Conv filter (and bias, when present) must be constants.
    - BatchNorm running_mean/running_var/weight/bias must be constants.
    - Conv output must feed only the BatchNorm op.
    - Dynamic/unsupported constant shapes are not fused.
  }];
  let constructor = "mlir::mfuse::createFuseConvBatchNormPass()";
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
