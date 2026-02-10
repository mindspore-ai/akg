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

def FuseMatMulCast : Pass<"fuse-matmul-cast", "::mlir::ModuleOp"> {
  let summary = "Fuse MatMul/Mm (f16) and Cast (f32) into MatMul with f32 result";
  let description = [{
    Fuses MatMul or Mm with float16 inputs followed by Cast to float32
    into a single MatMul/Mm with float32 result type.
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
