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

#ifndef MFUSE_DIALECT_MFUSE_ACLNN_OPS_TD
#define MFUSE_DIALECT_MFUSE_ACLNN_OPS_TD

include "mfusion/Dialect/Mfuse/IR/MfuseTypes.td"

//===----------------------------------------------------------------------===//
// MFUSE Aclnn Operations
//
// NOTE: All operations in this file are sorted alphabetically by their class name.
//===----------------------------------------------------------------------===//

def Mfuse_AclnnAddOp : Mfuse_Op<"aclnn.add", [Pure]> {
  let summary = "The aclnnAdd operation";
  let description = [{
    Element-wise addition of two tensors with an optional alpha scaling factor.
    Computes: result = x + y * alpha. Default alpha is 1.0.
  }];

  let arguments = (ins
    AnyRankedTensor:$x,
    AnyRankedTensor:$y,
    AnyScalar:$alpha
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $x `,` $y `,` $alpha attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_AclnnAddRmsNormOp : Mfuse_Op<"aclnn.add_rms_norm", [Pure]> {
  let summary = "The aclnnAddRmsNorm operation";
  let description = [{
    Operation for fused Add and RmsNorm computation.
    Accepts two input tensors for addition, a gamma scaling factor tensor, and epsilon value for numerical stability.
    Produces normalized output tensor, the standard deviation, and the intermediate addition result.
  }];

  let arguments = (ins
    AnyRankedTensor:$x1,
    AnyRankedTensor:$x2,
    AnyRankedTensor:$gamma,
    F64Attr:$epsilon
  );

  let results = (outs
    AnyRankedTensor:$yOut,
    AnyRankedTensor:$rstdOut,
    AnyRankedTensor:$xOut
  );

  let assemblyFormat = [{
    $x1 `,` $x2 `,` $gamma
    attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_AclnnBatchMatmulOp : Mfuse_Op<"aclnn.batch_matmul", [Pure]> {
  let summary = "CANN BatchMatMul operation.";
  let description = [{
    Performs batch matrix multiplication of two tensors (rank >= 3).
    Semantics align with CANN BatchMatMul: for each batch index,
    output[..., :, :] = matrix(self[..., :, :]) * matrix(mat2[..., :, :]).
    Input x shape (*B, N, C) or (*B, C, N) if trans_x1; y shape (*B, C, M) or (*B, M, C) if trans_x2.
    Output shape (*B, N, M). Optional trans_x1/trans_x2 swap the last two dimensions of each
    operand for the matmul without inserting mfuse.permute (DVM-friendly).
  }];

  let arguments = (ins
    AnyRankedTensor:$self,
    AnyRankedTensor:$mat2,
    DefaultValuedAttr<BoolAttr, "false">:$trans_x1,
    DefaultValuedAttr<BoolAttr, "false">:$trans_x2
  );

  let results = (outs AnyRankedTensor:$out);

  let assemblyFormat = [{
    $self `,` $mat2 attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_AclnnBatchNormOp : Mfuse_Op<"aclnn.batch_norm", [Pure]> {
  let summary = "Batch normalization (maps to torch.aten.batch_norm).";
  let description = [{
    Operand order matches `torch.aten.batch_norm`: input, weight (gamma), bias (beta),
    running_mean, running_var, then attributes training, momentum, epsilon, cudnn_enabled.
    For eval, use `training = false`; `fuse-batch-norm` folds the common mfuse
    decomposition into this op. External running_mean/running_var (args or
    constants) require pass option `allow-external-batch-norm-stats=true`; by
    default the pass requires E(x)/Var(x) style proofs from the input tensor.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weight,
    AnyRankedTensor:$bias,
    AnyRankedTensor:$running_mean,
    AnyRankedTensor:$running_var,
    DefaultValuedAttr<BoolAttr, "false">:$training,
    F64Attr:$momentum,
    F64Attr:$epsilon,
    DefaultValuedAttr<BoolAttr, "false">:$cudnn_enable
  );

  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $input `,` $weight `,` $bias `,` $running_mean `,` $running_var
    attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_AclnnConv2DOp : Mfuse_Op<"aclnn.conv2d", [Pure]> {
  let summary = "2D convolution (NCHW) aligned with torch.aten.convolution / CANN.";
  let description = [{
    Performs 2D convolution: input (NCHW) and weight (KCHW) produce output (NKHW').
    Spatial lists are two integers `[H, W]`. Defaults stride=[1,1], padding=[0,0],
    dilation=[1,1], output_padding=[0,0], groups=1, transposed=false.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weight,
    I64ArrayAttr:$stride,
    I64ArrayAttr:$padding,
    I64ArrayAttr:$dilation,
    BoolAttr:$transposed,
    I64ArrayAttr:$output_padding,
    I64Attr:$groups
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $input `,` $weight attr-dict `:` functional-type(operands, results)
  }];

  let builders = [
    OpBuilder<(ins "::mlir::Type":$resultType, "::mlir::Value":$input, "::mlir::Value":$weight), [{
      build($_builder, $_state, resultType, input, weight, $_builder.getI64ArrayAttr({1, 1}),
            $_builder.getI64ArrayAttr({0, 0}), $_builder.getI64ArrayAttr({1, 1}),
            $_builder.getBoolAttr(false), $_builder.getI64ArrayAttr({0, 0}), $_builder.getI64IntegerAttr(1));
    }]>
  ];
}

def Mfuse_AclnnConv2DWithBiasOp : Mfuse_Op<"aclnn.conv2d_with_bias", [Pure]> {
  let summary = "2D convolution with bias; same hyper-parameters as mfuse.aclnn.conv2d.";
  let description = [{
    Performs 2D convolution and adds bias: result = conv2d(input, weight) + broadcast(bias).
    Bias is 1D (output channels) and broadcasts to NCHW output.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weight,
    AnyRankedTensor:$bias,
    I64ArrayAttr:$stride,
    I64ArrayAttr:$padding,
    I64ArrayAttr:$dilation,
    BoolAttr:$transposed,
    I64ArrayAttr:$output_padding,
    I64Attr:$groups
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $input `,` $weight `,` $bias attr-dict `:` functional-type(operands, results)
  }];

  let builders = [
    OpBuilder<(ins "::mlir::Type":$resultType, "::mlir::Value":$input, "::mlir::Value":$weight, "::mlir::Value":$bias), [{
      build($_builder, $_state, resultType, input, weight, bias, $_builder.getI64ArrayAttr({1, 1}),
            $_builder.getI64ArrayAttr({0, 0}), $_builder.getI64ArrayAttr({1, 1}),
            $_builder.getBoolAttr(false), $_builder.getI64ArrayAttr({0, 0}), $_builder.getI64IntegerAttr(1));
    }]>
  ];
}

def Mfuse_AclnnClampOp : Mfuse_Op<"aclnn.clamp", [Pure]> {
  let summary = "Element-wise clamp operation";
  let description = [{ 
    Element-wise clamp operation. Clamps the value of each element in the input tensor to be within the 
    specified min and max range. Computes: result = max(min(input, max_value), min_value)
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$min,
    AnyRankedTensor:$max
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{ 
    $input `,` $min `,` $max attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_AclnnGeluBackwardOp : Mfuse_Op<"aclnn.gelu_backward", [Pure]> {
  let summary = "The aclnnGeluBackwardV2 operation";
  let description = [{
    Computes backward of gelu op element-wise.
  }];

  let arguments = (ins
    AnyRankedTensor:$grad_output,
    AnyRankedTensor:$self,
    StrAttr:$approximate
  );

  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $grad_output `,` $self attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_AclnnGeluOp : Mfuse_Op<"aclnn.gelu", [Pure]> {
  let summary = "The aclnnGelu operation";
  let description = [{
    Computes GELU(x) = x * Φ(x) element-wise, where Φ(x) is the cumulative
    distribution function of the standard normal distribution.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    StrAttr:$approximate
  );

  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_AclnnLayerNormOp : Mfuse_Op<"aclnn.layer_norm", [Pure]> {
  let summary = "The aclnnLayerNorm operation";
  let description = [{
    Operation for LayerNorm computation.
    Computes: y = (x - mean) / sqrt(var + eps) * gamma + beta
    Accepts input tensor x, gamma (weight), beta (bias), and epsilon for numerical stability.
    Produces normalized output tensor.
  }];

  let arguments = (ins
    AnyRankedTensor:$x,
    AnyRankedTensor:$gamma,
    AnyRankedTensor:$beta,
    I64ArrayAttr: $normalized_shape,
    F64Attr:$epsilon
  );

  let results = (outs AnyRankedTensor:$yOut);

  let assemblyFormat = [{
    $x `,` $gamma `,` $beta attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_AclnnMatmulOp : Mfuse_Op<"aclnn.matmul", [Pure]> {
  let summary = "CANN MatMul operation.";
  let description = [{
    Performs matrix multiplication of two tensors (supports 1D to 6D).
    Semantics align with CANN MatMul: out = self * mat2.
    Optional trans_x1/trans_x2 swap the last two dimensions of each operand for the matmul
    without inserting mfuse.permute (DVM-friendly).
  }];

  let arguments = (ins
    AnyRankedTensor:$self,
    AnyRankedTensor:$mat2,
    DefaultValuedAttr<BoolAttr, "false">:$trans_x1,
    DefaultValuedAttr<BoolAttr, "false">:$trans_x2
  );

  let results = (outs AnyRankedTensor:$out);

  let assemblyFormat = [{
    $self `,` $mat2 attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_AclnnMmOp : Mfuse_Op<"aclnn.mm", [Pure]> {
  let summary = "CANN mm operation (2D matrix multiplication).";
  let description = [{
    Performs matrix multiplication of two 2D tensors. Corresponds to the 2D
    case of CANN MatMul. Optional trans_x1/trans_x2 swap the last two dimensions
    of each operand without inserting mfuse.permute (DVM-friendly).
  }];

  let arguments = (ins
    AnyRankedTensor:$self,
    AnyRankedTensor:$mat2,
    DefaultValuedAttr<BoolAttr, "false">:$trans_x1,
    DefaultValuedAttr<BoolAttr, "false">:$trans_x2
  );

  let results = (outs AnyRankedTensor:$out);

  let assemblyFormat = [{
    $self `,` $mat2 attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_AclnnRmsNormOp : Mfuse_Op<"aclnn.rms_norm", [Pure]> {
  let summary = "The aclnnRmsNorm operation";
  let description = [{
    Operation for RmsNorm computation.
    Accepts input tensor, gamma scaling factor, and epsilon value for numerical stability.
    Produces normalized output tensor and reciprocal of standard deviation.
  }];

  let arguments = (ins
    AnyRankedTensor:$x,
    AnyRankedTensor:$gamma,
    F64Attr:$epsilon
  );

  let results = (outs
    AnyRankedTensor:$yOut,
    AnyRankedTensor:$rstdOut
  );

  let assemblyFormat = [{
    $x `,` $gamma
    attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_AclnnSiluOp : Mfuse_Op<"aclnn.silu", [Pure]> {
  let summary = "The aclnnSilu operation";
  let description = [{
    Applies the Sigmoid Linear Unit (Silu) function, element-wise.
    Computes: result = self * sigmoid(self)
  }];

  let arguments = (ins AnyRankedTensor:$self);
  let results = (outs AnyRankedTensor:$out);

  let assemblyFormat = [{
    $self attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_AclnnSplitWithSizeOp : Mfuse_Op<"aclnn.split_with_size", [Pure]> {
  let summary = "The aclnnSplitWithSize operation";
  let description = [{
    Splits a tensor into multiple tensors.
  }];

  let arguments = (ins
    AnyRankedTensor:$self,
    AnyRankedTensor:$splitSize,
    AnyScalar:$dim
  );

  let results = (outs Variadic<AnyRankedTensor>:$out);

  let assemblyFormat = [{
    $self `,` $splitSize `,` $dim attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_AclnnSubOp : Mfuse_Op<"aclnn.sub", [Pure]> {
  let summary = "The aclnnSub operation";
  let description = [{
    Element-wise subtraction of two tensors with an optional alpha scaling factor.
    Computes: result = x - y * alpha
  }];

  let arguments = (ins
    AnyRankedTensor:$x,
    AnyRankedTensor:$y,
    AnyScalar:$alpha
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $x `,` $y `,` $alpha attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_AclnnSigmoidOp : Mfuse_Op<"aclnn.sigmoid", [Pure]> {
  let summary = "The aclnnSigmoid operation";
  let description = [{
    Computes the sigmoid function of each element in the input tensor.
    Computes: result = 1 / (1 + exp(-x))
  }];

  let arguments = (ins
    AnyRankedTensor:$input
  );

  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_AclnnSwiGluOp : Mfuse_Op<"aclnn.swi_glu", [Pure]> {
  let summary = "The aclnnSwiGlu operation";
  let description = [{
    Applies the SwiGlu activation.
    Computes: SwiGlu(x, y) = x * Silu(y) where x and y are split from input.
  }];

  let arguments = (ins
    AnyRankedTensor:$x,
    AnyScalar:$dim
  );

  let results = (outs AnyRankedTensor:$out);

  let assemblyFormat = [{
    $x `,` $dim attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_AclnnTanhOp : Mfuse_Op<"aclnn.tanh", [
    Pure,
    SameOperandsAndResultType,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "The aclnnTanh operation";
  let description = [{
    Compute hyperbolic tangent of input element-wise.
  }];

  let arguments = (ins
    AnyRankedTensor:$input
  );

  let results = (outs
    AnyRankedTensor:$output
  );

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_AclnnVarOp : Mfuse_Op<"aclnn.var", [Pure]> {
  let summary = "The aclnnVar operation";
  let description = [{
    Operation for computing variance of a tensor with Bessel correction.
    Accepts input tensor, dimension list, correction factor, and keepdim flag.
    Returns variance as a single result.
  }];

  let arguments = (ins
    AnyRankedTensor:$self,
    I64ArrayAttr:$dim,
    I64Attr:$correction,
    I1Attr:$keepdim
  );

  let results = (outs
    AnyRankedTensor:$varianceOut
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $self attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_AclnnVarMeanOp : Mfuse_Op<"aclnn.var_mean", [Pure]> {
  let summary = "The aclnnVarMean operation";
  let description = [{
    Operation for computing variance and mean of a tensor.
    Accepts input tensor, dimension list, correction factor, and keepdim flag.
    Returns variance and mean as two results.
  }];

  let arguments = (ins
    AnyRankedTensor:$self,
    I64ArrayAttr:$dim,
    I64Attr:$correction,
    I1Attr:$keepdim
  );

  let results = (outs
    AnyRankedTensor:$varianceOut,
    AnyRankedTensor:$meanOut
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $self attr-dict `:` functional-type(operands, results)
  }];
}

#endif // MFUSE_DIALECT_MFUSE_ACLNN_OPS_TD
