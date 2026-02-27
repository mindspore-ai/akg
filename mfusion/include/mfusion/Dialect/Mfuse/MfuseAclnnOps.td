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

include "mfusion/Dialect/Mfuse/MfuseTypes.td"

//===----------------------------------------------------------------------===//
// MFUSE Aclnn Operations
//
// NOTE: All operations in this file are sorted alphabetically by their class name.
//===----------------------------------------------------------------------===//

def Mfuse_AclnnAddOp : Mfuse_Op<"aclnn.add", [Pure]> {
  let summary = "The aclnnAdd operation";
  let description = [{
    Element-wise addition of two tensors with an optional alpha scaling factor.
    Computes: result = x + y * alpha
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
    Input x shape (*B, N, C) or (*B, C, N) if transpose_a; y shape (*B, C, M) or (*B, M, C) if transpose_b.
    Output shape (*B, N, M). Trans is applied by upper Mfuse ops (Permute) before lowering.
  }];

  let arguments = (ins
    AnyRankedTensor:$self,
    AnyRankedTensor:$mat2
  );

  let results = (outs AnyRankedTensor:$out);

  let assemblyFormat = [{
    $self `,` $mat2 attr-dict `:` functional-type(operands, results)
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

def Mfuse_AclnnMatmulOp : Mfuse_Op<"aclnn.matmul", [Pure]> {
  let summary = "CANN MatMul operation.";
  let description = [{
    Performs matrix multiplication of two tensors (supports 1D to 6D).
    Semantics align with CANN MatMul: out = self * mat2; trans and bias
    are applied by upper Mfuse ops before lowering to this aclnn op.
  }];

  let arguments = (ins
    AnyRankedTensor:$self,
    AnyRankedTensor:$mat2
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
    case of CANN MatMul; trans is applied by upper Mfuse ops before lowering.
  }];

  let arguments = (ins
    AnyRankedTensor:$self,
    AnyRankedTensor:$mat2
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

def Mfuse_AclnnTanhOp : Mfuse_Op<"aclnn.tanh", [Pure]> {
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

#endif // MFUSE_DIALECT_MFUSE_ACLNN_OPS_TD
