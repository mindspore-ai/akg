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

#ifndef MUSE_DIALECT_MUSE_META_OPS_TD
#define MUSE_DIALECT_MUSE_META_OPS_TD

include "mfusion/Dialect/Muse/MuseTypes.td"

//===----------------------------------------------------------------------===//
// MUSE Meta Operations
//
// This file defines only meta operators supported by auto fusion.
//
// NOTE: All operations in this file are sorted alphabetically by their class name.
//===----------------------------------------------------------------------===//

def Muse_AddOp : Muse_Op<"add", [Pure, Commutative]> {
  let summary = "Element-wise addition of two tensors.";
  let description = [{
    Computes element-wise sum of two tensors: result = x + y.
    Both input tensors must have compatible shapes for broadcasting.
  }];

  let arguments = (ins
    AnyRankedTensor:$x,
    AnyRankedTensor:$y
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $x `,` $y attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_CastOp : Muse_Op<"cast", [Pure]> {
  let summary = "Cast tensor to a specified data type.";
  let description = [{
    Casts the input tensor to the specified data type.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    TypeAttr:$dtype
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_DivOp : Muse_Op<"div", [Pure]> {
  let summary = "Element-wise division of two tensors.";
  let description = [{
    Computes element-wise division of two tensors: result = self / other.
    Both input tensors must have compatible shapes for broadcasting.
  }];

  let arguments = (ins
    AnyRankedTensor:$self,
    AnyRankedTensor:$other
  );

  let results = (outs
    AnyRankedTensor:$out
  );

  let assemblyFormat = [{
    $self `,` $other
    attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_EqOp : Muse_Op<"eq", [Pure]> {
  let summary = "Element-wise equality comparison of two tensors.";
  let description = [{
    Returns a boolean tensor where each element is true if the corresponding element in self is equal to the element in other.
    Both input tensors must have compatible shapes for broadcasting.
  }];

  let arguments = (ins
    AnyRankedTensor:$self,
    AnyRankedTensor:$other
  );

  let results = (outs AnyRankedTensor:$out);

  let assemblyFormat = [{
    $self `,` $other attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_ExpOp : Muse_Op<"exp", [Pure]> {
  let summary = "The exp operation";
  let description = [{
    Operation for exponential computation.
    Computes the exponential value (e^x) for each element in the input tensor.
  }];

  let arguments = (ins
    AnyRankedTensor:$input
  );

  let results = (outs AnyRankedTensor:$out);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_GeOp : Muse_Op<"ge", [Pure]> {
  let summary = "Element-wise greater-than-or-equal comparison of two tensors.";
  let description = [{
    Returns a boolean tensor where each element is true if the corresponding element in self is greater than or equal to the element in other.
    Both input tensors must have compatible shapes for broadcasting.
  }];

  let arguments = (ins
    AnyRankedTensor:$self,
    AnyRankedTensor:$other
  );

  let results = (outs AnyRankedTensor:$out);

  let assemblyFormat = [{
    $self `,` $other attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_GtOp : Muse_Op<"gt", [Pure]> {
  let summary = "Element-wise greater-than comparison of two tensors.";
  let description = [{
    Returns a boolean tensor where each element is true if the corresponding element in self is greater than the element in other.
    Both input tensors must have compatible shapes for broadcasting.
  }];

  let arguments = (ins
    AnyRankedTensor:$self,
    AnyRankedTensor:$other
  );

  let results = (outs AnyRankedTensor:$out);

  let assemblyFormat = [{
    $self `,` $other attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_LeOp : Muse_Op<"le", [Pure]> {
  let summary = "Element-wise less-than-or-equal comparison of two tensors.";
  let description = [{
    Returns a boolean tensor where each element is true if the corresponding element in self is less than or equal to the element in other.
    Both input tensors must have compatible shapes for broadcasting.
  }];

  let arguments = (ins
    AnyRankedTensor:$self,
    AnyRankedTensor:$other
  );

  let results = (outs AnyRankedTensor:$out);

  let assemblyFormat = [{
    $self `,` $other attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_LtOp : Muse_Op<"lt", [Pure]> {
  let summary = "Element-wise less-than comparison of two tensors.";
  let description = [{
    Returns a boolean tensor where each element is true if the corresponding element in self is less than the element in other.
    Both input tensors must have compatible shapes for broadcasting.
  }];

  let arguments = (ins
    AnyRankedTensor:$self,
    AnyRankedTensor:$other
  );

  let results = (outs AnyRankedTensor:$out);

  let assemblyFormat = [{
    $self `,` $other attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_Matmul : Muse_Op<"matmul", [Pure]> {
  let summary = "Matrix multiplication of two tensors.";
  let description = [{
    Performs matrix multiplication of two tensors (supports 1D to 6D tensors).
  }];

  let arguments = (ins
    AnyRankedTensor:$self,
    AnyRankedTensor:$other
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $self `,` $other attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_Mm : Muse_Op<"mm", [Pure]> {
  let summary = "Matrix multiplication of two 2D tensors.";
  let description = [{
    Performs matrix multiplication of two 2D tensors.
  }];

  let arguments = (ins
    AnyRankedTensor:$self,
    AnyRankedTensor:$other
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $self `,` $other attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_MulOp : Muse_Op<"mul", [Pure, Commutative]> {
  let summary = "Element-wise multiplication of two tensors.";
  let description = [{
    Computes element-wise product of two tensors: result = lhs * rhs.
    Both input tensors must have compatible shapes for broadcasting.
  }];

  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_NegOp : Muse_Op<"neg", [Pure]> {
  let summary = "The neg operation";
  let description = [{
    Get the element-wise negation of the input tensor.
  }];

  let arguments = (ins
    AnyRankedTensor:$input
  );

  let results = (outs AnyRankedTensor:$out);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_NeOp : Muse_Op<"ne", [Pure]> {
  let summary = "Element-wise not-equal comparison of two tensors.";
  let description = [{
    Returns a boolean tensor where each element is true if the corresponding element in self is not equal to the element in other.
    Both input tensors must have compatible shapes for broadcasting.
  }];

  let arguments = (ins
    AnyRankedTensor:$self,
    AnyRankedTensor:$other
  );

  let results = (outs AnyRankedTensor:$out);

  let assemblyFormat = [{
    $self `,` $other attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_PermuteOp : Muse_Op<"permute", [Pure]> {
  let summary = "Tensor dimension permutation operation.";
  let description = [{
    Permutes the dimensions of a tensor according to the specified order.
    Reorders the axes of the input tensor based on the permutation array.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    I64ArrayAttr:$perm
  );

  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $input `,` $perm attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_PowOp : Muse_Op<"pow", [Pure]> {
  let summary = "Element-wise power operation.";
  let description = [{
    Raises each element of the base tensor to the power of the exponent.
    Exponent is represented as tensor.
  }];

  let arguments = (ins
    AnyRankedTensor:$base,
    AnyRankedTensor:$exponent
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $base `,` $exponent attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_ReciprocalOp : Muse_Op<"reciprocal", [Pure]> {
  let summary = "The reciprocal operation";
  let description = [{
    Get the the reciprocal of input tensor.
  }];

  let arguments = (ins
    AnyRankedTensor:$input
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_ReduceSumOp : Muse_Op<"reduce_sum", [Pure]> {
  let summary = "Sum operation";
  let description = [{
    Computes the sum of tensor elements along specified dimensions.
    The boolean parameter controls whether to keep dimensions,
    and the optional dtype parameter specifies the output data type.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    I64ArrayAttr:$dimensions,
    BoolAttr:$keepdim,
    TypeAttr:$dtype
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_ReshapeOp : Muse_Op<"reshape", [Pure]> {
  let summary = "Reshape operation";
  let description = [{
    Reshapes input tensor to the specified shape.
    Takes input tensor and shape attribute.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    I64ArrayAttr:$shape
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_SubOp : Muse_Op<"sub", [Pure]> {
  let summary = "Element-wise subtraction of two tensors.";
  let description = [{
    Computes element-wise difference of two tensors: result = x - y.
    Both input tensors must have compatible shapes for broadcasting.
  }];

  let arguments = (ins
    AnyRankedTensor:$x,
    AnyRankedTensor:$y
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $x `,` $y attr-dict `:` functional-type(operands, results)
  }];
}

#endif // MUSE_DIALECT_MUSE_META_OPS_TD
