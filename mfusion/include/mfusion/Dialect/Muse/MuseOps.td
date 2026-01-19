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

#ifndef MUSE_DIALECT_MUSE_OPS_TD
#define MUSE_DIALECT_MUSE_OPS_TD

include "mfusion/Dialect/Muse/MuseTypes.td"

//===----------------------------------------------------------------------===//
// MUSE Operations
//
// NOTE: All operations in this file are sorted alphabetically by their class name
// (the identifier after "def", e.g., Muse_AddRmsNormOp, Muse_BatchNormOp).
// When adding new operations, please maintain this alphabetical order to keep
// the file organized and easy to navigate.
//===----------------------------------------------------------------------===//

def Muse_AddOp : Muse_Op<"add", [Pure]> {
  let summary = "aclnnAdd";
  let description = [{
    Element-wise addition of two tensors with an optional alpha scaling factor.
    Computes: result = x + y * alpha
  }];

  let arguments = (ins
    MuseAnyTensor:$x,
    MuseAnyTensor:$y,
    Muse_ScalarType:$alpha
  );

  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $x `,` $y `,` $alpha attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnAdd";
}

def Muse_AddRmsNormOp : Muse_Op<"add_rms_norm", [Pure]> {
  let summary = "aclnnAddRmsNorm";
  let description = [{
    Operation for fused Add and RmsNorm computation.
    Accepts two input tensors for addition, a gamma scaling factor tensor, and epsilon value for numerical stability.
    Produces normalized output tensor, the standard deviation, and the intermediate addition result.
  }];

  let arguments = (ins
    MuseAnyTensor:$x1,
    MuseAnyTensor:$x2,
    MuseAnyTensor:$gamma,
    Muse_F64Type:$epsilon
  );

  let results = (outs
    MuseAnyTensor:$yOut,
    MuseAnyTensor:$rstdOut,
    MuseAnyTensor:$xOut
  );

  let assemblyFormat = [{
    $x1 `,` $x2 `,` $gamma `,` $epsilon
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnAddRmsNorm";
}

def Muse_ApplyRotaryPosEmbOp : Muse_Op<"apply_rotary_pos_emb"> {
  let summary = "aclnnApplyRotaryPosEmb";
  let description = [{
    Operation for rope computation.
    Accepts query, key, cos, sin  tensor, and layout string for rope calculation.
    Produces the result after rope.
  }];

  let arguments = (ins
    MuseAnyTensor:$query,
    MuseAnyTensor:$key,
    MuseAnyTensor:$cos,
    MuseAnyTensor:$sin,
    Muse_I64Type:$layout
  );

  let results = (outs
    MuseAnyTensor:$fake_query_out,
    MuseAnyTensor:$fake_key_out
  );

  let assemblyFormat = [{
    $query `,` $key `,` $cos `,` $sin `,` $layout
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnApplyRotaryPosEmb";
  let genAclnnCfg = (AclnnConfig (IORef 0, 0), (IORef 1, 1));
}

def Muse_BatchNormOp : Muse_Op<"batch_norm", [Pure]> {
  let summary = "Batch normalization operation";
  let description = [{
    Performs batch normalization on input tensors.
  }];

  let arguments = (ins
    MuseAnyTensor:$input,
    MuseAnyTensor:$scale,
    MuseAnyTensor:$offset,
    MuseAnyTensor:$mean,
    MuseAnyTensor:$variance,
    Muse_F64Type:$epsilon,
    Muse_BooleanType:$is_training
  );

  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $scale `,` $offset `,` $mean `,` $variance `,` $epsilon `,` $is_training
    attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_BitwiseAndTensorOp : Muse_Op<"bitwise_and_tensor", [Pure]> {
  let summary = "aclnnBitwiseAndTensor";
  let description = [{
    Operation for element-wise bitwise AND computation between two tensors.
    Performs bitwise AND operation on corresponding elements of input tensors.
    Both input tensors must have the same data type and compatible shapes for broadcasting.
  }];

  let arguments = (ins
    MuseAnyTensor:$self,
    MuseAnyTensor:$other
  );

  let results = (outs
    MuseAnyTensor:$out
  );

  let assemblyFormat = [{
    $self `,` $other
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnBitwiseAndTensor";
}

def Muse_BitwiseNotOp : Muse_Op<"bitwise_not", [Pure]> {
  let summary = "aclnnBitwiseNot";
  let description = [{
    Operation for element-wise bitwise NOT computation on a tensor.
    Performs bitwise NOT operation on each element of the input tensor.
    The input tensor must have integer data type.
  }];

  let arguments = (ins
    MuseAnyTensor:$self
  );

  let results = (outs
    MuseAnyTensor:$out
  );

  let assemblyFormat = [{
    $self
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnBitwiseNot";
}

def Muse_CastOp : Muse_Op<"cast", [Pure]> {
  let summary = "aclnnCast";
  let description = [{
    Casts a tensor to the specified data type.
  }];

  let arguments = (ins
    MuseAnyTensor:$input,
    Muse_DtypeType:$dtype
  );

  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $dtype attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnCast";
}

def Muse_CatOp : Muse_Op<"cat", [Pure]> {
  let summary = "Concatenation operation";
  let description = [{
    Concatenates tensors along the specified axis.
  }];

  let arguments = (ins
    MuseTensorList:$inputs,
    Muse_I64Type:$axis
  );

  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $inputs `,` $axis attr-dict `:` functional-type(operands, results)
  }];
  let genAclnnOp = "aclnnCat";
}

def Muse_CloneOp : Muse_Op<"clone", [Pure]> {
  let summary = "Clone operation";
  let description = [{
    Returns a clone of the input tensor.
  }];

  let arguments = (ins
    MuseAnyTensor:$self
  );

  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $self attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_ConvOp : Muse_Op<"conv", [Pure]> {
  let summary = "2D convolution operation";
  let description = [{
    Performs a 2D convolution operation on input tensors.
  }];

  let arguments = (ins
    MuseAnyTensor:$input,
    MuseAnyTensor:$kernel,
    MuseOptional<MuseAnyTensor>:$bias,
    Muse_I64ArrayType:$strides,
    Muse_I64ArrayType:$padding,
    Muse_I64ArrayType:$dilation,
    Muse_BooleanType:$has_bias
  );

  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $kernel `,` $bias `,` $strides `,` $padding `,` $dilation `,` $has_bias
    attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_CustomCallOp : Muse_Op<"custom_call"> {
  let summary = "Invoke a custom operator";
  let description = [{
    Invokes a custom operator by name.
    The op_name operand contains the name of the custom operator to invoke.
  }];

  let arguments = (ins
    Muse_StringType:$op_name,
    Variadic<AnyType>:$inputs
  );

  let results = (outs
    Variadic<AnyType>:$outputs
  );

  let assemblyFormat = [{
    $op_name `,` $inputs attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_DivOp : Muse_Op<"div", [Pure]> {
  let summary = "aclnnDiv";
  let description = [{
    Operation for element-wise division of two tensors.
    Performs division of corresponding elements of input tensors.
    Both input tensors must have compatible shapes for broadcasting.
  }];

  let arguments = (ins
    MuseAnyTensor:$self,
    MuseAnyTensor:$other
  );

  let results = (outs
    MuseAnyTensor:$out
  );

  let assemblyFormat = [{
    $self `,` $other
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnDiv";
}

def Muse_DivModOp : Muse_Op<"div_mod", [Pure]> {
  let summary = "aclnnDivMod";
  let description = [{
    Operation for element-wise division and modulus computation between two tensors.
    Performs division and modulus operations on corresponding elements of input tensors.
    The mode parameter controls the rounding behavior for integer division.
    Both input tensors must have compatible shapes for broadcasting.
  }];

  let arguments = (ins
    MuseAnyTensor:$self,
    MuseAnyTensor:$other,
    Muse_I64Type:$mode
  );

  let results = (outs
    MuseAnyTensor:$out
  );

  let assemblyFormat = [{
    $self `,` $other `,` $mode
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnDivMod";
}

def Muse_DvmCallOp : Muse_Op<"dvm_call"> {
  let summary = "Invoke a DVM kernel described by a JSON payload";
  let description = [{
    Invokes a DVM kernel described by a JSON payload.
    The payload operand contains a serialized instruction stream compatible with
    the runtime OpDvmCall JSON schema (see tests/st/inferrt/ops/test_op_dvm_call_e2e.py).

    Operand order:
    - payload_json: muse.string JSON payload
    - inputs: variadic tensor inputs bound to Load instructions in-order
  }];

  let arguments = (ins
    Muse_StringType:$payload_json,
    Variadic<MuseAnyTensor>:$inputs
  );

  let results = (outs
    Variadic<MuseAnyTensor>:$outputs
  );

  let assemblyFormat = [{
    $payload_json `,` $inputs attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_EmbeddingOp : Muse_Op<"embedding", [Pure]> {
  let summary = "aclnnEmbedding";
  let description = [{
    Operation for embedding computation.
    Accepts weight, indices tensor.
    Produces embedding output.
  }];

  let arguments = (ins
    MuseAnyTensor:$weight,
    MuseAnyTensor:$indices
  );

  let results = (outs
    MuseAnyTensor:$out
  );

  let genAclnnOp = "aclnnEmbedding";
}

def Muse_EmptyOp : Muse_Op<"empty", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "";
  let description = [{
    Operation for empty ops.
    Accept shape size.
    Produces the device the same shape tensor.
  }];

  let arguments = (ins
    Muse_I64ArrayType:$size,
    Muse_DtypeType:$dtype,
    Muse_StringType:$device
  );

  let results = (outs
    MuseAnyTensor:$yOut
  );

  let assemblyFormat = [{
    $size `,` $dtype `,` $device
    attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_EqOp : Muse_Op<"eq", [Pure]> {
  let summary = "Element-wise equality comparison";
  let description = [{
    Element-wise equality comparison of two tensors.
    Returns a boolean tensor where each element is true if the corresponding elements are equal.
  }];

  let arguments = (ins
    MuseAnyTensor:$self,
    MuseAnyTensor:$other
  );

  let results = (outs MuseAnyTensor:$out);

  let assemblyFormat = [{
    $self `,` $other attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnEqTensor";
}

def Muse_ExpandOp : Muse_Op<"expand", [Pure]> {
  let summary = "expand operation";
  let description = [{
    Expands the input tensor 'self' to the given 'size' by broadcasting.
    The output tensor will have the specified shape.
  }];

  let arguments = (ins
    MuseAnyTensor:$self,
    Muse_I64ArrayType:$size
  );

  let results = (outs MuseAnyTensor:$out);

  let assemblyFormat = [{
    $self `,` $size attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnExpand";
}

def Muse_FlattenOp : Muse_Op<"flatten", [Pure]> {
  let summary = "flatten operation";
  let description = [{
    Flatten input tensor to the specified shape.
    Takes two operands: input tensor and shape.
  }];

  let arguments = (ins
    MuseAnyTensor:$input,
    Muse_I64Type:$start_idx,
    Muse_I64Type:$end_idx
  );

  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $start_idx `,` $end_idx attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_GatherV2Op : Muse_Op<"gather_v2", [Pure]> {
  let summary = "gather_v2 operation";
  let description = [{
    Gathers elements from a tensor along a specified dimension using indices from another tensor.
    Takes three operands: input tensor, dimension index, and indices tensor.
    Produces a tensor containing elements gathered from the input tensor at the specified indices.
  }];

  let arguments = (ins
    MuseAnyTensor:$x,
    Muse_I64Type:$dim,
    MuseAnyTensor:$index
  );

  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $x `,` $dim `,` $index attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnGatherV2";
}

def Muse_GeOp : Muse_Op<"ge", [Pure]> {
  let summary = "Element-wise greater-than-or-equal comparison";
  let description = [{
    Element-wise greater-than-or-equal comparison of two tensors.
    Returns a boolean tensor where each element is true if the corresponding element in self is greater than or equal to the element in other.
  }];

  let arguments = (ins
    MuseAnyTensor:$self,
    MuseAnyTensor:$other
  );

  let results = (outs MuseAnyTensor:$out);

  let assemblyFormat = [{
    $self `,` $other attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnGeTensor";
}

def Muse_GetItemSliceOp : Muse_Op<"getitem_slice", [Pure]> {
  let summary = "getitem use aclnnSliceV2";
  let description = [{
    Slice operation with support for multiple axes, starts, ends, and steps.
    Extracts a slice from the input tensor along the specified axes with the given starts, ends, and steps.
  }];

  let arguments = (ins
    MuseAnyTensor:$x,
    Muse_I64ArrayType:$starts,
    Muse_I64ArrayType:$ends,
    Muse_I64ArrayType:$axes,
    Muse_I64ArrayType:$steps
  );

  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $x `,` $starts `,` $ends `,` $axes `,` $steps attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_GroupedMatmulOp : Muse_Op<"grouped_matmul", [Pure]> {
  let summary = "NPU grouped matmul operation";
  let description = [{
    Performs grouped matrix multiplication on NPU with optional bias, scale, offset, and activation.
  }];

  let arguments = (ins
    MuseTensorList:$x,
    MuseTensorList:$weight,
    MuseOptional<MuseTensorList>:$bias,
    MuseOptional<MuseTensorList>:$scale,
    MuseOptional<MuseTensorList>:$offset,
    MuseOptional<MuseTensorList>:$antiquant_scale,
    MuseOptional<MuseTensorList>:$antiquant_offset,
    MuseOptional<MuseTensorList>:$per_token_scale,
    MuseOptional<MuseAnyTensor>:$group_list,
    MuseOptional<MuseTensorList>:$activation_input,
    MuseOptional<MuseTensorList>:$activation_quant_scale,
    MuseOptional<MuseTensorList>:$activation_quant_offset,
    Muse_I64Type:$split_item,
    Muse_I64Type:$group_type,
    Muse_I64Type:$group_list_type,
    Muse_I64Type:$act_type,
    MuseOptional<Muse_I64ArrayType>:$tuning_config,
    Muse_DtypeType:$output_dtype
  );

  let results = (outs
    MuseTensorList:$output
  );

  let assemblyFormat = [{
    $x `,` $weight `,` $bias `,` $scale `,` $offset `,` $antiquant_scale `,` $antiquant_offset `,` $per_token_scale 
    `,` $group_list `,` $activation_input `,` $activation_quant_scale `,` $activation_quant_offset
    `,` $split_item `,` $group_type `,` $group_list_type `,` $act_type `,` $tuning_config `,` $output_dtype
    attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_GtOp : Muse_Op<"gt", [Pure]> {
  let summary = "Element-wise greater-than comparison";
  let description = [{
    Element-wise greater-than comparison of two tensors.
    Returns a boolean tensor where each element is true if the corresponding element in self is greater than the element in other.
  }];

  let arguments = (ins
    MuseAnyTensor:$self,
    MuseAnyTensor:$other
  );

  let results = (outs MuseAnyTensor:$out);

  let assemblyFormat = [{
    $self `,` $other attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnGtTensor";
}

def Muse_IndexTensorOp : Muse_Op<"index_tensor", [Pure]> {
  let summary = "Indexes a tensor using a list of index tensors.";
  let description = [{
    Operation for indexing a tensor using a list of index tensors.
    Performs advanced indexing where each tensor in the indices list
    corresponds to indexing along that dimension.
    This is the implementation of `aten.index.Tensor_hacked_twin`.
  }];

  let arguments = (ins
    MuseAnyTensor:$self,
    MuseTensorList:$indices
  );

  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $self `,` $indices attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnIndex";
}

def Muse_InplaceCopyOp : Muse_Op<"inplace_copy"> {
  let summary = "Copy operation";
  let description = [{
    Copies data from source tensor to destination tensor. The optional boolean parameter controls whether the copy is in-place.
  }];

  let arguments = (ins
    MuseAnyTensor:$dst,
    MuseAnyTensor:$src,
    Muse_BooleanType:$non_blocking
  );

  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $dst `,` $src `,` $non_blocking
    attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_InplaceFillScalarOp : Muse_Op<"inplace_fill_scalar"> {
  let summary = "Fill tensor with scalar value";
  let description = [{
    Fills the destination tensor with the specified scalar value.
    Computes: result = fill(x, value)
  }];

  let arguments = (ins
    MuseAnyTensor:$x,
    Muse_ScalarType:$value
  );

  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $x `,` $value attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnInplaceFillScalar";
  let genAclnnCfg = (AclnnConfig (IORef 0, 0));
}

def Muse_InplaceFillTensorOp : Muse_Op<"inplace_fill_tensor"> {
  let summary = "Fill tensor with tensor value";
  let description = [{
    Fills the destination tensor with the specified tensor value.
    Computes: result = fill(x, value)
  }];

  let arguments = (ins
    MuseAnyTensor:$x,
    MuseAnyTensor:$value
  );

  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $x `,` $value attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnInplaceFillTensor";
  let genAclnnCfg = (AclnnConfig (IORef 0, 0));
}

def Muse_InplaceMaskedFillScalarOp : Muse_Op<"inplace_masked_fill_scalar"> {
  let summary = "Masked fill operation";
  let description = [{
    Replaces elements in the input tensor with the fill value where the mask is True.
  }];

  let arguments = (ins
    MuseAnyTensor:$input,
    MuseAnyTensor:$mask,
    Muse_ScalarType:$value
  );

  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $mask `,` $value attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnInplaceMaskedFillScalar";
  let genAclnnCfg = (AclnnConfig (IORef 0, 0));
}

def Muse_InplaceMaskedFillTensorOp : Muse_Op<"inplace_masked_fill_tensor"> {
  let summary = "Masked fill operation";
  let description = [{
    Replaces elements in the input tensor with the fill value where the mask is True.
  }];

  let arguments = (ins
    MuseAnyTensor:$input,
    MuseAnyTensor:$mask,
    MuseAnyTensor:$value
  );

  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $mask `,` $value attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnInplaceMaskedFillTensor";
  let genAclnnCfg = (AclnnConfig (IORef 0, 0));
}

def Muse_LeOp : Muse_Op<"le", [Pure]> {
  let summary = "Element-wise less-than-or-equal comparison";
  let description = [{
    Element-wise less-than-or-equal comparison of two tensors.
    Returns a boolean tensor where each element is true if the corresponding element in self is less than or equal to the element in other.
  }];

  let arguments = (ins
    MuseAnyTensor:$self,
    MuseAnyTensor:$other
  );

  let results = (outs MuseAnyTensor:$out);

  let assemblyFormat = [{
    $self `,` $other attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnLeTensor";
}

def Muse_LinalgCallOp : Muse_Op<"linalg_call"> {
  let summary = "Invoke a Linalg MLIR kernel";
  let description = [{
    Invokes a Linalg MLIR kernel compiled through BiSheng compiler.
    The mlir_text operand contains a serialized Linalg MLIR module with a
    function marked with hacc.entry attribute. 

    Internal workflow:
    1. Init: Extract MLIR text, call LoweredOpHelper::CreateFromMlirText()
    2. InferShape/CalcWorkspace/Launch: Delegate to internal AutoLoweredOp
  }];

  let arguments = (ins
    Muse_StringType:$mlir_text,
    Variadic<MuseAnyTensor>:$inputs
  );

  let results = (outs
    Variadic<MuseAnyTensor>:$outputs
  );

  let assemblyFormat = [{
    $mlir_text `,` $inputs attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_LinearOp : Muse_Op<"linear", [Pure]> {
  let summary = "dense";
  let description = [{
    Operation for linear computation.
    Accepts weight, indices tensor.
    Produces embedding output.
  }];

  let arguments = (ins
    MuseAnyTensor:$x,
    MuseAnyTensor:$weight,
    Optional<MuseAnyTensor>:$bias
  );

  let results = (outs
    MuseAnyTensor:$out
  );

  let assemblyFormat = [{
    $x `,` $weight (`,` $bias^)?
    attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_LtOp : Muse_Op<"lt", [Pure]> {
  let summary = "Element-wise less-than comparison";
  let description = [{
    Element-wise less-than comparison of two tensors.
    Returns a boolean tensor where each element is true if the corresponding element in self is less than the element in other.
  }];

  let arguments = (ins
    MuseAnyTensor:$self,
    MuseAnyTensor:$other
  );

  let results = (outs MuseAnyTensor:$out);

  let assemblyFormat = [{
    $self `,` $other attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnLtTensor";
}

def Muse_MakeListOp : Muse_Op<"make_list", [Pure]> {
  let summary = "Construct a list from elements";
  let description = [{
    Creates a list from a list of elements, similar to torch.prim.ListConstruct.
    This operation takes variadic elements and constructs a list type containing them.
  }];

  let arguments = (ins
    Variadic<AnyType>:$elements
  );

  let results = (outs MuseAnyList:$result);

  let assemblyFormat = [{
    $elements attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_MaskedFillScalarOp : Muse_Op<"masked_fill_scalar", [Pure]> {
  let summary = "Masked fill operation";
  let description = [{
    Replaces elements in the input tensor with the fill value where the mask is True.
  }];

  let arguments = (ins
    MuseAnyTensor:$input,
    MuseAnyTensor:$mask,
    Muse_ScalarType:$value
  );

  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $mask `,` $value attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_MaskedFillTensorOp : Muse_Op<"masked_fill_tensor", [Pure]> {
  let summary = "Masked fill operation";
  let description = [{
    Replaces elements in the input tensor with the fill value where the mask is True.
  }];

  let arguments = (ins
    MuseAnyTensor:$input,
    MuseAnyTensor:$mask,
    MuseAnyTensor:$value
  );

  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $mask `,` $value attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_Matmul : Muse_Op<"matmul", [Pure]> {
  let summary = "aclnnMatmul";
  let description = [{
    Performs matrix multiplication of two tensors (supports 1D to 6D tensor).
  }];

  let arguments = (ins
    MuseAnyTensor:$self,
    MuseAnyTensor:$other
  );

  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $self `,` $other attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_Mm : Muse_Op<"mm", [Pure]> {
  let summary = "aclnnMm";
  let description = [{
    Performs matrix multiplication of two 2D tensors.
  }];

  let arguments = (ins
    MuseAnyTensor:$self,
    MuseAnyTensor:$other
  );

  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $self `,` $other attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_MoeGatingTopKSoftmaxOp : Muse_Op<"moe_gating_top_k_softmax", [Pure]> {
  let summary = "moe calc for input do Softmax, and do topK selection";
  let description = [{
    Performs a Softmax and TopK on input tensors.
  }];
  
  let arguments = (ins
    MuseAnyTensor:$x,
    MuseOptional<MuseAnyTensor>:$finished,
    Muse_I64Type:$k
  );
  
  let results = (outs 
    MuseAnyTensor:$y,
    MuseAnyTensor:$expert_idx,
    MuseAnyTensor:$row_idx
  );
  
  let assemblyFormat = [{
    $x `,` $finished `,` $k
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnMoeGatingTopKSoftmax";
}

def Muse_FusedInferAttentionScoreOp : Muse_Op<"fused_infer_attention_score", [Pure]> {
  let summary = "aclnnFusedInferAttentionScoreV3";
  let description = [{
    Support FlashAttention and PagedAttenton in prefill and decoder.
  }];

  let arguments = (ins
    MuseAnyTensor:$query,
    MuseTensorList:$key,
    MuseTensorList:$value,
    MuseOptional<MuseAnyTensor>:$pseShiftOptional,
    MuseOptional<MuseAnyTensor>:$attenMaskOptional,
    MuseOptional<Muse_I64ArrayType>:$actualSeqLengthsOptional,
    MuseOptional<Muse_I64ArrayType>:$actualSeqLengthsKvOptional,
    MuseOptional<MuseAnyTensor>:$deqScale1Optional,
    MuseOptional<MuseAnyTensor>:$quantScale1Optional,
    MuseOptional<MuseAnyTensor>:$deqScale2Optional,
    MuseOptional<MuseAnyTensor>:$quantScale2Optional,
    MuseOptional<MuseAnyTensor>:$quantOffset2Optional,
    MuseOptional<MuseAnyTensor>:$antiquantScaleOptional,
    MuseOptional<MuseAnyTensor>:$antiquantOffsetOptional,
    MuseOptional<MuseAnyTensor>:$blockTableOptional,
    MuseOptional<MuseAnyTensor>:$queryPaddingSizeOptional,
    MuseOptional<MuseAnyTensor>:$kvPaddingSizeOptional,
    MuseOptional<MuseAnyTensor>:$keyAntiquantScaleOptional,
    MuseOptional<MuseAnyTensor>:$keyAntiquantOffsetOptional,
    MuseOptional<MuseAnyTensor>:$valueAntiquantScaleOptional,
    MuseOptional<MuseAnyTensor>:$valueAntiquantOffsetOptional,
    MuseOptional<MuseAnyTensor>:$keySharedPrefixOptional,
    MuseOptional<MuseAnyTensor>:$valueSharedPrefixOptional,
    MuseOptional<MuseAnyTensor>:$actualSharedPrefixLenOptional,
    MuseOptional<MuseAnyTensor>:$queryRopeOptional,
    MuseOptional<MuseAnyTensor>:$keyRopeOptional,
    MuseOptional<MuseAnyTensor>:$keyRopeAntiquantScaleOptional,
    Muse_I64Type:$numHeads,
    Muse_F64Type:$scaleValue,
    Muse_I64Type:$preTokens,
    Muse_I64Type:$nextTokens,
    Muse_StringType:$inputLayout,
    Muse_I64Type:$numKeyValueHeads,
    Muse_I64Type:$sparseMode,
    Muse_I64Type:$innerPrecise,
    Muse_I64Type:$blockSize,
    Muse_I64Type:$antiquantMode,
    Muse_BooleanType:$softmaxLseFlag,
    Muse_I64Type:$keyAntiquantMode,
    Muse_I64Type:$valueAntiquantMode
  );

  let results = (outs
    MuseAnyTensor:$attentionOut,
    MuseAnyTensor:$softmaxLse
  );
}

def Muse_MoeInitRoutingV3Op : Muse_Op<"moe_init_routing_v3", [Pure]> {
  let summary = "aclnnMoeInitRoutingV3";
  let description = [{
    Operation for initializing Moe (Mixture of Experts) routing V3 using aclnn backend.
    Accepts input tensor, expert indices, scale/offset (optional), and configuration arguments.
    Produces expanded tensors for X, row indices, the expert token count or cumsum, and scale.
  }];

  let arguments = (ins
    MuseAnyTensor:$x,
    MuseAnyTensor:$expertIdx,
    MuseOptional<MuseAnyTensor>:$scaleOptional,
    MuseOptional<MuseAnyTensor>:$offsetOptional,
    Muse_I64Type:$activeNum,
    Muse_I64Type:$expertCapacity,
    Muse_I64Type:$expertNum,
    Muse_I64Type:$dropPadMode,
    Muse_I64Type:$expertTokensNumType,
    Muse_BooleanType:$expertTokensNumFlag,
    Muse_I64Type:$quantMode,
    Muse_I64ArrayType:$activeExpertRangeOptional,
    Muse_I64Type:$rowIdxType
  );

  let results = (outs
    MuseAnyTensor:$expandedXOut,
    MuseAnyTensor:$expandedRowIdxOut,
    MuseAnyTensor:$expertTokensCountOrCumsumOut,
    MuseAnyTensor:$expandedScaleOut
  );

  let assemblyFormat = [{
    $x `,` $expertIdx `,` $scaleOptional `,` $offsetOptional `,`
    $activeNum `,` $expertCapacity `,` $expertNum `,` $dropPadMode `,`
    $expertTokensNumType `,` $expertTokensNumFlag `,` $quantMode `,`
    $activeExpertRangeOptional `,` $rowIdxType
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnMoeInitRoutingV3";
}

def Muse_MoeTokenUnpermuteOp : Muse_Op<"moe_token_unpermute", [Pure]> {
  let summary = "aclnnMoeTokenUnpermute";
  let description = [{
    Operation for moe token unpermute computation.
    Accepts permuted_tokens, sorted_indices and optional probs tensor, padded_mode, optional restore_shape.
    Produces unpermute_tokens as result.
  }];

  let arguments = (ins
    MuseAnyTensor:$permuted_tokens,
    MuseAnyTensor:$sorted_indices,
    MuseOptional<MuseAnyTensor>:$probs,
    Muse_BooleanType:$padded_mode,
    MuseOptional<Muse_I64ArrayType>:$restore_shape
  );

  let results = (outs
    MuseAnyTensor:$Out
  );

  let assemblyFormat = [{
    $permuted_tokens `,` $sorted_indices `,` $probs `,` $padded_mode `,` $restore_shape
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnMoeTokenUnpermute";
}

def Muse_MulOp : Muse_Op<"mul", [Pure, Commutative]> {
  let summary = "Multiplication operation";
  let description = [{
    Element-wise multiplication of two tensors.
  }];

  let arguments = (ins
    MuseAnyTensor:$lhs,
    MuseAnyTensor:$rhs
  );

  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnMul";
}

def Muse_NeOp : Muse_Op<"ne", [Pure]> {
  let summary = "Element-wise not-equal comparison";
  let description = [{
    Element-wise not-equal comparison of two tensors.
    Returns a boolean tensor where each element is true if the corresponding elements are not equal.
  }];

  let arguments = (ins
    MuseAnyTensor:$self,
    MuseAnyTensor:$other
  );

  let results = (outs MuseAnyTensor:$out);

  let assemblyFormat = [{
    $self `,` $other attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnNeTensor";
}

def Muse_PermuteOp : Muse_Op<"permute", [Pure]> {
  let summary = "Tensor permutation operation";
  let description = [{
    Permutes the dimensions of a tensor according to the specified order.
    Reorders the axes of the input tensor based on the permutation array.
  }];

  let arguments = (ins
    MuseAnyTensor:$input,
    Muse_I64ArrayType:$perm
  );

  let results = (outs MuseAnyTensor:$output);

  let assemblyFormat = [{
    $input `,` $perm attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnPermute";
}

def Muse_PoolOp : Muse_Op<"pool", [Pure]> {
  let summary = "Pooling operation";
  let description = [{
    Performs pooling operations (max, average, etc.) on input tensors.
  }];

  let arguments = (ins
    MuseAnyTensor:$input,
    Muse_I64ArrayType:$kernel_size,
    Muse_I64ArrayType:$strides,
    Muse_I64ArrayType:$padding,
    Muse_StringType:$pool_type
  );

  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $kernel_size `,` $strides `,` $padding `,` $pool_type
    attr-dict `:` functional-type(operands, results)
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
    MuseAnyTensor:$input,
    Muse_I64ArrayType:$dimensions,
    Muse_BooleanType:$keepdim,
    Muse_DtypeType:$dtype
  );

  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $dimensions `,` $keepdim `,` $dtype
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnReduceSum";
}

def Muse_ReluOp : Muse_Op<"relu", [Pure]> {
  let summary = "ReLU activation function";
  let description = [{
    Applies the ReLU (Rectified Linear Unit) activation function.
  }];

  let arguments = (ins MuseAnyTensor:$input);
  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_RemainderTensorTensorOp : Muse_Op<"remainder_tensor_tensor", [Pure]> {
  let summary = "aclnnRemainderTensorTensor";
  let description = [{
    Operation for element-wise remainder computation between two tensors.
    Performs remainder operation on corresponding elements of input tensors.
    Computes the remainder of division of elements in self tensor by elements in other tensor.
    Both input tensors must have compatible shapes for broadcasting.
  }];

  let arguments = (ins
    MuseAnyTensor:$self,
    MuseAnyTensor:$other
  );

  let results = (outs
    MuseAnyTensor:$out
  );

  let assemblyFormat = [{
    $self `,` $other
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnRemainderTensorTensor";
}

def Muse_ReshapeOp : Muse_Op<"reshape", [Pure]> {
  let summary = "Reshape operation";
  let description = [{
    Reshapes input tensor to the specified shape.
    Takes two operands: input tensor and shape.
  }];

  let arguments = (ins
    MuseAnyTensor:$input,
    Muse_I64ArrayType:$shape
  );

  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $shape attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def Muse_RmsNormOp : Muse_Op<"rms_norm", [Pure]> {
  let summary = "aclnnRmsNorm";
  let description = [{
    Operation for RmsNorm computation.
    Accepts input tensor, gamma scaling factor, and epsilon value for numerical stability.
    Produces normalized output tensor and reciprocal of standard deviation.
  }];

  let arguments = (ins
    MuseAnyTensor:$x,
    MuseAnyTensor:$gamma,
    Muse_F64Type:$epsilon
  );

  let results = (outs
    MuseAnyTensor:$yOut,
    MuseAnyTensor:$rstdOut
  );

  let assemblyFormat = [{
    $x `,` $gamma `,` $epsilon
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnRmsNorm";
}

def Muse_ScatterNdUpdateOp : Muse_Op<"scatter_nd_update"> {
  let summary = "aclnnScatterNdUpdate";
  let description = [{
    Operation for ScatterNDUpdate computation.
    Accepts input, indices, updates tensors.
    Return the updated input tensor as result, which is a in-place ops.
  }];

  let arguments = (ins
    MuseAnyTensor:$input,
    MuseAnyTensor:$indices,
    MuseAnyTensor:$updates
  );

  let results = (outs
    MuseAnyTensor:$outs
  );

  let genAclnnOp = "aclnnScatterNdUpdate";
  let genAclnnCfg = (AclnnConfig (IORef 0, 0));
}

def Muse_SigmoidOp : Muse_Op<"sigmoid", [Pure]> {
  let summary = "Sigmoid activation function";
  let description = [{
    Applies the sigmoid activation function.
  }];

  let arguments = (ins MuseAnyTensor:$input);
  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnSigmoid";
}

def Muse_SizeOp : Muse_Op<"size", [Pure]> {
  let summary = "Get the size of a tensor dimension";
  let description = [{
    Returns the size of the specified dimension of the input tensor.
    Takes two operands: input tensor and dimension index.
    Returns a scalar tensor of type I64.
  }];

  let arguments = (ins
    MuseAnyTensor:$input,
    Muse_I64Type:$dim
  );

  let results = (outs Muse_I64Type:$result);

  let assemblyFormat = [{
    $input `,` $dim attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_SliceOp : Muse_Op<"slice", [Pure]> {
  let summary = "Selects a range of elements from an input tensor along a given dimension.";
  let description = [{
    Extracts a contiguous slice from the input tensor `self` along the specified dimension `dim`, 
    beginning at index `start`, and continuing up to (but not including) index `end`, 
    with elements selected every `step` indices.
  }];

  let arguments = (ins
    MuseAnyTensor:$self,
    Muse_I64Type:$dim,
    Muse_I64Type:$start,
    Muse_I64Type:$end,
    Muse_I64Type:$step
  );

  let results = (outs
    MuseAnyTensor:$result
  );

  let assemblyFormat = [{
    $self `,` $dim `,` $start `,` $end `,` $step
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnSlice";
}

def Muse_SoftmaxOp : Muse_Op<"softmax", [Pure]> {
  let summary = "Softmax activation function";
  let description = [{
    Applies the softmax activation function along the specified axis.
  }];

  let arguments = (ins
    MuseAnyTensor:$input,
    Muse_I64Type:$axis
  );
  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $axis attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_SplitOp : Muse_Op<"split_with_size", [Pure]> {
  let summary = "Split operation";
  let description = [{
    Splits tensor along the specified axis into multiple outputs.
  }];

  let arguments = (ins
    MuseAnyTensor:$input,
    Muse_I64ArrayType:$split_sizes,
    Muse_I64Type:$axis
  );

  let results = (outs MuseTensorList:$results);

  let assemblyFormat = [{
    $input `,` $split_sizes `,` $axis attr-dict `:` functional-type(operands, results)
  }];
  
  let genAclnnOp = "aclnnSplitWithSize";
}

def Muse_SplitTensorOp : Muse_Op<"split_tensor", [Pure]> {
  let summary = "Split tensor operation";
  let description = [{
    Splits tensor along the specified axis into multiple outputs.
  }];

  let arguments = (ins
    MuseAnyTensor:$input,
    Muse_I64Type:$split_sizes,
    Muse_I64Type:$axis
  );

  let results = (outs MuseTensorList:$results);

  let assemblyFormat = [{
    $input `,` $split_sizes `,` $axis attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnSplitTensor";
}

def Muse_SqueezeDimOp : Muse_Op<"squeeze_dim", [Pure]> {
  let summary = "Removes a single dimension of size one from the input tensor at the specified position (dim).";
  let description = [{
    The returned tensor shares the same underlying data with this tensor.
  }];

  let arguments = (ins
    MuseAnyTensor:$input,
    Muse_I64Type:$dim
  );

  let results = (outs
    MuseAnyTensor:$yOut
  );

  let assemblyFormat = [{
    $input `,` $dim attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_StridedSliceAssignOp : Muse_Op<"strided_slice_assign"> {
  let summary = "Strided slice assign operation";
  let description = [{
    Strided slice operation with support for updating values in the input tensor.
    Extracts a slice from the input tensor along the specified axes with the given starts, ends, and strides,
    and updates the extracted slice with the update tensor.
  }];

  let arguments = (ins
    MuseAnyTensor:$x,
    MuseAnyTensor:$update,
    Muse_I64ArrayType:$begin,
    Muse_I64ArrayType:$end,
    Muse_I64ArrayType:$strides,
    MuseOptional<Muse_I64ArrayType>:$axes
  );

  let results = (outs);

  let assemblyFormat = [{
    $x `,` $update `,` $begin `,` $end `,` $strides `,` $axes attr-dict `:` type(operands)
  }];
}

def Muse_SubOp : Muse_Op<"sub", [Pure]> {
  let summary = "aclnnSub";
  let description = [{
    Element-wise subtraction of two tensors with an optional alpha scaling factor.
    Computes: result = x - y * alpha
  }];

  let arguments = (ins
    MuseAnyTensor:$x,
    MuseAnyTensor:$y,
    Muse_ScalarType:$alpha
  );

  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $x `,` $y `,` $alpha attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnSub";
}

def Muse_SwiGluOp : Muse_Op<"swiglu"> {
  let summary = "aclnnSwiGlu";
  let description = [{
    Operation for swiglu computation.
    Accepts x tensor and the split dim.
    Produces swiglu output.
  }];

  let arguments = (ins
    MuseAnyTensor:$x,
    Muse_I64Type:$dim
  );

  let results = (outs
    MuseAnyTensor:$out
  );
  let assemblyFormat = [{
    $x `,` $dim
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnSwiGlu";
}

def Muse_UnpackListOp : Muse_Op<"unpack_list", [Pure]> {
  let summary = "Unpack a list into individual elements";
  let description = [{
    Unpacks a list into individual elements, similar to torch.prim.ListUnpack.
    This operation takes a list as input and produces variadic outputs representing
    each element of the list. The number of outputs must match the list length.
  }];

  let arguments = (ins
    MuseAnyList:$list
  );

  let results = (outs
    Variadic<AnyType>:$elements
  );

  let assemblyFormat = [{
    $list attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_UnsqueezeOp : Muse_Op<"unsqueeze", [Pure]> {
  let summary = "Returns a new tensor with a dimension of size one inserted at the specified position.";
  let description = [{
    The returned tensor shares the same underlying data with this tensor.
  }];

  let arguments = (ins
    MuseAnyTensor:$input,
    Muse_I64Type:$dim
  );

  let results = (outs MuseAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $dim attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_ViewOp : Muse_Op<"view", [Pure]> {
  let summary = "Returns a view tensor which shares the same memory with the original one.";

  let arguments = (ins
    MuseAnyTensor:$input,
    Muse_I64ArrayType:$shape
  );

  let results = (outs
    MuseAnyTensor:$yOut
  );

  let assemblyFormat = [{
    $input `,` $shape attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_ZerosOp : Muse_Op<"zeros", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "aclnnInplaceZero";
  let description = [{
    Operation for zeros ops.
    Accept shape size.
    Produces the device the same shape tensor, the value in tensor has been set zero.
  }];

  let arguments = (ins
    Muse_I64ArrayType:$size,
    Muse_DtypeType:$dtype,
    Muse_StringType:$device
  );

  let results = (outs
    MuseAnyTensor:$yOut
  );

  let assemblyFormat = [{
    $size `,` $dtype `,` $device
    attr-dict `:` functional-type(operands, results)
  }];
}

#endif // MUSE_DIALECT_MUSE_OPS_TD
