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

#ifndef MFUSE_DIALECT_MFUSE_META_OPS_TD
#define MFUSE_DIALECT_MFUSE_META_OPS_TD

include "mfusion/Dialect/Mfuse/IR/MfuseTypes.td"
include "mfusion/Dialect/Mfuse/IR/MfuseInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"

// This is the base class for binary element-wise ops that support automatic broadcasting.
// It provides a custom builder that automatically infers the symbolic shape during op construction,
// ensuring that each tensor operand is associated with a symbolic shape.
class Mfuse_BroadcastableBinaryOp<string mnemonic, list<Trait> traits = []>
    : Mfuse_Op<mnemonic, !listconcat([DeclareOpInterfaceMethods<Mfuse_InferSymbolicShapeInterface>], traits)> {
  string autoClassName = !subst("Mfuse_", "", NAME);
  let skipDefaultBuilders = 1;
  let builders = [
    // Explicit type builder (kept for callers that already compute resultType)
    OpBuilder<(ins "::mlir::Type":$resultType, "::mlir::Value":$lhs, "::mlir::Value":$rhs),
      "auto promoted = ::mlir::mfuse::promoteBinaryOperands<::mlir::mfuse::" # autoClassName # ">($_builder, $_state.location, lhs, rhs);\n" #
      "::mlir::mfuse::SymbolicBuilderHelper<::mlir::mfuse::" # autoClassName # ">::build($_builder, $_state, resultType, {std::get<0>(promoted), std::get<1>(promoted)}, {});"
    >,

    // New Builder: infer base RankedTensorType first, then delegate to SymbolicBuilderHelper.
    OpBuilder<(ins "::mlir::Value":$lhs, "::mlir::Value":$rhs),
      "auto promoted = ::mlir::mfuse::promoteBinaryOperands<::mlir::mfuse::" # autoClassName # ">($_builder, $_state.location, lhs, rhs);\n"
      "::mlir::Type resultType = " # autoClassName # "::inferResultType(std::get<0>(promoted), std::get<1>(promoted));\n"
      "::mlir::mfuse::SymbolicBuilderHelper<::mlir::mfuse::" # autoClassName # ">::build($_builder, $_state, resultType, {std::get<0>(promoted), std::get<1>(promoted)}, {});"
    >
  ];

  let extraClassDeclaration = [{
    static ::mlir::Type inferResultType(::mlir::Value lhs, ::mlir::Value rhs);
  }];
}

// This is the base class for reduce ops that support symbolic shape inference.
// It provides custom builders that handle the reduction attributes and delegate
// to SymbolicBuilderHelper, while requiring each concrete op to implement
// a static inferResultType.
class Mfuse_ReduceOp<string mnemonic, list<Trait> traits = []>
    : Mfuse_Op<mnemonic, !listconcat([DeclareOpInterfaceMethods<Mfuse_InferSymbolicShapeInterface>], traits)> {
  string autoClassName = !subst("Mfuse_", "", NAME);
  let skipDefaultBuilders = 1;

  let builders = [
    // Explicit type builder (kept for callers that already compute resultType)
    OpBuilder<(ins
        "::mlir::Type":$resultType,
        "::mlir::Value":$input,
        "::mlir::ArrayAttr":$dimensions,
        "::mlir::BoolAttr":$keepdim,
        CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attributes
      ),
      "llvm::SmallVector<mlir::NamedAttribute, 4> attrs(attributes.begin(), attributes.end());\n" #
      "attrs.push_back($_builder.getNamedAttr(\"dimensions\", dimensions));\n" #
      "attrs.push_back($_builder.getNamedAttr(\"keepdim\", keepdim));\n" #
      "::mlir::mfuse::SymbolicBuilderHelper<::mlir::mfuse::" # autoClassName # ">::build(" #
      "$_builder, $_state, resultType, ::mlir::ValueRange{input}, attrs);"
    >,

    // Builder that infers the result type from input, dimensions, keepdim and element type.
    OpBuilder<(ins
        "::mlir::Value":$input,
        "::mlir::ArrayAttr":$dimensions,
        "::mlir::BoolAttr":$keepdim,
        CArg<"mlir::Type", "nullptr">:$dtype
      ),
      "::mlir::Type elemType = dtype ? dtype : llvm::cast<::mlir::RankedTensorType>(input.getType()).getElementType();\n" #
      "::mlir::Type resultType = ::mlir::mfuse::" # autoClassName # "::inferResultType(input, dimensions, keepdim, elemType);\n" #
      "build($_builder, $_state, resultType, input, dimensions, keepdim);"
    >
  ];

  let extraClassDeclaration = [{
    static ::mlir::Type inferResultType(::mlir::Value input,
                                        ::mlir::ArrayAttr dimensions,
                                        ::mlir::BoolAttr keepdim,
                                        ::mlir::Type elementType);
  }];
}

//===----------------------------------------------------------------------===//
// MFUSE Meta Operations
//
// This file defines only meta operators supported by auto fusion.
//
// NOTE: All operations in this file are sorted alphabetically by their class name.
//===----------------------------------------------------------------------===//

def Mfuse_AbsOp : Mfuse_Op<"abs", [
    Pure,
    SameOperandsAndResultType,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Element-wise absolute value.";
  let description = [{
    Computes the absolute value of each element in the input tensor.
  }];

  let arguments = (ins
    AnyRankedTensor:$input
  );

  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_AddOp : Mfuse_BroadcastableBinaryOp<"add", [Pure, Commutative, SameOperandsAndResultElementType]> {
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

def Mfuse_BatchMatmulOp : Mfuse_Op<"batch_matmul", [Pure]> {
  let summary = "Batch matrix multiplication of two tensors with transpose options.";
  let description = [{
    Batch matrix multiplication with optional transpose on last two dimensions.
    Supports tensors with rank >= 2 and same rank.

    Default: (*B, N, C) @ (*B, C, M) -> (*B, N, M)
    transpose_a=True: (*B, C, N) @ (*B, C, M) -> (*B, N, M)
    transpose_b=True: (*B, N, C) @ (*B, M, C) -> (*B, N, M)

    Where *B represents multidimensional batch dimensions.
  }];

  let arguments = (ins
    AnyRankedTensor:$self,
    AnyRankedTensor:$mat2,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_a,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_b
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $self `,` $mat2 attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_BroadcastToOp : Mfuse_Op<"broadcast_to", [
    Pure,
    SameOperandsAndResultElementType,
    DeclareOpInterfaceMethods<Mfuse_InferSymbolicShapeInterface>
  ]> {
  let summary = "Broadcast tensor to a given shape.";
  let description = [{
    Broadcasts a tensor to the shape encoded by the result type,
    following NumPy broadcasting rules. Dimensions are added to the
    left (front) of the tensor shape as needed.

    Only semi-static output is supported:
    if both the input and output dimensions at a given axis are dynamic,
    the output dimension is interpreted as equal to the corresponding input dimension.
  }];

  let arguments = (ins AnyRankedTensor:$input);

  let results = (outs AnyRankedTensor:$output);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
        "mlir::Type":$resultType,
        "mlir::Value":$input,
        CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attributes
      ), [{
        ::mlir::mfuse::SymbolicBuilderHelper<::mlir::mfuse::BroadcastToOp>::build(
            $_builder, $_state, resultType, mlir::ValueRange(input), attributes);
      }]>
  ];
  let hasVerifier = 1;

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_CastOp : Mfuse_Op<"cast", [Pure, SameOperandsAndResultShape]> {
  let summary = "Cast tensor to a specified data type.";
  let description = [{
    Casts the input tensor to the specified data type.
  }];

  let arguments = (ins AnyRankedTensor:$input);

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;

  let extraClassDeclaration = [{
    static ::mlir::Type inferResultType(::mlir::Value input, ::mlir::Type elementType);
  }];

  let builders = [
    OpBuilder<(ins "::mlir::Value":$input, "::mlir::Type":$elementType), [{
      ::mlir::Type resultType = inferResultType(input, elementType);
      build($_builder, $_state, resultType, input);
    }]>
  ];
}

def Mfuse_CeilOp : Mfuse_Op<"ceil", [
    Pure,
    SameOperandsAndResultType,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Element-wise ceiling (round up) operation.";
  let description = [{
    Computes the ceiling of each element in the input tensor.
    Returns the smallest integer value that is greater than or equal to each element.
  }];

  let arguments = (ins
    AnyRankedTensor:$input
  );

  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_Conv2DOp : Mfuse_Op<"conv2d", [Pure]> {
  let summary = "2D convolution of input with weight.";
  let description = [{
    Performs 2D convolution: input (NCHW) and weight (KCHW) produce output (NKHW').
    Used for Conv2D+Cast fusion when Conv2D output is f16 and Cast target is f32.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weight
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $input `,` $weight attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_Conv2DWithBiasOp : Mfuse_Op<"conv2d_with_bias", [Pure]> {
  let summary = "2D convolution of input with weight and bias.";
  let description = [{
    Performs 2D convolution and adds bias: result = conv2d(input, weight) + broadcast(bias).
    Bias is 1D (output channels) and broadcasts to NCHW output.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weight,
    AnyRankedTensor:$bias
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $input `,` $weight `,` $bias attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_DivOp : Mfuse_BroadcastableBinaryOp<"div", [Pure, SameOperandsAndResultElementType]> {
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

def Mfuse_EqOp : Mfuse_BroadcastableBinaryOp<"eq", [Pure, SameOperandsElementType]> {
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

def Mfuse_ExpOp : Mfuse_Op<"exp", [
    Pure,
    SameOperandsAndResultType,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
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

def Mfuse_FloorOp : Mfuse_Op<"floor", [
    Pure,
    SameOperandsAndResultType,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Element-wise floor (round down) operation.";
  let description = [{
    Computes the floor of each element in the input tensor.
    Returns the largest integer value that is less than or equal to each element.
  }];

  let arguments = (ins
    AnyRankedTensor:$input
  );

  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_GeOp : Mfuse_BroadcastableBinaryOp<"ge", [Pure, SameOperandsElementType]> {
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

def Mfuse_GroupedMatmulOp : Mfuse_Op<"grouped_matmul", [Pure]> {
  let summary = "Grouped matrix multiplication operation.";
  let description = [{
    Performs multiple matmul operations in a single kernel with optional bias
    and quantization/dequantization parameters.

    split_item: 0=multiple tensors, 3=single tensor
    group_type: -1=no grouping, 0=group on m-axis
  }];

  let arguments = (ins
    MfuseTensorList:$x,
    MfuseTensorList:$weight,
    MfuseOptionalTensorList:$bias,
    MfuseOptionalTensorList:$scale,
    MfuseOptionalTensorList:$offset,
    MfuseOptionalTensorList:$antiquant_scale,
    MfuseOptionalTensorList:$antiquant_offset,
    AnyRankedTensor:$group_list,
    DefaultValuedAttr<I32Attr, "0">:$split_item,
    DefaultValuedAttr<I32Attr, "-1">:$group_type,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_a,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_b
  );

  let results = (outs MfuseTensorList:$y);

  let assemblyFormat = [{
    $x `,` $weight `,` $bias `,` $scale `,` $offset `,` $antiquant_scale `,` $antiquant_offset `,` $group_list attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_GtOp : Mfuse_BroadcastableBinaryOp<"gt", [Pure, SameOperandsElementType]> {
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

def Mfuse_IsFiniteOp : Mfuse_Op<"is_finite", [Pure]> {
  let summary = "Determines which elements are finite for each position.";
  let description = [{
    For each element in the input tensor, determines whether it is a finite value.
    An element is considered finite if it is not NaN, -INF, or INF.
  }];

  let arguments = (ins
    AnyRankedTensor:$input
  );

  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_LeOp : Mfuse_BroadcastableBinaryOp<"le", [Pure, SameOperandsElementType]> {
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

def Mfuse_LogOp : Mfuse_Op<"log", [
    Pure,
    SameOperandsAndResultType,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Element-wise natural logarithm operation.";
  let description = [{
    Computes the natural logarithm (base-e) of each element in the input tensor.
    For elements less than or equal to 0, returns NaN.
  }];

  let arguments = (ins
    AnyRankedTensor:$input
  );

  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_LogicalAndOp : Mfuse_BroadcastableBinaryOp<"logical_and", [Pure, Commutative, SameOperandsElementType]> {
  let summary = "Element-wise logical AND operation.";
  let description = [{
    Computes the element-wise logical AND of two tensors.
    Non-zero values are treated as true, zero values are treated as false.
  }];

  let arguments = (ins
    AnyRankedTensor:$self,
    AnyRankedTensor:$other
  );

  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $self `,` $other attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_LogicalNotOp : Mfuse_Op<"logical_not", [
    Pure,
    SameOperandsAndResultType,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Element-wise logical NOT operation.";
  let description = [{
    Computes the element-wise logical NOT of a tensor.
    Non-zero values are treated as true, zero values are treated as false.
  }];

  let arguments = (ins
    AnyRankedTensor:$input
  );

  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_LogicalOrOp : Mfuse_BroadcastableBinaryOp<"logical_or", [Pure, Commutative, SameOperandsElementType]> {
  let summary = "Element-wise logical OR operation.";
  let description = [{
    Computes the element-wise logical OR of two tensors.
    Non-zero values are treated as true, zero values are treated as false.
  }];

  let arguments = (ins
    AnyRankedTensor:$self,
    AnyRankedTensor:$other
  );

  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $self `,` $other attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_LtOp : Mfuse_BroadcastableBinaryOp<"lt", [Pure, SameOperandsElementType]> {
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

def Mfuse_MatmulOp : Mfuse_Op<"matmul", [Pure]> {
  let summary = "Matrix multiplication of two tensors (no bias).";
  let description = [{
    Performs matrix multiplication of two tensors (supports 1D to 6D tensors).
    Optional trans_x1/trans_x2 treat operand as transposed.
  }];

  let arguments = (ins
    AnyRankedTensor:$self,
    AnyRankedTensor:$other,
    DefaultValuedAttr<BoolAttr, "false">:$trans_x1,
    DefaultValuedAttr<BoolAttr, "false">:$trans_x2
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $self `,` $other attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_MatmulWithBiasOp : Mfuse_Op<"matmul_with_bias", [Pure]> {
  let summary = "Matrix multiplication of two tensors with bias.";
  let description = [{
    Performs matrix multiplication of two tensors (supports 1D to 6D) and adds
    broadcast(bias): result = self * other + broadcast(bias).
    Optional trans_x1/trans_x2 treat operand as transposed.
  }];

  let arguments = (ins
    AnyRankedTensor:$self,
    AnyRankedTensor:$other,
    AnyRankedTensor:$bias,
    DefaultValuedAttr<BoolAttr, "false">:$trans_x1,
    DefaultValuedAttr<BoolAttr, "false">:$trans_x2
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $self `,` $other `,` $bias attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_MaximumOp : Mfuse_BroadcastableBinaryOp<"maximum", [Pure, Commutative, SameOperandsAndResultElementType]> {
  let summary = "Element-wise maximum of two tensors.";
  let description = [{
    Computes the element-wise maximum of two tensors.
    For each element position, selects the maximum value from the corresponding elements.
  }];

  let arguments = (ins
    AnyRankedTensor:$x1,
    AnyRankedTensor:$x2
  );

  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $x1 `,` $x2 attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_MinimumOp : Mfuse_BroadcastableBinaryOp<"minimum", [Pure, Commutative, SameOperandsAndResultElementType]> {
  let summary = "Element-wise minimum of two tensors.";
  let description = [{
    Computes the element-wise minimum of two tensors.
    For each element position, selects the minimum value from the corresponding elements.
  }];

  let arguments = (ins
    AnyRankedTensor:$x1,
    AnyRankedTensor:$x2
  );

  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $x1 `,` $x2 attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_MulOp : Mfuse_BroadcastableBinaryOp<"mul", [Pure, Commutative, SameOperandsAndResultElementType]> {
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

def Mfuse_NeOp : Mfuse_BroadcastableBinaryOp<"ne", [Pure, SameOperandsElementType]> {
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

def Mfuse_NegOp : Mfuse_Op<"neg", [
    Pure,
    SameOperandsAndResultType,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
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

def Mfuse_PermuteOp : Mfuse_Op<"permute", [Pure, SameOperandsAndResultElementType]> {
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

def Mfuse_PowOp : Mfuse_BroadcastableBinaryOp<"pow", [Pure, SameOperandsAndResultElementType]> {
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

def Mfuse_RealDivOp : Mfuse_BroadcastableBinaryOp<"real_div", [Pure, SameOperandsAndResultElementType]> {
  let summary = "Element-wise real division of two tensors.";
  let description = [{
    Computes element-wise real division of two tensors: result = x / y.
    Performs floating-point division element-wise.
    Unlike integer division which truncates, this operation always returns floating-point results.
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

def Mfuse_ReciprocalOp : Mfuse_Op<"reciprocal", [
    Pure,
    SameOperandsAndResultType,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
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

def Mfuse_ReduceSumOp : Mfuse_ReduceOp<"reduce_sum", [Pure]> {
  let summary = "Sum operation";
  let description = [{
    Computes the sum of tensor elements along specified dimensions.
    The boolean parameter controls whether to keep dimensions,
    and the result tensor type specifies the output data type.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    I64ArrayAttr:$dimensions,
    BoolAttr:$keepdim
  );

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_ReshapeOp : Mfuse_Op<"reshape", [
    Pure,
    SameOperandsAndResultElementType,
    DeclareOpInterfaceMethods<Mfuse_InferSymbolicShapeInterface>
  ]> {
  let summary = "Reshape operation";
  let description = [{
    Reshapes input tensor to the output shape encoded by result type.
    Only semi-static output is supported: at most one dynamic dimension.
  }];

  let arguments = (ins
    AnyRankedTensor:$input
  );

  let results = (outs AnyRankedTensor:$result);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
        "mlir::Type":$resultType,
        "mlir::Value":$input,
        CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attributes
      ), [{
        ::mlir::mfuse::SymbolicBuilderHelper<::mlir::mfuse::ReshapeOp>::build(
            $_builder, $_state, resultType, mlir::ValueRange(input), attributes);
      }]>
  ];
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_RsqrtOp : Mfuse_Op<"rsqrt", [
    Pure,
    SameOperandsAndResultType,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Element-wise reciprocal square root operation.";
  let description = [{
    Computes the reciprocal of the square root of each element in the input tensor.
    Returns 1 / sqrt(x) for each element. For elements less than 0, returns NaN.
  }];

  let arguments = (ins
    AnyRankedTensor:$input
  );

  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_SelectOp : Mfuse_Op<"select", [Pure]> {
  let summary = "Select elements from two tensors based on a condition.";
  let description = [{
    Selects elements from two tensors based on a condition tensor.
    Returns a tensor with elements selected from 'on_true' or 'on_false'
    based on the corresponding boolean value in the 'condition' tensor.
    If condition[i] is true, output[i] = on_true[i].
    If condition[i] is false, output[i] = on_false[i].
    All three input tensors must be broadcastable to the same shape.
  }];

  let arguments = (ins
    AnyRankedTensor:$condition,
    AnyRankedTensor:$on_true,
    AnyRankedTensor:$on_false
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $condition `,` $on_true `,` $on_false attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_SliceOp : Mfuse_Op<"slice", [Pure]> {
  let summary = "Extract a slice along a single dimension.";
  let description = [{
    Extracts a slice from the input tensor along a single dimension with given start, end, and step.
    Only the dimension 'dim' has different size between input and output.
    Supports index normalization and automatic size calculation.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    I64Attr:$dim,
    I64Attr:$start,
    I64Attr:$end,
    I64Attr:$step
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $input 
    `dim` $dim 
    `start` $start 
    `end` $end 
    `step` $step
    attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_SqrtOp : Mfuse_Op<"sqrt", [
    Pure,
    SameOperandsAndResultType,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Element-wise square root operation.";
  let description = [{
    Computes the square root of each element in the input tensor.
    Returns the non-negative square root of each element.
    For elements less than 0, returns NaN.
  }];

  let arguments = (ins
    AnyRankedTensor:$input
  );

  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Mfuse_SubOp : Mfuse_BroadcastableBinaryOp<"sub", [Pure, SameOperandsAndResultElementType]> {
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

def Mfuse_TruncOp : Mfuse_Op<"trunc", [
    Pure,
    SameOperandsAndResultType,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Element-wise truncation (round toward zero) operation.";
  let description = [{
    Computes the truncation of each element in the input tensor by rounding toward zero.
    This is different from floor which always rounds down, or ceil which always rounds up.
  }];

  let arguments = (ins
    AnyRankedTensor:$input
  );

  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

#endif // MFUSE_DIALECT_MFUSE_META_OPS_TD
