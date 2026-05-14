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

#ifndef MFUSION_DIALECT_DVM_DVMOPS_TD
#define MFUSION_DIALECT_DVM_DVMOPS_TD

include "mfusion/Dialect/Dvm/IR/DvmDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// DVM Operations
//
// These operations map directly to the DVM C++ API (dvm::Kernel methods).
// We use generic ops (Unary, Binary, Reduce) with an `op_type` attribute
// to strictly align with the `dvm::Kernel::Unary/Binary/Reduce` interfaces.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// DVM Enum Definitions
//===----------------------------------------------------------------------===//

def Dvm_UnaryOpSqrt : I32EnumAttrCase<"Sqrt", 0>;
def Dvm_UnaryOpAbs  : I32EnumAttrCase<"Abs", 1>;
def Dvm_UnaryOpLog  : I32EnumAttrCase<"Log", 2>;
def Dvm_UnaryOpExp  : I32EnumAttrCase<"Exp", 3>;
def Dvm_UnaryOpReciprocal : I32EnumAttrCase<"Reciprocal", 4>;
def Dvm_UnaryOpIsFinite : I32EnumAttrCase<"IsFinite", 5>;
def Dvm_UnaryOpLogicalNot : I32EnumAttrCase<"LogicalNot", 6>;
def Dvm_UnaryOpRound : I32EnumAttrCase<"Round", 7>;
def Dvm_UnaryOpFloor : I32EnumAttrCase<"Floor", 8>;
def Dvm_UnaryOpCeil : I32EnumAttrCase<"Ceil", 9>;
def Dvm_UnaryOpTrunc : I32EnumAttrCase<"Trunc", 10>;

def Dvm_UnaryOpType : I32EnumAttr<"UnaryOpType", "DVM unary operation type",
    [Dvm_UnaryOpSqrt, Dvm_UnaryOpAbs, Dvm_UnaryOpLog, Dvm_UnaryOpExp,
     Dvm_UnaryOpReciprocal, Dvm_UnaryOpIsFinite, Dvm_UnaryOpLogicalNot,
     Dvm_UnaryOpRound, Dvm_UnaryOpFloor, Dvm_UnaryOpCeil, Dvm_UnaryOpTrunc]> {
  let cppNamespace = "::mlir::dvm";
  let description = [{
    Enum for UnaryOp input parameter.
    Maps directly to `dvm::UnaryOpType` enum in `dvm.h`.
  }];
}

def Dvm_BinaryOpEqual        : I32EnumAttrCase<"Equal", 0>;
def Dvm_BinaryOpNotEqual     : I32EnumAttrCase<"NotEqual", 1>;
def Dvm_BinaryOpGreater      : I32EnumAttrCase<"Greater", 2>;
def Dvm_BinaryOpGreaterEqual : I32EnumAttrCase<"GreaterEqual", 3>;
def Dvm_BinaryOpLess         : I32EnumAttrCase<"Less", 4>;
def Dvm_BinaryOpLessEqual    : I32EnumAttrCase<"LessEqual", 5>;
def Dvm_BinaryOpAdd          : I32EnumAttrCase<"Add", 6>;
def Dvm_BinaryOpSub          : I32EnumAttrCase<"Sub", 7>;
def Dvm_BinaryOpMul          : I32EnumAttrCase<"Mul", 8>;
def Dvm_BinaryOpDiv          : I32EnumAttrCase<"Div", 9>;
def Dvm_BinaryOpPow          : I32EnumAttrCase<"Pow", 10>;
def Dvm_BinaryOpMaximum      : I32EnumAttrCase<"Maximum", 11>;
def Dvm_BinaryOpMinimum      : I32EnumAttrCase<"Minimum", 12>;
def Dvm_BinaryOpLogicalAnd   : I32EnumAttrCase<"LogicalAnd", 13>;
def Dvm_BinaryOpLogicalOr    : I32EnumAttrCase<"LogicalOr", 14>;

def Dvm_BinaryOpType : I32EnumAttr<"BinaryOpType", "DVM binary operation type",
    [Dvm_BinaryOpEqual, Dvm_BinaryOpNotEqual, Dvm_BinaryOpGreater,
     Dvm_BinaryOpGreaterEqual, Dvm_BinaryOpLess, Dvm_BinaryOpLessEqual,
     Dvm_BinaryOpAdd, Dvm_BinaryOpSub, Dvm_BinaryOpMul, Dvm_BinaryOpDiv,
     Dvm_BinaryOpPow, Dvm_BinaryOpMaximum, Dvm_BinaryOpMinimum,
     Dvm_BinaryOpLogicalAnd, Dvm_BinaryOpLogicalOr]> {
  let cppNamespace = "::mlir::dvm";
  let description = [{
    Enum for BinaryOp input parameter.
    Maps directly to `dvm::BinaryOpType` enum in `dvm.h`.
  }];
}

def Dvm_ReduceOpSum : I32EnumAttrCase<"Sum", 0>;

def Dvm_ReduceOpType : I32EnumAttr<"ReduceOpType", "DVM reduce operation type",
    [Dvm_ReduceOpSum]> {
  let cppNamespace = "::mlir::dvm";
  let description = [{
    Enum for ReduceOp input parameter.
    Maps directly to `dvm::ReduceOpType` enum in `dvm.h`.
  }];
}

def Dvm_DTypeBool     : I32EnumAttrCase<"Bool", 0>;
def Dvm_DTypeFloat16  : I32EnumAttrCase<"Float16", 1>;
def Dvm_DTypeBFloat16 : I32EnumAttrCase<"BFloat16", 2>;
def Dvm_DTypeFloat32  : I32EnumAttrCase<"Float32", 3>;
def Dvm_DTypeInt32    : I32EnumAttrCase<"Int32", 4>;
def Dvm_DTypeInt64    : I32EnumAttrCase<"Int64", 5>;

def Dvm_DType : I32EnumAttr<"DType", "DVM data type",
    [Dvm_DTypeBool, Dvm_DTypeFloat16, Dvm_DTypeBFloat16, Dvm_DTypeFloat32,
     Dvm_DTypeInt32, Dvm_DTypeInt64]> {
  let cppNamespace = "::mlir::dvm";
  let description = [{
    Enum for DVM data types.
    Maps directly to `dvm::DType` enum in `dvm.h`.
  }];
}

def Dvm_GroupTypeSplitM : I32EnumAttrCase<"Split_M", 0>;
def Dvm_GroupTypeSplitN : I32EnumAttrCase<"Split_N", 1>;
def Dvm_GroupTypeSplitK : I32EnumAttrCase<"Split_K", 2>;

def Dvm_GroupType : I32EnumAttr<"GroupType", "DVM group type",
    [Dvm_GroupTypeSplitM, Dvm_GroupTypeSplitN, Dvm_GroupTypeSplitK]> {
  let cppNamespace = "::mlir::dvm";
  let description = [{
    Enum for GroupedMatMul input parameter.
    Maps directly to `dvm::GroupType` enum in `dvm.h`.
  }];
}

//===----------------------------------------------------------------------===//
// Load Operation
// Maps to: dvm::Kernel::Load(void* addr, ShapeRef* shape, DType type)
//===----------------------------------------------------------------------===//

def Dvm_LoadOp : Dvm_Op<"load", []> {
  let summary = "Load operation (Graph Input)";
  let description = [{
    Loads an input of DVM graph.
    Maps conceptually to `dvm::Kernel::Load(addr, shape, type)`.

    Note: `addr`, `shape`, and `type` arguments are omitted in the MLIR definition
    because this op provides a high-level semantic for graph inputs. Concrete
    memory addresses are bound at runtime by the backend, and shape/type metadata
    is inferred from the result tensor.
  }];

  let arguments = (ins AnyRankedTensor:$input);

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Store Operation
// Maps to: dvm::Kernel::Store(void* addr, NDObject* input)
//===----------------------------------------------------------------------===//

def Dvm_StoreOp : Dvm_Op<"store", []> {
  let summary = "Store operation (Graph Output)";
  let description = [{
    Store an output to the DVM graph.
    Maps conceptually to `dvm::Kernel::Store(addr, input)`.

    Note: `addr` argument is omitted in the MLIR definition because this op
    provides a high-level semantic for graph outputs. Concrete memory addresses
    are bound at runtime by the backend.
  }];

  let arguments = (ins
    AnyRankedTensor:$input
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Unary Operations
// Maps to: dvm::Kernel::Unary(UnaryOpType type, NDObject* input)
//===----------------------------------------------------------------------===//

def Dvm_UnaryOp : Dvm_Op<"unary", [Pure]> {
  let summary = "Generic unary operation";
  let description = [{
    Represents a unary operation in DVM.
    Maps directly to `dvm::Kernel::Unary(op_type, input)`.

    `op_type` is an enum: Sqrt, Abs, Exp, Log, Reciprocal,
    IsFinite, LogicalNot, Round, Floor, Ceil, Trunc.
  }];

  let arguments = (ins
    Dvm_UnaryOpType:$op_type,
    AnyRankedTensor:$input
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $op_type $input attr-dict `:` type($input) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Binary Operations
// Maps to: dvm::Kernel::Binary(BinaryOpType type, NDObject* lhs, NDObject* rhs)
//===----------------------------------------------------------------------===//

def Dvm_BinaryOp : Dvm_Op<"binary", [Pure]> {
  let summary = "Generic binary operation";
  let description = [{
    Represents a binary operation in DVM.
    Maps directly to `dvm::Kernel::Binary(op_type, lhs, rhs)`.

    `op_type` is an enum: Add, Sub, Mul, Div, Pow, Maximum, Minimum,
    LogicalAnd, LogicalOr, Equal, NotEqual, Greater, GreaterEqual,
    Less, LessEqual.
  }];

  let arguments = (ins
    Dvm_BinaryOpType:$op_type,
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $op_type $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Reduce Operations
// Maps to: dvm::Kernel::Reduce(ReduceOpType type, NDObject* input, ShapeRef* dims, bool keepdims)
//===----------------------------------------------------------------------===//

def Dvm_ReduceOp : Dvm_Op<"reduce", [Pure]> {
  let summary = "Generic reduce operation";
  let description = [{
    Represents a reduction operation in DVM.
    Maps directly to `dvm::Kernel::Reduce(op_type, input, dimensions, keepdims)`.

    `op_type` is an enum: Sum.
  }];

  let arguments = (ins
    Dvm_ReduceOpType:$op_type,
    AnyRankedTensor:$input,
    I64ArrayAttr:$dims,
    BoolAttr:$keepdims
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $op_type $input `dims` $dims `keepdims` $keepdims attr-dict `:` type($input) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Select Operation
// Maps to: dvm::Kernel::Select(NDObject* cond, NDObject* lhs, NDObject* rhs)
//===----------------------------------------------------------------------===//

def Dvm_SelectOp : Dvm_Op<"select", [Pure]> {
  let summary = "Select operation";
  let description = [{
    Element-wise selection: cond ? true_val : false_val.
    Maps directly to `dvm::Kernel::Select(cond, lhs, rhs)`.
  }];

  let arguments = (ins
    AnyRankedTensor:$cond,
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $cond `,` $lhs `,` $rhs attr-dict
    `:` `(` type($cond) `,` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Cast Operation
// Maps to: dvm::Kernel::Cast(NDObject* input, DType type)
//===----------------------------------------------------------------------===//

def Dvm_CastOp : Dvm_Op<"cast", [Pure]> {
  let summary = "Cast operation";
  let description = [{
    Casts input to a new type.
    Maps directly to `dvm::Kernel::Cast(input, result_type)`.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    Dvm_DType:$type
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = "$input `type` $type attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Broadcast Operation
// Maps to: dvm::Kernel::Broadcast(NDObject* input, ShapeRef* shape)
//===----------------------------------------------------------------------===//

def Dvm_BroadcastOp : Dvm_Op<"broadcast", [Pure]> {
  let summary = "Broadcast operation";
  let description = [{
    Broadcasts input to a new shape.
    Maps directly to `dvm::Kernel::Broadcast(input, shape)`.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    I64ArrayAttr:$shape
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = "$input `shape` $shape attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Reshape Operation
// Maps to: dvm::Kernel::Reshape(NDObject* input, ShapeRef* shape)
//===----------------------------------------------------------------------===//

def Dvm_ReshapeOp : Dvm_Op<"reshape", [Pure]> {
  let summary = "Reshape operation";
  let description = [{
    Reshapes input to a new shape.
    Maps directly to `dvm::Kernel::Reshape(input, shape)`.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    I64ArrayAttr:$shape
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = "$input `shape` $shape attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Copy Operation
// Maps to: dvm::Kernel::Copy(NDObject* input)
//===----------------------------------------------------------------------===//

def Dvm_CopyOp : Dvm_Op<"copy", [Pure]> {
  let summary = "Copy operation";
  let description = [{
    Copies the input tensor.
    Maps directly to `dvm::Kernel::Copy(input)`.
  }];

  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// OneHot Operation
// Maps to: dvm::Kernel::OneHot(NDObject* indices, ShapeRef* depth, int axis, T on_value, T off_value)
//===----------------------------------------------------------------------===//

def Dvm_OneHotOp : Dvm_Op<"onehot", [Pure]> {
  let summary = "OneHot operation";
  let description = [{
    Converts indices to one-hot encoded tensor.
    Maps directly to `dvm::Kernel::OneHot(indices, depth, axis, on_value, off_value)`.

    Note: depth is ShapeRef* in C++ API but typically contains a single value.
  }];

  let arguments = (ins
    AnyRankedTensor:$indices,
    I64ArrayAttr:$depth,
    I64Attr:$axis,
    AnyAttr:$on_value,
    AnyAttr:$off_value
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $indices `depth` $depth `axis` $axis attr-dict
    `:` type($indices) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// ElemAny Operation
// Maps to: dvm::Kernel::ElemAny(NDObject* input)
//===----------------------------------------------------------------------===//

def Dvm_ElemAnyOp : Dvm_Op<"elemany", [Pure]> {
  let summary = "Element-wise any operation";
  let description = [{
    Returns true if any element in the input is true.
    Maps directly to `dvm::Kernel::ElemAny(input)`.
  }];

  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// MatMul Operation
// Maps to: dvm::Kernel::MatMul(NDObject* lhs, NDObject* rhs, bool trans_a, bool trans_b, NDObject* bias)
//===----------------------------------------------------------------------===//

def Dvm_MatMulOp : Dvm_Op<"matmul", [Pure]> {
  let summary = "MatMul operation";
  let description = [{
    Matrix multiplication with optional bias.
    Maps directly to `dvm::Kernel::MatMul(lhs, rhs, trans_a, trans_b, bias)`.

    Supports 2D and multi-dimensional matrix multiplication with broadcasting.
    Does NOT support vector-vector, matrix-vector, or vector-matrix multiplication directly.

    `trans_a`: If true, transposes the last two dimensions of `lhs`.
    `trans_b`: If true, transposes the last two dimensions of `rhs`.
  }];

  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs,
    BoolAttr:$trans_a,
    BoolAttr:$trans_b,
    Optional<AnyRankedTensor>:$bias
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs `trans_a` $trans_a `trans_b` $trans_b (`bias` $bias^)? attr-dict
    `:` type($lhs) `,` type($rhs) (`,` type($bias)^)? `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// GroupedMatMul Operation
// Maps to: dvm::Kernel::GroupedMatMul(NDObject* lhs, NDObject* rhs, bool trans_a, bool trans_b,
//                                     NDObject* bias, NDObject* group_list, GroupType group_type)
//===----------------------------------------------------------------------===//

def Dvm_GroupedMatMulOp : Dvm_Op<"grouped_matmul", [Pure]> {
  let summary = "GroupedMatMul operation";
  let description = [{
    Grouped matrix multiplication with optional bias.
    Maps directly to `dvm::Kernel::GroupedMatMul(lhs, rhs, trans_a, trans_b, bias, group_list, group_type)`.
  }];

  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs,
    BoolAttr:$trans_a,
    BoolAttr:$trans_b,
    Optional<AnyRankedTensor>:$bias,
    AnyRankedTensor:$group_list,
    Dvm_GroupType:$group_type
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs `trans_a` $trans_a `trans_b` $trans_b
    `group_list` $group_list `group_type` $group_type (`bias` $bias^)? attr-dict
    `:` type($lhs) `,` type($rhs) `,` type($group_list) (`,` type($bias)^)? `->` type($result)
  }];
}

#endif // MFUSION_DIALECT_DVM_DVMOPS_TD
