/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef FUSION_OPS
#define FUSION_OPS

include "akg/Dialect/Fusion/IR/FusionBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

class Fusion_Op<string mnemonic, list<Trait> traits = []>
    : Op<Fusion_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

def LoadOp : Fusion_Op<"load", [
  MemRefsNormalizable,
  AttrSizedOperandSegments]> {
  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
                        [MemRead]>:$memref,
                        Variadic<Index>:$indices,
                        Optional<AnyType>:$padding,
                        OptionalAttr<BoolArrayAttr>:$in_bounds);
  let results = (outs AnyType : $result);

  let extraClassDeclaration = [{
    Value getMemRef() { return getOperand(0); }
    void setMemRef(Value value) { setOperand(0, value); }
    void setPadding(Value value) {
      auto operands = getODSOperands(2);
      if (operands.empty()) {
        return;
      }
      int index = getODSOperands(0).size() + getODSOperands(1).size();
      setOperand(index, value);
    }
    MemRefType getMemRefType() {
      return getMemRef().getType().cast<MemRefType>();
    }
  }];

  let hasFolder = 1;
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// MultiLoadOp
//===----------------------------------------------------------------------===//
def MultiLoadOp : Fusion_Op<"multi_load", [
  MemRefsNormalizable,
  AttrSizedOperandSegments
]> {
  let arguments = (ins Arg<AnyMemRef, "the reference to mark fusion",
                               [MemWrite]>:$memref,
                     Variadic<Index>:$indices,
                     Optional<AnyType>:$padding,
                     OptionalAttr<BoolArrayAttr>:$in_bounds);
  let results = (outs AnyType : $result);

  let extraClassDeclaration = [{
    Value getMemRef() { return getOperand(0); }
    void setMemRef(Value value) { setOperand(0, value); }
    void setPadding(Value value) {
      auto operands = getODSOperands(2);
      if (operands.empty()) {
        return;
      }
      int index = getODSOperands(0).size() + getODSOperands(1).size();
      setOperand(index, value);
    }
    MemRefType getMemRefType() {
      return getMemRef().getType().cast<MemRefType>();
    }
  }];

  let hasFolder = 1;
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// InsertOp
//===----------------------------------------------------------------------===//
def InsertOp : Fusion_Op<"insert", []> {
  let description = [{
    The `insert` operation takes a memref operand and a data operand. It means
    that fusion pass will insert operations between data operand and result
    for the function tensor argument which maps to the memref operand.
  }];

  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
                           [MemRead]>:$memref,
                       AnyType:$data);
  let results = (outs AnyType : $result);

  let assemblyFormat = "$memref `,` $data  attr-dict `:` type($memref) `,` "
                       "type($data) `to` type($result)";
}

//===----------------------------------------------------------------------===//
// SubViewOp
//===----------------------------------------------------------------------===//
def SubViewOp : Fusion_Op<"subview", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<ViewLikeOpInterface>,
    AttrSizedOperandSegments,
    OffsetSizeAndStrideOpInterface,
    Pure]> {
  let summary = "fusion subview operation";
  let description = [{
    The "subview" operation converts a memref type to another memref type
    which represents a reduced-size view of the original memref as specified by
    the operation's offsets, sizes and strides arguments.
  }];

  let arguments = (ins AnyMemRef:$source,
                       Variadic<Index>:$offsets,
                       Variadic<Index>:$sizes,
                       Variadic<Index>:$strides,
                       DenseI64ArrayAttr:$static_offsets,
                       DenseI64ArrayAttr:$static_sizes,
                       DenseI64ArrayAttr:$static_strides);
  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    $source ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($source) `to` type($result)
  }];

  let builders = [
    // Build a SubViewOp with mixed static and dynamic entries and custom
    // result type. If the type passed is nullptr, it is inferred.
    OpBuilder<(ins "Value":$source, "ArrayRef<OpFoldResult>":$offsets,
      "ArrayRef<OpFoldResult>":$sizes, "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a SubViewOp with mixed static and dynamic entries and inferred
    // result type.
    OpBuilder<(ins "MemRefType":$resultType, "Value":$source,
      "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<OpFoldResult>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a SubViewOp with static entries and custom result type. If the
    // type passed is nullptr, it is inferred.
    OpBuilder<(ins "Value":$source, "ArrayRef<int64_t>":$offsets,
      "ArrayRef<int64_t>":$sizes, "ArrayRef<int64_t>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a SubViewOp with static entries and inferred result type.
    OpBuilder<(ins "MemRefType":$resultType, "Value":$source,
      "ArrayRef<int64_t>":$offsets, "ArrayRef<int64_t>":$sizes,
      "ArrayRef<int64_t>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a SubViewOp with dynamic entries and custom result type. If the
    // type passed is nullptr, it is inferred.
    OpBuilder<(ins "Value":$source, "ValueRange":$offsets,
      "ValueRange":$sizes, "ValueRange":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a SubViewOp with dynamic entries and inferred result type.
    OpBuilder<(ins "MemRefType":$resultType, "Value":$source,
      "ValueRange":$offsets, "ValueRange":$sizes, "ValueRange":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = [{
    /// Returns the type of the base memref operand.
    MemRefType getSourceType() {
      return getSource().getType().cast<MemRefType>();
    }

    /// The result of a subview is always a memref.
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }

    /// A subview result type can be fully inferred from the source type and the
    /// static representation of offsets, sizes and strides. Special sentinels
    /// encode the dynamic case.
    static Type inferResultType(MemRefType sourceMemRefType,
                                ArrayRef<int64_t> staticOffsets,
                                ArrayRef<int64_t> staticSizes,
                                ArrayRef<int64_t> staticStrides);
    static Type inferResultType(MemRefType sourceMemRefType,
                                ArrayRef<OpFoldResult> staticOffsets,
                                ArrayRef<OpFoldResult> staticSizes,
                                ArrayRef<OpFoldResult> staticStrides);

    /// A rank-reducing result type can be inferred from the desired result
    /// shape. Only the layout map is inferred.
    ///
    /// Note: The result shape cannot be inferred with just the result rank and
    /// and the desired sizes. In case there are more "ones" among the sizes
    /// than the difference in source/result rank, it is not clear which dims of
    /// size one should be dropped.
    static Type inferRankReducedResultType(ArrayRef<int64_t> resultShape,
                                           MemRefType sourceMemRefType,
                                           ArrayRef<int64_t> staticOffsets,
                                           ArrayRef<int64_t> staticSizes,
                                           ArrayRef<int64_t> staticStrides);
    static Type inferRankReducedResultType(ArrayRef<int64_t> resultShape,
                                           MemRefType sourceMemRefType,
                                           ArrayRef<OpFoldResult> staticOffsets,
                                           ArrayRef<OpFoldResult> staticSizes,
                                           ArrayRef<OpFoldResult> staticStrides);

    /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getSourceType().getRank();
      return {rank, rank, rank};
    }

    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }

    /// Return the dimensions of the source type that are dropped when
    /// the result is rank-reduced.
    llvm::SmallBitVector getDroppedDims();

    /// Given a `value`, asserted to be of MemRefType, build a SubViewOp that
    /// results in a rank reduction to the desired memref shape and return the
    /// new value created.
    /// If the shape of `value` is already the `desiredShape`, just return
    /// `value`.
    /// If the shape of `value` cannot be rank-reduced to `desiredShape`, fail.
    static FailureOr<Value> rankReduceIfNeeded(
      OpBuilder &b, Location loc, Value value, ArrayRef<int64_t> desiredShape);

    /// Returns the dynamic sizes for this subview operation if specified.
    ::mlir::Operation::operand_range getDynamicSizes() { return getSizes(); }

    /// Return the list of Range (i.e. offset, size, stride). Each
    /// Range entry contains either the dynamic value or a ConstantIndexOp
    /// constructed with `b` at location `loc`.
    ::mlir::SmallVector<::mlir::Range, 8> getOrCreateRanges(
            ::mlir::OpBuilder &b, ::mlir::Location loc) {
        return ::mlir::getOrCreateRanges(*this, b, loc);
    }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//
def Fusion_BroadcastOp :
  Fusion_Op<"broadcast", [Pure,
     PredOpTrait<"source operand and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>]>,
    Arguments<(ins AnyType:$source)>,
    Results<(outs AnyVectorOfAnyRank:$vector)> {
  let summary = "broadcast operation";
  let description = [{
    Broadcasts the scalar or k-D vector value in the source operand
    to a n-D result vector.
  }];
  let extraClassDeclaration = [{
    Type getSourceType() { return getSource().getType(); }
    VectorType getVectorType() {
      return getVector().getType().cast<VectorType>();
    }

    /// Return the dimensions of the result vector that were formerly ones in the
    /// source tensor and thus correspond to "dim-1" broadcasting.
    llvm::SetVector<int64_t> computeBroadcastedUnitDims();

    /// Broadcast `value` to a vector of `dstShape`, knowing that exactly the
    /// `broadcastedDims` dimensions in the dstShape are broadcasted.
    /// This requires (and asserts) that the broadcast is free of dim-1
    /// broadcasting.
    /// Since vector.broadcast only allows expanding leading dimensions, an extra
    /// vector.transpose may be inserted to make the broadcast possible.
    /// `value`, `dstShape` and `broadcastedDims` must be properly specified or
    /// the helper will assert. This means:
    ///   1. `dstShape` must not be empty.
    ///   2. `broadcastedDims` must be confined to [0 .. rank(value.getVectorType)]
    ///   2. `dstShape` trimmed of the dimensions specified in `broadcastedDims`
    //       must match the `value` shape.
    static Value createOrFoldBroadcastOp(
      OpBuilder &b, Value value,
      ArrayRef<int64_t> dstShape,
      const llvm::SetVector<int64_t> &broadcastedDims);
  }];
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($vector)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//
def Fusion_TransposeOp :
  Fusion_Op<"transpose", [Pure,
    PredOpTrait<"operand and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>]>,
    Arguments<(ins AnyVectorOfAnyRank:$vector, I64ArrayAttr:$transp)>,
    Results<(outs AnyVectorOfAnyRank:$result)> {
  let summary = "vector transpose operation";
  let description = [{
    Takes a n-D vector and returns the transposed n-D vector defined by
    the permutation of ranks in the n-sized integer array attribute.
  }];
  let builders = [
    OpBuilder<(ins "Value":$vector, "ArrayRef<int64_t>":$transp)>
  ];
  let extraClassDeclaration = [{
    VectorType getVectorType() {
      return getVector().getType().cast<VectorType>();
    }
    VectorType getResultType() {
      return getResult().getType().cast<VectorType>();
    }
    void getTransp(SmallVectorImpl<int64_t> &results);
    static StringRef getTranspAttrStrName() { return "transp"; }
  }];
  let assemblyFormat = [{
    $vector `,` $transp attr-dict `:` type($vector) `to` type($result)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//
def StoreOp : Fusion_Op<"store", [
  MemRefsNormalizable
]> {
  let arguments = (ins
                       AnyType:$value,
                       Arg<AnyMemRef, "the reference to store to",
                         [MemWrite]>:$memref,
                       Variadic<Index>:$indices,
                       OptionalAttr<BoolArrayAttr>:$in_bounds);

  let extraClassDeclaration = [{
    Value getValueToStore() { return getOperand(0); }

    Value getMemRef() { return getOperand(1); }
    void setMemRef(Value value) { setOperand(1, value); }
    MemRefType getMemRefType() {
      return getMemRef().getType().cast<MemRefType>();
    }
  }];

  let hasFolder = 1;
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

#endif // FUSION_OPS
