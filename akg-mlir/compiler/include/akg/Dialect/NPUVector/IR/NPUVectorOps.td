/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef NPUVECTOR_OPS
#define NPUVECTOR_OPS

include "akg/Dialect/NPUVector/IR/NPUVectorBase.td"
include "akg/Dialect/NPUVector/IR/NPUVectorTypes.td"
include "mlir/Dialect/Vector/IR/VectorAttributes.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/Arith/IR/ArithOpsInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/BuiltinAttributes.td"

//===----------------------------------------------------------------------===//
// TransferReadOp
//===----------------------------------------------------------------------===//

def NPUVector_TransferReadOp :
  NPUVector_Op<"transfer_read", [
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      AttrSizedOperandSegments,
      TypesMatchWith<"padding type matches source element type",
                     "source", "padding", 
                     "getElementTypeOrSelf($_self)">
    ]>,
    Arguments<(ins AnyShaped:$source,
                   Variadic<Index>:$indices,
                   AnyType:$padding,
                   Optional<VectorOf<[I1]>>:$mask,
                   Variadic<Index>:$dynamicSizes,
                   Optional<Index>:$maxSize)>,
    Results<(outs AnyVectorOrNPUVectorOfAnyRank:$vector)> {

  let summary = "Reads a supervector from memory into an SSA vector value with dynamic shape support.";

  let description = [{
    The `npu_vector.transfer_read` op performs a read from a slice within a
    [MemRef](../LangRef.md#memref-type) or a Ranked
    [Tensor](../LangRef.md#tensor-type) supplied as its first operand
    into a [vector](../LangRef.md#vector-type) of the same base elemental type.

    This operation supports dynamic shapes, allowing vector dimensions to be
    specified as dynamic (using ShapedType::kDynamic).

    The slice is further defined by a full-rank index within the MemRef/Tensor,
    supplied as the operands `[1 .. 1 + rank(memref/tensor))` that defines the
    starting point of the transfer (e.g. `%A[%i0, %i1, %i2]`).

    The permutation_map [attribute](../LangRef.md#attributes) is an
    [affine-map](Affine.md#affine-maps) which specifies the transposition on the
    slice to match the vector shape. The permutation map may be implicit and
    omitted from parsing and printing if it is the canonical minor identity map
    (i.e. if it does not permute or broadcast any dimension).

    The size of the slice is specified by the size of the vector, given as the
    return type. Dynamic dimensions in the vector type are supported.

    An SSA value `padding` of the same elemental type as the MemRef/Tensor is
    provided to specify a fallback value in the case of out-of-bounds accesses
    and/or masking.

    An optional SSA value `mask` may be specified to mask out elements read from
    the MemRef/Tensor. The `mask` type is an `i1` vector with a shape that
    matches how elements are read from the MemRef/Tensor, *before* any
    permutation or broadcasting. Elements whose corresponding mask element is
    `0` are masked out and replaced with `padding`.

    For every vector dimension, the boolean array attribute `in_bounds`
    specifies if the transfer is guaranteed to be within the source bounds. If
    set to "false", accesses (including the starting point) may run
    out-of-bounds along the respective vector dimension as the index increases.
    Non-vector and broadcast dimensions *must* always be in-bounds. The
    `in_bounds` array length has to be equal to the vector rank. This attribute
    has a default value: `false` (i.e. "out-of-bounds"). When skipped in the
    textual IR, the default value is assumed.

    Example:

    ```mlir
    // Read with dynamic shape support
    %f0 = arith.constant 0.0f : f32
    %v = npu_vector.transfer_read %A[%i0, %i1], (%f0)
         {permutation_map: (d0, d1) -> (d0, d1)} :
         memref<?x?xf32>, vector<?x?xf32>
    ```
  }];

  let assemblyFormat = [{
    $source `[` $indices `]`
    (`[` $dynamicSizes^ `]`)?
    (`[` $maxSize^ `]`)?
    `,` $padding
    (`,` $mask^ `:` type($mask))?
    attr-dict
    `:` type($source) `,` type($vector)
  }];

  let extraClassDeclaration = [{
    // MaskableOpInterface methods.
    bool supportsPassthru() { return true; }

    MutableOperandRange getDpsInitsMutable() {
      return MutableOperandRange(getOperation(), /*start=*/0, /*length=*/0);
    }
    
    // Helper method to get shaped type
    ShapedType getShapedType() {
      return llvm::cast<ShapedType>(getSource().getType());
    }
  }];
}

//===----------------------------------------------------------------------===//
// TransferWriteOp
//===----------------------------------------------------------------------===//

def NPUVector_TransferWriteOp :
  NPUVector_Op<"transfer_write", [
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      AttrSizedOperandSegments,
      DestinationStyleOpInterface
  ]>,
    Arguments<(ins AnyVectorOrNPUVectorOfAnyRank:$vector,
                   AnyShaped:$source,
                   Variadic<Index>:$indices,
                   Optional<VectorOf<[I1]>>:$mask)>,
    Results<(outs Optional<AnyRankedTensor>:$result)> {

  let summary = "The npu_vector.transfer_write op writes a supervector to memory with dynamic shape support.";

  let description = [{
    The `npu_vector.transfer_write` op performs a write from a
    [vector](../LangRef.md#vector-type), supplied as its first operand, into a
    slice within a [MemRef](../LangRef.md#memref-type) or a Ranked
    [Tensor](../LangRef.md#tensor-type) of the same base elemental type,
    supplied as its second operand.

    This operation supports dynamic shapes, allowing vector dimensions to be
    specified as dynamic (using ShapedType::kDynamic).

    The slice is further defined by a full-rank index within the MemRef/Tensor,
    supplied as the operands `[2 .. 2 + rank(memref/tensor))` that defines the
    starting point of the transfer (e.g. `%A[%i0, %i1, %i2, %i3]`).

    The permutation_map [attribute](../LangRef.md#attributes) is an
    [affine-map](Affine.md#affine-maps) which specifies the transposition on the
    slice to match the vector shape. The permutation map may be implicit and
    omitted from parsing and printing if it is the canonical minor identity map
    (i.e. if it does not permute any dimension). In contrast to `transfer_read`,
    write ops cannot have broadcast dimensions.

    The size of the slice is specified by the size of the vector. Dynamic
    dimensions in the vector type are supported.

    An optional SSA value `mask` may be specified to mask out elements written
    to the MemRef/Tensor. The `mask` type is an `i1` vector with a shape that
    matches how elements are written into the MemRef/Tensor, *after* applying
    any permutation. Elements whose corresponding mask element is `0` are
    masked out.

    For every vector dimension, the boolean array attribute `in_bounds`
    specifies if the transfer is guaranteed to be within the source bounds. If
    set to "false", accesses (including the starting point) may run
    out-of-bounds along the respective vector dimension as the index increases.
    Non-vector and broadcast dimensions *must* always be in-bounds. The
    `in_bounds` array length has to be equal to the vector rank. This attribute
    has a default value: `false` (i.e. "out-of-bounds"). When skipped in the
    textual IR, the default value is assumed.

    Example:

    ```mlir
    // Write with dynamic shape support
    %val = `ssa-value` : vector<?x?xf32>
    npu_vector.transfer_write %val, %A[%i0, %i1]
      {permutation_map: (d0, d1) -> (d0, d1)} :
      vector<?x?xf32>, memref<?x?xf32>
    ```
  }];

  let assemblyFormat = [{
    $vector `,` $source `[` $indices `]`
    (`,` $mask^ `:` type($mask))?
    attr-dict
    `:` type($vector) `,` type($source)
    (`->` type($result)^)?
  }];

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() {
      return MutableOperandRange(getOperation(), /*start=*/1, /*length=*/1);
    }
    
    // Helper method to get shaped type
    ShapedType getShapedType() {
      return llvm::cast<ShapedType>(getSource().getType());
    }
  }];
}

//===----------------------------------------------------------------------===//
// ReductionOp
//===----------------------------------------------------------------------===//

def NPUVector_ReductionOp :
  NPUVector_Op<"reduction", [Pure,
     PredOpTrait<"source operand and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>,
     OptionalTypesMatchWith<"dest and acc have the same type",
                            "dest", "acc", "::llvm::cast<Type>($_self)">,
     DeclareOpInterfaceMethods<ArithFastMathInterface>
    ]>,
    Arguments<(ins Vector_CombiningKindAttr:$kind,
               AnyVectorOrNPUVectorOfAnyRank:$vector,
               Optional<AnyType>:$acc,
               DefaultValuedAttr<
                 Arith_FastMathAttr,
                 "::mlir::arith::FastMathFlags::none">:$fastmath)>,
    Results<(outs AnyType:$dest)> {
  let summary = "reduction operation with dynamic shape support";
  let description = [{
    Reduces a vector "horizontally" into a scalar using the given
    operation: `add`/`mul`/`minsi`/`minui`/`maxsi`/`maxui`/`and`/`or`/`xor` for
    integers, and `add`/`mul`/`minnumf`/`maxnumf`/`minimumf`/`maximumf` for
    floats. Reductions also allow an optional fused accumulator.

    This operation supports dynamic shapes, allowing the input vector to have
    dynamic dimensions. The reduction is performed along all dimensions of the
    vector, producing a scalar result.

    Example:

    ```mlir
    %1 = npu_vector.reduction <add>, %0 : vector<?xf32> into f32

    %3 = npu_vector.reduction <xor>, %2 : vector<?xi32> into i32

    %4 = npu_vector.reduction <mul>, %0, %1 : vector<?x?xf32> into f32
    ```
  }];
  let extraClassDeclaration = [{
    NPUVectorType getSourceVectorType() {
      return ::llvm::cast<NPUVectorType>(getVector().getType());
    }
  }];
  let builders = [
    // Builder that infers the type of `dest`.
    OpBuilder<(ins "::mlir::vector::CombiningKind":$kind, "Value":$vector, "Value":$acc,
                    CArg<"::mlir::arith::FastMathFlags",
                         "::mlir::arith::FastMathFlags::none">:$fastMathFlags)>,
    // Builder that infers the type of `dest` and has no accumulator.
    OpBuilder<(ins "::mlir::vector::CombiningKind":$kind, "Value":$vector,
                    CArg<"::mlir::arith::FastMathFlags",
                         "::mlir::arith::FastMathFlags::none">:$fastMathFlags)>,
  ];
  let assemblyFormat = "$kind `,` $vector (`,` $acc^)? (`fastmath` `` $fastmath^)?"
                       " attr-dict `:` type($vector) `into` type($dest)";
}

//===----------------------------------------------------------------------===//
// ExtFOp - Floating Point Extension
//===----------------------------------------------------------------------===//

def NPUVector_ExtFOp :
  NPUVector_Op<"extf", [Pure, SameOperandsAndResultShape,
                        DeclareOpInterfaceMethods<ArithFastMathInterface>]>,
    Arguments<(ins NPUVectorOfAnyRank:$in,
                   DefaultValuedAttr<Arith_FastMathAttr,
                     "::mlir::arith::FastMathFlags::none">:$fastmath)>,
    Results<(outs NPUVectorOfAnyRank:$out)> {
  let summary = "NPU vector floating point extension operation";
  let description = [{
    Cast a floating-point NPU vector to a larger floating-point NPU vector of the same shape.
    The destination type must be strictly wider than the source type.
    
    Example:
    ```mlir
    %1 = npuvector.extf %0 : !npuvector<?xbf16> to !npuvector<?xf32>
    %2 = npuvector.extf %0 : !npuvector<128xf16> to !npuvector<128xf32>
    ```
  }];
  let assemblyFormat = "$in (`fastmath` `` $fastmath^)? attr-dict `:` qualified(type($in)) `to` qualified(type($out))";
}

//===----------------------------------------------------------------------===//
// TruncFOp - Floating Point Truncation
//===----------------------------------------------------------------------===//

def NPUVector_TruncFOp :
  NPUVector_Op<"truncf", [Pure, SameOperandsAndResultShape,
                          DeclareOpInterfaceMethods<ArithFastMathInterface>]>,
    Arguments<(ins NPUVectorOfAnyRank:$in,
                   DefaultValuedAttr<Arith_FastMathAttr,
                     "::mlir::arith::FastMathFlags::none">:$fastmath)>,
    Results<(outs NPUVectorOfAnyRank:$out)> {
  let summary = "NPU vector floating point truncation operation";
  let description = [{
    Truncate a floating-point NPU vector to a smaller floating-point NPU vector of the same shape.
    The destination type must be strictly narrower than the source type.
    
    Example:
    ```mlir
    %1 = npuvector.truncf %0 : !npuvector<?xf32> to !npuvector<?xbf16>
    %2 = npuvector.truncf %0 : !npuvector<128xf32> to !npuvector<128xf16>
    ```
  }];
  let assemblyFormat = "$in (`fastmath` `` $fastmath^)? attr-dict `:` qualified(type($in)) `to` qualified(type($out))";
}

//===----------------------------------------------------------------------===//
// ExtSIOp - Signed Integer Extension
//===----------------------------------------------------------------------===//

def NPUVector_ExtSIOp :
  NPUVector_Op<"extsi", [Pure, SameOperandsAndResultShape]>,
    Arguments<(ins NPUVectorOfAnyRank:$in)>,
    Results<(outs NPUVectorOfAnyRank:$out)> {
  let summary = "NPU vector signed integer extension operation";
  let description = [{
    Cast a signed integer NPU vector to a wider signed integer NPU vector of the same shape.
    
    Example:
    ```mlir
    %1 = npuvector.extsi %0 : !npuvector<?xi8> to !npuvector<?xi32>
    %2 = npuvector.extsi %0 : !npuvector<128xi16> to !npuvector<128xi32>
    ```
  }];
  let assemblyFormat = "$in attr-dict `:` qualified(type($in)) `to` qualified(type($out))";
}

//===----------------------------------------------------------------------===//
// ExtUIOp - Unsigned Integer Extension
//===----------------------------------------------------------------------===//

def NPUVector_ExtUIOp :
  NPUVector_Op<"extui", [Pure, SameOperandsAndResultShape]>,
    Arguments<(ins NPUVectorOfAnyRank:$in)>,
    Results<(outs NPUVectorOfAnyRank:$out)> {
  let summary = "NPU vector unsigned integer extension operation";
  let description = [{
    Cast an unsigned integer NPU vector to a wider unsigned integer NPU vector of the same shape.
    
    Example:
    ```mlir
    %1 = npuvector.extui %0 : !npuvector<?xi8> to !npuvector<?xi32>
    %2 = npuvector.extui %0 : !npuvector<128xi16> to !npuvector<128xi32>
    ```
  }];
  let assemblyFormat = "$in attr-dict `:` qualified(type($in)) `to` qualified(type($out))";
}

//===----------------------------------------------------------------------===//
// TruncIOp - Integer Truncation
//===----------------------------------------------------------------------===//

def NPUVector_TruncIOp :
  NPUVector_Op<"trunci", [Pure, SameOperandsAndResultShape]>,
    Arguments<(ins NPUVectorOfAnyRank:$in)>,
    Results<(outs NPUVectorOfAnyRank:$out)> {
  let summary = "NPU vector integer truncation operation";
  let description = [{
    Truncate an integer NPU vector to a narrower integer NPU vector of the same shape.
    
    Example:
    ```mlir
    %1 = npuvector.trunci %0 : !npuvector<?xi32> to !npuvector<?xi8>
    %2 = npuvector.trunci %0 : !npuvector<128xi32> to !npuvector<128xi16>
    ```
  }];
  let assemblyFormat = "$in attr-dict `:` qualified(type($in)) `to` qualified(type($out))";
}

//===----------------------------------------------------------------------===//
// SIToFPOp - Signed Integer to Float
//===----------------------------------------------------------------------===//

def NPUVector_SIToFPOp :
  NPUVector_Op<"sitofp", [Pure, SameOperandsAndResultShape]>,
    Arguments<(ins NPUVectorOfAnyRank:$in)>,
    Results<(outs NPUVectorOfAnyRank:$out)> {
  let summary = "NPU vector signed integer to floating-point conversion";
  let description = [{
    Convert a signed integer NPU vector to a floating-point NPU vector of the same shape.
    
    Example:
    ```mlir
    %1 = npuvector.sitofp %0 : !npuvector<?xi32> to !npuvector<?xf32>
    ```
  }];
  let assemblyFormat = "$in attr-dict `:` qualified(type($in)) `to` qualified(type($out))";
}

//===----------------------------------------------------------------------===//
// UIToFPOp - Unsigned Integer to Float
//===----------------------------------------------------------------------===//

def NPUVector_UIToFPOp :
  NPUVector_Op<"uitofp", [Pure, SameOperandsAndResultShape]>,
    Arguments<(ins NPUVectorOfAnyRank:$in)>,
    Results<(outs NPUVectorOfAnyRank:$out)> {
  let summary = "NPU vector unsigned integer to floating-point conversion";
  let description = [{
    Convert an unsigned integer NPU vector to a floating-point NPU vector of the same shape.
    
    Example:
    ```mlir
    %1 = npuvector.uitofp %0 : !npuvector<?xi32> to !npuvector<?xf32>
    ```
  }];
  let assemblyFormat = "$in attr-dict `:` qualified(type($in)) `to` qualified(type($out))";
}

//===----------------------------------------------------------------------===//
// FPToSIOp - Float to Signed Integer
//===----------------------------------------------------------------------===//

def NPUVector_FPToSIOp :
  NPUVector_Op<"fptosi", [Pure, SameOperandsAndResultShape]>,
    Arguments<(ins NPUVectorOfAnyRank:$in)>,
    Results<(outs NPUVectorOfAnyRank:$out)> {
  let summary = "NPU vector floating-point to signed integer conversion";
  let description = [{
    Convert a floating-point NPU vector to a signed integer NPU vector of the same shape.
    
    Example:
    ```mlir
    %1 = npuvector.fptosi %0 : !npuvector<?xf32> to !npuvector<?xi32>
    ```
  }];
  let assemblyFormat = "$in attr-dict `:` qualified(type($in)) `to` qualified(type($out))";
}

//===----------------------------------------------------------------------===//
// FPToUIOp - Float to Unsigned Integer
//===----------------------------------------------------------------------===//

def NPUVector_FPToUIOp :
  NPUVector_Op<"fptoui", [Pure, SameOperandsAndResultShape]>,
    Arguments<(ins NPUVectorOfAnyRank:$in)>,
    Results<(outs NPUVectorOfAnyRank:$out)> {
  let summary = "NPU vector floating-point to unsigned integer conversion";
  let description = [{
    Convert a floating-point NPU vector to an unsigned integer NPU vector of the same shape.
    
    Example:
    ```mlir
    %1 = npuvector.fptoui %0 : !npuvector<?xf32> to !npuvector<?xi32>
    ```
  }];
  let assemblyFormat = "$in attr-dict `:` qualified(type($in)) `to` qualified(type($out))";
}

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

def NPUVector_BitcastOp :
  NPUVector_Op<"bitcast", [Pure, SameOperandsAndResultShape]>,
    Arguments<(ins NPUVectorOfAnyRank:$in)>,
    Results<(outs NPUVectorOfAnyRank:$out)> {
  let summary = "NPU vector bitcast between values of equal bit width";
  let description = [{
    Bitcast an integer or floating-point NPU vector to another integer or
    floating-point NPU vector of equal bit width. Casts elementwise and
    preserves shape.
    
    Semantics match `arith.bitcast`, but on `!npuvector` types.
    
    Example:
    ```mlir
    %1 = npuvector.bitcast %0 : !npuvector<?xbf16> to !npuvector<?xi16>
    %2 = npuvector.bitcast %0 : !npuvector<128xf32> to !npuvector<128xi32>
    ```
  }];
  let assemblyFormat = "$in attr-dict `:` qualified(type($in)) `to` qualified(type($out))";
}

//===----------------------------------------------------------------------===//
// IndexCastOp
//===----------------------------------------------------------------------===//

def NPUVector_IndexCastOp :
  NPUVector_Op<"index_cast", [Pure, SameOperandsAndResultShape]>,
    Arguments<(ins NPUVectorOfAnyRank:$in)>,
    Results<(outs NPUVectorOfAnyRank:$out)> {
  let summary = "NPU vector cast between index and integer element types";
  let description = [{
    Casts between NPU vectors of index elements and NPU vectors of signless
    integer elements, preserving shape. If casting to a wider integer, the
    value is sign-extended. If casting to a narrower integer, the value is
    truncated.
    
    Semantics match `arith.index_cast`.
    
    Example:
    ```mlir
    %1 = npuvector.index_cast %0 : !npuvector<?xindex> to !npuvector<?xi64>
    %2 = npuvector.index_cast %0 : !npuvector<128xi32> to !npuvector<128xindex>
    ```
  }];
  let assemblyFormat = "$in attr-dict `:` qualified(type($in)) `to` qualified(type($out))";
}

//===----------------------------------------------------------------------===//
// IndexCastUIOp
//===----------------------------------------------------------------------===//

def NPUVector_IndexCastUIOp :
  NPUVector_Op<"index_castui", [Pure, SameOperandsAndResultShape]>,
    Arguments<(ins NPUVectorOfAnyRank:$in)>,
    Results<(outs NPUVectorOfAnyRank:$out)> {
  let summary = "NPU vector unsigned cast between index and integer element types";
  let description = [{
    Casts between NPU vectors of index elements and NPU vectors of signless
    integer elements, preserving shape. If casting to a wider integer, the
    value is zero-extended. If casting to a narrower integer, the value is
    truncated.
    
    Semantics match `arith.index_castui`.
    
    Example:
    ```mlir
    %1 = npuvector.index_castui %0 : !npuvector<?xindex> to !npuvector<?xi64>
    %2 = npuvector.index_castui %0 : !npuvector<128xi32> to !npuvector<128xindex>
    ```
  }];
  let assemblyFormat = "$in attr-dict `:` qualified(type($in)) `to` qualified(type($out))";
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

def NPUVector_BroadcastOp :
  NPUVector_Op<"broadcast", [Pure, AttrSizedOperandSegments]>,
    Arguments<(ins AnyType:$source,
                   Variadic<Index>:$dynamicSizes,
                   Optional<Index>:$maxSize)>,
    Results<(outs NPUVectorOfAnyRank:$result)> {

  let summary = "broadcast a scalar or vector to an NPU vector";
  
  let description = [{
    The `npuvector.broadcast` operation broadcasts a scalar value or a 
    lower-dimensional vector to an NPU vector by replicating the value 
    across all elements.
    
    This operation supports dynamic shapes in the result vector, allowing
    dimensions to be specified as dynamic (using `?`).
    
    The legality rules are:
    - The source operand must have the same element type as the result type
    - A scalar can be broadcast to any NPU vector
    - A k-D vector can be broadcast to an n-D vector if k <= n and
      trailing dimensions match
    
    Example:
    
    ```mlir
    // Broadcast scalar to static vector
    %0 = arith.constant 2.0 : f32
    %1 = npuvector.broadcast %0 : f32 to npuvector<1024xf32>
    
    // Broadcast scalar to dynamic vector
    %2 = npuvector.broadcast %0 : f32 to npuvector<?xf32>
    
    // Broadcast vector to higher dimension
    %3 = npuvector.broadcast %1 : npuvector<1024xf32> to npuvector<8x1024xf32>
    ```
  }];
  
  let assemblyFormat = "$source (`[` $dynamicSizes^ `]`)? (`[` $maxSize^ `]`)? `:` type($source) `to` qualified(type($result)) attr-dict";
  
  let builders = [
    OpBuilder<(ins "Type":$resultType, "Value":$source, "ValueRange":$dynamicSizes), [{
      build($_builder, $_state, resultType, source, dynamicSizes, /*maxSize=*/Value());
    }]>
  ];
}

#endif // NPUVECTOR_OPS

