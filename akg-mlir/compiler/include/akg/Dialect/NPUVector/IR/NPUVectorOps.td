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
                   Variadic<Index>:$maxSizes)>,
    Results<(outs AnyVectorOrNPUVectorOfAnyRank:$vector)> {

  let summary = "NPU super-vector load from memref/tensor with explicit dynamic tile sizes.";

  let description = [{
    NPUVector wrapper around the same idea as `vector.transfer_read`: load a slice of a
    `memref` or ranked `tensor` into a `vector` or `!npuvector` value with matching
    element type.

    Compared to upstream `vector.transfer_read`, this op **emphasizes explicit dynamic
    tile metadata for Ascend/NPU**: result ranks may use `ShapedType::kDynamic`;
    `dynamicSizes` supplies index SSA values for those dynamic extents (one per `?`, in
    shape order); optional `maxSizes` supplies matching upper bounds (same arity as
    `dynamicSizes` when present). Semantics of `padding`, `mask`, `permutation_map`,
    and `in_bounds` follow the MLIR Vector dialect contract for `transfer_read`---see
    LLVM's `VectorOps.td` / LangRef rather than duplicating them here.

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
    (`[` $maxSizes^ `]`)?
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

  let summary = "NPU super-vector store to memref/tensor; supports dynamic tile shapes.";

  let description = [{
    NPUVector wrapper around the same idea as `vector.transfer_write`: store a
    `vector` or `!npuvector` into the matching slice of a `memref` or ranked `tensor`
    with the same element type.

    **Extension**: the value may have dynamic extents (`?`) so it lines up with
    `transfer_read` and vectorization passes that emit dynamic tiles. Semantics of
    `permutation_map`, `mask`, and `in_bounds` match upstream `vector.transfer_write`;
    refer to LLVM `VectorOps.td` for the full text.

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
  let summary = "Fold an n-D NPU/vector tile to a scalar; supports dynamic rank sizes.";
  let description = [{
    **Full-tensor reduction** of a `vector` or `!npuvector` to a scalar using the same
    `CombiningKind` enumeration as `vector.reduction`; an optional fused accumulator
    operand is allowed.

    Unlike the upstream `vector.reduction` docs (often phrased around 1-D), this op is
    defined for **multi-dimensional tiles**, including dynamic `?` extents, for NPU
    reduction lowering.

    Example:

    ```mlir
    %1 = npuvector.reduction <add>, %0 : !npuvector<?xf32> into f32
    %2 = npuvector.reduction <xor>, %v : !npuvector<?xi32> into i32
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
// CmpIOp
//===----------------------------------------------------------------------===//

def NPUVector_CmpIOp :
  NPUVector_Op<"cmpi", [Pure, SameOperandsAndResultShape,
                        TypesMatchWith<"rhs type matches lhs type",
                                       "lhs", "rhs", "$_self">]>,
    Arguments<(ins Arith_CmpIPredicateAttr:$predicate,
                   NPUVectorOfAnyRank:$lhs,
                   NPUVectorOfAnyRank:$rhs)>,
    Results<(outs NPUVectorOfAnyRank:$result)> {
  let summary = "NPU vector integer comparison operation";
  let description = [{
    Integer comparison for NPU vectors. Compares two NPU vectors elementwise
    according to the predicate (eq/ne/slt/sle/sgt/sge/ult/ule/ugt/uge).
    Result is an NPU vector of i1 with the same shape.
    
    Semantics match `arith.cmpi`, but on `!npuvector` types.
    
    Example:
    ```mlir
    %1 = npuvector.cmpi eq, %a, %b : !npuvector<128xi32> to !npuvector<128xi1>
    %2 = npuvector.cmpi slt, %x, %y : !npuvector<?xi16> to !npuvector<?xi1>
    ```
  }];
  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` qualified(type($lhs)) `to` qualified(type($result))";
}

//===----------------------------------------------------------------------===//
// CmpFOp
//===----------------------------------------------------------------------===//

def NPUVector_CmpFOp :
  NPUVector_Op<"cmpf", [Pure, SameOperandsAndResultShape,
                        TypesMatchWith<"rhs type matches lhs type",
                                       "lhs", "rhs", "$_self">]>,
    Arguments<(ins Arith_CmpFPredicateAttr:$predicate,
                   NPUVectorOfAnyRank:$lhs,
                   NPUVectorOfAnyRank:$rhs,
                   DefaultValuedAttr<Arith_FastMathAttr,
                     "::mlir::arith::FastMathFlags::none">:$fastmath)>,
    Results<(outs NPUVectorOfAnyRank:$result)> {
  let summary = "NPU vector floating-point comparison operation";
  let description = [{
    Floating-point comparison for NPU vectors. Compares two NPU vectors
    elementwise according to the predicate (oeq/ogt/oge/olt/ole/one/ord/
    ueq/ugt/uge/ult/ule/une/uno). Result is an NPU vector of i1 with the
    same shape.
    
    Semantics match `arith.cmpf`, but on `!npuvector` types.
    
    Example:
    ```mlir
    %1 = npuvector.cmpf oeq, %a, %b : !npuvector<128xf32> to !npuvector<128xi1>
    %2 = npuvector.cmpf ult, %x, %y : !npuvector<?xbf16> to !npuvector<?xi1>
    ```
  }];
  let assemblyFormat = "$predicate `,` $lhs `,` $rhs (`fastmath` `` $fastmath^)? attr-dict `:` qualified(type($lhs)) `to` qualified(type($result))";
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

def NPUVector_SelectOp :
  NPUVector_Op<"select", [Pure,
    AllTypesMatch<["true_value", "false_value", "result"]>,
    TypesMatchWith<"false_value type matches true_value type",
                   "true_value", "false_value", "$_self">
  ]>,
    Arguments<(ins NPUVectorOfAnyRank:$condition,
                   NPUVectorOfAnyRank:$true_value,
                   NPUVectorOfAnyRank:$false_value)>,
    Results<(outs NPUVectorOfAnyRank:$result)> {
  let summary = "NPU vector conditional selection operation";
  let description = [{
    Selects elements from two NPU vectors based on a condition vector.
    The condition must be an NPU vector of i1 with the same shape as the
    value operands. For each element, if condition[i] is 1, result[i] =
    true_value[i], otherwise result[i] = false_value[i].
    
    Semantics match `arith.select`, but on `!npuvector` types.
    
    Example:
    ```mlir
    %1 = npuvector.select %cond, %a, %b : !npuvector<128xi1>, !npuvector<128xf32>
    %2 = npuvector.select %mask, %x, %y : !npuvector<?xi1>, !npuvector<?xi32>
    ```
  }];
  let assemblyFormat = "$condition `,` $true_value `,` $false_value attr-dict `:` qualified(type($condition)) `,` qualified(type($result))";
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

def NPUVector_BroadcastOp :
  NPUVector_Op<"broadcast", [Pure, AttrSizedOperandSegments]>,
    Arguments<(ins AnyType:$source,
                   Variadic<Index>:$dynamicSizes,
                   Variadic<Index>:$maxSizes)>,
    Results<(outs NPUVectorOfAnyRank:$result)> {

  let summary = "Broadcast to !npuvector with optional runtime sizes and max tile.";
  
  let description = [{
    Builds on `vector.broadcast` (replicate a scalar or smaller vector to a target
    shape) by adding **per-dimension runtime sizes** via `dynamicSizes` and optional
    **per-dimension upper bounds** `maxSizes` (same arity as `dynamicSizes` when present),
    so broadcasts align with dynamic vectorization and `transfer_read` tiles.

    Element type must match the result; a scalar can broadcast to any-rank `!npuvector`;
    when promoting rank, trailing dimensions must be compatible in the same sense as
    MLIR's `vector.broadcast` legality rules (see upstream docs).

    Example:
    
    ```mlir
    %c = arith.constant 2.0 : f32
    %1 = npuvector.broadcast %c [%vs] [%max] : f32 to !npuvector<?xf32>
    ```
  }];
  
  let assemblyFormat = "$source (`[` $dynamicSizes^ `]`)? (`[` $maxSizes^ `]`)? `:` type($source) `to` qualified(type($result)) attr-dict";
  
  let builders = [
    OpBuilder<(ins "Type":$resultType, "Value":$source, "ValueRange":$dynamicSizes), [{
      build($_builder, $_state, resultType, source, dynamicSizes, /*maxSizes=*/ValueRange());
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

def NPUVector_TransposeOp :
  NPUVector_Op<"transpose", [Pure,
    PredOpTrait<"operand and result have same element type",
                TCresVTEtIsSameAsOpBase<0, 0>>]>,
    Arguments<(ins NPUVectorOfAnyRank:$vector,
                  DenseI64ArrayAttr:$permutation)>,
    Results<(outs NPUVectorOfAnyRank:$result)> {
  let summary = "Permute ranks of an !npuvector (dynamic dims preserved by position).";
  let description = [{
    Permutes axes of an `!npuvector`. **Semantics align with `vector.transpose`**
    (see LLVM `VectorOps.td`) but operands use the NPU vector type; static dimensions
    are reordered by shape, while dynamic `?` extents remain runtime-sized and move to
    the result axis given by `permutation`.

    `permutation` must be a permutation of `[0, n-1]`.

    Example:

    ```mlir
    %1 = npuvector.transpose %0, [1, 0]
      : !npuvector<2x3xf32> to !npuvector<3x2xf32>
    ```
  }];

  let builders = [
    OpBuilder<(ins "Value":$vec, "ArrayRef<int64_t>":$perm), [{
      auto srcType = ::llvm::cast<NPUVectorType>(vec.getType());
      SmallVector<int64_t> resultShape(perm.size());
      for (unsigned i = 0; i < static_cast<unsigned>(perm.size()); ++i)
        resultShape[i] = srcType.getShape()[perm[i]];
      build($_builder, $_state,
            NPUVectorType::get(resultShape, srcType.getElementType()),
            vec, $_builder.getDenseI64ArrayAttr(perm));
    }]>
  ];

  let extraClassDeclaration = [{
    NPUVectorType getSourceVectorType() {
      return ::llvm::cast<NPUVectorType>(getVector().getType());
    }
    NPUVectorType getResultVectorType() {
      return ::llvm::cast<NPUVectorType>(getResult().getType());
    }
  }];

  let assemblyFormat = [{
    $vector `,` $permutation attr-dict `:` qualified(type($vector)) `to` qualified(type($result))
  }];
}

#endif // NPUVECTOR_OPS

