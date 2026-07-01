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
                   Variadic<Index>:$maxSizes,
                   OptionalAttr<AffineMapAttr>:$permutation_map)>,
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

  let builders = [
    OpBuilder<(ins "Type":$vectorType, "Value":$source, "ValueRange":$indices, "Value":$padding,
                   "Value":$mask, "ValueRange":$dynamicSizes, "ValueRange":$maxSizes,
                   "AffineMap":$permutationMap), [{
      build($_builder, $_state, vectorType, source, indices, padding, mask, dynamicSizes, maxSizes,
            ::mlir::AffineMapAttr::get(permutationMap));
    }]>,
    OpBuilder<(ins "Type":$vectorType, "Value":$source, "ValueRange":$indices, "Value":$padding,
                   "Value":$mask, "ValueRange":$dynamicSizes, "ValueRange":$maxSizes), [{
      build($_builder, $_state, vectorType, source, indices, padding, mask, dynamicSizes, maxSizes,
            ::mlir::AffineMapAttr());
    }]>
  ];

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
                   Optional<VectorOf<[I1]>>:$mask,
                   OptionalAttr<AffineMapAttr>:$permutation_map)>,
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

    The optional `permutation_map` follows upstream `vector.transfer_write`:
    it maps destination memref/tensor dimensions to vector dimensions and may
    be omitted for the canonical minor identity map. For example,
    `affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>` means vector dim 0
    writes memref dim 1, vector dim 1 writes memref dim 3, and vector dim 2
    writes memref dim 4. When the attribute is omitted, vector dims map to the
    trailing memref dims `[rank - vectorRank, ..., rank - 1]`. Rank-0 vectors
    use an empty-result map.

    Example:

    ```mlir
    %val = `ssa-value` : !npuvector<?x6x64xf32>
    npuvector.transfer_write %val, %out[%i, %j, %k, %c0, %c0]
      {permutation_map = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>} :
      !npuvector<?x6x64xf32>, memref<4x197x3x6x64xf32>
    ```
  }];

  let assemblyFormat = [{
    $vector `,` $source `[` $indices `]`
    (`,` $mask^ `:` type($mask))?
    attr-dict
    `:` type($vector) `,` type($source)
    (`->` type($result)^)?
  }];

  let builders = [
    OpBuilder<(ins "Type":$resultType, "Value":$vector, "Value":$source, "ValueRange":$indices,
                   "AffineMap":$permutationMap), [{
      build($_builder, $_state, resultType, vector, source, indices, Value(),
            ::mlir::AffineMapAttr::get(permutationMap));
    }]>,
    OpBuilder<(ins "Type":$resultType, "Value":$vector, "Value":$source, "ValueRange":$indices), [{
      build($_builder, $_state, resultType, vector, source, indices, Value(),
            ::mlir::AffineMapAttr());
    }]>,
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$vector, "Value":$source, "ValueRange":$indices,
                   "Value":$mask), [{
      build($_builder, $_state, resultTypes, vector, source, indices, mask,
            ::mlir::AffineMapAttr());
    }]>
  ];

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() {
      return MutableOperandRange(getOperation(), /*start=*/1, /*length=*/1);
    }

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
                 "::mlir::arith::FastMathFlags::none">:$fastmath,
               OptionalAttr<DenseI64ArrayAttr>:$reduction_dims)>,
    Results<(outs AnyType:$dest)> {
  let summary = "Reduce specified dims of an NPU/vector tile; defaults to all dims (scalar).";
  let description = [{
    Reduction of a `vector` or `!npuvector` using the same `CombiningKind` enumeration
    as `vector.reduction`; an optional fused accumulator operand is allowed.

    When `reduction_dims` is absent or empty, **all** dimensions are folded and the result
    is a scalar (backward-compatible with the previous full-reduction semantics).

    When `reduction_dims` is present, only those axes are collapsed and the result is a
    lower-rank `!npuvector` retaining the non-reduced dimensions. Reducing every source
    dimension explicitly produces a rank-0 `!npuvector`.

    Example:

    ```mlir
    // Full reduction (default):
    %1 = npuvector.reduction <add>, %0 : !npuvector<?xf32> into f32

    // Partial reduction — fold only dim 1:
    %2 = npuvector.reduction <add>, %v {reduction_dims = array<i64: 1>}
           : !npuvector<?x?xf32> into !npuvector<?xf32>
    ```
  }];
  let extraClassDeclaration = [{
    NPUVectorType getSourceVectorType() {
      return ::llvm::cast<NPUVectorType>(getVector().getType());
    }
    bool isPartialReduction() {
      auto dims = getReductionDims();
      return dims && !dims->empty() && ::llvm::isa<NPUVectorType>(getDest().getType());
    }
  }];
  let builders = [
    // Full reduction to scalar with accumulator.
    OpBuilder<(ins "::mlir::vector::CombiningKind":$kind, "Value":$vector, "Value":$acc,
                    CArg<"::mlir::arith::FastMathFlags",
                         "::mlir::arith::FastMathFlags::none">:$fastMathFlags), [{
      auto elemType = ::llvm::cast<ShapedType>(vector.getType()).getElementType();
      build($_builder, $_state, elemType, kind, vector, acc,
            /*reductionDims=*/nullptr, fastMathFlags);
    }]>,
    // Full reduction to scalar, no accumulator.
    OpBuilder<(ins "::mlir::vector::CombiningKind":$kind, "Value":$vector,
                    CArg<"::mlir::arith::FastMathFlags",
                         "::mlir::arith::FastMathFlags::none">:$fastMathFlags), [{
      auto elemType = ::llvm::cast<ShapedType>(vector.getType()).getElementType();
      build($_builder, $_state, elemType, kind, vector, /*acc=*/Value(),
            /*reductionDims=*/nullptr, fastMathFlags);
    }]>,
    // Partial reduction with explicit result type and reduction_dims.
    OpBuilder<(ins "Type":$destType, "::mlir::vector::CombiningKind":$kind,
                    "Value":$vector, "Value":$acc,
                    "::mlir::DenseI64ArrayAttr":$reductionDims,
                    CArg<"::mlir::arith::FastMathFlags",
                         "::mlir::arith::FastMathFlags::none">:$fastMathFlags), [{
      $_state.addTypes(destType);
      $_state.addAttribute("kind",
        ::mlir::vector::CombiningKindAttr::get($_builder.getContext(), kind));
      $_state.addOperands(vector);
      if (acc) $_state.addOperands(acc);
      if (reductionDims)
        $_state.addAttribute("reduction_dims", reductionDims);
      $_state.addAttribute("fastmath",
        ::mlir::arith::FastMathFlagsAttr::get($_builder.getContext(), fastMathFlags));
    }]>,
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
                   Variadic<Index>:$maxSizes,
                   DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$dimension)>,
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

    Attribute `dimension` defaults to `{}` (empty): treat as **prefix
    mapping** `m[i]=i` in lowering when doing rank lift. When **non-scalar** source is
    promoted to a higher rank (`srcRank < dstRank`), set non-empty `m` to give **destination
    dimension index** for each **source dimension** `i` (i.e. result dimension `m[i]`).

    Example:

    ```mlir
    %c = arith.constant 2.0 : f32
    %1 = npuvector.broadcast %c [%vs] [%max] : f32 to !npuvector<?xf32>
    ```
  }];

  let assemblyFormat = "$source (`[` $dynamicSizes^ `]`)? (`[` $maxSizes^ `]`)? `:` type($source) `to` qualified(type($result)) attr-dict";

  let builders = [
    OpBuilder<(ins "Type":$resultType, "Value":$source, "ValueRange":$dynamicSizes), [{
      build($_builder, $_state, resultType, source, dynamicSizes,
            /*maxSizes=*/ValueRange(),
            /*dimension=*/$_builder.getDenseI64ArrayAttr({}));
    }]>
  ];

  let hasFolder = 1;
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

//===----------------------------------------------------------------------===//
// ExtractSliceOp
//===----------------------------------------------------------------------===//

def NPUVector_ExtractSliceOp :
  NPUVector_Op<"extract_slice", [Pure, AttrSizedOperandSegments]>,
    Arguments<(ins NPUVectorOfAnyRank:$source,
                   Variadic<Index>:$offsets,
                   Variadic<Index>:$sizes,
                   Variadic<Index>:$strides,
                   DenseI64ArrayAttr:$keep_dims)>,
    Results<(outs NPUVectorOfAnyRank:$result)> {
  let summary = "Extract a (optionally rank-reduced) sub-tile from an !npuvector.";
  let description = [{
    Projects a sub-tile out of an `!npuvector` value, analogous to
    `memref.subview` / `tensor.extract_slice` but acting on an SSA NPU vector tile.

    `offsets`, `sizes`, `strides` each have one `Index` SSA value per **source**
    dimension. `keep_dims` is a strictly increasing subsequence of `[0, srcRank)`
    whose length equals the result rank: result dimension `r` comes from source
    dimension `keep_dims[r]`. Source dimensions that are dropped (not in
    `keep_dims`) must be sliced with size `1`. Element type is preserved.

    This op is the canonical way for the vectorizer to forward a high-rank scratch
    tile (kept entirely in SSA after `AllocBufferShrink`) into the lower-rank tiles
    that downstream consumers expect, without materializing a physical write to the
    shrunk buffer.

    Example:

    ```mlir
    // keep dims 0,1; drop the trailing channel dim at offset %c1.
    %sub = npuvector.extract_slice %src [%c0, %c0, %c1] [%s0, %s1, %c1] [%c1, %c1, %c1]
             {keep_dims = array<i64: 0, 1>}
           : !npuvector<?x?x2xbf16> to !npuvector<?x?xbf16>
    ```
  }];

  let assemblyFormat = [{
    $source `[` $offsets `]` `[` $sizes `]` `[` $strides `]` attr-dict
    `:` qualified(type($source)) `to` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    NPUVectorType getSourceVectorType() {
      return ::llvm::cast<NPUVectorType>(getSource().getType());
    }
    NPUVectorType getResultVectorType() {
      return ::llvm::cast<NPUVectorType>(getResult().getType());
    }
  }];
}

#endif // NPUVECTOR_OPS

