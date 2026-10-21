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

#ifndef AKG_DIALECT_PYPTO_IR_PYPTOOPS_TD
#define AKG_DIALECT_PYPTO_IR_PYPTOOPS_TD

include "akg/Dialect/Pypto/IR/PyptoBase.td"
include "akg/Dialect/Pypto/IR/PyptoTypes.td"
include "akg/Dialect/Pypto/IR/PyptoAttrs.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Pypto ops definitions.
//
// Every op mirrors exactly one PyPTO runtime entry point; the emitpy lowering
// (`-convert-pypto-to-emitpy`) turns each op into the matching `pl.*`
// call with the argument/keyword layout the runtime prescribes. Ops stay
// dtype-agnostic: element types travel in the operand/result types.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// A. Constants and allocation
//===----------------------------------------------------------------------===//

def Pypto_MatmulOp : Pypto_Op<"matmul"> {
  let summary = "Tensor matrix multiplication with an optional initial accumulator";
  let description =
    [{
    Without `init`, computes lhs @ rhs. With `init`, preserves the Linalg
    destination semantics: init + lhs @ rhs. The result type selects the
    output dtype.
  }];
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs, Optional<AnyRankedTensor>:$accumulator);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat =
    [{
    $lhs `,` $rhs (`init` $accumulator^)? attr-dict `:` functional-type(operands, results)
  }];
}

def Pypto_ConstantOp : Pypto_Op<"constant"> {
  let summary = "Produce a Python scalar with an exact PyPTO dtype";
  let description =
    [{
    Syntax:

    ```
    %c = pypto.constant "1.0" : f32
    ```

    `value` carries the Python literal text (e.g. `"1.0"`, `"42"`, `"True"`).
    The result type pins the exact dtype; a bare Python literal would
    default to index/float64, and scalar ops such as `pl.write` require the
    value dtype to match the tensor dtype.

    The emitpy lowering emits `pl.const(value, dtype)`.
  }];
  let arguments = (ins StrAttr:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def Pypto_TileFullOp : Pypto_Op<"tile_full"> {
  let summary = "Allocate a tile filled with a splat value";
  let description =
    [{
    Syntax:

    ```
    %t = pypto.tile_full "0.0" : !pypto.tile<f32, [8, 8]>
    %d = pypto.tile_full "0" extents [%n] : !pypto.tile<i32, [?]>
    ```

    `value` carries the Python literal of the splat element. The tile shape
    and dtype are taken from the result type; `extents` names the runtime
    bound of each dynamic (`?`) dimension and is left out for statically
    shaped results.

    The emitpy lowering emits
    `pl.tile.full(shape, dtype, value)` with all three arguments positional.
  }];
  let arguments = (ins StrAttr:$value, Variadic<Index>:$extents);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$value (`extents` `[` $extents^ `]`)? attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def Pypto_CreateTileOp : Pypto_Op<"create_tile"> {
  let summary = "Allocate an uninitialized scratch tile";
  let description =
    [{
    Syntax:

    ```
    %t = pypto.create_tile : !pypto.tile<f32, [8, 8]>
    ```

    The shape and dtype are taken from the result type.

    The emitpy lowering emits
    `pl.create_tile(shape, dtype=..., target_memory=pl.MemorySpace.Vec)`.
  }];
  let results = (outs Pypto_TileType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// B. Tile views
//===----------------------------------------------------------------------===//

def Pypto_SliceOp : Pypto_Op<"slice"> {
  let summary = "Cut a statically shaped window out of a tile or tensor";
  let description =
    [{
    Syntax:

    ```
    %v = pypto.slice %src[%o0, %o1], [8, 8] : src-type to result-type
    %w = pypto.slice %src[%o0], [16] drop_dims = [0] : src-type to result-type
    ```

    `window` is full-rank over the source, with unit dims standing in for
    the axes a rank-reducing view drops; `drop_dims` lists those axes.
    `offsets` carry the start coordinates and may be loop-dependent.

    The emitpy lowering emits
    `pl.slice(src, [window], [offsets], drop_dims=[...])` where the last
    keyword argument only appears for rank-reducing slices.
  }];
  let arguments = (ins AnyType:$source, DenseI64ArrayAttr:$window, Variadic<Index>:$offsets,
    OptionalAttr<DenseI64ArrayAttr>:$dropDims);
  let results = (outs AnyType:$result);
  let assemblyFormat =
    [{
    $source `[` $offsets `]` `,` $window (`drop_dims` `=` $dropDims^)?
    attr-dict `:` type($source) `to` type($result)
  }];
  let hasVerifier = 1;
}

def Pypto_ReshapeOp : Pypto_Op<"reshape"> {
  let summary = "View a tile or tensor under a different shape";
  let description =
    [{
    Syntax:

    ```
    %v = pypto.reshape %src : src-type to result-type
    %w = pypto.reshape %src extents [%n, %c1] : src-type to result-type
    ```

    The new shape is the result type's shape; the element count must be
    preserved. `extents` carries the runtime bound of every dimension and is
    left out for statically shaped results (mirroring `pypto.tile_full`).

    The emitpy lowering emits `pl.reshape(src, [shape])`.
  }];
  let arguments = (ins AnyType:$source, Variadic<Index>:$extents);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$source (`extents` `[` $extents^ `]`)? attr-dict `:` type($source) `to` type($result)";
  let hasVerifier = 1;
}

def Pypto_TransposeOp : Pypto_Op<"transpose"> {
  let summary = "Swap two axes of a tile or tensor";
  let description =
    [{
    Syntax:

    ```
    %v = pypto.transpose %src, 0, 1 : src-type to result-type
    ```

    PyPTO swaps exactly two axes, so a general permutation must be lowered
    into a chain of swaps. Identity permutations fold away at conversion
    time and never produce this op.

    The emitpy lowering emits `pl.transpose(src, i, j)`.
  }];
  let arguments = (ins AnyType:$source, I64Attr:$i, I64Attr:$j);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$source `,` $i `,` $j attr-dict `:` type($source) `to` type($result)";
  let hasVerifier = 1;
}

def Pypto_ReinterpretViewOp : Pypto_Op<"reinterpret_view"> {
  let summary = "Reinterpret a tile's bits under a different dtype";
  let description =
    [{
    Syntax:

    ```
    %v = pypto.reinterpret_view %src : src-type to result-type
    ```

    The shape is preserved and the source and target element types must
    have the same bit width.

    The emitpy lowering emits `pl.reinterpret_view(src, dtype)`.
  }];
  let arguments = (ins AnyType:$source);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// C. Conversion, comparison and selection
//===----------------------------------------------------------------------===//

def Pypto_CastOp : Pypto_Op<"cast"> {
  let summary = "Convert a scalar or tile to another dtype";
  let description =
    [{
    Syntax:

    ```
    %v = pypto.cast %src : src-type to result-type
    ```

    The shape is preserved; the target dtype comes from the result type.
    Bit-preserving reinterpretation is `pypto.reinterpret_view` instead.

    The emitpy lowering emits `pl.cast(src, dtype)`.
  }];
  let arguments = (ins AnyType:$source);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($result)";
  let hasVerifier = 1;
}

def Pypto_CmpOp : Pypto_Op<"cmp"> {
  let summary = "Elementwise comparison of two tiles";
  let description =
    [{
    Syntax:

    ```
    %v = pypto.cmp lt, %lhs, %rhs : tile-type, tile-type -> i1-tile-type
    ```

    The predicate numeric values match the PyPTO `cmp_type` argument codes.

    The emitpy lowering emits `pl.cmp(lhs, rhs, cmp_type=...)` with the
    code passed as the `cmp_type` keyword argument. Scalar comparisons
    lower to the eq/ne/lt/le/gt/ge syntax helpers and never produce this op.
  }];
  let arguments = (ins Pypto_CmpPredicateAttr:$predicate, AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat =
    [{
    $predicate `,` $lhs `,` $rhs attr-dict `:`
    type($lhs) `,` type($rhs) `->` type($result)
  }];
  let hasVerifier = 1;
}

def Pypto_SelectOp : Pypto_Op<"select"> {
  let summary = "Elementwise selection between two tiles";
  let description =
    [{
    Syntax:

    ```
    %v = pypto.select %cond, %trueValue, %falseValue : result-type
    ```

    Scratch geometry is architecture-defined and synthesized by the
    runtime's composite-op lowering, so the result shape may be dynamic.
    Scalar selections lower to the `select` syntax helper and never
    produce this op. Exactly one branch may instead be a scalar: the
    runtime's composite materializes it with `tile.full`, which demands
    a compile-time literal.

    The emitpy lowering emits `pl.tile.select(condition, trueValue,
    falseValue)` with all arguments positional; a constant scalar branch
    rides inline as `pl.const(value, dtype)` text.
  }];
  let arguments = (ins AnyType:$condition, AnyType:$trueValue, AnyType:$falseValue);
  let results = (outs AnyType:$result);
  let assemblyFormat =
    [{
    $condition `,` $trueValue `,` $falseValue attr-dict `:`
    type($condition) `,` type($trueValue) `,` type($falseValue) `to`
    type($result)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// D. Elementwise binary ops
//===----------------------------------------------------------------------===//

class Pypto_BinaryOp<string mnemonic> : Pypto_Op<mnemonic> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `to` type($result)";
  let hasVerifier = 1;
}

def Pypto_AddOp : Pypto_BinaryOp<"add"> { let summary = "Elementwise or scalar addition (`pl.add`)"; }
def Pypto_SubOp : Pypto_BinaryOp<"sub"> { let summary = "Elementwise or scalar subtraction (`pl.sub`)"; }
def Pypto_MulOp : Pypto_BinaryOp<"mul"> { let summary = "Elementwise or scalar multiplication (`pl.mul`)"; }
def Pypto_DivOp : Pypto_BinaryOp<"div"> { let summary = "Elementwise or scalar division (`pl.div`)"; }
def Pypto_FmodOp : Pypto_BinaryOp<"fmod"> { let summary = "Elementwise or scalar remainder (`pl.fmod`)"; }
def Pypto_MaximumOp : Pypto_BinaryOp<"maximum"> { let summary = "Elementwise maximum (`pl.maximum`)"; }
def Pypto_MinimumOp : Pypto_BinaryOp<"minimum"> { let summary = "Elementwise minimum (`pl.minimum`)"; }
def Pypto_AndOp : Pypto_BinaryOp<"and"> { let summary = "Elementwise or scalar bitwise and (`pl.and_`)"; }
def Pypto_OrOp : Pypto_BinaryOp<"or"> { let summary = "Elementwise or scalar bitwise or (`pl.or_`)"; }
def Pypto_XorOp : Pypto_BinaryOp<"xor"> { let summary = "Elementwise or scalar bitwise xor (`pl.xor`)"; }
def Pypto_ShlOp : Pypto_BinaryOp<"shl"> { let summary = "Elementwise or scalar left shift (`pl.shl`)"; }
def Pypto_ShrOp : Pypto_BinaryOp<"shr"> { let summary = "Elementwise or scalar right shift (`pl.shr`)"; }

//===----------------------------------------------------------------------===//
// D2. Row-broadcast binary ops
//===----------------------------------------------------------------------===//

// The row_expand family: `tile OP row_vec` where the [rows, 1] row vector is
// broadcast across the tile's columns by the runtime (PTOAS-native row
// broadcast, valid semantics included). The verifier mirrors the runtime's
// DeduceTileRowExpandType contract: both operands carry at least two
// dimensions, the row vector's last dimension is one, and the row counts
// agree.
class Pypto_RowExpandOp<string mnemonic> : Pypto_Op<mnemonic> {
  let arguments = (ins AnyType:$tile, AnyType:$rowVec);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$tile `,` $rowVec attr-dict `:` type($tile) `,` type($rowVec) `to` type($result)";
  let hasVerifier = 1;
}

def Pypto_RowExpandAddOp : Pypto_RowExpandOp<"row_expand_add"> {
  let summary = "Row-broadcast addition (`pl.row_expand_add`)";
}
def Pypto_RowExpandSubOp : Pypto_RowExpandOp<"row_expand_sub"> {
  let summary = "Row-broadcast subtraction (`pl.row_expand_sub`)";
}
def Pypto_RowExpandMulOp : Pypto_RowExpandOp<"row_expand_mul"> {
  let summary = "Row-broadcast multiplication (`pl.row_expand_mul`)";
}
def Pypto_RowExpandDivOp : Pypto_RowExpandOp<"row_expand_div"> {
  let summary = "Row-broadcast division (`pl.row_expand_div`)";
}

//===----------------------------------------------------------------------===//
// E. Elementwise unary ops
//===----------------------------------------------------------------------===//

class Pypto_UnaryOp<string mnemonic> : Pypto_Op<mnemonic> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($result)";
  let hasVerifier = 1;
}

def Pypto_NegOp : Pypto_UnaryOp<"neg"> { let summary = "Elementwise negation (`pl.neg`)"; }
def Pypto_AbsOp : Pypto_UnaryOp<"abs"> { let summary = "Elementwise absolute value (`pl.abs`)"; }
def Pypto_ExpOp : Pypto_UnaryOp<"exp"> { let summary = "Elementwise exponential (`pl.exp`)"; }
def Pypto_LogOp : Pypto_UnaryOp<"log"> { let summary = "Elementwise natural logarithm (`pl.log`)"; }
def Pypto_SqrtOp : Pypto_UnaryOp<"sqrt"> { let summary = "Elementwise square root (`pl.sqrt`)"; }
def Pypto_RsqrtOp : Pypto_UnaryOp<"rsqrt"> { let summary = "Elementwise reciprocal square root (`pl.rsqrt`)"; }
def Pypto_SinOp : Pypto_UnaryOp<"sin"> { let summary = "Elementwise sine (`pl.sin`)"; }
def Pypto_CosOp : Pypto_UnaryOp<"cos"> { let summary = "Elementwise cosine (`pl.cos`)"; }

//===----------------------------------------------------------------------===//
// F. Scalar combines
//===----------------------------------------------------------------------===//

def Pypto_MaxOp : Pypto_BinaryOp<"max"> { let summary = "Scalar maximum (`pl.max`); the tile form is `pypto.maximum`"; }
def Pypto_MinOp : Pypto_BinaryOp<"min"> { let summary = "Scalar minimum (`pl.min`); the tile form is `pypto.minimum`"; }

//===----------------------------------------------------------------------===//
// G. Reduction and scalar IO
//===----------------------------------------------------------------------===//

def Pypto_ReductionOp : Pypto_Op<"reduction"> {
  let summary = "Reduce the first or last axis of a tile in place";
  let description =
    [{
    Syntax:

    ```
    %v = pypto.reduction sum, row, %vector, %scratch : tile-type to result-type
    ```

    `axis = row` folds the last axis onto the `pl.row_*` family, `col` the
    first axis onto `pl.col_*`. The reduction is keepdim: the folded axis
    reduces to 1 and every other dim is preserved. `scratch` is an
    uninitialized tile of the input shape the runtime works in. Full
    reductions decompose at conversion time into this op plus scalar reads
    and combines, so the op always carries both the scratch tile and the
    axis.

    The emitpy lowering emits `pl.row_<kind>(vector, scratch)` /
    `pl.col_<kind>(vector, scratch)` with both operands positional.
  }];
  let arguments = (ins Pypto_ReduceKindAttr:$kind, Pypto_ReduceAxisAttr:$axis, AnyType:$vector, AnyType:$scratch);
  let results = (outs AnyType:$result);
  let assemblyFormat =
    [{
    $kind `,` $axis `,` $vector `,` $scratch attr-dict `:`
    type($vector) `,` type($scratch) `->` type($result)
  }];
  let hasVerifier = 1;
}

def Pypto_ReadOp : Pypto_Op<"read"> {
  let summary = "Read one element out of a tile";
  let description =
    [{
    Syntax:

    ```
    %s = pypto.read %src[0, 1] : tile-type to scalar-type
    ```

    `indices` are constant coordinates into the tile.

    The emitpy lowering emits `pl.read(src, [indices])`.
  }];
  let arguments = (ins AnyType:$source, DenseI64ArrayAttr:$indices);
  let results = (outs AnyType:$result);
  let assemblyFormat =
    [{
    $source $indices attr-dict `:` type($source) `->` type($result)
  }];
  let hasVerifier = 1;
}

def Pypto_WriteOp : Pypto_Op<"write"> {
  let summary = "Write one scalar element into a tile";
  let description =
    [{
    Syntax:

    ```
    pypto.write %dest[0, 1], %value : tile-type, scalar-type
    ```

    The emitpy lowering emits `pl.write(dest, [offsets], value)`.
  }];
  let arguments = (ins AnyType:$dest, AnyType:$value, Variadic<Index>:$offsets);
  let assemblyFormat =
    [{
    $dest `[` $offsets `]` `,` $value attr-dict `:` type($dest) `,` type($value)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Tile movement between tensor memory and cores
//===----------------------------------------------------------------------===//

def Pypto_LoadOp : Pypto_Op<"load", [AttrSizedOperandSegments]> {
  let summary = "Load a tile from a PyPTO tensor at the given offsets";
  let description =
    [{
    Syntax:

    ```
    %tile = pypto.load %src[%offsets...] : src-type -> tile-type
    %dyn  = pypto.load %src[%o...] padded [%p...] valid [%v...]
              : src-type -> tile-type
    ```

    Reads a `!pypto.tile` from the source buffer starting at the given
    element offsets. The number of offsets must equal the source's rank:
    a rank-reduced load (source rank > tile rank, as produced by
    `vector.transfer_read` lowering) drops leading source dimensions that
    stay addressed by the offsets alone. The source is a PyPTO tensor-typed
    value (in practice an `!emitpy.opaque` carrying the `pl.Tensor[[...],
    dtype]` annotation).

    With empty `paddedShapes`/`validExtents` the tile is statically shaped
    and the emitpy lowering emits `pl.load(src, [offsets], [shape])`.
    The dynamic form passes the per-dimension 32-byte aligned allocation
    bounds (`paddedShapes`) and the actual extents (`validExtents`) and
    emits `pl.load(src, [offsets], [shapes], [valid])`.
  }];
  let arguments = (ins AnyType:$source, Variadic<Index>:$offsets, Variadic<Index>:$paddedShapes,
    Variadic<Index>:$validExtents);
  let results = (outs AnyType:$result);
  let assemblyFormat =
    [{
    $source `[` $offsets `]`
    (`padded` `[` $paddedShapes^ `]` `valid` `[` $validExtents `]`)?
    attr-dict `:` type($source) `->` type($result)
  }];
  let hasVerifier = 1;
}

def Pypto_StoreOp : Pypto_Op<"store"> {
  let summary = "Store a tile back to a PyPTO tensor at the given offsets";
  let description =
    [{
    Syntax:

    ```
    pypto.store %tile, %dest[%offsets...] : tile-type, dest-type
    ```

    Writes a `!pypto.tile` back to the destination buffer starting at the
    given element offsets. The number of offsets must equal the
    destination's rank (see `pypto.load` on rank-reduced accesses).

    The emitpy lowering emits `pl.store(tile, [offsets], dest)`.
  }];
  let arguments = (ins AnyType:$value, AnyType:$dest, Variadic<Index>:$offsets);
  let assemblyFormat =
    [{
    $value `,` $dest `[` $offsets `]` attr-dict `:` type($value) `,` type($dest)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// H. SPMD and loop structure
//
// The two structured ops below let the pypto layer carry the loop/SPMD
// skeleton end to end: `-convert-vector-to-pypto` lowers `scf.forall`
// onto `pypto.spmd` and `scf.for` onto `pypto.range`, so the intermediate
// IR is pure pypto. The emitpy lowering then maps them one-to-one onto
// `emitpy.spmd_scope`/`emitpy.for` with the bodies inlined unchanged.
//===----------------------------------------------------------------------===//

def Pypto_SpmdOp : Pypto_Op<"spmd", [SingleBlock, NoTerminator]> {
  let summary = "SPMD multi-core launch scope";

  let description =
    [{
    Syntax (mirrors `scf.forall`'s normalized spelling; the block index is
    bound inline instead of through an explicit `^bb0` header):

    ```
    pypto.spmd (%blk) in (48) {
      ...
    }
    ```

    The body executes once per core of a `core_num`-core grid; the single
    body block argument (index type) is the executing core's block index.
    `core_num` is always static: the vector-to-pypto lowering folds the
    forall bound (an IntegerAttr or a constant value) into this attribute.

    The emitpy lowering maps it onto `emitpy.spmd_scope` verbatim.
  }];

  let arguments = (ins I64Attr:$core_num);

  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
}

def Pypto_RangeOp : Pypto_Op<"range", [SingleBlock, NoTerminator]> {
  let summary = "Counted per-core loop with an induction variable";

  let description =
    [{
    Syntax (mirrors `scf.for`'s spelling; the induction variable is bound
    inline instead of through an explicit `^bb0` header):

    ```
    pypto.range %i = %lb to %ub step %step {
      ...
    }
    ```

    The three index operands are the lower bound, upper bound and step;
    the single body block argument (index type) is the induction variable.
    Loop-carried values are not represented: the lowering accepts only
    `scf.for` loops without iter_args and with an empty yield, so all
    results flow through tile stores.

    The emitpy lowering maps it onto `emitpy.for`, which prints as
    `for <iv> in pl.range(<lb>, <ub>, <step>)`.
  }];

  let arguments = (ins Index:$lower_bound, Index:$upper_bound, Index:$step);

  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
}

#endif // AKG_DIALECT_PYPTO_IR_PYPTOOPS_TD
