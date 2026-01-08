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

#ifndef MFUSION_DIALECT_MUSE_MUSESYMBOLICOPS
#define MFUSION_DIALECT_MUSE_MUSESYMBOLICOPS

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mfusion/Dialect/Muse/MuseDialect.td"
include "mfusion/Dialect/Muse/MuseTypes.td"

def Muse_SymbolicIntOp : Muse_Op<"symbolic_int", [Pure]> {
  let summary = "Symbolic int representing a dynamic dimension";
  let description = [{
    The `muse.symbolic_int` is a one-to-one mapping op of `torch.symbolic_int`.

    Example:
    ```
    %0 = muse.symbolic_int "s0" {min_val = 5, max_val = 10} : !muse.164
    %1 = muse.symbolic_int "s1" {min_val = 2, max_val = 20} : !muse.164
    ```
  }];
  let arguments = (ins
    StrAttr:$symbol_name,
    I64Attr:$min_val,
    I64Attr:$max_val
  );
  let results = (outs
    Muse_I64Type:$result
  );
  let assemblyFormat = [{
    $symbol_name ` ` `{` `min_val` `=` $min_val `,` `max_val` `=` $max_val `}` attr-dict `:` type($result)
  }];
}

def Muse_BindSymbolicShapeOp : Muse_Op<"bind_symbolic_shape"> {
  let summary = "Binds shape expressions to tensors using an affine map indexed by shape symbols";
  let description = [{
    The `muse.bind_symbolic_shape` is a one-to-one mapping op of `torch.bind_symbolic_shape`.

    Example:
    ```
    muse.bind_symbolic_shape %arg0, [%0, %1], affine_map<()[s0, s1] -> (s0, s1, 3)> : !muse.tensor<?x?x3xf32>
    muse.bind_symbolic_shape %out0, [%0, %1, %2], affine_map<()[s0, s1, s2] -> (s0, s1 * 2 + s2, 3)> : !muse.tensor<?x?x3xf32>
    ```
  }];
  let arguments = (ins
    MuseAnyTensor:$operand,
    Variadic<Muse_I64Type>:$shape_symbols,
    Builtin_AffineMapAttr:$shape_expressions
  );
  let results = (outs);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Muse_EvalSymbolicExprOp : Muse_Op<"eval_symbolic_expr", [Pure]> {
  let summary = "Evaluates a symbolic integer expression defined by an affine map";
  let description = [{
    The `muse.eval_symbolic_expr` operation evaluates an integer expression defined
    by an affine map applied to its input operands. This can be used to fuse
    sequences of integer arithmetic operations into a single symbolic expression.

    Example:
    ```
    // Computes: ((a + b) - c) floordiv d
    %0 = muse.eval_symbolic_expr [%a, %b, %c, %d], affine_map<()[s0, s1, s2, s3] -> ((s0 + s1) - s2) floordiv s3> : !muse.i64
    ```
  }];
  let arguments = (ins
    Variadic<Muse_I64Type>:$operands,
    Builtin_AffineMapAttr:$expr
  );
  let results = (outs
    Muse_I64Type:$result
  );
  let assemblyFormat = [{
    ` ` `[` $operands `]` `,` $expr attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
}

#endif // MFUSION_DIALECT_MUSE_MUSESYMBOLICOPS
