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

#ifndef AKG_DIALECT_MATH_IR_MATH_EXT_OPS
#define AKG_DIALECT_MATH_IR_MATH_EXT_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Dialect/Math/IR/MathBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/Arith/IR/ArithOpsInterfaces.td"

include "akg/Dialect/Math/IR/MathExtBase.td"
include "akg/Dialect/Math/IR/MathExtInterfaces.td"

//===----------------------------------------------------------------------===//
// MATH Ext Operator.
//===----------------------------------------------------------------------===//
// class MathExt_Op for the operation in this dialect
class MathExt_Op<string mnemonic, list<Trait> traits = []> :
    Op<MathExt_Dialect, mnemonic, traits # [Pure,
    DeclareOpInterfaceMethods<VectorUnrollOpInterface>] #
    ElementwiseMappable.traits>;

// Base class for unary math operations on floating point types. Require an
// operand and result of the same type. This type can be a floating point type,
// vector or tensor thereof.
class MathExt_FloatUnaryOp<string mnemonic, list<Trait> traits = []> :
    MathExt_Op<mnemonic,
        traits # [SameOperandsAndResultType,
                  DeclareOpInterfaceMethods<ArithFastMathInterface>]> {
  let arguments = (ins FloatLike:$operand,
      DefaultValuedAttr<Arith_FastMathAttr,
                        "::mlir::arith::FastMathFlags::none">:$fastmath);
  let results = (outs FloatLike:$result);

  let assemblyFormat = [{ $operand (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($result) }];
}

class MathExt_IntegerUnaryOp<string mnemonic, list<Trait> traits = []> :
     MathExt_Op<mnemonic,
      !listconcat([DeclareOpInterfaceMethods<ArithFastMathInterface>, Pure,
        TypesMatchWith<"result type has i1 element type and same shape as operands",
        "operand", "result", "::getI1SameShape($_self)">], traits)>,
    Arguments<(ins FloatLike:$operand,
      DefaultValuedAttr<
        Arith_FastMathAttr, "::mlir::arith::FastMathFlags::none">:$fastmath)>,
    Results<(outs BoolLike:$result)> {
  let assemblyFormat = [{ $operand (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($operand) }];
}

//===----------------------------------------------------------------------===//
// Operator: AcosOp
//===----------------------------------------------------------------------===//
def MathExt_AcosOp : MathExt_FloatUnaryOp<"acos"> {
  let summary = "arccos of the specified value";

  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `mathExt.acos` ssa-use `:` type
    ```

    The `acos` operation computes the arccosine of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    ```mlir
    // Scalar arccosine value.
    %a = mathExt.acos %b : f64
    ```
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator: AsinOp
//===----------------------------------------------------------------------===//
def MathExt_AsinOp : MathExt_FloatUnaryOp<"asin"> {
  let summary = "arcsine of the specified value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `mathExt.asin` ssa-use `:` type
    ```

    The `asin` operation computes the arcsine of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    ```mlir
    // Scalar arcsine value.
    %a = mathExt.asin %b : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator: IsinfOp
//===----------------------------------------------------------------------===//
def MathExt_IsinfOp : MathExt_IntegerUnaryOp<"isinf"> {
  let summary = "Returns true for infinite float inputs";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `mathExt.isinf` ssa-use `:` type
    ```
  }];
  //let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator: IsnanOp
//===----------------------------------------------------------------------===//
def MathExt_IsnanOp : MathExt_IntegerUnaryOp<"isnan"> {
  let summary = "Returns true for IEEE NaN inputs";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `mathExt.isnan` ssa-use `:` type
    ```
  }];
  //let hasFolder = 1;
}

#endif // AKG_DIALECT_MATH_IR_MATH_EXT_OPS