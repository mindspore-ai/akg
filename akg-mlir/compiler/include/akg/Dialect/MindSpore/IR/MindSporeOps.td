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

#ifndef AKG_DIALECT_MINDSPORE_IR_MINDSPORE_OPS
#define AKG_DIALECT_MINDSPORE_IR_MINDSPORE_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"

include "akg/Dialect/MindSpore/IR/MindSporeBase.td"
include "akg/Dialect/MindSpore/IR/MindSporeTypesBase.td"
include "akg/Dialect/MindSpore/IR/MindSporeInterfaces.td"


//===----------------------------------------------------------------------===//
// MindSpore Operator.
//===----------------------------------------------------------------------===//
// class MindSpore_Op for the operation in this dialect
class MindSpore_Op<string mnemonic, list<Trait> traits = []> :
    Op<MindSpore_Dialect, mnemonic, !listconcat(traits, [MindSporeOpInterface])> {
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: const
//===----------------------------------------------------------------------===//
def MindSpore_ConstOp : MindSpore_Op<"const", [Pure]>  {
  let summary = "MindSpore Constant op.";

  let description = [{
    MindSpore Constant op.
  }];

  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore  Operator: cast
//===----------------------------------------------------------------------===//
def MindSpore_CastOp: MindSpore_Op<"cast", [Pure]> {

  let summary = "Cast operation";

  let description = [{
    Perform a set of permissible cast operations
        Mode                    Input   Output
        ---------------------------------------
        signed 8 to bool        int8    Boolean
        signed 16 to bool       int16   Boolean
        signed 32 to bool       int32   Boolean
        bool to 8               Boolean int8
        bool to 16              Boolean int16
        bool to 32              Boolean int32
        signed 8 to signed 16   int8    int16
        signed 8 to signed 32   int8    int32
        signed 16 to signed 8   int16   int8
        signed 16 to signed 32  int16   int32
        signed 32 to signed 8   int32   int8
        signed 32 to signed 16  int32   int16
        float to signed 8       float   int8
        float to signed 16      float   int16
        signed 8 to float       int8    float
        signed 16 to float      int16   float
  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: add
//===----------------------------------------------------------------------===//
def MindSpore_AddOp : MindSpore_Op<"add", [Pure]> {
  let summary = "Generated op for mindspore.ops.add(input, other)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.add.html

    Add other value to input Tensor.

  }];

  let arguments = (ins
      MindSpore_Tensor:$input1,
      MindSpore_Tensor:$input2
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: sub
//===----------------------------------------------------------------------===//
def MindSpore_SubOp : MindSpore_Op<"sub", [Pure]> {
  let summary = "Generated op for mindspore.ops.sub(input, other)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.sub.html

    Subtract the second input tensor from the first input tensor element-wise.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input1,
    MindSpore_Tensor:$input2
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: mul
//===----------------------------------------------------------------------===//
def MindSpore_MulOp : MindSpore_Op<"mul", [Pure]> {
  let summary = "Generated op for mindspore.ops.mul(input, other)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.mul.html

    Multiplie two tensors element-wise.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input1,
    MindSpore_Tensor:$input2
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: div
//===----------------------------------------------------------------------===//
def MindSpore_DivOp : MindSpore_Op<"div", [Pure]> {
  let summary = "Generated op for mindspore.ops.div(input, other, *, rounding_mode=None)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.div.html

    Divide the first input tensor by the second input tensor in int/floating-point
    type element-wise.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input1,
    MindSpore_Tensor:$input2
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: floordiv
//===----------------------------------------------------------------------===//
def MindSpore_FloorDivOp : MindSpore_Op<"floordiv", [Pure]> {
  let summary = "Generated op for mindspore.ops.floordiv(input, other, *, rounding_mode=None)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.floor_divide.html

    Divides the first input tensor by the second input tensor element-wise and round down to the closest integer.

    Inputs of input and other comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar. When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar, the scalar could only be a constant.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input1,
    MindSpore_Tensor:$input2
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: floormod
//===----------------------------------------------------------------------===//
def MindSpore_FloorModOp : MindSpore_Op<"floormod", [Pure]> {
  let summary = "Generated op for mindspore.ops.floormod(input, other, *, rounding_mode=None)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.floor_mod.html

    Computes the remainder of division element-wise. It iss a flooring divide. E.g. floor(x/y) * y+mod(x,y)=x.

    Inputs of x and y comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, dtypes of them cannot be both bool, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar, the scalar could only be a constant.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input1,
    MindSpore_Tensor:$input2
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: pow
//===----------------------------------------------------------------------===//
def MindSpore_PowOp : MindSpore_Op<"pow", [Pure]> {
  let summary = "Generated op for mindspore.ops.pow(input, exponent)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.pow.html

    Calculate the exponent power of each element in input.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input1,
    MindSpore_Tensor:$input2
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: exp
//===----------------------------------------------------------------------===//
def MindSpore_ExpOp : MindSpore_Op<"exp", [Pure]> {
  let summary = "Generated op for mindspore.ops.exp(input)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.exp.html

    Return exponential of a tensor element-wise.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: expm1
//===----------------------------------------------------------------------===//
def MindSpore_ExpM1Op : MindSpore_Op<"expm1", [Pure]> {
  let summary = "Generated op for mindspore.ops.expm1(input)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.expm1.html

    Returns exponential then minus 1 of a tensor element-wise.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: sign
//===----------------------------------------------------------------------===//
def MindSpore_SignOp : MindSpore_Op<"sign", [Pure]> {
  let summary = "Generated op for mindspore.ops.sign(input)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.sign.html

    Performs sign on the tensor element-wise.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: addN
//===----------------------------------------------------------------------===//

def MindSpore_AddNOp : MindSpore_Op<"addn", [Pure, SameOperandsAndResultElementType]> {
  let summary = "Generated op for mindspore.ops.addn(x)";

  let description = [{
    https://www.mindspore.cn/docs/en/r2.0/api_python/ops/mindspore.ops.addn.html
    Compute addition of all input tensors element-wise.All input tensors
    must have the same shape.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input1,
    MindSpore_Tensor:$input2,
    Variadic<MindSpore_Tensor>:$others
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: assign
//===----------------------------------------------------------------------===//
def MindSpore_AssignOp: MindSpore_Op<"assign", [Pure]> {
  let summary = "Generated op for mindspore.ops.assign(variable, value)";

  let description = [{
    https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.assign.html

    Assign Parameter with a value.
    Args of variable and value comply with the implicit type conversion rules to
    make the data types consistent. If they have different data types, the lower
    priority data type will be converted to the relatively highest priority data
    type.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input0,
    MindSpore_Tensor:$input1
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}


//===----------------------------------------------------------------------===//
// MindSpore Operator: inplaceAssign
//===----------------------------------------------------------------------===//
def MindSpore_InplaceAssignOp: MindSpore_Op<"inplace_assign", [Pure]> {
  let summary = "inplace Assign operator";

  let description = [{
    Assign the value of 2nd input to 1st input.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input0,
    MindSpore_Tensor:$input1
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: concat
//===----------------------------------------------------------------------===//
def MindSpore_ConcatOp : MindSpore_Op<"concat", [Pure]> {
  let summary = "Generated op for mindspore.ops.cat(tensors, axis=0)";

  let description = [{
    Concatenate a variadic amount of input tensors along with the given axis.
  }];

  let arguments = (ins
    Variadic<MindSpore_Tensor>:$input,
    OptionalAttr<I64Attr>:$axis
  );

  let builders = [
    OpBuilder<(ins "mlir::Type":$output, "Value":$input)>,
  ];

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: sin
//===----------------------------------------------------------------------===//
def MindSpore_SinOp : MindSpore_Op<"sin", [Pure]> {
  let summary = "Generated op for mindspore.ops.sin(input)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.sin.html

    Compute sine of the input element-wise.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: cos
//===----------------------------------------------------------------------===//
def MindSpore_CosOp : MindSpore_Op<"cos", [Pure]> {
  let summary = "Generated op for mindspore.ops.cos(input)";
  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.cos.html

    Compute cosine of the input element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: asin
//===----------------------------------------------------------------------===//
def MindSpore_AsinOp : MindSpore_Op<"asin", [Pure]> {
  let summary = "Generated op for mindspore.ops.asin(input)";
  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.asin.html

    Compute arcsine of input tensors element-wise.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: acos
//===----------------------------------------------------------------------===//
def MindSpore_AcosOp : MindSpore_Op<"acos", [Pure]> {
  let summary = "Generated op for mindspore.ops.acos(input)";
  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.cos.html

    Compute arccosine of input tensors element-wise.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: tanh
//===----------------------------------------------------------------------===//
def MindSpore_TanhOp : MindSpore_Op<"tanh", [Pure]> {
  let summary = "Generated op for mindspore.ops.tanh(input)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.tanh.html

    Compute hyperbolic tangent of input element-wise.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: asinh
//===----------------------------------------------------------------------===//
def MindSpore_AsinhOp : MindSpore_Op<"asinh", [Pure]> {
  let summary = "Generated op for mindspore.ops.asinh(input)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.asinh.html

    Compute inverse hyperbolic sine of the input element-wise.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: acosh
//===----------------------------------------------------------------------===//
def MindSpore_AcoshOp : MindSpore_Op<"acosh", [Pure]> {
  let summary = "Generated op for mindspore.ops.acosh(input)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.acosh.html

    Compute inverse hyperbolic cosine of the inputs element-wise.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: atan
//===----------------------------------------------------------------------===//
def MindSpore_AtanOp : MindSpore_Op<"atan", [Pure]> {
  let summary = "Generated op for mindspore.ops.atan(input)";
  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.atan.html

    Compute the trigonometric inverse tangent of the input element-wise.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: atan2
//===----------------------------------------------------------------------===//
def MindSpore_Atan2Op : MindSpore_Op<"atan2", [Pure]> {
  let summary = "Generated op for mindspore.ops.atan2(input, other)";
  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.atan2.html

    Return arctangent of input/other element-wise.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    MindSpore_Tensor:$other
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: erf
//===----------------------------------------------------------------------===//
def MindSpore_ErfOp : MindSpore_Op<"erf", [Pure]> {
  let summary = "Generated op for mindspore.ops.erf(input)";
  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.erf.html

    Computes the Gauss error function of input element-wise.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: erfc
//===----------------------------------------------------------------------===//
def MindSpore_ErfcOp : MindSpore_Op<"erfc", [Pure]> {
  let summary = "Generated op for mindspore.ops.erfc(input)";
  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.erfc.html

    Computes the complementary error function of input element-wise.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: erfinv
//===----------------------------------------------------------------------===//
def MindSpore_ErfinvOp : MindSpore_Op<"erfinv", [Pure]> {
  let summary = "Generated op for mindspore.ops.erfinv(input)";
  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.erfinv.html

    Returns the result of the inverse error function with input, which is defined in the range (-1, 1) as:
    erfinv(erf(x))=x where x is the input.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: relu
//===----------------------------------------------------------------------===//
def MindSpore_ReluOp : MindSpore_Op<"relu", [Pure]> {
  let summary = "Generated op for mindspore.ops.relu(input)";
  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.relu.html

    Computes ReLU (Rectified Linear Unit activation function) of input tensors element-wise.

    It returns max(input, 0) element-wise.
    Specially, the neurons with the negative output will be suppressed and the active neurons will stay the same.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: sigmoid
//===----------------------------------------------------------------------===//
def MindSpore_SigmoidOp : MindSpore_Op<"sigmoid", [Pure]> {
  let summary = "Generated op for mindspore.ops.sigmoid(input)";
  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.sigmoid.html

    Computes Sigmoid of input element-wise. The Sigmoid function is defined as:
    sigmoid(input_i) = 1 / (1+exp( -input_i))
    where input_i is an element of the input.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: gelu
//===----------------------------------------------------------------------===//
def MindSpore_GeluOp : MindSpore_Op<"gelu", [Pure]> {
  let summary = "Generated op for mindspore.ops.gelu(input, approximate)";
  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.gelu.html

    Gaussian Error Linear Units activation function.

    GeLU is described in the paper Gaussian Error Linear Units (GELUs).
    And also please refer to BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.

    When approximate argument is none, GeLU is defined as : GELU(xi)=xi * P(X<xi),
    where P is the cumulative distribution function of the standard Gaussian distribution,
    xi is the input element.

    When approximate argument is tanh, GeLU is estimated with:

    GELU(xi)=0.5 * xi * (1+tanh((√2/π) * (xi+0.044715 * xi ^ 3)))

  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    OptionalAttr<StrAttr>:$approximate
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: tile
//===----------------------------------------------------------------------===//
def MindSpore_TileOp: MindSpore_Op<"tile", [Pure]> {
  let summary = "Generated op for mindspore.ops.tile(input, multiples)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.tile.html

    Replicate an input tensor with given multiples times.
    Creates a new tensor by replicating input multiples times. The i’th dimension
    of output tensor has input.shape[i] * multiples[i] elements, and the values of
    input are replicated multiples[i] times along the i’th dimension.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    DenseI64ArrayAttr:$multiples
  );

  let extraClassDeclaration = [{
    uint64_t getNewRankSize() {
      return getMultiples().size();
    }
  }];

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: equal
//===----------------------------------------------------------------------===//

def MindSpore_EqualOp : MindSpore_Op<"equal", [Pure]> {
  let summary = "Generated op for mindspore.ops.equal(input, other)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.equal.html

    Compute the boolean value of input=other element-wise.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input1,
    MindSpore_Tensor:$input2
  );

  let results = (outs
    I1Tensor:$output
  );
}
//===----------------------------------------------------------------------===//
// MindSpore Operator: greater
//===----------------------------------------------------------------------===//

def MindSpore_GreaterOp : MindSpore_Op<"greater", [Pure]> {
  let summary = "Generated op for mindspore.ops.greater(input, other)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.greater.html

    Compute the boolean value of input>other element-wise.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input1,
    MindSpore_Tensor:$input2
  );

  let results = (outs
    I1Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: greater_equal
//===----------------------------------------------------------------------===//

def MindSpore_GreaterEqualOp : MindSpore_Op<"greater_equal", [Pure]> {
  let summary = "Generated op for mindspore.ops.greater_equal(input, other)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.greater_equal.html

    Compute the boolean value of input≥other element-wise.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input1,
    MindSpore_Tensor:$input2
  );

  let results = (outs
    I1Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: less
//===----------------------------------------------------------------------===//
def MindSpore_LessOp : MindSpore_Op<"less", [Pure]> {
  let summary = "Generated op for mindspore.ops.less(x, y)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.less.html
    Computes the boolean value of x<y element-wise.Inputs of x and y comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar. 
    When the inputs are one tensor and one scalar, the scalar could only be a constant.
  }];

  let arguments = (ins
    MindSpore_Tensor:$x,
    MindSpore_Tensor:$y
  );

  let extraClassDeclaration = [{
    Value getInput1() {return getX();}
    Value getInput2() {return getY();}
  }];

  let results = (outs
    I1Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: less_equal
//===----------------------------------------------------------------------===//
def MindSpore_LessEqualOp : MindSpore_Op<"less_equal", [Pure]> {
  let summary = "Generated op for mindspore.ops.less_equal(input, other).";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.less_equal.html

    Computes the boolean value of input<=other element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    MindSpore_Tensor:$other
  );

  let extraClassDeclaration = [{
    Value getInput1() {return getInput();}
    Value getInput2() {return getOther();}
  }];

  let results = (outs
    I1Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: logical_and
//===----------------------------------------------------------------------===//
def MindSpore_LogicalAndOp : MindSpore_Op<"logical_and", [Pure]> {
  let summary = "Generated op for mindspore.ops.logical_and(input, other)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.logical_and.html

    Compute the “logical AND” of two tensors element-wise.

  }];

  let arguments = (ins
    I1Tensor:$input1,
    I1Tensor:$input2
  );

  let results = (outs
    I1Tensor:$z
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: logical_or
//===----------------------------------------------------------------------===//
def MindSpore_LogicalOrOp : MindSpore_Op<"logical_or", [Pure]> {
  let summary = "Generated op for mindspore.ops.logical_or(input, other)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.logical_or.html

    Computes the “logical OR” of two tensors element-wise.

  }];

  let arguments = (ins
    I1Tensor:$input1,
    I1Tensor:$input2
  );

  let results = (outs
    I1Tensor:$z
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: logical_not
//===----------------------------------------------------------------------===//
def MindSpore_LogicalNotOp : MindSpore_Op<"logical_not", [Pure]> {
  let summary = "Generated op for mindspore.ops.logical_not(input, other)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.logical_not.html

    Computes the “logical NOT” of a tensor element-wise.

  }];

  let arguments = (ins
    I1Tensor:$input
  );

  let results = (outs
    I1Tensor:$z
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: negate
//===----------------------------------------------------------------------===//
def MindSpore_NegateOp : MindSpore_Op<"neg", [Pure]> {
  let summary = "Generated op for mindspore.ops.neg(input)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.neg.html

    Returns a tensor with negative values of the input tensor element-wise.

  }];

  let arguments = (ins
      MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: reduce_max
//===----------------------------------------------------------------------===//
def MindSpore_ReduceMaxOp : MindSpore_Op<"reduce_max", [Pure]> {
  let summary = "Generated op for mindspore.ops.amax(input, axis=None, keepdims=False, *, initial=None, where=None)";

  let description = [{
    mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.amax.html

    Reduces all dimensions of a tensor by returning the maximum value in input,
    by default. And also can reduce a dimension of input along specified axis.
    keepdims determines whether the dimensions of output and input are the same.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    DenseI64ArrayAttr:$axis,
    OptionalAttr<BoolAttr>:$keepdims
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: reduce_min
//===----------------------------------------------------------------------===//
def MindSpore_ReduceMinOp : MindSpore_Op<"reduce_min", [Pure]> {
  let summary = "Generated op for mindspore.ops.amin(input, axis=None, keepdims=False, *, initial=None, where=None)";

  let description = [{
    mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.amin.html

    Reduces all dimensions of a tensor by returning the minimum value in input,
    by default. And also can reduce a dimension of input along specified axis.
    keepdims determines whether the dimensions of output and input are the same.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    DenseI64ArrayAttr:$axis,
    OptionalAttr<BoolAttr>:$keepdims
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: reduce_any
//===----------------------------------------------------------------------===//
def MindSpore_ReduceAnyOp : MindSpore_Op<"reduce_any", [Pure]> {
  let summary = "Generated op for mindspore.ops.any(input, axis=None, keep_dims=False)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.any.html

    Reduce a dimension of input by the “logical OR” of all elements in the dimension,
    by default. And also can reduce a dimension of input along the axis.
    Determine whether the dimensions of the output and input are the same by
    controlling keep_dims.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    DenseI64ArrayAttr:$axis,
    OptionalAttr<BoolAttr>:$keepdims
  );

  let results = (outs
    MindSpore_Tensor1Dto4D:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: reduce_sum
//===----------------------------------------------------------------------===//
def MindSpore_ReduceSumOp : MindSpore_Op<"reduce_sum", [Pure]> {
  let summary = "mindspore.ops.sum(input, dim=None, keepdim=False, *, dtype=None)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.sum.html

    Calculate sum of Tensor elements over a given dim.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    DenseI64ArrayAttr:$axis,
    OptionalAttr<BoolAttr>:$keepdims
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: argmax
//===----------------------------------------------------------------------===//
def MindSpore_ArgMaxOp : MindSpore_Op<"argmax", [Pure]> {
  let summary = "mindspore.ops.argmax(input, dim=None, keepdim=False)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.argmax.html

    Return the indices of the maximum values of a tensor across a dimension.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    DenseI64ArrayAttr:$axis,
    OptionalAttr<BoolAttr>:$keepdims
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: reduce_prod
//===----------------------------------------------------------------------===//
def MindSpore_ReduceProdOp : MindSpore_Op<"reduce_prod", [Pure]> {
  let summary = "mindspore.ops.cumprod(input, dim=None, keepdim=False)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.cumprod.htm

    Computes the cumulative product of the input tensor along dimension dim.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    DenseI64ArrayAttr:$axis,
    OptionalAttr<BoolAttr>:$keepdims
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: not_equal
//===----------------------------------------------------------------------===//
def MindSpore_NotEqualOp : MindSpore_Op<"not_equal", [Pure]> {
  let summary = "Generated op for mindspore.ops.not_equal(input, other).";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.not_equal.html

    Computes the non-equivalence of two tensors element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$x,
    MindSpore_Tensor:$y
  );

  let extraClassDeclaration = [{
    Value getInput1() {return getX();}
    Value getInput2() {return getY();}
  }];

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: reduce_all
//===----------------------------------------------------------------------===//
def MindSpore_ReduceAllOp : MindSpore_Op<"reduce_all", [Pure]> {
  let summary = "mindspore.ops.all(input, axis=None, keep_dims=False)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.all.html

    Reduces a dimension of input by the “logical AND” of all elements in the dimension,
    by default. And also can reduce a dimension of input along the axis.
    Determine whether the dimensions of the output and input are the same by
    controlling keep_dims.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    DenseI64ArrayAttr:$axis,
    OptionalAttr<BoolAttr>:$keepdims
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: unsorted_segment_sum
//===----------------------------------------------------------------------===//
def MindSpore_UnsortedSegmentSumOp : MindSpore_Op<"unsorted_segment_sum", [Pure]> {
  let summary = "Generated op for mindspore.ops.unsorted_segment_sum(x, segment_ids, num_segments)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.unsorted_segment_sum.html

    Compute the sum of a tensor along segments.

  }];

  let arguments = (ins
    MindSpore_Tensor:$x,
    MindSpore_Tensor:$segment_ids,
    OptionalAttr<AnyAttrOf<[I64Attr, DenseI64ArrayAttr]>>:$num_segments
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: unsorted_segment_max
//===----------------------------------------------------------------------===//
def MindSpore_UnsortedSegmentMaxOp : MindSpore_Op<"unsorted_segment_max", [Pure]> {
  let summary = "Generated op for mindspore.ops.unsorted_segment_max(x, segment_ids, num_segments)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.unsorted_segment_max.html

    Compute the maximum along segments of a tensor.

  }];

  let arguments = (ins
    MindSpore_Tensor:$x,
    MindSpore_Tensor1D:$segment_ids,
    MindSpore_Int:$num_segments
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: unsorted_segment_min
//===----------------------------------------------------------------------===//
def MindSpore_UnsortedSegmentMinOp : MindSpore_Op<"unsorted_segment_min", [Pure]> {
  let summary = "Generated op for mindspore.ops.unsorted_segment_min(x, segment_ids, num_segments)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.unsorted_segment_min.html

    Compute the minimum along segments of a tensor.

  }];

  let arguments = (ins
    MindSpore_Tensor:$x,
    MindSpore_Tensor1D:$segment_ids,
    MindSpore_Int:$num_segments
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: unsorted_segment_prod
//===----------------------------------------------------------------------===//
def MindSpore_UnsortedSegmentProdOp : MindSpore_Op<"unsorted_segment_prod", [Pure]> {
  let summary = "Generated op for mindspore.ops.unsorted_segment_prod(x, segment_ids, num_segments)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.unsorted_segment_prod.html

    Compute the product of a tensor along segments.

  }];

  let arguments = (ins
    MindSpore_Tensor:$x,
    MindSpore_Tensor1D:$segment_ids,
    MindSpore_Int:$num_segments
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: square
//===----------------------------------------------------------------------===//
def MindSpore_SquareOp : MindSpore_Op<"square", [Pure]> {
  let summary = "Generated op for mindspore.ops.square(input)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.square.html

    Return square of a tensor element-wise.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let extraClassDeclaration = [{
    Value getInput1() {return getInput();}
    Value getInput2() {return getInput();}
  }];

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: transpose
//===----------------------------------------------------------------------===//
def MindSpore_TransposeOp : MindSpore_Op<"transpose", [Pure]> {
  let summary = "Generated op for mindspore.ops.transpose(input, input_perm)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.transpose.html

    Permute the dimensions based on perm.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    MindSpore_Int32Or64Tensor:$perms
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: expand_dims
//===----------------------------------------------------------------------===//
def MindSpore_Expand_DimsOp: MindSpore_Op<"expand_dims", [Pure]> {
  let summary = "Generated op for mindspore.ops.expand_dims(input_x, axis)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.expand_dims.html

    Add an additional dimension to input_x at the given axis.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input_x,
    MindSpore_SignedInt:$axis
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: select
//===----------------------------------------------------------------------===//
def MindSpore_SelectOp : MindSpore_Op<"select", [Pure]> {
  let summary = "Generated op for mindspore.ops.select(cond, x, y)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.select.html

    The conditional tensor determines whether the corresponding element in the
    output must be selected from x (if true) or y (if false) based on the value
    of each element.
  }];

  let arguments = (ins
    I1Tensor:$pred,
    MindSpore_Tensor:$on_true,
    MindSpore_Tensor:$on_false
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: reshape
//===----------------------------------------------------------------------===//
def MindSpore_ReshapeOp: MindSpore_Op<"reshape", [Pure]> {
  let summary = "Generated op for mindspore.ops.Reshape";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.reshape.html

    Rearrange the input Tensor based on the given shape.
    The ‘input_shape’ can only have one -1 at most, in which case it’s inferred
    from the remaining dimensions and the number of elements in the input.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    Optional<AnyTypeOf<[MindSpore_Tensor, MindSpore_Index1D]>>:$newShapeValue,
    OptionalAttr<DenseI64ArrayAttr>:$new_shape
  );

  let results = (outs
    MindSpore_RankedTensor:$output
  );

  let builders = [
    OpBuilder<(ins "mlir::Type":$output, "Value":$input, "Value":$new_shape)>,
    OpBuilder<(ins "mlir::Type":$output, "Value":$input, "DenseI64ArrayAttr":$new_shape)>,
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: broadcast_to
//===----------------------------------------------------------------------===//
def MindSpore_BroadcastToOp: MindSpore_Op<"broadcast_to", [Pure]> {
  let summary = "Generated op for mindspore.ops.broadcast_to(input, shape)";

  let description = [{
    https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.broadcast_to.html

    Broadcast a tensor into the given dynamic shape by adding dimensions and
    Replicating input0 x multiplies times along each dimension.
    For example:
      %0 = ...: tensor<?x?xf32>
      %d0 = tensor.dim %0, %c0 : tensor<?x?xf32>
      %d1 = tensor.dim %0, %c1 : tensor<?x?xf32>
      %new_shape = tensor.from_elements %c32, %d0, %d1 : tensor<3xindex>
      %2 = "mindspore.broadcast_to"(%0, %new_shape) : (tensor<?x?xf32>, tensor<3xindex>)
      -> tensor<?x?x?xf32>
    The first dimensions are newly added. Then, the remaining dimension of %0 is replicated
    specific multiples times and forming the last two dimensions of new shape.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    Optional<MindSpore_Index1D>:$newShapeValue,
    OptionalAttr<DenseI64ArrayAttr>:$new_shape
  );

  let builders = [
    OpBuilder<(ins "mlir::Type":$output, "Value":$input, "Value":$new_shape)>,
    OpBuilder<(ins "mlir::Type":$output, "Value":$input, "DenseI64ArrayAttr":$new_shape)>,
  ];

   let extraClassDeclaration = [{
      uint64_t getNewRankSize();
    }];

  let results = (outs
    MindSpore_Tensor:$output
  );
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: slice
//===----------------------------------------------------------------------===//
def MindSpore_SliceOp: MindSpore_Op<"slice", [Pure]> {
  let summary = "Generated op for mindspore.ops.slice(input_x, begin, size)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.slice.html

    Extract a slice of the input on the given axis, beginning at the
    start coordinates, and extending for size elements in each direction. No
    data conversion happens during a slice operation.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    DenseI64ArrayAttr:$begin,
    DenseI64ArrayAttr:$size
  );

  let results = (outs
    MindSpore_Tensor:$result
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: strided_slice
//===----------------------------------------------------------------------===//
def MindSpore_Strided_SliceOp: MindSpore_Op<"strided_slice", [Pure]> {
  let summary = "Generated op for mindspore.ops.strided_slice(...)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.strided_slice.html

    Extract a strided slice of a Tensor based on begin/end index and strides.
    This operation extracts a fragment of size (end-begin)/strides from the
    given ‘input_tensor’. Starting from the beginning position, the fragment
    continues adding strides to the index until all dimensions are not less than
    the ending position.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    Variadic<MindSpore_Tensor1D>:$start_end_strides,
    OptionalAttr<DenseI64ArrayAttr>:$start,
    OptionalAttr<DenseI64ArrayAttr>:$end,
    OptionalAttr<DenseI64ArrayAttr>:$strides,
    OptionalAttr<I64Attr>:$begin_mask,
    OptionalAttr<I64Attr>:$end_mask,
    OptionalAttr<I64Attr>:$ellipsis_mask,
    OptionalAttr<I64Attr>:$new_axis_mask,
    OptionalAttr<I64Attr>:$shrink_axis_mask
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: pad
//===----------------------------------------------------------------------===//
def MindSpore_PadOp: MindSpore_Op<"pad", [Pure]> {
  let summary = "Generated op for mindspore.ops.pad(input_x, padding, mode='constant', value=None)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.pad.html

    Pad the input tensor according to the padding.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input_x,
    DenseI64ArrayAttr:$padding,
    OptionalAttr<StrAttr>:$mode,
    OptionalAttr<I64Attr>:$value
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: split
//===----------------------------------------------------------------------===//
def MindSpore_SplitOp: MindSpore_Op<"split", [Pure]> {
  let summary = "Generated op for mindspore.ops.split(tensor, split_size_or_sections, axis=0)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.split.html

    Split the Tensor into chunks along the given axis.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input_x,
    AnyAttrOf<[I64Attr, DenseI64ArrayAttr]>:$split_size_or_sections,
    OptionalAttr<I64Attr>:$axis
  );

  let results = (outs
    Variadic<MindSpore_Tensor>:$result
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: gather
//===----------------------------------------------------------------------===//
def MindSpore_GatherOp : MindSpore_Op<"gather", [Pure]> {
  let summary = "Generated op for mindspore.ops.gather(input_params, input_indices, axis, batch_dims=0)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.gather.html

    Return the slice of the input tensor corresponding to the elements of
    input_indices on the specified axis.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    MindSpore_Tensor:$indices,
    I64Attr:$axis,
    OptionalAttr<I64Attr>:$batch_dims
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: sqrt
//===----------------------------------------------------------------------===//
def MindSpore_SqrtOp: MindSpore_Op<"sqrt", [Pure]> {
  let summary = "Generated op for mindspore.ops.sqrt(value)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.sqrt.html

    Returns sqrt of a tensor element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: IsNan
//===----------------------------------------------------------------------===//
def MindSpore_IsnanOp : MindSpore_Op<"isnan", [Pure]> {
  let summary = "Generated op for mindspore.ops.isnan(x)";
  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.isnan.html

    Determines which elements are NaN for each position.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: isInf
//===----------------------------------------------------------------------===//
def MindSpore_IsinfOp : MindSpore_Op<"isinf", [Pure]> {
  let summary = "Generated op for mindspore.ops.isinf(input)";
  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.isinf.html

    Determines which elements are inf or -inf for each position.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: isInf
//===----------------------------------------------------------------------===//
def MindSpore_IsFiniteOp : MindSpore_Op<"isfinite", [Pure]> {
  let summary = "Generated op for mindspore.ops.isfinite(input)";
  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.isfinite.html

    Determines which elements are finite for each position. If elements are not NaN, -INF, INF, they are finite.
  }];

  let arguments = (ins
      MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: keepArgs
//===----------------------------------------------------------------------===//
def MindSpore_KeepArgsOp : MindSpore_Op<"keepargs", [MemoryEffects<[MemWrite]>]> {
  let summary = "Keep two arguments to record and avoid CSE.";
  let description = [{
    Keep two arguments with anytype to record and avoid CSE.
  }];

  let arguments = (ins
    AnyType:$input1,
    AnyType:$input2
  );

  let hasCanonicalizer = 0;
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: inv
//===----------------------------------------------------------------------===//
def MindSpore_InvOp : MindSpore_Op<"inv", [Pure]> {
  let summary = "Generated op for mindspore.ops.inv(input)";
  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.inv.html

    Computes Reciprocal of input tensor element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: rsqrt
//===----------------------------------------------------------------------===//
def MindSpore_RsqrtOp : MindSpore_Op<"rsqrt", [Pure]> {
  let summary = "Generated op for mindspore.ops.rsqrt(input)";
  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.rsqrt.html

    Computes reciprocal of square root of input tensor element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: log
//===----------------------------------------------------------------------===//
def MindSpore_LogOp : MindSpore_Op<"log", [Pure]> {
  let summary = "Generated op for mindspore.ops.log(input)";
  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.log.html

    Input Tensor of any dimension. The value must be greater than 0.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: log1p
//===----------------------------------------------------------------------===//
def MindSpore_Log1pOp : MindSpore_Op<"log1p", [Pure]> {
  let summary = "Generated op for mindspore.ops.log1p(input)";
  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.log1p.html

    Returns the natural logarithm of one plus the input tensor element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: xlogy
//===----------------------------------------------------------------------===//
def MindSpore_XLogYOp : MindSpore_Op<"xlogy", [Pure]> {
  let summary = "Generated op for mindspore.ops.xlogy(input, other)";
  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.log1p.html

    Computes the first input tensor multiplied by the logarithm of second input tensor element-wise.
    Returns zero when input is zero.

    out_i=input_i ln other_i

    Inputs of input and other comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar, the scalar could only be a constant.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    MindSpore_Tensor:$other
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: abs
//===----------------------------------------------------------------------===//
def MindSpore_AbsOp : MindSpore_Op<"abs", [Pure]> {
  let summary = "Generated op for mindspore.ops.abs(input)";
  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.abs.html

    Returns absolute value of a tensor element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: maximum
//===----------------------------------------------------------------------===//
def MindSpore_MaximumOp : MindSpore_Op<"maximum", [Pure]> {
  let summary = "Generated op for mindspore.ops.maximum(input)";
  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.maximum.html

    Computes the maximum of input tensors element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input1,
    MindSpore_Tensor:$input2
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: minimum
//===----------------------------------------------------------------------===//
def MindSpore_MinimumOp : MindSpore_Op<"minimum", [Pure]> {
  let summary = "Generated op for mindspore.ops.minimum(input)";
  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.minimum.html

    Computes the minimum of input tensors element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input1,
    MindSpore_Tensor:$input2
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: floor
//===----------------------------------------------------------------------===//
def MindSpore_FloorOp : MindSpore_Op<"floor", [Pure]> {
  let summary = "Generated op for mindspore.ops.floor(input)";
  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.floor.html

    Rounds a tensor down to the closest integer element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: ceil
//===----------------------------------------------------------------------===//
def MindSpore_CeilOp : MindSpore_Op<"ceil", [Pure]> {
  let summary = "Generated op for mindspore.ops.ceil(input)";
  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.ceil.html

    Rounds a tensor up to the closest integer element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: round
//===----------------------------------------------------------------------===//
def MindSpore_RoundOp : MindSpore_Op<"round", [Pure]> {
  let summary = "Generated op for mindspore.ops.round(input)";
  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.round.html

    Returns half to even of a tensor element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: trunc
//===----------------------------------------------------------------------===//
def MindSpore_TruncOp : MindSpore_Op<"trunc", [Pure]> {
  let summary = "Generated op for mindspore.ops.trunc(input)";
  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.trunc.html

    Returns a new tensor with the truncated integer values of the elements of the input tensor.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: onehot
//===----------------------------------------------------------------------===//
def MindSpore_OneHotOp : MindSpore_Op<"onehot", [Pure]> {
  let summary = "Generated op for mindspore.ops.onehot(input, depth, on_value, off_value)";
  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.OneHot.html

    Computes a one-hot tensor.

    The locations represented by indices in indices take value on_value,
    while all other locations take value off_value.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    DenseI64ArrayAttr:$depth,
    MindSpore_Tensor:$on_value,
    MindSpore_Tensor:$off_value
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: cumsum
//===----------------------------------------------------------------------===//
def MindSpore_CumSumOp : MindSpore_Op<"cumsum", [Pure]> {
  let summary = "mindspore.ops.cumsum(x, axis, dtype=None)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.cumsum.html

    Computes the cumulative sum of input Tensor along axis.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    DenseI64ArrayAttr:$axis
  );

  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: unknown
//===----------------------------------------------------------------------===//
def MindSpore_UnknownOp : MindSpore_Op<"unknown", [Pure]> {
  let summary = "Generated op for mindspore.ops.unknown(input)";
  let description = [{
    Unknown Op Placeholder
  }];

  let arguments = (ins
    Variadic<MindSpore_Tensor>:$others
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: matmul
//===----------------------------------------------------------------------===//
def MindSpore_MatMulOp : MindSpore_Op<"matmul", [Pure]> {
  let summary = "Matrix multiplication";

  let description = [{
    Performs a two dimensional matrix multiplication.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input_a,
    MindSpore_Tensor:$input_b,
    OptionalAttr<BoolAttr>:$transpose_a,
    OptionalAttr<BoolAttr>:$transpose_b
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: batchmatmul
//===----------------------------------------------------------------------===//
def MindSpore_BatchMatMulOp : MindSpore_Op<"batchmatmul", [Pure]> {
  let summary = "Matrix multiplication with batches";

  let description = [{
    Performs a two dimensional matrix multiplication with one or two batches.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input_a,
    MindSpore_Tensor:$input_b,
    OptionalAttr<BoolAttr>:$transpose_a,
    OptionalAttr<BoolAttr>:$transpose_b
  );

  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: argmin
//===----------------------------------------------------------------------===//
def MindSpore_ArgMinOp : MindSpore_Op<"argmin", [Pure]> {
  let summary = "mindspore.ops.argmin(input, dim=None, keepdim=False)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.argmin.html

    Return the indices of the minimum values of a tensor across a dimension.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    DenseI64ArrayAttr:$axis,
    OptionalAttr<BoolAttr>:$keepdims
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: adaptive_avg_pool2d
//===----------------------------------------------------------------------===//
def MindSpore_AdaptiveAvgPool2dOp : MindSpore_Op<"adaptive_avg_pool2d", [Pure]> {
  let summary = "mindspore.ops.adaptive_avg_pool2d(input, output_size)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.adaptive_avg_pool2d.html

    Performs 2D adaptive average pooling on a multi-plane input signal.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    DenseI64ArrayAttr:$output_size
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: adaptive_avg_pool3d
//===----------------------------------------------------------------------===//
def MindSpore_AdaptiveAvgPool3dOp : MindSpore_Op<"adaptive_avg_pool3d", [Pure]> {
  let summary = "mindspore.ops.adaptive_avg_pool3d(input, output_size)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.adaptive_avg_pool3d.html

    Performs 3D adaptive average pooling on a multi-plane input signal.

  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    DenseI64ArrayAttr:$output_size
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: log_softmax
//===----------------------------------------------------------------------===//
def MindSpore_LogSoftmaxOp : MindSpore_Op<"log_softmax", [Pure]> {
  let summary = "mindspore.ops.log_softmax(logits, axis=- 1)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.log_softmax.html

    Applies the Log Softmax function to the input tensor on the specified axis.

  }];

  let arguments = (ins
    MindSpore_Tensor:$logits,
    DefaultValuedAttr<I64Attr, "-1">:$axis
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: softmax
//===----------------------------------------------------------------------===//
def MindSpore_SoftmaxOp : MindSpore_Op<"softmax", [Pure]> {
  let summary = "mindspore.ops.softmax(input, axis=- 1, *, dtype=None)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.softmax.html

    Applies the Softmax operation to the input tensor on the specified axis.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    DefaultValuedAttr<I64Attr, "-1">:$axis,
    OptionalAttr<TypeAttr>:$dtype
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: adaptive_avg_pool1d
//===----------------------------------------------------------------------===//
def MindSpore_AdaptiveAvgPool1dOp : MindSpore_Op<"adaptive_avg_pool1d", [Pure]> {
  let summary = "mindspore.ops.adaptive_avg_pool1d(input, output_size)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.adaptive_avg_pool1d.html

    Applies a 1D adaptive average pooling over an input Tensor which can be regarded as a composition of 1D input planes.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    I64Attr:$output_size
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: addmm
//===----------------------------------------------------------------------===//
def MindSpore_AddmmOp : MindSpore_Op<"addmm", [Pure]> {
  let summary = "mindspore.ops.addmm(input, mat1, mat2, *, beta=1, alpha=1)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.addmm.html

    Multiply matrix mat1 and matrix mat2.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    MindSpore_Tensor:$mat1,
    MindSpore_Tensor:$mat2,
    OptionalAttr<AnyAttrOf<[I64Attr, F64Attr]>>:$beta,
    OptionalAttr<AnyAttrOf<[I64Attr, F64Attr]>>:$alpha
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: amax
//===----------------------------------------------------------------------===//
def MindSpore_AmaxOp : MindSpore_Op<"amax", [Pure]> {
  let summary = "mindspore.ops.amax(input, axis=None, keepdims=False, *, initial=None, where=None)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.amax.html

    Return the maximum values along the given axis of the tensor.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    OptionalAttr<DenseI64ArrayAttr>:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims,
    OptionalAttr<AnyAttrOf<[I64Attr, F64Attr]>>:$initial,
    Optional<TensorOf<[I1]>>:$where
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: amin
//===----------------------------------------------------------------------===//
def MindSpore_AminOp : MindSpore_Op<"amin", [Pure]> {
  let summary = "mindspore.ops.amin(input, axis=None, keepdims=False, *, initial=None, where=None)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.amin.html

    Return the minimum values along the given axis of the tensor.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    OptionalAttr<DenseI64ArrayAttr>:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims,
    OptionalAttr<AnyAttrOf<[I64Attr, F64Attr]>>:$initial,
    Optional<TensorOf<[I1]>>:$where
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: any
//===----------------------------------------------------------------------===//
def MindSpore_AnyOp : MindSpore_Op<"any", [Pure]> {
  let summary = "mindspore.ops.any(input, axis=None, keep_dims=False)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.any.html

    Tests if any element in input evaluates to True along the given axes.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    OptionalAttr<DenseI64ArrayAttr>:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: atanh
//===----------------------------------------------------------------------===//
def MindSpore_AtanhOp : MindSpore_Op<"atanh", [Pure]> {
  let summary = "mindspore.ops.atanh(input)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.atanh.html

    Computes inverse hyperbolic tangent of the input element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: avg_pool1d
//===----------------------------------------------------------------------===//
def MindSpore_AvgPool1dOp : MindSpore_Op<"avg_pool1d", [Pure]> {
  let summary = "mindspore.ops.avg_pool1d(input_x, kernel_size=1, stride=1, padding=0, ceil_mode=False, count_include_pad=True)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.avg_pool1d.html

    Applies a 1D average pooling over an input Tensor which can be regarded as a composition of 1D input planes.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input_x,
    DefaultValuedAttr<I64Attr, "1">:$kernel_size,
    DefaultValuedAttr<I64Attr, "1">:$stride,
    DefaultValuedAttr<I64Attr, "0">:$padding,
    DefaultValuedAttr<BoolAttr, "false">:$ceil_mode,
    DefaultValuedAttr<BoolAttr, "true">:$count_include_pad
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: avg_pool2d
//===----------------------------------------------------------------------===//
def MindSpore_AvgPool2dOp : MindSpore_Op<"avg_pool2d", [Pure]> {
  let summary = "mindspore.ops.avg_pool2d(input_x, kernel_size=1, stride=1, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=0)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.avg_pool2d.html

    Applies a 2D average pooling over an input Tensor which can be regarded as a composition of 2D input planes.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input_x,
    DefaultValuedAttr<I64Attr, "1">:$kernel_size,
    DefaultValuedAttr<I64Attr, "1">:$stride,
    DefaultValuedAttr<I64Attr, "0">:$padding,
    DefaultValuedAttr<BoolAttr, "false">:$ceil_mode,
    DefaultValuedAttr<BoolAttr, "true">:$count_include_pad,
    DefaultValuedAttr<I64Attr, "0">:$divisor_override
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: avg_pool3d
//===----------------------------------------------------------------------===//
def MindSpore_AvgPool3dOp : MindSpore_Op<"avg_pool3d", [Pure]> {
  let summary = "mindspore.ops.avg_pool3d(input_x, kernel_size=1, stride=1, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=0)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.avg_pool3d.html

    Applies a 3D average pooling over an input Tensor which can be regarded as a composition of 3D input planes.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input_x,
    DefaultValuedAttr<I64Attr, "1">:$kernel_size,
    DefaultValuedAttr<I64Attr, "1">:$stride,
    DefaultValuedAttr<I64Attr, "0">:$padding,
    DefaultValuedAttr<BoolAttr, "false">:$ceil_mode,
    DefaultValuedAttr<BoolAttr, "true">:$count_include_pad,
    DefaultValuedAttr<I64Attr, "0">:$divisor_override
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: bmm
//===----------------------------------------------------------------------===//
def MindSpore_BmmOp : MindSpore_Op<"bmm", [Pure]> {
  let summary = "mindspore.ops.bmm(input_x, mat2)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.bmm.html

    Perform a batch matrix-matrix product of matrices in input tensors.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input_x,
    MindSpore_Tensor:$mat2
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: cat
//===----------------------------------------------------------------------===//
def MindSpore_CatOp : MindSpore_Op<"cat", [Pure]> {
  let summary = "mindspore.ops.cat(tensors, axis=0)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.cat.html

    Connect input tensors along with the given axis.
  }];

  let arguments = (ins
    Variadic<MindSpore_Tensor>:$tensors,
    DefaultValuedAttr<I64Attr, "0">:$axis
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: clamp
//===----------------------------------------------------------------------===//
def MindSpore_ClampOp : MindSpore_Op<"clamp", [Pure, AttrSizedOperandSegments]> {
  let summary = "mindspore.ops.clamp(input, min=None, max=None)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.clamp.html

    Clamp all elements of the input tensor within the range [min, max].
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    Optional<MindSpore_Tensor>:$min,
    Optional<MindSpore_Tensor>:$max
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: col2im
//===----------------------------------------------------------------------===//
def MindSpore_Col2ImOp : MindSpore_Op<"col2im", [Pure]> {
  let summary = "mindspore.ops.col2im(input_x, output_size, kernel_size, dilation, padding_value, stride)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.col2im.html

    Combines an array of sliding local blocks into a large containing tensor.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input_x,
    RankedTensorOf<[I64]>:$output_size,
    DenseI64ArrayAttr:$kernel_size,
    DenseI64ArrayAttr:$dilation,
    DenseI64ArrayAttr:$padding_value,
    DenseI64ArrayAttr:$stride
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: cosh
//===----------------------------------------------------------------------===//
def MindSpore_CoshOp : MindSpore_Op<"cosh", [Pure]> {
  let summary = "mindspore.ops.cosh(input)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.cosh.html

    Computes hyperbolic cosine of input element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: diagonal
//===----------------------------------------------------------------------===//
def MindSpore_DiagonalOp : MindSpore_Op<"diagonal", [Pure]> {
  let summary = "mindspore.ops.diagonal(input, offset=0, dim1=0, dim2=1)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.diagonal.html

    Returns diagonals of the input tensor along specified dimension.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    DefaultValuedAttr<I64Attr, "0">:$offset,
    DefaultValuedAttr<I64Attr, "0">:$dim1,
    DefaultValuedAttr<I64Attr, "1">:$dim2
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: elu
//===----------------------------------------------------------------------===//
def MindSpore_EluOp : MindSpore_Op<"elu", [Pure]> {
  let summary = "mindspore.ops.elu(input_x, alpha=1.0)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.elu.html

    Exponential Linear Unit activation function.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input_x,
    DefaultValuedAttr<F64Attr, "1.0">:$alpha
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: embedding
//===----------------------------------------------------------------------===//
def MindSpore_EmbeddingOp : MindSpore_Op<"embedding", [Pure]> {
  let summary = "mindspore.ops.embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.embedding.html

    Retrieve the word embeddings in weight using indices specified in input.
  }];

  let arguments = (ins
    TensorOf<[I32, I64]>:$input,
    MindSpore_Tensor:$weight,
    OptionalAttr<I64Attr>:$padding_idx,
    OptionalAttr<F64Attr>:$max_norm,
    DefaultValuedAttr<F64Attr, "2.0">:$norm_type,
    DefaultValuedAttr<BoolAttr, "false">:$scale_grad_by_freq
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: flip
//===----------------------------------------------------------------------===//
def MindSpore_FlipOp : MindSpore_Op<"flip", [Pure]> {
  let summary = "mindspore.ops.flip(input, dims)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.flip.html

    Reverses elements in a tensor along the given dims.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    DenseI64ArrayAttr:$dims
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: full
//===----------------------------------------------------------------------===//
def MindSpore_FullOp : MindSpore_Op<"full", [Pure]> {
  let summary = "mindspore.ops.full(size, fill_value, *, dtype=None)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.full.html

    Create a tensor filled with a specified value.
  }];

  let arguments = (ins
    DenseI64ArrayAttr:$size,
    AnyAttrOf<[I64Attr, F64Attr]>:$fill_value,
    OptionalAttr<TypeAttr>:$dtype
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: full_like
//===----------------------------------------------------------------------===//
def MindSpore_FullLikeOp : MindSpore_Op<"full_like", [Pure]> {
  let summary = "mindspore.ops.full_like(input, fill_value, *, dtype=None)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.full_like.html

    Return  tensor of the same shape as input and filled with a specified value.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    AnyAttrOf<[I64Attr, F64Attr]>:$fill_value,
    OptionalAttr<TypeAttr>:$dtype
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: hardtanh
//===----------------------------------------------------------------------===//
def MindSpore_HardtanhOp : MindSpore_Op<"hardtanh", [Pure]> {
  let summary = "mindspore.ops.hardtanh(input, min_val=- 1.0, max_val=1.0)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.hardtanh.html

    Applies the hardtanh activation function element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    DefaultValuedAttr<F64Attr, "-1.0">:$min_val,
    DefaultValuedAttr<F64Attr, "1.0">:$max_val
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: index_select
//===----------------------------------------------------------------------===//
def MindSpore_IndexSelectOp : MindSpore_Op<"index_select", [Pure]> {
  let summary = "mindspore.ops.index_select(input, axis, index)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.index_select.html

    Select the input tensor according to the specified axis and index and return a new tensor.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    I64Attr:$axis,
    TensorOf<[I32, I64]>:$index
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: leaky_relu
//===----------------------------------------------------------------------===//
def MindSpore_LeakyReluOp : MindSpore_Op<"leaky_relu", [Pure]> {
  let summary = "mindspore.ops.leaky_relu(input, alpha=0.2)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.leaky_relu.html

    leaky_relu activation function.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    DefaultValuedAttr<F64Attr, "0.2">:$alpha
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: log10
//===----------------------------------------------------------------------===//
def MindSpore_Log10Op : MindSpore_Op<"log10", [Pure]> {
  let summary = "mindspore.ops.log10(input)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.log10.html

    Compute the logarithm to the base 10 of the input tensor element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: log2
//===----------------------------------------------------------------------===//
def MindSpore_Log2Op : MindSpore_Op<"log2", [Pure]> {
  let summary = "mindspore.ops.log2(input)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.log2.html

    Compute the logarithm to the base 2 of the input tensor element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: logical_xor
//===----------------------------------------------------------------------===//
def MindSpore_LogicalXorOp : MindSpore_Op<"logical_xor", [Pure]> {
  let summary = "mindspore.ops.logical_xor(input, other)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.logical_xor.html

    Compute the "logical XOR" of two tensors element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    MindSpore_Tensor:$other
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: mean
//===----------------------------------------------------------------------===//
def MindSpore_MeanXorOp : MindSpore_Op<"mean", [Pure]> {
  let summary = "mmindspore.ops.mean(x, axis=None, keep_dims=False)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.mean.html

    Compute the mean(s) of the tensor along the specified axis(axes).
  }];

  let arguments = (ins
    MindSpore_Tensor:$x,
    OptionalAttr<DenseI64ArrayAttr>:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: mm
//===----------------------------------------------------------------------===//
def MindSpore_MmOp : MindSpore_Op<"mm", [Pure]> {
  let summary = "mindspore.ops.mm(input, mat2)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.mm.html

    Returns the matrix product of two arrays.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    MindSpore_Tensor:$mat2
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: nonzero
//===----------------------------------------------------------------------===//
def MindSpore_NonZeroOp : MindSpore_Op<"nonzero", [Pure]> {
  let summary = "mindspore.ops.nonzero(input, *, as_tuple=False)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.nonzero.html

    Return the positions of all non-zero values.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    DefaultValuedAttr<BoolAttr, "false">:$as_tuple
  );
  
  let results = (outs
    Variadic<TensorOf<[I64]>>:$outputs
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: permute
//===----------------------------------------------------------------------===//
def MindSpore_PermuteOp : MindSpore_Op<"permute", [Pure]> {
  let summary = "mindspore.ops.permute(input, axis)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.permute.html

    Permute the input tensor along the specified axis.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    DenseI64ArrayAttr:$axis
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: prod
//===----------------------------------------------------------------------===//
def MindSpore_ProdOp : MindSpore_Op<"prod", [Pure]> {
  let summary = "mindspore.ops.prod(input, axis=None, keep_dims=False, dtype=None)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.prod.html

    Return the product(s) of the tensor along the specified axis(axes).
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    OptionalAttr<DenseI64ArrayAttr>:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$keep_dims,
    OptionalAttr<TypeAttr>:$dtype
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: rand
//===----------------------------------------------------------------------===//
def MindSpore_RandOp : MindSpore_Op<"rand", [Pure]> {
  let summary = "mindspore.ops.rand(*size, dtype=None, seed=None)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.rand.html

    Return a new tensor that fills numbers from the uniform distribution over an interval [0,1) based on the given size and dtype.
  }];

  let arguments = (ins
    DenseI64ArrayAttr:$size,
    OptionalAttr<TypeAttr>:$dtype,
    OptionalAttr<I64Attr>:$seed
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: randn
//===----------------------------------------------------------------------===//
def MindSpore_RandnOp : MindSpore_Op<"randn", [Pure]> {
  let summary = "mindspore.ops.randn(*size, dtype=None, seed=None)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.randn.html

    Return a new tensor with given shape and dtype, filled with random numbers from the standard normal distribution.
  }];

  let arguments = (ins
    DenseI64ArrayAttr:$size,
    OptionalAttr<TypeAttr>:$dtype,
    OptionalAttr<I64Attr>:$seed
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: randperm
//===----------------------------------------------------------------------===//
def MindSpore_RandPermOp : MindSpore_Op<"randperm", [Pure]> {
  let summary = "mindspore.ops.randperm(n, seed=0, offset=0, dtype=mstype.int64)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.randperm.html

    Generates random permutation of integers from 0 to n-1.
  }];

  let arguments = (ins
    DenseI64ArrayAttr:$n,
    DefaultValuedAttr<I64Attr, "0">:$seed,
    DefaultValuedAttr<I64Attr, "0">:$offset,
    OptionalAttr<TypeAttr>:$dtype
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: reciprocal
//===----------------------------------------------------------------------===//
def MindSpore_ReciprocalOp : MindSpore_Op<"reciprocal", [Pure]> {
  let summary = "mindspore.ops.reciprocal(input)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.reciprocal.html

    Returns reciprocal of a tensor element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: scatter_add
//===----------------------------------------------------------------------===//
def MindSpore_ScatterAddOp : MindSpore_Op<"scatter_add", [Pure]> {
  let summary = "mindspore.ops.scatter_add(input_x, indices, updates)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.scatter_add.html

    Perform an addition update on input_x based on the specified indices and update values.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input_x,
    TensorOf<[I32, I64]>:$indices,
    MindSpore_Tensor:$updates
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: select_scatter
//===----------------------------------------------------------------------===//
def MindSpore_SelectScatterOp : MindSpore_Op<"select_scatter", [Pure]> {
  let summary = "mindspore.ops.select_scatter(input, src, axis, index)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.select_scatter.html

    On the specified dimension axis of input , src is scattered into input on the specified index of input .
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    MindSpore_Tensor:$src,
    I64Attr:$axis,
    I64Attr:$index
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: sinh
//===----------------------------------------------------------------------===//
def MindSpore_SinhOp : MindSpore_Op<"sinh", [Pure]> {
  let summary = "mindspore.ops.sinh(input)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.sinh.html

    Compute hyperbolic sine of the input element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: slice_scatter
//===----------------------------------------------------------------------===//
def MindSpore_SliceScatterOp : MindSpore_Op<"slice_scatter", [Pure]> {
  let summary = "mindspore.ops.slice_scatter(input, src, axis=0, start=None, end=None, step=1)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.slice_scatter.html

    Embed src into the sliced input along the specified axis .
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    MindSpore_Tensor:$src,
    DefaultValuedAttr<I64Attr, "0">:$axis,
    OptionalAttr<I64Attr>:$start,
    OptionalAttr<I64Attr>:$end,
    DefaultValuedAttr<I64Attr, "1">:$step
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: sort
//===----------------------------------------------------------------------===//
def MindSpore_SortOp : MindSpore_Op<"sort", [Pure]> {
  let summary = "mindspore.ops.sort(input_x, axis=- 1, descending=False)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.sort.html

    Sort the elements of the input tensor along the given axis.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input_x,
    DefaultValuedAttr<I64Attr, "-1">:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$descending
  );
  
  let results = (outs
    MindSpore_Tensor:$sorted_tensor,
    TensorOf<[I64]>:$indices
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: tan
//===----------------------------------------------------------------------===//
def MindSpore_TanOp : MindSpore_Op<"tan", [Pure]> {
  let summary = "mindspore.ops.tan(input)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.tan.html

    Computes tangent of input element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input

  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: topk
//===----------------------------------------------------------------------===//
def MindSpore_TopkOp : MindSpore_Op<"topk", [Pure]> {
  let summary = "mindspore.ops.topk(input, k, dim=None, largest=True, sorted=True)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.topk.html

    Return the top k largest or smallest elements of the input tensor along a specified dimension.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    I64Attr:$k,
    OptionalAttr<I64Attr>:$dim,
    DefaultValuedAttr<BoolAttr, "true">:$largest,
    DefaultValuedAttr<BoolAttr, "true">:$sorted
  );
  
  let results = (outs
    MindSpore_Tensor:$values,
    TensorOf<[I64]>:$indices
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: unsqueeze
//===----------------------------------------------------------------------===//
def MindSpore_UnsqueezeOp : MindSpore_Op<"unsqueeze", [Pure]> {
  let summary = "mindspore.ops.unsqueeze(input, dim)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.unsqueeze.html

    Adds an additional dimension to the input tensor at the given dimension.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    I64Attr:$dim
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: bitwise_and
//===----------------------------------------------------------------------===//
def MindSpore_BitwiseAndOp : MindSpore_Op<"bitwise_and", [Pure]> {
  let summary = "mindspore.ops.bitwise_and(input, other)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.bitwise_and.html

    Compute the bitwise AND of two input tensors.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    MindSpore_Tensor:$other
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: bitwise_or
//===----------------------------------------------------------------------===//
def MindSpore_BitwiseOrOp : MindSpore_Op<"bitwise_or", [Pure]> {
  let summary = "mindspore.ops.bitwise_or(input, other)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.bitwise_or.html

    Compute the bitwise OR of two input tensors.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    MindSpore_Tensor:$other
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// MindSpore Operator: bitwise_xor
//===----------------------------------------------------------------------===//
def MindSpore_BitwiseXorOp : MindSpore_Op<"bitwise_xor", [Pure]> {
  let summary = "mindspore.ops.bitwise_xor(input, other)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.bitwise_xor.html

    Compute the bitwise XOR of two input tensors.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    MindSpore_Tensor:$other
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: eq
//===----------------------------------------------------------------------===//
def MindSpore_EqOp : MindSpore_Op<"eq", [Pure]> {
  let summary = "mindspore.ops.eq(input, other)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.eq.html

    Compute the equivalence of the two inputs element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    MindSpore_Tensor:$other
  );
  
  let results = (outs
    TensorOf<[I1]>:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: fmod
//===----------------------------------------------------------------------===//
def MindSpore_FmodOp : MindSpore_Op<"fmod", [Pure]> {
  let summary = "mindspore.ops.fmod(input, other)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.fmod.html

    Compute the remainder of element-wise division of first input by the second input.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    MindSpore_Tensor:$other
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: ge
//===----------------------------------------------------------------------===//
def MindSpore_GeOp : MindSpore_Op<"ge", [Pure]> {
  let summary = "mindspore.ops.ge(input, other)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.ge.html

    Compute the value of input >= other element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    MindSpore_Tensor:$other
  );
  
  let results = (outs
    TensorOf<[I1]>:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: le
//===----------------------------------------------------------------------===//
def MindSpore_LeOp : MindSpore_Op<"le", [Pure]> {
  let summary = "mindspore.ops.le(input, other)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.le.html

    Compute the value of input <= other element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    MindSpore_Tensor:$other
  );
  
  let results = (outs
    TensorOf<[I1]>:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: lt
//===----------------------------------------------------------------------===//
def MindSpore_LtOp : MindSpore_Op<"lt", [Pure]> {
  let summary = "mindspore.ops.lt(input, other)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.lt.html

    Alias for mindspore.ops.less() .
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    MindSpore_Tensor:$other
  );
  
  let results = (outs
    TensorOf<[I1]>:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: gt
//===----------------------------------------------------------------------===//
def MindSpore_GtOp : MindSpore_Op<"gt", [Pure]> {
  let summary = "mindspore.ops.gt(input, other)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.gt.html

    Compute the value of input > other element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    MindSpore_Tensor:$other
  );
  
  let results = (outs
    TensorOf<[I1]>:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: ne
//===----------------------------------------------------------------------===//
def MindSpore_NeOp : MindSpore_Op<"ne", [Pure, Commutative]> {
  let summary = "mindspore.ops.ne(input, other)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.ne.html

    Compute the non-equivalence of two inputs element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    MindSpore_Tensor:$other
  );
  
  let results = (outs
    TensorOf<[I1]>:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: remainder
//===----------------------------------------------------------------------===//
def MindSpore_RemainderOp : MindSpore_Op<"remainder", [Pure]> {
  let summary = "mindspore.ops.remainder(input, other)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.remainder.html

    Compute the remainder of division for the input tensor element-wise.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    MindSpore_Tensor:$other
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: arange
//===----------------------------------------------------------------------===//
def MindSpore_ArangeOp : MindSpore_Op<"arange", [Pure]> {
  let summary = "mindspore.ops.arange(start=0, end=None, step=1, *, dtype=None)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.arange.html

    Returns a tensor with a step length of step in the interval [ start , end ).
  }];

  let arguments = (ins
    OptionalAttr<AnyAttrOf<[I64Attr, F64Attr]>>:$start,
    OptionalAttr<AnyAttrOf<[I64Attr, F64Attr]>>:$end,
    OptionalAttr<AnyAttrOf<[I64Attr, F64Attr]>>:$step,
    OptionalAttr<TypeAttr>:$dtype

  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: fill
//===----------------------------------------------------------------------===//
def MindSpore_FillOp : MindSpore_Op<"fill", [Pure]> {
  let summary = "mindspore.ops.fill(type, shape, value)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.fill.html

    Create a tensor filled with a specified value.
  }];

  let arguments = (ins
    TypeAttr:$type,
    MindSpore_Tensor:$shape,
    MindSpore_Tensor:$value
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: squeeze
//===----------------------------------------------------------------------===//
def MindSpore_SqueezeOp : MindSpore_Op<"squeeze", [Pure]> {
  let summary = "mindspore.ops.squeeze(input, axis=None)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.squeeze.html

    Remove length one axes from input tensor.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    OptionalAttr<AnyAttrOf<[I64Attr, DenseI64ArrayAttr]>>:$axis
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: var
//===----------------------------------------------------------------------===//
def MindSpore_VarOp : MindSpore_Op<"var", [Pure]> {
  let summary = "mindspore.ops.var(input, axis=None, ddof=0, keepdims=False)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.var.html

    Compute the variance of the tensor along a specified axis.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    OptionalAttr<AnyAttrOf<[I64Attr, DenseI64ArrayAttr]>>:$axis,
    OptionalAttr<AnyAttrOf<[I64Attr, BoolAttr]>>:$ddof,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: where
//===----------------------------------------------------------------------===//
def MindSpore_WhereOp : MindSpore_Op<"where", [Pure]> {
  let summary = "mindspore.ops.where(condition, input, other)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.where.html

    Return a tensor in which the elements are selected from input or other based on the condition.
  }];

  let arguments = (ins
    TensorOf<[I1]>:$condition,
    MindSpore_Tensor:$input,
    MindSpore_Tensor:$other
  );
  
  let results = (outs
    MindSpore_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: max
//===----------------------------------------------------------------------===//
def MindSpore_MaxOp : MindSpore_Op<"max", [Pure]> {
  let summary = "mindspore.ops.max(input, axis=None, keepdims=False, *, initial=None, where=None)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.max.html

    Return the maximum values and their indices along the given axis of the tensor.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    OptionalAttr<I64Attr>:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims,
    OptionalAttr<AnyAttrOf<[I64Attr, F64Attr]>>:$initial,
    Optional<TensorOf<[I1]>>:$where
  );
  
  let results = (outs
    MindSpore_Tensor:$max,
    TensorOf<[I64]>:$max_indices
  );
}

//===----------------------------------------------------------------------===//
// MindSpore Operator: min
//===----------------------------------------------------------------------===//
def MindSpore_MinOp : MindSpore_Op<"min", [Pure]> {
  let summary = "mindspore.ops.min(input, axis=None, keepdims=False, *, initial=None, where=None)";

  let description = [{
    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.min.html

    Return the minimum values and their indices along the given axis of the tensor.
  }];

  let arguments = (ins
    MindSpore_Tensor:$input,
    OptionalAttr<I64Attr>:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims,
    OptionalAttr<AnyAttrOf<[I64Attr, F64Attr]>>:$initial,
    Optional<TensorOf<[I1]>>:$where
  );
  
  let results = (outs
    MindSpore_Tensor:$min,
    TensorOf<[I64]>:$min_indices
  );
}

#endif // AKG_DIALECT_MINDSPORE_IR_MINDSPORE_OPS
