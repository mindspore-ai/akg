/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef GE_OP_MATRIX_CALCULATION_OPS_H
#define GE_OP_MATRIX_CALCULATION_OPS_H

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Multiplies matrix "a" by matrix "b", producing "a * b".

*@par Inputs:
*Two inputs, including:
* @li x1: A matrix Tensor. 2D. Must be one of the following types: float16,
* float32, int32. Has format [ND, NHWC, FRACTAL_NZ].
* @li x2: A matrix Tensor. 2D. Must be one of the following types: float16,
* float32, int32. Has format [ND, NHWC, FRACTAL_NZ].
* @li bias: A 1D Tensor. Must be one of the following types: float16,
* float32, int32. Has format [ND, NHWC].

*@par Attributes:
*@li transpose_a: A bool. If True, changes the shape of "x1" from [M, K] to [K, M].
*@li transpose_b: A bool. If True, changes the shape of "x2" from [M, K] to [K, M].

*@par Outputs:
*y: The result matrix Tensor. 2D. Must be one of the following types: float16,
* float32, int32. Has format [ND, NHWC, FRACTAL_NZ].
*/
REG_OP(MatMul)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .OP_END_FACTORY_REG(MatMul)

/**
*@brief Performs Matrix-to-matrix Multiply, producing c=alpha[0]*a*b+beta[0]*c.

*@par Inputs:
*Five inputs, including:
*@li a: A matrix Tensor. 4D. Must be one of the following types:\n float16, int8. Has format [FRACTAL_NZ].
*@li b: A matrix Tensor. 4D. Must be one of the following types:\n float16, int8. When type is int8, has format [FRACTAL_Z], \n otherwise has format [FRACTAL_NZ].
*@li c: A matrix Tensor. 2D or higher. Must be one of the following types: \n float16, int32, float32. When type is int32, has format [ND], \n otherwise has format [FRACTAL_NZ].
*@li alpha: A 1D Tensor. The shape of alpha is [1].\n Must be one of the following types: float16, int32, float32. Has format [ND].
*@li beta: A 1D Tensor. The shape of beta is [1].\n Must be one of the following types: float16, int32, float32. Has format [ND].

*@par Attributes:
*Two attributes, including:
*@li transpose_a: Optional. A bool.\n If True, changes the shape of "a" from [M, K] to [K, M].\n Reserved parameters, not used for now.
*@li transpose_b: Optional. A bool.\n If True, changes the shape of "b" from [M, K] to [K, M].\n Reserved parameters, not used for now.

*@par Outputs:
*@out: The result matrix Tensor. 4D. Must be one of the following types:\n float16, float32, int32. Has format [FRACTAL_NZ].
*/

REG_OP(Gemm)
    .INPUT(a, TensorType({DT_FLOAT16, DT_INT8}))
    .INPUT(b, TensorType({DT_FLOAT16, DT_INT8}))
    .INPUT(c, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(alpha, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(beta, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(out, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .ATTR(transpose_a, Bool, false)
    .ATTR(transpose_b, Bool, false)
    .OP_END_FACTORY_REG(Gemm)

/**
*@brief Multiplies matrix "a" by matrix "b", producing "a * b".

*@par Inputs:
*Three inputs, including:
* @li x1: A matrix Tensor. Must be one of the following types: float16,
* float32, int32. 2D or higher. Has format [ND, NHWC, FRACTAL_NZ].
* @li x2: A matrix Tensor. Must be one of the following types: float16,
* float32, int32. 2D or higher. Has format [ND, NHWC, FRACTAL_NZ].

*@par Attributes:
*@li adj_x: A bool. If True, changes the shape of "x1" from [B, M, K] to [B, K, M].
*@li adj_y: A bool. If True, changes the shape of "x2" from [B, M, K] to [B, K, M].

*@par Outputs:
*y: The result matrix Tensor. 2D or higher. Must be one of the following types: float16,
* float32, int32. 2D or higher. Has format [ND, NHWC, FRACTAL_NZ]. Has the same shape length as "x1" and "x2".
*/

REG_OP(BatchMatMul)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .ATTR(adj_x1, Bool, false)
    .ATTR(adj_x2, Bool, false)
    .OP_END_FACTORY_REG(BatchMatMul)

REG_OP(MeanCCE)
    .INPUT(x, TensorType::ALL())
    .INPUT(indices, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(keep_dims, Bool, false)
    .ATTR(value1, ListInt, {})
    .ATTR(mode, Int, 3)                 // 0:max pooling or 1:avg pooling
    .ATTR(pad_mode, Int, 0)
    .ATTR(global_pooling, Bool, true)
    .ATTR(window, ListInt, {1,1})      // kernel size
    .ATTR(pad, ListInt, {0,0,0,0})     // pad size
    .ATTR(stride, ListInt, {1,1})      // stride size
    .ATTR(ceil_mode, Int, 0)
    .ATTR(data_mode, Int, 1)
    .ATTR(nan_opt, Int, 0)
    .ATTR(fomart, Int, 0)
    .OP_END_FACTORY_REG(MeanCCE)

REG_OP(MeanGrad)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(mode, Int, 1)                 // 0:max pooling or 1:avg pooling
    .ATTR(pad_mode, Int, 0)
    .ATTR(global_pooling, Bool, false)
    .ATTR(window, ListInt, {1,1})      // kernel size
    .ATTR(pad, ListInt, {0,0,0,0})     // pad size
    .ATTR(stride, ListInt, {1,1})      // stride size
    .ATTR(ceil_mode, Int, 0)
    .ATTR(data_mode, Int, 1)
    .ATTR(nan_opt, Int, 0)
    .ATTR(mean_grad_output_shape_value, ListInt, {1,1,1,1})
    .ATTR(mean_grad_output_shape_format, Int, 1) //must be NHWC
    .OP_END_FACTORY_REG(MeanGrad)

REG_OP(MatMulCCE)
    .INPUT(x1, TensorType({DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(x3, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(transpose_a, Bool, false)
    .ATTR(transpose_b, Bool, false)
    .ATTR(has_bias, Bool, false)
    .OP_END_FACTORY_REG(MatMulCCE)

/**
*@brief Computes half the L2 norm of a tensor without the sqrt.

*@par Inputs:

* x: A Tensor.
*     TensorType::FloatingDataType().

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*/
REG_OP(L2Loss)
    .INPUT(x, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(L2Loss)

/**
*@brief: Returns a batched diagonal tensor with a given batched diagonal values.

*@par Inputs:
*x: A Tensor. Must be one of the following types: float16, float32, int32, int8, uint8.

*@par Outputs:
*y: A Tensor. Has the same type as "x".

*/
REG_OP(MatrixDiag)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixDiag)

/**
*@brief: Returns a batched diagonal tensor with a given batched diagonal values.

*@par Inputs:
* Two inputs, including:
*@li x: A Tensor. Must be one of the following types: float16, float32, int32, int8, uint8.
*@li assist: A Tensor of the same type as "x".

*@par Outputs:
*y: A Tensor. Has the same type as "x".

*/
REG_OP(MatrixDiagD)
    .INPUT(x, TensorType::BasicType())
    .INPUT(assist, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixDiagD)

/**
*@brief: Returns the batched diagonal part of a batched tensor.

*@par Inputs:
*x: A Tensor. Must be one of the following types: float16, float32, int32, int8, uint8.

*@par Outputs:
*y: A Tensor. Has the same type as "x".

*/
REG_OP(MatrixDiagPart)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixDiagPart)

/**
*@brief: Returns the batched diagonal part of a batched tensor.

*@par Inputs:
* Two inputs, including:
*@li x: A Tensor. Must be one of the following types: float16, float32, int32, int8, uint8.
*@li assist: A Tensor of the same type as "x".

*@par Outputs:
*y: A Tensor. Has the same type as "x".

*/
REG_OP(MatrixDiagPartD)
    .INPUT(x, TensorType::BasicType())
    .INPUT(assist, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixDiagPartD)

/**
*@brief: Returns a batched matrix tensor with new batched diagonal values.

*@par Inputs:
* Two inputs, including:
*@li x: A Tensor. Must be one of the following types: float16, float32, int32, int8, uint8.
*@li diagonal: A Tensor of the same type as "x".

*@par Outputs:
*y: A Tensor. Has the same type as "x".

*/
REG_OP(MatrixSetDiag)
    .INPUT(x, TensorType::BasicType())
    .INPUT(diagonal, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixSetDiag)

/**
*@brief: Returns a batched matrix tensor with new batched diagonal values.

*@par Inputs:
* Three inputs, including:
*@li x: A Tensor. Must be one of the following types: float16, float32, int32, int8, uint8.
*@li diagonal: A Tensor of the same type as "x".
*@li assist: A Tensor of the same type as "x".

*@par Outputs:
*y: A Tensor. Has the same type as "x".

*/
REG_OP(MatrixSetDiagD)
    .INPUT(x, TensorType::BasicType())
    .INPUT(diagonal, TensorType::BasicType())
    .INPUT(assist, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixSetDiagD)

/**
*@brief Applies sparse "updates" to individual values or slices in a Variable.

*@par Inputs:
* Three inputs, including:
*@li var: An ND Tensor. \n

*Must be one of the following types: float16, float32, int8, uint8, bool
*@li indices: An ND Tensor. \n

*Must be one of the following types: int32
*@li updates: An ND Tensor. \n

*Must be one of the following types: float16, float32, int8, uint8, bool

*@par Attributes:
*use_locking: An optional bool. Defaults to "False". If "True", the operation will be protected by a lock.

*@par Outputs:
*var: A Tensor. Has the same type and format as input "var".

*/
REG_OP(ScatterNdUpdate)
    .INPUT(var, TensorType::BasicType())
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType::BasicType())
    .OUTPUT(var,  TensorType::BasicType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterNdUpdate)

/**
*@brief Applies sparse addition to individual values or slices in a Variable.

*@par Inputs:
* Three inputs, including:
*@li x: An ND Tensor. \n

*Must be one of the following types: float16, float32, int32, int8, uint8
*@li indices: An ND Tensor. \n

*Must be one of the following types: int32
*@li updates: An ND Tensor. \n

*Must be one of the following types: float16, float32, int32, int8, uint8

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x".

*/
REG_OP(TensorScatterUpdate)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OP_END_FACTORY_REG(TensorScatterUpdate)

/**
*@brief Adds sparse "updates" to a variable reference.

*@par Inputs:
* Three inputs, including:
*@li var: An ND Tensor. \n

*Must be one of the following types: float16, float32, int32, int8, uint8
*@li indices: An ND Tensor of type int32.


*@li updates: An ND Tensor. \n

*Must be one of the following types: float16, float32, int32, int8, uint8

*@par Attributes:
*use_locking: An optional bool. Defaults to "False". If "True", the operation will be protected by a lock.

*@par Outputs:
*var: A Tensor. Has the same type and format as input "var".

*/
REG_OP(ScatterAdd)
    .INPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterAdd)

/**
*@brief Divides a variable reference by sparse updates.

*@par Inputs:
* Three inputs, including:
*@li var: An NCHW, NHWC, or ND Tensor. \n

*Must be one of the following types: float16, float32, int32, int8, uint8
*@li indices: An NCHW, NHWC, or ND Tensor. \n

*Must be one of the following types: int32
*@li updates: An NCHW, NHWC, or ND Tensor. \n

*Must be one of the following types: float16, float32, int32, int8, uint8

*@par Attributes:
*@li use_locking: An optional bool. Defaults to "False". If "True", the operation will be protected by a lock.
*@li isRef: An optional bool. Defaults to "True"

*@par Outputs:
*var: A Tensor. Has the same type and format as input "var".

*/
REG_OP(ScatterDiv)
    .INPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterDiv)

/**
*@brief Applies sparse addition to individual values or slices in a Variable.

*@par Inputs:
* Three inputs, including:
*@li var: An ND Tensor. \n

*Must be one of the following types: float16, float32, int32, int8, uint8
*@li indices: An ND Tensor. \n

*Must be one of the following types: int32
*@li updates: An ND Tensor. \n

*Must be one of the following types: float16, float32, int32, int8, uint8

*@par Attributes:
*use_locking: An optional bool. Defaults to "False". If "True", the operation will be protected by a lock.

*@par Outputs:
*var: A Tensor. Has the same type and format as input "var".

*/
REG_OP(ScatterNdAdd)
    .INPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterNdAdd)

/**
*@brief Applies sparse addition to individual values or slices in a Variable.

*@par Inputs:
* Three inputs, including:
*@li x: An ND Tensor. \n

*Must be one of the following types: float16, float32, int32, int8, uint8
*@li indices: An ND Tensor. \n

*Must be one of the following types: int32
*@li updates: An ND Tensor. \n

*Must be one of the following types: float16, float32, int32, int8, uint8

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x".

*/
REG_OP(TensorScatterAdd)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OP_END_FACTORY_REG(TensorScatterAdd)

/**
*@brief Applies sparse subtraction to individual values or slices in a Variable.

*@par Inputs:
* Three inputs, including:
*@li var: An ND Tensor. \n

*Must be one of the following types: float16, float32, int32, int8, uint8
*@li indices: An ND Tensor. \n

*Must be one of the following types: int32
*@li updates: An ND Tensor. \n

*Must be one of the following types: float16, float32, int32, int8, uint8

*@par Attributes:
*use_locking: An optional bool. Defaults to "False". If "True", the operation will be protected by a lock.

*@par Outputs:
*var: A Tensor. Has the same type and format as input "var".

*/
REG_OP(ScatterNdSub)
    .INPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterNdSub)

/**
*@brief Applies sparse addition to individual values or slices in a Variable.

*@par Inputs:
* Three inputs, including:
*@li x: An ND Tensor. \n

*Must be one of the following types: float16, float32, int32, int8, uint8
*@li indices: An ND Tensor. \n

*Must be one of the following types: int32
*@li updates: An ND Tensor. \n

*Must be one of the following types: float16, float32, int32, int8, uint8

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x".

*/
REG_OP(TensorScatterSub)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OP_END_FACTORY_REG(TensorScatterSub)

/**
*@brief Subtracts sparse updates to a variable reference.

*@par Inputs:
* Three inputs, including:
*@li var: An ND Tensor. \n

*Must be one of the following types: float16, float32, int32, int8, uint8
*@li indices: An ND Tensor. \n

*Must be one of the following types: int32
*@li updates: An ND Tensor. \n

*Must be one of the following types: float16, float32, int32, int8, uint8

*@par Attributes:
*use_locking: An optional bool. Defaults to "False". If "True", the operation will be protected by a lock.

*@par Outputs:
*var: A Tensor. Has the same type and format as input "var".

*/
REG_OP(ScatterSub)
    .INPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterSub)

/**
*@brief: Returns the batched diagonal part of a batched tensor with "assist".

*@par Inputs:
* Two inputs, including:
* @li x: A Tensor of type float16, float32, or int32.
* @li assist: A Tensor of the same type as "x".

*@par Outputs:
*y: A Tensor. Has the same type as "x".

*/
REG_OP(DiagPartD)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(assist, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(DiagPartD)

/**
*@brief: Returns the batched diagonal part of a batched tensor.

*@par Inputs:\n
*x: A Tensor. Must be one of the following types: float16, float32, int32, int64, double, complex64, complex128.

*@par Outputs:
*y: A Tensor. Has the same type as "x".

*/
REG_OP(DiagPart)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT64, DT_DOUBLE,
                          DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT64, DT_DOUBLE,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(DiagPart)

/**
*@brief Also known as a "fully-connected" layer, computes an inner product with a set of learned weights, and (optionally) adds biases.

*@par Inputs:
* Four inputs, including:
*@li x: A Tensor of type float16, int8.
*@li w: A weight matrix of type float16, int8.
*@li b: A Tensor of type float16, int32, float32.
*@li offset_w: A Tensor of type int8.

*@par Attributes:
*@li num_output: Reserved.
*@li transpose: A bool, specifying whether to transpose, either "true" or "false". Defaults to "false".
*@li axis: Reserved.
*@li offset_x: Reserved.

*@par Outputs:
*y: The result tensor of type float16, int8, float32.

*@par Quantization supported or not
* Yes
*/
REG_OP(FullyConnection)
    .INPUT(x, TensorType({DT_FLOAT16, DT_INT8}))
    .INPUT(w, TensorType({DT_FLOAT16, DT_INT8}))
    .OPTIONAL_INPUT(b, TensorType({DT_FLOAT16, DT_INT32,DT_FLOAT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT32,DT_FLOAT32}))
    .REQUIRED_ATTR(num_output, Int)
    .ATTR(transpose, Bool, false)
    .ATTR(axis, Int, 1)
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(FullyConnection)

/**
*@brief Computes the confusion matrix from predictions and labels.

*@par Inputs:
* Three inputs, including:
*@li labels: A Tensor. Must be one of the following types: float16, float32, int32, int8.
*@li predictions: A Tensor. Must be one of the following types: float16, float32, int32, int8.
*@li weights: A Tensor. Must be one of the following types: float16, float32, int32, int8.

*@par Attributes:
*@li num_classes: An integer for the shape of the output matrix. No default value.
*@li dtype: Data type of the confusion matrix. No default value.

*@par Outputs:
*y: A Tensor. Has the same type and format as input "labels"

*@attention Constraints:
*@li "weights", "labels", and "predictions" are 1D tensors.
*@li The output is with shape (num_classes, num_classes), where, 1 <= num_classes <= 4096.

*@see Region()

*/
REG_OP(ConfusionMatrix)
    .INPUT(labels, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16, DT_INT8, DT_UINT8}))
    .INPUT(predictions, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16, DT_INT8, DT_UINT8}))
    .OPTIONAL_INPUT(weights, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16, DT_INT8, DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16, DT_INT8, DT_UINT8}))
    .REQUIRED_ATTR(num_classes, Int)
    .REQUIRED_ATTR(dtype, String)
    .OP_END_FACTORY_REG(ConfusionMatrix)

/**
*@brief Multiplies sparse updates into a variable reference.

*@par Inputs:
* Three inputs, including:
*@li var: An NCHW, NHWC, or ND Tensor. \n

*Must be one of the following types: float16, float32, int32, int8, uint8
*@li indices: An NCHW, NHWC, or ND Tensor. \n

*Must be one of the following types: int32
*@li updates: An NCHW, NHWC, or ND Tensor. \n

*Must be one of the following types: float16, float32, int32, int8, uint8

*@par Attributes:
*use_locking: An optional bool. Defaults to "False". If "True", the operation will be protected by a lock.

*@par Outputs:
*var: A Tensor. Has the same type and format as input "var".

*/
REG_OP(ScatterMul)
    .INPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(updates, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterMul)

/**
*@brief Reduces sparse updates into a variable reference using the "min" operation.

*@par Inputs:
* Three inputs, including:
*@li var: An NCHW, NHWC, or ND Tensor. \n

*Must be one of the following types: float16, float32, int32
*@li indices: An NCHW, NHWC, or ND Tensor. \n

*Must be one of the following types: int32
*@li updates: An NCHW, NHWC, or ND Tensor. \n

*Must be one of the following types: float16, float32, int32

*@par Attributes:
*use_locking: An optional bool. Defaults to "False". If "True", the operation will be protected by a lock.

*@par Outputs:
*var: A Tensor. Has the same type and format as input "var".

*/
REG_OP(ScatterMin)
    .INPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(updates, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .OUTPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterMin)

/**
*@brief Reduces sparse updates into a variable reference using the "max" operation.

*@par Inputs:
* Three inputs, including:
*@li var: An NCHW, NHWC, or ND Tensor. \n

*Must be one of the following types: float16, float32, int32
*@li indices: An NCHW, NHWC, or ND Tensor. \n

*Must be one of the following types: int32
*@li updates: An NCHW, NHWC, or ND Tensor. \n

*Must be one of the following types: float16, float32, int32

*@par Attributes:
*use_locking: An optional bool. Defaults to "False". If "True", the operation will be protected by a lock.

*@par Outputs:
*var: A Tensor. Has the same type and format as input "var".

*/
REG_OP(ScatterMax)
    .INPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(updates, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .OUTPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterMax)

/**
*@brief Applies sparse updates to a variable reference.

*@par Inputs:
* Three inputs, including:
*@li var: An NCHW, NHWC, or ND Tensor. \n

*Must be one of the following types: float16, float32, int32, int8, uint8
*@li indices: An NCHW, NHWC, or ND Tensor. \n

*Must be one of the following types: int32
*@li updates: An NCHW, NHWC, or ND Tensor. \n

*Must be one of the following types: float16, float32, int32, int8, uint8

*@par Attributes:
*use_locking: An optional bool. Defaults to "False". If "True", the operation will be protected by a lock.

*@par Outputs:
*var: A Tensor. Has the same type and format as input "var".

*/
REG_OP(ScatterUpdate)
    .INPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(updates, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterUpdate)

/**
*@brief Returns a tensor with the `k[0]`-th to `k[1]`-th diagonals of the batched `input`.

*@par Inputs:
* Three inputs, including:
*@li input: Rank `r` tensor where `r >= 2`. \n

*@li k: \n
*Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main \n
*diagonal, and negative value means subdiagonals. `k` can be a single integer \n
*(for a single diagonal) or a pair of integers specifying the low and high ends \n
*of a matrix band. `k[0]` must not be larger than `k[1]`. \n

*@li padding_value: The value to fill the area outside the specified diagonal band with. \n

*@par Outputs:
*diagonal: The extracted diagonal(s).

*/
REG_OP(MatrixDiagPartV2)
    .INPUT(input, TensorType::BasicType())
    .INPUT(k, TensorType({DT_INT32}))
    .INPUT(padding_value, TensorType::BasicType())
    .OUTPUT(diagonal, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixDiagPartV2)

/**
*@brief Returns a batched matrix tensor with new batched diagonal values.

*@par Inputs:
* Three inputs, including:
*@li input: "Rank `r+1`, where `r >= 1`. \n

*@li diagonal: Rank `r` when `k` is an integer or `k[0] == k[1]`. Otherwise, it has rank `r+1`. \n

*@li k:
*Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main \n
*diagonal, and negative value means subdiagonals. `k` can be a single integer \n
*(for a single diagonal) or a pair of integers specifying the low and high ends \n
*of a matrix band. `k[0]` must not be larger than `k[1]`. \n

*@par Outputs:
*output: Rank `r+1`, with `output.shape = input.shape`.

*/
REG_OP(MatrixSetDiagV2)
    .INPUT(input, TensorType::BasicType())
    .INPUT(diagonal, TensorType::BasicType())
    .INPUT(k, TensorType({DT_INT32}))
    .OUTPUT(output, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixSetDiagV2)

/**
*@brief Returns a batched diagonal tensor with given batched diagonal values.

*@par Inputs:
* Five inputs, including:
*@li diagonal: Rank `r`, where `r >= 1` \n

*@li k:
*Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main \n
*diagonal, and negative value means subdiagonals. `k` can be a single integer \n
*(for a single diagonal) or a pair of integers specifying the low and high ends \n
*of a matrix band. `k[0]` must not be larger than `k[1]`. \n

*@li num_rows:
*The number of rows of the output matrix. If it is not provided, the op assumes \n
*the output matrix is a square matrix and infers the matrix size from k and the \n
*innermost dimension of `diagonal`. \n

*@li num_cols: An NCHW, NHWC, or ND Tensor.
*The number of columns of the output matrix. If it is not provided, the op \n
*assumes the output matrix is a square matrix and infers the matrix size from \n
*k and the innermost dimension of `diagonal`. \n

*@li padding_value: The number to fill the area outside the specified diagonal band with. \n

*@par Outputs:
*output: Has rank `r+1` when `k` is an integer or `k[0] == k[1]`, rank `r` otherwise.

*/
REG_OP(MatrixDiagV2)
    .INPUT(diagonal, TensorType::BasicType())
    .INPUT(k, TensorType({DT_INT32}))
    .INPUT(num_rows, TensorType({DT_INT32}))
    .INPUT(num_cols, TensorType({DT_INT32}))
    .INPUT(padding_value, TensorType::BasicType())
    .OUTPUT(output, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixDiagV2)

}  // namespace ge

#endif  // GE_OP_MATRIX_CALCULATION_OPS_H
