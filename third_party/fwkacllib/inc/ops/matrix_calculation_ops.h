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

/*!
 * \file matrix_calculation_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_MATRIX_CALCULATION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_MATRIX_CALCULATION_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Multiplies matrix "a" by matrix "b", producing "a * b" . \n

*@par Inputs:
*Three inputs, including:
* @li x1: A matrix Tensor. 2D. Must be one of the following types: float16,
* float32, int32. Has format [ND, NHWC, FRACTAL_NZ].
* @li x2: A matrix Tensor. 2D. Must be one of the following types: float16,
* float32, int32. Has format [ND, NHWC, FRACTAL_NZ].
* @li bias: A optional 1D Tensor. Must be one of the following types: float16,
* float32, int32. Has format [ND, NHWC] . \n

*@par Attributes:
*@li transpose_x1: A bool. If True, changes the shape of "x1" from [M, K] to [K, M].
*@li transpose_x2: A bool. If True, changes the shape of "x2" from [M, K] to [K, M] . \n

*@par Outputs:
*y: The result matrix Tensor. 2D. Must be one of the following types: float16,
* float32, int32. Has format [ND, NHWC, FRACTAL_NZ] . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator BatchMatmul.
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
*@brief Multiplies matrix "a" by matrix "b", producing "a * b" . \n

*@par Inputs:
*Two inputs, including:
* @li x1: A matrix Tensor. 2D. Must be one of the following types: float16,
* float32, int32. Has format [ND, NHWC, FRACTAL_NZ].
* @li x2: A matrix Tensor. 2D. Must be one of the following types: float16,
* float32, int32. Has format [ND, NHWC, FRACTAL_NZ].
* @li bias: A 1D Tensor. Must be one of the following types: float16,
* float32, int32. Has format [ND, NHWC] . \n

*@par Attributes:
*@li transpose_x1: A bool. If True, changes the shape of "x1" from [M, K] to [K, M].
*@li transpose_x2: A bool. If True, changes the shape of "x2" from [M, K] to [K, M] . \n

*@par Outputs:
*y: The result matrix Tensor. 2D. Must be one of the following types: float16,
* float32, int32. Has format [ND, NHWC, FRACTAL_NZ] . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator BatchMatmul.
*/
REG_OP(MatMulV2)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(MatMulV2)


/**
*@brief Performs Matrix-to-matrix Multiply, producing c=alpha[0]*a*b+beta[0]*c . \n

*@attention Constraints:
* For better performance, The k-axis must be aligned to 16 (input type
* is float16) or 32 (input type is int8). \n

*@par Inputs:
*Five inputs, including:
*@li a: A matrix Tensor. Must be one of the following types: float16, int8.
* Has format [ND, FRACTAL_NZ]. 2D(ND) or 4D(FRACTAL_NZ).
*@li b: A matrix Tensor. Must be one of the following types: float16, int8.
* Has format [ND, FRACTAL_NZ, FRACTAL_Z]. 2D(ND) or 4D(FRACTAL_NZ, FRACTAL_Z).
*@li c: A matrix Tensor. Must be one of the following types: float16, int32,
* float32. has format [ND, FRACTAL_NZ]. 2D(ND) or 4D(FRACTAL_NZ).
*@li alpha: A 1D Tensor. The shape of alpha is [1].Must be one of the following
* types: float16, int32, float32. Has format [ND].
*@li beta: A 1D Tensor. The shape of beta is [1]. Must be one of the following
* types: float16, int32, float32. Has format [ND].
* The format of a, b, c has restriction:\n
* When type of a is int8 and type of c is int32, the format of a, b, c should
* all be ND, or a is FRACTAL_NZ and b is FRACTAL_Z and c is ND.\n
* When type of a is int8 and type of c is float32, the format of a, b, c should
* all be ND or a is FRACTAL_NZ and b is FRACTAL_Z and c is FRACTAL_NZ.\n
* When type of a is float16 and type of c is float16, the format of a, b, c
* should all be ND or FRACTAL_NZ.\n
* When type of a is float16 and type of c is float32, the format of a, b, c
* should all be ND or FRACTAL_NZ . \n

*@par Attributes:
*Two attributes, including:
*@li transpose_a: Optional. A bool. If True, changes the shape of "a" from
* [M, K] to [K, M].
*@li transpose_b: Optional. A bool. If True, changes the shape of "b" from
* [K, N] to [N, K] . \n

*@par Outputs:
*y: The result matrix Tensor. Must be one of the following types: float16,
* float32, int32. Has format [ND, FRACTAL_NZ], the format should be equal to a.
* 2D(ND) or 4D(FRACTAL_NZ).
*/

REG_OP(GEMM)
    .INPUT(a, TensorType({DT_FLOAT16, DT_INT8}))
    .INPUT(b, TensorType({DT_FLOAT16, DT_INT8}))
    .INPUT(c, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(alpha, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(beta, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .ATTR(transpose_a, Bool, false)
    .ATTR(transpose_b, Bool, false)
    .OP_END_FACTORY_REG(GEMM)

/**
*@brief Multiplies matrix "a" by matrix "b", producing "a * b" . \n

*@par Inputs:
*Three inputs, including:
* @li x1: A matrix Tensor. Must be one of the following types: float16,
* float32, int32. 2D or higher. Has format [ND, NHWC, FRACTAL_NZ].
* @li x2: A matrix Tensor. Must be one of the following types: float16,
* float32, int32. 2D or higher. Has format [ND, NHWC, FRACTAL_NZ] . \n

*@par Attributes:
*@li adj_x1: A bool. If True, changes the shape of "x1" from [B, M, K] to [B, K, M].
*@li adj_x2: A bool. If True, changes the shape of "x2" from [B, M, K] to [B, K, M] . \n

*@par Outputs:
*y: The result matrix Tensor. 2D or higher. Must be one of the following types: float16,
* float32, int32. 2D or higher. Has format [ND, NHWC, FRACTAL_NZ]. Has the same shape length as "x1" and "x2" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator BatchMatmul.
*/

REG_OP(BatchMatMul)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .ATTR(adj_x1, Bool, false)
    .ATTR(adj_x2, Bool, false)
    .OP_END_FACTORY_REG(BatchMatMul)


/**
* @brief Multiplies matrix "a" by matrix "b", producing "a * b" . \n

* @par Inputs:
* Three inputs, including:
* @li x1: A matrix Tensor. Must be one of the following types: float16,
* float32, int32. 2D or higher. Has format [ND, NHWC, FRACTAL_NZ].
* @li x2: A matrix Tensor. Must be one of the following types: float16,
* float32, int32. 2D or higher. Has format [ND, NHWC, FRACTAL_NZ] . \n
* @li bias: A matrix Tensor. Must be one of the following types: float16,
* float32, int32. 2D or higher. Has format [ND, NHWC, FRACTAL_NZ] . \n

* @par Attributes:
* @li adj_x: A bool. If True, changes the shape of "x1" from [B, M, K] to [B, K, M].
* @li adj_y: A bool. If True, changes the shape of "x2" from [B, M, K] to [B, K, M] . \n

* @par Outputs:
* y: The result matrix Tensor. 2D or higher. Must be one of the following types: float16,
* float32, int32. 2D or higher. Has format [ND, NHWC, FRACTAL_NZ]. Has the same shape length as "x1" and "x2" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator BatchMatmul.
*/

REG_OP(BatchMatMulV2)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .ATTR(adj_x1, Bool, false)
    .ATTR(adj_x2, Bool, false)
    .OP_END_FACTORY_REG(BatchMatMulV2)


/**
*@brief Computes half the L2 norm of a tensor without the sqrt . \n

*@par Inputs:

* x: A Tensor.
*     TensorType::FloatingDataType() . \n

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator L2Loss.
*/
REG_OP(L2Loss)
    .INPUT(x, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(L2Loss)

/**
*@brief: Returns a batched diagonal tensor with a given batched diagonal values . \n

*@par Inputs:
*x: A Tensor. Must be one of the following types:
*   float16, float32, double, int32, uint8, int16, int8, complex64, int64,
*   qint8, quint8, qint32, uint16, complex128, uint32, uint64 . \n

*@par Outputs:
*y: A Tensor. Has the same type as "x" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator MatrixDiag.
*/
REG_OP(MatrixDiag)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixDiag)

/**
*@brief: Returns a batched diagonal tensor with a given batched diagonal values . \n

*@par Inputs:
* Two inputs, including:
*@li x: A Tensor. Must be one of the following types: float16, float32, int32, int8, uint8.
*@li assist: A Tensor of the same type as "x" . \n

*@par Outputs:
*y: A Tensor. Has the same type as "x" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator MatrixDiag.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use MatrixDiag instead.
*/
REG_OP(MatrixDiagD)
    .INPUT(x, TensorType::BasicType())
    .INPUT(assist, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixDiagD)

/**
*@brief: Returns the batched diagonal part of a batched tensor . \n

*@par Inputs:
*x: A Tensor. Must be one of the following types:
*   float16, float32, double, int32, uint8, int16, int8, complex64, int64,
*   qint8, quint8, qint32, uint16, complex128, uint32, uint64 . \n

*@par Outputs:
*y: A Tensor. Has the same type as "x" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator MatrixDiagPart.
*/
REG_OP(MatrixDiagPart)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixDiagPart)

/**
*@brief: Returns the batched diagonal part of a batched tensor . \n

*@par Inputs:
* Two inputs, including:
*@li x: A Tensor. Must be one of the following types: float16, float32, int32, int8, uint8.
*@li assist: A Tensor of the same type as "x" . \n

*@par Outputs:
*y: A Tensor. Has the same type as "x" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator MatrixDiagPart.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use MatrixDiagPart instead.
*/
REG_OP(MatrixDiagPartD)
    .INPUT(x, TensorType::BasicType())
    .INPUT(assist, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixDiagPartD)

/**
*@brief: Returns a batched matrix tensor with new batched diagonal values . \n

*@par Inputs:
* Two inputs, including:
*@li x: A Tensor. Must be one of the following types:
*    float16, float32, double, int32, uint8, int16, int8, complex64, int64,
*    qint8, quint8, qint32, uint16, complex128, uint32, uint64.
*@li diagonal: A Tensor of the same type as "x" . \n

*@par Outputs:
*y: A Tensor. Has the same type as "x" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator MatrixSetDiag.
*/
REG_OP(MatrixSetDiag)
    .INPUT(x, TensorType::BasicType())
    .INPUT(diagonal, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixSetDiag)

/**
*@brief: Returns a batched matrix tensor with new batched diagonal values . \n

*@par Inputs:
* Three inputs, including:
*@li x: A Tensor. Must be one of the following types: float16, float32, int32, int8, uint8.
*@li diagonal: A Tensor of the same type as "x".
*@li assist: A Tensor of the same type as "x" . \n

*@par Outputs:
*y: A Tensor. Has the same type as "x" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator MatrixSetDiag.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use MatrixSetDiag instead.
*/
REG_OP(MatrixSetDiagD)
    .INPUT(x, TensorType::BasicType())
    .INPUT(diagonal, TensorType::BasicType())
    .INPUT(assist, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixSetDiagD)

/**
*@brief Applies sparse "updates" to individual values or slices in a Variable . \n

*@par Inputs:
* Three inputs, including:
*@li var: An ND Tensor.
*Must be one of the following types: float16, float32, int8, uint8, double,
 * int64, complex64, qint8, quint8, qint32, uint16, complex128, half, uint32,
 * uint64
*@li indices: An ND Tensor.
*Must be one of the following types: int32, int64
*@li updates: An ND Tensor.
*Must be one of the following types: float16, float32, int8, uint8, double,
 * int64, complex64, qint8, quint8, qint32, uint16, complex128, half, uint32,
 * uint64

*@par Attributes:
*use_locking: An optional bool. Defaults to "False". If "True",
 * the operation will be protected by a lock . \n

*@par Outputs:
*var: A Tensor. Has the same type and format as input "var" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterNdUpdate.
*/
REG_OP(ScatterNdUpdate)
    .INPUT(var, TensorType::BasicType())
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType::BasicType())
    .OUTPUT(var,  TensorType::BasicType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterNdUpdate)

/**
*@brief Applies sparse addition to individual values or slices in a Variable . \n

*@par Inputs:
* Three inputs, including:
*@li x: An ND Tensor. \n

*Must be one of the following types: float16, float32, bool, int8, uint8
*@li indices: An ND Tensor. \n

*Must be one of the following types: int32
*@li updates: An ND Tensor. \n

*Must be one of the following types: float16, float32, bool, int8, uint8

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator TensorScatterUpdate.
*/
REG_OP(TensorScatterUpdate)
    .INPUT(x, TensorType::BasicType())
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(TensorScatterUpdate)

/**
*@brief Adds sparse "updates" to a variable reference . \n

*@par Inputs:
* Three inputs, including:
*@li var: An ND Tensor . \n

*Must be one of the following types: float16, float32, int32, int8, uint8
*@li indices: An ND Tensor of type int32 or int64.


*@li updates: An Tensor. format:NCHW, NHWC . \n

*Must be one of the following types: float16, float32, int32, int8, uint8

*@par Attributes:
* use_locking: An optional bool. Defaults to "False". If "True", the operation
* will be protected by a lock . \n

*@par Outputs:
*var: A Tensor. Has the same type and format as input "var" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterAdd.
*/
REG_OP(ScatterAdd)
    .INPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterAdd)

/**
*@brief Divides a variable reference by sparse updates . \n

*@par Inputs:
* Three inputs, including:
*@li var: An ND Tensor.
*Must be one of the following types: float16, float, int32, int8, uint8

*@li indices: An ND Tensor.
*Must be one of the following types: int32
*@li updates: An ND Tensor.
*Must be one of the following types: float16, float, int32, int8, uint8

*@par Attributes:
*@li use_locking: An optional bool. Defaults to "False". If "True",
* the operation will be protected by a lock . \n

*@par Outputs:
*var: A Tensor. Has the same type and format as input "var" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterDiv.
*/
REG_OP(ScatterDiv)
    .INPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterDiv)

/**
*@brief Applies sparse addition to individual values or slices in a Variable . \n

*@par Inputs:
* Three inputs, including:
*@li var: An ND Tensor.
*Must be one of the following types: float16, float, int32, int8, uint8
*@li indices: An ND Tensor.
*Must be one of the following types: int32
*@li updates: An ND Tensor.
*Must be one of the following types: float16, float, int32, int8, uint8
*@par Attributes:
*use_locking: An optional bool. Defaults to "False". If "True",
* the operation will be protected by a lock . \n

*@par Outputs:
*var: A Tensor. Has the same type and format as input "var" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterNdAdd.
*/
REG_OP(ScatterNdAdd)
    .INPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterNdAdd)

/**
*@brief Applies sparse addition to individual values or slices in a Variable . \n

*@par Inputs:
* Three inputs, including:
*@li x: An ND Tensor. \n

*Must be one of the following types: float16, float32, int32, int8, uint8
*@li indices: An ND Tensor. \n

*Must be one of the following types: int32
*@li updates: An ND Tensor. \n

* Must be one of the following types: float16, float32, int32, int8, uint8

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator TensorScatterAdd.
*/
REG_OP(TensorScatterAdd)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OP_END_FACTORY_REG(TensorScatterAdd)

/**
*@brief Applies sparse subtraction to individual values or slices in a Variable . \n

*@par Inputs:
* Three inputs, including:
*@li var: An ND Tensor.
*Must be one of the following types: float16, float, int32, int8, uint8
*@li indices: An ND Tensor.
*Must be one of the following types: int32, int64
*@li updates: An ND Tensor.
*Must be one of the following types: float16, float, int32, int8, uint8

*@par Attributes:
*use_locking: An optional bool. Defaults to "False". If "True",
* the operation will be protected by a lock . \n

*@par Outputs:
* var: A Tensor. Has the same type and format as input "var" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterNdSub.
*/
REG_OP(ScatterNdSub)
    .INPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterNdSub)

/**
*@brief Applies sparse addition to individual values or slices in a Variable . \n

*@par Inputs:
* Three inputs, including:
*@li x: An ND Tensor. \n

*Must be one of the following types: float16, float32, int32, int8, uint8
*@li indices: An ND Tensor. \n

*Must be one of the following types: int32
*@li updates: An ND Tensor. \n

*Must be one of the following types: float16, float32, int32, int8, uint8

*@par Outputs:
* y: A Tensor. Has the same type and format as input "x" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator TensorScatterSub.
*/
REG_OP(TensorScatterSub)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OP_END_FACTORY_REG(TensorScatterSub)

/**
*@brief Subtracts sparse updates to a variable reference . \n

*@par Inputs:
* Three inputs, including:
*@li var: An ND Tensor.
*Must be one of the following types: float16, float, int32, int8, uint8
*@li indices: An ND Tensor.
*Must be one of the following types: int32, int64
*@li updates: An ND Tensor.
*Must be one of the following types: float16, float, int32, int8, uint8
*@par Attributes:
*use_locking: An optional bool. Defaults to "False". If "True",
* the operation will be protected by a lock . \n

*@par Outputs:
* var: A Tensor. Has the same type and format as input "var" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterSub.
*/
REG_OP(ScatterSub)
    .INPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterSub)

/**
*@brief: Returns the batched diagonal part of a batched tensor with "assist" . \n

*@par Inputs:
* Two inputs, including:
* @li x: A Tensor of type float16, float32, or int32.
* @li assist: A Tensor of the same type as "x" . \n

*@par Outputs:
*y: A Tensor. Has the same type as "x" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator DiagPart.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use DiagPart instead.
*/
REG_OP(DiagPartD)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(assist, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(DiagPartD)

/**
*@brief: Returns the batched diagonal part of a batched tensor . \n

*@par Inputs:
*x: A Tensor. Must be one of the following types:
*    float16, float32, int32, int64, double, complex64, complex128 . \n

*@par Outputs:
*y: A Tensor. Has the same type as "x" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator DiagPart.
*/
REG_OP(DiagPart)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT64, DT_DOUBLE,
                          DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT64, DT_DOUBLE,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(DiagPart)

/**
*@brief Also known as a "fully-connected" layer, computes an inner product with a set of learned weights, and (optionally) adds biases . \n

*@par Inputs:
* Four inputs, including:
*@li x: A Tensor of type float16, int8.
*@li w: A weight matrix of type float16, int8.
*@li b: A Tensor of type float16, int32, float32.
*@li offset_w: A Tensor of type int8 . \n

*@par Attributes:
*@li num_output: Reserved.
*@li transpose: A bool, specifying weight whether to transpose, either "true" or "false". Defaults to "false".
*@li axis: Optional. A int, 1 or 2, specifying which dimension the input "K" starts from. Defaults to 1.
* The product of the subsequent dimensions starting form first dimension or the second dimension is "K".
*@li offset_x: Reserved . \n

*@par Outputs:
*y: The result tensor of type float16, int32, float32 . \n

*@par Third-party framework compatibility
* Compatible with the Caffe operator InnerProduct . \n

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
*@brief Also known as a "fully-connected-compress" layer, computes an inner product with a set of learned weights, and (optionally) adds biases . \n

*@par Inputs:
* Four inputs, including:
*@li x: A Tensor of type uint8, int8.
*@li w: A weight matrix of type int8, int8.
*@li w: A compress index matrix of type int8, int8.
*@li b: A Tensor of type float16, int32, int32.
*@li offset_w: A Tensor of type int8.i

*@par Attributes:
*@li num_output: Reserved.
*@li transpose: A bool, specifying whether to transpose, either "true" or "false". Defaults to "false".
*@li axis: Reserved.
*@li offset_x: Reserved . \n

*@par Outputs:
*y: The result tensor of type int32 . \n

*@par Third-party framework compatibility
* Compatible with the Caffe operator InnerProduct . \n

*@par Quantization supported or not
* Yes
*/
REG_OP(FullyConnectionCompress)
    .INPUT(x, TensorType({DT_UINT8, DT_INT8}))
    .INPUT(w, TensorType({DT_INT8}))
    .INPUT(comress_index, TensorType({DT_INT8}))
    .OPTIONAL_INPUT(b, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .REQUIRED_ATTR(num_output, Int)
    .ATTR(transpose, Bool, false)
    .ATTR(axis, Int, 1)
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(FullyConnectionCompress)

/**
*@brief Computes the confusion matrix from predictions and labels . \n

*@par Inputs:
* Three inputs, including:
*@li labels: A Tensor. Must be one of the following types: float16, float32,
* int32, int8, uint8.
*@li predictions: A Tensor. Must be one of the following types: float16,
* float32, int32, int8, uint8.
*@li weights: A Tensor. Must be one of the following types: float16, float32,
* int32, int8, uint8 . \n

*@par Attributes:
*@li num_classes: An integer for the shape of the output matrix.
* No default value.
*@li dtype: Data type of the confusion matrix. No default value . \n

*@par Outputs:
*y: A Tensor. Has the same type and format as input "labels"

*@attention Constraints:
*@li "weights", "labels", and "predictions" are 1D tensors.
*@li The output is with shape (num_classes, num_classes),
* where, 1 <= num_classes <= 4096 . \n

*@see Region()

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ConfusionMatrix.
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
*@brief Multiplies sparse updates into a variable reference . \n

*@par Inputs:
* Three inputs, including:
*@li var: An ND Tensor.
*Must be one of the following types: float16, float, int32, int8, uint8
*@li indices: An ND Tensor.
*Must be one of the following types: int32
*@li updates: An ND Tensor . \n

*Must be one of the following types: float16, float, int32, int8, uint8

*@par Attributes:
*use_locking: An optional bool. Defaults to "False". If "True", the operation
* will be protected by a lock . \n

*@par Outputs:
*var: A Tensor. Has the same type and format as input "var" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterMul.
*/
REG_OP(ScatterMul)
    .INPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(updates, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterMul)

/**
*@brief Reduces sparse updates into a variable reference using
 * the "min" operation . \n

*@par Inputs:
* Three inputs, including:
*@li var: An ND Tensor.
*Must be one of the following types: float16, float, int32

*@li indices: An ND Tensor.
*Must be one of the following types: int32

*@li updates: An ND Tensor.
*Must be one of the following types: float16, float, int32

*@par Attributes:
*use_locking: An optional bool. Defaults to "False". If "True", the operation
* will be protected by a lock . \n

*@par Outputs:
*var: A Tensor. Has the same type and format as input "var" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterMin.
*/
REG_OP(ScatterMin)
    .INPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(updates, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .OUTPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterMin)

/**
*@brief Reduces sparse updates into a variable reference using the "max" operation . \n

*@par Inputs:
* Three inputs, including:
*@li var: An ND Tensor . \n

*Must be one of the following types: float16, float, int32
*@li indices: An NCHW, NHWC, or ND Tensor . \n

*Must be one of the following types: int32
*@li updates: An NCHW, NHWC, or ND Tensor . \n

*Must be one of the following types: float16, float, int32

*@par Attributes:
*use_locking: An optional bool. Defaults to "False".
* If "True", the operation will be protected by a lock . \n

*@par Outputs:
*var: A Tensor. Has the same type and format as input "var" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterMax.
*/
REG_OP(ScatterMax)
    .INPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(updates, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .OUTPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterMax)

/**
*@brief Applies sparse updates to a variable reference . \n

*@par Inputs:
* Three inputs, including:
*@li var: An ND Tensor . \n

*Must be one of the following types: float16, float, int32, int8, uint8
*@li indices: An ND Tensor . \n

*Must be one of the following types: int32
*@li updates: An ND Tensor . \n

*Must be one of the following types: float16, float, int32, int8, uint8

*@par Attributes:
*use_locking: An optional bool. Defaults to "False". If "True",
* the operation will be protected by a lock . \n

*@par Outputs:
*var: A Tensor. Has the same type and format as input "var" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterUpdate.
*/
REG_OP(ScatterUpdate)
    .INPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(updates, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterUpdate)

/**
*@brief Returns a tensor with the `k[0]`-th to `k[1]`-th diagonals of the batched `input` . \n

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
*diagonal: The extracted diagonal(s) . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterUpdate.
*/
REG_OP(MatrixDiagPartV2)
    .INPUT(input, TensorType::BasicType())
    .INPUT(k, TensorType({DT_INT32}))
    .INPUT(padding_value, TensorType::BasicType())
    .OUTPUT(diagonal, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixDiagPartV2)

/**
*@brief Returns a batched matrix tensor with new batched diagonal values . \n

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
*output: Rank `r+1`, with `output.shape = input.shape` . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterUpdate.
*/
REG_OP(MatrixSetDiagV2)
    .INPUT(input, TensorType::BasicType())
    .INPUT(diagonal, TensorType::BasicType())
    .INPUT(k, TensorType({DT_INT32}))
    .OUTPUT(output, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixSetDiagV2)

/**
*@brief Returns a batched diagonal tensor with given batched diagonal values . \n

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
*output: Has rank `r+1` when `k` is an integer or `k[0] == k[1]`, rank `r` otherwise . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterUpdate.
*/
REG_OP(MatrixDiagV2)
    .INPUT(diagonal, TensorType::BasicType())
    .INPUT(k, TensorType({DT_INT32}))
    .INPUT(num_rows, TensorType({DT_INT32}))
    .INPUT(num_cols, TensorType({DT_INT32}))
    .INPUT(padding_value, TensorType::BasicType())
    .OUTPUT(output, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixDiagV2)

REG_OP(IndexAdd)
    .INPUT(var, TensorType({DT_INT32, DT_INT8, DT_UINT8, DT_FLOAT32, DT_FLOAT16}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(updates, TensorType({DT_INT32, DT_INT8, DT_UINT8, DT_FLOAT32, DT_FLOAT16}))
    .OUTPUT(var_out, TensorType({DT_INT32, DT_INT8, DT_UINT8, DT_FLOAT32, DT_FLOAT16}))
    .ATTR(axis, Int, 0)
    .OP_END_FACTORY_REG(IndexAdd)

}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_MATRIX_CALCULATION_OPS_H_
