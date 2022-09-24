/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
* @brief Backprop W of AttentionLnQKV + ReduceSumD \n
* @par Inputs:
* Four inputs, including:
* @li x: A Tensor. Must be one of the following types: float16.
* @li query_dx: A Tensor. Must be one of the following types: float16.
* @li key_dw: A Tensor. Must be one of the following types: float16.
* @li value_dw: A Tensor. Must be one of the following types: float16.

* @par Attributes:
* @li trans_a: A optional attribute, the type is bool. Defaults to True.
* @li trans_b: A optional attribute, the type is bool. Defaults to False. \n

* @par Outputs:
* Six outputs, including:
* @li dw_query: A Tensor. Must be one of the following types: float16.
* @li dw_key: A Tensor. Must be one of the following types: float16.
* @li dw_value: A Tensor. Must be one of the following types: float16.
* @li dbias_query: A Tensor. Must be one of the following types: float16.
* @li dbias_key: A Tensor. Must be one of the following types: float16.
* @li dbias_value: A Tensor. Must be one of the following types: float16. \n

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use. \n
*/
REG_OP(AttentionQKVGradW)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(query_dx, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(key_dw, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(value_dw, TensorType({DT_FLOAT16}))
    .OUTPUT(dw_query, TensorType({DT_FLOAT16}))
    .OUTPUT(dw_key, TensorType({DT_FLOAT16}))
    .OUTPUT(dw_value, TensorType({DT_FLOAT16}))
    .OUTPUT(dbias_query, TensorType({DT_FLOAT16}))
    .OUTPUT(dbias_key, TensorType({DT_FLOAT16}))
    .OUTPUT(dbias_value, TensorType({DT_FLOAT16}))
    .ATTR(trans_a, Bool, true)
    .ATTR(trans_b, Bool, false)
    .OP_END_FACTORY_REG(AttentionQKVGradW)

/**
* @brief Backprop X of AttentionLnQKV + AddN \n
* @par Inputs:
* Seven inputs, including:
* @li ln_dx: A Tensor. Must be one of the following types: float16.
* @li query_dx: A Tensor. Must be one of the following types: float16.
* @li key_dw: A Tensor. Must be one of the following types: float16.
* @li value_dw: A Tensor. Must be one of the following types: float16.
* @li kernel_query: A Tensor. Must be one of the following types: float16.
* @li kernel_key: A Tensor. Must be one of the following types: float16.
* @li kernel_value: A Tensor. Must be one of the following types: float16. \n

* @par Attributes:
* @li trans_a: A optional attribute, the type is bool. Defaults to False.
* @li trans_b: A optional attribute, the type is bool. Defaults to True. \n

* @par Outputs:
* One outputs, including:
* @li dx: A Tensor. Must be one of the following types: float16. \n

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use. \n
*/
REG_OP(AttentionQKVGradX)
    .INPUT(ln_dx, TensorType({DT_FLOAT16}))
    .INPUT(query_dx, TensorType({DT_FLOAT16}))
    .INPUT(key_dw, TensorType({DT_FLOAT16}))
    .INPUT(value_dw, TensorType({DT_FLOAT16}))
    .INPUT(kernel_query, TensorType({DT_FLOAT16}))
    .INPUT(kernel_key, TensorType({DT_FLOAT16}))
    .INPUT(kernel_value, TensorType({DT_FLOAT16}))
    .OUTPUT(dx, TensorType({DT_FLOAT16}))
    .ATTR(trans_a, Bool, false)
    .ATTR(trans_b, Bool, true)
    .OP_END_FACTORY_REG(AttentionQKVGradX)

/**
* @brief
             / (MatMul -> ConfusionTransposeD).
   LayerNorm - (MatMul -> ConfusionTransposeD).
             \ (MatMul -> ConfusionTransposeD). \n
* @par Inputs:
* Nine inputs, including:
* @li x: A Tensor. Must be one of the following types: float16.
* @li kernel_query: A Tensor. Must be one of the following types: float16.
* @li kernel_key: A Tensor. Must be one of the following types: float16.
* @li kernel_value: A Tensor. Must be one of the following types: float16.
* @li gamma: A Tensor. Must be one of the following types: float16.
* @li beta: A Tensor. Must be one of the following types: float16.
* @li bias_query: A Tensor. Must be one of the following types: float16.
* @li bias_key: A Tensor. Must be one of the following types: float16.
* @li bias_value: A Tensor. Must be one of the following types: float16. \n

* @par Attributes:
* @li epsilon: A optional attribute, the type is float32. Defaults to 1e-7.
* @li trans_a: A optional attribute, the type is bool. Defaults to False.
* @li trans_b: A optional attribute, the type is bool. Defaults to False. \n

* @par Outputs:
* Six outputs, including:
* @li norm: A Tensor. Must be one of the following types: float16.
* @li query_output: A Tensor. Must be one of the following types: float16.
* @li key_output: A Tensor. Must be one of the following types: float16.
* @li value_output: A Tensor. Must be one of the following types: float16.
* @li mean: A Tensor. Must be one of the following types: float16.
* @li variance: A Tensor. Must be one of the following types: float16. \n

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use. \n
*/
REG_OP(AttentionLnQKV)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(kernel_query, TensorType({DT_FLOAT16}))
    .INPUT(kernel_key, TensorType({DT_FLOAT16}))
    .INPUT(kernel_value, TensorType({DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT16}))
    .INPUT(beta, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(bias_query, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(bias_key, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(bias_value, TensorType({DT_FLOAT16}))
    .OUTPUT(norm, TensorType({DT_FLOAT16}))
    .OUTPUT(query_output, TensorType({DT_FLOAT16}))
    .OUTPUT(key_output, TensorType({DT_FLOAT16}))
    .OUTPUT(value_output, TensorType({DT_FLOAT16}))
    .OUTPUT(mean, TensorType({DT_FLOAT16}))
    .OUTPUT(variance, TensorType({DT_FLOAT16}))
    .ATTR(epsilon, Float, 0.0000001)
    .ATTR(trans_a, Bool, false)
    .ATTR(trans_b, Bool, false)
    .OP_END_FACTORY_REG(AttentionLnQKV)

/**
* @brief
   swin_transformer model specific structure.Operator only supports swin_transformer. \n
* @par Inputs:
* Five inputs, including:
* @li x: A Tensor. Must be one of the following types: float16.
* @li gamma: A Tensor. Must be one of the following types: float16.
* @li beta: A Tensor. Must be one of the following types: float16.
* @li weight: A Tensor. Must be one of the following types: float16.
* @li bias: A Tensor. Must be one of the following types: float16. \n

* @par Attributes:
* @li head_num: A optional attribute, the type is int.
* @li head_dim: A optional attribute, the type is int.
* @li seq_length: A optional attribute, the type is int.
* @li shifts: A optional attribute, the type is list int. Defaults to ().
* @li epsilon: A optional attribute, the type is float. Defaults to 1e-7. \n

* @par Outputs:
* Three outputs, including:
* @li query_output: A Tensor. Must be one of the following types: float16.
* @li key_output: A Tensor. Must be one of the following types: float16.
* @li value_output: A Tensor. Must be one of the following types: float16. \n

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use. \n
*/
REG_OP(SwinTransformerLnQKV)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT16}))
    .INPUT(beta, TensorType({DT_FLOAT16}))
    .INPUT(weight, TensorType({DT_FLOAT16}))
    .INPUT(bias, TensorType({DT_FLOAT16}))
    .OUTPUT(query_output, TensorType({DT_FLOAT16}))
    .OUTPUT(key_output, TensorType({DT_FLOAT16}))
    .OUTPUT(value_output, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(head_num, Int)
    .REQUIRED_ATTR(head_dim, Int)
    .REQUIRED_ATTR(seq_length, Int)
    .ATTR(shifts, ListInt, {})
    .ATTR(epsilon, Float, 0.0000001)
    .OP_END_FACTORY_REG(SwinTransformerLnQKV)

/**
* @brief Multiplies matrix "a" by matrix "b", producing "a * b". \n
* @par Inputs:
* Three inputs, including:
* @li x1: A matrix Tensor. 2D. Must be one of the following types: float16,
* float32, int32. Has format [ND, NHWC].
* @li x2: A matrix Tensor. 2D. Must be one of the following types: float16,
* float32, int32. Has format [ND, NHWC].
* @li bias: A optional 1D Tensor. Must be one of the following types: float16,
* float32, int32. Has format [ND, NHWC]. \n

* @par Attributes:
* @li transpose_x1: A bool. If True, changes the shape of "x1" from [K, M] to
* [M, K] before multiplication.
* @li transpose_x2: A bool. If True, changes the shape of "x2" from [N, K] to
* [K, N] before multiplication. \n

* @par Outputs:
* y: The result matrix Tensor. 2D. Must be one of the following types: float16,
* float32, int32. Has format [ND, NHWC]. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator BatchMatmul.
*/
REG_OP(MatMul)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .OP_END_FACTORY_REG(MatMul)

/**
* @brief Multiplies matrix "a" by matrix "b", producing "a * b". \n
* @par Inputs:
* Four inputs, including:
* @li x1: A matrix Tensor. 2D. Must be one of the following types: float32,
* float16, int32, int8, int4. Has format [ND, NHWC].
* @li x2: A matrix Tensor. 2D. Must be one of the following types: float32,
* float16, int32, int8, int4. Has format [ND, NHWC].
* @li bias: A 1D Tensor. Must be one of the following types: float32,
* float16, int32. Has format [ND, NHWC].
* @li offset_w: A Optional 1D Tensor for quantized inference. Type is int8, int4.
* Reserved. \n

* @par Attributes:
* @li transpose_x1: A bool. If True, changes the shape of "x1" from [K, M] to
* [M, K] before multiplication.
* @li transpose_x2: A bool. If True, changes the shape of "x2" from [N, K] to
* [K, N] before multiplication.
* @li offset_x: An optional integer for quantized MatMulV2.
* The negative offset added to the input x1 for int8 type. Ensure offset_x
* within the effective range of int8 [-128, 127]. Defaults to "0". \n

* @par Outputs:
* y: The result matrix Tensor. 2D. Must be one of the following types: float32,
* float16, int32. Has format [ND, NHWC]. \n

* @attention Constraints:
* if performances better in format NZ, please close
* "MatmulTransdataFusionPass" in fusion configuration. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator BatchMatmul.
*/
REG_OP(MatMulV2)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT4, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT4, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8, DT_INT4}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(MatMulV2)

/**
* @brief Multiplies matrix "a" by matrix "b", producing "a * b". \n
* @par Inputs:
* Five inputs, including:
* @li x1: A matrix Tensor. 2D. Must be one of the following types: int8.
* @li x2: A matrix Tensor. 2D. Must be one of the following types: int8.
* @li compress_index: A compress index matrix of type int8.
* @li bias: An optional Tensor. 1D. Must be one of the following types: int32,
* float16.
* @li offset_w: An optional matrix Tensor. 2D. Must be one of the following
* types: int8. \n

* @par Attributes:
* @li transpose_x1: A bool. If True, changes the shape of "x1" from [K, M] to
* [M, K] before multiplication.
* @li transpose_x2: A bool. If True, changes the shape of "x2" from [N, K] to
* [K, N] before multiplication.
* @li offset_x: An optional integer for quantized MatMulV2Compress.
* The negative offset added to the input x1 for int8 type. Ensure offset_x
* within the effective range of int8 [-128, 127]. Defaults to "0". \n

* @par Outputs:
* y: The result matrix Tensor. 2D. Must be one of the following types: int32,
* float16. \n

* @attention Constraints:
* if performances better in format NZ, please close
* "MatmulTransdataFusionPass" in fusion configuration.

*/
REG_OP(MatMulV2Compress)
    .INPUT(x1, TensorType({DT_INT8}))
    .INPUT(x2, TensorType({DT_INT8}))
    .INPUT(compress_index, TensorType({DT_INT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_INT32, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_INT32, DT_FLOAT16}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(MatMulV2Compress)

/**
* @brief Performs Matrix-to-matrix Multiply,
* producing y=alpha[0]*a*b+beta[0]*c. \n
* @attention Constraints:
* For better performance, The k-axis must be aligned to 16 (input type
* is float16) or 32 (input type is int8). \n

* @par Inputs:
* Five inputs, including:
* @li a: A matrix Tensor. Must be one of the following types:float32, float16,
* int8, int32. Has format ND.
* @li b: A matrix Tensor. Must be one of the following types:float32, float16,
* int8, int32. Has format ND.
* @li c: A matrix Tensor. Must be one of the following types:float32, float16,
* int8, int32. Has format ND.
* @li alpha: A 1D Tensor. The shape of alpha is [1].Must be one of the
* following types: float32, float16, int8, int32. Has format ND.
* @li beta: A 1D Tensor. The shape of beta is [1]. Must be one of the following
* types: float32, float16, int8, int32. Has format ND. \n

* @par Attributes:
* Two attributes, including:
* @li transpose_a: Optional. A bool. If True, changes the shape of "a" from
* [K, M] to [M, K] before multiplication.
* @li transpose_b: Optional. A bool. If True, changes the shape of "b" from
* [N, K] to [K, N] before multiplication. \n

* @par Outputs:
* y: The result matrix Tensor. Must be one of the following types: float32,
* float16, int8, int32. Has format [ND], the format should be equal to a.
*/

REG_OP(GEMM)
    .INPUT(a, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT32}))
    .INPUT(b, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT32}))
    .INPUT(c, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT32}))
    .INPUT(alpha, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT32}))
    .INPUT(beta, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT32}))
    .ATTR(transpose_a, Bool, false)
    .ATTR(transpose_b, Bool, false)
    .OP_END_FACTORY_REG(GEMM)

/**
* @brief Multiplies matrix "a" by matrix "b", producing "a * b". \n
* @par Inputs:
* Two inputs, including:
* @li x1: A matrix Tensor. Must be one of the following types: float16,
* float32, int32. 2D or higher. Has format [ND, NHWC].
* @li x2: A matrix Tensor. Must be one of the following types: float16,
* float32, int32. 2D or higher. Has format [ND, NHWC]. \n

* @par Attributes:
* @li adj_x1: A bool. If True, changes the shape of "x1" from [B, K, M]
* to [B, M, K] before multiplication.
* @li adj_x2: A bool. If True, changes the shape of "x2" from [B, N, K]
* to [B, K, N] before multiplication. \n

* @par Outputs:
* y: The result matrix Tensor. 2D or higher. Must be one of the following
* types: float16,
* float32, int32. 2D or higher. Has format [ND, NHWC]. Has the same shape
* length as "x1" and "x2". \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator BatchMatmul.
*/

REG_OP(BatchMatMul)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .ATTR(adj_x1, Bool, false)
    .ATTR(adj_x2, Bool, false)
    .OP_END_FACTORY_REG(BatchMatMul)


/**
* @brief Multiplies matrix "a" by matrix "b", producing "a * b" . \n
* @par Inputs:
* Three inputs, including:
* @li x1: A matrix Tensor. Must be one of the following types: float16,
* float32, int32, int8, int4. 2D or higher. Has format [ND, NHWC].
* @li x2: A matrix Tensor. Must be one of the following types: float16,
* float32, int32, int8, int4. 2D or higher. Has format [ND, NHWC].
* @li bias: A optional Tensor. Must be one of the following types:
* float16,
* float32, int32. Has format [ND, NHWC].
* @li offset_w: A optional Tensor. Must be one of the following types:
* int8, int4. Has format [ND, NHWC]. \n

* @par Attributes:
* @li adj_x1: A bool. If True, changes the shape of "x1" from [B, K, M] to
* [B, M, K] before multiplication.
* @li adj_x2: A bool. If True, changes the shape of "x2" from [B, N, K] to
* [B, K, N] before multiplication. \n

* @par Outputs:
* y: The result matrix Tensor. 2D or higher. Must be one of the following
* types: float16,
* float32, int32. 2D or higher. Has format [ND, NHWC]. Has the same shape
* length as "x1" and "x2". \n

* @attention Constraints:
* if performances better in format NZ, please close
* "MatmulTransdataFusionPass" in fusion configuration. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator BatchMatmul.
*/

REG_OP(BatchMatMulV2)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT4, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT4, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8, DT_INT4}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .ATTR(adj_x1, Bool, false)
    .ATTR(adj_x2, Bool, false)
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(BatchMatMulV2)

/**
* @brief Computes half the L2 norm of a tensor without the sqrt . \n

* @par Inputs:

* x: A Tensor.
*     TensorType::FloatingDataType() . \n

* @par Outputs:
* y: A Tensor. Has the same type as "x". \n

* @attention Constraints:
* if performances better in format NZ, please close
* "MatmulTransdataFusionPass" in fusion configuration. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator L2Loss.
*/
REG_OP(L2Loss)
    .INPUT(x, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(L2Loss)

/**
* @brief: Returns a batched diagonal tensor with a given batched diagonal values . \n

* @par Inputs:
* x: A Tensor. Must be one of the following types:
*   float16, float32, double, int32, uint8, int16, int8, complex64, int64,
*   qint8, quint8, qint32, uint16, complex128, uint32, uint64 . \n

* @par Outputs:
* y: A Tensor. Has the same type as "x" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MatrixDiag.
*/
REG_OP(MatrixDiag)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixDiag)

/**
* @brief: Returns a batched diagonal tensor with a given batched diagonal values . \n

* @par Inputs:
* Two inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32, int32, int8, uint8.
* @li assist: A Tensor of the same type as "x" . \n

* @par Outputs:
*y: A Tensor. Has the same type as "x" . \n

* @par Third-party framework compatibility
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
* @brief: Returns the batched diagonal part of a batched tensor . \n

* @par Inputs:
* x: A Tensor. Must be one of the following types:
* float16, float32, double, int32, uint8, int16, int8, complex64, int64,
* qint8, quint8, qint32, uint16, complex128, uint32, uint64 . \n

* @par Outputs:
* y: A Tensor. Has the same type as "x" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MatrixDiagPart.
*/
REG_OP(MatrixDiagPart)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixDiagPart)

/**
* @brief: Returns the batched diagonal part of a batched tensor . \n

* @par Inputs:
* Two inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32, int32, int8, uint8.
* @li assist: A Tensor of the same type as "x" . \n

* @par Outputs:
* y: A Tensor. Has the same type as "x" . \n

* @par Third-party framework compatibility
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
* @brief: Returns a batched matrix tensor with new batched diagonal values . \n

* @par Inputs:
* Two inputs, including:
* @li x: A Tensor. Must be one of the following types:
*    float16, float32, double, int32, uint8, int16, int8, complex64, int64,
*    qint8, quint8, qint32, uint16, complex128, uint32, uint64.
* @li diagonal: A Tensor of the same type as "x" . \n

* @par Outputs:
* y: A Tensor. Has the same type as "x" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MatrixSetDiag.
*/
REG_OP(MatrixSetDiag)
    .INPUT(x, TensorType::BasicType())
    .INPUT(diagonal, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixSetDiag)

/**
* @brief: Returns a batched matrix tensor with new batched diagonal values . \n

* @par Inputs:
* Three inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32, int32, int8, uint8.
* @li diagonal: A Tensor of the same type as "x".
* @li assist: A Tensor of the same type as "x" . \n

* @par Outputs:
* y: A Tensor. Has the same type as "x" . \n

* @par Third-party framework compatibility
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
* @brief Function AttentionScore. \n

* @par Inputs:
* six inputs, including:
* @li query: A matrix Tensor. The type only support float16.
* @li key: A matrix Tensor. The type only support float16.
* @li value: A matrix Tensor. The type only support float16.
* @li padding_mask: A matrix Tensor. The type only support float16.
* @li scale: A scalar. The type only support float16.
* @li drop_mask: A matrix Tensor. The type only support uint8. \n

* @par Attributes:
* @li keep_prob: A mutable Tensor. Must met all of the following rules:
 shape of "keep_prob" should be (1,) or [1,].
* @li query_transpose: A bool. If True, changes the shape of "query" from [K, M] to
 [M, K].
* @li key_transpose: A bool. If True, changes the shape of "key" from [N, K] to
 [K, N].
* @li bmm_score_transpose_a: A bool. If True, changes the shape of "mid_data" from [K, M] to
 [M, K].
* @li bmm_score_transpose_b: A bool. If True, changes the shape of "value" from [N, K] to
 [K, N].
* @li axes: A list of int. The dimension softmax would be performed on. Defaults
 to "[-1]" . \n

* @par Outputs:
* attention_score: The result matrix Tensor. The type only support float16.
* softmax_output: The result matrix Tensor. The type only support float16.

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(AttentionScore)
    .INPUT(query, TensorType({DT_FLOAT16}))
    .INPUT(key, TensorType({DT_FLOAT16}))
    .INPUT(value, TensorType({DT_FLOAT16}))
    .INPUT(padding_mask, TensorType({DT_FLOAT16}))
    .INPUT(scale, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(drop_mask, TensorType({DT_INT8}))
    .OUTPUT(attention_score, TensorType({DT_FLOAT16}))
    .OUTPUT(softmax_output, TensorType({DT_FLOAT16}))
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(query_transpose, Bool, false)
    .ATTR(key_transpose, Bool, false)
    .ATTR(bmm_score_transpose_a, Bool, false)
    .ATTR(bmm_score_transpose_b, Bool, false)
    .ATTR(softmax_axes, ListInt, {-1})
    .OP_END_FACTORY_REG(AttentionScore)

/**
* @brief Function AttentionScoreGrad. \n

* @par Inputs:
* seven inputs, including:
* @li attention_score: A matrix Tensor. The type only support float16.
* @li dx: A matrix Tensor. The type only support float16.
* @li query: A matrix Tensor. The type only support float16.
* @li key: A matrix Tensor. The type only support float16.
* @li value: A matrix Tensor. The type only support float16.
* @li scale: A scalar. The type only support float16.
* @li drop_mask: A matrix Tensor. The type only support uint8. \n

* @par Attributes:
* @li keep_prob: A mutable Tensor. Must met all of the following rules:
 shape of "keep_prob" should be (1,) or [1,].
* @li query_transpose: A bool. If True, changes the shape of "query" from [K, M] to
 [M, K].
* @li key_transpose: A bool. If True, changes the shape of "key" from [N, K] to
 [K, N].
* @li value_transpose: A bool. If True, changes the shape of "mid_data" from [K, M] to
 [M, K].
* @li dx_transpose: A bool. If True, changes the shape of "value" from [N, K] to
 [K, N].
* @li softmax_axes: A int. The dimension softmax would be performed on. Defaults
 to "-1" . \n

* @par Outputs:
* value_dw: The result matrix Tensor. The type only support float16.
* query_dx: The result matrix Tensor. The type only support float16.
* key_dw: The result matrix Tensor. The type only support float16.

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(AttentionScoreGrad)
    .INPUT(attention_score, TensorType({DT_FLOAT16}))
    .INPUT(dx, TensorType({DT_FLOAT16}))
    .INPUT(query, TensorType({DT_FLOAT16}))
    .INPUT(key, TensorType({DT_FLOAT16}))
    .INPUT(value, TensorType({DT_FLOAT16}))
    .INPUT(scale, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(drop_mask, TensorType({DT_INT8}))
    .OUTPUT(value_dw, TensorType({DT_FLOAT16}))
    .OUTPUT(query_dx, TensorType({DT_FLOAT16}))
    .OUTPUT(key_dw, TensorType({DT_FLOAT16}))
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(query_transpose, Bool, false)
    .ATTR(key_transpose, Bool, false)
    .ATTR(value_transpose, Bool, false)
    .ATTR(dx_transpose, Bool, false)
    .ATTR(softmax_axes, Int, -1)
    .OP_END_FACTORY_REG(AttentionScoreGrad)

/**
* @brief Applies sparse "updates" to individual values or slices in a Variable . \n

* @par Inputs:
* Three inputs, including:
* @li var: An ND Tensor.
* Must be one of the following types: float16, float32, int8, uint8, double,
 * int64, complex64, qint8, quint8, qint32, uint16, complex128, half, uint32,
 * uint64
* @li indices: An ND Tensor.
* Must be one of the following types: int32 or int64
* @li updates: An ND Tensor.
* Must be one of the following types: float16, float32, int8, uint8, double,
 * int64, complex64, qint8, quint8, qint32, uint16, complex128, half, uint32,
 * uint64

* @par Attributes:
* use_locking: An optional bool. Defaults to "False". If "True",
 * the operation will be protected by a lock . \n

* @par Outputs:
* var: A Tensor. Has the same type and format as input "var" . \n

* @par Third-party framework compatibility
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
* @brief Applies sparse addition to individual values or slices in a Variable . \n

* @par Inputs:
* Three inputs, including:
* @li x: An ND Tensor. \n

* Must be one of the following types: float16, float32, bool, int8, uint8
* @li indices: An ND Tensor. \n

* Must be one of the following types: int32
* @li updates: An ND Tensor. \n

* Must be one of the following types: float16, float32, bool, int8, uint8

* @par Outputs:
* y: A Tensor. Has the same type and format as input "x" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator TensorScatterUpdate.

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(TensorScatterUpdate)
    .INPUT(x, TensorType::BasicType())
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(TensorScatterUpdate)

/**
* @brief Uses "updates" to update tensor "data" by "indices". \n

* @par Inputs:
* Three inputs, including:
* @li data: An ND Tensor . \n
* Must be one of the following types: float16, float32, int32, int8, uint8
* @li indices: An ND Tensor of type int32 or int64
* @li updates: An Tensor. Same shape as indices. format:NCHW, NHWC . \n
* Must be one of the following types: float16, float32, int32, int8, uint8

* @par Attributes:
* @li axis: An optional attribute. Defaults to 0.
* @li reduction: An optional attribute. Defaults to string "none" and can be
* "add" or "mul". \n

* @par Outputs:
* y: A Tensor. Has the same type and format as input "data" . \n

* @par Third-party framework compatibility
* Compatible with the ONNX operator ScatterElements.
*/
REG_OP(ScatterElements)
    .INPUT(data, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(axis, Int, 0)
    .ATTR(reduction, String, "none")
    .OP_END_FACTORY_REG(ScatterElements)

/**
* @brief Uses "updates" to update tensor "data" by "indices". \n

* @par Inputs:
* Three inputs, including:
* @li var: A Tensor of type BasicType.
* @li indices: An ND Tensor of type int32 or int64.
* @li updates: An Tensor with the same dtype as 'var'. Same shape as indices. \n

* @par Attributes:
* @li use_locking: An optional bool. Defaults to "False". If "True",
* the operation will be protected by a lock . \n

* @par Outputs:
* var: A Tensor. Has the same type and format as input "var" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterNdMax.
*/
REG_OP(ScatterNdMax)
    .INPUT(var, TensorType::BasicType())
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType::BasicType())
    .OUTPUT(var,  TensorType::BasicType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterNdMax)

/**
* @brief Adds sparse "updates" to a variable reference . \n

* @par Inputs:
* Three inputs, including:
* @li var: An ND Tensor .

* Must be one of the following types: float16, float, int32, int8, uint8
* @li indices: An ND Tensor . \n

* Must be one of the following types: int32 or int64
* @li updates: An ND Tensor .

* Must be one of the following types: float16, float, int32, int8, uint8

* @par Attributes:
* use_locking: An optional bool. Defaults to "False". If "True",
* the operation will be protected by a lock . \n

* @par Outputs:
* var: A Tensor. Has the same type and format as input "var" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterAdd.
*/
REG_OP(ScatterAdd)
    .INPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterAdd)

/**
* @brief Adds sparse "updates" to a variable reference . \n

* @par Inputs:
* Three inputs, including:
* @li var: An ND Tensor .
* Must be one of the following types: float16, float32, int32, int8, uint8

* @li indices: An ND Tensor of type int32 or int64

* @li updates: An ND Tensor .
* Must be one of the following types: float16, float32, int32, int8, uint8

* @par Attributes:
* axis: An required int. The axis along which to index. \n

* @par Outputs:
* var: A Tensor. Has the same type and format as input "var" . \n

* @par Third-party framework compatibility
* Compatible with the pytorch operator ScatterAdd.
*/
REG_OP(ScatterAddWithAxis)
    .INPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .REQUIRED_ATTR(axis, Int)
    .OP_END_FACTORY_REG(ScatterAddWithAxis)

/**
* @brief Divides a variable reference by sparse updates . \n

* @par Inputs:
* Three inputs, including:
* @li var: An ND Tensor.
* Must be one of the following types: float16, float, int32, int8, uint8

* @li indices: An ND Tensor.
* Must be one of the following types: int32 or int64
* @li updates: An ND Tensor.
* Must be one of the following types: float16, float, int32, int8, uint8

* @par Attributes:
* use_locking: An optional bool. Defaults to "False". If "True",
* the operation will be protected by a lock . \n

* @par Outputs:
* var: A Tensor. Has the same type and format as input "var" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterDiv.
*/
REG_OP(ScatterDiv)
    .INPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterDiv)

/**
* @brief Applies sparse addition to individual values or slices in a Variable . \n

* @par Inputs:
* Three inputs, including:
* @li var: An ND Tensor.
* Must be one of the following types: float16, float, int32, int8, uint8
* @li indices: An ND Tensor.
* Must be one of the following types: int32 or int64
* @li updates: An ND Tensor.
* Must be one of the following types: float16, float, int32, int8, uint8
* @par Attributes:
* use_locking: An optional bool. Defaults to "False". If "True",
* the operation will be protected by a lock . \n

* @par Outputs:
* var: A Tensor. Has the same type and format as input "var" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterNdAdd.
*/
REG_OP(ScatterNdAdd)
    .INPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterNdAdd)

/**
* @brief Applies sparse addition to individual values or slices in a Variable . \n

* @par Inputs:
* Three inputs, including:
* @li x: An ND Tensor. \n

* Must be one of the following types: float16, float32, int32, int8, uint8
* @li indices: An ND Tensor. \n

* Must be one of the following types: int32
* @li updates: An ND Tensor. \n

* Must be one of the following types: float16, float32, int32, int8, uint8

* @par Outputs:
* y: A Tensor. Has the same type and format as input "x" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator TensorScatterAdd.

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(TensorScatterAdd)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OP_END_FACTORY_REG(TensorScatterAdd)

/**
* @brief Applies sparse subtraction to individual values or slices in a Variable . \n

* @par Inputs:
* Three inputs, including:
* @li var: An ND Tensor.
* Must be one of the following types: float16, float, int32, int8, uint8
* @li indices: An ND Tensor.
* Must be one of the following types: int32 or int64
* @li updates: An ND Tensor.
* Must be one of the following types: float16, float, int32, int8, uint8

* @par Attributes:
*use_locking: An optional bool. Defaults to "False". If "True",
* the operation will be protected by a lock . \n

* @par Outputs:
* var: A Tensor. Has the same type and format as input "var" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterNdSub.
*/
REG_OP(ScatterNdSub)
    .INPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterNdSub)

/**
* @brief Uses "updates" to update tensor "data" by "indices". \n

* @par Inputs:
* Three inputs, including:
* @li var: A Tensor of type BasicType.
* @li indices: A ND Tensor of type int32 or int64.
* @li updates: A Tensor with the same dtype as 'var'. Same shape as indices. \n

* @par Attributes:
* use_locking: An optional bool. Defaults to "False". If "True",
* the operation will be protected by a lock . \n

* @par Outputs:
* var: A Tensor. Has the same type and format as input "var" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterNdMin.
*/
REG_OP(ScatterNdMin)
    .INPUT(var, TensorType::BasicType())
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType::BasicType())
    .OUTPUT(var,  TensorType::BasicType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterNdMin)

/**
* @brief Applies sparse addition to individual values or slices in a Variable . \n

* @par Inputs:
* Three inputs, including:
* @li x: An ND Tensor. \n

* Must be one of the following types: float16, float32, int32, int8, uint8
* @li indices: An ND Tensor. \n

* Must be one of the following types: int32
* @li updates: An ND Tensor. \n

* Must be one of the following types: float16, float32, int32, int8, uint8

* @par Outputs:
* y: A Tensor. Has the same type and format as input "x" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator TensorScatterSub.

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(TensorScatterSub)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OP_END_FACTORY_REG(TensorScatterSub)

/**
* @brief Subtracts sparse updates to a variable reference . \n

* @par Inputs:
* Three inputs, including:
* @li var: An ND Tensor.
* Must be one of the following types: float16, float, int32, int8, uint8
* @li indices: An ND Tensor.
* Must be one of the following types: int32 or int64
* @li updates: An ND Tensor.
* Must be one of the following types: float16, float, int32, int8, uint8
* @par Attributes:
* use_locking: An optional bool. Defaults to "False". If "True",
* the operation will be protected by a lock . \n

* @par Outputs:
* var: A Tensor. Has the same type and format as input "var" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterSub.
*/
REG_OP(ScatterSub)
    .INPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterSub)

/**
* @brief: Returns the batched diagonal part of a batched tensor with "assist" . \n

* @par Inputs:
* Two inputs, including:
* @li x: A Tensor of type float16, float32, or int32.
* @li assist: A Tensor of the same type as "x" . \n

* @par Outputs:
* y: A Tensor. Has the same type as "x" . \n

* @par Third-party framework compatibility
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
* @brief: Returns the batched diagonal part of a batched tensor . \n

* @par Inputs:
* x: A Tensor. Must be one of the following types:
*    float16, float32, int32, int64, double, complex64, complex128 . \n

* @par Outputs:
* y: A Tensor. Has the same type as "x" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator DiagPart.
*/
REG_OP(DiagPart)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT64, DT_DOUBLE,
                          DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT64, DT_DOUBLE,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(DiagPart)

/**
* @brief Also known as a "fully-connected" layer, computes an inner product
* with a set of learned weights, and (optionally) adds biases. \n
* @par Inputs:
* Four inputs, including:
* @li x: A Tensor of type float16, int8, int4.
* @li w: A weight matrix of type float16, int8, int4, float32.
* @li b: An optional Tensor of type float16, int32, float32.
* @li offset_w: An optional Tensor of type int8, int4.
* Reserved. Only None Supported. \n

* @par Attributes:
* @li num_output: Required. An int, output neuron number. Reserved.
* @li transpose: A bool, specifying weight whether to transpose input w,
* either "true" or "false". Defaults to "false".
* @li axis: Optional. An int, 1 or 2, specifying which dimension the input
* "K" starts from. Defaults to 1.
* The product of the subsequent dimensions starting form first dimension
* or the second dimension is "K".
* @li offset_x: An optional integer for quantized FullyConnection.
* The negative offset added to the input image for int8 type. Ensure offset_x
* within the effective range of int8 [-128, 127]. Defaults to "0". \n

* @par Outputs:
* y: The result tensor of type float16, int32, float32. \n

* @par Third-party framework compatibility
* Compatible with the Caffe operator InnerProduct. \n

* @par Quantization supported or not
* Yes
*/
REG_OP(FullyConnection)
    .INPUT(x, TensorType({DT_FLOAT16, DT_INT8, DT_INT4, DT_FLOAT, DT_BF16}))
    .INPUT(w, TensorType({DT_FLOAT16, DT_INT8, DT_INT4, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(b, TensorType({DT_FLOAT16, DT_INT32, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8, DT_INT4}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT32, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(num_output, Int)
    .ATTR(transpose, Bool, false)
    .ATTR(axis, Int, 1)
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(FullyConnection)

/**
* @brief Also known as a "fully-connected-compress" layer, computes an inner
* product with a set of learned weights, and (optionally) adds biases. \n
* @par Inputs:
* Five inputs, including:
* @li x: A Tensor of type uint8, int8.
* @li w: A weight matrix of type int8.
* @li compress_index: A compress index matrix of type int8.
* @li b: A optional Tensor of type int32.
* @li offset_w: A optional Tensor of type int8.

* @par Attributes:
* @li num_output: A int, specifying the number of outputs.
* @li transpose: A bool, specifying whether to transpose input w, either "true"
* or "false". Defaults to "false".
* @li axis: Optional. A int, 1 or 2, specifying which dimension the input "K"
* starts from. Defaults to "1".
* The product of the subsequent dimensions starting form first dimension or the
* second dimension is "K".
* @li offset_x: An optional integer for quantized FullyConnectionCompress.
* The negative offset added to the input image for int8 type. Ensure offset_x
* within the effective range of int8 [-128, 127]. Defaults to "0". \n

* @par Outputs:
* y: The result tensor of type int32. \n

* @par Third-party framework compatibility
* Compatible with the Caffe operator InnerProduct. \n

* @par Quantization supported or not
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
* @brief Computes the confusion matrix from predictions and labels . \n

* @par Inputs:
* Three inputs, including:
* @li labels: A Tensor. Must be one of the following types: float16, float32,
* int32, int8, uint8.
* @li predictions: A Tensor. Must be one of the following types: float16,
* float32, int32, int8, uint8.
* @li weights: A Tensor. Must be one of the following types: float16, float32,
* int32, int8, uint8 . \n

* @par Attributes:
* @li num_classes: An integer for the shape of the output matrix.
* No default value.
* @li dtype: Data type of the confusion matrix. No default value . \n

* @par Outputs:
* y: A Tensor. Has the same type and format as input "labels"

* @attention Constraints:
* @li "weights", "labels", and "predictions" are 1D tensors.
* @li The output is with shape (num_classes, num_classes),
* where, 1 <= num_classes <= 4096 . \n

* @see Region()

* @par Third-party framework compatibility
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
* @brief Multiplies sparse updates into a variable reference . \n

* @par Inputs:
* Three inputs, including:
* @li var: An ND Tensor.
* Must be one of the following types: float16, float, int32, int8, uint8
* @li indices: An ND Tensor.
* Must be one of the following types: int32 or int64
* @li updates: An ND Tensor . \n

* Must be one of the following types: float16, float, int32, int8, uint8

* @par Attributes:
* use_locking: An optional bool. Defaults to "False". If "True", the operation
* will be protected by a lock . \n

* @par Outputs:
* var: A Tensor. Has the same type and format as input "var" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterMul.
*/
REG_OP(ScatterMul)
    .INPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterMul)

/**
* @brief Reduces sparse updates into a variable reference using
* the "min" operation . \n

* @par Inputs:
* Three inputs, including:
* @li var: An ND Tensor.
* Must be one of the following types: float16, float, int32, int8, uint8

* @li indices: An ND Tensor.
* Must be one of the following types: int32 or int64

* @li updates: An ND Tensor.
* Must be one of the following types: float16, float, int32, int8, uint8

* @par Attributes:
* use_locking: An optional bool. Defaults to "False". If "True", the operation
* will be protected by a lock . \n

* @par Outputs:
* var: A Tensor. Has the same type and format as input "var" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterMin.
*/
REG_OP(ScatterMin)
    .INPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterMin)

/**
* @brief Reduces sparse updates into a variable reference using the "max" operation . \n

* @par Inputs:
* Three inputs, including:
* @li var: An ND Tensor .

* Must be one of the following types: float16, float, int32, int8, uint8
* @li indices: An NCHW, NHWC, or ND Tensor . \n

* Must be one of the following types: int32 or int64
* @li updates: An NCHW, NHWC, or ND Tensor .

* Must be one of the following types: float16, float, int32, int8, uint8

* @par Attributes:
* use_locking: An optional bool. Defaults to "False".
* If "True", the operation will be protected by a lock . \n

* @par Outputs:
* var: A Tensor. Has the same type and format as input "var" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterMax.
*/
REG_OP(ScatterMax)
    .INPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterMax)

/**
* @brief Applies sparse updates to a variable reference . \n

* @par Inputs:
* Three inputs, including:
* @li var: An ND Tensor .

* Must be one of the following types: float16, float, int32, int8, uint8
* @li indices: An ND Tensor . \n

* Must be one of the following types: int32 or int64
* @li updates: An ND Tensor .

* Must be one of the following types: float16, float, int32, int8, uint8

* @par Attributes:
* use_locking: An optional bool. Defaults to "False". If "True",
* the operation will be protected by a lock . \n

* @par Outputs:
* var: A Tensor. Has the same type and format as input "var" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterUpdate.
*/
REG_OP(ScatterUpdate)
    .INPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterUpdate)

/**
* @brief Returns a tensor with the `k[0]`-th to `k[1]`-th diagonals of the batched `input` . \n

* @par Inputs:
* Three inputs, including:
* @li input: Rank `r` tensor where `r >= 2`. \n

* @li k: \n
* Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main \n
* diagonal, and negative value means subdiagonals. `k` can be a single integer \n
* (for a single diagonal) or a pair of integers specifying the low and high ends \n
* of a matrix band. `k[0]` must not be larger than `k[1]`. \n

* @li padding_value: The value to fill the area outside the specified diagonal band with. \n

* @par Outputs:
* diagonal: The extracted diagonal(s) . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterUpdate.
*/
REG_OP(MatrixDiagPartV2)
    .INPUT(input, TensorType::BasicType())
    .INPUT(k, TensorType({DT_INT32}))
    .INPUT(padding_value, TensorType::BasicType())
    .OUTPUT(diagonal, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixDiagPartV2)

/**
* @brief Returns a batched matrix tensor with new batched diagonal values . \n

* @par Inputs:
* Three inputs, including:
* @li input: "Rank `r+1`, where `r >= 1`. \n

* @li diagonal: Rank `r` when `k` is an integer or `k[0] == k[1]`. Otherwise, it has rank `r+1`. \n

* @li k:
* Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main \n
* diagonal, and negative value means subdiagonals. `k` can be a single integer \n
* (for a single diagonal) or a pair of integers specifying the low and high ends \n
* of a matrix band. `k[0]` must not be larger than `k[1]`. \n

* @par Outputs:
* output: Rank `r+1`, with `output.shape = input.shape` . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterUpdate.
*/
REG_OP(MatrixSetDiagV2)
    .INPUT(input, TensorType::BasicType())
    .INPUT(diagonal, TensorType::BasicType())
    .INPUT(k, TensorType({DT_INT32}))
    .OUTPUT(output, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixSetDiagV2)

/**
* @brief Returns a batched matrix tensor with new batched diagonal values . \n

* @par Inputs:
* Three inputs, including:
* @li input: "Rank `r+1`, where `r >= 1`. \n

* @li diagonal: Rank `r` when `k` is an integer or `k[0] == k[1]`. Otherwise, it has rank `r+1`. \n

* @li k:
* Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main \n
* diagonal, and negative value means subdiagonals. `k` can be a single integer \n
* (for a single diagonal) or a pair of integers specifying the low and high ends \n
* of a matrix band. `k[0]` must not be larger than `k[1]`. \n

* @par Attributes:
* @li align: An optional string. Defaults to RIGHT_LEFT. It is a string specifying \n
* how superdiagonals and subdiagonals should be aligned, respectively. \n
* other optional: LEFT_RIGHT, LEFT_LEFT, and RIGHT_RIGHT.\n

* @par Outputs:
* output: Rank `r+1`, with `output.shape = input.shape` . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterUpdate.
*/
REG_OP(MatrixSetDiagV3)
    .INPUT(input, TensorType::BasicType())
    .INPUT(diagonal, TensorType::BasicType())
    .INPUT(k, TensorType({DT_INT32}))
    .OUTPUT(output, TensorType::BasicType())
    .ATTR(align, String, "RIGHT_LEFT")
    .OP_END_FACTORY_REG(MatrixSetDiagV3)

/**
* @brief Returns a batched diagonal tensor with given batched diagonal values . \n

* @par Inputs:
* Five inputs, including:
* @li diagonal: Rank `r`, where `r >= 1` \n

* @li k:
* Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main \n
* diagonal, and negative value means subdiagonals. `k` can be a single integer \n
* (for a single diagonal) or a pair of integers specifying the low and high ends \n
* of a matrix band. `k[0]` must not be larger than `k[1]`. \n

* @li num_rows:
* The number of rows of the output matrix. If it is not provided, the op assumes \n
* the output matrix is a square matrix and infers the matrix size from k and the \n
* innermost dimension of `diagonal`. \n

* @li num_cols: An NCHW, NHWC, or ND Tensor.
* The number of columns of the output matrix. If it is not provided, the op \n
* assumes the output matrix is a square matrix and infers the matrix size from \n
* k and the innermost dimension of `diagonal`. \n

* @li padding_value: The number to fill the area outside the specified diagonal band with. \n

* @par Outputs:
* output: Has rank `r+1` when `k` is an integer or `k[0] == k[1]`, rank `r` otherwise . \n

* @par Third-party framework compatibility
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

/**
* @brief Add updates to var_out according to axis and indices.

* @par Inputs:
* Three inputs, including:
* @li var: A Tensor. Must be one of the following types:
*     float16, float32, int32, int8, uint8.
* @li indices: A Tensor of the indices, type should be int32.
* @li updates: A Tensor of the same type as "var".

* @par Attributes:
* @li axis: An required int to specify the axis to perform indices add.

* @par Outputs:
* @li var_out: A Tensor. Same as input "var".

* @par Third-party framework compatibility
* Compatible with the Pytorch operator index_add.

* @par Restrictions:
* Warning:THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(IndexAdd)
    .INPUT(var, TensorType({DT_INT32, DT_INT8, DT_UINT8, DT_FLOAT32, DT_FLOAT16}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(updates, TensorType({DT_INT32, DT_INT8, DT_UINT8, DT_FLOAT32, DT_FLOAT16}))
    .OUTPUT(var_out, TensorType({DT_INT32, DT_INT8, DT_UINT8, DT_FLOAT32, DT_FLOAT16}))
    .ATTR(axis, Int, 0)
    .OP_END_FACTORY_REG(IndexAdd)

/**
* @brief According to the index number of indexes, replace the value
* corresponding to X1 with the value in x2.

* @par Inputs:
* Three inputs, including:
* @li x1:  A Tensor. Must be one of the following types:
* float16, float32, double, int32, uint8, int16, int8, complex64, int64,
* qint8, quint8, qint32, uint16, complex128, uint32, uint64. \n

* @li x2: A Tensor of the same type as "x1".
* @li indices: A Tensor of the indices,

* @par Attributes:
* @li accumulate: Does it support self accumulation.Defaults to 0.

* @par Outputs:
* @li y: A Tensor. Same as input "x1".

* @par Third-party framework compatibility
* Compatible with the Pytorch operator index_put.

* @par Restrictions:
* Warning:THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(IndexPut)
    .INPUT(x1, TensorType::BasicType())
    .INPUT(x2, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(indices, ListInt)
    .ATTR(accumulate, Int, 0)
    .OP_END_FACTORY_REG(IndexPut)

/**
* @brief: Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices input \n

* @par Inputs:
* x: A Tensor. Must be one of the following types:
* float16, float32, double, int32, uint8, int16, int8, complex64, int64,
* qint8, quint8, qint32, uint16, complex128, uint32, uint64. \n

* @par Attributes:
* diagonal: An optional attribute indicates the diagonal to consider. \n

* @par Outputs:
* y: A Tensor. Has the same type as "x" . \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator Triu.
*/
REG_OP(Triu)
    .INPUT(x, TensorType::BasicType())
    .ATTR(diagonal, Int, 0)
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(Triu)

/**
* @brief: Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices input \n

*@par Inputs:
* x: A Tensor. Must be one of the following types:
* float16, float32, double, int32, uint8, int16, int8, complex64, int64,
* qint8, quint8, qint32, uint16, complex128, uint32, uint64. \n

* @par Attributes:
* diagonal: An optional attribute indicates the diagonal to consider. \n

* @par Outputs:
* y: A Tensor. Has the same type as "x" . \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator Tril.
*/
REG_OP(Tril)
    .INPUT(x, TensorType::BasicType())
    .ATTR(diagonal, Int, 0)
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(Tril)
/**
* @brief Concatenates a list of N tensors along the first dimension.
* @par Inputs:
* @li x: A list of Tensors. Must be one of the following types:  int32,
* float16, float32. Tensors to be concatenated. All must have size 1 in
*  the first dimension and same shape. It's a dynamic input. \n

* @par Attributes:
* @li equation: The subscripts for the Einstein summation. \n
* @li N: tensor size of input. \n

* @par Outputs:
* @li y: Sums the product of the elements of the input operands along
* dimensions specified
* using a notation based on the Einstein summation convention. \n

* @attention Constraints:
* Input N must be Int. \n

* @par Third-party framework compatibility
* Compatible with Tensorflow 2.x einsum operator.
*/
REG_OP(Einsum)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .REQUIRED_ATTR(equation, String)
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(Einsum)

/**
* @brief Returns a 2-D tensor with ones on the diagonal and zeros elsewhere. \n

* @par Inputs:
* No inputs

* @par Attributes:
* @li num_rows: An required int. \n
* @li num_columns: An optional int.Defaults to 0. \n
* @li batch_shape: An optional ListInt.Defaults to []. \n
* @li dtype: An optional int.Defaults to 0. \n

* @par Outputs:
* y: A Tensor with targeted type and shape. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator Eye. \n
*/
REG_OP(Eye)
    .OUTPUT(y, TensorType::BasicType())    /* "Result, has targeted element type" */
    .REQUIRED_ATTR(num_rows, Int)
    .ATTR(num_columns, Int, 0)
    .ATTR(batch_shape, ListInt, {})
    .ATTR(dtype, Int, 0)
    .OP_END_FACTORY_REG(Eye)

/**
* @brief: Fill diagonal of at least 2 dimension tensors with value . \n

* @par Inputs:
* x: A Tensor. Must be one of the following types:
*    float32, int32, int64 . \n

* @par Outputs:
*y: A Tensor. Has the same type as "x" . \n

* @par Attributes:
* fill_value:The value to fill in
* wrap: An optional bool. Defaults to "False". If "True", Use recursive fill. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator FillDiagonal.
*/
REG_OP(FillDiagonal)
    .INPUT(x, TensorType({DT_FLOAT, DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(fill_value, Float)
    .ATTR(wrap, Bool, false)
    .OP_END_FACTORY_REG(FillDiagonal)

/**
* @brief: Returns the sum of the elements of the diagonal of the input 2-D matrix. \n

* @par Inputs:
* x: A Tensor. Must be one of the following types:
*    float16, float. \n

* @par Outputs:
* y: A Tensor. Has the same type as "x" . \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator Trace.
*/

REG_OP(Trace)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(Trace)

/**
* @brief  Computes the generalized inverse of any matrix. \n

* @par Inputs:
* @li x: input matrix. Must be one of the following types:
*     double, float. \n

* @par Attributes:
* @li rcond: An optional float >= 0 or inf. Defaults to 1e-15. \n

* @par Outputs:
* y: A Tensor with the same type and shape of x's transpose. \n

*/
REG_OP(Pinverse)
    .INPUT(x, TensorType({ DT_FLOAT, DT_DOUBLE }))
    .OUTPUT(y, TensorType({ DT_FLOAT, DT_DOUBLE }))
    .ATTR(rcond, Float, 1e-15)
    .OP_END_FACTORY_REG(Pinverse)

/**
* @brief  From the input tensor and updates tensor, select the maximum value according to indices to output. \n

* @par Inputs:
* Three inputs, including:
* @li input: Must be one of the following types:
*       float16, float32, double, int32, uint8, int16, int8, complex64, int64,
*       qint8, quint8, qint32, uint16, complex128, uint32, uint64.
* @li indices: Must be one of the following types:
*       int32, int64.
* @li updates: Must have the same type as input. \n

* @par Outputs:
* output: A Tensor with the same type as input. \n
*/
REG_OP(TensorScatterMax)
    .INPUT(input, TensorType::BasicType())
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType::BasicType())
    .OUTPUT(output, TensorType::BasicType())
    .OP_END_FACTORY_REG(TensorScatterMax)

/**
* @brief  From the input tensor and updates tensor, select the minimum value according to indices to output. \n

* @par Inputs:
* Three inputs, including:
* @li input: Must be one of the following types:
*       float16, float32, double, int32, uint8, int16, int8, complex64, int64,
*       qint8, quint8, qint32, uint16, complex128, uint32, uint64.
* @li indices: Must be one of the following types:
*       int32, int64.
* @li updates: Must have the same type as input. \n

* @par Outputs:
* output: A Tensor with the same type as input. \n
*/
REG_OP(TensorScatterMin)
    .INPUT(input, TensorType::BasicType())
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType::BasicType())
    .OUTPUT(output, TensorType::BasicType())
    .OP_END_FACTORY_REG(TensorScatterMin)

/**
* @brief: Returns the batched diagonal part of a batched tensor. \n

* @par Inputs:
* @li x: A Tensor. Rank r tensor where r >= 2.
* @li k: A Tensor of type int32. Diagonal offset(s). Positive value means superdiagonal,
         0 refers to the main diagonal, and negative value means subdiagonals. k can be a
         single integer (for a single diagonal) or a pair of integers specifying the low and
         high ends of a matrix band. k[0] must not be larger than k[1].
* @li padding_value:A Tensor. Must have the same type as input. The value to fill the area
                    outside the specified diagonal band with. Default is 0. \n

* @par Outputs:
* @li y: A Tensor. Has the same type as "input". \n

* @par Attributes:
* @li align:An optional string from: "LEFT_RIGHT", "RIGHT_LEFT", "LEFT_LEFT", "RIGHT_RIGHT". Defaults to "RIGHT_LEFT".

* @par Third-party framework compatibility
* Compatible with the Tensorflow  operator FillDiagonal.
*/
 REG_OP(MatrixDiagPartV3)
    .INPUT(x, TensorType::BasicType())
    .INPUT(k, TensorType({DT_INT32}))
    .INPUT(padding_value, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(align,String ,"RIGHT_LEFT")
    .OP_END_FACTORY_REG(MatrixDiagPartV3)

/**
* @brief Returns a batched diagonal tensor with given batched diagonal values . \n

* @par Inputs:
* Five inputs, including:
* @li x: Rank `r`, where `r >= 1` \n

* @li k:
* Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main
* diagonal, and negative value means subdiagonals. `k` can be a single integer
* (for a single diagonal) or a pair of integers specifying the low and high ends
* of a matrix band. `k[0]` must not be larger than `k[1]`. \n

* @li num_rows:
* The number of rows of the output matrix. If it is not provided, the op assumes
* the output matrix is a square matrix and infers the matrix size from k and the
* innermost dimension of `diagonal`. \n

* @li num_cols: An NCHW, NHWC, or ND Tensor.
* The number of columns of the output matrix. If it is not provided, the op
* assumes the output matrix is a square matrix and infers the matrix size from
* k and the innermost dimension of `diagonal`. \n

* @li padding_value: The number to fill the area outside the specified diagonal band with. \n

* @par Attributes:
* @li align: An optional string from: "LEFT_RIGHT", "RIGHT_LEFT", "LEFT_LEFT", "RIGHT_RIGHT".
* Defaults to "RIGHT_LEFT" \n

* @par Outputs:
* @li y: Has rank `r+1` when `k` is an integer or `k[0] == k[1]`, rank `r` otherwise . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterUpdate.
*/
REG_OP(MatrixDiagV3)
    .INPUT(x, TensorType({BasicType(), DT_BOOL}))
    .INPUT(k, TensorType({DT_INT32}))
    .INPUT(num_rows, TensorType({DT_INT32}))
    .INPUT(num_cols, TensorType({DT_INT32}))
    .INPUT(padding_value, TensorType({BasicType(), DT_BOOL}))
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL}))
    .ATTR(align, String, "RIGHT_LEFT")
    .OP_END_FACTORY_REG(MatrixDiagV3)

/**
* @brief Function SwinAttentionScore. \n

* @par Inputs:
* six inputs, including:
* @li query: A matrix Tensor. The type only support float16.
* @li key: A matrix Tensor. The type only support float16.
* @li value: A matrix Tensor. The type only support float16.
* @li padding_mask1: A matrix Tensor. The type only support float16.
* @li padding_mask2: A matrix Tensor. The type only support float16.
* @li scale: A scalar. The type only support float16.
* @li drop_mask: A matrix Tensor. The type only support uint8. \n

* @par Attributes:
* @li keep_prob: A mutable Tensor. Must met all of the following rules:
 shape of "keep_prob" should be (1,) or [1,].
* @li query_transpose: A bool. If True, changes the shape of "query" from [K, M] to
 [M, K].
* @li key_transpose: A bool. If True, changes the shape of "key" from [N, K] to
 [K, N].
* @li bmm_score_transpose_a: A bool. If True, changes the shape of "mid_data" from [K, M] to
 [M, K].
* @li bmm_score_transpose_b: A bool. If True, changes the shape of "value" from [N, K] to
 [K, N].
* @li axes: A list of int. The dimension softmax would be performed on. Defaults
 to "[]" . \n

* @par Outputs:
* attention_score: The result matrix Tensor. The type only support float16.
* softmax: The result matrix Tensor. The type only support float16.

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(SwinAttentionScore)
    .INPUT(query, TensorType({DT_FLOAT16}))
    .INPUT(key, TensorType({DT_FLOAT16}))
    .INPUT(value, TensorType({DT_FLOAT16}))
    .INPUT(padding_mask1, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(padding_mask2, TensorType({DT_FLOAT16}))
    .INPUT(scale, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(drop_mask, TensorType({DT_INT8}))
    .OUTPUT(attention_score, TensorType({DT_FLOAT16}))
    .OUTPUT(softmax, TensorType({DT_FLOAT16}))
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(query_transpose, Bool, false)
    .ATTR(key_transpose, Bool, false)
    .ATTR(bmm_score_transpose_a, Bool, false)
    .ATTR(bmm_score_transpose_b, Bool, false)
    .ATTR(softmax_axes, ListInt, {})
    .OP_END_FACTORY_REG(SwinAttentionScore)

/**
* @brief
   swin_transformer model specific structure.Operator only supports swin_transformer. \n
* @par Inputs:
* Three inputs, including:
* @li x: A Tensor. Must be one of the following types: float16.
* @li weight: A Tensor. Must be one of the following types: float16.
* @li bias: A Tensor. Must be one of the following types: float16. \n

* @par Attributes:
* @li shifts: A optional attribute, the type is list int. Defaults to (). \n

* @par Outputs:
* One output, including:
* @li y: A Tensor. Must be one of the following types: float16. \n

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use. \n
*/
REG_OP(SwinAttentionFFN)
    .INPUT(x1, TensorType({DT_FLOAT16}))
    .INPUT(x2, TensorType({DT_FLOAT16}))
    .INPUT(bias, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .ATTR(shifts, ListInt, {})
    .OP_END_FACTORY_REG(SwinAttentionFFN)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_MATRIX_CALCULATION_OPS_H_
