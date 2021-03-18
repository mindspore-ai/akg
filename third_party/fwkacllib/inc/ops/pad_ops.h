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

#ifndef GE_OP_PAD_OPS_H
#define GE_OP_PAD_OPS_H

#include "graph/operator_reg.h"
namespace ge {

/**
*@brief Creates a tensor filled with a scalar value.\n
* This operation creates a tensor of shape "dims" and fills it with "value".
*
*@par Inputs:
*@li dims: A 1D tensor of types int32 or int64. Represents the shape of the output tensor.

*@li value: A 0D scalar. Specifies the value to fill the returned tensor.
*
*@par Outputs:
* y: A tensor. Has the same type as "value".
*
*/
REG_OP(Fill)
    .INPUT(dims, TensorType::IndexNumberType())
    .INPUT(value, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(Fill)

/**
*@brief Creates a tensor filled with a scalar value.\n
* This operation creates a tensor of shape "dims" and fills it with "value".
*
*@par Inputs:
* value: A 0D scalar for the value to fill the returned tensor.
*
*@par Attributes:
* dims: A tensor. Must be one of the following types:"int32"
*     1-D. Represents the shape of the output tensor.
*
*@par Outputs:
* y: A tensor. Has the same type as "value".
*
*/
REG_OP(FillD)
    .INPUT(value, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16,
                              DT_UINT16, DT_UINT8, DT_INT32, DT_INT64,
                              DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16,
                           DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                           DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .REQUIRED_ATTR(dims, ListInt)
    .OP_END_FACTORY_REG(FillD)

/**
*@brief Broadcasts an array for a compatible shape.\n
* Broadcasting is the process of making arrays to have compatible shapes
* for arithmetic operations. Two shapes are compatible if for each
*  dimension pair they are either equal or one of them is one. When trying
*  to broadcast a Tensor to a shape, it starts with the trailing dimensions,
*  and works its way forward.
*
*@par Inputs:
*@li x: A tensor.
*@li shape: A tensor of type int32 or int64.
*     A 1D tensor of type int32, for the shape of the desired output.
*
*@par Outputs:
* y: A tensor. Has the same type as "x".
*/
REG_OP(BroadcastTo)
    .INPUT(x, TensorType::BasicType())
    .INPUT(shape, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(BroadcastTo)

/**
*@brief Broadcasts an array for a compatible shape.\n
* Broadcasting is the process of making arrays to have compatible shapes
* for arithmetic operations. Two shapes are compatible if for each
*  dimension pair they are either equal or one of them is one. When trying
*  to broadcast a Tensor to a shape, it starts with the trailing dimensions,
*  and works its way forward.
*
*@par Inputs:
* x: A tensor. A tensor to broadcast.
*
*@par Attributes:
* shape: A tensor of type int32.
*     A 1D tensor of type int32, for the shape of the desired output.
*
*@par Outputs:
* y: A tensor. Has the same type as "x".
*
*/
REG_OP(BroadcastToD)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(shape, ListInt)
    .OP_END_FACTORY_REG(BroadcastToD)

/**
*@brief Pads a tensor.

*@par Inputs:
*Two inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32, double, int32,
*     uint8, int16, int8, complex64, int64, qint8, quint8, qint32, qint16, quint16, uint16,
*     complex128, uint32, uint64.
* @li paddings: A Tensor of type int32 or int64.

*@par Outputs:
*y: A Tensor of the same type as "x".
*/
REG_OP(Pad)
    .INPUT(x, TensorType::BasicType())
    .INPUT(paddings, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(Pad)

/**
*@brief Pads a tensor.

*@par Inputs:
*x: A Tensor. Must be one of the following types: float16, float32, int8, uint8, int32.

*@par Attributes:
*paddings: An optional "vector<vector<int>>". Defaults to "{}".
*     For each dimension D of input, paddings[D, 0] indicates how many
*     values to add before the contents of tensor in that dimension,
*     and paddings[D, 1] indicates how many values to add after the
*     contents of tensor in that dimension.

*@par Outputs:
*y: A Tensor of the same type as "x".
*/
REG_OP(PadD)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_UINT8, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_UINT8, DT_FLOAT}))
    .REQUIRED_ATTR(paddings, ListListInt)
    .OP_END_FACTORY_REG(PadD)

/**
*@brief Create a diagonal tensor

*@par Inputs:
*Two inputs, including:
* @li x: A mutable Tensor. Must be one of the following types:
*     float16, float32, int32.

* @li assist: A mutable Tensor of the same type as "x".

*@par Outputs:
*y: A mutable Tensor. Has the same type as "x".
*/
REG_OP(DiagD)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .INPUT(assist, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OP_END_FACTORY_REG(DiagD)

/**
*@brief Create a diagonal tensor

*@par Inputs:
*One input, include:
* x: A mutable Tensor. Must be one of the following types:
*     float16, float32, double, int32, int64, complex64, complex128.

*@par Outputs:
*y: A mutable Tensor. Has the same type as "x".
*/
REG_OP(Diag)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32,
                          DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32,
                           DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Diag)
} // namespace ge

#endif //GE_OP_PAD_OPS_H
