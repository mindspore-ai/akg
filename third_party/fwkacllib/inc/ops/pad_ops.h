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
 * \file pad_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_PAD_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_PAD_OPS_H_

#include "graph/operator_reg.h"
namespace ge {

/**
*@brief Creates a tensor filled with a scalar value.
* This operation creates a tensor of shape "dims" and fills it with "value".
*
*@par Inputs:
*@li dims: A 1D tensor of types int32 or int64. Represents the shape of the output tensor . \n

*@li value: A 0D scalar. Specifies the value to fill the returned tensor.
*    Must be one of the following types:
*    float16, float32, double, int32, uint8, int16, int8, complex64, int64, bool, 
*    qint8, quint8, qint32, qint16, quint16, uint16, complex128, uint32, uint64, .
*
*@par Outputs:
* y: A tensor. Has the same type as "value".
*
*@par Third-party framework compatibility
*@li Compatible with the TensorFlow operator Fill.
*@li Compatible with the Caffe operator Filler.
*
*/
REG_OP(Fill)
    .INPUT(dims, TensorType::IndexNumberType())
    .INPUT(value, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16,
                              DT_INT8, DT_COMPLEX64, DT_INT64, DT_BOOL, DT_QINT8,
                              DT_QUINT8, DT_QINT32, DT_QINT16, DT_QUINT16, DT_UINT16,
                              DT_COMPLEX128, DT_FLOAT16, DT_UINT32, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16,
                              DT_INT8, DT_COMPLEX64, DT_INT64, DT_BOOL, DT_QINT8,
                              DT_QUINT8, DT_QINT32, DT_QINT16, DT_QUINT16, DT_UINT16,
                              DT_COMPLEX128, DT_FLOAT16, DT_UINT32, DT_UINT64}))
    .OP_END_FACTORY_REG(Fill)

/**
*@brief Creates a tensor filled with a scalar value.
* This operation creates a tensor of shape "dims" and fills it with "value".
*
*@par Inputs:
* value: A 0D scalar for the value to fill the returned tensor. Must be one of
*    the following types:
*    float16, float32, uint8, int8, int16, int32, int64, quint8, qint8, qint32
*
*@par Attributes:
* dims: A tensor. Must be one of the following types:"int32"
*     1-D. Represents the shape of the output tensor.
*
*@par Outputs:
* y: A tensor. Has the same type as "value".
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use Fill instead.
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
*@brief Broadcasts an array for a compatible shape.
*  Broadcasting is the process of making arrays to have compatible shapes
*  for arithmetic operations. Two shapes are compatible if for each
*  dimension pair they are either equal or one of them is one. When trying
*  to broadcast a Tensor to a shape, it starts with the trailing dimensions,
*  and works its way forward.
*
*@par Inputs:
*@li x: A tensor.
*@li shape: A tensor of type int32.
*     A 1D tensor of type int32, for the shape of the desired output.
*
*@par Outputs:
* y: A tensor. Has the same type as "x".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator BroadcastTo.
*
*/
REG_OP(BroadcastTo)
    .INPUT(x, TensorType::BasicType())
    .INPUT(shape, TensorType({DT_INT32,DT_INT64}))
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(BroadcastTo)

/**
*@brief Broadcasts an array for a compatible shape.
*  Broadcasting is the process of making arrays to have compatible shapes
*  for arithmetic operations. Two shapes are compatible if for each
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
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator BroadcastTo.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use BroadcastTo instead.
*/
REG_OP(BroadcastToD)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(shape, ListInt)
    .OP_END_FACTORY_REG(BroadcastToD)

/**
*@brief Pads a tensor . \n

*@par Inputs:
*Two inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32, double, int32,
*     uint8, int16, int8, complex64, int64, qint8, quint8, qint32, qint16, quint16, uint16,
*     complex128, uint32, uint64.
* @li paddings: A Tensor of type int32 or int64 . \n

*@par Outputs:
*y: A Tensor of the same type as "x" . \n

*@par Third-party framework compatibility:
* Compatible with TensorFlow operator Pad.
*/
REG_OP(Pad)
    .INPUT(x, TensorType::BasicType())
    .INPUT(paddings, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(Pad)

/**
*@brief Pads a tensor . \n

*@par Inputs:
*x: A Tensor. Must be one of the following types: float16, float32, int32 . \n

*@par Attributes:
*paddings: An optional "vector<vector<int>>". Defaults to "{}".
*     For each dimension D of input, paddings[D, 0] indicates how many
*     values to add before the contents of tensor in that dimension,
*     and paddings[D, 1] indicates how many values to add after the
*     contents of tensor in that dimension . \n

*@par Outputs:
*y: A Tensor of the same type as "x" . \n

*@par Third-party framework compatibility:
* Compatible with TensorFlow operator Pad.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use Pad instead.
*/
REG_OP(PadD)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .REQUIRED_ATTR(paddings, ListListInt)
    .OP_END_FACTORY_REG(PadD)

/**
*@brief Pads a tensor . \n

*@par Inputs:
*Three inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32, double, int32,
*     uint8, int16, int8, complex64, int64, qint8, quint8, qint32, qint16, quint16, uint16,
*     complex128, uint32, uint64.
* @li constant_values: A Tensor. Must have the same type as input.
* @li paddings: A Tensor of type int32 or int64 . \n

*@par Outputs:
*y: A Tensor of the same type as "x" . \n

*@par Third-party framework compatibility:
* Compatible with TensorFlow operator Pad.
*/
REG_OP(PadV2)
    .INPUT(x, TensorType::BasicType())
    .INPUT(paddings, TensorType::IndexNumberType())
    .INPUT(constant_values, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(PadV2)

/**
*@brief Pads a tensor . \n

*@par Inputs:
*@li x: A Tensor. Must be one of the following types: float16, float32, int32 . \n
*@li constant_values: A Tensor. Must have the same type as input.

*@par Attributes:
*paddings: A required Attribute.
*     For each dimension D of input, paddings[D, 0] indicates how many
*     values to add before the contents of tensor in that dimension,
*     and paddings[D, 1] indicates how many values to add after the
*     contents of tensor in that dimension . \n

*@par Outputs:
*y: A Tensor of the same type as "x" . \n

*@par Third-party framework compatibility:
* Compatible with TensorFlow operator PadV2.
*/
REG_OP(PadV2D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .INPUT(constant_values, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .REQUIRED_ATTR(paddings, ListListInt)
    .OP_END_FACTORY_REG(PadV2D)

/**
*@brief Pads a tensor.

*@par Inputs:
*Two inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32, double, int32,
*     uint8, int16, int8, complex64, int64, qint8, quint8, qint32, qint16, quint16, uint16,
*     complex128, uint32, uint64.
* @li paddings: A Tensor of type int32 or int64.
* @li constant_values: A optional Tensor of int32 or int64

*@par Attributes:
* @li mode: An optional string, Defaults to "constant", indicates paddings mode,
*     support "constant", "reflect", "edge"
* @li paddings_contiguous: An optional bool value, Defaults to true.
*     If true, paddings is arranged as [[begin0, end0], [begin1, end1], ...]
*     If false, paddings is arranged as [[begin0, begin1], ..., [end0, end1], ...]

*@par Outputs:
*y: A Tensor of the same type as "x".

*@par Third-party framework compatibility:
* Compatible with ONNX operator Pad.
*/
REG_OP(PadV3)
    .INPUT(x, TensorType::BasicType())
    .INPUT(paddings, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(constant_values, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(mode, String, "constant")
    .ATTR(paddings_contiguous, Bool, true)
    .OP_END_FACTORY_REG(PadV3)
	
 /**
*@brief Cal the grad of Pads.

*@par Inputs:
*Two inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32, double, int32,
*     uint8, int16, int8, complex64, int64, qint8, quint8, qint32, qint16, quint16, uint16,
*     complex128, uint32, uint64.
* @li paddings: A Tensor of type int32 or int64.

*@par Attributes:
* @li mode: An optional string, Defaults to "reflect", indicates paddings mode,
*     support "reflect", "edge"
* @li paddings_contiguous: An optional bool value, Defaults to true.
*     If true, paddings is arranged as [[begin0, end0], [begin1, end1], ...]
*     If false, paddings is arranged as [[begin0, begin1], ..., [end0, end1], ...]

*@par Outputs:
*y: A Tensor of the same type as "x".

*@par Third-party framework compatibility:
* Compatible with ONNX operator PadGrad.
*/

REG_OP(PadV3Grad)
    .INPUT(x, TensorType::BasicType())
    .INPUT(paddings, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(mode, String, "reflect")
    .ATTR(paddings_contiguous, Bool, true)
    .OP_END_FACTORY_REG(PadV3Grad)

  /**
  *@brief Pads a tensor.

  *@par Inputs:
  *x: A Tensor. Must be one of the following types: float16, float32, int8, uint8, int32.

  *@par Attributes:
  * @li paddings: An required "vector<vector<int>>".
  *     For each dimension D of input, paddings[D, 0] indicates how many
  *     values to add before the contents of tensor in that dimension,
  *     and paddings[D, 1] indicates how many values to add after the
  *     contents of tensor in that dimension.
  * @li constant_values: An optional int value for pad.
  * @li mode: An optional string, Defaults to "constant", indicates paddings mode,
  *     support "constant", "reflect", "edge"
  * @li paddings_contiguous: An optional bool value, Defaults to true.
  *     If true, paddings is arranged as [[begin0, end0], [begin1, end1], ...]
  *     If false, paddings is arranged as [[begin0, begin1], ..., [end0, end1], ...]

  *@par Outputs:
  *y: A Tensor of the same type as "x".

  *@par Third-party framework compatibility:
  * Compatible with ONNX operator Pad.

  * @par Restrictions:
  * Warning: THIS FUNCTION IS DEPRECATED. Please use PadV3 instead.
  */
  REG_OP(PadV3D)
      .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_UINT8}))
      .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_UINT8}))
      .REQUIRED_ATTR(paddings, ListListInt)
      .ATTR(constant_values, Int, 0)
      .ATTR(mode, String, "constant")
      .ATTR(paddings_contiguous, Bool, true)
      .OP_END_FACTORY_REG(PadV3D)

/**
*@brief Create a diagonal tensor

*@par Inputs:
*Two inputs, including:
* @li x: A mutable Tensor. Must be one of the following types:
*     float16, float32, int32 . \n

* @li assist: A mutable Tensor with rank k is at most 1,
*     Has the same type as "x" . \n

*@par Outputs:
*y: A mutable Tensor. Has the same type as "x" . \n

*@see Diag()
*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Diag.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use Diag instead.
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
* x: A mutable Tensor with rank k, where k is at most 1. Must be one of the
*     following types:
*     float16, float32, double, int32, int64, complex64, complex128 . \n

*@par Outputs:
*y: A mutable Tensor. Has the same type as "x" . \n

*@see DiagD()
*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Diag.
*/
REG_OP(Diag)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32,
                          DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32,
                           DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Diag)

/**
*@brief Ascend Padding, pad the last dimension of input

*@par Inputs:
*One input, include:
*x: Tensor which last dimension must be 1. For example: [624000, 1] . \n

*@par Outputs:
*y: Padding the last dimension of x to padDimSize, [624000, padDimSize] . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Diag.
*/
REG_OP(AscendPadding)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(pad_dim_size, Int, 8)
    .OP_END_FACTORY_REG(AscendPadding)


/**
*@brief EmbeddingRankId, traverse the index calculation server and its position in the server . \n

*@par Restrictions:
*Warning:THIS FUNCTION IS DEPRECATED. Please do not use. \n

*@par Inputs:
*One input, include:
*addr_table: Tensor which last dimension must be 3. For example: [8, 3].
*index: Tensor  For example: [640000].
*@par Outputs:
*rank_id: Tensor the first dimension of index to Size, [size, 3].
 Tensor which last dimension must be 3.For example: [640000, 3]
*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Diag.
*/
REG_OP(EmbeddingRankId)
    .INPUT(addr_table, TensorType({DT_UINT64}))
    .INPUT(index, TensorType({DT_INT64,DT_INT32,DT_UINT64}))
    .OUTPUT(rank_id, TensorType({DT_UINT64}))
    .ATTR(row_memory, Int, 320)
    .ATTR(mode, String, "mod")
    .OP_END_FACTORY_REG(EmbeddingRankId)

/**
*@brief EmbeddingLocalIndex, Sort statistics index according to rank_id \n

*@par Inputs:
* @li addr_table: A 2D tensor which last dimension must be 3.
* @li index: A tensor with data type int32, int64, uint32, uint64.

*@par Attributes:
* @li row_memory: The size of Embedding vector in a row, the default is 320.
* @li mode: String type, currently there are two options: 'mod' and 'order'

*@par Outputs:
* @li local_idx:Index on each server.
* @li nums:The number of local_idx found on each server.
* @li recover_idx:The sorted local_idx element is at the position corresponding
* to the original input index.

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Diag.
*/
REG_OP(EmbeddingLocalIndex)
    .INPUT(addr_table, TensorType({DT_UINT64}))
    .INPUT(index, TensorType({DT_INT64,DT_INT32,DT_UINT32,DT_UINT64}))
    .OUTPUT(local_idx, TensorType({DT_INT64,DT_INT32,DT_UINT32,DT_UINT64}))
    .OUTPUT(nums, TensorType({DT_INT64,DT_INT32,DT_UINT32,DT_UINT64}))
    .OUTPUT(recover_idx, TensorType({DT_INT64,DT_INT32,DT_UINT32,DT_UINT64}))
    .ATTR(row_memory, Int, 320)
    .ATTR(mode, String, "mod")
    .OP_END_FACTORY_REG(EmbeddingLocalIndex)

/**
* @brief Fill the value to a tensor has the specified shape.

* @par Inputs:
* One inputs, including:
* @li dims: An Tensor, specify the shape that the value to fill.

* @par Attributes:
* @li value: An optional float value. Defaults to 0.0.

* @par Outputs:
* @li y: A Tensor. Has the shape specify by attr shape, and full of the value specify by attr value.

* @par Third-party framework compatibility
* Compatible with the ONNX operator ConstantOfShape.
*/
REG_OP(FillV2)
    .INPUT(dims, TensorType({DT_INT16, DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64}))
    .ATTR(value, Float, 0)
    .OP_END_FACTORY_REG(FillV2)

/**
* @brief Fill the value to a tensor has the specified shape.

* @par Attributes:
* @li value: An optional float value. Defaults to 0.0.

* @li dims: An required listInt to specify the shape that the value to fill.

* @par Outputs:
* y: A Tensor. Has the shape specify by attr shape, and full of the value specify by attr value.

* @par Third-party framework compatibility
* Compatible with the ONNX operator ConstantOfShape.
*/
REG_OP(FillV2D)
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64}))
    .ATTR(value, Float, 0)
    .REQUIRED_ATTR(dims, ListInt)
    .OP_END_FACTORY_REG(FillV2D)
} // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_PAD_OPS_H_
