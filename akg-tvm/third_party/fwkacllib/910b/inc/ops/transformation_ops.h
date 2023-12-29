/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file transformation_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_TRANSFORMATION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_TRANSFORMATION_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief This operation convert output dataType and shape

*@par Inputs:
*The input handle must have the resource type. Inputs include:
*x:A list of Tensor objects. One or more tensors from which
the enqueued tensors should be taken . \n

*@par Outputs:
*y:A list of Tensor objects. One or more tensors from which
the enqueued tensors should be taken . \n

*@par Attributes:
*type: An optional ge::DataType. It refers to the target data type of outputs . \n

*@par Third-party framework compatibility
*Compatible with tensorflow QueueIsClosed operator.
*/

REG_OP(Bitcast)
    .INPUT(x, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8,
                          DT_INT64, DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64,
                          DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32}))
    .OUTPUT(y, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8,
                           DT_INT64, DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64,
                           DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32}))
    .REQUIRED_ATTR(type, Type)
    .OP_END_FACTORY_REG(Bitcast)

/**
*@brief Convert tensor format from HWCN to C1HWNCoC0 . \n

*@par Inputs:
*x: A Tensor. Must be 4D Tensor of type float16, float32, int32, uint16, with format HWCN . \n

*@par Outputs:
*y: A 6D Tensor. Has the same type as "x", with format C1HWNCoC0. \n

*@attention Constraints:
*THIS OPERATOR IS DEPRECATED. It will be removed in a future version.
*/
REG_OP(DepthwiseWeight4DTo6D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT16}))
    .OP_END_FACTORY_REG(DepthwiseWeight4DTo6D)

/**
*@brief Convert tensor format from C1HWNCoC0 to HWCN . \n

*@par Inputs:
*x: A Tensor. Must be 6D Tensor of type float16, float32, int32, uint16, with format C1HWNCoC0 . \n

*@par Attributes:
*channel_size: An optional int, specifying the channel size of 4D Tensor with format HWCN . \n

*@par Outputs:
*y: A 4D Tensor. Has the same type as "x", with format HWCN. \n

*@attention Constraints:
*THIS OPERATOR IS DEPRECATED. It will be removed in a future version.
*/
REG_OP(DepthwiseWeight6DTo4D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT16}))
    .ATTR(channel_size, Int, 16)
    .OP_END_FACTORY_REG(DepthwiseWeight6DTo4D)

/**
*@brief Permutes the dimensions according to perm.
        The returned tensor's dimension i will correspond to the input dimension perm[i] . \n

*@par Inputs:
*x: A Tensor. Must be one of the following types: float16, float32, int8, int16, int32, int64, uint8, uint16, uint32, uint64 . \n

*@par Attributes:
*perm: A permutation of the dimensions of "x" . \n

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*@par Restrictions:
*Warning: THIS FUNCTION IS DEPRECATED. Please use Transpose instead.
*/
REG_OP(TransposeD)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                        DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                         DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(perm, ListInt)
    .OP_END_FACTORY_REG(TransposeD)

/**
* @brief Permutes the dimensions according to perm.
         The returned tensor's dimension i will correspond to the input dimension perm[i] . \n

* @par Inputs:
* Two inputs, including:
* @li x: A Tensor. Must be one of the following types:
* bfloat16, float16, float32, double, int64, int32, uint8, uint16, uint32, uint64, int8,
* int16, complex64, complex128, qint8, quint8, qint16, quint16, qint32 . \n
* @li perm: A Tensor of type int32 or int64. A permutation of the dimensions of "x" . \n

* @par Outputs:
* y: A Tensor. Has the same type as "x" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Transpose.
*/
REG_OP(Transpose)
    .INPUT(x, TensorType::BasicType())
    .INPUT(perm, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(Transpose)

/**
* @brief Do format transfer for various data format.
* In general, the framework will insert it atomatically . \n

* @par Inputs:
* src: A Tensor. For all branches can be types: bfloat16, float16, float32, int32, int8, bool.
*      For branches without padding also can be types: int16, int64, uint8, uint16, uint32, uint64 . \n

* @par Attributes:
* @li src_format: A string source data format, can be "NHWC", "NCHW" etc.
* @li dst_format: A string target data format, can be "NCHW" etc.
* @li src_subformat: A optional int32 for source sub-format, default value is 0.
* @li dst_subformat: A optional int32 for target sub-format, default value is 0.
* @li groups: A optional int32, default value is 1. \n

* @par Outputs:
* dst: A Tensor. Has the same type as "src".
*\n
*\n
* The following value ranges must be met.
* '<===>' indicates that foramt is bidirectionly supported, either input or output.
* '===>' indicates that foramt is unbidirectionly supported, and the input and
* output data types must be correspond to each other. \n
*\n
*\n
| src_format <===> dst_format | dtype                              | C0    | groups |\n
| :-------------------------: | :--------------------------------: |:-----:| :----: |\n
| NCHW <====> NC1HWC0         | float32, int32,uint32              | 8,16  | 1      |\n
| NCHW <====> NC1HWC0         | bfloat16, float16                  | 16    | 1      |\n
| NCHW <====> NC1HWC0         | int8, uint8                        | 32    | 1      |\n
| NHWC <====> NC1HWC0         | float32, int32,uint32              | 8,16  | 1      |\n
| NHWC <====> NC1HWC0         | bfloat16, float16                  | 16    | 1      |\n
| NHWC <====> NC1HWC0         | int8,  uint8                       | 32    | 1      |\n
| ND <====> FRACTAL_NZ        | float32, int32,uint32              | 16    | 1      |\n
| ND <====> FRACTAL_NZ        | bfloat16, float16                  | 16    | 1      |\n
| ND <====> FRACTAL_NZ        | int8, uint8                        | 32    | 1      |\n
| NCHW <====> FRACTAL_Z       | float32, int32,uint32              | 8,16  | 1      |\n
| NCHW <====> FRACTAL_Z       | bfloat16, float16                  | 16    | 1      |\n
| NCHW <====> FRACTAL_Z       | int8,  uint8                       | 32    | 1      |\n
| HWCN <====> FRACTAL_Z       | float32, int32,uint32              | 8,16  | 1      |\n
| HWCN <====> FRACTAL_Z       | bfloat16, float16                  | 16    | 1      |\n
| HWCN <====> FRACTAL_Z       | int8, uint8                        | 32    | 1      |\n
| NCDHW <====> NDC1HWC0       | float32, int32,uint32              | 16    | 1      |\n
| NCDHW <====> NDC1HWC0       | bfloat16, float16                  | 16    | 1      |\n
| NCDHW <====> NDC1HWC0       | int8, uint8                        | 32    | 1      |\n
| NDHWC <====> NDC1HWC0       | float32, int32,uint32              | 16    | 1      |\n
| NDHWC <====> NDC1HWC0       | bfloat16, float16                  | 16    | 1      |\n
| NDHWC <====> NDC1HWC0       | int8, uint8                        | 32    | 1      |\n
| NCDHW <====> FRACTAL_Z_3D   | float32, int32,uint32              | 16    | 1      |\n
| NCDHW <====> FRACTAL_Z_3D   | bfloat16, float16                  | 16    | 1      |\n
| NCDHW <====> FRACTAL_Z_3D   | int8, uint8                        | 32    | 1      |\n
| DHWCN <====> FRACTAL_Z_3D   | float32, int32,uint32              | 16    | 1      |\n
| DHWCN <====> FRACTAL_Z_3D   | bfloat16, float16                  | 16    | 1      |\n
| DHWCN <====> FRACTAL_Z_3D   | int8, uint8                        | 32    | 1      |\n
| NCHW <====> FRACTAL_Z       | float32, uint32, int32             | 8     | >1     |\n
| NCHW <====> FRACTAL_Z       | float16, bfloat16, uint16, int16   | 16    | >1     |\n
| HWCN ====> FRACTAL_Z        | float16, bfloat16, uint16, int16   | 16    | >1     |\n
| NCDHW <====> FRACTAL_Z_3D   | float32, uint32, int32             | 8     | >1     |\n
| NCDHW <====> FRACTAL_Z_3D   | float16, bfloat16, uint16, int16   | 16    | >1     |\n
| FRACTAL_Z_3D ====> DHWCN    | float32, uint32, int32             | 8     | >1     |\n
| FRACTAL_Z_3D ====> DHWCN    | float16, bfloat16, uint16, int16   | 16    | >1     |\n
*\n
*
*/
REG_OP(TransData)
    .INPUT(src, TensorType::BasicType())
    .OUTPUT(dst, TensorType::BasicType())
    .REQUIRED_ATTR(src_format, String)
    .REQUIRED_ATTR(dst_format, String)
    .ATTR(src_subformat, Int, 0)
    .ATTR(dst_subformat, Int, 0)
    .ATTR(groups, Int, 1)
    .OP_END_FACTORY_REG(TransData)

/**
*@brief Do format transfer for various data format only 
support "ND" to "ND_RNN_BIAS" and "ND" to "FRACTAL_ZN_RNN"

*@par Inputs:
*src: A Tensor. For all branches can be types: float16, float32, int32, int8, bool.
* For branches without padding also can be types: int16, int64, uint8, uint16, uint32, uint64 . \n

*@par Attributes:
*@li src_format: A string source data format, can be "ND", "ND_RNN_BIAS", "FRACTAL_ZN_RNN" etc.
*@li dst_format: A string target data format, can be "ND", "ND_RNN_BIAS", "FRACTAL_ZN_RNN" etc.
*@li input_size: A mental int32.
*@li hidden_size: A mental int32.

*@par Outputs:
*dst: A Tensor. Has the same type as "src".
*/
REG_OP(TransDataRNN)
    .INPUT(src, TensorType::BasicType())
    .OUTPUT(dst, TensorType::BasicType())
    .REQUIRED_ATTR(src_format, String)
    .REQUIRED_ATTR(dst_format, String)
    .REQUIRED_ATTR(input_size, Int)
    .REQUIRED_ATTR(hidden_size, Int)
    .OP_END_FACTORY_REG(TransDataRNN)

/**
*@brief Permutes the dimensions according to order.
        The returned tensor's dimension i will correspond to the input dimension order[i] . \n

*@par Inputs:
*x: A Tensor. Must be one of the following types: float16, float32 . \n

*@par Attributes:
*order: A permutation of the dimensions of "x".Type is int32.support any axis transformation.Defaults to "{0}"

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*/
REG_OP(Permute)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(order, ListInt, {0})
    .OP_END_FACTORY_REG(Permute)

/**
* @brief Flattens the inputs tensor into a 2D matrix. If input tensor has shape (d_0, d_1,..., d_n),
*        then the output will have shape (d_0 X d_1 ... d_(axis-1), d_axis X d_(axis + 1)...X d_n)\n

* @par Inputs:
* One input:
* x: A multi-dimensional Tensor. Must be one of the following types:
*    int8, uint8, int16, uint16, int32, uint32, int64,uint64, float16, float32, bfloat16.

* @par Outputs:
* y: A 2D flattened Tensor with the contents of the input tensor, with input dimensions up to axis flattened 
* to the outer dimension of the output and remaining input dimensions flattened into the inner dimension of the output.
* Must be one of the following data types:
  int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, bfloat16 .

* @par Attributes:
* axis: A optional int32, default value is 1. Indicate up to which input dimensions (exclusive) should be flattened 
* to the outer dimension of the output. The value for axis must be in the range [-r, r], where r is the rank of 
* the input tensor. Negative value means counting dimensions from the back. When axis = 0, the shape of 
* the output tensor is (1, (d_0 X d_1 ... d_n), where the shape of the input tensor is (d_0, d_1, ... d_n).

* @par Third-party framework compatibility
* Compatible with TensorFlow / ONNX operator Flatten.
*/
REG_OP(Flatten)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64,
                          DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64,
                          DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64,
                           DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64,
                           DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(axis, Int, 1)
    .OP_END_FACTORY_REG(Flatten)

/**
* @brief Permutes data from batch into blocks of spatial data and then prunes them.
* The values from the batch dimension are moved in spatial blocks to the height and width dimensions.
* And then prunes the height and width dimensions. \n

* @par Inputs:
* @li x: A N-D tensor, Must be one of the following types:
* float16, float32, double, int64, int32, uint8, uint16, uint32, uint64, int8,
* int16, complex64, complex128, qint8, quint8, qint16, quint16, qint32, bfloat16.
* @li crops: A 2D tensor with shape [M, 2], support int32 or int64. \n

* @par Attributes:
* block_size: Must be one of the following types: `int32`, `int64`. \n

* @par Outputs:
* y: A N-D tensor, the same type as "x". \n

* @attention Constraints:
* @li If N is 4 and M is 2:
* @li The size of the first dimension of input "x" must be divisible by (block_size * block_size).
* @li "y" is a 4D shape [batch, height, width, depth], height = height_pad - crop_top - crop_bottom,
* width = width_pad - crop_left - crop_right.

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator BatchToSpaceND.
*/
REG_OP(BatchToSpaceND)
    .INPUT(x, TensorType::BasicType())
    .INPUT(block_shape, TensorType::IndexNumberType())
    .INPUT(crops, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(BatchToSpaceND)

/**
* @brief Permutes data from batch into blocks of spatial data and then prunes them.
* The values from the batch dimension are moved in spatial blocks to the height and width dimensions.
* And then prunes the height and width dimensions. \n

* @par Inputs:
* @li x: A N-D tensor, Must be one of the following types: float16, float32. \n

* @par Attributes:
* @li block_size: Must be one of the following types: `int32`, `int64`.
* @li crops: Must be one of the following types: int32, int64.
* 2D list with non negative integer of shape [M, 2]. It specifies how many
* elements are clipped from the intermediate result of spatial dimension . \n

* @par Outputs:
* y: A N-D tensor, the same type as "x". \n

* @attention Constraints:
* @li If N is 4 and M is 2:
* @li The size of the first dimension of input "x" must be divisible by (block_size * block_size).
* @li "y" is a 4D shape [batch, height, width, depth], height = height_pad - crop_top - crop_bottom,
* width = width_pad - crop_left - crop_right.

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator BatchToSpaceND.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use BatchToSpaceND instead.
*/
REG_OP(BatchToSpaceNDD)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(block_shape, ListInt)
    .REQUIRED_ATTR(crops, ListInt)
    .OP_END_FACTORY_REG(BatchToSpaceNDD)

/**
* @brief Zeros-pads and then permutes blocks of spatial data into batch.
* The values from the height and width dimensions are moved in spatial blocks to the batch dimension.
* After zeros-pads the height and width dimensions. \n

* @par Inputs:
* @li x: A N-D tensor, Must be one of the following types:
* float16, float32, double, int64, int32, uint8, uint16, uint32, uint64, int8,
* int16, complex64, complex128, qint8, quint8, qint16, quint16, qint32, bfloat16.
* @li paddings: A 2D tensor with shape [M, 2], support int32 or int64. \n

* @par Attributes:
* block_size: Must be one of the following types: `int32`, `int64`. \n

* @par Outputs:
* y: A N-D tensor, the same type as "x". \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SpaceToBatchND.
*/
REG_OP(SpaceToBatchND)
    .INPUT(x, TensorType::BasicType())
    .INPUT(block_shape, TensorType::IndexNumberType())
    .INPUT(paddings, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(SpaceToBatchND)

/**
* @brief Zeros-pads and then permutes blocks of spatial data into batch.
* The values from the height and width dimensions are moved in spatial blocks to the batch dimension.
* After zeros-pads the height and width dimensions. \n

* @par Inputs:
* @li x: A N-D tensor, Must be one of the following types: float16, float32. \n

* @par Attributes:
* @li block_size: Must be one of the following types: `int32`, `int64`.
* @li paddings: Must be one of the following types: int32, int64. \n
* 2D list with non negative integer of shape [M, 2]. It specifies how many
* elements are padded from the intermediate result of spatial dimension . \n

* @par Outputs:
* y: A N-D tensor, the same type as "x". \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SpaceToBatchND.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use SpaceToBatchND instead.
*/
REG_OP(SpaceToBatchNDD)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(block_shape, ListInt)
    .REQUIRED_ATTR(paddings, ListInt)
    .OP_END_FACTORY_REG(SpaceToBatchNDD)

/**
* @brief Outputs a copy of the input tensor where values from the "height" and
* "width" dimensions are moved to the "depth" dimension . \n

* @par Inputs:
* x: An NHWC Tensor. Must be one of the following types:
* float16, float32, double, int64, int32, uint8, uint16, uint32, uint64, int8,
* int16, complex64, complex128, qint8, quint8, qint16, quint16, qint32, bfloat16.


* @par Attributes:
* @li block_size: A required int, specifying the input block size.
* @li data_format: An optional string, specifying the data format. Defaults to
* "NHWC" . \n

* @par Outputs:
* y: A Tensor. Has the same type as input "x".
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SpaceToDepth.
*/
REG_OP(SpaceToDepth)
  .INPUT(x, TensorType::BasicType())
  .OUTPUT(y, TensorType::BasicType())
  .REQUIRED_ATTR(block_size, Int)
  .ATTR(data_format, String, "NHWC")
  .OP_END_FACTORY_REG(SpaceToDepth)

/**
* @brief Rearranges data from depth into blocks of spatial data . \n

* @par Inputs:
* x: A Tensor. Must be one of the following types: float16, float32, double, int32, uint8,
*     int16, int8, complex64, int64, qint8, quint8, qint32, qint16, quint16, uint16,
*     complex128, uint32, uint64, bfloat16

* @par Attributes:
* Three attributes, including:
* @li block_size: An int >= 2, specifying the size of the spatial block.
* @li mode: An optional string, specifying the mode. Defaults to "DCR".
* @li data_format: An optional string, specifying the data format. Defaults to "NHWC" . \n

* @par Outputs:
* y: A Tensor of the same type as "x" . \n

* @par Third-party framework compatibility:
* Compatible with TensorFlow operator DepthToSpace.
*/
REG_OP(DepthToSpace)
  .INPUT(x, TensorType::BasicType())
  .OUTPUT(y, TensorType::BasicType())
  .REQUIRED_ATTR(block_size, Int)
  .ATTR(mode, String, "DCR")
  .ATTR(data_format, String, "NHWC")
  .OP_END_FACTORY_REG(DepthToSpace)

/**
* @brief Permutes data from batch into blocks of spatial data and then prunes them.
* The values from the batch dimension are moved in spatial blocks to the height and width dimensions.
* And then prunes the height and width dimensions. \n

* @par Inputs:
* @li x: A 4D tensor, Must be one of the following types:
* float16, float32, double, int64, int32, uint8, uint16, uint32, uint64, int8,
* int16, complex64, complex128, qint8, quint8, qint16, quint16, qint32, bfloat16.
* @li crops: A 2D tensor with shape [2, 2], support int32 or int64. \n

* @par Attributes:
* block_size: Must be one of the following types: `int32`, `int64`. \n

* @par Outputs:
* y: A 4D tensor, the same type as "x". \n

* @attention Constraints:
* @li The size of the first dimension of input "x" must be divisible by (block_size * block_size).
* @li "y" is a 4D shape [batch, height, width, depth], height = height_pad - crop_top - crop_bottom,
* width = width_pad - crop_left - crop_right.

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator BatchToSpace.
*/
REG_OP(BatchToSpace)
    .INPUT(x, TensorType::BasicType())
    .INPUT(crops, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(block_size, Int)
    .OP_END_FACTORY_REG(BatchToSpace)

/**
* @brief Permutes data from batch into blocks of spatial data and then prunes them.
* The values from the batch dimension are moved in spatial blocks to the height and width dimensions.
* And then prunes the height and width dimensions. \n

* @par Inputs:
* @li x: A 4D tensor, Must be one of the following types: float16, float32. \n

* @par Attributes:
* @li block_size: Must be one of the following types: `int32`, `int64`.
* @li crops: Must be one of the following types: int32, int64.
* 2D list with non negative integer of shape [2, 2]. It specifies how many
* elements are clipped from the intermediate result of spatial dimension . \n

* @par Outputs:
* y: A 4D tensor, the same type as "x". \n

* @attention Constraints:
* @li The size of the first dimension of input "x" must be divisible by (block_size * block_size).
* @li "y" is a 4D shape [batch, height, width, depth], height = height_pad - crop_top - crop_bottom,
* width = width_pad - crop_left - crop_right.

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator BatchToSpace.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use BatchToSpace instead.
*/
REG_OP(BatchToSpaceD)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(block_size, Int)
    .REQUIRED_ATTR(crops, ListInt)
    .OP_END_FACTORY_REG(BatchToSpaceD)

/**
* @brief Zeros-pads and then permutes blocks of spatial data into batch.
* The values from the height and width dimensions are moved in spatial blocks to the batch dimension.
* After zeros-pads the height and width dimensions. \n

* @par Inputs:
* @li x: A 4D tensor, Must be one of the following types:
* float16, float32, double, int64, int32, uint8, uint16, uint32, uint64, int8,
* int16, complex64, complex128, qint8, quint8, qint16, quint16, qint32, bfloat16.
* @li paddings: A 2D tensor with shape [2, 2], support int32 or int64. \n

* @par Attributes:
* block_size: Must be one of the following types: `int32`, `int64`. \n

* @par Outputs:
* y: A 4D tensor, the same type as "x". \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SpaceToBatch.
*/
REG_OP(SpaceToBatch)
    .INPUT(x, TensorType::BasicType())
    .INPUT(paddings, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(block_size, Int)
    .OP_END_FACTORY_REG(SpaceToBatch)

/**
* @brief Zeros-pads and then permutes blocks of spatial data into batch.
* The values from the height and width dimensions are moved in spatial blocks to the batch dimension.
* After zeros-pads the height and width dimensions. \n

* @par Inputs:
* @li x: A 4D tensor, Must be one of the following types: float16, float32. \n

* @par Attributes:
* @li block_size: Must be one of the following types: `int32`, `int64`.
* @li paddings: Must be one of the following types: int32, int64. \n
* 2D list with non negative integer of shape [2, 2]. It specifies how many
* elements are padded from the intermediate result of spatial dimension . \n

* @par Outputs:
* y: A 4D tensor, the same type as "x". \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SpaceToBatch.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use SpaceToBatch instead.
*/
REG_OP(SpaceToBatchD)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(block_size, Int)
    .REQUIRED_ATTR(paddings, ListInt)
    .OP_END_FACTORY_REG(SpaceToBatchD)

/**
* @brief Unpacks the given dimension of a rank-R Tensor "x" into rank-(R-1)
* tensors . \n

* @par Inputs:
* x: A rank-R tensor (R > 0) of type BasicType. \n

* @par Attributes:
* @li num: A required int, specifying the number of tensors to be unpacked to.
* Defaults to "None".
* @li axis: An optional int, specifying the axis to unpack along. The value range
* is [-R, R) . \n

* @par Outputs:
* y: Dynamic output. The list of Tensor objects unpacked from "x", of type BasicType . \n

* @attention Constraints:
* @li If "num" is not specified, it is inferred from the shape of "x".
* @li For the ND format, "axis" is in the range [-R, R). \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Unpack.
*/

REG_OP(Unpack)
    .INPUT(x, TensorType::BasicType())
    .DYNAMIC_OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(num, Int)
    .ATTR(axis, Int, 0)
    .OP_END_FACTORY_REG(Unpack)

/**
* @brief Extract "patches" from "images" and stacks them in the "depth"
* dimension of the output . \n

* @par Inputs:
* x: A 4D Tensor with shape [batch, in_rows, in_cols, depth], Must be one of the
*    following types: int8, int16, int32, int64, uint8, uint16, uint32, uint64,
*    float16, float32, double, bfloat16. The inputs must have data_format with one of follows:
*    NHWC, NCHW.

* @par Attributes:
* @li ksizes: A required list or tuple. The size of the sliding window for each
* dimension of images.
* @li strides: A required list or tuple. How far the centers of two consecutive
* patches are in the images. Must be: [1, stride_rows, stride_cols, 1].
* @li rates: A required list or tuple. Must be: [1, rate_rows, rate_cols, 1].
* This is the input stride, specifying how far two consecutive patch
* samples are in the input. Equivalent to extracting patches
* with patch_sizes_eff = patch_sizes + (patch_sizes - 1) *
* (rates - 1), followed by subsampling them spatially by a factor of rates.
* This is equivalent to rate in dilated (a.k.a. Atrous) convolutions.
* @li padding: A required string. The type of padding algorithm to use,
  support "SAME" or "VALID". \n

* @par Outputs:
* y: A 4D Tensor with shape [batch, out_rows, out_cols, ksize_rows *
* ksize_cols * depth] containing image patches with size ksize_rows x ksize_cols
* x depth vectorized in the "depth" dimension. Note "out_rows" and "out_cols"
* are the dimensions of the output patches . \n

* @attention Constraints:
* "ksizes", "strides" and "rates" are lists of integers . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ExtractImagePatches.
*/
REG_OP(ExtractImagePatches)
    .INPUT(x, TensorType::RealNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .REQUIRED_ATTR(ksizes, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(rates, ListInt)
    .REQUIRED_ATTR(padding, String)
    .OP_END_FACTORY_REG(ExtractImagePatches)

/**
* @brief Extract "patches" from "input" and put them in the "depth"
* dimension of the output . \n

* @par Inputs:
* x: A 5D Tensor with shape [batch, in_planes, in_rows, in_cols, depth] . \n
*    The inputs must have data_format with one of follows: NDHWC, NCDHW. \n

* @par Attributes:
* @li ksizes: A required list or tuple. The size of the sliding window for each
* dimension of "x".
* @li strides: A required list or tuple. How far the centers of two consecutive
* patches are in "x". Must be: [1, stride_planes, stride_rows, stride_cols, 1].
* @li padding: A required string. The type of padding algorithm to use ,
* support "SAME" or "VALID" . \n

* @par Outputs:
* Output: A 5D Tensor with shape [batch, out_planes, out_rows, out_cols, ksize_planes *
* ksize_rows * ksize_cols * depth] containing patches with size (ksize_rows * ksize_cols
* * depth) vectorized in the "depth" dimension. Note "out_planes", "out_rows" and "out_cols"
* are the dimensions of the output patches . \n

* @attention Constraints:
* "ksizes" and "strides" are lists of integers.
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ExtractVolumePatches.
*/
REG_OP(ExtractVolumePatches)
    .INPUT(x, TensorType::REALNUMBERTYPE())
    .OUTPUT(y, TensorType::REALNUMBERTYPE())
    .REQUIRED_ATTR(ksizes, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .OP_END_FACTORY_REG(ExtractVolumePatches)

/**
* @brief Confuse reshape and transpose . \n

* @par Inputs:
* x: A Tensor. Must be one of the following types: bfloat16, float16, float32, int8, int16,
                int32, int64, uint8, uint16, uint32, uint64 . \n

* @par Attributes:
* @li perm: A permutation of the dimensions of "x".
* @li shape: The shape of the input.
* @li transpose_first: If True, the transpose is first, otherwise the reshape is first . \n

* @par Outputs:
* y: A Tensor. Has the same type as "x".
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ConfusionTranspose instead.
*/
REG_OP(ConfusionTransposeD)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                          DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                           DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(perm, ListInt)
    .REQUIRED_ATTR(shape, ListInt)
    .REQUIRED_ATTR(transpose_first, Bool)
    .OP_END_FACTORY_REG(ConfusionTransposeD)

/**
* @brief Confuse reshape and transpose . \n

* @par Inputs:
* @li x: A Tensor. Must be one of the following types: bfloat16, float16, float32, int8, int16,
                   int32, int64, uint8, uint16, uint32, uint64.
* @li shape: The shape of the input . \n

* @par Attributes:
* @li perm: A permutation of the dimensions of "x".
* @li transpose_first: If True, the transpose is first, otherwise the reshape is first . \n

* @par Outputs:
* y: A Tensor. Has the same type as "x".

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(ConfusionTranspose)
    .INPUT(x, TensorType::BasicType())
    .INPUT(shape, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(perm, ListInt)
    .REQUIRED_ATTR(transpose_first, Bool)
    .OP_END_FACTORY_REG(ConfusionTranspose)

/**
*@brief Flattens the input tensor to one-dimensional . \n

*@par Inputs:
*x: An ND tensor. All data types are supported . \n

*@par Attributes:
*@li axis: An optional int32, specifying the first axis to flatten. All preceding axes are retained in the output. Defaults to "1".
*@li end_axis: An optional int32, specifying the last axis to flatten. All following axes are retained in the output. Defaults to "-1" . \n

*@par Outputs:
*y: The flattened ND tensor. All data types are supported . \n

*@attention Constraints:
* "axis" and "end_axis" must be within the dimension range of the input. This operator cannot be directly called by the acllopExecute API.
*@par Third-party framework compatibility
* Compatible with the Caffe operator Flatten.
*/
REG_OP(FlattenV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16,
                          DT_INT32, DT_UINT32, DT_INT64, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16,
                           DT_INT32, DT_UINT32, DT_INT64, DT_UINT64}))
    .ATTR(axis, Int, 1)
    .ATTR(end_axis, Int, -1)
    .OP_END_FACTORY_REG(FlattenV2)

/**
*@brief Compress large weight to small one. Usually inserted before Conv2d.
*
*@par Inputs:
*weight: A tensor before compress. Must be one of the following types: DT_INT8, DT_FLOAT16
*
*@par Outputs:
*@li weight_compress: A tensor after compress. Must be one of the following types: DT_INT8, DT_FLOAT16
*@li compress_index: A tensor. Must be one of the following types: DT_INT8
*
*@par Attributes:
*compress_parameters: A required int8, specifying the compressing block.
*
*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(Compress)
.INPUT(weight, TensorType({DT_INT8, DT_FLOAT16}))
.OUTPUT(weight_compress, TensorType({DT_INT8, DT_FLOAT16}))
.OUTPUT(compress_index, TensorType({DT_INT8}))
.REQUIRED_ATTR(compress_parameters, ListInt)
.OP_END_FACTORY_REG(Compress)

/**
*@brief Compress large weight to small one. Usually inserted before FullyConnection.
*
*@par Inputs:
*weight: A tensor before compress. Must be one of the following types: DT_INT8, DT_FLOAT16
*
*@par Outputs:
*@li weight_compress: A tensor after compress. Must be one of the following types: DT_INT8, DT_FLOAT16
*@li compress_index: A tensor. Must be one of the following types: DT_INT8
*
*@par Attributes:
*compress_parameters: A required int8, specifying the compressing block.
*
*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(CompressFcOp)
.INPUT(weight, TensorType({DT_INT8}))
.OUTPUT(weight_compress, TensorType({DT_INT8}))
.OUTPUT(compress_index, TensorType({DT_INT8}))
.OUTPUT(compress_info, TensorType({DT_UINT32}))
.REQUIRED_ATTR(compress_parameters, ListInt)
.OP_END_FACTORY_REG(CompressFcOp)

/**
*@brief Performs Col2im for each batch entry. \n

*@par Inputs:
*@li x: The Col Tensor. 4-D, shape: `(n, c, kernel_h*kernel_w, ho*wo)`. 
where ho/wo is do = (output_d + 2*padding_d - dilation_d*(kernel_d - 1) - 1)//stride_d + 1.
*@li output_size: The img shape Tensor. 1-D, shape:`(2)`, value: (output_h, output_w).  \n

*@par Outputs:
*y: The img Tensor. 4-D, shape: `(n, c, output_h, output_w)`. \n

*@par Attributes:
*@li kernel_shape: ListInt, value: `(kernel_h, kernel_w)`, the shape of kernel in convolution.
*@li dilation: ListInt, value: `(dilation_h, dilation_w)`, the dilation in convolution.
*@li padding: ListInt, value: `(padding_h, padding_w)`, the dilation in convolution.
*@li stride:  ListInt, value: `(stride_h, stride_w)`, the dilation in convolution.  \n

*@par Third-party framework compatibility
* Compatible with Pytorch col2im/im2col_backward operator.
*/
REG_OP(Col2im)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(output_size, TensorType({DT_INT32, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(kernel_size, ListInt)
    .REQUIRED_ATTR(dilation, ListInt)
    .REQUIRED_ATTR(padding, ListInt)
    .REQUIRED_ATTR(stride, ListInt)
    .OP_END_FACTORY_REG(Col2im)

/**
* @brief Performs Col2ImV2 for each batch entry. \n

* @par Inputs:
* @li x: The Col Tensor. 3-D, shape: `(n, c*kernel_h*kernel_w, ho*wo)`.
where ho/wo is do = (output_d + 2*padding_d - dilation_d*(kernel_d - 1) - 1)//stride_d + 1.
* @li output_size: The img shape Tensor. 1-D, shape:`(2)`, value: (output_h, output_w).
* @li kernel_shape: The kernel size Tensor. 1-D , value: `(kernel_h, kernel_w)`, the shape of kernel in convolution.  \n

* @par Outputs:
* y: The img Tensor. 4-D, shape: `(n, c, output_h, output_w)`. \n

* @par Attributes:

* @li dilation: ListInt, value: `(dilation_h, dilation_w)`, the dilation in convolution.
* @li padding: ListInt, value: `(padding_h, padding_w)`, the dilation in convolution.
* @li stride:  ListInt, value: `(stride_h, stride_w)`, the dilation in convolution.  \n

* @par Third-party framework compatibility
* Compatible with ONNX Col2Im operator.
*/
REG_OP(Col2ImV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(output_size, TensorType({DT_INT32, DT_INT32}))
    .INPUT(kernel_size, TensorType({DT_INT32, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(dilation, ListInt)
    .REQUIRED_ATTR(padding, ListInt)
    .REQUIRED_ATTR(stride, ListInt)
    .OP_END_FACTORY_REG(Col2ImV2)

/**
* @brief Performs Im2col for each batch entry. \n

* @par Inputs:
* x: A 4D Tensor with shape [batch, in_rows, in_cols, depth], Must be one of the
*    following types:float32, int8, float16. The inputs must have data_format with
*    one of follows:NHWC, NCHW.

* @par Attributes:
* @li ksizes: A required list or tuple. The size of the sliding window for each
* dimension of images.
* @li strides: A optional list or tuple. How far the centers of two consecutive
* patches are in the images. Defaults to "{1}".
* @li dilations: A optional list or tuple. Defaults to "{1}".
* This is the input stride, specifying how far two consecutive patch
* samples are in the input. Equivalent to extracting patches
* with patch_sizes_eff = patch_sizes + (patch_sizes - 1) *
* (dilations - 1), followed by subsampling them spatially by a factor of dilations.
* This is equivalent to rate in dilated (a.k.a. Atrous) convolutions.
* @li padding_mode: A optional String. The type of padding algorithm to use,
* support "SAME", "VALID", "CALCULATED". Among the three modes, only the "CALCULATED"
* means to use the pads below. Defaults to "CALCULATED".
* @li pads: A optional list or tuple. The pad distance. Defaults to "{0}". \n

* @par Outputs:
* y: A 4D Tensor with shape [batch, out_rows, out_cols, ksize_rows *
* ksize_cols * depth] containing image patches with size ksize_rows x ksize_cols
* x depth vectorized in the "depth" dimension. Note "out_rows" and "out_cols"
* are the dimensions of the output patches . \n

* @attention Constraints:
* "ksizes", "strides", "dilations" and "pads" are lists of integers . \n

* @par Third-party framework compatibility
* Compatible with Pytorch Im2col operator.
*/
REG_OP(Im2col)
    .INPUT(x, TensorType::RealNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .REQUIRED_ATTR(ksizes, ListInt)
    .ATTR(strides, ListInt, {1})
    .ATTR(dilations, ListInt, {1})
    .ATTR(padding_mode, String, "CALCULATED")
    .ATTR(pads, ListInt, {0})
    .OP_END_FACTORY_REG(Im2col)

/**
*@brief Generates a 2D or 3D flow field (sampling grid), given a batch of affine
matrices theta. \n

*@par Inputs:
*Input theta must be float16 or float, output_size must be int32 type.Inputs
include:
*@li theta: input batch of affine matrices with shape (N,2,3) for 2D or (N,3,4)
for 3D
*@li output_size: the target output image size. (N×C×H×W for 2D or N×C×D×H×W for
3D) Example: torch.Size((32, 3, 24, 24)) . \n


*@par Attributes:
*align_corners: if True, consider -1 and 1 to refer to the centers of the corner
pixels rather than the image corners.Refer to grid_sample() for a more complete
description. A grid generated by affine_grid() should be passed to grid_sample()
with the same setting for this option. Default: False \n

*@par Outputs:
*@li y: A 2-D integer tensor of shape [M] representing the
selected indices from the boxes tensor, where M <= max_output_size. \n

*@attention Constraints:
*Input theta must be float16 or float, output_size must be int32 type .
The current implementation of AffineGrid operator AiCore adopts 
BatchMatMul's FP16 fusion operator scheme, and the accuracy will 
decrease when the theta range exceeds [-10,10].If the model requires 
high accuracy of AffineGrid, it is recommended to use AICPU. \n

*@par Third-party framework compatibility
*Compatible with Pytorch affine_grid operator.
*/

REG_OP(AffineGrid)
    .INPUT(theta, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(output_size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(align_corners, Bool, false)
    .OP_END_FACTORY_REG(AffineGrid)

/**
*@brief  Make memory of a view be contiguous. \n

*@par Inputs:
*Four inputs, including:
*@li x: The input tensor.
*@li size: The shape of output tensor. 
*@li stride: The stride of output tensor.
*@li storage_offset: The offset in the underlying storage of the output tensor. \n

*@par Outputs:
*y: A Tensor. Has the same type as "x" . \n

*@par Third-party framework compatibility
*Compatible with the pytorch operator as_strided.
*/
REG_OP(AsStrided)
    .INPUT(x, TensorType::BasicType())
    .INPUT(size, TensorType::IndexNumberType())
    .INPUT(stride, TensorType::IndexNumberType())
    .INPUT(storage_offset, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(AsStrided)

/**
*@brief This transform extracts n-grams from the input sequence and save them as a
vector. \n

*@par Inputs:
*@li input: can be either a 1-D or 2-D tensor for n-gram extraction, It is ether string UTF-8 or int32/int64 . \n

*@par Attributes:
*@li max_gram_length : int (required)
*Maximum n-gram length. If this value is 3, 3-grams will be used to generate the output .
*@li max_skip_count : int (required)
*Maximum number of items (integers/strings) to be skipped when constructing an n-gram from X.
If max_skip_count=1, min_gram_length=2, max_gram_length=3, this operator may generate 2-grams
with skip_count=0 and skip_count=1, and 3-grams with skip_count=0 and skip_count=1.
*@li min_gram_length : int (required)
*Minimum n-gram length. If this value is 2 and max_gram_length is 3, output may contain counts of
2-grams and 3-grams.
*@li mode : string (required)
*The weighting criteria. It can be one of "TF" (term frequency), "IDF" (inverse document frequency),
and "TFIDF" (the combination of TF and IDF).
*@li ngram_counts : list of ints (required)
*The starting indexes of 1-grams, 2-grams, and so on in pool. It is useful when determining the boundary
between two consecutive collections of n-grams. For example, if ngram_counts is [0, 17, 36],
the first index (zero-based) of 1-gram/2-gram/3-gram in pool are 0/17/36. This format is essentially identical
to CSR (or CSC) sparse matrix format, and we choose to use this due to its popularity.
*@li ngram_indexes : list of ints (required)
*list of int64s (type: AttributeProto::INTS). This list is parallel to the specified 'pool_*' attribute. The i-th element
in ngram_indexes indicate the coordinate of the i-th n-gram in the output tensor.
*@li pool_int64s : list of ints
*List of int64 n-grams learned from the training set. Either this or pool_strings attributes must be present but not both.
It's an 1-D tensor starting with the collections of all 1-grams and ending with the collections of n-grams. The i-th element
in pool stores the n-gram that should be mapped to coordinate ngram_indexes[i] in the output vector.
*@li pool_strings : list of strings
*List of strings n-grams learned from the training set. Either this or pool_int64s attributes must be present but not both.
It's an 1-D tensor starting with the collections of all 1-grams and ending with the collections of n-grams. The i-th element
in pool stores the n-gram that should be mapped to coordinate ngram_indexes[i] in the output vector.
*@li weights : list of floats
*list of floats. This attribute stores the weight of each n-gram in pool. The i-th element in weights is the weight of
the i-th n-gram in pool. Its length equals to the size of ngram_indexes. By default, weights is an all-one tensor.This attribute
is used when mode is "IDF" or "TFIDF" to scale the associated word counts. \n

*@par Outputs:
*@li output: tensor(float)
*For 1-D input, output is the n-gram representation of that input. For 2-D input, the output is also a 2-D tensor
whose i-th row is the n-gram representation of the i-th input row. More specifically, if input shape is [C], the corresponding
output shape would be [max(ngram_indexes) + 1]. If input shape is [N, C], this operator produces a [N, max(ngram_indexes) + 1]-tensor. \n

*@attention Constraints:
*@li input can be either a 1-D or 2-D tensor, shape is [C] or [N, C].
*@li max(ngram_indexes) + 1 == len(weights), len(y) == len(weights).
*@li ngram_counts and pool(pool_int64s or pool_strings) must match.
*@li either pool_strings or pool_int64s attributes must be present but not both.
*/

REG_OP(TfIdfVectorizer)
    .INPUT(input, TensorType({DT_INT32, DT_INT64, DT_STRING}))
    .OUTPUT(output, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(max_gram_length, Int)
    .REQUIRED_ATTR(max_skip_count, Int)
    .REQUIRED_ATTR(min_gram_length, Int)
    .REQUIRED_ATTR(mode, String)
    .REQUIRED_ATTR(ngram_counts, ListInt)
    .REQUIRED_ATTR(ngram_indexes, ListInt)
    .ATTR(pool_int64s, ListInt, {})
    .ATTR(pool_strings, ListString, {})
    .ATTR(weights, ListFloat, {})
    .OP_END_FACTORY_REG(TfIdfVectorizer)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_TRANSFORMATION_OPS_H_
