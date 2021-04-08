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
*@li x:A list of Tensor objects. One or more tensors from which
the enqueued tensors should be taken . \n

*@par Outputs:
*@li y:A list of Tensor objects. One or more tensors from which
the enqueued tensors should be taken . \n

*@par Attributes:
*@li type: An optional ge::DataType. It refers to the target data type of outputs . \n

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
*y: A 6D Tensor. Has the same type as "x", with format C1HWNCoC0.
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
*y: A 4D Tensor. Has the same type as "x", with format HWCN.
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
*@brief Permutes the dimensions according to perm.
        The returned tensor's dimension i will correspond to the input dimension perm[i] . \n

*@par Inputs:
*Two inputs, including:
*@li x: A Tensor. Must be one of the following types: float16, float32, int8, int16, int32, int64, uint8, uint16, uint32, uint64.
*@li perm: A Tensor of type int32 or int64. A permutation of the dimensions of "x" . \n

*@par Outputs:
*y: A Tensor. Has the same type as "x" . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Transpose.
*/
REG_OP(Transpose)
    .INPUT(x, TensorType::BasicType())
    .INPUT(perm, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(Transpose)

/**
*@brief Doing format_transfer for various data format only
support "NHWC/NCHW" to "NC1HWC0" and "NC1HWC0" to "NHWC/NCHW"
"NCHW" to "FRACTAL_Zn" or "FRACTAL_Zn" to "NCHW".
"HWCN" to "FRACTAL_Zn" or "FRACTAL_Zn" to "HWCN" . \n

*@par Inputs:
*src: A Tensor dtype of all types . \n

*@par Attributes:
*@li src_format: A string source data format, can be "NHWC", "NCHW", "FRACTAL_Zn" etc.
*@li dst_format: A string target data format, can be "NC1HWC0", "NCHW", "FRACTAL_Zn" etc.
*@li group: A optional int32, default value is 1. \n

*@par Outputs:
*dst: A Tensor dtype of all types.
*/
REG_OP(TransData)
    .INPUT(src, TensorType::BasicType())
    .OUTPUT(dst, TensorType::BasicType())
    .REQUIRED_ATTR(src_format, String)
    .REQUIRED_ATTR(dst_format, String)
    .ATTR(groups, Int, 1)
    .OP_END_FACTORY_REG(TransData)

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
*@brief Flattens the inputs. Reserves axis 0 and flattens the input tensors
* along axis 1 . \n

*@par Inputs:
*One input:
*x: A multi-dimensional Tensor. Must be one of the following types:
* int8, uint8, int16, uint16, int32, uint32, int64,uint64, float16, float32 . \n

*@par Outputs:
*y: A 2D flattened Tensor (Reserves axis 0 and flattens the input tensors
* along axis 1). Must be one of the following data types: int8, uint8, int16,
* uint16, int32, uint32, int64,uint64, float16, float32 . \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator Flatten.
*/
REG_OP(Flatten)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64,
                          DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64,
                          DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64,
                           DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64,
                           DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(Flatten)

/**
*@brief Permutes and crops the input tensor . \n

*@par Inputs:
* Three inputs, including:
*@li x: A 5D Tensor of type float16 or int8 or uint8, with format NC1HWC0.
*@li block_shape: A 1D list or tuple of int32 or int64.
*@li crops: A 2D list or tuple of int32 or int64. Specifies the amount to
*crop from start and end dimensions after permutation . \n

*@par Outputs:
*y: A Tensor with format NC1HWC0. Has the same type as input "x" . \n

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
*@brief Permutes and crops the input tensor . \n

*@par Inputs:
* One input:
*x: A 5D Tensor of type float16 or int8 or uint8, with format NC1HWC0 . \n

*@par Attributes:
*@li block_shape: A required 1D list or tuple of int32 or int64.
*@li crops: A required 2D list or tuple of int32 or int64. Specifies the amount to crop
* from the start and end dimensions after permutation . \n

*@par Outputs:
*y: A Tensor with format NC1HWC0. Has the same type as input "x".


*@par Third-party framework compatibility
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
*@brief Pads and permutes the input tensor . \n

*@par Inputs:
* Three inputs, including:
*@li x: A 5D Tensor of type float16 or float32, with format NC1HWC0.
*@li block_shape: A 1D list or tuple of int32 or int64.
*@li paddings: A 2D list or tuple of int32 or int64. Specifies the padding for the start and end dimensions after permutation . \n

*@par Outputs:
*y: A Tensor with format NC1HWC0. Has the same type as input "x" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SpaceToBatchND.
*/
REG_OP(SpaceToBatchND)
    .INPUT(x, TensorType::BasicType())
    .INPUT(block_shape, TensorType::IndexNumberType())
    .INPUT(paddings, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(SpaceToBatchND)

/**
*@brief Pads and permutes the input tensor . \n

*@par Inputs:
* One input:
*x: A 5D Tensor of type float16 or float32, with format NC1HWC0 . \n

*@par Attributes:
*@li block_shape: A required 1D list or tuple of int32 or int64.
*@li paddings: A required 2D list or tuple of int32 or int64. Specifies the padding for the start and end dimensions after permutation . \n

*@par Outputs:
*y: A Tensor with format NC1HWC0. Has the same type as input "x" . \n

*@par Third-party framework compatibility
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
*@brief Outputs a copy of the input tensor where values from the "height" and
* "width" dimensions are moved to the "depth" dimension . \n

*@par Inputs:
*x: An NHWC Tensor. Must be one of the following types:
* float16, float32, double, int64, int32, uint8, uint16, uint32, uint64, int8,
* int16, complex64, complex128, qint8, quint8, qint16, quint16, qint32.


*@par Attributes:
*@li block_size: A required int, specifying the input block size.
*@li data_format: An optional string, specifying the data format. Defaults to
* "NHWC" . \n

*@par Outputs:
*y: A Tensor. Has the same type as input "x".
*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SpaceToDepth.
*/
REG_OP(SpaceToDepth)
  .INPUT(x, TensorType::BasicType())
  .OUTPUT(y, TensorType::BasicType())
  .REQUIRED_ATTR(block_size, Int)
  .ATTR(data_format, String, "NHWC")
  .OP_END_FACTORY_REG(SpaceToDepth)

/**
*@brief Rearranges data from depth into blocks of spatial data . \n

*@par Inputs:
*x: A Tensor. Must be one of the following types: float16, float32, double, int32, uint8,
*     int16, int8, complex64, int64, qint8, quint8, qint32, qint16, quint16, uint16,
*     complex128, uint32, uint64

*@par Attributes:
*Two attributes, including:
* @li block_size: An int >= 2, specifying the size of the spatial block.
* @li data_format: An optional string, specifying the data format. Defaults to "NHWC" . \n

*@par Outputs:
*y: A Tensor of the same type as "x" . \n

*@par Third-party framework compatibility:
* Compatible with TensorFlow operator DepthToSpace.
*/
REG_OP(DepthToSpace)
  .INPUT(x, TensorType::BasicType())
  .OUTPUT(y, TensorType::BasicType())
  .REQUIRED_ATTR(block_size, Int)
  .ATTR(data_format, String, "NHWC")
  .OP_END_FACTORY_REG(DepthToSpace)

/**
*@brief Permutes data into spatial data blocks and then prunes them . \n

*@par Inputs:
*@li x: A 4D Tensor with format. Must set the format, supported format list ["NCHW, NHWC"]
*@li crops: A 1D list or tuple of int32 or int64 . \n

*Must be one of the following types: float16, float32

*@par Attributes:
*block_size: A required int8, int16, int32, or int64. No default value . \n

*@par Outputs:
*y: A 4D Tensor with format NHWC,

* of type float16 or float32 . \n

*@attention Constraints:
*@li The size of the first dimension of input "x" must be divisible by (block_size * block_size).
*@li "crops" is a 4Dshape [batch, height, width, depth], height = height_pad - crop_top - crop_bottom,
*width = width_pad - crop_left - crop_right.
*@li block_size > 2

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator BatchToSpace.
*/
REG_OP(BatchToSpace)
    .INPUT(x, TensorType::BasicType())
    .INPUT(crops, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(block_size, Int)
    .OP_END_FACTORY_REG(BatchToSpace)

/**
*@brief Rearrange the batch (permutes) data into spatial data blocks, and then crop them . \n

*@par Inputs:
* One input:
*x: An Tensor of shape [batch*block_size*block_size, height_pad/block_size, width_pad/block_size, depth].
*The batch size of the input tensor must be divisible by (block size * block size).
*Must be one of the following types:  float16, float32, double, int64, int32, uint8, uint16, uint32, uint64,
*int8, int16, complex64, complex128, qint8, quint8, qint16, quint16, qint32 . \n

*@par Attributes:
*@li block_size: Must be one of the following types: `int32`, `int64`.
*@li crops: An Tensor. Must be one of the following types: int32, Int64.
*2D tensor with non negative integer of shape [2, 2]. It specifies how many
*elements are clipped from the intermediate result of spatial dimension . \n

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x" . \n

*@attention Constraints:
*@li The size of the first dimension of input "x" must be divisible by (block_size * block_size).
*@li "crops" is a 2D tensor of non-negative integers with shape (2, 2).
*@li block_size > 2


*@par Third-party framework compatibility
* Compatible with the TensorFlow operator BatchToSpace.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use BatchToSpace instead.
*/
REG_OP(BatchToSpaceD)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT32, DT_UINT8,
                        DT_UINT16, DT_UINT32, DT_UINT64, DT_INT8, DT_INT16, DT_COMPLEX64,
                        DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT32, DT_UINT8,
                        DT_UINT16, DT_UINT32, DT_UINT64, DT_INT8, DT_INT16, DT_COMPLEX64,
                        DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32}))
    .REQUIRED_ATTR(block_size, Int)
    .REQUIRED_ATTR(crops, ListInt)
    .OP_END_FACTORY_REG(BatchToSpaceD)

/**
*@brief Outputs a copy of the input tensor where values from the "height" and
* "width" dimensions are padded and rearranged to the "batch" dimension . \n

*@par Inputs:
* Two inputs, including:
*@li x: An 4D Tensor. Must be one of the following types:
* float16, float32, double, int64, int32, uint8, uint16, uint32, uint64, int8,
* int16, complex64, complex128, qint8, quint8, qint16, quint16, qint32.
* Must set the format, supported format list ["NCHW, NHWC"]
*@li paddings: A 2D tensor of type int, specifying the input . \n

*@par Attributes:
*block_size: A required int, specifying the input block size . \n

*@par Outputs:
*y: A Tensor. Has the same type as input "x".
*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SpaceToBatch.
*/
REG_OP(SpaceToBatch)
    .INPUT(x, TensorType::BasicType())
    .INPUT(paddings, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(block_size, Int)
    .OP_END_FACTORY_REG(SpaceToBatch)

/**
*@brief Outputs a copy of the input tensor where values from the "height" and "width" dimensions are padded and rearranged to the "batch" dimension . \n

*@par Inputs:
*x: An NHWC Tensor. Must be one of the following types: float16, float32, double, int64, int32, uint8, uint16, uint32, uint64, int8, int16, complex64, complex128, qint8, quint8, qint16, quint16, qint32.


*@par Attributes:
*@li block_size: A required int, specifying the input block size.
*@li paddings: A 2D tensor. All data types are supported . \n

*@par Outputs:
*y: A Tensor. Has the same type as input "x".
*@par Third-party framework compatibility
*@ Compatible with the TensorFlow operator SpaceToBatch.
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
* x: A rank-R tensor (R > 0) of type BasicType, with format ND or NC1HWC0 . \n

* @par Attributes:
* @li num: A required int, specifying the number of tensors to be unpacked to.
* Defaults to "None".
* @li axis: An optional int, specifying the axis to unpack along. The value range
* is [-R, R) . \n

* @par Outputs:
* y: Dynamic output. The list of Tensor objects unpacked from "x", of type BasicType . \n

* @attention Constraints:
* @li If "num" is not specified, it is inferred from the shape of "x".
* @li For the ND format, "axis" is in the range [-R, R); For the NC1HWC0 format,
* "axis" must not be 2, 3, -2, or -3 . \n

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
*    following types:float32, double, int32, uint8, int16, int8, int64, uint16,
*    float16, uint32, uint64. The inputs must have data_format with one of follows:
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
*@brief Confuse reshape and transpose . \n

*@par Inputs:
*x: A Tensor. Must be one of the following types: float16, float32, int8, int16, int32, int64, uint8, uint16, uint32, uint64 . \n

*@par Attributes:
*@li perm: A permutation of the dimensions of "x".
*@li shape: The shape of the input.
*@li transpose_first: If True, the transpose is first, otherwise the reshape is first . \n

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ConfusionTranspose instead.
*/
REG_OP(ConfusionTransposeD)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                        DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                         DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(perm, ListInt)
    .REQUIRED_ATTR(shape, ListInt)
    .REQUIRED_ATTR(transpose_first, Bool)
    .OP_END_FACTORY_REG(ConfusionTransposeD)

/**
*@brief Confuse reshape and transpose . \n

*@par Inputs:
*@li x: A Tensor. Must be one of the following types: float16, float32, int8, int16, int32, int64, uint8, uint16, uint32, uint64.
*@li shape: The shape of the input . \n

*@par Attributes:
*@li perm: A permutation of the dimensions of "x".
*@li transpose_first: If True, the transpose is first, otherwise the reshape is first . \n

*@par Outputs:
*y: A Tensor. Has the same type as "x".

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
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
.REQUIRED_ATTR(compress_parameters, ListInt)
.OP_END_FACTORY_REG(CompressFcOp)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_TRANSFORMATION_OPS_H_
