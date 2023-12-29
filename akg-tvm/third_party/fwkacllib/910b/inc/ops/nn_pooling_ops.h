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
 * \file nn_pooling_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_POOLING_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_POOLING_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief Performs pooling on the input.
*@par Inputs:
* x: An NCHW tensor of type float16, float32, int8.
*@par Attributes:
*@li mode: An optional int32, specifying the pooling algorithm, either "0" (max pooling) or "1" (avg pooling). Defaults to "0".
*@li global_pooling: An optional bool. Defaults to "false".
*@li window: Optional, including:
*window[0]: An optional int32, specifying the window size along in the H dimension. The value range is [1, 32768]. Defaults to "1".
*window[1]: An optional int32, specifying the window size along in the W dimension. The value range is [1, 32768]. Defaults to "1".
*@li stride: Optional, including:
*stride[0]: An optional int32, specifying the stride along in the H dimension. The value range is [1, 63]. Defaults to "1".
*stride[1]: An optional int32, specifying the stride along in the W dimension. The value range is [1, 63]. Defaults to "1".
*@li pad: Optional, including:
*pad[0]: An optional int32, specifying the up padding. Defaults to "0".
*pad[1]: An optional int32, specifying the bottom padding. Defaults to "0".
*pad[2]: An optional int32, specifying the left padding. Defaults to "0".
*pad[3]: An optional int32, specifying the right padding. Defaults to "0".
*@li dilation: Optional, including:
*dilation[0]: An optional int32, specifying the up dilation. Defaults to "1".
*dilation[1]: An optional int32, specifying the bottom dilation. Defaults to "1".
*dilation[2]: An optional int32, specifying the left dilation. Defaults to "1".
*dilation[3]: An optional int32, specifying the right dilation. Defaults to "1".
*@li ceil_mode: An optional int32, either "0" (ceil mode) or "1" (floor mode). Defaults to "0".
*@li data_format: An optional string, Specify the data format of the input and output data. With the default format "NCHW".
*@par Outputs:
*y: An NCHW tensor of type float16, float32, int32.
*@attention Constraints:
*@li window[0] * window[1] < 256;
*@li 1<=input_h<=4096,1<=input_w<=4096
*@li If input tensor N is a prime number, it should be less than 65535.
*@par Third-party framework compatibility
*@li Compatible with the Caffe operator Pooling.
*@li Compatible with the TensorFlow operator Pooling.
*/
REG_OP(Pooling)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_INT32}))
    .ATTR(mode, Int, 0)                 // 0:max pooling or 1:avg pooling
    .ATTR(global_pooling, Bool, false)
    .ATTR(window, ListInt, {1,1})       // kernel size
    .ATTR(stride, ListInt, {1,1})       // stride size
    .ATTR(pad, ListInt, {0,0,0,0})      // pad size
    .ATTR(dilation, ListInt, {1,1,1,1})
    .ATTR(ceil_mode, Int, 0)
    .ATTR(data_format, String, "NCHW")
    .OP_END_FACTORY_REG(Pooling)

/**
* @brief Performs average pooling on the input. \n
* @par Inputs:
* x: A tensor of type float16, float32, double. \n

* @par Attributes:
* @li ksize: A required list of 4 ints, specifying the size (N, C, H, and W)
* of the sliding window, where N = C = 1, and H and W are positive integers
*  within the range [1, 255].
* @li strides: A required list of 4 ints, specifying the stride of the
* sliding window. The strides of the N and C dimensions are 1. The strides of
*  the H and W dimensions are positive integers within the range [1, 63].
* @li padding: A required string, specifying the padding algorithm,
 * either "VALID" or "SAME". With "SAME" means that the outputs will have the
 * same spatial dimensions as its inputs. With "VALID" means no padding.
* @li data_format: An optional string, specifying the data format of "ksize"
* and "strides", either "NCHW", or "NHWC" (default). \n

* @par Outputs:
* y: The average pooled output tensor. Has the same type and format
* as input "x". \n

* @attention Constraints:
* @li This operator applies only to a TensorFlow network.
* @li Only single input and single output are supported.
* @li Global pooling is supported.
* @li "ksize_H" and "ksize_W" are positive integers within the range [1, 255].
* ksize_H * ksize_W < 256
* @li Due to instruction restrictions,
 * the values of "strides_h" and "strides_w" are positive integers within
 * the range [1, 63].
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator AvgPool.
*/
REG_OP(AvgPool)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(AvgPool)

/**
* @brief Performs average pooling on the input.
* @par Inputs:
* x: A tensor of type float16, float32, double.

* @par Attributes:
* @li ksize: A required list of 4 ints, specifying the size (N, C, H, and W)
* of the sliding window, where N = C = 1,
 * and H and W are positive integers within the range [1, 255].
* @li strides: A required list of 4 ints, specifying the stride of the
 * sliding window. The strides of the N and C dimensions are 1.
 * The strides of the H and W dimensions are positive integers within
 * the range [1, 63].
* @li padding_mode: A required string, specifying the padding algorithm,
 * either "VALID", "SAME" and "CALCULATED".
 * With "SAME" means that the outputs will have the same spatial dimensions
 * as its inputs. With "VALID" means no padding.
* @li pads: Pad value when padding_mode is "CALCULATED".
* @li data_format: An optional string, specifying the data format of "ksize"
 * and "strides", either "NCHW", or "NHWC" (default).
* @li global_pooling: Global or not. If true, pads will change to {0,0,0,0}
* and ksize will change to [input_h, input_w].
* @li ceil_mode: Use ceil or floor to calculate the output size when
* padding_mode is "CALCULATED".
* @li exclusive: Ignore padding area or not when calculating average.
* @li divisor_override: An optional Int, its valid range is [1, 255], and the default value is zero.
* if specified, it will be used as divisor, otherwise size of the pooling region will be used.

* @par Outputs:
* y: The average pooled output tensor. Has the same type and format as
* input "x".

* @attention Constraints:
* @li Only single input and single output are supported.
* @li Global pooling is supported.
* @li "ksize_H" and "ksize_W" are positive integers within the range [1, 255].
* ksize_H * ksize_W < 256
* @li Due to instruction restrictions,
 * the values of "strides_h" and "strides_w" are positive integers within
 * the range [1, 63].
* @li If the sliding window range exceeds the original width and height of the input feature map,
 * and the calculation result of count_include_pad is False, the behavior of dividing by 0 will appear.
 * This scenario does not conform to the normal logic of the operator.
 * It is recommended to modify attributes such as ceil_mode or stride to satisfy that the sliding window
 * always has an intersection with the input feature map. In this abnormal scenario,
 * different chips may return different results, and four abnormal results may appear: 0, 65504, Nan, and INF.
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator AvgPoolV2.
*/
REG_OP(AvgPoolV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(padding_mode, String, "CALCULATED")
    .ATTR(pads, ListInt, {0, 0, 0, 0})
    .ATTR(data_format, String, "NCHW")
    .ATTR(global_pooling, Bool, false)
    .ATTR(ceil_mode, Bool, false)
    .ATTR(exclusive, Bool, true)
    .ATTR(divisor_override, Int, 0)
    .OP_END_FACTORY_REG(AvgPoolV2)

/**
* @brief Performs average pooling on the input. \n
* @par Inputs:
* @li x: A 5-D Tensor of shape [batch, depth, height, width, channels] and
* type float16. \n

* @par Attributes:
* @li ksize: List of ints that has length 1, 3 or 5. The size of the window
* for each dimension of the input tensor.
* @li strides:List of ints that has length 1, 3 or 5. The stride of the sliding
* window for each dimension of the input tensor.
* @li pads: List of ints, implicit zero paddings on both sides of the input.
* @li ceil_mode: When true, will use ceil instead of floor in the formula to
* compute the output shape.
* @li count_include_pad: When true, will include the zero-padding in the
* averaging calculation.
* @li divisor_override: if specified, it will be used as divisor, otherwise
* size of the pooling region will be used.
* @li data_format: A string, format of input data. \n

* @par Outputs:
* y: The average pooled output tensor. \n

* @attention Constraints:
* @li "ksize" is in the range [1, 255]. "strides" is in the range [1, 63].

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator AvgPool3D.
*/
REG_OP(AvgPool3D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(ceil_mode, Bool, false)
    .ATTR(count_include_pad, Bool, true)
    .ATTR(divisor_override, Int, 0)
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(AvgPool3D)


/**
* @brief Performs average pooling on the input.
* @par Inputs:
* @li x: A 5-D Tensor of shape [batch, depth, height, width, channels] and type float16.
* @li filter: An optional tensor of type float16, fractal_z_3d layout.
* @li multiplier: An optional tensor of float16.

* @par Attributes:
* @li ksize: List of ints that has length 1, 3 or 5. The size of the window for each dimension of the input tensor.
* @li strides:List of ints that has length 1, 3 or 5. The stride of the sliding window for each dimension of the input tensor.
* @li pads: List of ints, implicit zero paddings on both sides of the input.
* @li ceil_mode: When true, will use ceil instead of floor in the formula to compute the output shape.
* @li count_include_pad: When true, will include the zero-padding in the averaging calculation.
* @li divisor_override: if specified, it will be used as divisor, otherwise size of the pooling region will be used.
* @li data_format: A string, format of input data . \n

* @par Outputs:
* y: The average pooled output tensor . \n

* @attention Constraints:
* "ksize" is in the range [1, 255]. "strides" is in the range [1, 63]

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator AvgPool3D.

* @attention Constraints:
* The operator will not be enhanced in the future.
*/
REG_OP(AvgPool3DD)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OPTIONAL_INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OPTIONAL_INPUT(multiplier, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(ceil_mode, Bool, false)
    .ATTR(count_include_pad, Bool, true)
    .ATTR(divisor_override, Int, 0)
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(AvgPool3DD)

/**
* @brief Computes AvgPool3DGrad function. \n
* @par Inputs:
* @li orig_input_shape: An NDHWC tensor of type int32.
* @li grads: An NDHWC tensor of type float16. \n

* @par Attributes:
* @li ksize: List of ints that has length 5. The size of the window for
* each dimension of the input tensor.
* @li strides:List of ints that has length 5. The stride of the sliding
* window for each dimension of the input tensor.
* @li pads: List of ints, implicit zero paddings on both sides of the input.
* @li ceil_mode: When true, will use ceil instead of floor in the formula to
* compute the output shape.
* @li count_include_pad: When true, will include the zero-padding in the
* averaging calculation.
* @li divisor_override: if specified, it will be used as divisor, otherwise
* size of the pooling region will be used.
* @li data_format: A string, format of input data. \n

* @par Outputs:
* output: A mutable tensor with the same shape and type as "orig_input_shape".

* @attention Constraints:
* @li "ksize" is in the range [1, 255]. "strides" is in the range [1, 63]. \n

* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator AvgPoolGrad.
*/

REG_OP(AvgPool3DGrad)
    .INPUT(orig_input_shape, TensorType({DT_INT32}))
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(output, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(ceil_mode, Bool, false)
    .ATTR(count_include_pad, Bool, true)
    .ATTR(divisor_override, Int, 0)
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(AvgPool3DGrad)

/**
* @brief Performs average pooling on the input.
* @par Inputs:
* @li grads: An NDHWC tensor of type float16.
* @li filter: An optional tensor of type float16.
* @li multiplier: An optional tensor of float16.

* @par Attributes:
* @li orig_input_shape: List of ints that has length 5.
* The size of the window for each dimension of the input tensor.
* @li ksize: List of ints that has length 5.
* The size of the window for each dimension of the input tensor.
* @li strides:List of ints that has length 5.
* The stride of the sliding window for each dimension of the input tensor.
* @li pads: List of ints, implicit zero paddings on both sides of the input.
* @li ceil_mode: When true, will use ceil instead of floor
* in the formula to compute the output shape.
* @li count_include_pad: When true, will include the zero-padding
* in the averaging calculation.
* @li divisor_override: if specified, it will be used as divisor,
* otherwise size of the pooling region will be used.
* @li data_format: A string, format of input data. \n

* @par Outputs:
* output: The average pooled output tensor . \n

* @attention Constraints:
* "ksize" is in the range [1, 255]. "strides" is in the range [1, 63]

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator AvgPool3DGradD.

* @attention Constraints:
* The operator will not be enhanced in the future.
*/
REG_OP(AvgPool3DGradD)
    .INPUT(grads, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(filter, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(multiplier, TensorType({DT_FLOAT16}))
    .OUTPUT(output, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(orig_input_shape, ListInt)
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(ceil_mode, Bool, false)
    .ATTR(count_include_pad, Bool, true)
    .ATTR(divisor_override, Int, 0)
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(AvgPool3DGradD)

/**
* @brief Performs max_pool_ext2 on the input . \n

* @par Inputs:
* One input:
* x: A Tensor of type float16.


* @par Attributes:
* @li ksize: A required list of int8, int16, int32, or int64 values,
* specifying the size of the window for each dimension of the input tensor. No default value.
* @li strides: A required list of int8, int16, int32, or int64 values,
* specifying the stride of the sliding window for each dimension of the input tensor. No default value.
* @li padding: A required string. No default value.
* @li data_format: An optional string . \n

* @par Outputs:
* y: A Tensor. Has the same type and format as input "x" . \n

* @attention Constraints:
* @li "ksize" is a list that has length 4: ksize[0] = 1 or ksize[3] = 1, ksize[1] * ksize[2] <= 255.
* @li "stride is a list that has length 4: strides[0] = 1 or strides[3] = 1,
 * strides[1] <= 63, strides[0] >= 1, strides[2] <= 63, strides[2] >= 1.
* @li "padding" is either "SAME" or "VALID" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPoolV2.
*/
REG_OP(MaxPoolExt2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE, DT_INT8,
                          DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                          DT_UINT16, DT_QINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE, DT_INT8,
                           DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                           DT_UINT16, DT_QINT8}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(MaxPoolExt2)

/**
* @brief Performs max pooling on the input . \n

* @par Inputs:
* One input:
* x: A Tensor. Supported type:float16, float32, double, int8, int16,
* int32, int64, uint8, uint16, qint8

* @par Attributes:
* @li ksize: A required list of int8, int16, int32, or int64 values,
* specifying the size of the window for each dimension of the input tensor.
* No default value.
* @li strides: A required list of int8, int16, int32, or int64 values,
* specifying the stride of the sliding window for each dimension of
* the input tensor. No default value.
* @li padding: A required string. No default value.
* @li data_format: An optional string. Defaults to "NHWC" . \n

* @par Outputs:
* y: A Tensor. Has the same type and format as input "x" . \n

* @attention Constraints:
* @li "ksize" is a list that has length 4: ksize[0] = 1 or ksize[3] = 1,
* ksize[1] * ksize[2] <= 255.
* @li "stride is a list that has length 4: strides[0] = 1 or strides[3] = 1,
* strides[1] <= 63, strides[0] >= 1, strides[2] <= 63, strides[2] >= 1.
* @li "padding" is either "SAME" or "VALID".


* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPool.
*/
REG_OP(MaxPool)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE, DT_INT8,
                          DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                          DT_UINT16, DT_QINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE, DT_INT8,
                           DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_QINT8}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(MaxPool)

/**
* @brief Performs max 3d pooling on the input . \n

* @par Inputs:
* x: A Tensor. Supported type float16, float32, double . \n

* @par Attributes:
* @li ksize: A required list of int8, int16, int32, or int64 values,
specifying the size of the window for each dimension of the input tensor.
No default value.
* @li strides: A required list of int8, int16, int32, or int64 values,
specifying the stride of the sliding window for each dimension of
the input tensor. No default value.
* @li padding: A required string type of float16.
* @li pads: A list type of int32. Default value {0,0,0,0,0,0}.
* @li dilation: A list type of int32. Default value {1,1,1,1,1,1}.
* @li ceil_mode: A ceil mode number of int32 . Default value 0.
* @li data_format: An optional string. Defaults to "NDHWC" . \n

* @par Outputs:
* y: A Tensor. Has the same type and format as input "x" . \n

* @attention Constraints:
* @li "ksize" is a list that has length 4: ksize[0] = 1 or ksize[3] = 1,
 * ksize[1] * ksize[2] <= 255.
* @li "stride is a list that has length 4: strides[0] = 1 or strides[3] = 1,
 * strides[1] <= 63, strides[0] >= 1, strides[2] <= 63, strides[2] >= 1.
* @li "padding" is either "SAME" or "VALID" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPool3D.
*/
REG_OP(MaxPool3D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(pads, ListInt, {0,0,0,0,0,0})
    .ATTR(dilation, ListInt, {1, 1, 1, 1, 1})
    .ATTR(ceil_mode, Int, 0)
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(MaxPool3D)

/**
* @brief Performs max pooling3d on both max values and indices.
*
* @par Inputs:
*  One input:
*  x: An 6D tensor. Supported type: float16. Format as NDC1HWC0.
* @par Attributes:
*  @li ksize: A required list of int32 values,
*   specifying the size of the window for each dimension of the input tensor.
*   No default value.
*  @li strides: A required list of int32 values,
*   specifying the stride of the sliding window for each dimension of
*   the input tensor. No default value.
*  @li pads: A required 3*2-dimension-list of int32 values.
*   specifying the pad of three dimension of input, implement with 0.
*  @li dilation: dilation of kernel. default value is {1,1,1,1,1}.
*  @li ceil_mode: default value is false.
*  @li data_format: the format of torch input, default value is "NCDHW".
*  @li argmax_type: the function of this field is to determine the type of
*   output argmax, "bitmask" is the default value, the argmax will return
*   a img2col bitmask. "index_int32" and "index_int64" represent the torch
*   output indices.
* @par Outputs:
*  y: An 6D tensor. the maxpool3d output(max value), format as NDoC1HoWoC0.
* @par Outputs:
*  argmax: A 5D uint16 tensor. the indice output.
*/
REG_OP(MaxPool3DWithArgmax)
    .INPUT(x, TensorType::RealNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .OUTPUT(argmax, TensorType::IndexNumberType())
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilation, ListInt, {1, 1, 1, 1, 1})
    .ATTR(ceil_mode, Bool, false)
    .ATTR(data_format, String, "NCDHW")
    .ATTR(argmax_type, String, "bitmask")
    .OP_END_FACTORY_REG(MaxPool3DWithArgmax)

/**
* @brief Applies a 2D adaptive max pooling over an input signal conposed of several input planes. \n
* The output is of size H x W, for any input size.

* @par Inputs:
* One input, including:
* @li x: A Tensor. Must be one of the following data types:
*     float16, float32, float64. \n

* @par Attributes:
* @li output_size: A required list of 2 ints
*    specifying the size (H,W) of the output tensor. \n

* @par Outputs:
* @li y: A Tensor. Has the same data type as "x" \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator AdaptiveMaxPool2d.
*/
REG_OP(AdaptiveMaxPool2d)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE}))
    .OUTPUT(argmax, TensorType::IndexNumberType())
    .REQUIRED_ATTR(output_size, ListInt)
    .OP_END_FACTORY_REG(AdaptiveMaxPool2d)

/**
* @brief Computes second-order gradients of the maxpooling3d function . \n

* @par Inputs:
* @li orig_x: Original forward input tensor(NDC1HWC0) of type float16
* @li orig_y: Original forward output tensor(NDC1HWC0) of type float16
* @li grads: Gradient tensor(NDC1HWC0) of type float16
* @li assist: Assist tensor(NDC1HWC0) of type float16

* @par Attributes:
* @li ksize: A required list or tuple,
* specifying the size of the sliding window.
* @li strides: A required list or tuple,
* specifying the stride of the sliding window.
* @li pads: A required list or tuple
* @li padding: A required string, window sliding mode. Either SAME or VALID.
* @li data_format: An optional string.
* Format of the original input, either NCDHW or NDHWC. Defaults to NDHWC . \n

* @attention Constraints:
* @li Only the Ascend 910 platform is supported.
* @li "orig_x" and "grads" must have the same shape.
* @li "orig_y" and "y" must have the same shape. Otherwise, an error is reported.
* @li "orig_x", "orig_y", "grads", and "y" must be NDC1HWC0 tensors . \n

* @par Outputs:
* @li y: Result tensor of type float16

* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator MaxPool3DGradGrad.
*/

REG_OP(MaxPool3DGradGrad)
    .INPUT(orig_x, TensorType::RealNumberType())
    .INPUT(orig_y, TensorType::RealNumberType())
    .INPUT(grads, TensorType::RealNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(MaxPool3DGradGrad)


/**
* @brief Computes gradients of the maxpooling function . \n

* @par Inputs:
* @li x1: A mutable tensor of type RealNumberType.
* @li x2: A mutable tensor of type RealNumberTypex.
* @li grad: A mutable tensor of type RealNumberType . \n

* @par Attributes:
* @li ksize: A required tuple or list, specifying the size of the window for
* each dimension of the input tensor.
* @li strides: A required tuple or list, specifying the stride of the sliding
* window for each dimension of the input tensor.
* @li padding: A required string, specifying the type of padding algorithm
* to use.
* @li data_format: An optional string, Specify the data format of the input and
* output data. With the default format "NHWC" . \n

* @par Outputs:
* y: A mutable tensor. Has the same shape and type as "x1" . \n

* @attention Constraints:
* @li ksize is limited by buffer with full tiling.
* @li "ksize" is in the range [1, 255]. "strides" is in the range [1, 63]

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPoolGrad.
*/
REG_OP(MaxPoolGrad)
    .INPUT(x1, TensorType::RealNumberType())
    .INPUT(x2, TensorType::RealNumberType())
    .INPUT(grad, TensorType::RealNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(MaxPoolGrad)

/**
* @brief Computes second-order gradients of the maxpooling function . \n

* @par Inputs:
* @li x1: Original forward input tensor. Supported type:float, double, int32,
 * uint8, int16, int8, int64, uint16, half, uint32, uint64.
* @li x2: Has the same type and format as input "x1".
* @li grad:Has the same type and format as input "x1" . \n

* @par Attributes:
* @li ksize: A required list or tuple,
* specifying the size of the sliding window.
* @li strides: A required list or tuple,
* specifying the stride of the sliding window.
* @li padding: A required string, window sliding mode. Either SAME or VALID.
* @li data_format: An optional string.
* Format of the original input, either NCHW or NHWC. Defaults to NHWC . \n

* @attention Constraints:
* @li Only the Ascend 910 platform is supported.
* @li "x1" and "grads" must have the same shape.
* @li "x2" and "y" must have the same shape. Otherwise, an error is reported.
* @li "x1", "x2", "grads", and "y" must be 5D tensors.
* @li ksize[H] and ksize[W] is in the range [1, 255].
* @li strides[H] and strides[W] is in the range [1, 63].
* @li Other dimensions of ksize and strides is 1 . \n

* @par Outputs:
* y: Has the same type and format as input "x1" . \n

* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator MaxPoolGradGrad.
*/
REG_OP(MaxPoolGradGrad)
    .INPUT(x1, TensorType::RealNumberType())
    .INPUT(x2, TensorType::RealNumberType())
    .INPUT(grad, TensorType::RealNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(MaxPoolGradGrad)

/**
*@brief Performs max_pool_ext2 on the input . \n

*@par Inputs:
* Three inputs:
*@li x: A Tensor of type float16.
*@li strides: A required type of int32 values,
 * specifying the stride of the sliding window for each dimension of the input tensor. No default value.
*@li ksize: A required type of int32 values,
 * specifying the size of the window for each dimension of the input tensor. No default value.


*@par Attributes:
*@li padding: A required string. No default value.
*@li data_format: An optional string. \n

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x" . \n

*@attention Constraints:
*@li "ksize" is a list that has length 4: ksize[0] = 1 or ksize[3] = 1, ksize[1] * ksize[2] <= 255.
*@li "stride is a list that has length 4: strides[0] = 1 or strides[3] = 1, strides[1] <= 63, strides[0] >= 1,
 * strides[2] <= 63, strides[2] >= 1.
*@li "padding" is either "SAME" or "VALID" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPoolV2.
*/
REG_OP(MaxPoolV2)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(ksize, TensorType({DT_INT32}))
    .INPUT(strides, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(padding, String)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(MaxPoolV2)

/**
* @brief Performs max pooling on the input and outputs both max values and
 * indices . \n

* @par Inputs:
* One input:
* x: An 4D Tensor. Supported type: float, double, int32,
 * uint8, int16, int8, int64, uint16, half, uint32, uint64.
 * Must set the format, supported format list ["NCHW, NHWC"]. \n

* @par Attributes:
* @li ksize: A required list of int8, int16, int32, or int64 values,
 * specifying the size of the window for each dimension of the input tensor.
 * No default value.
* @li strides: A required list of int8, int16, int32, or int64 values,
 * specifying the stride of the sliding window for each dimension of
 * the input tensor. No default value.
* @li padding: A required string. No default value .
* @li Targmax:An optional int with default value 7 . \n

* @par Outputs:
* @li y: A Tensor. Has the same type and format as input "x".
* @li argmax: A Tensor. Has the same type and format as input "x".
* @attention Constraints:
* @li "ksize" is a list that has length 4: ksize[0] = 1 or ksize[3] = 1,
 * ksize[1] * ksize[2] <= 255.
* @li "stride is a list that has length 4: strides[0] = 1 or strides[3] = 1,
 * strides[1] <= 63, strides[0] >= 1, strides[2] <= 63, strides[2] >= 1.
* @li "padding" is either "SAME" or "VALID" .

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPoolWithArgmax.
*/
REG_OP(MaxPoolWithArgmax)
    .INPUT(x, TensorType::RealNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .OUTPUT(argmax, TensorType::IndexNumberType())
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(Targmax, Int, 7)
    .OP_END_FACTORY_REG(MaxPoolWithArgmax)

/**
* @brief Performs the backpropagation of MaxPoolWithArgmax . \n

* @par Inputs:
* Three inputs, including:
* @li x: An 4d tensor. Supported type: float, double, int32,
 * uint8, int16, int8, int64, uint16, half, uint32, uint64.
 * Must set the format, supported format list ["NCHW, NHWC"]
* @li grad: An 4d tensor. Supported type: float, double, int32,
 * uint8, int16, int8, int64, uint16, half, uint32, uint64.
 * Must set the format, supported format list ["NCHW, NHWC"]
*@li argmx: A tensor of type int32 or int64 . \n

* @par Attributes:
* @li ksize: A required list of int8, int16, int32, or int64 values,
 * specifying the size of the window for each dimension of the input tensor.
 * No default value.
* @li strides: A required list of int8, int16, int32, or int64 values,
 * specifying the stride of the sliding window for each dimension of
 * the input tensor. No default value.
* @li padding: A required string. No default value . \n

* @par Outputs:
* y: A Tensor. Has the same type and format as input "x" . \n

* @attention Constraints:
* @li "ksize" is a list that has length 4: ksize[0] = 1 or ksize[3] = 1,
 * ksize[1] * ksize[2] <= 255.
* @li "strides" is a list that has length 4: strides[0] = 1 or strides[3] = 1
* @li "padding" is either "SAME" or "VALID". \n


* @see max_pool_with_argmax
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPoolGradWithArgmax.
*/
REG_OP(MaxPoolGradWithArgmax)
    .INPUT(x, TensorType::RealNumberType())
    .INPUT(grad, TensorType::RealNumberType())
    .INPUT(argmax, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .OP_END_FACTORY_REG(MaxPoolGradWithArgmax)

/**
* @brief Performs transform mask to argmax . \n

* @par Inputs:
* Two inputs:
* @li x: A Tensor of type float16.
* @li mask: A Tensor of type uint16 . \n

* @par Attributes:
* @li ksize: A required list of int8, int16, int32, or int64 values, specifying the size of the window for each dimension of the input tensor. No default value.
* @li strides: A required list of int8, int16, int32, or int64 values, specifying the stride of the sliding window for each dimension of the input tensor. No default value.
* @li padding: A required string. No default value .
* @li originshape:A required list of int8, int16, int32, or int64 values, No default value. \n

* @par Outputs:
*argmax: A Tensor of type int32 . \n

* @attention Constraints:
*@li "ksize" is a list that has length 4: ksize[0] = 1 or ksize[3] = 1, ksize[1] * ksize[2] <= 255.
*@li "stride is a list that has length 4: strides[0] = 1 or strides[3] = 1, strides[1] <= 63, strides[0] >= 1, strides[2] <= 63, strides[2] >= 1.
*@li "padding" is either "SAME" or "VALID" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Mask2Argmax.
*/
REG_OP(Mask2Argmax)
    .INPUT(x, TensorType::RealNumberType())
    .INPUT(mask, TensorType::IndexNumberType())
    .OUTPUT(argmax, TensorType::IndexNumberType())
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .REQUIRED_ATTR(originshape, ListInt)
    .OP_END_FACTORY_REG(Mask2Argmax)

/**
* @brief Computes second-order gradients of the maxpooling function . \n

* @par Inputs:
* @li x: Original forward input tensor. Supported type: float, double, int32,
 * uint8, int16, int8, int64, uint16, half, uint32, uint64.
* @li grad: Gradient tensor. Supported type: float, double, int32,
 * uint8, int16, int8, int64, uint16, half, uint32, uint64.
* @li argmax: An tensor of type int32 or int64.
* @par Attributes:
* @li ksize: A required list, specifying the size of the sliding window.
* @li strides: A required list, specifying the stride of the sliding window.
* @li padding: A required string, window sliding mode. Either SAME or VALID.
* @par Outputs:
* y:Result tensor. Supported type: float, double, int32,
 * uint8, int16, int8, int64, uint16, half, uint32, uint64

* @attention Constraints:
* @li Only the cloud platform is supported.
* @li "x1" and "grads" must have the same shape.
* @li length of the shape of x, grads, argmax, y must be 5.
* @li shape of argmax must be (fmap_n, fmap_c1, kernel_h * kernel_w,
* (shape_max_pool[2] * shape_max_pool[3] + 15) // 16 * 16, 1),
* or (fmap_n, fmap_c1, kernel_h * kernel_w,
* (shape_max_pool[2] * shape_max_pool[3] + 31) // 16, 16), else failed . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPoolGradGradWithArgmax.
*/
REG_OP(MaxPoolGradGradWithArgmax)
    .INPUT(x, TensorType::RealNumberType())
    .INPUT(grad, TensorType::RealNumberType())
    .INPUT(argmax, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .OP_END_FACTORY_REG(MaxPoolGradGradWithArgmax)

/**
* @brief Computes avgpoograd function. \n
* @par Inputs:
* @li orig_input_shape: A tensor of type int32.
* @li input_grad: A tensor of type float16. \n

* @par Attributes:
* @li ksize: A required tuple or list, specifying the size of the window for
* each dimension of the input tensor.
* @li strides: A required tuple or list, specifying the stride of the sliding
* window for each dimension of the input tensor.
* @li padding: A required string, specifying the type of
* the padding algorithm to use.
* @li data_format: An optional string. Defaults to "NHWC". \n

* @par Outputs:
* out_grad: A mutable tensor with the same shape and type as "orig_input_shape". \n

* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator AvgPoolGrad.
* TensorFlow operator AvgPoolGrad are difference from our operator AvgPoolGrad in
* following two case:
* -- kernel_h > input_h && hernel_h // 2 < input_h - 1
* -- kernel_w > input_w && hernel_w // 2 < input_w - 1
*/
REG_OP(AvgPoolGrad)
    .INPUT(orig_input_shape, TensorType({DT_INT32}))
    .INPUT(input_grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(out_grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(AvgPoolGrad)

/**
* @brief Computes gradients of average pooling function . \n
* @par Inputs:
* @li input_grad: An NHWC tensor of type float16.
* @li mean_matrix: Assist matrix, an NHWC tensor of type float16.
* @li kernel_matrix: Assist matrix, an NHWC tensor of type float16.

* @par Attributes:
* @li orig_input_shape: A required Original input dimensions.
* @li ksize: A required tuple or list, specifying the size of the window
* for each dimension of the input tensor.
* @li strides: A required tuple or list, specifying the stride of
* the sliding window for each dimension of the input tensor.
* @li padding: A required string, specifying the type of the padding algorithm
* to use.
* @li data_format: An optional string. Defaults to "NHWC" . \n

* @par Outputs:
* @li out_grad: A mutable tensor with the same shape and type as "orig_input".
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use AvgPoolGrad instead.
*/
REG_OP(AvgPoolGradD)
    .INPUT(input_grad, TensorType({DT_FLOAT16}))
    .INPUT(mean_matrix, TensorType({DT_FLOAT16}))
    .INPUT(kernel_matrix, TensorType({DT_FLOAT16}))
    .OUTPUT(out_grad, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(orig_input_shape, ListInt)
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(AvgPoolGradD)

/**
* @brief Computes avgpoolv2grad function. \n
* @par Inputs:
* @li orig_input_shape: An NHWC tensor of type int32.
* @li input_grad: An NHWC tensor of type float16, float32, or double. \n

* @par Attributes:
* @li ksize: A required tuple or list, specifying the size of the window for
* each dimension of the input tensor.
* @li strides: A required tuple or list, specifying the stride of the sliding
* window for each dimension of the input tensor.
* @li padding_mode: A required string, specifying the type of
* the padding algorithm to use.
* @li global_pooling: Whether to use the global pooling. If global_pooling =
* true, ksize and pads will be ignored. Default False.
* @li ceil_mode: Whether to use the ceil function to calculate output height
* and width. Default False.
* @li exclusive: Whether to exclude padding points. default is true.
* @li data_format: An optional string. Defaults to "NHWC". \n

* @par Outputs:
* @li out_grad: A mutable tensor with the same shape and type as "orig_input". \n

* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator AvgPoolGrad.
*/
REG_OP(AvgPoolV2Grad)
    .INPUT(orig_input_shape, TensorType({DT_INT32}))
    .INPUT(input_grad, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE}))
    .OUTPUT(out_grad, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(padding_mode, String, "CALCULATED")
    .ATTR(pads, ListInt, {0,0,0,0})
    .ATTR(data_format, String, "NCHW")
    .ATTR(global_pooling, Bool, false)
    .ATTR(ceil_mode, Bool, false)
    .ATTR(exclusive, Bool, true)
    .OP_END_FACTORY_REG(AvgPoolV2Grad)
/**
* @brief Computes gradients of averagev2 pooling function.
* @par Inputs:
* input_grad: An NHWC tensor of type float16, float32, or double.

* @par Attributes:
* @li orig_input_shape: A required tuple or list of type int32.
* @li ksize: A required tuple or list, specifying the size of the window for
* each dimension of the input tensor.
* @li strides: A required tuple or list, specifying the stride of the sliding
* window for each dimension of the input tensor.
* @li padding_mode: A required string, specifying the type of
* the padding algorithm to use.
* @li global_pooling: Whether to use the global pooling. If global_pooling=true,
* ksize and pads will be ignored. Default False.
* @li ceil_mode: Whether to use the ceil function to calculate output height and
* width. Default False.
* @li exclusive: Whether to exclude padding points. default is true.
* @li data_format: An optional string. Defaults to "NHWC".

* @par Outputs:
* out_grad: A mutable tensor with the same shape and type as "orig_input".

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator AvgPoolGrad.

* @attention Constraints:
* The operator will not be enhanced in the future.
*/
REG_OP(AvgPoolV2GradD)
    .INPUT(input_grad, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(mean_matrix, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(kernel_matrix, TensorType({DT_FLOAT16}))
    .OUTPUT(out_grad, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(orig_input_shape, ListInt)
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(padding_mode, String, "CALCULATED")
    .ATTR(pads, ListInt, {0,0,0,0})
    .ATTR(data_format, String, "NCHW")
    .ATTR(global_pooling, Bool, false)
    .ATTR(ceil_mode, Bool, false)
    .ATTR(exclusive, Bool, true)
    .OP_END_FACTORY_REG(AvgPoolV2GradD)

/**
* @brief upsample the layer, similar to the nearest-neighbor difference scaling algorithm.

* @par Inputs:
* one input, including:
* x: A tensor of type float16 or float32.
* @par Attributes:
* @li  scale: A optional float32, scale factor of x. Defaults to "1.0".
* @li  stride_h: An optional int32, broadcast the axis of h. Defaults to "2".
* @li  stride_w: An optional int32, broadcast the axis of w. Defaults to "2".
* @par Outputs:
*y: A tensor of type float16 or float32.
*/
REG_OP(Upsample)
   .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
   .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
   .ATTR(scale, Float, 1)
   .ATTR(stride_h, Int, 2)
   .ATTR(stride_w, Int, 2)
   .OP_END_FACTORY_REG(Upsample)

/**
* @brief Computes gradient of the FractionalMaxPool function . \n

* @par Inputs:
* Inputs include:
* @li orig_input: A Tensor. Must be one of the following types: float32, float64, int32, int64.
* @li orig_output: A Tensor. Must have the same type as orig_input.
* @li out_backprop: A Tensor. Must have the same type as orig_input.
      4-D with shape [batch, height, width, channels].
* @li row_pooling_sequence: A Tensor of type int64.
* @li col_pooling_sequence: A Tensor of type int64 . \n

* @par Attributes:
* overlapping: An optional bool. Defaults to False . \n

* @par Outputs:
* y: A Tensor. Has the same type as orig_input . \n

* @attention Constraints:
* The implementation for FractionalMaxPoolGrad on Ascend uses AICPU, with bad performance.

* @par Third-party framework compatibility
* @li compatible with tensorflow FractionalMaxPoolGrad operator.
*/
REG_OP(FractionalMaxPoolGrad)
    .INPUT(orig_input, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .INPUT(orig_output, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .INPUT(out_backprop, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .INPUT(row_pooling_sequence, TensorType({ DT_INT64 }))
    .INPUT(col_pooling_sequence, TensorType({ DT_INT64 }))
    .OUTPUT(y, TensorType({ DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64 }))
    .ATTR(overlapping, Bool, false)
    .OP_END_FACTORY_REG(FractionalMaxPoolGrad)

/**
*@brief Performs fractional average pooling on the input . \n

*@par Inputs:
*Inputs include:
*x: A Tensor. Must be one of the following types: float32, float64, int32, int64.
 4-D with shape [batch, height, width, channels] . \n

*@par Attributes:
*@li pooling_ratio: A list of floats that has length >= 4.
*@li pseudo_random: An optional bool. Defaults to False.
*@li overlapping: An optional bool. Defaults to False. When set to True, it means when pooling.
*@li deterministic: An optional bool. Defaults to False.
*@li seed: An optional int. Defaults to 0.
*@li seed2: An optional int. Defaults to 0 . \n

*@par Outputs:
*@li y: A Tensor. Has the same type as x.
*@li row_pooling_sequence: A Tensor of type int64.
*@li col_pooling_sequence: A Tensor of type int64 . \n

*@attention Constraints:
*The implementation for FractionalAvgPool on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with tensorflow FractionalAvgPool operator.
*/
REG_OP(FractionalAvgPool)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .OUTPUT(row_pooling_sequence, TensorType({DT_INT64}))
    .OUTPUT(col_pooling_sequence, TensorType({DT_INT64}))
    .ATTR(pooling_ratio, ListFloat, {})
    .ATTR(pseudo_random, Bool, false)
    .ATTR(overlapping, Bool, false)
    .ATTR(deterministic, Bool, false)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(FractionalAvgPool)

/**
*@brief Performs fractional max pooling on the input . \n

*@par Inputs:
*Inputs include:
*x: A Tensor. Must be one of the following types: float32, float64, int32, int64.
 4-D with shape [batch, height, width, channels] . \n

*@par Attributes:
*@li pooling_ratio: A list of floats that has length >= 4. Pooling ratio for each dimension of value.
*@li pseudo_random: An optional bool. Defaults to False.
*@li overlapping: An optional bool. Defaults to False.
*@li deterministic: An optional bool. Defaults to False.
*@li seed: An optional int. Defaults to 0.
*@li seed2: An optional int. Defaults to 0 . \n

*@par Outputs:
*@li y: A Tensor. Has the same type as x.
*@li row_pooling_sequence: A Tensor of type int64.
*@li col_pooling_sequence: A Tensor of type int64 . \n

*@attention Constraints:
*The implementation for FractionalMaxPool on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with tensorflow FractionalMaxPool operator.
*/
REG_OP(FractionalMaxPool)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .OUTPUT(row_pooling_sequence, TensorType({DT_INT64}))
    .OUTPUT(col_pooling_sequence, TensorType({DT_INT64}))
    .ATTR(pooling_ratio, ListFloat, {})
    .ATTR(pseudo_random, Bool, false)
    .ATTR(overlapping, Bool, false)
    .ATTR(deterministic, Bool, false)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(FractionalMaxPool)

/**
*@brief Finds values of the n-th order statistic for the last dimension . \n

*@par Inputs:
*Inputs include:
* @li x: A Tensor. Must be one of the following types: float32, float64, int32, uint8,
      int16, int8, int64, bfloat16, uint16, half, uint32, uint64.
* @li n: A Tensor of type int32. 0-D . \n

*@par Attributes:
*reverse: An optional bool. Defaults to False . \n

*@par Outputs:
*y: A Tensor. Has the same type as x . \n

*@attention Constraints:
*The implementation for NthElement on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with tensorflow NthElement operator.
*/
REG_OP(NthElement)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16,
                          DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .INPUT(n, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16,
                          DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .ATTR(reverse, Bool, false)
    .OP_END_FACTORY_REG(NthElement)

/**
*@brief Computes gradient of the FractionalAvgPool function . \n

*@par Inputs:
*Inputs include:
* @li orig_input_tensor_shape: A Tensor of type int64.
* @li out_backprop: A Tensor. Must be one of the following types: float32, float64,
      int32, int64. 4-D with shape [batch, height, width, channels].
* @li row_pooling_sequence: A Tensor of type int64.
* @li col_pooling_sequence: A Tensor of type int64 . \n

*@par Attributes:
*overlapping: An optional bool. Defaults to False . \n

*@par Outputs:
*y: A Tensor. Has the same type as out_backprop . \n

*@attention Constraints:
*The implementation for FractionalAvgPoolGrad on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with tensorflow FractionalAvgPoolGrad operator.
*/
REG_OP(FractionalAvgPoolGrad)
    .INPUT(orig_input_tensor_shape, TensorType({DT_INT64}))
    .INPUT(out_backprop, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .INPUT(row_pooling_sequence, TensorType({DT_INT64}))
    .INPUT(col_pooling_sequence, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .ATTR(overlapping, Bool, false)
    .OP_END_FACTORY_REG(FractionalAvgPoolGrad)

/**
*@brief Returns the permuted vector/tensor in the destination data format given the . \n

*@par Inputs:
*Inputs include:
*x: A Tensor. Must be one of the following types: int32, int64. Vector of size 4
 or Tensor of shape (4, 2) in source data format . \n

*@par Attributes:
*@li src_format: An optional string. Defaults to "NHWC". source data format.
*@li dst_format: An optional string. Defaults to "NCHW". destination data format . \n

*@par Outputs:
*y: A Tensor. Has the same type as x . \n

*@attention Constraints:
*The implementation for DataFormatVecPermute on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with tensorflow DataFormatVecPermute operator.
*/
REG_OP(DataFormatVecPermute)
    .INPUT(x, TensorType({ DT_INT32, DT_INT64 }))
    .OUTPUT(y, TensorType({ DT_INT32, DT_INT64 }))
    .ATTR(src_format, String, "NHWC")
    .ATTR(dst_format, String, "NCHW")
    .OP_END_FACTORY_REG(DataFormatVecPermute)

/**
* @brief Computes gradients of the MaxPool3D function . \n

* @par Inputs:
* @li orig_x: A mutable NDC1HWC0 tensor of type float16.
* @li orig_y: A mutable NDC1HWC0 tensor of type float16.
* @li grads: A mutable NDC1HWC0 tensor of type float16 . \n

* @par Attributes:
* @li ksize: A required tuple or list, specifying the size of the window for
* each dimension of the input tensor.
* @li strides: A required tuple or list, specifying the stride of the sliding
* window for each dimension of the input tensor.
* @li pads: A list of 6 ints. Supports only padding along the D,
* H and W dimensions in sequence of head, tail, top, bottom, left and right.
* to use.
* @li data_format: An optional string, Specify the data format of the input and
* output data. With the default format "NDHWC" . \n

* @par Outputs:
* y: A mutable tensor. Has the same shape as "orig_x", but type is float32 . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPool3DGrad.
*/
REG_OP(MaxPool3DGrad)
    .INPUT(orig_x, TensorType::RealNumberType())
    .INPUT(orig_y, TensorType::RealNumberType())
    .INPUT(grads, TensorType::RealNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(padding, String, "SAME")
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(MaxPool3DGrad)

/**
*@brief Performs AvgPool1D on the input . \n
*@par Inputs:
*x: A Tensor. Must be one of the following types: int8, uint8, int16, int32, int64, float16, float32, float64 . \n

*@par Attributes:
*@li ksize: An required int, specifying the size of the window.
*@li strides: An required int.
*@li pads: A required tuple or list.
*@li ceil_mode: An optional bool. Defaults to False.
*@li count_include_pad: An optional bool. Defaults to False . \n

*@par Outputs:
*y: A Tensor. Has the same type as x . \n

*@par Third-party framework compatibility
*@li compatible with pytorch AvgPool1D operator.
*/
REG_OP(AvgPool1D)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(ksize, Int)
    .REQUIRED_ATTR(strides, Int)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(ceil_mode, Bool, false)
    .ATTR(count_include_pad, Bool, false)
    .OP_END_FACTORY_REG(AvgPool1D)

/**
*@brief Performs AvgPool1D on the input . \n
*@par Inputs:
*x: A Tensor. Must be one of the following types: int8, uint8, int16, int32, int64, float16, float32, float64 . \n

*@par Attributes:
*@li ksize: An required int, specifying the size of the window.
*@li strides: An required int.
*@li pads: A required tuple or list.
*@li ceil_mode: An optional bool. Defaults to False.
*@li count_include_pad: An optional bool. Defaults to False . \n

*@par Outputs:
*y: A Tensor. Has the same type as x . \n

*@par Third-party framework compatibility
*@li compatible with pytorch AvgPool1D operator.
*
*@par Restrictions:
*Warning: THIS FUNCTION IS DEPRECATED. Please use AvgPool1D instead.
*/
REG_OP(AvgPool1DD)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(assist_matrix, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(ksize, Int)
    .REQUIRED_ATTR(strides, Int)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(ceil_mode, Bool, false)
    .ATTR(count_include_pad, Bool, false)
    .OP_END_FACTORY_REG(AvgPool1DD)
/**
* @brief Performs max pooling on the input and outputs both max values and indices . \n

* @par Inputs:
* One input:
* x: An 5hd Tensor of type float16.
* Must set the format, supported format list ["NC1HWC0"].
* @par Attributes:
* @li ksize: A required list of int8, int16, int32, or int64 values,
* specifying the size of the window for each dimension of the input tensor. No default value.
* @li strides: A required list of int8, int16, int32, or int64 values,
* specifying the stride of the sliding window for each dimension of the input tensor. No default value.
* @li pads: A required list of int8, int16, int32, or int64 values,
* specifying the pad of the input feature map. No default value. \n
* @li dtype: A optional int. default value is 3.
* @li dilation: A optional list of int8, int16, int32, or int64 values.
* @li ceil_mode: A optional bool. default value is false . \n

* @par Outputs:
* y: A Tensor. Has the same type and format as input "x".
* argmax:  A Tensor. type:uint16.
* @attention Constraints:
* @li ksize: a list that has length 4:
* ksize[0] = 1, ksize[1] = 1, ksize[2] * ksize[3] <= (ub_size-8)*1024//6//2//16.
* @li strides: a list that has length 4:
* strides[0] = 1, strides[1] = 1, 1 <= strides[2] <= 2048, 1 <= strides[3] <= 2048.
* @li pads: a list that has length 4:
* pads[0] = 1, pads[1] = 1, 1 <= pads[2] <= (ksize[2]//2), 1 <= pads[3] <= (ksize[3]//2).
* @li dilation: a list that has length 4.
* @li ceil_mode: is a bool, default is false . \n

* @par Third-party framework compatibility
* Compatible with the PyTorch operator max_pool2d_with_indices.
*/
REG_OP(MaxPoolWithArgmaxV2)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .OUTPUT(argmax, TensorType({DT_UINT16}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dtype, Int, 3)
    .ATTR(dilation, ListInt, {1, 1, 1, 1})
    .ATTR(ceil_mode, Bool, false)
    .OP_END_FACTORY_REG(MaxPoolWithArgmaxV2)

/**
* @brief Performs the backpropagation of MaxPoolWithArgmaxV2. \n

* @par Inputs:
* Three inputs, including:
* @li x: An 5hd tensor of type float16.
* Must set the format, supported format list ["NC1HWC0"]
* @li grad: An 5hd tensor of type float16.
* Must set the format, supported format list ["NC1HWC0"]
* @li argmax: An 5hd tensor of type uint16 or int64.
* Must set the format, supported format list ["NC1HWC0"] \n

* @par Attributes:
* @li ksize: A required list of int8, int16, int32, or int64 values,
* specifying the size of the window for each dimension of the input tensor. No default value.
* @li strides: A required list of int8, int16, int32, or int64 values,
* specifying the stride of the sliding window for each dimension of the input tensor. No default value.
* @li pads: A required list of int8, int16, int32, or int64 values,
* specifying the pad of the input feature map. No default value. \n
* @li dtype: A optional int. default value is 3.
* @li dilation: A optional list of int8, int16, int32, or int64 values.
* @li ceil_mode: A optional bool. default value is false. \n

* @par Outputs:
* y: A Tensor. Has the same type and format as input "x". \n

* @attention Constraints:
* @li ksize: a list that has length 4:
* ksize[0] = 1, ksize[1] = 1, ksize[2] * ksize[3] <= (ub_size-8)*1024//7//2//16.
* @li strides: a list that has length 4:
* strides[0] = 1, strides[1] = 1, 1 <= strides[2] <= 2048, 1 <= strides[3] <= 2048.
* @li pads: a list that has length 4:
* pads[0] = 1, pads[1] = 1, 1 <= pads[2] <= (ksize[2]//2), 1 <= pads[3] <= (ksize[3]//2).
* @li dilation: a list that has length 4.
* @li ceil_mode: is a bool, default is false. \n

* @see max_pool_grad_with_argmaxv2
* @par Third-party framework compatibility
* Compatible with the PyTorch backward operator of max_pool2d_with_indices.
*/

REG_OP(MaxPoolGradWithArgmaxV2)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(grad, TensorType({DT_FLOAT16}))
    .INPUT(argmax, TensorType({DT_UINT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dtype, Int, 3)
    .ATTR(dilation, ListInt, {1,1,1,1})
    .ATTR(ceil_mode, Bool, false)
    .OP_END_FACTORY_REG(MaxPoolGradWithArgmaxV2)

/**
* @brief Performs max pooling on the input . \n

* @par Inputs:
* One input:
* x: A Tensor. Supported type:float16, float32, double, int32, int64,
* uint8, int16, int8, uint16, qint8

* @par Attributes:
* @li ksize: A required list of int8, int16, int32, or int64 values,
* specifying the size of the window for each dimension of the input tensor.
* No default value.
* @li strides: A required list of int8, int16, int32, or int64 values,
* specifying the stride of the sliding window for each dimension of
* the input tensor. No default value.
* @li padding_mode: A required string. Defaults to "CALCULATED".
* @li pads:A required list of int8, int16, int32, or int64 values,
* a data to calculate when padding_mode is "CALCULATED".
* @li data_format: An optional string. Defaults to "NHWC" .
* @li global_pooling bool, Whether to use the global pooling.
* If global_pooling = true, kernel size and paddings will be ignored.
* Default False
* @li ceil_mode: Whether to use the ceil function to calculate output
* height and width. False is the default. If it is set to False,
* the floor function will be used. Default False \n

* @par Outputs:
* y: A Tensor. Has the same type and format as input "x" . \n

* @attention Constraints:
* @li "ksize" is a list that has length 4: ksize[0] = 1 or ksize[3] = 1,
* ksize[1] * ksize[2] <= 255.
* @li "stride is a list that has length 4: strides[0] = 1 or strides[3] = 1,
* strides[1] <= 63, strides[0] >= 1, strides[2] <= 63, strides[2] >= 1.
* @li "padding" is  "SAME" "VALID" or "CALCULATE" .


* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPool.
*/
REG_OP(MaxPoolV3)
    .INPUT(x,TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE, DT_INT32, DT_INT64, DT_UINT8, DT_INT16, DT_INT8, DT_UINT16, DT_QINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE, DT_INT32, DT_INT64, DT_UINT8, DT_INT16, DT_INT8, DT_UINT16, DT_QINT8}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(padding_mode, String, "CALCULATED")
    .ATTR(pads, ListInt, {0,0,0,0})
    .ATTR(data_format, String, "NCHW")
    .ATTR(global_pooling,Bool,false)
    .ATTR(ceil_mode, Bool, false)
    .OP_END_FACTORY_REG(MaxPoolV3)

/**
* @brief Computes gradients of the maxpooling function . \n

* @par Inputs:
* @li orig_input: A mutable tensor of type RealNumberType.
* @li orig_output: A mutable tensor of type RealNumberTypex.
* @li grad: A mutable tensor of type RealNumberType . \n

* @par Attributes:
* @li ksize: A required list of int8, int16, int32, or int64 values,
* specifying the size of the window for each dimension of the input tensor.
* No default value.
* @li strides: A required list of int8, int16, int32, or int64 values,
* specifying the stride of the sliding window for each dimension of
* the input tensor. No default value.
* @li padding_mode: A required string. Defaults to "CALCULATED".
* @li pads:A required list of int8, int16, int32, or int64 values,
* a data to caculate when padding_mode is "CALCULATED".
* @li data_format: An optional string. Defaults to "NHWC" .
* @li global_pooling bool, Whether to use the global pooling.
* If global_pooling = true, kernel size and paddings will be ignored.
* Default False
* @li ceil_mode: Whether to use the ceil function to calculate output
* height and width. False is the default. If it is set to False,
* the floor function will be used. Default False \n

* @par Outputs:
* out_grad: A mutable tensor. Has the same shape and type as "x1" . \n

* @attention Constraints:
* @li Computing gradients of global pooling is not supported, which means
* "ksize < x1".
* @li "ksize" is in the range [1, 255]. "strides" is in the range [1, 63]

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPoolGrad.
*/
REG_OP(MaxPoolV3Grad)
    .INPUT(orig_input, TensorType::RealNumberType())
    .INPUT(orig_output, TensorType::RealNumberType())
    .INPUT(grad, TensorType::RealNumberType())
    .OUTPUT(out_grad, TensorType::RealNumberType())
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(padding_mode, String, "CALCULATED")
    .ATTR(pads, ListInt, {0, 0, 0, 0})
    .ATTR(data_format, String, "NCHW")
    .ATTR(global_pooling, Bool, false)
    .ATTR(ceil_mode, Bool, false)
    .OP_END_FACTORY_REG(MaxPoolV3Grad)

/**
*@brief Performs Dilation2D on the input . \n

*@par Inputs:
* @li x: A tensor of shape is 4d, format is support NHWC.
*@li filter: A tensor of shape is 3d, the type is same with x, and the c dimension is same with x. \n

*@par Attributes:
*@li strides: A required list of 4 ints, specifying the stride of the sliding window. The strides of the N and C dimensions are 1.
*@li rates: A required list of 4 ints. The rates of the N and C dimensions are 1.
*@li padding_mode: A optional string. Defaults to "SAME", it support SAME and VALID.
*@li pads: An optional list of 4 ints.
*@li ceil_mode: An optional bool. Defaults to "false". Use ceil or floor to calculate the output size when padding_mode is "CALCULATED".
*@li data_format: An optional string, specifying the data format of "rates" and "strides", either "NCHW" or "NHWC" (default). \n

*@par Outputs:
*y: The output tensor. Has the same type and format as input "x" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Dilation2D.
*/
REG_OP(Dilation2D)
    .INPUT(x,TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_UINT8, DT_INT16, DT_INT8, DT_UINT16}))
    .INPUT(filter,TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_UINT8, DT_INT16, DT_INT8, DT_UINT16}))
    .OUTPUT(y,TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_UINT8, DT_INT16, DT_INT8, DT_UINT16}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(rates, ListInt)
    .ATTR(padding_mode, String, "SAME")
    .ATTR(pads, ListInt, {0,0,0,0})
    .ATTR(ceil_mode, Bool, false)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(Dilation2D)

/**
*@brief Performs Dilation2DBackpropFilter on the input. \n

*@par Inputs:
*@li x: A tensor of shape is 4d, format is support NHWC.
* @li filter: A tensor of shape is 3d, the type is same with x, and the c dimension is same with x.
*@li out_backprop: Has the same type and format as input x and the c dimension is same with x. \n

*@par Attributes
*@li strides: A required list of 4 ints, specifying the stride of the sliding window. The strides of the N and C dimension are 1.
* @li rates: A required list of 4 ints, the rates of the N and C dimensions are 1.
*@li padding_mode: A optional string. Defaults to "SAME", it support SAME and VALID.
*@li pads: A optional list of 4 ints.
*@li ceil_mode: An optional bool. Defaults to "false". Use ceil or floor to calculate the output size when padding_mode is "CALCULATED".
*@li data_format: An optional string, specifying the data format of "rates" and "strides", either "NCHW" or "NHWC" (default). \n

*@par Outputs:
*y: The output tensor. Has the same type and format as input "filter" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Dilation2DBackpropFilter.
*/

REG_OP(Dilation2DBackpropFilter)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_UINT8, DT_INT16, DT_INT8, DT_UINT16}))
    .INPUT(filter,
           TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_UINT8, DT_INT16, DT_INT8, DT_UINT16}))
    .INPUT(out_backprop,
           TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_UINT8, DT_INT16, DT_INT8, DT_UINT16}))
    .OUTPUT(y,
            TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_UINT8, DT_INT16, DT_INT8, DT_UINT16}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(rates, ListInt)
    .ATTR(padding_mode, String, "SAME")
    .ATTR(pads, ListInt, {0, 0, 0, 0})
    .ATTR(ceil_mode, Bool, false)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(Dilation2DBackpropFilter)

/**
*@brief Performs Dilation2DBackpropInput on the input. \n

*@par Inputs:
*@li x: A tensor of shape is 4d, format is support NHWC.
* @li filter: A tensor of shape is 3d, the type is same with x, and the c dimension is same with x.
*@li out_backprop: Has the same type and format as input x and the c dimension is same with x. \n

*@par Attributes
*@li strides: A required list of 4 ints, specifying the stride of the sliding window. The strides of the N and C dimension are 1.
*@li rates: A required list of 4 ints, the rates of the N and C dimensions are 1.
*@li padding_mode: A optional string. Defaults to "SAME", it support SAME and VALID.
*@li pads: A optional list of 4 ints.
*@li ceil_mode: An optional bool. Defaults to "false". Use ceil or floor to calculate the output size when padding_mode is "CALCULATED".
*@li data_format: An optional string, specifying the data format of "rates" and "strides", either "NCHW" or "NHWC" (default). \n

*@par Outputs:
*y: The output tensor. Has the same type and format as input "x" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Dilation2DBackpropInput.
*/

REG_OP(Dilation2DBackpropInput)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_UINT8, DT_INT16, DT_INT8, DT_UINT16}))
    .INPUT(filter,
           TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_UINT8, DT_INT16, DT_INT8, DT_UINT16}))
    .INPUT(out_backprop,
           TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_UINT8, DT_INT16, DT_INT8, DT_UINT16}))
    .OUTPUT(y,
            TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_UINT8, DT_INT16, DT_INT8, DT_UINT16}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(rates, ListInt)
    .ATTR(padding_mode, String, "SAME")
    .ATTR(pads, ListInt, {0, 0, 0, 0})
    .ATTR(ceil_mode, Bool, false)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(Dilation2DBackpropInput)

/**
* @brief Applies a 1D/2D/3D adaptive average pooling over
*       an input signal composed of several input planes.  \n

* @par Inputs:
* Two input, including:
* @li x: A Tensor. Must be one of the following data types:
*     float16, float32. \n
* @li output_size: A required tensor of shape must be 1 or 2 or 3 ,
*  specifying the size  of the output tensor. \n

* @par Outputs:
* @li y: A Tensor. Has the same data type as "x" \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator AdaptiveAvgPool.
*/
REG_OP(AdaptiveAvgPool)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(output_size, TensorType({DT_INT64, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(AdaptiveAvgPool)

/**
* @brief Applies a 2D adaptive average pooling over
*       an input signal composed of several input planes.  \n

* @par Inputs:
* One input, including:
* @li x: A Tensor. Must be one of the following data types:
*     float16, float32. \n

* @par Attributes:
* @li output_size: A required list of 2 ints
*    specifying the size (H,W) of the output tensor. \n

* @par Outputs:
* @li y: A Tensor. Has the same data type as "x" \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator AdaptiveAvgPool2d.
*/
REG_OP(AdaptiveAvgPool2d)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(output_size, ListInt)
    .OP_END_FACTORY_REG(AdaptiveAvgPool2d)

/**
* @brief Compute gradients of adaptive averagev2 pooling function.

* @par Inputs:
* @li input_grad: A Tensor. Must be one of the following data types:
* float16, float32.

* @par Attributes:
* @li orig_input_shape: A required tuple or list of type int32.

* @par Outputs:
* @li output_grad: A tensor with the same type as "input_grad".

* @par Third-party framework compatibility
* Compatible with the Pytorch operator AdaptiveAvgPool2dGrad.
*/
REG_OP(AdaptiveAvgPool2dGrad)
    .INPUT(input_grad, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(output_grad, TensorType({DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(orig_input_shape, ListInt)
    .OP_END_FACTORY_REG(AdaptiveAvgPool2dGrad)

/**
* @brief Performs the backpropagation of MaxPoolWithGradArgmaxV1.

* @par Inputs:
* Three inputs, including:
* @li x: A tensor of type float16,float32.
* @li grad: A tensor of type float16,float32.
* @li argmax: A tensor of type uint16,int32. \n

* @par Attributes:
* @li ksize: A required list of int8, int16, int32, or int64 values,
* specifying the size of the window for each dimension of the input tensor. No default value.
* @li strides: A required list of int8, int16, int32, or int64 values,
* specifying the stride of the sliding window for each dimension of the input tensor. No default value.
* @li pads: A required list of int8, int16, int32, or int64 values,
* specifying the pad of the input feature map. No default value. \n

* @par Outputs:
* y: A Tensor. Has the same type and format as input "x". \n

* @attention Constraints:
* @li The MaxPoolGradWithArgmaxV2 operator has the same function, and it is recommended to use the V2 operator.
* @li ksize: a list that has length 4:
* ksize[0] = 1, ksize[3] = 1, ksize[1] * ksize[2] <= (ub_size-8)*1024//7//2//16.
* @li strides: a list that has length 4:
* strides[0] = 1, strides[3] = 1, 1 <= strides[1] <= 2048, 1 <= strides[2] <= 2048.
* @li pads: a list that has length 4:
* pads[0] = 1, pads[3] = 1, 1 <= pads[2] <= (ksize[1]//2), 1 <= pads[2] <= (ksize[3]//2).
* @li ceil_mode: defaults to False.\n

* @par Third-party framework compatibility
* Compatible with the Pytorch backward operator of max_pool2d_with_indices.
*/

REG_OP(MaxPoolGradWithArgmaxV1)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(argmax, TensorType({DT_UINT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dtype, Int, 3)
    .ATTR(dilation, ListInt, {1, 1, 1, 1})
    .ATTR(ceil_mode, Bool, false)
    .OP_END_FACTORY_REG(MaxPoolGradWithArgmaxV1)

/**
* @brief Performs max pooling on the input and outputs both max values and indices.

* @par Inputs:
* One input:
* x: A Tensor of type float16, float32. \n

* @par Attributes:
* @li ksize: A required list of int8, int16, int32, or int64 values,
* specifying the size of the window for each dimension of the input tensor. No default value.
* @li strides: A required list of int8, int16, int32, or int64 values,
* specifying the stride of the sliding window for each dimension of the input tensor. No default value.
* @li pads: A required list of int8, int16, int32, or int64 values,
* specifying the pad of the input feature map. No default value. \n

* @par Outputs:
* y: A Tensor. Has the same type and format as input "x".
* argmax:  A Tensor. type:uint16, int32. \n

* @attention Constraints:
* @li The MaxPoolWithArgmaxV2 operator has the same function, and it is recommended to use the V2 operator.
* @li ksize: a list that has length 4:
* ksize[0] = 1, ksize[3] = 1, ksize[1] * ksize[2] <= (ub_size-8)*1024//6//2//16.
* @li strides: a list that has length 4:
* strides[0] = 1, strides[3] = 1, 1 <= strides[1] <= 2048, 1 <= strides[2] <= 2048.
* @li pads: a list that has length 4:
* pads[0] = 1, pads[3] = 1, 1 <= pads[1] <= (ksize[1]//2), 1 <= pads[2] <= (ksize[2]//2).
* @li ceil_mode: defaults to False.

* @par Third-party framework compatibility
* Compatible with the PyTorch operator max_pool2d_with_indices.
*/
REG_OP(MaxPoolWithArgmaxV1)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OUTPUT(argmax, TensorType({DT_UINT16, DT_INT32}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dtype, Int, 3)
    .ATTR(dilation, ListInt, {1, 1, 1, 1})
    .ATTR(ceil_mode, Bool, false)
    .OP_END_FACTORY_REG(MaxPoolWithArgmaxV1)

/**
* @brief Randomly sample a subset of positive and negative examples,and overwrite
the label vector to the ignore value (-1) for all elements that are not
included in the sample.\n

* @par Inputs:
* One input:
* labels: shape of labels,(N, ) label vector with values. \n

* @par Attributes:
* @li batch_size_per_images: A require attribute of type int.
* @li positive_fraction: A require attribute of type float.

* @par Outputs:
* y: The result of subSample. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator SubSample.

* @attention Constraints:
* Warning: This operator can be integrated only by MaskRcnn. Please do not use it directly.
*/
REG_OP(SubSample)
    .INPUT(labels, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .REQUIRED_ATTR(batch_size_per_images, Int)
    .REQUIRED_ATTR(positive_fraction, Float)
    .OP_END_FACTORY_REG(SubSample)

/**
* @brief Randomly sample a subset of positive and negative examples,and overwrite
the label vector to the ignore value (-1) for all elements that are not
included in the sample.\n

* @par Inputs:
* two inputs, including:
* @li labels: shape of labels,(N, ) label vector with values:.
* @li shuffle_matrix: random matrix with shape (N, ). \n

* @par Attributes:
* @li batch_size_per_images: A require attribute of type int.
* @li positive_fraction: A require attribute of type float.

* @par Outputs:
* y: The result of subSample. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator SubSampleLabels.

* @attention Constraints:
* Warning: This operator can be integrated only by MaskRcnn. Please do not use it directly.
*/
REG_OP(SubSampleLabels)
    .INPUT(labels, TensorType({DT_INT32}))
    .INPUT(shuffle_matrix, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .REQUIRED_ATTR(batch_size_per_images, Int)
    .REQUIRED_ATTR(positive_fraction, Float)
    .OP_END_FACTORY_REG(SubSampleLabels)

/**
* @brief Computes GlobalLpPool, GlobalLpPool consumes an input tensor X and applies lp pool pooling across the
values in the same channel. \n

* @par Inputs:
* x: A Tensor of type float16 or float32 . \n

* @par Attributes:
* @li p: Optional. Must be one of the following types: float32. Defaults to 2.0. \n

* @par Outputs:
* y: A Tensor. Has the same type as "x", when shape of x is [N,C,H,W], shape of y is [N,C,1,1].
* @par Third-party framework compatibility
* Compatible with the onnx operator GlobalLpPool.
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED.
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/

REG_OP(GlobalLpPool)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(p, Float, 2.0)
    .OP_END_FACTORY_REG(GlobalLpPool)

/**
*@brief GlobalAveragePool consumes an input tensor X and applies average pooling across the values in the same channel.
This is equivalent to AveragePool with kernel size equal to the spatial dimension of input tensor \n

*@par Inputs:
*@li x: Input data tensor from the previous operator; dimensions for image case are (N x C x H x W),
where N is the batch size, C is the number of channels, and H and W are the height and the width of the data.
For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.

*@par Outputs:
*y: Output data tensor from pooling across the input tensor. The output tensor has the same rank as the input.
The first two dimensions of output shape are the same as the input (N x C), while the other dimensions are all 1

*@par Restrictions:
*Warning: This operator can be integrated only by configuring INSERT_OP_FILE of aclgrphBuildModel. Please do not use it directly.
*/
REG_OP(GlobalAveragePool)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(GlobalAveragePool);

}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_POOLING_OPS_H
