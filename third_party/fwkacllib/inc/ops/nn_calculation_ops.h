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
 * \file nn_calculation_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_CALCULATION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_CALCULATION_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Computes the gradients of depthwise convolution with respect to
* the filter . \n

* @par Inputs:
* Three inputs include: \n
* @li input: 4D origin shape of input tensor [N, C, H, W] or [N, H, W, C],
* support float16, float32, double
* @li filter_size: A 4D tensor of type int32, with shape [H, W, C, K]
* @li out_backprop: 4D tensor with shape [N, C, H, W] or [N, H, W, C].
* Must be one of the following types: float16, float32, double . \n

* @par Attributes:
* @li strides: A required list or tuple. The stride of the sliding window
* for height and width of input "x" of the convolution.
* Must be with shape [1, 1, stride_height, stride_width] or
* [1, stride_height, stride_width, 1].
* @li dilations: An optional list or tuple. The dilation factor for each
* dimension of input "x".
* If set to k > 1, there will be k-1 skipped cells between each filter element
* on that dimension. Must be with shape [1, 1, dilation_height, dilation_width]
* or [1, dilation_height, dilation_width, 1].
* @li pads: A required list or tuple. Padding added to each dimension of the
* input.
* @li data_format: An optional string. Input data format, either "NHWC" or
* "NCHW" . \n

* @par Outputs:
* filter_grad: Gradient of the deep convolution relative to the filter with
* shape [H, W, C, K]. Must be one of the following types: float16, float32,
* double . \n

* @attention Constraints:\n
* The feature map is 4D with shape [N, C, Hi, Wi] or [N, Hi, Wi, C], but
* the data is 5D with shape [N, C1, Hi, Wi, C0], where C0 is 16.\n
* The filter is 4D with shape [Hf, Wf, C, K], but the data is 6D with shape
* [C1, Hf, Wf, K, Co, C0],
* where K is fixed at 1, and Co and C0 are 16.\n
* Output backprop is 4D with shape [N, C, Ho, Wo] or [N, Ho, Wo, C], but the
* data is 5D with shape [N, C1, Ho, Wo, C0],
* where C is the same as that of the feature map and C0 is 16.\n
* Limited by Tiling and L1 / L0 buffer memory: 512 * ceil(Wo, 16) + (480 *
* stride_h + 32 * filter_h) * ceil(Wi, 16) <= l1_size and Hf*Wf <= l0b_size/512 . \n

* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator DepthwiseConv2DBackpropFilter.
* @li Compatible with the Caffe operator DepthwiseConv2DBackpropFilter.
*/
REG_OP(DepthwiseConv2DBackpropFilter)
    .INPUT(input, TensorType({float16}))
    .INPUT(filter_size, TensorType({DT_INT32, DT_INT64}))
    .INPUT(out_backprop, TensorType({float16}))
    .OUTPUT(filter_grad, TensorType({float32}))
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(DepthwiseConv2DBackpropFilter)

/**
* @brief Computes the gradients of depthwise convolution with respect to
* the filter . \n

* @par Inputs:
* Two inputs include: \n
* @li input: 4D tensor with shape [N, C, H, W] or [N, H, W, C], of type float16
* @li out_backprop: 4D tensor with shape [N, C, H, W] or [N, H, W, C],
* of type float16

* @par Attributes:
* @li filter_size: A required list or tuple. Shape of filter.
* @li strides: A required list or tuple. The stride of the sliding window for
* height and width of input "x" of the convolution.
* Must be with shape [1, 1, stride_height, stride_width] or [1, stride_height,
* stride_width, 1].
* @li dilations: An optional list or tuple. The dilation factor for each
* dimension of input "x".
* If set to k > 1, there will be k-1 skipped cells between each filter element
* on that dimension. Must be with shape [1, 1, dilation_height, dilation_width]
* or [1, dilation_height, dilation_width, 1].
* @li pads: A required list or tuple. Padding added to each dimension of the
* input.
* @li data_format: An optional string. Input data format, either "NHWC" or
* "NCHW" . \n

* @par Outputs:
* filter_grad: Gradient of the deep convolution relative to the filter with
* shape [H, W, C, K]. Must be of type float32 . \n

* @attention Constraints:\n
* The feature map is 4D with shape [N, C, Hi, Wi] or [N, Hi, Wi, C], but
* the data is 5D with shape [N, C1, Hi, Wi, C0], where C0 is 16.\n
* The filter is 4D with shape [Hf, Wf, C, K], but the data is 6D with shape
* [C1, Hf, Wf, K, Co, C0],
* where K is fixed at 1, and Co and C0 are 16.\n
* Output backprop is 4D with shape [N, C, Ho, Wo] or [N, Ho, Wo, C], but the
* data is 5D with shape [N, C1, Ho, Wo, C0],
* where C is the same as that of the feature map and C0 is 16.\n
* Limited by Tiling and L1 / L0 buffer memory: 512 * ceil(Wo, 16) + (480 *
* stride_h + 32 * filter_h) * ceil(Wi, 16) <= l1_size and Hf*Wf <= l0b_size/512 . \n

* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator DepthwiseConv2DBackpropFilter.
* @li Compatible with the Caffe operator DepthwiseConv2DBackpropFilter.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use DepthwiseConv2DBackpropFilter
* instead.
*/
REG_OP(DepthwiseConv2DBackpropFilterD)
    .INPUT(input, TensorType({float16}))
    .INPUT(out_backprop, TensorType({float16}))
    .OUTPUT(filter_grad, TensorType({float32}))
    .REQUIRED_ATTR(filter_size, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(DepthwiseConv2DBackpropFilterD)

/**
* @brief Computes the gradients of depthwise convolution with respect to the
* input . \n

* @par Inputs:
* Three inputs include: \n
* @li input_size: 4D shape of input tensor [N, C, H, W] or [N, H, W, C],
* support int32, int64
* @li filter: 4D filter tensor with shape of [H, W, C, K], support float16.
* @li out_backprop: 4D tensor with shape [N, C, H, W] or [N, H, W, C].
* Must be one of the following types: float16 . \n

* @par Attributes:
* @li strides: A required list or tuple of int32. The stride of the sliding window for
* height and width of input "x" of the convolution.
* Must be with shape [1, 1, stride_height, stride_width] or [1, stride_height,
* stride_width, 1].
* @li dilations: An optional list or tuple of int32. The dilation factor for each
* dimension of input "x". Defaults to "[1, 1, 1, 1]".
* If set to k > 1, there will be k-1 skipped cells between each filter element
* on that dimension. Must be with shape [1, 1, dilation_height, dilation_width]
* or [1, dilation_height, dilation_width, 1].
* @li pads: A required list or tuple of int32. Padding added to each dimension of the
* input.
* @li data_format: An optional string. Input data format, either "NHWC" or
* "NCHW". Defaults to "NHWC" . \n

* @par Outputs:
* input_grad: Gradient of the deep convolution relative to the input with shape
* [N, C, H, W] or [N, H, W, C] Must be one of the following types: float16 . \n

* @attention Constraints:\n
* The feature map is 4D with shape [N, C, Hi, Wi] or [N, Hi, Wi, C], but
* the data is 5D with shape [N, C1, Hi, Wi, C0], where C0 is 16.\n
* The filter is 4D with shape [Hf, Wf, C, K], but the data is 6D with shape
* [C1, Hf, Wf, K, Co, C0],
* where K is fixed at 1, and Co and C0 are 16.\n
* Output backprop is 4D with shape [N, C, Ho, Wo] or [N, Ho, Wo, C], but the
* data is 5D with shape [N, C1, Ho, Wo, C0],
* where C is the same as that of the feature map and C0 is 16.\n
* Limited by Tiling: max_h_in_l1 >= C0, where max_h_in_l1 = (l1_size - Hf *
* Wf * C0 * C0 * 2) / (2 * Wo *C0).\n

* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator DepthwiseConv2DBackpropInput.
* @li Compatible with the Caffe operator DepthwiseConv2DBackpropInput.
*/
REG_OP(DepthwiseConv2DBackpropInput)
    .INPUT(input_size, TensorType({DT_INT32, DT_INT64}))
    .INPUT(filter, TensorType({DT_FLOAT16}))
    .INPUT(out_backprop, TensorType({DT_FLOAT16}))
    .OUTPUT(input_grad, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(DepthwiseConv2DBackpropInput)

/**
* @brief Computes the gradients of depthwise convolution with respect to the
* input . \n

* @par Inputs:
* Two inputs include: \n
* @li filter: A 4D tensor of type float16, with shape [H, W, C, K]
* @li out_backprop: 4D tensor with shape [N, C, H, W] or [N, H, W, C], of
* type float16

* @par Attributes:
* @li input_size: A required list or tuple. The origin shape of input.
* @li strides: A required list or tuple. The stride of the sliding window for
* height and width of input "x" of the convolution.
* Must be with shape [1, 1, stride_height, stride_width] or [1, stride_height,
* stride_width, 1].
* @li dilations: An optional list or tuple. The dilation factor for each
* dimension of input "x".
* If set to k > 1, there will be k-1 skipped cells between each filter element
* on that dimension. Must be with shape [1, 1, dilation_height, dilation_width]
* or [1, dilation_height, dilation_width, 1].
* @li pads: A required list or tuple. Padding added to each dimension of the
* input.
* @li data_format: An optional string. Input data format, either "NHWC" or
* "NCHW" . \n

* @par Outputs:
* input_grad: Gradient of the deep convolution relative to the input with
* shape [N, C, H, W] or [N, H, W, C]. Must be of type float16 . \n

* @attention Constraints:\n
* The feature map is 4D with shape [N, C, Hi, Wi] or [N, Hi, Wi, C], but
* the data is 5D with shape [N, C1, Hi, Wi, C0], where C0 is 16.\n
* The filter is 4D with shape [Hf, Wf, C, K], but the data is 6D with shape
* [C1, Hf, Wf, K, Co, C0],
* where K is fixed at 1, and Co and C0 are 16.\n
* Output backprop is 4D with shape [N, C, Ho, Wo] or [N, Ho, Wo, C], but the
* data is 5D with shape [N, C1, Ho, Wo, C0],
* where C is the same as that of the feature map and C0 is 16.\n
* Limited by Tiling: max_h_in_l1 >= C0, where max_h_in_l1 = (l1_size - Hf *
* Wf * C0 * C0 * 2) / (2 * Wo *C0).\n

* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator DepthwiseConv2DBackpropInput.
* @li Compatible with the Caffe operator DepthwiseConv2DBackpropInput.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use DepthwiseConv2DBackpropInput
* instead.
*/
REG_OP(DepthwiseConv2DBackpropInputD)
    .INPUT(filter, TensorType({DT_FLOAT16}))
    .INPUT(out_backprop, TensorType({DT_FLOAT16}))
    .OUTPUT(input_grad, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(input_size, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(DepthwiseConv2DBackpropInputD)

/**
*@brief Computes a 2D deep convolution given a 4D input tensor and a filter
* tensor . \n

*@par Inputs:
*Two required inputs and two optional inputs, including: \n
* @li x: A 4D tensor of type float16 or int8, with shape [N, C, H, W] or [N, H, W, C]
* @li filter: A 4D tensor of type float16 or int8, with shape [H, W, C, K]
* @li bias: An optional tensor of type float16 or int32
* @li offset_w: An optional float16 or int8, used for quantized inference

* @par Attributes:
* @li strides: A required list or tuple. The stride of the sliding window for
* height and width of input "x" of the convolution.
* Must be with shape [1, 1, stride_height, stride_width] or [1, stride_height,
* stride_width, 1].
* @li dilations: An optional list or tuple. The dilation factor for each
* dimension of input "x".
* If set to k > 1, there will be k-1 skipped cells between each filter element
* on that dimension. Must be with shape [1, 1, dilation_height, dilation_width]
* or [1, dilation_height, dilation_width, 1]. Defaults to "[1, 1, 1, 1]".
* @li pads: A required list or tuple of int32. Padding added to each dimension of the
* input.
* @li data_format: An optional string. Input data format, either "NHWC" or
* "NCHW". Defaults to "NHWC".
* @li offset_x: An optional int. Input offset, used for quantized inference.
* Defaults to 0 . \n

* @par Outputs:
* y: 4D tensor of type float16 or int32, with shape [N, C, H, W] or [N, H, W, C]

* @attention Constraints:\n
* The feature map is 4D with shape [N, C, Hi, Wi] or [N, Hi, Wi, C], but
* the data is 5D with shape [N, C1, Hi, Wi, C0], where C0 is 16.\n
* The filter is 4D with shape [Hf, Wf, C, K], but the data is 6D with shape
* [C1, Hf, Wf, K, Co, C0],
* where K is fixed at 1, and Co and C0 are 16.\n
* Limited by the size of L1 buffer memory: \n
* (l1_size - filter_h*filter_w*BLOCK_SIZE*BLOCK_SIZE*data_size) // (Wi *
* BLOCK_SIZE * data_size) >= (BLOCK_SIZE * strides_h + filter_h - strides_h).\n

* @par Quantization supported or not
* Yes

* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator DepthwiseConv2D.
* @li Compatible with the Caffe operator DepthwiseConv2D.
*/
REG_OP(DepthwiseConv2D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_INT8}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_INT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_FLOAT16, DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT32}))
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(data_format, String, "NHWC")
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(DepthwiseConv2D)

/**
*@brief Performs the the backward operation for "BiasAdd" on the "bias" tensor.
*        It accumulates all the values from out_backprop into the feature
*        dimension. For NHWC data format, the feature dimension is the last.
*        For NCHW data format, the feature dimension is the third-to-last . \n

*@par Inputs:
*x: A Tensor of type NumberType . \n

*@par Attributes:
*data_format: Data format. Defaults to "NHWC" . \n

*@par Outputs:
*y: A Tensor.Has the same type as "x" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator BiasAddGrad.
*/
REG_OP(BiasAddGrad)
    .INPUT(x, TensorType::NumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(BiasAddGrad)

/**
*@brief Computes the gradients of convolution with respect to the input.
*@par Inputs:
 * Three inputs:
 * @li input_size: A const Tensor of type int32. Currently does not support
 * data tensor. An integer vector representing the shape of input, where
 * input is a 4-D tensor [batch, height, width, channels]
 * or [batch, channels, height, width].
 * @li filter: A Tensor. Must be one of the following types: float16, float32,
 * float64. 4-D with shape
 * [filter_height, filter_width, in_channels, out_channels]
 * or [out_channels, filter_height, filter_width, in_channels]
 * or [out_channels, in_channel, filter_height, filter_width].
 * @li out_backprop: A Tensor. Must have the same type as filter.
 * 4-D with shape [batch, out_height, out_width, out_channels]
 * or [batch, out_channels, out_height, out_width].
 * Gradients with respect to the output of the convolution.
 *\n
 *\n
 * The following are the supported data types and data formats:
*@verbatim
    | Tensor    | out_bckprop | filter  | y
    ------------|-------------|---------|--------
    | Data Type | float16     | float16 | float16
    |           |-------------|---------|--------
    |           | float32     | float32 | float32
    |           |-------------|---------|--------
    |           | float64     | float64 | float64
    ------------|-------------|---------|--------
    | Format    | NCHW        | NCHW    | NCHW
    |           | NHWC        | HWCN    | NHWC
@endverbatim
 * For float32 and float64 type, the actual calculation on the chip is based on
 * float16.
 *\n
 *
*@par Attributes:
 * Five attributes:
 * @li strides: A tuple/list of 4 integers. The stride of the sliding window
 * for H/W dimension. The index of H/W is same as data_format.
 * @li pads: A tuple/list of 4 integers, [top, bottom, left, right] pads
 * on feature map
 * @li dilations: A tuple/list of 4 integers, The dilation factor for each
 * dimension of input, defaults to [1,1,1,1].
 * @li groups: Number of blocked connections from input channels to output
 * channels.
 * @li data_format: An optional string from: "NHWC", "NCHW". Defaults to
 * "NHWC". Specify the data format of the input and output data.
 *\n
 *\n
 * The following value range restrictions must be met:
*@verbatim
    | Name             | Field    | Scope
    -------------------|----------|--------------
    | input_size       | H        | [1, 4096]
    |                  | W        | [1, 4096]
    -------------------|----------|--------------
    | Filter           | H        | [1, 255]
    |                  | W        | [1, 255]
    -------------------|----------|--------------
    | out_backprop     | H        | [1, 4096]
    |                  | W        | [1, 4096]
    -------------------|----------|--------------
    | y(fmap)          | H        | [1, 4096]
    |                  | W        | [1, 4096]
    -------------------|----------|--------------
    | Stride           | H        | [1, 63]
    |                  | W        | [1, 63]
    -------------------|----------|--------------
    | Padding          | Top      | [0, 255]
    |                  | Bottom   | [0, 255]
    |                  | Left     | [0, 255]
    |                  | Right    | [0, 255]
    -------------------|----------|--------------
    | Dilation         | H        | [1, 255]
    |                  | W        | [1, 255]

@endverbatim
 * In Ascend910, fmap or out_backprop's H and W not support 1 when
 * fmap_h + pad_top + pad_bottom != (filter_height - 1) * dilation_h + 1
 *\n
 *
*@par Outputs:
 * y: A Tensor. Has the same type as filter,and has same format as input_size.
 *\n
 *     out_backprop_height = (fmap_height + pad_top + pad_bottom -
 *                           (dilation_h * (filter_height - 1) + 1))
 *                           / stride_h + 1
 *\n
 *     out_backprop_width = (fmap_width + pad_left + pad_right -
 *                          (dilation_w * (filter_width - 1) + 1))
 *                          / stride_w + 1
 *\n
 *
*@par Third-party framework compatibility
 * Compatible with Tensorflow's conv2d_backprop_input
*/
REG_OP(Conv2DBackpropInput)
    .INPUT(input_size, TensorType({DT_INT32}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(out_backprop, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(Conv2DBackpropInput)

/**
*@brief Computes the gradients of convolution with respect to the input.
*@par Inputs:
 * Two inputs:
 * @li filter: A Tensor. Types is float16.
 * 4-D with shape [filter_height, filter_width, in_channels, out_channels]
 * or [out_channels, filter_height, filter_width, in_channels]
 * or [out_channels, in_channel, filter_height, filter_width].
 * @li out_backprop: A Tensor. Must have the same type as filter.
 * 4-D with shape [batch, out_height, out_width, out_channels]
 * or [batch, out_channels, out_height, out_width].
 * Gradients with respect to the output of the convolution.
*@par Attributes:
 * Six attributes:
 * @li input_size A Tensor of type int32. An integer vector representing the
 * shape of input, where input is a 4-D tensor [batch, height, width, channels]
 * or [batch, channels, height, width].
 * @li strides: A tuple/list of 4 integers. The stride of the sliding window
 * for H/W dimension. The index of H/W is same as data_format.
 * @li pads: A tuple/list of 4 integers, [top, bottom, left, right] pads on
 * feature map
 * @li dilations: A tuple/list of 4 integers, The dilation factor for each
 * dimension of input, defaults to [1,1,1,1].
 * @li groups: Number of blocked connections from input channels to output
 * channels.
 * @li data_format: An optional string from: "NHWC", "NCHW". Defaults to
 * "NHWC". Specify the data format of the input and output data.
*@par Outputs:
 * y: A Tensor. Has the same type as filter,4-D tensor [batch, height, width,
 * channels] or [batch, channels, height, width].
*@par Third-party framework compatibility
 * Compatible with Tensorflow's conv2d_backprop_input
*@par Restrictions:
 * Warning: THIS FUNCTION IS DEPRECATED. Please use Conv2DBackpropInput instead.
*/
REG_OP(Conv2DBackpropInputD)
    .INPUT(filter, TensorType({DT_FLOAT16, DT_INT8}))
    .INPUT(out_backprop, TensorType({DT_FLOAT16, DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT32}))
    .REQUIRED_ATTR(input_size, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(Conv2DBackpropInputD)

/**
*@brief Computes the Deconvolution with respect to the input.
*@par Inputs:
 * Three inputs:
 * @li x: A Tensor of type float16 or int8.  4D with shape
 * [batch, out_channels, out_height, out_width]. Gradients with respect
 * to the output of the convolution.
 * @li filter: A Tensor. Must have the same type as "x".
 * 4D with shape [out_channels, in_channel, filter_height, filter_width].\n
 * Two optional inputs:
 * @li bias: An optional tensor. Must have the same type as "y".
 * @li offset_w: An optional 1D tensor for quantized deconvolution.
 * Type is int8. Reserved.\n
 *\n
 *\n
 * The following are the supported data types and data formats:
*@verbatim
    | Tensor    | x       | filter  | bias    | y
    ------------|---------|---------|---------|--------
    | Data Type | float16 | float16 | float16 | float16
    |           |---------|---------|---------|--------
    |           | int8    | int8    | int32   | int32
    ------------|---------|---------|---------|--------
    | Format    | NCHW    | NCHW    | ND      | NCHW
@endverbatim
 * For int8, a dequant or requant operator must be followed.
 *\n
 *
*@par Attributes:
 * Six attributes:
 * @li strides: A tuple or list of 2 integers. The stride of the sliding window
 * for H/W dimension, defaults to [1,1].
 * @li pads: A tuple or list of 4 integers. The [top, bottom, left, right]
 * padding on the feature map, defaults to [0,0,0,0].
 * @li dilations: A tuple or list of 4 integers. The dilation factor for each
 * dimension of input, defaults to [1,1,1,1].
 * @li groups: Number of blocked connections from input channels to
 output channels. Defaults to "1".
 * @li data_format: An optional string from: "NCHW". Defaults to "NCHW". \n
  Specify the data format of the input and output data.
 * @li offset_x: An optional integer for quantized deconvolution.
 * Defaults to "0".
 *\n
 *\n
 * The following value range restrictions must be met:
*@verbatim
    | Name             | Field    | Scope
    -------------------|----------|--------------
    | x (out_backprop) | H        | [1, 4096]
    |                  | W        | [1, 4096]
    -------------------|----------|--------------
    | Filter           | H        | [1, 255]
    |                  | W        | [1, 255]
    -------------------|----------|--------------
    | y (fmap)         | H        | [1, 4096]
    |                  | W        | [1, 4096]
    -------------------|----------|--------------
    | Stride           | H        | [1, 63]
    |                  | W        | [1, 63]
    -------------------|----------|--------------
    | Padding          | Top      | [0, 255]
    |                  | Bottom   | [0, 255]
    |                  | Left     | [0, 255]
    |                  | Right    | [0, 255]
    -------------------|----------|--------------
    | Dilation         | H        | [1, 255]
    |                  | W        | [1, 255]
    -------------------|----------|--------------
    | Offset_x         |          | [-128, 127]

@endverbatim
 * In Ascend910, fmap or out_backprop's H and W not support 1 when
 * fmap_h + pad_top + pad_bottom != (filter_height - 1) * dilation_h + 1
 *\n
 *
*@par Outputs:
 * y: A Tensor. 4D tensor with shape [batch, channels, height, width].
 *\n
 *     out_backprop_height = (fmap_height + pad_top + pad_bottom -
 *                           (dilation_h * (filter_height - 1) + 1))
 *                           / stride_h + 1
 *\n
 *     out_backprop_width = (fmap_width + pad_left + pad_right -
 *                          (dilation_w * (filter_width - 1) + 1))
 *                          / stride_w + 1
 *\n
 *
 * When type of x is float16, the type of y must be float16.
 * When type of x is int8, the type of y must be int32.
*/
REG_OP(Deconvolution)
    .INPUT(x, TensorType({DT_FLOAT16, DT_INT8}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_INT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT32}))
    .ATTR(strides, ListInt, {1, 1})
    .ATTR(pads, ListInt, {0, 0, 0, 0})
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NCHW")
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(Deconvolution)
/**
*@brief Computes the gradients of convolution with respect to the filter
*@par Inputs:
 * Three inputs:
 * @li x: A Tensor. Must be one of the following types: float16, float32,
 * float64.4-D with shape [batch, in_height, in_width, in_channels] or
 * [batch, in_channels, in_height, in_width].
 * @li filter_size: A const Tensor of type int32. Currently does not support
 * data tensor. An integer vector representing the tensor shape of filter,
 * where filter is a 4-D tensor [filter_height, filter_width, in_channels,
 * out_channels] or [out_channels, filter_height, filter_width, in_channels]
 * or [out_channels, in_channel, filter_height, filter_width].
 * @li out_backprop: A Tensor. Must have the same type as x. 4-D with shape
 * [batch, out_height, out_width, out_channels] or [batch, out_channels,
 * out_height, out_width]. Gradients with respect to the output of the
 * convolution.
 *\n
 *\n
 * The following are the supported data types and data formats:
*@verbatim
    | Tensor    | x       | out_backprop | y
    ------------|---------|--------------|---------
    | Data Type | float16 |    float16   | float16
    |           |---------|--------------|---------
    |           | float32 |    float32   | float32
    |           |---------|--------------|---------
    |           | float64 |    float64   | float64
    |-----------|---------|--------------|---------
    | Format    | NCHW    |     NCHW     | NCHW
    |           | NHWC    |     NHWC     | HWCN
@endverbatim
 * For float32 and float64 type of x and outbackprop, the actual calculation on the chip
 * is based on float16.
 *\n
 *
*@par Attributes:
 * Five attributes:
 * @li strides: A tuple/list of 4 integers. The stride of the sliding window
 * for H/W dimension. The index of H/W is same as data_format.
 * @li pads: A tuple/list of 4 integers, [top, bottom, left, right] pads on
 * feature map.
 * @li dilations: A tuple/list of 4 integers, The dilation factor for each
 * dimension of input, defaults to [1,1,1,1].
 * @li groups: Number of blocked connections from input channels to output
 * channels.
 * @li data_format: An optional string from: "NHWC", "NCHW". Defaults to
 * "NHWC". Specify the data format of the input and output data.
 *\n
*\n
* The following value range restrictions must be met:
*@verbatim
    | Name             | Field    | Scope
    -------------------|----------|--------------
    | x(fmap)          | H        | [1, 4096]
    |                  | W        | [1, 4096]
    -------------------|----------|--------------
    | Filter Size      | H        | [1, 255]
    |                  | W        | [1, 255]
    -------------------|----------|--------------
    | out_backprop     | H        | [1, 4096]
    |                  | W        | [1, 4096]
    -------------------|----------|--------------
    | y                | H        | [1, 4096]
    |                  | W        | [1, 4096]
    -------------------|----------|--------------
    | Stride           | H        | [1, 63]
    |                  | W        | [1, 63]
    -------------------|----------|--------------
    | Padding          | Top      | [0, 255]
    |                  | Bottom   | [0, 255]
    |                  | Left     | [0, 255]
    |                  | Right    | [0, 255]
    -------------------|----------|--------------
    | Dilation         | H        | [1, 255]
    |                  | W        | [1, 255]

@endverbatim
 * In Ascend910, out_backprop's H and W not support 1 when
 * fmap_h + pad_top + pad_bottom != (filter_height - 1) * dilation_h + 1
 *\n
 *
*@par Outputs:
 * y: A Tensor. Has the same type as x, has the same format as filter_size.
 *\n
 *     out_backprop_height = (in_height + pad_top + pad_bottom -
 *                           (dilation_h * (filter_height - 1) + 1))
 *                           / stride_h + 1
 *\n
 *     out_backprop_width = (in_width + pad_left + pad_right -
 *                          (dilation_w * (filter_width - 1) + 1))
 *                          / stride_w + 1
 *\n
 *
*@par Third-party framework compatibility
 * Compatible with Tensorflow's conv2d_backprop_filter
*/
REG_OP(Conv2DBackpropFilter)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(filter_size, TensorType({DT_INT32}))
    .INPUT(out_backprop, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(Conv2DBackpropFilter)

/**
*@brief Computes the gradients of convolution with respect to the filter.
*@par Inputs:
 * Two inputs:
 * @li x: A Tensor. Type is float16.
 * 4-D with shape [batch, in_height, in_width, in_channels] or [batch,
 * in_channels, in_height, in_width].
 * @li out_backprop: A Tensor. Must have the same type as x. 4-D with shape
 * [batch, out_height, out_width, out_channels] or [batch, out_channels,
 * out_height, out_width]. Gradients with respect to the output of the
 * convolution.
*@par Attributes:
 * Six attributes:
 * @li filter_size: A Tensor of type integers. An integer vector representing
 * the tensor shape of filter,
 * where filter is a 4-D tensor [filter_height, filter_width, in_channels,
 * out_channels] or [out_channels, filter_height, filter_width, in_channels]
 * or [out_channels, in_channel, filter_height, filter_width].
 * @li strides: A tuple/list of 4 integers. The stride of the sliding window
 * for H/W dimension. The index of H/W is same as data_format.
 * @li pads: A tuple/list of 4 integers, [top, bottom, left, right] pads on
 * feature map
 * @li dilations: A tuple/list of 4 integers, The dilation factor for each
 * dimension of input, defaults to [1,1,1,1].
 * @li groups: Number of blocked connections from input channels to output
 * channels.
 * @li data_format: An optional string from: "NHWC", "NCHW". Defaults to
 * "NHWC". Specify the data format of the input and output data.
*@par Outputs:
 * y: A Tensor. Type is float32, a 4-D tensor [filter_height, filter_width,
 * in_channels, out_channels] or [out_channels, filter_height, filter_width,
 * in_channels] or [out_channels, in_channel, filter_height, filter_width].
 * Compatible with Tensorflow's conv2d_backprop_filter
*@par Restrictions:
 * Warning: THIS FUNCTION IS DEPRECATED. Please use Conv2DBackpropFilter instead.
*/
REG_OP(Conv2DBackpropFilterD)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(out_backprop, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(filter_size, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(Conv2DBackpropFilterD)

/**
*@brief Computes a 2D convolution given 4D "x" and "filter" tensors.
*@par Inputs:
*@li x: A 4D tensor of input image. With the format "NHWC", the data is stored
* in the order of: [batch, in_height, in_width, in_channels].
*@li filter: A 4D tensor of learnable filters. Must have the same type as "x".
* With the format "HWCN" , the data is stored in the order of: [filter_height,
* filter_width, in_channels / groups, out_channels].
*@li bias: An optional 1D tensor of additive biases to the filter outputs.
* The data is stored in the order of: [out_channels].
*@li offset_w: Reserved.
*\n
*\n
* The following are the supported data types and data formats:
*@verbatim
    | Tensor    | x       | filter  | bias    | y
    ------------|---------|---------|---------|--------
    | Data Type | float16 | float16 | float16 | float16
    |           |---------|---------|---------|--------
    |           | float32 | float32 | float32 | float32
    |           |---------|---------|---------|--------
    |           | int8    | int8    | int32   | int32
    ------------|---------|---------|---------|--------
    | Format    | NCHW    | NCHW    | ND      | NCHW
    |           | NHWC    | HWCN    |         | NHWC
@endverbatim
* For float32 type, the actual calculation on the chip is based on
* float16. For int8, a dequant or requant operator must be followed.
*\n
*
*@par Attributes:
*@li strides: Required. A list of 4 integers. The stride of the sliding window
* for each dimension of input. The dimension order is determined by the data
* format of "x". The N and C dimensions must be set to 1.
*@li pads: Required. A list of 4 integers. The number of pixels to add to each
* (top, bottom, left, right) side of the input.
*@li dilations: Optional. A list of 4 integers. The dilation factor for each
* dimension of input. The dimension order is determined by the data format of
* "x". The N and C dimensions must be set to 1. Defaults to [1, 1, 1, 1].
*@li groups: Optional. An integer of type int32. The number of blocked
* connections from input channels to output channels. In_channels and
* out_channels must both be divisible by "groups". Defaults to 1.
*@li offset_x: Optional. An integer of type int32. The negative offset added
* to the input image for int8 type. Ensure that the output is within the
* effective range. Defaults to 0.
*@li data_format: Reserved.
*\n
*\n
* The following value range restrictions must be met:
*@verbatim
    | Name             | Field    | Scope
    -------------------|----------|--------------
    | Input Image Size | H        | [1, 100000]
    |                  | W        | [1, 4096]
    -------------------|----------|--------------
    | Filter Size      | H        | [1, 255]
    |                  | W        | [1, 255]
    -------------------|----------|--------------
    | Stride           | H        | [1, 63]
    |                  | W        | [1, 63]
    -------------------|----------|--------------
    | Padding          | Top      | [0, 255]
    |                  | Bottom   | [0, 255]
    |                  | Left     | [0, 255]
    |                  | Right    | [0, 255]
    -------------------|----------|--------------
    | Dilation         | H        | [1, 255]
    |                  | W        | [1, 255]
    -------------------|----------|--------------
    | Offset_x         |          | [-128, 127]

@endverbatim
* The W dimension of the input image supports cases exceeding 4096, but it may
* cause compilation errors.
*\n
*
*@par Outputs:
*@li y: A 4D Tensor of output feature map. Has the same type as "x". With the
* format "NHWC", the data is stored in the order of: [batch, out_height,
* out_width, out_channels].
*\n
*     out_height = (in_height + pad_top + pad_bottom -
*                   (dilation_h * (filter_height - 1) + 1))
*                  / stride_h + 1
*\n
*     out_width = (in_width + pad_left + pad_right -
*                  (dilation_w * (filter_width - 1) + 1))
*                 / stride_w + 1
*\n
*
*@par Quantization supported or not
*@li Yes
*
*@par Third-party framework compatibility
*@li Compatible with the TensorFlow operator "conv2d".
*@li Compatible with the Caffe operator 2D "Convolution".
*/
REG_OP(Conv2D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NHWC")
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(Conv2D)

/**
*@brief Computes a 2D convolution given 4D "x" and "filter_compress" tensors.
*@par Inputs:
* @li x: A 4D tensor of input images.
* @li filter_compress: A 4D tensor of compressed filters.
* @li compress_index: A 1D Tensor dtype of int8.
* @li bias: An optional 1D tensor.
* @li offset_w: An optional 1D tensor for quantized convolution. Reserved.
*
* The input and output tensor attributes are listed as follows:
* @verbatim
    |Tensor    | x       | filter_compress  | bias    | offset_w | y
    -----------|---------|---------|---------|----------|--------
    |Data Type | float16 | float16 | float16 | _        | float16
    |          |---------|---------|---------|----------|--------
    |          | float32 | float32 | float32 | _        | float32
    |          |---------|---------|---------|----------|--------
    |          | int8    | int8    | int32   | int8     | int32
    -----------|---------|---------|---------|----------|--------
    |Format    | NCHW    | NCHW    | ND      | ND       | NCHW
    |          | NHWC    | NHWC    |         |          | NHWC
    |          |         | HWCN    |         |          |
@endverbatim
* It should be noted that the data types must correspond to each other, but the
* format does not need to . \n

*@par Attributes:
* @li strides: A list of 4 integers. Specifying the strides of the
* convolution along the height and width. The dimension order is determined
* by the data format of "x". By default the N and C dimensions are set to 1.
* @li pads: A list of 4 integers. Specifying the top, bottom, left and right
* padding.
* @li dilations: A list of 4 integers. Specifying the dilation rate to use
* for dilated convolution. Has the same dimension order and value as "strides".
* @li groups: Number of blocked connections from input channels to output
* channels. Input channels and output channels must both be divisible by
* "groups".Type is int32.
* @li offset_x: An optional integer for quantized convolution. Type is int32.
* Defaults to "0".
* @li data_format: An optional string from: "NHWC", "NCHW". Specifying the
* data format of the input and output images. Type is string.
* Defaults to "NHWC". Reserved . \n

*@par Outputs:
* @li y: A 4D Tensor of output images . \n

*@par Restrictions:
*Warning: THIS FUNCTION IS DEPRECATED.
*/
REG_OP(Conv2DCompress)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8}))
    .INPUT(filter_compress, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8}))
    .INPUT(compress_index, TensorType({DT_INT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NHWC")
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(Conv2DCompress)

/**
*@brief Computes a 2D deformable convolution given 4D "x", "filter" and
* "offsets" tensors.
*@par Inputs:
*@li x: A 4D tensor of input image. With the format "NHWC", the data is stored
* in the order of: [batch, in_height, in_width, in_channels].
*@li filter: A 4D tensor of learnable filters. Must have the same type as "x".
* With the format "HWCN" , the data is stored in the order of: [filter_height,
* filter_width, in_channels / groups, out_channels].
*@li offsets: A 4D tensor of x-y coordinates offset and mask. With the format
* "NHWC", the data is stored in the order of: [batch, out_height, out_width,
* deformable_groups * filter_height * filter_width * 3].
*@li bias: An optional 1D tensor of additive biases to the filter outputs.
* The data is stored in the order of: [out_channels].
*\n
*\n
* The following are the supported data types and data formats:
*@verbatim
    | Tensor    | x       | filter  | offsets | bias     | y
    ------------|---------|---------|---------|----------|--------
    | Data Type | float16 | float16 | float16 | float16  | float16
    |           |---------|---------|---------|----------|--------
    |           | float32 | float32 | float32 | float32  | float32
    ------------|---------|---------|---------|----------|--------
    | Format    | NCHW    | NCHW    | NCHW    | ND       | NCHW
    |           | NHWC    | HWCN    | NHWC    |          | NHWC
@endverbatim
* For float32 type, the actual convolution calculation part on the chip is
* based on float16.
*\n
*
*@par Attributes:
*@li strides: Required. A list of 4 integers. The stride of the sliding window
* for each dimension of input. The dimension order is interpreted according to
* the data format of "x". The N and C dimensions must be set to 1.
*@li pads: Required. A list of 4 integers. The number of pixels to add to each
* (top, bottom, left, right) side of the input.
*@li dilations: Optional. A list of 4 integers. The dilation factor for each
* dimension of input. The dimension order is interpreted according to the data
* format of "x". The N and C dimensions must be set to 1. Defaults to
* [1, 1, 1, 1].
*@li groups: Optional. An integer of type int32. The number of blocked
* connections from input channels to output channels. In_channels and
* out_channels must both be divisible by "groups". Defaults to 1.
*@li data_format: Reserved.
*@li deformable_groups: Optional. An integer of type int32. The number of
* deformable group partitions. In_channels must be divisible by
* "deformable_groups". Defaults to 1.
*\n
*\n
* The following value range restrictions must be met:
*@verbatim
    | Name              | Field  | Scope
    --------------------|--------|----------------------------
    | Input Image Size  | H      | [1, 100000 / filter_height]
    |                   | W      | [1, 4096 / filter_width]
    --------------------|--------|----------------------------
    | Filter Size       | H      | [1, 63]
    |                   | W      | [1, 63]
@endverbatim
*\n
*
*@par Outputs:
*@li y:  A 4D Tensor of output feature map. Has the same type as "x". With the
* format "NHWC", the data is stored in the order of: [batch, out_height,
* out_width, out_channels].
*\n
*     out_height = (in_height + pad_top + pad_bottom -
*                   (dilation_h * (filter_height - 1) + 1))
*                  / stride_h + 1
*\n
*     out_width = (in_width + pad_left + pad_right -
*                  (dilation_w * (filter_width - 1) + 1))
*                 / stride_w + 1
*\n
*
*@par Quantization supported or not
*@li No
*
*@par Third-party framework compatibility
*@li Compatible with the Mxnet operator "DeformableConvolution".
*@li Compatible with the Paddlepaddle operator "deformable_conv".
*@li Compatible with the Mmcv operator "deform_conv".
*/
REG_OP(DeformableConv2D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(offsets, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NHWC")
    .ATTR(deformable_groups, Int, 1)
    .OP_END_FACTORY_REG(DeformableConv2D)

/**
*@brief Computes a 3D convolution given 5D "x" and "filter" tensors.
 *@par Inputs:
 * @li x: A 5D tensor. Must be one of the following types: float16,
 * (Currently does not support int8). The format of x is NCDHW or NDHWC.
 * @li filter: A 5D tensor of the same type as "x".
 * (Currently does not support int8).
 * The format is NCDHW, NDHWC or DHWCN . \n

*@par Optional input:
 * @li bias: An optional 1D tensor of the same type as "x".
 * @li offset_w: An optional 1D tensor for quantized deconvolution. Reserved . \n

*@par Required Attributes:
 * @li strides: A list of 5 integers. Specifies the stride of the sliding window
 * for each dimension of "x".
 * The N and C dimensions must be 1. Has the same format as "x".
 * @li pads: A list of 6 integers.
 * Supports only padding along the D, H and W dimensions in sequence of head,
 * tail, top, bottom, left and right . \n

*@par Attributes:
 * @li groups: Number of blocked connections from input channels to output
 * channels. Reserved.
 * @li data_format: An optional string from: "NDHWC", "NCDHW".
 * Defaults to "NDHWC". Specify the data format of the input and output data.
 * @li dilations: A list of 5 integers. Specifies the dilation factor for each
 * dimension of "x", now only support [1,1,1,1,1]
 * The N and C dimensions must be 1. Has the same format as "x".
 * @li offset_x: An optional int. Input offset, used for quantized inference.
 * Defaults to 0. Reserved . \n

*@par Outputs:
 *y: A Tensor. Has the same type and data format as "x". \n

*@attention Constraints:
 *The image size after padding is greater than the filter size . \n

*@par Third-party framework compatibility
 * @li Compatible with the TensorFlow operator conv3d.
 * @li Compatible with the Caffe operator Convolution.
*/
REG_OP(Conv3D)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(filter, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NDHWC")
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(Conv3D)


/**
*@brief Computes the gradients of convolution 3d with respect to the input.
*@par Inputs:
 * Three inputs:
 * @li input_size: A Tensor of type int32, int64. An integer vector representing
 * the shape of input, where input is a 5-D tensor
 * [batch, depth, height, width, channels] or
 * [batch, channels, depth, height, width].
 * @li filter: A Tensor. Must be one of the following types: float16, float32.
 * Currently does not support double.
 * @li out_backprop: A Tensor. Must have the same type as filter.
 * 5-D with shape [batch, depth, out_height, out_width, out_channels]
 * or [batch, out_channels, depth, out_height, out_width]. Gradients with
 * respect to the output of the convolution . \n

*@par Required Attributes:
 * @li strides: A list of 5 integers. Specifies the stride of the sliding window
 * for each dimension of "x".
 * The N and C dimensions must be 1. Has the same format as "x".
 * @li pads: A list of 6 integers.
 * Supports only padding along the D, H and W dimensions in sequence of head,
 * tail, top, bottom, left and right . \n

*@par Attributes:
 * Three attributes:
 * @li groups: Number of blocked connections from input channels to output
 * channels. Reserved.
 * @li data_format: An optional string from: "NDHWC", "NCDHW".
 * Defaults to "NDHWC". Specify the data format of the input and output data.
 * @li dilations: A tuple/list of 5 integers, The dilation factor for each
 * dimension of the input, now only support [1,1,1,1,1]

*@par Outputs:
 * y: A Tensor. Has the same type as filter,and has same format as input_size

*@par Third-party framework compatibility
 * Compatible with Tensorflow's conv3d_backprop_input
*/
REG_OP(Conv3DBackpropInput)
    .INPUT(input_size, TensorType({DT_INT32, DT_INT64}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(out_backprop, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(Conv3DBackpropInput)

/**
*@brief Computes the gradients of convolution 3d with respect to the input.
*@par Inputs:
 * Two inputs:
 * @li filter: A Tensor whose type is float16. The format of filter is NCDHW,
 * NDHWC or DHWCN.
 * @li out_backprop: A Tensor. Must have the same type as filter. The format is
 * NDHWC or NCDHW.  \n

*@par Required Attributes:
 * @li strides: A list of 5 integers. Specifies the stride of the sliding window
 * for each dimension of "x".
 * The N and C dimensions must be 1. Has the same format as "x".
 * @li pads: A list of 6 integers. Supports only padding along the D, H and W
 * dimensions in sequence of head, tail, top, bottom, left and right.
 * @li input_size: A tuple/list of type int32, int64. An integer vector
 * representing the shape of input, where input is a 5-D tensor
 * [batch, depth, height, width, channels] or
 * [batch, channels, depth, height, width] . \n

*@par Attributes:
 * Three attributes:
 * @li groups: Number of blocked connections from input channels to output
 * channels. Reserved.
 * @li data_format: An optional string from: "NDHWC", "NCDHW".
 * Defaults to "NDHWC". Specify the data format of the input and output data.
 * @li dilations: A tuple/list of 5 integers, The dilation factor for each
 * dimension of input, now only support [1,1,1,1,1]
*@par Outputs:
 * y: A Tensor. Has the same type and data format as out_backprop.
*@par Third-party framework compatibility
 * Compatible with Tensorflow's conv3d_backprop_input

*@par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use Conv3DBackpropInput instead.
*/
REG_OP(Conv3DBackpropInputD)
    .INPUT(filter, TensorType({DT_FLOAT16}))
    .INPUT(out_backprop, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(input_size, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(Conv3DBackpropInputD)

/**
*@brief Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence . \n

*@par Inputs:
* @li x: A Tensor dtype of float16.
* @li cont: A Tensor dtype of float16, float32.
* @li w_x: A Tensor dtype of float16.
* @li bias: A Tensor dtype of int16, int32, float16, float32.
* @li w_h: A Tensor dtype of float16.
* @li x_static: A optinal Tensor dtype of float16.
* @li h_0: A optinal Tensor dtype of float16, float32.
* @li c_0: A optinal Tensor dtype of float16, float32.
* @li w_x_static: A optinal Tensor dtype of float16 . \n

*@par Attributes:
*@li num_output: A Scalar of output size dtype of int.
*@li expose_hidden: A Scalar(bool) of features hidden . \n

*@par Outputs:
*@li h: A Tensor dtype of float16, float32.
* @li h_t: A optinal Tensor dtype of float16, float32. The hidden state at time t.
* @li c_t: A optinal Tensor dtype of float16, float32. The cell state at time t . \n

*@par Third-party framework compatibility:
* Compatible with the Caffe operator LSTM.
*/
REG_OP(LSTM)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(cont, TensorType({DT_FLOAT32,DT_FLOAT16}))
    .INPUT(w_x, TensorType({DT_FLOAT16}))
    .INPUT(bias, TensorType({DT_FLOAT16,DT_FLOAT32,DT_INT16,DT_INT32}))
    .INPUT(w_h, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(x_static, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(h_0, TensorType({DT_FLOAT16,DT_FLOAT32}))
    .OPTIONAL_INPUT(c_0, TensorType({DT_FLOAT16,DT_FLOAT32}))
    .OPTIONAL_INPUT(w_x_static, TensorType({DT_FLOAT16}))
    .OUTPUT(h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(h_t, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(c_t, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(num_output, Int, 0)
    .ATTR(expose_hidden, Bool, false)
    .OP_END_FACTORY_REG(LSTM)

/**
*@brief Computes the gradients of convolution3D with respect to the filter
*@par Inputs:
 * Three inputs:
 * @li x: A Tensor. Must be one of the following types: float16, float32.
 * Currently does not support double.
 * 5-D with shape [batch, in_depth, in_height, in_width, in_channels]
 * or [batch, in_channels, in_depth, in_height, in_width].
 * @li filter_size: A Tensor of type int32. An integer vector representing the
 * tensor shape of filter, where filter is a 5-D tensor
 * [filter_depth, filter_height, filter_width, in_channels, out_channels]
 * [out_channels, in_channels, filter_depth, filter_height, filter_width]
 * or [out_channels, filter_depth, filter_height, filter_width, in_channels].
 * @li out_backprop: A Tensor. Must have the same type as x.
 * 5-D with shape [batch, out_depth, out_height, out_width, out_channels]
 * or [batch, out_channels, out_depth, out_height, out_width].
 * Gradients with respect to the output of the convolution. \n

*@par Required Attributes:
 * @li strides: A tuple/list of 5 integers. Specifies the stride of the sliding
 * window for each dimension of "x". The N and C dimensions must be 1.
 * Has the same format as "x".
 * @li pads: A tuple/list of 6 integers, [front, back, top, bottom, left, right]
 * pads on feature map . \n

*@par Attributes:
 * Three attributes:
 * @li dilations: A tuple/list of 5 integers, The dilation factor for each
 * dimension of input, now only support [1,1,1,1,1].
 * @li groups: Number of blocked connections from input channels to output
 * channels. Reserved.
 * @li data_format: An optional string from: "NDHWC", "NCDHW".
 * Defaults to "NDHWC". Specify the data format of the input and output data.

*@par Outputs:
 * y: A Tensor that has the same type as x
 * and the format is NDHWC, NCDHW or DHWCN.
*@par Third-party framework compatibility
 * Compatible with Tensorflow's conv3d_backprop_filter
*/
REG_OP(Conv3DBackpropFilter)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(filter_size, TensorType({DT_INT32}))
    .INPUT(out_backprop, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(Conv3DBackpropFilter)

/**
*@brief Computes the gradients of convolution with respect to the filter.
*@par Inputs:
 * Two inputs:
 * @li x: A Tensor of type float16.
 * 5-D with shape [batch, in_depth, in_height, in_width, in_channels]
 * or [batch, in_channels, in_depth, in_height, in_width].
 * @li out_backprop: A Tensor. Must have the same type as x.
 * 5-D with shape [batch, out_depth, out_height, out_width, out_channels]
 * or [batch, out_channels, out_depth, out_height, out_width].
 * Gradients with respect to the output of the convolution. \n

*@par Required Attributes:
 * @li filter_size: A tuple/list of type integers. An integer vector
 * representing the tensor shape of filter, where filter is a 5-D tensor
 * [filter_depth, filter_height, filter_width, in_channels, out_channels],
 * [out_channels, filter_depth, filter_height, filter_width, in_channels]
 * or [out_channels, in_channels, filter_depth, filter_height, filter_width].
 * @li strides: A tuple/list of 5 integers. Specifies the stride of the sliding
 * window for each dimension of "x".
 * The N and C dimensions must be 1. Has the same format as "x".
 * @li pads: A tuple/list of 6 integers, [front, back, top, bottom, left, right]
 * pads on feature map. \n

*@par Attributes:
 * Three attributes:
 * @li dilations: A tuple/list of 5 integers, The dilation factor for each
 * dimension of input, now only support [1,1,1,1,1].
 * @li groups: Number of blocked connections from input channels to output
 * channels. Reserved.
 * @li data_format: An optional string from: "NDHWC", "NCDHW".
 * Defaults to "NDHWC". Specify the data format of the input and output data.

*@par Outputs:
 * y: A Tensor of type float32 and the format is NDHWC, NCDHW or DHWCN.
*@par Third-party framework compatibility
 * Compatible with Tensorflow's conv3d_backprop_filter
*@par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use Conv3DBackpropFilter instead.
*/


REG_OP(Conv3DBackpropFilterD)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(out_backprop, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(filter_size, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(Conv3DBackpropFilterD)

/**
*@brief Computes the transpose of convolution 3d with respect to the input.
*@par Inputs:
 * Three inputs:
 * @li input_size: A Tensor of type int32. An integer vector representing the
 * shape of input.
 * @li x: A Tensor of type float16, currently does not support int8. The format
 * is NDHWC or NCDHW.
 * @li filter: A Tensor of type float16, currently does not support int8.
 * The format is NDHWC, NCDHW or DHWCN.

*@par Optional input:
 * Two optional inputs
 * @li bias: An optional 1D tensor of the same type as "x". Reserved.
 * @li offset_w: An optional 1D tensor for quantized deconvolution. Reserved . \n

*@par Required Attributes:
 * @li strides: A tuple/list of 5 integers. Specifies the stride of the sliding
 * window for each dimension of "x".
 * The N and C dimensions must be 1. Has the same format as "x".
 * @li pads: A tuple/list of 6 integers

*@par Attributes:
 * Five attributes:
 * @li groups: Number of blocked connections from input channels to output
 * channels. Reserved.
 * @li dilations: A tuple/list of 5 integers,
 * The dilation factor for each dimension of input, now only support [1,1,1,1,1]
 * @li data_format: An optional string from: "NDHWC", "NCDHW".
 * Defaults to "NDHWC". Specify the data format of the input and output data.
 * @li output_padding: The size will be added in the output shape.
 * @li offset_x: Input offset_x value. Reserved.
*@par Outputs:
 * y: A Tensor. Has the same type and format as x.
*/
REG_OP(Conv3DTranspose)
    .INPUT(input_size, TensorType({DT_INT32, DT_INT64}))
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(filter, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NDHWC")
    .ATTR(output_padding, ListInt, {0, 0, 0, 0, 0})
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(Conv3DTranspose)

/**
*@brief Computes the transpose of convolution 3d with respect to the input.
*@par Inputs:
 * @li x: A Tensor of type float16, currently does not support int8.
 * The format is NDHWC or NCDHW.
 * @li filter: A Tensor of type float16, currently does not support int8.
 * The format is NDHWC, NCDHW or DHWCN.

*@par Optional inputs:
 * @li bias: An optional 1D tensor of the same type as "x". Reserved.
 * @li offset_w: An optional 1D tensor for quantized deconvolution. Reserved . \n

*@par Required Attributes:
 * @li input_size: A tuple/list of type int32.
 * An integer vector representing the shape of input
 * @li strides: A tuple/list of 5 integers.
 * Specifies the stride of the sliding window for each dimension of "x".
 * The N and C dimensions must be 1. Has the same format as "x".
 * @li pads: A tuple/list of 6 integers . \n

*@par Attributes:
 * Five attributes:
 * @li dilations: A tuple/list of 5 integers, The dilation factor for each
 * dimension of input, now only support [1,1,1,1,1]
 * @li groups: Number of blocked connections from input channels to output
 * channels. Reserved.
 * @li data_format: An optional string from: "NDHWC", "NCDHW".
 * Defaults to "NDHWC". Specify the data format of the input and output data.
 * @li output_padding: The size will be added in the output shape.
 * @li offset_x: Input offset_x value. Reserved.
*@par Outputs:
 * y: A Tensor. Has the same type and format as x.
*@par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use Conv3DTranspose instead.
*/
REG_OP(Conv3DTransposeD)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(filter, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(input_size, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NDHWC")
    .ATTR(output_padding, ListInt, {0, 0, 0, 0, 0})
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(Conv3DTransposeD)

/**
*@brief Computes the transpose of convolution 2d with respect to the input.
*@par Inputs:
 * Five inputs:
 * @li input_size: A Tensor of type int32 or int64. An integer vector
 * representing the shape of input, where input is a 4-D tensor
 * [batch, height, width, channels] or [batch, channels, height, width].
 * @li x: A Tensor of type float16, int8. 4-D with shape [batch, out_height,
 * out_width, out_channels] or [batch, out_channels, out_height, out_width].
 * @li filter: A Tensor of type float16, int8. Must have the same type as "x".
 * 4-D with shape [filter_height, filter_width, in_channels, out_channels]
 * or [out_channels, filter_height, filter_width, in_channels]
 * or [out_channels, in_channel, filter_height, filter_width].
 * @li bias: An optional 1D tensor of type float16 or int32. Format is "ND".
 * @li offset_w: An optional 1D tensor for quantized inference. Reserved.
 *\n
 *\n
 * The following are the supported data types and data formats:
*@verbatim
    | Tensor    | x       | filter  | bias    | y
    ------------|---------|---------|---------|--------
    | Data Type | float16 | float16 | float16 | float16
    |           |---------|---------|---------|--------
    |           | int8    | int8    | int32   | int32
    ------------|---------|---------|---------|--------
    | Format    | NCHW    | NCHW    | ND      | NCHW
    |           | NHWC    | HWCN    |         | NHWC
@endverbatim
 * For int8, a dequant or requant operator must be followed.
 *\n
 *
*@par Required Attributes:
 * @li strides: A required tuple/list of 4 integers. The stride of the sliding
 * window for H/W dimension. The index of H/W is same as data_format.
 * @li pads: A required tuple/list of 4 integers, [top, bottom, left, right]
 * pads on feature map.
*@par Attributes:
 * Five attributes:
 * @li groups: Number of blocked connections from input channels to output
 * channels.
 * Defaults to "1".
 * @li dilations: A tuple/list of 4 integers, The dilation factor for each
 * dimension of input. Must be [1, 1, 1, 1].
 * @li data_format: An optional string from: "NHWC", "NCHW". Defaults to "NHWC".
 * Specify the data format of the input and output data.
 * @li output_padding: The size will be added in the output shape. Defaults
 * to [0, 0, 0, 0].
 * @li offset_x: An optional int. Input offset, used for quantized inference.
 * Defaults to "0".
 *\n
 *\n
 * The following value range restrictions must be met:
*@verbatim
    | Name             | Field    | Scope
    -------------------|----------|--------------
    | input_size       | H        | [1, 4096]
    |                  | W        | [1, 4096]
    -------------------|----------|--------------
    | x (out_backprop) | H        | [1, 4096]
    |                  | W        | [1, 4096]
    -------------------|----------|--------------
    | filter           | H        | [1, 255]
    |                  | W        | [1, 255]
    -------------------|----------|--------------
    | y (fmap)         | H        | [1, 4096]
    |                  | W        | [1, 4096]
    -------------------|----------|--------------
    | Stride           | H        | [1, 63]
    |                  | W        | [1, 63]
    -------------------|----------|--------------
    | Padding          | Top      | [0, 255]
    |                  | Bottom   | [0, 255]
    |                  | Left     | [0, 255]
    |                  | Right    | [0, 255]
    -------------------|----------|--------------
    | Dilation         | H        | [1, 255]
    |                  | W        | [1, 255]
    -------------------|----------|--------------
    | Offset_x         |          | [-128, 127]

@endverbatim
 * In Ascend910, fmap or out_backprop's H and W not support 1 when
 * fmap_h + pad_top + pad_bottom != (filter_height - 1) * dilation_h + 1
 *\n
 *
*@par Outputs:
 * y: A Tensor. A Tensor of type float16 or int32, and has same format as
 * input_size.
 *\n
 *     out_backprop_height = (fmap_height + pad_top + pad_bottom -
 *                           (dilation_h * (filter_height - 1) + 1))
 *                           / stride_h + 1
 *\n
 *     out_backprop_width = (fmap_width + pad_left + pad_right -
 *                          (dilation_w * (filter_width - 1) + 1))
 *                          / stride_w + 1
 *\n
 *
*/
REG_OP(Conv2DTranspose)
    .INPUT(input_size, TensorType({DT_INT32, DT_INT64}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_INT8}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_INT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT32}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NHWC")
    .ATTR(output_padding, ListInt, {0, 0, 0, 0})
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(Conv2DTranspose)

/**
*@brief Computes the transpose of convolution 2d with respect to the input.
*@par Inputs:
 * Four inputs:
 * @li x: A Tensor of type float16, int8.
 * @li filter: A Tensor of type float16, int8. Must have the same type as "x".
 * @li bias: An optional 1D tensor of the same type as "x".
 * @li offset_w: An optional 1D tensor for quantized inference. Type is int8. Reserved.
*@par Required Attributes:
 * @li input_size: A Tensor of type int32 or int64. An integer vector representing the
 * shape of input.
 * @li strides: A required list or tuple. The stride of the sliding window for
 * height and width for H/W dimension.
 * @li pads: A required list or tuple of int32. Padding added to each dimension
 * of the input.
*@par Attributes:
 * Five attributes:
 * @li groups: Number of blocked connections from input channels to output channels.
 * Defaults to "1".
 * @li dilations: A tuple/list of 4 integers, The dilation factor for each dimension
 * of input. Must be [1, 1, 1, 1].
 * @li data_format: An optional string from: "NHWC", "NCHW". Defaults to "NHWC".
 * Specify the data format of the input and output data.
 * @li output_padding: The size will be added in the output shape. Defaults
 * to [0, 0, 0, 0].
 * @li offset_x: An optional int. Input offset, used for quantized inference.
 * Defaults to "0".
*@par Outputs:
 * y: A Tensor. Has the same type as "filter".
*@par Restrictions:
 * Warning: THIS FUNCTION IS DEPRECATED. Please use Conv2DTranspose instead.
*/
REG_OP(Conv2DTransposeD)
    .INPUT(x, TensorType({DT_FLOAT16, DT_INT8}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_INT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT32}))
    .REQUIRED_ATTR(input_size, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NHWC")
    .ATTR(output_padding, ListInt, {0, 0, 0, 0})
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(Conv2DTransposeD)

/**
*@brief Computes the deformed convolution output with the expected input
*@par Inputs:
 * Two inputs:
 * @li x: A Tensor of type float16,float32
 * @li offsets: A Tensor of type float16,float32.Deformation offset parameter.
*@par Required Attributes:
 * @li strides: A tuple/list of 4 integers.The stride of the sliding window for
 * height and width for H/W dimension.
 * @li pads: A tuple/list of 4 integers.Padding added to H/W dimension
 * of the input.
 * @li ksize: A tuple/list of 2 integers.kernel size.
*@par Attributes:
 * Three attributes:
 * @li dilations: A tuple/list of 4 integers, The dilation factor for each dimension
 * of input.  Defaults to [1, 1, 1, 1]
 * @li data_format: An optional string from: "NCHW", "NHWC". Defaults to "NCHW". Specify the data format of the input x.
 * @li deformable_groups: Specify the c-axis grouping number of input x.
 * @li modulated: Specify version of DeformableConv2D, true means v2, false means v1
*@par Outputs:
 * y: A Tensor. A Tensor of type float16, float32.
*/
REG_OP(DeformableOffsets)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(offsets, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .REQUIRED_ATTR(ksize, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(data_format, String, "NCHW")
    .ATTR(deformable_groups, Int, 1)
    .ATTR(modulated, Bool, true)
    .OP_END_FACTORY_REG(DeformableOffsets)

/**
*@brief Computes the gradients of DeformableOffsets with respect to input and offsets
*@par Inputs:
 * Three inputs:
 * @li grad: A Tensor of type float16,float32. gradients with respect to DeformableOffsets output
 * @li x: A Tensor of type float16,float32.
 * @li offsets: A Tensor of type float16,float32.Deformation offset parameter.
*@par Required Attributes:
 * @li strides: A tuple/list of 4 integers.The stride of the sliding window for
 * height and width for H/W dimension.
 * @li pads: A tuple/list of 4 integers.Padding added to H/W dimension
 * of the input.
 * @li ksize: A tuple/list of 2 integers.kernel size.
*@par Attributes:
 * Three attributes:
 * @li dilations: A tuple/list of 4 integers, The dilation factor for each dimension
 * of input.  Defaults to [1, 1, 1, 1]
 * @li data_format: An optional string from: "NCHW", "NHWC". Defaults to "NCHW". Specify the data format of the input x.
 * @li deformable_groups: Specify the c-axis grouping number of input x.
 * @li modulated: Specify version of DeformableConv2D, true means v2, false means v1.
*@par Outputs:
 * grad_x: A Tensor of type float16, float32. Gradients with respect to input_x
 * grad_offsets: A Tensor of type float16, float32. Gradients with respect to input_offsets
*/
REG_OP(DeformableOffsetsGrad)
    .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(offsets, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(grad_x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(grad_offsets, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .REQUIRED_ATTR(ksize, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(data_format, String, "NCHW")
    .ATTR(deformable_groups, Int, 1)
    .ATTR(modulated, Bool, true)
    .OP_END_FACTORY_REG(DeformableOffsetsGrad)

/**
*@brief Computes the deformed dilation output with the expected input
*@par Inputs:
 * One inputs:
 * @li x: A Tensor of type int8, float16, float32
*@par Required Attributes:
 * @li dilations: A tuple/list of integers.
*@par Attributes:
 * Two attributes:
 * @li padding_value: default value filling in blank
 * @li pads: A tuple/list of integers.
*@par Outputs:
 * y: A Tensor. A Tensor of type int8, float16, float32.
*/
REG_OP(Dilation)
    .INPUT(x, TensorType({DT_INT8, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_INT8, DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(dilations, ListInt)
    .ATTR(pads, ListInt, {})
    .ATTR(padding_value, Float, 0.0)
    .OP_END_FACTORY_REG(Dilation)

}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_CALCULATION_OPS_H_
