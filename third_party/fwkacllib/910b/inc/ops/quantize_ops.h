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
 * \file quantize_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_QUANTIZE_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_QUANTIZE_OPS_H_
#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Dequantizes the input tensor into a float tensor.
* [min_range, max_range] are float32 tensors that specify the range
* for "y".
* The "mode" attribute controls exactly which calculations are used to convert
* the float values to their quantized equivalents.
* @par Inputs:
* @li x: A Tensor. Must be one of the following types: int8, uint8,
* int32.
* @li min_range: A Tensor of type float32.
* Specifies the minimum scalar value possibly produced for the input.
* @li max_range: A Tensor of type float32.
* Specifies the maximum scalar value possibly produced for the input . \n

* @par Attributes:
* mode: An optional string from: "MIN_COMBINED", "MIN_FIRST", and "SCALED".
* Defaults to "MIN_COMBINED" . \n

* @par Outputs:
* y: A dictionary of type float32 . \n

* @attention Constraints:
* @li "min_range" and "max_range" have the same shapes.
* @li "x" and "y" have the same shapes . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Dequantize.
*/
REG_OP(Dequantize)
    .INPUT(x, TensorType(DT_QINT8, DT_QUINT8, DT_QINT32, DT_QINT16, DT_QUINT16))
    .INPUT(min_range, TensorType{DT_FLOAT})
    .INPUT(max_range, TensorType{DT_FLOAT})
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(mode, String, "MIN_COMBINED")
    .OP_END_FACTORY_REG(Dequantize)

/**
* @brief Quantizes the input . \n
* @par Inputs:
* @li x: shape and dtype of input_x. \n
* @li scales: shape and dtype of input_scales. \n
* @li zero_points: shape and dtype of input_zero_points \n
* @par Attributes:
* @li dtype: required, type.
* @li axis: the processed dim. \n
* @par Outputs:
* y: shape and dtype of output_y, should be same shape as input, dtype is same as the quantified type . \n
*/
REG_OP(Quantize)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(scales, TensorType({DT_FLOAT}))
    .INPUT(zero_points, TensorType({DT_INT8,DT_UINT8,DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT8,DT_UINT8,DT_INT32}))
    .REQUIRED_ATTR(dtype, String)
    .ATTR(axis, Int, 1)
    .OP_END_FACTORY_REG(Quantize)

/**
* @brief Quantizes the input . \n

* @par Inputs:
* x: An tensor of type float16 or float32, specifying the input . \n

* @par Attributes:
* @li scale: A required float32, specifying the scaling ratio.
* @li offset: A required float16, specifying the offset.
* @li sqrt_mode: A optional bool, specifying whether to perform square root on "scale", either "True" or "False".
* Defaults to "False".
* @li round_mode: An optional string, specifying the float16 to int8 cast type.
* The value range is [Round, Floor, Ceil, Trunc]. Defaults to "Round" .
* @li dst_type: A optional int32, specifying the output data type. Defaults to "DT_INT8" . \n

* @par Outputs:
* y: The quantized output tensor of type int8 or int4. \n

* @attention Constraints:
* round_mode value range is [Round, Floor, Ceil, Trunc].
* @li Round: round to nearest, tie to even(c language rint).
* @li Floor: round to minus infinity(c language floor).
* @li Ceil: round to positive infinity(c language ceil).
* @li Trunc: round to zero(c language trunc). \n

* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe.
*/
REG_OP(AscendQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT4}))
    .REQUIRED_ATTR(scale, Float)
    .REQUIRED_ATTR(offset, Float)
    .ATTR(sqrt_mode, Bool, false)
    .ATTR(round_mode, String, "Round")
    .ATTR(dst_type, Int, DT_INT8)
    .OP_END_FACTORY_REG(AscendQuant)

/**
* @brief Dequantizes the input . \n

 *@par Inputs:
* @li x: An tensor of type int32, specifying the input.
* @li deq_scale: An tensor of type uint64, specifying the scaling ratio . \n

* @par Attributes:
* @li sqrt_mode: A optional bool, specifying whether to perform square root on "scale", either "True" or "False".
* Defaults to "False".
* @li relu_flag: A optional bool, specifying whether to perform ReLU, either "True" or "False". Defaults to "False".
* @li dtype: A optional int32, specifying the output data type. Defaults to "DT_FLOAT" . \n

* @par Outputs:
* y: The dequantized output tensor of type float16 or float32. \n

* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe.
*/
REG_OP(AscendDequant)
    .INPUT(x, TensorType({DT_INT32}))
    .INPUT(deq_scale, TensorType({DT_FLOAT16, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(sqrt_mode, Bool, false)
    .ATTR(relu_flag, Bool, false)
    .ATTR(dtype, Int, DT_FLOAT)
    .OP_END_FACTORY_REG(AscendDequant)

/**
* @brief Anti quantizes the input . \n

* @par Inputs:
* x: An tensor of type int8, specifying the input . \n

* @par Attributes:
* @li scale: A required float32 scale.
* @li offset: A required float32 offset.
* @li dtype: A optional int32, specifying the output data type. Defaults to "DT_FLOAT".
* @li sqrt_mode: A optional bool, specifying whether to perform square root on "scale", either "True" or "False".
* Defaults to "False" . \n

* @par Outputs:
* y: The dequantized output tensor of type float16 or float32. \n

* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe.
*/
REG_OP(AscendAntiQuant)
    .INPUT(x, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(scale, Float)
    .REQUIRED_ATTR(offset, Float)
    .ATTR(dtype, Int, DT_FLOAT)
    .ATTR(sqrt_mode, Bool, false)
    .OP_END_FACTORY_REG(AscendAntiQuant)

/**
* @brief Dequantizes the input of int16 . \n

* @par Inputs:
* @li x0: An tensor of type int32, specifying the input.
* @li deq_scale: An tensor of type uint64, specifying the scaling ratio.
* @li x1: An tensor of type int16, specifying the input . \n

* @par Attributes:
* relu_flag: A optional bool, specifying whether to perform ReLU, either "True" or "False". Defaults to "False" . \n

* @par Outputs:
* y: The dequantized output tensor of type int16. \n

* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe.
*/
REG_OP(AscendDequantS16)
  .INPUT(x0, TensorType({DT_INT32}))
  .INPUT(deq_scale, TensorType({DT_UINT64}))
  .OPTIONAL_INPUT(x1, TensorType({DT_INT16}))
  .OUTPUT(y, TensorType({DT_INT16}))
  .ATTR(relu_flag, Bool, false)
  .OP_END_FACTORY_REG(AscendDequantS16)

/**
* @brief Requantizes the input . \n

* @par Inputs:
* @li x: An tensor of type int32, specifying the input.
* @li req_scale: An tensor of type uint64, specifying the scaling ratio . \n

* @par Attributes:
* relu_flag: A optional bool, specifying whether to perform ReLU, either "True" or "False". Defaults to "False" . \n

* @par Outputs:
* y: The dequantized output tensor of type int8. \n

* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe.
*/
REG_OP(AscendRequant)
  .INPUT(x, TensorType({DT_INT32}))
  .INPUT(req_scale, TensorType({DT_UINT64}))
  .OUTPUT(y, TensorType({DT_INT8}))
  .ATTR(relu_flag, Bool, false)
  .OP_END_FACTORY_REG(AscendRequant)

/**
* @brief Requantizes the input of int16 . \n

* @par Inputs:
* @li x0: An tensor of type int16, specifying the input.
* @li req_scale: An tensor of type uint64, specifying the scaling ratio.
* @li x1: An tensor of type int16 . \n

* @par Attributes:
* @li dual_output: A optional bool, specifying whether to perform dual ouput, either "True" or "False".
* Defaults to "False".
* @li relu_flag: A optional bool, specifying whether to perform ReLU, either "True" or "False". Defaults to "False" . \n

* @par Outputs:
* @li y0: The dequantized output tensor of type int8.
* @li y1: The dequantized output tensor of type int16. \n

* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe.
*/
REG_OP(AscendRequantS16)
  .INPUT(x0, TensorType({DT_INT16}))
  .INPUT(req_scale, TensorType({DT_UINT64}))
  .OPTIONAL_INPUT(x1, TensorType({DT_INT16}))
  .OUTPUT(y0, TensorType({DT_INT8}))
  .OUTPUT(y1, TensorType({DT_INT16}))
  .ATTR(dual_output, Bool, false)
  .ATTR(relu_flag, Bool, false)
  .OP_END_FACTORY_REG(AscendRequantS16)

/**
* @brief Quantizes the input of int8 . \n

* @par Inputs:
* @li x: A tensor of type int8, specifying the input.
* @li offset: A tensor of type int8.

* @par Attributes:
* @li dst_type: A optional int from: DT_INT8, DT_INT4. Defaults to DT_INT8.

* @par Outputs:
* @li y: output tensor of type int4 or int8.

* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe, Onnx, Tensorflow or Pythorch.
*/
REG_OP(AscendWeightQuant)
  .INPUT(x, TensorType({DT_INT8}))
  .INPUT(offset, TensorType({DT_INT8}))
  .OUTPUT(y, TensorType({DT_INT8, DT_INT4}))
  .ATTR(dst_type, Int, DT_INT8)
  .OP_END_FACTORY_REG(AscendWeightQuant)
} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_QUANTIZE_OPS_H_
