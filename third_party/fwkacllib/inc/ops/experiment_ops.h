/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file experiment_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_EXPERIMENT_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_EXPERIMENT_OPS_H_

#include "graph/operator_reg.h"
namespace ge {
/**
* @brief Updates "var" according to the AdamW algorithm.
*
* @attention Constraints:
*  The input tensors must have the same shape.*
*
* @par Inputs:
* @li var: A mutable Tensor of the type TensorType::NumberType().
*     Should be from a Variable().
* @li m: A mutable Tensor of the same type as "var".
*     Should be from a Variable().
* @li v: A mutable Tensor of the same type as "var".
*     Should be from a Variable().
* @li beta1_power: A scalar of the same type as "var".
* @li beta2_power: A scalar of the same type as "var".
* @li lr: learning_rate. A scalar of the same type as "var".
* @li weight_decay: learning_rate. A scalar of the same type as "var".
* @li beta1: A scalar of the same type as "var".
* @li beta2: A scalar of the same type as "var".
* @li epsilon: A scalar of the same type as "var".
* @li grad: A Tensor of the same type as "var", for the gradient.
* @li max_grad_norm: A mutable Tensor of the same type as "var", an optional input.
*     Should be from a Variable().
*
* @par Attributes:
* @li amsgrad: An optional bool. Defaults to "False".
*     If "True", max_grad_norm input and output must be entered.
* @li maximize: An optional bool. Defaults to "False".
*
* @par Outputs:
* @li var: A mutable tensor. Has the same type as input "var".
* @li m: A mutable tensor. Has the same type as input "m".
* @li v: A mutable tensor. Has the same type as input "v". \n
*/
REG_OP(ApplyAdamW)
    .INPUT(var, TensorType::NumberType())
    .INPUT(m, TensorType::NumberType())
    .INPUT(v, TensorType::NumberType())
    .INPUT(beta1_power, TensorType::NumberType())
    .INPUT(beta2_power, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(weight_decay, TensorType::NumberType())
    .INPUT(beta1, TensorType::NumberType())
    .INPUT(beta2, TensorType::NumberType())
    .INPUT(epsilon, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OPTIONAL_INPUT(max_grad_norm, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(m, TensorType::NumberType())
    .OUTPUT(v, TensorType::NumberType())
    .ATTR(amsgrad, Bool, false)
    .ATTR(maximize, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdamW)

/**
* @brief Calculate SQ distance. \n
*
* @par Inputs:
* @li ivf: A Tensor, dtype is uint8.
* @li query: A Tensor, dtype is float16 or float32.
* @li bucket_list: A Tensor, dtype is int32 or int64.
* @li bucket_limits: A Tensor, dtype is int32 or int64.
* @li bucket_offsets: A Tensor, dtype is int32 or int64.
* @li vmin: A Tensor, dtype is float16 or float32.
* @li vdiff: A Tensor, dtype is float16 or float32. \n
*
* @par Outputs:
* @li actual_count: A Tensor, dtype is int32 or int64, the actual number of sq_distance.
* @li sq_distance: A Tensor, dtype is float16 or float32.
* @li grouped_extreme_distance: A Tensor, dtype is float16 or float32, the extremum in each group of sq_distance.
* @li sq_ivf: A Tensor, dtype is int32 or int64.
* @li sq_index: A Tensor, dtype is int32 or int64. \n
*
* @par Attributes:
* @li total_limit: A Int, indicates the max length of the output sq_distance.
* @li group_size: A Int, indicates the group size of the extremum.
* @li extreme_mode: A Int, indicates the type of extremum, 0 means minimum, and 1 means maximum. \n
*
*/
REG_OP(ScanSQCodes)
    .INPUT(ivf, TensorType({DT_UINT8}))
    .INPUT(query, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(bucket_list, TensorType({DT_INT32, DT_INT64}))
    .INPUT(bucket_limits, TensorType({DT_INT32, DT_INT64}))
    .INPUT(bucket_offsets, TensorType({DT_INT32, DT_INT64}))
    .INPUT(vmin, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(vdiff, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(actual_count, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(sq_distance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(grouped_extreme_distance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(sq_ivf, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(sq_index, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(total_limit, Int)
    .ATTR(group_size, Int, 64)
    .ATTR(extreme_mode, Int, 0)
    .OP_END_FACTORY_REG(ScanSQCodes)

/**
* @brief Multiplies matrix "a" by matrix "b", producing "a * b". \n
* @par Inputs:
* Four inputs, including:
* @li x1: A matrix Tensor. Must be one of the following types: float32,
* float16, int32, int8, int4, bf16. 3D. Has format ND.
* @li x2: A matrix Tensor. Must be one of the following types: float32,
* float16, int32, int8, int4, bf16. 3D. Has format ND.
* @li bias: A optional Tensor. Must be one of the following types:
* float32, float16, int32, bf16. 1D. Has format ND.
* @li offset_w: A optional Tensor. Must be one of the following types:
* int8, int4. Has format ND. \n

* @par Attributes:
* Three attributes, including:
* @li perm_x1: A list int. "x1" is permuted to shape [B, M, K] before multiplication.
* @li perm_x2: A list int. "x2" is permuted to shape [B, K, N] before multiplication.
* @li perm_y: A list int. "y" is permuted after multiplication.
* @li offset_x: An optional integer for quantized TransposeBatchMatMul.
* The negative offset added to the input "x1" for int8, int4 type. Ensure offset_x
* within the effective range of input data type. Defaults to "0". \n

* @par Outputs:
* y: The result matrix Tensor. 3D. Must be one of the following
* types: float32, float16, int32, bf16. 3D. Has format ND. \n
*/
REG_OP(TransposeBatchMatMul)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT4, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT4, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8, DT_INT4}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .ATTR(perm_x1, ListInt, {})
    .ATTR(perm_x2, ListInt, {})
    .ATTR(perm_y, ListInt, {})
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(TransposeBatchMatMul)

/**
* @brief Performs non-maximum suppression (NMS) on the rotated boxes according
* to their intersection-over-union (IoU). Rotated NMS interatively removes lower
* scoring rotated boxes which have an IoU greater than iou_threshold with
* another (higher scoring) rotated box.

* @par Inputs:
* Three inputs, including:
* @li boxes: A 2D Tensor of float16 or float32 with shape (N, 5). Rotated boxes to
* perform NMS on. They are expected to be in (x1, y1, x2, y2, angle_degress) format.
* @li scores: A 1D Tensor of float16 or float32 with shape (N). Scores for each one of
* the rotated boxes.
* @li labels: A 1D Tensor of int32 or int64 with shape (N). Labels for each one of
* the rotated boxes.

* @par Attributes:
* iou_threshold: A required float attribute. Discards all overlapping rotated
* boxes with IoU < iou_threshold.

* @par Outputs:
* Two outputs, including:
* @li selected_detections: A 2D Tensor of float16 or float32 with shape (N, 5).
* The selected boxes that kept by Rotated NMS, sorted in decreasing order of scores.
* @li keep_indices: A 1D Tensor of int32 or int64 with shape (N). The indices of
* selected_detections.

* @attention Constraints:
* Currently, the tensor type of input (boxes, scores) only support float.
* The tensor type of keep_indices only support int32.
*/
REG_OP(RotatedNMS)
    .INPUT(boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(labels, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(selected_detections, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(keep_indices, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(iou_threshold, Float)
    .OP_END_FACTORY_REG(RotatedNMS)

/**
* @brief Performs average pooling on the input. Used in the combination of conv + avgpoolupdate to replace avgpool
* @par Inputs:
* x1: Output of upstream Conv2d. A tensor of type float16, float32.
* x2: Input feature map of upstream Conv2d. A tensor of type int8, float16, float32.

* @par Attributes:
* @li ksize: A required list of 4 ints, specifying the size (N, C, H, and W) of the sliding window,
* where N = C = 1, and H and W are positive integers within the range [1, 255].
* @li strides: A required list of 4 ints, specifying the stride of the sliding window.
* The strides of the N and C dimensions are 1.
* The strides of the H and W dimensions are positive integers within the range [1, 63].
* @li padding_mode: A required string, specifying the padding algorithm,
* either "VALID", "SAME" and "CALCULATED".
* With "SAME" means that the outputs will have the same spatial dimensions as its inputs.
* With "VALID" means no padding.
* @li pads: Pad value when padding_mode is "CALCULATED".
* @li data_format: An optional string, specifying the data format of "ksize" and "strides",
* either "NCHW", or "NHWC" (default).
* @li ceil_mode: Use ceil or floor to calculate the output size when padding_mode is "CALCULATED".
* @li exclusive: Ignore padding area or not when calculating average.

* @par Outputs:
* y: The average pooled output tensor. Has the same type and format as input "x1".

* @attention Constraints:
* @li Only single input and single output are supported.
* @li "ksize_H" and "ksize_W" are positive integers within the range [1, 255]. ksize_H * ksize_W < 256
* @li Due to instruction restrictions,
* the values of "strides_h" and "strides_w" are positive integers within the range [1, 63].
* @par Third-party framework compatibility
* Compatible with the TensorFlow/Pytorch/Onnx operator AvgPoolV2.
*/
REG_OP(AvgPoolUpdate)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x2, TensorType({DA_INT4, DT_INT8, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(padding_mode, String, "CALCULATED")
    .ATTR(pads, ListInt, {0, 0, 0, 0})
    .ATTR(data_format, String, "NHWC")
    .ATTR(ceil_mode, Bool, false)
    .ATTR(exclusive, Bool, true)
    .OP_END_FACTORY_REG(AvgPoolUpdate)

/**
* @brief batch input by time
* @par Inputs:
* x: A list of input tensors. It's a dynamic input

* @par Attributes:
* @li window: time window, [-1, int64_max], if -1 will batch by input data flag,
* else will batch by input timestamp and data flag.
* @li batch_dim: [-1, input_shape_range), if -1 input shape:[x, ..., x] ---> output shape:[-1, x, ..., x],
* else output shape:[x, ..., -1(batch_dim), ..., x];
* @li drop_remainder: a bool flag, take effect when window > -1,
* if true when batch data window < window, will drop data.

* @par Outputs:
* y: A list of output tensors. It's a dynamic input, the same size as "x".

* @attention Constraints:
* @li Only support in helper udf
*/
REG_OP(TimeBatch)
    .DYNAMIC_INPUT(x, TensorType::RealNumberType())
    .DYNAMIC_OUTPUT(y, TensorType::RealNumberType())
    .REQUIRED_ATTR(window, Int)
    .ATTR(batch_dim, Int, -1)
    .ATTR(drop_remainder, Bool, false)
    .OP_END_FACTORY_REG(TimeBatch)

/**
* @brief Auto Batch process. \n

* @par Inputs:
* @li x: A list of input tensor objects. It's a dynamic input. \n

* @par Outputs:
* @li y: A list of output tensor objects. It's a dynamic output. \n

* @par Attributes:
* @li batch_size: auto batch size.
* @li timeout: auto batch wait timeout(unit:ms).
* @li padding: weather to pad when batch is insufficient.
* @li slide_stride: sliding window step.
*/
REG_OP(AutoBatch)
    .DYNAMIC_INPUT(x, TensorType::RealNumberType())
    .DYNAMIC_OUTPUT(y, TensorType::RealNumberType())
    .REQUIRED_ATTR(batch_size, Int)
    .ATTR(timeout, Int, 0)
    .ATTR(padding, Bool, false)
    .ATTR(slide_stride, Int, 0)
    .OP_END_FACTORY_REG(AutoBatch)

/**
* @brief YUVToRGB

* @par Inputs:
* @li x: A 4-D uint8 Tensor.
*        Must set the format, supported format list ["NYUV"].
* @li matrix: A 1-D float tensor of 2x3x3 elements

* @par Outputs:
* @li y: A 4-D uint8 Tensor.
*        Must set the format, supported format list ["NCHW, NHWC"].

* @par Attributes:
* @li matrix_type: An Int attr, Defaults to 0.
*                  support list [ 0: CSC_MATRIX_BT601_WIDE,
*                                 1: CSC_MATRIX_BT601_NARROW,
*                                 2: CSC_MATRIX_BT709_WIDE,
*                                 3: CSC_MATRIX_BT709_NARROW,
*                                 4: CSC_MATRIX_BT2020_WIDE,
*                                 5: CSC_MATRIX_BT2020_NARROW,
*                                 6: CSC_MATRIX_USR_DEFINE ]
* @li rb_swap: An Int attr, Defaults to 0.
*              support list [ 0: RGB, 1: BGR ]

* @attention Constraints:
* @li Only support in dvpp
*/

REG_OP(YUVToRGB)
    .INPUT(x, TensorType({DT_UINT8}))
    .OPTIONAL_INPUT(matrix, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_UINT8}))
    .ATTR(matrix_type, Int, 0)
    .ATTR(rb_swap, Int, 0)
    .OP_END_FACTORY_REG(YUVToRGB)

/**
* @brief DecodeJpegPre

* @par Inputs:
* @li contents: A Tensor of type string. 0-D. The JPEG-encoded image.

* @par Outputs:
* @li dvpp_support: indicates if the dvpp support this jpeg image decode.

* @par Attributes:
* @li w_range: An required listInt contains width [min, max].
* @li h_range: An required listInt contains height [min, max].

* @attention Constraints:
* @li Only support in dvpp
*/

REG_OP(DecodeJpegPre)
    .INPUT(contents, TensorType({DT_STRING}))
    .OUTPUT(dvpp_support, BOOL)
    .REQUIRED_ATTR(w_range, ListInt)
    .REQUIRED_ATTR(h_range, ListInt)
    .OP_END_FACTORY_REG(DecodeJpegPre)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_EXPERIMENT_OPS_H_
