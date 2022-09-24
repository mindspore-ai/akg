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
 * \file nn_batch_norm_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_BATCH_NORM_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_BATCH_NORM_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Normalizes elements of a specific dimension of eigenvalues (L2) . \n

*@par Inputs:
*One input:
*x: A multi-dimensional Tensor of type float16 or float32, specifying the eigenvalue . \n

*@par Attributes:
*@li axis: A required attribute of type list, specifying the axis for normalization.
*@li eps: An optional attribute of type float, specifying the lower limit of normalization. Defaults to "1e-4" . \n

*@par Outputs:
*y: A multi-dimensional Tensor of type float16 or float32, specifying the eigenvalue for normalization . \n

*@par Third-party framework compatibility
* Compatible with the L2 scenario of PyTorch operator Normalize.
*/
REG_OP(L2Normalize)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(axis, ListInt, {})
    .ATTR(eps, Float, 1e-4)
    .OP_END_FACTORY_REG(L2Normalize)

/**
*@brief Performs the backpropagation of L2Normalize for training scenarios . \n

*@par Inputs:
* Three inputs, including:
*@li x: A multi-dimensional Tensor of type float16 or float32, specifying
* the eigenvalue of forward inputs.
*@li y: A multi-dimensional Tensor of type float16 or float32, specifying
* the normalization result of the forward output.
*@li dy: A multi-dimensional Tensor of type float16 or float32, specifying
* the reverse input gradient . \n

*@par Attributes:
*@li axis: A required attribute of type int, specifying the axis to be
* normalized.
*@li eps: An optional attribute of type float, specifying the lower limit of
* normalization. Defaults to "1e-4" . \n

*@par Outputs:
*dx: Reverse gradient of eigenvalue "x". Has the same dimensions as "x" . \n

*@par Third-party framework compatibility
* Compatible with the L2 scenario of PyTorch operator NormalizeGrad.
*/
REG_OP(L2NormalizeGrad)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(dx, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(dim, ListInt, {})
    .ATTR(eps, Float, 0.0001)
    .OP_END_FACTORY_REG(L2NormalizeGrad)

/**
*@brief Performs batch normalization . \n

*@par Inputs:
* Five inputs, including: (NHWC, NCHW)
*@li x: A 4D or 5D Tensor of type float16 or float32, with format NHWC or NCHW.
*@li scale: A Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW. 
Specifies the scaling factor.
*@li offset: A Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW. Specifies the offset.
*@li mean: A Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW. 
Specifies the mean used for inference. Must be "None" if the
operation is used for training.
*@li variance: A Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW. 
Specifies the variance used for inference. Must be "None"
if the operation is used for training . \n

*@par Attributes:
*@li epsilon: An optional float32, specifying the small value added to variance to avoid dividing by zero.
* Defaults to "0.0001".
*@li data_format: An optional string, specifying the format of "x". Defaults to "NHWC".
*@li is_training: An optional bool, specifying if the operation is used for training or inference. 
Defaults to "True" . \n

*@par Outputs:
* Five outputs, including: (NHWC, NCHW)
*@li y: A 4D or 5D Tensor of type float16 or float32 for the normalized "x", with format NHWC or NCHW.
*@li batch_mean: A Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW. 
Specifies the mean of "x".
*@li batch_variance: A Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW.
Specifies the variance of "x".
*@li reserve_space_1: An optional Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW.
Specifies the mean of "x" for gradient computation. Pass "None" to skip this output.
*@li reserve_space_2: An optional Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW.
*@li reserve_space_3: An optional Tensor of type float32. For compatibility with tensorflow, 
only has one useless emement. \n

*@attention Constraints:
*@li If the operation is used for inference and outputs "reserve_space_1" and "reserve_space_2" are available,
then "reserve_space_1" has the same value as "mean" and "reserve_space_2" has the same value as "variance".
*@li For Ascend 310, the result accuracy fails to reach 1‰ due to the square root instruction . \n

*@par Third-party framework compatibility
*@li Compatible with the TensorFlow operator fused_batch_norm.
*@li Compatible with the TensorFlow operator fused_batch_norm_v2.
*/
REG_OP(BatchNorm)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_1, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_2, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_3, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001)
    .ATTR(data_format, String, "NHWC")
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(BatchNorm)

/**
* @brief After the mean and reciprocal of standard deviation(invert_std) are separately calculated on each device,
* the mena and reciprocal of standard deviation(invert_std) data on each device are normlized,
* a total mean and reciprocal of standard deviation(invert_std) are returned, and running_var are updated.

* @par Inputs:
* include:
* @li mean_all: A Tensor. The mean of each device. Must be one of the following types: float16, float32.
* @li invert_std_all: A Tensor. Reciprocal of the variances of each device. Must be one of the following types: float16, float32.
* @li count_all: A Tensor. Number of data for each device. Must be one of the following types: float16, float32.
* @li mean_broadcast: A Tensor. The overall average and broadcast. Must be one of the following types: float16, float32.
* @li count_sum: A Tensor. General statistics. Must be one of the following types: float16, float32.
* @li running_var: A Tensor. Runtime variance. Must be one of the following types: float16, float32. \n

* @par Attributes:
* Two Attributes, including:
* @li momentum: A optional float. Defaults to 0.01. \n
* @li epsilon: An optional float. Defaults to 0.00001. \n

* @par Outputs:
* include:
* @li invert_std: A Tensor. It's inverse of total variance.
* @li running_var_update: A Tensor. It's moving variance of each device after the update. \n

* @par Third-party framework compatibility
* ReduceMeanWithCount and SyncBatchNormGatherStatsWithCounts and SyncBNTrainingUpdate
* compatible with the Pytorch operator BatchNormGatherStatsWithCounts.
*/
REG_OP(SyncBatchNormGatherStatsWithCounts)
    .INPUT(mean_all, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(invert_std_all, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(count_all, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(mean_broadcast, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(count_sum, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(running_var, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(invert_std, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(running_var_update, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(momentum, Float, 0.1)
    .ATTR(epsilon, Float, 0.001)
    .OP_END_FACTORY_REG(SyncBatchNormGatherStatsWithCounts)

/**
* @brief update running_mean.

* @par Inputs:
* include:
* @li mean: A Tensor. The mean of each device. Must be one of the following types: float16, float32.
* @li running_mean: A Tensor. Runtime Mean. Must be one of the following types: float16, float32. \n

* @par Attributes:
* One Attribute, including:
* @li momentum: A optional float. Defaults to 0.01. \n

* @par Outputs:
* include:
* @li running_mean_update: A Tensor. It's moving mean of each device after the update. \n

* @par Third-party framework compatibility
* ReduceMeanWithCount and SyncBatchNormGatherStatsWithCounts and SyncBNTrainingUpdate
* compatible with the Pytorch operator BatchNormGatherStatsWithCounts.
*/
REG_OP(SyncBNTrainingUpdate)
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(running_mean, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(running_mean_update, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(momentum, Float, 0.1)
    .OP_END_FACTORY_REG(SyncBNTrainingUpdate)

/**
*@brief part of SyncBatchNormBackward . \n

*@par Inputs:
* Three inputs, including:
*@li sum_dy: A Tensor. Must be one of the following types: float16, float32 .
*@li sum_dy_dx_pad: A Tensor. Must be one of the following types: float16, float32 .
*@li mean: A Tensor. Must be one of the following types: float16, float32 .
*@li invert_std: A Tensor. Must be one of the following types: float16, float32 . \n

*@par Outputs:
*@li sum_dy_xmu: A Tensor. Has the same type and format as input "sum_dy"
*@li y: A Tensor. Has the same type and format as input "sum_dy" . \n
*/
REG_OP(SyncBatchNormBackwardReduce)
    .INPUT(sum_dy, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(sum_dy_dx_pad, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(invert_std, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(sum_dy_xmu, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(SyncBatchNormBackwardReduce)

/**
*@brief part of SyncBatchNormBackward . \n

*@par Inputs:
* Three inputs, including:
*@li grad_output: A Tensor. Must be one of the following types: float16, float32 .
*@li save_input: A Tensor. Must be one of the following types: float16, float32 .
*@li mean: A Tensor. Must be one of the following types: float16, float32 .
*@li invstd: A Tensor. Must be one of the following types: float16, float32 .
*@li weight: A Tensor. Must be one of the following types: float16, float32 .
*@li mean_dy: A Tensor. Must be one of the following types: float16, float32 .
*@li mean_dy_xmu: A Tensor. Must be one of the following types: float16, float32 . \n

*@par Outputs:
*@li grad_input: A Tensor. Has the same type and format as input "grad_output" . \n
*/
REG_OP(SyncBatchNormBackwardElemt)
    .INPUT(grad_output, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(save_input, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(invstd, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(weight, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(mean_dy, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(mean_dy_xmu, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(grad_input, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(SyncBatchNormBackwardElemt)
    
/**
*@brief Performs batch normalization . \n

*@par Inputs:
* Five inputs, including: (NHWC, NCHW)
*@li x: A 3D or 6D Tensor of type float16 or float32, with format NDHWC or NCDHW.
*@li scale: A Tensor of type float32. Must be 1D if input "x" is with format NDHWC or NCDHW. 
Specifies the scaling factor.
*@li offset: A Tensor of type float32. Must be 3D if input "x" is with format NDHWC or NCDHW.
Specifies the offset.
*@li mean: A Tensor of type float32. Must be 3D if input "x" is with format NDHWC or NCDHW.
Specifies the mean used for inference. Must be "None" if the
operation is used for training.
*@li variance: A Tensor of type float32. Must be 3D if input "x" is with format NHWC or NCHW.
Specifies the variance used for inference. Must be "None"
if the operation is used for training . \n

*@par Attributes:
*@li epsilon: An optional float32, specifying the small value added to variance to avoid dividing by zero. Defaults to "0.0001".
*@li data_format: An optional string, specifying the format of "x". Defaults to "NHWC".
*@li is_training: An optional bool, specifying if the operation is used for training or inference. Defaults to "True" . \n

*@par Outputs:
* Five outputs, including: (NHWC, NCHW)
*@li y: A 3D or 6D Tensor of type float16 or float32 for the normalized "x", with format NDHWC or NCDHW.
*@li batch_mean: A Tensor of type float32. Must be 3D if input "x" is with format NDHWC or NCDHW.
Specifies the mean of "x".
*@li batch_variance: A Tensor of type float32. Must be 1D if input "x" is with format NDHWC or NCDHW.
Specifies the variance of "x".
*@li reserve_space_1: An optional Tensor of type float32. Must be 1D if input "x" is with format NDHWC or NCDHW.
Specifies the mean of "x" for gradient computation. Pass "None" to skip this output.
*@li reserve_space_2: An optional Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW.
Specifies the variance of "x" for gradient computation. Pass "None" to skip this output . \n

*@attention Constraints:
*@li If the operation is used for inference and outputs "reserve_space_1" and "reserve_space_2" are available,
then "reserve_space_1" has the same value as "mean" and "reserve_space_2" has the same value as "variance".
*@li For Ascend 310, the result accuracy fails to reach 1‰ due to the square root instruction . \n

*@par Third-party framework compatibility
*@li Compatible with the TensorFlow operator fused_batch_norm.
*@li Compatible with the TensorFlow operator fused_batch_norm_v2.
*/
REG_OP(BatchNorm3D)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_1, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_2, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001)
    .ATTR(data_format, String, "NCDHW")
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(BatchNorm3D)
/**
*@brief Performs batch normalization . \n

*@par Inputs:
* Five inputs, including: (NHWC or NCHW supported)
*@li x: A 4D Tensor of type float16 or float32.
*@li scale: A 1D Tensor of type float32, for the scaling factor.
*@li offset: A 1D Tensor of type float32, for the scaling offset.
*@li mean: A 1D Tensor of type float32, for the mean used for inference.
Must be "None" if the operation is used for training.
*@li variance: A 1D Tensor of type float32, for the variance used for inference.
Must be "None" if the operation is used for training . \n

*@par Attributes:
*@li epsilon: An optional float32, specifying the small value
added to variance to avoid dividing by zero. Defaults to "0.0001".
*@li data_format: An optional string, specifying the format of "x". Defaults to "NHWC".
*@li is_training: An optional bool, specifying if the operation
is used for training or inference. Defaults to "True" . \n

*@par Outputs:
* Five outputs, including: (NHWC or NCHW supported)
*@li y: A 4D Tensor of type float16 or float32, for the normalized "x".
*@li batch_mean: A 1D Tensor of type float32, for the mean of "x".
*@li batch_variance: A 1D Tensor of type float32, for the variance of "x".
*@li reserve_space_1: A 1D Tensor of type float32, for the mean of "x" for gradient computation.
*@li reserve_space_2: A 1D Tensor of type float32, for the variance of "x" for gradient computation . \n

*@attention Constraints:
*@li If the operation is used for inference, then output "reserve_space_1"
has the same value as "mean" and output "reserve_space_2" has the same value as "variance".
*@li For Ascend 310, the result accuracy fails to reach 1‰ due to the square root instruction . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator fused_batch_norm_v2.
*/
REG_OP(BatchNormExt2)
    .INPUT(input_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input_scale, TensorType({DT_FLOAT}))
    .INPUT(input_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(input_mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(input_variance, TensorType({DT_FLOAT}))
    .OUTPUT(output_y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(output_mean, TensorType({DT_FLOAT}))
    .OUTPUT(output_variance, TensorType({DT_FLOAT}))
    .OUTPUT(output_reserve_space_1, TensorType({DT_FLOAT}))
    .OUTPUT(output_reserve_space_2, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001)
    .ATTR(data_format, String, "NHWC")
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(BatchNormExt2)

/**
*@brief Performs the backpropagation of BatchNorm . \n

*@par Inputs:
* Five inputs, including:
*@li y_backprop: A 4D or 5D Tensor of type float16 or float32, with format NHWC, NCHW, for the gradient.
*@li x: A 4D or 5D Tensor of type float16 or float32, with format NHWC, NCHW.
*@li scale: A 4D or 5D Tensor of type float32, with format NHWC, NCHW.
*@li reserve_space_1: A 4D or 5D Tensor of type float32, with format NHWC, NCHW. It is an output of BatchNorm.
*@li reserve_space_2: A 4D or 5D Tensor of type float32, with format NHWC, NCHW. It is an output of BatchNorm .
*@li reserve_space_3: A 1D optional Tensor of type float32. It is an output of BatchNorm . \n

*@par Attributes:
*@li epsilon: An optional float32. Defaults to "0.0001". A small float number added to the variance of "x".
*@li data_format: An optional string. Defaults to "NHWC".
*@li is_training: An optional bool. Defaults to "true". Specifies the operation is for training (default) or inference . \n

*@par Outputs:
*@li x_backprop: A Tensor of type float16 or float32, with format NHWC, NCHW, for the offset of "x".
*@li scale_backprop: A Tensor of type float32, with format NHWC, NCHW, for the offset of "scale".
*@li *offset_backprop: A Tensor of type float32, with format NHWC, NCHW, for the offset of "offset".
*@li *reserve_space_4: A Tensor of type float32, with shape NHWC, NCHW. Pass "None" to skip this output.
*@li *reserve_space_5: A Tensor of type float32, with shape NHWC, NCHW. Pass "None" to skip this output . \n

*@attention Constraints:
* The preceding layer of this operator must be operator BatchNorm . \n

*@see BatchNorm
*@par Third-party framework compatibility
* Compatible with the TensorFlow operators FusedBatchNormGradV2 and FusedBatchNormGrad.
*/
REG_OP(BatchNormGrad)
    .INPUT(y_backprop, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(reserve_space_1, TensorType({DT_FLOAT}))
    .INPUT(reserve_space_2, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(reserve_space_3, TensorType({DT_FLOAT}))
    .OUTPUT(x_backprop, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(scale_backprop, TensorType({DT_FLOAT}))
    .OUTPUT(offset_backprop, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_4, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_5, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001)
    .ATTR(data_format, String, "NHWC")
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(BatchNormGrad)

/**
*@brief Performs the backpropagation of BatchNorm . \n

*@par Inputs:
* Five inputs, including:
*@li y_backprop: A 3D or 6D Tensor of type float16 or float32, with format NDHWC, NCDHW, for the gradient.
*@li x: A 3D or 6D Tensor of type float16 or float32, with format NDHWC, NCDHW.
*@li scale: A 3D or 6D Tensor of type float32, with format NDHWC, NCDHW.
*@li reserve_space_1: A 3D or 6D Tensor of type float32, with format NDHWC, NCDHW. It is an output of BatchNorm.
*@li reserve_space_2: A 3D or 6D Tensor of type float32, with format NDHWC, NCDHW. It is an output of BatchNorm . \n

*@par Attributes:
*@li epsilon: An optional float32. Defaults to "0.0001". A small float number added to the variance of "x".
*@li data_format: An optional string. Defaults to "NCDHW".
*@li is_training: An optional bool. Defaults to "true". Specifies the operation is for training (default) or inference . \n

*@par Outputs:
*@li x_backprop: A Tensor of type float16 or float32, with format NHWC, NCHW, for the offset of "x".
*@li scale_backprop: A Tensor of type float32, with format NDHWC, NCDHW, for the offset of "scale".
*@li *offset_backprop: A Tensor of type float32, with format NDHWC, NCDHW, for the offset of "offset".
*@li *reserve_space_4: A Tensor of type float32, with shape NDHWC, NCDHW. Pass "None" to skip this output.
*@li *reserve_space_5: A Tensor of type float32, with shape NDHWC, NCDHW. Pass "None" to skip this output . \n

*@attention Constraints:
* The preceding layer of this operator must be operator BatchNorm . \n

*@see BatchNorm
*@par Third-party framework compatibility
* Compatible with the TensorFlow operators FusedBatchNormGradV2 and FusedBatchNorm3DGrad.
*/
REG_OP(BatchNorm3DGrad)
    .INPUT(y_backprop, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(reserve_space_1, TensorType({DT_FLOAT}))
    .INPUT(reserve_space_2, TensorType({DT_FLOAT}))
    .OUTPUT(x_backprop, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(scale_backprop, TensorType({DT_FLOAT}))
    .OUTPUT(offset_backprop, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_4, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_5, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001)
    .ATTR(data_format, String, "NCDHW")
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(BatchNorm3DGrad)

/**
*@brief Performs the backpropagation of BatchNorm . \n

*@par Inputs:
* Five inputs, including:
*@li y_backprop: A 4D Tensor of type float16 or float32, with format NHWC or NCHW, for the gradient.
*@li x: A 4D Tensor of type float16 or float32, with format NHWC or NCHW.
*@li scale: A 4D Tensor of type float32, with format NHWC or NCHW.
*@li reserve_space_1: A 4D Tensor of type float32, with format NHWC or NCHW. It is an output of BatchNormExt2.
*@li reserve_space_2: A 4D Tensor of type float32, with format NHWC or NCHW. It is an output of BatchNormExt2 . \n

*@par Attributes:
*@li epsilon: A required float32. A small float number added to the variance of "x".
*@li data_format: A required string for the format.
*@li is_training: A required bool for specifying the operation is for training (true) or inference (false) . \n

*@par Outputs:
*@li x_backprop: A Tensor of type float16 or float32, with format NHWC or NCHW, for the offset of "x".
*@li scale_backprop: A Tensor of type float32, with format NHWC or NCHW, for the offset of "scale".
*@li offset_backprop: A Tensor of type float32, with format NHWC or NCHW, for the offset of "offset".
*@li reserve_space_3: A Tensor of type float32, with format NHWC or NCHW.
*@li reserve_space_4: A Tensor of type float32, with format NHWC or NCHW . \n

*@attention Constraints:
* The preceding layer of this operator must be BatchNormExt2 . \n

*@see BatchNormExt2
*@par Third-party framework compatibility
* Compatible with the TensorFlow operator FusedBatchNormGradV2.
*/
REG_OP(BatchNormGradExt2)
    .INPUT(y_backprop, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(reserve_space_1, TensorType({DT_FLOAT}))
    .INPUT(reserve_space_2, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001)
    .ATTR(data_format, String, "NHWC")
    .ATTR(is_training, Bool, true)
    .OUTPUT(x_backprop, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(scale_backprop, TensorType({DT_FLOAT}))
    .OUTPUT(offset_backprop, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_3, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_4, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BatchNormGradExt2)


/**
*@brief Performs batch normalization . \n

*@par Inputs:
*@li x: A 4D or 5D Tensor of type float16 or float32, with format NHWC or NCHW.
*@li mean: A Tensor of type float32 or float16. Must be 1D if input "x"  Specifies the mean used for inference.
*@li variance: A Tensor of type float32 or float16 . Must be 1D if input "x"  Specifies the variance used for inference.
*@li momentum: A Tensor,represents the mean and the variance's scale factor
*@li scale: An optional tensor of type float16 or float32, no use
*@li offset: An optional tensor of type float16 or float32, no use
*@par Attributes:
*@li epsilon: An optional float32, specifying the small value added to variance to avoid dividing by zero. Defaults to "0.00001".
*@li use_global_stats: mean inference mode , only can be "True".
*@li mode: An optional input, not use
*@par Outputs:
*@li y: A 4D or 5D Tensor of type float16 or float32 for the normalized "x"
*/
REG_OP(BNInference)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(variance, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(momentum, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .ATTR(epsilon, Float,1e-5f)
    .ATTR(use_global_stats, Bool,true)
    .ATTR(mode, Int,1)
    .OP_END_FACTORY_REG(BNInference)

/**
*@brief Performs batch normalization .

*@par Inputs:
*@li x: A 4D or 5D Tensor of type float16 or float32, with format NHWC or NCHW.
*@li mean: A Tensor of type float32 or float16. Must be 1D if input "x"
* Specifies the mean used for inference.
*@li variance: A Tensor of type float32 or float16 . Must be 1D if input "x"
* Specifies the variance used for inference.
*@li scale: An optional tensor of type float16 or float32, no use.
*@li offset: An optional tensor of type float16 or float32, no use. \n

*@par Attributes:
*@li momentum: An optional float32 num, represents the mean and
* the variance's scale factor.
*@li epsilon: An optional float32, specifying the small value
* added to variance to avoid dividing by zero. Defaults to "0.00001".
*@li use_global_stats: mean inference mode , only can be "True".
*@li mode: An optional attr, not use. \n

*@par Outputs:
*@li y: A 4D or 5D Tensor of type float16 or float32 for the normalized "x". \n

*@par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use BNInference instead.
*/
REG_OP(BNInferenceD)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(variance, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OPTIONAL_INPUT(b, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .ATTR(momentum, Float,0.9)
    .ATTR(epsilon, Float,1e-5f)
    .ATTR(use_global_stats, Bool,true)
    .ATTR(mode, Int,1)
    .OP_END_FACTORY_REG(BNInferenceD)

}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_BATCH_NORM_OPS_H_
