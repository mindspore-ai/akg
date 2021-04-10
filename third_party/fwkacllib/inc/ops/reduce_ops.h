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
 * \file reduce_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_REDUCE_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_REDUCE_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Performs reduced batch normalization . \n

*@par Inputs:
*x: A 5D Tensor of type float16 or float32, with format NC1HWC0 . \n

*@par Outputs:
*@li sum: A 1D Tensor of type float32 for SUM reduced "x".
*@li square_sum: A 1D Tensor of type float32 for SUMSQ reduced "x" . \n

*@attention Constraints:
* This operator is a BatchNorm fusion operator for updating the moving
* averages for training.
* This operator is used in conjunction with BNTrainingUpdate.
*/
REG_OP(BNTrainingReduce)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(sum, TensorType({DT_FLOAT}))
    .OUTPUT(square_sum, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BNTrainingReduce)

/**
*@brief Performs the backpropagation of BatchNorm . \n

*@par Inputs:
* Seven inputs, including:
*@li grads: A 5D Tensor of type float16 or float32, with format NC1HWC0, for
* the gradient.
*@li x: A 5D Tensor of type float16 or float32, with format NC1HWC0.
*@li diff_scale: A 5D Tensor of type float32, with format NC1HWC0,
* for the mean of "x".
*@li diff_offset: A 5D Tensor of type float32, with format NC1HWC0,
* for the variance of "x".
*@li scale: A 5D Tensor of type float32, with format NC1HWC0.
*@li batch_mean: A 5D Tensor of type float32, with format NC1HWC0,
* for the mean of "x".
*@li batch_variance: A 5D Tensor of type float32, with format NC1HWC0,
* for the variance of "x" . \n

*@par Attributes:
*epsilon: An optional float32. Defaults to "0.0001". A small float number
* added to the variance of "x" . \n

*@par Outputs:
*y: A Tensor of type float16 or float32, with format NC1HWC0, for the offset
* of "x" . \n

*@attention Constraints:
* The preceding layer of this operator must be BNTrainingUpdateGrad . \n

*@see BNTrainingUpdateGrad
*/
REG_OP(BNTrainingReduceGrad)
    .INPUT(grads, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(diff_scale, TensorType({DT_FLOAT}))
    .INPUT(diff_offset, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(batch_mean, TensorType({DT_FLOAT}))
    .INPUT(batch_variance, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001)
    .OP_END_FACTORY_REG(BNTrainingReduceGrad)

/**
*@brief Performs reduced batch normalization . \n

*@par Inputs:
* Seven inputs, including: (NC1HWC0 supported)
*@li x: A 5D Tensor of type float16 or float32.
*@li sum: A 1D Tensor of type float32 for the output of operator
* BNTrainingReduce.
*@li square_sum: A 1D Tensor of type float32 for the output of operator
* BNTrainingReduce.
*@li scale: A 1D Tensor of type float32, for the scaling factor.
*@li offset: A 1D Tensor of type float32, for the scaling offset.
*@li mean: A 1D Tensor of type float32, for the updated mean.
*@li variance: A 1D Tensor of type float32, for the updated variance . \n

*@par Attributes:
*@li epsilon: A required float32, specifying the small value added to variance
* to avoid dividing by zero.
*@li factor: A required float32, specifying the weight for updating the mean
* and variance . \n

*@par Outputs:
* Five outputs, including: (NC1HWC0 supported)
*@li y: A 5D Tensor of type float16 or float32, for normalized "x".
*@li mean: A 5D Tensor of type float32, for the updated mean.
*@li variance: A 5D Tensor of type float32, for the updated variance.
*@li batch_mean: A 1D Tensor of type float32, for the mean of "x".
*@li batch_variance: A 1D Tensor of type float32, for the variance of "x" . \n

*@attention Constraints:
*@li This operator is a BatchNorm fusion operator for updating the moving
averages for training.
*This operator is used in conjunction with BNTrainingReduce.
*@li For Ascend 310, the result accuracy fails to reach 1‰ due to the square
* root instruction.
*/
REG_OP(BNTrainingUpdate)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(sum, TensorType({DT_FLOAT}))
    .INPUT(square_sum, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT}))
    .INPUT(variance, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(factor, Float)
    .REQUIRED_ATTR(epsilon, Float)
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(mean, TensorType({DT_FLOAT}))
    .OUTPUT(variance, TensorType({DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BNTrainingUpdate)

/**
*@brief Performs batch normalization for inference . \n

*@par Inputs:
* Five inputs, including: (NC1HWC0 supported)
*@li x: A 5D Tensor of type float16 or float32.
*@li scale: A 5D Tensor of type float32, for the scaling factor.
*@li offset: A 5D Tensor of type float32, for the scaling offset.
*@li mean: A 5D Tensor of type float32, for the mean.
*@li variance: A 5D Tensor of type float32, for the variance . \n

*@par Attributes:
*epsilon: An optional float32, specifying the small value added to variance to
* avoid dividing by zero. Defaults to "0.0001" . \n

*@par Outputs:
*y: A 5D Tensor of type float16 or float32 for the normalized "x" . \n

*@attention Constraints:
*For Ascend 310, the result accuracy fails to reach 1‰ due to the square root
* instruction.
*/
REG_OP(BNInfer)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT}))
    .INPUT(variance, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(epsilon, Float)
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(BNInfer)

/**
*@brief Performs reduced batch normalization. For some scene which don't contain
assignmoving average . \n

*@par Inputs:
*Five inputs, including: (NC1HWC0 supported)
*@li x: A 5D Tensor of type float16 or float32.
*@li sum: A 5D Tensor of type float32 for the output of operator BNTrainingReduce.
*@li square_sum: A 5D Tensor of type float32 for the output of operator BNTrainingReduce.
*@li scale: A 5D Tensor of type float32, for the scaling factor.
*@li offset: A 5D Tensor of type float32, for the scaling offset . \n

*@par Attributes:
*epsilon: A required float32, specifying the small value added to variance to avoid dividing by zero . \n

*@par Outputs:
*Three outputs, including: (NC1HWC0 supported)
*@li y: A 5D Tensor of type float16 or float32, for normalized "x".
*@li batch_mean: A 5D Tensor of type float32, for the mean of "x".
*@li batch_variance: A 5D Tensor of type float32, for the variance of "x" . \n

*@attention Constraints:
*This operator is used in conjunction with BNTrainingReduce.
For Ascend 310, the result accuracy fails to reach 1‰ due to the square root instruction.
*/
REG_OP(BNTrainingUpdateV2)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(sum, TensorType({DT_FLOAT}))
    .INPUT(square_sum, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(epsilon, Float)
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BNTrainingUpdateV2)

/**
*@brief Performs reduced batch normalization v3. For some scene which don't contain
assign moving average . \n

*@par Inputs:
* Five inputs, including: (NC1HWC0 supported)
*@li x: A 5D Tensor of type float16 or float32.
*@li sum: A 5D Tensor of type float32 for the output of operator BNTrainingReduce.
*@li square_sum: A 5D Tensor of type float32 for the output of operator BNTrainingReduce.
*@li scale: A 5D Tensor of type float32, for the scaling factor.
*@li offset: A 5D Tensor of type float32, for the scaling offset . \n

*@par Attributes:
*epsilon: A required float32, specifying the small value added to variance to avoid dividing by zero . \n

*@par Outputs:
*@li y: A 5D Tensor of type float16 or float32, for normalized "x".
*@li batch_mean: A 5D Tensor of type float32, for the mean of "x".
*@li batch_variance: A 5D Tensor of type float32, for the variance of "x".
*@li reserve_1: A 5D Tensor of type float32, for the mean of batch "x". Has the same type as batch_mean.
*@li reserve_2: A 5D Tensor of type float32, for the variance of batch "x". Has the same type as batch_mean . \n

*@attention Constraints:
*@li This operator is used in conjunction with BNTrainingReduce.
*@li For Ascend 310, the result accuracy fails to reach 1‰ due to the square root instruction.
*/
REG_OP(BNTrainingUpdateV3)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(sum, TensorType({DT_FLOAT}))
    .INPUT(square_sum, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(epsilon, Float)
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_1, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_2, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BNTrainingUpdateV3)

/**
*@brief Performs the backpropagation of BatchNorm . \n

*@par Inputs:
* Four inputs, including:
*@li grads: A 5D Tensor of type float16 or float32, with format NC1HWC0,
* for the gradient.
*@li x: A 5D Tensor of type float16 or float32, with format NC1HWC0.
*@li batch_mean: A 5D Tensor of type float32, with format NC1HWC0,
* for the mean of "x".
*@li batch_variance: A 5D Tensor of type float32, with format NC1HWC0,
* for the variance of "x" . \n

*@par Attributes:
*epsilon: An optional float32. Defaults to "0.0001". A small float number
* added to the variance of "x" . \n

*@par Outputs:
*@li diff_scale: A Tensor of type float32, with format NC1HWC0,
* for the offset of "scale".
*@li diff_offset: A Tensor of type float32, with format NC1HWC0,
* for the offset of "offset" . \n

*/
REG_OP(BNTrainingUpdateGrad)
    .INPUT(grads, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(batch_mean, TensorType({DT_FLOAT}))
    .INPUT(batch_variance, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001)
    .OUTPUT(diff_scale, TensorType({DT_FLOAT}))
    .OUTPUT(diff_offset, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BNTrainingUpdateGrad)

/**
*@brief Performs the backpropagation of BatchNorm for inference . \n

*@par Inputs:
* Three inputs, including:
*@li grads: A 5D Tensor of type loat16 or float32, with format NC1HWC0, for the gradient.
*@li scale: A 5D Tensor of type float32, with format NC1HWC0.
*@li batch_variance: A 5D Tensor of type float32, with format NC1HWC0. It is an output of BatchNorm . \n

*@par Attributes:
*epsilon: An optional float32. Defaults to "0.0001". A small float number added to the variance of "x" . \n

*@par Outputs:
*x_backprop: A Tensor of type float16 or float32, with format NC1HWC0, for the offset of "x" . \n

*@attention Constraints:
* The preceding layer of this operator must be operator BatchNorm.
*/
REG_OP(BNInferGrad)
    .INPUT(grads, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(batch_variance, TensorType({DT_FLOAT}))
    .OUTPUT(x_backprop, TensorType({DT_FLOAT16,DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001)
    .OP_END_FACTORY_REG(BNInferGrad)

/**
*@brief Computes the sum of elements across dimensions of a tensor . \n

*@par Inputs:
* Two inputs, including:
*@li x: A Tensor. Must be one of the following types:
*     float32, float64, int32, uint8, int16, int8,
*     complex64, int64, qint8, quint8, qint32, uint16,
*     complex128, float16, uint32, uint64, complex64, complex128.
*@li axes: A 1D list or tuple of int32 or int64. Specifies the dimensions to reduce . \n

*@par Attributes:
*keep_dims: An optional bool. If "true", retains reduced dimensions with length 1. Defaults to "false" . \n

*@par Outputs:
*y: The reduced tensor. Has the same type and format as input "x" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Sum.
*/
REG_OP(ReduceSum)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceSum)

/**
*@brief Computes the sum of elements across dimensions of a tensor . \n

*@par Inputs:
* One input:
*x: A Tensor. Up to 8D. Must be one of the following types: float16, float32. \n

*@par Attributes:
*@li axes: A required 1D list or tuple of int32 or int64. Specifies the dimensions to reduce.
*@li keep_dims: An optional bool. If "true", retains reduced dimensions with length 1. Defaults to "false" . \n

*@par Outputs:
*y: The reduced tensor. Has the same type and format as input "x" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Sum.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ReduceSum instead.
*/
REG_OP(ReduceSumD)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(axes, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceSumD)

/**
*@brief Calculates the "logical sum" of elements of a tensor in a dimension . \n

*@par Inputs:
*One input:
*x: The boolean tensor to reduce . \n

*@par Attributes:
*@li keep_dims: A bool. If true, retains reduced dimensions with length 1.
*@li axis: The dimensions to reduce. If None, reduces all dimensions.
*Must be in the range [- rank (input_sensor), rank (input_sensor)) . \n

*@par Outputs:
*y: The reduced tensor . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ReduceAll.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ReduceAll instead.
*/
REG_OP(ReduceAllD)
    .INPUT(x, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .REQUIRED_ATTR(axes, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceAllD)

/**
*@brief Calculates the "logical sum" of elements of a tensor in a dimension . \n

*@par Inputs:
*Two inputs, including:
*@li x: The boolean tensor to reduce.
*@li axis: A mutable Tensor. The dimensions to reduce. If None, reduces all dimensions. Must be in the range [- rank (input_sensor), rank (input_sensor)) . \n

*@par Attributes:
*keep_dims: A bool. If true, retains reduced dimensions with length 1 . \n

*@par Outputs:
*y: The reduced tensor . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ReduceAll.
*/
REG_OP(ReduceAll)
    .INPUT(x, TensorType({DT_BOOL}))
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceAll)

/**
*@brief  Reduce a tensor on a certain axis based on product. . \n

*@par Inputs:
*Two inputs, including:
*@li x: A mutable Tensor. Must be the type of NumberType.
*@li axis: A mutable Tensor. The dimensions to reduce . \n

*@par Attributes:
*@li keep_dims: A bool. If true, retains reduced dimensions with length 1. Defaults to "False" . \n

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ReduceProd.
*/
REG_OP(ReduceProd)
    .INPUT(x,TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y,TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceProd)

/**
*@brief Computes the product of elements across dimensions of a tensor . \n

*@par Inputs:
* One input:
*x: A Tensor. Must be one of the following types: float16, float, int8, uint8 . \n

*@par Attributes:
*@li axes: A required int8, int16, int32, or int64. Specifies the dimensions to reduce. No default value.
*@li keep_dims: An optional bool. If "True", retains reduced dimensions with length 1. Defaults to "False" . \n

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x" . \n

*@attention Constraints:
* "keep_dims" is in the range [-rank(input_tensor), rank(input_tensor)] . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ReduceProd.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ReduceProd instead.
*/
REG_OP(ReduceProdD)
    .INPUT(x,TensorType({DT_FLOAT, DT_UINT8, DT_INT8, DT_INT32, DT_FLOAT16}))
    .OUTPUT(y,TensorType({DT_FLOAT, DT_UINT8, DT_INT8, DT_INT32, DT_FLOAT16}))
    .REQUIRED_ATTR(axes, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceProdD)

/**
*@brief Reduces "x" along the dimensions according to "axis" . \n

*@par Inputs:
*Two inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32, int8, uint8.
* @li axes: The dimensions to reduce. Must be one of the following types: int, list, tuple, NoneType.
*   - If None (the default), reduces all dimensions.
*   - Must be in the range [-rank(x), rank(x)) . \n

*@par Attributes:
*keep_dims: A bool or NoneType.
* - If true, retains reduced dimensions with length 1.
* - If false, the rank of the tensor is reduced by 1 for each entry in axis.
*@par Outputs:
*y: A Tensor. Has the same type as "x" . \n

*@par Third-party framework compatibility:
* Compatible with the TensorFlow operator ReduceMean.
*/
REG_OP(ReduceMean)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceMean)

/**
*@brief Reduces "x" along the dimensions according to "axis" . \n

*@par Inputs:
*One input:
* @li x: A Tensor. Must be one of the following types: float16, float32 . \n

*@par Attributes:
*@li axes: The dimensions to reduce. Must be one of the following types: int, list, tuple, NoneType.
* If None (the default), reduces all dimensions.
* Must be in the range [-rank(x), rank(x)).
*@li keep_dims: A bool or NoneType.
* - If true, retains reduced dimensions with length 1.
* - If false, the rank of the tensor is reduced by 1 for each entry in axis.
*@par Outputs:
*y: A Tensor. Has the same type as "x" . \n

*@par Third-party framework compatibility:
* Compatible with the TensorFlow operator ReduceMean.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ReduceMean instead.
*/
REG_OP(ReduceMeanD)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(axes, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceMeanD)

/**
*@brief Returns the maximum of elements across dimensions of a Tensor . \n

*@par Inputs:
* Two inputs, including:
*@li x: A multi-dimensional Tensor of type float16, float32, or int16.
*@li axes: A Scalar of type int32, specifying the axes information of the index with the maximum value . \n

*@par Attributes:
*keep_dims: A bool, specifying whether to keep dimensions for the output Tensor. Defaults to "false" . \n

*@par Outputs:
*y: A multi-dimensional Tensor, specifying the maximum value of the corresponding axis in the tensor. Has the same type as "x". (If "keep_dims" is set to "false", the output dimensions are reduced by "dimension" compared with that of "x". Otherwise, the output has one fewer dimension than "x".)

*@attention Constraints:
* The value range of "axes" is [-dims, dims - 1]. "dims" indicates the dimension length of "x" . \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator Max.
*/
REG_OP(ReduceMax)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceMax)

/**
*@brief Returns the maximum of elements across dimensions of a Tensor . \n

*@par Inputs:
*x: A multi-dimensional Tensor of type float16, float32, or int16 . \n

*@par Attributes:
* Two attributes, including:
*@li axes: A required listint, specifying the axes information of the index with the maximum value.
*@li keep_dims: A bool, specifying whether to keep dimensions for the output Tensor. Defaults to "false" . \n

*@par Outputs:
*y: A multi-dimensional Tensor, specifying the maximum value of the corresponding axis in the tensor. Has the same type as "x". (If "keep_dims" is set to "false", the output dimensions are reduced by "dimension" compared with that of "x". Otherwise, the output has one fewer dimension than "x".)

*@attention Constraints:
* The value range of "axis" is [-dims, dims - 1]. "dims" indicates the dimension length of "x" . \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator Max.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ReduceMax instead.
*/
REG_OP(ReduceMaxD)
    .INPUT(x, TensorType({DT_FLOAT, DT_UINT8, DT_INT8,
                          DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_UINT8, DT_INT8,
                           DT_FLOAT16, DT_INT32}))
    .REQUIRED_ATTR(axes, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceMaxD)

/**
*@brief Computes the minimum of elements across dimensions of a tensor . \n

*@par Inputs:
*@li input_tensor: A Tensor. Must be one of the following types: float16, float32, int8, uint8.
*@li axes: A Tensor of type int8 or int32. Specifies the dimensions to reduce. Defaults to "None".

*@par Attributes:
*keep_dims: An optional bool. If "True", reduced dimensions will be retained. Defaults to "False".

*@par Outputs:
*output_tensor: A Tensor. Must be one of the following types: float16, float32, int8, uint8 . \n

*@attention Constraints:
* If "axes = None", all dimensions will be reduced. "axes" must be in the range [-rank(input_shape), rank(input_shape)) . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator reduce_min.
*/
REG_OP(ReduceMin)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceMin)

/**
*@brief Computes the minimum of elements across dimensions of a tensor . \n

*@par Inputs:
*input_min: A Tensor. Must be one of the following types: float16, float32, int8, uint8 . \n

*@par Attributes:
*@li axes: An optional int32, list, tuple, or NoneType value. Specifies the dimensions to reduce. Defaults to "None".
*@li keep_dims: An optional bool or NoneType value. If "True", reduced dimensions will be retained. Defaults to "None" (equivalent to "False").

*@par Outputs:
*output_min: A Tensor. Must be one of the following types: float16, float32, int8, uint8 . \n

*@attention Constraints:
* If "axes = None", all dimensions will be reduced. "axes" must be in the range [-rank(input_shape), rank(input_shape)) . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator reduce_min.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ReduceMin instead.
*/
REG_OP(ReduceMinD)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT8,DT_UINT8,DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT8,DT_UINT8,DT_INT32}))
    .REQUIRED_ATTR(axes, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceMinD)
/**
*@brief Computes the "logical or" of elements across dimensions of a tensor.
* Reduces "x" along the dimensions given in "axes".
* Unless "keep_dims" is true, the rank of the tensor is reduced by 1 for each
* entry in "axes". If "keep_dims" is true, the reduced dimensions
* are retained with length 1.
*
* If "axes" is None, all dimensions are reduced, and a
* tensor with a single element is returned.
*
*@attention Constraints:
* Only support bool
*
*@par Inputs:
*@li x : The boolean tensor to reduce.
*@li axes: The dimensions to reduce. If "None" (default), reduces all
*          dimensions. Must be in the range "[-rank(x), rank(x))".
*
*@par Attributes:
* keep_dims: If true, retains reduced dimensions with length 1.
*
*@par Outputs:
* y: The reduced tensor
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator reduce_any.
*
*/
REG_OP(ReduceAny)
    .INPUT(x, TensorType({DT_BOOL}))
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceAny)
/**
*@brief Computes the "logical or" of elements across dimensions of a tensor.
* Reduces "x" along the dimensions given in "axes".
* Unless "keep_dims" is true, the rank of the tensor is reduced by 1 for each
* entry in "axes". If "keep_dims" is true, the reduced dimensions
* are retained with length 1.
*
* If "axis" is None, all dimensions are reduced, and a
* tensor with a single element is returned.
*
*@attention Constraints:
*  Only support bool
*
*@par Inputs:
* x: The boolean tensor to reduce.
*
*@par Attributes:
*@li axes: The dimensions to reduce. Must be in the range "[-rank(x), rank(x))".
*@li keep_dims: If true, retains reduced dimensions with length 1.
*
*@par Outputs:
* y: The reduced tensor
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator reduce_any.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ReduceAny instead.
*/
REG_OP(ReduceAnyD)
    .INPUT(x, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .REQUIRED_ATTR(axes, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceAnyD)

/**
*@brief Compute reduction on dimensions specified by "axis".
*Four reduction operations are provided:
*SUM     Computes the sum of elements across specified dimensions of a tensor.
*ASUM    Computes the sum of absolute values of elements across specified dimensions of a tensor.
*SUMSQ   Computes the sum of squares of elements across specified dimensions of a tensor.
*SUMSQ   Computes the mean values of elements across specified dimensions of a tensor . \n

*@par Inputs:
*x: A Tensor of type float16 or float32

*@par Attributes:
*@li operation: An optional int32 from 1(SUM), 2(ASUM), 3(SUMSQ), and 4(MEAN),
*specifying the reduction algorithm. Defaults to "1".
*@li axis: An optional int32, specifying the first axis to reduce. Defaults to "0".
*The value range is [-N, N-1], where N is the input tensor rank.
*@li coeff: An optional float32, specifying the scale coefficient. Defaults to "1.0" . \n

*@par Outputs:
*y: A Tensor. Has the same type as "x" . \n

*@attention Constraints: The Reduction operator supports type float16 only on the device chip.
*@par Third-party framework compatibility
* Compatible with the Caffe operator Reduction.
*/
REG_OP(Reduction)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(operation, Int, 1)
    .ATTR(axis, Int, 0)
    .ATTR(coeff, Float, 1.0)
    .OP_END_FACTORY_REG(Reduction);

/**
*@brief Computes the euclidean norm of elements across dimensions of a tensor . \n

*@par Inputs:
*@li input_tensor: A Tensor. Must be one of the following types: float16, float32, int32.
*@li axes: A Tensor of type int8 or int32. Specifies the dimensions to reduce. Defaults to "None" . \n

*@par Attributes:
*keep_dims: An optional bool. If "True", reduced dimensions will be retained. Defaults to "False" . \n

*@par Outputs:
*output_tensor: A Tensor. Must be one of the following types: float16, float32, int32 . \n

*@attention Constraints:
* If "axes = None", all dimensions will be reduced. "axes" must be in the range [-rank(input_shape), rank(input_shape)) . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator EuclideanNorm.
*/
REG_OP(EuclideanNorm)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(EuclideanNorm)

/**
*@brief Computes the euclidean norm of elements across dimensions of a tensor . \n

*@par Inputs:
*input_min: A Tensor. Must be one of the following types: float16, float32, int32 . \n

*@par Attributes:
*@li axes: An optional int32, list, tuple, or NoneType value. Specifies the dimensions to reduce. Defaults to "None".
*@li keep_dims: An optional bool or NoneType value. If "True", reduced dimensions will be retained. Defaults to "None" (equivalent to "False") . \n

*@par Outputs:
*output_min: A Tensor. Must be one of the following types: float16, float32, int32 . \n

*@attention Constraints:
* If "axes = None", all dimensions will be reduced. "axes" must be in the range [-rank(input_shape), rank(input_shape)) . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator EuclideanNorm.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use EuclideanNorm instead.
*/
REG_OP(EuclideanNormD)
    .INPUT(x, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16}))
    .ATTR(axes, ListInt, {})
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(EuclideanNormD)



/**
*@brief Performs instance normalization for inference . \n

*@par Inputs:
* Five inputs, including: (NC1HWC0 supported)
*@li x: A Tensor of type float16 or float32.
*@li gamma: A [N, C1, 1, 1, C0] Tensor of type float32, for the scaling gamma.
*@li beta: A [N, C1, 1, 1, C0] Tensor of type float32, for the scaling beta.
*@li mean: A [N, C1, 1, 1, C0] ensor of type float32, for the mean.
*@li variance: A [N, C1, 1, 1, C0] Tensor of type float32, for the variance . \n

*@par Attributes:
*epsilon: An optional float32, specifying the small value added to variance to avoid dividing by zero.
Defaults to "0.00001" . \n

*@par Outputs:
*y: A Tensor of type float16 or float32 for the normalized "x".
*batch_mean: A Tensor of type float32 for the result mean.
*batch_ variance: A Tensor of type float32 for the result variance . \n

*@attention Constraints:
*For Ascend 310, the result accuracy fails to reach 0.001 due to the square root instruction.
*/
REG_OP(INInferV2)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OPTIONAL_INPUT(gamma, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(beta, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.00001)
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(INInferV2)

/**
*@brief Performs reduced instance normalization . \n

*@par Inputs:
*x: A Tensor of type float16 or float32, with format NC1HWC0 . \n

*@par Outputs:
*@li sum: A Tensor of type float32 for SUM reduced "x".
*@li square_sum: A Tensor of type float32 for SUMSQ reduced "x" . \n

*@attention Constraints:
* This operator is a InstanceNorm fusion operator for updating the moving averages for training.
* This operator is used in conjunction with INTrainingUpdateV2.
*/
REG_OP(INTrainingReduceV2)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(sum, TensorType({DT_FLOAT}))
    .OUTPUT(square_sum, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(INTrainingReduceV2)


/**
*@brief Performs update instance normalization . \n

*@par Inputs:
* Seven inputs, including: (NC1HWC0supported)
*@li x: A Tensor of type float16 or float32.
*@li sum: A T [N, C1, 1, 1, C0] ensor of type float32 for the output of operator INTrainingReduceV2.
*@li square_sum: A  [N, C1, 1, 1, C0] Tensor of type float32 for the output of operator INTrainingReduceV2.
*@li gamma: A  [N, C1, 1, 1, C0] Tensor of type float32, for the scaling gamma.
*@li beta: A  [N, C1, 1, 1, C0] Tensor of type float32, for the scaling beta.
*@li mean: A  [N, C1, 1, 1, C0] Tensor of type float32, for the updated mean.
*@li variance: A  [N, C1, 1, 1, C0] Tensor of type float32, for the updated variance . \n

*@par Attributes:
*@li momentum: A required float32, specifying the momentum to update mean and var.
*@li epsilon: A required float32, specifying the small value added to variance to avoid dividing by zero . \n

*@par Outputs:
* Three outputs, including: (NC1HWC0 supported)
*@li y: A Tensor of type float16 or float32, for normalized "x".
*@li batch_mean: A Tensor of type float32, for the updated mean.
*@li batch_variance: A Tensor of type float32, for the updated variance . \n

*@attention Constraints:
*@li This operator is a InstanceNorm fusion operator for updating the moving averages for training.
* This operator is used in conjunction with INTrainingReduceV2.
*@li For Ascend 310, the result accuracy fails to reach 1‰ due to the square root instruction.
*/
REG_OP(INTrainingUpdateV2)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(sum, TensorType({DT_FLOAT}))
    .INPUT(square_sum, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(gamma, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(beta, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))
    .ATTR(momentum, Float, 0.1)
    .ATTR(epsilon, Float, 0.00001)
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(INTrainingUpdateV2)


/**
*@brief Performs reduced group normalization . \n

*@par Inputs:
*x: A Tensor of type float16 or float32, with format NCHW NHWC . \n

*@par Outputs:
*@li sum: A Tensor of type float32 for SUM reduced "x".
*@li square_sum: A Tensor of type float32 for SUMSQ reduced "x".


*@par Attributes:
*@li num_groups: Int, specifying the num of groups. required, same to GNTrainingUpdate . \n

*@attention Constraints:
* This operator is a GroupNorm fusion operator for updating the moving averages for training.
* This operator is used in conjunction with GNTrainingUpdate.
*/
REG_OP(GNTrainingReduce)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(sum, TensorType({DT_FLOAT}))
    .OUTPUT(square_sum, TensorType({DT_FLOAT}))
    .ATTR(num_groups, Int, 2)
    .OP_END_FACTORY_REG(GNTrainingReduce)


/**
*@brief Performs update group normalization . \n

*@par Inputs:
* Eight inputs, including: (NCHW NHWC supported)
*@li x: A Tensor of type float16 or float32.
*@li sum: A 5D Tensor of type float32,
shape is [N, G, 1, 1, 1] for NCHW, [N, 1, 1, G, 1] for NHWC
for the output of operator GNTrainingReduce.
*@li square_sum: A 5D Tensor of type float32,
shape is [N, G, 1, 1, 1] for NCHW, [N, 1, 1, G, 1] for NHWC
for the output of operator GNTrainingReduce.
*@li scale: A 5D Tensor of type float32,
shape is [1, G, 1, 1, 1] for NCHW, [1, 1, 1, G, 1] for NHWC
is for the scaling gamma.
*@li offset: A 5D Tensor of type float32,
shape is [1, G, 1, 1, 1] for NCHW, [1, 1, 1, G, 1] for NHWC
for the scaling beta.
*@li mean: A 5D Tensor of type float32,
shape is [N, G, 1, 1, 1] for NCHW, [N, 1, 1, G, 1] for NHWC
for the updated mean.
*@li variance: A 5D Tensor of type float32,
shape is [N, G, 1, 1, 1] for NCHW, [N, 1, 1, G, 1] for NHWC
for the updated variance.


*@par Attributes:
*@li epsilon: A float32, specifying the small value added to variance to avoid dividing by zero.
*@li num_groups: Int, specifying the num of groups. required, same to GNTrainingReduce

*@par Outputs:
* Three outputs, including: (NC1HWC0 supported)
*@li y: A Tensor of type float16 or float32, for normalized "x".
*@li batch_mean: A Tensor of type float32, for the updated mean.
*@li batch_variance: A Tensor of type float32, for the updated variance . \n

*@attention Constraints:
*@li This operator is a InstanceNorm fusion operator for updating the moving averages for training.
* This operator is used in conjunction with GNTrainingUpdate.
*@li For Ascend 310, the result accuracy fails to reach 1‰ due to the square root instruction.
*/
REG_OP(GNTrainingUpdate)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(sum, TensorType({DT_FLOAT}))
    .INPUT(square_sum, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))
    .ATTR(num_groups, Int, 2)
    .ATTR(epsilon, Float, 0.0001)
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(GNTrainingUpdate)

/**
* @brief Calculates the standard deviation and average value of Tensors.

* @par Inputs:
* @li x: A Tensor. Must be one of the following types:
*     float16, float32. \n

* @par Attributes:
* Three Attributes, including:
* @li dim: An optional listint, Defaults to "None". \n

* @li unbiased: An optional bool. Defaults to "True".
*     If "True", Use Bessel Correction.
*     If "False", Do not use Bessel Correction. \n

* @li keepdim: An optional bool. Defaults to "False".
*     If "True", Keep the original tensor dimension.
*     If "False", Do not keep the original tensor dimension. \n

* @par Outputs:
* Two Outputs, including:
* @li y1: A Tensor. Has the same type as "x".
* @li y2: A Tensor. Has the same type as "x". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator ReduceStd.
*/
REG_OP(ReduceStd)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y1, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y2, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(dim, ListInt, {})
    .ATTR(unbiased, Bool, true)
    .ATTR(keepdim, Bool, false)
    .OP_END_FACTORY_REG(ReduceStd)
} //namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_REDUCE_OPS_H_
