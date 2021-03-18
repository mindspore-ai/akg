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

#ifndef GE_OP_REDUCE_OPS_H
#define GE_OP_REDUCE_OPS_H

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Performs reduced batch normalization.

*@par Inputs:\n
*x: A 5D Tensor of type float16 or float32, with format NC1HWC0.

*@par Outputs:
*@li sum: A 1D Tensor of type float32 for SUM reduced "x".
*@li square_sum: A 1D Tensor of type float32 for SUMSQ reduced "x".

*@attention Constraints:\n
* This operator is a BatchNorm fusion operator for updating the moving averages for training. \n This operator is used in conjunction with BNTrainingUpdate.
*/
REG_OP(BNTrainingReduce)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(sum, TensorType({DT_FLOAT}))
    .OUTPUT(square_sum, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BNTrainingReduce)

/**
*@brief Performs the backpropagation of BatchNorm.

*@par Inputs:
* Seven inputs, including: \n
*@li grads: A 5D Tensor of type float16 or float32, with format NC1HWC0, for the gradient.
*@li x: A 5D Tensor of type float16 or float32, with format NC1HWC0.
*@li diff_scale: A 5D Tensor of type float32, with format NC1HWC0, for the mean of "x".
*@li diff_offset: A 5D Tensor of type float32, with format NC1HWC0, for the variance of "x".
*@li scale: A 5D Tensor of type float32, with format NC1HWC0.
*@li batch_mean: A 5D Tensor of type float32, with format NC1HWC0, for the mean of "x".
*@li batch_variance: A 5D Tensor of type float32, with format NC1HWC0, for the variance of "x".

*@par Attributes:
*epsilon: An optional float32. Defaults to "0.0001". A small float number added to the variance of "x".

*@par Outputs:
*y: A Tensor of type float16 or float32, with format NC1HWC0, for the offset of "x".

*@attention Constraints:
* The preceding layer of this operator must be BNTrainingUpdateGrad.

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
*@brief Performs reduced batch normalization.

*@par Inputs:\n
* Seven inputs, including: (NC1HWC0 supported)
*@li x: A 5D Tensor of type float16 or float32.
*@li sum: A 1D Tensor of type float32 for the output of operator BNTrainingReduce.
*@li square_sum: A 1D Tensor of type float32 for the output of operator BNTrainingReduce.
*@li scale: A 1D Tensor of type float32, for the scaling factor.
*@li offset: A 1D Tensor of type float32, for the scaling offset.
*@li mean: A 1D Tensor of type float32, for the updated mean.
*@li variance: A 1D Tensor of type float32, for the updated variance.

*@par Attributes:
*@li epsilon: A required float32, specifying the small value added to variance to avoid dividing by zero.
*@li factor: A required float32, specifying the weight for updating the mean and variance.

*@par Outputs:\n
* Five outputs, including: (NC1HWC0 supported)
*@li y: A 5D Tensor of type float16 or float32, for normalized "x".
*@li mean: A 5D Tensor of type float32, for the updated mean.
*@li variance: A 5D Tensor of type float32, for the updated variance.
*@li batch_mean: A 1D Tensor of type float32, for the mean of "x".
*@li batch_variance: A 1D Tensor of type float32, for the variance of "x".

*@attention Constraints:
*@li This operator is a BatchNorm fusion operator for updating the moving averages for training. \n This operator is used in conjunction with BNTrainingReduce.
*@li For Ascend 310, the result accuracy fails to reach 1‰ due to the square root instruction.
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
*@brief Performs batch normalization for inference.

*@par Inputs:\n
* Five inputs, including: (NC1HWC0 supported)
*@li x: A 5D Tensor of type float16 or float32.
*@li scale: A 5D Tensor of type float32, for the scaling factor.
*@li offset: A 5D Tensor of type float32, for the scaling offset.
*@li mean: A 5D Tensor of type float32, for the mean.
*@li variance: A 5D Tensor of type float32, for the variance.

*@par Attributes:
*epsilon: An optional float32, specifying the small value added to variance to avoid dividing by zero. Defaults to "0.0001".

*@par Outputs:\n
*y: A 5D Tensor of type float16 or float32 for the normalized "x".

*@attention Constraints:
*For Ascend 310, the result accuracy fails to reach 1‰ due to the square root instruction.
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
assignmoving average.

*@par Inputs:\n
* Five inputs, including: (NC1HWC0 supported)
*@li x: A 5D Tensor of type float16 or float32.
*@li sum: A 5D Tensor of type float32 for the output of operator BNTrainingReduce.
*@li square_sum: A 5D Tensor of type float32 for the output of operator BNTrainingReduce.
*@li scale: A 5D Tensor of type float32, for the scaling factor.
*@li offset: A 5D Tensor of type float32, for the scaling offset.

*@par Attributes:
*epsilon: A required float32, specifying the small value added to variance to avoid dividing by zero.

*@par Outputs:\n
* Three outputs, including: (NC1HWC0 supported)
*@li y: A 5D Tensor of type float16 or float32, for normalized "x".
*@li batch_mean: A 5D Tensor of type float32, for the mean of "x".
*@li batch_variance: A 5D Tensor of type float32, for the variance of "x".

*@attention Constraints:
*@li This operator is used in conjunction with BNTrainingReduce.
*@li For Ascend 310, the result accuracy fails to reach 1‰ due to the square root instruction.
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
*@brief Performs the backpropagation of BatchNorm.

*@par Inputs:
* Four inputs, including: \n
*@li grads: A 5D Tensor of type float16 or float32, with format NC1HWC0, for the gradient.
*@li x: A 5D Tensor of type float16 or float32, with format NC1HWC0.
*@li batch_mean: A 5D Tensor of type float32, with format NC1HWC0, for the mean of "x".
*@li batch_variance: A 5D Tensor of type float32, with format NC1HWC0, for the variance of "x".

*@par Attributes:
*epsilon: An optional float32. Defaults to "0.0001". A small float number added to the variance of "x".

*@par Outputs:
*@li diff_scale: A Tensor of type float32, with format NC1HWC0, for the offset of "scale".
*@li diff_offset: A Tensor of type float32, with format NC1HWC0, for the offset of "offset".

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
*@brief Performs the backpropagation of BatchNorm for inference.

*@par Inputs:
* Three inputs, including: \n
*@li grads: A 5D Tensor of type loat16 or float32, with format NC1HWC0, for the gradient.
*@li scale: A 5D Tensor of type float32, with format NC1HWC0.
*@li batch_variance: A 5D Tensor of type float32, with format NC1HWC0. It is an output of BatchNorm.

*@par Attributes:
*epsilon: An optional float32. Defaults to "0.0001". A small float number added to the variance of "x".

*@par Outputs:
*x_backprop: A Tensor of type float16 or float32, with format NC1HWC0, for the offset of "x".

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
*@brief Computes the sum of elements across dimensions of a tensor.

*@par Inputs:
* Two inputs, including: \n
*@li x: A Tensor of type float16 or float32. Up to 8D.
*@li axes: A 1D list or tuple of int32 or int64. Specifies the dimensions to reduce.

*@par Attributes:
*keep_dims: An optional bool. If "true", retains reduced dimensions with length 1. Defaults to "false".

*@par Outputs:
*y: The reduced tensor. Has the same type and format as input "x".

*/
REG_OP(ReduceSum)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceSum)

/**
*@brief Computes the sum of elements across dimensions of a tensor.

*@par Inputs:
* One input: \n
*x: A Tensor. Up to 8D. Must be one of the following types: float16, float32, int32, int8, uint8.

*@par Attributes:
*@li axes: A required 1D list or tuple of int32 or int64. Specifies the dimensions to reduce.
*@li keep_dims: An optional bool. If "true", retains reduced dimensions with length 1. Defaults to "false".

*@par Outputs:
*y: The reduced tensor. Has the same type and format as input "x".

*/
REG_OP(ReduceSumD)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_UINT8, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_UINT8, DT_INT32}))
    .REQUIRED_ATTR(axes, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceSumD)

/**
*@brief Calculates the "logical sum" of elements of a tensor in a dimension.

*@par Inputs:
*One input:
*x: A mutable Tensor. Must be one of the following types: float16,
* float32, double. Should be a Variable Tensor.

*@par Attributes:
*@li keep_dims: A bool. If true, retains reduced dimensions with length 1.
*@li axis: The dimensions to reduce. If None, reduces all dimensions.
*Must be in the range [- rank (input_sensor), rank (input_sensor)).

*@par Outputs:
*y: The reduced tensor.
*/
REG_OP(ReduceAllD)
    .INPUT(x, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .REQUIRED_ATTR(axes, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceAllD)

/**
*@brief Calculates the "logical sum" of elements of a tensor in a dimension.

*@par Inputs:
*Two inputs, including:
*@li x: A mutable Tensor. Must be one of the following types: float16, float32, double. Should be a Variable Tensor.
*@li axis: A mutable Tensor. The dimensions to reduce. If None, reduces all dimensions. Must be in the range [- rank (input_sensor), rank (input_sensor)).

*@par Attributes:
*keep_dims: A bool. If true, retains reduced dimensions with length 1.

*@par Outputs:
*y: The reduced tensor.
*/
REG_OP(ReduceAll)
    .INPUT(x, TensorType({DT_BOOL}))
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceAll)

/**
*@brief  Reduce a tensor on a certain axis based on product..

*@par Inputs:
*Two inputs, including:
*@li x: A mutable Tensor. Must be the type of NumberType.
*@li axis: A mutable Tensor. The dimensions to reduce.

*@par Attributes:
*@li keep_dims: A bool. If true, retains reduced dimensions with length 1. Defaults to "False".

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x".

*/
REG_OP(ReduceProd)
    .INPUT(x,TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y,TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceProd)

/**
*@brief Computes the product of elements across dimensions of a tensor.

*@par Inputs:
* One input: \n
*x: A Tensor. Must be one of the following types: float16, float, int8, uint8.

*@par Attributes:
*@li axes: A required int8, int16, int32, or int64. Specifies the dimensions to reduce. No default value.
*@li keep_dims: An optional bool. If "True", retains reduced dimensions with length 1. Defaults to "False".

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x".

*@attention Constraints:
* "keep_dims" is in the range [-rank(input_tensor), rank(input_tensor)].

*/
REG_OP(ReduceProdD)
    .INPUT(x,TensorType({DT_FLOAT, DT_UINT8, DT_INT8, DT_INT32, DT_FLOAT16}))
    .OUTPUT(y,TensorType({DT_FLOAT, DT_UINT8, DT_INT8, DT_INT32, DT_FLOAT16}))
    .REQUIRED_ATTR(axes, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceProdD)

/**
*@brief Reduces "x" along the dimensions according to "axis".

*@par Inputs:
*Two inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32, int8, uint8.
* @li axes: The dimensions to reduce. Must be one of the following types: int, list, tuple, NoneType.\n
*   - If None (the default), reduces all dimensions.\n
*   - Must be in the range [-rank(x), rank(x)).

*@par Attributes:
*keep_dims: A bool or NoneType. \n
* - If true, retains reduced dimensions with length 1. \n
* - If false, the rank of the tensor is reduced by 1 for each entry in axis.
*@par Outputs:
*y: A Tensor. Has the same type as "x".
*/
REG_OP(ReduceMean)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceMean)

/**
*@brief Reduces "x" along the dimensions according to "axis".

*@par Inputs:
*One input:
* @li x: A Tensor. Must be one of the following types: float16, float32, int8, uint8.

*@par Attributes:
*@li axes: The dimensions to reduce. Must be one of the following types: int, list, tuple, NoneType. \n
* If None (the default), reduces all dimensions. \n
* Must be in the range [-rank(x), rank(x)). \n
*@li keep_dims: A bool or NoneType. \n
* - If true, retains reduced dimensions with length 1. \n
* - If false, the rank of the tensor is reduced by 1 for each entry in axis.
*@par Outputs:
*y: A Tensor. Has the same type as "x".
*/
REG_OP(ReduceMeanD)
    .INPUT(x, TensorType({DT_FLOAT16, DT_INT32, DT_FLOAT, DT_INT8, DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT32, DT_FLOAT, DT_INT8, DT_UINT8}))
    .REQUIRED_ATTR(axes, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceMeanD)

/**
*@brief Returns the maximum of elements across dimensions of a Tensor.

*@par Inputs:
* Two inputs, including: \n
*@li x: A multi-dimensional Tensor of type float16, float32, or int16.
*@li axes: A Scalar of type int32, specifying the axes information of the index with the maximum value.

*@par Attributes:
*keep_dims: A bool, specifying whether to keep dimensions for the output Tensor. Defaults to "false".

*@par Outputs:
*y: A multi-dimensional Tensor, specifying the maximum value of the corresponding axis in the tensor. Has the same type as "x". (If "keep_dims" is set to "false", the output dimensions are reduced by "dimension" compared with that of "x". Otherwise, the output has one fewer dimension than "x".)

*@attention Constraints:
* The value range of "axes" is [-dims, dims - 1]. "dims" indicates the dimension length of "x".

*/
REG_OP(ReduceMax)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceMax)

/**
*@brief Returns the maximum of elements across dimensions of a Tensor.

*@par Inputs:
*x: A multi-dimensional Tensor of type float16, float32, or int16.

*@par Attributes:
* Two attributes, including: \n
*@li axes: A required listint, specifying the axes information of the index with the maximum value.
*@li keep_dims: A bool, specifying whether to keep dimensions for the output Tensor. Defaults to "false".

*@par Outputs:
*y: A multi-dimensional Tensor, specifying the maximum value of the corresponding axis in the tensor. Has the same type as "x". (If "keep_dims" is set to "false", the output dimensions are reduced by "dimension" compared with that of "x". Otherwise, the output has one fewer dimension than "x".)

*@attention Constraints:
* The value range of "axis" is [-dims, dims - 1]. "dims" indicates the dimension length of "x".
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
*@brief Computes the minimum of elements across dimensions of a tensor.

*@par Inputs:
*@li input_tensor: A Tensor. Must be one of the following types: float16, float32, int8, uint8.
*@li axes: A Tensor of type int8 or int32. Specifies the dimensions to reduce. Defaults to "None". 

*@par Attributes:\n
*keep_dims: An optional bool. If "True", reduced dimensions will be retained. Defaults to "False". 

*@par Outputs:\n
*output_tensor: A Tensor. Must be one of the following types: float16, float32, int8, uint8.

*@attention Constraints:\n
* If "axes = None", all dimensions will be reduced. "axes" must be in the range [-rank(input_shape), rank(input_shape)).

*/
REG_OP(ReduceMin)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceMin)

/**
*@brief Computes the minimum of elements across dimensions of a tensor.

*@par Inputs:\n
*input_min: A Tensor. Must be one of the following types: float16, float32, int8, uint8.

*@par Attributes:
*@li axes: An optional int32, list, tuple, or NoneType value. Specifies the dimensions to reduce. Defaults to "None". 
*@li keep_dims: An optional bool or NoneType value. If "True", reduced dimensions will be retained. Defaults to "None" (equivalent to "False"). 

*@par Outputs:\n
*output_min: A Tensor. Must be one of the following types: float16, float32, int8, uint8.

*@attention Constraints:\n
* If "axes = None", all dimensions will be reduced. "axes" must be in the range [-rank(input_shape), rank(input_shape)).

*/
REG_OP(ReduceMinD)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT8,DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT8,DT_UINT8}))
    .REQUIRED_ATTR(axes, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceMinD)
/**
*@brief Computes the "logical or" of elements across dimensions of a tensor.\n
* Reduces "x" along the dimensions given in "axes".
* Unless "keep_dims" is true, the rank of the tensor is reduced by 1 for each
* entry in "axes". If "keep_dims" is true, the reduced dimensions
* are retained with length 1.
*
* If "axes" is None, all dimensions are reduced, and a
* tensor with a single element is returned.
*
*@attention Constraints:\n
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
*/
REG_OP(ReduceAny)
    .INPUT(x, TensorType({DT_BOOL}))
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceAny)
/**
*@brief Computes the "logical or" of elements across dimensions of a tensor.\n
* Reduces "x" along the dimensions given in "axes".
* Unless "keep_dims" is true, the rank of the tensor is reduced by 1 for each
* entry in "axes". If "keep_dims" is true, the reduced dimensions
* are retained with length 1.
*
* If "axis" is None, all dimensions are reduced, and a
* tensor with a single element is returned.
*
*@attention Constraints:\n
*  Only support bool
*
*@par Inputs:
* x: The boolean tensor to reduce.
*
*@par Attributes:
*@li axes: The dimensions to reduce. If "None" (default), reduces all
*          dimensions. Must be in the range "[-rank(x), rank(x))".
*@li keep_dims: If true, retains reduced dimensions with length 1.
*
*@par Outputs:
* y: The reduced tensor
*
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
*SUMSQ   Computes the mean values of elements across specified dimensions of a tensor.

*@par Inputs: 
*x: A Tensor of type float16 or float32

*@par Attributes:
*@li operation: An optional int32 from 1(SUM), 2(ASUM), 3(SUMSQ), and 4(MEAN), 
*specifying the reduction algorithm. Defaults to 1.
*@li axis: An optional int32, specifying the first axis to reduce. Defaults to "0". 
*The value range is [-N, N-1], where N is the input tensor rank.
*@li coeff: An optional float32, specifying the scale coefficient. Defaults to "1.0".

*@par Outputs: 
*y: A Tensor. Has the same type as "x".

*@attention Constraints: The Reduction operator supports type float16 only on the device chip.
*/
REG_OP(Reduction)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(operation, Int, 1)
    .ATTR(axis, Int, 0)
    .ATTR(coeff, Float, 1.0)
    .OP_END_FACTORY_REG(Reduction);

/**
*@brief Computes the euclidean norm of elements across dimensions of a tensor.

*@par Inputs:
*@li input_tensor: A Tensor. Must be one of the following types: float16, float32, int32.
*@li axes: A Tensor of type int8 or int32. Specifies the dimensions to reduce. Defaults to "None".

*@par Attributes:\n
*keep_dims: An optional bool. If "True", reduced dimensions will be retained. Defaults to "False".

*@par Outputs:\n
*output_tensor: A Tensor. Must be one of the following types: float16, float32, int32.

*@attention Constraints:\n
* If "axes = None", all dimensions will be reduced. "axes" must be in the range [-rank(input_shape), rank(input_shape)).

*/
REG_OP(EuclideanNorm)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(EuclideanNorm)

/**
*@brief Computes the euclidean norm of elements across dimensions of a tensor.

*@par Inputs:\n
*input_min: A Tensor. Must be one of the following types: float16, float32, int32.

*@par Attributes:
*@li axes: An optional int32, list, tuple, or NoneType value. Specifies the dimensions to reduce. Defaults to "None".
*@li keep_dims: An optional bool or NoneType value. If "True", reduced dimensions will be retained. Defaults to "None" (equivalent to "False").

*@par Outputs:\n
*output_min: A Tensor. Must be one of the following types: float16, float32, int32.

*@attention Constraints:\n
* If "axes = None", all dimensions will be reduced. "axes" must be in the range [-rank(input_shape), rank(input_shape)).

*/
REG_OP(EuclideanNormD)
    .INPUT(x, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16}))
    .ATTR(axes, ListInt, {})
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(EuclideanNormD)

} //namespace ge


#endif /* GE_OP_REDUCE_OPS_H */
