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

#ifndef GE_OP_RANDOM_OPS_H_
#define GE_OP_RANDOM_OPS_H_

#include <vector>

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Draws samples from a multinomial distribution.

*@par Inputs:
*Inputs include: \n
* @li logits: A Tensor. Must be one of the following types: float32, float64, int32, uint8, int16, int8, \n
      int64, bfloat16, uint16, half, uint32, uint64. 2-D Tensor with shape [batch_size, num_classes].
* @li num_samples: A Tensor of type int32. 0-D. Number of independent samples to draw for each row slice.

*@par Attributes:
*@li output_dtype: An optional type from: int32, int64. Defaults to int64.
*@li seed: An optional int. Defaults to 0.
*@li seed2: An optional int. Defaults to 0.

*@par Outputs:
*y_indices: A Tensor of type output_dtype.

*@attention Constraints:\n
*-The implementation for Multinomial on Ascend uses AICPU, with bad performance.\n

*/
REG_OP(Multinomial)
    .INPUT(logits, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(num_samples, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Type, DT_INT64)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(Multinomial)

/**
*@brief Outputs random values from a normal distribution.

*@par Inputs:
*Inputs include: \n
* @li shape: A Tensor. Must be one of the following types: int32, int64. \n
      The shape of the output tensor. Batches are indexed by the 0th dimension.
* @li means: A Tensor. Must be one of the following types: half, bfloat16, float32, float64.
* @li stdevs: A Tensor. Must have the same type as means.
* @li min: A Tensor. Must have the same type as means. The minimum cutoff. May be -infinity.
* @li max: A Tensor. Must have the same type as means.

*@par Attributes:
*@li seed: An optional int. Defaults to 0.
*@li seed2: An optional int. Defaults to 0.

*@par Outputs:
*y: A Tensor. Has the same type as means.

*@attention Constraints:\n
*-The implementation for ParameterizedTruncatedNormal on Ascend uses AICPU, with bad performance.\n

*/
REG_OP(ParameterizedTruncatedNormal)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(means, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(stdevs, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(min, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(max, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(ParameterizedTruncatedNormal)

/**
*@brief Computes the derivative of a Gamma random sample w.r.t. alpha.

*@par Inputs:
*Inputs include: \n
* @li alpha: A Tensor. Must be one of the following types: float32, float64.
* @li sample: A Tensor. Must have the same type as alpha.

*@par Outputs:
*y: A Tensor. Has the same type as alpha.

*@attention Constraints:\n
*-The implementation for RandomGammaGrad on Ascend uses AICPU, with bad performance.\n

*/
REG_OP(RandomGammaGrad)
    .INPUT(alpha, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(sample, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(RandomGammaGrad)

/**
*@brief Outputs random values from the Gamma distribution(s) described by alpha.

*@par Inputs:
*Inputs include: \n
* @li shape: A Tensor. Must be one of the following types: int32, int64. 1-D integer tensor.
* @li alpha: A Tensor. Must be one of the following types: half, float32, float64.

*@par Attributes:
*@li seed: An optional int. Defaults to 0.
*@li seed2: An optional int. Defaults to 0.

*@par Outputs:
*y: A Tensor. Has the same type as alpha.

*@attention Constraints:\n
*-The implementation for RandomGamma on Ascend uses AICPU, with bad performance.\n

*/
REG_OP(RandomGamma)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(alpha, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(RandomGamma)

/**
*@brief Outputs random values from the Poisson distribution(s) described by rate.

*@par Inputs:
*Inputs include: \n
* @li shape: A Tensor. Must be one of the following types: int32, int64. 1-D integer tensor.
* @li rate: A Tensor. Must be one of the following types: half, float32, float64, int32, int64.

*@par Attributes:
*@li dtype: An optional type from: half, float32, float64, int32, int64. Defaults to int64.
*@li seed: An optional int. Defaults to 0.
*@li seed2: An optional int. Defaults to 0.

*@par Outputs:
*y: A Tensor of type dtype.

*@attention Constraints:\n
*-The implementation for RandomPoisson on Ascend uses AICPU, with bad performance.\n

*/
REG_OP(RandomPoisson)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(rate, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, \
        DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, \
        DT_INT32, DT_INT64}))
    .ATTR(dtype, Type, DT_INT64)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(RandomPoisson)

/**
*@brief Randomly shuffles a tensor along its first dimension.

*@par Inputs:
*Inputs include: \n
*x: A Tensor. The tensor to be shuffled.

*@par Attributes:
*@li seed: An optional int. Defaults to 0.
*@li seed2: An optional int. Defaults to 0.

*@par Outputs:
*y: A Tensor. Has the same type as x.

*@attention Constraints:\n
*-The implementation for RandomShuffle on Ascend uses AICPU, with bad performance.\n

*/
REG_OP(RandomShuffle)
    .INPUT(x, TensorType({DT_INT64, DT_INT32, DT_UINT16, DT_INT16,
        DT_UINT8, DT_INT8, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
        DT_COMPLEX128, DT_BOOL, DT_STRING, DT_RESOURCE}))
    .OUTPUT(y, TensorType({DT_INT64, DT_INT32, DT_UINT16, DT_INT16,
        DT_UINT8, DT_INT8, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
        DT_COMPLEX128, DT_BOOL, DT_STRING, DT_RESOURCE}))
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(RandomShuffle)

/**
*@brief Outputs random values from a normal distribution.

*@par Inputs:
*Inputs include: \n
*shape: A Tensor. Must be one of the following types: int32, int64. The shape of the output tensor.

*@par Attributes:
*@li dtype: A type from: half, float16, float32, float64. The type of the output.
*@li seed: An optional int. Defaults to 0.
*@li seed2: An optional int. Defaults to 0.

*@par Outputs:
*y: A Tensor of type dtype.

*@attention Constraints:\n
*-The implementation for RandomStandardNormal on Ascend uses AICPU, with bad performance.\n

*/
REG_OP(RandomStandardNormal)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(dtype, Type)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(RandomStandardNormal)

/**
*@brief Outputs random integers from a uniform distribution.

*@par Inputs:
*Inputs include: \n
* @li shape: A Tensor. Must be one of the following types: int32, int64. The shape of the output tensor.
* @li min: A Tensor. Must be one of the following types: int32, int64. 0-D.
* @li max: A Tensor. Must have the same type as minval. 0-D.

*@par Attributes:
*@li seed: An optional int. Defaults to 0.
*@li seed2: An optional int. Defaults to 0.

*@par Outputs:
*y: A Tensor. Has the same type as min.

*@attention Constraints:\n
*-The implementation for RandomUniformInt on Ascend uses AICPU, with bad performance.\n

*/
REG_OP(RandomUniformInt)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(min, TensorType({DT_INT32, DT_INT64}))
    .INPUT(max, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(RandomUniformInt)

/**
*@brief Outputs random values from a uniform distribution.

*@par Inputs:
*Inputs include: \n
*shape: A Tensor. Must be one of the following types: int32, int64. The shape of the output tensor.

*@par Attributes:
*@li dtype: A type from: half, float16, float32, float64. The type of the output.
*@li seed: An optional int. Defaults to 0.
*@li seed2: An optional int. Defaults to 0.

*@par Outputs:
*y: A Tensor of type dtype.

*@attention Constraints:\n
*-The implementation for RandomUniform on Ascend uses AICPU, with bad performance.\n

*/
REG_OP(RandomUniform)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(dtype, Type)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(RandomUniform)

/**
*@brief Outputs random values from a truncated normal distribution.

*@par Inputs:
*Inputs include: \n
*shape: A Tensor. Must be one of the following types: int32, int64.

*@par Attributes:
*@li seed: An optional int. Defaults to 0.
*@li seed2: An optional int. Defaults to 0.

*@par Outputs:
*size: A Tensor of types: float16, float32, double.

*@attention Constraints:\n
*-The implementation for TruncatedNormal on Ascend uses AICPU, with bad performance.\n

*/
REG_OP(TruncatedNormal)
    .INPUT(shape, TensorType({ DT_INT32, DT_INT64 }))
    .OUTPUT(y, TensorType({ DT_FLOAT16, DT_FLOAT, DT_DOUBLE }))
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(TruncatedNormal)

/**
*@brief Generate random bit mask for dropout.

*@par Inputs:
include: \n
*@li shape:The shape of the output tensor.
*@li prob:0-D. Number of bit 1.

*@par Attributes:
*@li seed:If either seed or seed2 are set to be non-zero, the random number\n
*generator is seeded by the given seed. Otherwise, it is seeded by a random seed.
*@li seed2:A second seed to avoid seed collision.

*@par Outputs:
*y:Output (1-D) random number using uint data format.

*@attention Constraints:\n
*The output is aligned with 128 bits

*@see DropOutGenMask()
*/
REG_OP(DropOutGenMask)
    .INPUT(shape, TensorType({ DT_INT32, DT_INT64 }))
    .INPUT(prob, TensorType({ DT_FLOAT16, DT_FLOAT }))
    .OUTPUT(y, TensorType({ DT_UINT8 }))
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(DropOutGenMask)

/**
*@brief Generates values in an interval.

*@par Inputs:\n
* Four ND inputs, including:
*@li input_assist: A 1D Tensor of type float32.
*@li input_start: A 1D Tensor of type float32, for the first entry in the range.
*@li input_stop: A 1D Tensor of type float32, for the last entry in the range.
*@li input_num: A 1D Tensor of type int32, for the common difference of the entries.

*@par Outputs:\n
*output_op: A 1D Tensor of type float32.

*@attention Constraints:\n
* "input_assist" is a sequence of "input_num" evenly-spaced values beginning at 0 with an common difference of 1.
*/
REG_OP(LinSpaceD)
    .INPUT(assist, TensorType({DT_FLOAT}))
    .INPUT(start, TensorType({DT_FLOAT}))
    .INPUT(stop, TensorType({DT_FLOAT}))
    .INPUT(num, TensorType::IndexNumberType())
    .OUTPUT(output, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(LinSpaceD)

/**
*@brief Generates values in an interval.

*@par Inputs:\n
* Four ND inputs, including:
*@li input_assist: A 1D Tensor of type float32.
*@li input_start: A 1D Tensor of type float32, for the first entry in the range.
*@li input_stop: A 1D Tensor of type float32, for the last entry in the range.
*@li input_num: A 1D Tensor of type int32, for the common difference of the entries.

*@par Outputs:\n
*output_op: A 1D Tensor of type float32.

*@attention Constraints:\n
* "input_assist" is a sequence of "input_num" evenly-spaced values beginning at 0 with an common difference of 1.

*/
REG_OP(LinSpace)
    .INPUT(start, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(stop, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(num, TensorType::IndexNumberType())
    .OUTPUT(output, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(LinSpace)

REG_OP(Dropout)
    .INPUT(x, TensorType{DT_FLOAT})
    .OUTPUT(y, TensorType{DT_FLOAT})
    .ATTR(dropout_ratio, Float, 0.5)
    .ATTR(scale_train, Bool, true)
    .ATTR(alpha, Float, 1.0)
    .ATTR(beta, Float, 0.0)
    .OP_END_FACTORY_REG(Dropout)

/**
*@brief Shuffle index of no-zero element.

*@par Inputs:
include: \n
*x:A tensor <= 5-D.

*@par Attributes:
*@li count:the count of output, if 0, out all no-zero elements.
*@li seed:If either seed or seed2 are set to be non-zero, the random number generator is seeded by the given seed.
          Otherwise, it is seeded by a random seed.
*@li seed2:A second seed to avoid seed collision.

*@par Outputs:
*@li y:2-D tensor, no-zero element index.
*@li mask:1-D, whether the corresponding index is valid.

*@see RandomChoiceWithMask()
*/
REG_OP(RandomChoiceWithMask)
    .INPUT(x, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .OUTPUT(mask, TensorType({DT_BOOL}))
    .ATTR(count, Int, 0)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(RandomChoiceWithMask)

/**
*@brief Permutes data in the channel dimension of the input

*@par Inputs:
*Inputs including: \n
* @li x: A required Tensor. Must be one of the following types:
         float16, float32, int8, uint8, int16, uint16, int32, uint32, int64, uint64.

*@par Attributes:
*@li group: A required int32, specifying the number of groups to split the channel dimension into. Defaults to "1".

*@par Outputs:
*y: A required Tensor. Has same type and shape as "x". Must be one of the following types:
    float16, float32, int8, uint8, int16, uint16, int32, uint32, int64, uint64.

*@attention Constraints:\n
*@li "group" must be greater than 0 and must evenly divide the channel dimension size.
*@li The format of input "x" must be NCHW.
*/
REG_OP(ShuffleChannel)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT8, DT_UINT8, DT_INT16,
                          DT_UINT16, DT_INT32, DT_UINT32,DT_INT64,DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT8, DT_UINT8, DT_INT16,
                           DT_UINT16, DT_INT32, DT_UINT32,DT_INT64,DT_UINT64}))
    .ATTR(group, Int, 1)
    .OP_END_FACTORY_REG(ShuffleChannel)
}   // namespace ge

#endif  // GE_OP_RANDOM_OPS_H_
