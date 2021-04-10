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
 * \file random_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_RANDOM_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_RANDOM_OPS_H_

#include <vector>

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Draws samples from a multinomial distribution . \n

*@par Inputs:
*Inputs include:
* @li logits: A Tensor. Must be one of the following types: float16, float, double.
2-D Tensor with shape [batch_size, num_classes].
* @li num_samples: A Tensor of type int32. 0-D. Number of independent samples to draw for each row slice . \n

*@par Attributes:
*@li output_dtype: An optional type from: int32, int64. Defaults to int64.
*@li seed: An optional int. Defaults to 0.
*@li seed2: An optional int. Defaults to 0 . \n

*@par Outputs:
*y_indices: A Tensor of type output_dtype . \n

*@attention Constraints:
*The implementation for Multinomial on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with tensorflow Multinomial operator.
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
*@brief Outputs random values from a normal distribution . \n

*@par Inputs:
*Inputs include:
* @li shape: A Tensor. Must be one of the following types: int32, int64.
      The shape of the output tensor. Batches are indexed by the 0th dimension.
* @li means: A Tensor. Must be one of the following types: half, bfloat16, float32, float64.
* @li stdevs: A Tensor. Must have the same type as means.
* @li min: A Tensor. Must have the same type as means. The minimum cutoff. May be -infinity.
* @li max: A Tensor. Must have the same type as means . \n

*@par Attributes:
*@li seed: An optional int. Defaults to 0.
*@li seed2: An optional int. Defaults to 0 . \n

*@par Outputs:
*y: A Tensor. Has the same type as means . \n

*@attention Constraints:
*The implementation for ParameterizedTruncatedNormal on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with tensorflow ParameterizedTruncatedNormal operator.
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
*@brief Computes the derivative of a Gamma random sample w.r.t. alpha . \n

*@par Inputs:
*Inputs include:
* @li alpha: A Tensor. Must be one of the following types: float32, float64.
* @li sample: A Tensor. Must have the same type as alpha . \n

*@par Outputs:
*y: A Tensor. Has the same type as alpha . \n

*@attention Constraints:
*The implementation for RandomGammaGrad on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with tensorflow RandomGammaGrad operator.
*/
REG_OP(RandomGammaGrad)
    .INPUT(alpha, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(sample, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(RandomGammaGrad)

/**
*@brief Outputs random values from the Gamma distribution(s) described by alpha . \n

*@par Inputs:
*Inputs include:
* @li shape: A Tensor. Must be one of the following types: int32, int64. 1-D integer tensor.
* @li alpha: A Tensor. Must be one of the following types: half, float32, float64 . \n

*@par Attributes:
*@li seed: An optional int. Defaults to 0.
*@li seed2: An optional int. Defaults to 0 . \n

*@par Outputs:
*y: A Tensor. Has the same type as alpha . \n

*@attention Constraints:
*The implementation for RandomGamma on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with tensorflow RandomGamma operator.
*/
REG_OP(RandomGamma)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(alpha, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(RandomGamma)

/**
*@brief Outputs random values from the Poisson distribution(s) described by rate . \n

*@par Inputs:
*Inputs include:
* @li shape: A Tensor. Must be one of the following types: int32, int64. 1-D integer tensor.
* @li rate: A Tensor. Must be one of the following types: half, float32, float64, int32, int64 . \n

*@par Attributes:
*@li dtype: An optional type from: half, float32, float64, int32, int64. Defaults to int64.
*@li seed: An optional int. Defaults to 0.
*@li seed2: An optional int. Defaults to 0 . \n

*@par Outputs:
*y: A Tensor of type dtype . \n

*@attention Constraints:
*The implementation for RandomPoisson on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with tensorflow RandomPoisson operator.
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
*@brief Randomly shuffles a tensor along its first dimension . \n

*@par Inputs:
*Inputs include:
*x: A Tensor. The tensor to be shuffled . \n

*@par Attributes:
*@li seed: An optional int. Defaults to 0.
*@li seed2: An optional int. Defaults to 0 . \n

*@par Outputs:
*y: A Tensor. Has the same type as x . \n

*@attention Constraints:
*The implementation for RandomShuffle on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with tensorflow RandomShuffle operator.
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
*@brief Outputs random values from a normal distribution . \n

*@par Inputs:
*Inputs include:
*shape: A Tensor. Must be one of the following types: int32, int64. The shape of the output tensor . \n

*@par Attributes:
*@li dtype: A type from: half, float16, float32, float64. The type of the output.
*@li seed: An optional int. Defaults to 0.
*@li seed2: An optional int. Defaults to 0 . \n

*@par Outputs:
*y: A Tensor of type dtype . \n

*@attention Constraints:
*The implementation for RandomStandardNormal on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with tensorflow RandomStandardNormal operator.
*/
REG_OP(RandomStandardNormal)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(dtype, Type)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(RandomStandardNormal)

/**
*@brief Outputs random integers from a uniform distribution . \n

*@par Inputs:
*Inputs include:
* @li shape: A Tensor. Must be one of the following types: int32, int64. The shape of the output tensor.
* @li min: A Tensor. Must be one of the following types: int32, int64. 0-D.
* @li max: A Tensor. Must have the same type as minval. 0-D . \n

*@par Attributes:
*@li seed: An optional int. Defaults to 0.
*@li seed2: An optional int. Defaults to 0 . \n

*@par Outputs:
*y: A Tensor. Has the same type as min . \n

*@attention Constraints:
*The implementation for RandomUniformInt on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with tensorflow RandomUniformInt operator.
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
*@brief Outputs random values from a uniform distribution . \n

*@par Inputs:
*Inputs include:
*shape: A Tensor. Must be one of the following types: int32, int64. The shape of the output tensor . \n

*@par Attributes:
*@li dtype: A type from: half, float16, float32, float64. The type of the output.
*@li seed: An optional int. Defaults to 0.
*@li seed2: An optional int. Defaults to 0 . \n

*@par Outputs:
*y: A Tensor of type dtype . \n

*@attention Constraints:
*The implementation for RandomUniform on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with tensorflow RandomUniform operator.
*/
REG_OP(RandomUniform)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(dtype, Type)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(RandomUniform)

/**
*@brief Outputs random values from a truncated normal distribution . \n

*@par Inputs:
*Inputs include:
*shape: A Tensor. Must be one of the following types: int32, int64 . \n

*@par Attributes:
*@li seed: An optional int. Defaults to 0.
*@li seed2: An optional int. Defaults to 0 . \n

*@par Outputs:
*size: A Tensor of types: float16, float32, double . \n

*@attention Constraints:
*The implementation for TruncatedNormal on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with tensorflow TruncatedNormal operator.
*/
REG_OP(TruncatedNormal)
    .INPUT(shape, TensorType({ DT_INT32, DT_INT64 }))
    .OUTPUT(y, TensorType({ DT_FLOAT16, DT_FLOAT, DT_DOUBLE }))
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(TruncatedNormal)

/**
*@brief Generate random bit mask for dropout . \n

*@par Inputs:
include:
*@li shape:The shape of the output tensor.
*@li prob:0-D. Number of bit 1 . \n

*@par Attributes:
*@li seed:If either seed or seed2 are set to be non-zero, the random number
*generator is seeded by the given seed. Otherwise, it is seeded by a random seed.
*@li seed2:A second seed to avoid seed collision . \n

*@par Outputs:
*y:Output (1-D) random number using uint data format . \n

*@attention Constraints:
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
*@brief Generates values in an interval . \n

*@par Inputs:
* Four ND inputs, including:
*@li assist: A 1D Tensor of type float32.
*@li start: A 1D Tensor of type float32, for the first entry in the range.
*@li stop: A 1D Tensor of type float32, for the last entry in the range.
*@li num: A 1D Tensor of type int32 or int64, for the common difference of the entries . \n

*@par Outputs:
*output_op: A 1D Tensor of type float32 . \n

*@attention Constraints:
* "input_assist" is a sequence of "input_num" evenly-spaced values beginning at 0 with an common difference of 1 . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator lin_space.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use LinSpace instead.
*/
REG_OP(LinSpaceD)
    .INPUT(assist, TensorType({DT_FLOAT}))
    .INPUT(start, TensorType({DT_FLOAT}))
    .INPUT(stop, TensorType({DT_FLOAT}))
    .INPUT(num, TensorType::IndexNumberType())
    .OUTPUT(output, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(LinSpaceD)

/**
*@brief Generates values in an interval . \n

*@par Inputs:
* Four ND inputs, including:
*@li start: A 1D Tensor of type float32, for the first entry in the range.
*@li stop: A 1D Tensor of type float32, for the last entry in the range.
*@li num: A 1D Tensor of type int32 or int64, for the common difference of the entries . \n

*@par Outputs:
*output_op: A 1D Tensor of type float32 . \n

*@attention Constraints:
* "input_assist" is a sequence of "input_num" evenly-spaced values beginning at 0 with an common difference of 1 . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator lin_space.
*/
REG_OP(LinSpace)
    .INPUT(start, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(stop, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(num, TensorType::IndexNumberType())
    .OUTPUT(output, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(LinSpace)



/**
*@brief The dropout operator randomly sets (according to the given dropout probability)
*the outputs of some units to zero, while others are remain unchanged. . \n

*@par Inputs:
*One input, including:
*@li x:The input tensor variable. The data type is float32. \n

*@par Attributes:
*@li dropout_ratio:Float between 0 and 1. Fraction of the input units to drop.Defaults to "0.5".
*@li scale_train: Bool,default to true.
*@li alpha: An optional float32. A scaling factor. Defaults to "1.0".
*@li beta: An optional float32. An exponent. Defaults to "0.0". \n

*@par Outputs:
*y: A Variable holding Tensor representing the dropout, has same shape and data type with x. \n
*/
REG_OP(Dropout)
    .INPUT(x, TensorType{DT_FLOAT})
    .OUTPUT(y, TensorType{DT_FLOAT})
    .ATTR(dropout_ratio, Float, 0.5)
    .ATTR(scale_train, Bool, true)
    .ATTR(alpha, Float, 1.0)
    .ATTR(beta, Float, 0.0)
    .OP_END_FACTORY_REG(Dropout)

/**
*@brief Shuffle index of no-zero element . \n

*@par Inputs:
include:
*x:A tensor <= 5-D . \n

*@par Attributes:
*@li count:the count of output, if 0, out all no-zero elements.
*@li seed:If either seed or seed2 are set to be non-zero, the random number generator is seeded by the given seed.
          Otherwise, it is seeded by a random seed.
*@li seed2:A second seed to avoid seed collision . \n

*@par Outputs:
*@li y:2-D tensor, no-zero element index.
*@li mask:1-D, whether the corresponding index is valid . \n

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
*Inputs including:
* @li x: A required Tensor. Must be one of the following types:
         float16, float32, int8, uint8, int16, uint16, int32, uint32, int64, uint64 . \n

*@par Attributes:
*@li group: A required int32, specifying the number of groups to split the channel dimension into. Defaults to "1" . \n

*@par Outputs:
*y: A required Tensor. Has same type and shape as "x". Must be one of the following types:
    float16, float32, int8, uint8, int16, uint16, int32, uint32, int64, uint64 . \n

*@attention Constraints:
*@li "group" must be greater than 0 and must evenly divide the channel dimension size.
*@li The format of input "x" must be NCHW.
*@par Third-party framework compatibility
* Compatible with the Caffe operator ShuffleChannel.
*/
REG_OP(ShuffleChannel)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT8, DT_UINT8, DT_INT16,
                          DT_UINT16, DT_INT32, DT_UINT32,DT_INT64,DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT8, DT_UINT8, DT_INT16,
                           DT_UINT16, DT_INT32, DT_UINT32,DT_INT64,DT_UINT64}))
    .ATTR(group, Int, 1)
    .OP_END_FACTORY_REG(ShuffleChannel)

/**
 * @briefGenerate a tensor of samples from a multinomial 
 * distribution according to the probabilities of each of 
 * the possible outcomes.
 * 
 * @par inputs
 * one input including:
 * @li x:Input tensor with shape [batch_size, class_size], 
 * where class_size is the number of all possible outcomes.
 * Each value along the axis zero represents the unnormalized 
 * log-probability of each corresponding outcome in a batch.
 * 
 * @par output
 * one output including:
 * @li y:Output tensor with shape [batch_size, sample_size], 
 * where sample_size is the number of times to sample. 
 * Each value along the axis zero represents the outcome of 
 * the corresponding sample in a batch.
 * 
 */
REG_OP(MultinomialFuss)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_FLOAT64}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Int, 6)
    .ATTR(sample_size, Int, 1)
    .ATTR(seed, Float, 0)
    .OP_END_FACTORY_REG(MultinomialFuss)

/**
* @brief During training, randomly zeroes some of the elements of the input tensor
* with probability
*
* @par Inputs:
* @li x: A ND Tensor. Must be one of the following data types: Float, Float16
* @li seed: A ND Tensor. Must be one of the following data types: Float
*
* @par Attributes:
* @li p: probability of an element to be zeroed
*
* @par Outputs:
* @li y: A tensor with the same shape and type as "x".
* @li mask: A tensor with the same shape and type as "x".
* @li new_seed: A tensor with the same shape and type as "seed".
*/

REG_OP(DropoutV2)
    .INPUT(x, TensorType({ DT_FLOAT16, DT_FLOAT }))
    .INPUT(seed, TensorType({ DT_FLOAT }))
    .OUTPUT(y, TensorType({ DT_FLOAT16, DT_FLOAT }))
    .OUTPUT(mask, TensorType({ DT_FLOAT }))
    .OUTPUT(seed, TensorType({ DT_FLOAT }))
    .REQUIRED_ATTR(p, Float)
    .OP_END_FACTORY_REG(DropoutV2)
}   // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_RANDOM_OPS_H_
