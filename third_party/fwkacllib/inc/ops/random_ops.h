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
*@brief Creates a multinomial distribution. \n

*@par Inputs:
*Inputs include:
* @li q: A Tensor. Must be one of the following types: float, double.
1-D Tensor with shape [num_classes].
* @li j: A Tensor. Must be one of the following types: int64.
1-D Tensor with shape [num_classes].
* @li num_samples: A Tensor of type int32. 0-D. Number of independent samples to draw for each row slice . \n

*@par Attributes:
*@li output_dtype: An optional type from: int32, int64. Defaults to int64.
*@li seed: An optional int. Defaults to 0.
*@li seed2: An optional int. Defaults to 0. \n

*@par Outputs:
*y: A Tensor of type int32 or int64. \n

*@attention Constraints:
*The implementation for MultinomialAliasDraw on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with torch _multinomial_alias_draw operator.
*/
REG_OP(MultinomialAliasDraw)
    .INPUT(q, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(j, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT64}))
    .REQUIRED_ATTR(num_samples, Int)
    .ATTR(seed, Int, 0)
    .OP_END_FACTORY_REG(MultinomialAliasDraw)

/**
*@brief Prepares for MultinomialAliasDraw to create a multinomial distribution. \n

*@par Inputs:
*Inputs include:
* @li probs: A Tensor. Must be one of the following types: float, double.
1-D Tensor with shape [num_classes]. \n

*@par Outputs:
*j: A Tensor. Must be one of the following types: int64.
1-D Tensor with shape [num_classes].
*q: A Tensor. Must be one of the following types: float, double.
1-D Tensor with shape [num_classes]. \n

*@attention Constraints:
*The implementation for MultinomialAliasSetup on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with torch _multinomial_alias_setup operator.
*/
REG_OP(MultinomialAliasSetup)
    .INPUT(probs, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(j, TensorType({DT_INT64}))
    .OUTPUT(q, TensorType({DT_FLOAT, DT_DOUBLE})) 
    .OP_END_FACTORY_REG(MultinomialAliasSetup)

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
*@brief Returns the random permutation of integers from 0 to n-1. \n

*@par Attributes:
*@li n: An required int.
*@li dtype: An optional str. Defaults to int64 .
*@li layout: An optional int. Defaults to 0 . \n

*@par Outputs:
*out: A required Tensor. Must be one of the following types:
         float16, float32, float32, int8, uint8, int16, int32, int64. \n

*@attention Constraints:
*The implementation for Randperm on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with Pytorch Randperm operator.
*/
REG_OP(Randperm)
    .OUTPUT(out, TensorType({DT_INT64, DT_INT32, DT_INT16,
        DT_UINT8, DT_INT8, DT_FLOAT16, DT_FLOAT32, DT_DOUBLE}))
    .REQUIRED_ATTR(n, Int)
    .ATTR(layout, Int, 0)
    .ATTR(dtype, Type, DT_INT64)
    .OP_END_FACTORY_REG(Randperm)

/**
*@brief Fills a tensor with elements drawn from the poisson distribution. \n

*@par Inputs:
*x:  A Tensor. Must be one of the following types: float16, float. \n

*@par Attributes:
*@li seed: An optional int. Defaults to 0. \n

*@par Outputs:
*y: A Tensor list with same type as "x" . \n

*@par Third-party framework compatibility
*@ Compatible with the Pytorch operator Poisson.
*/
REG_OP(Poisson)
    .INPUT(x, TensorType({ DT_FLOAT16,DT_FLOAT }))
    .OUTPUT(y, TensorType({ DT_FLOAT16,DT_FLOAT }))
    .ATTR(seed, Int, 0)
    .OP_END_FACTORY_REG(Poisson)   
 
/**
*@brief Outputs random values from the Poisson distribution(s) described by rate . \n

*@par Inputs:
*Inputs include:
* @li shape: A Tensor. Must be one of the following types: int32, int64. 1-D integer tensor.
* @li rate: A Tensor. Must be one of the following types: half, float32, float64, int32, int64 . \n

*@par Attributes:
*@li dtype: An optional type from: half, float32, float64, int32, int64. Defaults to int64.
*@li seed: An optional int. Defaults to 0. If either seed or seed2 are set to be non-zero, 
the random number generator is seeded by the given seed. Otherwise, it is seeded by a random seed.
*@li seed2: An optional int. Defaults to 0 . A second seed to avoid seed collision. \n

*@par Outputs:
*y: A Tensor of type dtype float16, float, double, int32, int64. \n

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
*@li seed: An optional int. Defaults to 0. If either seed or seed2 are set to be non-zero, 
the random number generator is seeded by the given seed. Otherwise, it is seeded by a random seed.
*@li seed2: An optional int. Defaults to 0 . A second seed to avoid seed collision. \n

*@par Outputs:
*y: A Tensor. Has the same type as x . A Tensor of type float16, float, 
*double, int32, int64, int16, uint16, int8, uint8, int32,int64. \n

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
*@li seed: An optional int. Defaults to 0. If either seed or seed2 are set to be non-zero, 
the random number generator is seeded by the given seed. Otherwise, it is seeded by a random seed.
*@li seed2: An optional int. Defaults to 0 . A second seed to avoid seed collision. \n

*@par Outputs:
*y: A Tensor of type float32, float16, double. \n

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
*@brief Output random value from  separate normal distribution. \n

*@par Inputs:
*Inputs include:
*mean: The mean is a tensor with the mean of each output element’s normal distribution . 
*std: The std is a tensor with the standard deviation of each output element’s normal distribution. \n
*@par Outputs:
*y: A Tensor of type dtype . \n

*@attention Constraints:
*The implementation for Normal on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with Pytorch Normal operator.
*/
REG_OP(Normal)
    .INPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(std, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(Normal)

/**
*@brief Outputs random integers from a uniform distribution . \n

*@par Inputs:
*Inputs include:
* @li shape: A Tensor. Must be one of the following types: int32, int64. The shape of the output tensor.
* @li min: A Tensor. Must be one of the following types: int32, int64. 0-D.
* @li max: A Tensor. Must have the same type as minval. 0-D . \n

*@par Attributes:
*@li seed: An optional int. Defaults to 0. If either seed or seed2 are set to be non-zero, 
the random number generator is seeded by the given seed. Otherwise, it is seeded by a random seed.
*@li seed2: An optional int. Defaults to 0 . A second seed to avoid seed collision. \n

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
*@li seed: An optional int. Defaults to 0. If either seed or seed2 are set to be non-zero, 
the random number generator is seeded by the given seed. Otherwise, it is seeded by a random seed.
*@li seed2: An optional int. Defaults to 0 . A second seed to avoid seed collision. \n

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
*@li seed: An optional int. Defaults to 0.If either `seed` or `seed2` 
are set to be non-zero, the random number generator is seeded by the given 
seed. Otherwise, it is seeded by a random seed.
*@li seed2: An optional int. Defaults to 0 . A second seed to avoid seed collision. \n

*@par Outputs:
*y: A Tensor of types: float16, float32, double . A tensor of the specified shape
filled with random truncated normal values. \n

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
*@brief Generate random uint8 mask for dropout v3 . \n

*@par Inputs:
include:
*@li shape:The shape of the output tensor.
*@li prob:0-D. Prob of 1 . \n

*@par Attributes:
*@li seed:If either seed or seed2 are set to be non-zero, the random number
*generator is seeded by the given seed. Otherwise, it is seeded by a random seed.
*@li seed2:A second seed to avoid seed collision . \n

*@par Outputs:
*y:Output (1-D) random number using uint8 data format . \n

*@attention Constraints:
*The output is aligned with 16

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.

*@see DropOutGenMaskV3()
*/
REG_OP(DropOutGenMaskV3)
    .INPUT(shape, TensorType({ DT_INT32, DT_INT64 }))
    .INPUT(prob, TensorType({ DT_FLOAT16, DT_FLOAT }))
    .OUTPUT(y, TensorType({ DT_UINT8 }))
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(DropOutGenMaskV3)

    
/**
* @brief Generate stateless random bit mask for dropout . \n

* @par Inputs:
include:
* @li shape:The shape of the output tensor.
* @li prob:0-D. Number of bit 1 . \n
* @li seed:Frist seed to avoid seed collision.
* @li seed1:Second seed to avoid seed collision . \n
* @li offset:Initial offset of random number . \n

* @par Outputs:
*y:Output (1-D) random number using uint data format . \n

* @attention Constraints:
*The output is aligned with 128 bits

* @see StatelessDropOutGenMask()
*/
REG_OP(StatelessDropOutGenMask)
    .INPUT(shape, TensorType({ DT_INT32, DT_INT64 }))
    .INPUT(prob, TensorType({ DT_FLOAT16, DT_FLOAT }))
    .INPUT(seed, TensorType({ DT_INT32, DT_INT64 }))
    .INPUT(seed1, TensorType({ DT_INT32, DT_INT64 }))
    .OPTIONAL_INPUT(offset, TensorType({ DT_INT64 }))
    .OUTPUT(y, TensorType({ DT_UINT8 }))
    .OP_END_FACTORY_REG(StatelessDropOutGenMask)

/**
* @brief Generate bernoulli distribution for tensor input . \n

* @par Inputs:
include:
* @li shape:The shape of the output tensor. A Tensor of type int32, int64.
* @li prob:0-D. Number of bit 1 . \n
* @li seed:If seed is set to be -1, and offset is set to be 0, the random number
* generator is seeded by arandom seed. Otherwise, it is seeded by the given seed.
* @li offset:To avoid seed collision . \n

* @par Outputs:
* y:A Tensor. A Tensor of type int8, uint8, int16, uint16, 
*  int32, uint32, int64, uint64, bool, float16, float, double, bf16. \n
*/
REG_OP(StatelessBernoulli)
    .INPUT(shape, TensorType({ DT_INT32, DT_INT64}))
    .INPUT(prob, TensorType({ DT_FLOAT16, DT_FLOAT, DT_DOUBLE }))
    .INPUT(seed, TensorType({ DT_INT64 }))
    .INPUT(offset, TensorType({ DT_INT64 }))
    .OUTPUT(y, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32,
        DT_INT64, DT_UINT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .OP_END_FACTORY_REG(StatelessBernoulli)
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
* x: A required Tensor. Must be one of the following types:
     float16, float32, int8, uint8, int16, uint16, int32, uint32, int64, uint64 . \n

*@par Attributes:
* group: A required int32, specifying the number of groups to split the channel dimension into. Defaults to "1" . \n

*@par Outputs:
* y: A required Tensor. Has same type and shape as "x". Must be one of the following types:
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
 * @par Restrictions:
 * Warning:THIS FUNCTION IS EXPERIMENTAL. Please do not use.
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

/**
* @brief The Bernoulli distribution with probability . \n

* @par Inputs:
* @li x: A ND Tensor. Must be one of the following data types: 
         int8, uint8, int16, int32, int64, bool, float32, float64 . 
* @li p: A ND Tensor. The probability of an element to be zeroed. 
        Must be one of the following data types: float32, float64. \n

* @par Attributes:
* seed: An Integer, the seed of the random generator. Default value -1 
    to use current timestamp, otherwise it should be a positive integer.

* @par Outputs:
* y: A tensor with the same shape and type as "x".
*/

REG_OP(Bernoulli)
    .INPUT(x, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT, DT_DOUBLE}))
    .INPUT(p, TensorType({ DT_FLOAT, DT_DOUBLE }))
    .OUTPUT(y, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT, DT_DOUBLE}))
    .ATTR(seed, Int, -1)
    .OP_END_FACTORY_REG(Bernoulli)

/**
 * @brief: Fill the input tensor with values drawn from the uniform distribution U(from, to). \n
 
 * @par Inputs:
 * x: A Tensor. Must be one of the following types: float16, float, double. \n

 * @par Attributes:
 * @li from: The lower bound of the uniform. Defaults: 0.0
 * @li to: The upper bound of the uniform. Defaults: 1.0  \n

 * @par Outputs:
 * y: A Tensor has the same type as x. \n
 */
REG_OP(Uniform)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(from, Float, 0.0)
    .ATTR(to, Float, 1.0)
    .OP_END_FACTORY_REG(Uniform)

/**
*@brief Outputs integers consisting of 0 and 1, used for lstm etc. \n
*@par Inputs
* @li time_step: A tensor with data type int64. 0-D.
* @li batch_size: A tensor with data type int64. 0-D.

*@par Outputs:
*y: A Tensor. Has the  type float16 or float, 2-D, [time_step,batch_size]. \n

*@attention Constraints:
* Compatible with the Caffe operator ContinuationIndicator.
*/
REG_OP(ContinuationIndicator)
    .REQUIRED_ATTR(time_step, Int)
    .REQUIRED_ATTR(batch_size, Int)
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(ContinuationIndicator)

/**
*@brief Outputs random values from the Exponential distribution(s) described by rate . \n

*@par Inputs:
*Inputs include:
* @li x: A Tensor. Must be one of the following types: half, float32, float64. \n

*@par Attributes:
*@li lambda: An optional float. Defaults to 1.
*@li seed: An optional int. Defaults to 0.The random number generator is seeded by the given seed.
 Otherwise, it is seeded by a random seed. \n

*@par Outputs:
*y: A Tensor of type dtype float16, float, double. \n

*@attention Constraints:
*The implementation for Exponential on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with tensorflow Exponential operator.
*/
REG_OP(Exponential)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(lambda, Float, 1)
    .ATTR(seed, Int, 0)
    .OP_END_FACTORY_REG(Exponential)

/**
*@brief Fills a tensor with elements drawn from the geometric distribution. \n

*@par Inputs:
*x:  A Tensor. Must be one of the following types: float16, float. \n

*@par Attributes:
*@li p: The probability of experimental success in Bernoulli's experiment.
*@li seed: An optional int. Defaults to 0. \n

*@par Outputs:
*y: A Tensor list with same type as "x" . \n

*@par Third-party framework compatibility
*@ Compatible with the Pytorch operator Geometric.
*/
REG_OP(Geometric)
    .INPUT(x, TensorType({ DT_FLOAT16,DT_FLOAT }))
    .OUTPUT(y, TensorType({ DT_FLOAT16,DT_FLOAT }))
    .REQUIRED_ATTR(p, Float)
    .ATTR(seed, Int, 0)
    .OP_END_FACTORY_REG(Geometric)

}   // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_RANDOM_OPS_H_
