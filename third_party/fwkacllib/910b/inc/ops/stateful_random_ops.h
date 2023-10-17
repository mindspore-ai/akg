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
 * \file stateful_random_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_STATEFUL_RANDOM_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_STATEFUL_RANDOM_OPS_H_

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Non-deterministically generates some integers . \n

*@par Inputs:
*This op may use some OS-provided source of non-determinism (e.g. an RNG),
*so each execution will give different results. Inputs included:
*shape: The shape of the output tensor . \n

*@par Attributes:
*dtype: required, type. \n

*@par Outputs:
*y:A Returns Non-deterministic integer values with specified shape . \n

*@par Third-party framework compatibility
*Compatible with tensorflow NonDeterministicInts operator.
*/

REG_OP(NonDeterministicInts)
    .INPUT(shape, TensorType({DT_INT32,DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32,DT_INT64}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(NonDeterministicInts)

/**
*@brief Advance the counter of a counter-based RNG. The state of the RNG after
*`rng_skip(n)` will be the same as that after `stateful_uniform([n])`
*(or any other distribution). The actual increment added to the
*counter is an unspecified implementation detail . \n

*@par Inputs:
*@li x: The handle of the resource variable that stores the state of the RNG.
*@li algorithm: The RNG algorithm.
*@li delta: The amount of advancement . \n

*@par Third-party framework compatibility
* Compatible with tensorflow RngSkip operator.
*/

REG_OP(RngSkip)
    .INPUT(x, TensorType({DT_RESOURCE}))
    .INPUT(algorithm, TensorType({DT_INT64}))
    .INPUT(delta, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(RngSkip)

/**
*@brief Outputs random integers from a uniform distribution.
The generated values are uniform integers in the range `[minval, maxval)`.
The lower bound `minval` is included in the range, while the upper bound
`maxval` is excluded.
The random integers are slightly biased unless `maxval - minval` is an exact
power of two.  The bias is small for values of `maxval - minval` significantly
smaller than the range of the output (either `2^32` or `2^64`) . \n

*@par Inputs:
*@li x: The handle of the resource variable that stores the state of the RNG.
*@li algorithm: The RNG algorithm.
*@li shape: The shape of the output tensor.
*@li counts: A 0/1-D Tensor or Python value. The counts of the binomial
distribution.  Must be broadcastable with the leftmost dimension defined by `shape`.
*@li probs: A 0/1-D Tensor or Python value. The probability of success for the
binomial distribution.  Must be broadcastable with the leftmost dimension defined by `shape`.\n

*@par Attributes:
*dtype: required, type. \n

*@par Outputs:
*y:A Returns Random values with specified shape . \n

*@par Third-party framework compatibility
* Compatible with tensorflow StatefulRandomBinomial operator.
*/

REG_OP(StatefulRandomBinomial)
    .INPUT(x, TensorType({DT_RESOURCE}))
    .INPUT(algorithm, TensorType({DT_INT64}))
    .INPUT(shape, TensorType({DT_INT32}))
    .INPUT(counts, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(probs, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(StatefulRandomBinomial)

/**
*@brief Outputs random values from a normal distribution.
*The generated values will have mean 0 and standard deviation 1 . \n

*@par Inputs:
*@li x: The handle of the resource variable that stores the state of the RNG.
*@li algorithm: The RNG algorithm.
*@li shape: The shape of the output tensor . \n

*@par Outputs:
*y:A Returns A tensor of the specified shape filled with random normal values . \n

*@par Third-party framework compatibility
* Compatible with tensorflow StatefulStandardNormalV2 operator.
*/

REG_OP(StatefulStandardNormalV2)
    .INPUT(x, TensorType({DT_RESOURCE}))
    .INPUT(algorithm, TensorType({DT_INT64}))
    .INPUT(shape, TensorType({DT_INT32,DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(StatefulStandardNormalV2)

/**
*@brief Outputs random values from a truncated normal distribution.
*The generated values follow a normal distribution with mean 0 and standard
*deviation 1, except that values whose magnitude is more than 2 standard
*deviations from the mean are dropped and re-picked . \n

*@par Inputs:
*@li x: The handle of the resource variable that stores the state of the RNG.
*@li algorithm: The RNG algorithm.
*@li shape: The shape of the output tensor . \n

*@par Outputs:
*y:A Returns Random values with specified shape . \n

*@par Third-party framework compatibility
* Compatible with tensorflow StatefulTruncatedNormal operator.
*/

REG_OP(StatefulTruncatedNormal)
    .INPUT(x, TensorType({DT_RESOURCE}))
    .INPUT(algorithm, TensorType({DT_INT64}))
    .INPUT(shape, TensorType({DT_INT32,DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(StatefulTruncatedNormal)

/**
*@brief Outputs random values from a uniform distribution.
The generated values follow a uniform distribution in the range `[0, 1)`. The
lower bound 0 is included in the range, while the upper bound 1 is excluded.

*@par Inputs:
*@li x: The handle of the resource variable that stores the state of the RNG.
*@li algorithm: The RNG algorithm.
*@li shape: The shape of the output tensor . \n

*@par Outputs:
*y:A Returns Random values with specified shape . \n

*@par Third-party framework compatibility
* Compatible with tensorflow StatefulUniform operator.
*/

REG_OP(StatefulUniform)
    .INPUT(x, TensorType({DT_RESOURCE}))
    .INPUT(algorithm, TensorType({DT_INT64}))
    .INPUT(shape, TensorType({DT_INT32,DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(StatefulUniform)

/**
*@brief Outputs random integers from a uniform distribution.
The generated values are uniform integers covering the whole range of `dtype` . \n

*@par Inputs:
*@li x: The handle of the resource variable that stores the state of the RNG.
*@li algorithm: The RNG algorithm.
*@li shape: The shape of the output tensor . \n

*@par Outputs:
*y:A  Returns Random values with specified shape . \n

*@par Third-party framework compatibility
* Compatible with tensorflow StatefulUniformFullInt operator.
*/

REG_OP(StatefulUniformFullInt)
    .INPUT(x, TensorType({DT_RESOURCE}))
    .INPUT(algorithm, TensorType({DT_INT64}))
    .INPUT(shape, TensorType({DT_INT32,DT_INT64}))
    .OUTPUT(y, TensorType({DT_UINT64}))
    .OP_END_FACTORY_REG(StatefulUniformFullInt)

/**
*@brief Outputs random integers from a uniform distribution.
The generated values are uniform integers in the range `[minval, maxval)`.
The lower bound `minval` is included in the range, while the upper bound
`maxval` is excluded.
The random integers are slightly biased unless `maxval - minval` is an exact
power of two.  The bias is small for values of `maxval - minval` significantly
smaller than the range of the output (either `2^32` or `2^64`) . \n

*@par Inputs:
*@li x: The handle of the resource variable that stores the state of the RNG.
*@li algorithm: The RNG algorithm.
*@li shape: The shape of the output tensor.
*@li minval: Minimum value (inclusive, scalar).
*@li maxval: Maximum value (exclusive, scalar) . \n

*@par Outputs:
*y:A Returns Random values with specified shape . \n

*@par Third-party framework compatibility
* Compatible with tensorflow StatefulUniformInt operator.
*/

REG_OP(StatefulUniformInt)
    .INPUT(x, TensorType({DT_RESOURCE}))
    .INPUT(algorithm, TensorType({DT_INT64}))
    .INPUT(shape, TensorType({DT_INT32,DT_INT64}))
    .INPUT(minval, TensorType({DT_INT64}))
    .INPUT(maxval, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(StatefulUniformInt)

/**
* @brief Advance the counter of a counter-based RNG. The state of the RNG after
* `rng_skip(n)` will be the same as that after `stateful_uniform([n])`
* (or any other distribution). The actual increment added to the
* counter is an unspecified implementation detail . \n

* @par Inputs:
* @li value: Stores the state of the RNG.
* @li algorithm: The RNG algorithm.
* @li delta: The amount of advancement . \n

* @par Outputs:
* value:A Returns Random values with specified shape . \n

* @par Third-party framework compatibility
* Compatible with tensorflow RngReadAndSkipV2 operator.
*/

REG_OP(RngReadAndSkipV2)
    .INPUT(value, TensorType({DT_INT64}))
    .INPUT(algorithm, TensorType({DT_INT32}))
    .INPUT(delta, TensorType({DT_UINT64}))
    .OUTPUT(value, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(RngReadAndSkipV2)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_STATEFUL_RANDOM_OPS_H_