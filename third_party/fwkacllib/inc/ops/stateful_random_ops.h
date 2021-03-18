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

#ifndef GE_OP_STATEFUL_RANDOM_OPS_H
#define GE_OP_STATEFUL_RANDOM_OPS_H

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Non-deterministically generates some integers.

*@par Inputs:
*This op may use some OS-provided source of non-determinism (e.g. an RNG), \n
*so each execution will give different results. Inputs included:
*@li shape: The shape of the output tensor.

*@par Outputs:
*y:A Returns Non-deterministic integer values with specified shape.

*/

REG_OP(NonDeterministicInts)
    .INPUT(shape, TensorType({DT_INT32,DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32,DT_INT64}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(NonDeterministicInts)

/**
*@brief Advance the counter of a counter-based RNG. The state of the RNG after \n
*`rng_skip(n)` will be the same as that after `stateful_uniform([n])` \n
*(or any other distribution). The actual increment added to the \n
*counter is an unspecified implementation detail.

*@par Inputs:
*@li resource: The handle of the resource variable that stores the state of the RNG.
*@li algorithm: The RNG algorithm.
*@li delta: The amount of advancement.

*@par Outputs:
*y:A Returns the created operation.

*/

REG_OP(RngSkip)
    .INPUT(x, TensorType({DT_RESOURCE}))
    .INPUT(algorithm, TensorType({DT_INT64}))
    .INPUT(delta, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(RngSkip)

/**
*@brief Outputs random integers from a uniform distribution. \n
The generated values are uniform integers in the range `[minval, maxval)`. \n
The lower bound `minval` is included in the range, while the upper bound \n
`maxval` is excluded. \n
The random integers are slightly biased unless `maxval - minval` is an exact \n
power of two.  The bias is small for values of `maxval - minval` significantly \n
smaller than the range of the output (either `2^32` or `2^64`).

*@par Inputs:
*@li resource: The handle of the resource variable that stores the state of the RNG.
*@li algorithm: The RNG algorithm.
*@li shape: The shape of the output tensor.
*@li minval: Minimum value (inclusive, scalar).
*@li maxval: Maximum value (exclusive, scalar).

*@par Outputs:
*y:A Returns Random values with specified shape.

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
*@brief Outputs random values from a normal distribution. \n
*The generated values will have mean 0 and standard deviation 1.

*@par Inputs:
*@li resource: The handle of the resource variable that stores the state of the RNG.
*@li algorithm: The RNG algorithm.
*@li shape: The shape of the output tensor.

*@par Outputs:
*y:A Returns A tensor of the specified shape filled with random normal values.

*/

REG_OP(StatefulStandardNormalV2)
    .INPUT(x, TensorType({DT_RESOURCE}))
    .INPUT(algorithm, TensorType({DT_INT64}))
    .INPUT(shape, TensorType({DT_INT32,DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(StatefulStandardNormalV2)

/**
*@brief Outputs random values from a truncated normal distribution. \n
*The generated values follow a normal distribution with mean 0 and standard \n
*deviation 1, except that values whose magnitude is more than 2 standard \n
*deviations from the mean are dropped and re-picked.

*@par Inputs:
*@li resource: The handle of the resource variable that stores the state of the RNG.
*@li algorithm: The RNG algorithm.
*@li shape: The shape of the output tensor.

*@par Outputs:
*y:A Returns Random values with specified shape.

*/

REG_OP(StatefulTruncatedNormal)
    .INPUT(x, TensorType({DT_RESOURCE}))
    .INPUT(algorithm, TensorType({DT_INT64}))
    .INPUT(shape, TensorType({DT_INT32,DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(StatefulTruncatedNormal)

/**
*@brief Outputs random values from a uniform distribution. \n
The generated values follow a uniform distribution in the range `[0, 1)`. The \n
lower bound 0 is included in the range, while the upper bound 1 is excluded. \n

*@par Inputs:
*@li resource: The handle of the resource variable that stores the state of the RNG.
*@li algorithm: The RNG algorithm.
*@li shape: The shape of the output tensor.

*@par Outputs:
*y:A Returns Random values with specified shape.

*/

REG_OP(StatefulUniform)
    .INPUT(x, TensorType({DT_RESOURCE}))
    .INPUT(algorithm, TensorType({DT_INT64}))
    .INPUT(shape, TensorType({DT_INT32,DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(StatefulUniform)

/**
*@brief Outputs random integers from a uniform distribution. \n
The generated values are uniform integers covering the whole range of `dtype`.

*@par Inputs:
*@li resource: The handle of the resource variable that stores the state of the RNG.
*@li algorithm: The RNG algorithm.
*@li shape: The shape of the output tensor.

*@par Outputs:
*y:A  Returns Random values with specified shape.

*/

REG_OP(StatefulUniformFullInt)
    .INPUT(x, TensorType({DT_RESOURCE}))
    .INPUT(algorithm, TensorType({DT_INT64}))
    .INPUT(shape, TensorType({DT_INT32,DT_INT64}))
    .OUTPUT(y, TensorType({DT_UINT64}))
    .OP_END_FACTORY_REG(StatefulUniformFullInt)

/**
*@brief Outputs random integers from a uniform distribution. \n
The generated values are uniform integers in the range `[minval, maxval)`. \n
The lower bound `minval` is included in the range, while the upper bound \n
`maxval` is excluded. \n
The random integers are slightly biased unless `maxval - minval` is an exact \n
power of two.  The bias is small for values of `maxval - minval` significantly \n
smaller than the range of the output (either `2^32` or `2^64`).

*@par Inputs:
*@li resource: The handle of the resource variable that stores the state of the RNG.
*@li algorithm: The RNG algorithm.
*@li shape: The shape of the output tensor.
*@li minval: Minimum value (inclusive, scalar).
*@li maxval: Maximum value (exclusive, scalar).

*@par Outputs:
*y:A Returns Random values with specified shape.

*/

REG_OP(StatefulUniformInt)
    .INPUT(x, TensorType({DT_RESOURCE}))
    .INPUT(algorithm, TensorType({DT_INT64}))
    .INPUT(shape, TensorType({DT_INT32,DT_INT64}))
    .INPUT(minval, TensorType({DT_INT64}))
    .INPUT(maxval, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(StatefulUniformInt)

}  // namespace ge

#endif //GE_OP_STATELESS_RANDOM_OPS_H