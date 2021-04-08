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
 * \file stateless_random_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_STATELESS_RANDOM_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_STATELESS_RANDOM_OPS_H_

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Draws samples from a multinomial distribution . \n

*@par Inputs:
include:
*@li logits:2-D Tensor with shape [batch_size, num_classes]. Each slice [i, :]
*represents the unnormalized log probabilities for all classes.
*@li num_samples:0-D. Number of independent samples to draw for each row slice.
*@li seed:The seed to generate random . \n

*@par Attributes:
*output_dtype:Output data type . \n

*@par Outputs:
*y:Output random number . \n

*@see StatelessMultinomial()

*@par Third-party framework compatibility
*compatible with StatelessMultinomial op of tensorflow
*/
REG_OP(StatelessMultinomial)
    .INPUT(logits, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE}))
    .INPUT(num_samples, TensorType({DT_INT32}))
    .INPUT(seed, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(output_dtype, Type, DT_INT64)
    .OP_END_FACTORY_REG(StatelessMultinomial)

/**
*@brief Outputs deterministic pseudorandom random integers from a uniform distribution . \n

*@par Inputs:
*@li shape: The shape of the output tensor.
*@li seed: 2 seeds (shape [2]).
*@li minval: Minimum value (inclusive, scalar).
*@li maxval: Maximum value (exclusive, scalar) . \n

*@par Outputs:
*y: Returns Random values with specified shape . \n

*@par Third-party framework compatibility
* Compatible with TensorFlow StatelessRandomUniformInt operator.
*/

REG_OP(StatelessRandomUniformInt)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(seed, TensorType({DT_INT32, DT_INT64}))
    .INPUT(minval, TensorType({DT_INT32, DT_INT64}))
    .INPUT(maxval, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(StatelessRandomUniformInt)

}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_STATELESS_RANDOM_OPS_H_