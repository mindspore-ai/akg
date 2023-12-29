/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file randomdsa_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_RANDOMDSA_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_RANDOMDSA_OPS_H_

#include <vector>
#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief Generate DSA random bit mask for dropout. \n

* @par Inputs:
* @li count:The shape of the input tensor. Must be int64.
* @li seed:If seed is set to be non-zero, the random number
* generator is seeded by the given seed. Otherwise, it is seeded by a random seed. Must be int64.
* @li dropout:0-D. Number of bit 1. Must be one of the following dtypes:float16, float32, bf16.\n

* @par Attributes:
* @li random_algorithm:The default value is "Philox".
* @li output_dtype:The dtype of output. The default value is "uint1". \n

* @par Outputs:
* y:If the dtype of y is uint8, output (1-D) random number using uint8 data format. Else if the dtype of y is uint1,
* output random number with the shape of count using uint1 data format.
* Must be one of the following types:uint1, uint8.\n

* @see DSAGenBitMask()
*/
REG_OP(DSAGenBitMask)
    .INPUT(count, TensorType({DT_INT64}))
    .INPUT(seed, TensorType({DT_UINT64}))
    .INPUT(dropout, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(out, TensorType({DT_UINT1, DT_UINT8}))
    .ATTR(random_algorithm, String, "Philox")
    .ATTR(output_dtype, String, "uint8")
    .OP_END_FACTORY_REG(DSAGenBitMask)

/**
* @brief Generate DSA truncatenormal data in random. \n

* @par Inputs:
include:
* @li count: The shape of the input tensor.
* @li seed: If seed is set to be non-zero, the random number
* generator is seeded by the given seed. Otherwise, it is seeded by a random seed
* @li mean: A Tensor. Must be one of the following types: float16, float32, double
* @li stdev: A Tensor. Must be one of the following types: float16, float32, double. \n

* @par Attributes:
* @li random_algorithm:The default value is "Philox". \n

* @par Outputs:
* y:Output (1-D) random number using float and bf data format . \n

* @see DSARandomTruncatedNormal()
*/
REG_OP(DSARandomTruncatedNormal)
    .INPUT(count, TensorType({DT_INT64}))
    .INPUT(seed, TensorType({DT_UINT64}))
    .INPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(stdev, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(out, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .ATTR(random_algorithm, String, "Philox")
    .OP_END_FACTORY_REG(DSARandomTruncatedNormal)

/**
* @brief Generate DSA stateless truncatenormal data in random. \n

* @par Inputs:
include:
* @li count: The shape of the input tensor.
* @li seed: If seed is set to be non-zero, the random number
* generator is seeded by the given seed. Otherwise, it is seeded by a random seed
* @li mean: A Tensor. Must be one of the following types: float16, float32, double
* @li stdev: A Tensor. Must be one of the following types: float16, float32, double. \n
* @li counter: Counter of RNG algorithm. Shape[2] for philox, shape[1] for threefry. \n

* @par Attributes:
* @li random_algorithm:The default value is "Philox". \n

* @par Outputs:
* y:Output (1-D) random number using float and bf data format . \n

* @see DSAStatelessRandomTruncatedNormal()
*/
REG_OP(DSAStatelessRandomTruncatedNormal)
    .INPUT(count, TensorType({DT_INT64}))
    .INPUT(seed, TensorType({DT_UINT64}))
    .INPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(stdev, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(counter, TensorType({DT_UINT64}))
    .OUTPUT(out, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .ATTR(random_algorithm, String, "Philox")
    .OP_END_FACTORY_REG(DSAStatelessRandomTruncatedNormal)

/**
* @brief Generate DSA normal data in random. \n

* @par Inputs:
include:
* @li count: The shape of the input tensor.
* @li seed: If seed is set to be non-zero, the random number
* generator is seeded by the given seed. Otherwise, it is seeded by a random seed
* @li mean: A Tensor. Must be one of the following types: float16, float32, double
* @li stdev: A Tensor. Must be one of the following types: float16, float32, double. \n

* @par Attributes:
* @li random_algorithm:The default value is "Philox". \n

* @par Outputs:
* y:Output (1-D) random number using float and bf data format . \n

* @see DSARandomNormal()
*/
REG_OP(DSARandomNormal)
    .INPUT(count, TensorType({DT_INT64}))
    .INPUT(seed, TensorType({DT_UINT64}))
    .INPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(stdev, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(out, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .ATTR(random_algorithm, String, "Philox")
    .OP_END_FACTORY_REG(DSARandomNormal)

/**
* @brief Generate State less DSA dropout data in random. \n

* @par Inputs:
include:
* @li count: The shape of the input tensor.
* @li seed: If seed is set to be non-zero, the random number
* generator is seeded by the given seed. Otherwise, it is seeded by a random seed
* @li dropout: A Tensor. Must be one of the following types: float16, float32, bfloat16
* @li offset: A Tensor. Must be one of the following types: uint64. \n

* @par Attributes:
* @li random_algorithm:The default value is "Philox". \n

* @par Outputs:
* y:Output (1-D) random number using float and bf data format . \n

* @see DSAStatelessGenBitMask()
*/
REG_OP(DSAStatelessGenBitMask)
    .INPUT(count, TensorType({DT_INT64}))
    .INPUT(seed, TensorType({DT_UINT64}))
    .INPUT(dropout, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(offset, TensorType({DT_UINT64}))
    .OUTPUT(out, TensorType({DT_UINT1, DT_UINT8}))
    .ATTR(random_algorithm, String, "Philox")
    .ATTR(output_dtype, String, "uint8")
    .OP_END_FACTORY_REG(DSAStatelessGenBitMask)

/**
 * @brief Generate State less DSA normal data in random. \n

    * @par Inputs:
    include:
    * @li count: The shape of the input tensor.
    * @li seed: If seed is set to be non-zero, the random number
    * generator is seeded by the given seed. Otherwise, it is seeded by a random seed
    * @li mean: A Tensor, value should be 0. Must be one of the following types: float16, float32, bfloat16.
    * @li stdev: A Tensor, value should be 1. Must be one of the following types: float16, float32, bfloat16.
    * @li counter: A Tensor. Must be one of the following types: uint64. \n

    * @par Attributes:
    * @li random_algorithm:The default value is "Philox". \n

    * @par Outputs:
    * y:Output (1-D) random number using float and bf data format . \n

    * @see DSAStatelessRandomNormal()
    */
REG_OP(DSAStatelessRandomNormal)
    .INPUT(count, TensorType({DT_INT64}))
    .INPUT(seed, TensorType({DT_UINT64}))
    .INPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(stdev, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(counter, TensorType({DT_UINT64}))
    .OUTPUT(out, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(random_algorithm, String, "Philox")
    .OP_END_FACTORY_REG(DSAStatelessRandomNormal)

/**
* @brief Generate State less DSA uniform data in random. \n

* @par Inputs:
include:
* @li count: The shape of the input tensor.
* @li seed: If seed is set to be non-zero, the random number
* generator is seeded by the given seed. Otherwise, it is seeded by a random seed
* @li low: A Tensor, value should be 0.. Must be one of the following types: float16, float32, bfloat16.
* @li high: A Tensor, value should be 1.. Must be one of the following types: float16, float32, bfloat16.
* @li counter: A Tensor. Must be one of the following types: uint64. \n

* @par Attributes:
* @li random_algorithm:The default value is "Philox". \n

* @par Outputs:
* y:Output (1-D) random number using float and bf data format . \n

* @see DSAStatelessRandomUniform()
*/
REG_OP(DSAStatelessRandomUniform)
    .INPUT(count, TensorType({DT_INT64}))
    .INPUT(seed, TensorType({DT_UINT64}))
    .INPUT(low, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64}))
    .INPUT(high, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64}))
    .INPUT(counter, TensorType({DT_UINT64}))
    .OUTPUT(out, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64}))
    .ATTR(random_algorithm, String, "Philox")
    .OP_END_FACTORY_REG(DSAStatelessRandomUniform)

/**
* @brief Generate DSA uniform data in random. \n

* @par Inputs:
include:
* @li count: The shape of the input tensor.
* @li seed: If seed is set to be non-zero, the random number
* generator is seeded by the given seed. Otherwise, it is seeded by a random seed
* @li low: A Tensor. Must be one of the following types: int, float, bf
* @li high: A Tensor. Must be one of the following types: int, float, bf. \n

* @par Attributes:
* @li random_algorithm:The default value is "Philox". \n

* @par Outputs:
* y:Output (1-D) random number using float int and bf data format . \n

* @see DSARandomUniform()
*/
REG_OP(DSARandomUniform)
    .INPUT(count, TensorType({DT_INT64}))
    .INPUT(seed, TensorType({DT_UINT64}))
    .INPUT(low, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64}))
    .INPUT(high, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64}))
    .OUTPUT(out, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64}))
    .ATTR(random_algorithm, String, "Philox")
    .OP_END_FACTORY_REG(DSARandomUniform)
}
#endif  // OPS_BUILT_IN_OP_PROTO_INC_RANDOMDSA_OPS_H
