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

#ifndef GE_OP_RAGGED_MATH_OPS_H
#define GE_OP_RAGGED_MATH_OPS_H

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Returns a `RaggedTensor` containing the specified sequences of numbers.

*@par Inputs:
*@li starts: The starts of each range.
*@li limits: The limits of each range.
*@li deltas: The deltas of each range.

*@par Outputs:
*y:A Returns The `row_splits` for the returned `RaggedTensor`.The `flat_values` for the returned `RaggedTensor`.

*@attention Constraints: \n
*The input tensors `starts`, `limits`, and `deltas` may be scalars or vectors. \n
*The vector inputs must all have the same size.  Scalar inputs are broadcast \n
*to match the size of the vector inputs.

*/

REG_OP(RaggedRange)
    .INPUT(starts, TensorType({DT_FLOAT,DT_DOUBLE,DT_INT32,DT_INT64}))
    .INPUT(limits, TensorType({DT_FLOAT,DT_DOUBLE,DT_INT32,DT_INT64}))
    .INPUT(deltas, TensorType({DT_FLOAT,DT_DOUBLE,DT_INT32,DT_INT64}))
    .OUTPUT(rt_nested_splits, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(rt_dense_values, TensorType({DT_FLOAT,DT_DOUBLE,DT_INT32,DT_INT64}))
    .REQUIRED_ATTR(Tsplits, Type)
    .OP_END_FACTORY_REG(RaggedRange)

}  // namespace ge

#endif //GE_OP_RAGGED_MATH_OPS_H