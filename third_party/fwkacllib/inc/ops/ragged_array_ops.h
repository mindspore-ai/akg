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

#ifndef GE_OP_RAGGED_ARRAY_OPS_H
#define GE_OP_RAGGED_ARRAY_OPS_H

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Gather ragged slices from `params` axis `0` according to `indices`.

*@par Inputs:
*@li params_nested_splits: The `nested_row_splits` tensors that define the row-partitioning for the \n
*params` RaggedTensor input.
*@li params_dense_values: The `flat_values` for the `params` RaggedTensor. There was a terminology change \n
*at the python level from dense_values to flat_values, so dense_values is the \n
*deprecated name.
*@li indices: Indices in the outermost dimension of `params` of the values that should be \n
*gathered.
*@li OUTPUT_RAGGED_RANK: The ragged rank of the output RaggedTensor. `output_nested_splits` will contain \n
*this number of `row_splits` tensors. This value should equal \n
*`indices.shape.ndims + params.ragged_rank - 1`.

*@par Outputs:
*y:A Returns The `nested_row_splits` tensors that define the row-partitioning for the \n
*returned RaggedTensor.The `flat_values` for the returned RaggedTensor.

*/

REG_OP(RaggedGather)
    .DYNAMIC_INPUT(params_nested_splits, TensorType({DT_INT32, DT_INT64}))
    .INPUT(params_dense_values, TensorType({DT_INT32, DT_INT64}))
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .DYNAMIC_OUTPUT(output_nested_splits, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(output_dense_values, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(Tsplits, Type)
    .ATTR(PARAMS_RAGGED_RANK, Int, 1)
    .ATTR(OUTPUT_RAGGED_RANK, Int, 0)
    .OP_END_FACTORY_REG(RaggedGather)

}  // namespace ge

#endif //GE_OP_RAGGED_ARRAY_OPS_H