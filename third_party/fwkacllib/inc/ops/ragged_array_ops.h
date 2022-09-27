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
 * \file ragged_array_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_RAGGED_ARRAY_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_RAGGED_ARRAY_OPS_H_

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Gather ragged slices from `params` axis `0` according to `indices` . \n

*@par Inputs:
*@li params_nested_splits: The `nested_row_splits` tensors that define the row-partitioning for the
*params` RaggedTensor input. It's a dynamic input.
*@li params_dense_values: The `flat_values` for the `params` RaggedTensor. There was a terminology change
*at the python level from dense_values to flat_values, so dense_values is the
*deprecated name.
*@li indices: Indices in the outermost dimension of `params` of the values that should be
*gathered.

*@par Attributes:
*@li PARAMS_RAGGED_RANK:The ragged rank of the params_nested_splits.
*@li Tsplits:A type of output_nested_splits.
*@li OUTPUT_RAGGED_RANK: The ragged rank of the output RaggedTensor. `output_nested_splits` will contain
*this number of `row_splits` tensors. This value should equal
*`indices.shape.ndims + params.ragged_rank - 1` . \n

*@par Outputs:
*@li output_nested_splits:A Returns The `nested_row_splits` tensors that define the row-partitioning for the
*returned RaggedTensor.The `flat_values` for the returned RaggedTensor . 
*@li output_dense_values:The `flat_values` for the returned RaggedTensor. \n

*@par Third-party framework compatibility
* Compatible with tensorflow RaggedGather operator.
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

#endif  // OPS_BUILT_IN_OP_PROTO_INC_RAGGED_ARRAY_OPS_H_