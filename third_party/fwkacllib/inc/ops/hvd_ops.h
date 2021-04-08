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
 * \file hvd_ops.h
 * \brief Horovod collective communication library ops.
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_HVD_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_HVD_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Outputs a tensor gathering all input tensors.
 * @par Inputs:
 * x: A tensor. Must be one of the following types: uint8, int8, uint16, int16, int32,
 int64, float16, bool.
 * @par Attributes:
 * @li rank_size: A required integer identifying the number of ranks
 participating in the op.
 * @par Outputs:
 * y: A Tensor. Has the same type as "x".
 */
REG_OP(HorovodAllgather)
    // GE not support float64 currently
    .INPUT(x, TensorType({DT_UINT8, DT_INT8, DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_UINT8, DT_INT8, DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_BOOL}))
    // add rank_size attr
    .REQUIRED_ATTR(rank_size, Int)
    .OP_END_FACTORY_REG(HorovodAllgather)

/**
 * @brief Outputs a tensor containing the reduction across all input tensors
 passed to op.
 * @par Inputs:
 * x: A tensor. Must be one of the following types: int32, int64, float16, float32
 @par Attributes:
 * @li reduce_op: A required int identifying the reduction operation to
 perform.The supported operation are: "sum", "max", "min", "prod".
 * @par Outputs:
 * y: A Tensor. Has the same type as "x".
 */
REG_OP(HorovodAllreduce)
    .INPUT(x, TensorType({DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(reduce_op, Int)
    .OP_END_FACTORY_REG(HorovodAllreduce)

/**
 * @brief Broadcasts the input tensor in root rank to all ranks.
 * @par Inputs:
 * x: A list of dynamic input tensor. Must be one of the following types:
 int8, int32, float16, float32.
 * @par Attributes:
 * @li root_rank: A required integer identifying the root rank in the op
 input of this rank will be broadcast to other ranks.
 * @par Outputs:
 * y: A list of dynamic output tensor. Has the same type and length as "x".
 */
REG_OP(HorovodBroadcast)
    .INPUT(x, TensorType({DT_UINT8, DT_INT8, DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_UINT8, DT_INT8, DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_BOOL}))
    .REQUIRED_ATTR(root_rank, Int)
    .OP_END_FACTORY_REG(HorovodBroadcast)

} // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_HVD_OPS_H_
