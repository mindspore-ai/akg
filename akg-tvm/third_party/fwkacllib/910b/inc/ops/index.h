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
 * \file index.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_INDEX_H_
#define OPS_BUILT_IN_OP_PROTO_INC_INDEX_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Tutel dispatch function in moe.
*
* @par Inputs:
* @li x: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.
* @li gates: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.
* @li indices: A mutable Tensor of the type DT_INT32, for topk's k size.
* @li locations: A mutable Tensor of the type DT_INT32, for token size.
*
* @par Attributes:
* capacity: expert capacity.
*
* @par Outputs:
* y: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.\n
*/
REG_OP(MoeTutelDispatch)
    .INPUT(x, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .INPUT(gates, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .INPUT(indices, TensorType({ DT_INT32 }))
    .INPUT(locations, TensorType({ DT_INT32 }))
    .OUTPUT(y, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .REQUIRED_ATTR(capacity, Int)
    .OP_END_FACTORY_REG(MoeTutelDispatch)

/**
* @brief Tutel combine function in moe.
*
* @par Inputs:
* @li y_grad: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.
* @li gates: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.
* @li indices: A mutable Tensor of the type DT_INT32, for topk's k size.
* @li locations: A mutable Tensor of the type DT_INT32, for token size.
*
* @par Outputs:
* @li x_grad: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.\n
*/
REG_OP(MoeTutelCombineX)
    .INPUT(y_grad, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .INPUT(gates, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .INPUT(indices, TensorType({ DT_INT32 }))
    .INPUT(locations, TensorType({ DT_INT32 }))
    .OUTPUT(x_grad, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .OP_END_FACTORY_REG(MoeTutelCombineX)

/**
* @brief Tutel combine function in moe.
*
* @par Inputs:
* @li x: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.
* @li y_grad: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.
* @li indices: A mutable Tensor of the type DT_INT32, for topk's k size.
* @li locations: A mutable Tensor of the type DT_INT32, for token size.
*
* @par Outputs:
* gates_grad: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.\n
*/
REG_OP(MoeTutelCombineGates)
    .INPUT(x, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .INPUT(y_grad, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .INPUT(indices, TensorType({ DT_INT32 }))
    .INPUT(locations, TensorType({ DT_INT32 }))
    .OUTPUT(gates_grad, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .OP_END_FACTORY_REG(MoeTutelCombineGates)
}
#endif  // OPS_BUILT_IN_OP_PROTO_INC_INDEX_H_
