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
 * \file reduce.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_REDUCE_H_
#define OPS_BUILT_IN_OP_PROTO_INC_REDUCE_H_

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Insert a Cast node for the ReduceMean operator . \n

* @par Inputs:
* Two inputs, including:
*  @li x: A Tensor. Must be one of the following types: float16, float32, int8, uint8.
*  @li axes: The dimensions to reduce. Must be one of the following types: int, list, tuple, NoneType.
*    - If None (the default), reduces all dimensions.
*    - Must be in the range [-rank(x), rank(x)) . \n

* @par Attributes:
* keep_dims: A bool or NoneType.
*  - If true, retains reduced dimensions with length 1.
*  - If false, the rank of the tensor is reduced by 1 for each entry in axis.
* noop_with_empty_axes: A bool.
*  - If true, when axes = [], not reduce.
*  - If false, when axes = [], reduce all.
* dtype: enum.
*  - optional attr, could be one of the following types: DT_FLOAT16, DT_FLOAT, DT_INT8, DT_UINT8.
* @par Outputs:
* y: A Tensor. Has the same type as "x" . \n

* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator ReduceMeanWithCast.
*/
REG_OP(ReduceMeanWithCast)
    .INPUT(x, "T1")
    .INPUT(axes, "T2")
    .OUTPUT(y, "T3")
    .ATTR(keep_dims, Bool, false)
    .ATTR(noop_with_empty_axes, Bool, true)
    .ATTR(dtype, Type, DT_UNDEFINED)
    .DATATYPE(T1, TensorType::NumberType())
    .DATATYPE(T2, TensorType::IndexNumberType())
    .DATATYPE(T3, TensorType::NumberType())
    .OP_END_FACTORY_REG(ReduceMeanWithCast)
} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_REDUCE_H_
