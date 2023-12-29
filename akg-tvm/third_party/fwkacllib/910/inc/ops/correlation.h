/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file correlation.h
 * \brief
 */
#ifndef GE_OP_CORRELATION_OPS_H
#define GE_OP_CORRELATION_OPS_H

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Computes a 2D Correlation given 4D "x" and "filter" tensors.
*
*@par Inputs:
* @li filter: A 4D tensor of filters.
* @li x: A 4D tensor of input images, batch number must equal to batch
* number of "filter", and channel must equal to channel of "filter".
*
*@par Attributes:
* @li groups: set correlation mode, must be 1 or channel.
*
*@par Outputs:
*y: A Tensor. Has the same type as "x".

*@par Third-party framework compatibility
* Compatible with caffe correlation custom operator.
*/
REG_OP(Correlation)
    .INPUT(filter, TensorType({DT_FLOAT16, DT_INT8}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT32}))
    .ATTR(groups, Int, 1)
    .OP_END_FACTORY_REG(Correlation)
}  // namespace ge

#endif  // GE_OP_NN_CALCULATION_OPS_H
