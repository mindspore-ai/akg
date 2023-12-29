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
 * \file arithmetic.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_ARITHMETIC_H_
#define OPS_BUILT_IN_OP_PROTO_INC_ARITHMETIC_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Computes the logarithm of the sum of exponentiations of the inputs element-wise. y = ln(e^x1 + e^x2). \n
*
* @par Inputs:
* Two inputs, including:
* @li x1: A Tensor. Must be one of the following types: bfloat16, float16, float32, double, complex64, complex128.
* @li x2: A Tensor. Must be the same type and shape as "x1". \n
*
* @par Attributes:
* @li base: An optional attribute of type float32, specifying the base gamma. Defaults to "-1.0".
* @li scale: An optional attribute of type float32, specifying the scale alpha. Defaults to "1.0".
* @li shift: An optional attribute of type float32, specifying the shift beta. Defaults to "0.0". \n
*
*
* @par Outputs:
* y: A Tensor of the same type as "x1". \n

* @par Third-party framework compatibility
* Compatible with pytorch operator LogAddExp.
*/
REG_OP(LogAddExp)
    .INPUT(x1, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(x2, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(base, Float, -1.0)
    .ATTR(scale, Float, 1.0)
    .ATTR(shift, Float, 0.0)
    .OP_END_FACTORY_REG(LogAddExp)
}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_ARITHMETIC_H_
