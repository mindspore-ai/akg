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

#ifndef GE_OP_NN_LOSS_SCALE_OPS_H
#define GE_OP_NN_LOSS_SCALE_OPS_H
#include "graph/operator_reg.h"

namespace ge {
REG_OP(NPUAllocFloatStatusOperator)
    .OUTPUT(data, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(NPUAllocFloatStatusOperator)

REG_OP(NPUClearFloatStatusOperator)
    .INPUT(addr, TensorType{DT_FLOAT})
    .OUTPUT(data, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(NPUClearFloatStatusOperator)

REG_OP(NPUGetFloatStatusOperator)
    .INPUT(addr, TensorType{DT_FLOAT})
    .OUTPUT(data, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(NPUGetFloatStatusOperator)

/**
*@brief Produces a variable with 0 in memory.

*@par Outputs:
*y: A Tensor of type int32, output eight numbers with a value of zero.
*/
REG_OP(NPUAllocFloatStatus)
    .OUTPUT(data, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(NPUAllocFloatStatus)

/**
*@brief Set the value of address 0x40000 to 0 in each core.

*@par Inputs:
*@li addr: A tensor of type float32.

*@par Outputs:
*data: A Tensor of type float32.
*/
REG_OP(NPUClearFloatStatus)
    .INPUT(addr, TensorType{DT_FLOAT})
    .OUTPUT(data, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(NPUClearFloatStatus)

/**
*@brief Get the value of address 0x40000.

*@par Inputs:
*@li addr: A tensor of type float32.

*@par Outputs:
*data: A Tensor of type float32.
*/
REG_OP(NPUGetFloatStatus)
    .INPUT(addr, TensorType{DT_FLOAT})
    .OUTPUT(data, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(NPUGetFloatStatus)
}  // namespace ge

#endif  // GE_OP_NN_LOSS_SCALE_OPS_H
