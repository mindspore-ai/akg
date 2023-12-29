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
 * \file npu_loss_scale_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NPU_LOSS_SCALE_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NPU_LOSS_SCALE_OPS_H_
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Computes NPU alloc float status operator function . \n

*@par Outputs:
*data: A Tensor of data value. Must be float32.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(NPUAllocFloatStatusOperator)
    .OUTPUT(data, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(NPUAllocFloatStatusOperator)

/**
*@brief Computes NPU clear float status operator function . \n

*@par Inputs:
*addr: A Tensor of data memory address. Must be float32 . \n

*@par Outputs:
*data: A Tensor of data value. Must be float32.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(NPUClearFloatStatusOperator)
    .INPUT(addr, TensorType{DT_FLOAT})
    .OUTPUT(data, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(NPUClearFloatStatusOperator)

/**
*@brief Computes NPU get float status operator function . \n

*@par Inputs:
*addr: A Tensor of data memory address. Must be float32 . \n

*@par Outputs:
*data: A Tensor of data value. Must be float32.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(NPUGetFloatStatusOperator)
    .INPUT(addr, TensorType{DT_FLOAT})
    .OUTPUT(data, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(NPUGetFloatStatusOperator)

/**
*@brief Produces a variable with 0 in memory . \n

*@par Outputs:
*y: A Tensor of type int32, output eight numbers with a value of zero.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(NPUAllocFloatStatus)
    .OUTPUT(data, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(NPUAllocFloatStatus)

/**
*@brief Set the value of address 0x40000 to 0 in each core . \n

*@par Inputs:
*addr: A tensor of type float32 . \n

*@par Outputs:
*data: A Tensor of type float32.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(NPUClearFloatStatus)
    .INPUT(addr, TensorType{DT_FLOAT})
    .OUTPUT(data, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(NPUClearFloatStatus)

/**
*@brief Get the value of address 0x40000 . \n

*@par Inputs:
*addr: A tensor of type float32 . \n

*@par Outputs:
*data: A Tensor of type float32.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(NPUGetFloatStatus)
    .INPUT(addr, TensorType{DT_FLOAT})
    .OUTPUT(data, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(NPUGetFloatStatus)


/**
*@brief Set the value of global workspace to 0. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(NPUClearFloatStatusV2)
    .OP_END_FACTORY_REG(NPUClearFloatStatusV2)

/**
*@brief Set the value of global workspace to 0. \n

*@par Inputs:
*addr: A nested structure of Tensors of type float32 . \n

*@par Outputs:
*data: A Tensor of type float32.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(NPUGetFloatStatusV2)
    .OUTPUT(data, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(NPUGetFloatStatusV2)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_NPU_LOSS_SCALE_OPS_H_
