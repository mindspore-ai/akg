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
 * \file resource_variable_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_RESOURCE_VARIABLE_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_RESOURCE_VARIABLE_OPS_H_

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Creates a handle to a Variable resource. \n

*@par Outputs:
*y:A Tensor of type resource. \n

*@par Attributes:
* @li container: optional, string. the container this 
variable is placed in.
* @li shared_name: optional, string.the name by which
 this variable is referred to.
* @li dtype: required, type. the output of type.
* @li shape: optional, ListInt. the output of shape. \n

*@see VarHandleOp.
*/

REG_OP(VarHandleOp)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .REQUIRED_ATTR(dtype, Type)
    .ATTR(shape, ListInt, ge::UNKNOWN_SHAPE)
    .OUTPUT(y, TensorType({DT_RESOURCE}))
    .OP_END_FACTORY_REG(VarHandleOp)

/**
*@brief Assigns a new value to a variable. \n

*@par Inputs:
*@li resource:Handle to the resource in which to store the variable.
*@li value:The value to set the new tensor to use. \n

*@par Attributes:
* dtype: required, type. \n

*@see AssignVariableOp.
*/

REG_OP(AssignVariableOp)
    .INPUT(resource, TensorType({DT_RESOURCE}))
    .INPUT(value, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(AssignVariableOp)

/**
*@brief Adds a value to the current value of a variable. \n

*@par Inputs:
*@li resource:Handle to the resource in which to store the variable.
*@li value:The value by which the variable will be incremented. \n

*@par Attributes:
* dtype: required, type. \n

*@see AssignAddVariableOp.
*/

REG_OP(AssignAddVariableOp)
    .INPUT(resource, TensorType({DT_RESOURCE}))
    .INPUT(value, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(AssignAddVariableOp)

/**
*@brief Subtracts a value to the current value of a variable. \n

*@par Inputs:
*@li resource:Handle to the resource in which to store the variable.
*@li value:The value by which the variable will be incremented. \n

*@par Attributes:
* dtype: required, type. \n

*@see AssignSubVariableOp.
*/

REG_OP(AssignSubVariableOp)
    .INPUT(resource, TensorType({DT_RESOURCE}))
    .INPUT(value, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(AssignSubVariableOp)

}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_RESOURCE_VARIABLE_OPS_H_