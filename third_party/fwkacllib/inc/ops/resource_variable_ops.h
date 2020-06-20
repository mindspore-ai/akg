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

#ifndef GE_OP_RESOURCE_VARIABLE_OPS_H
#define GE_OP_RESOURCE_VARIABLE_OPS_H

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

REG_OP(VarHandleOp)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .REQUIRED_ATTR(dtype, Type)
    .ATTR(shape, ListInt, ge::UNKNOWN_SHAPE)
    .OUTPUT(y, TensorType({DT_RESOURCE}))
    .OP_END_FACTORY_REG(VarHandleOp)

REG_OP(AssignVariableOp)
    .INPUT(resource, TensorType({DT_RESOURCE}))
    .INPUT(value, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(AssignVariableOp)

REG_OP(AssignAddVariableOp)
    .INPUT(resource, TensorType({DT_RESOURCE}))
    .INPUT(value, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(AssignAddVariableOp)

REG_OP(AssignSubVariableOp)
    .INPUT(resource, TensorType({DT_RESOURCE}))
    .INPUT(value, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(AssignSubVariableOp)

}  // namespace ge

#endif //GE_OP_RESOURCE_VARIABLE_OPS_H