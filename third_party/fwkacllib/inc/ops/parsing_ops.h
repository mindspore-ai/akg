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

#ifndef GE_OP_PARSING_OPS_H
#define GE_OP_PARSING_OPS_H

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief Converts each string in the input Tensor to the specified numeric type.

*@par Inputs:
*Inputs include: \n
*x: A Tensor. Must be one of the following types: string.

*@par Attributes:
*out_type: The numeric type to interpret each string in string_tensor as.

*@par Outputs:
*y: A Tensor. Has the same type as x.

*@attention Constraints:\n
*-The implementation for StringToNumber on Ascend uses AICPU, with bad performance.\n

*/
REG_OP(StringToNumber)
    .INPUT(x, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .ATTR(out_type, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(StringToNumber)

}  // namespace ge

#endif  // GE_OP_PARSING_OPS_H
