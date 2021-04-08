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
 * \file condtake_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_CONDTAKE_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_CONDTAKE_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
*@brief Take elements from data if specific condition is satisfied on mask. \n

*@par Inputs:
*@li data: input tensor from which to take elements, High-dimension input would
first be flattened.
*@li mask: condition param; must be the same shape with data. \n

*@par Attributes:
*@li mode:convert by convert in Mode.
*@li val:convert by <class 'float'>
*@li eps:convert by <class 'float'> (default: 1e-06) \n

*@par Outputs:
*@li out_data: the elements taken
*@li out_index: the indices corresponding to those elements
*@li valid_num: elements of out_data and out_index from zeros to valid_num is valid.
*/

REG_OP(CondTake)
    .INPUT(data, TensorType({DT_FLOAT}))
    .INPUT(mask, TensorType({DT_FLOAT}))
    .OUTPUT(out_data, TensorType({DT_FLOAT}))
    .OUTPUT(out_index, TensorType({DT_INT32}))
    .OUTPUT(valid_num, TensorType({DT_INT32}))
    .REQUIRED_ATTR(mode, String)
    .REQUIRED_ATTR(val, Float)
    .ATTR(eps, Float, 1e-06)
    .OP_END_FACTORY_REG(CondTake)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_CONDTAKE_OPS_H_
