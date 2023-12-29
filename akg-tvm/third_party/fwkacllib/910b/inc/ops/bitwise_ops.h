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
 * \file bitwise_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_BITWISE_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_BITWISE_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Element-wise computes the bitwise left-shift of x and y . \n

*@par Inputs:
*Input "x" is a k-dimensional tensor. Inputs "num_lower" and "num_upper"
are 0D scalars.
* @li x: A Tensor. Must be one of the following types: int8, int16, int32,
int64, uint8, uint16, uint32, uint64.
* @li y: A Tensor. Has the same type as "x".  \n

*@par Outputs:
* z: A Tensor. Has the same type as "x".  \n

*@attention Constraints:
*Unique runs on the Ascend AI CPU, which delivers poor performance.  \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator LeftShift.
*/

REG_OP(LeftShift)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, \
           DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .INPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, \
           DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .OUTPUT(z, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, \
            DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .OP_END_FACTORY_REG(LeftShift)

/**
*@brief Element-wise computes the bitwise right-shift of x and y . \n

*@par Inputs:
*Input "x" is a k-dimensional tensor. Inputs "num_lower" and "num_upper"
are 0D scalars.
* @li x: A Tensor. Must be one of the following types: int8, int16, int32,
int64, uint8, uint16, uint32, uint64.
* @li y: A Tensor. Has the same type as "x".  \n

*@par Outputs:
* z: A Tensor. Has the same type as "x".  \n

*@attention Constraints:
*Unique runs on the Ascend AI CPU, which delivers poor performance.  \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator RightShift.
*/

REG_OP(RightShift)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, \
           DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .INPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, \
           DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .OUTPUT(z, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, \
            DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .OP_END_FACTORY_REG(RightShift)

}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_BITWISE_OPS_H_
