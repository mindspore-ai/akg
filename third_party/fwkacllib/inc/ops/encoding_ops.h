/*
 * Copyright (c) Huawei Technologies Co., Ltd 2022-2022. All rights reserved.
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
 * \file encoding_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_ENCODING_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_ENCODING_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief An op to decode indices for LDPC code. \n

* @par Inputs:
* @li valid_num: an int32 tensor indicates index limit for each line.
* @li matrix_info: an int32 2D-tensor store the block indices info of connection H matrix. \n

* @par Outputs:
* indices: an int32 2D-tensor store the concrete indices value.
*
* @par Restrictions:
* Warning:THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/

REG_OP(LDPCDecode)
    .INPUT(valid_num, TensorType({DT_INT32}))
    .INPUT(matrix_info, TensorType({DT_INT32}))
    .OUTPUT(indices, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(LDPCDecode)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_ENCODING_OPS_H_
