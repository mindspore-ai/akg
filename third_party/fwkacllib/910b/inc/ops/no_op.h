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
 * \file no_op.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NO_OP_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NO_OP_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief Does nothing. Only useful as a placeholder for control edges . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator NoOp.
*/

REG_OP(NoOp)
    .OP_END_FACTORY_REG(NoOp)

}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_NO_OP_H_
