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
 * \file save_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_SAVE_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SAVE_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Mark which tensors need to be saved to the ckpt file.
*@par Inputs:
*tensors: A list of input tensor.It's a dynamic input.
*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(Save)
    .DYNAMIC_INPUT(tensors, TensorType:ALL())
    .OP_END_FACTORY_REG(Save)

} // namespace ge


#endif  // OPS_BUILT_IN_OP_PROTO_INC_SAVE_OPS_H_
