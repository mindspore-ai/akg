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

#ifndef GE_OP_SWAP_CO_OPS_H_
#define GE_OP_SWAP_CO_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Folds the convolution input weight constant of the preceding layer \n
* of PSROIPooling to convert the N dimension of the weight from \n
* (output_dim, group_size*group_size) to \n
* (group_size*group_size, int((output_dim+15)/C0)*C0).
*@see PSROIPooling

*@par Inputs:
* One input:
*x: An NCHW tensor of type float16 or float32, describing the weight of\n
* convolution. Dim N must equal output_dim*group_size*group_size.

*@par Attributes:
*@li output_dim: A required int32, specifying the number of output channels.\n
* Must be greater than "0".
*@li group_size: A required int32, specifying the number of groups to encode\n
* position-sensitive score maps. Must be within the range (0, 128).

*@par Outputs:
*y: An NCHW tensor of type float16 or float32, describing the result weight\n
* of convolution.
*/

REG_OP(SwapCo)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(output_dim, Int, 0)
    .ATTR(group_size, Int, 0)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(SwapCo)

}  // namespace ge

#endif  // GE_OP_SWAP_CO_OPS_H_
