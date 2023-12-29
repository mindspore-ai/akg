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
 * \file avg_pool_1d_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_AVGPOOL1DOPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_AVGPOOL1DOPS_H_
#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Generate an auxiliary matrix .  \n

*@par Inputs:
* @li x: A tensor. Must be one of the following types:uint8, int8,int16, int32,
 int64, float16, float, double.The format must be NHWC/NCHW.

*@par Attributes:
*@li ksize: Kernel size. Input type is int.
*@li strides: Input type is int.
*@li pads: Input type is listInt .
*@li ceil_mode: Bool, default value is false.
*@li count_include_pad: Bool, default value is false.  \n

*@par Outputs:
*y_tensor: A  tensor with the same types as "x" .  \n
*@par Third-party framework compatibility

*Compatible with the TensorFlow operator Unbatch.
*/
REG_OP(AvgPool1DAvgMatrix)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT8,
                          DT_INT32, DT_INT64, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT8,
                           DT_INT32, DT_INT64, DT_DOUBLE}))
    .REQUIRED_ATTR(ksize, Int)
    .REQUIRED_ATTR(strides, Int)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(ceil_mode, Bool, false)
    .ATTR(count_include_pad, Bool, false)
    .OP_END_FACTORY_REG(AvgPool1DAvgMatrix)
}
#endif