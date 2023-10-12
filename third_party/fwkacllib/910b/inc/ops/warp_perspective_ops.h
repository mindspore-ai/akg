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
 * \file warp_perspective_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_WARP_PERSPECTIVE_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_WARP_PERSPECTIVE_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
*@brief Applies a perspective transformation to an image . \n

* @par Inputs:
* @li x: input tensor, format could be NHWC or NCHW, type could be float or uint8.
* @li matrix: transformation matrix, format ND , shape must be (N, 9),
type could be float or double. \n

* @par Attributes:
* @li out_height: output height, required.
* @li out_width: output width, required.
* @li interpolation_mode: interpolation method, support "bilinear" and "nearest",
defaults to "bilinear"
* @li border_type: border processing method, support "BORDER_CONSTANT" and "BORDER_REPLICATE",
default BORDER_CONSTANT.
* @li constant: border processed value, used when border_type is BORDER_CONSTANT.
* @li data_format: the data format of input tensor and output tensor, support "CHW" and "HWC",
defaults to "CHW" \n

* @par Outputs:
* @li y: output tensor, format could be NHWC or NCHW, type could be float or uint8.
*/

REG_OP(WarpPerspective)
    .INPUT(x, TensorType({DT_FLOAT, DT_UINT8}))
    .INPUT(matrix, TensorType({DT_DOUBLE, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_UINT8}))
    .REQUIRED_ATTR(out_height, Int)
    .REQUIRED_ATTR(out_width, Int)
    .ATTR(interpolation_mode, String, "bilinear")
    .ATTR(border_type, String, "BORDER_CONSTANT")
    .ATTR(constant, Float, 0)
    .ATTR(data_format, String, "CHW")
    .OP_END_FACTORY_REG(WarpPerspective)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_WARP_PERSPECTIVE_OPS_H_
