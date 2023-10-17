/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file nn_detect.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_DETECT_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_DETECT_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief Performs the backpropagation of ROI Align Rotated . \n

* @par Inputs:
* @li x: A tensor of type float32, describing the feature_map.
* @li rois: A tensor of type float32, with shape(n, 6) with each roi decoded as 
  (batch_index, center_x, center_y, w, h, angle).
* The angle is in radian.

* @par Attributes:
* @li pooled_h: A required int32, specifying the pooled H. Must be greater
* than 0.
* @li pooled_w: A required int32, specifying the pooled W. Must be greater
* than 0.
* @li spatial_scale: An required scaling factor for mapping the input coordinates
* to the ROI coordinates.
* @li sampling_ratio: An required number of inputs samples to take for each output sample. 
* 0 to take samples densely for current models.
* @li aligned: A required bool, if False, use the legacy implementation.
* If True, align the results more perfectly. Default: True.
* @li clockwise: A required bool, if True, the angle in each proposal follows a clockwise 
  fashion in image space,
* Otherwise, the angle is counterclockwise. Default: False. \n

* @par Outputs:
* @li y: A tensor of type float32, describing the result. \n

* @par Third-party framework compatibility
* It has a corresponding operator in MMCV.
*/
REG_OP(RoiAlignRotated)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(rois, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(pooled_h, Int)
    .REQUIRED_ATTR(pooled_w, Int)
    .REQUIRED_ATTR(spatial_scale, Float)
    .ATTR(sampling_ratio, Int, 0)
    .ATTR(aligned, Bool, true)
    .ATTR(clockwise, Bool, false)
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(RoiAlignRotated)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_DETECT_H_
