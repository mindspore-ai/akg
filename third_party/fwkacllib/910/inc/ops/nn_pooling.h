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
 * \file nn_pooling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_POOLING_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_POOLING_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief Performs the backpropagation of ROI Pooling . \n

* @par Inputs:
* Five inputs, including:
* @li grad: A tensor of type float16 or float32, describing the gradient input.
* @li x: A tensor of type float16 or float32, describing the feature
* map.
* @li rois: A tensor of type float16 or float32, with 2D shape
* [batch, 5], describing the RIOs. Each ROI consists of five
* elements: "batch_id", "x1", "y1", "x2", and "y2", which "batch_id" indicates
* the index of the input feature map, "x1", "y1", "x2", or "y2" must be
* greater than or equal to "0.0".
* @li roi_actual_num: A  optional tensor of type int32, specifying
* the number of ROIs per batch.
* @li argmax: A tensor of type int32, describing the index of grad. \n

* @par Attributes:
* @li pooled_h: A required int32, specifying the pooled H. Must be greater
* than 0.
* @li pooled_w: A required int32, specifying the pooled W. Must be greater
* than 0.
* @li spatial_scale_h: An required scaling factor for mapping the input
* coordinates of height to the ROI coordinates.
* @li spatial_scale_w: An required scaling factor for mapping the input
* coordinates of width to the ROI coordinates .
* @li pool_channel: A required int32, secifying the pooling channel. \n

* @par Outputs:
* @li y: A tensor of type float16 or float32, describing the result. \n

* @attention Constraints:
* "pool_channel" only support equal to the channel of "x".
* "roi_actual_num" only support equal to the number of "rois". \n

* @par Third-party framework compatibility
* It has a corresponding operator in MMCV.
*/
REG_OP(RoiPoolingGradWithArgMax)
    .INPUT(grad, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(rois, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OPTIONAL_INPUT(roi_actual_num, TensorType({DT_INT32}))
    .INPUT(argmax, TensorType({DT_INT32}))
    .REQUIRED_ATTR(pooled_h, Int)
    .REQUIRED_ATTR(pooled_w, Int)
    .REQUIRED_ATTR(spatial_scale_h, Float)
    .REQUIRED_ATTR(spatial_scale_w, Float)
    .REQUIRED_ATTR(pool_channel, Int)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(RoiPoolingGradWithArgMax)
}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_POOLING_H_
