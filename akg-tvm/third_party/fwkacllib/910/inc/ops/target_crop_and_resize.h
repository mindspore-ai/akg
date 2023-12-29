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
 * \file target_crop_and_resize.h
 * \brief
 */
#ifndef GE_OP_TARGET_CROP_AND_RESIZE_H
#define GE_OP_TARGET_CROP_AND_RESIZE_H

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Performs crop and resize on images.

*@par Inputs:
*@li x: An NCHW tensor of type uint8, specifying the input to the data layer.
*@li boxes: Crop parameters of type int32. \n
*@li box_index: Batch index parameters of type int32. The batch of the input x to be cropped and resize. \n

*@par Attributes:
*output_h: A required int, specifying the height of output. \n
*output_w: A required int, specifying the width of output. \n
*input_format: A required string, specifying the input format. \n

*@par Outputs:
*y: The output tensor of type uint8.
*@par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe.
*
*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(TargetCropAndResize)
    .INPUT(x, TensorType({DT_UINT8}))
    .INPUT(boxes, TensorType({DT_INT32}))
    .INPUT(box_index, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_UINT8}))
    .ATTR(output_h, Int, 224)
    .ATTR(output_w, Int, 224)
    .ATTR(input_format, String, "YUV420SP_U8")
    .OP_END_FACTORY_REG(TargetCropAndResize)
}
#endif //GE_OP_TARGET_CROP_AND_RESIZE_H
