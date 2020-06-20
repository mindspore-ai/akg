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

#ifndef GE_OP_SSDDETECTIONOUTPUT_OPS_H_
#define GE_OP_SSDDETECTIONOUTPUT_OPS_H_
#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Returns detection result.

*@par Inputs:
* Four inputs, including:
*@li mbox_conf: An ND tensor of type floa16 or float32, specifying the box confidences data, used as the input of operator SSDDetectionOutput.
*@li mbox_loc: An ND tensor of type floa16 or float32, specifying the box loc predictions, used as the input of operator SSDDetectionOutput.
*@li mbox_priorbox: An ND tensor of type floa16 or float32, output from operator PriorBoxD, used as the input of operator SSDDetectionOutput.
*@par Attributes:
*@li num_classes: An optional int32, specifying the number of classes to be predicted. Defaults to "2". The value must be greater than 1 and lesser than 1025.
*@li share_location: An option bool, specify the shared location. Defaults to True
*@li background_label_id: An option int32, specify the background label id. Must be 0
*@li nms_threshold: An option float32, specify the nms threshold
*@li top_k: An option int32, specify the topk value. Defaults to 200
*@li eta: An option float32, specify the eta value. Defaults to 1
*@li variance_encoded_in_target: An option bool, specify whether variance encoded in target or not. Defaults to False
*@li code_type: An option int32, specify the code type. Defaults to 1(only supports 2). The corner is 1, center_size is 2, corner_size is 3
*@li keep_top_k: An option int32, specify the topk value after nms. Defaults to -1
*@li confidence_threshold: An option float32, specify the topk filter threshold. Only consider detections with confidence greater than the threshold
*@li kernel_name: An optional string, specifying the operator name. Defaults to "ssd_detection_output".
*@par Outputs:
*out_boxnum: An NCHW tensor of type int32, specifying the number of output boxes.
*y: An NCHW tensor of type float16, describing the information of each output box, including the coordinates, class, and confidence.

*/
REG_OP(SSDDetectionOutput)
    .INPUT(bbox_delta, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(score, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(anchors, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(out_boxnum, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(num_classes, Int, 2)
    .ATTR(share_location, Bool, true)
    .ATTR(background_label_id, Int, 0)
    .ATTR(iou_threshold, Float, 0.3)
    .ATTR(top_k, Int, 200)
    .ATTR(eta, Float, 1.0)
    .ATTR(variance_encoded_in_target, Bool, false)
    .ATTR(code_type, Int, 1)
    .ATTR(keep_top_k, Int, -1)
    .ATTR(confidence_threshold, Float, 0.0)
    .OP_END_FACTORY_REG(SSDDetectionOutput)
}
#endif
