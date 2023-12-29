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
 * \file rpn_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_RPN_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_RPN_OPS_H_

#include "graph/operator_reg.h"
namespace ge {
/**
*@brief Iteratively removes lower scoring boxes which have an IoU greater than
* iou_threshold with higher scoring box according to their
* intersection-over-union (IoU) . \n

* @par Inputs:
* box_scores: 2-D tensor with shape of [N, 8], including proposal boxes and
* corresponding confidence scores . \n

* @par Attributes:
* iou_threshold: An optional float. The threshold for deciding whether boxes
* overlap too much with respect to IOU . \n

* @par Outputs:
* @li selected_boxes: 2-D tensor with shape of [N,5], representing filtered
* boxes including proposal boxes and corresponding confidence scores.
* @li selected_idx: 1-D tensor with shape of [N], representing the index of
* input proposal boxes.
* @li selected_mask: 1-D tensor with shape of [N], the symbol judging whether
* the output proposal boxes is valid . \n

* @attention Constraints:
* The 2nd-dim of input box_scores must be equal to 8.\n

*/
REG_OP(NMSWithMask)
    .INPUT(box_scores, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(selected_boxes, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(selected_idx, TensorType({DT_INT32}))
    .OUTPUT(selected_mask, TensorType({DT_UINT8}))
    .ATTR(iou_threshold, Float, 0.5)
    .OP_END_FACTORY_REG(NMSWithMask)

/**
* @brief Greedily selects a subset of bounding boxes in descending order of
* score . \n

* @par Inputs:
* Input boxes and scores must be float type. Inputs include:
* @li boxes: A 2-D float tensor of shape [num_boxes, 4]. They are expected to be in (x1, y1, x2, y2) format
* with 0 <= x1 < x2 and 0 <= y1 < y2.
* @li sorted_scores: A 1-D float tensor of shape [num_boxes] representing boxes' scores, which is sorted
* by descending order.
* @li input_indices: A 1-D integer tensor of shape [num_boxes] representing the indices for each row of
* boxes that would sort row of boxes by scores in descending order.
* @li max_output_size: A scalar integer tensor representing the maximum number
* of boxes to be selected by non max suppression.
* @li iou_threshold: A 0-D float tensor representing the threshold for deciding
* whether boxes overlap too much with respect to IOU.
* @li score_threshold: A 0-D float tensor representing the threshold for
* deciding when to remove boxes based on score . \n

* @par Attributes:
* offset: An optional int. Defaults to 0. \n

* @par Outputs:
* @li selected_indices: A 1-D integer tensor of shape [M] representing the selected
* indices from the boxes tensor, where M <= max_output_size . \n

* @attention Constraints:
* Input boxes and scores must be float type . \n
*/

REG_OP(SortedNMS)
    .INPUT(boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(sorted_scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input_indices, TensorType({DT_INT32}))
    .INPUT(max_output_size, TensorType({DT_INT32}))
    .INPUT(iou_threshold, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(score_threshold, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(selected_indices, TensorType({DT_INT32}))
    .ATTR(offset, Int, 0)
    .OP_END_FACTORY_REG(SortedNMS)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_RPN_OPS_H_
