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

 #ifndef GE_OP_RPN_PROPOSAL_POST_PROCESSING_H
 #define GE_OP_RPN_PROPOSAL_POST_PROCESSING_H

 #include "graph/operator_reg.h"

namespace ge {
    REG_OP(RpnProposalPostProcessing)
    .INPUT(sorted_proposal, TensorType({DT_FLOAT16}))
    .INPUT(proposal_num, TensorType({DT_UINT32}))
    .OUTPUT(sorted_box, TensorType({ DT_FLOAT16}))
    .REQUIRED_ATTR(img_size, ListInt)
    .REQUIRED_ATTR(score_threshold, Float)
    .REQUIRED_ATTR(k, Int)
    .REQUIRED_ATTR(min_size, Float)
    .REQUIRED_ATTR(nms_threshold, Float)
    .REQUIRED_ATTR(post_nms_num, Int)
    .ATTR(box_filter, Bool, true)
    .ATTR(core_max_num, Int, 8)
    .OP_END_FACTORY_REG(RpnProposalPostProcessing)
    } // namespace ge

     #endif // GE_OP_GENERATE_RPN_PROPOSAL_POST_PROCESSING_H

