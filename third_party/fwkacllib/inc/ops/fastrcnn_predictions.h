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

 #ifndef GE_OP_FASTRCNN_PREDICTIONS_H
 #define GE_OP_FASTRCNN_PREDICTIONS_H

 #include "graph/operator_reg.h"

 namespace ge {

 REG_OP(FastrcnnPredictions)
     .INPUT(rois, TensorType({DT_FLOAT16}))
     .INPUT(score, TensorType({DT_FLOAT16}))
     .REQUIRED_ATTR(nms_threshold, Float)
     .REQUIRED_ATTR(score_threshold, Float)
     .REQUIRED_ATTR(k, Int)
     .OUTPUT(sorted_rois, TensorType({DT_FLOAT16}))
     .OUTPUT(sorted_scores, TensorType({DT_FLOAT16}))
     .OUTPUT(sorted_classes, TensorType({DT_FLOAT16}))
     .OP_END_FACTORY_REG(FastrcnnPredictions)
 } // namespace ge

 #endif // GE_OP_FASTRCNN_PREDICTIONS_H
