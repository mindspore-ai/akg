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

#ifndef GE_OP_CTC_OPS_H
#define GE_OP_CTC_OPS_H

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Calculates the CTC Loss (log probability) for each batch entry. \n
Also calculates the gradient. 

*@par Inputs:
*@li inputs: 3-D, shape: `(max_time x batch_size x num_classes)`, the logits.
*@li labels_indices: The indices of a `SparseTensor<int32, 2>`. \n
`labels_indices(i, :) == [b, t]` means `labels_values(i)` stores the id for \n
`(batch b, time t)`.
*@li labels_values: The values (labels) associated with the given batch and time.
*@li sequence_length: A vector containing sequence lengths (batch).

*@par Outputs:
*@li loss: A vector (batch) containing log-probabilities.
*@li gradient: The gradient of `loss`.  3-D, shape: `(max_time x \n
batch_size x num_classes)`.

*@par Attributes:
*@li preprocess_collapse_repeated: Scalar, if true then repeated labels are collapsed prior to \n
the CTC calculation.If not specified, defaults to false
*@li ctc_merge_repeated: Scalar. If set to false, *during* CTC calculation \n
repeated non-blank labels will not be merged and are interpreted as \n
individual labels.  This is a simplified version of CTC. \n
If not specified, defaults to true

*/

REG_OP(CTCLoss)
    .INPUT(inputs, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(labels_indices, TensorType({DT_INT64}))
    .INPUT(labels_values, TensorType({DT_INT32}))
    .INPUT(sequence_length, TensorType({DT_INT32}))
    .OUTPUT(loss, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(gradient, TensorType({DT_FLOAT, DT_DOUBLE}))
    .ATTR(preprocess_collapse_repeated, Bool, false)
    .ATTR(ctc_merge_repeated, Bool, true)
    .ATTR(ignore_longer_outputs_than_inputs, Bool, false)
    .OP_END_FACTORY_REG(CTCLoss)

}  // namespace ge

#endif //GE_OP_CTC_OPS_H