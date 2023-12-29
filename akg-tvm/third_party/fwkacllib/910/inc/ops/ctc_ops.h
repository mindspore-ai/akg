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
 * \file ctc_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_CTC_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_CTC_OPS_H_

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Calculates the CTC Loss (log probability) for each batch entry.
Also calculates the gradient. \n

*@par Inputs:
*@li inputs: 3-D, shape: `(max_time x batch_size x num_classes)`, the logits.
*@li labels_indices: The indices of a `SparseTensor<int32, 2>`.
`labels_indices(i, :) == [b, t]` means `labels_values(i)` stores the id for
`(batch b, time t)`.
*@li labels_values: The values (labels) associated with the given batch and time.
*@li sequence_length: A vector containing sequence lengths (batch). \n

*@par Outputs:
*@li loss: A vector (batch) containing log-probabilities.
*@li gradient: The gradient of `loss`.  3-D, shape: `(max_time x
batch_size x num_classes)`. \n

*@par Attributes:
*@li preprocess_collapse_repeated: Scalar, if true then repeated labels are collapsed prior to
the CTC calculation.If not specified, defaults to false
*@li ctc_merge_repeated: Scalar. If set to false, *during* CTC calculation
repeated non-blank labels will not be merged and are interpreted as
individual labels.  This is a simplified version of CTC.
If not specified, defaults to true. \n

*@par Third-party framework compatibility
* Compatible with TensorFlow CTCLoss operator.
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

/**
*@brief Performs greedy decoding on the logits given in inputs. \n

*@par Inputs:
*@li inputs: 3-D, shape: `(max_time x batch_size x num_classes)`, the logits.
*@li sequence_length: A vector containing sequence lengths, size `(batch_size)`. \n

*@par Attributes:
* merge_repeated: If True, merge repeated classes in output. \n

*@par Outputs:
*@li decoded_indices: Indices matrix, size `(total_decoded_outputs x 2)`,
of a `SparseTensor<int64, 2>`.  The rows store: [batch, time].
*@li decoded_values: Values vector, size: `(total_decoded_outputs)`,
of a `SparseTensor<int64, 2>`.  The vector stores the decoded classes.
*@li decoded_shape: Shape vector, size `(2)`, of the decoded SparseTensor.
Values are: `[batch_size, max_decoded_length]`.
*@li log_probability: Matrix, size `(batch_size x 1)`, containing sequence
log-probabilities. \n

*@par Third-party framework compatibility
* Compatible with TensorFlow CTCGreedyDecoder operator.
*/
REG_OP(CTCGreedyDecoder)
    .INPUT(inputs, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(sequence_length, TensorType({DT_INT32}))
    .ATTR(merge_repeated, Bool, false)
    .OUTPUT(decoded_indices, TensorType({DT_INT64}))
    .OUTPUT(decoded_values, TensorType({DT_INT64}))
    .OUTPUT(decoded_shape, TensorType({DT_INT64}))
    .OUTPUT(log_probability, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(CTCGreedyDecoder)

/**
*@brief Performs beam search decoding on the logits given in input. \n

*@par Inputs:
*@li inputs: 3-D, shape: `(max_time x batch_size x num_classes)`, the logits.
*@li sequence_length: A vector containing sequence lengths, size `(batch_size)`. \n

*@par Attributes:
*@li merge_repeated: If True, merge repeated classes in output. \n
*@li beam_width:A scalar >= 0 (beam search beam width).
*@li top_paths:A scalar >= 0, <= beam_width (controls output size).

*@par Outputs:
*@li decoded_indices: A list (length: top_paths) of indices matrices.  Matrix j,
size `(total_decoded_outputs[j] x 2)`, has indices of a
`SparseTensor<int64, 2>`.  The rows store: [batch, time].
*@li decoded_values: A list (length: top_paths) of values vectors.  Vector j,
size `(length total_decoded_outputs[j])`, has the values of a
`SparseTensor<int64, 2>`.  The vector stores the decoded classes for beam j.
*@li decoded_shape: A list (length: top_paths) of shape vector.  Vector j,
size `(2)`, stores the shape of the decoded `SparseTensor[j]`.
Its values are: `[batch_size, max_decoded_length[j]]`.
*@li log_probability: A matrix, shaped: `(batch_size x top_paths)`.  The
sequence log-probabilities. \n

*@par Third-party framework compatibility
* Compatible with TensorFlow CTCBeamSearchDecoder operator.
*/
REG_OP(CTCBeamSearchDecoder)
    .INPUT(inputs, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(sequence_length, TensorType({DT_INT32}))
    .REQUIRED_ATTR(beam_width, Int)
    .REQUIRED_ATTR(top_paths, Int)
    .ATTR(merge_repeated, Bool, true)
    .DYNAMIC_OUTPUT(decoded_indices, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(decoded_values, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(decoded_shape, TensorType({DT_INT64}))
    .OUTPUT(log_probability, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(CTCBeamSearchDecoder)

/**
*@brief The Connectionist Temporal Classification loss.

*@par Inputs:
*@li log_probs: Tensor of size (T, N, C), where T =input length, N =batch size,
                and C = number of classes (including blank).
                It represent the logarithmized probabilities of the outputs.
*@li targets: Tensor of size (N, S) or sum(target_lengths), where S = max target length.
             It represent the target sequences.
*@li input_lengths: Tuple or tensor of size (N). It represent the lengths of the inputs.
*@li target_lengths: Tuple or tensor of size (N). It represent lengths of the targets.

*@par Outputs:
*@li neg_log_likelihood: A loss value which is differentiable with respect to each input node.
*@li log_alpha: The probability of possible trace of input to target.

*@par Attributes:
*@li blank: Blank label. Default 0.
*@li reduction: Specifies the reduction to apply to the output. Default: 'mean'.
*@li zero_infinity: Whether to zero infinite losses and the associated gradients.

* @par Third-party framework compatibility:
* Compatible with Pytorch CTCLoss operator.

*@attention Constraints:
* The limit of Label’s length is 1K.
*/
REG_OP(CTCLossV2)
    .INPUT(log_probs, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(targets, TensorType({DT_INT32, DT_INT64}))
    .INPUT(input_lengths, TensorType({DT_INT32, DT_INT64}))
    .INPUT(target_lengths, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(neg_log_likelihood, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(log_alpha, TensorType({DT_FLOAT, DT_DOUBLE}))
    .ATTR(blank, Int, 0)
    .ATTR(reduction, String, "mean")
    .ATTR(zero_infinity, Bool, false)
    .OP_END_FACTORY_REG(CTCLossV2)

/**
*@brief The Connectionist Temporal Classification loss grad.

* @par Inputs:
*@li grad_out: Gradient renewal coefficient. Tensor of size (N), where N = batch size.
* @li log_probs: Tensor of size (T, N, C), where T =input length, N =batch size,
                and C = number of classes (including blank).
                It represent the logarithmized probabilities of the outputs.
*@li targets: Tensor of size (N, S) or sum(target_lengths), where S = max target length.
             It represent the target sequences.
* @li input_lengths: Tuple or tensor of size (N). It represent the lengths of the inputs.
*@li target_lengths: Tuple or tensor of size (N). It represent lengths of the targets.
* @li neg_log_likelihood: A loss value which is differentiable with respect to each input node.
* @li log_alpha: The probability of possible trace of input to target.

* @par Outputs:
*@li grad: Tensor of size (T, N, C), The grad of Connectionist Temporal Classification loss.

* @par Attributes:
*@li blank: Blank label. Default 0.
* @li reduction: Specifies the reduction to apply to the output. Default: 'mean'.
* @li zero_infinity: Whether to zero infinite losses and the associated gradients.

* @par Third-party framework compatibility:
* Compatible with Pytorch CTCLoss operator.

* @attention Constraints:
* The limit of Label’s length is 1K.
*/
REG_OP(CTCLossV2Grad)
    .INPUT(grad_out, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(log_probs, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(targets, TensorType({DT_INT32, DT_INT64}))
    .INPUT(input_lengths, TensorType({DT_INT32, DT_INT64}))
    .INPUT(target_lengths, TensorType({DT_INT32, DT_INT64}))
    .INPUT(neg_log_likelihood, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(log_alpha, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(grad, TensorType({DT_FLOAT, DT_DOUBLE}))
    .ATTR(blank, Int, 0)
    .ATTR(reduction, String, "mean")
    .ATTR(zero_infinity, Bool, false)
    .OP_END_FACTORY_REG(CTCLossV2Grad)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_CTC_OPS_H_