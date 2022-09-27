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
 * \file nn_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_OPS_H_
#include "graph/operator_reg.h"
#include "nn_pooling_ops.h"

namespace ge {
/**
* @brief Says whether the targets are in the top "k" predictions . \n

* @par Inputs:
* Three inputs, including:
* @li predictions: A 2D Tensor of type float32. A "batch_size * classes" tensor.
* @li targets: A 1D Tensor of type IndexNumberType. A batch_size tensor of class ids.
* @li k: A 1D Tensor of the same type as "targets".
* Specifies the number of top elements to look at for computing precision . \n

* @par Outputs:
* precision: A Tensor of type bool . \n

* @attention Constraints:
* @li targets must be non-negative tensor.

* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator InTopKV2.
*/
REG_OP(InTopKV2)
    .INPUT(predictions, TensorType({DT_FLOAT}))
    .INPUT(targets, TensorType(IndexNumberType))
    .INPUT(k, TensorType({IndexNumberType}))
    .OUTPUT(precision, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(InTopKV2)

/**
*@brief Performs batch normalization . \n

*@par Inputs:
* Five inputs, including: (NHWC, NCHW supported)
*@li x: A 4D or 5D Tensor of type float16 or float32, with format NHWC or NCHW for 4D.
*@li scale: A Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW.
Specifies the scaling factor.
*@li offset: A Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW.
Specifies the offset.
*@li mean: A Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW.
Specifies the mean used for inference. Must be "None" if the
operation is used for training.
*@li variance: A Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW.
Specifies the variance used for inference. Must be "None"
if the operation is used for training . \n

*@par Attributes:
*@li epsilon: An optional float32, specifying the small value added to variance to avoid dividing by zero. Defaults to "0.0001".
*@li data_format: An optional string, specifying the format of "x". Defaults to "NHWC".
*@li is_training: An optional bool, specifying if the operation is used for training or inference. Defaults to "True" . \n

*@par Outputs:
* Five outputs, including: (NHWC, NCHWsupported)
*@li y: A 4D or 5D Tensor of type float16 or float32 for the normalized "x", with format NHWC or NCHW for 4D.
*@li batch_mean: A Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW.
Specifies the mean of "x".
*@li batch_variance: A Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW.
pecifies the variance of "x".
*@li reserve_space_1: An optional Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW.
Specifies the mean of "x" for gradient computation. Pass "None" to skip this output.
*@li reserve_space_2: An optional Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW.
Specifies the variance of "x" for gradient computation. Pass "None" to skip this output . \n

*@attention Constraints:
*@li If the operation is used for inference and outputs "reserve_space_1" and "reserve_space_2" are available,
then "reserve_space_1" has the same value as "mean" and "reserve_space_2" has the same value as "variance".
*@li For Ascend 310, the result accuracy fails to reach 1‰ due to the square root instruction . \n
*/
REG_OP(FusedBatchNormV2)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_1, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_2, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001)
    .ATTR(data_format, String, "NHWC")
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(FusedBatchNormV2)

/**
 * @brief Large amount of data sort.First operator of TopK.
 * @par Inputs:
 * two input, including:
 * @li input_data: A Tensor. Data to be sorted. Support float16 or float32.
 * @li input_index: A Tensor. Range(0, 2048). Support float16 or int32.
 * @par Attributes:
 * @li k_num: Int.Number to be sorted.
 * @li largest: An optional bool, controls whether to return largest or smallest elements. Defaults to true.
 * If "True", the "k" largest elements are returned in descending order.
 * If "False", the "k" smallest elements are returned in ascending order.
 * @par Outputs:
 * One output, including:
 * output_proposal: A Tensor. Datatype and format is same as input_data. Proposal sorted for each channel.
 * @par Restrictions:
 * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
 */
REG_OP(SegmentSort)
    .INPUT(input_data, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input_index, TensorType({DT_FLOAT16,DT_INT32}))
    .OUTPUT(output_proposal, TensorType({DT_FLOAT16,DT_FLOAT}))
    .REQUIRED_ATTR(k_num, Int)
    .ATTR(largest, Bool, true)
    .OP_END_FACTORY_REG(SegmentSort)

/**
 * @brief: Large amount of data sort.Second operator of TopK.
 * @par Inputs:
 * One input, including:
 * input_proposal: A Tensor. Proposal sorted for each channel. Support float16 or float32
 * @par Attributes:
 * @li k_num: Int.Number to be sorted.
 * @li include_index: Bool.include_index is false,output proposal. include_index is true, output data and index.
 * @li largest: An optional bool, controls whether to return largest or smallest elements. Defaults to true.
 * If "True", the "k" largest elements are returned in descending order.
 * If "False", the "k" smallest elements are returned in ascending order.
 * @par Outputs:
 * Two output, including:
 * output_proposal: A Tensor. Datatype and format is same as input_data. Proposal sorted for each channel.
 * output_index: A Tensor.If include_index is true, output index.
 * @par Restrictions:
 * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
 */
REG_OP(MultiMerge)
    .INPUT(input_proposal, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(output_proposal, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(output_index, TensorType({DT_INT32}))
    .REQUIRED_ATTR(k_num, Int)
    .ATTR(include_index, Bool, false)
    .ATTR(largest, Bool, true)
    .OP_END_FACTORY_REG(MultiMerge)

/**
 * @brief Large amount of data sort.Third operator of TopK.
 * @par Inputs:
 * One input, including:
 * input_proposal: A Tensor. Proposal sorted for each channel. Support float16
 * @par Attributes:
 * @li k_num: Int.Number to be sorted.
 * @li largest: An optional bool, controls whether to return largest or smallest elements. Defaults to true.
 * If "True", the "k" largest elements are returned in descending order.
 * If "False", the "k" smallest elements are returned in ascending order.
 * @par Outputs:
 * Two output, including:
 * @li output_data: A Tensor. Datatype and format is same as input_data. Data sorted.
 * @li output_index: A Tensor. int32. Data index.
 * @par Restrictions:
 * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
 */
REG_OP(SingleMerge)
    .INPUT(input_proposal, TensorType({ DT_FLOAT16 }))
    .OUTPUT(output_data, TensorType({ DT_FLOAT16 }))
    .OUTPUT(output_index, TensorType({ DT_INT32 }))
    .REQUIRED_ATTR(k_num, Int)
    .ATTR(largest, Bool, true)
    .OP_END_FACTORY_REG(SingleMerge)

/**
 * @brief MultiHeadAttention.
 * @par Inputs:
 * thirteen input, including:
 * @li query: A Tensor. Query of Attention. Support float16
 * @li key: A Tensor. Key of Attention. Support float16
 * @li value: A Tensor. Value of Attention. Support float16
 * @li query_weight: A Tensor. QueryWeight of Attention. Support float16
 * @li key_weight: A Tensor. KeyWeight of Attention. Support float16
 * @li value_weight: A Tensor. ValueWeight of Attention. Support float16
 * @li attn_mask: A Tensor. AttentionMask of Attention. Support float16
 * @li out_proj_weight: A Tensor. OutProjWeight of Attention. Support float16
 * @li query_bias: Optional Tensor. QueryBias of Attention. Support float16
 * @li key_bias: Optional Tensor. KeyBias of Attention. Support float16
 * @li value_bias: Optional Tensor. ValueBias of Attention. Support float16
 * @li out_proj_bias: Optional Tensor. OutProjBias of Attention. Support float16
 * @li dropout_mask_input: Optional Tensor. DropOutMask of Attention. Support uint8 \n

 * @par Attributes:
 * @li attn_head_num: Attention Head numbers, Support int
 * @li attn_dim_per_head: Attention dim of a Head, Support int
 * @li src_len: source length, Support int
 * @li tgt_len: target length, Support int
 * @li keep_prob: dropout keep probability, Support float
 * @li softmax_use_float: SoftMax Use Float32 to keep precision, Support bool \n

 * @par Outputs:
 * Eight output, including:
 * @li y: A Tensor. Result of Attention. Support float16
 * @li dropout_mask: DropOutMask of Attention. Support uint8
 * @li query_res: Query Result of Attention. Support float16
 * @li key_res: Key Result of Attention. Support float16
 * @li value_res: Value Result of Attention. Support float16
 * @li attn_scores: Attention Scores of SoftMax. Support float16, float
 * @li attn_res: Attention Result of SoftMax. Support float16
 * @li context: Context of Attention. Support float16

 * @par Restrictions:
 * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
 */
REG_OP(MultiHeadAttention)
    .INPUT(query, TensorType({DT_FLOAT16}))
    .INPUT(key, TensorType({DT_FLOAT16}))
    .INPUT(value, TensorType({DT_FLOAT16}))
    .INPUT(query_weight, TensorType({DT_FLOAT16}))
    .INPUT(key_weight, TensorType({DT_FLOAT16}))
    .INPUT(value_weight, TensorType({DT_FLOAT16}))
    .INPUT(attn_mask, TensorType({DT_FLOAT16}))
    .INPUT(out_proj_weight, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(query_bias, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(key_bias, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(value_bias, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(out_proj_bias, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(dropout_mask_input, TensorType({DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .OUTPUT(dropout_mask, TensorType({DT_UINT8}))
    .OUTPUT(query_res, TensorType({DT_FLOAT16}))
    .OUTPUT(key_res, TensorType({DT_FLOAT16}))
    .OUTPUT(value_res, TensorType({DT_FLOAT16}))
    .OUTPUT(attn_scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(attn_res, TensorType({DT_FLOAT16}))
    .OUTPUT(context, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(attn_head_num, Int)
    .REQUIRED_ATTR(attn_dim_per_head, Int)
    .REQUIRED_ATTR(src_len, Int)
    .REQUIRED_ATTR(tgt_len, Int)
    .REQUIRED_ATTR(keep_prob, Float)
    .REQUIRED_ATTR(softmax_use_float, Bool)
    .OP_END_FACTORY_REG(MultiHeadAttention)

/**
 * @brief MultiHeadAttentionGrad.
 * @par Inputs:
 * thirteen input, including:
 * @li query: A Tensor. Query of Attention. Support float16
 * @li key: A Tensor. Key of Attention. Support float16
 * @li value: A Tensor. Value of Attention. Support float16
 * @li query_weight: A Tensor. QueryWeight of Attention. Support float16
 * @li key_weight: A Tensor. KeyWeight of Attention. Support float16
 * @li value_weight: A Tensor. ValueWeight of Attention. Support float16
 * @li out_proj_weight: A Tensor. OutProjWeight of Attention. Support float16
 * @li query_res: A Tensor. Query Result of Attention. Support float16
 * @li key_res: A Tensor. Key Result of Attention. Support float16
 * @li value_res: A Tensor. Value Result of Attention. Support float16
 * @li attn_scores: A Tensor. Attention Scores of Attention. Support float16, float
 * @li attn_res: A Tensor. Attention Result of Attention. Support float16
 * @li context: A Tensor. Context of Attention. Support float16
 * @li y_grad: A Tensor. Grad of Attention. Support float16
 * @li dropout_mask: : A Tensor. Query Result of Attention. Support uint8 \n

 * @par Attributes:
 * @li attn_head_num: Attention Head numbers, Support int
 * @li attn_dim_per_head: Attention dim of a Head, Support int
 * @li src_len: source length, Support int
 * @li tgt_len: target length, Support int
 * @li keep_prob: dropout keep probability, Support float
 * @li softmax_use_float: SoftMax Use Float32 to keep precision, Support bool
 * @li bias_grad_mask: mask for attention has bias grad, Support list bool  \n

 * @par Outputs:
 * Eight output, including:
 * @li query_weight_grad: QueryWeight Grad of Attention. Support float16
 * @li key_weight_grad: KeyWeight Grad of Attention. Support float16
 * @li value_weight_grad: ValueWeight Grad of Attention. Support float16
 * @li out_proj_weight_grad: OutProjWeight Grad of Attention. Support float16
 * @li query_grad: Query Grad of Attention. Support float16
 * @li key_grad: Key Grad of Attention. Support float16
 * @li value_grad: Value Grad of Attention. Support float16
 * @li query_bias_grad: QueryBias Grad of Attention. Support float16
 * @li key_bias_grad: KeyBias Grad of Attention. Support float16
 * @li value_bias_grad: ValueBias Grad of Attention. Support float16
 * @li out_proj_bias_grad: OutProjBias Grad of Attention. Support float16

 * @par Restrictions:
 * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
 */
REG_OP(MultiHeadAttentionGrad)
    .INPUT(query, TensorType({DT_FLOAT16}))
    .INPUT(key, TensorType({DT_FLOAT16}))
    .INPUT(value, TensorType({DT_FLOAT16}))
    .INPUT(query_weight, TensorType({DT_FLOAT16}))
    .INPUT(key_weight, TensorType({DT_FLOAT16}))
    .INPUT(value_weight, TensorType({DT_FLOAT16}))
    .INPUT(out_proj_weight, TensorType({DT_FLOAT16}))
    .INPUT(query_res, TensorType({DT_FLOAT16}))
    .INPUT(key_res, TensorType({DT_FLOAT16}))
    .INPUT(value_res, TensorType({DT_FLOAT16}))
    .INPUT(attn_scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(attn_res, TensorType({DT_FLOAT16}))
    .INPUT(context, TensorType({DT_FLOAT16}))
    .INPUT(y_grad, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(dropout_mask, TensorType({DT_UINT8}))
    .OUTPUT(query_weight_grad, TensorType({DT_FLOAT16}))
    .OUTPUT(key_weight_grad, TensorType({DT_UINT8}))
    .OUTPUT(value_weight_grad, TensorType({DT_FLOAT16}))
    .OUTPUT(out_proj_weight_grad, TensorType({DT_FLOAT16}))
    .OUTPUT(query_grad, TensorType({DT_FLOAT16}))
    .OUTPUT(key_grad, TensorType({DT_FLOAT16}))
    .OUTPUT(value_grad, TensorType({DT_FLOAT16}))
    .OUTPUT(query_bias_grad, TensorType({DT_FLOAT16}))
    .OUTPUT(key_bias_grad, TensorType({DT_FLOAT16}))
    .OUTPUT(value_bias_grad, TensorType({DT_FLOAT16}))
    .OUTPUT(out_proj_bias_grad, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(attn_head_num, Int)
    .REQUIRED_ATTR(attn_dim_per_head, Int)
    .REQUIRED_ATTR(src_len, Int)
    .REQUIRED_ATTR(tgt_len, Int)
    .REQUIRED_ATTR(keep_prob, Float)
    .REQUIRED_ATTR(softmax_use_float, Bool)
    .REQUIRED_ATTR(bias_grad_mask, ListBool)
    .OP_END_FACTORY_REG(MultiHeadAttentionGrad)
}// namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_OPS_H_
