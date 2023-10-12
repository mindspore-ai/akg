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
 * \file sdca_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_SDCA_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SDCA_OPS_H_

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Distributed version of Stochastic Dual Coordinate Ascent (SDCA) optimizer for
*linear models with L1 + L2 regularization. As global optimization objective is
*strongly-convex, the optimizer optimizes the dual objective at each step. The
*optimizer applies each update one example at a time. Examples are sampled
*uniformly, and the optimizer is learning rate free and enjoys linear convergence
*rate . \n

*@par Inputs:
*@li sparse_example_indices: a list of vectors which contain example indices.It's a dynamic input.
*@li sparse_feature_indices: a list of vectors which contain feature indices.It's a dynamic input.
*@li sparse_feature_values: a list of vectors which contains feature value associated with each feature group.It's a dynamic input.
*@li dense_features: a list of matrices which contains the dense feature values.It's a dynamic input.
*@li example_weights: a vector which contains the weight associated with each example.
*@li example_labels: a vector which contains the label/target associated with each example.
*@li sparse_indices: a list of vectors where each value is the indices which has
*corresponding weights in sparse_weights. This field maybe omitted for the dense approach.It's a dynamic input.
*@li sparse_weights: a list of vectors where each value is the weight associated with a sparse feature group.
*@li dense_weights: a list of vectors where the values are the weights associated with a dense feature group.It's a dynamic input.
*@li example_state_data: a list of vectors containing the example state data. \n

*@par Attributes:
*@li adaptive: the type is bool default false.
*@li num_sparse_features:The num of sparse.
*@li num_sparse_features_with_values: The num of sparse_feature_values
*@li num_dense_features:The num of dense.
*@li loss_type: Type of the primal loss. Currently SdcaSolver supports logistic, squared and hinge losses.
*@li l1: Symmetric l1 regularization strength.
*@li l2: Symmetric l2 regularization strength.
*@li num_loss_partitions: Number of partitions of the global loss function.
*@li num_inner_iterations: Number of iterations per mini-batch . \n

*@par Outputs:
*@li out_example_state_data: A Returns a list of vectors containing the updated example state
*data.a list of vectors where each value is the delta
*@li out_delta_sparse_weights:weights associated with a sparse feature group.a list of vectors where the values are the delta
*@li out_delta_dense_weights:weights associated with a dense feature group . \n

*@par Third-party framework compatibility
* Compatible with tensorflow SdcaOptimizerV2 operator.
*/

REG_OP(SdcaOptimizerV2)
    .DYNAMIC_INPUT(sparse_example_indices, TensorType({DT_INT64}))
    .DYNAMIC_INPUT(sparse_feature_indices, TensorType({DT_INT64}))
    .DYNAMIC_INPUT(sparse_feature_values, TensorType({DT_FLOAT}))
    .DYNAMIC_INPUT(dense_features, TensorType({DT_FLOAT}))
    .INPUT(example_weights, TensorType({DT_FLOAT}))
    .INPUT(example_labels, TensorType({DT_FLOAT}))
    .DYNAMIC_INPUT(sparse_indices, TensorType({DT_INT64}))
    .DYNAMIC_INPUT(sparse_weights, TensorType({DT_FLOAT}))
    .DYNAMIC_INPUT(dense_weights, TensorType({DT_FLOAT}))
    .INPUT(example_state_data, TensorType({DT_FLOAT}))
    .OUTPUT(out_example_state_data, TensorType({DT_FLOAT}))
    .DYNAMIC_OUTPUT(out_delta_sparse_weights, TensorType({DT_FLOAT}))
    .DYNAMIC_OUTPUT(out_delta_dense_weights, TensorType({DT_FLOAT}))
    .ATTR(adaptive, Bool, false)
    .ATTR(num_sparse_features, Int, 0)
    .ATTR(num_sparse_features_with_values, Int, 0)
    .ATTR(num_dense_features, Int, 0)
    .ATTR(num_loss_partitions, Int, 1)
    .ATTR(num_inner_iterations, Int, 1)
    .ATTR(loss_type, String, "logistic_loss")
    .ATTR(l1, Float, 0.5)
    .ATTR(l2, Float, 0.5)
    .OP_END_FACTORY_REG(SdcaOptimizerV2)

}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_SDCA_OPS_H_