/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file nn_training_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_TRAINING_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_TRAINING_OPS_H_

#include "graph/operator_reg.h"
namespace ge {
/**
*@brief Updates "var" according to the AdaMax algorithm.
*  t-1 mean previous period.
*  m_t <- beta1 * m{t-1} + (1 - beta1) * grad
*  v_t <- max(beta2 * v{t-1}, abs(grad))
*  var <- var - lr / (1 - beta1^t) * m_t / (v_t + epsilon)
*
*@attention Constraints:
*  the input tensors must have the same shape.
*
*@par Inputs:
*@li var: A mutable tensor. Must be one of the following types: TensorType::NumberType().
*     Should be from a Variable().
*@li m: A mutable tensor. Has the same type as "var".
*     Should be from a Variable().
*@li v: A mutable tensor. Has the same type as "var".
*     Should be from a Variable().
*@li beta1_power: A scalar. Has the same type as "var".
*@li lr: learning_rate. A scalar. Has the same type as "var".
*@li beta1: A scalar. Has the same type as "var".
*@li beta2: A scalar. Has the same type as "var".
*@li epsilon: A scalar. Has the same type as "var".
*@li grad: A tensor for the gradient. Has the same type as "var".
*
*@par Attributes:
* use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var", "ms", and "mom" tensors is protected
*     by a lock; otherwise the behavior is undefined, but may exhibit less
*     contention.
*
*@par Outputs:
* var: A mutable tensor. Has the same type as input "var".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyAdaMax.
*
*/
REG_OP(ApplyAdaMax)
    .INPUT(var, TensorType::NumberType())
    .INPUT(m, TensorType::NumberType())
    .INPUT(v, TensorType::NumberType())
    .INPUT(beta1_power, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(beta1, TensorType::NumberType())
    .INPUT(beta2, TensorType::NumberType())
    .INPUT(epsilon, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdaMax)

/**
*@brief Updates "var" according to the AdaMax algorithm.
*  t-1 mean previous period.
*  m_t <- beta1 * m{t-1} + (1 - beta1) * grad
*  v_t <- max(beta2 * v{t-1}, abs(grad))
*  var <- var - lr / (1 - beta1^t) * m_t / (v_t + epsilon)
*
*@attention Constraints:
*  the input tensors must have the same shape.
*
*@par Inputs:
*@li var: A mutable tensor. Must be one of the following types: TensorType::NumberType().
*     Should be from a Variable().
*@li m: A mutable tensor. Has the same type as "var".
*     Should be from a Variable().
*@li v: A mutable tensor. Has the same type as "var".
*     Should be from a Variable().
*@li beta1_power: A scalar. Has the same type as "var".
*@li lr: learning_rate. A scalar. Has the same type as "var".
*@li beta1: A scalar. Has the same type as "var".
*@li beta2: A scalar. Has the same type as "var".
*@li epsilon: A scalar. Has the same type as "var".
*@li grad: A tensor for the gradient. Has the same type as "var".
*
*@par Attributes:
* use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var", "ms", and "mom" tensors is protected
*     by a lock; otherwise the behavior is undefined, but may exhibit less
*     contention.
*
*@par Outputs:
*@li var: A mutable tensor. Has the same type as input "var".
*@li m: A mutable tensor. Has the same type as input "m".
*@li v: A mutable tensor. Has the same type as input "v".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyAdaMax.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ApplyAdaMax instead.
*/
REG_OP(ApplyAdaMaxD)
    .INPUT(var, TensorType::NumberType())
    .INPUT(m, TensorType::NumberType())
    .INPUT(v, TensorType::NumberType())
    .INPUT(beta1_power, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(beta1, TensorType::NumberType())
    .INPUT(beta2, TensorType::NumberType())
    .INPUT(epsilon, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(m, TensorType::NumberType())
    .OUTPUT(v, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdaMaxD)

/**
*@brief Updates relevant entries in "var" and "accum" according to the adagrad scheme . \n

*@par Inputs:
* Five inputs, including:
*@li var: An NCHW, NHWC, or ND Tensor of type float32.
*@li accum: An NCHW, NHWC, or ND Tensor of type float32.
*@li lr: An NCHW, NHWC, or ND Tensor of type float32.
*@li grad: An NCHW, NHWC, or ND Tensor of type float32.
*@li indices: An NCHW, NHWC, or ND Tensor of type float32 . \n

*@par Attributes:
*@li use_locking: An optional bool. Defaults to "False". If "True", the operation will be protected by a lock.
*@li update_slots: An optional bool. Defaults to "True". If "True", the calcution will be different as "False" . \n

*@par Outputs:
*var: A Tensor. Has the same type and format as input "var" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseApplyAdagrad.
*/
REG_OP(SparseApplyAdagrad)
    .INPUT(var, TensorType({DT_FLOAT}))
    .INPUT(accum, TensorType({DT_FLOAT}))
    .INPUT(lr, TensorType({DT_FLOAT}))
    .INPUT(grad, TensorType({DT_FLOAT}))
    .INPUT(indices, TensorType({DT_INT32}))
    .OUTPUT(var, TensorType({DT_FLOAT}))
    .OUTPUT(accum, TensorType({DT_FLOAT}))
    .ATTR(use_locking, Bool, false)
    .ATTR(update_slots, Bool, true)
    .OP_END_FACTORY_REG(SparseApplyAdagrad)

/**
*@brief Updates relevant entries in "var" and "accum" according to the adagrad scheme . \n

*@par Inputs:
* Four inputs, including:
*@li var: An NCHW, NHWC, or ND Tensor of type float32.
*@li accum: An NCHW, NHWC, or ND Tensor of type float32.
*@li grad: An NCHW, NHWC, or ND Tensor of type float32.
*@li indices: An NCHW, NHWC, or ND Tensor of type int32 . \n

*@par Attributes:
*@li lr: Required, used for computation.
*@li use_locking: An optional bool. Defaults to "False". If "True", the operation will be protected by a lock.
*@li update_slots: An optional bool. Defaults to "True". If "True", the calcution will be different as "False" . \n

*@par Outputs:
*@li var: A Tensor. Has the same type and format as input "var".
*@li accum: A Tensor. Has the same type and format as input "var" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseApplyAdagrad. \n
*
*@par Restrictions:
*Warning: THIS FUNCTION IS DEPRECATED. Please use SparseApplyAdagrad instead.
*/
REG_OP(SparseApplyAdagradD)
    .INPUT(var, TensorType({DT_FLOAT}))
    .INPUT(accum, TensorType({DT_FLOAT}))
    .INPUT(grad, TensorType({DT_FLOAT}))
    .INPUT(indices, TensorType({DT_INT32}))
    .OUTPUT(var, TensorType({DT_FLOAT}))
    .OUTPUT(accum, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(lr, Float)
    .ATTR(use_locking, Bool, false)
    .ATTR(update_slots, Bool, true)
    .OP_END_FACTORY_REG(SparseApplyAdagradD)

/**
*@brief Updates relevant entries in "var" and "accum" according to the adagrad scheme . \n

*@par Inputs:
*Six inputs, including:
*@li var: An NCHW, NHWC, or ND Tensor of type float32.
*@li accum: An NCHW, NHWC, or ND Tensor of type float32.
*@li lr: An NCHW, NHWC, or ND Tensor of type float32.
*@li epsilon: An NCHW, NHWC, or ND Tensor of type float32.
*@li grad: An NCHW, NHWC, or ND Tensor of type float32.
*@li indices: An NCHW, NHWC, or ND Tensor of type float32 . \n

*@par Attributes:
*@li use_locking: An optional bool. Defaults to "False". If "True", the operation will be protected by a lock.
*@li update_slots: An optional bool. Defaults to "True". If "False", the computation logic will be different . \n

*@par Outputs:
*@li var: A Tensor. Has the same type and format as input "var" .
*@li accum: A Tensor. Has the same type and format as input "accum" . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator SparseApplyAdagradV2.
*/
REG_OP(SparseApplyAdagradV2)
    .INPUT(var, TensorType({DT_FLOAT}))
    .INPUT(accum, TensorType({DT_FLOAT}))
    .INPUT(lr, TensorType({DT_FLOAT}))
    .INPUT(epsilon, TensorType({DT_FLOAT}))
    .INPUT(grad, TensorType({DT_FLOAT}))
    .INPUT(indices, TensorType({DT_INT32}))
    .OUTPUT(var, TensorType({DT_FLOAT}))
    .OUTPUT(accum, TensorType({DT_FLOAT}))
    .ATTR(use_locking, Bool, false)
    .ATTR(update_slots, Bool, true)
    .OP_END_FACTORY_REG(SparseApplyAdagradV2)

/**
*@brief Updates relevant entries in "var" and "accum" according to the adagrad scheme . \n

*@par Inputs:
*Four inputs, including:
*@li var: An NCHW, NHWC, or ND Tensor of type float32.
*@li accum: An NCHW, NHWC, or ND Tensor of type float32.
*@li grad: An NCHW, NHWC, or ND Tensor of type float32.
*@li indices: An NCHW, NHWC, or ND Tensor of type int32 . \n

*@par Attributes:
*@li lr: Required, used for computation.
*@li epsilon: Required, used for computation.
*@li use_locking: An optional bool. Defaults to "False". If "True", the operation will be protected by a lock.
*@li update_slots: An optional bool. Defaults to "True". If "False", the computation logic will be different . \n

*@par Outputs:
*@li var: A Tensor. Has the same type and format as input "var".
*@li accum: A Tensor. Has the same type and format as input "accum" . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator SparseApplyAdagradV2. \n
*
*@par Restrictions:
*Warning: THIS FUNCTION IS DEPRECATED. Please use SparseApplyAdagradV2 instead.
*/
REG_OP(SparseApplyAdagradV2D)
    .INPUT(var, TensorType({DT_FLOAT}))
    .INPUT(accum, TensorType({DT_FLOAT}))
    .INPUT(grad, TensorType({DT_FLOAT}))
    .INPUT(indices, TensorType({DT_INT32}))
    .OUTPUT(var, TensorType({DT_FLOAT}))
    .OUTPUT(accum, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(lr, Float)
    .REQUIRED_ATTR(epsilon, Float)
    .ATTR(use_locking, Bool, false)
    .ATTR(update_slots, Bool, true)
    .OP_END_FACTORY_REG(SparseApplyAdagradV2D)

/**
*@brief Updates "var" according to the momentum scheme. Set use_nesterov = True if you
*   want to use Nesterov momentum.
*  computing process:
*  accum = accum * momentum + grad
*  var -= lr * accum
*
*@attention Constraints:
*  the input tensors must have the same shape.
*
*@par Inputs:
*@li var: A mutable tensor. Should be from a Variable().
*@li accum: A mutable tensor. Has the same type as "var".
*     Should be from a Variable().
*@li lr: A scalar. Has the same type as "var".
*@li grad: A tensor for the gradient. Has the same type as "var".
*@li momentum: Momentum. Must be a scalar.

*@par Attributes:
*@li use_nesterov: An optional bool. Defaults to "False".
*     If "True", the tensor passed to compute grad will be
*     var - lr * momentum * accum, so in the end, the var you get is actually
*     var - lr * momentum * accum.
*
*@li use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var", "ms", and "mom" tensors is protected by a lock;
*     otherwise the behavior is undefined, but may exhibit less contention.
*
*@par Outputs:
* var: A mutable tensor. Has the same type as input "var".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyMomentum.
*
*/

REG_OP(ApplyMomentum)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .INPUT(momentum, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_nesterov, Bool, false)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyMomentum)


/**
*@brief Updates "var" according to the momentum scheme. Set use_nesterov = True if you
*   want to use Nesterov momentum.
*  computing process:
*  accum = accum * momentum + grad
*  var -= lr * accum
*
*@attention Constraints:
*  the input tensors must have the same shape.
*
*@par Inputs:
*@li var: A mutable tensor. Should be from a Variable().
*@li accum: A mutable tensor. Has the same type as "var".
*     Should be from a Variable().
*@li lr: A scalar. Has the same type as "var".
*@li grad: A tensor for the gradient. Has the same type as "var".
*
*@par Attributes:
*@li use_nesterov: An optional bool. Defaults to "False".
*     If "True", the tensor passed to compute grad will be
*     var - lr * momentum * accum, so in the end, the var you get is actually
*     var - lr * momentum * accum.
*
*@li use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var", "ms", and "mom" tensors is protected by a lock;
*     otherwise the behavior is undefined, but may exhibit less contention.
*
*@par Outputs:
* var: A mutable tensor. Has the same type as input "var".
* accum: A mutable tensor. Has the same type as input "accum".
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyMomentum.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ApplyMomentum instead.
*/

REG_OP(ApplyMomentumD)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .INPUT(momentum, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(accum, TensorType::NumberType())
    .ATTR(use_nesterov, Bool, false)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyMomentumD)

/**
*@brief Updates '*var' according to the momentum scheme.
*   accum = accum * momentum - grad * lr
*   if use_nesterov is True:
*       var += accum * momentum - grad * lr
*   else:
*       var += accum
*
*@par Inputs:
*@li var: A mutable tensor. Must be one of the data types defined in
*    TensorType::NumberType(). Should be from a Variable().
*@li accum: A mutable tensor. Has the same type as "var". Should be from a
*    Variable().
*@li lr: A tensor for the learning rate. Has the same type as "var". Should be
*    from a Variable().
*@li grad: A tensor for the gradient. Has the same type as "var". Should be
*    from a Variable().
*@li momentum: A scalar. Has the same type as "var".
*
*@par Attributes:
*@li use_nesterov: An optional bool. Defaults to "False".
*    If "True", var will be updated by using Nesterov momentum.
*@li use_locking: An optional bool. Defaults to "False".
*    If "True", updating of the "var" tensor is protected by a lock;
*    otherwise the behavior is undefined, but may exhibit less contention.
*
*@par Outputs:
* var: A mutable tensor. Has the same type as input "var".
*
*@attention Constraints:
* The input tensors must have the same shape.
*
*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ResourceApplyKerasMomentum.
*
*/
REG_OP(ApplyKerasMomentum)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .INPUT(momentum, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .ATTR(use_nesterov, Bool, false)
    .OP_END_FACTORY_REG(ApplyKerasMomentum)


/**
*@brief Updates '*var' according to the momentum scheme.
*   accum = accum * momentum - grad * lr
*   if use_nesterov is True:
*       var += accum * momentum - grad * lr
*   else:
*       var += accum
*
*@par Inputs:
*@li var: A mutable tensor. Must be one of the data types defined in
*    TensorType::NumberType(). Should be from a Variable().
*@li accum: A mutable tensor. Has the same type as "var". Should be from a
*    Variable().
*@li lr: A tensor for the learning rate. Has the same type as "var". Should be
*    from a Variable().
*@li grad: A tensor for the gradient. Has the same type as "var". Should be
*    from a Variable().
*@li momentum: A scalar. Has the same type as "var". Should be from a
*    Variable().
*
*@par Attributes:
*@li use_nesterov: An optional bool. Defaults to "False".
*    If "True", var will be updated by using nesterov momentum
*@li use_locking: An optional bool. Defaults to "False".
*    If "True", updating of the "var" tensor is protected by a lock;
*    otherwise the behavior is undefined, but may exhibit less contention.
*
*@par Outputs:
*@li var: A mutable tensor. Has the same type as input "var".
*@li accum: A mutable tensor. Has the same type as input "var"
*
*@attention Constraints:
* The input tensors must have the same shape.
*
*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ResourceApplyKerasMomentum.
*
*@par Restrictions:
*Warning: THIS FUNCTION IS DEPRECATED. Please use ApplyKerasMomentum instead.
*/
REG_OP(ApplyKerasMomentumD)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .INPUT(momentum, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(accum, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .ATTR(use_nesterov, Bool, false)
    .OP_END_FACTORY_REG(ApplyKerasMomentumD)


/**
*@brief Updates '*var' according to the Adam algorithm.
*   lr_t := {learning_rate} * sqrt{1 - beta_2^t} / (1 - beta_1^t)
*   m_t := beta_1 * m_{t-1} + (1 - beta_1) * g
*   v_t := beta_2 * v_{t-1} + (1 - beta_2) * g * g
*   vhat_t := max{vhat_{t-1}, v_t}
*   variable := variable - lr_t * m_t / (sqrt{vhat_t} + epsilon)
*
*@par Inputs:
*@li var: A mutable tensor. Must be one of the data types defined in
*    TensorType::NumberType(). Should be from a Variable().
*@li m: A mutable tensor. Has the same type as "var". Should be from a
*    Variable().
*@li v: A mutable tensor. Has the same type as "var". Should be from a
*    Variable().
*@li vhat: A mutable tensor. Has the same type as "var". Should be from a
*    Variable().
*@li beta1_power: A mutable tensor. Has the same type as "var". Should be from a
*    Variable().
*@li beta2_power: A mutable tensor. Has the same type as "var". Should be from a
*    Variable().
*@li lr: A tensor for the learning rate. Has the same type as "var". Should be
*    from a Variable().
*@li grad: A tensor for the gradient. Has the same type as "var". Should be
*    from a Variable().
*
*@par Attributes:
*@li beta1: A scalar. Has the same type as "var".
*@li beta2: A scalar. Has the same type as "var".
*@li epsilon: A scalar. Has the same type as "var".
*@li use_locking: An optional bool. Defaults to "False".
*    If "True", updating of the "var" tensor is protected by a lock;
*    otherwise the behavior is undefined, but may exhibit less contention.
*
*@par Outputs:
*@li var: A mutable tensor. Has the same type as input "var".
*@li m: A mutable tensor. Has the same type as input "var"
*@li v: A mutable tensor. Has the same type as input "var"
*@li vhat: A mutable tensor. Has the same type as input "var"
*
*@attention Constraints:
* The input tensors must have the same shape.
*
*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ResourceApplyKerasMomentum.
*
*@par Restrictions:
*Warning: THIS FUNCTION IS DEPRECATED. Please use ApplyAdamWithAmsgrad instead.
*
*/
REG_OP(ApplyAdamWithAmsgradD)
    .INPUT(var, TensorType::NumberType())
    .INPUT(m, TensorType::NumberType())
    .INPUT(v, TensorType::NumberType())
    .INPUT(vhat, TensorType::NumberType())
    .INPUT(beta1_power, TensorType::NumberType())
    .INPUT(beta2_power, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(m, TensorType::NumberType())
    .OUTPUT(v, TensorType::NumberType())
    .OUTPUT(vhat, TensorType::NumberType())
    .REQUIRED_ATTR(beta1, Float)
    .REQUIRED_ATTR(beta2, Float)
    .REQUIRED_ATTR(epsilon, Float)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdamWithAmsgradD)


/**
*@brief Updates '*var' according to the Adam algorithm..
*   lr_t := {learning_rate} * sqrt{1 - beta_2^t} / (1 - beta_1^t)
*   m_t := beta_1 * m_{t-1} + (1 - beta_1) * g
*   v_t := beta_2 * v_{t-1} + (1 - beta_2) * g * g
*   vhat_t := max{vhat_{t-1}, v_t}
*   variable := variable - lr_t * m_t / (sqrt{vhat_t} + epsilon)
*
*@par Inputs:
*@li var: A mutable tensor. Must be one of the data types defined in
*    TensorType::NumberType(). Should be from a Variable().
*@li m: A mutable tensor. Has the same type as "var". Should be from a
*    Variable().
*@li v: A mutable tensor. Has the same type as "var". Should be from a
*    Variable().
*@li vhat: A mutable tensor. Has the same type as "var". Should be from a
*    Variable().
*@li beta1_power: A mutable tensor. Has the same type as "var". Should be from a
*    Variable().
*@li beta2_power: A mutable tensor. Has the same type as "var". Should be from a
*    Variable().
*@li lr: A tensor for the learning rate. Has the same type as "var". Should be
*    from a Variable().
*@li grad: A tensor for the gradient. Has the same type as "var". Should be
*    from a Variable().
*
*@par Attributes:
*@li beta1: A scalar. Has the same type as "var".
*@li beta2: A scalar. Has the same type as "var".
*@li epsilon: A scalar. Has the same type as "var".
*@li use_locking: An optional bool. Defaults to "False".
*    If "True", updating of the "var" tensor is protected by a lock;
*    otherwise the behavior is undefined, but may exhibit less contention.
*
*@par Outputs:
*@li var: A mutable tensor. Has the same type as input "var".
*@li m: A mutable tensor. Has the same type as input "var"
*@li v: A mutable tensor. Has the same type as input "var"
*@li vhat: A mutable tensor. Has the same type as input "var"
*
*@attention Constraints:
* The input tensors must have the same shape.
*
*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ResourceApplyKerasMomentum.
*
*/
REG_OP(ApplyAdamWithAmsgrad)
    .INPUT(var, TensorType::NumberType())
    .INPUT(m, TensorType::NumberType())
    .INPUT(v, TensorType::NumberType())
    .INPUT(vhat, TensorType::NumberType())
    .INPUT(beta1_power, TensorType::NumberType())
    .INPUT(beta2_power, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(beta1, TensorType::NumberType())
    .INPUT(beta2, TensorType::NumberType())
    .INPUT(epsilon, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdamWithAmsgrad)


/**
*@brief Updates '*var' according to the Adam algorithm..
*   lr_t := {learning_rate} * sqrt{1 - beta_2^t} / (1 - beta_1^t)
*   m_t := beta_1 * m_{t-1} + (1 - beta_1) * g
*   v_t := beta_2 * v_{t-1} + (1 - beta_2) * g * g
*   vhat_t := max{vhat_{t-1}, v_t}
*   variable := variable - lr_t * m_t / (sqrt{vhat_t} + epsilon)
*
*@par Inputs:
*Eleven inputs, including:
*@li var: A mutable tensor. Must be one of the data types defined in
*    TensorType::NumberType(). Should be from a Variable().
*@li m: A mutable tensor. Has the same type as "var". Should be from a
*    Variable().
*@li v: A mutable tensor. Has the same type as "var". Should be from a
*    Variable().
*@li vhat: A mutable tensor. Has the same type as "var". Should be from a
*    Variable().
*@li beta1_power: A mutable tensor. Has the same type as "var". Should be from a
*    Variable().
*@li beta2_power: A mutable tensor. Has the same type as "var". Should be from a
*    Variable().
*@li lr: A tensor for the learning rate. Has the same type as "var". Should be
*    from a Variable().
*@li beta1: A mutable tensor. Has the same type as "var". Should be
*    from a Variable().
*@li beta2: A mutable tensor. Has the same type as "var". Should be
*    from a Variable().
*@li epsilon: A mutable tensor. Has the same type as "var". Should be
*    from a Variable().
*@li grad: A tensor for the gradient. Has the same type as "var". Should be
*    from a Variable().
*
*@par Attribute:
*one attribute, including:
*@li use_locking: An optional bool. Defaults to "False".
*    If "True", updating of the "var" tensor is protected by a lock;
*    otherwise the behavior is undefined, but may exhibit less contention.
*
*@par Outputs:
*four outputs, including:
*@li var: A mutable tensor. Has the same type as input "var".
*@li m: A mutable tensor. Has the same type as input "var"
*@li v: A mutable tensor. Has the same type as input "var"
*@li vhat: A mutable tensor. Has the same type as input "var"
*
*@attention Constraints:
* The input tensors must have the same shape.
*
*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ResourceApplyKerasMomentum.
*
*/
REG_OP(ApplyAdamWithAmsgradV2)
    .INPUT(var, TensorType({DT_FLOAT}))
    .INPUT(m, TensorType({DT_FLOAT}))
    .INPUT(v, TensorType({DT_FLOAT}))
    .INPUT(vhat, TensorType({DT_FLOAT}))
    .INPUT(beta1_power, TensorType({DT_FLOAT}))
    .INPUT(beta2_power, TensorType({DT_FLOAT}))
    .INPUT(lr, TensorType({DT_FLOAT}))
    .INPUT(beta1, TensorType({DT_FLOAT}))
    .INPUT(beta2, TensorType({DT_FLOAT}))
    .INPUT(epsilon, TensorType({DT_FLOAT}))
    .INPUT(grad, TensorType({DT_FLOAT}))
    .OUTPUT(var, TensorType({DT_FLOAT}))
    .OUTPUT(m, TensorType({DT_FLOAT}))
    .OUTPUT(v, TensorType({DT_FLOAT}))
    .OUTPUT(vhat, TensorType({DT_FLOAT}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdamWithAmsgradV2)

/**
*@brief Updates "var" according to the AddSign update.
*  t-1 mean previous period.
*  m_t <- beta1 * m_{t-1} + (1 - beta1) * grad
*  update <- exp(logbase * sign_decay * sign(grad) * sign(m_t)) * grad
*  var <- var - lr * update
*
*@attention Constraints:
*  the input tensors must have the same shape.
*
*@par Inputs:
*@li var: A mutable tensor. Should be from a Variable().
*@li m: A mutable tensor. Has the same type as "var".
*     Should be from a Variable().
*@li lr: A scalar. Has the same type as "var".
*@li logbase: A scalar. Has the same type as "var".
*@li sign_decay: A scalar. Has the same type as "var".
*@li beta: A scalar. Has the same type as "var".
*@li grad: A tensor for the gradient. Has the same type as "var".
*
*@par Attributes:
* use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var", "ms", and "mom" tensors is protected
*     by a lock; otherwise the behavior is undefined, but may exhibit less
*     contention.
*
*@par Outputs:
* var: A mutable tensor. Has the same type as input "var".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyPowerSign.
*
*/
REG_OP(ApplyPowerSign)
    .INPUT(var, TensorType::NumberType())
    .INPUT(m, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(logbase, TensorType::NumberType())
    .INPUT(sign_decay, TensorType::NumberType())
    .INPUT(beta, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyPowerSign)

/**
*@brief Updates "var" according to the AddSign update.
*  t-1 mean previous period.
*  m_t <- beta1 * m_{t-1} + (1 - beta1) * grad
*  update <- exp(logbase * sign_decay * sign(grad) * sign(m_t)) * grad
*  var <- var - lr * update
*
*@attention Constraints:
*  the input tensors must have the same shape.
*
*@par Inputs:
*@li var: A mutable tensor. Should be from a Variable().
*@li m: A mutable tensor. Has the same type as "var".
*     Should be from a Variable().
*@li lr: A scalar. Has the same type as "var".
*@li logbase: A scalar. Has the same type as "var".
*@li sign_decay: A scalar. Has the same type as "var".
*@li beta: A scalar. Has the same type as "var".
*@li grad: A tensor for the gradient. Has the same type as "var".
*
*@par Attributes:
* use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var", "ms", and "mom" tensors is protected
*     by a lock; otherwise the behavior is undefined, but may exhibit less
*     contention.
*
*@par Outputs:
*@li var: A mutable tensor. Has the same type as input "var".
*@li m: A mutable tensor. Has the same type as input "var".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyPowerSign.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ApplyPowerSign instead.
*/
REG_OP(ApplyPowerSignD)
    .INPUT(var, TensorType::NumberType())
    .INPUT(m, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(logbase, TensorType::NumberType())
    .INPUT(sign_decay, TensorType::NumberType())
    .INPUT(beta, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(m, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyPowerSignD)

/**
*@brief Updates "var" as FOBOS algorithm with fixed learning rate.
*  prox_v = var - alpha * delta
*  var = sign(prox_v)/(1+alpha * l2) * max{|prox_v|-alpha * l1,0}
*
*@attention Constraints:
*  the input tensors must have the same shape.
*
*@par Inputs:
*@li var: A mutable tensor. Should be from a Variable().
*@li alpha: A scalar. Has the same type as "var".
*@li l1: A scalar. Has the same type as "var".
*@li l2: A scalar. Has the same type as "var".
*@li delta: A tensor. Has the same type as "var". The change.
*
*@par Attributes:
* use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var", "ms", and "mom" tensors is protected
*     by a lock; otherwise the behavior is undefined, but may exhibit less
*     contention.
*
*@par Outputs:
* var: A mutable tensor. Has the same type as input "var".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyProximalGradientDescent.
*
*/
REG_OP(ApplyProximalGradientDescent)
    .INPUT(var, TensorType::NumberType())
    .INPUT(alpha, TensorType::NumberType())
    .INPUT(l1, TensorType::NumberType())
    .INPUT(l2, TensorType::NumberType())
    .INPUT(delta, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyProximalGradientDescent)

/**
*@brief Updates "var" according to the AddSign update . \n

*@par Inputs:
*Seven inputs, including:
* @li var: A mutable Tensor of type TensorType::NumberType().
*     Should be a Variable Tensor.
* @li m: A mutable Tensor of the same type as "var".
*     Should be a Variable Tensor.
* @li lr: A Tensor of the same type as "var", for the scaling factor. Must be a scalar.
* @li alpha: A Tensor of the same type as "var". Must be a scalar.
* @li sign_decay: A Tensor of the same type as "var". Must be a scalar.
* @li beta: A Tensor of the same type as "var". Must be a scalar.
* @li grad: A Tensor of the same type as "var", for the gradient.


*@par Attributes:
*use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var" and "m" tensors will be
*     protected by a lock; otherwise the behavior is undefined,
*     but may exhibit less contention . \n

*@par Outputs:
*var: A mutable Tensor. Has the same type as "var" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ApplyAddSign.
*/
REG_OP(ApplyAddSign)
    .INPUT(var, TensorType::NumberType())
    .INPUT(m, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(alpha, TensorType::NumberType())
    .INPUT(sign_decay, TensorType::NumberType())
    .INPUT(beta, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyAddSign)

/**
*@brief Updates "var" according to the AddSign update . \n

*@par Inputs:
*Seven inputs, including:
* @li var: A mutable Tensor of type TensorType::NumberType().
*     Should be a Variable Tensor.
* @li m: A mutable Tensor of the same type as "var".
*     Should be a Variable Tensor.
* @li lr: A Tensor of the same type as "var", for the scaling factor. Must be a scalar.
* @li alpha: A Tensor of the same type as "var". Must be a scalar.
* @li sign_decay: A Tensor of the same type as "var". Must be a scalar.
* @li beta: A Tensor of the same type as "var". Must be a scalar.
* @li grad: A Tensor of the same type as "var", for the gradient.


*@par Attributes:
*use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var" and "m" tensors will be
*     protected by a lock; otherwise the behavior is undefined,
*     but may exhibit less contention . \n

*@par Outputs:
*@li var: A mutable Tensor. Has the same type as "var".
*@li m: A mutable Tensor. Has the same type as "m" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ApplyAddSign.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ApplyAddSign instead.
*/
REG_OP(ApplyAddSignD)
    .INPUT(var, TensorType::NumberType())
    .INPUT(m, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(alpha, TensorType::NumberType())
    .INPUT(sign_decay, TensorType::NumberType())
    .INPUT(beta, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(m, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyAddSignD)

/**
*@brief Updates "var" according to the centered RMSProp algorithm.
*  The centered RMSProp algorithm uses an estimate of the centered second moment
*  (i.e., the variance) for normalization, as opposed to regular RMSProp, which
*  uses the (uncentered) second moment. This often helps with training, but is
*  slightly more expensive in terms of computation and memory.
*
*  t-1 mean previous period.
*  mg <- rho * mg{t-1} + (1-rho) * grad
*  ms <- rho * ms{t-1} + (1-rho) * grad * grad
*  mom <- momentum * mom{t-1} + lr * grad / sqrt(ms - mg * mg + epsilon)
*  var <- var - mom
*
*@attention Constraints:
*@li in dense implementation of this algorithm, mg, ms, and mom will
*    update even if the grad is zero, but in this sparse implementation, mg, ms,
*    and mom will not update in iterations during which the grad is zero.
*@li the input tensors must have the same shape.
*
*@par Inputs:
*@li var: A mutable tensor. Should be from a Variable().
*@li mg: A mutable tensor. Has the same type as "var".
*     Should be from a Variable().
*@li ms: A mutable tensor. Has the same type as "var".
*     Should be from a Variable().
*@li mom: A mutable tensor. Has the same type as "var".
*     Should be from a Variable().
*@li lr: A scalar. Has the same type as "var".
*@li rho: A scalar. Has the same type as "var".
*@li momentum: A tensor. Has the same type as "var".
*@li epsilon: A scalar. Has the same type as "var".
*@li grad: A tensor for the gradient. Has the same type as "var".
*
*@par Attributes:
* use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var", "ms", and "mom" tensors is protected
*     by a lock; otherwise the behavior is undefined, but may exhibit less
*     contention.
*
*@par Outputs:
* var: A mutable tensor. Has the same type as input "var".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyCenteredRMSProp.
*
*/
REG_OP(ApplyCenteredRMSProp)
    .INPUT(var, TensorType::NumberType())
    .INPUT(mg, TensorType::NumberType())
    .INPUT(ms, TensorType::NumberType())
    .INPUT(mom, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(rho, TensorType::NumberType())
    .INPUT(momentum, TensorType::NumberType())
    .INPUT(epsilon, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyCenteredRMSProp)

/**
*@brief Updates "var" according to the centered RMSProp algorithm.
*  The centered RMSProp algorithm uses an estimate of the centered second moment
*  (i.e., the variance) for normalization, as opposed to regular RMSProp, which
*  uses the (uncentered) second moment. This often helps with training, but is
*  slightly more expensive in terms of computation and memory.
*
*  t-1 mean previous period.
*  mg <- rho * mg{t-1} + (1-rho) * grad
*  ms <- rho * ms{t-1} + (1-rho) * grad * grad
*  mom <- momentum * mom{t-1} + lr * grad / sqrt(ms - mg * mg + epsilon)
*  var <- var - mom
*
*@attention Constraints:
*@li in dense implementation of this algorithm, mg, ms, and mom will
*    update even if the grad is zero, but in this sparse implementation, mg, ms,
*    and mom will not update in iterations during which the grad is zero.
*@li the input tensors must have the same shape.
*
*@par Inputs:
*@li var: A mutable tensor. Should be from a Variable().
*@li mg: A mutable tensor. Has the same type as "var".
*     Should be from a Variable().
*@li ms: A mutable tensor. Has the same type as "var".
*     Should be from a Variable().
*@li mom: A mutable tensor. Has the same type as "var".
*     Should be from a Variable().
*@li lr: A scalar. Has the same type as "var".
*@li rho: A scalar. Has the same type as "var".
*@li momentum: A tensor. Has the same type as "var".
*@li epsilon: A scalar. Has the same type as "var".
*@li grad: A tensor for the gradient. Has the same type as "var".
*
*@par Attributes:
* use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var", "ms", and "mom" tensors is protected
*     by a lock; otherwise the behavior is undefined, but may exhibit less
*     contention.
*
*@par Outputs:
*@li var: A mutable Tensor. Has the same type as "var".
*@li mg: A mutable Tensor. Has the same type as "mg".
*@li ms: A mutable Tensor. Has the same type as "ms".
*@li mom: A mutable Tensor. Has the same type as "mom" . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyCenteredRMSPropD.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ApplyCenteredRMSProp instead.
*/
REG_OP(ApplyCenteredRMSPropD)
    .INPUT(var, TensorType::NumberType())
    .INPUT(mg, TensorType::NumberType())
    .INPUT(ms, TensorType::NumberType())
    .INPUT(mom, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(rho, TensorType::NumberType())
    .INPUT(momentum, TensorType::NumberType())
    .INPUT(epsilon, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(mg, TensorType::NumberType())
    .OUTPUT(ms, TensorType::NumberType())
    .OUTPUT(mom, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyCenteredRMSPropD)

/**
*@brief Updates "var" by subtracting 'alpha' * 'delta' from it.
*   var -= delta * alpha
*
*@attention Constraints:
*  the input tensors must have the same shape.
*
*@par Inputs:
*@li var: A mutable tensor. Should be from a Variable().
*@li alpha: A scalar. Has the same type as "var".
*@li delta: A tensor for the change. Has the same type as "var".
*
*@par Attributes:
* use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var" tensors is protected
*     by a lock; otherwise the behavior is undefined, but may exhibit less
*     contention.
*
*@par Outputs:
* var: A mutable tensor. Has the same type as input "var".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyGradientDescent.
*
*/
REG_OP(ApplyGradientDescent)
    .INPUT(var, TensorType::NumberType())
    .INPUT(alpha, TensorType::NumberType())
    .INPUT(delta, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyGradientDescent)

/**
*@brief Updates "var" according to the adagrad scheme.
*   accum += grad * grad
*   var -= lr * grad * (1 / sqrt(accum))
*
*@attention Constraints:
*  the input tensors must have the same shape.
*
*@par Inputs:
*@li var: A mutable tensor. Should be from a Variable().
*@li accum: A mutable tensor. Has the same type as "var".
*     Should be from a Variable().
*@li lr: A scalar. Has the same type as "var".
*@li grad: A tensor for the gradient. Has the same type as "var".
*
*@par Attributes:
*@li update_slots: An optional bool. Defaults to "True". If "True", the calcution will be different as "False".
*@li use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var", "ms", and "mom" tensors is protected
*     by a lock; otherwise the behavior is undefined, but may exhibit less
*     contention.
*
*@par Outputs:
* var: A mutable tensor. Has the same type as input "var".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyAdagrad.
*
*/
REG_OP(ApplyAdagrad)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(update_slots, Bool, true)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdagrad)

/**
*@brief Updates "var" according to the adagrad scheme.
*   accum += grad * grad
*   var -= lr * grad * (1 / sqrt(accum))
*
*@attention Constraints:
*  the input tensors must have the same shape.
*
*@par Inputs:
*@li var: A mutable tensor. Should be from a Variable().
*@li accum: A mutable tensor. Has the same type as "var".
*     Should be from a Variable().
*@li lr: A scalar. Has the same type as "var".
*@li grad: A tensor for the gradient. Has the same type as "var".
*
*@par Attributes:
*@li update_slots: An optional bool. Defaults to "True". If "True", the calcution will be different as "False".
*@li use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var", "ms", and "mom" tensors is protected
*     by a lock; otherwise the behavior is undefined, but may exhibit less
*     contention.
*
*@par Outputs:
*@li var: A mutable tensor. Has the same type as input "var".
*@li accum: A mutable tensor. Has the same type as input "var".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyAdagrad.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ApplyAdagrad instead.
*/
REG_OP(ApplyAdagradD)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(accum, TensorType::NumberType())
    .ATTR(update_slots, Bool, true)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdagradD)

/**
* @brief Updates "var" according to the adagradv2 scheme.
*   accum += grad * grad
*   var -= lr * grad * (1 / sqrt(accum) + epsilon)
*
* @par Inputs:
* @li var: A mutable tensor. Must be one of the data types defined in
* TensorType::NumberType(). Should be from a Variable().
* @li accum: A mutable tensor. Has the same type as "var". Should be from a
* Variable().
* @li lr: A tensor for the learning rate. Has the same type as "var". Should be
* from a Variable().
* @li grad: A tensor for the gradient. Has the same type as "var". Should be
* from a Variable().
* @li epsilon: A scalar. Has the same type as "var".
*
* @par Attributes:
* @li update_slots: An optional bool. Defaults to "True".
* If "True", "accum" will be updated
* @li use_locking: An optional bool. Defaults to "False".
* If "True", updating of the "var" tensor is protected by a lock;
* otherwise the behavior is undefined, but may exhibit less contention.
*
* @par Outputs:
* var: A mutable tensor. Has the same type as input "var".
*
* @attention Constraints:
* The input tensors must have the same shape.
*
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ApplyAdagrad.
*
*/
REG_OP(ApplyAdagradV2)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(epsilon, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(update_slots, Bool, true)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdagradV2)


/**
* @brief Updates "var" according to the adagradv2 scheme.
* accum += grad * grad
* var -= lr * grad * (1 / sqrt(accum) + epsilon)
*
* @par Inputs:
* @li var: A mutable tensor. Must be one of the data types defined in
* TensorType::NumberType(). Should be from a Variable().
* @li accum: A mutable tensor. Has the same type as "var". Should be from a
* Variable().
* @li lr: A tensor for the learning rate. Has the same type as "var". Should be
* from a Variable().
* @li grad: A tensor for the gradient. Has the same type as "var". Should be
* from a Variable().
*
* @par Attributes:
* @li epsilon: A scalar. Has the same type as "var".
* @li update_slots: An optional bool. Defaults to "True".
* If "True", "accum" will be updated
* @li use_locking: An optional bool. Defaults to "False".
* If "True", updating of the "var" tensor is protected by a lock;
* otherwise the behavior is undefined, but may exhibit less contention.
*
* @par Outputs:
* var: A mutable tensor. Has the same type as input "var".
*
* @attention Constraints:
* The input tensors must have the same shape.
*
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ApplyAdagrad.
*
*@par Restrictions:
*Warning: THIS FUNCTION IS DEPRECATED. Please use ApplyAdagradV2 instead.
*/
REG_OP(ApplyAdagradV2D)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(accum, TensorType::NumberType())
    .REQUIRED_ATTR(epsilon, Float)
    .ATTR(update_slots, Bool, true)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdagradV2D)

/**
*@brief Updates "var" according to the proximal adagrad scheme . \n

*@par Inputs:
*Eight inputs, including:
* @li var: A mutable Tensor. Must be one of the following types:
*     TensorType::NumberType(). Should be a Variable Tensor.
* @li gradient_accumulator: A mutable Tensor. Must have the same
*     type as "var". Should be a Variable Tensor.
* @li gradient_squared_accumulator: A mutable Tensor of the same type as "var".
*     Should be a Variable Tensor.
* @li grad: A Tensor of the same type as "var", for the gradient.
* @li lr: A Tensor of the same type as "var".
*     Scaling factor. Must be a scalar.
* @li l1: A Tensor of the same type as "var".
*     L1 regulariation. Must be a scalar.
* @li l2: A Tensor of the same type as "var".
*     L2 regulariation. Must be a scalar.
* @li global_step: A Tensor of type int32 or int64.
*     Training step number. Must be a scalar . \n

*@par Attributes:
*use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the var and accum tensors will be
*     protected by a lock; otherwise the behavior is undefined,
*     but may exhibit less contention . \n

*@par Outputs:
*var: A mutable Tensor. Has the same type as "var" . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyAdagradDA.
*/
REG_OP(ApplyAdagradDA)
    .INPUT(var, TensorType::NumberType())
    .INPUT(gradient_accumulator, TensorType::NumberType())
    .INPUT(gradient_squared_accumulator, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(l1, TensorType::NumberType())
    .INPUT(l2, TensorType::NumberType())
    .INPUT(global_step, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdagradDA)

/**
*@brief Updates "var" according to the proximal adagrad scheme . \n

*@par Inputs:
*Eight inputs, including:
* @li var: A mutable Tensor. Must be one of the following types:
*     TensorType::NumberType(). Should be a Variable Tensor.
* @li gradient_accumulator: A mutable Tensor. Must have the same
*     type as "var". Should be a Variable Tensor.
* @li gradient_squared_accumulator: A mutable Tensor of the same type as "var".
*     Should be a Variable Tensor.
* @li grad: A Tensor of the same type as "var", for the gradient.
* @li lr: A Tensor of the same type as "var".
*     Scaling factor. Must be a scalar.
* @li l1: A Tensor of the same type as "var".
*     L1 regulariation. Must be a scalar.
* @li l2: A Tensor of the same type as "var".
*     L2 regulariation. Must be a scalar.
* @li global_step: A Tensor of type int32 or int64.
*     Training step number. Must be a scalar . \n

*@par Attributes:
*use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the var and accum tensors will be
*     protected by a lock; otherwise the behavior is undefined,
*     but may exhibit less contention . \n

*@par Outputs:
*var: A mutable Tensor. Has the same type as "var".
*gradient_accumulator: A mutable Tensor. Has the same type as "var".
*gradient_squared_accumulator: A mutable Tensor. Has the same type as "var" . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyAdagradDA.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ApplyAdagradDA instead.
*/
REG_OP(ApplyAdagradDAD)
    .INPUT(var, TensorType::NumberType())
    .INPUT(gradient_accumulator, TensorType::NumberType())
    .INPUT(gradient_squared_accumulator, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(l1, TensorType::NumberType())
    .INPUT(l2, TensorType::NumberType())
    .INPUT(global_step, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(gradient_accumulator, TensorType::NumberType())
    .OUTPUT(gradient_squared_accumulator, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdagradDAD)

/**
*@brief Returns the dimension index in the destination data format given the one in
* the source data format.
*
*@par Inputs:
* x: A tensor of type int32 or int64.
*     A Tensor with each element as a dimension index in source data format.
*     Must be in the range [-4, 4).
*
*@par Attributes:
*@li src_format: An optional string. Defaults to NHWC.
*     source data format. Must of length 4.
*@li dst_format: An optional string. Defaults to NCHW.
*     destination data format. Must of length 4.
*
*@par Outputs:
* y: A tensor. Has the same type as "x". Must be in the range [0, 4).
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator DataFormatDimMap.
*
*/
REG_OP(DataFormatDimMap)
    .INPUT(x, TensorType::IndexNumberType())
    .ATTR(src_format, String, "NHWC")
    .ATTR(dst_format, String, "NCHW")
    .OUTPUT(y, TensorType::IndexNumberType())
    .OP_END_FACTORY_REG(DataFormatDimMap)

/**
* @brief Implements stochastic gradient descent (optionally with momentum).
* Nesterov momentum is based on the formula from
* On the importance of initialization and momentum in deep learning.

* @par Inputs:
* @li parameters: A mutable tensor of type float16 or float32.
* Specifies the iterable of parameters to optimize or dicts defining parameter
* groups.
* @li gradient: A tensor of type float16 or float32.
* Specifies the gradient of training step.
* @li learning_rate: A tensor of type float16 or float32.
* Specifies the learing_rate of training step.
* @li accum: A tensor of type float16 or float32.
* Specifies the velocity of training step.
* @li momentum: A tensor of type float16 or float32.
* Specifies the momentum factor.
* @li stat: A tensor of type float16 or float32.
* Specifies the status representing the first step or not . \n

* @par Attributes:
* @li dampening: An optional float, specifying the dampening for momentum.
* Defaults to "0.0".
* @li weight_decay: An optional float, specifying the L2 penalty. Defaults to
* "0.0".
* @li nesterov: An optional bool, specifying whether to enable Nesterov
* momentum. Defaults to "False" . \n

* @par Outputs:
* parameters: A mutable tensor same as input "parameters" . \n

* @see ApplyMomentum()

* @par Third-party framework compatibility
* @li Compatible with the PyTorch operator SGD.
*/
REG_OP(SGD)
    .INPUT(parameters, TensorType(DT_FLOAT, DT_FLOAT16))
    .INPUT(gradient, TensorType(DT_FLOAT, DT_FLOAT16))
    .INPUT(learning_rate, TensorType(DT_FLOAT, DT_FLOAT16))
    .INPUT(accum, TensorType(DT_FLOAT, DT_FLOAT16))
    .INPUT(momentum, TensorType(DT_FLOAT, DT_FLOAT16))
    .INPUT(stat, TensorType(DT_FLOAT, DT_FLOAT16))
    .OUTPUT(parameters, TensorType(DT_FLOAT, DT_FLOAT16))
    .ATTR(dampening, Float, 0.0)
    .ATTR(weight_decay, Float, 0.0)
    .ATTR(nesterov, Bool, false)
    .OP_END_FACTORY_REG(SGD)

/**
* @brief Updates "var" according to the RMSProp algorithm.
*    mean_square = decay * mean_square + (1-decay) * gradient ** 2
*    Delta = learning_rate * gradient / sqrt(mean_square + epsilon)
*    ms <- rho * ms_{t-1} + (1-rho) * grad * grad
*    mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
*    var <- var - mom
*
* @par Inputs:
* @li var: A mutable tensor. Must be one of the data types defined in
* TensorType::NumberType(). Should be from a Variable().
* @li ms: A mutable tensor. Must have the same type as "var". Should be from a
* Variable().
* @li mom: A mutable tensor. Must have the same type as "var". Should be from a
* Variable().
* @li lr: A scalar. Must have the same type as "var".
* @li rho: A scalar. Must have the same type as "var".
* @li momentum: A scalar. Must have the same type as "var".
* @li epsilon: A scalar. Must have the same type as "var".
* @li grad: A tensor, specifying the gradient. Must have the same type as "var".
*
* @par Attributes:
* use_locking: An optional "bool". Defaults to "False". If "True", updating of
* the "var", "ms", and "mom" tensors will be protected by a lock; otherwise the
* behavior is undefined, but may exhibit less contention.
*
* @par Outputs:
* var: A mutable tensor. Has the same type as input "var".
*
* @attention Constraints:
* @li Note that in dense implementation of this algorithm, "ms" and "mom" will
* update even if "grad" is 0, but in this sparse implementation, "ms" and "mom"
* will not update in iterations during which "grad" is 0.
* @li The input tensors "var", "ms", "mom" and "grad" must have the same shape.
*
* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator ApplyRMSProp.
*/
REG_OP(ApplyRMSProp)
    .INPUT(var, TensorType::NumberType())
    .INPUT(ms, TensorType::NumberType())
    .INPUT(mom, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(rho, TensorType::NumberType())
    .INPUT(momentum, TensorType::NumberType())
    .INPUT(epsilon, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyRMSProp)

/**
* @brief Updates "var" according to the RMSProp algorithm, a const input will be
* considered as an attribute.
*     mean_square = decay * mean_square + (1-decay) * gradient ** 2
*     Delta = learning_rate * gradient / sqrt(mean_square + epsilon)
*     ms <- rho * ms_{t-1} + (1-rho) * grad * grad
*     mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
*     var <- var - mom
*
* @par Inputs:
* @li var: A mutable tensor. Must be one of the data types defined in
* TensorType::NumberType(). Should be from a Variable().
* @li ms: A mutable tensor. Must have the same type as "var". Should be from a
* Variable().
* @li mom: A mutable tensor. Must have the same type as "var". Should be from a
* Variable().
* @li lr: A scalar. Must have the same type as "var".
* @li grad: A tensor, specifying the gradient. Must have the same type as "var".
*
* @par Attributes:
* @li use_locking: An optional "bool". Defaults to "False". If "True", updating
* of the "var", "ms", and "mom" tensors will be protected by a lock;
* otherwise the behavior is undefined, but may exhibit less contention.
* @li rho: A required scalar. Must have the same type as "var".
* @li momentum: A required scalar. Must have the same type as "var".
* @li epsilon: A required scalar. Must have the same type as "var".
*
* @par Outputs:
* var: A mutable tensor. Must have the same type as input "var".
*
* @attention Constraints:
* @li Note that in dense implementation of this algorithm, "ms" and "mom" will
* update even if "grad" is 0, but in this sparse implementation, "ms" and "mom"
* will not update in iterations during which "grad" is 0.
* @li The input tensors "var", "ms", "mom" and "grad" must have the same shape.
*
* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator ApplyRMSProp.
*
*@par Restrictions:
*Warning: THIS FUNCTION IS DEPRECATED. Please use ApplyRMSProp instead.
*/
REG_OP(ApplyRMSPropD)
    .INPUT(var, TensorType::NumberType())
    .INPUT(ms, TensorType::NumberType())
    .INPUT(mom, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(ms, TensorType::NumberType())
    .OUTPUT(mom, TensorType::NumberType())
    .REQUIRED_ATTR(rho, Float)
    .REQUIRED_ATTR(momentum, Float)
    .REQUIRED_ATTR(epsilon, Float)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyRMSPropD)

/**
*@brief Update "var" and "accum" according to FOBOS with Adagrad learning rate . \n

*@par Inputs:
*Six inputs, including:
* @li var: A mutable Tensor of type TensorType::NumberType().
*    Should be from a Variable().
* @li accum: A mutable Tensor of the same type as "var". Should be from a Variable().
* @li lr: A Tensor of the same type as "var", for the scaling factor. Must be a scalar.
* @li l1: A Tensor of the same type as "var", for L1 regulariation. Must be a scalar.
* @li l2: A Tensor of the same type as "var", for L2 regulariation. Must be a scalar.
* @li grad: A Tensor of the same type as "var", for the gradient . \n

*@par Attributes:
*use_locking: An optional bool. Defaults to "False". If "True", updating of the "var" and "accum" *tensors will be protected by a lock; otherwise the behavior is undefined, but may exhibit less *contention . \n

*@par Outputs:
*var: A mutable tensor. Must have the same type as input "var" . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyProximalAdagrad.
*/
REG_OP(ApplyProximalAdagrad)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(l1, TensorType::NumberType())
    .INPUT(l2, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyProximalAdagrad)

/**
*@brief Update "var" and "accum" according to FOBOS with Adagrad learning rate . \n

*@par Inputs:
*Six inputs, including:
* @li var: A mutable Tensor of type TensorType::NumberType().
*    Should be from a Variable().
* @li accum: A mutable Tensor of the same type as "var". Should be from a Variable().
* @li lr: A Tensor of the same type as "var", for the scaling factor. Must be a scalar.
* @li l1: A Tensor of the same type as "var", for L1 regulariation. Must be a scalar.
* @li l2: A Tensor of the same type as "var", for L2 regulariation. Must be a scalar.
* @li grad: A Tensor of the same type as "var", for the gradient . \n

*@par Attributes:
*use_locking: An optional bool. Defaults to "False". If "True", updating of the "var" and "accum" *tensors will be protected by a lock; otherwise the behavior is undefined, but may exhibit less *contention . \n

*@par Outputs:
* @li var: A mutable Tensor. Has the same type as "var".
* @li accum: A mutable Tensor. Has the same type as "var" . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyProximalAdagradD.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ApplyProximalAdagrad instead.
*/
REG_OP(ApplyProximalAdagradD)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(l1, TensorType::NumberType())
    .INPUT(l2, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(accum, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyProximalAdagradD)

/**
*@brief Updates entries in 'var' and 'accum' according to the Proximal Adagrad algorithm.
* Compared with op ApplyProximalAdagrad, an additional index tensor is input,
* Only the indices into the first dimensions of "var" and "accum" are updated . \n

*@par Inputs:
* Seven inputs, including:
* @li var: A mutable Tensor.
*     TensorType::NumberType(). Should be a Variable Tensor.
* @li accum: A mutable Tensor of the same type as "var".
*     Should be a Variable Tensor. Should be greater than or equal to zero.
*     Accum and grad cannot be equal to zero at the same time.
* @li lr: A Tensor of the same type as "var".
*     Scaling factor. Must be a scalar. Should be greater than zero.
* @li l1: A Tensor of the same type as "var".
*     L1 regulariation. Must be a scalar. Should be greater than or equal to zero.
* @li l2: A Tensor of the same type as "var".
*     L2 regulariation. Must be a scalar. Should be greater than or equal to zero.
* @li grad: A Tensor. Has the same type as "var".
*     The gradient.
* @li indices: A vector of indices into the first dimension of "var" and "accum".
*     TensorType::IndexNumberType(). Can contain duplicate values . \n

* @par Attributes:
* use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the var and accum tensors will be protected by a lock;
*     If "False", the behavior is undefined, but may exhibit less contention.

* @par Outputs:
* @li var: A mutable Tensor. Has the same type as "var" .
* @li accum: A mutable Tensor. Has the same type as "accum" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseApplyProximalAdagrad.
*/
REG_OP(SparseApplyProximalAdagrad)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(l1, TensorType::NumberType())
    .INPUT(l2, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .INPUT(indices, TensorType::IndexNumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(accum, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(SparseApplyProximalAdagrad)

/**
*@brief Updates entries in 'var' and 'accum' according to the Proximal Adagrad algorithm.\ n
* Compared with op ApplyProximalAdagrad, an additional index tensor is input,
* Only the indices into the first dimensions of "var" and "accum" are updated . \n

*@par Inputs:
* Seven inputs, including:
* @li var: A mutable Tensor.
*     TensorType::NumberType(). Should be a Variable Tensor.
* @li accum: A mutable Tensor of the same type as "var".
*     Should be a Variable Tensor. Should be greater than or equal to zero.
*     Accum and grad cannot be equal to zero at the same time.
* @li lr: A Tensor of the same type as "var".
*     Scaling factor. Must be a scalar. Should be greater than zero.
* @li l1: A Tensor of the same type as "var".
*     L1 regulariation. Must be a scalar. Should be greater than or equal to zero.
* @li l2: A Tensor of the same type as "var".
*     L2 regulariation. Must be a scalar. Should be greater than or equal to zero.
* @li grad: A Tensor. Has the same type as "var".
*     The gradient.
* @li indices: A vector of indices into the first dimension of "var" and "accum".
*     TensorType::IndexNumberType(). Can contain duplicate values . \n

*@par Attributes:
*use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the var and accum tensors will be protected by a lock;
*     If "False", the behavior is undefined, but may exhibit less contention . \n

*@par Outputs:
*@li var: A mutable Tensor. Has the same type as "var".
*@li accum:  A mutable Tensor. Has the same type as "var" . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator SparseApplyProximalAdagrad.

* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use SparseApplyProximalAdagrad instead.
*/
REG_OP(SparseApplyProximalAdagradD)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(l1, TensorType::NumberType())
    .INPUT(l2, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .INPUT(indices, TensorType::IndexNumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(accum, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(SparseApplyProximalAdagradD)

/**
*@brief Updates "var" according to the Ftrl-proximal scheme . \n

*@par Inputs:
*Eight inputs, including:
* @li var: A mutable Tensor. Must be of type TensorType::NumberType().
*     Should be a Variable Tensor.
* @li accum: A mutable Tensor of the same type as "var".
*     Should be a Variable Tensor.
* @li linear: A mutable Tensor of the same type as "var".
*     Should be a Variable Tensor.
* @li grad: A Tensor of the same type as "var", for the gradient.
* @li lr: A Tensor of the same type as "var", for the scaling factor. Must be a scalar.
* @li l1: A Tensor of the same type as "var", for L1 regulariation. Must be a scalar.
* @li l2: A Tensor of the same type as "var", for L2 regulariation. Must be a scalar.
* @li lr_power: A Tensor of the same type as "var", for the scaling factor. Must be a scalar . \n

*@par Attributes:
*use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var" and "accum" tensors will be
*     protected by a lock; otherwise the behavior is undefined,
*     but may exhibit less contention . \n

*@par Outputs:
*var: A mutable Tensor. Has the same type as "var" . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyFtrl.
*/
REG_OP(ApplyFtrl)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(linear, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(l1, TensorType::NumberType())
    .INPUT(l2, TensorType::NumberType())
    .INPUT(lr_power, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyFtrl)

/**
*@brief Updates "var" according to the Ftrl-proximal scheme . \n

*@par Inputs:
*Eight inputs, including:
* @li var: A mutable Tensor. Must be of type TensorType::NumberType().
*     Should be a Variable Tensor.
* @li accum: A mutable Tensor of the same type as "var".
*     Should be a Variable Tensor.
* @li linear: A mutable Tensor of the same type as "var".
*     Should be a Variable Tensor.
* @li grad: A Tensor of the same type as "var", for the gradient.
* @li lr: A Tensor of the same type as "var", for the scaling factor. Must be a scalar.
* @li l1: A Tensor of the same type as "var", for L1 regulariation. Must be a scalar.
* @li l2: A Tensor of the same type as "var", for L2 regulariation. Must be a scalar.
* @li lr_power: A Tensor of the same type as "var", for the scaling factor. Must be a scalar . \n

*@par Attributes:
*use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var" and "accum" tensors will be
*     protected by a lock; otherwise the behavior is undefined,
*     but may exhibit less contention . \n

*@par Outputs:
*@li var: A mutable Tensor. Has the same type as "var".
*@li accum: A mutable Tensor. Has the same type as "accum".
*@li linear: A mutable Tensor. Has the same type as "linear" . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyFtrl.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ApplyFtrl instead.
*/
REG_OP(ApplyFtrlD)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(linear, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(l1, TensorType::NumberType())
    .INPUT(l2, TensorType::NumberType())
    .INPUT(lr_power, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(accum, TensorType::NumberType())
    .OUTPUT(linear, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyFtrlD)

/**
*@brief Update "var" according to the Ftrl-proximal scheme . \n

*@par Inputs:
*Nine inputs, including:
* @li var: A mutable Tensor. Must be of type TensorType::NumberType().
*     Should be a Variable Tensor.
* @li accum: A mutable Tensor of the same type as "var".
*     Should be a Variable Tensor.
* @li linear: A mutable Tensor of the same type as "var".
*     Should be a Variable Tensor.
* @li grad: A Tensor of the same type as "var", for the gradient.
* @li lr: A Tensor of the same type as "var", for the scaling factor. Must be a scalar.
* @li l1: A Tensor of the same type as "var", for L1 regulariation. Must be a scalar.
* @li l2: A Tensor of the same type as "var", for L2 regulariation. Must be a scalar.
* @li l2_shrinkage: A Tensor of the same type as "var".
* @li lr_power: A Tensor of the same type as "var", for the scaling factor. Must be a scalar . \n

*@par Attributes:
*use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var" and "accum" tensors will be
*     protected by a lock; otherwise the behavior is undefined,
*     but may exhibit less contention . \n

*@par Outputs:
*var: A mutable Tensor. Has the same type as "var" . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyFtrlV2.
*/
REG_OP(ApplyFtrlV2)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(linear, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(l1, TensorType::NumberType())
    .INPUT(l2, TensorType::NumberType())
    .INPUT(l2_shrinkage, TensorType::NumberType())
    .INPUT(lr_power, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyFtrlV2)

/**
*@brief Update "var" according to the Ftrl-proximal scheme . \n

*@par Inputs:
*Nine inputs, including:
* @li var: A mutable Tensor. Must be of type TensorType::NumberType().
*     Should be a Variable Tensor.
* @li accum: A mutable Tensor of the same type as "var".
*     Should be a Variable Tensor.
* @li linear: A mutable Tensor of the same type as "var".
*     Should be a Variable Tensor.
* @li grad: A Tensor of the same type as "var", for the gradient.
* @li lr: A Tensor of the same type as "var", for the scaling factor. Must be a scalar.
* @li l1: A Tensor of the same type as "var", for L1 regulariation. Must be a scalar.
* @li l2: A Tensor of the same type as "var", for L2 regulariation. Must be a scalar.
* @li l2_shrinkage: A Tensor of the same type as "var".
* @li lr_power: A Tensor of the same type as "var", for the scaling factor. Must be a scalar . \n

*@par Attributes:
*use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var" and "accum" tensors will be
*     protected by a lock; otherwise the behavior is undefined,
*     but may exhibit less contention . \n

*@par Outputs:
*var: A mutable Tensor. Has the same type as "var".
*accum: A mutable Tensor. Has the same type as "accum".
*linear: A mutable Tensor. Has the same type as "linear" . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyFtrlV2.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ApplyFtrlV2 instead.
*/
REG_OP(ApplyFtrlV2D)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(linear, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(l1, TensorType::NumberType())
    .INPUT(l2, TensorType::NumberType())
    .INPUT(l2_shrinkage, TensorType::NumberType())
    .INPUT(lr_power, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(accum, TensorType::NumberType())
    .OUTPUT(linear, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyFtrlV2D)

/**
*@brief Updates "var" according to the Adam algorithm.
*  lr_t <- text{learning\_rate} * sqrt{1 - beta_2^t} / (1 - beta_1^t)
*  m_t <- beta_1 * m_{t-1} + (1 - beta_1) * g
*  v_t <- max(beta2 * v{t-1}, abs(g))
*  variable <- variable - lr_t * m_t / (sqrt{v_t} + epsilon)
*
*@attention Constraints:
*  *The input tensors must have the same shape.*
*
*@par Inputs:
*@li var: A mutable Tensor of the type TensorType::NumberType().
*     Should be from a Variable().
*@li m: A mutable Tensor of the same type as "var".
*     Should be from a Variable().
*@li v: A mutable Tensor of the same type as "var".
*     Should be from a Variable().
*@li beta1_power: A scalar of the same type as "var".
*@li beta2_power: A scalar of the same type as "var".
*@li lr: learning_rate. A scalar of the same type as "var".
*@li beta1: A scalar of the same type as "var".
*@li beta2: A scalar of the same type as "var".
*@li epsilon: A scalar of the same type as "var".
*@li grad: A Tensor of the same type as "var", for the gradient.
*
*@par Attributes:
*@li use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var", m", and "v" tensors will be protected
*     by a lock; otherwise the behavior is undefined, but may exhibit less
*     contention.
*@li use_nesterov: An optional bool. Defaults to "False".
      If "True", uses the nesterov update.
*
*@par Outputs:
* var: A mutable Tensor. Has the same type as intput "var" . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyAdam.
*/
REG_OP(ApplyAdam)
    .INPUT(var, TensorType::NumberType())
    .INPUT(m, TensorType::NumberType())
    .INPUT(v, TensorType::NumberType())
    .INPUT(beta1_power, TensorType::NumberType())
    .INPUT(beta2_power, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(beta1, TensorType::NumberType())
    .INPUT(beta2, TensorType::NumberType())
    .INPUT(epsilon, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .ATTR(use_nesterov, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdam)

/**
*@brief Updates "var" according to the Adam algorithm.
*  lr_t <- text{learning\_rate} * sqrt{1 - beta_2^t} / (1 - beta_1^t)
*  m_t <- beta_1 * m_{t-1} + (1 - beta_1) * g
*  v_t <- max(beta2 * v{t-1}, abs(g))
*  variable <- variable - lr_t * m_t / (sqrt{v_t} + epsilon)
*
*@attention Constraints:
*  *The input tensors must have the same shape.*
*
*@par Inputs:
*@li var: A mutable Tensor of the type TensorType::NumberType().
*     Should be from a Variable().
*@li m: A mutable Tensor of the same type as "var".
*     Should be from a Variable().
*@li v: A mutable Tensor of the same type as "var".
*     Should be from a Variable().
*@li beta1_power: A scalar of the same type as "var".
*@li beta2_power: A scalar of the same type as "var".
*@li lr: learning_rate. A scalar of the same type as "var".
*@li beta1: A scalar of the same type as "var".
*@li beta2: A scalar of the same type as "var".
*@li epsilon: A scalar of the same type as "var".
*@li grad: A Tensor of the same type as "var", for the gradient.
*
*@par Attributes:
*@li use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var", m", and "v" tensors will be protected
*     by a lock; otherwise the behavior is undefined, but may exhibit less
*     contention.
*@li use_nesterov: An optional bool. Defaults to "False".
      If "True", uses the nesterov update.
*
*@par Outputs:
*@li var: A mutable tensor. Has the same type as input "var".
*@li m: A mutable tensor. Has the same type as input "m".
*@li v: A mutable tensor. Has the same type as input "v" . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyAdam.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ApplyAdam instead.
*/
REG_OP(ApplyAdamD)
    .INPUT(var, TensorType::NumberType())
    .INPUT(m, TensorType::NumberType())
    .INPUT(v, TensorType::NumberType())
    .INPUT(beta1_power, TensorType::NumberType())
    .INPUT(beta2_power, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(beta1, TensorType::NumberType())
    .INPUT(beta2, TensorType::NumberType())
    .INPUT(epsilon, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(m, TensorType::NumberType())
    .OUTPUT(v, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .ATTR(use_nesterov, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdamD)

/**
*@brief Updates "var" according to the proximal adadelta scheme . \n

*@par Inputs:
*Seven inputs, including:
* @li var: A mutable Tensor of type TensorType::NumberType().
*     Should be a Variable Tensor.
* @li accum: A mutable Tensor of the same type as "var".
*     Should be a Variable Tensor.
* @li accum_update: A mutable Tensor of the same type as "var".
*     Should be a Variable Tensor.
* @li lr: A scalar of the same type as "var", for the scaling factor.
* @li rho: A scalar of the same type as "var", for the decay factor.
* @li epsilon: A scalar of the same type as "var", for the constant factor.
* @li grad: A Tensor of the same type as "var", for the gradient . \n

*@par Attributes:
*use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var", "accum" and "accum_update" tensors will be
*     protected by a lock; otherwise the behavior is undefined,
*     but may exhibit less contention . \n

*@par Outputs:
*var: A mutable Tensor. Has the same type as "var" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ApplyAdadelta.
*/
REG_OP(ApplyAdadelta)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(accum_update, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(rho, TensorType::NumberType())
    .INPUT(epsilon, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdadelta)

/**
*@brief Updates "var" according to the proximal adadelta scheme . \n

*@par Inputs:
*Seven inputs, including:
* @li var: A mutable Tensor of type TensorType::NumberType().
*     Should be a Variable Tensor.
* @li accum: A mutable Tensor of the same type as "var".
*     Should be a Variable Tensor.
* @li accum_update: A mutable Tensor of the same type as "var".
*     Should be a Variable Tensor.
* @li lr: A scalar of the same type as "var", for the scaling factor.
* @li rho: A scalar of the same type as "var", for the decay factor.
* @li epsilon: A scalar of the same type as "var", for the constant factor.
* @li grad: A Tensor of the same type as "var", for the gradient . \n

*@par Attributes:
*use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var", "accum" and "accum_update" tensors will be
*     protected by a lock; otherwise the behavior is undefined,
*     but may exhibit less contention . \n

*@par Outputs:
*@li var: A mutable Tensor. Has the same type as "var".
*@li accum: A mutable Tensor. Has the same type as "var".
*@li accum_update: A mutable Tensor. Has the same type as "var" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ApplyAdadelta.

* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ApplyAdadelta instead.
*/
REG_OP(ApplyAdadeltaD)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(accum_update, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(rho, TensorType::NumberType())
    .INPUT(epsilon, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(accum, TensorType::NumberType())
    .OUTPUT(accum_update, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdadeltaD)

/**
*@brief Updates "var" according to the ApplyMomentum algorithm.
* accum = accum * momentum + x1 * x2
* if use_nesterov is True:
* var -= x1 * x2 * lr + accum * momentum * lr
* else: var -= accum * lr
*
*@par Inputs:
* Six inputs, including:
*@li var: A mutable Tensor has type TensorType::NumberType().
* Should be a Variable Tensor.
*@li accum: A mutable Tensor has the same type as "var".
* Should be a Variable Tensor.
*@li lr: A scalar has the same type as "var", for the scaling factor.
*@li x1: A Tensor has type TensorType::NumberType().
*@li momentum: A scalar has the same type as "var".
*@li x2: A scalar has the same type as "var". \n
*
*@par Attributes:
* Two attributes, including:
*@li use_nesterov: An optional bool. Defaults to "False".
* If True, the tensor passed to compute grad will be
* var - lr * momentum * accum, so in the end,
* the var you get is actually var - lr * momentum * accum.
*@li use_locking: An optional bool. Defaults to "False".
* If "True", updating of the "var", m", and "v" tensors will be protected
* by a lock; otherwise the behavior is undefined, but may exhibit
* less contention. \n
*
*@par Outputs:
* Two outputs, including:
*@li var: A mutable Tensor has the same type as "var".
*@li accum: A mutable Tensor has the same type as "var". \n

*@par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(FusedMulApplyMomentum)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(x1, TensorType::NumberType())
    .INPUT(momentum, TensorType::NumberType())
    .INPUT(x2, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(accum, TensorType::NumberType())
    .ATTR(use_nesterov, Bool, false)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(FusedMulApplyMomentum)

/**
* @brief Updates "var" according to the ApplyMomentum algorithm.
*   accum = accum * momentum + x1 * x2
*   if use_nesterov is True:
*       var -= x1 * x2 * lr + accum * momentum * lr
*   else:
*       var -= accum * lr
*
* @par Inputs:
*   Seven inputs, including:
*  @li var: A mutable Tensor of type float32.
*     Should be a Variable Tensor.
*  @li accum: A mutable Tensor has type TensorType::NumberType().
*     Should be a Variable Tensor.
*  @li lr: A scalar has the same type as "accum", for the scaling factor.
*  @li x1: A Tensor has the same type as "accum".
*  @li momentum: A scalar has the same type as "accum".
*  @li x2: A scalar has the same type as "accum".
*  @li var_copy: A Tensor has type float16.
*
* @par Attributes:
*   Two Attributes, including:
*  @li use_nesterov: An optional bool. Defaults to "False".
*     If True, the tensor passed to compute grad will be var - lr * momentum * accum,
*     so in the end, the var you get is actually var - lr * momentum * accum.
*  @li use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var", m", and "v" tensors will be protected
*     by a lock; otherwise the behavior is undefined, but may exhibit less contention.
*
* @par Outputs:
*   Three outputs, including:
*  @li var: A Tensor has the type float32.
*  @li var_copy: A Tensor has the type float16.
*  @li accum: A Tensor has the same type as input "accum".

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(FusedMulApplyMomentumExtern)
    .INPUT(var, TensorType(DT_FLOAT))
    .INPUT(accum, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(x1, TensorType::NumberType())
    .INPUT(momentum, TensorType::NumberType())
    .INPUT(x2, TensorType::NumberType())
    .INPUT(var_copy, TensorType(DT_FLOAT16))
    .OUTPUT(var, TensorType(DT_FLOAT))
    .OUTPUT(var_copy, TensorType(DT_FLOAT16))
    .OUTPUT(accum, TensorType::NumberType())
    .ATTR(use_nesterov, Bool, false)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(FusedMulApplyMomentumExtern)

/**
*@brief Updates '*var' according to the momentum scheme.
*   accum = accum * momentum - x1 * x2 * lr
*   if use_nesterov is True:
*       var += accum * momentum - x1 * x2 * lr
*   else:
*       var += accum
*
*@par Inputs:
*@li var: A mutable tensor. Must be one of the data types defined in
*    TensorType::NumberType(). Should be from a Variable().
*@li accum: A mutable tensor. Has the same type as "var". Should be from a
*    Variable().
*@li lr: A tensor for the learning rate. Has the same type as "var". Should be
*    from a Variable().
*@li x1: A Tensor has type TensorType::NumberType().
*@li momentum: A scalar. Has the same type as "var".
*@li x2: A scalar has the same type as "var".
*
*@par Attributes:
*@li use_nesterov: An optional bool. Defaults to "False".
*    If "True", var will be updated by using Nesterov momentum.
*@li use_locking: An optional bool. Defaults to "False".
*    If "True", updating of the "var" tensor is protected by a lock;
*    otherwise the behavior is undefined, but may exhibit less contention.
*
*@par Outputs:
* @li var: A mutable tensor. Has the same type as input "var".
* @li accum: A mutable tensor. Has the same type as input "accum".
*
*@attention Constraints:
* @li var: A mutable tensor. Has the same type as input "var".
* @li accum: A mutable tensor. Has the same type as input "accum".
*
*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ResourceApplyKerasMomentum.
*
*/
REG_OP(FusedMulApplyKerasMomentum)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(x1, TensorType::NumberType())
    .INPUT(momentum, TensorType::NumberType())
    .INPUT(x2, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(accum, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .ATTR(use_nesterov, Bool, false)
    .OP_END_FACTORY_REG(FusedMulApplyKerasMomentum)

/**
*@brief Update "g" according to the LARS algorithm . \n

*@par Inputs:
*Four inputs, including:
* @li w: A Tensor. Must be of type TensorType::DT_FLOAT.
* @li g: A Tensor of the same type and shape as "w".
* @li weight_decay: A Tensor of the same type as "w",  Must be a scalar.
* @li learning_rate: A Tensor of the same type as "w", Must be a scalar . \n

*@par Attributes:
*Three Attributes, including:
* @li hyperpara: An optional float. Default value is 0.001.
* @li epsilon: An optional float. Default value is 1e-5.Avoid denominator is 0.
* @li use_clip: An optional bool. Defaults to "False".
*     If "True", updating learning rate . \n

*@par Outputs:
*g_new: Tensor of the same type as "w".
*/
REG_OP(LarsV2)
    .INPUT(w, TensorType(DT_FLOAT))
    .INPUT(g, TensorType(DT_FLOAT))
    .INPUT(weight_decay, TensorType(DT_FLOAT))
    .INPUT(learning_rate, TensorType(DT_FLOAT))
    .OUTPUT(g_new, TensorType(DT_FLOAT))
    .ATTR(hyperpara, Float, 0.001)
    .ATTR(epsilon, Float, 0.00001)
    .ATTR(use_clip, Bool, false)
    .OP_END_FACTORY_REG(LarsV2)

/**
*@brief Update "g" according to the LARS algorithm . \n

*@par Inputs:
*Six inputs, including:
* @li w: A Tensor. Must be of type TensorType::DT_FLOAT.
* @li g: A Tensor of the same type and shape as "w".
* @li w_square_sum: A Tensor of  square_sum(w), has the same type as "w",  Must be a scalar.
* @li g_square_sum: A Tensor of  square(g), has the same type as "w", Must be a scalar.
* @li weight_decay: A Tensor of the same type as "w",  Must be a scalar.
* @li learning_rate: A Tensor of the same type as "w", Must be a scalar . \n

*@par Attributes:
*Three Attributes, including:
* @li hyperpara: An optional float. Default value is 0.001.
* @li epsilon: An optional float. Default value is 1e-5.Avoid denominator is 0.
* @li use_clip: An optional bool. Defaults to "False".
*     If "True", updating learning rate . \n

*@par Outputs:
*g_new: Tensor of the same type as "w".
*/
REG_OP(LarsV2Update)
    .INPUT(w, TensorType(DT_FLOAT))
    .INPUT(g, TensorType(DT_FLOAT))
    .INPUT(w_square_sum, TensorType(DT_FLOAT))
    .INPUT(g_square_sum, TensorType(DT_FLOAT))
    .INPUT(weight_decay, TensorType(DT_FLOAT))
    .INPUT(learning_rate, TensorType(DT_FLOAT))
    .OUTPUT(g_new, TensorType(DT_FLOAT))
    .ATTR(hyperpara, Float, 0.001)
    .ATTR(epsilon, Float, 0.00001)
    .ATTR(use_clip, Bool, false)
    .OP_END_FACTORY_REG(LarsV2Update)

/**
* @brief Update relevant entries in '*var' according to the Ftrl-proximal scheme . \n

* @par Inputs:
* Nine inputs, including:
* @li var: A mutable Tensor. Must be of type TensorType::NumberType().
* Should be a Variable Tensor.
* @li accum: A mutable Tensor of the same type as "var".
* Should be a Variable Tensor. The value of accum must be greater than 0.
* @li linear: A mutable Tensor of the same type as "var".
* Should be a Variable Tensor.
* @li grad: A Tensor of the same type as "var", for the gradient.
* @li indices: A vector of indices into the first dimension of var and accum.
* The value of indices must be unique. Otherwise, the result is unpredictable.
* @li lr: A Tensor of the same type as "var", for the scaling factor. Must be a scalar.
* @li l1: A Tensor of the same type as "var", for L1 regulariation. Must be a scalar.
* @li l2: A Tensor of the same type as "var", for L2 regulariation. Must be a scalar.
* @li lr_power: A Tensor of the same type as "var", for the scaling factor. Must be a scalar . \n

* @par Attributes:
* use_locking: An optional bool. Defaults to "False".
* If "True", updating of the "var" and "accum" tensors will be
* protected by a lock; otherwise the behavior is undefined,
* but may exhibit less contention . \n

* @par Outputs:
* @li var: A Tensor. Has the same type and format as input "var" .
* @li accum: A Tensor. Has the same type and format as input "accum".
* @li linear: A Tensor. Has the same type and format as input "linear" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseApplyFtrl.
*/
REG_OP(SparseApplyFtrl)
    .INPUT(var, TensorType({DT_FLOAT}))
    .INPUT(accum, TensorType({DT_FLOAT}))
    .INPUT(linear, TensorType({DT_FLOAT}))
    .INPUT(grad, TensorType({DT_FLOAT}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(lr, TensorType({DT_FLOAT}))
    .INPUT(l1, TensorType({DT_FLOAT}))
    .INPUT(l2, TensorType({DT_FLOAT}))
    .INPUT(lr_power, TensorType({DT_FLOAT}))
    .OUTPUT(var, TensorType({DT_FLOAT}))
    .OUTPUT(accum, TensorType({DT_FLOAT}))
    .OUTPUT(linear, TensorType({DT_FLOAT}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(SparseApplyFtrl)

/**
* @brief Update relevant entries in '*var' according to the Ftrl-proximal scheme . \n

* @par Inputs:
* Five inputs, including:
* @li var: A mutable Tensor. Must be of type TensorType::NumberType().
* Should be a Variable Tensor.
* @li accum: A mutable Tensor of the same type as "var".
* Should be a Variable Tensor. The value of accum must be greater than 0.
* @li linear: A mutable Tensor of the same type as "var".
* Should be a Variable Tensor.
* @li grad: A Tensor of the same type as "var", for the gradient.
* @li indices: A vector of indices into the first dimension of var and accum.
* The value of indices must be unique. Otherwise, the result is unpredictable . \n

* @par Attributes:
* @li lr: A Tensor of the same type as "var", for the scaling factor. Must be a scalar.
* @li l1: A Tensor of the same type as "var", for L1 regulariation. Must be a scalar.
* @li l2: A Tensor of the same type as "var", for L2 regulariation. Must be a scalar.
* @li lr_power: A Tensor of the same type as "var", for the scaling factor. Must be a scalar.
* @li use_locking: An optional bool. Defaults to "False".
* If "True", updating of the "var" and "accum" tensors will be
* protected by a lock; otherwise the behavior is undefined,
* but may exhibit less contention . \n

* @par Outputs:
* @li var: A Tensor. Has the same type and format as input "var".
* @li accum: A Tensor. Has the same type and format as input "accum".
* @li linear: A Tensor. Has the same type and format as input "linear" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseApplyFtrl.
*
*@par Restrictions:
*Warning: THIS FUNCTION IS DEPRECATED. Please use SparseApplyFtrl instead.
*/
REG_OP(SparseApplyFtrlD)
    .INPUT(var, TensorType({DT_FLOAT}))
    .INPUT(accum, TensorType({DT_FLOAT}))
    .INPUT(linear, TensorType({DT_FLOAT}))
    .INPUT(grad, TensorType({DT_FLOAT}))
    .INPUT(indices, TensorType({DT_INT32}))
    .OUTPUT(var, TensorType({DT_FLOAT}))
    .OUTPUT(accum, TensorType({DT_FLOAT}))
    .OUTPUT(linear, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(lr, Float)
    .REQUIRED_ATTR(l1, Float)
    .REQUIRED_ATTR(l2, Float)
    .REQUIRED_ATTR(lr_power, Float)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(SparseApplyFtrlD)

/**
* @brief Updates relevant entries in '*var' according to the Ftrl-proximal scheme.
* That is for rows we have grad for, "var", "accum" and "linear" are updated . \n

* @par Inputs:
* Ten inputs, including:
* @li var: A mutable Tensor. Must be of type TensorType::NumberType().
*     Should be a Variable Tensor.
* @li accum: A mutable Tensor of the same type as "var".
*     Should be a Variable Tensor.
* @li linear: A mutable Tensor of the same type as "var".
*     Should be a Variable Tensor.
* @li grad: A Tensor of the same type as "var", for the gradient.
* @li indices: A vector of indices into the first dimension of "var" and "accum".
* @li lr: A Tensor of the same type as "var", for the scaling factor. Must be a scalar.
* @li l1: A Tensor of the same type as "var", for L1 regulariation. Must be a scalar.
* @li l2: A Tensor of the same type as "var", for L2 regulariation. Must be a scalar.
* @li l2_shrinkage: A Tensor of the same type as "var", L2 shrinkage regulariation. Must be a scalar.
* @li lr_power: A Tensor of the same type as "var", for the scaling factor. Must be a scalar . \n

* @par Attributes:
* use_locking: An optional bool. Defaults to "False".
* If "True", updating of the "var" and "accum" tensors will be
* protected by a lock; otherwise the behavior is undefined,
* but may exhibit less contention . \n

* @par Outputs:
* @li var: A Tensor. Has the same type and format as input "var" .
* @li accum: A Tensor. Has the same type and format as input "accum".
* @li linear: A Tensor. Has the same type and format as input "linear" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseApplyFtrlV2.
*/
REG_OP(SparseApplyFtrlV2)
    .INPUT(var, TensorType({DT_FLOAT}))
    .INPUT(accum, TensorType({DT_FLOAT}))
    .INPUT(linear, TensorType({DT_FLOAT}))
    .INPUT(grad, TensorType({DT_FLOAT}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(lr, TensorType({DT_FLOAT}))
    .INPUT(l1, TensorType({DT_FLOAT}))
    .INPUT(l2, TensorType({DT_FLOAT}))
    .INPUT(l2_shrinkage, TensorType({DT_FLOAT}))
    .INPUT(lr_power, TensorType({DT_FLOAT}))
    .OUTPUT(var, TensorType({DT_FLOAT}))
    .OUTPUT(accum, TensorType({DT_FLOAT}))
    .OUTPUT(linear, TensorType({DT_FLOAT}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(SparseApplyFtrlV2)

/**
* @brief Updates relevant entries in '*var' according to the Ftrl-proximal scheme.
* That is for rows we have grad for, "var", "accum" and "linear" are updated . \n

* @par Inputs:
* Five inputs, including:
* @li var: A mutable Tensor. Must be of type TensorType::NumberType().
* Should be a Variable Tensor.
* @li accum: A mutable Tensor of the same type as "var".
* Should be a Variable Tensor.
* @li linear: A mutable Tensor of the same type as "var".
* Should be a Variable Tensor.
* @li grad: A Tensor of the same type as "var", for the gradient.
* @li indices: A vector of indices into the first dimension of "var" and "accum" . \n

* @par Attributes:
* @li lr: A Tensor of the same type as "var", for the scaling factor. Must be a scalar.
* @li l1: A Tensor of the same type as "var", for L1 regulariation. Must be a scalar.
* @li l2: A Tensor of the same type as "var", for L2 regulariation. Must be a scalar.
* @li l2_shrinkage: A Tensor of the same type as "var", L2 shrinkage regulariation. Must be a scalar.
* @li lr_power: A Tensor of the same type as "var", for the scaling factor. Must be a scalar.
* @li use_locking: An optional bool. Defaults to "False".
* If "True", updating of the "var" and "accum" tensors will be
* protected by a lock; otherwise the behavior is undefined,
* but may exhibit less contention . \n

* @par Outputs:
* @li var: A Tensor. Has the same type and format as input "var".
* @li accum: A Tensor. Has the same type and format as input "accum".
* @li linear: A Tensor. Has the same type and format as input "linear" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseApplyFtrlV2D.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use SparseApplyFtrlV2 instead.
*/
REG_OP(SparseApplyFtrlV2D)
    .INPUT(var, TensorType({DT_FLOAT}))
    .INPUT(accum, TensorType({DT_FLOAT}))
    .INPUT(linear, TensorType({DT_FLOAT}))
    .INPUT(grad, TensorType({DT_FLOAT}))
    .INPUT(indices, TensorType({DT_INT32}))
    .OUTPUT(var, TensorType({DT_FLOAT}))
    .OUTPUT(accum, TensorType({DT_FLOAT}))
    .OUTPUT(linear, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(lr, Float)
    .REQUIRED_ATTR(l1, Float)
    .REQUIRED_ATTR(l2, Float)
    .REQUIRED_ATTR(l2_shrinkage, Float)
    .REQUIRED_ATTR(lr_power, Float)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(SparseApplyFtrlV2D)

/**
* @brief Updates "var" in specified index according to the RMSProp algorithm.
*    mean_square = decay * mean_square + (1-decay) * gradient ** 2
*    Delta = learning_rate * gradient / sqrt(mean_square + epsilon)
*    ms <- rho * ms_{t-1} + (1-rho) * grad * grad
*    mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
*    var <- var - mom
*
* @par Inputs:
* Nine inputs, including:
* @li var: A mutable tensor. Must be one of the data types defined in
* TensorType::NumberType(). Should be from a Variable().
* @li ms: A mutable tensor. Must have the same type as "var". Should be from a
* Variable().
* @li mom: A mutable tensor. Must have the same type as "var". Should be from a
* Variable().
* @li lr: A scalar. Must have the same type as "var".
* @li rho: A scalar. Must have the same type as "var".
* @li momentum: A scalar. Must have the same type as "var".
* @li epsilon: A scalar. Must have the same type as "var".
* @li grad: A tensor, specifying the gradient.
* @li indices: A vector of indices into the first dimension of "var", "mom" and "ms".
*
* @par Attributes:
* use_locking: An optional "bool". Defaults to "False". If "True", updating of
* the "var", "ms", and "mom" tensors will be protected by a lock; otherwise the
* behavior is undefined, but may exhibit less contention.
*
* @par Outputs:
* @li var: A mutable tensor. Has the same type as input "var".
* @li ms:  A mutable tensor. Must have the same type as input "ms".
* @li mom: A mutable tensor. Must have the same type as input "mom".
*
* @attention Constraints:
* @li Note that in this sparse implementation, "ms" and "mom" will not update
* in iterations during which "grad" is 0.
* @li The input tensors "var", "ms", and "mom" must have the same shape.
*
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseApplyRMSProp.
*/
REG_OP(SparseApplyRMSProp)
    .INPUT(var, TensorType::NumberType())
    .INPUT(ms, TensorType::NumberType())
    .INPUT(mom, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(rho, TensorType::NumberType())
    .INPUT(momentum, TensorType::NumberType())
    .INPUT(epsilon, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .INPUT(indices, TensorType::IndexNumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(ms, TensorType::NumberType())
    .OUTPUT(mom, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(SparseApplyRMSProp)

/**
* @brief Updates "var" in specified index according to the RMSProp algorithm.
* a const input will be considered as an attribute.
*     mean_square = decay * mean_square + (1-decay) * gradient ** 2
*     Delta = learning_rate * gradient / sqrt(mean_square + epsilon)
*     ms <- rho * ms_{t-1} + (1-rho) * grad * grad
*     mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
*     var <- var - mom
*
* @par Inputs:
* Six inputs, including:
* @li var: A mutable tensor. Must be one of the data types defined in
* TensorType::NumberType(). Should be from a Variable().
* @li ms: A mutable tensor. Must have the same type as "var". Should be from a
* Variable().
* @li mom: A mutable tensor. Must have the same type as "var". Should be from a
* Variable().
* @li lr: A scalar. Must have the same type as "var".
* @li grad: A tensor, specifying the gradient.
*
* @par Attributes:
* @li use_locking: An optional "bool". Defaults to "False". If "True",
* updating of the "var", "ms", and "mom" tensors will be protected by a lock;
* otherwise the behavior is undefined, but may exhibit less contention.
* @li rho: A required scalar. Must have the same type as "var".
* @li momentum: A required scalar. Must have the same type as "var".
* @li epsilon: A required scalar. Must have the same type as "var".
*
* @par Outputs:
* @li var: A mutable tensor. Must have the same type as input "var".
* @li ms:  A mutable tensor. Must have the same type as input "ms".
* @li mom: A mutable tensor. Must have the same type as input "mom".
*
* @attention Constraints:
* @li Note that in this sparse implementation, "ms" and "mom" will not update
* in iterations during which "grad" is 0.
* @li The input tensors "var", "ms" and "mom" must have the same shape.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use SparseApplyRMSProp instead.
*/
REG_OP(SparseApplyRMSPropD)
    .INPUT(var, TensorType::NumberType())
    .INPUT(ms, TensorType::NumberType())
    .INPUT(mom, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .INPUT(indices, TensorType::IndexNumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(ms, TensorType::NumberType())
    .OUTPUT(mom, TensorType::NumberType())
    .REQUIRED_ATTR(rho, Float)
    .REQUIRED_ATTR(momentum, Float)
    .REQUIRED_ATTR(epsilon, Float)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(SparseApplyRMSPropD)

/**
* @brief Updates "var" in specified index according to the Adadelta algorithm.
*    accum <- rho * accum + (1 - rho) * grad.square()
*    update <- (accum_update + epsilon).sqrt() * (accum + epsilon()).rsqrt() * grad
*    var <- var - update * lr
*    accum_update <- rho() * accum_update + (1 - rho()) * update.square()
*
* @par Inputs:
* Eight inputs, including:
* @li var: A mutable tensor. Must be one of the data types defined in
* TensorType::NumberType(). Should be from a Variable().
* @li accum: A mutable tensor. Must have the same type as "var". Should be from a
* Variable().
* @li accum_update: A mutable tensor. Must have the same type as "var". Should be from a
* Variable().
* @li lr: A scalar. Must have the same type as "var".
* @li rho: A scalar. Must have the same type as "var".
* @li epsilon: A scalar. Must have the same type as "var".
* @li grad: A tensor, specifying the gradient.
* @li indices: A vector of indices into the first dimension of "var", "accum" and "accum_update".
*
* @par Attributes:
* use_locking: An optional "bool". Defaults to "False". If "True", updating of
* the "var", "accum", and "accum_update" tensors will be protected by a lock; otherwise the
* behavior is undefined, but may exhibit less contention.
*
* @par Outputs:
* @li var: A mutable tensor. Has the same type as input "var".
* @li accum:  A mutable tensor. Must have the same type as input "accum".
* @li accum_update: A mutable tensor. Must have the same type as input "accum_update".
*
* @attention Constraints:
* @li Note that in this sparse implementation, "accum" and "accum_update" will not update
* in iterations during which "grad" is 0.
* @li The input tensors "var", "accum", and "accum_update" must have the same shape.
*
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseApplyAdadelta.
*/
REG_OP(SparseApplyAdadelta)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(accum_update, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(rho, TensorType::NumberType())
    .INPUT(epsilon, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .INPUT(indices, TensorType::IndexNumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(accum, TensorType::NumberType())
    .OUTPUT(accum_update, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(SparseApplyAdadelta)

/**
* @brief Updates "var" in specified index according to the Adadelta algorithm.
* a const input will be considered as an attribute.
*    accum <- rho * accum + (1 - rho) * grad.square()
*    update <- (accum_update + epsilon).sqrt() * (accum + epsilon()).rsqrt() * grad
*    var <- var - update * lr
*    accum_update <- rho() * accum_update + (1 - rho()) * update.square()
*
* @par Inputs:
* Seven inputs, including:
* @li var: A mutable tensor. Must be one of the data types defined in
* TensorType::NumberType(). Should be from a Variable().
* @li accum: A mutable tensor. Must have the same type as "var". Should be from a
* Variable().
* @li accum_update: A mutable tensor. Must have the same type as "var". Should be from a
* Variable().
* @li lr: A scalar. Must have the same type as "var".
* @li rho: A scalar. Must have the same type as "var".
* @li grad: A tensor, specifying the gradient.
* @li indices: A vector of indices into the first dimension of "var", "accum" and "accum_update".
*
* @par Attributes:
* @li use_locking: An optional "bool". Defaults to "False". If "True",
* updating of the "var", "accum", and "accum_update" tensors will be protected by a lock;
* otherwise the behavior is undefined, but may exhibit less contention.
* @li epsilon: A required scalar. Must have the same type as "var".
*
* @par Outputs:
* @li var: A mutable tensor. Must have the same type as input "var".
* @li accum:  A mutable tensor. Must have the same type as input "accum".
* @li accum_update: A mutable tensor. Must have the same type as input "accum_update".
*
* @attention Constraints:
* @li Note that in this sparse implementation, "accum" and "accum_update" will not update
* in iterations during which "grad" is 0.
* @li The input tensors "var", "accum" and "accum_update" must have the same shape.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use SparseApplyAdadelta instead.
*/
REG_OP(SparseApplyAdadeltaD)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(accum_update, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(rho, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .INPUT(indices, TensorType::IndexNumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(accum, TensorType::NumberType())
    .OUTPUT(accum_update, TensorType::NumberType())
    .REQUIRED_ATTR(epsilon, Float)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(SparseApplyAdadeltaD)


/**
*@brief Clean memory of workspace list . \n

*@par Attributes:
* @li automic_add_mem_size: sizes of workspaces . \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(AtomicAddrClean)
    .ATTR(automic_add_mem_size, ListInt, {})
    .OP_END_FACTORY_REG(AtomicAddrClean)

/**
*@brief Clean memory of workspace list . \n

*@par Attributes:
* @li workspace_size: sizes of workspaces . \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(DynamicAtomicAddrClean)
    .ATTR(automic_add_mem_size, ListInt, {})
    .OP_END_FACTORY_REG(DynamicAtomicAddrClean)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_TRAINING_OPS_H_
