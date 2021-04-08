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

/*!
 * \file nn_norm_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_OPS_H_

#include "graph/operator_reg.h"
namespace ge {

/**
*@brief Computes the gradient for log softmax activations . \n

*@par Inputs:
*@li grad: A Tensor. Must be one of the following types: float16, float32.
*@li x: A Tensor. Must be one of the following types: float16, float32 . \n

*@par Attributes:
* axis: An optional list of ints. Defaults to "{-1}" . \n

*@par Outputs:
* y: A Tensor. Has the same type as "grad" . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator LogSoftmaxGrad.
*/

REG_OP(LogSoftmaxGrad)
    .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(axis, ListInt, {-1})
    .OP_END_FACTORY_REG(LogSoftmaxGrad)

/**
*@brief Computes sparse softmax cross entropy cost and gradients to backpropagate . \n

*@par Inputs:
*Two inputs, including:
* @li features: A Tensor. Must be one of the following types: half, float32, double.
*    A "batch_size * num_classes" matrix.
* @li labels: A Tensor of the same type as "features". batch_size vector with values in [0, num_classes).


*@par Outputs:
*loss: A Tensor for per example loss (a "batch_size" vector). Has the same type as "features".
*backprop: A Tensor for the backpropagated gradients (a batch_size * num_classes matrix). Has the same type as "features" . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator SparseSoftmaxCrossEntropyWithLogits.
*/
REG_OP(SparseSoftmaxCrossEntropyWithLogits)
    .INPUT(features, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(labels, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(loss, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(backprop, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(SparseSoftmaxCrossEntropyWithLogits)

/**
*@brief Computes softmax cross entropy cost and gradients to backpropagate . \n

*@par Inputs:
*Two inputs, including:
* @li features: A Tensor. Must be one of the following types: half, float32, double.
*    A "batch_size * num_classes" matrix.
* @li labels: A Tensor of the same type as "features". A "batch_size * num_classes" matrix . \n

*@par Outputs:
*loss: A Tensor for per example loss (a "batch_size" vector). Has the same type as "features".
*backprop: A Tensor for the backpropagated gradients (a batch_size * num_classes matrix). Has the same type as "features" . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator SoftmaxCrossEntropyWithLogits.
*/
REG_OP(SoftmaxCrossEntropyWithLogits)
    .INPUT(features, TensorType({DT_DOUBLE,DT_FLOAT16,DT_FLOAT}))
    .INPUT(labels, TensorType({DT_DOUBLE,DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(loss, TensorType({DT_DOUBLE,DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(backprop, TensorType({DT_DOUBLE,DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(SoftmaxCrossEntropyWithLogits)

/**
*@brief Computes gradients for a softmax operation . \n

*@par Inputs:
* Two inputs, including:
* @li softmax: Output of the softmax operator. Must be one of the following
* types: float16, float31, int32, int8, uint8. The format is NC1HWC0 or DN.
* @li grad_softmax: A Tensor. Has the same shape and type as "softmax".
* The format is NC1HWC0 or DN . \n

*@par Outputs:
*grad_x: A Tensor. Has the same shape and type as "softmax" . \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator SoftmaxGrad.
*/
REG_OP(SoftmaxGrad)
    .INPUT(softmax, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(grad_softmax, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(grad_x, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OP_END_FACTORY_REG(SoftmaxGrad)

/**
*@brief Computes the sigmoid cross entropy loss of "predict" and "target" . \n

*@par Inputs:
* Two inputs, including:
*@li predict: A multi-dimensional Tensor of type float16 or float32, specifying the predictive value.
*@li target: A multi-dimensional Tensor of type float16 or float32, specifying the target value . \n

*@par Outputs:
*loss: Sigmoid cross entropy between the predictive value and target value. Has the same dimensions as "predict" . \n

*@par Third-party framework compatibility
* Compatible with the scenario where "reduction" is set to "none"of PyTorch operator SigmoidCrossEntropyWithLogitsGrad.
*/
REG_OP(SigmoidCrossEntropyWithLogitsGrad)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(dout, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(gradient, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(SigmoidCrossEntropyWithLogitsGrad)

/**
*@brief Performs the backpropagation of SigmoidCrossEntropyWithLogits for training scenarios . \n

*@par Inputs:
* Three inputs, including:
*@li predict: A multi-dimensional Tensor of type float16 or float32, specifying the predictive value.
*@li target: A multi-dimensional Tensor of type float16 or float32, specifying the target value.
*@li dout: A multi-dimensional Tensor of float16 or float32, specifying the gradient transferred from the upper layer . \n

*@par Outputs:
*gradient: Return gradient. Has the same dimensions and type as "predict" . \n

*@par Third-party framework compatibility
* Compatible with the scenario where "reduction" is set to "none"of PyTorch operator SigmoidCrossEntropyWithLogits.
*/
REG_OP(SigmoidCrossEntropyWithLogits)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(loss, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(SigmoidCrossEntropyWithLogits)

/**
*@brief Computes the sigmoid cross entropy loss of "predict" and "target" . \n

*@par Inputs:
* four inputs, including:
*@li predict: A multi-dimensional Tensor of type float16 or float32, specifying the predictive value.
*@li target: A multi-dimensional Tensor of type float16 or float32, specifying the target value . \n
*@li weight: An multi-dimensional Tensor, specifying the weight value. \n
*@li pos_weight: An multi-dimensional Tensor, specifying the pos weight value. \n

*@par Attributes:
*reduction: A character string from "none", "mean", and "sum", specifying the reduction type to be applied to the output. Defaults to "mean" . \n

*@par Outputs:
*loss: Sigmoid cross entropy between the predictive value and target value. Has the same dimensions as "predict" . \n

*@par Third-party framework compatibility
* Compatible with PyTorch operator BCEWithLogitsLoss.
*/
REG_OP(SigmoidCrossEntropyWithLogitsV2)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(pos_weight, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(loss, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(SigmoidCrossEntropyWithLogitsV2)

/**
*@brief Computes the regression box of the RPN. It is a FasterRCNN operator . \n

*@par Inputs:
* Two inputs, including:
*@li predict: A multi-dimensional Tensor of type float16 or float32, specifying the predictive value.
*@li label: A multi-dimensional Tensor of type float16 or float32, specifying the target value . \n

*@par Attributes:
* sigma: Must be a floating point number. Defaults to "1.0" . \n

*@par Outputs:
*loss: Indicates the loss between the predictive value and target value. Has the same dimensions as "predict" . \n

*@attention Constraints:
* This operator does not perform the "reduce" operation on the loss value. Call other reduce operators to perform "reduce" operation on the loss if required . \n

*@par Third-party framework compatibility
* Compatible with the scenario where "reduction" is set to "none"of PyTorch operator SmoothL1Loss.
*/
REG_OP(SmoothL1Loss)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(label, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(loss, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(sigma, Float, 1.0)
    .OP_END_FACTORY_REG(SmoothL1Loss)

/**
*@brief Performs the backpropagation of SmoothL1Loss for training scenarios . \n

*@par Inputs:
* Three inputs, including:
*@li predict: A multi-dimensional Tensor of type float16 or float32, specifying the predictive value.
*@li label: A multi-dimensional Tensor of float16 or float32, specifying the target value.
*@li dout: A multi-dimensional Tensor of float16 or float32, specifying the gradient transferred from the upper layer . \n

*@par Attributes:
* sigma: Must be a floating point number. Defaults to "1.0" . \n

*@par Outputs:
*gradient: Return gradient. Has the same dimensions and type as "predict" . \n

*@par Third-party framework compatibility
* Compatible with the scenario where "reduction" is set to "none"of PyTorch operator SmoothL1LossGrad.
*/
REG_OP(SmoothL1LossGrad)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(label, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(dout, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(gradient, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(sigma, Float, 1.0)
    .OP_END_FACTORY_REG(SmoothL1LossGrad)

/**
*@brief Creates a criterion that measures the Binary Cross Entropy between the target and the output . \n

*@par Inputs:
* Three inputs, including:
*@li x: A 1D or 2D Tensor of type float16 or float32, specifying a predictive value.
*@li y: A 1D or 2D Tensor of type float16 or float32, indicating a tag.
*@li weight: An optional 1D or 2D Tensor, specifying the weight . \n

*@par Attributes:
*reduction: A character string from "none", "mean", and "sum", specifying the reduction type to be applied to the output. Defaults to "mean" . \n

*@par Outputs:
*output: Output loss. Has the same dimension with the inputs. When "reduction" is set to "none", a Tensor with the same size as "x" is output. Otherwise, a Scalar is output . \n

*@attention Constraints:
*@li The value of "x" must range from 0 to 1.
*@li The value of "y" must be "0" or "1" . \n

*@par Third-party framework compatibility
* Compatible with PyTorch operator BCELoss.
*/
REG_OP(BinaryCrossEntropy)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(output, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(BinaryCrossEntropy)

/**
*@brief Performs the backpropagation of BinaryCrossEntropy for training scenarios . \n

*@par Inputs:
* Four inputs, including:
*@li x: A 1D or 2D Tensor of type float16 or float32, specifying a predictive value.
*@li y: A 1D or 2D Tensor of type float16 or float32, indicating a tag.
*@li grad_output: A 1D or 2D Tensor of type float16 or float32, specifying the backpropagation gradient.
*@li weight: An optional 1D or 2D Tensor, specifying the weight . \n

*@par Attributes:
*reduction: A character string from "none", "mean", and "sum", specifying the gradient output mode. Defaults to "mean" . \n

*@par Outputs:
*output: A 1D or 2D Tensor. When "reduction" is set to "none", a Tensor with the same size as "x" is output. Otherwise, a Scalar is output . \n

*@attention Constraints:
*@li The value of "x" must range from 0 to 1.
*@li The value of "y" must be "0" or "1" . \n

*@par Third-party framework compatibility
* Compatible with PyTorch operator BCELossGrad.
*/
REG_OP(BinaryCrossEntropyGrad)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(grad_output, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(output, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(BinaryCrossEntropyGrad)

/**
*@brief Applies the Softmax function to an n-dimensional input Tensor
* rescaling them. so that the elements of the n-dimensional output Tensor lie
* in the range [0,1] and sum to 1 . \n

*@par Inputs:
*One input:
*x: A mutable Tensor. Must be one of the following types: float16, float32,
* double. Should be a Variable Tensor . \n

*@par Attributes:
*axes: A list of int. The dimension softmax would be performed on. Defaults
* to "[-1]" . \n

*@par Outputs:
*y: A Tensor. Has the same dimensionality and shape as the "x" with values in
* the range [0, 1]. Must be one of the following types: float16, float32,
* double . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Softmax.
*/
REG_OP(SoftmaxV2)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT}))
    .ATTR(axes, ListInt, {-1})
    .OP_END_FACTORY_REG(SoftmaxV2)

/**
*@brief Computes log softmax activations . \n

*@par Inputs:
*One input:
* logits: A Tensor. Must be one of the following types: double, float16, float32 . \n

*@par Attributes:
* axes: An optional list of ints. Defaults to "{-1}" . \n

*@par Outputs:
* logsoftmax: A Tensor. Has the same type as "logits" . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator LogSoftmax.
*/
REG_OP(LogSoftmaxV2)
    .INPUT(logits, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(logsoftmax, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT}))
    .ATTR(axes, ListInt, {-1})
    .OP_END_FACTORY_REG(LogSoftmaxV2)

/**
*@brief Confuse mul, sum and sub . \n

*@par Inputs:
*Two inputs, including:
* @li grad: A Tensor. Must be one of the following types: float16, float32.
* @li x: A Tensor. Must be one of the following types: float16, float32 . \n

*@par Outputs:
* y: A Tensor of the same type as "grad" . \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(ConfusionSoftmaxGrad)
  .INPUT(grad, TensorType({DT_FLOAT16,DT_FLOAT}))
  .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
  .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
  .OP_END_FACTORY_REG(ConfusionSoftmaxGrad)

/**
*@brief Function softmax gradients ext . \n

*@par Inputs:
* @li grad: A Tensor dtype of float16, float32.
* @li x1: A Tensor dtype of float16, float32.
* @li x2: A Tensor dtype of float16, float32 . \n

*@par Attributes:
*@li axis: A int Scalar. The axis for reduce.
*@li keepdims: A bool Scalar. If true, retains reduced dimensions with length 1 . \n

*@par Outputs:
*y: A Tensor dtype of float16, float32.
*/
REG_OP(SoftmaxGradExt)
  .INPUT(grad, TensorType({DT_FLOAT16,DT_FLOAT}))
  .INPUT(x1, TensorType({DT_FLOAT16,DT_FLOAT}))
  .INPUT(x2, TensorType({DT_FLOAT16,DT_FLOAT}))
  .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
  .ATTR(axes, Int, 1)
  .ATTR(keep_dims, Bool, false)
  .OP_END_FACTORY_REG(SoftmaxGradExt)

/**
*@brief Normalizes the input . \n

*@par Inputs:
* One input:
*x: An NCHW tensor of type float16 or float32 . \n

*@par Attributes:
*@li normalize_variance: An optional bool specifying whether to normalize the variance, either "true" (default) or "false"
* the value "false" indicates only to subtract the mean.
*@li across_channels: An optional bool specifying whether to perform across-channel MVN, either "true" or "false" (default)
* The value "true" indicates "CHW" is treated as a vector.
*@li eps: An optional float32 epsilon for not dividing by zero. Defaults to "1e-9" . \n

*@par Outputs:
*y: An NCHW tensor of type float16 or float32 . \n

*@attention Constraints:
* The input tensor must have the NCHW format, whose shape length must be 4.
*@par Third-party framework compatibility
* Compatible with the Caffe operator MVN.
*/

REG_OP(MVN)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16})) /* "First operand." */
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))  /* "Result, has same element type as inputs" */
    .ATTR(normalize_variance, Bool, true)
    .ATTR(across_channels, Bool, false)
    .ATTR(eps, Float, 1e-9)
    .OP_END_FACTORY_REG(MVN)

/**
*@brief Normalizes the input "x1" . \n

*@par Inputs:
* Two inputs, including:
*@li x1: A required NCHW or NHWC tensor of type float32, float16, or int8.
*@li x2: A required ND tensor of type float32, float16, or int8, specifying
* the scaling factor. If "channel_shared" is "true", "x2" is a [1]-dimensional
* vector. If "channel_shared" is "false", "x2" is a [C]-dimensional vector . \n

*@par Attributes:
*@li across_spatial: An optional bool, specifying the dimension of input "x1"
* to be summed. The value "true" (default) indicates dimensions C, H, W, and
* the value "false" indicates dimension C.
*@li channel_shared: An optional bool, specifying the dimension count of input
* "x2". The value "true" (default) indicates 1, and the value "false" indicates
* dimension C of "x1".
*@li eps: An optional float32, specifying the bias when "across_spatial" is
* "true". Defaults to "1e-10" . \n

*@par Outputs:
*y: A Tensor. Has the same type and format as "x1" . \n

*@par Third-party framework compatibility
* Compatible with the Caffe operator Normalize.
*/
REG_OP(Normalize)
     .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8}))
     .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8}))
     .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8}))
     .ATTR(across_spatial, Bool, true)
     .ATTR(channel_shared, Bool, true)
     .ATTR(eps, Float, 1e-10)
     .OP_END_FACTORY_REG(Normalize);

/**
*@brief Layernorm operator interface implementation
*  calculating: x, gamma, beta
*  mean  = np.mean(x, reduce_axis, keepdims=True)
*  variance = np.mean(np.power((x - mean),2), reduce_axis, keepdims=True)
*  y = gamma*((x - mean) / np.sqrt(variance + 0.001)) + beta

*@par Inputs:
*Three inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32.
* @li gamma: A Tensor. Must be one of the following types: float16, float32.
* @li beta: A Tensor. Must be one of the following types: float16, float32 . \n

*@par Attributes:
* @li begin_norm_axis: A optional attribute, the type is int32. Defaults to 0.
* @li begin_params_axis: A optional attribute, the type is int32. Defaults to 0.
* @li epsilon: A optional attribute, the type is float32. Defaults to 1e-7 . \n

*@par Outputs:
*Three outputs, including:
* @li y: A Tensor. Must be one of the following types: float16, float32.
* @li mean: A Tensor. Must be one of the following types: float16, float32.
* @li variance: A Tensor. Must be one of the following types: float16, float32.
*/
REG_OP(LayerNorm)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(begin_norm_axis, Int, 0)
    .ATTR(begin_params_axis, Int, 0)
    .ATTR(epsilon, Float, 0.0000001)
    .OP_END_FACTORY_REG(LayerNorm)

/**
*@brief LayerNormGrad operator interface implementation
*  calculating: dy, x, variance, mean, gamma
*  pd_xl = data_dy*data_gamma
*  pd_var = np.sum(((-0.5)*pd_xl*(data_x - data_mean)
*           np.power((data_variance + EPSLON), (-1.5))),
*           reduce_axis, keepdims=True)
*  pd_mean = np.sum(((-1.0)*pd_xl
*            np.power((data_variance + EPSLON), (-0.5))),
*            reduce_axis, keepdims=True)
*            + pd_var*(1.0/m)
*            np.sum(((-2.0)*(data_x - data_mean)), reduce_axis, keepdims=True)
*  pd_x = pd_xl*np.power((data_variance + EPSLON), (-0.5)) +
*         pd_var*(2.0/m)*(data_x - data_mean) + pd_mean*(1.0/m)
*  pd_gamma = np.sum((data_dy*(data_x - data_mean)
*             np.power((data_variance + EPSLON), (-0.5))), param_axis, keepdims=True)
*  pd_beta = np.sum(data_dy, param_axis, keepdims=True)

*@par Inputs:
*Five inputs, including:
* @li dy: A Tensor. Must be one of the following types: float16, float32.
* @li x: A Tensor. Must be one of the following types: float16, float32.
* @li variance: A Tensor. Must be one of the following types: float16, float32.
* @li mean: A Tensor. Must be one of the following types: float16, float32.
* @li gamma: A Tensor. Must be one of the following types: float16, float32 . \n

*@par Outputs:
*Three outputs, including:
* @li pd_x: A Tensor. Must be one of the following types: float16, float32.
* @li pd_gamma: A Tensor. Must be one of the following types: float16, float32.
* @li pd_beta: A Tensor. Must be one of the following types: float16, float32.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(LayerNormGrad)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_beta, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(LayerNormGrad)

/**
*@brief LayerNormXBackprop operator interface implementation
*  calculating: dy, x, variance, mean, gamma
*  pd_xl = data_dy*data_gamma
*  pd_var = np.sum(((-0.5)*pd_xl*(data_x - data_mean)
*           np.power((data_variance + EPSLON), (-1.5))),
*           reduce_axis, keepdims=True)
*  pd_mean = np.sum(((-1.0)*pd_xl
*            np.power((data_variance + EPSLON), (-0.5))),
*            reduce_axis, keepdims=True)
*            + pd_var*(1.0/m)
*            np.sum(((-2.0)*(data_x - data_mean)), reduce_axis, keepdims=True)
*  pd_x = pd_xl*np.power((data_variance + EPSLON), (-0.5)) +
*         pd_var*(2.0/m)*(data_x - data_mean) + pd_mean*(1.0/m)
*  pd_gamma = np.sum((data_dy*(data_x - data_mean)
*             np.power((data_variance + EPSLON), (-0.5))), param_axis, keepdims=True)
*  pd_beta = np.sum(data_dy, param_axis, keepdims=True)

*@par Inputs:
*Five inputs, including:
* @li dy: A Tensor. Must be one of the following types: float16, float32.
* @li x: A Tensor. Must be one of the following types: float16, float32.
* @li variance: A Tensor. Must be one of the following types: float16, float32.
* @li mean: A Tensor. Must be one of the following types: float16, float32.
* @li gamma: A Tensor. Must be one of the following types: float16, float32 . \n

*@par Outputs:
*Three outputs, including:
* @li pd_x: A Tensor. Must be one of the following types: float16, float32.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(LayerNormXBackprop)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(LayerNormXBackprop)

/**
*@brief LayerNormBetaGammaBackprop operator interface implementation
*  calculating: dy, x, variance, mean
*  pd_xl = data_dy*data_gamma
*  pd_var = np.sum(((-0.5)*pd_xl*(data_x - data_mean)
*           np.power((data_variance + EPSLON), (-1.5))),
*           reduce_axis, keepdims=True)
*  pd_mean = np.sum(((-1.0)*pd_xl
*            np.power((data_variance + EPSLON), (-0.5))),
*            reduce_axis, keepdims=True)
*            + pd_var*(1.0/m)
*            np.sum(((-2.0)*(data_x - data_mean)), reduce_axis, keepdims=True)
*  pd_x = pd_xl*np.power((data_variance + EPSLON), (-0.5)) +
*         pd_var*(2.0/m)*(data_x - data_mean) + pd_mean*(1.0/m)
*  pd_gamma = np.sum((data_dy*(data_x - data_mean)
*             np.power((data_variance + EPSLON), (-0.5))), param_axis, keepdims=True)
*  pd_beta = np.sum(data_dy, param_axis, keepdims=True)

*@par Inputs:
*Three inputs, including:
* @li dy: A Tensor. Must be one of the following types: float16, float32.
* @li x: A Tensor. Must be one of the following types: float16, float32.
* @li variance: A Tensor. Must be one of the following types: float16, float32.
* @li mean: A Tensor. Must be one of the following types: float16, float32 . \n

*@par Outputs:
*Three outputs, including:
* @li pd_gamma: A Tensor. Must be one of the following types: float16, float32.
* @li pd_beta: A Tensor. Must be one of the following types: float16, float32.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(LayerNormBetaGammaBackprop)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_beta, TensorType({DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(shape_gamma, ListInt)
    .OP_END_FACTORY_REG(LayerNormBetaGammaBackprop)

/**
*@brief Return "output" according to the algorithm of dropout_do_mask:
*  scale_x = x *(1 / keep_prob)
*  output = select(mask == 1, scale_x, 0)

*@par Inputs:
*Three inputs, including:
* @li x: A mutable Tensor. Must be one of the following types:
*     float16, float32
* @li mask: A mutable Tensor. Must met all of the following rules:
*     shape of mask should be 1D.
*     dtype of mask should be uint8.
*     value of shape should met the following algorithm:
*     value = (size(x) + 128 - 1) // 128 * 128 //8
* @li keep_prob: A mutable Tensor. Must met all of the following rules:
*     shape of "keep_prob" should be (1,) or [1,].
*     Has the same type as "x" . \n

*@par Output:
*y: A mutable Tensor. Has the same type as "x".
*/
REG_OP(DropOutDoMask)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(mask, TensorType({DT_UINT8}))
    .INPUT(keep_prob, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(DropOutDoMask)
	
/**
*@brief Scales the input . \n

*@par Inputs:
* Three inputs, including:
*@li x: An ND tensor of type float16 or float32.
*@li scale: An ND tensor of type float16 or float32.
*@li bias: An optional ND tensor of type float16 or float32 . \n

*@par Attributes:
*@li axis: An optional int32 used to compute the shape of scale and bias input from the online bottoms. Defaults to "1".
*@li num_axes: An optional int32 used to compute the shape of scale and bias input from a Caffe model trained offline. Defaults to "1".
*@li scale_from_blob: An optional bool. If "true", scale and bias are input from a Caffe model trained offline. If "false", scale and bias are input from online bottoms. Defaults to "true" . \n

*@par Outputs:
*y: An ND tensor of type float16 or float32 . \n

*@attention Constraints:
* Assume that the shape length of "x" is "n" and that of "scale" is "m".
*@li "axis" is within the range [-n, n-1]. num_axes >= -1.
*@li If "scale_from_blob = true", "num_axes = -1", and "axis >= 0", the ith axis of "scale" and the (i+"axis")th axis of "x" must have the same size (0 <= i < n-axis).
* If "axis < 0", the ith axis of "scale" and the (i+n+"axis")th axis of "x" must have the same size (0 <= i < -axis).
*@li If "scale_from_blob = true" and "num_axes = 0", "scale" is a scalar with shape length 1 and dimension size 1.
*@li If "scale_from_blob = true", "num_axes > 0, and "axis >= 0", "axis + num_axes" must be less than or equal to "n" and the ith axis of "scale" and the (i+"axis")th axis of "x" must have the same size (0 <= i < num_axes).
* If "axis < 0", "n + axis + num_axes" must be less than or equal to "n" and the ith axis of "scale" and the (i+n+"axis")th axis of "x" must have the same size (0 <= i < num_axes).
*@li If "scale_from_blob = false", "scale" is not a scalar, and "axis >= 0","axis + m" must be less than or equal to "n" and the ith axis of "scale" and the (i+"axis")th axis of "x" must have the same size (0 <= i < m).
* If "axis < 0", "n + axis + m" must be less than or equal to "n" and the ith axis of "scale" and the (i+n+"axis")th axis of "x" must have the same size (0 <= i < m).
*@li If "bias" is not None, the constraints for "bias" is the same as that for "scale".
*@par Third-party framework compatibility
* Compatible with the Caffe operator Scale.
*/
REG_OP(Scale)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16})) /* "First operand." */
    .INPUT(scale, TensorType({DT_FLOAT, DT_FLOAT16})) /* "Second operand." */
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16})) /* "Third operand." */
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))  /* "Result, has same element type as x" */
    .ATTR(axis, Int, 1)
    .ATTR(num_axes, Int, 1)
    .ATTR(scale_from_blob, Bool, true)
    .OP_END_FACTORY_REG(Scale)

/**
*@brief Local Response Normalization . \n

*@par Inputs:
*One input, including:
*@li x: A Tensor. Must be 4-D shape, and only support the following types: float16, float32 . \n

*@par Attributes:
*@li depth_radius: An optional int32, specifying the half-width of the normalization window. Defaults to "5".
* under the caffe framework, if local_size is provided and is an odd number,
* depth_radius = (local_size - 1) / 2. local_size is the number of channels to sum over (for ACROSS_CHANNELS)
* or the side length of the square region to sum over (for WITHIN_CHANNEL).
*@li bias: An optional float32. An offset, usually > 0 to avoid dividing by 0.
* Defaults to "1.0".
*@li alpha: An optional float32. A scaling factor, usually positive.
* Defaults to "1.0".
*@li beta: An optional float32. An exponent. Defaults to "0.75" for the caffe framework, Defaults to "0.5" for others.
*@li norm_region: An optional string. A mode option. "ACROSS_CHANNELS":0. Defaults to "ACROSS_CHANNELS" . \n

*@par Outputs:
*y: A Tensor. Has the same data type and shape as "x" . \n

*@par Third-party framework compatibility:
* Compatible with the TensorFlow operator LRN.
*/
REG_OP(LRN)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .ATTR(depth_radius, Int, 5)
    .ATTR(bias, Float, 1.0)
    .ATTR(alpha, Float, 1.0)
    .ATTR(beta, Float, 0.5)
    .ATTR(norm_region, String, "ACROSS_CHANNELS")
    .OP_END_FACTORY_REG(LRN)

/**
* @brief Computes the gradient for Local Response Normalization . \n

* @par Inputs:
* @li grads: A 4D Tensor of type float16 or float32.
* @li x: A 4D Tensor of type float16 or float32.
* @li y: A 4D Tensor of type float16 or float32 . \n

* @par Attributes:
* @li depth_radius: An optional int, specifying the half-width of the
* normalization window. Defaults to "5".
* @li bias: An optional float32. An offset, usually > 0 to avoid dividing by 0.
* Defaults to "1".
* @li alpha: An optional float32. A scaling factor, usually positive.
* Defaults to "1".
* @li beta: An optional float32. An exponent. Defaults to "0.5" . \n

* @par Outputs:
* z: A Tensor. Has the same type and shape as "grads" . \n

* @attention Constraints:
* "x" and "y" must have the same shape and type as "grads" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator LRNGrad.
*/
REG_OP(LRNGrad)
    .INPUT(grads, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(z, TensorType({DT_FLOAT16,DT_FLOAT}))
    .ATTR(depth_radius, Int, 5)
    .ATTR(bias, Float, 1.0)
    .ATTR(alpha, Float, 1.0)
    .ATTR(beta, Float, 0.5)
    .OP_END_FACTORY_REG(LRNGrad)

 /**
 *@brief Calculates the RNNT Loss (log probability) for each batch entry.
 Also calculates the gradient.

 *@par Inputs:
 *@li acts: 4-D, shape: `(batch x seqLength x labelLength x outputDim)`, the logits.
 *@li labels: 2-D Tensor containing all the targets of the batch with zero padded.
 *@li input_lengths: Tensor of size (batch) containing size of each output sequence.
 *@li label_lengths: Tensor of (batch) containing label length of each example.

 *@par Outputs:
 *@li costs: 1-D Tensor, the cost of each example in the batch.
 *@li grads: A Tensor. Has the same type as acts.

 *@par Attributes:
 *@li blank_label: An optional attribute. Defaults to 0.

 *@par Third-party framework compatibility
 * Compatible with TensorFlow RNNTLoss operator.
 */
REG_OP(RNNTLoss)
    .INPUT(acts, TensorType({DT_FLOAT}))
    .INPUT(labels, TensorType({DT_INT32}))
    .INPUT(input_lengths, TensorType({DT_INT32}))
    .INPUT(label_lengths, TensorType({DT_INT32}))
    .ATTR(blank_label, Int, 0)
    .OUTPUT(costs, TensorType({DT_FLOAT}))
    .OUTPUT(grads, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(RNNTLoss)

/**
*@brief Performs group normalization . \n

*@par Inputs:
* Five inputs, including: (NHWC, NCHW supported)
*@li x: A 4D Tensor of type float16 or float32, with format NHWC or
NCHW for 4D.
*@li scale: A Tensor of type float32. Must be 1D if input "x" is with format
NHWC or NCHW. Specifies the scaling factor.
*@li offset: A Tensor of type float32. Must be 1D if input "x" is with
format NHWC or NCHW. Specifies the offset.
*@li mean: A Tensor of type float32. Must be 1D if input "x" is with format
NHWC or NCHW. Reserved. Mu
st be "None" if the operation is used for training.
*@li variance: A Tensor of type float32. Must be 1D if input "x" is with
format NHWC or NCHW. Specifies the variance used for inference. Reserved . \n

*@par Attributes:
*@li epsilon: An optional float32, specifying the small value added to
variance to avoid dividing by zero. Defaults to "0.0001".
*@li data_format: An optional string, specifying the format of "x".
Defaults to "NHWC".
*@li is_training: An optional bool, specifying if the operation is used for
training or inference. Defaults to "True" . \n

*@par Outputs:
* Five outputs, including: (NHWC, NCHW supported)
*@li y: A 4D Tensor of type float16 or float32 for the normalized "x",
with format NHWC or NCHW for 4D.
*@li batch_mean: A Tensor of type float32. Must be 1D if input "x" is with
format NHWC or NCHW. Specifies the mean of "x".
*@li batch_variance: A Tensor of type float32. Must be 1D if input "x" is
with format NHWC or NCHW. Specifies the variance of "x".
*@li reserve_space_1: An optional Tensor of type float32. Must be 1D if
input "x" is with format NHWC or NCHW. Specifies the mean o
f "x" for gradient computation. Pass "None" to skip this output.
*@li reserve_space_2: An optional Tensor of type float32. Must be 1D if
input "x" is with format NHWC or NCHW. Specifies the varian
ce of "x" for gradient computation. Pass "None" to skip this output . \n

*@attention Constraints:
*@li If the operation is used for inference and outputs "reserve_space_1"
and "reserve_space_2" are available, then "reserve_space_1" has the same
value as "mean" and "reserve_spa
ce_2" has the same value as "variance".
*@li For Ascend 310, the result accuracy fails  due to the square root
instruction . \n

*@par Third-party framework compatibility
*@li Compatible with the PyTorch operator GroupNorm.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(GroupNorm)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT,}))
    .INPUT(offset, TensorType({DT_FLOAT,}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_1, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_2, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001)
    .ATTR(data_format, String, "NHWC")
    .ATTR(is_training, Bool, true)
    .ATTR(num_groups, Int, 2)
    .OP_END_FACTORY_REG(GroupNorm)

/**
*@brief Performs instance normalization . \n

*@par Inputs:
* Five inputs, including: (NC1HWC0, supported)
*@li x: A 5D Tensor of type float16 or float32, NC1HWC0.
*@li gamma: A Tensor of type float32.
A 5D Tensor for scaling factor, to scale the normalized x.
*@li beta: A Tensor of type float32.
A 5D Tensor for offset, to shift to the normalized x.
*@li mean: A Tensor of type float32.
A 5D Tensor Specifies the mean used for inference. Reserved.
*@li variance: A Tensor of type float32.
A 5D Tensor Specifies the variance used for inference. Reserved . \n

*@par Attributes:
*@li is_training: An optional bool, specifying if the operation is used for
training or inference. Defaults to "True".
*@li momentum: An optional float32,
the value used for the running_mean and running_var computation. Default: "0.1".
*@li epsilon: An optional float32, specifying the small value added to
variance to avoid dividing by zero. Defaults to "0.00001" . \n

*@par Outputs:
* Three outputs, including: (NHWC, NCHW NC1HWC0 supported)
*@li y: A 5D tensor of type float16 or float32 for the normalized "x",
*@li batch_mean: A Tensor of type float32.
Specifies the mean of "x".
*@li batch_variance: A Tensor of type float32.
Specifies the variance of "x" . \n

*@par Third-party framework compatibility
*@li Compatible with the PyTorch operator InstanceNorm.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(InstanceNormV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(gamma, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(beta, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))

    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))

    .ATTR(is_training, Bool, true)
    .ATTR(momentum, Float, 0.1)
    .ATTR(epsilon, Float, 0.00001)
    .OP_END_FACTORY_REG(InstanceNormV2)

/**
*@brief Performs instance normalization for inference.

*@par Inputs:\n
* Five inputs, including: (NC1HWC0 supported)
*@li x: A Tensor of type float16 or float32.
*@li gamma: A [N, C1, 1, 1, C0] Tensor of type float32, for the scaling gamma.
*@li beta: A [N, C1, 1, 1, C0] Tensor of type float32, for the scaling beta.
*@li mean: A [N, C1, 1, 1, C0] ensor of type float32, for the mean.
*@li variance: A [N, C1, 1, 1, C0] Tensor of type float32, for the variance.
*@li variance_sqrt: A [N, C1, 1, 1, C0] Tensor of type float32, for the variance_sqrt.

*@par Outputs:\n
*y: A Tensor of type float16 or float32 for the normalized "x".
*batch_mean: A Tensor of type float32 for the result mean.
*batch_ variance: A Tensor of type float32 for the result variance.

*@attention Constraints:
*For Ascend 310, the result accuracy fails to reach 1<89> due to the square root instruction.

* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use INInferV2 instead.
*/
REG_OP(INInferV2D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(gamma, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(beta, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance_sqrt, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(INInferV2D)

/**
*@brief Performs instance normalization for inference of InHost part.

*@par Inputs:\n
* One input, including: (NC1HWC0 supported)
* variance: A [N, C1, 1, 1, C0] Tensor of type float32, for the variance.

*@par Attributes:
* epsilon: An optional float32, specifying the small value added to
variance to avoid dividing by zero. Defaults to "0.00001" . \n

*@par Outputs:\n
* variance_sqrt: A [N, C1, 1, 1, C0] Tensor of type float32, for the variance_sqrt.
*/
REG_OP(InHost)
     .INPUT(variance, TensorType({DT_FLOAT}))
     .OUTPUT(variance_sqrt, TensorType({DT_FLOAT}))
     .ATTR(epsilon, Float, 0.00001)
     .OP_END_FACTORY_REG(InHost)

/**
* @brief perform instance normalization to x. \n

* @par Inputs:
* Three inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32, format is NC1HWC0.
* @li gamma: A Tensor. Must be one of the following types: float16, float32, format is ND.
* @li beta: A Tensor. Must be one of the following types: float16, float32, format is ND.

* @par Attributes:
* @li data_format: An attribute of type String \n
* @li epsilon: An attribute of type Float, . \n

* @par Outputs:
* @li y: A Tensor. Has the same type as "x", format is NC1HWC0. \n
* @li mean: A Tensor. Has the same type as "x", format is NC1HWC0 and the shape is [N, C1, 1, 1, C0]. \n
* @li variance: A Tensor. Has the same type as "x", format is NC1HWC0 and the shape is [N, C1, 1, 1, C0]. \n

* @par Third-party framework compatibility
* Can be used by onnx InstanceNormalization
*/
REG_OP(InstanceNorm)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(variance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(data_format, String)
    .REQUIRED_ATTR(epsilon, Float)
    .OP_END_FACTORY_REG(InstanceNorm)

REG_OP(KlDivLossGrad)
    .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(reduction, String, "mean")
    .ATTR(log_target, Bool, false)
    .OP_END_FACTORY_REG(KlDivLossGrad)

/**
* @brief Computes l1_loss_grad or l1_loss_backward. \n

* @par Inputs:
* Three inputs, including:
* @li grads: A Tensor. Must be one of the following types: float16, float32.
* Required.
* @li predict: A Tensor. Has the same type as "grads". Required.
* @li label: A Tensor. Has the same type as "grads". Required. \n

* @par Attributes:
* @li reduction: An optional attribute of type String. Defaults to "mean". \n

* @par Outputs:
* @li y: A Tensor. Has the same type as "x". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator L1LossGrad.
*/
REG_OP(L1LossGrad)
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(label, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(L1LossGrad)

/**
* @brief Computes loss of lp, p=1,2,3....

* @par Inputs:
* @li predict: An ND tensor of type float16, float32.
* @li label: An ND tensor of type float16, float32. \n

* @par Attributes:
* @li p: A required int attribute that decides which loss to compute, now the p only can be 1 to compute l1_loss.
* @li reduction: An optional string.Defaults to "mean". \n

* @par Outputs:
* @li y: An ND tensor tensor with the same shape and type as "predict". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator LpLoss.
*/
REG_OP(LpLoss)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(label, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(p, Int)
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(LpLoss)

/**
* @brief Computes gradients of mse loss.

* @par Inputs:
* @li predict: An ND tensor of type float16, float32.
* @li label: An ND tensor of type float16, float32.
* @li dout: An ND tensor of type float16, float32. \n

* @par Attributes:
* @li reduction: An optional string.Defaults to "mean". \n

* @par Outputs:
* @li y: An ND tensor tensor with the same shape and type as "predict". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator MseLossGrad.
*/
REG_OP(MseLossGrad)
    .INPUT(predict, TensorType({DT_FLOAT32, DT_FLOAT16}))
    .INPUT(label, TensorType({DT_FLOAT32, DT_FLOAT16}))
    .INPUT(dout, TensorType({DT_FLOAT32, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT32, DT_FLOAT16}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(MseLossGrad)

/**
* @brief Computes mse loss.
* @par Inputs:
* two inputs, including:
*  @li predict: An ND Tensor of dtype float16 or float32.
*  @li label: An ND Tensor of dtype float16 or float32.\n
*
* @par Attributes:
*  @li reduction:An optional str from sum, none, mean, Defaults to "mean".\n
*
* @par Outputs:
*  @li y: when reduction=sum/mean, y is scale. when reduction=none, y has
*    same type and shape as "predict".\n
*/
REG_OP(MseLoss)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(label, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(MseLoss)

/**
* @brief Calculates the reversed outputs of the function "smooth_l1_loss_v2". \n

* @par Inputs:
* Three Inputs, including:
* @li predict: A Tensor. Must be one of the following types:
*     float16, float32.
* @li label: A Tensor. Has the same type as "predict".
* @li dout: A Tensor. Has the same type as "predict". \n

* @par Attributes:
* Two Attributes, including:
* @li sigma: An optional float. Defaults to 1.0. \n

* @li reduction: An optional string. Defaults to "mean",
*    Must be one of the following: "none", "mean", "sum". \n

* @par Outputs:
* @li gradient: A Tensor. Has the same type as "predict". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator SmoothL1LossBackward.
*/
REG_OP(SmoothL1LossGradV2)
    .INPUT(predict, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(label, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(dout, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(gradient, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(sigma, Float, 1.0)
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(SmoothL1LossGradV2)

/**
* @brief Creates a criterion that uses a squared term if the absolute
* element-wise error falls below beta and an L1 term otherwise. It is
* less sensitive to outliers than the MSELoss and in some cases prevents
* exploding gradients.

* @par Inputs:
* @li predict: A multi-dimensional Tensor of type float16 or float32,
* specifying the predictive value. \n
* @li label: A multi-dimensional Tensor of type float16 or float32,
* specifying the target value. \n

* @par Attributes:
* @li sigma: An optional int. Specifies the threshold of loss. Defaults
* to "1.0". \n
* @li reduction: An optional str. Specifies the reduction to apply to
* the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
* 'mean': the sum of the output will be divided by the number of elements in
* the output,'sum': the output will be summed. Default: 'mean'. \n

* @par Outputs:
* @li loss: Indicates the loss between the predictive value and target value.
* Has the same dimensions as "predict". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator smooth_l1_loss. \n
*/
REG_OP(SmoothL1LossV2)
    .INPUT(predict, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .INPUT(label, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .OUTPUT(loss, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .ATTR(sigma, Float, 1.0)
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(SmoothL1LossV2)

/**
* @brief Computes gradients of sigmoid_cross_entropy_with_logits_v2.

* @par Inputs:
* @li predict: An ND tensor of type float16, float32.
* @li target: An ND tensor of type float16, float32.
* @li dout: An ND tensor of type float16, float32.
* @li weight: An optional ND tensor of type float16, float32.
* @li pos_weight: An optional ND tensor of type float16, float32. \n

* @par Attributes:
* @li reduction: An optional string.Defaults to "mean". \n

* @par Outputs:
* @li gradient: An ND tensor tensor with the same shape and type as "predict". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator SigmoidCrossEntropyWithLogitsGrad.
*/
REG_OP(SigmoidCrossEntropyWithLogitsGradV2)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(dout, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(pos_weight, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(gradient, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(SigmoidCrossEntropyWithLogitsGradV2)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_OPS_H_
