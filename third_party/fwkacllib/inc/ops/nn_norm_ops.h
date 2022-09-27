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
*A "batch_size * num_classes" matrix.
* @li labels: A Tensor. Must be one of the following types: 'int32', 'int64'.
*batch_size vector with values in [0, num_classes).
*This is the label for the given minibatch entry. \n


*@par Outputs:
*@li loss: A Tensor for per example loss (a "batch_size" vector). Has the same type as "features".
*@li backprop: A Tensor for the backpropagated gradients (a batch_size * num_classes matrix). 
Has the same type as "features" . \n

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
* @li loss: A Tensor for per example loss (a "batch_size" vector). Has the same type as "features".
* @li backprop: A Tensor for the backpropagated gradients (a batch_size * num_classes matrix). Has the same type as "features" . \n

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
* types: float16, float31, int32, int8, uint8.
* @li grad_softmax: A Tensor. Has the same shape and type as "softmax".\n

*@par Attributes:
* axes: An optional list of ints. Defaults to "{-1}" . \n

*@par Outputs:
*grad_x: A Tensor. Has the same shape and type as "softmax" . \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator SoftmaxGrad.
*/
REG_OP(SoftmaxGrad)
    .INPUT(softmax, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(grad_softmax, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(grad_x, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(axes, ListInt, {-1})
    .OP_END_FACTORY_REG(SoftmaxGrad)

/**
* @brief Computes the sigmoid cross entropy loss of "predict" and "target" .

*@par Inputs:
* Three inputs, including:
*@li predict: A multi-dimensional Tensor of type float16 or float32, specifying the predictive value.
*@li target: A multi-dimensional Tensor of type float16 or float32, specifying the target value .
*@li dout:A multi-dimensional Tensor of float16 or float32,specifying the gradient transferred from the upper layer. \n

*@par Outputs:
*gradient: Sigmoid cross entropy between the predictive value and target value. Has the same dimensions as "predict" . \n

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
* @brief Performs the backpropagation of SigmoidCrossEntropyWithLogits for training scenarios .

*@par Inputs:
* Two inputs, including:
*@li predict: A multi-dimensional Tensor of type float16 or float32, specifying the predictive value.
*@li target: A multi-dimensional Tensor of type float16 or float32, specifying the target value. \n

*@par Outputs:
*loss: Return loss. Has the same dimensions and type as "predict" . \n

*@par Third-party framework compatibility
* Compatible with the scenario where "reduction" is set to "none"of PyTorch operator SigmoidCrossEntropyWithLogits.
*/
REG_OP(SigmoidCrossEntropyWithLogits)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(loss, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(SigmoidCrossEntropyWithLogits)

/**
*@brief Computes the sigmoid cross entropy loss of "predict" and "target".

*@par Inputs:
* four inputs, including:
*@li predict: A multi-dimensional Tensor of type float16 or float32, specifying the predictive value.
*@li target: A multi-dimensional Tensor of type float16 or float32, specifying the target value.
*@li weight: An multi-dimensional Tensor, specifying the weight value.
*@li pos_weight: An multi-dimensional Tensor, specifying the pos weight value. \n

*@par Attributes:
*reduction: A character string from "none", "mean", and "sum", specifying the reduction type to be applied to the output. Defaults to "mean". \n

*@par Outputs:
*loss: Sigmoid cross entropy between the predictive value and target value. Has the same dimensions as "predict". \n

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
* @brief Computes the sigmoid focal loss of "pred" and "target".

* @par Inputs:
* Three inputs, including:
* @li pred: A 2-dimensional Tensor of type float16 or float32, specifying the predicted value.
* @li target: A 1-dimensional Tensor of type int32, specifying the target value.
* @li weight: A 1-dimensional Tensor, specifying the weight value. \n

* @par Attributes:
* @li gamma: An optional float, specifying the exponent of the modulating factor (1 - pt)
* to balance easy/hard examples. Defaults to 2.0. 
* @li alpha: An optional float, specifying the weighting factor in range (1, 0) to balance
* the importance of positive/negative examples or less than 0 for ignore. Defaults to 0.25. 
* @li reduction: A optional character string from "none", "mean", and "sum", specifying the
* reduction type to be applied to the output. Defaults to "mean".  \n

* @par Outputs:
* loss: Sigmoid focal loss between the predicted value and target value. Has the same dimensions as "pred". \n

* @par Third-party framework compatibility
* Compatible with mmcv operator SigmoidFocalLoss.
*/
REG_OP(SigmoidFocalLoss)
    .INPUT(pred, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(target, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(loss, TensorType({DT_FLOAT16,DT_FLOAT}))
    .ATTR(gamma, Float, 2.0)
    .ATTR(alpha, Float, 0.25)
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(SigmoidFocalLoss)

/**
* @brief Computes the regression box of the RPN. It is a FasterRCNN operator .

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
* @brief Performs the backpropagation of SmoothL1Loss for training scenarios .

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
*@brief Function softmax with dropoutDoMaskV3D

*@par Inputs:
*Two inputs, including:
* @li x: A mutable Tensor. The type only support float16.
* @li mask: A mutable Tensor. Must met all of the following rules:
*     shape of mask should be 1D.
*     dtype of mask should be uint8.
*     value of shape should met the following algorithm:
*     value = (size(x) + 128 - 1) // 128 * 128

*@par Attributes:
* @li keep_prob: A mutable Tensor. Must met all of the following rules:
*     shape of "keep_prob" should be (1,) or [1,].
*     Has the same type as "x" . \n
* @li axes: A list of int. The dimension softmax would be performed on. Defaults
*     to "[-1]" . \n

*@par Outputs:
*y1: A mutable Tensor. Has the same type as "x".
*y2: A mutable Tensor. Has the same type as "x". \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(SoftmaxV2WithDropOutDoMaskV3D)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(mask, TensorType({DT_UINT8}))
    .OUTPUT(y1, TensorType({DT_FLOAT16}))
    .OUTPUT(y2, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(keep_prob, Float)
    .ATTR(axes, ListInt, {-1})
    .OP_END_FACTORY_REG(SoftmaxV2WithDropOutDoMaskV3D)

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
* y: A Tensor dtype of float16, float32. \n

*@attention Constraints:
* THIS OPERATOR IS DEPRECATED. It will be removed in a future version.
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
*@brief Normalizes the input . \n

*@par Inputs:
* One input:
*x: An NCHW tensor of type float16 or float32 . \n

*@par Attributes:
*@li eps: An optional float32 epsilon for not dividing by zero. Defaults to "1e-9" . \n
*@li axes: A list of Intefers, along which axis to reduce. Defaults to "[0, 2, 3]" . \n

*@par Outputs:
*y: An NCHW tensor of type float16 or float32 . \n

*@attention Constraints:
* The input tensor must have the NCHW format, whose shape length must be 4.
*@par Third-party framework compatibility
* Compatible with the ONNX operator MeanVarianceNormalization.
*/

REG_OP(MVNV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16})) /* "First operand." */
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))  /* "Result, has same element type as inputs" */
    .ATTR(eps, Float, 1e-9)
    .ATTR(axes, ListInt, {0, 2, 3})
    .OP_END_FACTORY_REG(MVNV2)

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
*@brief Returns a tensor where each sub-tensor of input along dimension 
*       dim is normalized such that the p-norm of the sub-tensor is lower than the value maxnorm. \n

*@par Inputs:
*One input, including:
* x: A Tensor. Must be one of the following types: float16, float32 . \n

*@par Attributes:
* @li p: Specify L_p norm, the type is float. 
* @li dim: The processed dim, the type is int.
* @li maxnorm: Threshold for comparison, the type is float.  \n

*@par Outputs:
*One outputs, including:
* y: shape and dtype of output, should be same shape and type as input.
*/
REG_OP(Renorm)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(p, Float)
    .REQUIRED_ATTR(dim, Int)
    .REQUIRED_ATTR(maxnorm, Float)
    .OP_END_FACTORY_REG(Renorm)

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
*@brief LayerNormXBackpropV2 operator interface implementation
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
*  res_for_gamma = (data_x - data_mean) * np.power((data_variance + EPSLON), (-0.5))

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
* @li res_for_gamma: A Tensor. Must be one of the following types: float32.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(LayerNormXBackpropV2)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(res_for_gamma, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(LayerNormXBackpropV2)

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
*@brief LayerNormBetaGammaBackpropV2 operator interface implementation
*  calculating: dy, x, variance, mean
*  pd_gamma = np.sum((data_dy*res_for_gamma), param_axis, keepdims=True)
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
REG_OP(LayerNormBetaGammaBackpropV2)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(res_for_gamma, TensorType({DT_FLOAT}))
    .OUTPUT(pd_gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_beta, TensorType({DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(shape_gamma, ListInt)
    .OP_END_FACTORY_REG(LayerNormBetaGammaBackpropV2)

/**
* @brief LNDropoutGrad operator interface implementation
*   calculating: dy, x, variance, mean, gamma
*   pd_xl = dy*gamma
*   sub_x_mean = x - mean
*   var_elta_2 = np.power((variance + EPSLON), (-0.5))
*   pd_var = sum(pd_xl * sub_x_mean, reduce_axis, keepdims=True) * var_elta_2 * var_elta_2 * var_elta_2 * (-0.5)
*   pd_mean = sum(pd_xl, reduce_axis, keepdims=True) * var_elta_2 * (-1.0)
*   pd_x = pd_xl * var_elta_2 + pd_var * (2.0 / m) * sub_x_mean + pd_mean * (1.0 / m)
*   pd_x_dropout = pd_x * mask * (1 / keep_prob)
*   pd_gamma = sum(dy * sub_x_mean * var_elta_2, param_axis, keepdims=True)
*   pd_beta = sum(dy, param_axis, keepdims=True)

* @par Inputs:
* Six inputs, including:
*  @li dy: A Tensor. Must be one of the following types: float16, float32.
*  @li x: A Tensor. Must be one of the following types: float16, float32.
*  @li variance: A Tensor. Must be one of the following types: float16, float32.
*  @li mean: A Tensor. Must be one of the following types: float16, float32.
*  @li gamma: A Tensor. Must be one of the following types: float16, float32.
*  @li mask: A Tensor. Must be one of the following types: uint8.\n

* @par Outputs:
* Four outputs, including:
*  @li pd_x: A Tensor. Must be one of the following types: float16, float32.
*  @li pd_x_dropout: A Tensor. Must be one of the following types: float16, float32.
*  @li pd_gamma: A Tensor. Must be one of the following types:  float16, float32.
*  @li pd_beta: A Tensor. Must be one of the following types:  float16, float32.

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(LNDropoutGrad)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(mask, TensorType({DT_UINT8}))
    .OUTPUT(pd_x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_x_dropout, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_beta, TensorType({DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(keep_prob, Float)
    .OP_END_FACTORY_REG(LNDropoutGrad)

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

*@par Outputs:
*y: A mutable Tensor. Has the same type as "x".
*/
REG_OP(DropOutDoMask)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(mask, TensorType({DT_UINT8}))
    .INPUT(keep_prob, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(DropOutDoMask)

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
*     value = (size(x) + 128 - 1) // 128 * 128
* @li keep_prob: A mutable Tensor. Must met all of the following rules:
*     shape of "keep_prob" should be (1,) or [1,].
*     Has the same type as "x" . \n

*@par Outputs:
*y: A mutable Tensor. Has the same type as "x".
*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(DropOutDoMaskV3)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(mask, TensorType({DT_UINT8}))
    .INPUT(keep_prob, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(DropOutDoMaskV3)

/**
*@brief Return "output" according to the algorithm of dropout_do_mask:
*  scale_x = x *(1 / keep_prob)
*  output = select(mask == 1, scale_x, 0)

*@par Inputs:
*Two inputs, including:
* @li x: A mutable Tensor. Must be one of the following types:
*     float16, float32
* @li mask: A mutable Tensor. Must met all of the following rules:
*     shape of mask should be 1D.
*     dtype of mask should be uint8.
*     value of shape should met the following algorithm:
*     value = (size(x) + 128 - 1) // 128 * 128
*@par Attributes:
* @li keep_prob: A mutable Tensor. Must met all of the following rules:
*     shape of "keep_prob" should be (1,) or [1,].
*     Has the same type as "x" . \n

*@par Output:
*y: A mutable Tensor. Has the same type as "x".
*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(DropOutDoMaskV3D)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(mask, TensorType({DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(keep_prob, Float)
    .OP_END_FACTORY_REG(DropOutDoMaskV3D)

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
*x: A Tensor. Must be 4-D shape, and only support the following types: float16, float32 . \n

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
 *blank_label: An optional attribute. Defaults to 0.

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
* @brief Performs group normalization . \n

* @par Inputs:
* Three inputs
* @li x: A ND Tensor of type float16 or float32, with format NCHW for 4D.
* @li gamma: A Tensor of type float16 or float32. Must be 1D. Specifies the scaling factor.
* @li beta: A Tensor of type float16 or float32. Must be 1D. Specifies the offset. \n

* @par Attributes:
* @li num_groups: An required int32, specifying the number of group.
* @li eps: An optional float32, specifying the small value added to
variance to avoid dividing by zero. Defaults to "0.0001".
* @li data_format: An optional string, specifying the format of "x".
Defaults to "NHWC".
* @li is_training: An optional bool, specifying if the operation is used for
training or inference. Defaults to "True" . \n

* @par Outputs:
* Three outputs
* @li y: A ND Tensor of type float16 or float32 for the normalized "x",
with format NCHW for 4D.
* @li mean: A Tensor of type float16 or float32. Must be 1D. Specifies the mean of "x".
* @li variance: A Tensor of type float16 or float32. Must be 1D. Specifies the variance of "x". \n

* @attention Constraints:
* @li For Ascend 310, only support NCHW which can be trans to 5HD. \n

* @par Third-party framework compatibility
* @li Compatible with the PyTorch operator GroupNorm.

*/
REG_OP(GroupNorm)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(variance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(data_format, String, "NHWC")
    .ATTR(eps, Float, 0.0001)
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(GroupNorm)

/**
*@brief Performs instance normalization . \n

*@par Inputs:
* Five inputs, including:
*@li x: A 5D Tensor of type float16 or float32.
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
* Three outputs, including: (NHWC, NCHW supported)
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
* Five inputs, including:
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
* @brief InstanceNorm operator interface implementation.

* @par Inputs:
* Three inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32.
* @li gamma: A Tensor. Must be one of the following types: float16, float32.
* @li beta: A Tensor. Must be one of the following types: float16, float32.

* @par Attributes:
* @li data_format: An attribute of type String \n
* @li epsilon: An attribute of type Float. \n

* @par Outputs:
* Three outputs, including:
* @li y: A Tensor. Has the same type as "x". \n
* @li mean: A Tensor. Has the same type as "x". \n
* @li variance: A Tensor. Has the same type as "x". \n

*/
REG_OP(InstanceNorm)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(variance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(data_format, String, "NDHWC")
    .ATTR(epsilon, Float, 1e-6)
    .OP_END_FACTORY_REG(InstanceNorm)

/**
* @brief InstanceNormGrad operator interface implementation.

* @par Inputs:
* Five inputs, including:
* @li dy: A Tensor. Must be one of the following types: float16, float32.
* @li x: A Tensor. Must be one of the following types: float16, float32.
* @li variance: A Tensor. Must be one of the following types: float16, float32.
* @li mean: A Tensor. Must be one of the following types: float16, float32.
* @li gamma: A Tensor. Must be one of the following types: float16, float32 . \n

* @par Outputs:
* Three outputs, including:
* @li pd_x: A Tensor. Must be one of the following types: float16, float32.
* @li pd_gamma: A Tensor. Must be one of the following types: float16, float32.
* @li pd_beta: A Tensor. Must be one of the following types: float16, float32.

*/
REG_OP(InstanceNormGrad)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_beta, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(InstanceNormGrad)

/**
* @brief Computes Kl_div_loss_grad or Kl_div_loss_backward. \n

* @par Inputs:
* Three inputs, including:
* @li grad: A Tensor. Must be one of the following types: float16, float32.
* Required.
* @li input: A Tensor. Has the same type as "grad". Required.
* @li target: A Tensor. Has the same type as "grad". Required. \n

* @par Attributes:
* @li reduction: An optional attribute of type String. Defaults to "mean". \n
* @li log_target: An optional attribute of type Bool. Defaults to false. \n

* @par Outputs:
* @li y: A Tensor. Has the same type as "grad". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator KlDivLossGrad.
*/
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
* reduction: An optional attribute of type String. Defaults to "mean". \n

* @par Outputs:
* y: A Tensor. Has the same type as "x". \n

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
*  y: An ND tensor tensor with the same shape and type as "predict". \n

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
* reduction: An optional string.Defaults to "mean". \n

* @par Outputs:
* y: An ND tensor tensor with the same shape and type as "predict". \n

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
* reduction:An optional str from sum, none, mean, Defaults to "mean".\n
*
* @par Outputs:
* y: when reduction=sum/mean, y is scale. when reduction=none, y has
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
*  gradient: A Tensor. Has the same type as "predict". \n

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
* loss: Indicates the loss between the predictive value and target value.
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
* @brief Computes Centralization. result = x - mean(x, axes)

* @par Inputs:
*  x: An ND tensor of type float16, float32.
* @par Attributes:
* axes: The dimensions to reduce. Must be one of the following types: int, list, tuple, NoneType.
* Must be in the range [-rank(x), rank(x)).
* @par Outputs:
* y: A Tensor. Has the same type as "x". \n

* @par Third-party framework compatibility
* custom operator \n
*/
REG_OP(Centralization)
    .INPUT(x, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .OUTPUT(y, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .ATTR(axes, ListInt, {-1})
    .OP_END_FACTORY_REG(Centralization)

/**
*@brief Roll the tensor along the given dimension(s).
* Elements that are shifted beyond the last position are re-introduced at the first position.
* If a dimension is not specified, the tensor will be flattened before rolling and then restored to the original shape. \n

*@par Inputs:
*One inputs, including:
* x: A tensor . Must be one of the following types:
*     float16, float32, int32, uint32, int8, uint8. \n

*@par Attributes:
* @li shifts: The number of places by which the elements of the tensor are shifted. \n
* @li dims: Axis along which to roll. \n

*@par Outputs:
* y: A Tensor with the same type and shape of x's. \n

*@par Third-party framework compatibility
*Compatible with the Pytorch operator Roll. \n
*/
REG_OP(Roll)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_UINT32,DT_INT8,DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_UINT32,DT_INT8,DT_UINT8}))
    .REQUIRED_ATTR(shifts, ListInt)
    .ATTR(dims, ListInt, {})
    .OP_END_FACTORY_REG(Roll)

/**
* @brief Roll the tensor along the given dimension(s).

* @par Inputs:
* One inputs, including:
* x: A tensor

* @par Attributes:
* @li shift: The number of places by which the elements of the tensor are shifted. \n
* @li axes: Axis along which to roll. \n

* @par Outputs:
* y: A Tensor with the same type and shape of x's. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator Roll. \n
*/
REG_OP(RollV2)
    .INPUT(input, TensorType({DT_INT8,DT_UINT8,DT_INT16,DT_UINT16,DT_INT32,DT_INT64,DT_FLOAT16, \
                            DT_FLOAT,DT_DOUBLE}))
    .INPUT(shift, TensorType({DT_INT32,DT_INT64}))
    .INPUT(axes, TensorType({DT_INT32,DT_INT64}))
    .OUTPUT(output, TensorType({DT_INT8,DT_UINT8,DT_INT16,DT_UINT16,DT_INT32,DT_INT64,DT_FLOAT16, \
                            DT_FLOAT,DT_DOUBLE}))
    .OP_END_FACTORY_REG(RollV2)

/**
 * @brief Calculate the loss. Creates a criterion that optimizes a two-class classification
 * logistic loss between input_x and input_y (containing 1 or -1). \n

 * @par Inputs:
 * Tow inputs, including:
 * @li input_x: A tensor. Must be one of the following types:
 *     float16, float32. \n
 * @li input_y: A tensor. Must be one of the following types:
 *     float16, float32. \n

 * @par Attributes:
 * reduction: An optional string.Defaults to "mean". \n

 * @par Outputs:
 * output_z: while reduction == "none", A Tensor with the same type and shape of input_x's. \n
 *          while reduction == "sum" or "mean", A Tensor with the same type of input_x , shape of which is (1,)

 * @par Third-party framework compatibility
 * Compatible with the Pytorch operator SoftMarginLoss. \n
 */
REG_OP(SoftMarginLoss)
    .INPUT(input_x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(input_y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(reduction, String, "mean")
    .OUTPUT(output_z, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(SoftMarginLoss)

/**
* @brief Computes gradients of sigmoid_cross_entropy_with_logits_v2.

* @par Inputs:
* @li predict: An ND tensor of type float16, float32.
* @li target: An ND tensor of type float16, float32.
* @li dout: An ND tensor of type float16, float32.
* @li weight: An optional ND tensor of type float16, float32.
* @li pos_weight: An optional ND tensor of type float16, float32. \n

* @par Attributes:
* reduction: An optional string.Defaults to "mean". \n

* @par Outputs:
* gradient: An ND tensor tensor with the same shape and type as "predict". \n

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
/**
 * @brief Calculate the PoissonNllLoss function. 
 *        target∼Poisson(input)loss(input,target)=input−target∗log(input)+log(target!) \n

 * @par Inputs:
 * Two inputs, including:
 * @li input_x: A tensor. Must be one of the following types: float16, float32.
 * @li target: A tensor. Must be one of the following types: float16, float32. \n

 * @par Attributes:
 * four Attributes, including:
 * @li log_input: An optional bool. Defaults to "True"
 * @li full: An optional bool. Defaults to "False"
 * @li eps: An optional float. Defaults to "1e-8"
 * @li reduction: An optional string. Defaults to "mean" \n

 * @par Outputs:
 * loss: A Tensor has same element type as two inputs. \n

 * @par Third-party framework compatibility
 * Compatible with the Pytorch operator PoissonNllLoss. \n
 */
REG_OP(PoissonNllLoss)
    .INPUT(input_x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(loss, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(log_input, Bool, true)
    .ATTR(full, Bool, false)
    .ATTR(eps, Float, 1e-8)
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(PoissonNllLoss)
/**
 *@brief rnn_gen_mask
 * @par Inputs:
 * seq_length: A ND Tensor of type int32. Recoed the current length of each batch.\n
 *
 * @par Attributes:
 * @li num_step: A required int.\n
 * @li hidden_size: A required int. \n
 *
 * 
 * @par Ouputs:
 * y: A mutable Tensor of type float16, with the shape of [num_step, batch_size, hidden_size]. \n
 *
 */
REG_OP(RnnGenMask)
    .INPUT(seq_length, TensorType({DT_INT32}))
    .OUTPUT(seq_mask, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(num_step, Int)
    .REQUIRED_ATTR(hidden_size, Int)
    .OP_END_FACTORY_REG(RnnGenMask)

/**
* @brief Creates a criterion that optimizes a multi-class multi-classification hinge loss (margin-based loss) 
*        between input x (a 2D mini-batch Tensor) and output y (which is a 2D Tensor of target class indices) \n
 
* @par Inputs:
* Two inputs, including:
* @li x: A tensor. Must be one of the following types:
*     float16, float32.
* @li target: A tensor. Must be the following types:
*     int32. \n

* @par Attributes:
* reduction: An optional string. Defaults to "mean" \n

* @par Outputs:
* @li y: A Tensor has same element type as input x. \n
* @li is_target: A Tensor has same element type as input target. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator MultiLabelMarginLoss. \n
*/
REG_OP(MultilabelMarginLoss)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(target, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(is_target, TensorType({DT_INT32}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(MultilabelMarginLoss)

/**
* @brief Performs batch normalization . \n
* @par Inputs:
* Two inputs
* @li input_x: A Tensor. Support float32. shape (n, c, d).
* @li seq_len: A Tensor. Each batch normalize data num. Support Int32. Shape (n, ). \n
* @par Attributes:
* @li normalize_type: Str. Support "per_feature" or "all_features".
* @li epsilon: An optional float32, specifying the small value added to
* variance to avoid dividing by zero. Defaults to "0.00001" . \n
* @par Outputs:
* One outputs
* @li output_y: A Tensor for the normalized "x".Support float32. shape (n, c, d).\n
*/
REG_OP(NormalizeBatch)
    .INPUT(input_x, TensorType({ DT_FLOAT }))
    .INPUT(seq_len, TensorType({ DT_INT32 }))
    .OUTPUT(output_y, TensorType({ DT_FLOAT }))
    .REQUIRED_ATTR(normalize_type, String)
    .ATTR(epsilon, Float, 0.00001)
    .OP_END_FACTORY_REG(NormalizeBatch)

/**
*@brief GroupNorm and Reul operator
*  calculating: x, gamma, beta
*  y = relu(gamma*((x - mean) / np.sqrt(variance + 0.001)) + beta)

* @par Inputs:
* Three inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32.
* @li gamma: A Tensor. Must be one of the following types: float16, float32.
* @li beta: A Tensor. Must be one of the following types: float16, float32 . \n

* @par Attributes:
* @li num_groups: A require attribute, the type is int32.
* @li eps: A optional attribute, the type is float32. Defaults to 0.00001. \n

* @par Outputs:
* One outputs, including:
* @li y: A Tensor. Must be one of the following types: float16, float32.
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use/
*/
REG_OP(GroupNormRelu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(eps, Float, 0.00001)
    .OP_END_FACTORY_REG(GroupNormRelu)

/**
* @brief Function dropout with softmaxgrad and muls

* @par Inputs:
* Two inputs, including:
* @li y_grad: A mutable Tensor. The type only support float16.
* @li mask: A mutable Tensor. Must met all of the following rules:
*     shape of mask should be 1D.
*     dtype of mask should be uint8.
*     value of shape should met the following algorithm:
*     value = (size(x) + 128 - 1) // 128 * 128
* @li softmax_output: A mutable Tensor. Must met all of the following rules:
*     shape of softmax_output should be NZ.
*     dtype of softmax_output should be float16.
*     it is the output of softmax

* @par Attributes:
* @li input_keep_prob:A attribute used to judge which units should be keep.
*     Has the same type as "x" . \n
* @li alpha: A attribute used to scale tensor.
*     Has the same type as "x" . \n
* @li axes: A list of int. The dimension softmax would be performed on. Defaults
*     to "[-1]" . \n

* @par Outputs:
* x_grad: A mutable Tensor. Has the same type as "x". \n

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(DropoutWithMulsAndSoftmaxGrad)
    .INPUT(y_grad, TensorType({ DT_FLOAT16 }))
    .INPUT(mask, TensorType({ DT_UINT8 }))
    .INPUT(softmax_output, TensorType({ DT_FLOAT16 }))
    .OUTPUT(x_grad, TensorType({ DT_FLOAT16 }))
    .REQUIRED_ATTR(input_keep_prob, Float)
    .REQUIRED_ATTR(alpha, Float)
    .ATTR(axes, ListInt, { -1 })
    .OP_END_FACTORY_REG(DropoutWithMulsAndSoftmaxGrad)

/**
* @brief Loss function that measures the softmax cross entropy. \n

* @par Inputs:
* Three inputs, including:
* @li scores: A Tensor. Must be one of the following types: half, float32, double.
* A "batch_size * num_classes" matrix.
* @li labels: A Tensor. Must be one of the following types: "int32", "int64".
* @li weights: A manual rescaling weight given to each class. 
* If given, it has to be a 1D Tensor assigning weight to each of the classes.
* Otherwise, it is treated as if having all ones. \n

* @par Attributes:
* ignore_index:Specifies a target value that is ignored and does not contribute to the input gradient.
* It's an optional value.
* reduction: A character string from "none", "mean", and "sum", specifying the gradient output mode. Defaults to "mean" . \n

* @par Outputs:
* @li loss: A Tensor for per example loss (a "batch_size" vector). Has the same type as "scores".
* @li log_prop: A Tensor. Has the same type as "scores" . \n

* @par Third-party framework compatibility
* Compatible with the ONNX operator SoftmaxCrossEntropyLoss.
*/
REG_OP(SoftmaxCrossEntropyLoss)
    .INPUT(scores, TensorType({DT_DOUBLE,DT_FLOAT16,DT_FLOAT,DT_BFLOAT16}))
    .INPUT(labels, TensorType({DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(weights, TensorType({DT_DOUBLE,DT_FLOAT16,DT_FLOAT,DT_BFLOAT16}))
    .ATTR(ignore_index, Int, 0)
    .ATTR(reduction, String, "mean")
    .OUTPUT(loss, TensorType({DT_DOUBLE,DT_FLOAT16,DT_FLOAT,DT_BFLOAT16}))
    .OUTPUT(log_prop, TensorType({DT_DOUBLE,DT_FLOAT16,DT_FLOAT,DT_BFLOAT16}))
    .OP_END_FACTORY_REG(SoftmaxCrossEntropyLoss)

/**
* @brief Function axpy with softmax and dropoutdomask . \n

* @par Inputs:
* Three inputs, including:
* @li x1: A mutable Tensor. The type only support float16.
* @li x2: A mutable Tensor. The type only support float16.
* @li mask: A mutable Tensor. Must meet all of the following rules:
*     shape of mask should be 1D.
*     dtype of mask should be uint8.
*     value of shape should meet the following algorithm:
*     value = (size(x) + 128 - 1) // 128 * 128 . \n

* @par Attributes:
* @li alpha: A attribute used to scale tensor. The type is float . \n
* @li input_keep_prob: A attribute used to judge which units should be keep.
*     The type is float . \n
* @li axis: A list of int. The dimension softmax would be performed on. Defaults
*     to "[-1]" . \n

* @par Outputs:
* y1: A mutable Tensor. Has the same type as "x1". \n
* y2: A mutable Tensor. Has the same type as "x1". \n

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(AxpyWithSoftmaxAndDropOutDoMask)
    .INPUT(x1, TensorType({DT_FLOAT16}))
    .INPUT(x2, TensorType({DT_FLOAT16}))
    .INPUT(mask, TensorType({DT_UINT8}))
    .OUTPUT(y1, TensorType({DT_FLOAT16}))
    .OUTPUT(y2, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(alpha, Float)
    .REQUIRED_ATTR(input_keep_prob, Float)
    .ATTR(axis, ListInt, {-1})
    .OP_END_FACTORY_REG(AxpyWithSoftmaxAndDropOutDoMask)

/**
* @brief MMCV Function: sigmoid_focal_loss_grad  . \n

* @par Inputs:
* Three inputs, including:
* @li pred: the predicted tensor. The type support float16 and float32.
* @li target: the target label Tensor. The type support Int32.
* @li dout: the grad of previous op grad, which has the sampe shape wth pred. The type support float16 and float32.
* @li weight: A optioanl input Tensor, default is None, which helps to calculate the loss by supplying sample weights:
*     shape of pred should be (B，D), B means batch size, D means the number of labels.
*     shape of target should be (D, ).
*     shape of weight should be (D, ) \n

* @par Attributes:
* @li alpha: A attribute is used to reweight the sample. The type is float . \n
* @li gamma: A attribute is used to calculate the power of the probability.
*     The type is float . \n
* @li reduction: a type of the reduce method. default is 'mean', which means computing the average loss. 
                'sum' means computing the sum of the loss, 'none' means no reducing .\n

* @par Outputs:
* grad: A mutable Tensor. Has the same type and shape as "pred". \n

* @par Third-party framework compatibility
* Compatible with the MMCV operator SigmoidFocalLoss.
*/
REG_OP(SigmoidFocalLossGrad)
    .INPUT(pred, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_INT32}))
    .INPUT(dout, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(alpha, Float, 0.25)
    .ATTR(gamma, Float, 2.0)
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(SigmoidFocalLossGrad)

/**
* @brief MMCV Function: softmax_focal_loss_grad  . \n

* @par Inputs:
* Three inputs, including:
* @li pred: the predicted tensor. The type support float16 and float32.
* @li target: the target label Tensor. The type support Int32.
* @li dout: the grad of previous op grad, which has the sampe shape wth pred. The type support float16 and float32.
* @li weight: A optioanl input Tensor, default is None, which helps to calculate the loss by supplying sample weights:
*     shape of pred should be (B，D), B means batch size, D means the number of labels.
*     shape of target should be (B, D).
*     shape of weight should be (D, ) \n

* @par Attributes:
* @li alpha: A attribute is used to reweight the sample. The type is float . \n
* @li gamma: A attribute is used to calculate the power of the probability.
*     The type is float . \n
* @li reduction: a type of the reduce method. default is 'mean', which means computing the average loss. 
                'sum' means computing the sum of the loss, 'none' means no reducing .\n

* @par Outputs:
* grad: A mutable Tensor. Has the same type and shape as "pred". \n

* @par Third-party framework compatibility
* Compatible with the MMCV operator SoftmaxFocalLossGrad.
*/
REG_OP(SoftmaxFocalLossGrad)
    .INPUT(pred, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_INT32}))
    .INPUT(dout, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(alpha, Float, 0.25)
    .ATTR(gamma, Float, 2.0)
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(SoftmaxFocalLossGrad)
}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_OPS_H_
