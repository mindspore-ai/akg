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

#ifndef GE_OP_NN_NORM_OPS_H
#define GE_OP_NN_NORM_OPS_H

#include "graph/operator_reg.h"
namespace ge {

/**
*@brief Computes the gradient for log softmax activations.

*@par Inputs:
*@li grad: A Tensor. Must be one of the following types: float16, float32.
*@li x: A Tensor. Must be one of the following types: float16, float32.

*@par Attributes:
* axis: An optional list of ints. Defaults to "{-1}".

*@par Outputs:
* y: A Tensor. Has the same type as "grad".
*/

REG_OP(LogSoftmaxGrad)
    .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(axis, ListInt, {-1})
    .OP_END_FACTORY_REG(LogSoftmaxGrad)

REG_OP(SparseSoftmaxCrossEntropyWithLogitsCCE)
    .INPUT(features, TensorType{DT_FLOAT})
    .INPUT(labels, TensorType{DT_FLOAT})
    .OUTPUT(out, TensorType{DT_FLOAT})
    .OUTPUT(non, TensorType{DT_FLOAT})
    .ATTR(cross_entropy_is_grad, Bool, 0)
    .ATTR(cross_entropy_mode, Int, 1)
    .ATTR(softmax_cross_entropy_lossscale_div_batch, Float, 1.0)
    .OP_END_FACTORY_REG(SparseSoftmaxCrossEntropyWithLogitsCCE)

/**
*@brief Computes sparse softmax cross entropy cost and gradients to backpropagate.

*@par Inputs:
*Two inputs, including:
* @li features: A Tensor. Must be one of the following types: half, float32, double.
*    A "batch_size * num_classes" matrix.
* @li labels: A Tensor of the same type as "features". batch_size vector with values in [0, num_classes).


*@par Outputs:
*loss: A Tensor for per example loss (a "batch_size" vector). Has the same type as "features".
*backprop: A Tensor for the backpropagated gradients (a batch_size * num_classes matrix). Has the same type as "features".
*/
REG_OP(SparseSoftmaxCrossEntropyWithLogits)
    .INPUT(features, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(labels, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(loss, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(backprop, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(SparseSoftmaxCrossEntropyWithLogits)

/**
*@brief Computes softmax cross entropy cost and gradients to backpropagate.

*@par Inputs:
*Two inputs, including:
* @li features: A Tensor. Must be one of the following types: half, float32, double.
*    A "batch_size * num_classes" matrix.
* @li labels: A Tensor of the same type as "features". A "batch_size * num_classes" matrix.

*@par Outputs:
*loss: A Tensor for per example loss (a "batch_size" vector). Has the same type as "features".
*backprop: A Tensor for the backpropagated gradients (a batch_size * num_classes matrix). Has the same type as "features".
*/
REG_OP(SoftmaxCrossEntropyWithLogits)
    .INPUT(features, TensorType({DT_DOUBLE,DT_FLOAT16,DT_FLOAT}))
    .INPUT(labels, TensorType({DT_DOUBLE,DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(loss, TensorType({DT_DOUBLE,DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(backprop, TensorType({DT_DOUBLE,DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(SoftmaxCrossEntropyWithLogits)

/**
*@brief Computes gradients for a softmax operation.

*@par Inputs:
* Two inputs, including: \n
* @li softmax: Output of the softmax operator. Must be one of the following types: float16, float31, int32, int8, uint8. The format is NC1HWC0 or DN.
* @li grad_softmax: A Tensor. Has the same shape and type as "softmax". The format is NC1HWC0 or DN.

*@par Outputs:
*grad_x: A Tensor. Has the same shape and type as "softmax".

*/
REG_OP(SoftmaxGrad)
    .INPUT(softmax, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(grad_softmax, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(grad_x, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OP_END_FACTORY_REG(SoftmaxGrad)

/**
*@brief Computes the sigmoid cross entropy loss of "predict" and "target".

*@par Inputs:
* Two inputs, including: \n
*@li predict: A multi-dimensional Tensor of type float16 or float32, specifying the predictive value.
*@li target: A multi-dimensional Tensor of type float16 or float32, specifying the target value.

*@par Outputs:
*loss: Sigmoid cross entropy between the predictive value and target value. Has the same dimensions as "predict".

*/
REG_OP(SigmoidCrossEntropyWithLogitsGrad)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(dout, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(gradient, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(SigmoidCrossEntropyWithLogitsGrad)

/**
*@brief Performs the backpropagation of SigmoidCrossEntropyWithLogits for training scenarios.

*@par Inputs:
* Three inputs, including: \n
*@li predict: A multi-dimensional Tensor of type float16 or float32, specifying the predictive value.
*@li target: A multi-dimensional Tensor of type float16 or float32, specifying the target value.
*@li dout: A multi-dimensional Tensor of float16 or float32, specifying the gradient transferred from the upper layer.

*@par Outputs: \n
*gradient: Return gradient. Has the same dimensions and type as "predict".

*/
REG_OP(SigmoidCrossEntropyWithLogits)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(loss, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(SigmoidCrossEntropyWithLogits)

/**
*@brief Computes the regression box of the RPN. It is a FasterRCNN operator.

*@par Inputs:
* Two inputs, including: \n
*@li predict: A multi-dimensional Tensor of type float16 or float32, specifying the predictive value.
*@li label: A multi-dimensional Tensor of type float16 or float32, specifying the target value.

*@par Attributes:
* sigma: Must be a floating point number. Defaults to "1.0".

*@par Outputs:
*loss: Indicates the loss between the predictive value and target value. Has the same dimensions as "predict".

*@attention Constraints:
* This operator does not perform the "reduce" operation on the loss value. Call other reduce operators to perform "reduce" operation on the loss if required.

*/
REG_OP(SmoothL1Loss)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(label, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(loss, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(sigma, Float, 1.0)
    .OP_END_FACTORY_REG(SmoothL1Loss)

/**
*@brief Performs the backpropagation of SmoothL1Loss for training scenarios.

*@par Inputs:
* Three inputs, including: \n
*@li predict: A multi-dimensional Tensor of type float16 or float32, specifying the predictive value.
*@li label: A multi-dimensional Tensor of float16 or float32, specifying the target value.
*@li dout: A multi-dimensional Tensor of float16 or float32, specifying the gradient transferred from the upper layer.

*@par Attributes:
* sigma: Must be a floating point number. Defaults to "1.0".

*@par Outputs:
*gradient: Return gradient. Has the same dimensions and type as "predict".

*/
REG_OP(SmoothL1LossGrad)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(label, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(dout, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(gradient, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(sigma, Float, 1.0)
    .OP_END_FACTORY_REG(SmoothL1LossGrad)

/**
*@brief Creates a criterion that measures the Binary Cross Entropy between the target and the output.

*@par Inputs:
* Three inputs, including: \n
*@li x: A 1D or 2D Tensor of type float16 or float32, specifying a predictive value.
*@li y: A 1D or 2D Tensor of type float16 or float32, indicating a tag.
*@li weight: An optional 1D or 2D Tensor, specifying the weight.

*@par Attributes:
*reduction: A character string from "none", "mean", and "sum", specifying the reduction type to be applied to the output. Defaults to "mean".

*@par Outputs:
*output: Output loss. Has the same dimension with the inputs. When "reduction" is set to "none", a Tensor with the same size as "x" is output. Otherwise, a Scalar is output.

*@attention Constraints:
*@li The value of "x" must range from 0 to 1.
*@li The value of "y" must be "0" or "1".

*/
REG_OP(BinaryCrossEntropy)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(output, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(BinaryCrossEntropy)

/**
*@brief Performs the backpropagation of BinaryCrossEntropy for training scenarios.

*@par Inputs:
* Four inputs, including: \n
*@li x: A 1D or 2D Tensor of type float16 or float32, specifying a predictive value.
*@li y: A 1D or 2D Tensor of type float16 or float32, indicating a tag.
*@li grad_output: A 1D or 2D Tensor of type float16 or float32, specifying the backpropagation gradient.
*@li weight: An optional 1D or 2D Tensor, specifying the weight.

*@par Attributes: \n
*reduction: A character string from "none", "mean", and "sum", specifying the gradient output mode. Defaults to "mean".

*@par Outputs: \n
*output: A 1D or 2D Tensor. When "reduction" is set to "none", a Tensor with the same size as "x" is output. Otherwise, a Scalar is output.

*@attention Constraints:
*@li The value of "x" must range from 0 to 1.
*@li The value of "y" must be "0" or "1".

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
*@brief Applies the Softmax function to an n-dimensional input Tensor rescaling them \n so 
that the elements of the n-dimensional output Tensor lie in the range [0,1] and sum to 1.

*@par Inputs:
*One input:
*x: A mutable Tensor. Must be one of the following types: float16,
*float32, double. Should be a Variable Tensor.

*@par Attributes:
*axes: A list of ints. The dimension softmax would be performed on.

*@par Outputs:
*y: A Tensor. Has the same dimensionality and shape as the "x" with values in the range [0, 1]. Must be one of the following types: float16, float32, int32.
*/
REG_OP(SoftmaxV2)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT}))
    .ATTR(axes, ListInt, {-1})
    .OP_END_FACTORY_REG(SoftmaxV2)

/**
*@brief Computes log softmax activations.

*@par Inputs:
*One input:
* logits: A Tensor. Must be one of the following types: double, float16, float32.

*@par Attributes:
* axes: An optional list of ints. Defaults to "{-1}".

*@par Outputs:
* logsoftmax: A Tensor. Has the same type as "logits".
*/
REG_OP(LogSoftmaxV2)
    .INPUT(logits, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(logsoftmax, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT}))
    .ATTR(axes, ListInt, {-1})
    .OP_END_FACTORY_REG(LogSoftmaxV2)

REG_OP(FusedBatchNormV2)
    .INPUT(x, TensorType{DT_FLOAT})                  /* Input data tensor from the previous operator"" */
    .INPUT(scale, TensorType{DT_FLOAT})              /* If spatial is true, the dimension of bias is (C) If spatial is false, the dimensions of scale are (C x D1 x ... x Dn)*/
    .INPUT(b, TensorType{DT_FLOAT})                  /* If spatial is true, the dimension of bias is (C) If spatial is false, the dimensions of scale are (C x D1 x ... x Dn)*/
    .OPTIONAL_INPUT(mean, TensorType{DT_FLOAT})               /* If spatial is true, the dimension of the running mean (training) or the estimated mean (testing) is (C).If spatial is false, the dimensions of the running mean (training) or the estimated mean (testing) are (C x D1 x ... x Dn)*/
    .OPTIONAL_INPUT(variance, TensorType{DT_FLOAT})           /* If spatial is true, the dimension of the running variance(training) or the estimated variance (testing) is (C). If spatial is false, the dimensions of the running variance(training) or the estimated variance (testing) are (C x D1 x ... x Dn).*/
    .OUTPUT(y, TensorType{DT_FLOAT})                 /* The output tensor of the same shape as X */
    .ATTR(momentum, Float, 0.9)            // Factor used in computing the running mean and variance.
    .ATTR(epsilon, Float, 1e-5f)           // The epsilon value to use to avoid division by zero
    .ATTR(mode, Int, 1)                    // 1 means using "CC_BATCHNORM_SPATIAL"; 0 means using "CC_BATCHNORM_PER_ACTIVATION"; only support 1 now
    .ATTR(use_global_stats, Bool, true)
    .ATTR(alpha, Float, 1)
    .ATTR(beta, Float, 0)
    .OP_END_FACTORY_REG(FusedBatchNormV2)


/**
*@brief Confuse mul, sum and sub.

*@par Inputs:
*Two inputs, including:
* @li grad: A Tensor. Must be one of the following types: float16, float32.
* @li x: A Tensor. Must be one of the following types: float16, float32.

*@par Outputs:
* y: A Tensor of the same type as "grad".

*/
REG_OP(ConfusionSoftmaxGrad)
  .INPUT(grad, TensorType({DT_FLOAT16,DT_FLOAT}))
  .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
  .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
  .OP_END_FACTORY_REG(ConfusionSoftmaxGrad)

REG_OP(SoftmaxGradExt)
  .INPUT(grad, TensorType({DT_FLOAT16,DT_FLOAT}))
  .INPUT(x1, TensorType({DT_FLOAT16,DT_FLOAT}))
  .INPUT(x2, TensorType({DT_FLOAT16,DT_FLOAT}))
  .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
  .ATTR(axes, Int, 1)
  .ATTR(keep_dims, Bool, false)
  .OP_END_FACTORY_REG(SoftmaxGradExt)
  
/**
*@brief Normalizes the input.

*@par Inputs:
* One input:
*x: An NCHW tensor of type float16 or float32.

*@par Attributes:
*@li normalize_variance: An optional bool specifying whether to normalize the variance, either "true" (default) or "false"
* the value "false" indicates only to subtract the mean.
*@li across_channels: An optional bool specifying whether to perform across-channel MVN, either "true" or "false" (default)
* The value "true" indicates "CHW" is treated as a vector.
*@li eps: An optional float32 epsilon for not dividing by zero. Defaults to "1e-9".

*@par Outputs:
*y: An NCHW tensor of type float16 or float32.

*@attention Constraints:\n
* The input tensor must have the NCHW format, whose shape length must be 4.
*/

REG_OP(MVN)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16})) /* "First operand." */
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))  /* "Result, has same element type as inputs" */
    .ATTR(normalize_variance, Bool, true)
    .ATTR(across_channels, Bool, false)
    .ATTR(eps, Float, 1e-9)
    .OP_END_FACTORY_REG(MVN)

/**
*@brief Normalizes the input "x1".

*@par Inputs:
* Two inputs, including:
*@li x1: A required NCHW or NHWC tensor of type float32, float16, or int8.
*@li x2: A required ND tensor of type float32, float16, or int8, specifying
* the scaling factor. If "channel_shared" is "true", "x2" is a [1]-dimensional
* vector. If "channel_shared" is "false", "x2" is a [C]-dimensional vector.

*@par Attributes:
*@li across_spatial: An optional bool, specifying the dimension of input "x1"
* to be summed. The value "true" (default) indicates dimensions C, H, W, and
* the value "false" indicates dimension C.
*@li channel_shared: An optional bool, specifying the dimension count of input
* "x2". The value "true" (default) indicates 1, and the value "false" indicates
* dimension C of "x1".
*@li eps: An optional float32, specifying the bias when "across_spatial" is
* "true". Defaults to "1e-10".

*@par Outputs:
*y: A Tensor. Has the same type and format as "x1".

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
* @li beta: A Tensor. Must be one of the following types: float16, float32.

*@par Attributes:
* @li begin_norm_axis: A required attribute, the type is int32.
* @li begin_params_axis: A required attribute,the type is int32.

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
*Three inputs, including:
* @li dy: A Tensor. Must be one of the following types: float16, float32.
* @li x: A Tensor. Must be one of the following types: float16, float32.
* @li variance: A Tensor. Must be one of the following types: float16, float32.
* @li mean: A Tensor. Must be one of the following types: float16, float32.
* @li gamma: A Tensor. Must be one of the following types: float16, float32.

*@par Outputs:
*Three outputs, including:
* @li pd_x: A Tensor. Must be one of the following types: float16, float32.
* @li pd_gamma: A Tensor. Must be one of the following types: float16, float32.
* @li pd_beta: A Tensor. Must be one of the following types: float16, float32.
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
*Three inputs, including:
* @li dy: A Tensor. Must be one of the following types: float16, float32.
* @li x: A Tensor. Must be one of the following types: float16, float32.
* @li variance: A Tensor. Must be one of the following types: float16, float32.
* @li mean: A Tensor. Must be one of the following types: float16, float32.
* @li gamma: A Tensor. Must be one of the following types: float16, float32.

*@par Outputs:
*Three outputs, including:
* @li pd_x: A Tensor. Must be one of the following types: float16, float32.
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
* @li mean: A Tensor. Must be one of the following types: float16, float32.

*@par Outputs:
*Three outputs, including:
* @li pd_gamma: A Tensor. Must be one of the following types: float16, float32.
* @li pd_beta: A Tensor. Must be one of the following types: float16, float32.
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
*@brief Return "output" according to the algorithm of dropout_do_mask: \n
*  scale_x = x *(1 / keep_prob)
*  output = select(mask == 1, scale_x, 0)

*@par Inputs:
*Three inputs, including: \n
* @li x: A mutable Tensor. Must be one of the following types:
*     float16, float32
* @li mask: A mutable Tensor. Must met all of the following rules:
*     shape of mask should be 1D.
*     dtype of mask should be uint8.
*     value of shape should met the following algorithm:
*     value = (size(x) + 128 - 1) // 128 * 128 //8
* @li keep_prob: A mutable Tensor. Must met all of the following rules:
*     shape of "keep_prob" should be (1,) or [1,].
*     Has the same type as "x".

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
*@brief Scales the input.

*@par Inputs:
* Three inputs, including:
*@li x: An ND tensor of type float16 or float32.
*@li scale: An ND tensor of type float16 or float32.
*@li bias: An ND tensor of type float16 or float32.

*@par Attributes:
*@li axis: An optional int32 used to compute the shape of scale and bias input from the online bottoms. Defaults to "1".

*@par Outputs:
*y: An ND tensor of type float16 or float32.

*@attention Constraints:\n
* Assume that the shape length of "x" is "n" and that of "scale" is "m".
*@li "axis" is within the range [-n, n-1]. num_axes >= -1.
*@li If "scale_from_blob = true", "num_axes = -1", and "axis >= 0", the ith axis of "scale" and the (i+"axis")th axis of "x" must have the same size (0 <= i < n-axis).\n  
* If "axis < 0", the ith axis of "scale" and the (i+n+"axis")th axis of "x" must have the same size (0 <= i < -axis).
*@li If "scale_from_blob = true" and "num_axes = 0", "scale" is a scalar with shape length 1 and dimension size 1.
*@li If "scale_from_blob = true", "num_axes > 0, and "axis >= 0", "axis + num_axes" must be less than or equal to "n" and the ith axis of "scale" and the (i+"axis")th axis of "x" must have the same size (0 <= i < num_axes).\n
* If "axis < 0", "n + axis + num_axes" must be less than or equal to "n" and the ith axis of "scale" and the (i+n+"axis")th axis of "x" must have the same size (0 <= i < num_axes).
*@li If "scale_from_blob = false", "scale" is not a scalar, and "axis >= 0","axis + m" must be less than or equal to "n" and the ith axis of "scale" and the (i+"axis")th axis of "x" must have the same size (0 <= i < m).\n
* If "axis < 0", "n + axis + m" must be less than or equal to "n" and the ith axis of "scale" and the (i+n+"axis")th axis of "x" must have the same size (0 <= i < m).
*@li If "bias" is not None, the constraints for "bias" is the same as that for "scale".
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
*@brief Local Response Normalization.

*@par Inputs:
*One input, including:
*@li x: A Tensor. Must be 4-D shape, and only support the following types: float16, float32.

*@par Attributes:
* depth_radius = (local_size + 1) / 2. Defaults to "5".
*@li bias: An optional float32. An offset, usually > 0 to avoid dividing by 0.
* Defaults to "1".
*@li alpha: An optional float32. A scaling factor, usually positive.
* Defaults to "1".
*@li norm_region: An optional string. A mode option. "ACROSS_CHANNELS":0, "WITHIN_CHANNEL":1. Defaults to "ACROSS_CHANNELS".

*@par Outputs:
*y: A Tensor. Has the same data type and shape as "x".

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
* @brief Computes the gradient for Local Response Normalization.

* @par Inputs:
* @li grads: A 4D Tensor of type float16 or float32.
* @li x: A 4D Tensor of type float16 or float32.
* @li y: A 4D Tensor of type float16 or float32.

* @par Attributes:
* @li depth_radius: An optional int, specifying the half-width of the
* normalization window. Defaults to "5".
* @li bias: An optional float32. An offset, usually > 0 to avoid dividing by 0.
* Defaults to "1".
* @li alpha: An optional float32. A scaling factor, usually positive.
* Defaults to "1".
* @li beta: An optional float32. An exponent. Defaults to "0.5".

* @par Outputs:
* z: A Tensor. Has the same type and shape as "grads".

* @attention Constraints:
* "x" and "y" must have the same shape and type as "grads".

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
 *@brief Calculates the RNNT Loss (log probability) for each batch entry. \n
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
*@brief Performs group normalization.

*@par Inputs:\n
* Five inputs, including: (NHWC, NCHW supported)
*@li x: A 4D Tensor of type float16 or float32, with format NHWC or \n
NCHW for 4D.
*@li scale: A Tensor of type float32. Must be 1D if input "x" is with format \n
NHWC or NCHW. Specifies the scaling factor.
*@li offset: A Tensor of type float32. Must be 1D if input "x" is with \n
format NHWC or NCHW. Specifies the offset.
*@li mean: A Tensor of type float32. Must be 1D if input "x" is with format \n
NHWC or NCHW. Reserved. Mu
st be "None" if the operation is used for training.
*@li variance: A Tensor of type float32. Must be 1D if input "x" is with \n
format NHWC or NCHW. Specifies the variance used for inference. Reserved.

*@par Attributes:
*@li epsilon: An optional float32, specifying the small value added to \n
variance to avoid dividing by zero. Defaults to "0.0001".
*@li data_format: An optional string, specifying the format of "x". \n
Defaults to "NHWC".
*@li is_training: An optional bool, specifying if the operation is used for \n
training or inference. Defaults to "True".

*@par Outputs:\n
* Five outputs, including: (NHWC, NCHW supported)
*@li y: A 4D Tensor of type float16 or float32 for the normalized "x", \n
with format NHWC or NCHW for 4D.
*@li batch_mean: A Tensor of type float32. Must be 1D if input "x" is with \n
format NHWC or NCHW. Specifies the mean of "x".
*@li batch_variance: A Tensor of type float32. Must be 1D if input "x" is \n
with format NHWC or NCHW. Specifies the variance of "x".
*@li reserve_space_1: An optional Tensor of type float32. Must be 1D if \n
input "x" is with format NHWC or NCHW. Specifies the mean o
f "x" for gradient computation. Pass "None" to skip this output.
*@li reserve_space_2: An optional Tensor of type float32. Must be 1D if \n
input "x" is with format NHWC or NCHW. Specifies the varian
ce of "x" for gradient computation. Pass "None" to skip this output.

*@attention Constraints:
*@li If the operation is used for inference and outputs "reserve_space_1" \n
and "reserve_space_2" are available, then "reserve_space_1" has the same \n
value as "mean" and "reserve_spa
ce_2" has the same value as "variance".
*@li For Ascend 310, the result accuracy fails  due to the square root \n
instruction.

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

}  // namespace ge

#endif  //GE_OP_NN_NORM_OPS_H
