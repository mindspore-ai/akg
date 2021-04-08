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
 * \file nonlinear_fuc_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NONLINEAR_FUC_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NONLINEAR_FUC_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Computes the for the gelu of "x" . \n

*@par Inputs:
*Two inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Gelu
*/
REG_OP(Gelu)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(Gelu)

/**
*@brief Computes the gradient for the gelu of "x" . \n

*@par Inputs:
*Three inputs, including:
* @li dy: A Tensor. Must be one of the following types: float16, float32
* @li x: A Tensor of the same type as "dy".
* @li y: A Tensor of the same type as "dy" . \n

*@par Outputs:
*z: A Tensor. Has the same type as "dy".
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator GeluGrad
*/
REG_OP(GeluGrad)
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(z, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(GeluGrad)

/**
*@brief Computes the for the fast_gelu of "x" . \n

*@par Inputs:
*Two inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator FastGelu
*/
REG_OP(FastGelu)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(FastGelu)

/**
*@brief Computes the gradient for the fast_gelu of "x" . \n

*@par Inputs:
*Three inputs, including:
* @li dy: A Tensor. Must be one of the following types: float16, float32
* @li x: A Tensor of the same type as "dy" . \n

*@par Outputs:
*z: A Tensor. Has the same type as "dy".
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator FastGeluGrad
*/
REG_OP(FastGeluGrad)
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(z, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(FastGeluGrad)


/**
*@brief Computes the gradient for the tanh of "x" . \n

*@par Inputs:
*Two inputs, including:
* @li y: A Tensor. Must be one of the following types: float16, float32,
*     double, complex64, complex128.
* @li dy: A Tensor of the same type as "y" . \n

*@par Outputs:
*z: A Tensor. Has the same type as "y".
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator TanhGrad.
*/
REG_OP(TanhGrad)
    .INPUT(y, TensorType::UnaryDataType())
    .INPUT(dy, TensorType::UnaryDataType())
    .OUTPUT(z, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(TanhGrad)

/**
*@brief: Computes hyperbolic tangent of "x" element-wise . \n

*@par Inputs:
*One input:
*x: A Tensor. Must be one of the following types: float16, float32, complex64, complex128, double . \n

*@par Outputs:
*y: A Tensor. Has the same type as "x" . \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator Tanh.
*/
REG_OP(Tanh)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Tanh)

/**
* @brief Computes rectified linear: "max(x, 0)".
*
* @par Inputs:
* x: A tensor. Must be one of the following types: float32, float64, int32, uint8,
*     int16, int8, int64, uint16, float16, qint8.
*
* @par Outputs:
* y: A tensor. Has the same type as "x".
*
* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator Relu.
* @li Compatible with the Caffe operator ReLULayer.
*
*/
REG_OP(Relu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE,
                          DT_INT8, DT_INT32, DT_INT16, DT_INT64,
                          DT_UINT8, DT_UINT16, DT_QINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE,
                           DT_INT8, DT_INT32, DT_INT16, DT_INT64,
                           DT_UINT8, DT_UINT16, DT_QINT8}))
    .OP_END_FACTORY_REG(Relu)

/**
* @brief Computes rectified linear 6.
* activations = min(max(x, 0), 6) . \n

* @par Inputs:
* x: A Tensor of type RealNumberType . \n

* @par Outputs:
* y: A Tensor of type RealNumberType . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Relu6.
*/
REG_OP(Relu6)
    .INPUT(x, TensorType::RealNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .OP_END_FACTORY_REG(Relu6)

/**
* @brief Computes rectified linear 6*scale.
* activations = min(max(x, 0), 6*scale) . \n

* @par Inputs:
* x: A Tensor of type RealNumberType . \n

* @par Attributes:
* epsilon: A required scalar. The data type is float32 . \n

* @par Outputs:
* y: A Tensor of type RealNumberType . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Relu6.
*
*@par Restrictions:
*Warning: THIS FUNCTION IS DEPRECATED. Please use Relu6 instead.
*/
REG_OP(Relu6D)
    .INPUT(x, TensorType::RealNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .ATTR(scale, Float, 1.0)
    .OP_END_FACTORY_REG(Relu6D)

/**
* @brief Computes rectified linear 6 gradients for a Relu6 operation.
*     backprops = gradients * (features > 0) * (features < 6) . \n

* @par Inputs:
* @li features: A Tensor of type RealNumberType.
* @li gradients: A Tensor of type RealNumberType . \n

* @par Outputs:
* backprops: A Tensor of type RealNumberType . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Relu6Grad.
*/
REG_OP(Relu6Grad)
    .INPUT(gradients, TensorType::RealNumberType())
    .INPUT(features, TensorType::RealNumberType())
    .OUTPUT(backprops, TensorType::RealNumberType())
    .OP_END_FACTORY_REG(Relu6Grad)

/**
* @brief Compute sigmoid of "x" element-wise . \n

* @par Inputs:
* A Tensor of type complex64, complex128, float16, float32 or double . \n

* @par Outputs:
* A Tensor. Has the same type as "x" . \n

* @see Relu()

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Sigmoid.
*/
REG_OP(Sigmoid)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Sigmoid)

/**
* @brief Computes z = (y - y*y)*dy . \n

* @par Inputs:
* @li y: The input is Tensor, dtype is UnaryDataType.
* @li dy: The input is Tensor, dtype is UnaryDataType . \n

* @par Outputs:
* z: The shape of output, dtype is UnaryDataType.
*/
REG_OP(SigmoidGrad)
    .INPUT(y, TensorType(UnaryDataType))
    .INPUT(dy, TensorType(UnaryDataType))
    .OUTPUT(z, TensorType(UnaryDataType))
    .OP_END_FACTORY_REG(SigmoidGrad)

/**
*@brief Computes the binomial normal log likelihood (BNLL) output:
*if x>0, x+log(1+exp(-x)); otherwise log(1+exp(x)) . \n

*@par Inputs:
*x: A Tensor of type double, float16 or float32 . \n

*@par Outputs:
*y: A tensor. Has the same type and format as input "x" . \n

*@par Third-party framework compatibility
* Compatible with the Caffe operator BNLL.
*/
REG_OP(BNLL)
    .INPUT(x, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(BNLL)

/**
*@brief Computes softplus: log(exp(x) + 1) . \n

*@par Inputs:
* One input:
*x: A Tensor of type float16 or float32. Up to 8D . \n

*@par Outputs:
*y: The activations tensor. Has the same type and format as input "x"

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Softplus.
*/
REG_OP(Softplus)
    .INPUT(x, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(Softplus)

/**
*@brief Computes softplus gradients for a softplus operation . \n

*@par Inputs:
*Two inputs:
* @li gradients: An NC1HWC0 or ND Tensor of type float16 or float32.
* @li features: An NC1HWC0 or ND Tensor of type float16 or float32.


*@par Outputs:
*backprops: A Tensor. Has the same type and format as input "gradients" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SoftplusGrad.
*/
REG_OP(SoftplusGrad)
    .INPUT(gradients, TensorType::FloatingDataType())
    .INPUT(features, TensorType::FloatingDataType())
    .OUTPUT(backprops, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(SoftplusGrad)

/**
*@brief Computes softsign: x/(abs(x) + 1) . \n

*@par Inputs:
* One input:
*x: A Tensor of type float16 or float32. Up to 8D . \n

*@par Outputs:
*y: The activations tensor. Has the same type and format as "x"

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Softsign.
*/
REG_OP(Softsign)
    .INPUT(x, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(Softsign)

/**
*@brief Computes scaled exponential linear: scale * alpha * (exp(x) - 1) . \n

*@par Inputs:
* One input:
*x: A Tensor. Must be one of the following types: float16, float, double
 * int32, int8. format:ND, NC1HWC0 . \n

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x". format:ND, NC1HWC0 . \n

*@see Region()

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Selu.
*/
REG_OP(Selu)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE,
                                 DT_INT8,DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE,
                                     DT_INT8,DT_INT32}))
    .OP_END_FACTORY_REG(Selu)

/**
*@brief Computes rectified linear gradients for a ReLU operation . \n

*@par Inputs:
* Two inputs, including:
*@li gradients: A Tensor. Must be one of the following types: float32, double,
 * int32, int8, int16, int64, uint16, float16, uint32, uint64
*@li features: A Tensor. Must be one of the following types: float32, double,
 * int32, int8, int16, int64, uint16, float16, uint32, uint64

*@par Outputs:
*backprops: A Tensor. Must have the same type as"gradients" . \n

*@attention Constraints:
* The corresponding Relu operator needs to be called before using this operator on the network . \n

*@see Relu

*@par Third-party framework compatibility
* Compatible with TensorFlow operator ReluGrad.
*/
REG_OP(ReluGrad)
    .INPUT(gradients, TensorType::RealNumberType())
    .INPUT(features, TensorType::RealNumberType())
    .OUTPUT(backprops, TensorType::RealNumberType())
    .OP_END_FACTORY_REG(ReluGrad)

/**
*@brief Computes rectified linear gradients for a ReLU operation . \n

*@par Inputs:
* Two inputs, including:
*@li gradients: A Tensor. Must be one of the following types: float32, double, int32, int8, int16,  int8, int64, uint16, float16, uint32, uint64
*@li mask: A Tensor. Must be the following types: uint8

*@par Outputs:
*backprops: A Tensor. Must have the same type as"gradients" . \n

*@attention Constraints:
* The corresponding Relu operator needs to be called before using this operator on the network . \n

*@see Relu

*@par Third-party framework compatibility
* Compatible with TensorFlow operator ReluGradV2.
*/
REG_OP(ReluGradV2)
    .INPUT(gradients, TensorType::RealNumberType())
    .INPUT(mask, TensorType({DT_UINT8}))
    .OUTPUT(backprops, TensorType::RealNumberType())
    .OP_END_FACTORY_REG(ReluGradV2)

/**
*@brief Computes rectified linear: "max(x, 0)".
*
*@attention Constraints:
* The last dimension must be divisible by 8.
* The second output "mask" is "1" (for y >= 0) or "0" ( for y < 0).
*
*@par Inputs:
* x: A tensor. Must be one of the following types: float32, float64, int32, uint8,
*     int16, int8, int64, uint16, float16, qint8.
*
*@par Outputs:
*@li y: A tensor. Has the same type as "x".
*@li mask: A tensor of type uint8.
*
*@par Third-party framework compatibility
* Incompatible with TensorFlow or Caffe.
*
*/
REG_OP(ReluV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, DT_INT32, DT_INT16, DT_INT64, DT_UINT8, DT_UINT16, DT_QINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, DT_INT32, DT_INT16, DT_INT64, DT_UINT8, DT_UINT16, DT_QINT8}))
    .OUTPUT(mask, TensorType({DT_UINT8}))
    .OP_END_FACTORY_REG(ReluV2)

/**
*@brief Performs parametric ReLU . \n

*@par Inputs:
* Two inputs, including:
*@li x: A multi-dimensional Tensor of type float16 or float32.
*@li weight: A Scalar or 1D Tensor of type float16 or float32, specifying the weight, the initial value of "a". The number of dimensions must be the same as the number of channels . \n

*@par Outputs:
*y: An activated Tensor. Has the same dimensions with "x" . \n

*@par Third-party framework compatibility
* Compatible with PyTorch and Caffe operator PReLU.
*/
REG_OP(PRelu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(weight, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(PRelu)

/**
*@brief Performs the backpropagation of PRelu for training scenarios . \n

*@par Inputs:
* Three inputs, including:
*@li grads: Input gradient. Multi-dimensional Tensors are supported. The data type can be float16 or float32.
*@li features: A multi-dimensional Tensor of type float16 or float32.
*@li weights: A Scalar or 1D Tensor of type float16 or float32, specifying the weight. The number of dimensions must be the same as the number of channels . \n

*@par Outputs:
*@li dx: Reverse gradient of "features". Has the same dimensions and type as "features".
*@li da: Reverse gradient of "weight". Has the same dimensions and type as "features" . \n

*@par Third-party framework compatibility
* Compatible with PyTorch operator PReluGrad.
*/
REG_OP(PReluGrad)
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(features, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(weights, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(da, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(PReluGrad)

/**
*@brief Activation function fused from sigmoid and ReLU, with soft saturation
*    on the left and no saturation on the right . \n

*@par Inputs:
*x: A float16, float32 or double, for the input data type . \n

*@par Attributes:
*alpha: A float32. Defines at which negative value the ELU saturates. Defaults to "1.0" . \n

*@par Outputs:
*y: A float16, float32 or double, for the normalized result . \n

*@attention Constraints:
*@li The input is of type float16 or float32 . \n

*@par Multiple batches supported or not
*Supported
*@par Third-party framework compatibility
*@li Compatible with Tensorflow's Elu operator
*@li Compatible with Caffe's ELULayer operator
*
*@since V100R001C33
*/
REG_OP(Elu)
    .INPUT(x, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType::FloatingDataType())
    .ATTR(alpha, Float, 1.0)
    .OP_END_FACTORY_REG(Elu)

/**
*@brief Computes gradients for the exponential linear (Elu) operation.
*
*@par Inputs:
*@li grads: A tensor. Must be one of the following types: float16, float32, float64.
*     The backpropagated gradients to the corresponding Elu operation.
*@li activations: A tensor. Has the same type as "grads".
*     The outputs of the corresponding Elu operation.
*
*@par Outputs:
* y: A tensor. Has the same type as "grads".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator EluGrad.
*
*/
REG_OP(EluGrad)
    .INPUT(grads, TensorType::FloatingDataType())
    .INPUT(activations, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(EluGrad)

/**
*@brief Computes the output as x if x > 0 and negative_slope * x if x <= 0 . \n

*@par Inputs:
* One input:
* x: A Tensor. Must be one of the following types: float32, float16, double.
*
*@par Attributes:
*negative_slope: A float32. Defaults to "0.0".
*
*@par Outputs:
*y: A Tensor. Has the same type as "x".
*@par Third-party framework compatibility
* Compatible with the Caffe operator ReLU.
*/
REG_OP(LeakyRelu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE}))
    .ATTR(negative_slope, Float, 0.0)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE}))
    .OP_END_FACTORY_REG(LeakyRelu)

/**
*@brief Computes the output as gradients if features > 0 and negative_slope * gradients if features <= 0 . \n

*@par Inputs:
* Two inputs, including:
* @li gradients: A Tensor. Must be one of the following types: float16, float32, double.
* @li features: A Tensor. Has the same type as "gradients" . \n

*@par Attributes:
*negative_slope: A float32. Defaults to "0.0" . \n

*@par Outputs:
*backprops: A Tensor. Has the same type as "gradients" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator LeakyReluGrad.
*/
REG_OP(LeakyReluGrad)
    .INPUT(gradients, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(features, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(negative_slope, Float, 0.0)
    .OUTPUT(backprops, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(LeakyReluGrad)

/**
*@brief Thresholds grad each element of the input Tensor . \n

*@par Inputs:
* @li gradients: A Tensor shape and dtype of input gradients. Support float16, int32.
* @li features: A Tensor shape and dtype of input features. Support float16, int32 . \n

*@par Attributes:
*threshold: A float32 scale value to threshold at . \n

*@par Outputs:
*backprops: A Tensor of shape and dtype of output backprops, should be same shape and type as inputs . \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(ThresholdGradV2D)
    .INPUT(gradients, TensorType({DT_INT32, DT_FLOAT16}))
    .INPUT(features, TensorType({DT_INT32, DT_FLOAT16}))
    .OUTPUT(backprops, TensorType({DT_INT32, DT_FLOAT16}))
    .REQUIRED_ATTR(threshold, Float)
    .OP_END_FACTORY_REG(ThresholdGradV2D)

/**
*@brief Thresholds each element of the input Tensor y = (x > threshold) ? x : value . \n

*@par Inputs:
*x: A Tensor dtype of real number . \n

*@par Attributes:
*@li threshold: A float32 scale value to threshold at.
*@li value: A float32 scale value to replace with . \n

*@par Outputs:
*y: A Tensor of shape and dtype of output, should be same shape and type as input . \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(ThresholdV2D)
    .INPUT(x, TensorType::RealNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .REQUIRED_ATTR(threshold, Float)
    .REQUIRED_ATTR(value, Float)
    .OP_END_FACTORY_REG(ThresholdV2D)

/**
*@brief: Computes hyperbolic tangent of "x" element-wise . \n

*@par Inputs:
*One input:
*x: A Tensor. Must be one of the following types: float16, float32 . \n

*@par Outputs:
*y: A Tensor. Has the same type as "x" . \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator Mish.
*/

REG_OP(Mish)
    .INPUT(x, TensorType({ DT_FLOAT,DT_FLOAT16 }))
    .OUTPUT(y, TensorType({ DT_FLOAT,DT_FLOAT16 }))
    .OP_END_FACTORY_REG(Mish)

/**
 * @brief pytorch hardtanh_backward operator.
 *
 * @par Inputs:
 * 2 inputs, including:
 * @li result, minimum tensor of the linear region range,
 * datatype: float16/float32, format:ND/5HD.
 * @li grad, maximum tensor of the linear region range,
 * datatype:float16/float32, format:ND/5HD. \n

 * @par Attributes:
 * 2 attributes, including:
 * @li min_val, minimum value of the linear region range, datatype:float.
 * @li max_val, maximum value of the linear region range, datatype:float. \n

 * @par Outputs:
 * 1 output, including:
 * @li y, hardtanh_backward output tensor, datatype and format is same as
 * input result. \n

 * @attention Constraints:
 * This operator only supports dataType: float16/float32, format: ND/5HD. \n

 * @par Third-party framework compatibility
 * Compatible with the Pytorch operator HardtanhGrad.
 */
REG_OP(HardtanhGrad)
    .INPUT(result, TensorType({ DT_FLOAT16, DT_FLOAT })) /* "First operand." */
    .INPUT(grad, TensorType({ DT_FLOAT16, DT_FLOAT }))   /* "Second operand." */
    .OUTPUT(y, TensorType({ DT_FLOAT16, DT_FLOAT }))     /* "Result, has same element type as two inputs" */
    .ATTR(min_val, Float, -1.0)
    .ATTR(max_val, Float, 1.0)
    .OP_END_FACTORY_REG(HardtanhGrad)

/**
* @brief Calculates the softplus loss function with attributes of beta and threshold. \n

* @par Inputs:
* One inputs, including:
* @li x: A mutable Tensor. Must be one of the following types:
*     float16, float32. \n

* @par Attributes:
* @li beta: An optional float. Defaults to "1.0" \n

* @li threshold: An optional float. Defaults to "20.0" \n

* @par Outputs:
* @li y: A mutable Tensor. Has the same type as "x" \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator Softplus.
*/
REG_OP(SoftplusV2)
    .INPUT(x, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .OUTPUT(y, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .ATTR(beta, Float, 1.0)
    .ATTR(threshold, Float, 20.0)
    .OP_END_FACTORY_REG(SoftplusV2)

/**
* @brief Calculates the reversed outputs of the function "softplus_v2". \n

* @par Inputs:
* Two inputs, including:
* @li input_gradients: A mutable Tensor. Must be one of the following types:
*     float16, float32.
* @li input_features: A mutable Tensor of the same type as "input_gradients" \n

* @par Attributes:
* @li beta: An optional float. Defaults to "1.0" \n

* @li threshold: An optional float. Defaults to "20.0" \n

* @par Outputs:
* @li output_backprops: A mutable Tensor. Has the same type as "input_gradients" \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator SoftplusGrad.
*/
REG_OP(SoftplusV2Grad)
    .INPUT(input_gradients, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .INPUT(input_features, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .OUTPUT(output_backprops, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .ATTR(beta, Float, 1.0)
    .ATTR(threshold, Float, 20.0)
    .OP_END_FACTORY_REG(SoftplusV2Grad)

/**
 * @brief ThresholdedRelu takes one input data (Tensor) and produces one output data (Tensor)
 *  where the rectified linear function, y = x for x > alpha, y = 0 otherwise, is applied to the tensor elementwise.
 * 
 * @par inputs
 * one input including:
 * @li x: input A Tensor. Must be one of the following types: float32, float16
 * 
 * @par output
 * one output including:
 * @li y:A Tensor of the same type as x
 * 
 */
REG_OP(ThresholdedRelu)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(alpha, Float, 1.0)
    .OP_END_FACTORY_REG(ThresholdedRelu)

/**
* @brief Calculate the hard shrinkage function. \n

* @par Inputs:
* One inputs, including:
* @li input_x: A tensor. Must be one of the following types:
*     float16, float32. \n

* @par Attributes:
* @li lambd: An optional float. Defaults to 0.5. \n

* @par Outputs:
* y: A Tensor with the same dtype and shape of input_x's. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator Hardshrink. \n
*/
REG_OP(HardShrink)
    .INPUT(input_x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(output_y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(lambd, Float, 0.5)
    .OP_END_FACTORY_REG(HardShrink)

/**
* @brief Calculate the hard sigmoid function. \n

* @par Inputs:
* One inputs, including:
* @li input_x: A tensor. Must be one of the following types:
*     float16, float32, int32. \n

* @par Attributes:
* @li alpha: An optional float. Defaults to 0.16666666. \n
* @li beta: An optional float. Defaults to 0.5. \n

* @par Outputs:
* y: A Tensor with the same dtype and shape of input_x's. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator Hardsigmoid. \n
*/    
REG_OP(HardSigmoid)
    .INPUT(input_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(output_y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(alpha, Float, 0.16666666)
    .ATTR(beta, Float, 0.5)
    .OP_END_FACTORY_REG(HardSigmoid)

/**
* @brief Calculate the soft shrinkage function. \n

* @par Inputs:
* One inputs, including:
* @li input_x: A tensor. Must be one of the following types:
*     float16, float32. \n

* @par Attributes:
* @li lambd: An optional float. Defaults to 0.5. \n

* @par Outputs:
* y: A Tensor with the same dtype and shape of input_x's. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator Softshrink. \n
*/
REG_OP(SoftShrink)
     .INPUT(input_x, TensorType({DT_FLOAT16, DT_FLOAT}))
     .OUTPUT(output_y, TensorType({DT_FLOAT16, DT_FLOAT}))
     .ATTR(lambd, Float, 0.5)
     .OP_END_FACTORY_REG(SoftShrink)

/**
* @brief Calculate the reversed outputs of the function "soft_shrink". \n

* @par Inputs:
* Two inputs, including:
* @li input_grad: A tensor. Must be one of the following types:
*     float16, float32. \n
* @li input_x: A tensor of the same dtype as "input_grad". \n

* @par Attributes:
* @li lambd: An optional float. Defaults to 0.5. \n

* @par Outputs:
* y: A Tensor of the same dtype and shape as "input_graxd". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator SoftShrinkGrad. \n
*/
REG_OP(SoftShrinkGrad)
     .INPUT(input_grad, TensorType({DT_FLOAT16, DT_FLOAT}))
     .INPUT(input_x, TensorType({DT_FLOAT16, DT_FLOAT}))
     .OUTPUT(output_y, TensorType({DT_FLOAT16, DT_FLOAT}))
     .ATTR(lambd, Float, 0.5)
     .OP_END_FACTORY_REG(SoftShrinkGrad)
} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_NONLINEAR_FUC_OPS_H_
