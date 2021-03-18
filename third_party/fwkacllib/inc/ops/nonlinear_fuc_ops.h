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

#ifndef GE_OP_NONLINEAR_FUC_OPS_H
#define GE_OP_NONLINEAR_FUC_OPS_H

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Computes the for the gelu of "x".

*@par Inputs:
*Two inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*/
REG_OP(Gelu)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(Gelu)

/**
*@brief Computes the gradient for the gelu of "x".

*@par Inputs:
*Two inputs, including:
* @li dy: A Tensor. Must be one of the following types: float16, float32
* @li x: A Tensor of the same type as "dy".
* @li y: A Tensor of the same type as "dy".

*@par Outputs:
*z: A Tensor. Has the same type as "dy".
*/
REG_OP(GeluGrad)
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(z, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(GeluGrad)

/**
*@brief Computes the gradient for the tanh of "x".

*@par Inputs:
*Two inputs, including:
* @li y: A Tensor. Must be one of the following types: float16, float32,
*     double, complex64, complex128.
* @li dy: A Tensor of the same type as "y".

*@par Outputs:
*z: A Tensor. Has the same type as "y".
*/
REG_OP(TanhGrad)
    .INPUT(y, TensorType::UnaryDataType())
    .INPUT(dy, TensorType::UnaryDataType())
    .OUTPUT(z, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(TanhGrad)

/**
*@brief: Computes hyperbolic tangent of "x" element-wise.

*@par Inputs:
*One input:
*x: A Tensor. Must be one of the following types: float16, float32, double, complex64, complex128, int32, int64

*@par Outputs:
*y: A Tensor. Has the same type as "x".

*/
REG_OP(Tanh)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Tanh)

/**
* @brief Computes rectified linear: "max(x, 0)".
*
* @par Inputs:
* x: A tensor. Must be one of the following types: float32, float64, int32, uint8,\n
*     int16, int8, int64, uint16, float16, qint8.
*
* @par Outputs:
* y: A tensor. Has the same type as "x".
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
* activations = min(max(x, 0), 6).

* @par Inputs:
* x: A Tensor of type RealNumberType.

* @par Outputs:
* y: A Tensor of type RealNumberType.
*/
REG_OP(Relu6)
    .INPUT(x, TensorType::RealNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .OP_END_FACTORY_REG(Relu6)

/**
* @brief Computes rectified linear 6 gradients for a Relu6 operation.
*     backprops = gradients * (features > 0) * (features < 6).

* @par Inputs:
* @li features: A Tensor of type RealNumberType.
* @li gradients: A Tensor of type RealNumberType.

* @par Outputs:
* backprops: A Tensor of type RealNumberType.
*/
REG_OP(Relu6Grad)
    .INPUT(gradients, TensorType::RealNumberType())
    .INPUT(features, TensorType::RealNumberType())
    .OUTPUT(backprops, TensorType::RealNumberType())
    .OP_END_FACTORY_REG(Relu6Grad)

/**
* @brief Compute sigmoid of "x" element-wise.

* @par Inputs:
* A Tensor of type UnaryDataType.

* @par Outputs:
* A Tensor. Has the same type as "x".

* @see Relu()
*/
REG_OP(Sigmoid)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Sigmoid)

/**
* @brief Computes z = (y - y*y)*dy.

* @par Inputs:
* @li y: the input is tensor , dtype is UnaryDataType.
* @li dy the input is tensor , dtype is UnaryDataType.

* @par Outputs:
* z: the shape of output, dtype is UnaryDataType.
*/
REG_OP(SigmoidGrad)
    .INPUT(y, TensorType(UnaryDataType))
    .INPUT(dy, TensorType(UnaryDataType))
    .OUTPUT(z, TensorType(UnaryDataType))
    .OP_END_FACTORY_REG(SigmoidGrad)

/**
*@brief Computes the binomial normal log likelihood (BNLL) output:\n
*if x>0, x+log(1+exp(-x)); otherwise log(1+exp(x)).

*@par Inputs:
*x: A Tensor of type float16 or float32.

*@par Outputs:
*y: A tensor. Has the same type and format as input "x".

*/
REG_OP(BNLL)
    .INPUT(x, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(BNLL)

/**
*@brief Computes softplus: log(exp(x) + 1).

*@par Inputs:
* One input:\n
*x: A Tensor of type float16 or float32. Up to 8D.

*@par Outputs:
*y: The activations tensor. Has the same type and format as input "x"

*/
REG_OP(Softplus)
    .INPUT(x, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(Softplus)

/**
*@brief Computes softplus gradients for a softplus operation.

*@par Inputs:
*Two inputs:
* @li gradients: An NC1HWC0 or ND Tensor of type float16 or float32.
* @li features: An NC1HWC0 or ND Tensor of type float16 or float32.


*@par Outputs:
*backprops: A Tensor. Has the same type and format as input "gradients".

*/
REG_OP(SoftplusGrad)
    .INPUT(gradients, TensorType::FloatingDataType())
    .INPUT(features, TensorType::FloatingDataType())
    .OUTPUT(backprops, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(SoftplusGrad)

/**
*@brief Computes softsign: x/(abs(x) + 1).

*@par Inputs:
* One input:\n
*x: A Tensor of type float16 or float32. Up to 8D.

*@par Outputs:
*y: The activations tensor. Has the same type and format as "x"

*/
REG_OP(Softsign)
    .INPUT(x, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(Softsign)

/**
*@brief Computes scaled exponential linear: scale * alpha * (exp(x) - 1).

*@par Inputs:
* One input: \n
*x: A Tensor. Must be one of the following types: float16, float32, int32, int8.

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x".

*@see Region()

*/
REG_OP(Selu)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE,
                                 DT_INT8,DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE,
                                     DT_INT8,DT_INT32}))
    .OP_END_FACTORY_REG(Selu)

/**
*@brief Computes rectified linear gradients for a ReLU operation.

*@par Inputs:
* Two inputs, including:
*@li gradients: A Tensor. Must be one of the following types: float32, double, int32, int8, int16, int8, int64, uint16, float16, uint32, uint64
*@li features: A Tensor. Must be one of the following types: float32, double, int32, int8, int16, int8, int64, uint16, float16, uint32, uint64

*@par Outputs:
*backprops: A Tensor. Must have the same type as"gradients".

*@attention Constraints:
* The corresponding Relu operator needs to be called before using this operator on the network.

*@see Relu

*/
REG_OP(ReluGrad)
    .INPUT(gradients, TensorType::RealNumberType())
    .INPUT(features, TensorType::RealNumberType())
    .OUTPUT(backprops, TensorType::RealNumberType())
    .OP_END_FACTORY_REG(ReluGrad)

/**
*@brief Computes rectified linear gradients for a ReLU operation.

*@par Inputs:
* Two inputs, including:
*@li gradients: A Tensor. Must be one of the following types: float32, double, int32, int8, int16,\n int8, int64, uint16, float16, uint32, uint64
*@li mask: A Tensor. Must be the following types: uint8

*@par Outputs:
*backprops: A Tensor. Must have the same type as"gradients".

*@attention Constraints:
* The corresponding Relu operator needs to be called before using this operator on the network.

*@see Relu
*/
REG_OP(ReluGradV2)
    .INPUT(gradients, TensorType::RealNumberType())
    .INPUT(mask, TensorType({DT_UINT8}))
    .OUTPUT(backprops, TensorType::RealNumberType())
    .OP_END_FACTORY_REG(ReluGradV2)

/**
*@brief Computes rectified linear: "max(x, 0)".
*
*@attention Constraints:\n
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
*/
REG_OP(ReluV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, DT_INT32, DT_INT16, DT_INT64, DT_UINT8, DT_UINT16, DT_QINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, DT_INT32, DT_INT16, DT_INT64, DT_UINT8, DT_UINT16, DT_QINT8}))
    .OUTPUT(mask, TensorType({DT_UINT8}))
    .OP_END_FACTORY_REG(ReluV2)

/**
*@brief Performs parametric ReLU.

*@par Inputs:
* Two inputs, including: \n
*@li x: A multi-dimensional Tensor of type float16 or float32.
*@li weight: A Scalar or 1D Tensor of type float16 or float32, specifying the weight, the initial value of "a". The number of dimensions must be the same as the number of channels.

*@par Outputs:
*y: An activated Tensor. Has the same dimensions with "x".

*/
REG_OP(PRelu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(weight, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(PRelu)

/**
*@brief Performs the backpropagation of PRelu for training scenarios.

*@par Inputs:
* Three inputs, including: \n
*@li grads: Input gradient. Multi-dimensional Tensors are supported. The data type can be float16 or float32.
*@li features: A multi-dimensional Tensor of type float16 or float32.
*@li weights: A Scalar or 1D Tensor of type float16 or float32, specifying the weight. The number of dimensions must be the same as the number of channels.

*@par Outputs:
*@li dx: Reverse gradient of "features". Has the same dimensions and type as "features".
*@li da: Reverse gradient of "weight". Has the same dimensions and type as "features".

*/
REG_OP(PReluGrad)
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(features, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(weights, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(da, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(PReluGrad)

/**
*@brief Activation function fused from sigmoid and ReLU, with soft saturation on the left and no saturation on the right.

*@par Inputs:
*x: A float16 or float32, for the input data type.

*@par Attributes:
*alpha: A float. Defines at which negative value the ELU saturates. Defaults to "1.0".

*@par Outputs:
*y: A float16 or float32, for the normalized result.

*@attention Constraints:
*@li The input is of type float16 or float32.

*@par Multiple batches supported or not
*Supported
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
*/
REG_OP(EluGrad)
    .INPUT(grads, TensorType::FloatingDataType())
    .INPUT(activations, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(EluGrad)

/**
*@brief Computes the output as x if x > 0 and negative_slope * x if x <= 0.

*@par Inputs:
* One input:
* x: A Tensor. Must be one of the following types: float32, float16, double.
*
*@par Attributes:
*negative_slope: A float32. Defaults to "0.0".
*
*@par Outputs:
*y: A Tensor. Has the same type as "x".
*/
REG_OP(LeakyRelu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE}))
    .ATTR(negative_slope, Float, 0.0)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE}))
    .OP_END_FACTORY_REG(LeakyRelu)

/**
*@brief Computes the output as gradients if features > 0 and negative_slope * gradients if features <= 0.

*@par Inputs:
* Two inputs, including:
* @li gradients: A Tensor. Must be one of the following types: float16, float32, double.
* @li features: A Tensor. Has the same type as "gradients".

*@par Attributes:
*negative_slope: A float32. Defaults to "0.0".

*@par Outputs:
*backprops: A Tensor. Has the same type as "gradients".
*/
REG_OP(LeakyReluGrad)
    .INPUT(gradients, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(features, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(negative_slope, Float, 0.0)
    .OUTPUT(backprops, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(LeakyReluGrad)

REG_OP(threshold_grad_v2_d)
    .INPUT(input_x, TensorType({DT_INT32, DT_FLOAT16}))
    .INPUT(input_y, TensorType({DT_INT32, DT_FLOAT16}))
    .OUTPUT(output_z, TensorType({DT_INT32, DT_FLOAT16}))
    .OP_END_FACTORY_REG(threshold_grad_v2_d)

REG_OP(ThresholdV2D)
    .INPUT(x, TensorType::RealNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .OP_END_FACTORY_REG(ThresholdV2D)

} // namespace ge

#endif // GE_OP_NONLINEAR_FUC_OPS_H
