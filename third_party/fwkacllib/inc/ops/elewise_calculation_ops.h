/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file elewise_calculation_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_ELEWISE_CALCULATION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_ELEWISE_CALCULATION_OPS_H_
#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Adds all input tensors element-wise. \n

*@par Inputs:
*Dynamic inputs, including:
*x: A list of Tensor objects, each with same shape and type. The supported types are:
*   float16, float32, double, int32, uint8, int16, int8, complex64, int64,
*   qint8, quint8, qint32, uint16, complex128, uint32, uint64. It's a dynamic input. \n

*@par Attributes:
*N: An required attribute of type int32, means nums of inputs. \n

*@par Outputs:
*y: A Tensor. Has the same shape and type as the elements of "x". \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator AddN.
*/
REG_OP(AddN)
    .DYNAMIC_INPUT(x, TensorType({NumberType(), DT_VARIANT}))
    .OUTPUT(y, TensorType({NumberType(), DT_VARIANT}))
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(AddN)

/**
* @brief Calculates the reversed outputs of the function "maximum".

* @par Inputs:
* Three inputs, including:
* @li grads: A mutable Tensor. Must be one of the following types:
* float16, float32, int32.
* @li x1: A mutable Tensor of the same type as "grads".
* @li x2: A mutable Tensor of the same type as "grads". \n

* @par Attributes:
* @li grad_x: An optional bool. Defaults to "True".
* If "True", "y1" will be output.
* If "False", "y1" will not be output. \n

* @li grad_y: An optional bool. Defaults to "True".
* If "True", "y2" will be output.
* If "False", "y2" will not be output. \n

* @par Outputs:
* @li y1: A mutable Tensor. Has the same type as "grads".
* @li y2: A mutable Tensor. Has the same type as "grads". \n

* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator MaximumGrad.
*/
REG_OP(MaximumGrad)
    .INPUT(grads, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .ATTR(grad_x, Bool, true)
    .ATTR(grad_y, Bool, true)
    .OP_END_FACTORY_REG(MaximumGrad)

/**
* @brief Calculates the reversed outputs of the function "minimum".

* @par Inputs:
* Three inputs, including:
* @li grads: A mutable Tensor. Must be one of the following types:
* float16, float32, int32.
* @li x1: A mutable Tensor of the same type as "grads".
* @li x2: A mutable Tensor of the same type as "grads". \n

* @par Attributes:
* @li grad_x: An optional bool. Defaults to "True".
* If "True", "y1" will be output.
* If "False", "y1" will not be output. \n

* @li grad_y: An optional bool. Defaults to "True".
* If "True", "y2" will be output.
* If "False", "y2" will not be output. \n

* @par Outputs:
* @li y1: A mutable Tensor. Has the same type as "grads".
* @li y2: A mutable Tensor. Has the same type as "grads". \n

* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator MinimumGrad.
*/
REG_OP(MinimumGrad)
    .INPUT(grads, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .ATTR(grad_x, Bool, true)
    .ATTR(grad_y, Bool, true)
    .OP_END_FACTORY_REG(MinimumGrad)

/**
*@brief Cast a tensor form src data type to dst data type. \n

*@par Inputs:
*One input:
*x:A Tensor. Must be one of the following types: bool, float16, float, int8, int32, uint32, uint8,
   int64, uint64, int16, uint16, double, complex64, complex128, qint8, quint8, qint16, quint16, qint32, uint1.
   For float32 type, the actual calculation on the chip is based on float16.  \n

*@par Attributes:
*dst_type: An required attribute of type int32, specifying the dst data type. \n

*@par Outputs:
*y:A Tensor with same shape as x, and data type is specified by dst_type.
*/
REG_OP(Cast)
    .INPUT(x, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8,
                          DT_INT64, DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64,
                          DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32, DT_BF16, DT_UINT1}))
    .OUTPUT(y, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8,
                           DT_INT64, DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64,
                           DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32, DT_BF16}))
    .REQUIRED_ATTR(dst_type, Int)
    .OP_END_FACTORY_REG(Cast)

/**
*@brief Returns the truth value of (x1 >= x2) element-wise. \n
*when input is int32 and (x2 - x1) > 2**31 or < -2**31
*aicore accuracy is not guaranteed \n

*@par Inputs:
*Two inputs, including:
* @li x1: A Tensor. Must be one of the following types: float16, float32,
*    double, int32, int8, uint8, int64, uint16, uint32, uint64.
* @li x2: A Tensor of the same type as "x1". \n

*@par Outputs:
*y: A Tensor. Has the same type as "x1". \n

*@par Third-party framework compatibility:
* Compatible with the TensorFlow operator GreaterEqual.
*/
REG_OP(GreaterEqual)
    .INPUT(x1, TensorType::RealNumberType())
    .INPUT(x2, TensorType::RealNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(GreaterEqual)

/**
*@brief Returns the truth value of (x1 < x2) element-wise. \n
*when input is int32 and (x2 - x1) > 2**31 or < -2**31
*aicore accuracy is not guaranteed \n

*@par Inputs:
*Two inputs, including:
* @li x1: A Tensor. Must be one of the following types: float16, float32, double, int32,
*     uint8, int16, int8, int64, uint16, uint32, uint64.
* @li x2: A Tensor with the same type as "x1". \n

*@par Outputs:
*y: A Tensor of type bool. \n

*@par Third-party framework compatibility:
* Compatible with TensorFlow operator Less.
*/
REG_OP(Less)
    .INPUT(x1, TensorType::RealNumberType())
    .INPUT(x2, TensorType::RealNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(Less)

/**
*@brief Returns x1/x2 element-wise for real types. \n

*@par Inputs:
* Two inputs, including:
*@li x1: A Tensor. Must be one of the following types: float16, float32, double, uint16,
         int8, uint8, int16, int32, int64, complex64, DT_COMPLEX128.
*@li x2: A Tensor. Must be one of the following types: float16, float32, double, uint16,
         int8, uint8, int16, int32, int64, complex64, DT_COMPLEX128. \n

*@par Outputs:
* y: A Tensor. Has the same type and format as input "x1". \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator RealDiv.
*/
REG_OP(RealDiv)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DT_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DT_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DT_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(RealDiv)

/**
*@brief Computes square root of x element-wise. \n

*@par Inputs:
*  x: A Tensor. Must be one of the following types: float16, float32, complex128, complex64, float64. \n

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Sqrt.
*/
REG_OP(Sqrt)
    .INPUT(x, TensorType{(DT_FLOAT. DT_FLOAT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128)})
    .OUTPUT(y, TensorType{(DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128)})
    .OP_END_FACTORY_REG(Sqrt)

/**
*@brief Returns the max of "x" and "y" (i.e. x > y ? x: y) element-wise. \n

*@par Inputs:
*Two inputs, including:
* @li x1: A Tensor. Must be one of the following types: float16, float32, double, int32, int64.
* @li x2: A Tensor of the same type as "x1". \n

*@par Outputs:
*y: A Tensor. Has the same type as "x1". \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Maximum.
*/
REG_OP(Maximum)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32,
                           DT_INT64}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32,
                           DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32,
                           DT_INT64}))
    .OP_END_FACTORY_REG(Maximum)

/**
*@brief Returns the min of x and y (i.e. x1 < x2 ? x1 : x2) element-wise. \n

*@par Inputs:
*Two inputs, include:
* @li x1: A Tensor. Must be one of the following types: float32, float16, double, int32, int64.
* @li x2: A Tensor of the same type as "x1". \n

*@par Outputs:
*y: A Tensor of the same type as "x1". \n

*@par Third-party framework compatibility:
* Compatible with the TensorFlow operator Minimum.
*/
REG_OP(Minimum)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT32,
                           DT_INT64}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT32,
                           DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT32,
                           DT_INT64}))
    .OP_END_FACTORY_REG(Minimum)

/**
*@brief: Computes the reciprocal of "x". \n

*@par Inputs:
*One inputs, include:
*x:A Tensor of type float16, float32, int32, int64, double,
*     complex64, complex128.the format can be [NCHW,NHWC,ND]

*@par Outputs:
*y:A Tensor with same type as "x". \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Reciprocal.
*/
REG_OP(Reciprocal)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_FLOAT16,
                          DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_FLOAT16
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Reciprocal)

/**
*@brief Returns x - y element-wise.
*@par Inputs:
*Two inputs, including:
* @li x1: A Tensor. Must be one of the following types: int8, int16, int32, int64, uint8, float64,
*     float16, float32, complex128, complex64, uint16.
* @li x2: A Tensor of the same type as "x1". \n

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Subtract.
*/
REG_OP(Sub)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DT_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DT_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DT_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Sub)

/**
*@brief computes the absolute value of a tensor. \n

*@par Inputs:
*One input, including: \n
*x: A Tensor. Must be one of the following types: float16, float32, double, int8, int16, int32, int64. \n

*@par Outputs:
*y: A Tensor. Has the same type as "x". \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Abs.
*/
REG_OP(Abs)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16,
                          DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16,
                           DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(Abs)

/**
*@brief Computes gradients for absolute operation. \n

*
*@par Inputs:
*@li y: A tensor of type float16 or float32.
*@li dy: A tensor of the same type as "y".
*
*@attention Constraints:
* "dy" has the same type as "y".
*
*@par Outputs:
* z: A tensor. Has the same type as "y".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator AbsGrad.
*
*/
REG_OP(AbsGrad)
    .INPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(z, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(AbsGrad)

/**
*@brief: Computes the sign  of "x". \n

*@par Inputs:
*x:An ND Tensor of type float16, float32, int32, int64, double,
*     complex64, complex128. \n

*@par Outputs:
*y:An ND Tensor with same type as "x". \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Sign.
*/
REG_OP(Sign)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT, DT_DOUBLE, DT_INT32,
                          DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32,
                           DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Sign)

/**
*@brief Returns (x1 - x2)(x1 - x2) element-wise. \n

*@par Inputs:
*Two inputs, including: \n
*@li x1: A Tensor. Must be one of the following types: float16, float32, float64, int32, int64, complex64,complex128
*@li x2: A Tensor. Has the same type as "x1". \n

*@par Outputs:
*y: A Tensor. Has the same type as "x1". \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator SquaredDifference.
*/
REG_OP(SquaredDifference)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32,
                           DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32,
                           DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32,
                           DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(SquaredDifference)

/**
*@brief Computes cosine of "x" element-wise. \n

*@par Inputs:
*x: A Tensor of type float16, float32, double, complex64, complex128.
* the format can be [NCHW,NHWC,ND]

*@par Outputs:
*y: A Tensor of the same type as "x". \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Cos. \n

*/
REG_OP(Cos)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Cos)

/**
*@brief Returns x1/x2 element-wise. \n

*@par Inputs:
* Two inputs, including:
*@li x1: A Tensor. Must be one of the following types:
*    float16, float32, int32, int8, uint8, float64, int64, uint16, int16,
*    complex64, complex128, the format can be [NCHW,NHWC,ND].
*@li x2: A Tensor. Has the same type and format as input "x1". \n

*@par Outputs:
* y: A Tensor. Has the same type and format as input "x1". \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Div.
*/
REG_OP(Div)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT32,
                           DT_DOUBLE, DT_INT64, DT_UINT16, DT_INT16,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT32,
                           DT_DOUBLE, DT_INT64, DT_UINT16, DT_INT16,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT32,
                           DT_DOUBLE, DT_INT64, DT_UINT16, DT_INT16,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Div)

/**
*@brief: Returns the truth value of (x = y) element-wise. \n

*@par Inputs:
* Two inputs, including:
*@li x1: A Tensor. Must be one of the following types:
*    float16, float32, int32, int8, uint8, double, int16, int64, complex64,
*    complex128, quint8, qint8, qint32, string, bool. the format can be
*    [NCHW, NHWC, ND]
*@li x2: A Tensor of the same type and format as "x1". \n

*@par Outputs:
*y: A Tensor of type bool. \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Equal.
*/
REG_OP(Equal)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_UINT8,
                           DT_DOUBLE, DT_INT16, DT_INT64, DT_COMPLEX64,
                           DT_COMPLEX128, DT_QUINT8, DT_QINT8, DT_QINT32,
                           DT_STRING, DT_BOOL}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_UINT8,
                           DT_DOUBLE, DT_INT16, DT_INT64, DT_COMPLEX64,
                           DT_COMPLEX128, DT_QUINT8, DT_QINT8, DT_QINT32,
                           DT_STRING, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(Equal)

/**
*@brief Computes the exponential of "x" element-wise. \n

*@par Inputs:
*One input:\n
*x: A Tensor. Must be one of the following types: float16, float32, double, complex64, complex128. \n

*@par Attributes:
*@li base: An optional attribute of type float32, specifying the base gamma. Defaults to "-1.0".
*@li scale: An optional attribute of type float32, specifying the scale alpha. Defaults to "1.0".
*@li shift: An optional attribute of type float32, specifying the shift beta. Defaults to "0.0". \n

*@par Outputs:
*y: A Tensor of the same type as "x". \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator Exp.
*/
REG_OP(Exp)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .ATTR(base, Float, -1.0)
    .ATTR(scale, Float, 1.0)
    .ATTR(shift, Float, 0.0)
    .OP_END_FACTORY_REG(Exp)

/**
*@brief Computes the exp(x) - 1 element-wise, y = e^x - 1. \n

*@par Inputs:
*One input:
*x: A Tensor. Must be one of the following types: float16, float32, double, complex64, complex128. \n

*@par Outputs:
*y: A Tensor of the same type as "x". \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator Expm1.
*/
REG_OP(Expm1)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Expm1)

/**
* @brief Computes the expint(x). \n

* @par Inputs:
* One input:
* x: A Tensor. Must be one of the following types: bfloat16, half, float32, double. \n

* @par Outputs:
* y: A Tensor of the same type as "x". \n

* @par Third-party framework compatibility
* Compatible with TensorFlow operator Expint.
*/
REG_OP(Expint)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(Expint)

/**
* @brief: Computes the reciprocal of "x".

* @par Inputs:
* x: A Tensor. Must be one of the following types: float16, float32,
* int32, int64, double, complex64, complex128. \n

* @par Outputs:
* y: A Tensor. Must be one of the following type: float16, float32, int32. \n

* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator Inv.
*/
REG_OP(Inv)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE,DT_INT32,DT_INT64,DT_COMPLEX64,DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .OP_END_FACTORY_REG(Inv)

/**
* @brief: Computes "x" reciprocal grad, dx = -1*dy*y*y, where, "y = 1/x",
* and "dy" is the corresponding input gradient.

* @par Inputs:
* Two inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32,
* int32, int8.
* @li grad: A Tensor. Has the same type as "x". \n

* @par Outputs:
* y: A Tensor, Has the same type as "x". \n

* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator InvGrad.
*/
REG_OP(InvGrad)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8}))
    .INPUT(grad, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8}))
    .OP_END_FACTORY_REG(InvGrad)

/**
*@brief: Returns the truth value of (x <= y) element-wise. \n
*when input is int32 and (x2 - x1) > 2**31 or < -2**31
*aicore accuracy is not guaranteed \n

*@par Inputs:
* Two inputs, including:
*@li x1: A Tensor. Must be one of the following types: float32, float64,
* int32, uint8, int16, int8, int64, qint8, quint8, qint32, uint16,
* float16, uint32, uint64.
*@li x2: A Tensor of the same type as "x1". \n

*@par Outputs:
*y: A Tensor of type bool. \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator LessEqual.
*/
REG_OP(LessEqual)
    .INPUT(x1, TensorType::RealNumberType())
    .INPUT(x2, TensorType::RealNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(LessEqual)

/**
*@brief Computes the logarithm of (x + 1) element-wise, y = ln(x + 1). \n

*@par Inputs:
*One input:\n
*x: A Tensor. Must be one of the following types: float16, float32, double, complex64, complex128. \n

*@par Outputs:
*y: A Tensor of the same type as "x". \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator Log1p.
*/
REG_OP(Log1p)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Log1p)

/**
* @brief Returns element-wise remainder of division.

* @par Inputs:
* Two inputs, including:
* @li x1: A Tensor. Must be one of the following types: float16, float32,
* int32, int64, int8, uint8, double.
* @li x2: A Tensor of the same type as "x1". \n

* @par Outputs:
* y: A Tensor. Has the same type as "x1". \n

* @attention Constraints:
* @li x2: The input data does not support 0.
* @li When NUM exceeds 2048 , the accuracy of operator cannot guarantee the
* requirement of double thousandths in the mini form.
* @li Due to different architectures, the calculation results of this operator
* on NPU and CPU may be inconsistent.
* @li If shape is expressed as (D1,D2... ,Dn),
* then D1*D2... *DN<=1000000,n<=8. \n

* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator Mod.
*/
REG_OP(Mod)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8,
                           DT_INT64, DT_DOUBLE}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8,
                           DT_INT64, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8,
                           DT_INT64, DT_DOUBLE}))
    .OP_END_FACTORY_REG(Mod)

/**
* @brief Returns the truth value of (x != y) element-wise.

* @par Inputs:
* Two inputs, including:
* @li x1: A Tensor. Must be one of the following types: float16, float32, int32,
* int8, uint8, double, int16, int64, uint16, half, uint32, uint64.
* @li x2: A Tensor of the same type as "x1". \n

* @par Outputs:
* y: A Tensor of type bool. \n

* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator NotEqual.
*/
REG_OP(NotEqual)
    .INPUT(x1, TensorType::RealNumberType())
    .INPUT(x2, TensorType::RealNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(NotEqual)

/**
* @brief Computes ndtri element-wise (y = sqrt(2) * erfinv(2 * x - 1)).

* @par Inputs:
* One input, including: \n
* x: A Tensor. Must be one of the following types: bfloat16, float16,
* float32, double. \n

* @par Outputs:
* y: A Tensor. Has the same type and format as input "x". \n

* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator Ndtri.
*/
REG_OP(Ndtri)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(Ndtri)

/**
*@brief Computes numerical negative value element-wise (y = -x)

*@par Inputs:
* One input:
*x: A Tensor. Must be one of the following types: float16, float32, int32,
 * int64, complex64, complex128. \n

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x". \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Neg.
*/
REG_OP(Neg)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Neg)

/**
* @brief Returns x1/x2 element-wise for integer types.

* @par Inputs:
* @li x1: A Tensor. Must be one of the following types:
*     float32, float16, int8, uint8, int32, int16,
*     uint16, double, int64, complex64, complex128.
* @li x2: A Tensor of the same data type as "x1". \n

* @par Outputs:
* y: A Tensor. Has the same type as "x1".

* @attention Constraints:
* Broadcasting is supported. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator TruncateDiv. \n

*/
REG_OP(TruncateDiv)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT32,
                           DT_DOUBLE, DT_UINT16, DT_INT16, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT32,
                           DT_DOUBLE, DT_UINT16, DT_INT16, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT32,
                           DT_DOUBLE, DT_UINT16, DT_INT16, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(TruncateDiv)

/**
*@brief Computes x1/x2 element-wise, if x1 == 0, return 0.

*@par Inputs:
* Two inputs, including:
* @li x1: A Tensor. Must be one of the following types: float16, float32,
* double, complex64, complex128.
* @li x2: A Tensor. Has the same type as "x1". \n

*@par Outputs:
*y: A Tensor. Has the same type as "x1". \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator Xdivy.
*/
REG_OP(Xdivy)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
                           DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
                           DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
                           DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Xdivy)

/**
* @brief Computes "x" multiplied by the logarithm of y element-wise,
* if "x" == 0, return "0".

* @par Inputs:
* Two inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32,
* double, complex64, complex128.
* @li y: A Tensor. Has the same type as "x". \n

* @par Outputs:
* z: A Tensor. Has the same type as "x". \n

* @par Third-party framework compatibility
* Compatible with TensorFlow operator Xlog1py.
*/
REG_OP(Xlog1py)
    .INPUT(x, TensorType({DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
                          DT_COMPLEX128}))
    .INPUT(y, TensorType({DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
                          DT_COMPLEX128}))
    .OUTPUT(z, TensorType({DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
                           DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Xlog1py)

/**
*@brief Computes "x" multiplied by the logarithm of y element-wise,
* if "x" == 0, return "0".

*@par Inputs:
* Two inputs, including:
* @li x1: A Tensor. Must be one of the following types: float16, float32,
* double, complex64, complex128.
* @li x2: A Tensor. Has the same type as "x1". \n

*@par Outputs:
*y: A Tensor. Has the same type as "x1". \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator Xlogy.
*/
REG_OP(Xlogy)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
                           DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
                           DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
                           DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Xlogy)

/**
*@brief Computes square of "x" element-wise. \n

*@par Inputs:
*One input: \n
*x: A Tensor. Must be one of the following types: float16, float32, float64, int32, int64, complex64, complex128

*@par Outputs:
*y: A Tensor. Has the same type as "x". \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator Square.
*/
REG_OP(Square)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT,
                          DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT,
                           DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Square)


/**
*@brief Computes reciprocal of square root of "x" element-wise: y = 1/sqrt{x}. \n

*
*@par Inputs:
* x: An ND or 5HD tensor. Must be one of the following types: float, double, half,
 * complex64, complex128.
*
*@par Outputs:
* y: An ND or 5HD tensor. Has the same type as "x".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Rsqrt.
*
*/
REG_OP(Rsqrt)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Rsqrt)

/**
*@brief Computes the trignometric inverse sine of "x" element-wise. \n

*
*@par Inputs:
* x: A tensor. Must be one of the following types: float16, float32, float64, int32, int64, complex64, complex128.
*
*@par Outputs:
* y: A tensor. Has the same type as "x".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Asin.
*
*/
REG_OP(Asin)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                          DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                           DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Asin)

/**
*@brief Computes gradients for Asin operation. \n

*
*@par Inputs:
*@li y: A tensor of type float16, float32, float64, int32, int64, complex64, complex128.
*@li dy: A tensor of the same type as "y".
*
*@attention Constraints:
* "dy" has the same type as "y".
*
*@par Outputs:
* z: A tensor. Has the same type as "y".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator AsinGrad.
*
*/
REG_OP(AsinGrad)
  .INPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                        DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
  .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                         DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
  .OUTPUT(z, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                         DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
  .OP_END_FACTORY_REG(AsinGrad)

/**
*@brief Computes acos of x element-wise. \n

*
*@par Inputs:
* x: A tensor. Must be one of the following types: float16, float32, float64, int32, int64, complex64, complex128.
*
*@par Outputs:
* y: A tensor. Has the same type as "x".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Acos.
*
*/
REG_OP(Acos)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                          DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                           DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Acos)

/**
*@brief Computes gradients for Acos operation. \n

*
*@par Inputs:
*@li y: A tensor of type float16 or float32.
*@li dy: A tensor of the same type as "y".
*
*@attention Constraints:
* "dy" has the same shape as "y".
*
*@par Outputs:
* z: A tensor. Has the same type as "y".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator AcosGrad.
*
*/
REG_OP(AcosGrad)
  .INPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
  .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT}))
  .OUTPUT(z, TensorType({DT_FLOAT16, DT_FLOAT}))
  .OP_END_FACTORY_REG(AcosGrad)

/**
*@brief Computes inverse hyperbolic cosine of x element-wise. \n

*
*@par Inputs:
* x: A tensor. Must be one of the following types: float16, float32, float64, complex64, complex128.
*
*@attention Constraints:
* x Given an input tensor, the function computes inverse hyperbolic cosine of every element.\n
*   Input range is [1, inf].
*
*@par Outputs:
* y: A tensor. Has the same type as "x".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Acosh.
*
*/
REG_OP(Acosh)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Acosh)

/**
*@brief Computes gradients for Acosh operation. \n

*
*@par Inputs:
*@li y: A tensor of type float16 or float32.
*@li dy: A tensor of the same type as "y".
*
*@attention Constraints:
* "dy" has the same type as "y".
*
*@par Outputs:
* z: A tensor. Has the same type as "y".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator AcoshGrad.
*
*/
REG_OP(AcoshGrad)
  .INPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
  .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT}))
  .OUTPUT(z, TensorType({DT_FLOAT16, DT_FLOAT}))
  .OP_END_FACTORY_REG(AcoshGrad)

/**
*@brief Returns the truth value of x1 OR x2 element-wise. \n

*
*@par Inputs:
*@li x1: A tensor of type bool.
*@li x2: A tensor of the same type as "x1".
*
*@attention Constraints:
* LogicalOr supports broadcasting.
*
*@par Outputs:
* y: A tensor of the same type as "x1".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator LogicalOr.
*
*/
REG_OP(LogicalOr)
    .INPUT(x1, TensorType({DT_BOOL}))
    .INPUT(x2, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(LogicalOr)

/**
* @brief Computes spence of x element-wise.

*
* @par Inputs:
*  x: A tensor. Must be one of the following types: bfloat16, float16, float32, double.
*
* @par Outputs:
*  y: A tensor. Has the same type as "x".
*
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Spence.
*
*/
REG_OP(Spence)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(Spence)

/**
*@brief Returns the truth value of x1 AND x2 element-wise. \n

*
*@par Inputs:
*@li x1: A tensor of type bool.
*@li x2: A tensor of the same type as "x1".
*
*@attention Constraints:
* LogicalAnd supports broadcasting.
*
*@par Outputs:
* y: A tensor of the same type as "x1".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator LogicalAnd.
*
*/
REG_OP(LogicalAnd)
    .INPUT(x1, TensorType({DT_BOOL}))
    .INPUT(x2, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(LogicalAnd)

/**
*@brief Computes the Bessel i0e function of "x" element-wise.
* Exponentially scaled modified Bessel function of order 0
* defined as: bessel_i0e(x) = exp(-abs(x)) bessel_i0(x).
* This function is faster and numerically stabler than "bessel_i0(x)".
*
*@par Inputs:
* x: A tensor of type float16, float32, or float64.
*
*@par Outputs:
* y: A tensor. Has the same type as "x".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator BesselI0e.
*
*/
REG_OP(BesselI0e)
    .INPUT(x, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(BesselI0e)

/**
*@brief Computes the Bessel i1e function of "x" element-wise.
* Exponentially scaled modified Bessel function of order 0
* defined as: bessel_i1e(x) = exp(-abs(x)) bessel_i1(x).
* This function is faster and numerically stabler than "bessel_i1(x)".
*
*@par Inputs:
* x: A tensor of type float16, float32, or float64.
*
*@par Outputs:
* y: A tensor. Has the same type as "x".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator BesselI1e.
*
*/
REG_OP(BesselI1e)
    .INPUT(x, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(BesselI1e)

/**
* @brief Computes logarithm of x element-wise.
* y = log_base(shift + scale * x), with "base" > 0. \n

* @par Inputs:
* x: A Tensor of type complex64, complex128, float16, float32 or double. \n

* @par Attributes:
* @li base: An optional float32, specifying the base "e". Defaults to "-1.0"

* @li scale: An optional float32, specifying the scale of input "x". Defaults
* to "1.0"
* @li shift: An optional float32, specifying the shift. Defaults to "0.0"

* @par Outputs:
* y: A Tensor has same type as "x". \n

* @attention Constraints:
* @li "base" is supposed to be greater than 0. Retaining the default
* value "-1" sets "base" to "e".
* @li If the input value of operator Log is within the range (0, 0.01] or
* [0.95, 1.05], the output accuracy is subject to change. \n

* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator Log.
* @li Compatible with the Caffe operator Log.
*/
REG_OP(Log)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .ATTR(base, Float, -1.0)
    .ATTR(scale, Float, 1.0)
    .ATTR(shift, Float, 0.0)
    .OP_END_FACTORY_REG(Log)

/**
* @brief Returns x1 * x2 element-wise.
* y = x1 * x2

* @par Inputs:
* @li x1: A Tensor. Must be one of the following types: float16, float32,
* float64, uint8, int8, uint16, int16, int32, int64, complex64, complex128.
* @li x2: A Tensor. Must be one of the following types: float16, float32,
* float64, uint8, int8, uint16, int16, int32, int64, complex64, complex128. \n

* @par Outputs:
* y: A Tensor. Must be one of the following types: float16, float32, float64,
* uint8, int8, uint16, int16, int32, int64, complex64, complex128. \n

* @attention Constraints:
* "x1" and "x2" have incompatible shapes or types. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Multiply.
*/
REG_OP(Mul)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Mul)

/**
* @brief Computes the gradient of the square root of "x" with regard to its
* input. grad = dy * 0.5/y, where y = sqrt(x), and "dy" is the corresponding
* input gradient. \n

* @par Inputs:
* Two inputs, including:
* @li y: A Tensor of type float32 or float16.
* @li dy: A Tensor. Has the same type as "y". \n

* @par Outputs:
* z: A Tensor. Has the same type as "y". \n

* @attention Constraints:
* "dy" has the same shape and type as "y".
*/
REG_OP(SqrtGrad)
    .INPUT(y, TensorType(UnaryDataType))
    .INPUT(dy, TensorType(UnaryDataType))
    .OUTPUT(z, TensorType(UnaryDataType))
    .OP_END_FACTORY_REG(SqrtGrad)

/**
*@brief Returns x + y element-wise.
*@par Inputs:
*Two inputs, including:
* @li x1: A Tensor. Must be one of the following types: int8, int16, int32, int64, uint8, float64,
*     float16, float32, complex128, complex64, string.
* @li x2: A Tensor of the same type as "x1". \n

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Add.
*/
REG_OP(Add)
    .INPUT(x1, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                           DT_COMPLEX64, DT_STRING}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                           DT_COMPLEX64, DT_STRING}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                           DT_COMPLEX64, DT_STRING}))
    .OP_END_FACTORY_REG(Add)

/**
*@brief Confuse broadcast, add and mul. \n

*@par Inputs:
*Five inputs, including:
* @li x1: A Tensor. Must be one of the following types:int32 float16, float32.
* @li x2: A Tensor of the same type as "x1".
* @li x3: A Tensor of the same type as "x1". \n

*@par Outputs:
*@li y: A Tensor. Has the same type as "x1". \n

*@par Third-party framework compatibility:
* Compatible with the TensorFlow operator LRN.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/

REG_OP(FusedMulAdd)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(FusedMulAdd)

/**
*@brief Confuse mul+add+add with broadcast. \n

*@par Inputs:
*Four inputs, including:
* @li x1: A Tensor. Must be one of the following types:int32, float16, float32.
* @li x2: A Tensor of the same type as "x1".
* @li x3: A Tensor of the same type as "x1".
* @li x4: A Tensor of the same type as "x1". \n

*@par Outputs:
* y: A Tensor. Has the same type as "x1". \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/

REG_OP(FusedMulAddAdd)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(x4, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(FusedMulAddAdd)
	
/**
*@brief Returns x1 + x2 element-wise. \n

*
*@par Inputs:
*@li x1: A tensor. Must be one of the following types: float16, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128.
*@li x2: A tensor of the same type as "x1".
*
*@attention Constraints:
* AddV2 supports broadcasting.
*
*@par Outputs:
* y: A tensor. Has the same type as "x1".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator AddV2.
*
*/
REG_OP(AddV2)
    .INPUT(x1, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX64,
                           DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX64,
                           DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX64,
                           DT_COMPLEX128}))
    .OP_END_FACTORY_REG(AddV2)

/**
*@brief Updates "ref" by adding "value" to it. \n

*@par Inputs:
*@li ref: A Tensor. Must be one of the following types: float16, float32, int8, int16, int32, int64, uint8, uint16, uint32, uint64.
*@li value: A Tensor of the same type as "ref". \n

*@par Attributes:
*use_locking: An optional bool. Defaults to "False".
              If "True", the addition will be protected by a lock;
              otherwise the behavior is undefined, but may exhibit less contention.
*             This attribute is reserved. \n

*@par Outputs:
*ref: A Tensor that holds the new value of ref after the value has been added. \n

*@attention Constraints:
*An input tensor of type int64 must have a shape with size 1. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator AssignAdd.
*/
REG_OP(AssignAdd)
    .INPUT(ref, TensorType::BasicType())
    .INPUT(value,TensorType::BasicType())
    .OUTPUT(ref, TensorType::BasicType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(AssignAdd)

/**
*@brief Updates "ref" by assigning "value" to it. \n

*@par Inputs:
*@li ref: A Tensor. Must be one of the following types: float16, float32, int8, int16, int32, int64, uint8, uint16, uint32, uint64.
*@li value: A Tensor of the same type as "ref". \n

*@par Attributes:
*@li validate_shape: An optional bool. Defaults to "true".
                     If "true", the operation will validate that the shape of "value" matches the shape of the Tensor being assigned to.
*                    If "false", "ref" will take on the shape of "value".
*                    This attribute is reserved.
*@li use_locking: An optional bool. Defaults to True.
                  If True, the assignment will be protected by a lock;
                  otherwise the behavior is undefined, but may exhibit less contention.
*                 This attribute is reserved. \n

*@par Outputs:
*ref: A Tensor that holds the new value of ref after the value has been assigned. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Assign.
*/
REG_OP(Assign)
    .INPUT(ref, TensorType::BasicType())
    .INPUT(value,TensorType::BasicType())
    .OUTPUT(ref, TensorType::BasicType())
    .ATTR(validate_shape, Bool, true)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(Assign)

/**
*@brief Updates "var" by subtracting "value" from it.\n
* This operation outputs "var" after the update is done. \n
* This makes it easier to chain operations that need to use the reset value. \n

*
*@par Inputs:
*@li var: A tensor. Must be one of the following types: float32, float64, int32, uint8, int16, int8, complex64, int64, qint8, quint8, qint32, uint16, complex128, uint32, uint64
*@li value: A tensor of the same type as "var".
*
*@par Attributes:
* use_locking: An optional bool. Defaults to "False". If "True", the subtraction will be protected \n
* by a lock; otherwise the behavior is undefined, but may exhibit less contention.
*
*@par Outputs:
* y: A tensor. Has the same type as "var".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator AssignSub.
*
*/
REG_OP(AssignSub)
    .INPUT(var, TensorType::NumberType())
    .INPUT(value,TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(AssignSub)

/**
*@brief: Computes the backpropagation of the square root operation. \n

*@par Inputs:
* Two inputs, including:
*@li y: An NCHW, NHWC, ND Tensor. Must be one of the following types: \
 * float, int32, int8, double, complex64, complex128, half.
*@li dy: A Tensor of the same type and format as "y". \n

*@par Outputs:
*z: A Tensor of the same type and format as "y". \n

*@see Matmul() | Rsqrt ()

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator RsqrtGrad.
*/
REG_OP(RsqrtGrad)
    .INPUT(y, TensorType({UnaryDataType,int32,int8}))
    .INPUT(dy, TensorType({UnaryDataType,int32,int8}))
    .OUTPUT(z, TensorType({UnaryDataType,int32,int8}))
    .OP_END_FACTORY_REG(RsqrtGrad)

/**
* @brief Computes hyperbolic sine of "x" element-wise.

* @par Inputs:
* x: An NCHW, NHWC,or ND Tensor of type float, double, complex64,
* complex128, half. \n

* @par Outputs:
* y: A NCHW, NHWC,or ND Tensor of type float, double, complex64,
* complex128, half. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Sinh. \n

*/
REG_OP(Sinh)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Sinh)

/**
*@brief: Clips tensor values to a specified min and max. \n

*@par Inputs:
* Three inputs, including:
*@li x: A Tensor of type  float32, float64, int32, uint8, int16, int8, complex64, int64,
*qint8, quint8, qint32, uint16, complex128, float16, uint32, uint64.
*@li clip_value_min: A Tensor of the same type as "x".
*@li clip_value_max: A Tensor of the same type as "x". \n

*@par Outputs:
*y: A Tensor. Has the same type as "x". \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ClipByValue.
*/
REG_OP(ClipByValue)
    .INPUT(x, TensorType::NumberType())
    .INPUT(clip_value_min, TensorType::NumberType())
    .INPUT(clip_value_max, TensorType::NumberType())
    .OUTPUT(y, TensorType::NumberType())
    .OP_END_FACTORY_REG(ClipByValue)

/**
*@brief Computes cosine of "x" element-wise. \n

*@par Inputs:
*x: A Tensor of type float16, float32, double, complex64, complex128.
* the format can be [NCHW,NHWC,ND]. \n

*@par Outputs:
*y: A Tensor. Has the same type as "x". \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Cosh. \n

*/
REG_OP(Cosh)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Cosh)

/**
*@brief: Returns 0 if the denominator is zero, else, like Div. \n

*@par Inputs:
* Two inputs, including:
*@li x1: A Tensor. Must be one of the following types:float16, float32, int32,
*    int8, uint8, double, the format can be [NCHW,NHWC,ND].
*@li x2: A Tensor of the same type as "x1". \n

*@par Outputs:
*y: A Tensor. Has the same type as "x1". \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator DivNoNan.
*/
REG_OP(DivNoNan)
    .INPUT(x1, TensorType({DT_FLOAT, DT_UINT8, DT_INT8, DT_INT32, DT_FLOAT16,
                           DT_DOUBLE}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_UINT8, DT_INT8, DT_INT32, DT_FLOAT16,
                           DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_UINT8, DT_INT8, DT_INT32, DT_FLOAT16,
                           DT_DOUBLE}))
    .OP_END_FACTORY_REG(DivNoNan)

/**
* @brief Reverses specific dimensions of a tensor.

* @par Inputs:
* One input: \n
* x: A Tensor, Must be one of the following types:
* int32, uint8, int16, int8, int64, int64, uint16, uint32, uint64,
* and format can be [NCHW,NHWC,ND]. \n

* @par Outputs:
* y: A Tensor. Has the same type and format as "x". \n

* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator Invert.
*/
REG_OP(Invert)
    .INPUT(x, TensorType::IntegerDataType())
    .OUTPUT(y, TensorType::IntegerDataType())
    .OP_END_FACTORY_REG(Invert)

/**
*@brief Returns a tensor of the same shape and type with all elements set to one.
*@par Inputs:
*One input: \n
*x: A Tensor. Must be one of the following types: float16, float32, int8, uint8,
 * int16, uint16, int32, int64, complex128, bool. \n

*@par Outputs:
*y: A Tensor of the same type as "x". \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator OnesLike.
*/
REG_OP(OnesLike)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8,
                          DT_UINT8, DT_INT16, DI_UINT16, DT_INT32,
                          DT_INT64, DT_COMPLEX128, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8,
                           DT_UINT8, DT_INT16, DI_UINT16, DT_INT32,
                           DT_INT64, DT_COMPLEX128, DT_BOOL}))
    .OP_END_FACTORY_REG(OnesLike)

/**
*@brief Computes the gradient for the inverse of "x" with regard its input. \n

*@par Inputs:
*@li input_y: A Tensor. Must be one of the following types: float, double,
 * complex64, complex128, half.
*@li input_dy: A Tensor. Must be one of the following types: float, double,
 * complex64, complex128, half. \n

*@par Outputs:
*output_data: A Tensor. Must be one of the following types: float, double,
 * complex64, complex128, half. \n

*@attention Constraints:
* "input_dy" has the same shape and type as "input_y". \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator reciprocal_grad.
*/
REG_OP(ReciprocalGrad)
    .INPUT(y, TensorType::UnaryDataType())
    .INPUT(dy, TensorType::UnaryDataType())
    .OUTPUT(z, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(ReciprocalGrad)

/**
*@brief Returns the truth value of (x1 > x2) element-wise. \n
*when input is int32 and (x2 - x1) > 2**31 or < -2**31
*aicore accuracy is not guaranteed \n

*@par Inputs:
*@li x1: A Tensor of type float16, float32, double, int64, int32, int16, int8,
*    uint8, uint16, uint32, uint64.
*@li x2: A Tensor of the same data type as "x1". \n

*@par Outputs:
*y: A Tensor of type bool.

*@attention Constraints:
* Broadcasting is supported. \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Greater. \n

*/
REG_OP(Greater)
    .INPUT(x1, TensorType::RealNumberType())
    .INPUT(x2, TensorType::RealNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(Greater)

/**
*@brief Returns a tensor of the same type and shape as the input tensor with all elements set to zero. \n

*@par Inputs:
*x: A Tensor. Must be one of the following types:
*     float32, float64, int32, uint8, int16, int8,
*     complex64, int64, qint8, quint8, qint32, qint16, quint16, uint16,
*     complex128, float16, uint32, uint64, complex64, complex128. \n

*@par Outputs:
*y: A Tensor of the same data type as "x". \n

*@attention Constraints:
* The output has the same shape and type as the input. \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator zeros_like.
*/
REG_OP(ZerosLike)
    .INPUT(x, TensorType({BasicType(), DT_VARIANT}))
    .OUTPUT(y, TensorType({BasicType(), DT_VARIANT}))
    .OP_END_FACTORY_REG(ZerosLike)

/**
*@brief Returns the truth value of NOT "x" element-wise. \n

*@par Inputs:
*x: A Tensor of type bool. \n

*@par Outputs:
*y: A Tensor of type bool. \n

*@attention Constraints:
* The input and output values are "1" or "0", corresponding to bool values "true" and "false". \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator logical_not.
*/
REG_OP(LogicalNot)
    .INPUT(x, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(LogicalNot)

/**
*@brief Computes inverse hyperbolic sine of x element-wise.
* Given an input tensor, this function computes inverse hyperbolic sine for every element in the tensor. \n

*
*@par Inputs:
* x: A tensor. Must be one of the following types: float16, float32, float64, complex64, complex128.
*
*@par Outputs:
* y: A tensor. Has the same type as "x".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Asinh.
*
*/
REG_OP(Asinh)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Asinh)

/**
*@brief Computes gradients for Asinh operation. \n

*
*@par Inputs:
*@li y: A tensor. Must be one of the following types: float16, float32.
*@li dy: A tensor of the same type as "y"
*
*@par Outputs:
* z: A tensor. Has the same type as "y".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator AsinhGrad.
*
*/
REG_OP(AsinhGrad)
  .INPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
  .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT}))
  .OUTPUT(z, TensorType({DT_FLOAT16, DT_FLOAT}))
  .OP_END_FACTORY_REG(AsinhGrad)

/**
*@brief Computes inverse hyperbolic tangent of x element-wise.\n
* Given an input tensor, this function computes inverse hyperbolic tangent for every element in the tensor. \n Input range is [-1,1] and output range is [-inf, inf]. If input is -1, \n output will be -inf and if the input is 1, output will be inf.\n  Values outside the range will have nan as output. \n

*
*@par Inputs:
* x: A tensor. Must be one of the following types: float16, float32, float64, complex64, complex128.
*
*@par Outputs:
* y: A tensor. Has the same type as "x".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Atanh.
*
*/
REG_OP(Atanh)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Atanh)

/**
*@brief Computes the trignometric inverse tangent of x element-wise.
* The atan operation returns the inverse of tan, such that if y = tan(x) then, x = atan(y). \n

*
*@par Inputs:
* x: A tensor. Must be one of the following types: float16, float32, float64, complex64, complex128.
*
*@par Outputs:
* y: A tensor. Has the same type as "x". The output of atan will lie within the invertible range of tan, i.e (-pi/2, pi/2).
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Atan.
*
*/
REG_OP(Atan)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Atan)

/**
*@brief Computes gradients for Atan operation. \n

*
*@par Inputs:
*@li y: A tensor of type float16 or float32.
*@li dy: A tensor of the same type as "y"
*
*@par Outputs:
* z: A tensor. Has the same type as "y".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator AtanGrad.
*
*/
REG_OP(AtanGrad)
  .INPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
  .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT}))
  .OUTPUT(z, TensorType({DT_FLOAT16, DT_FLOAT}))
  .OP_END_FACTORY_REG(AtanGrad)

/**
*@brief Computes arctangent of x1/x2 element-wise, respecting signs of the arguments. \n

*
*@par Inputs:
*@li x1: A tensor. Must be one of the following types: float16, float32, float64
*@li x2: A tensor of the same type as "x1".
*
*@par Outputs:
* y: A tensor. Has the same type as "x1".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Atan2.
*
*/
REG_OP(Atan2)
    .INPUT(x1, TensorType::FloatingDataType())
    .INPUT(x2, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(Atan2)

/**
* @brief Computes fresnel_cos of x element-wise.
* 
* @par Inputs:
* x: A tensor. Must be one of the following types: bfloat16, float16, float32,
* double. \n
* 
* @par Outputs:
* y: A tensor. Has the same type as "x". \n
* 
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator FresnelCos.
* 
*/
REG_OP(FresnelCos)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(FresnelCos)

/**
* @brief Computes fresnel_sin of x element-wise.
 
* 
* @par Inputs:
* x: A tensor. Must be one of the following types: bfloat16, float16, float32,
* double. \n
* 
* @par Outputs:
* y: A tensor. Has the same type as "x". \n
* 
* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator FresnelSin.
* 
*/
REG_OP(FresnelSin)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(FresnelSin)

/**
*@brief Returns the truth value of abs(x1-x2) < tolerance element-wise. \n

*
*@par Inputs:
*@li x1: A tensor. Must be one of the following types: float32, float64, int32, uint8, int16, int8, complex64, int64, qint8, quint8, qint32, uint16, complex128, float16, uint32, uint64
*@li x2: A tensor of the same type as "x1".
*
*@par Attributes:
* tolerance: Defaults to "1e-05".
*
*@par Outputs:
* y: A tensor of type bool.
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApproximateEqual.
*
*/
REG_OP(ApproximateEqual)
  .INPUT(x1, TensorType::NumberType())
  .INPUT(x2, TensorType::NumberType())
  .OUTPUT(y, TensorType({DT_BOOL}))
  .ATTR(tolerance, Float, 1e-5)
  .OP_END_FACTORY_REG(ApproximateEqual)

/**
*@brief Returns the element-wise sum of a list of tensors.\n
* AccumulateNV2 performs the same operation as AddN, but does not wait for all of its inputs
to be ready before beginning to sum.\n This can save memory if inputs are ready at different times,
since minimum temporary storage is proportional to the output size rather than the inputs size.
 Returns a Tensor of same shape and type as the elements of inputs. \n

*
*@par Inputs:
*Dynamic inputs, including:
* x: A tensor. Must be one of the following types: float32, float64, int32, uint8, int16, int8, complex64, int64,
qint8, quint8, qint32, uint16, complex128, float16, uint32, uint64. It's a dynamic input. \n
*
*@par Outputs:
* y: A tensor. Has the same type as "x".
*
*@par Attributes:
* N: the size of x.
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator AccumulateNV2.
*
*/
REG_OP(AccumulateNV2)
   .DYNAMIC_INPUT(x, TensorType::NumberType())
   .OUTPUT(y, TensorType::NumberType())
   .REQUIRED_ATTR(N, Int)
   .OP_END_FACTORY_REG(AccumulateNV2)

/**
*@brief Fake-quantizes the input Tensor, type float to output a Tensor of same type.
*  [min, max] define the clamping range for the "inputs" data.\n
*  the values of "x" are quantized into the quantization range ([0, 2^num_bits - 1] \n
*  when "narrow_range" is "false" or [1, 2^num_bits - 1] when it is "true") and \n
*  then de-quantized and output as float32 in [min; max] interval.\n
*  num_bits is the bit width of the quantization, between 2 and 16, inclusive. \n
*  Quantization is called fake since the output is still in floating point. \n

*@par Inputs:
*One input:
*x: A Tensor of type float32. \n

*@par Attributes:
*@li min: An optional attribute. Defaults to "-6.0".
*@li max: An optional attribute. Defaults to "6.0".
*@li num_bits: An optional attribute. Defaults to "8".
*@li narrow_range: An optional bool. Defaults to "false". \n

*@par Outputs:
*y: A Tensor. Has the same shape and type of "x". \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator FakeQuantWithMinMaxArgs.
*/
REG_OP(FakeQuantWithMinMaxArgs)
    .INPUT(x, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(min, Float, -6.0)
    .ATTR(max, Float, 6.0)
    .ATTR(num_bits, Int, 8)
    .ATTR(narrow_range, Bool, false)
    .OP_END_FACTORY_REG(FakeQuantWithMinMaxArgs)

/**
*@brief Computes gradients for a FakeQuantWithMinMaxArgs operation. \n

*@par Inputs:
*Two inputs, including: \n
*@li gradients: A Tensor of type float32. Backpropagated gradients above the FakeQuantWithMinMaxArgs operation.
*@li x: A Tensor of type float32. Has the same type and format as "gradients".\n
* This is the input Tensor of the FakeQuantWithMinMaxArgs operator.\n

*@par Attributes:
*@li min: An optional attribute. Defaults to "-6.0".
*@li max: An optional attribute. Defaults to "6.0".
*@li num_bits: An optional attribute. Defaults to "8".
*@li narrow_range: An optional bool. Defaults to "False". \n

*@par Outputs:
*y: A Tensor of type float32. \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator FakeQuantWithMinMaxArgsGradient.
*/
REG_OP(FakeQuantWithMinMaxArgsGradient)
    .INPUT(gradients, TensorType({DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(min, Float, -6.0)
    .ATTR(max, Float, 6.0)
    .ATTR(num_bits, Int, 8)
    .ATTR(narrow_range, Bool, false)
    .OP_END_FACTORY_REG(FakeQuantWithMinMaxArgsGradient)

/**
*@brief Fake-quantize the 'inputs' tensor of type float via global float scalars. \n

*@par Inputs:
*Three inputs, including:
*@li x: A Tensor of type float32.
*@li min: A Tensor of type float32. Has the same type and format as "x".
*@li max: A Tensor of type float32. Has the same type and format as "x".\n
*[min; max] define the clamping range for the inputs data

*@par Attributes:
*@li num_bits: An optional attribute. Defaults to "8".
*@li narrow_range: An optional bool. Defaults to "False". \n

*@par Outputs:
*y: A Tensor of type float32. \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator FakeQuantWithMinMaxVars.
*/
REG_OP(FakeQuantWithMinMaxVars)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(min, TensorType({DT_FLOAT}))
    .INPUT(max, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(num_bits, Int, 8)
    .ATTR(narrow_range, Bool, false)
    .OP_END_FACTORY_REG(FakeQuantWithMinMaxVars)

/**
*@brief Computes gradients for a FakeQuantWithMinMaxVars operation. \n

*@par Inputs:
*Four inputs, including:
*@li gradients: A Tensor of type float32.
*@li x: A Tensor of type float32.
*@li min: A Tensor of type float32.
*@li max: A Tensor of type float32. \n

*@par Attributes:
*@li num_bits: An integer specifying the quantization bit width. Defaults to "8".
*@li narrow_range: A Boolean specifying whether to use a narrow range for quantization. Defaults to "False". \n

*@par Outputs:
*@li backprops_wrt_x: A Tensor. Has the same type as input "x".
*@li backprops_wrt_min: A Tensor. Has the same type as input "min".
*@li backprops_wrt_max: A Tensor. Has the same type as input "max". \n

*@attention Constraints:
*@li "gradients" has the same shape as "x".
*@li "min" and "max" are scalars.
*@li "num_bits" is between 2 and 16

*@see Region()

*@par Third-party framework compatibility
* Compatible with the operator FakeQuantWithMinMaxVarsGradient.
*/
REG_OP(FakeQuantWithMinMaxVarsGradient)
    .INPUT(gradients, TensorType({DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(min, TensorType({DT_FLOAT}))
    .INPUT(max, TensorType({DT_FLOAT}))
    .OUTPUT(backprops_wrt_x, TensorType({DT_FLOAT}))
    .OUTPUT(backprops_wrt_min, TensorType({DT_FLOAT}))
    .OUTPUT(backprops_wrt_max, TensorType({DT_FLOAT}))
    .ATTR(num_bits, Int, 8)
    .ATTR(narrow_range, Bool, false)
    .OP_END_FACTORY_REG(FakeQuantWithMinMaxVarsGradient)

/**
*@brief Fake-quantizes the "inputs" tensor of type float
via per-channel floats min and max of shape [d] to "outputs" \n
tensor of same shape as inputs

*@par Inputs:
*Three inputs, including:
*@li x: A Tensor of type float32.
*@li min: A Tensor of type float32.
*@li max: A Tensor of type float32. \n

*@par Attributes:
*@li num_bits: An integer specifying the quantization bit width. Defaults to "8".
*@li narrow_range: A Boolean specifying whether to use a narrow range for quantization. Defaults to "False". \n

*@par Outputs:
*y: A Tensor. Has the same type as input "x".


*@attention Constraints:
*@li "min" and "max" have one-dimensional shapes.
*@li "min" has the same last dimension size as "x". "max" has the same last dimension size as "x".
*@li "num_bits" is between 2 and 16

*@see Region()

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator FakeQuantWithMinMaxVarsPerChannel.
*/
REG_OP(FakeQuantWithMinMaxVarsPerChannel)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(min, TensorType({DT_FLOAT}))
    .INPUT(max, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(num_bits, Int, 8)
    .ATTR(narrow_range, Bool, false)
    .OP_END_FACTORY_REG(FakeQuantWithMinMaxVarsPerChannel)

/**
*@brief Computes gradients for a FakeQuantWithMinMaxVarsPerChannel operation. \n

*@par Inputs:
*Four inputs, including:
*@li gradients: A Tensor of type float32.
*@li x: A Tensor of type float32.
*@li min: A Tensor of type float32.
*@li max: A Tensor of type float32. \n

*@par Attributes:
*@li num_bits: An integer specifying the quantization bit width. Defaults to "8".
*@li narrow_range: A Boolean specifying whether to use a narrow range for quantization. Defaults to "False". \n

*@par Outputs:
*@li backprops_wrt_x: A Tensor. Has the same type as input "x".
*@li backprops_wrt_min: A Tensor. Has the same type as input "min".
*@li backprops_wrt_max: A Tensor. Has the same type as input "max". \n

*@attention Constraints:
*@li "gradients" has the same shape as "x".
*@li "min" and "max" have one-dimensional shapes.
*@li "min" has the same last dimension size as "x". "max" has the same last dimension size as "x". "gradients" has the same last dimension size as "x".
*@li "num_bits" is between 2 and 16

*@see Region()

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator FakeQuantWithMinMaxVarsPerChannelGradient.
*/
REG_OP(FakeQuantWithMinMaxVarsPerChannelGradient)
    .INPUT(gradients, TensorType({DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(min, TensorType({DT_FLOAT}))
    .INPUT(max, TensorType({DT_FLOAT}))
    .OUTPUT(backprops_wrt_x, TensorType({DT_FLOAT}))
    .OUTPUT(backprops_wrt_min, TensorType({DT_FLOAT}))
    .OUTPUT(backprops_wrt_max, TensorType({DT_FLOAT}))
    .ATTR(num_bits, Int, 8)
    .ATTR(narrow_range, Bool, false)
    .OP_END_FACTORY_REG(FakeQuantWithMinMaxVarsPerChannelGradient)

/**
*@brief Element-wise computes the bitwise AND of "x1" and "x2". \n

*@par Inputs:
*Two inputs, including:
* @li x1: A Tensor. Must be one of the following types: int8, int16,
*     int32, int64, uint8, uint16, uint32, uint64. Broadcasting is supported.
* @li x2: A Tensor of the same type as "x1". \n

*@par Outputs:
*y: A Tensor. Has the same type as "x1". \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator BitwiseAnd.
*/
REG_OP(BitwiseAnd)
    .INPUT(x1, TensorType::IntegerDataType())
    .INPUT(x2, TensorType::IntegerDataType())
    .OUTPUT(y, TensorType::IntegerDataType())
    .OP_END_FACTORY_REG(BitwiseAnd)

/**
*@brief Element-wise computes the bitwise OR of "x1" and "x2". \n

*@par Inputs:
*Two inputs, including:
* @li x1: A Tensor. Must be one of the following types: int8, int16,
*     int32, int64, uint8, uint16, uint32, uint64. Broadcasting is supported.
* @li x2: A Tensor of the same type as "x1". \n

*@par Outputs:
*y: A Tensor. Has the same type as "x1". \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator BitwiseOr.
*/
REG_OP(BitwiseOr)
    .INPUT(x1, TensorType::IntegerDataType())
    .INPUT(x2, TensorType::IntegerDataType())
    .OUTPUT(y, TensorType::IntegerDataType())
    .OP_END_FACTORY_REG(BitwiseOr)

/**
*@brief Elementwise computes the bitwise XOR of "x1" and "x2". \n

*@par Inputs:
*Two inputs, including:
*@li x1: A Tensor. Must be one of the following types: int8, int16, int32, int64, uint8, uint16, uint32, uint64.
*       The format is ND. Broadcasting is supported.
*@li x2: A Tensor. Has the same type and format as "x1". \n

*@par Outputs:
*y: Output result. Has the same type as "x1". \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator BitwiseXor.
*/
REG_OP(BitwiseXor)
    .INPUT(x1, TensorType::IntegerDataType())
    .INPUT(x2, TensorType::IntegerDataType())
    .OUTPUT(y, TensorType::IntegerDataType())
    .OP_END_FACTORY_REG(BitwiseXor)

/**
*@brief Returns element-wise smallest integer not less than "x". \n

*@par Inputs:
* x: A Tensor of type float16 or float32 or float64. \n

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Ceil.
*/
REG_OP(Ceil)
  .INPUT(x, TensorType::FloatingDataType())
  .OUTPUT(y, TensorType::FloatingDataType())
  .OP_END_FACTORY_REG(Ceil)

/**
*@brief Returns element-wise largest integer not greater than "x". \n

*@par Inputs:
*x: A Tensor of type float16, float32 or double. \n

*@par Outputs:
*y: A Tensor of the same type as "x". \n

*@par Third-party framework compatibility:
* Compatible with TensorFlow operator Floor.
*/
REG_OP(Floor)
  .INPUT(x, TensorType::FloatingDataType())
  .OUTPUT(y, TensorType::FloatingDataType())
  .OP_END_FACTORY_REG(Floor)

/**
*@brief Divides "x1/x2" element-wise, rounding toward the
*        most negative integer. \n

*@par Inputs:
*Two inputs, including:
* @li x1: A Tensor. Must be one of the following types: float16, float32, int32, int64, int8,
*     uint8, int16, uint16, double, complex64, complex128.
* @li x2: A Tensor of the same type as "x1". \n

*@par Outputs:
*y: A Tensor. Has the same type as "x1". \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator FloorDiv.
*/
REG_OP(FloorDiv)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8,
                           DT_INT64, DT_INT16, DT_UINT16, DT_DOUBLE,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8,
                           DT_INT64, DT_INT16,DT_UINT16, DT_DOUBLE,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8,
                           DT_INT64, DT_INT16,DT_UINT16, DT_DOUBLE,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(FloorDiv)

/**
*@brief Returns element-wise remainder of division. Consistent with: floor(x1/x2) * x2 + mod(x1, x2) = x1. \n

*@par Inputs:
* Two inputs, including:
*@li x1: A Tensor. Must be one of the following types:
*    int32, int64, float, float16, double
*@li x2: A Tensor. Must have the same type as "x1".
*
*@par Outputs:
*y: Result remainder.

*@attention Constraints:
*@li x2: The input data does not support 0
*@li When NUM exceeds 2048 , the accuracy of operator cannot guarantee the
*requirement of double thousandths in the mini form
*@li Due to different architectures, the calculation results of this operator
*on NPU and CPU may be inconsistent
*@li If shape is expressed as (D1,D2... ,Dn), then D1*D2... *DN<=1000000,n<=8

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator FloorMod.
*/
REG_OP(FloorMod)
    .INPUT(x1, TensorType({DT_INT32,  DT_INT64, DT_FLOAT, DT_FLOAT16,
                           DT_DOUBLE}))
    .INPUT(x2, TensorType({DT_INT32,  DT_INT64, DT_FLOAT, DT_FLOAT16,
                           DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT32,  DT_INT64, DT_FLOAT, DT_FLOAT16,
                           DT_DOUBLE}))
    .OP_END_FACTORY_REG(FloorMod)

/**
*@brief Computes the power of "x1" to "x2". \n

*@par Inputs:
*Two inputs, including:
* @li x1: A Tensor. Must be one of the following types:
*     float16, float32, int32, int64, int8, uint8, double, complex64, complex128.
* @li x2: A Tensor of the same type as "x1". \n

*@par Outputs:
*y: A Tensor. Has the same type as "x1". \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Pow.
*/
REG_OP(Pow)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_INT8,
                           DT_UINT8, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_INT8,
                           DT_UINT8, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_INT8,
                           DT_UINT8, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Pow)

/**
*@brief Return element-wise integer closest to x. \n

*@par Inputs:
*One input, include:
*x: A mutable Tensor. Must be one of the following types:
*     float16, float32, double. \n

*@par Outputs:
*y: A mutable Tensor. Has the same type as "x". \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Rint.
*/
REG_OP(Rint)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(Rint)

/**
*@brief Rounds the values of a tensor to the nearest integer, element-wise.
 * Rounds half to even. \n

*@par Inputs:
*Inputs including:
*x: A required ND Tensor of type float16, float, int64, double, complex64,
 * complex128 or int32.
*@par Outputs:
*y: A required ND Tensor. Has the same data type and shape as "x".
*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Round.
*/
REG_OP(Round)
    .INPUT(x, TensorType(DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64,
                         DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128))
    .OUTPUT(y, TensorType(DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64,
                          DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128))
    .OP_END_FACTORY_REG(Round)

/**
*@brief: Computes sine of "x" element-wise. \n

*@par Inputs:
*One input:
*x: An ND Tensor. Must be one of the following types: float16, float32, double,
 * complex64, complex128, int32, int64

*@par Outputs:
*y: An ND Tensor. Has the same type as "x". \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator Sin.
*/
REG_OP(Sin)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Sin)

/**
* @brief: Computes tan of "x" element-wise.

* @par Inputs:
* One input:
* x: A Tensor. Must be one of the following types: float16, float32, double, complex64, complex128, int32, int64

* @par Outputs:
* y: A Tensor. Has the same type as "x". \n

* @par Third-party framework compatibility
* Compatible with TensorFlow operator Tan.
*/
REG_OP(Tan)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_COMPLEX64,
                          DT_COMPLEX128, DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_COMPLEX64,
                           DT_COMPLEX128, DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(Tan)

/**
* @brief Returns element-wise remainder of division.

* @par Inputs:
* Two inputs, including:
* @li x1: A Tensor. Must be one of the following types: float16, float32,
* double, int32, int64.
* @li x2: A Tensor of the same type as "x1". \n

* @par Outputs:
* y: A Tensor. Has the same type as "x1". \n

* @attention Constraints:
* @li x2: The input data does not support 0
* @li When NUM exceeds 2048 , the accuracy of operator cannot guarantee the
* requirement of double thousandths in the mini form
* @li Due to different architectures, the calculation results of this operator
* on NPU and CPU may be inconsistent
* @li If shape is expressed as (D1,D2... ,Dn), then D1*D2... *DN<=1000000,n<=8

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator TruncateMod.
*/
REG_OP(TruncateMod)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64,
                           DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64,
                           DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64,
                           DT_INT32}))
    .OP_END_FACTORY_REG(TruncateMod)

/**
*@brief Adds 'bias' to 'x'. \n

*@par Inputs:
*Two inputs, including:
* @li x: A Tensor of type NumberType. Must be one of the following types: float32, float64, int32, uint8, int16,
*int8, complex64, int64, qint8, quint8, qint32, bfloat16, uint16, complex128, float16, uint32, uint64.
* @li bias: A 1D Tensor with size the C dimension of value. \n

*@par Attributes:
*data_format: An optional string. Defaults to "NHWC". \n

*@par Outputs:
*y: A Tensor with same type as "x". \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator BiasAdd.
*/
REG_OP(BiasAdd)
    .INPUT(x, TensorType::NumberType())
    .INPUT(bias, TensorType::NumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(BiasAdd)

/**
*@brief Returns the index with the smallest value across dimensions of a tensor. \n

*@par Inputs:
*Two inputs, including:
*@li x: A Tensor. Must be one of the following types: float32, float64, int32, uint8, int16, int8, complex64, int64, qint8, quint8, qint32, bfloat16, uint16, complex128, float16, uint32, uint64.
*format is ND.
*@li dimension: A Tensor. Must be one of the following types: int32, int64. Must be in the range [-rank(input x), rank(input x)]. Describes which dimension of the input Tensor to reduce across.
* The format is ND.
*@par Attributes:
*dtype: The output type, either "int32" or "int64". Defaults to "int64". \n

*@par Outputs:
*y: A Tensor of type "dtype". \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator ArgMin.
*/
REG_OP(ArgMin)
    .INPUT(x, TensorType::NumberType())
    .INPUT(dimension, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Type, DT_INT64)
    .OP_END_FACTORY_REG(ArgMin)

/**
*@brief Returns the index with the smallest value across dimensions of a tensor. \n

*@par Inputs:
*One input:

*x: A Tensor of type float16 or float32 in ND format. \n

*@par Attributes:
*@li dimension: The dimension of the input Tensor to reduce across.
*@li dtype: An optional attribute, specifying the output data type. Must be "int32". Defaults to "int64". \n

*@par Outputs:
*y: A Tensor of type dtype. \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator ArgMin.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ArgMin instead.
*/
REG_OP(ArgMinD)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .REQUIRED_ATTR(dimension, Int)
    .ATTR(dtype, Type, DT_INT64)
    .OP_END_FACTORY_REG(ArgMinD)

/**
*@brief Returns the index with the largest value across axes of a tensor. \n

*@par Inputs:
* Two inputs, including:
*@li x: A multi-dimensional Tensor of type float16, float32, or int16.
*@li dimension: A Scalar of type int32, specifying the index with the largest value. \n

*@par Attributes:
*dtype: The output type, either "int32" or "int64". Defaults to "int64". \n

*@par Outputs:
*y: A multi-dimensional Tensor of type int32 or int64, specifying the index with the largest value. The dimension is one less than that of "x". \n

*@attention Constraints:
*@li x: If there are multiple maximum values, the index of the first maximum value is used.
*@li The value range of "dimension" is [-dims, dims - 1]. "dims" is the dimension length of "x". \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator ArgMax.
*/
REG_OP(ArgMaxV2)
    .INPUT(x, TensorType::NumberType())
    .INPUT(dimension, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Type, DT_INT64)
    .OP_END_FACTORY_REG(ArgMaxV2)

/**
*@brief Returns the index with the largest value across axes of a tensor. \n

*@par Inputs:
* One input, including:
*x: A multi-dimensional Tensor of type float16, float32. \n

*@par Attributes:
*@li dimension: An integer of type int32, specifying the axis information of the index with the maximum value.
*@li dtype: The output type, either "int32" or "int64". Defaults to "int64". \n

*@par Outputs:
*y: A multi-dimensional Tensor of type int32, specifying the index with the largest value. The dimension is one less than that of "x". \n

*@attention Constraints:
*@li x: If there are multiple maximum values, the index of the first maximum value is used.
*@li The value range of "dimension" is [-dims, dims - 1]. "dims" is the dimension length of "x". \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator ArgMax.
*
* @par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(ArgMaxD)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .REQUIRED_ATTR(dimension, Int)
    .ATTR(dtype, Type, DT_INT64)
    .OP_END_FACTORY_REG(ArgMaxD)

/**
*@brief Returns the maximum value of all elements in the input in the given
* dimension. \n

*@par Inputs:
*One input: \n
*x: A multi-dimensional Tensor of type float16 or float32. \n

*@par Attributes:
*@li dimension: An integer of type int32, specifying the axis information of
* the index with the maximum value.
*@li keep_dims: A bool, specifying whether to keep dimensions for the output
* Tensor. Defaults to "false". \n

*@par Outputs:
*@li indice: A multi-dimensional Tensor of type int32, specifying the index.
* (If "keep_dims" is set to "false", the output dimensions are reduced by
* "dimension" compared with that of "x". Otherwise, the output has one fewer
* dimension than "x".)
*@li values: A Tensor, specifying the maximum value. Has the same dimensions
* as "indice" and the same type as "x". \n

*@attention Constraints:
*@li If there are multiple maximum values, the index of the first maximum
* value is used.
*@li The value range of "dimension" is [-dims, dims - 1]. "dims" is the
* dimension length of "x". \n

*@par Third-party framework compatibility
* Compatible with the two output scenarios of PyTorch operator Max (the output
* sequence is opposite to that of PyTorch).
*/
REG_OP(ArgMaxWithValue)
    .INPUT(x, TensorType({DT_FLOAT,DT_FLOAT16}))
    .OUTPUT(indice,TensorType({DT_INT32}))
    .OUTPUT(values, TensorType({DT_FLOAT,DT_FLOAT16}))
    .REQUIRED_ATTR(dimension, Int)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ArgMaxWithValue)

/**
*@par Inputs:
*One input: \n
*x: A multi-dimensional Tensor of type float16 or float32. \n

*@par Attributes:
*@li dimension: An integer of type int32, specifying the axis information of
* the index with the maximum value.
*@li keep_dims: A bool, specifying whether to keep dimensions for the output
* Tensor. Defaults to "false". \n

*@par Outputs:
*@li indice: A multi-dimensional Tensor of type int32, specifying the index.
* (If "keep_dims" is set to "false", the output dimensions are reduced by
* "dimension" compared with that of "x". Otherwise, the output has one fewer
* dimension than "x".)
*@li values: A Tensor, specifying the minimum value. Has the same dimensions
* as "indice" and the same type as "x". \n

*@attention Constraints:
*@li If there are multiple minimum values, the index of the first minimum
* value is used.
*@li The value range of "dimension" is [-dims, dims - 1]. "dims" is the
* dimension length of "x".
*@li Performing the ArgMinWithValue operation on the last axis of float32 data
* is not supported on a mini platform. \n

*@par Third-party framework compatibility
* Compatible with the two output scenarios of PyTorch operator Min (the output
* sequence is opposite to that of PyTorch).
*/
REG_OP(ArgMinWithValue)
    .INPUT(x, TensorType({DT_FLOAT,DT_FLOAT16}))
    .OUTPUT(indice,TensorType({DT_INT32}))
    .OUTPUT(values, TensorType({DT_FLOAT,DT_FLOAT16}))
    .REQUIRED_ATTR(dimension, Int)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ArgMinWithValue)

/**
*@brief Compute elementwise modes, such as 0: PRODUCT, 1: SUM, 2: MAX

*@par Inputs:
*One input: \n
*x: the list of input data, the type of element in Tensor should be same.
*   the max size of x is 32.
*   should met one of the following types: float16, float32. It's a dynamic input. \n

*@par Outputs:
*y: A Tensor. Has the same type and format as "x". \n

*@par Attributes:
*@li N: A required attribute. the number of input x, max size is 32. Type is int.
*@li model: An optional attribute. Type is int. Defaults to "1".
*    "0": product, "1": sum, "2": max.
*@li coeff: A required attribute. Must met all of following rules:
*    size of "coeff" must be equal to len("x") or is null.
*    the absolute value of "coeff" must less than or equal to 1.
*/
REG_OP(Eltwise)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(N, Int)
    .ATTR(mode, Int, 1)
    .ATTR(coeff, ListFloat, {})
    .OP_END_FACTORY_REG(Eltwise)

/**
 *@brief Computes the inverse error function of each element of input. \n

 *@par Inputs:
 *One inputs, including:
 * input_x: A tensor. Must be one of the following types:
 *     float16, float32. \n

 *@par Outputs:
 *output_y: A Tensor with the same type and shape of input_x's. \n

 *@par Third-party framework compatibility
 *Compatible with the Pytorch operator Erfinv. \n
 */
REG_OP(Erfinv)
    .INPUT(input_x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(output_y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(Erfinv)

/**
*@brief Computes element-wise population count. \n

*@par Inputs:
*x: A Tensor of type TensorType::IntegerDataType(). \n

*@par Outputs:
*y: A Tensor of type uint8. \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator PopulationCount.
*/
REG_OP(PopulationCount)
  .INPUT(x, TensorType::IntegerDataType())
  .OUTPUT(y, TensorType({DT_UINT8}))
  .OP_END_FACTORY_REG(PopulationCount)

/**
*@brief A fusion operator for bert lamb. \n

*@par Inputs:
*Thirteen inputs, including:
* @li input_mul3: A Tensor. Must be one of the following types: float16, float32.
* @li input_mul2: A Tensor. Must be one of the following types: float16, float32.
* @li input_realdiv1: A Tensor. Must be one of the following types: float16, float32.
* @li input_mul1: A Tensor. Must be one of the following types: float16, float32.
* @li input_mul0: A Tensor. Must be one of the following types: float16, float32.
* @li input_realdiv0: A Tensor. Must be one of the following types: float16, float32.
* @li input_mul4: A Tensor. Must be one of the following types: float16, float32.
* @li mul0_x: A Tensor. Must be one of the following types: float16, float32.
* @li mul1_sub: A Tensor. Must be one of the following types: float16, float32.
* @li mul2_x: A Tensor. Must be one of the following types: float16, float32.
* @li mul3_sub1: A Tensor. Must be one of the following types: float16, float32.
* @li mul4_x: A Tensor. Must be one of the following types: float16, float32.
* @li add2_y: A Tensor. Must be one of the following types: float16, float32. \n

*@par Outputs:
*Four outputs, including:
* @li y1: A Tensor. Must be one of the following types: float16, float32.
* @li y2: A Tensor. Must be one of the following types: float16, float32.
* @li y3: A Tensor. Must be one of the following types: float16, float32.
* @li y4: A Tensor. Must be one of the following types: float16, float32. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(LambNextMVWithDecay)
    .INPUT(input_mul3, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input_mul2, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input_realdiv1, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input_mul1, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input_mul0, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input_realdiv0, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input_mul4, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul0_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul1_sub, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul2_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul3_sub1, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul4_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(add2_y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(y1, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(y2, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(y3, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(y4, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(LambNextMVWithDecay)

/**
*@brief Confuse real_div, rsqrt, sqrt, maximum, minimum, sub and add. \n

*@par Inputs:
*Thirteen inputs, including:
* @li input_mul3: A Tensor. Must be one of the following types: float16, float32.
* @li input_mul2: A Tensor of the same type as "input1".
* @li input_realdiv1: A Tensor of the same type as "input1".
* @li input_mul1: A Tensor of the same type as "input1".
* @li input_mul0: A Tensor of the same type as "input1".
* @li input_realdiv0: A Tensor. Must be one of the following types: float16, float32.
* @li input_mul4: A Tensor of the same type as "input1".
* @li mul0_x: A Tensor of the same type as "input1".
* @li mul1_sub: A Tensor of the same type as "input1".
* @li mul2_x: A Tensor of the same type as "input1".
* @li mul3_sub1: A Tensor. Must be one of the following types: float16, float32.
* @li mul4_x: A Tensor of the same type as "input1".
* @li add2_y: A Tensor of the same type as "input1". \n

*@par Outputs:
*Four outputs, including:
*@li y1: A Tensor. Has the same type as "input_mul3".
*@li y2: A Tensor. Has the same type as "input_mul3".
*@li y3: A Tensor. Has the same type as "input_mul3".
*@li y4: A Tensor. Has the same type as "input_mul3".

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(LambNextMV)
    .INPUT(input_mul3, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input_mul2, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input_realdiv1, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input_mul1, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input_mul0, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input_realdiv0, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input_mul4, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul0_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul1_sub, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul2_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul3_sub1, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul4_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(add2_y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(y1, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(y2, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(y3, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(y4, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(LambNextMV)

/**
*@brief A fusion operator for bert lamb. \n

*@par Inputs:
*Six inputs, including:
* @li input_square: A Tensor. Must be one of the following types: float16, float32.
* @li input_mul2: A Tensor. Must be one of the following types: float16, float32.
* @li mul2_x: A Tensor. Must be one of the following types: float16, float32.
* @li mul3_x: A Tensor. Must be one of the following types: float16, float32.
* @li truediv1_recip: A Tensor. Must be one of the following types: float16, float32.
* @li add2_y: A Tensor. Must be one of the following types: float16, float32. \n

*@par Outputs:
*Two outputs, including:
* @li y1: A Tensor of the same type as "input_square".
* @li y2: A Tensor of the same type as "input_square". \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(LambNextRight)
    .INPUT(input_square, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input_mul2, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul2_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul3_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(truediv1_recip, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(add2_y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(y1, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(y2, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(LambNextRight)

/**
*@brief A fusion operator for bert lamb. \n

*@par Inputs:
*Six inputs, including:
* @li input_greater1: A Tensor. Must be one of the following types: float16, float32.
* @li input_greater_realdiv: A Tensor. Must be one of the following types: float16, float32.
* @li input_realdiv: A Tensor. Must be one of the following types: float16, float32.
* @li input_mul0: A Tensor. Must be one of the following types: float16, float32.
* @li input_mul1: A Tensor. Must be one of the following types: float16, float32.
* @li input_sub: A Tensor. Must be one of the following types: float16, float32.
* @li greater_y: A Tensor. Must be one of the following types: float16, float32.
* @li select_e: A Tensor. Must be one of the following types: float16, float32.
* @li minimum_y: A Tensor. Must be one of the following types: float16, float32. \n

*@par Outputs:
*y: A Tensor of the same type as "input_greater1". \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(LambUpdateWithLr)
    .INPUT(input_greater1, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input_greater_realdiv, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input_realdiv, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input_mul0, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input_mul1, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input_sub, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(greater_y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(select_e, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(minimum_y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(LambUpdateWithLr)

/**
*@brief A fusion operator for bert lamb. \n

*@par Inputs:
*Seven inputs, including:
* @li x1: A Tensor. Must be one of the following types: float16, float32.
* @li x2: A Tensor. Must be one of the following types: float16, float32.
* @li x3: A Tensor. Must be one of the following types: float16, float32.
* @li x4: A Tensor. Must be one of the following types: float16, float32.
* @li x5: A Tensor. Must be one of the following types: float16, float32.
* @li greater_y: A Tensor. Must be one of the following types: float16, float32.
* @li select_e: A Tensor. Must be one of the following types: float16, float32. \n

*@par Outputs:
*y: A Tensor of the same type as input. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(LambUpdateWithLrV2)
    .INPUT(x1, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(x3, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(x4, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(x5, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(greater_y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(select_e, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(LambUpdateWithLrV2)

/**
*@brief A fusion operator for bert lamb. \n

*@par Inputs:
*Eleven inputs, including:
* @li input0: A Tensor. Must be one of the following types: float16, float32.
* @li input1: A Tensor. Must be one of the following types: float16, float32.
* @li input2: A Tensor. Must be one of the following types: float16, float32.
* @li input3: A Tensor. Must be one of the following types: float16, float32.
* @li input4: A Tensor. Must be one of the following types: float16, float32.
* @li mul0_x: A Tensor. Must be one of the following types: float16, float32.
* @li mul1_x: A Tensor. Must be one of the following types: float16, float32.
* @li mul2_x: A Tensor. Must be one of the following types: float16, float32.
* @li mul3_x: A Tensor. Must be one of the following types: float16, float32.
* @li mul4_x: A Tensor. Must be one of the following types: float16, float32.
* @li add2_y: A Tensor. Must be one of the following types: float16, float32. \n

*@par Outputs:
*Three outputs, including:
* @li output0: A Tensor. Must be one of the following types: float16, float32.
* @li output1: A Tensor. Must be one of the following types: float16, float32.
* @li output2: A Tensor. Must be one of the following types: float16, float32. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(AdamApplyOneWithDecay)
    .INPUT(input0, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input1, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input2, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input3, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input4, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul0_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul1_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul2_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul3_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul4_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(add2_y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(output0, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(output1, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(output2, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(AdamApplyOneWithDecay)

/**
*@brief A fusion operator for bert lamb. \n

*@par Inputs:
*Ten inputs, including:
* @li input0: A Tensor. Must be one of the following types: float16, float32.
* @li input1: A Tensor. Must be one of the following types: float16, float32.
* @li input2: A Tensor. Must be one of the following types: float16, float32.
* @li input3: A Tensor. Must be one of the following types: float16, float32.
* @li input4: A Tensor. Must be one of the following types: float16, float32.
* @li mul0_x: A Tensor. Must be one of the following types: float16, float32.
* @li mul1_x: A Tensor. Must be one of the following types: float16, float32.
* @li mul2_x: A Tensor. Must be one of the following types: float16, float32.
* @li mul3_x: A Tensor. Must be one of the following types: float16, float32.
* @li add2_y: A Tensor. Must be one of the following types: float16, float32. \n

*@par Outputs:
*Three outputs, including:
* @li output0: A Tensor. Must be one of the following types: float16, float32.
* @li output1: A Tensor. Must be one of the following types: float16, float32.
* @li output2: A Tensor. Must be one of the following types: float16, float32. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(AdamApplyOne)
    .INPUT(input0, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input1, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input2, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input3, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input4, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul0_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul1_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul2_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul3_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(add2_y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(output0, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(output1, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(output2, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(AdamApplyOne)

/**
*@brief A fusion operator for bert lamb. \n

*@par Inputs:
*Eleven inputs, including:
* @li input0: A Tensor. Must be one of the following types: float16, float32.
* @li input1: A Tensor. Must be one of the following types: float16, float32.
* @li input2: A Tensor. Must be one of the following types: float16, float32.
* @li input3: A Tensor. Must be one of the following types: float16, float32.
* @li input4: A Tensor. Must be one of the following types: float16, float32.
* @li mul0_x: A Tensor. Must be one of the following types: float16, float32.
* @li mul1_x: A Tensor. Must be one of the following types: float16, float32.
* @li mul2_x: A Tensor. Must be one of the following types: float16, float32.
* @li mul3_x: A Tensor. Must be one of the following types: float16, float32.
* @li mul4_x: A Tensor. Must be one of the following types: float16, float32.
* @li add2_y: A Tensor. Must be one of the following types: float16, float32. \n

*@par Outputs:
*Three outputs, including:
* @li output0: A Tensor. Must be one of the following types: float16, float32.
* @li output1: A Tensor. Must be one of the following types: float16, float32.
* @li output2: A Tensor. Must be one of the following types: float16, float32. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(AdamApplyOneWithDecayAssign)
    .INPUT(input0, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input1, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input2, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input3, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input4, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul0_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul1_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul2_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul3_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul4_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(add2_y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(output0, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(output1, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(output2, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(AdamApplyOneWithDecayAssign)

/**
*@brief A fusion operator for bert lamb. \n

*@par Inputs:
*Ten inputs, including:
* @li input0: A Tensor. Must be one of the following types: float16, float32.
* @li input1: A Tensor. Must be one of the following types: float16, float32.
* @li input2: A Tensor. Must be one of the following types: float16, float32.
* @li input3: A Tensor. Must be one of the following types: float16, float32.
* @li input4: A Tensor. Must be one of the following types: float16, float32.
* @li mul0_x: A Tensor. Must be one of the following types: float16, float32.
* @li mul1_x: A Tensor. Must be one of the following types: float16, float32.
* @li mul2_x: A Tensor. Must be one of the following types: float16, float32.
* @li mul3_x: A Tensor. Must be one of the following types: float16, float32.
* @li add2_y: A Tensor. Must be one of the following types: float16, float32. \n

*@par Outputs:
*Three outputs, including:
* @li output0: A Tensor. Must be one of the following types: float16, float32.
* @li output1: A Tensor. Must be one of the following types: float16, float32.
* @li output2: A Tensor. Must be one of the following types: float16, float32. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(AdamApplyOneAssign)
    .INPUT(input0, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input1, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input2, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input3, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input4, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul0_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul1_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul2_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul3_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(add2_y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(output0, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(output1, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(output2, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(AdamApplyOneAssign)

/**
*@brief A fusion operator for bert lamb. \n

*@par Inputs:
*Ten inputs, including:
* @li input0: A Tensor. Must be one of the following types: float16, float32.
* @li input1: A Tensor. Must be one of the following types: float16, float32.
* @li input2: A Tensor. Must be one of the following types: float16, float32.
* @li input3: A Tensor. Must be one of the following types: float16, float32.
* @li input4: A Tensor. Must be one of the following types: float16, float32.
* @li mul0_x: A Tensor. Must be one of the following types: float16, float32.
* @li mul1_x: A Tensor. Must be one of the following types: float16, float32.
* @li mul2_x: A Tensor. Must be one of the following types: float16, float32.
* @li mul3_x: A Tensor. Must be one of the following types: float16, float32.
* @li steps: A Tensor. Must be one of the following types: float16, float32.
* @li do_use_weight: A Tensor. Must be one of the following types: float16, float32.
* @li weight_decay_rate: A Tensor. Must be one of the following types: float16, float32.
* @li add2_y: A Tensor. Must be one of the following types: float16, float32. \n

*@par Outputs:
*Three outputs, including:
* @li output0: A Tensor. Must be one of the following types: float16, float32. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(LambApplyOptimizerAssign)
    .INPUT(grad, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(inputv, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(inputm, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input3, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul0_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul1_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul2_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mul3_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(add2_y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(steps, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(do_use_weight, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(weight_decay_rate, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(output0, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(inputv, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(inputm, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(LambApplyOptimizerAssign)

/**
*@brief A fusion operator for bert lamb. \n

*@par Inputs:
*Ten inputs, including:
* @li input0: A Tensor. Must be one of the following types: float16, float32.
* @li input1: A Tensor. Must be one of the following types: float16, float32.
* @li input2: A Tensor. Must be one of the following types: float16, float32.
* @li input3: A Tensor. Must be one of the following types: float16, float32.
* @li input4: A Tensor. Must be one of the following types: float16, float32.
* @li mul0_x: A Tensor. Must be one of the following types: float16, float32.
* @li mul1_x: A Tensor. Must be one of the following types: float16, float32.
* @li mul2_x: A Tensor. Must be one of the following types: float16, float32.
* @li mul3_x: A Tensor. Must be one of the following types: float16, float32.
* @li steps: A Tensor. Must be one of the following types: float16, float32.
* @li do_use_weight: A Tensor. Must be one of the following types: float16, float32.
* @li weight_decay_rate: A Tensor. Must be one of the following types: float16, float32.
* @li add2_y: A Tensor. Must be one of the following types: float16, float32. \n

*@par Outputs:
*No outputs
*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(LambApplyWeightAssign)
    .INPUT(input0, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input1, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input2, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input3, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input_param, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(input_param, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(LambApplyWeightAssign)

/**
*@brief Confuse select, maximum, greater and sqrt. \n

*@par Inputs:
*Four inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32.
* @li greater_zeros: A Tensor. Must be one of the following types: float16, float32.
* @li select_ones: A Tensor. Must be one of the following types: float16, float32.
* @li maximum_ones: A Tensor. Must be one of the following types: float16, float32. \n

*@par Outputs:
*y: A Tensor of the same type as "x". \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(ClipByNormNoDivSum)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(greater_zeros, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(select_ones, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(maximum_ones, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(ClipByNormNoDivSum)

/**
*@brief Confuse reducesumd and square. \n

*@par Inputs:
*x: A Tensor of type float16, float32. \n

*@par Attributes:
* Two attributes, including: \n
*@li axis: A optional listint, specifies the dimensions to reduce.
*@li keep_dims: A bool, specifying whether to keep dimensions for the output Tensor. Defaults to "false". \n

*@par Outputs:
*Two outputs, including: \n
*@li y1: A Tensor. Has the same type as "x".
*@li y2: A Tensor. Has the same type as "x".

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(SquareSumV2)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(y1, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(y2, TensorType({DT_FLOAT16,DT_FLOAT}))
    .REQUIRED_ATTR(axis, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(SquareSumV2)

/**
* @brief Confuse reducesumd and square.

* @par Inputs:
* x: A Tensor of type float16, float32. \n

* @par Attributes:
* Two attributes, including: \n
* @li axis: A optional listint, specifies the dimensions to reduce.
* @li keep_dims: A bool, specifying whether to keep dimensions for the output Tensor. Defaults to "false". \n

* @par Outputs:
* y: A Tensor. Has the same type as "x".

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(SquareSumV1)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .REQUIRED_ATTR(axis, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(SquareSumV1)

/**
*@brief Calculate square of Tensor and then reducesum

*@par Inputs:
*x1: A Tensor of type float32.
*x2: A Tensor of type float32. \n

*@par Outputs:
y1: A Tensor. Has the same type as "x1".The result of "x1".
y2: A Tensor. Has the same type as "x2".The result of "x2".

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(SquareSumAll)
    .INPUT(x1, TensorType({DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT}))
    .OUTPUT(y1, TensorType({DT_FLOAT}))
    .OUTPUT(y2, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(SquareSumAll)

/**
* @brief Confuse broadcast, addn and mul.

* @par Inputs:
* Three inputs, including:
* @li x1: A Tensor. Must be one of the following types:int32, int16,
* float16, float32.
* @li x2: A Tensor of the same type as "x1".
* @li x3: A Tensor of the same type as "x1". \n

* @par Outputs:
* y: A Tensor. Has the same type as "x1". \n

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(FusedMulAddN)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT16}))
    .INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT16}))
    .OP_END_FACTORY_REG(FusedMulAddN)

/**
* @brief Add 'bias' to 'x'.

* @par Inputs:
* Two inputs, including:
* @li x: An ND tensor of type float16 or float32.
* @li bias: An ND tensor of type float16 or float32. \n

* @par Attributes:
* @li axis: An optional int32 used to compute the shape of bias input from the online bottoms. Defaults to "1".
* @li num_axes: An optional int32 used to compute the shape of
* bias input from a Caffe model trained offline. Defaults to "1".
* @li bias_from_blob: An optional bool. If "true", bias is input from a Caffe model trained offline.
* If "false", bias is input from online bottoms. Defaults to "true". \n

* @par Outputs:
* y: An ND tensor of type float16 or float32. \n

* @attention Constraints:
* Assume that the shape length of "x" is "n" and that of "bias" is "m".
* @li "axis" is within the range [-n, n-1]. num_axes >= -1.
* @li If "bias_from_blob = true", "num_axes = -1", and "axis >= 0",
* the ith axis of "bias" and the (i+"axis")th axis of "x" must have the same size (0 <= i < n-axis).
* If "axis < 0", the ith axis of "bias" and the (i+n+"axis")th axis of "x" must have the same size (0 <= i < -axis).
* @li If "bias_from_blob = true" and "num_axes = 0", "bias" is a scalar with shape length 1 and dimension size 1.
* @li If "bias_from_blob = true", "num_axes > 0, and "axis >= 0",
* "axis + num_axes" must be less than or equal to "n" and the ith axis of "bias" and
* the (i+"axis")th axis of "x" must have the same size (0 <= i < num_axes).
* If "axis < 0", "n + axis + num_axes" must be less than or equal to "n" and
* the ith axis of "bias" and the (i+n+"axis")th axis of "x" must have the same size (0 <= i < num_axes).
* @li If "bias_from_blob = false", "bias" is not a scalar, and "axis >= 0",
* "axis + m" must be less than or equal to "n" and the ith axis of "bias" and
* the (i+"axis")th axis of "x" must have the same size (0 <= i < m).
* If "axis < 0", "n + axis + m" must be less than or equal to "n" and
* the ith axis of "bias" and the (i+n+"axis")th axis of "x" must have the same size (0 <= i < m). \n
* @par Third-party framework compatibility
* Compatible with the Caffe operator Bias.
*/

REG_OP(Bias)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16})) /* "First operand." */
    .INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16})) /* "Second operand." */
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))  /* "Result, has same element type as x" */
    .ATTR(axis, Int, 1)
    .ATTR(num_axes, Int, 1)
    .ATTR(bias_from_blob, Bool, true)
    .OP_END_FACTORY_REG(Bias)

/**
*@brief Function multiply gradients calculation.
output0 is the result of which input0 dot multily input1.
output1 is the result of which input0 dot multily input1, then reducesum it. \n

*@par Inputs:
*@li input0: A Tensor of input of mul, and dtype supports float16, float32.
*@li input1: A Tensor of input of mul and mul_1, and dtype supports float16, float32.
*@li input2: A Tensor of input of mul_1, and dtype supports float16, float32. \n

*@par Attributes:
*@li axes: The dimensions to reduce. Default:(), reduce all dimensions. \n
Only constant value is allowed.
*@li keep_dims: If true, keep these reduced dimensions and the length is 1. \n
If false, don’t keep these dimensions. Default:False. \n

*@par Outputs:
*@li output0: A Tensor result of which input0 dot multily input1.
*@li output1: A Tensor result of which input0 dot multily input1, then reducesum it.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(ConfusionMulGrad)
    .INPUT(input0, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input1, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input2, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(output0, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(output1, TensorType({DT_FLOAT16,DT_FLOAT}))
    .ATTR(axes, ListInt, {})
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ConfusionMulGrad)

/**
*@brief Function fused multiply l2 loss calculation. \n

*@par Inputs:
*@li x1: A Tensor of type float16, float32.
*@li x2: A Tensor of type float16, float32.
*@li x3: A Tensor of type float16, float32. \n

*@par Outputs:
*@li y1: A Tensor of shape and dtype of first output, which should have \n
shape (1,) and dtype as input.
*@li y2: A Tensor of shape and dtype of second output, should be same shape and type as input.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(FusedMulAddNL2loss)
    .INPUT(x1, TensorType::NumberType())
    .INPUT(x2, TensorType::NumberType())
    .INPUT(x3, TensorType::NumberType())
    .OUTPUT(y1, TensorType::NumberType())
    .OUTPUT(y2, TensorType::NumberType())
    .OP_END_FACTORY_REG(FusedMulAddNL2loss)

/**
*@brief Tests whether the input exceeds a threshold. \n

*@par Inputs:
* x: A Tensor with any format. Must be one of the following types: float16, float32. \n

*@par Attributes:
* threshold: A required float32. Defaults to "0.0". "x" is compared with "threshold", outputs "1" for inputs above threshold; "0" otherwise. \n

*@par Outputs:
* y: A Tensor with any format. Has the same type as the input. Must be one of the following types: float16, float32.
*@par Third-party framework compatibility
* Compatible with the Caffe operator Threshold.
*/

 REG_OP(Threshold)
     .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
     .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
     .ATTR(threshold, Float, 0.0)
     .OP_END_FACTORY_REG(Threshold);

/**
*@brief Returns the index number corresponding to the maximum value entered. \n

*@par Inputs:
*x: A tensor. Must be one of the following types: float16, float32. \n

*@par Attributes:
*@li axis: An optional int. Specify the axis to be cut at the input tensor. If this parameter is not provided, find the topk for each batch. Defaults to 10000
*@li out_max_val: An optional bool. Whether to output the maximum value. If it is True, the maximum value and index are output, otherwise only the index is output.
* Defaults to False
*@li topk: An optional int. It means the number of top tok in each axis (the value is greater than or equal to 1), and the value range must be in [1,x.shape(axis)].
* Defaults to 1

*@par Outputs:
*@li indices: A tensor of type float16, float32, int32. The index of the maximum value of the output.
*@li values: A tensor of type float16, float32.Output tensor, including maximum index or maximum value.
*@par Third-party framework compatibility
* Compatible with the Caffe operator ArgMax.
*/
REG_OP(ArgMaxWithK)
     .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
     .OUTPUT(indices, TensorType({DT_INT32, DT_FLOAT, DT_FLOAT16}))
     .OUTPUT(values, TensorType({DT_FLOAT, DT_FLOAT16}))
     .ATTR(axis, Int, 10000)
     .ATTR(out_max_val, Bool, false)
     .ATTR(topk, Int, 1)
     .OP_END_FACTORY_REG(ArgMaxWithK)

/**
*@brief Multiply tensor with scale. \n

*@par Inputs:
*One input, including:
*x: A Tensor. Must be one of the following types:int32,int16, float16, float32.

*@par Outputs:
*y: A Tensor. Has the same type and shape as "x1". \n

*@par Third-party framework compatibility:
* Compatible with the Pytorch operator muls.
*/
REG_OP(Muls)
     .INPUT(x, TensorType({DT_FLOAT,DT_INT16,DT_INT32,DT_FLOAT16}))
     .OUTPUT(y, TensorType({DT_FLOAT,DT_INT16,DT_INT32,DT_FLOAT16}))
     .REQUIRED_ATTR(value, Float)
     .OP_END_FACTORY_REG(Muls)

/**
*@brief Fill tensor with scale. \n

*@par Inputs:
*One input, including:
*x: A Tensor. Must be one of the following types:float32, float16, int64, int32, int16, bool.

*@par Outputs:
*y: A Tensor. Has the same type and shape as "x1". \n

*@par Third-party framework compatibility:
* Compatible with the Pytorch operator fills.
*/
REG_OP(Fills)
     .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT64, DT_INT32, DT_INT16, DT_BOOL}))
     .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT64, DT_INT32, DT_INT16, DT_BOOL}))
     .REQUIRED_ATTR(value, Float)
     .OP_END_FACTORY_REG(Fills)

/**
*@brief Add tensor with scale. \n

*@par Inputs:
*One input, including: \n
*x: A Tensor. Must be one of the following types:int32,int16, float16, float32. \n

*@par Attributes:
*value: A scale. Must be float. \n

*@par Outputs:
*y: A Tensor. Has the same type and shape as "x1". \n

*@par Third-party framework compatibility:
* Compatible with the Pytorch operator adds.
*/
 REG_OP(Adds)
     .INPUT(x, TensorType({DT_FLOAT,DT_INT16,DT_INT32,DT_FLOAT16}))
     .OUTPUT(y, TensorType({DT_FLOAT,DT_INT16,DT_INT32,DT_FLOAT16}))
     .REQUIRED_ATTR(value,Float)
     .OP_END_FACTORY_REG(Adds)

/**
* @brief Computes the product of x and y and returns 0 if the y is zero,
* even if x is NaN or infinite.

* @par Inputs:
* Two inputs, including: \n
* @li x1: A Tensor. Must be one of the following types:float16, float32,
* double, complex64, complex128.
* @li x2: A Tensor. Has the same type and shape as "x1". \n

* @par Outputs:
* y: A Tensor. Has the same type and shape as "x1". \n

* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator MulNoNan.
*/
 REG_OP(MulNoNan)
     .INPUT(x1, TensorType::NumberType()) /* "First operand." */
     .INPUT(x2, TensorType::NumberType()) /* "Second operand." */
     .OUTPUT(y, TensorType::NumberType())  /* "Result, has same element type as two inputs" */
     .OP_END_FACTORY_REG(MulNoNan)

/**
*@brief Add tensor with scale. \n

*@par Inputs:
* @li x1: A Tensor dtype of int32, float16, float32.
* @li x2: A Tensor dtype of int32, float16, float32. \n

*@par Attributes:
*alpha: Float scalar apply to x2:x2*alpha

*@par Outputs:
*y: A Tensor. should be same shape and type as "x1". \n

*@par Third-party framework compatibility:
* Compatible with the Pytorch operator Axpy.
*/
REG_OP(Axpy)
    .INPUT(x1, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16}))
    .REQUIRED_ATTR(alpha, Float)
    .OP_END_FACTORY_REG(Axpy)

/**
*@brief Creates a criterion that measures the loss given input tensors x1 x2 and a Tensor label y with values 1 or -1. \n

*@par Inputs:
*@li x1: A ND Tensor with one of the following types: int8, uint8, int32, float16, float32.
*@li x2: A ND Tensor with one of the following types: int8, uint8, int32, float16, float32.
*@li target: A ND Tensor with one of the following types: int8, int32, float16, float32. \n

*@par Attributes:
*@li margin: A optional float32. Defaults to "0.0".
*@li reduction: A optional string. Defaults to "mean". \n

*@par Outputs:
*@li y: A ND Tensor with Must be float32.
*@par Third-party framework compatibility
* Compatible with the PyTorch operator CosineEmbeddingLoss.
*/
REG_OP(CosineEmbeddingLoss)
    .INPUT(x1, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(x2, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(target, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(margin, Float, 0)
    .ATTR(reduction, String, "mean")
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(CosineEmbeddingLoss)

/**
*@brief Kullback-Leibler divergence. \n

*@par Inputs:
* Two inputs, including:
*@li x: Tensor of arbitrary shape.
*@li target: Tensor of the same shape and dtype as x. \n

*@par Attributes:
*reduction: An required "string", Specifies the reduction to apply to the output;
* Reduction only supports the two modes of "sum" and "batchmean". \n

*@par Outputs:
*y: A ND Tensor of the same dtype as x.
*@par Third-party framework compatibility
*Compatible with the PyTorch operator kl_div.
*/
REG_OP(KLDiv)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(target, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(reduction, String)
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(KLDiv)

/**
*@brief copy data from x to y.. \n

*@par Inputs:
*One inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32, int8, uint8, int32, bool. \n

*@par Outputs:
*y: A Tensor. Has the same type as "x". \n

*@par Third-party framework compatibility

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(TensorMove)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8, DT_BOOL}))
    .OP_END_FACTORY_REG(TensorMove)

/**
*@brief copy data from x to x. \n

*@par Inputs:
*One inputs, including:
*x: A Tensor. Must be one of the following types: float16, float32, int8, uint8, int16, uint16, int32, uint32, int64, uint64. \n

*@par Outputs:
*output_x: A Tensor. Has the same type as "x". \n

*@par Third-party framework compatibility
*/
REG_OP(TensorRedirect)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8,
                           DT_INT64, DT_INT16, DT_UINT16, DT_UINT64, DT_UINT32}))
    .OUTPUT(output_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8,
                           DT_INT64, DT_INT16, DT_UINT16, DT_UINT64, DT_UINT32}))
    .OP_END_FACTORY_REG(TensorRedirect)

/**
* @brief Performs the element-wise division of tensor x1 by tensor x2,
* multiply the result by the scalar value and add it to tensor input_data.

* @par Inputs:
* Four inputs, including:
* @li input_data: A mutable input Tensor. Must be one of the following types:
*     float16, float32, double, int64.
* @li x1: A mutable input Tensor of the same type as input_data.
* @li x2: A mutable input Tensor of the same type as input_data.
* @li value: A mutable input Tensor. Must be one of the following types:
*     float16, float32, double, int64, int32. \n


* @par Outputs:
* y: A mutable Tensor. Has the same type as input_data. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator Addcdiv(version-1.5.0).
*/
REG_OP(Addcdiv)
    .INPUT(input_data, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64}))
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64}))
    .INPUT(value, TensorType({ DT_FLOAT16, DT_FLOAT, DT_INT32, DT_DOUBLE, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64}))
    .OP_END_FACTORY_REG(Addcdiv)

/**
* @brief Performs the element-wise multiplication of tensor x1 by tensor x2,
* multiply the result by the scalar value and add it to tensor input_data

* @par Inputs:
* Four inputs, including:
* @li input_data: A mutable input Tensor. Must be one of the following types:
*     float16, float32, double, int64, int8, int32, uint8.
* @li x1: A mutable input Tensor of the same type as input_data.
* @li x2: A mutable input Tensor of the same type as input_data.
* @li value: A tensor which includes only one element of the same type as input_data. \n

* @par Outputs:
* y: A mutable output Tensor. Has the same type as input_data. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator Addcmul.
*/
REG_OP(Addcmul)
    .INPUT(input_data, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8, DT_DOUBLE, DT_INT64}))
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8, DT_DOUBLE, DT_INT64}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8, DT_DOUBLE, DT_INT64}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8, DT_DOUBLE, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8, DT_DOUBLE, DT_INT64}))
    .OP_END_FACTORY_REG(Addcmul)

/**
* @brief Computes the result of x2 * alpha + x1.

* @par Inputs:
* @li x1: An ND tensor of type float16, float32, int32.
* @li x2: An ND tensor of type float16, float32, int32.
* @li alpha: A scalar tensor of type float16, float32. \n

* @par Outputs:
* y: An ND tensor tensor with the same shape and type as "x1". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator Axpy.
*/
REG_OP(AxpyV2)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .INPUT(alpha, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OP_END_FACTORY_REG(AxpyV2)

/**
* @brief Add the partial values of two tensors.

* @par Inputs:
* @li x1: A Tensor in 5HD, and must be one of the following types: float16,
* float32. \n
* @li x2: A Tensor of the same type as "x1", and the same shape as "x1",
* except for the C1 value. \n

* @par Attributes:
* @li x1_c1_offset: A required int. Offset value of C1 in "x1". \n
* @li x2_c1_offset: A required int. Offset value of C1 in "x2". \n
* @li c1_len: A required int. C1 len of "y". The value must be less than
* the difference between C1 and offset in "x1" and "x2". \n

* @par Outputs:
* y:  A Tensor of the same type as "x1", and the same shape as "x1",
* except for the C1 value. Record the result after adding. \n
*/
REG_OP(StrideAdd)
    .INPUT(x1, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .INPUT(x2, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .OUTPUT(y, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .REQUIRED_ATTR(x1_c1_offset, Int)
    .REQUIRED_ATTR(x2_c1_offset, Int)
    .REQUIRED_ATTR(c1_len, Int)
    .OP_END_FACTORY_REG(StrideAdd)

/**
* @brief Compare two tensors are totally equal or not, only output a bool value"

* @par Inputs:
* Two inputs, including:
* @li input_x: A Tensor. the first tensor. \n
* @li input_y: A Tensor. the second tensor. \n

* @par Outputs:
*output_z: A Tensor. Bool type, compare result of the two inputs. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch equal operator. \n
*/
REG_OP(TensorEqual)
    .INPUT(input_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT32, DT_INT8, DT_UINT8}))
    .INPUT(input_y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT32, DT_INT8, DT_UINT8}))
    .OUTPUT(output_z, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(TensorEqual)

/**
 * @brief Element-wise min of each of the input tensors (with Numpy-style broadcasting support).
 * All inputs and outputs must have the same data type. This operator supports multidirectional
 * (i.e., Numpy-style) broadcasting
 *
 * @par Inputs:
 * one input including:
 * x: dynamic input A Tensor. Must be one of the following types: float32, float16, double, int32, int64
 *
 * @par Outputs:
 * one output including:
 * y:A Tensor of the same type as x
 *
 */
REG_OP(MaxN)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_FLOAT64, DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_FLOAT64, DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(MaxN)


/**
 * @brief Calculates x * maske * value.
 *
 * @par Inputs:
 * @li x: An tensor of type float16 or float32, specifying the input to the data layer.
 * @li mask: An tensor of type int8 or float16 or float32, be same shape with x. \n
 *
 * @par Attributes:
 * value: A optional float. \n
 *
 * @par Outputs:
 * y: The output tensor of type float16 or float32.
 @ li y:A Tensor of the same type and shape as x
 *
 */
REG_OP(MaskedScale)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .INPUT(mask, TensorType({DT_INT8, DT_FLOAT16, DT_FLOAT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .REQUIRED_ATTR(value, Float)
    .OP_END_FACTORY_REG(MaskedScale)

/**
 * @brief Calculate the lerp function. \n

 * @par Inputs:
 * Three inputs, including:
 * @li start: A tensor. Must be one of the following types:
 *     float16, float32. \n
 * @li end: A tensor. Must be one of the following types:
 *     float16, float32. \n
 * @li weight: A tensor. Must be one of the following types:
 *     float16, float32. \n

 * @par Outputs:
 * y: A Tensor with the same type and shape of input_x's. \n

 * @par Third-party framework compatibility
 * Compatible with the Pytorch operator Lerp. \n
 */
REG_OP(Lerp)
    .INPUT(start, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(end, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(weight, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(Lerp)

/**
*@brief Returns the num value of abs(x1-x2) > atol+rtol*abs(x2) element-wise. \n

*
*@par Inputs:
*@li x1: A tensor. Must be one of the following types: float32, int32, uint8, int8, float16
*@li x2: A tensor of the same type as "x1".
*
*@par Attributes:
* atol: Defaults to "1e-05".
* rtol: Defaults to "1e-03".
*
*@par Outputs:
* num: A tensor of type float32.
*
*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*
*/
REG_OP(DataCompare)
  .INPUT(x1, TensorType({ DT_FLOAT16, DT_FLOAT,DT_INT8, DT_UINT8, DT_INT32 }))
  .INPUT(x2, TensorType({ DT_FLOAT16, DT_FLOAT,DT_INT8, DT_UINT8, DT_INT32 }))
  .OUTPUT(num, TensorType({DT_FLOAT}))
  .ATTR(atol, Float, 1e-5)
  .ATTR(rtol, Float, 1e-3)
  .OP_END_FACTORY_REG(DataCompare)

/**
*@brief Hardmax(element in input, axis) = 1 if the element is the first maximum value along the specified axis, 0
*otherwise The input does not need to explicitly be a 2D vector.The "axis" attribute indicates the dimension along
*which Hardmax will be performed.The output tensor has the same shape and contains the Hardmax values of the
*corresponding input.
*
*@par Inputs:
*one input including:
*x: input A Tensor.Must be one of the following types:float32,float16
*
*@par Attributes:
*axis:A required int attribute that decides which dimension will be used to cal the hard_max
*
*@par Outputs:
*one output including:
*y:A Tensor of the same type as x
*
*/
REG_OP(HardMax)
    .INPUT(x, TensorType({ DT_FLOAT16, DT_FLOAT }))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(axis, Int, -1)
    .OP_END_FACTORY_REG(HardMax)

/**
* @brief Computes the dot product (inner product) of two tensors. This function does not broadcast.

* @par Inputs:
* Two inputs, including:
* @li input_x: A Tensor. the first tensor must be 1d. \n
* @li input_y: A Tensor. the second tensor must be 1d. \n

* @par Outputs:
* output: A Tensor. Result of the two inputs, must be 1d. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch dot operator. \n
*/
REG_OP(Dot)
    .INPUT(input_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_UINT8, DT_INT8, DT_INT32}))
    .INPUT(input_y, TensorType({DT_FLOAT, DT_FLOAT16, DT_UINT8, DT_INT8, DT_INT32}))
    .OUTPUT(output, TensorType({DT_FLOAT, DT_FLOAT16, DT_UINT8, DT_INT8, DT_INT32}))
    .OP_END_FACTORY_REG(Dot)

/**
*@brief Returns a new tensor with boolean elements representing \n
*if each element of input is “close” to the corresponding element of other \n

*@par Inputs:
*Two inputs, including:
* @li x1: A tensor. Must be one of the following types:
*     float16, float32, int32. \n
* @li x2: A tensor with the same type and shape of x1's. \n

*@par Attributes:
*@li rtol: An optional float.Defaults to 1e-05. \n
*@li atol: An optional float.Defaults to 1e-08. \n
*@li equal_nan: An optional bool.Defaults to false. \n

*@par Outputs:
*y: A Tensor bool with the same shape of x1's. \n

*@par Third-party framework compatibility
*Compatible with the Pytorch operator isclose. \n
*/
REG_OP(IsClose)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .ATTR(rtol, Float, 1e-05)
    .ATTR(atol, Float, 1e-08)
    .ATTR(equal_nan, Bool, false)
    .OP_END_FACTORY_REG(IsClose)

/**
* @brief Returns the reverse tensor of the ArgMax operator of a tensor. \n

* @par Inputs:
* three input, including:
* var: A Tensor of type float16, float32, int32 or int8. \n
* indices: A Tensor of type int32. \n
* updates: A Tensor of type float16, float32, int32 or int8. \n

* @par Attributes:
* @li dimension: An integer of type int, specifying the axis information of the index with the maximum value.\n

* @par Outputs:
* y: A Tensor of type float16, float32, int32 or int8. \n
*
*@attention Constraints:
*@li indices: only support int32,and shape same to "updates"
*@li The value range of "dimension" is [-dims, dims - 1]. "dims" is the dimension length of "x".
*@li y:A Tensor, the type and shape is same to "var" \n

*@par Third-party framework compatibility
* not support all scene like pytorch operator scatter
* exp:
* var.shape=[2,3,4,5], dim=2, the shape of indices and updates should be [2,3,5]
* not support the shape of indices and updates is [2,3,2,5] like pytorch operator scatter. \n
*/
REG_OP(ArgMaxGrad)
    .INPUT(var, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(updates, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8}))
    .REQUIRED_ATTR(dimension, Int)
    .OP_END_FACTORY_REG(ArgMaxGrad)

/**
* @brief Returns the reverse tensor of the ArgMax operator of a tensor. \n

* @par Inputs:
* three input, including:
* var: A Tensor of type float16, float32, int32 or int8. \n
* indices: A Tensor of type int32. \n
* updates: A Tensor of type float16, float32, int32 or int8. \n
* assist: A Tensor of int32,also a assist matrix and it's shape must match the shape of var \n

* @par Attributes:
* @li dimension: An integer of type int, specifying the axis information of the index with the maximum value.\n

* @par Outputs:
* y: A Tensor of type float16, float32, int32 or int8. \n

*@attention Constraints:
*@li indices: only support int32,and shape same to "updates"
*@li The value range of "dimension" is [-dims, dims - 1]. "dims" is the dimension length of "x".
*@li y:A Tensor, the type and shape is same to "var" \n

*@par Third-party framework compatibility
* not support all scene like pytorch operator scatter
* exp:
* var.shape=[2,3,4,5], dim=2, the shape of indices and updates should be [2,3,5]
* not support the shape of indices and updates is [2,3,2,5] like pytorch operator scatter. \n
*/
REG_OP(ArgMaxGradD)
    .INPUT(var, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(updates, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8}))
    .INPUT(assist, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8}))
    .REQUIRED_ATTR(dimension, Int)
    .OP_END_FACTORY_REG(ArgMaxGradD)

/**
*@brief Calculates the reversed outputs of the function "AddMatMatElements"
*  c = c * beta + alpha * a * b

*@par Inputs:
*Three inputs, including:
* @li c: A mutable Tensor. Must be one of the following types:
*     float16, float32.
* @li a: A mutable Tensor of the same type as "c".
* @li b: A mutable Tensor of the same type as "c".
* @li beta: A mutable scalar of the same type as "c".
* @li alpha: A mutable scalar of the same type as "c". \n

*@par Outputs:
* @li c: A mutable Tensor. Has the same type as "c". \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator AddMatMatElements.
*/
REG_OP(AddMatMatElements)
    .INPUT(c, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(a, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(b, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(alpha, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(c, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(AddMatMatElements)

/**
*@brief Returns cosine similarity between x1 and x2,computed along dim. \n

*@par Inputs:
*Two inputs, including:
* @li input_x1: A tensor. Must be the following types: float32.
* @li input_x2: A tensor. Must of the following types: float32. \n

* @par Attributes:
* @li dim:The type is Int and the default value is 1.
* @li eps:The type is Float and the default value is 1e-8. \n

*@par Outputs:
* output_y: A Tensor with the same type of input_x's. \n

*@par Third-party framework compatibility
*Compatible with the Pytorch operator CosineSimilarity. \n
*/
REG_OP(CosineSimilarity)
    .INPUT(input_x1, TensorType({DT_FLOAT}))  /* "First operand." */
    .INPUT(input_x2, TensorType({DT_FLOAT}))  /* "Second operand." */
    .OUTPUT(output_y, TensorType({DT_FLOAT})) /* "Result, has same element type as two inputs" */
    .ATTR(dim, Int, 1)
    .ATTR(eps, Float, 1e-8)
    .OP_END_FACTORY_REG(CosineSimilarity)

/**
*@brief count adam result. \n

*@par Inputs:
*eleven inputs, including:
* @li var: A Tensor. Support float16/float32.\n
* @li m: A Tensor. Datatype and shape are same as exp_avg.\n
* @li v: A Tensor. Datatype and shape are same as exp_avg.\n
* @li lr: A Tensor. Datatype is same as exp_avg. Shape (1, ).\n
* @li beta1: A Tensor. Datatype is same as exp_avg. Shape (1, ).\n
* @li beta2: A Tensor. Datatype is same as exp_avg. Shape (1, ).\n
* @li epsilon: A Tensor. Datatype is same as exp_avg. Shape (1, ).\n
* @li grad: A Tensor. Datatype and shape are same as exp_avg.\n
* @li max_grad_norm: A Tensor. Datatype is same as exp_avg. Shape (1, ).\n
* @li global_grad_norm: A Tensor. Datatype is same as exp_avg. Shape (1, ).\n
* @li weight_decay: A Tensor. Datatype is same as exp_avg. Shape (1, ).\n
* @li step_size: A Optional Tensor. Datatype is same as exp_avg. Shape (1, ).\n

* @par Attributes:
* @li adam_mode: An optional bool. Defaults to "adam". \n

*@par Outputs:
*three inputs, including:
* @li var: A Tensor. Datatype and shape are same as exp_avg.\n
* @li m: A Tensor. Datatype and shape are same as exp_avg.\n
* @li v: A Tensor. Datatype and shape are same as exp_avg.\n
*/
REG_OP(ApplyAdamV2)
    .INPUT(var, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .INPUT(m, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .INPUT(v, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .INPUT(lr, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .INPUT(beta1, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .INPUT(beta2, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .INPUT(epsilon, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .INPUT(grad, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .INPUT(max_grad_norm, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .INPUT(global_grad_norm, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .INPUT(weight_decay, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .OPTIONAL_INPUT(step_size, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .OUTPUT(var, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .OUTPUT(m, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .OUTPUT(v, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .ATTR(adam_mode, String, "adam")
    .OP_END_FACTORY_REG(ApplyAdamV2)

/**
* @brief Computes Dawsn operation.  \n

*
* @par Inputs:
* x: A tensor. Must be one of the following types: bfloat16, float16, float32, float64.
*
* @par Outputs:
* y: A tensor. Has the same type as "x".
*
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Dawsn.
*
*/
REG_OP(Dawsn)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(Dawsn)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_ELEWISE_CALCULATION_OPS_H_
