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

#ifndef GE_OP_MATH_OPS_H_
#define GE_OP_MATH_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief Computes the output as (shift + scale * x) ^ power.

*@par Inputs:
* x: A Tensor of type float16 or float32.

*@par Attributes:
*@li power: Optional. Defaults to 1.0.
*@li scale: Optional. Defaults to 1.0.
*@li shift: Optional. Defaults to 0.0.

*@par Outputs:
* y: A Tensor. Has the same type and shape as "x".
*/

REG_OP(Power)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(power, Float, 1.0)
    .ATTR(scale, Float, 1.0)
    .ATTR(shift, Float, 0.0)
    .OP_END_FACTORY_REG(Power);

/**
*@brief Compute the lower regularized incomplete Gamma function P(a, x).

*@par Inputs:
*The input a and x must have the same type. Inputs include: \n
*@li a:A Tensor. Must be one of the following types: float, double.
*@li x:A Tensor. Must have the same type as a.

*@par Outputs:
*z:A Tensor. Has the same type as a.

*/

REG_OP(Igamma)
    .INPUT(a, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(z, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(Igamma)

/**
*@brief Compute the upper regularized incomplete Gamma function Q(a, x).

*@par Inputs:
*The input a and x must have the same type. Inputs include: \n
*@li a:A Tensor. Must be one of the following types: float, float64.
*@li x:A Tensor. Must have the same type as a.

*@par Outputs:
*z:A Tensor. Has the same type as a.

*/

REG_OP(Igammac)
    .INPUT(a, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(z, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(Igammac)

/**
*@brief Compare values of input to threshold and pack resulting bits into \n
a uint8.

*@par Inputs:
*The input size must be a non-negative int32 scalar Tensor. Inputs include: \n
*@li input:Values to compare against threshold and bitpack.
*@li threshold:Threshold to compare against.

*@par Outputs:
*y:The bitpacked comparisons.

*@attention Constraints: \n
*Currently, the innermost dimension of the tensor must be divisible by 8. \n

*/

REG_OP(CompareAndBitpack)
    .INPUT(x, TensorType({ DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, \
        DT_INT16, DT_INT32, DT_INT64, DT_BOOL }))
    .INPUT(threshold, TensorType({ DT_FLOAT, DT_FLOAT16, DT_DOUBLE, \
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_BOOL }))
    .OUTPUT(y, TensorType(DT_UINT8))
    .OP_END_FACTORY_REG(CompareAndBitpack)

/**
*@brief Counts the number of occurrences of each value in an integer array. \n
Outputs a vector with length size and the same dtype as weights. If weights \n
are empty, then index i stores the number of times the value i is counted in \n
arr. If weights are non-empty, then index i stores the sum of the value in \n
weights at each index.

*@par Inputs:
*The input size must be a non-negative int32 scalar Tensor. Inputs include: \n
*@li array:int32 Tensor.
*@li size:non-negative int32 scalar Tensor.
*@li weights: is an int32, int64, float32, or double Tensor with the same \n
shape as arr, or a length-0 Tensor, in which case it acts as all weights \n
equal to 1.

*@par Outputs:
*bins:1D Tensor with length equal to size. The counts or summed weights for \n
each value in the range [0, size).

*/

REG_OP(Bincount)
    .INPUT(array, TensorType(DT_INT32))
    .INPUT(size, TensorType(DT_INT32))
    .INPUT(weights, TensorType({ DT_FLOAT, DT_INT32, DT_INT64, DT_DOUBLE }))
    .OUTPUT(bins, TensorType({ DT_FLOAT, DT_INT32, DT_INT64, DT_DOUBLE }))
    .OP_END_FACTORY_REG(Bincount)

/**
*@brief Compute the regularized incomplete beta integral.

*@par Inputs:
*The input b and x must have the same types as a. Inputs include: \n
*@li a:A Tensor. Must be one of the following types: float32, double.
*@li b:A Tensor. Must have the same type as a.
*@li x:A Tensor. Must have the same type as a.

*@par Outputs:
*z:A Tensor. Has the same type as a.

*/

REG_OP(Betainc)
    .INPUT(a, TensorType({DT_DOUBLE, DT_FLOAT}))
    .INPUT(b, TensorType({DT_DOUBLE, DT_FLOAT}))
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT}))
    .OUTPUT(z, TensorType({DT_DOUBLE, DT_FLOAT}))
    .OP_END_FACTORY_REG(Betainc)

/**
*@brief Compute the Hurwitz zeta function

*@par Inputs:
*The input q must be the same type as x. Inputs include: \n
*@li x:A Tensor. Must be one of the following types: float32, double.
*@li q:A Tensor. Must have the same type as x.

*@par Outputs:
*z:A Tensor. Has the same type as x.

*@attention Constraints: \n
*The implementation for Zeta on Ascend uses ai cpu, with bad performance. \n

*/

REG_OP(Zeta)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT}))
    .INPUT(q, TensorType({DT_DOUBLE, DT_FLOAT}))
    .OUTPUT(z, TensorType({DT_DOUBLE, DT_FLOAT}))
    .OP_END_FACTORY_REG(Zeta)

/**
*@brief Bucketizes 'input' based on 'boundaries'. For example, if the inputs \n
are boundaries = [0, 10, 100] input = [[-5, 10000] [150, 10] [5, 100]] then \n
the output will be output = [[0, 3] [3, 2] [1, 3]]

*@par Inputs:
*The dtype of input x must be int or float. Inputs include: \n
*x:Any shape of Tensor contains with int or float type.

*@par Attributes:
*boundaries:A sorted list of floats gives the boundary of the buckets.

*@par Outputs:
*y:Same shape with 'input', each value of input replaced with bucket index.

*/

REG_OP(Bucketize)
    .INPUT(x, TensorType({DT_INT32, DT_INT64, DT_DOUBLE, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .REQUIRED_ATTR(boundaries, ListFloat)
    .OP_END_FACTORY_REG(Bucketize)

/**
*@brief Computes the sum along sparse segments of a tensor.

*@par Inputs:
*The input indices and segment_ids must have same rank. Inputs include: \n
*@li x:A Tensor. Must be one of the following types: float, double, int32, \n
uint8, int16, int8, int64, uint16, uint32, uint64.
*@li indices: A Tensor. Must be one of the following types: int32, int64. \n
A 1-D tensor. Has same rank as segment_ids.
*@li segment_ids: A Tensor of type int32. A 1-D tensor. Values should be \n
sorted and can be repeated.

*@par Outputs:
*y:A Tensor. Has the same type as x.

*/

REG_OP(SparseSegmentSum)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16,
        DT_INT32, DT_INT64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(segment_ids, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16,
        DT_INT32, DT_INT64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(SparseSegmentSum)

/**
*@brief Computes the mean along sparse segments of a tensor.

*@par Inputs:
*The input indices and segment_ids must have same rank. Inputs include: \n
*@li x: A Tensor. Must be one of the following types: float, double.
*@li indices: A Tensor. Must be one of the following types: int32, int64. \n
A 1-D tensor. Has same rank as segment_ids.
*@li segment_ids: A Tensor of type int32. A 1-D tensor. Values should be \n
sorted and can be repeated.

*@par Outputs:
*y:A Tensor. Has the same type as x.

*/

REG_OP(SparseSegmentMean)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(segment_ids, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(SparseSegmentMean)

/**
*@brief Computes gradients for SparseSegmentMean.

*@par Inputs:
*The input grad must have be type float or double. Inputs include: \n
*@li grad: A Tensor. Must be one of the following types: float, double. \n
gradient propagated to the SparseSegmentMean op.
*@li indices: A Tensor. Must be one of the following types: int32, int64. \n
indices passed to the corresponding SparseSegmentMean op.
*@li segment_ids: A Tensor of type int32. segment_ids passed to the \n
corresponding SparseSegmentMean op.
*@li output_dim0: A Tensor of type int32. dimension 0 of "x" passed to \n
SparseSegmentMean op.

*@par Outputs:
*y:A Tensor. Has the same type as grad.

*/

REG_OP(SparseSegmentMeanGrad)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(segment_ids, TensorType({DT_INT32}))
    .INPUT(output_dim0, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(SparseSegmentMeanGrad)

/**
*@brief Computes the gradient of igamma(a, x) wrt a

*@par Inputs:
*The input a and x must have the same type. Inputs include: \n
*@li a:A Tensor. Must be one of the following types: float32, double.
*@li x:A Tensor. Must have the same type as a.

*@par Outputs:
*y:A Tensor. Has the same type as a.

*/

REG_OP(IgammaGradA)
    .INPUT(a, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(z, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(IgammaGradA)

/**
*@brief Initialize data process channel.

*@par Attributes:
*channel_name: A string. Default "".

*/

REG_OP(InitData)
    .ATTR(channel_name, String, "")
    .OP_END_FACTORY_REG(InitData)

/**
*@brief Get the next batch of data in data processing.

*@par Attributes:
*@li output_types: A nested structure of DType objects corresponding to each \n
component of an element of this dataset.
*@li output_shapes: A nested structure of TensorShape objects corresponding \n
to each component of an element of this dataset.
*@li channel_name: A string. Default "".

*@par Outputs:
*y:A nested structure of Tensor objects.

*/

REG_OP(GetNext)
    .DYNAMIC_OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64,
                                        DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL}))
    .ATTR(output_types, ListInt, {})
    .ATTR(output_shapes, ListListInt, {})
    .ATTR(output_num, Int, 1)
    .ATTR(channel_name, String, "")
    .OP_END_FACTORY_REG(GetNext)
/**
*@brief: Computes the Gauss error function of `x` element-wise.

*@par Inputs:\n
*x: A Tensor of type float16 or float32.

*@par Outputs:
*y: A Tensor. Has the same type as "x".

*/
REG_OP(Erf)
    .INPUT(x, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(Erf)

/**
*@brief: Computes the Gauss complementary error function of "x" element-wise.

*@par Inputs:\n
*x: A Tensor of type float16 or float32.

*@par Outputs:
*y: A Tensor. Has the same type as "x".

*/
REG_OP(Erfc)
    .INPUT(x, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(Erfc)

/**
*@brief This operation returns a rank 1 histogram counting the number of entries in `values` \n
*  that fell into every bin.The bins are equal width and determined by the arguments \n
*  'value_range' and 'nbins'. \n

*@par Inputs:
*Three inputs, including: \n
*@li x: A Tensor of type float32,float16,int32.
*@li range: A Tensor of type float32,float16,int32.
*@li nbins: A Tensor of type int32.

*@par Attributes:
* dtype: An optional attribute. Defaults to "int32".

*@par Outputs:
*y: A Tensor. A Tensor of type int32.

*/
REG_OP(HistogramFixedWidth)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .INPUT(range, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .INPUT(nbins, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .ATTR(dtype, String, "int32")
    .OP_END_FACTORY_REG(HistogramFixedWidth)

/**
*@brief This operation returns a rank 1 histogram counting the number of entries in `values` \n
*  that fell into every bin.The bins are equal width and determined by the arguments \n
*  'value_range' and 'nbins'. \n

*@par Inputs:
*Two inputs, including: \n
*@li x: A Tensor of type float32,float16,int32.
*@li range: A Tensor of type float32,float16,int32.

*@par Attributes:
*@li dtype: An optional attribute. Defaults to "int32".
*@li nbins: A required attribute,the type is int32.

*@par Outputs:
*y: A Tensor. A Tensor of type int32.

*/
REG_OP(HistogramFixedWidthD)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .INPUT(range, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .REQUIRED_ATTR(nbins, Int)
    .ATTR(dtype, String, "int32")
    .OP_END_FACTORY_REG(HistogramFixedWidthD)

/**
*@brief Returns the next representable value of x1 in the direction of x2, element-wise.

*@par Inputs:
*The input X1 and x2 must have the same type. Inputs include: \n
*@li x1:A Tensor. Must be one of the following types: float32, double.
*@li x2:A Tensor. Must have the same type as x1.

*@par Outputs:
*output:A Tensor. Has the same type as x1.

*/
REG_OP(NextAfter)
    .INPUT(x1, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(output, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(NextAfter)

/**
 * *@brief Compute element-wise finiteness, return a boolean tensor.
 *
 * *@par Inputs:
 * *x:A Tensor.
 *
 * *@par Outputs:
 * *y:A Tensor. Has the same shape as x.
 *
 * */
REG_OP(IsFinite)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(IsFinite)

/**
 * *@brief Computes the complex absolute value of a tensor.
 *
 * *@par Inputs:
 * *x:A Tensor.
 *
 * *@par Outputs:
 * *y:A tensor of type `float` or `double` that is the absolute value of each element in `x`.
 *
 * */
REG_OP(ComplexAbs)
    .INPUT(x, TensorType({DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .ATTR(Tout, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(ComplexAbs)

/**
 * *@brief Returns which elements of x are NaN.
 *
 * *@par Inputs:
 * *x:A Tensor.
 *
 * *@par Outputs:
 * *y:A Tensor. Has the same shape as x.
 *
 * */
REG_OP(IsNan)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(IsNan)

/**
 * *@brief Returns the real part of a complex number.
 *
 * *@par Inputs:
 * *input:A Tensor.
 *
 * *@par Outputs:
 * *output:A Tensor. Has the same shape as input.
 *
 * */
REG_OP(Real)
    .INPUT(input, TensorType({DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(output, TensorType({DT_FLOAT, DT_DOUBLE}))
    .ATTR(Tout, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(Real)

/**
 * *@brief Returns the complex conjugate of a complex number.
 *
 * *@par Inputs:
 * *input:A Tensor.
 *
 * *@par Outputs:
 * *output:A Tensor. Has the same shape as input.
 *
 * */
REG_OP(Conj)
    .INPUT(input, TensorType({DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(output, TensorType({DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Conj)

/**
 * *@brief The negative log likelihood loss.
 *
 * *@par Inputs:
 * *The input x and weight must have the same type. Inputs include: \n
 * *@li x:A Tensor. Must be the type: float32.
 * *@li target:A Tensor. Must be the type: int32.
 * *@li weight:A Tensor. Must be the type: float32.
 *
 * *@par Attributes:
 * *@li reduction: An optional attribute. Defaults to "mean".
 *
 * *@par Outputs:
 * *Two outputs, including:
 * *@li y: A Tensor. Must be the following type: float32.
 * *@li total_weight: A Tensor. Must be the type: float32.
 *
 * */
REG_OP(NLLLoss)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(target, TensorType({DT_INT32}))
    .INPUT(weight, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OUTPUT(total_weight, TensorType({DT_FLOAT}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(NLLLoss)

/**
 * *@brief The negative log likelihood loss grad.

 * *@par Inputs:
 * *Inputs include:
 * *@li x:A Tensor. Must be the type: float32.
 * *@li y_grad:A Tensor. Must be the type: float32.
 * *@li target:A Tensor. Must be the type: int32.
 * *@li weight:A Tensor. Must be the type: float32.
 * *@li total_weight:A Tensor. Must be the type: float32.
 *
 * *@par Attributes:
 * *@li reduction: An optional attribute. Defaults to "mean".
 *
 * *@par Outputs:
 * *One outputs, including:
 * *@li x_grad: A Tensor. Must be the following type: float32.
 *
 * */
REG_OP(NLLLossGrad)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(y_grad, TensorType({DT_FLOAT}))
    .INPUT(target, TensorType({DT_INT32}))
    .INPUT(weight, TensorType({DT_FLOAT}))
    .INPUT(total_weight, TensorType({DT_FLOAT}))
    .OUTPUT(x_grad, TensorType({DT_FLOAT}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(NLLLossGrad)
}  // namespace ge

#endif  // GE_OP_MATH_OPS_H_
