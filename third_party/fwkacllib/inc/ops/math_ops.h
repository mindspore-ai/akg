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
 * \file math_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_MATH_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_MATH_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief Computes the output as (shift + scale * x) ^ power . \n

*@par Inputs:
* x: A Tensor of type float16 or float32 . \n

*@par Attributes:
*@li power: Optional. Must be one of the following types: float32. Defaults to 1.0.
*@li scale: Optional. Must be one of the following types: float32. Defaults to 1.0.
*@li shift: Optional. Must be one of the following types: float32. Defaults to 0.0 . \n

*@par Outputs:
* y: A Tensor. Has the same type and shape as "x".
*@par Third-party framework compatibility
* Compatible with the Caffe operator Power.
*/

REG_OP(Power)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(power, Float, 1.0)
    .ATTR(scale, Float, 1.0)
    .ATTR(shift, Float, 0.0)
    .OP_END_FACTORY_REG(Power);

/**
*@brief Compute the lower regularized incomplete Gamma function P(a, x) . \n

*@par Inputs:
*The input a and x must have the same type. Inputs include:
*@li a:A Tensor. Must be one of the following types: float, double.
*@li x:A Tensor. Must have the same type as a . \n

*@par Outputs:
*z:A Tensor. Has the same type as a . \n

*@par Third-party framework compatibility.
*Compatible with tensorflow Igamma operator.
*/

REG_OP(Igamma)
    .INPUT(a, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(z, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(Igamma)

/**
*@brief Compute the upper regularized incomplete Gamma function Q(a, x) . \n

*@par Inputs:
*The input a and x must have the same type. Inputs include:
*@li a:A Tensor. Must be one of the following types: float, float64.
*@li x:A Tensor. Must have the same type as a . \n

*@par Outputs:
*z:A Tensor. Has the same type as a . \n

*@par Third-party framework compatibility.
*Compatible with tensorflow Igammac operator.
*/

REG_OP(Igammac)
    .INPUT(a, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(z, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(Igammac)

/**
*@brief Compare values of input to threshold and pack resulting bits into
a uint8 . \n

*@par Inputs:
*The input size must be a non-negative int32 scalar Tensor. Inputs include:
*@li input:Values to compare against threshold and bitpack.
*@li threshold:Threshold to compare against . \n

*@par Outputs:
*y:The bitpacked comparisons . \n

*@attention Constraints:
*Currently, the innermost dimension of the tensor must be divisible by 8. \n

*@par Third-party framework compatibility
*Compatible with tensorflow CompareAndBitpack operator
*/

REG_OP(CompareAndBitpack)
    .INPUT(x, TensorType({ DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, \
        DT_INT16, DT_INT32, DT_INT64, DT_BOOL }))
    .INPUT(threshold, TensorType({ DT_FLOAT, DT_FLOAT16, DT_DOUBLE, \
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_BOOL }))
    .OUTPUT(y, TensorType(DT_UINT8))
    .OP_END_FACTORY_REG(CompareAndBitpack)

/**
*@brief Counts the number of occurrences of each value in an integer array.
Outputs a vector with length size and the same dtype as weights. If weights
are empty, then index i stores the number of times the value i is counted in
arr. If weights are non-empty, then index i stores the sum of the value in
weights at each index . \n

*@par Inputs:
*The input size must be a non-negative int32 scalar Tensor. Inputs include:
*@li array:int32 Tensor.
*@li size:non-negative int32 scalar Tensor.
*@li weights: is an int32, int64, float32, or double Tensor with the same
shape as arr, or a length-0 Tensor, in which case it acts as all weights
equal to 1 . \n

*@par Outputs:
*bins:1D Tensor with length equal to size. The counts or summed weights for
each value in the range [0, size) . \n

*@par Third-party framework compatibility
*Compatible with tensorflow Bincount operator
*/

REG_OP(Bincount)
    .INPUT(array, TensorType(DT_INT32))
    .INPUT(size, TensorType(DT_INT32))
    .INPUT(weights, TensorType({ DT_FLOAT, DT_INT32, DT_INT64, DT_DOUBLE }))
    .OUTPUT(bins, TensorType({ DT_FLOAT, DT_INT32, DT_INT64, DT_DOUBLE }))
    .OP_END_FACTORY_REG(Bincount)

/**
*@brief Compute the regularized incomplete beta integral . \n

*@par Inputs:
*The input b and x must have the same types as a. Inputs include:
*@li a:A Tensor. Must be one of the following types: float32, double.
*@li b:A Tensor. Must have the same type as a.
*@li x:A Tensor. Must have the same type as a . \n

*@par Outputs:
*z:A Tensor. Has the same type as a . \n

*@par Third-party framework compatibility.
*Compatible with tensorflow Betainc operator.
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
*The input q must be the same type as x. Inputs include:
*@li x:A Tensor. Must be one of the following types: float32, double.
*@li q:A Tensor. Must have the same type as x . \n

*@par Outputs:
*z:A Tensor. Has the same type as x . \n

*@attention Constraints:
*The implementation for Zeta on Ascend uses ai cpu, with bad performance.

*@par Third-party framework compatibility.
*Compatible with tensorflow Zeta operator.
*/

REG_OP(Zeta)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT}))
    .INPUT(q, TensorType({DT_DOUBLE, DT_FLOAT}))
    .OUTPUT(z, TensorType({DT_DOUBLE, DT_FLOAT}))
    .OP_END_FACTORY_REG(Zeta)

/**
*@brief Bucketize 'input' based on 'boundaries'. For example, if the inputs
are boundaries = [0, 10, 100] input = [[-5, 10000] [150, 10] [5, 100]] then
the output will be output = [[0, 3] [3, 2] [1, 3]]

*@par Inputs:
*The dtype of input x  int float double. Inputs include:
*x:Any shape of Tensor contains with int or float type . \n

*@par Attributes:
*boundaries:A sorted list of floats gives the boundary of the buckets . \n

*@par Outputs:
*y:Same shape with 'input', each value of input replaced with bucket index . \n

*@par Third-party framework compatibility.
*Compatible with tensorflow Bucketize operator.
*/

REG_OP(Bucketize)
    .INPUT(x, TensorType({DT_INT32, DT_INT64, DT_DOUBLE, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .REQUIRED_ATTR(boundaries, ListFloat)
    .OP_END_FACTORY_REG(Bucketize)

/**
*@brief Computes the sum along sparse segments of a tensor . \n

*@par Inputs:
*The input indices and segment_ids must have same rank. Inputs include:
*@li x:A Tensor. Must be one of the following types: float, double, int32,
uint8, int16, int8, int64, uint16, uint32, uint64.
*@li indices: A Tensor. Must be one of the following types: int32, int64.
A 1-D tensor. Has same rank as segment_ids.
*@li segment_ids: A Tensor of type int32. A 1-D tensor. Values should be
sorted and can be repeated . \n

*@par Outputs:
*y:A Tensor. Has the same type as x . \n

*@par Third-party framework compatibility
*Compatible with tensorflow SparseSegmentSum operator
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
*@brief Computes the mean along sparse segments of a tensor . \n

*@par Inputs:
*The input indices and segment_ids must have same rank. Inputs include:
*@li x: A Tensor. Must be one of the following types: float, double.
*@li indices: A Tensor. Must be one of the following types: int32, int64.
A 1-D tensor. Has same rank as segment_ids.
*@li segment_ids: A Tensor of type int32. A 1-D tensor. Values should be
sorted and can be repeated . \n

*@par Outputs:
*y:A Tensor. Has the same type as x . \n

*@par Third-party framework compatibility
*Compatible with tensorflow SparseSegmentMean operator
*/

REG_OP(SparseSegmentMean)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(segment_ids, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(SparseSegmentMean)

/**
*@brief Computes gradients for SparseSegmentMean . \n

*@par Inputs:
*The input grad must have be type float or double. Inputs include:
*@li grad: A Tensor. Must be one of the following types: float, double.
gradient propagated to the SparseSegmentMean op.
*@li indices: A Tensor. Must be one of the following types: int32, int64.
indices passed to the corresponding SparseSegmentMean op.
*@li segment_ids: A Tensor of type int32. segment_ids passed to the
corresponding SparseSegmentMean op.
*@li output_dim0: A Tensor of type int32. dimension 0 of "x" passed to
SparseSegmentMean op . \n

*@par Outputs:
*y:A Tensor. Has the same type as grad . \n

*@par Third-party framework compatibility
*Compatible with tensorflow SparseSegmentMeanGrad operator
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
*The input a and x must have the same type. Inputs include:
*@li a:A Tensor. Must be one of the following types: float32, double.
*@li x:A Tensor. Must have the same type as a . \n

*@par Outputs:
*y:A Tensor. Has the same type as a . \n

*@par Third-party framework compatibility
*Compatible with tensorflow IgammaGradA operator
*/

REG_OP(IgammaGradA)
    .INPUT(a, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(z, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(IgammaGradA)

/**
*@brief Initialize data process channel . \n

*@par Attributes:
*channel_name: A string. Default "" . \n

*@par Third-party framework compatibility
*Compatible with tensorflow InitData operator
*/

REG_OP(InitData)
    .ATTR(channel_name, String, "")
    .OP_END_FACTORY_REG(InitData)

/**
*@brief Get the next batch of data in data processing . \n

*@par Attributes:
*@li output_types: A nested structure of DType objects corresponding to each
component of an element of this dataset.
*@li output_shapes: A nested structure of TensorShape objects corresponding
to each component of an element of this dataset.
*@li channel_name: A string. Default "" . \n

*@par Outputs:
*y:A nested structure of Tensor objects . \n

*@par Third-party framework compatibility
*Compatible with tensorflow GetNext operator
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
*@brief Get dynamic dims after GetNext. \n

*@par Inputs:
*input: A nested structure of Tensor objects, from GetNext's output. \n

*@par Attributes:
*@li shape_info: GE shape_info for each inputs, -1 means unknow dim.
*@li N: Inputs number. \n

*@par Outputs:
*dims: GE unknow dims, a vector of int64. \n
*/

REG_OP(GetDynamicDims)
    .DYNAMIC_INPUT(input, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(dims, TensorType({DT_INT64}))
    .REQUIRED_ATTR(shape_info, ListInt)
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(GetDynamicDims)

/**
*@brief End of sequence . \n

*@par Inputs:
*x: A Tensor of type uint8 . \n

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*/

REG_OP(EndOfSequence)
    .INPUT(x, TensorType({DT_UINT8}))
    .OUTPUT(y, TensorType({DT_UINT8}))
    .OP_END_FACTORY_REG(EndOfSequence)

/**
*@brief: Computes the Gauss error function of `x` element-wise . \n

*@par Inputs:
*x: A Tensor of type float16, float32 or double. the format can be
*    [NCHW,NC1HWC0,NHWC,ND]

*@par Outputs:
*y: A Tensor. Has the same type and format as "x" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Erf.
*/
REG_OP(Erf)
    .INPUT(x, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(Erf)

/**
*@brief: Computes the Gauss complementary error function of "x" element-wise . \n

*@par Inputs:
*x: A Tensor of type float16 ,float32, double . \n

*@par Outputs:
*y: A Tensor. Has the same type as "x" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Erfc.
*/
REG_OP(Erfc)
    .INPUT(x, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(Erfc)

/**
*@brief This operation returns a rank 1 histogram counting the number of entries in `values`
*  that fell into every bin.The bins are equal width and determined by the arguments
*  'value_range' and 'nbins' . \n

*@par Inputs:
*Three inputs, including:
*@li x: A Tensor of type float32, float16, int32, int64.
*@li range: A Tensor of type float32,float16,int32, int64.
*@li nbins: A Tensor of type int32 . \n

*@par Attributes:
* dtype: An optional attribute. Defaults to "int32" . \n

*@par Outputs:
*y: A Tensor. A Tensor of type int32 or int64 . \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator HistogramFixedWidth.
*/
REG_OP(HistogramFixedWidth)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64}))
    .INPUT(range, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64}))
    .INPUT(nbins, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .ATTR(dtype, String, "int32")
    .OP_END_FACTORY_REG(HistogramFixedWidth)

/**
*@brief This operation returns a rank 1 histogram counting the number of entries in `values`
*  that fell into every bin.The bins are equal width and determined by the arguments
*  'value_range' and 'nbins' . \n

*@par Inputs:
*Two inputs, including:
*@li x: A Tensor of type float32,float16,int32, int64.
*@li range: A Tensor of type float32,float16,int32, int64 . \n

*@par Attributes:
*@li dtype: An optional attribute. Defaults to "int32".
*@li nbins: A required attribute,the type is int32 . \n

*@par Outputs:
*y: A Tensor. A Tensor of type int32 . \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator HistogramFixedWidth.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use HistogramFixedWidth instead.
*/
REG_OP(HistogramFixedWidthD)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64}))
    .INPUT(range, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .REQUIRED_ATTR(nbins, Int)
    .ATTR(dtype, String, "int32")
    .OP_END_FACTORY_REG(HistogramFixedWidthD)

/**
*@brief Returns the next representable value of x1 in the direction of x2, element-wise . \n

*@par Inputs:
*The input X1 and x2 must have the same type. Inputs include:
*@li x1:A Tensor. Must be one of the following types: float32, double.
*@li x2:A Tensor. Must have the same type as x1 . \n

*@par Outputs:
*output:A Tensor. Has the same type as x1 . \n

*@par Third-party framework compatibility
*Compatible with tensorflow NextAfter operator
*/
REG_OP(NextAfter)
    .INPUT(x1, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(output, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(NextAfter)

/**
 *@brief Compute element-wise finiteness, return a boolean tensor.

 *@par Inputs:
 *x:A Tensor.

 *@par Outputs:
 *y:A Tensor. Has the same shape as x.

 *@par Third-party framework compatibility.
 *Compatible with tensorflow IsFinite operator.
 */
REG_OP(IsFinite)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(IsFinite)

/**
 *@brief Compute element-wise infiniteness, return a boolean tensor.

 *@par Inputs:
 *x:A Tensor.

 *@par Outputs:
 *y:A Tensor. Has the same shape as x.

 *@par Third-party framework compatibility.
 *Compatible with tensorflow IsInf operator.
 */
REG_OP(IsInf)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(IsInf)

/**
 *@brief Computes the complex absolute value of a tensor.

 *@par Inputs:
 *x:A Tensor.

 *@par Outputs:
 *y:A tensor of type `float` or `double` that is the absolute value of each element in `x`.

 *@par Third-party framework compatibility.
 *Compatible with tensorflow ComplexAbs operator.
 */
REG_OP(ComplexAbs)
    .INPUT(x, TensorType({DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .ATTR(Tout, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(ComplexAbs)

/**
 *@brief Returns which elements of x are NaN.

 *@par Inputs:
 *x:A Tensor.

 *@par Outputs:
 *y:A Tensor. Has the same shape as x.

 *@par Third-party framework compatibility.
 *Compatible with tensorflow IsNan operator.
 */
REG_OP(IsNan)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(IsNan)

/**
 *@brief Returns the real part of a complex number.

 *@par Inputs:
 *input:A Tensor.

 *@par Outputs:
 *output:A Tensor. Has the same shape as input.

 *@par Third-party framework compatibility.
 *Compatible with tensorflow Real operator.
 */
REG_OP(Real)
    .INPUT(input, TensorType({DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(output, TensorType({DT_FLOAT, DT_DOUBLE}))
    .ATTR(Tout, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(Real)

/**
 *@brief Returns the complex conjugate of a complex number.

 *@par Inputs:
 *input:A Tensor.

 *@par Outputs:
 *output:A Tensor. Has the same shape as input.

 *@par Third-party framework compatibility.
 *Compatible with tensorflow output operator.
 */
REG_OP(Conj)
    .INPUT(input, TensorType({DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(output, TensorType({DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Conj)

/**
*@brief The negative log likelihood loss . \n

*@par Inputs:
*The input x and weight must have the same type. Inputs include:
*@li x: A Tensor dtype of float32.
*@li target: A Tensor dtype of int32.
*@li weight: A Tensor dtype of float32 . \n

*@par Attributes:
*reduction: An optional attribute. Defaults to "mean" . \n

*@par Outputs:
*@li y: A Tensor dtype of float32.
*@li total_weight: A Tensor dtype of float32 . \n

*@par Third-party framework compatibility
*Compatible with pytorch NLLLoss operator
*/
REG_OP(NLLLoss)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(target, TensorType({DT_INT32}))
    .INPUT(weight, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OUTPUT(total_weight, TensorType({DT_FLOAT}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(NLLLoss)

/**
*@brief The negative log likelihood loss grad . \n

*@par Inputs:
*@li x:A Tensor dtype of float32.
*@li y_grad:A Tensor dtype of float32.
*@li target:A Tensor dtype of int32.
*@li weight:A Tensor dtype of float32.
*@li total_weight:A Tensor dtype of float32 . \n

*@par Attributes:
*reduction: An optional attribute. Defaults to "mean" . \n

*@par Outputs:
*x_grad: A Tensor. Must be the following type: float32 . \n

*@par Third-party framework compatibility
*Compatible with pytorch NLLLossGrad operator
*/
REG_OP(NLLLossGrad)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(y_grad, TensorType({DT_FLOAT}))
    .INPUT(target, TensorType({DT_INT32}))
    .INPUT(weight, TensorType({DT_FLOAT}))
    .INPUT(total_weight, TensorType({DT_FLOAT}))
    .OUTPUT(x_grad, TensorType({DT_FLOAT}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(NLLLossGrad)

/**
*@brief The ifmr . \n

*@par Inputs:
*@li data:A Tensor of feature map
*@li data_min:A Tensor of min value of feature map.
*@li data_max:A Tensor of max value of feature map.
*@li cumsum:A Tensor of cumsum bin of data . \n

*@par Attributes:
*min_percentile: min init percentile.
*max_percentile: max init percentile.
*search_range: search range.
*search_step: step size of searching.
*with_offset: whether using offset . \n

*@par Outputs:
*scale: optimal scale.
*offset: optimal offset . \n

*@par Third-party framework compatibility
*Compatible with mindspore
*/

REG_OP(IFMR)
  .INPUT(data, TensorType({DT_FLOAT16, DT_FLOAT}))
  .INPUT(data_min, TensorType({DT_FLOAT16, DT_FLOAT}))
  .INPUT(data_max, TensorType({DT_FLOAT16, DT_FLOAT}))
  .INPUT(cumsum, TensorType({DT_INT32}))
  .OUTPUT(scale, TensorType({DT_FLOAT}))
  .OUTPUT(offset, TensorType({DT_FLOAT}))
  .REQUIRED_ATTR(min_percentile, Float)
  .REQUIRED_ATTR(max_percentile, Float)
  .REQUIRED_ATTR(search_range, ListFloat)
  .REQUIRED_ATTR(search_step, Float)
  .REQUIRED_ATTR(with_offset, Bool)
  .OP_END_FACTORY_REG(IFMR)

/**
*@brief weights adaptive range quantization. \n

*@par Inputs:
*@li w:A Tensor of weights. \n
*@li w_min:A Tensor of weights reduce_min. \n
*@li w_max:A Tensor of weights reduce_max. \n

*@par Attributes:
*num_bits: the bits num used for quantize.
*offset_flag: whether using offset. \n

*@par Outputs:
*y: fake quantized weights. \n

*@par Third-party framework compatibility
*Compatible with mindspore

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/

REG_OP(WtsARQ)
  .INPUT(w, TensorType({DT_FLOAT16, DT_FLOAT}))
  .INPUT(w_min, TensorType({DT_FLOAT16, DT_FLOAT}))
  .INPUT(w_max, TensorType({DT_FLOAT16, DT_FLOAT}))
  .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
  .ATTR(num_bits, Int, 8)
  .ATTR(offset_flag, Bool, false)
  .OP_END_FACTORY_REG(WtsARQ)

/**
*@brief The acts_ulq. \n

*@par Inputs:
*@li x:A Tensor of feature map
*@li clamp _min:A Tensor of min clamp value of feature map.
*@li clamp _max:A Tensor of max clamp value of feature map.

*@par Attributes:
*fixed_min: fix min to zero.
*num_bits: quant bits. \n

*@par Outputs:
*y: output fake quant feature map.
*clamp_min_mask: where x > clamp_min
*clamp_min_mask: where x < clamp_max
*x_clamped_loss: clamp loss. \n

*@par Third-party framework compatibility
*Compatible with mindspore

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/

REG_OP(ActsULQ)
  .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
  .INPUT(clamp_min, TensorType({DT_FLOAT16, DT_FLOAT}))
  .INPUT(clamp_max, TensorType({DT_FLOAT16, DT_FLOAT}))
  .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
  .OUTPUT(clamp_min_mask, TensorType({DT_BOOL}))
  .OUTPUT(clamp_max_mask, TensorType({DT_BOOL}))
  .OUTPUT(x_clamped_loss, TensorType({DT_FLOAT16, DT_FLOAT}))
  .ATTR(fixed_min, Bool, false)
  .ATTR(num_bits, Int, 8)
  .OP_END_FACTORY_REG(ActsULQ)

/**
*@brief The acts_ulq_input_grad. \n

*@par Inputs:
*@li y_grad: A Tensor of gradient
*@li clamp_min_mask: A Tensor of boolean mask indicating whether an additional one is needed'
*@li clamp_max_mask: A Tensor of boolean mask indicating whether an additional one is needed'

*@par Outputs:
*x_grapd: The gradient of inpust. \n

*@par Third-party framework compatibility
*Compatible with mindspore

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/

REG_OP(ActsULQInputGrad)
  .INPUT(y_grad, TensorType({DT_FLOAT16, DT_FLOAT}))
  .INPUT(clamp_min_mask, TensorType({DT_BOOL}))
  .INPUT(clamp_max_mask, TensorType({DT_BOOL}))
  .OUTPUT(x_grad, TensorType({DT_FLOAT16, DT_FLOAT}))
  .OP_END_FACTORY_REG(ActsULQInputGrad)

/**
*@brief The act_ulq_clamp_max_grad. \n

*@par Inputs:
*@li y_grad: A Tensor of gradient
*@li clamp_max_mask: A Tensor of boolean mask indicating whether an additional one is needed.
*@li x_clamped_loss: A Tensor of gradient. \n

*@par Outputs:
*clamp_max_grad: The gradient of clamp max. \n

*@par Third-party framework compatibility
*Compatible with mindspore

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/

REG_OP(ActULQClampMaxGrad)
  .INPUT(y_grad, TensorType({DT_FLOAT16, DT_FLOAT}))
  .INPUT(clamp_max_mask, TensorType({DT_BOOL}))
  .INPUT(x_clamped_loss, TensorType({DT_FLOAT16, DT_FLOAT}))
  .OUTPUT(clamp_max_grad, TensorType({DT_FLOAT16, DT_FLOAT}))
  .OP_END_FACTORY_REG(ActULQClampMaxGrad)

/**
*@brief The act_ulq_clamp_min_grad. \n

*@par Inputs:
*@li y_grad: A Tensor of gradient
*@li clamp_min_mask: A Tensor of boolean mask indicating whether an additional one is needed.
*@li x_clamped_loss: A Tensor of gradient. \n

*@par Outputs:
*clamp_min_grad: The gradient of clamp min. \n

*@par Third-party framework compatibility
*Compatible with mindspore

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/

REG_OP(ActULQClampMinGrad)
  .INPUT(y_grad, TensorType({DT_FLOAT16, DT_FLOAT}))
  .INPUT(clamp_min_mask, TensorType({DT_BOOL}))
  .INPUT(x_clamped_loss, TensorType({DT_FLOAT16, DT_FLOAT}))
  .OUTPUT(clamp_min_grad, TensorType({DT_FLOAT16, DT_FLOAT}))
  .OP_END_FACTORY_REG(ActULQClampMinGrad)

/**
* @brief Computes Lp norm.

* @par Inputs:
* @li x: An ND tensor of type float16, float32. \n
*
* @par Attributes:
* @li p: Int, "inf" or "-inf", default value is 2.
* @li axes: ListInt, {} means all axes will be computed.
* @li keepdim: Bool, default is false.
* @li epsilon: Float, default is 1e-12. \n

* @par Outputs:
* @li y: An ND tensor of type float16, float32. The shape of y is depending
* on axes and keepdim. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator LpNorm.
*/
REG_OP(LpNorm)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(p, Int, 2)
    .ATTR(axes, ListInt, {})
    .ATTR(keepdim, Bool, false)
    .ATTR(epsilon, Float, 1e-12)
    .OP_END_FACTORY_REG(LpNorm)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_MATH_OPS_H_
