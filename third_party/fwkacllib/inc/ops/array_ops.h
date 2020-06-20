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

#ifndef GE_OP_ARRAY_OPS_H_
#define GE_OP_ARRAY_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief Applies lower_bound(sorted_search_values, values) along each row.

*@par Inputs:
*The input sorted_x and values can be one-dimensional vector. Inputs include: \n
* @li sorted_x:A `Tensor`. 2-D Tensor where each row is ordered.
* @li values:A `Tensor`. Must have the same type as `sorted_x`.

*@par Attributes:
*@li out_type:An optional `DType` from: `int32, int64`. \n
Defaults to `int32`.

*@par Outputs:
*y: A `Tensor` of type `out_type`.

*@attention Constraints: \n
*-The implementation for LowerBound on Ascend uses AI CPU, with bad performance. \n

*@par Quantization supported or not
*Not supported
*@par Quantized inference supported or not
*Supported
*@par L2 convergence supported or not
*@par Multiple batches supported or not
*/

REG_OP(LowerBound)
    .INPUT(sorted_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, \
        DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .INPUT(values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, \
        DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(out_type, Type, DT_INT32)
    .OP_END_FACTORY_REG(LowerBound)

/**
*@brief Reverses variable length slices.

*@par Inputs:
*Input "x" is a k-dimensional tensor. Inputs "num_lower" and "num_upper" \n
are 0D scalars.
* @li x: A Tensor. The input to reverse.
* @li seq_lengths: A 1D Tensor of type int32 or int64.

*@par Attributes:
*@li seq_dim: An optional int. Defaults to "0". The dimension along which \n
reversal is performed.
*@li batch_dim: An optional int. Defaults to "0". The dimension along which \n
reversal is performed.

*@par Outputs:
*y: A rank k tensor. Has the same shape as input. The extracted banded tensor.

*@attention Constraints: \n
*ReverseSequence runs on the Ascend AI CPU, which delivers poor performance.
*/

REG_OP(ReverseSequence)
    .INPUT(x,
        TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(seq_lengths, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y,
        TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .REQUIRED_ATTR(seq_dim, Int)
    .ATTR(batch_dim, Int, 0)
    .OP_END_FACTORY_REG(ReverseSequence)

/**
*@brief Copies a tensor setting everything outside a central band in each innermost matrix.

*@par Inputs:
*Input "x" is a k-dimensional tensor. Inputs "num_lower" and "num_upper" \n
are 0D scalars.
* @li x: A rank k tensor.
* @li num_lower: A 0D tensor. Number of superdiagonals to keep. If negative, \n
keeps entire upper triangle.
* @li num_upper: A 0D tensor. Number of superdiagonals to keep. If negative, \n
keeps entire upper triangle.

*@par Outputs:
*y: A rank k tensor. Has the same shape as input. The extracted banded tensor.

*@attention Constraints: \n
*MatrixBandPart runs on the Ascend AI CPU, which delivers poor performance. \n
*/

REG_OP(MatrixBandPart)
    .INPUT(x, TensorType({ DT_INT8, DT_UINT8, \
           DT_INT16, DT_UINT16, DT_INT32, DT_INT64,
           DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL,
           DT_COMPLEX64, DT_COMPLEX128 }))
    .INPUT(num_lower, TensorType({ DT_INT32, DT_INT64 }))
    .INPUT(num_upper, TensorType({ DT_INT32, DT_INT64 }))
    .OUTPUT(y, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
           DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL,
           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(MatrixBandPart)

/**
*@brief Finds unique elements in a 1D tensor.

*@par Inputs:
*x: 1D tensor. \n
*Input "x" is a k-dimensional tensor. Inputs "num_lower" and "num_upper" \n
are 0D scalars.

*@par Attributes:
*out_idx: An optional DType from: "int32, int64". \n
Defaults to "int32".

*@par Outputs:
*@li y: A Tensor. Has the same type as "x".
*@li idx: A Tensor of type "out_idx".
*@li count: A Tensor of type "out_idx".

*@attention Constraints: \n
*UniqueWithCounts runs on the Ascend AI CPU, which delivers poor performance. \n
*/

REG_OP(UniqueWithCounts)
    .INPUT(x, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
           DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_STRING }))
    .OUTPUT(y, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
           DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_STRING }))
    .OUTPUT(idx, TensorType({ DT_INT32, DT_INT64 }))
    .OUTPUT(count, TensorType({ DT_INT32, DT_INT64 }))
    .REQUIRED_ATTR(out_idx, Type)
    .OP_END_FACTORY_REG(UniqueWithCounts)

/**
*@brief Finds unique elements in a 1D tensor.

*@par Inputs:
*x: 1D tensor. \n
*Input "x" is a k-dimensional tensor. Inputs "num_lower" and "num_upper" \n
are 0D scalars.

*@par Attributes:
*out_idx: An optional DType from: "int32, int64". Defaults to "int32".

*@par Outputs:
*@li y: "x" in the unique output "y".
*@li idx: A tensor the same size as "x". The index of each value of "x".

*@attention Constraints: \n
*Unique runs on the Ascend AI CPU, which delivers poor performance. \n
*/

REG_OP(Unique)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
           DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
           DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OUTPUT(idx, TensorType({DT_INT32, DT_INT64}))
    .ATTR(out_idx, Type, DT_INT32)
    .OP_END_FACTORY_REG(Unique)

/**
*@brief Finds unique elements in a 1D tensor.

*@par Inputs:
*Input "x" is a k-dimensional tensor. Inputs "num_lower" and "num_upper" \n
are 0D scalars. \n
*Including:
* @li x: 1D tensor.
* @li axis: A Tensor of type int32. Defaults to "None".

*@par Attributes:
*out_idx: An optional DType from: "int32, int64". \n
Defaults to "int32".

*@par Outputs:
*@li y: "x" in the unique output "y".
*@li idx: A tensor the same size as "x". The index of each value of "x".

*@attention Constraints: \n
*UniqueExt2 runs on the Ascend AI CPU, which delivers poor performance. \n
*/

REG_OP(UniqueExt2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
           DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .INPUT(axis, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
           DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OUTPUT(idx, TensorType({DT_INT32, DT_INT64}))
    .ATTR(out_idx, Type, DT_INT32)
    .OP_END_FACTORY_REG(UniqueExt2)

/**
*@brief Computes the inverse permutation of a tensor.

*@par Inputs:
*x: A k-dimensional tensor. \n

*@par Outputs:
*y: A 1D tensor.

*@attention Constraints: \n
*InvertPermutation runs on the Ascend AI CPU, which delivers poor performance. \n
*/

REG_OP(InvertPermutation)
    .INPUT(x, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(InvertPermutation)

/**
*@brief Checks a tensor for NaN and Inf values.

*@par Inputs:
*x: A k-dimensional tensor. \n

*@par Attributes:
*message: Prefix of the error message.

*@par Outputs:
*y: The output tensor.

*@attention Constraints: \n
*CheckNumerics runs on the Ascend AI CPU, which delivers poor performance. \n
*/

REG_OP(CheckNumerics)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(message, String)
    .OP_END_FACTORY_REG(CheckNumerics)

/**
*@brief Converts an array of flat indices into a tuple of coordinate arrays.

*@par Inputs:
*Input "indices" is a 0D or 1D tensor. Input "dims" is a 1D tensor. \n
* @li indices: A 0D or 1D int Tensor whose elements are indices into \n
the flattened version of an array of dimensions "dims".
* @li dims: A 1D int Tensor of the same type as "indices". \n
*The shape of the array to use for unraveling indices.

*@par Outputs:
*y: A Tensor. Has the same type as "indices".

*@attention Constraints: \n
*UnravelIndex runs on the Ascend AI CPU, which delivers poor performance. \n
*/

REG_OP(UnravelIndex)
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(dims, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(UnravelIndex)

/**
*@brief Applies upper_bound(sorted_search_values, values) along each row.

*@par Inputs:
*Inputs "sorted_x" and "values" are 2D tensors.
* @li sorted_x: A 2D Tensor where each row is ordered.
* @li values: A 2D Tensor with the same numbers of rows as "sorted_x.

*@par Attributes:
*out_type: sets the optional out_type attribute to value.

*@par Outputs:
*y: A Tensor with the same shape as "values".

*@attention Constraints: \n
*UpperBound runs on the Ascend AI CPU, which delivers poor performance. \n
*/

REG_OP(UpperBound)
    .INPUT(sorted_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
      DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .INPUT(values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
      DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(out_type, Type)
    .OP_END_FACTORY_REG(UpperBound)

/**
*@brief Finds unique elements in a 1D tensor.

*@par Inputs:
*Inputs "x" and "axis" are 1D vectors. \n
* @li x: A 1D tensor.
* @li axis: A 1D tensor.

*@par Attributes:
*out_idx: An optional DType from: "int32, int64". \n
Defaults to "int32".

*@par Outputs:
*@li y: "x" in the unique output "y".
*@li idx: A tensor the same size as "x". The index of each value of "x".
*@li count: A tensor the same size as "x". The index of each value of "x".

*@attention Constraints: \n
*UniqueWithCountsExt2 runs on the Ascend AI CPU, which delivers poor performance. \n
*/

REG_OP(UniqueWithCountsExt2)
    .INPUT(x, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
      DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_STRING }))
    .INPUT(axis, TensorType({ DT_INT32, DT_INT64 }))
    .OUTPUT(y, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
      DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_STRING }))
    .OUTPUT(idx, TensorType({ DT_INT32, DT_INT64 }))
    .OUTPUT(count, TensorType({ DT_INT32, DT_INT64 }))
    .REQUIRED_ATTR(out_idx, Type)
    .OP_END_FACTORY_REG(UniqueWithCountsExt2)

/**
*@brief Fills the tensor with the mirror value.

*@par Inputs:
*Inputs "x" and "paddings" are 1D scalars. \n
* @li x: The tensor to be padded.
* @li paddings: A two-column matrix specifying the padding sizes. \n
The number of rows Has the same rank as "x".

*@par Attributes:
*mode: Either "REFLECT" or "SYMMETRIC". In reflect mode the padded regions \n
do not include the borders, while in symmetric mode the padded regions \n
do include the borders.

*@par Outputs:
*y: The padded tensor.

*@attention Constraints: \n
*MirrorPad runs on the Ascend AI CPU, which delivers poor performance. \n
*/

REG_OP(MirrorPad)
    .INPUT(x, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
      DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL, \
      DT_COMPLEX64, DT_COMPLEX128 }))
    .INPUT(paddings, TensorType({ DT_INT32, DT_INT64 }))
    .OUTPUT(y, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
      DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL, \
      DT_COMPLEX64, DT_COMPLEX128 }))
    .REQUIRED_ATTR(mode, String)
    .OP_END_FACTORY_REG(MirrorPad)

/**
*@brief Calculates the difference between two numbers or a list of strings.

*@par Inputs:
*Inputs "x" and "y" are 1D vectors. \n
* @li x: A Tensor. 1D. Values to keep.
* @li y: A Tensor. Must have the same type as x. 1D. Values to remove.

*@par Attributes:
*out_idx: An optional DType from: "int32, int64". Defaults to "int32".

*@par Outputs:
*@li out: A Tensor. Has the same type as "x".
*@li idx: A Tensor of type "out_idx".

*@attention Constraints: \n
*ListDiff runs on the Ascend AI CPU, which delivers poor performance. \n
*/

REG_OP(ListDiff)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8,
        DT_INT16, DT_UINT16, DT_INT32, DT_INT64}))
    .INPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8,
        DT_INT16, DT_UINT16, DT_INT32, DT_INT64}))
    .OUTPUT(out, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8,
        DT_INT16, DT_UINT16, DT_INT32, DT_INT64}))
    .OUTPUT(idx, TensorType({DT_INT32, DT_INT64}))
    .ATTR(out_idx, Type, DT_INT32)
    .OP_END_FACTORY_REG(ListDiff)

/**
*@brief Create an empty tensor, using the shape and dtype specified in attributes.

*@par Attributes:
*@li dtype: Specify the data type of the empty tensor.
*@li shape: Specify the shape of the empty tensor.

*@par Outputs:
*y: The empty constant tensor.

*/
REG_OP(_ParallelConcatStart)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
                          DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(dtype, Type, DT_INT32)
    .ATTR(shape, ListInt, {})
    .OP_END_FACTORY_REG(_ParallelConcatStart)

/**
*@brief Creates a constant tensor from a tensor-like object. This operator is used for inference. \n
Operator Const has the same definition as operator Constant.

*@par Attributes:
*value: Required. The value and type of the resulting tensor, and no restrictions on type.

*@par Outputs:
*y: A constant tensor.
*/
REG_OP(Const)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Const)

/**
*@brief Creates a constant tensor for training.

*@par Attributes:
*value: Required. The value and type of the resulting tensor, and no restrictions on type.

*@par Outputs:
*y: The constant tensor.
*/
REG_OP(Constant)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Constant)

/**
*@brief Returns a copy of the input tensor.

*@par Inputs:
*x: A tensor.

*@par Outputs:
*y: A tensor.
*/
REG_OP(Snapshot)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OP_END_FACTORY_REG(Snapshot)

/**
*@brief Gives a guarantee to the runtime that the input tensor is a constant.

*@par Inputs:
*x: A tensor.

*@par Outputs:
*y: The input tensor.
*/
REG_OP(GuaranteeConst)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
                          DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
                          DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OP_END_FACTORY_REG(GuaranteeConst)

/**
*@brief Returns the target shape for broadcasting shapes "x1" and "x2".

*@par Inputs:
*@li x1: A tensor of type int32 or int64. A shape.
*@li x2: A tensor of the same type as "x1". The other shape.

*@par Outputs:
*y: A tensor. The broadcasted shape.
*/
REG_OP(BroadcastArgs)
    .INPUT(x1, TensorType({DT_INT32, DT_INT64}))
    .INPUT(x2, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(BroadcastArgs)

/**
*@brief Outputs its input tensor as is and triggers an error if a gradient is requested.

*@par Inputs:
*x: A tensor.

*@par Attributes:
*message: Will be printed in the error at the attempt to request a gradient.

*@par Outputs:
*y: The input tensor.
*/
REG_OP(PreventGradient)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(message, String, "")
    .OP_END_FACTORY_REG(PreventGradient)

/**
*@brief Returns the reduction indices for computing gradients of "x1" and "x2" with broadcast.

*@par Inputs:
*@li x1: A tensor of type int32 or int64.
*@li x2: A tensor of type int32 or int64. \n
"x2" has the same type as "x1".

*@par Outputs:
*@li y1: A tensor. Reduction indices of "x1".
*@li y2: A tensor. Reduction indices of "x2".
*/
REG_OP(BroadcastGradientArgs)
    .INPUT(x1, TensorType({DT_INT32, DT_INT64}))
    .INPUT(x2, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y1, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y2, TensorType({DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(BroadcastGradientArgs)

/**
*@brief Stops gradient computation. None is returned for the node where the gradient computation is stopped.


*@par Inputs:
*x: A tensor.

*@par Outputs:
*y: The input tensor.
*/
REG_OP(StopGradient)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OP_END_FACTORY_REG(StopGradient)

/**
*@brief Return a tensor with the same shape and contents as input.

*@par Inputs:
*x: A tensor.

*@par Outputs:
*y: A tensor.
*/
REG_OP(Identity)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OP_END_FACTORY_REG(Identity)

/**
*@brief Returns a list of tensors with the same shapes and contents as the input tensors.

*@par Inputs:
*x: A list of input tensors.

*@par Outputs:
*y: A list of Tensor objects, with the same length as the input tensor list.
*/
REG_OP(IdentityN)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OP_END_FACTORY_REG(IdentityN)

/**
*@brief Inserts a dimension of 1 into a tensor's shape. Only the tensor shape is changed, without changing the data.

*@par Inputs:
*@li x: A tensor.
*@li axis: The dimension index at which to expand.

*@par Outputs:
*y: A tensor.
*/
REG_OP(ExpandDims)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
        DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .INPUT(axis, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
        DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OP_END_FACTORY_REG(ExpandDims)

/**
*@brief Reshapes a tensor. Only the tensor shape is changed, without changing the data.

*@par Inputs:
*@li x: A tensor.
*@li shape: A tensor. Defines the shape of the output tensor.

*@par Attributes:
*@li axis: An optional int32 or int64. The first dimension to reshape. Defaults to "0".
*@li num_axes: An optional int32 or int64. The extent of the reshape. Defaults to "-1".

*@par Outputs:
*y: A tensor.

*@par Attention:
*This operator cannot be directly called by the acllopExecute API.
*/
REG_OP(Reshape)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
        DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
        DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(axis, Int, 0)
    .ATTR(num_axes, Int, -1)
    .OP_END_FACTORY_REG(Reshape)

/**
*@brief Removes dimensions of size 1 from the shape of a tensor.

*@par Inputs:
*x: A tensor.

*@par Attributes:
*axis: An optional list of int32 or int64. If not specified, squeezes all dimensions of size 1. \n If specified, only squeezes the dimensions listed. It is an error to squeeze a dimension that is not 1.

*@par Outputs:
*y: A tensor.
*/
REG_OP(Squeeze)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axis, ListInt, {})
    .OP_END_FACTORY_REG(Squeeze)

/**
*@brief Returns an integer representing the rank of input tensor. The rank of a tensor is the number of indices required to uniquely select each element of the tensor, that is, the dimension size of the tensor.

*@par Inputs:
*x: A tensor.

*@par Outputs:
*y: A tensor. The rank of input tensor.
*/
REG_OP(Rank)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(Rank)

/**
*@brief Returns the size of a tensor, that is, an integer of the number of elements of the tensor.

*@par Inputs:
*x: A tensor.

*@par Attributes:
*out_type: An optional int32 or int64. The output data type. Defaults to "int32".

*@par Outputs:
*y: A tensor. The size of the input tensor.
*/
REG_OP(Size)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT32,DT_INT64}))
    .ATTR(dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(Size)

/**
*@brief Input data for other operators.

*@par Inputs:
*x: A tensor.

*@par Attributes:
*index: Index of the input tensor of type int32 or int64.

*@par Outputs:
*y: A tensor.

*/
REG_OP(Data)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(Data)

/**
*@brief Inserts a placeholder for a tensor that will be always fed.

*@par Inputs:
*x: A tensor.

*@par Attributes:
*@li peerIndex: An integer type. The index of the corresponding "end" node connected to.
*@li parentId: A string, used to check if the nodes are from the saved parent node.
*@li parentOpType: A string. Op type of the original node.
*@li anchorIndex: An integer, used to check if the node is from the saved anchor.

*@par Outputs:
*y: The created placeholder tensor.
*/
REG_OP(PlaceHolder)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(peerIndex, Int, 0) // the index of the corresponding 'end' node it's connected to
    .ATTR(parentId, String, "")     // check if these node are from save parent node
    .ATTR(parentOpType, String, "") // op type of original node
    .ATTR(anchorIndex, Int, 0)  // check if these node are from save anchor
    .OP_END_FACTORY_REG(PlaceHolder)

/**
*@brief Inserts a placeholder with default value for a tensor.

*@par Inputs:
*x: A tensor.

*@par Attributes:
*@li dtype: data type of tensor.
*@li shape: tensor shape.

*@par Outputs:
*y: The created placeholder tensor.

*/
REG_OP(PlaceholderWithDefault)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .REQUIRED_ATTR(shape, ListInt)
    .OP_END_FACTORY_REG(PlaceholderWithDefault)

/**
*@brief Reads and returns the value of the input variable tensor.

*@par Inputs:
*x: A tensor.

*@par Attributes:
*dtype: An optional int32 or int64. The output data type. Defaults to int32.

*@par Outputs:
*y: A tensor.

*/
REG_OP(ReadVariableOp)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
                          DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
                           DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(ReadVariableOp)

REG_OP(End)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(peerIndex, Int, 0) // the index of the corresponding 'placeholder' node it's connected to
    .ATTR(parentOpType, String, "") // op type of original node
    .OP_END_FACTORY_REG(End)

REG_OP(Summary)
    .INPUT(x, TensorType::ALL())
    .OP_END_FACTORY_REG(Summary)

/**
*@brief Returns the shape of a tensor.

*@par Inputs:
*x: A tensor.

*@par Attributes:
*dtype: An optional int32 or int64. The output data type. Defaults to int32.

*@par Outputs:
*y: A tensor. The shape of the input tensor.
*/
REG_OP(Shape)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(Shape)

/**
*@brief Returns shape of tensors.

*@par Inputs:
*x: A list of input tensors.

*@par Attributes:
*dtype: An optional int32 or int64. The output data type. Defaults to "int32".

*@par Outputs:
*y: A list of tensors with the same length as the input list of tensors.
*/
REG_OP(ShapeN)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(ShapeN)

/**
*@brief Creates a tensor with the given "shape" and "dtype".

*@par Inputs:
*shape: The shape of the output tensor.

*@par Attributes:
*@li dtype: Optional. The data type of the output tensor. Defaults to "int32".
*@li init: An optional bool. If true, initializes the returned tensor with the default value of "dtype". Defaults to "false".

*@par Outputs:
*y: A tensor.
*/
REG_OP(Empty)
    .INPUT(shape, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(dtype, Int, DT_INT32)
    .ATTR(init, Bool, 0)
    .OP_END_FACTORY_REG(Empty)

/**
*@brief Gradient op for MirrorPad op. Folds a mirror-padded tensor.

*@par Inputs:
*Inputs "x" and "y" are 1D vectors. \n
* @li x: A Tensor. The input tensor to be folded.
* @li paddings: A Tensor of type int32 or int64. A two-column matrix \n
specifying the padding sizes.

*@par Attributes:
*mode: A string from: "REFLECT", "SYMMETRIC". The mode used in the MirrorPad op.

*@par Outputs:
*y: A Tensor. Has the same type as "x".

*@attention Constraints: \n
*MirrorPadGrad runs on the Ascend AI CPU, which delivers poor performance. \n
*/

REG_OP(MirrorPadGrad)
    .INPUT(x, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
              DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, \
              DT_COMPLEX64, DT_COMPLEX128 }))
    .INPUT(paddings, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
              DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, \
              DT_COMPLEX64, DT_COMPLEX128 }))
    .REQUIRED_ATTR(mode, String)
    .OP_END_FACTORY_REG(MirrorPadGrad)

/**
*@brief Returns locations of nonzero / true values in a tensor.

*@par Inputs:
*Including: \n
*x: A Tensor. Must be one of the following types: \n
DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT16, \n
DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL.

*@par Outputs:
*y: A Tensor of type DT_INT64.

*@attention Constraints:\n
*Where runs on the Ascend AI CPU, which delivers poor performance.\n

*/

REG_OP(Where)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT16, \
              DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(Where)

/**
*    multiple output blobs for feeding a blob into multiple output layers. \n
*The Split node is removed from the graph after the split operation is completed.

*@par Inputs:
*x: A Tensor. Must be one of the following types: \n
fp16, fp32, int8, uint8, int16, uint16, int32, uint32, int64, uint64.

*@par Outputs:
*y: A Tensor. Has the same type as "x".It's required and the value should equal to output_num.
*/
REG_OP(Copy)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT16, \
              DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT16, \
              DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64}))
    .OP_END_FACTORY_REG(Copy);

/**
*@brief Generates fingerprint values.

*@par Inputs:
*@li data: Must have rank 1 or higher.
*@li method: Fingerprint method used by this op. Currently available method is \n
`farmhash::fingerprint64`.

*@par Outputs:
`data`'s first dimension, and the second dimension size depends on the \n
fingerprint algorithm.

*/

REG_OP(Fingerprint)
    .INPUT(data, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT16, \
              DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL}))
    .INPUT(method, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_UINT8}))
    .OP_END_FACTORY_REG(Fingerprint)

/**
*@brief Change the shape of output according to the attr outShape
*

*@par Inputs:
*x: A Tensor.

*@par Outputs:
*y: A Tensor. Has the same type as "x".It's required and the value should equal to output_num.

*@par Attributes:
*outShape: The shape of output will be inferred according to the attribute
*/
REG_OP(TransShape)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(outShape,ListInt ,{})
    .OP_END_FACTORY_REG(TransShape);

}  // namespace ge

#endif  // GE_OP_ARRAY_OPS_H_
