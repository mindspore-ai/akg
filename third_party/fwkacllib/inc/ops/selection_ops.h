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

#ifndef GE_OP_SELECTION_OPS_H
#define GE_OP_SELECTION_OPS_H
#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Creates a sequence of numbers.

*@par Inputs:
*Three inputs, including:
* @li start: A 0D Tensor (scalar). Acts as first entry in the range if "limit"
*   is not "None"; otherwise, acts as range limit and first entry defaults to "0".
*   The supported types are: float32, int32, double, int64.
* @li limit: A 0D Tensor (scalar). Upper limit of sequence, exclusive. If "None",
*   defaults to the value of "start" while the first entry of the range
*   defaults to "0". The supported types are: float32, int32, double, int64.
* @li delta: A 0D Tensor (scalar). Number that increments "start".
*   Defaults to "1". The supported types are: float32, int32, double, int64.

*@par Outputs:
*y: A 1D Tensor.
*/
REG_OP(Range)
    .INPUT(start, TensorType({DT_FLOAT,DT_INT32,DT_DOUBLE,DT_INT64}))
    .INPUT(limit, TensorType({DT_FLOAT,DT_INT32,DT_DOUBLE,DT_INT64}))
    .INPUT(delta, TensorType({DT_FLOAT,DT_INT32,DT_DOUBLE,DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT,DT_INT32,DT_DOUBLE,DT_INT64}))
    .OP_END_FACTORY_REG(Range)

/**
*@brief: Creates a sequence of numbers.

*@par Inputs:
*Four inputs, including:
* @li x: A 1D Tensor of type float32 or int32. The assistant data.
* @li start: A 0D Tensor (scalar) of type float32 or int32. Acts as first entry in the range if "limit"
*   is not "None"; otherwise, acts as range limit and first entry defaults to "0".
* @li limit: A 0D Tensor (scalar) of type float32 or int32.
*   Upper limit of sequence, exclusive. If "None",
*   defaults to the value of "start" while the first entry of the range
*   defaults to "0".
* @li delta: A 0D Tensor (scalar) of type float32 or int32.
*   Number that increments "start". Defaults to "1".

*@par Outputs:
*y: A 1D Tensor.

*@par Quantization supported or not
*Not supported

*@par Quantized inference supported or not
*Not supported

*@par Multiple batches supported or not
*Supported

*@see Range()
*@since V100R001C33
*/
REG_OP(RangeD)
    .INPUT(x, TensorType({DT_FLOAT,DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT,DT_INT32}))
    .REQUIRED_ATTR(start, Float)
    .REQUIRED_ATTR(limit, Float)
    .REQUIRED_ATTR(delta, Float)
    .OP_END_FACTORY_REG(RangeD)

/**
*@brief Constructs a tensor by tiling a given tensor.

*@par Inputs:
*Two inputs, including:
* @li x: A Tensor of type TensorType::BasicType().
* @li multiples: A 1D Tensor of type int32 or int64.
*     The length must be the same as the number of dimensions in "input"

*@par Outputs:
*y: A Tensor. Has the same type as "x".

*@see TileD()
*/
REG_OP(Tile)
    .INPUT(x, TensorType::BasicType())
    .INPUT(multiples, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(Tile)

/**
*@brief Constructs a tensor by tiling a given tensor.

*@par Inputs:
*x: A Tensor. Must be one of the following types: float32, float16, int32.

*@par Attributes:
*multiples: A required Tensor of type int32 or int64.
*     Number of replication times.

*@par Outputs:
*y: A Tensor. Has the same type as "x".

*@see Tile()
*/
REG_OP(TileD)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .REQUIRED_ATTR(multiples, ListInt)
    .OP_END_FACTORY_REG(TileD)

/**
* @brief Gather slices from "x" into a tensor with shape specified by\n
* "indices". "indices" is an K-dimensional integer tensor, best thought of as a\n
* (K-1)-dimensional tensor of "indices" into "params", where each element\n
* defines a slice of "params":\n
*   output[\\(i_0, ..., i_{K-2}\\)] = params[indices[\\(i_0, ..., i_{K-2}\\)]]\n
* "indices" defines slices into the first N dimensions of\n
* "params", where\n
*           N = indices.shape[-1]\n
*     indices = [[0, 0], [1, 1]]\n
*      x = [['a', 'b'], ['c', 'd']]\n
*      output = ['a', 'd']\n

* @par Inputs:
* @li x: A Tensor of type BasicType.
* @li indices: A Tensor of type IndexNumberType.

* @par Outputs:
* output: A Tensor of type BasicType.
* @see GatherNd()

* @attention Constraints:
* @li "x" is one of the following types: float16, float32, int32, int8,
* uint8.
*/
REG_OP(GatherNd)
    .INPUT(x, TensorType::BasicType())
    .INPUT(indices, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(GatherNd)

/**
*@brief Gather slices from "x" according to "indices" by corresponding axis.

*@par Inputs:
*Three inputs, including:
* @li x: A Tensor. Must be one of the following types: float32, float64, int32, uint8, int16, int8, \n
*     complex64, int64, qint8, quint8, qint32, qint16, quint16, uint16, \n
*     complex128, float16, uint32, uint64, complex64, complex128.
* @li indices: A Tensor of type int32 or int64.
* @li axis: A Tensor of type as int32.

*@par Outputs:
*y: A Tensor. Has the same type as "x".

*@attention Constraints:
*Value in indices must be in range [0, x.shape[axis])
*/
REG_OP(GatherV2)
    .INPUT(x, TensorType::BasicType())
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(axis, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(GatherV2)

/**
*@brief Gather slices from "x" according to "indices" by corresponding axis.

*@par Inputs:
*Two inputs, including:
* @li x: A Tensor. Must be one of the following types: float32, float16, int32, uint32, int8, uint8, \n
*     int16, uint16, int64, uint64.
* @li indices: A Tensor of type int32 or int64.

*@par Attributes:
*axis: A int32 specifying the axis to gather from.

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*/
REG_OP(GatherV2D)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_UINT32, DT_INT8, DT_UINT8,
                          DT_INT16, DT_UINT16, DT_INT64, DT_UINT64}))
    .INPUT(indices, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_UINT32, DT_INT8, DT_UINT8,
                          DT_INT16, DT_UINT16, DT_INT64, DT_UINT64}))
    .REQUIRED_ATTR(axis, Int)
    .OP_END_FACTORY_REG(GatherV2D)

/**
*@brief Extracts a strided slice of a tensor. Roughly speaking, this op \n
    extracts a slice of size (end-begin)/stride from the given input tensor. \n
    Starting at the location specified by begin the slice continues by \n
    adding stride to the index until all dimensions are not less than end. \n

*@par Inputs:
*Four inputs, including:
* @li x: A Tensor. Must be one of the following types: float32, float64, int32, uint8, int16, int8, \n
*     complex64, int64, qint8, quint8, qint32, qint16, quint16, uint16, \n
*     complex128, float16, uint32, uint64, complex64, complex128. \n
* @li begin: A Tensor of type int32 or int64, for the index of the first value to select.

* @li end: A Tensor of type int32 or int64, for the index of the last value to select.

* @li strides: A Tensor of type int32 or int64, for the increment.

*@par Attributes:
* @li begin_mask: A Tensor of type int32. \n
      A bitmask where a bit "i" being "1" means to ignore the begin \n
      value and instead use the largest interval possible.
* @li end_mask: A Tensor of type int32. \n
      Analogous to "begin_mask".
* @li ellipsis_mask: A Tensor of type int32. \n
      A bitmask where bit "i" being "1" means the "i"th position \n
      is actually an ellipsis.
* @li new_axis_mask: A Tensor of type int32. \n
      A bitmask where bit "i" being "1" means the "i"th \n
      specification creates a new shape 1 dimension.
* @li shrink_axis_mask: A Tensor of type int32. \n
      A bitmask where bit "i" implies that the "i"th \n
      specification should shrink the dimensionality.

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*/
REG_OP(StridedSlice)
    .INPUT(x, TensorType::BasicType())
    .INPUT(begin, TensorType::IndexNumberType())
    .INPUT(end, TensorType::IndexNumberType())
    .INPUT(strides, TensorType::IndexNumberType())
    .ATTR(begin_mask, Int, 0)
    .ATTR(end_mask, Int, 0)
    .ATTR(ellipsis_mask, Int, 0)
    .ATTR(new_axis_mask, Int, 0)
    .ATTR(shrink_axis_mask, Int, 0)
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(StridedSlice)

/**
*@brief Extracts a strided slice of a tensor. Roughly speaking, this op \n
    extracts a slice of size "(end-begin)/stride" from the given input tensor. \n
    Starting at the location specified by "begin" the slice continues by \n
    adding "stride" to the index until all dimensions are not less than "end".

*@par Inputs:
*x: A Tensor. Must be one of the following types: float32, float64, int32, uint8, int16, int8, \n
*  complex64, int64, qint8, quint8, qint32, qint16, quint16, uint16, \n
*  complex128, float16, uint32, uint64, complex64, complex128.

*@par Attributes:
* @li begin: A Tensor of type int32 or int64. \n
      The index of the first value to select.
* @li end: A Tensor of type int32 or int64. \n
      The index of the last value to select.
* @li strides: A Tensor of type int32 or int64, for the increment. \n
* @li begin_mask: A Tensor of type int32. \n
      A bitmask where a bit "i" being "1" means to ignore the begin \n
      value and instead use the largest interval possible.
* @li end_mask: Analogous to "begin_mask". A Tensor of type as int32.
* @li ellipsis_mask: A Tensor of type int32. \n
      A bitmask where bit "i" being "1" means the "i"th position \n
      is actually an ellipsis.
* @li new_axis_mask: A Tensor of type int32. \n
      A bitmask where bit "i" being "1" means the "i"th \n
      specification creates a new shape 1 dimension.
* @li shrink_axis_mask: A Tensor of type int32. \n
      A bitmask where bit "i" implies that the "i"th \n
      specification should shrink the dimensionality.

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*/
REG_OP(StridedSliceD)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_UINT8, DT_INT8,
                          DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_UINT8, DT_INT8,
                          DT_BOOL}))
    .REQUIRED_ATTR(begin, ListInt)
    .REQUIRED_ATTR(end, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(begin_mask, Int, 0)
    .ATTR(end_mask, Int, 0)
    .ATTR(ellipsis_mask, Int, 0)
    .ATTR(new_axis_mask, Int, 0)
    .ATTR(shrink_axis_mask, Int, 0)
    .OP_END_FACTORY_REG(StridedSliceD)

/**
*@brief Since StridedSlice cuts out pieces of its "input" which is size "dy", \n
    its gradient will have the same shape (which is passed here as "shape"). \n
    The gradient will be zero in any element that the slice does not select.

*@par Inputs:
*dy: A Tensor. Must be one of the following types: float32, float64, int32, uint8, int16, int8, \n
*   complex64, int64, qint8, quint8, qint32, qint16, quint16, uint16, \n
*   complex128, float16, uint32, uint64, complex64, complex128.

*@par Attributes:
* @li shape: A Tensor of type int32 or int64.
* @li begin: A Tensor of type int32 or int64. \n
      The index of the first value to select.
* @li end: A Tensor of type int32 or int64. \n
      The index of the last value to select.
* @li strides: A Tensor of type int32 or int64, for the increment.
* @li begin_mask: A Tensor of type int32. \n
      A bitmask where a bit "i" being "1" means to ignore the begin \n
      value and instead use the largest interval possible.
* @li end_mask: A Tensor of type int32. \n
      Analogous to "begin_mask".
* @li ellipsis_mask: A Tensor of type int32. \n
      A bitmask where bit "i" being "1" means the "i"th position \n
      is actually an ellipsis.
* @li new_axis_mask: A Tensor of type int32. \n
      A bitmask where bit "i" being "1" means the "i"th \n
      specification creates a new shape 1 dimension.
* @li shrink_axis_mask: A Tensor of type int32. \n
      A bitmask where bit "i" implies that the "i"th \n
      specification should shrink the dimensionality.

*@par Outputs:
*output: A Tensor. Has the same type as "dy".
*/
REG_OP(StridedSliceGradD)
    .INPUT(dy, TensorType::BasicType())
    .OUTPUT(output, TensorType::BasicType())
    .REQUIRED_ATTR(shape, ListInt)
    .REQUIRED_ATTR(begin, ListInt)
    .REQUIRED_ATTR(end, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(begin_mask, Int, 0)
    .ATTR(end_mask, Int, 0)
    .ATTR(ellipsis_mask, Int, 0)
    .ATTR(new_axis_mask, Int, 0)
    .ATTR(shrink_axis_mask, Int, 0)
    .OP_END_FACTORY_REG(StridedSliceGradD)

/**
*@brief Since StridedSlice cuts out pieces of its "input" which is size "dy", \n
    its gradient will have the same shape (which is passed here as "shape"). \n
    The gradient will be zero in any element that the slice does not select.

*@par Inputs:
*Five inputs, including:
* @li shape: A Tensor of type int32 or int64.
* @li begin: A Tensor of type int32 or int64. \n
      The index of the first value to select.
* @li end: A Tensor of type int32 or int64. \n
      The index of the last value to select.
* @li strides: A Tensor of type int32 or int64, for the increment.
* @li dy: A Tensor. Must be one of the following types: \n
*     float32, float64, int32, uint8, int16, int8, \n
*     complex64, int64, qint8, quint8, qint32, qint16, quint16, uint16, \n
*     complex128, float16, uint32, uint64, complex64, complex128.

*@par Attributes:
* @li begin_mask: A Tensor of type int32. \n
      A bitmask where a bit "i" being "1" means to ignore the begin \n
      value and instead use the largest interval possible.
* @li end_mask: A Tensor of type int32. \n
      Analogous to "begin_mask".
* @li ellipsis_mask: A Tensor of type int32. \n
      A bitmask where bit "i" being "1" means the "i"th position \n
      is actually an ellipsis.
* @li new_axis_mask: A Tensor of type int32. \n
      A bitmask where bit "i" being "1" means the "i"th \n
      specification creates a new shape 1 dimension.
* @li shrink_axis_mask: A Tensor of type int32. \n
      A bitmask where bit "i" implies that the "i"th \n
      specification should shrink the dimensionality.

*@par Outputs:
*output: A Tensor has the same type as "dy".
*/
REG_OP(StridedSliceGrad)
    .INPUT(shape, TensorType::IndexNumberType())
    .INPUT(begin, TensorType::IndexNumberType())
    .INPUT(end, TensorType::IndexNumberType())
    .INPUT(strides, TensorType::IndexNumberType())
    .INPUT(dy, TensorType::BasicType())
    .OUTPUT(output, TensorType::BasicType())
    .ATTR(begin_mask, Int, 0)
    .ATTR(end_mask, Int, 0)
    .ATTR(ellipsis_mask, Int, 0)
    .ATTR(new_axis_mask, Int, 0)
    .ATTR(shrink_axis_mask, Int, 0)
    .OP_END_FACTORY_REG(StridedSliceGrad)

/**
*@brief Computes the sum along segments of a tensor.

*@par Inputs:
*Three inputs, including:
* @li x: A Tensor of type NumberType.
* @li segment_ids: A 1D Tensor of type IndexNumberType, whose shape is a prefix
* of "x.shape".
* @li num_segments: A Tensor of type IndexNumberType.

*@par Outputs:
*y: A Tensor of type RealNumberType.
*/
REG_OP(UnsortedSegmentSum)
    .INPUT(x, TensorType::NumberType())
    .INPUT(segment_ids, TensorType::IndexNumberType())
    .INPUT(num_segments, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .OP_END_FACTORY_REG(UnsortedSegmentSum)

/**
*@brief Computes the sum along segments of a tensor.

*@par Inputs:
*Two inputs, including:
* @li x: A Tensor of type float16, float32, int32, int8, uint8.
* @li segment_ids: A 1D Tensor of type int32, whose shape is a prefix
* of "x.shape".

*@par Attributes:
*num_segments: An int32, specifying the number of distinct segment IDs.

*@par Outputs:
*y: A Tensor with same type as "x".
*/
REG_OP(UnsortedSegmentSumD)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_UINT8}))
    .INPUT(segment_ids, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_UINT8}))
    .REQUIRED_ATTR(num_segments, Int)
    .OP_END_FACTORY_REG(UnsortedSegmentSumD)

/**
*@brief Reverses specific dimensions of a tensor.

*@par Inputs:
* Two inputs, including:\n
*@li x: An ND Tensor (up to 8D). \n
*Must be one of the following types: int8, uint8, int16, uint16, int32, int64, bool, float32, double
*@li axis: A 1D Tensor.\n
*Must be one of the following types: int32, int64

*@par Outputs:
*y: A Tensor. Has the same type and format as "x"

*@attention Constraints:
"axis" must be within the rank of "x".
*/
REG_OP(ReverseV2)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                          DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                          DT_COMPLEX64, DT_COMPLEX128, DT_STRING}))
    .INPUT(axis, TensorType({DT_INT32,DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                           DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                           DT_COMPLEX64, DT_COMPLEX128, DT_STRING}))
    .OP_END_FACTORY_REG(ReverseV2)

/**
*@brief Reverses specific dimensions of a tensor.

*@par Inputs:
* One input:
*@li x: An ND Tensor (up to 8D). \n
*Must be one of the following types: int8, uint8, int16, uint16, int32, int64, bool, float32, float64

*@par Attributes:
*axis: The indices of the dimensions to reverse.

*@par Outputs:
*y: A Tensor. Has the same type and format as "x"

*@attention Constraints:
"axis" must be within the rank of "x".
*/
REG_OP(ReverseV2D)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                          DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                          DT_COMPLEX64, DT_COMPLEX128, DT_STRING}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                           DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                           DT_COMPLEX64, DT_COMPLEX128, DT_STRING}))
    .REQUIRED_ATTR(axis, ListInt)
    .OP_END_FACTORY_REG(ReverseV2D)

/**
*@brief: Selects elements from "x1" or "x2", depending on "condition".

*@par Inputs:
* Three inputs, including:
* @li condition: A Tensor of type bool.
* @li x1: A Tensor. Must be one of the following types: float16, float32, int32, int8, uint8.
* @li x2: A Tensor of the same type as "x1".

*@par Outputs:
*y: A Tensor. Has the same type as "x1".
*/
REG_OP(Select)
    .INPUT(condition, TensorType({DT_BOOL}))
    .INPUT(x1,TensorType::BasicType())
    .INPUT(x2,TensorType::BasicType())
    .OUTPUT(y,TensorType::BasicType())
    .OP_END_FACTORY_REG(Select)

/**
*@brief: SelectV2s elements from "then" or "else", depending on "condition".

*@par Inputs:
* Three inputs, including:
* @li condition: A Tensor of type bool.
* @li then: A Tensor. Must be one of the following types: float16, float32, int32, int8, uint8.
* @li else: A Tensor of the same type as "then".

*@par Outputs:
*result: A Tensor. Has the same type as "then".

*/
REG_OP(SelectV2)
    .INPUT(condition, TensorType({DT_BOOL}))
    .INPUT(then,TensorType::BasicType())
    .INPUT(else,TensorType::BasicType())
    .OUTPUT(result,TensorType::BasicType())
    .OP_END_FACTORY_REG(SelectV2)


/**
*@brief: Computes the maximum along segments of a tensor.
*Computes a tensor such that output[i]=(data[i]) where max is over j such that segment_ids[j] == i.
*If the max is empty for a given segment ID i, output[i] = 0

*@par Inputs:
*Two inputs, include:
* @li x:A Tensor of type float16, float32, int32,int8,uint8.
* @li segment_ids:should be the size of the first dimension
        must sorted and need not cover all values in the full range of valid values
        must be positive intege

*@par Outputs:
*y:A Tensor with same type as "x".
*/
REG_OP(SegmentMax)
    .INPUT(x, TensorType::RealNumberType())
    .INPUT(segment_ids, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .OP_END_FACTORY_REG(SegmentMax)

/**
*@brief: Computes the maximum along segments of a tensor.
*Computes a tensor such that output[i]=(data[i]) where max is over j such that segment_ids[j] == i.
*If the max is empty for a given segment ID i, output[i] = 0

*@par Inputs:
*One inputs, include:
* @li x:A Tensor of type float16, float32, int32, int8,uint8 .

*@par Attributes:
* @li segment_ids:should be the size of the first dimension
        must sorted and need not cover all values in the full range of valid values
        must be positive intege

*@par Outputs:
*y:A Tensor with same type as "x".
*/
REG_OP(SegmentMaxD)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .REQUIRED_ATTR(segment_ids, ListInt)
    .OP_END_FACTORY_REG(SegmentMaxD)

/**
*@brief Returns a one-hot tensor. The locations represented by index in "x" take value "on_value",
*         while all other locations take value "off_value".

*@par Inputs:
*Four inputs, including:
* @li x: A Tensor of indices. Must be one of the following types: int32, uint8, int64.
* @li depth: A scalar of type int32. The depth of the one hot dimension.
* @li on_value: A scalar. The value to fill in output when indices[j] = i,
*     Must be one of the following types: float16, float32, int32, int8, uint8.
* @li off_value: A scalar. The value to fill in output when indices[j] != i,
*     Has the same type as "on_value".

*@par Attributes:
*axis: An int. The axis to fill. Defaults to "-1".

*@par Outputs:
*y: A Tensor. Has the same type as "on_value".
*/
REG_OP(OneHot)
    .INPUT(x, TensorType({DT_UINT8, DT_INT32, DT_INT64}))
    .INPUT(depth, TensorType({DT_INT32}))
    .INPUT(on_value, TensorType::BasicType())
    .INPUT(off_value, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(axis, Int, -1)
    .OP_END_FACTORY_REG(OneHot)

/**
*@brief Returns a one-hot tensor. The locations represented by index in "x" take value "on_value",
*         while all other locations take value "off_value".

*@par Inputs:
*Three inputs, including:
*@li x: A Tensor of indices. Must be one of the following types: int32, uint8, int64.
*@li on_value: A scalar. The value to fill in output when indices[j] = i,
*     Must be one of the following types: float16, float32, int32, int8, uint8.
*@li off_value: A scalar. The value to fill in output when indices[j] != i,
*     Has the same type as "on_value".

*@par Attributes:
*@li depth: A scalar of type int32. The depth of the one hot dimension.
*@li axis: An int. The axis to fill. Defaults to "-1".

*@par Outputs:
*y: A Tensor. Has the same type as "on_value".
*/
REG_OP(OneHotD)
    .INPUT(x, TensorType({DT_UINT8, DT_INT32}))
    .INPUT(on_value, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT8,
                                 DT_INT8}))
    .INPUT(off_value, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT8,
                                  DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT8, DT_INT8}))
    .REQUIRED_ATTR(depth, Int)
    .ATTR(axis, Int, -1)
    .OP_END_FACTORY_REG(OneHotD)

/**
*@brief Extracts a slice from a tensor.\n
        This operation extracts a slice of size "size" from a tensor "x" starting at the location specified by "begin".

*@par Inputs:
*@li x: A Tensor. Must be one of the following types: float16, float32, int8, int16, int32, int64, uint8, uint16, uint32, uint64.
*@li offsets: A Tensor of type int32 or int64. The starting location for the slice.
*@li size: A Tensor of type int32 or int64. The tensor shape.

*@par Outputs:
*y: A Tensor. Has the same type as "x". The slice extracted from the tensor.
*/
REG_OP(Slice)
    .INPUT(x, TensorType::BasicType())
    .INPUT(offsets, TensorType::IndexNumberType())
    .INPUT(size, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(Slice)

/**
*@brief Extracts a slice from a tensor.\n
        This operation extracts a slice of size "size" from a tensor "x" starting at the location specified by "begin".

*@par Inputs:
*x: A Tensor. Must be one of the following types: float16, float32, int8, int16, int32, int64, uint8, uint16, uint32, uint64.

*@par Attributes:
*@li offsets: The starting location for the slice.
*@li size: The tensor shape.

*@par Outputs:
*y: A Tensor. Has the same type as "x". The slice extracted from the tensor.
*/
REG_OP(SliceD)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(offsets, ListInt)
    .REQUIRED_ATTR(size, ListInt)
    .OP_END_FACTORY_REG(SliceD)

/**
* @brief Finds values and indices of the "k" largest elements for the last
* dimension.

* @par Inputs:
* @li x: A 1D or higher tensor of type float16, with the last dimension at
* least "k".
* Specifies the data to sort.
* @li assist_seq: A 1D tensor of type float16.
* with size of 2N, which "N" is the last dimension.
* The first N numbers is indices, and the next N numbers is deviation of casting
* float16 to int32.

* @par Attributes:
* @li k: A required int that is at least 0, specifying the number of top elements \n
* to look for along the last dimension (along each row for matrices).
* @li sorted: An optional bool. Defaults to true.\n
* If true, the resulting "k" elements will be sorted by the values in descending
* order.
* @li dim: An optional int. Defaults to -1. For reserved use.\n
* @li largest: An optional bool. Defaults to true. For reserved use.\n

* @par Outputs:
* @li values: A Tensor, specifying the sorted data. Has the same type as "input".
* @li indices: A Tensor of type int32, specifying the indices of sorted data.

* @attention Constraints:
* @li k =< 4096
* @li Size of the last dimension =< 65500
* @li sorted = true
* @li Don't support to get score on the platform of Ascend310
*/
REG_OP(TopKD)
    .INPUT(x, TensorType::RealNumberType())
    .INPUT(assist_seq, TensorType({DT_FLOAT16}))
    .OUTPUT(values, TensorType::RealNumberType())
    .OUTPUT(indices, TensorType({DT_INT32}))
    .REQUIRED_ATTR(k, Int)
    .ATTR(sorted, Bool, true)
    .ATTR(dim, Int, -1)
    .ATTR(largest, Bool, true)
    .OP_END_FACTORY_REG(TopKD)

/**
* @brief Finds values and indices of the "k" largest elements for the last
* dimension.

* @par Inputs:
* @li x: A 1D or higher tensor of type BasicType, with the last dimension
* at least "k".
* @li k: A 0D Tensor of type int32.\n
* Number of top elements to look for along the last dimension (along each row
* for matrices).

* @par Attributes:
* @li sorted: An optional bool. Defaults to true.\n
* If true, the resulting "k" elements will be sorted by the values in descending
* order.
* @li T: Indicator of indices type.

* @par Outputs:
* @li values: A Tensor, specifying the sorted data. Has the same type as
* "input".
* @li indices: A Tensor of type int32, specifying the indices of sorted data.

* @see TopK()
*/
REG_OP(TopK)
    .INPUT(x, TensorType::RealNumberType())
    .INPUT(k, TensorType({DT_INT32}))
    .OUTPUT(values, TensorType::RealNumberType())
    .OUTPUT(indices, TensorType({DT_INT32}))
    .ATTR(sorted, Bool, true)
    .OP_END_FACTORY_REG(TopK)
/**
*@brief Creates a new tensor by applying sparse "updates" to individual values or slices within a tensor (initially zero for numeric, empty for string) of the given "shape" according to "indices".

*@par Inputs:
*Inputs including: \n
* @li indices: A required index tensor. Must be one of the following types: float32, float16, int32, int8, uint8.
* @li x: A required slice tensor. Must be one of the following types: float32, float16, int32, int8, uint8.
* @li shape: A required list of int32, specifying the output shape.
*@par Outputs:
*y:A output Tensor with same datatype as "updates".

*@attention Constraints:\n
*@li "y" has the same shape as "shape".
*@li "y" has the same type as "x".
*/
REG_OP(ScatterNd)
    .INPUT(indices, TensorType::BasicType())
    .INPUT(x, TensorType::BasicType())
    .INPUT(shape, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(ScatterNd)
/**
*@brief Creates a new tensor by applying sparse "updates" to individual values or slices within a tensor (initially zero for numeric, empty for string) of the given "shape" according to "indices".

*@par Inputs:
*Inputs including: \n
* @li indices: A required index tensor. Must be one of the following types: float32, float16, int32, int8, uint8.
* @li x: A required slice tensor. Must be one of the following types: float32, float16, int32, int8, uint8.
*@par Attributes:
* @li shape: A required list of int32, specifying the output shape.
*@par Outputs:
*y: A Tensor. Has the same type as "updates".

*@attention Constraints:\n
*@li "y" has the same shape as "shape".
*@li "y" has the same type as "x".
*/
REG_OP(ScatterNdD)
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT16}))
    .REQUIRED_ATTR(shape, ListInt)
    .OP_END_FACTORY_REG(ScatterNdD)

/**
* @brief Says whether the targets are in the top "k" predictions.

* @par Inputs:
* Three inputs, including:
* @li x1: A 2D Tensor of type float32. A "batch_size * classes" tensor.
* @li x2: A 1D Tensor of type IndexNumberType. A batch_size tensor of class ids.

* @par Attributes:
* @li k: A required int32, specifying the number of top elements to look at for
* computing precision.

* @par Outputs:
* y: A Tensor of type bool.

* @see InTopK()
*/
REG_OP(InTopKD)
    .INPUT(x1, TensorType({DT_FLOAT}))
    .INPUT(x2, TensorType({IndexNumberType}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .REQUIRED_ATTR(k, Int)
    .OP_END_FACTORY_REG(InTopKD)

/**
* @brief Says whether the targets are in the top "k" predictions.

* @par Inputs:
* Two inputs, including:
* @li x1: A 2D Tensor of type float32. A "batch_size * classes" tensor.
* @li x2: A 1D Tensor of type IndexNumberType. A batch_size tensor of class ids.
* @li k: A 1D Tensor of the same type as "x2".
* Specifies the number of top elements to look at for computing precision.

* @par Outputs:
* y: A Tensor of type bool.
*/
REG_OP(InTopK)
    .INPUT(x1, TensorType({DT_FLOAT}))
    .INPUT(x2, TensorType(IndexNumberType))
    .INPUT(k, TensorType({IndexNumberType}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(InTopK)

/**
* @brief Assigns "value" to the sliced l-value reference of "var".\n
* The values of "value" are assigned to the positions in the variable. "var"\n
* that are selected by the slice parameters. The slice parameters "begin, "end",\n
* "strides", etc. work exactly as in "StridedSlice".

* @par Inputs:
* @li var: A mutable ND Tensor of type BasicType.
* @li begin: A mutable ND Tensor of type IndexNumberType.
* Specifies the index of the first value to select.
* @li end: A mutable ND Tensor of type IndexNumberType.
* Specifies the index of the last value to select.
* @li strides: A mutable ND Tensor of type IndexNumberType.
* Specifies the stride to select.
* @li input_value: A mutable ND Tensor of type BasicType.

* @par Attributes:
* @li begin_mask: An optional int. Defaults to "0".
* @li end_mask: An optional int. Defaults to "0".
* @li ellipsis_mask: An optional int. Defaults to "0".
* @li new_axis_mask: An optional int. Defaults to "0".
* @li shrink_axis_mask: An optional int. Defaults to "0".

* @par Outputs:
* var: A mutable Tensor. Has the same type as "var".

* @attention Constraints:
* This operator currently does not support broadcasting. Therefore, the shape
* of "value" must be exactly the shape produced by the slice of "var".

* @see StridedSlice()
*/
REG_OP(StridedSliceAssign)
    .INPUT(var, TensorType(BasicType))
    .INPUT(begin, TensorType(IndexNumberType))
    .INPUT(end, TensorType(IndexNumberType))
    .INPUT(strides, TensorType(IndexNumberType))
    .INPUT(input_value, TensorType(BasicType))
    .OUTPUT(var, TensorType(BasicType))
    .ATTR(begin_mask, Int, 0)
    .ATTR(end_mask, Int, 0)
    .ATTR(ellipsis_mask, Int, 0)
    .ATTR(new_axis_mask, Int, 0)
    .ATTR(shrink_axis_mask, Int, 0)
    .OP_END_FACTORY_REG(StridedSliceAssign)

/**
* @brief Assigns "value" to the sliced l-value reference of "var".\n
* The values of "value" are assigned to the positions in the variable. "var"\n
* that are selected by the slice parameters. The slice parameters "begin, "end",\n
* "strides", etc. work exactly as in "StridedSlice".

* @par Inputs:
* @li var: A mutable ND Tensor of type BasicType.
* @li input_value: A mutable ND "Tensor" of type BasicType.


* @par Attributes:
* @li begin: A required list of ints.
* Specifies the index of the first value to select.
* @li end: A required list of ints.
* Specifies the index of the last value to select.
* @li strides: A required list of ints. Specifies the stride to select.
* @li begin_mask: An optional int. Defaults to "0".
* @li end_mask: An optional int. Defaults to "0".
* @li ellipsis_mask: An optional int. Defaults to "0".
* @li new_axis_mask: An optional int. Defaults to "0".
* @li shrink_axis_mask: An optional int. Defaults to "0".

* @par Outputs:
* var: A mutable Tensor. Has the same type as input "var".

* @attention Constraints:
* This operator currently does not support broadcasting. Therefore, the shape of
* "value" shape must be exactly the shape produced by the slice of "var".

* @see StridedSlice()
*/
REG_OP(StridedSliceAssignD)
    .INPUT(var, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .INPUT(input_value, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OUTPUT(var, TensorType(BasicType))
    .REQUIRED_ATTR(begin, ListInt)
    .REQUIRED_ATTR(end, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(begin_mask, Int, 0)
    .ATTR(end_mask, Int, 0)
    .ATTR(ellipsis_mask, Int, 0)
    .ATTR(new_axis_mask, Int, 0)
    .ATTR(shrink_axis_mask, Int, 0)
    .OP_END_FACTORY_REG(StridedSliceAssignD)

/**
*@brief Gather slices from "params" according to "indices"."indices" must be \n
    an integer tensor of any dimension(usually 0-D or 1-D). \n
    Produces an output tensor with shape "indices.shape + params.shape[1:]".

*@par Inputs:
*Two inputs, including:
* @li x: A Tensor. Must be one of the following types: float32, float64, int32, uint8, int16, int8, \n
*     complex64, int64, qint8, quint8, qint32, qint16, quint16, uint16, \n
*     complex128, float16, uint32, uint64, complex64, complex128.
* @li indices: A Tensor of type int32 or int64.

*@par Attributes:
*validate_indices: A bool specifying whether to verify the argument of "indice".

*@par Outputs:
*y: A Tensor. Has the same type as "x".

*@attention Constraints:
* "indices" is in the range [0, x.shape[0]).
*/
REG_OP(Gather)
    .INPUT(x, TensorType::BasicType())
    .INPUT(indices, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(validate_indices, Bool, true)
    .OP_END_FACTORY_REG(Gather)

/**
*@brief Computes the cumulative product of the tensor "x" along "axis".

*@par Inputs:
* Two inputs, including:
*@li x: A Tensor. Must be one of the following types: int32, float32, float16, int8, uint8.
*@li axis A Tensor of type int32. Defaults to "0".
*
*@par Attributes:
*@li exclusive: If "False", performs inclusive cumprod, which means that the first element of the input is identical to the first element of the output. If "True", performs exclusive cumprod.
*@li reverse: A bool. Defaults to "False".
*
*@par Outputs:
*y: A Tensor. Has the same type as "x".
*/
REG_OP(Cumprod)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axis, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(exclusive, Bool, false)
    .ATTR(reverse, Bool, false)
    .OP_END_FACTORY_REG(Cumprod)

/**
*@brief Computes the cumulative product of the tensor "x" along "axis".

*@par Inputs:
* One input:
*x: A Tensor. Must be one of the following types: int32, float32, float16, int8, uint8.
*
*@par Attributes:
*@li axis A Tensor of type int32. Defaults to "0".
*@li exclusive: If "False", performs inclusive cumprod, which means that the first element of the input is identical to the first element of the output. If "True", performs exclusive cumprod.
*@li reverse: A bool. Defaults to "False".
*
*@par Outputs:
*y: A Tensor. Has the same type as "x".
*/
REG_OP(CumprodD)
    .INPUT(x, TensorType::NumberType())
    .OUTPUT(y, TensorType::NumberType())
    .REQUIRED_ATTR(axis, Int)
    .ATTR(exclusive, Bool, false)
    .ATTR(reverse, Bool, false)
    .OP_END_FACTORY_REG(CumprodD)

/**
*@brief Computes the cumulative sum of the tensor "x" along "axis".

*@par Inputs:
* Two inputs, including:
*@li x: A Tensor. Must be one of the following types: int32, float32, float16, int8, uint8.
*@li axis A Tensor of type int32. Defaults to "0".
*
*@par Attributes:
*@li exclusive: If "False", performs inclusive cumsum, which means that the first element of the input is identical to the first element of the output. If "True", performs exclusive cumsum.
*@li reverse: A bool. Defaults to "False".
*
*@par Outputs:
*@li y: A Tensor. Has the same type as "x".
*/
REG_OP(Cumsum)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axis, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(exclusive, Bool, false)
    .ATTR(reverse, Bool, false)
    .OP_END_FACTORY_REG(Cumsum)

/**
*@brief Computes the cumulative sum of the tensor "x" along "axis".
*
*@par Inputs:
* One input:
*x: A Tensor. Must be one of the following types: int32, float32, float16, int8, uint8.
*
*@par Attributes:
*@li axis A Tensor of type int32. Defaults to "0".
*@li exclusive: If "False", performs inclusive cumsum, which means that the first element of the input is identical to the first element of the output. If "True", performs exclusive cumsum.
*@li reverse: A bool. Defaults to "False".
*
*@par Outputs:
*y: A Tensor. Has the same type as "x".
*/
REG_OP(CumsumD)
    .INPUT(x, TensorType::NumberType())
    .OUTPUT(y, TensorType::NumberType())
    .REQUIRED_ATTR(axis, Int)
    .ATTR(exclusive, Bool, false)
    .ATTR(reverse, Bool, false)
    .OP_END_FACTORY_REG(CumsumD)

/**
*@brief Updates specified rows with values in v. \n
*Computes x[i, :] = v; return x.
*@par Inputs:
*Three inputs, including:
* @li x: A Tensor. \n
*     TensorType::NumberType().
* @li indices: A vector of type int32. \n
*     Indices into the left-most dimension of "x".
* @li v: A Tensor of the same type as "x". \n
*     Same dimension sizes as x except the first dimension, \n
*     which must be the same as the size of "indices".

*@par Outputs:
*y: A Tensor of the same type as "x". \n
*   An alias of "x". The content of "y" is undefined if there are duplicates in indices.
*/
REG_OP(InplaceUpdate)
    .INPUT(x, TensorType::BasicType())
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(v, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(InplaceUpdate)

/**
*@brief Updates specified rows with values in v. \n
*Computes x[i, :] = v; return x.
*@par Inputs:
*Two inputs, including:
* @li x: A Tensor. \n
*     TensorType::NumberType().
* @li v: A Tensor of the same type as "x". \n
* Same dimension sizes as "x" except the first dimension, which must be the same as the size of "indices".

*@par Attributes:
*indices: A required list of ints. Indices into the left-most dimension of "x".

*@par Outputs:
*y: A Tensor of the same type as "x". \n
*   An alias of "x". The content of "y" is undefined if there are duplicates in indices.
*/
REG_OP(InplaceUpdateD)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(v, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .REQUIRED_ATTR(indices, ListInt)
    .OP_END_FACTORY_REG(InplaceUpdateD)

/**
*@brief Adds "v" into specified rows of "x". \n
*Computes y = x; y[i, :] += v.
*@par Inputs:
*Three inputs, including:
* @li x: A Tensor. \n
*     TensorType::NumberType().
* @li indices: A vector of type int32. \n
*     Indices into the left-most dimension of "x".
* @li v: A Tensor of the same type as "x". \n
*     Same dimension sizes as x except the first dimension, \n
*     which must be the same as the size of "indices".

*@par Outputs:
*y: A Tensor of the same type as "x". \n
*  An alias of "x". The content of "y" is undefined if there are duplicates in indices.
*/
REG_OP(InplaceAdd)
    .INPUT(x, TensorType::BasicType())
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(v, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(InplaceAdd)

/**
*@brief Adds "v" into specified rows of "x". \n
*Computes y = x; y[i, :] += v.
*@par Inputs:
*Two inputs, including:
* @li x: A Tensor. \n
*     TensorType::NumberType().
* @li v: A Tensor of the same type as "x". \n
* Same dimension sizes as "x" except the first dimension, which must be the same as the size of "indices".

*@par Attributes:
*indices: A required list of ints. Indices into the left-most dimension of "x".

*@par Outputs:
*y: A Tensor of the same type as "x". \n
*  An alias of "x". The content of "y" is undefined if there are duplicates in indices.
*/
REG_OP(InplaceAddD)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(v, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .REQUIRED_ATTR(indices, ListInt)
    .OP_END_FACTORY_REG(InplaceAddD)

/**
*@brief Subtracts "v" into specified rows of "x". \n
*Computes y = x; y[i, :] -= v; return y.
*@par Inputs:
**Three inputs, including:
* @li x: A Tensor. TensorType::NumberType().
* @li indices: A vector of type int32. Indices into the left-most dimension of x.
* @li v: A Tensor of the same type as "x". \n
* Same dimension sizes as "x" except the first dimension, which must be the same as the size of "indices".

*@par Outputs:
*y: A Tensor. Has the same type as "x".\n
*  An alias of "x". The content of "y" is undefined if there are duplicates in indices.
*/
REG_OP(InplaceSub)
    .INPUT(x, TensorType::BasicType())
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(v, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(InplaceSub)

/**
*@brief Subtracts "v" into specified rows of "x". \n
*Computes y = x; y[i, :] -= v.

*@par Inputs:
**Two inputs, including:
* @li x: A Tensor. TensorType::NumberType().
* @li v: A Tensor of the same type as "x". \n
* Same dimension sizes as "x" except the first dimension, which must be the same as the size of "indices".

*@par Attributes:
*indices: A required list of ints. Indices into the left-most dimension of "x".

*@par Outputs:
*y: A Tensor. Has the same type as "x".\n
* An alias of x. The content of y is undefined if there are duplicates in indices.
*/
REG_OP(InplaceSubD)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(v, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .REQUIRED_ATTR(indices, ListInt)
    .OP_END_FACTORY_REG(InplaceSubD)

/**
* @brief Applies sparse addition to input "x" using individual values or slices\n
* from "updates" according to "indices". The updates are non-aliasing: "x" is\n
* only modified in-place if no other operations will use it. Otherwise, a copy\n
* of "x" is made. This operation has a gradient with respect to both "x" and
* "updates".

* @par Inputs:
* Three inputs, including:
* @li x: A Tensor of type NumberType. A batch_size x classes tensor.
* @li indices: A Tensor of type IndexNumberType. Specifies the indices into "x".
* @li updates: A Tensor. Must have the same type as "x".
* Specifies the updated values to add to "x".

* @par Outputs:
* y: A Tensor with the same shape as "x", containing values of "x" updated with
* "updates".

* @see ScatterNd(),ScatterNdAdd()
*/
REG_OP(ScatterNonAliasingAdd)
    .INPUT(x, TensorType::NumberType())
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType::NumberType())
    .OUTPUT(y, TensorType::NumberType())
    .OP_END_FACTORY_REG(ScatterNonAliasingAdd)

/**
* @brief Computes the minimum along segments of a tensor.

* @par Inputs:
* Three inputs, including:
* @li x: A Tensor of type RealNumberType.
* @li segment_ids: A 1D Tensor of type IndexNumberType, whose shape is a prefix
* of "x.shape".
* @li k: A Tensor.

* @par Outputs:
* y: A Tensor of type RealNumberType.

* @see UnsortedSegmentSum(), UnsortedSegmentProd(),
*/
REG_OP(UnsortedSegmentMin)
    .INPUT(x, TensorType::RealNumberType())
    .INPUT(segment_ids, TensorType::IndexNumberType())
    .INPUT(num_segments, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .OP_END_FACTORY_REG(UnsortedSegmentMin)

/**
* @brief Computes the minimum along segments of a tensor.

* @par Inputs:
* Three inputs, including:
* @li x: A Tensor of type RealNumberType.
* @li segment_ids: A 1D Tensor of type IndexNumberType, whose shape is a prefix
* of "x.shape".
* @li k: A Tensor.

* @par Attributes:
* num_segments: A required int32, specifying the number of distinct segment IDs.

* @par Outputs:
* y: A Tensor of type RealNumberType.

* @see UnsortedSegmentProdD(),
*/
REG_OP(UnsortedSegmentMinD)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(segment_ids, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .REQUIRED_ATTR(num_segments, Int)
    .OP_END_FACTORY_REG(UnsortedSegmentMinD)

/**
* @brief Computes the product along segments of a tensor.

* @par Inputs:
* Three inputs, including:
* @li x: A Tensor of type RealNumberType.
* @li segment_ids: A 1D Tensor of type IndexNumberType, whose shape is a prefix
* of "x.shape".
* @li k: A Tensor.

* @par Outputs:
* y: A Tensor of type RealNumberType.

* @see UnsortedSegmentSum(), UnsortedSegmentMin(),
*/
REG_OP(UnsortedSegmentProd)
    .INPUT(x, TensorType::NumberType())
    .INPUT(segment_ids, TensorType::IndexNumberType())
    .INPUT(num_segments, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .OP_END_FACTORY_REG(UnsortedSegmentProd)

/**
* @brief Computes the product along segments of a tensor.

* @par Inputs:
* Three inputs, including:
* @li x: A Tensor of type RealNumberType.
* @li segment_ids: A 1D Tensor of type IndexNumberType, whose shape is a prefix
* of "x.shape".
* @li k: A Tensor.

* @par Attributes:
* num_segments: An int32, specifying the number of distinct segment IDs.

* @par Outputs:
* y: A Tensor of type RealNumberType.

* @see UnsortedSegmentMinD()
*/
REG_OP(UnsortedSegmentProdD)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(segment_ids, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .REQUIRED_ATTR(num_segments, Int)
    .OP_END_FACTORY_REG(UnsortedSegmentProdD)

/**
*@brief Performs object detection.

*@par Inputs:
*@li cls_prob: An NCHW tensor of type float16 or float32, specifying the probability of the proposal is the background class.
*@li bbox_delta: An NCHW tensor of type float16 or float32, specifying the coordinates of the proposals bounding boxes.
*@li im_info: An ND tensor of type float16 or float32, specifying the Image information.

*@par Attributes:
*@li feat_stride: A optional float32, specifying the stride of the sliding window. Must be greater than "0".Defaults to "16".
*@li base_size: A optional float32, specifying the size of the generated base box. Must be greater than "0". Defaults to "16".
*@li min_size: A optional float32, specifying the minimum edge length of a proposal. A box with any edge less than this value is removed. Must be greater than "0". Defaults to "16".
*@li ratio: A optional list of floats, specifying the aspect ratio of the generated base box. Defaults to [0.5, 1, 2].
*@li scale: A optional list of floats, specifying the ratio of the size of the generated base box to "base_size". Defaults to [8, 16, 32].
*@li pre_nms_topn: A required int, specifying top K boxes before NMS. For float16 input, pre_nms_topn <= 6000. For float32 input, pre_nms_topn <= 3000. Defaults to "3000".
*@li post_nms_topn: A required int, specifying the number of boxes to be output after NMS. The value is a multiple of 16. For float16 input, post_nms_topn <= 6000. For float32 input, post_nms_topn <= 3000 (the maximum multiple of 16 is 2992 within the range). Defaults to "304".
*@li iou_threshold: A required float32, specifying the NMS threshold. The value range is (0,1]. Defaults to "0.7".
*@li output_actual_rois_num: An optional bool. Defaults to "false".

*@par Outputs:
*@li rois: A Tensor with shape [batch, 5, post_nms_topn], of type float16, specifying the output box information. "post_nms_topn" must be a multiple of 16. The dimension "5" indicates (batchID, x1, y1, x2, y2). The number of BBoxes output per batch is determined by "actual_rois_num".
*@li actual_rois_num: A Tensor with shape [batch, 8], of type int32, specifying the number of BBoxes output per batch.
*/
 REG_OP(Proposal)
     .INPUT(cls_prob, TensorType({DT_FLOAT16, DT_FLOAT}))
     .INPUT(bbox_delta, TensorType({DT_FLOAT16, DT_FLOAT}))
     .INPUT(im_info, TensorType({DT_FLOAT16, DT_FLOAT}))
     .OUTPUT(rois, TensorType({DT_FLOAT16, DT_FLOAT}))
     .OUTPUT(actual_rois_num, TensorType({DT_INT32}))
     .ATTR(feat_stride, Float, 16)
     .ATTR(base_size, Float, 16)
     .ATTR(min_size, Float, 16)
     .ATTR(ratio, ListFloat, {0.5, 1, 2})
     .ATTR(scale, ListFloat, {8, 16, 32})
     .ATTR(pre_nms_topn, Int, 3000)
     .ATTR(post_nms_topn, Int, 304)
     .ATTR(iou_threshold, Float, 0.7)
     .ATTR(output_actual_rois_num, Bool, false)
     .OP_END_FACTORY_REG(Proposal)

/**
*@brief Performs object detection. Different from Proposal, this is an internal API called after FE fusion and has an additional "rpn_bbox" attribute. The suffix "D" in the API name will be removed from the generated model.

*@par Inputs:
*@li cls_prob: An NCHW tensor of type float16, specifying the probability of the proposal is the background class.
*@li bbox_delta: An NCHW tensor of type float16, specifying the coordinates of the proposals bounding boxes.
*@li im_info: An ND tensor of type float16 or float32, specifying the Image information.
*@li rpn_bbox: An NCHW tensor of type float16, specifying the coordinates of the proposals bounding boxes.

*@par Attributes:
*@li feat_stride: A required float32, specifying the stride of the sliding window. Must be greater than "0".Defaults to "16".
*@li base_size: A required float32, specifying the size of the generated base box. Must be greater than "0". Defaults to "16".
*@li min_size: A required float32, specifying the minimum edge length of a proposal. A box with any edge less than this value is removed. Must be greater than "0". Defaults to "16".
*@li ratio: A required list of floats, specifying the aspect ratio of the generated base box. Defaults to [0.5, 1, 2].
*@li scale: A required list of floats, specifying the ratio of the size of the generated base box to "base_size". Defaults to [8, 16, 32].
*@li pre_nms_topn: A required int, specifying top K boxes before NMS. For float16 input, pre_nms_topn <= 6000. For float32 input, pre_nms_topn <= 3000. Defaults to "3000".
*@li post_nms_topn: A required int, specifying the number of boxes to be output after NMS. The value is a multiple of 16. For float16 input, post_nms_topn <= 6000. For float32 input, post_nms_topn <= 3000 (the maximum multiple of 16 is 2992 within the range). Defaults to "304".
*@li iou_threshold: A required float32, specifying the NMS threshold. The value range is (0,1]. Defaults to 0.7.
*@li output_actual_rois_num: An optional bool. Defaults to "false".

*@par Outputs:
*@li rois: A Tensor with shape [batch, 5, post_nms_topn], of type float16, specifying the output box information. "post_nms_topn" must be a multiple of 16. The dimension "5" indicates (batchID, x1, y1, x2, y2). The number of BBoxes output per batch is determined by "actual_rois_num".
*@li actual_rois_num: A Tensor with shape [batch, 8], of type int32, specifying the number of BBoxes output per batch.
*/
REG_OP(ProposalD)
     .INPUT(cls_prob, TensorType({DT_FLOAT16, DT_FLOAT}))
     .INPUT(bbox_delta, TensorType({DT_FLOAT16, DT_FLOAT}))
     .INPUT(im_info, TensorType({DT_FLOAT16, DT_FLOAT}))
     .INPUT(rpn_bbox, TensorType({DT_FLOAT16, DT_FLOAT}))
     .OUTPUT(rois, TensorType({DT_FLOAT16, DT_FLOAT}))
     .OUTPUT(actual_rois_num, TensorType({DT_INT32}))
     .ATTR(feat_stride, Float, 16)
     .ATTR(base_size, Float, 16)
     .ATTR(min_size, Float, 16)
     .ATTR(ratio, ListFloat, {0.5, 1, 2})
     .ATTR(scale, ListFloat, {8, 16, 32})
     .ATTR(pre_nms_topn, Int, 3000)
     .ATTR(post_nms_topn, Int, 304)
     .ATTR(iou_threshold, Float, 0.7)
     .ATTR(output_actual_rois_num, Bool, false)
     .OP_END_FACTORY_REG(ProposalD)

/**
*@brief Performs plane or channel conversion on YoloV2.
* If reverse=true: (N, H, W, C)->(N, H*stride, W*stride, C/(stride*stride))
* If reverse=false: (N, H, W, C)->(N, H/stride, W/stride, C*(stride*stride))

*@par Inputs:
*x: An (N, H, W, C) tensor. All data types are supported.

*@par Attributes:
*@li stride: An optional int32, specifying the plane or channel scaling factor. Defaults to "2".
*@li reverse: An optional bool, specifying the conversion mode. If "true", depth to space conversion is performed. If "false", space to depth conversion is performed. Defaults to "false".

*@par Outputs:
*y: An (N, H, W, C) tensor. All data types are supported.

*@attention Constraints:
*@li If reverse=true: C/(stride*stride) yields an integer result. If reverse=false: W/stride and H/stride yield integer results.
*/
REG_OP(PassThrough)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64}))
    .ATTR(stride, Int, 2)
    .ATTR(reverse, Bool, false)
    .OP_END_FACTORY_REG(PassThrough)

/**
*@brief Crops the input tensor x to the shape of size. For example: \n
*(1) x: bottom to be cropped, with shape (20, 50, 512, 512);\n
*(2) size: reference input for cropping, with shape (20, 10, 256, 256);\n
*(3) axis = 1;\n
*(4) offset = (25, 128, 128);\n
*(5) y = x[:, 25:25 + size.shape[1], 128:128 + size.shape[2], 128:128 + size.shape[3]].

*@par Inputs:
*Inputs include: \n
* @li x: A required Tensor. Must be one of the following types: float16, float32, int8, uint8, int16, uint16, int32, uint32,int64, uint64.
* @li size: A required Tensor. Must be one of the following types: float16, float32, int8, uint8, int16, uint16, int32, uint32, int64, uint64.
*@par Attributes:
*@li axis: A required int32, specifying the first dimension to crop.
*@li offset: A required array, specifying the shift for all/each dimension to align the cropped bottom with the reference bottom. Must be one of the following types: float16, float32, int8, uint8, int16, uint16, int32, uint32, int64, uint64.
*@par Outputs:
*y: A required Tensor. Has the same type and shape as "size".

*@attention Constraints:\n
*@li "y" must have the same type and shape as "size". "x" must have the same type as "size".
*@li "axis" must be less than the rank of "x".
*@li The "offset" for each dimension must not exceed the maximum value of the corresponding dimension of "x".
*@li The array length of "offset" plus the value of "axis" equals to the rank of "y".
*/
REG_OP(Crop)
      .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32,DT_INT64,DT_UINT64}))
      .INPUT(size, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32,DT_INT64,DT_UINT64}))
      .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32,DT_INT64,DT_UINT64}))
      .ATTR(axis, Int, 2)
      .REQUIRED_ATTR(offsets, ListInt)
      .OP_END_FACTORY_REG(Crop)

/**
*@brief Extends the input with copies of data along a specified dimension. For example: \n
*(1) If x = [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], with shape (2, 3, 2);\n
*(2) axis = 1;\n
*(3) tiles = 2;\n
*(4) Then, y = [[[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12], [7, 8], [9, 10], [11, 12]]], with shape (2, 6, 2).

*@par Inputs:
* One input:
*input_x: A Tensor with any format. Must be one of the following types: float16, float32, int8, int16, int32, int64, uint8, uint16, uint32, uint64.

*@par Attributes:
*@li axis: An optional int32, specifying the axis to tile. Defaults to 1.
*@li tiles: A required int32, specifying the number of copies (tiles) to output.

*@par Outputs:
*output_y: A Tensor of any format. Must be one of the following types: float16, float32, int8, int16, int32, int64, uint8, uint16, uint32, uint64.

*@attention Constraints:\n
*@li "axis" must be within the rank of the input tensor.
*@li "tiles" must be greater than 1.
*/
REG_OP(TileWithAxis)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT64, DT_INT32,
    DT_INT16, DT_INT8, DT_UINT64, DT_UINT32, DT_UINT16, DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT64, DT_INT32,
    DT_INT16, DT_INT8, DT_UINT64, DT_UINT32, DT_UINT16, DT_UINT8}))
    .ATTR(axis, Int, 1)
    .REQUIRED_ATTR(tiles, Int)
    .OP_END_FACTORY_REG(TileWithAxis)

/**
*@brief Read data with offset and stride.

*@par Inputs:
*One input:\n
*x: A Tensor. Must be one of the following types: float16, int8.

*@par Attributes:
*@li stride_list: An optional 5D list of type int32. Defaults to "[1,1,1,1,1]".

*@par Outputs:
*y: A Tensor of the same type as "x".
*/
REG_OP(ReadSelect)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(stride_list, ListInt, {1,1,1,1,1})
    .OP_END_FACTORY_REG(ReadSelect)

/**
*@brief: Write data with offset.

*@par Inputs:\n
*x: A Tensor. Must be one of the following types: int32, float32, float16, int8.

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*/
REG_OP(WriteSelect)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(WriteSelect)

/**
*@brief Read data by stride.

*@par Inputs:
*One input:\n
*x: A Tensor. Must be one of the following types: float16, int8.

*@par Attributes:
*@li axis: A required int32, specifying the index of axis to read by stride.

*@par Attributes:
*@li stride: A required int32, specifying the value of reading stride.

*@par Outputs:
*y: A Tensor of the same type as "x".
*/
REG_OP(StridedRead)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axis, Int, 1)
    .ATTR(stride, Int, 1)
    .OP_END_FACTORY_REG(StridedRead)

/**
*@brief: Write data by stride.

*@par Inputs:\n
*x: A Tensor. Must be one of the following types: float16, int8.

*@par Attributes:
*@li axis: A required int32, specifying the index of axis to write by stride.

*@par Attributes:
*@li stride: A required int32, specifying the value of writing stride.

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*/
REG_OP(StridedWrite)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axis, Int, 1)
    .ATTR(stride, Int, 1)
    .OP_END_FACTORY_REG(StridedWrite)

/**
*@brief Computes the cumulative log sum exp of the tensor "x" along "axis".

*@par Inputs:
* Two inputs, including:
*@li x: A Tensor. Must be one of the following types: float32, float16.
*@li axis A Tensor of type int32 or int16. Defaults to "0".
*
*@par Attributes:
*@li exclusive: If "False", performs inclusive CumulativeLogsumexp, which means that the first element of the input is identical to the first element of the output. If "True", performs exclusive CumulativeLogsumexp.
*@li reverse: A bool. Defaults to "False".
*
*@par Outputs:
*@li y: A Tensor. Has the same type as "x".
*/
REG_OP(CumulativeLogsumexp)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
    .INPUT(axis, TensorType({DT_INT32, DT_INT16}))
    .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
    .ATTR(exclusive, Bool, false)
    .ATTR(reverse, Bool, false)
    .OP_END_FACTORY_REG(CumulativeLogsumexp)

/**
*@brief Computes the cumulative log sum exp of the tensor "x" along "axis".
*
*@par Inputs:
* One input:
*x: A Tensor. Must be one of the following types: float32, float16.
*
*@par Attributes:
*@li axis A Tensor of type int32 or int16. Defaults to "0".
*@li exclusive: If "False", performs inclusive cumulativeLogsumexp, which means that the first element of the input is identical to the first element of the output. If "True", performs exclusive CumulativeLogsumexp.
*@li reverse: A bool. Defaults to "False".
*
*@par Outputs:
*y: A Tensor. Has the same type as "x".
*/
REG_OP(CumulativeLogsumexpD)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(axis, Int)
    .ATTR(exclusive, Bool, false)
    .ATTR(reverse, Bool, false)
    .OP_END_FACTORY_REG(CumulativeLogsumexpD)

} // namespace ge
#endif // GE_OP_SELECTION_OPS_H
