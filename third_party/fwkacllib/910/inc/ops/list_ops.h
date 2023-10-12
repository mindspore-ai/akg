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
 * \file list_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_LIST_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_LIST_OPS_H_

#include <algorithm>
#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief Creates and returns an empty tensor list. \n

*@par Inputs:
*@li element_shape: A shape compatible with that of elements in the list.
*@li max_num_elements: The maximum number of elements. \n

*@par Attributes:
*element_dtype: The type of elements in the list. \n

*@par Outputs:
*handle: An empty tensor list . \n

*@par Third-party framework compatibility.
*Compatible with tensorflow EmptyTensorList operator.
*/
REG_OP(EmptyTensorList)
    .INPUT(element_shape, TensorType({DT_INT32,DT_INT64}))
    .INPUT(max_num_elements, TensorType({DT_INT32}))
    .OUTPUT(handle, TensorType({DT_VARIANT}))
    .ATTR(element_dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(EmptyTensorList)

/**
*@brief Returns a list which has the passed-in `Tensor` as last element
and the other elements of the given list in `input_handle`. \n

*@par Inputs:
*@li input_handle: The old list.
*@li tensor: The tensor to put on the list. \n

*@par Attributes:
*element_dtype: The type of elements in the list. \n

*@par Outputs:
*output_handle:A list with the elements of old list followed by tensor. \n

*@par Third-party framework compatibility.
*Compatible with tensorflow TensorListPushBack operator.
*/
REG_OP(TensorListPushBack)
    .INPUT(input_handle, TensorType({DT_VARIANT}))
    .INPUT(tensor, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE,DT_INT8,
        DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_QINT8,DT_QUINT8,
        DT_QINT16,DT_QUINT16,DT_QINT32,DT_BOOL,DT_RESOURCE,
        DT_STRING,DT_COMPLEX64,DT_COMPLEX128}))
    .OUTPUT(output_handle, TensorType({DT_VARIANT}))
    .ATTR(element_dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(TensorListPushBack)

/**
*@brief The last element of the input list as well as a
list with all but that element. \n

*@par Inputs:
*@li input_handle: The input list.
*@li element_shape: A shape compatible with that of elements in the list. \n

*@par Attributes:
*element_dtype: The type of elements in the list. \n

*@par Outputs:
*@li output_handle:A list with the elements of the old list followed by tensor.
*@li tensor:The withdrawn last element of the list. \n

*@par Third-party framework compatibility.
*Compatible with tensorflow TensorListPopBack operator.
*/
REG_OP(TensorListPopBack)
    .INPUT(input_handle, TensorType({DT_VARIANT}))
    .INPUT(element_shape, TensorType({DT_INT32}))
    .OUTPUT(output_handle, TensorType({DT_VARIANT}))
    .OUTPUT(tensor, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE,DT_INT8,
        DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_QINT8,DT_QUINT8,
        DT_QINT16,DT_QUINT16,DT_QINT32,DT_BOOL,DT_RESOURCE,
        DT_STRING,DT_COMPLEX64,DT_COMPLEX128}))
    .ATTR(element_dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(TensorListPopBack)

/**
*@brief The number of tensors in the input tensor list. \n

*@par Inputs:
*input_handle: The input list. \n

*@par Outputs:
*length:The number of tensors in the list. \n

*@par Third-party framework compatibility.
*Compatible with tensorflow TensorListLength operator.
*/
REG_OP(TensorListLength)
    .INPUT(input_handle, TensorType({DT_VARIANT}))
    .OUTPUT(length, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(TensorListLength)

/**
*@brief The shape of elements in the input tensor list. \n

*@par Inputs:
*input_handle: The input list. \n

*@par Attributes:
*shape_type: The type of shape in the list. \n

*@par Outputs:
*element_shape:A shape compatible with that of elements in the list. \n

*@par Third-party framework compatibility.
*Compatible with tensorflow TensorListElementShape operator.
*/
REG_OP(TensorListElementShape)
    .INPUT(input_handle, TensorType({DT_VARIANT}))
    .OUTPUT(element_shape, TensorType({DT_INT32,DT_INT64}))
    .ATTR(shape_type, Type, DT_INT32)
    .OP_END_FACTORY_REG(TensorListElementShape)

/**
*@brief List of the given size with empty elements. \n

*@par Inputs:
*@li element_shape: A shape compatible with that of elements in the list.
*@li num_elements: The number of elements to reserve. \n

*@par Attributes:
*@li element_dtype: The type of elements in the list.
*@li shape_type: The type of shape in the list. \n

*@par Outputs:
*handle: An output tensor list . \n

*@par Third-party framework compatibility.
*Compatible with tensorflow TensorListReserve operator.
*/
REG_OP(TensorListReserve)
    .INPUT(element_shape, TensorType({DT_INT32,DT_INT64}))
    .INPUT(num_elements, TensorType({DT_INT32}))
    .OUTPUT(handle, TensorType({DT_VARIANT}))
    .ATTR(element_dtype, Type, DT_INT32)
    .ATTR(shape_type, Type, DT_INT32)
    .OP_END_FACTORY_REG(TensorListReserve)

/**
*@brief Get input tensor list elements of index position. \n

*@par Inputs:
*@li input_handle: The input list.
*@li index: A tensor of position.
*@li element_shape: A shape compatible with that of elements in the list. \n

*@par Attributes:
*element_dtype: The type of elements in the list. \n

*@par Outputs:
*item: An output tensor value of index position . \n

*@par Third-party framework compatibility.
*Compatible with tensorflow TensorListGetItem operator.
*/
REG_OP(TensorListGetItem)
    .INPUT(input_handle, TensorType({DT_VARIANT}))
    .INPUT(index, TensorType({DT_INT32}))
    .INPUT(element_shape, TensorType({DT_INT32}))
    .OUTPUT(item, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE,DT_INT8,
        DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_QINT8,DT_QUINT8,
        DT_QINT16,DT_QUINT16,DT_QINT32,DT_BOOL,
        DT_STRING,DT_COMPLEX64,DT_COMPLEX128}))
    .ATTR(element_dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(TensorListGetItem)

/**
*@brief Sets the index-th position of the list to contain the given tensor. \n

*@par Inputs:
*@li input_handle: The input list.
*@li index: The position in the list to which the tensor will be assigned.
*@li item: The element to be assigned to that position. \n

*@par Attributes:
*element_dtype: The type of elements in the list. \n

*@par Outputs:
*output_handle: An output tensor list . \n

*@par Third-party framework compatibility.
*Compatible with tensorflow TensorListSetItem operator.
*/
REG_OP(TensorListSetItem)
    .INPUT(input_handle, TensorType({DT_VARIANT}))
    .INPUT(index, TensorType({DT_INT32}))
    .INPUT(item, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE,DT_INT8,
        DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_QINT8,DT_QUINT8,
        DT_QINT16,DT_QUINT16,DT_QINT32,DT_BOOL,DT_RESOURCE,
        DT_STRING,DT_COMPLEX64,DT_COMPLEX128}))
    .OUTPUT(output_handle, TensorType({DT_VARIANT}))
    .ATTR(element_dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(TensorListSetItem)

/**
*@brief Push tensor to list. \n

*@par Inputs:
*@li input_handles: The input tensor lists.
*@li tensor: The tensor push into tensor list. \n

*@par Attributes:
*element_dtype: The type of elements in the list. \n

*@par Outputs:
*output_handles: The output tensor lists. \n

*@par Third-party framework compatibility.
*Compatible with tensorflow TensorListPushBackBatch operator.
*/
REG_OP(TensorListPushBackBatch)
    .INPUT(input_handles, TensorType({DT_VARIANT}))
    .INPUT(tensor, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE,DT_INT8,
        DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_QINT8,DT_QUINT8,
        DT_QINT16,DT_QUINT16,DT_QINT32,DT_BOOL,
        DT_STRING,DT_COMPLEX64,DT_COMPLEX128}))
    .OUTPUT(output_handles, TensorType({DT_VARIANT}))
    .ATTR(element_dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(TensorListPushBackBatch)

/**
*@brief Stacks all tensors in the list. \n

*@par Inputs:
*@li input_handle: The input tensor list.
*@li element_shape: A shape compatible with that of elements in the tensor. \n

*@par Attributes:
*@li element_dtype: The type of elements in the list.
*@li num_elements: The number of elements in the list. \n

*@par Outputs:
*tensor: The tensor of list. \n

*@par Third-party framework compatibility.
*Compatible with tensorflow TensorListStack operator.
*/
REG_OP(TensorListStack)
    .INPUT(input_handle, TensorType({DT_VARIANT}))
    .INPUT(element_shape, TensorType({DT_INT32}))
    .OUTPUT(tensor, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE,DT_INT8,
        DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_QINT8,DT_QUINT8,
        DT_QINT16,DT_QUINT16,DT_QINT32,DT_BOOL,
        DT_STRING,DT_COMPLEX64,DT_COMPLEX128}))
    .ATTR(element_dtype, Type, DT_INT32)
    .ATTR(num_elements, Int, -1)
    .OP_END_FACTORY_REG(TensorListStack)

/**
*@brief Concats all tensors in the list along the 0th dimension.
Requires that all tensors have the same shape except the first dimension. \n

*@par Inputs:
*@li input_handle: The input list.
*@li element_shape: The shape of the uninitialized elements in the list.
If the first dimension is not -1, it is assumed that all list elements have
the same leading dim.
*@li leading_dims: The list of leading dims of uninitialized list elements. Used if
the leading dim of input_handle.element_shape or the element_shape input arg
is not already set. \n

*@par Attributes:
*element_dtype: The type of elements in the list. \n

*@par Outputs:
*@li tensor: The concated result.
*@li lengths: Output tensor containing sizes of the 0th dimension of tensors
in the list, used for computing the gradient. \n

*@par Third-party framework compatibility.
*Compatible with tensorflow TensorListConcatV2 operator.
*/
REG_OP(TensorListConcatV2)
    .INPUT(input_handle, TensorType({DT_VARIANT}))
    .INPUT(element_shape, TensorType({DT_INT32,DT_INT64}))
    .INPUT(leading_dims, TensorType({DT_INT64}))
    .OUTPUT(tensor, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE,DT_INT8,
        DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_QINT8,DT_QUINT8,
        DT_QINT16,DT_QUINT16,DT_QINT32,DT_BOOL,
        DT_STRING,DT_COMPLEX64,DT_COMPLEX128}))
    .OUTPUT(lengths, TensorType({DT_INT64}))
    .ATTR(element_dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(TensorListConcatV2)

/**
*@brief Splits a tensor into a list. \n

*@par Inputs:
*@li tensor: The input tensor.
*@li element_shape: A shape compatible with that of elements in the tensor.
*@li lengths: Vector of sizes of the 0th dimension of tensors in the list. \n

*@par Attributes:
*element_dtype: The type of elements in the list. \n

*@par Outputs:
*output_handle: The list. \n

*@par Third-party framework compatibility.
*Compatible with tensorflow TensorListSplit operator.
*/
REG_OP(TensorListSplit)
    .INPUT(tensor, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE,DT_INT8,
        DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_QINT8,DT_QUINT8,
        DT_QINT16,DT_QUINT16,DT_QINT32,DT_BOOL,
        DT_STRING,DT_COMPLEX64,DT_COMPLEX128}))
    .INPUT(element_shape, TensorType({DT_INT32,DT_INT64}))
    .INPUT(lengths, TensorType({DT_INT64}))
    .OUTPUT(output_handle, TensorType({DT_VARIANT}))
    .ATTR(element_dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(TensorListSplit)

/**
*@brief Creates a TensorList which, when stacked, has the value of `tensor`. \n

*@par Inputs:
*@li tensor: The input tensor.
*@li element_shape: The shape of elements in the list. \n

*@par Attributes:
*element_dtype: The type of elements in the list. \n

*@par Outputs:
*output_handle: An output tensor list . \n

*@par Third-party framework compatibility.
*Compatible with tensorflow TensorListFromTensor operator.
*/
REG_OP(TensorListFromTensor)
    .INPUT(tensor, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE,DT_INT8,
        DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_QINT8,DT_QUINT8,
        DT_QINT16,DT_QUINT16,DT_QINT32,DT_BOOL,
        DT_STRING,DT_COMPLEX64,DT_COMPLEX128}))
    .INPUT(element_shape, TensorType({DT_INT32,DT_INT64}))
    .OUTPUT(output_handle, TensorType({DT_VARIANT}))
    .ATTR(element_dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(TensorListFromTensor)

/**
*@brief Resizes the list. \n

*@par Inputs:
*@li input_handle: The input tensor list.
*@li size: size of the output list. \n

*@par Outputs:
*output_handle: The output tensor list. \n

*@par Third-party framework compatibility.
*Compatible with tensorflow TensorListResize operator.
*/
REG_OP(TensorListResize)
    .INPUT(input_handle, TensorType({DT_VARIANT}))
    .INPUT(size, TensorType({DT_INT32}))
    .OUTPUT(output_handle, TensorType({DT_VARIANT}))
    .OP_END_FACTORY_REG(TensorListResize)

/**
*@brief Creates a Tensor by indexing into the TensorList. \n

*@par Inputs:
*@li input_handle: The input tensor list.
*@li indices: The indices used to index into the list.
*@li element_shape: The shape of elements in the list. \n

*@par Attributes:
*element_dtype: The type of elements in the list. \n

*@par Outputs:
*values: The tensor. \n

*@par Third-party framework compatibility.
*Compatible with tensorflow TensorListGather operator.
*/
REG_OP(TensorListGather)
    .INPUT(input_handle, TensorType({DT_VARIANT}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(element_shape, TensorType({DT_INT32}))
    .OUTPUT(values, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE,DT_INT8,
        DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_QINT8,DT_QUINT8,
        DT_QINT16,DT_QUINT16,DT_QINT32,DT_BOOL,
        DT_STRING,DT_COMPLEX64,DT_COMPLEX128}))
    .ATTR(element_dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(TensorListGather)

/**
*@brief Creates a TensorList by indexing into a Tensor. \n

*@par Inputs:
*@li tensor: The input tensor.
*@li indices: The indices used to index into the list.
*@li element_shape: The shape of the elements in the list (can be less specified than
the shape of the tensor).
*@li num_elements: The size of the output list. Must be large enough to accommodate
the largest index in indices. If -1, the list is just large enough to include
the largest index in indices. \n

*@par Attributes:
*element_dtype: The type of elements in the list. \n

*@par Outputs:
*output_handle: The TensorList. \n

*@par Third-party framework compatibility.
*Compatible with tensorflow TensorListScatterV2 operator.
*/
REG_OP(TensorListScatterV2)
    .INPUT(tensor, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE,DT_INT8,
        DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_QINT8,DT_QUINT8,
        DT_QINT16,DT_QUINT16,DT_QINT32,DT_BOOL,
        DT_STRING,DT_COMPLEX64,DT_COMPLEX128}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(element_shape, TensorType({DT_INT32,DT_INT64}))
    .INPUT(num_elements, TensorType({DT_INT32}))
    .OUTPUT(output_handle, TensorType({DT_VARIANT}))
    .ATTR(element_dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(TensorListScatterV2)

/**
*@brief Scatters tensor at indices in an input list. \n

*@par Inputs:
*@li input_handle: The input tensor list.
*@li tensor: The input tensor.
*@li indices: The indices used to index into the list. \n

*@par Attributes:
*element_dtype: The type of elements in the list. \n

*@par Outputs:
*output_handle: The TensorList. \n

*@par Third-party framework compatibility.
*Compatible with tensorflow TensorListScatterIntoExistingList operator.
*/
REG_OP(TensorListScatterIntoExistingList)
    .INPUT(input_handle, TensorType({DT_VARIANT}))
    .INPUT(tensor, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE,DT_INT8,
        DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_QINT8,DT_QUINT8,
        DT_QINT16,DT_QUINT16,DT_QINT32,DT_BOOL,
        DT_STRING,DT_COMPLEX64,DT_COMPLEX128}))
    .INPUT(indices, TensorType({DT_INT32}))
    .OUTPUT(output_handle, TensorType({DT_VARIANT}))
    .ATTR(element_dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(TensorListScatterIntoExistingList)

/**
*@brief Concat two tensor lists to a new tensor list. \n

*@par Inputs:
*@li input_a: The input tensor list A.
*@li input_b: The input tensor list B. \n

*@par Attributes:
*element_dtype: The type of elements in the list. \n

*@par Outputs:
*output: The output list. \n

*@par Third-party framework compatibility.
*Compatible with tensorflow TensorListConcatLists operator.
*/
REG_OP(TensorListConcatLists)
    .INPUT(input_a, TensorType({DT_VARIANT}))
    .INPUT(input_b, TensorType({DT_VARIANT}))
    .OUTPUT(output, TensorType({DT_VARIANT}))
    .ATTR(element_dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(TensorListConcatLists)
}   // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_LIST_OPS_H_
