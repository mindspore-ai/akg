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
 * \file sparse_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_SPARSE_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SPARSE_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Applies softmax to a batched ND SparseTensor . \n

*@par Inputs:
*The input must be a batched ND SparseTensor.
* @li indices: A matrix Tensor of type int64. 2D. The indices of the SparseTensor.
* @li values: A vector Tensor of type float or double. 1D. The values of the SparseTensor.
* @li shape: A vector Tensor of type int64. 1D. The shape of the SparseTensor . \n

*@par Outputs:
*y: A vector Tensor. 1D. Has the same type as "values" . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator SparseSoftmax.
*/
REG_OP(SparseSoftmax)
    .INPUT(indices, TensorType({DT_INT64}))
    .INPUT(values, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(shape, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(SparseSoftmax)

/**
*@brief Adds up a SparseTensor and a dense Tensor, producing a dense Tensor . \n

*@par Inputs:
*Inputs "x1_*" must be SparseTensors and "x2" must be a dense Tensor.
* @li x1_indices: A matrix Tensor of type int32 or int64. 2D. The indices of the SparseTensor.
* @li x1_values: The values of the SparseTensor. A vector Tensor. 1D.
* @li x1_shape: A vector Tensor of type int32 or int64. 1D. The shape of the SparseTensor.
* @li x2: A matrix Tensor. Has the same type and same shape as the SparseTensors . \n

*@par Outputs:
*y: A matrix Tensor. Has the same type and same shape as "x2" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseTensorDenseAdd.
*/

REG_OP(SparseTensorDenseAdd)
    .INPUT(x1_indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(x1_values, TensorType({DT_INT64, DT_INT32, DT_UINT16, DT_INT16, DT_UINT8, DT_INT8, \
        DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x1_shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(x2, TensorType({DT_INT64, DT_INT32, DT_UINT16, DT_INT16, DT_UINT8, DT_INT8, \
        DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_INT64, DT_INT32, DT_UINT16, DT_INT16, DT_UINT8, DT_INT8, \
        DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(SparseTensorDenseAdd)

/**
*@brief Reorders a SparseTensor into the canonical, row-major ordering . \n

*@par Inputs:
* @li indices: A matrix Tensor of type int32 or int64. 2D. The indices of the SparseTensor.
* @li values: Values of the SparseTensor. A vector Tensor. 1D.
* @li shape: A vector Tensor of type int32 or int64. 1D. The shape of the SparseTensor . \n

*@par Outputs:
*@li y_indices: The indices of the SparseTensor. Has the same type as "indices".
*@li y_values: The values of the SparseTensorr. Has the same type as "values" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseReorder.
*/
REG_OP(SparseReorder)
    .INPUT(indices, TensorType({DT_INT64}))
    .INPUT(values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, \
        DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING}))
    .INPUT(shape, TensorType({DT_INT64}))
    .OUTPUT(y_indices, TensorType({DT_INT64}))
    .OUTPUT(y_values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, \
        DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING}))
    .OP_END_FACTORY_REG(SparseReorder)

/**
* @brief Reshapes a SparseTensor to represent values in a new dense shape . \n

* @par Inputs:
* The input of int32 support only static input
* @li indices: A matrix Tensor of type int64 or type int32. 2D. The indices of the SparseTensor.
* @li shape: A vector Tensor of type int64 or type int32. 1D. The shape of the SparseTensor.
* @li new_shape: A 1D Tensor of type int64 or type int32. The requested new dense shape . \n

* @par Outputs:
* @li y_indices: A Tensor of type int64. The indices of the new dense shape.
* @li y_shape: A Tensor of type int64. The shape of the new dense shape . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseReshape.
*/
REG_OP(SparseReshape)
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(new_shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y_indices, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y_shape, TensorType({DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(SparseReshape)

/**
*@brief Adds up a SparseTensor and a dense Tensor.
*@par Inputs:
*(1) Broadcasts the dense side to have the same shape as the sparse side, if eligible;
*(2) Then, only the dense values pointed to by the indices of the SparseTensor participate in the cwise addition.
* @li x1_indices: A matrix Tensor of type int64. 2D. The indices of the SparseTensor.
* @li x1_values: The values of the SparseTensor. A vector Tensor. 1D.
* @li x1_shape: A 1D Tensor of type int64. The requested new dense shape.
* @li x2: A dense Tensor of the same type as "x1_values" . \n

*@par Outputs:
*y: A Tensor. Has the same type as "x1_values" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseDenseCwiseAdd.
*/
REG_OP(SparseDenseCwiseAdd)
    .INPUT(x1_indices, TensorType({DT_INT64}))
    .INPUT(x1_values, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
                                  DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, \
                                  DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x1_shape, TensorType({DT_INT64}))
    .INPUT(x2, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, \
                          DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, \
                          DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, \
                           DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, \
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(SparseDenseCwiseAdd)

/**
*@brief Divides a SparseTensor by a dense Tensor . \n

*@par Inputs:
* @li x1_indices: A matrix Tensor of type int64. 2D. The indices of the SparseTensor.
* @li x1_values: The values of the SparseTensor. A vector Tensor. 1D.
* @li x1_shape: A 1D Tensor of type int64. The requested new dense shape.
* @li x2: A dense Tensor of the same type as "x1_values" . \n

*@par Outputs:
*y: A Tensor. Has the same type as "x1_values" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseDenseCwiseDiv.
*/
REG_OP(SparseDenseCwiseDiv)
    .INPUT(x1_indices, TensorType({DT_INT64}))
    .INPUT(x1_values, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
                                  DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, \
                                  DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x1_shape, TensorType({DT_INT64}))
    .INPUT(x2, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, \
                          DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, \
                          DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, \
                           DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, \
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(SparseDenseCwiseDiv)

/**
*@brief Multiplies a SparseTensor by a dense Tensor . \n

*@par Inputs:
* @li x1_indices: A matrix Tensor of type int64. 2D. The indices of the SparseTensor.
* @li x1_values: The values of the SparseTensor. A vector Tensor. 1D.
* @li x1_shape: A 1D Tensor of type int64. The requested new dense shape.
* @li x2: A dense Tensor of the same type as "x1_values" . \n

*@par Outputs:
*y: A Tensor. Has the same type as "x1_values" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseDenseCwiseMul.
*/
REG_OP(SparseDenseCwiseMul)
    .INPUT(x1_indices, TensorType({DT_INT64}))
    .INPUT(x1_values, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
                                  DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, \
                                  DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x1_shape, TensorType({DT_INT64}))
    .INPUT(x2, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, \
                          DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, \
                          DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, \
                           DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, \
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(SparseDenseCwiseMul)

/**
*@brief Adds a SparseTensor to a SparseTensorsMap . \n

*@par Inputs:
* The input tensor must be a SparseTensor.
* @li x1_indices: A matrix Tensor of type int64. 2D. The indices of the SparseTensor.
* @li x1_values: The values of the SparseTensor. A vector Tensor. 1D.
* @li x1_shape: A 1D Tensor of type int64. The requested new dense shape . \n

*@par Attributes:
*@li container: An optional string. Defaults to " ".
*@li shared_name: An optional string. Defaults to " " . \n

*@par Outputs:
*handle: A Tensor of type int64 . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator AddSparseToTensorsMap.
*/
REG_OP(AddSparseToTensorsMap)
    .INPUT(indices, TensorType({DT_INT64}))
    .INPUT(values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, \
        DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING}))
    .INPUT(shape, TensorType({DT_INT64}))
    .OUTPUT(handle, TensorType({DT_INT64}))
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(AddSparseToTensorsMap)

/**
*@brief The gradient operator for the SparseSlice op . \n

*@par Inputs:
* @li backprop_val_grad: A Tensor.
* @li indices: A matrix Tensor of type int64. 2D. The indices of the SparseTensor.
* @li start: A 1D Tensor of type int64. The start of the slice.
* @li new_indices: A matrix Tensor of type int64. 2D. The indices of the sliced SparseTensor . \n

*@par Outputs:
*y_grad: A Tensor of type int64 . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseSliceGrad.
*/
REG_OP(SparseSliceGrad)
    .INPUT(backprop_val_grad, TensorType({ DT_INT8, DT_UINT8, DT_INT16,
        DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT, DT_FLOAT16, DT_DOUBLE,
        DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(indices, TensorType({DT_INT64}))
    .INPUT(start, TensorType({DT_INT64}))
    .INPUT(new_indices, TensorType({DT_INT64}))
    .OUTPUT(y_grad, TensorType({ DT_INT8, DT_UINT8, DT_INT16,
        DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT, DT_FLOAT16, DT_DOUBLE,
        DT_COMPLEX64, DT_COMPLEX128 }))
    .OP_END_FACTORY_REG(SparseSliceGrad)

/**
*@brief Slices a SparseTensor based on the "start" and "size" . \n

*@par Inputs:
* @li indices: A 2D Tensor of type int64. The indices of the SparseTensor.
* @li values: A 1D Tensor. The values of the SparseTensor.
* @li shape: A 2D Tensor of type int64. The shape of the SparseTensor.
* @li start:  A 1D Tensor of type int64. The start of the slice.
* @li size: A 1D Tensor of type int64. The size of the slice . \n

*@par Outputs:
*@li y_indices: A Tensor of type int64.
*@li y_values: A Tensor. Has the same type as "values".
*@li y_shape: A Tensor of type int64 . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseSlice.
*/
REG_OP(SparseSlice)
    .INPUT(indices, TensorType({DT_INT64}))
    .INPUT(values, TensorType({DT_INT64, DT_INT32, DT_UINT16, DT_INT16, \
        DT_UINT8, DT_INT8, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, \
        DT_COMPLEX128, DT_BOOL, DT_STRING, DT_RESOURCE}))
    .INPUT(shape, TensorType({DT_INT64}))
    .INPUT(start, TensorType({DT_INT64}))
    .INPUT(size, TensorType({DT_INT64}))
    .OUTPUT(y_indices, TensorType({DT_INT64}))
    .OUTPUT(y_values, TensorType({DT_INT64, DT_INT32, DT_UINT16, DT_INT16, \
        DT_UINT8, DT_INT8, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, \
        DT_COMPLEX128, DT_BOOL, DT_STRING, DT_RESOURCE}))
    .OUTPUT(y_shape, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(SparseSlice)

/**
*@brief The gradient operator for the SparseAdd op . \n

*@par Inputs:
* @li backprop_val_grad: A 1D Tensor with shape [nnz(sum)]. The gradient with respect to the non-empty values of the sum.
* @li x1_indices: A 2D Tensor of type int64. The indices of the SparseTensor A, with size [nnz(A), ndims].
* @li x2_indices: A 2D Tensor of type int64. The indices of the SparseTensor B, with size [nnz(B), ndims].
* @li sum_indices: A 2D Tensor of type int64. The indices of the sum SparseTensor, with size [nnz(sum), ndims] . \n

*@par Outputs:
*@li x1_val_grad: A Tensor. Has the same type as "backprop_val_grad".
*@li x2_val_grad: A Tensor. Has the same type as "backprop_val_grad" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseAddGrad.
*/
REG_OP(SparseAddGrad)
    .INPUT(backprop_val_grad, TensorType({DT_INT8, DT_INT16, DT_INT32,
                  DT_INT64, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x1_indices, TensorType({DT_INT64}))
    .INPUT(x2_indices, TensorType({DT_INT64}))
    .INPUT(sum_indices, TensorType({DT_INT64}))
    .OUTPUT(x1_val_grad, TensorType({DT_INT8, DT_INT16, DT_INT32,
                  DT_INT64, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(x2_val_grad, TensorType({DT_INT8, DT_INT16, DT_INT32,
                  DT_INT64, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(SparseAddGrad)

/**
*@brief The gradient of SparseFillEmptyRows . \n

*@par Inputs:
* @li reverse_index_map: A 1D Tensor of type int64. The reverse index map from SparseFillEmptyRows.
* @li grad_values: A 1D Tensor. The gradients from backprop . \n

*@par Outputs:
*@li y_value: A Tensor. Has the same type as "grad_values".
*@li y_default_value: A Tensor. Has the same type as "grad_values" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseFillEmptyRowsGrad.
*/
REG_OP(SparseFillEmptyRowsGrad)
    .INPUT(reverse_index_map, TensorType({DT_INT64}))
    .INPUT(grad_values, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, \
        DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING}))
    .OUTPUT(y_value, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, \
        DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING}))
    .OUTPUT(y_default_value, TensorType({DT_INT8, DT_UINT8, DT_INT16, \
        DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, \
        DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING}))
    .OP_END_FACTORY_REG(SparseFillEmptyRowsGrad)

/**
*@brief Multiplies SparseTensor A (of rank 2) by dense matrix B . \n

*@par Inputs:
* @li x1_indices: A 2D Tensor of type int32 or int64.
*The indices of the matrix "SparseTensor", with size [nnz, 2].
* @li x1_values: A 1D Tensor. The values of the SparseTensor, with size [nnz].
* @li x1_shape: A 1D Tensor of type int64. The shape of the SparseTensor, with size [2].
* @li x2: A dense matrix Tensor of the same type as "x1_values". 2D . \n

*@par Outputs:
*y: A "Tensor". Has the same type as "x1_values" . \n

*@par Attributes:
*@li adjoint_a: An optional bool. Defaults to "False".Use the adjoint of A in the matrix multiply.
*If A is complex, this is transpose(conj(A)). Otherwise it is transpose(A).
*@li adjoint_b: An optional bool. Defaults to "False".Use the adjoint of B in the matrix multiply.
*If B is complex, this is transpose(conj(B)). Otherwise it is transpose(B) . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseTensorDenseMatMul.
*/
REG_OP(SparseTensorDenseMatMul)
    .INPUT(x1_indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(x1_values, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, \
        DT_COMPLEXT64, DT_COMPLEX128, DT_FLOAT16, DT_INT64}))
    .INPUT(x1_shape, TensorType({DT_INT64}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT32, DT_COMPLEXT64, \
        DT_COMPLEX128, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT32, DT_COMPLEXT64, \
        DT_COMPLEX128, DT_FLOAT16}))
    .ATTR(adjoint_a, Bool, false)
    .ATTR(adjoint_b, Bool, false)
    .OP_END_FACTORY_REG(SparseTensorDenseMatMul)

/**
*@brief Converts a sparse representation into a dense tensor . \n

*@par Inputs:
* @li indices: A 0D, 1D, or 2D Tensor of type int32 or int64.
* @li output_shape: A 1D Tensor of the same type as "sparse_indices". The shape of the dense output tensor.
* @li values: A 1D Tensor. Values corresponding to each row of "sparse_indices",
or a scalar value to be used for all sparse indices.
* @li default_value: A Tensor of the same type as "sparse_values" . \n

*@par Attributes:
*validate_indices: If true, indices are checked to make sure they are sorted in
lexicographic order and that there are no repeats. \n

*@par Outputs:
*y: A Tensor. Has the same type as "values" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseToDense.
*/
REG_OP(SparseToDense)
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(output_shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(values, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_BOOL, DT_DOUBLE}))
    .INPUT(default_value, TensorType({DT_INT8, DT_UINT8, DT_INT16, \
        DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_BOOL, \
        DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_BOOL, DT_DOUBLE}))
    .ATTR(validate_indices, Bool, true)
    .OP_END_FACTORY_REG(SparseToDense)

/**
*@brief Concatenates a list of `SparseTensor` along the specified dimension.
*Concatenation is with respect to the dense versions of these sparse tensors . \n

*@par Inputs:
* @li indices:A list of at least 2 `Tensor` objects with type `int64`.2-D.
*Indices of each input `SparseTensor`.It's a dynamic input.
* @li values:A list with the same length as `indices` of `Tensor` objects with the same type.
It's a dynamic input.
* @li shapes:A list with the same length as `indices` of `Tensor` objects with type `int64`.1-D.
* Shapes of each `SparseTensor`. It's a dynamic input. \n

*@par Attributes:
*@li concat_dim: An `int` Dimension to concatenate along
*@li N:Number of sparse

*@par Outputs:
* @li y_indices:A `Tensor` of type `int64`.
* @li y_values:A `Tensor`. Has the same type as `values`.
* @li y_shape:A `Tensor` of type `int64` . \n

*@par Third-party framework compatibility
* Compatible SparseConcat operator in Tensorflow
*/
REG_OP(SparseConcat)
    .DYNAMIC_INPUT(indices, TensorType({DT_INT64}))
    .DYNAMIC_INPUT(values,
        TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
                    DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, \
                    DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING}))
    .DYNAMIC_INPUT(shapes, TensorType({DT_INT64}))
    .OUTPUT(y_indices, TensorType({DT_INT64}))
    .OUTPUT(y_values,
        TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
                    DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, \
                    DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING}))
    .OUTPUT(y_shape, TensorType({DT_INT64}))
    .ATTR(concat_dim, Int, 0)
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(SparseConcat)

/**
*@brief Adds two `SparseTensor` objects to produce another `SparseTensor` . \n

*@par Inputs:
*7 inputs, contains:
* @li x1_indices:A `Tensor` of type `int64`.2-D.
* The `indices` of the first `SparseTensor`, size `[nnz, ndims]` Matrix.
* @li x1_values:A `Tensor`. Must be one of the following types:float,int8,int16,int32,int64, float64.
* @li x1_shape:A `Tensor` of type `int64`.1-D. The `shape` of the first `SparseTensor`,
* size `[ndims]` Vector.
* @li x2_indices:A `Tensor` of type `int64`.2-D.The `indices` of the second `SparseTensor`,
* size `[nnz, ndims]` Matrix.
* @li x2_values:A `Tensor`. Must have the same type as `a_values`.1-D.
* The `values` of the second `SparseTensor`, size `[nnz]` Vector.
* @li x2_shape:A `Tensor` of type `int64`.1-D.
* The `shape` of the second `SparseTensor`, size `[ndims]` Vector.
* @li thresh:A `Tensor` 0-D.The magnitude threshold that determines if an output value/index pair takes space . \n

*@par Outputs:
* @li sum_indices:A `Tensor` of type `int64`.
* @li sum_values:A `Tensor`. Has the same type as `x1_values`.
* @li sum_shape:A `Tensor` of type `int64` . \n

*@par Third-party framework compatibility
* Compatible SparseAdd operator in Tensorflow
*/
REG_OP(SparseAdd)
    .INPUT(x1_indices, TensorType({DT_INT64}))
    .INPUT(x1_values, TensorType({DT_FLOAT, DT_INT8, DT_INT16, \
        DT_INT32, DT_INT64, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x1_shape, TensorType({DT_INT64}))
    .INPUT(x2_indices, TensorType({DT_INT64}))
    .INPUT(x2_values, TensorType({DT_FLOAT, DT_INT8, DT_INT16, DT_INT32, \
        DT_INT64, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2_shape, TensorType({DT_INT64}))
    .INPUT(thresh, TensorType({DT_FLOAT, DT_INT8, DT_INT16, DT_INT32, \
        DT_INT64, DT_DOUBLE}))
    .OUTPUT(sum_indices, TensorType({DT_INT64}))
    .OUTPUT(sum_values, TensorType({DT_FLOAT, DT_INT8, DT_INT16, \
        DT_INT32, DT_INT64, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(sum_shape, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(SparseAdd)

/**
*@brief Fills empty rows in the input 2-D `SparseTensor` with a default value . \n

*@par Inputs:
*4 inputs,contains:
* @li indices: A `Tensor` of type `int64`.2-D. the indices of the sparse tensor.
* @li values: A `Tensor`. 1-D. the values of the sparse tensor.
* @li dense_shape: A `Tensor` of type `int64`.1-D. the shape of the sparse tensor.
* @li default_value: `Tensor`. Must have the same type as `values`.
*0-D. default value to insert into location `[row, 0, ..., 0]`
*for rows missing from the input sparse tensor . \n

*@par Outputs:
* @li y_indices:A `Tensor` of type `int64`.
* @li y_values:A `Tensor`. Has the same type as `values`.
* @li empty_row_indicator:A `Tensor` of type `bool`.
* @li reverse_index_map:A `Tensor` of type `int64` . \n

*@par Third-party framework compatibility
* Compatible SparseFillEmptyRows operator in Tensorflow
*/
REG_OP(SparseFillEmptyRows)
    .INPUT(indices, TensorType({DT_INT64}))
    .INPUT(values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, \
        DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING}))
    .INPUT(dense_shape, TensorType({DT_INT64}))
    .INPUT(default_value, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, \
        DT_INT16, DT_UINT16, DT_UINT8, \
        DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, \
        DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING}))
    .OUTPUT(y_indices, TensorType({DT_INT64}))
    .OUTPUT(y_values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, \
        DT_INT16, DT_UINT16, DT_UINT8, \
        DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, \
        DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING}))
    .OUTPUT(empty_row_indicator, TensorType({DT_BOOL}))
    .OUTPUT(reverse_index_map, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(SparseFillEmptyRows)

/**
*@brief Returns the element-wise max of two SparseTensors . \n

*@par Inputs:
*6 inputs,contains:
* @li x1_indices:A `Tensor` of type `int64`.2-D.
*`N x R` matrix with the indices of non-empty values in a SparseTensor,
* in the canonical lexicographic ordering.
* @li x1_values:A `Tensor`. 1-D. the values of the sparse tensor.
* @li x1_shape:A `Tensor` of type `int64`.1-D. the shape of the sparse tensor.
* @li x2_indices:A `Tensor` of type `int64`.2-D. the indices of the sparse tensor.
* @li x2_values:A `Tensor`. 1-D. Must have the same type as `x1_values`.
* @li x2_shape:A `Tensor` of type `int64`.1-D.
*counterpart to `a_shape` for the other operand; the two shapes must be equal . \n

*@par Outputs:
* @li y_indices:A `Tensor` of type `int64`.
* @li y_values:A `Tensor`. Has the same type as `x1_values` . \n

*@par Third-party framework compatibility
* Compatible SparseSparseMaximum operator in Tensorflow
*/
REG_OP(SparseSparseMaximum)
    .INPUT(x1_indices, TensorType({DT_INT64}))
    .INPUT(x1_values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .INPUT(x1_shape, TensorType({DT_INT64}))
    .INPUT(x2_indices, TensorType({DT_INT64}))
    .INPUT(x2_values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .INPUT(x2_shape, TensorType({DT_INT64}))
    .OUTPUT(y_indices, TensorType({DT_INT64}))
    .OUTPUT(y_values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OP_END_FACTORY_REG(SparseSparseMaximum)

/**
*@brief Returns the element-wise min of two SparseTensors . \n

*@par Inputs:
*6 inputs,contains:
* @li x1_indices:A `Tensor` of type `int64`.2-D.
*`N x R` matrix with the indices of non-empty values in a SparseTensor,
* in the canonical lexicographic ordering.
* @li x1_values:A `Tensor`. 1-D. the values of the sparse tensor.
* @li x1_shape:A `Tensor` of type `int64`.1-D. the shape of the sparse tensor.
* @li x2_indices:A `Tensor` of type `int64`.2-D. the indices of the sparse tensor.
* @li x2_values:A `Tensor`. 1-D. Must have the same type as `x1_values`.
* @li x2_shape:A `Tensor` of type `int64`.1-D.
*counterpart to `a_shape` for the other operand; the two shapes must be equal . \n

*@par Outputs:
* @li y_indices:A `Tensor` of type `int64`.
* @li y_values:A `Tensor`. Has the same type as `x1_values` . \n

*@par Third-party framework compatibility
* Compatible SparseSparseMinimum operator in Tensorflow
*/
REG_OP(SparseSparseMinimum)
    .INPUT(x1_indices, TensorType({DT_INT64}))
    .INPUT(x1_values, TensorType({DT_INT64, DT_INT32, \
        DT_UINT16, DT_INT16, DT_UINT8, DT_INT8, DT_FLOAT16, \
        DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x1_shape, TensorType({DT_INT64}))
    .INPUT(x2_indices, TensorType({DT_INT64}))
    .INPUT(x2_values, TensorType({DT_INT64, DT_INT32, \
        DT_UINT16, DT_INT16, DT_UINT8, DT_INT8, DT_FLOAT16, \
        DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2_shape, TensorType({DT_INT64}))
    .OUTPUT(y_indices, TensorType({DT_INT64}))
    .OUTPUT(y_values, TensorType({DT_INT64, DT_INT32, \
        DT_UINT16, DT_INT16, DT_UINT8, DT_INT8, DT_FLOAT16, \
        DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(SparseSparseMinimum)

/**
*@brief Computes the max of elements across dimensions of a SparseTensor . \n

*@par Inputs:
*4 or 5 inputs,contains:
* @li x_indices:A `Tensor` of type `int64`.2-D.
*`N x R` matrix with the indices of non-empty values in a
*SparseTensor, possibly not in canonical ordering.
* @li x_values:A `Tensor`. 1-D. the values of the sparse tensor.
*`N` non-empty values corresponding to `input_indices`.
* @li x_shape:A `Tensor` of type `int64`.1-D.  Shape of the input SparseTensor.
* @li reduction_axes:A `Tensor` of type `int32`.1-D.
*Length-`K` vector containing the reduction axes . \n

*@par Attributes:
* keep_dims:An optional `bool`. Defaults to `False`.
*If true, retain reduced dimensions with length 1 . \n

*@par Outputs:
* y:A `Tensor`. Has the same type as `input_values` . \n

*@par Third-party framework compatibility
* Compatible SparseReduceMax operator in Tensorflow
*/
REG_OP(SparseReduceMax)
    .INPUT(x_indices, TensorType({DT_INT64}))
    .INPUT(x_values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .INPUT(x_shape, TensorType({DT_INT64}))
    .INPUT(reduction_axes, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16,
                           DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(SparseReduceMax)

/**
*@brief Computes the max of elements across dimensions of a SparseTensor . \n

*@par Inputs:
*4 or 5 inputs,contains:
* @li x_indices:A `Tensor` of type `int64`.2-D.
*`N x R` matrix with the indices of non-empty values in a
*SparseTensor, possibly not in canonical ordering.
* @li x_values:A `Tensor`. 1-D. the values of the sparse tensor.
*`N` non-empty values corresponding to `input_indices`.
* @li x_shape:A `Tensor` of type `int64`.1-D.  Shape of the input SparseTensor.
* @li reduction_axes:A `Tensor` of type `int32`.1-D.
*Length-`K` vector containing the reduction axes . \n

*@par Attributes:
* keep_dims:An optional `bool`. Defaults to `False`.
*If true, retain reduced dimensions with length 1 . \n

*@par Outputs:
* @li y_indices:A `Tensor` of type `int64`.
* @li y_values:A `Tensor`. Has the same type as `input_values`.
* @li y_shape:A `Tensor` of type `int64` . \n

*@par Third-party framework compatibility
* Compatible SparseReduceMaxSparse operator in Tensorflow
*/
REG_OP(SparseReduceMaxSparse)
    .INPUT(x_indices, TensorType({DT_INT64}))
    .INPUT(x_values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .INPUT(x_shape, TensorType({DT_INT64}))
    .INPUT(reduction_axes, TensorType({DT_INT32}))
    .OUTPUT(y_indices, TensorType({DT_INT64}))
    .OUTPUT(y_values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OUTPUT(y_shape, TensorType({DT_INT64}))
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(SparseReduceMaxSparse)

/**
*@brief Computes the sum of elements across dimensions of a SparseTensor . \n

*@par Inputs:
* @li x_indices: A 2D Tensor of type int64.
*"N x R" matrix with the indices of non-empty values in a
*SparseTensor, possibly not in canonical ordering.
* @li x_values: A 1D Tensor. The values of the SparseTensor.
*"N" non-empty values corresponding to "input_indices".
* @li x_shape: A 1D Tensor of type int64. Shape of the input SparseTensor.
* @li reduction_axes: A 1D Tensor of type int32.
*A length-"K" vector containing the reduction axes . \n

*@par Attributes:
*keep_dims: An optional bool. Defaults to "False".
*If true, retains reduced dimensions with length 1 . \n

*@par Outputs:
*y: A Tensor. Has the same type as "x_values". \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseReduceSum.
*/
REG_OP(SparseReduceSum)
    .INPUT(x_indices, TensorType({DT_INT64}))
    .INPUT(x_values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
                      DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE, \
                      DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x_shape, TensorType({DT_INT64}))
    .INPUT(reduction_axes, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16,
                           DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE, \
                           DT_COMPLEX64, DT_COMPLEX128}))
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(SparseReduceSum)

/**
*@brief Computes the sum of elements across dimensions of a SparseTensor . \n

*@par Inputs:
*4 or 5 inputs, including:
* @li x_indices: A 2D Tensor of type int64.
*"N x R" matrix with the indices of non-empty values in a
*SparseTensor, possibly not in canonical ordering.
* @li x_values: A 1D Tensor. The values of the SparseTensor.
*"N" non-empty values corresponding to "input_indices".
* @li x_shape: A 1D Tensor of type int64. Shape of the input SparseTensor.
* @li reduction_axes: A 1D Tensor of type int32.
* A length-"K" vector containing the reduction axes . \n

*@par Attributes:
* keep_dims: An optional bool. Defaults to "False".
*If true, retains reduced dimensions with length 1 . \n

*@par Outputs:
* @li y_indices: A Tensor of type int64.
* @li y_values: A Tensor. Has the same type as "input_values".
* @li y_shape: A Tensor of type int64 . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseReduceSumSparse.
*/
REG_OP(SparseReduceSumSparse)
    .INPUT(x_indices, TensorType({DT_INT64}))
    .INPUT(x_values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE, \
        DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x_shape, TensorType({DT_INT64}))
    .INPUT(reduction_axes, TensorType({DT_INT32}))
    .OUTPUT(y_indices, TensorType({DT_INT64}))
    .OUTPUT(y_values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE, \
        DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y_shape, TensorType({DT_INT64}))
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(SparseReduceSumSparse)

/**
*@brief Splits a SparseTensor into "num_split" tensors along one dimension . \n

*@par Inputs:
*4 or 5 inputs, including:
* @li split_dim: A 0D Tensor of type int64.
*The dimension along which to split. Must be in the range "[0, rank(shape))".
* @li indices: A 2D Tensor of type int64.
* The indices of the SparseTensor.
* @li values: A 1D Tensor. The values of the SparseTensor.
* @li shape: A 1D Tensor of type int64. Shape of the SparseTensor . \n

*@par Attributes:
* num_split: An int that is >= 1. The number of ways to split . \n

*@par Outputs:
* @li y_indices: A list of "num_split" Tensor objects of type int64.
* @li y_values: A list of "num_split" Tensor objects with the same type as "values".
* @li y_shape: A list of "num_split" Tensor objects of type int64 . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseSplit.
*/
REG_OP(SparseSplit)
    .INPUT(split_dim, TensorType({DT_INT64}))
    .INPUT(indices, TensorType({DT_INT64}))
    .INPUT(values, TensorType({DT_INT64, DT_INT32, DT_UINT16, DT_INT16, \
        DT_UINT8, DT_INT8, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, \
        DT_COMPLEX128, DT_BOOL, DT_STRING, DT_RESOURCE}))
    .INPUT(shape, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(y_indices, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(y_values, TensorType({DT_INT64, DT_INT32, DT_UINT16, \
        DT_INT16, DT_UINT8, DT_INT8, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, \
        DT_COMPLEX64, DT_COMPLEX128, DT_BOOL, DT_STRING, DT_RESOURCE}))
    .DYNAMIC_OUTPUT(y_shape, TensorType({DT_INT64}))
    .ATTR(num_split, Int, 1)
    .OP_END_FACTORY_REG(SparseSplit)

/**
*@brief Generates sparse cross from a list of sparse and dense tensors . \n

*@par Inputs:
* @li indices: A list of 2D Tensor objects of type int64.
* Indices of each input SparseTensor.It's a dynamic input.
* @li values: A list of 1D Tensor objects of type int64 or string.
* Values of each SparseTensor.It's a dynamic input.
* @li shapes: A list with the same length as "indices" of 1D Tensor objects of type int64.
* Shapes of each SparseTensor.It's a dynamic input.
* @li dense_inputs: A list of 2D Tensor objects of type int64 or string.
* Columns represented by dense Tensor .It's a dynamic input. \n

*@par Attributes:
* @li N: number of sparse.
* @li hashed_output: A bool. If true, returns the hash of the cross instead of the string.
* @li num_buckets: An int that is >= 0. It is used if "hashed_output" is true.
*output = hashed_value%num_buckets if num_buckets > 0 else "hashed_value".
* @li hash_key: An int. Specify the hash_key that will be used by the "FingerprintCat64"
*function to combine the crosses fingerprints.
* @li out_type: An int64 or string.
* @li internal_type: An int64 or string . \n

*@par Outputs:
* @li output_indices: A Tensor of type int64.
* @li output_values: A Tensor of type "out_type".
* @li output_shape: A Tensor of type int64 . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseCross.
*/
REG_OP(SparseCross)
    .DYNAMIC_INPUT(indices, TensorType({DT_INT64}))
    .DYNAMIC_INPUT(values, TensorType({DT_INT64, DT_STRING}))
    .DYNAMIC_INPUT(shapes, TensorType({DT_INT64}))
    .DYNAMIC_INPUT(dense_inputs, TensorType({DT_INT64, DT_STRING}))
    .OUTPUT(output_indices, TensorType({DT_INT64}))
    .OUTPUT(output_values, TensorType({DT_INT64, DT_STRING}))
    .OUTPUT(output_shape, TensorType({DT_INT64}))
    .ATTR(N, Int, 0)
    .REQUIRED_ATTR(hashed_output, Bool)
    .ATTR(num_buckets, Int, 0)
    .REQUIRED_ATTR(hash_key, Int)
    .REQUIRED_ATTR(out_type, Type)
    .REQUIRED_ATTR(internal_type, Type)
    .OP_END_FACTORY_REG(SparseCross)

/**
*@brief Generates sparse cross from a list of sparse and dense tensors . \n

*@par Inputs:
*3 or 5 inputs, including:
* @li indices: A 2D Tensor of type int64.
* The "indices" of the minibatch SparseTensor.
* @li values: A 1D Tensor. The "values" of the minibatch SparseTensor.
* @li shape: A 1D Tensor of type int64. The "shape" of the minibatch SparseTensor . \n

*@par Attributes:
* @li container: An optional string. Defaults to "".
*The container name for the "SparseTensorsMap" created by this op.
* @li shared_name: An optional string. Defaults to "".
*The shared name for the "SparseTensorsMap" created by this op . \n

*@par Outputs:
* handles: A Tensor of type int64 . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator AddManySparseToTensorsMap.
*/
REG_OP(AddManySparseToTensorsMap)
    .INPUT(indices, TensorType({DT_INT64}))
    .INPUT(values, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, \
        DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING}))
    .INPUT(shape, TensorType({DT_INT64}))
    .OUTPUT(handles, TensorType({DT_INT64}))
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(AddManySparseToTensorsMap)

/**
*@brief Reads SparseTensors from a "SparseTensorsMap" and concatenate them . \n

*@par Inputs:
* handles: A 1D Tensor of type int64.
*The "N" serialized SparseTensor objects . \n

*@par Attributes:
* @li dtype: A tf.DType. The "dtype" of the SparseTensor objects stored in the "SparseTensorsMap".
* @li container: An optional string. Defaults to "".
*The container name for the "SparseTensorsMap" read by this op.
* @li shared_name: An optional string. Defaults to "".
*The shared name for the "SparseTensorsMap" read by this op . \n

*@par Outputs:
* @li indices: A Tensor of type int64.2-D. The `indices` of the minibatch `SparseTensor`.
* @li values: A Tensor of type "dtype". 1-D. The `values` of the minibatch `SparseTensor`.
* @li shape: A Tensor of type int64 . 1-D. The `shape` of the minibatch `SparseTensor`. \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator TakeManySparseFromTensorsMap.
*/
REG_OP(TakeManySparseFromTensorsMap)
    .INPUT(handles, TensorType({DT_INT64}))
    .OUTPUT(indices, TensorType({DT_INT64}))
    .OUTPUT(values, TensorType({DT_BOOL, DT_INT8, DT_UINT8, DT_INT16, \
        DT_UINT16, DT_INT32, DT_INT64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(shape, TensorType({DT_INT64}))
    .REQUIRED_ATTR(dtype, Type)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(TakeManySparseFromTensorsMap)

/**
*@brief Serializes a SparseTensor into a [3] Tensor object . \n

*@par Inputs:
*3 or 4 inputs, including:
* @li indices: A 2D Tensor of type int64. The indices of the SparseTensor.
* @li values: A 1D Tensor. The values of the SparseTensor.
* @li shape: A 1D Tensor of type int64. The shape of the SparseTensor . \n

*@par Attributes:
* out_type: An optional type. Defaults to "string" . \n

*@par Outputs:
* serialized_sparse: A Tensor of type "out_type" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SerializeSparse.
*/
REG_OP(SerializeSparse)
    .INPUT(indices, TensorType({DT_INT64}))
    .INPUT(values, TensorType({DT_BOOL, DT_INT8, DT_UINT8, DT_INT16, \
        DT_UINT16, DT_INT32, DT_INT64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, \
        DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING}))
    .INPUT(shape, TensorType({DT_INT64}))
    .OUTPUT(serialized_sparse, TensorType({DT_STRING, DT_VARIANT}))
    .ATTR(out_type, Type, DT_STRING)
    .OP_END_FACTORY_REG(SerializeSparse)

/**
*@brief Serializes an "N"-minibatch SparseTensor into an [N, 3] Tensor object . \n

*@par Inputs:
*3 or 4 inputs, including:
* @li indices: A 2D Tensor of type int64. The "indices" of the minibatch SparseTensor.
* @li values: A 1D Tensor. The "values" of the minibatch SparseTensor.
* @li shape: A 1D Tensor of type int64. The "shape" of the minibatch SparseTensor . \n

*@par Attributes:
* out_type: An optional type. Defaults to "string" . \n

*@par Outputs:
* serialized_sparse: A Tensor of type "out_type" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SerializeManySparse.
*/
REG_OP(SerializeManySparse)
    .INPUT(indices, TensorType({DT_INT64}))
    .INPUT(values, TensorType({DT_BOOL, DT_INT8, DT_UINT8, DT_INT16, \
        DT_UINT16, DT_INT32, DT_INT64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, \
        DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING}))
    .INPUT(shape, TensorType({DT_INT64}))
    .OUTPUT(serialized_sparse, TensorType({DT_STRING, DT_VARIANT}))
    .ATTR(out_type, Type, DT_STRING)
    .OP_END_FACTORY_REG(SerializeManySparse)

/**
*@brief Deserializes SparseTensor objects . \n

*@par Inputs:
*serialized_sparse: A Tensor. The serialized SparseTensor objects.
*The last dimension must have 3 columns . \n

*@par Attributes:
* dtype: An optional type. The type of the serialized SparseTensor objects . \n

*@par Outputs:
* @li indices: A Tensor of type int64.
* @li values: A Tensor of type "dtype".
* @li shape: A Tensor of type int64 . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator DeserializeSparse.
*/
REG_OP(DeserializeSparse)
    .INPUT(serialized_sparse, TensorType({DT_STRING, DT_VARIANT}))
    .OUTPUT(indices, TensorType({DT_INT64}))
    .OUTPUT(values, TensorType({DT_BOOL, DT_INT8, DT_UINT8, DT_INT16, \
        DT_UINT16, DT_INT32, DT_INT64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, \
        DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING}))
    .OUTPUT(shape, TensorType({DT_INT64}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(DeserializeSparse)

/**
*@brief Deserializes and concatenates SparseTensors from a serialized minibatch . \n

*@par Inputs:
*Two inputs, including:
* serialized_sparse: A 2D Tensor of type string.
*The "N" serialized SparseTensor objects. Must have 3 columns . \n

*@par Attributes:
* dtype: An optional type. The type of the serialized SparseTensor objects . \n

*@par Outputs:
* @li indices: A Tensor of type int64.
* @li values: A Tensor of type "dtype".
* @li shape: A Tensor of type int64 . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator DeserializeManySparse.
*/
REG_OP(DeserializeManySparse)
    .INPUT(serialized_sparse, TensorType({DT_STRING}))
    .OUTPUT(indices, TensorType({DT_INT64}))
    .OUTPUT(values, TensorType({DT_BOOL, DT_INT8, DT_UINT8, DT_INT16, \
        DT_UINT16, DT_INT32, DT_INT64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, \
        DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING}))
    .OUTPUT(shape, TensorType({DT_INT64}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(DeserializeManySparse)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_SPARSE_OPS_H_