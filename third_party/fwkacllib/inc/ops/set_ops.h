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

#ifndef GE_OP_SET_OPS_H_
#define GE_OP_SET_OPS_H_

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Applies set operation along last dimension of 2 Tensor inputs.

*@par Inputs:
*Inputs include: \n
* @li x1: A Tensor. Must be one of the following types: int8, int16, int32, int64, uint8, uint16, string.
* @li x2: A Tensor. Must have the same type as x1.

*@par Attributes:
*@li set_operation: A string.
*@li validate_indices: An optional bool. Defaults to True.

*@par Outputs:
*@li y_indices: A Tensor of type int64.
*@li y_values: A Tensor. Has the same type as x1.
*@li y_shape: A Tensor of type int64.

*@attention Constraints:\n
*-The implementation for DenseToDenseSetOperation on Ascend uses AICPU, with bad performance.\n

*/
REG_OP(DenseToDenseSetOperation)
  .INPUT(x1, TensorType({DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, \
                         DT_INT32, DT_INT64, DT_STRING}))
  .INPUT(x2, TensorType({DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, \
                         DT_INT32, DT_INT64, DT_STRING}))
  .OUTPUT(y_indices, TensorType({DT_INT64}))
  .OUTPUT(y_values, TensorType({DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, \
                                DT_INT32, DT_INT64, DT_STRING}))
  .OUTPUT(y_shape, TensorType({DT_INT64}))
  .ATTR(set_operation, String, "")
  .ATTR(validate_indices, Bool, true)
  .OP_END_FACTORY_REG(DenseToDenseSetOperation)

/**
*@brief Applies set operation along last dimension of Tensor and SparseTensor.

*@par Inputs:
*Inputs include: \n
* @li x1: A Tensor. Must be one of the following types: int8, int16, int32, int64, uint8, uint16, string.
* @li x2_indices: A Tensor of type int64. 2D Tensor, indices of a SparseTensor.
* @li x2_values: A Tensor. Must have the same type as set1. 1D Tensor, values of a SparseTensor.
* @li x2_shape: A Tensor of type int64. 1D Tensor, shape of a SparseTensor.

*@par Attributes:
*@li set_operation: A string.
*@li validate_indices: An optional bool. Defaults to True.

*@par Outputs:
*@li y_indices: A Tensor of type int64.
*@li y_values: A Tensor. Has the same type as x1.
*@li y_shape: A Tensor of type int64.

*@attention Constraints:\n
*-The implementation for DenseToSparseSetOperation on Ascend uses AICPU, with bad performance.\n

*/
REG_OP(DenseToSparseSetOperation)
    .INPUT(x1, TensorType({DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, \
                           DT_INT32, DT_INT64, DT_STRING}))
    .INPUT(x2_indices, TensorType({DT_INT64}))
    .INPUT(x2_values, TensorType({DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, \
                                  DT_INT32, DT_INT64, DT_STRING}))
    .INPUT(x2_shape, TensorType({DT_INT64}))
    .OUTPUT(y_indices, TensorType({DT_INT64}))
    .OUTPUT(y_values, TensorType({DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, \
                                  DT_INT32, DT_INT64, DT_STRING}))
    .OUTPUT(y_shape, TensorType({DT_INT64}))
    .ATTR(set_operation, String, "")
    .ATTR(validate_indices, Bool, true)
    .OP_END_FACTORY_REG(DenseToSparseSetOperation)

/**
*@brief Applies set operation along last dimension of 2 SparseTensor inputs.

*@par Inputs:
*Inputs include: \n
* @li x1_indices: A Tensor of type int64. 2D Tensor, indices of a SparseTensor.
* @li x1_values: A Tensor. Must be one of the following types: int8, int16, \n
      int32, int64, uint8, uint16, string. 1D Tensor, values of a SparseTensor.
* @li x1_shape: A Tensor of type int64. 1D Tensor, shape of a SparseTensor.
* @li x2_indices: A Tensor of type int64. 2D Tensor, indices of a SparseTensor.
* @li x2_values: A Tensor. Must have the same type as set1_values. 1D Tensor, values of a SparseTensor.
* @li x2_shape: A Tensor of type int64. 1D Tensor, shape of a SparseTensor.

*@par Attributes:
*@li set_operation: A string.
*@li validate_indices: An optional bool. Defaults to True.

*@par Outputs:
*@li y_indices: A Tensor of type int64.
*@li y_values: A Tensor. Has the same type as x1_values.
*@li y_shape: A Tensor of type int64.

*@attention Constraints:\n
*-The implementation for SparseToSparseSetOperation on Ascend uses AICPU, with bad performance.\n

*/
REG_OP(SparseToSparseSetOperation)
    .INPUT(x1_indices, TensorType({DT_INT64}))
    .INPUT(x1_values, TensorType({DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, \
                                  DT_INT32, DT_INT64, DT_STRING}))
    .INPUT(x1_shape, TensorType({DT_INT64}))
    .INPUT(x2_indices, TensorType({DT_INT64}))
    .INPUT(x2_values, TensorType({DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, \
                                  DT_INT32, DT_INT64, DT_STRING}))
    .INPUT(x2_shape, TensorType({DT_INT64}))
    .OUTPUT(y_indices, TensorType({DT_INT64}))
    .OUTPUT(y_values, TensorType({DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, \
                                  DT_INT32, DT_INT64, DT_STRING}))
    .OUTPUT(y_shape, TensorType({DT_INT64}))
    .ATTR(set_operation, String, "")
    .ATTR(validate_indices, Bool, true)
    .OP_END_FACTORY_REG(SparseToSparseSetOperation)

/**
*@brief Number of unique elements along last dimension of input set.

*@par Inputs:
*Inputs include: \n
* @li set_indices: A Tensor of type int64. 2D Tensor, indices of a SparseTensor.
* @li set_values: A Tensor. Must be one of the following types: int8, int16, int32, int64, uint8, uint16.
* @li set_shape: A Tensor of type int64. 1D Tensor, shape of a SparseTensor.

*@par Attributes:
*validate_indices: An optional bool. Defaults to True.

*@par Outputs:
*size: A Tensor of type int32.

*@attention Constraints:\n
*-The implementation for SetSize on Ascend uses AICPU, with bad performance.\n

*/
REG_OP(SetSize)
    .INPUT(set_indices, TensorType({DT_INT64}))
    .INPUT(set_values, TensorType({DT_INT8, DT_INT16, \
        DT_UINT8, DT_UINT16, DT_INT32, DT_INT64, DT_STRING}))
    .INPUT(set_shape, TensorType({DT_INT64}))
    .OUTPUT(size, TensorType({DT_INT32}))
    .ATTR(validate_indices, Bool, true)
    .OP_END_FACTORY_REG(SetSize)
}  // namespace ge

#endif  // GE_OP_SET_OPS_H_
