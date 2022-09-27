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
 * \file state_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_STATE_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_STATE_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Creates a variable tensor . \n

*@par Inputs:
*x: A tensor, used to assign a value to the variable tensor internally.
The caller does not need to pass the value of the variable tensor . \n

*@par Attributes:
*@li index: An integer. Index of the input tensor.
*@li value: A tensor, used to pass and record the value of the variable tensor.
*@li container: A string. The container of the variable tensor.
*@li shared_name: A string. The shared name of the variable tensor . \n

*@par Outputs:
*y: The created variable tensor . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Variable.
*/
REG_OP(Variable)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(index, Int, 0)
    .ATTR(value, Tensor, Tensor())
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(Variable)

/**
*@brief Returns a temporary variable tensor. After the use of TemporaryVariable,
pass the reference to the variable tensor to the matching DestroyTemporaryVariable op for destruction . \n

*@par Attributes:
*@li shape: A required list of int32 or int64. The shape of the variable tensor.
*@li dtype: Required. The type of elements in the variable tensor.
*@li var_name: An optional string. The name of the variable to be created . \n

*@par Outputs:
*y: The created variable tensor . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator TemporaryVariable.
*/
REG_OP(TemporaryVariable)
    .OUTPUT(y, TensorType::ALL())
    .REQUIRED_ATTR(shape, ListInt)
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(var_name, String, "")
    .OP_END_FACTORY_REG(TemporaryVariable)

/**
*@brief Destroys the temporary variable and returns its final value.
All other uses of the temporary variable must have been executed before this op . \n

*@par Inputs:
*x: A reference to the temporary variable tensor . \n

*@par Attributes:
*var_name: A required string. Name of the temporary variable.
Must be the same as the "var_name" attribute of the reference to the temporary variable tensor . \n

*@par Outputs:
*y: Final value of the reference to the temporary variable tensor . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator DestroyTemporaryVariable.
*/
REG_OP(DestroyTemporaryVariable)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(var_name, String, "")
    .OP_END_FACTORY_REG(DestroyTemporaryVariable)

/**
*@brief Checks whether a tensor has been initialized. Outputs boolean scalar indicating whether the tensor has been initialized . \n

*@par Inputs:
*x: A Tensor of type float16, float32, double, bool, int8, uint8, uint16, int16, int32, uint32, uint64, int64. 

*@par Outputs:
*y: A tensor, indicating whether "x" has been initialized . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator IsVariableInitialized.
*/
REG_OP(IsVariableInitialized)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
                          DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(IsVariableInitialized)

/**
*@brief Checks whether a tensor has been initialized. Outputs boolean scalar indicating whether the tensor has been initialized . \n

*@par Inputs:
*x: A tensor . \n

*@par Outputs:
*y: A tensor, indicating whether "x" has been initialized, and the data type is boolean . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator VarIsInitializedOp.
*/
REG_OP(VarIsInitializedOp)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
                          DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(VarIsInitializedOp)

/**
*@brief Increments 'ref' until it reaches 'limit' . \n

*@par Inputs:
*Inputs include:
*ref: A mutable Tensor. Must be one of the following types: int32, int64 . \n

*@par Attributes:
*limit: An int. If incrementing ref would bring it above limit, instead
 generates an 'OutOfRange' error . \n

*@par Outputs:
*y: A Tensor. Has the same type as ref . \n

*@attention Constraints:
*The implementation for CountUpTo on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with tensorflow CountUpTo operator.
*/
REG_OP(CountUpTo)
    .INPUT(ref, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(limit, Int, 0)
    .OP_END_FACTORY_REG(CountUpTo)

}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_STATE_OPS_H_
