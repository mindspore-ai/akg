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
 * \file logging_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_LOGGING_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_LOGGING_OPS_H_

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Provides the time since epoch in seconds . \n

*@par Outputs:
*y: A Tensor of type float64. The timestamp as a double for seconds since
the Unix epoch . \n

*@attention Constraints:
*The timestamp is computed when the op is executed, not when it is added to
the graph . \n

*@par Third-party framework compatibility
*Compatible with tensorflow Timestamp operator . \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(Timestamp)
  .OUTPUT(y, TensorType({DT_DOUBLE}))
  .OP_END_FACTORY_REG(Timestamp)

/**
*@brief Asserts that the given condition is true . \n

*@par Inputs:
*If input_condition evaluates to false, print the list of tensors in data.
*Inputs include:
*@li input_condition: The condition to evaluate.
*@li input_data: The tensors to print out when condition is false .
 It's a dynamic input.  \n

*@par Attributes:
*summarize: Print this many entries of each tensor . \n

*@par Third-party framework compatibility
*Compatible with tensorflow Assert operator . \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(Assert)
  .INPUT(input_condition, TensorType{DT_BOOL})
  .DYNAMIC_INPUT(input_data, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8,
      DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
      DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING}))
  .ATTR(summarize, Int, 3)
  .OP_END_FACTORY_REG(Assert)

/**
*@brief Prints a tensor . \n

*@par Inputs:
*x: The tensor to print, it is a dynamic_input . \n

*Compatible with aicpu Print operator . \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(Print)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
    DT_INT64, DT_UINT32, DT_UINT64, DT_DOUBLE, DT_STRING}))
.OP_END_FACTORY_REG(Print)

/**
*@brief Prints a string scalar . \n

*@par Inputs:
*The dtype of input x must be string. Inputs include:
*x: The string scalar to print . \n

*@par Attributes:
*output_stream: A string specifying the output stream or logging level
to print to . \n

*@par Third-party framework compatibility
*Compatible with tensorflow PrintV2 operator . \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(PrintV2)
  .INPUT(x, TensorType({DT_STRING}))
  .ATTR(output_stream, String, "stderr")
  .OP_END_FACTORY_REG(PrintV2)

/**
*@brief Prints a list of tensors. \n

*@par Inputs:
*x: A tensor passwd through this op . \n
*data: A list of tensors to print out when op is evaluated. \n

*@par Attributes:
*message: A string, prefix of the error message. \n
*first_n: Only log first_n number of times. Negative numbers 
*log always; this is the default. \n
*summarize: Only print this many entries of each tensor. 
*If None, then a maximum of 3 elements are printed per input tensor. \n

*@par Third-party framework compatibility
*Compatible with tensorflow Print operator . \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(PrintV3)
  .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
    DT_INT64, DT_UINT32, DT_UINT64, DT_DOUBLE, DT_STRING}))
  .DYNAMIC_INPUT(data, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
    DT_INT64, DT_UINT32, DT_UINT64, DT_DOUBLE, DT_STRING}))
  .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
    DT_INT64, DT_UINT32, DT_UINT64, DT_DOUBLE, DT_STRING}))
  .ATTR(message, String, "")
  .ATTR(first_n, Int, -1)
  .ATTR(summarize, Int, 3)
  .OP_END_FACTORY_REG(PrintV3)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_LOGGING_OPS_H_
