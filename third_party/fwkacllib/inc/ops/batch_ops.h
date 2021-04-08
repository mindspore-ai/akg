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
 * \file batch_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_BATCH_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_BATCH_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Creates batches of tensors in "x_tensors" .   \n

*@par Inputs:
*Input "x_tensors" is a list or a dictionary of tensors.
*x_tensors: The list or dictionary of tensors to enqueue .
It's a dynamic input  \n

*@par Attributes:
*@li num_batch_threads: The number of threads enqueuing "x_tensors".
The batching will be nondeterministic if "num_batch_threads" > 1.
*@li max_batch_size: The maximum batch size pulled from the queue.
*@li max_enqueued_batches: The maximum number of batches pulled from the queue.
*@li batch_timeout_micros: The batch processing timeout, in microseconds.
*@li allowed_batch_sizes: The allowed batch size pulled from the queue.
*@li grad_timeout_micros: The gradient batch processing timeout,
in microseconds.
*@li container: If non-empty, this queue is placed in the given container.
Otherwise, a default container is used.
*@li shared_name: If set, this queue will be shared under the given name
across multiple sessions.
*@li batching_queue: The queue resource container .   \n

*@par Outputs:
*@li y_index: A Tensor. The index of a BatchTensor. Must be in row-major order.
*@li y_id: A Tensor. The ID of a BatchTensor. Must be in row-major order.
*@li y_tensors: A list or dictionary of tensors with
the same types as "x_tensors" .  It's a dynamic output.  \n

*@attention Constraints:
*Batch runs on the Ascend AI CPU, which delivers poor performance.   \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Batch.
*/

REG_OP(Batch)
  .DYNAMIC_INPUT(x_tensors, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, \
      DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE}))
  .OUTPUT(y_index, TensorType({ DT_INT64 }))
  .OUTPUT(y_id, TensorType({ DT_INT64 }))
  .DYNAMIC_OUTPUT(y_tensors, TensorType({DT_INT8, DT_UINT8, DT_INT16, \
      DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_BOOL}))
  .REQUIRED_ATTR(num_batch_threads, Int)
  .REQUIRED_ATTR(max_batch_size, Int)
  .ATTR(max_enqueued_batches, Int, 10)
  .REQUIRED_ATTR(batch_timeout_micros, Int)
  .ATTR(allowed_batch_sizes, ListInt, {})
  .REQUIRED_ATTR(grad_timeout_micros, Int)
  .ATTR(container, String, "")
  .ATTR(shared_name, String, "")
  .ATTR(batching_queue, String, "")
  .OP_END_FACTORY_REG(Batch)

/**
*@brief Reverses the operation of Batch for a single output Tensor .   \n

*@par Inputs:
*Input "x_tensors" is a list or a dictionary of tensors.
* @li x_tensors: The list or dictionary of tensors to enqueue.
* @li index: The matching "batch_index" obtained from Batch.
* @li id: The "id" scalar emitted by Batch .   \n

*@par Attributes:
*@li timeout_micros: The unbatch processing timeout, in microseconds.
*@li container: If non-empty, this queue is placed in the given container.
Otherwise, a default container is used.
*@li shared_name: If set, this queue will be shared under the given name
across multiple sessions .   \n

*@par Outputs:
*y_tensor: A list or dictionary of tensors with the same types as "x_tensors" .   \n

*@attention Constraints:
*Unbatch runs on the Ascend AI CPU, which delivers poor performance.   \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Unbatch.
*/

REG_OP(Unbatch)
  .INPUT(x_tensor, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
      DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT, DT_DOUBLE}))
  .INPUT(index, TensorType({DT_INT64}))
  .INPUT(id, TensorType({DT_INT64}))
  .OUTPUT(y_tensor, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
      DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT, DT_DOUBLE}))
  .REQUIRED_ATTR(timeout_micros, Int)
  .ATTR(container, String, "")
  .ATTR(shared_name, String, "")
  .OP_END_FACTORY_REG(Unbatch)

/**
*@brief Acts like Batch but using the given "batch_index" index of batching
things as they become available .   \n

*@par Inputs:
*Input "x_input" is a list or a dictionary of tensors.
* @li x_input: The input to the Unbatch operation.
* @li index: The batch_index given to the Unbatch operation.
* @li id: The "id" scalar emitted by Batch.
* @li grad: The downstream gradient .   \n

*@par Attributes:
*@li container: If non-empty, this queue is placed in the given container.
Otherwise, a default container is used.
*@li shared_name: If set, this queue will be shared under the given name
across multiple sessions .   \n

*@par Outputs:
*y_grad: The return value, either an empty tensor or the batched gradient .   \n

*@attention Constraints:
*UnbatchGrad runs on the Ascend AI CPU, which delivers poor performance.   \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator UnbatchGrad.
*/

REG_OP(UnbatchGrad)
  .INPUT(x_input, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
      DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT, DT_DOUBLE}))
  .INPUT(index, TensorType({DT_INT64}))
  .INPUT(grad, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
      DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT, DT_DOUBLE}))
  .INPUT(id, TensorType({DT_INT64}))
  .OUTPUT(y_grad, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
      DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT, DT_DOUBLE}))
  .ATTR(container, String, "")
  .ATTR(shared_name, String, "")
  .OP_END_FACTORY_REG(UnbatchGrad)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_BATCH_OPS_H_
