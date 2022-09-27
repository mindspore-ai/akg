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
 * \file data_flow_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_DATA_FLOW_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_DATA_FLOW_OPS_H_

#include <algorithm>
#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief This operation returns true if the queue is closed and false if
the queue is open. \n

*@par Inputs:
*The input handle must have the resource type. Inputs include:
*handle:A Tensor of type resource. The handle to a queue. \n

*@par Outputs:
*is_closed:A Tensor of type bool. \n

*@par Third-party framework compatibility
*Compatible with tensorflow QueueIsClosed operator.
*/

REG_OP(QueueIsClosed)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .OUTPUT(is_closed, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(QueueIsClosed)

/**
*@brief Computes the number of elements in the given queue. \n

*@par Inputs:
*The input handle must have the resource type. Inputs include:
*handle:A Tensor of type mutable resource. The handle to a queue. \n

*@par Outputs:
*size:A Tensor of type int32. \n

*@par Third-party framework compatibility
*Compatible with tensorflow QueueSize operator.
*/

REG_OP(QueueSize)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .OUTPUT(size, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(QueueSize)

/**
*@brief A queue that produces elements in first-in first-out order. \n

*@par Attributes:
*@li component_types: A list of DType objects. The length of component_types
must equal the number of tensors in each queue element.
*@li shapes:(Optional.) A list of fully-defined TensorShape objects with the
same length as dtypes, or None.
*@li capacity:An integer. The upper bound on the number of elements that may
be stored in this queue.
*@li container: An optional string. Defaults to "". If non-empty, this queue
is placed in the given container. Otherwise, a default container is used.
*@li shared_name:(Optional.) If non-empty, this queue will be shared under
the given name across multiple sessions. \n

*@par Outputs:
*handle:A Tensor of type mutable resource. The handle to a queue. \n

*@par Third-party framework compatibility
*Compatible with tensorflow FIFOQueue operator.
*/

REG_OP(FIFOQueue)
    .OUTPUT(handle, TensorType({DT_RESOURCE}))
    .REQUIRED_ATTR(component_types, ListType)
    .ATTR(shapes, ListListInt, {})
    .ATTR(capacity, Int, -1)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(FIFOQueue)

/**
*@brief Enqueues a tuple of one or more tensors in the given queue. \n

*@par Inputs:
*The input handle must have the resource type. Inputs include:
*@li handle:A Tensor of type mutable resource. The handle to a queue.
*@li components: A list of Tensor objects. One or more tensors from which
the enqueued tensors should be taken. It's a dynamic input. \n

*@par Attributes:
*timeout_ms: An optional int. Defaults to -1. If the queue is full, this
operation will block for up to timeout_ms milliseconds. Note: This option
is not supported yet. \n

*@par Third-party framework compatibility
*Compatible with tensorflow QueueEnqueue operator.
*/

REG_OP(QueueEnqueue)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .DYNAMIC_INPUT(components, TensorType({DT_FLOAT, DT_FLOAT16, \
        DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, \
        DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, DT_RESOURCE, \
        DT_STRING, DT_COMPLEX64, DT_COMPLEX128, DT_QINT16, DT_QUINT16, \
        DT_QINT8, DT_QUINT8, DT_QINT32}))
    .ATTR(timeout_ms, Int, -1)
    .OP_END_FACTORY_REG(QueueEnqueue)

/**
*@brief Enqueues zero or more tuples of one or more tensors in the given queue. \n

*@par Inputs:
*The input handle must have the resource type. Inputs include:
*@li handle:A Tensor of type mutable resource. The handle to a queue.
*@li components: A list of Tensor objects. One or more tensors from which
the enqueued tensors should be taken. It's a dynamic input. \n

*@par Attributes:
*timeout_ms: An optional int. Defaults to -1. If the queue is full, this
operation will block for up to timeout_ms milliseconds. Note: This option
is not supported yet. \n

*@par Third-party framework compatibility
*Compatible with tensorflow QueueEnqueueMany operator.
*/

REG_OP(QueueEnqueueMany)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .DYNAMIC_INPUT(components, TensorType({DT_FLOAT, DT_FLOAT16, \
        DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, \
        DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, DT_RESOURCE, \
        DT_STRING, DT_COMPLEX64, DT_COMPLEX128, DT_QINT16, DT_QUINT16, \
        DT_QINT8, DT_QUINT8, DT_QINT32}))
    .ATTR(timeout_ms, Int, -1)
    .OP_END_FACTORY_REG(QueueEnqueueMany)

/**
*@brief Dequeues n tuples of one or more tensors from the given queue. \n

*@par Inputs:
*The input handle must have the resource type. Inputs include:
*handle:A Tensor of type mutable resource. The handle to a queue. \n

*@par Attributes:
*@li timeout_ms: An optional int. Defaults to -1. If the queue is empty, this
operation will block for up to timeout_ms milliseconds. Note: This option is
not supported yet.
*@li component_types: A list of DTypes that has length >= 1. The type of each
component in a tuple. \n

*@par Outputs:
*components:A list of Tensor objects of type component_types. \n

*@par Third-party framework compatibility
*Compatible with tensorflow QueueDequeue operator.
*/

REG_OP(QueueDequeue)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .DYNAMIC_OUTPUT(components, TensorType({DT_FLOAT, DT_FLOAT16, \
        DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, \
        DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, DT_RESOURCE, \
        DT_STRING, DT_COMPLEX64, DT_COMPLEX128, DT_QINT16, DT_QUINT16, \
        DT_QINT8, DT_QUINT8, DT_QINT32}))
    .ATTR(timeout_ms, Int, -1)
    .REQUIRED_ATTR(component_types, ListType)
    .OP_END_FACTORY_REG(QueueDequeue)

/**
*@brief Dequeues n tuples of one or more tensors from the given queue. \n

*@par Inputs:
*The input handle must have the resource type. Inputs include:
*@li handle:A Tensor of type mutable resource. The handle to a queue.
*@li n: A Tensor of type int32. The number of tuples to dequeue. \n

*@par Attributes:
*@li timeout_ms: An optional int. Defaults to -1. If the queue has fewer than
n elements, this operation will block for up to timeout_ms milliseconds.
Note: This option is not supported yet.
*@li component_types: A list of DTypes that has length >= 1. The type of each
component in a tuple. \n

*@par Outputs:
*components:A list of Tensor objects of type component_types. \n

*@par Third-party framework compatibility
*Compatible with tensorflow QueueDequeueMany operator.
*/

REG_OP(QueueDequeueMany)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(n, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(components, TensorType({DT_FLOAT, DT_FLOAT16, \
        DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, \
        DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, DT_RESOURCE, \
        DT_STRING, DT_COMPLEX64, DT_COMPLEX128, DT_QINT16, DT_QUINT16, \
        DT_QINT8, DT_QUINT8, DT_QINT32}))
    .ATTR(timeout_ms, Int, -1)
    .REQUIRED_ATTR(component_types, ListType)
    .OP_END_FACTORY_REG(QueueDequeueMany)

/**
*@brief Dequeues n tuples of one or more tensors from the given queue. \n

*@par Inputs:
*The input handle must have the resource type. Inputs include:
*@li handle:A Tensor of type mutable resource. The handle to a queue.
*@li n: A Tensor of type int32. The number of tuples to dequeue. \n

*@par Attributes:
*@li timeout_ms: An optional int. Defaults to -1. If the queue has fewer than
n elements, this operation will block for up to timeout_ms milliseconds.
Note: This option is not supported yet.
*@li component_types: A list of DTypes that has length >= 1. The type of each
component in a tuple. \n

*@par Outputs:
*components:A list of Tensor objects of type component_types. \n

*@par Third-party framework compatibility
*Compatible with tensorflow QueueDequeueUpTo operator.
*/

REG_OP(QueueDequeueUpTo)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(n, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(components, TensorType({DT_FLOAT, DT_FLOAT16, \
        DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, \
        DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, DT_RESOURCE, \
        DT_STRING, DT_COMPLEX64, DT_COMPLEX128, DT_QINT16, DT_QUINT16, \
        DT_QINT8, DT_QUINT8, DT_QINT32}))
    .ATTR(timeout_ms, Int, -1)
    .REQUIRED_ATTR(component_types, ListType)
    .OP_END_FACTORY_REG(QueueDequeueUpTo)

/**
*@brief Stage values similar to a lightweight Enqueue. \n

*@par Inputs:
*The input values must be a list of Tensor objects. Inputs include:
*values: A list of Tensor objects. A list of data types that inserted values
should adhere to. It's a dynamic input. \n

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to 0. Maximum number of
elements in the Staging Area. If > 0, inserts on the container will block
when the capacity is reached.
*@li memory_limit: An optional int that is >= 0. Defaults to 0. The maximum
number of bytes allowed for Tensors in the Staging Area. If > 0, inserts will
block until sufficient space is available.
*@li container: An optional string. Defaults to "". If non-empty, this queue
is placed in the given container. Otherwise, a default container is used.
*@li shared_name: An optional string. Defaults to "". It is necessary to
match this name to the matching Unstage Op. \n

*@see Unstage

*@par Third-party framework compatibility
*Compatible with tensorflow Stage operator.
*/

REG_OP(Stage)
    .DYNAMIC_INPUT(values, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, \
        DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, \
        DT_DOUBLE, DT_UINT32, DT_UINT64}))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(Stage)

/**
*@brief Op removes all elements in the underlying container. \n

*@par Attributes:
*@li capacity: A list of DTypes
*@li memory_limit: An optional int that is >= 0. Defaults to 0.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "".
*@li dtypes: A list of DTypes. \n

*@see Stage

*@par Third-party framework compatibility
*Compatible with tensorflow StageClear operator.
*/

REG_OP(StageClear)
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .ATTR(dtypes, ListType, {})
    .OP_END_FACTORY_REG(StageClear)

/**
*@brief Op peeks at the values at the specified index. If the underlying
container does not contain sufficient elements this op will block until it does. \n

*@par Inputs:
*The input values must be type int32. Inputs include:
*values: A Tensor of type int32. \n

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to 0.
*@li memory_limit: An optional int that is >= 0. Defaults to 0.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "".
*@li dtypes: A list of DTypes that has length >= 1. \n

*@par Outputs:
*y:A list of Tensor objects of type dtypes. \n

*@par Third-party framework compatibility
*Compatible with tensorflow StagePeek operator.
*/

REG_OP(StagePeek)
    .INPUT(index, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, \
                    DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, \
                    DT_DOUBLE, DT_UINT32, DT_UINT64}))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .ATTR(dtypes, ListType, {})
    .OP_END_FACTORY_REG(StagePeek)

/**
*@brief Op returns the number of elements in the underlying container. \n

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to 0.
*@li memory_limit: An optional int that is >= 0. Defaults to 0.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "".
*@li dtypes: A list of DTypes that has length >= 1. \n

*@par Outputs:
*size:A Tensor of type int32. \n

*@par Third-party framework compatibility
*Compatible with tensorflow StageSize operator.
*/

REG_OP(StageSize)
    .OUTPUT(size, TensorType({DT_INT32}))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .ATTR(dtypes, ListType, {})
    .OP_END_FACTORY_REG(StageSize)

/**
*@brief Pop the element at the top of the stack. \n

*@par Inputs:
*The input handle must be type resource. Inputs include:
*handle: A Tensor of type resource. The handle to a stack. \n

*@par Attributes:
*elem_type: A DType. The type of the elem that is popped. \n

*@par Outputs:
*element:A Tensor of type elem_type. \n

*@par Third-party framework compatibility
*Compatible with tensorflow StackPop operator.
*/

REG_OP(StackPop)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .OUTPUT(element, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, \
                     DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, \
                     DT_DOUBLE, DT_UINT32, DT_UINT64}))
    .REQUIRED_ATTR(elem_type, Type)
    .OP_END_FACTORY_REG(StackPop)

/**
*@brief Push an element onto the stack. \n

*@par Inputs:
*The input handle must be type resource. Inputs include:
*@li handle: A Tensor of type resource. The handle to a stack.
*@li elem: A Tensor. The tensor to be pushed onto the stack. \n

*@par Attributes:
*swap_memory: An optional bool. Defaults to False. Swap elem to CPU. Default
to false. \n

*@par Outputs:
*y:A Tensor. Has the same type as elem. \n

*@par Third-party framework compatibility
*Compatible with tensorflow StackPush operator.
*/

REG_OP(StackPush)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(element, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, \
                     DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, \
                     DT_DOUBLE, DT_UINT32, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, \
                     DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, \
                     DT_DOUBLE, DT_UINT32, DT_UINT64}))
    .ATTR(swap_memory, Bool, false)
    .OP_END_FACTORY_REG(StackPush)

/**
*@brief Close the stack. \n

*@par Inputs:
*The input handle must be type resource. Inputs include:
*handle: A Tensor of type resource. The handle to a stack. \n

*@par Third-party framework compatibility
*Compatible with tensorflow StackClose operator.
*/

REG_OP(StackClose)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .OP_END_FACTORY_REG(StackClose)

/**
*@brief Create a stack. \n

*@par Inputs:
*The input max_size must be type int32. Inputs include:
*max_size: A Tensor of type int32. The number of elements of a stack. \n

*@par Attributes:
*@li stack_name: An optional string. Defaults to "".
*@li elem_type: The elements type of the created Stack. \n

*@par Outputs:
*handle: A Tensor of type resource. The handle to a stack. \n

*@par Third-party framework compatibility
*Compatible with tensorflow Stack operator.
*/

REG_OP(Stack)
    .INPUT(max_size, TensorType({DT_INT32}))
    .OUTPUT(handle, TensorType({DT_RESOURCE}))
    .ATTR(stack_name, String, "")
    .REQUIRED_ATTR(elem_type, Type)
    .OP_END_FACTORY_REG(Stack)

/**
*@brief Partitions "x" into "num_partitions" tensors using indices from "partitions". \n

*@par Inputs:
*Including:
* @li x: The Tensor to be sliced. Must be one of the following types:
DT_INT8, DT_UINT8, DT_INT16, DT_UINT16,
DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, \
DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING.
* @li partitions: A Tensor of type DT_INT32, with any shape. The indices. \n

*@par Attributes:
*num_partitions: The number of partitions to output. \n

*@par Outputs:
*y: A list of tensors of type DT_INT32. \n

*@attention Constraints:
*DynamicPartition runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator DynamicPartition. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(DynamicPartition)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, \
        DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING}))
    .INPUT(partitions, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, \
        DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING}))
    .ATTR(num_partitions, Int, 1)
    .OP_END_FACTORY_REG(DynamicPartition)

/**
*@brief Interleaves the values from the "x" tensors into a single tensor. \n

*@par Inputs:
*Including:
* @li indices: A list of at least 1 Tensor objects with type DT_INT32. It's a dynamic input.
* @li x: A list with the same length as "indices" of Tensor objects.
Must be one of the following types: DT_INT8, DT_UINT8, DT_INT16, DT_UINT16,
DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_QINT32,
DT_QUINT8, DT_QINT8, DT_STRING, DT_COMPLEX64, DT_COMPLEX128. It's a dynamic input. \n

*@par Attributes:
*N: An int that is >= 1. Defaults to "1". \n

*@par Outputs:
*y: A Tensor. Has the same type as "x". \n

*@attention Constraints:
*DynamicStitch runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator DynamicStitch. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(DynamicStitch)
    .DYNAMIC_INPUT(indices, TensorType({DT_INT32}))
    .DYNAMIC_INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, \
        DT_QINT32, DT_QUINT8, DT_QINT8, DT_STRING, DT_COMPLEX64, \
        DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, \
        DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, \
        DT_QINT32, DT_QUINT8, DT_QINT8, DT_STRING, DT_COMPLEX64, \
        DT_COMPLEX128}))
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(DynamicStitch)

/**
*@brief Interleaves the values from the "x" tensors into a single tensor. \n

*@par Inputs:
*Including:
* @li indices: A list of at least 1 Tensor objects with type DT_INT32. It's a dynamic input.
* @li x: A list with the same length as "indices" of Tensor objects. It's a dynamic input.
Must be one of the following types: DT_INT8, DT_UINT8, DT_INT16, DT_UINT16,
DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_STRING,
DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT32. \n

*@par Attributes:
*N: An int that is >= 1. Defaults to "1". \n

*@par Outputs:
*y: A Tensor. Has the same type as "x". \n

*@attention Constraints:
*ParallelDynamicStitch runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ParallelDynamicStitch.
*/

REG_OP(ParallelDynamicStitch)
    .DYNAMIC_INPUT(indices, TensorType({DT_INT32}))
    .DYNAMIC_INPUT(x,
        TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, \
        DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_STRING, DT_COMPLEX64, DT_COMPLEX128, \
        DT_QINT8, DT_QUINT8, DT_QINT32 }))
    .OUTPUT(y,
        TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, \
        DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_STRING, DT_COMPLEX64, DT_COMPLEX128, \
        DT_QINT8, DT_QUINT8, DT_QINT32 }))
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(ParallelDynamicStitch)

/**
*@brief Removes all elements in the underlying container. \n

*@par Attributes:An optional int that is >= 0. Defaults to "0".
*@li capacity: An optional int that is >= 0. Defaults to "0".
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li dtypes: A list of tf.DTypes.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "". \n

*@attention Constraints:
*MapClear runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator MapClear.
*/

REG_OP(MapClear)
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(MapClear)

/**
*@brief Returns the number of incomplete elements in the underlying container. \n

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to "0".
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li dtypes: A list of tf.DTypes.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "". \n

*@par Outputs:
*size: A Tensor of type DT_INT32. \n

*@attention Constraints:
*MapIncompleteSize runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator MapIncompleteSize.
*/

REG_OP(MapIncompleteSize)
    .OUTPUT(size, TensorType({DT_INT32}))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(MapIncompleteSize)

/**
*@brief Unstage Op is similar to a lightweight Dequeue. \n

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to 0.
*@li memory_limit: An optional int that is >= 0. Defaults to 0.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "".
*@li dtypes: A list of DTypes that has length >= 1. \n

*@par Outputs:
*y: A list of Tensor objects of type dtypes. \n

*@par Third-party framework compatibility
*Compatible with tensorflow Unstage operator.
*/

REG_OP(Unstage)
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, \
            DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, \
            DT_DOUBLE, DT_UINT32, DT_UINT64}))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .REQUIRED_ATTR(dtypes, ListType)
    .OP_END_FACTORY_REG(Unstage)

/**
*@brief Stage (key, values) in the underlying container which behaves like a hashtable. \n

*@par Inputs:
*Including:
* @li key: A Tensor of type DT_INT64.
* @li indices: A Tensor of type DT_INT32.
* @li values: A list of Tensor objects for tensor dtypes.
A list of data types that inserted values should adhere to of.
Must be one of the following types: DT_FLOAT, DT_FLOAT16,
DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
DT_INT64, DT_BOOL, DT_DOUBLE, DT_UINT32, DT_UINT64,
DT_RESOURCE, DT_STRING, DT_COMPLEX64, DT_COMPLEX128,
DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32.
It's a dynamic input. \n

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to "0".
Maximum number of elements in the Staging Area. If > 0,
inserts on the container will block when the capacity is reached.
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li dtypes: A list of tf.DTypes.
*@li container: An optional string. Defaults to "".
If non-empty, this queue is placed in the given container.
Otherwise, a default container is used.
*@li shared_name: An optional string. Defaults to "".
It is necessary to match this name to the matching Unstage Op. \n

*@attention Constraints:
*MapStage runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator MapStage.
*/

REG_OP(MapStage)
    .INPUT(key, TensorType({DT_INT64}))
    .INPUT(indices, TensorType({DT_INT32}))
    .DYNAMIC_INPUT(values,
        TensorType({ DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, DT_UINT32, \
        DT_UINT64, DT_RESOURCE, DT_STRING, DT_COMPLEX64, DT_COMPLEX128, \
        DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32 }))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(MapStage)

/**
*@brief Removes and returns the values associated with the key. \n

*@par Inputs:
*Including:
* @li key: A Tensor of type DT_INT64.
* @li indices: A Tensor of type DT_INT32. \n

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to "0".
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li dtypes: A list of DTypes that has length >= 1.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "". \n

*@par Outputs:
*values: A list of Tensor objects. Must be one of the following types:
DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
DT_INT64, DT_BOOL, DT_DOUBLE, DT_UINT32, DT_UINT64, DT_RESOURCE,
DT_STRING, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8,
DT_QINT16, DT_QUINT16, DT_QINT32. \n

*@attention Constraints:
*MapUnstage runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator MapUnstage.
*/

REG_OP(MapUnstage)
    .INPUT(key, TensorType({DT_INT64}))
    .INPUT(indices, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(values,
        TensorType({ DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, DT_UINT32, \
        DT_UINT64, DT_RESOURCE, DT_STRING, DT_COMPLEX64, DT_COMPLEX128, \
        DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32 }))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(MapUnstage)

/**
*@brief Removes and returns a random (key, value). \n

*@par Inputs:
*Including:
*indices: A Tensor of type DT_INT32. \n

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to "0".
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li dtypes: A list of DTypes that has length >= 1.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "". \n

*@par Outputs:
*@li key: A Tensor of type DT_INT64.
*@li values: A list of Tensor objects.
Must be one of the following types: DT_FLOAT, DT_FLOAT16, DT_INT8,
DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE,
DT_UINT32, DT_UINT64, DT_RESOURCE, DT_STRING, DT_COMPLEX64, DT_COMPLEX128,
DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32. \n

*@attention Constraints:
*MapUnstageNoKey runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator MapUnstageNoKey.
*/

REG_OP(MapUnstageNoKey)
    .INPUT(indices, TensorType({DT_INT32}))
    .OUTPUT(key, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(values,
        TensorType({ DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, DT_UINT32, \
        DT_UINT64, DT_RESOURCE, DT_STRING, DT_COMPLEX64, DT_COMPLEX128, \
        DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32 }))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(MapUnstageNoKey)

/**
*@brief Peeks at the values at the specified key. \n

*@par Inputs:
*Including:
* @li key: A Tensor of type DT_INT64.
* @li indices: A Tensor of type DT_INT32. \n

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to "0".
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li dtypes: A list of tf.DTypes that has length >= 1.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "". \n

*@par Outputs:
*values: A list of Tensor objects of type "dtypes".
Must be one of the following types: DT_FLOAT, DT_FLOAT16, DT_INT8,
DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL,
DT_DOUBLE, DT_UINT32, DT_UINT64, DT_RESOURCE, DT_STRING, DT_COMPLEX64,
DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32. \n

*@attention Constraints:
*MapPeek runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator MapPeek.
*/

REG_OP(MapPeek)
    .INPUT(key, TensorType({DT_INT64}))
    .INPUT(indices, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(values,
        TensorType({ DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, DT_UINT32, \
        DT_UINT64, DT_RESOURCE, DT_STRING, DT_COMPLEX64, DT_COMPLEX128, \
        DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32 }))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(MapPeek)

/**
*@brief Returns the number of elements in the underlying container. \n

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to "0".
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li dtypes: A list of tf.DTypes.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "". \n

*@par Outputs:
*size: A Tensor of type DT_INT32. \n

*@attention Constraints:
*MatMul runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator MapSize.
*/

REG_OP(MapSize)
    .OUTPUT(size, TensorType({DT_INT32}))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(MapSize)

/**
*@brief Class wrapping dynamic-sized, per-time-step, write-once Tensor arrays. \n

*@par Inputs:
*The input size must be type int32. Inputs include:
*@li size: int32 scalar Tensor: the size of the TensorArray. Required if
handle is not provided. \n

*@par Attributes:
*@li dtype: The data type of this TensorArray.
*@li element_shape: The TensorShape of elements in this TensorArray.
*@li dynamic_size: A boolean that determines whether writes to the
TensorArray are allowed to grow the size.
*@li clear_after_read: Boolean (optional, default: True). If True, clear
TensorArray values
after reading them. This disables read-many semantics, but allows early
release of memory.
*@li identical_element_shapes: If true (default is false), then all elements
in the TensorArray will be expected to have have identical shapes.
*@li tensor_array_name: String: the name of the TensorArray. \n

*@par Outputs:
*@li handle: The handle to the TensorArray.
*@li flow: A scalar used to control gradient flow. \n

*@par Third-party framework compatibility
*Compatible with tensorflow TensorArray operator.
*/

REG_OP(TensorArray)
    .INPUT(size, TensorType({DT_INT32}))
    .OUTPUT(handle, TensorType({DT_RESOURCE}))
    .OUTPUT(flow, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(dtype, Type)
    .ATTR(element_shape, ListInt, ge::UNKNOWN_RANK)
    .ATTR(dynamic_size, Bool, false)
    .ATTR(clear_after_read, Bool, true)
    .ATTR(identical_element_shapes, Bool, false)
    .ATTR(tensor_array_name, String, "")
    .OP_END_FACTORY_REG(TensorArray)

/**
*@brief Delete the TensorArray from its resource container. \n

*@par Inputs:
*The input handle must be type resource. Inputs include:
*handle: A Tensor of type resource. The handle to a TensorArray
(output of TensorArray or TensorArrayGrad). \n

*@par Third-party framework compatibility
*Compatible with tensorflow TensorArrayClose operator.
*/

REG_OP(TensorArrayClose)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .OP_END_FACTORY_REG(TensorArrayClose)

/**
*@brief Concat the elements from the TensorArray into value value. \n

*@par Inputs:
*The input handle must be type resource. Inputs include:
*@li handle: The handle to a TensorArray.
*@li flow_in: A float scalar that enforces proper chaining of operations. \n

*@par Attributes:
*@li dtype: The type of the elem that is returned.
*@li element_shape_except0: The expected shape of an element, if known,
excluding the first dimension. \n

*@par Outputs:
*@li value: All of the elements in the TensorArray, concatenated along
the first axis.
*@li lengths: A vector of the row sizes of the original T elements in the
value output. \n

*@par Third-party framework compatibility
*Compatible with tensorflow TensorArrayConcat operator.
*/

REG_OP(TensorArrayConcat)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(flow_in, TensorType({DT_FLOAT}))
    .OUTPUT(value, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8,
        DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL,
        DT_STRING, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8,
        DT_QUINT8, DT_QINT32}))
    .OUTPUT(lengths, TensorType({DT_INT64}))
    .REQUIRED_ATTR(dtype, Type)
    .ATTR(element_shape_except0, ListInt, ge::UNKNOWN_RANK)
    .OP_END_FACTORY_REG(TensorArrayConcat)

/**
*@brief All elements selected by indices must have the same shape. \n

*@par Inputs:
*The input handle must be type resource. Inputs include:
*@li handle: The handle to a TensorArray.
*@li indices: The locations in the TensorArray from which to read tensor
elements.
*@li flow_in: A float scalar that enforces proper chaining of operations. \n

*@par Attributes:
*@li dtype: The type of the elem that is returned.
*@li element_shape: The expected shape of an element, if known. Used to
validate the shapes of TensorArray elements. If this shape is not fully
specified, gathering zero-size TensorArrays is an error. \n

*@par Outputs:
*value:  All of the elements in the TensorArray, concatenated along a new
axis (the new dimension 0). \n

*@par Third-party framework compatibility
*Compatible with tensorflow TensorArrayGather operator.
*/

REG_OP(TensorArrayGather)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(flow_in, TensorType({DT_FLOAT}))
    .OUTPUT(value, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8,
        DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL,
        DT_STRING, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8,
        DT_QUINT8, DT_QINT32}))
    .REQUIRED_ATTR(dtype, Type)
    .ATTR(element_shape, ListInt, ge::UNKNOWN_RANK)
    .OP_END_FACTORY_REG(TensorArrayGather)

/**
*@brief Creates a TensorArray for storing the gradients of values in the
given handle. \n

*@par Inputs:
*The input handle must be type resource. Inputs include:
*@li handle: The handle to a TensorArray.
*@li flow_in: A float scalar that enforces proper chaining of operations. \n

*@par Attributes:
*source: The gradient source string, used to decide which gradient
TensorArray to return. \n

*@par Outputs:
*@li grad_handle: A Tensor of type resource.
*@li flow_out: A Tensor of type float. \n

*@par Third-party framework compatibility
*Compatible with tensorflow TensorArrayGrad operator.
*/

REG_OP(TensorArrayGrad)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(flow_in, TensorType({DT_FLOAT}))
    .OUTPUT(grad_handle, TensorType({DT_RESOURCE}))
    .OUTPUT(flow_out, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(source, String)
    .OP_END_FACTORY_REG(TensorArrayGrad)

/**
*@brief Push an element onto the tensor_array. \n

*@par Inputs:
*The input handle must be type resource. Inputs include:
*@li handle: The handle to a TensorArray.
*@li index: The position to write to inside the TensorArray.
*@li value: The tensor to write to the TensorArray.
*@li flow_in: A float scalar that enforces proper chaining of operations. \n

*@par Outputs:
*flow_out: A float scalar that enforces proper chaining of operations. \n

*@par Third-party framework compatibility
*Compatible with tensorflow TensorArrayWrite operator.
*/

REG_OP(TensorArrayWrite)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(index, TensorType({DT_INT32}))
    .INPUT(value, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8,
        DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL,
        DT_STRING, DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(flow_in, TensorType({DT_FLOAT}))
    .OUTPUT(flow_out, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(TensorArrayWrite)

/**
*@brief Creates a TensorArray for storing multiple gradients of values in
the given handle. \n

*@par Inputs:
*The input handle must be type resource. Inputs include:
*@li handle: A Tensor of type resource. The handle to the forward TensorArray.
*@li flow_in: A Tensor of type float. A float scalar that enforces proper
chaining of operations.
*@li shape_to_prepend: A Tensor of type int32. An int32 vector representing
a shape. \n

*@par Attributes:
*source: A string. The gradient source string, used to decide which gradient
TensorArray to return. \n

*@par Outputs:
*@li grad_handle: A Tensor of type resource.
*@li flow_out: A Tensor of type float. \n

*@par Third-party framework compatibility
*Compatible with tensorflow TensorArrayGradWithShape operator.
*/

REG_OP(TensorArrayGradWithShape)
    .INPUT(handle, TensorType({ DT_RESOURCE }))
    .INPUT(flow_in, TensorType({ DT_FLOAT }))
    .INPUT(shape_to_prepend, TensorType({ DT_INT32 }))
    .OUTPUT(grad_handle, TensorType({ DT_RESOURCE }))
    .OUTPUT(flow_out, TensorType({ DT_FLOAT }))
    .ATTR(source, String, "")
    .OP_END_FACTORY_REG(TensorArrayGradWithShape)

/**
*@brief Read an element from the TensorArray into output value. \n

*@par Inputs:
*The input handle must be type resource. Inputs include:
*@li handle: A Tensor of type resource. The handle to a TensorArray.
*@li index: A Tensor of type int32.
*@li flow_in: A Tensor of type float. \n

*@par Attributes:
*dtype: A DType. \n

*@par Outputs:
*y: A Tensor of type dtype. \n

*@par Third-party framework compatibility
*Compatible with tensorflow TensorArrayRead operator.
*/

REG_OP(TensorArrayRead)
    .INPUT(handle, TensorType({ DT_RESOURCE }))
    .INPUT(index, TensorType({ DT_INT32 }))
    .INPUT(flow_in, TensorType({ DT_FLOAT }))
    .OUTPUT(y, TensorType({ DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16,
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE,
        DT_STRING, DT_COMPLEX64, DT_COMPLEX128}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(TensorArrayRead)

/**
*@brief Scatter the data from the input value into specific TensorArray
elements. \n

*@par Inputs:
*The input handle must be type resource. Inputs include:
*@li handle: The handle to a TensorArray.
*@li indices: The locations at which to write the tensor elements.
*@li value: The concatenated tensor to write to the TensorArray.
*@li flow_in: A float scalar that enforces proper chaining of operations. \n

*@par Outputs:
*flow_out: A float scalar that enforces proper chaining of operations. \n

*@par Third-party framework compatibility
*Compatible with tensorflow TensorArrayScatter operator.
*/

REG_OP(TensorArrayScatter)
    .INPUT(handle, TensorType({ DT_RESOURCE }))
    .INPUT(indices, TensorType({ DT_INT32 }))
    .INPUT(value, TensorType({ DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16,
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE,
        DT_STRING, DT_COMPLEX64, DT_COMPLEX128 }))
    .INPUT(flow_in, TensorType({ DT_FLOAT }))
    .OUTPUT(flow_out, TensorType({ DT_FLOAT }))
    .OP_END_FACTORY_REG(TensorArrayScatter)

/**
*@brief Split the data from the input value into TensorArray elements. \n

*@par Inputs:
*The input handle must be type resource. Inputs include:
*@li handle: The handle to a TensorArray.
*@li value: The concatenated tensor to write to the TensorArray.
*@li lengths: The vector of lengths, how to split the rows of value into
the TensorArray.
*@li flow_in: A float scalar that enforces proper chaining of operations. \n

*@par Outputs:
*flow_out: A float scalar that enforces proper chaining of operations. \n

*@par Third-party framework compatibility
*Compatible with tensorflow TensorArraySplit operator.
*/

REG_OP(TensorArraySplit)
    .INPUT(handle, TensorType({ DT_RESOURCE }))
    .INPUT(value, TensorType({ DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16,
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE,
        DT_STRING, DT_COMPLEX64, DT_COMPLEX128 }))
    .INPUT(lengths, TensorType({ DT_INT64 }))
    .INPUT(flow_in, TensorType({ DT_FLOAT }))
    .OUTPUT(flow_out, TensorType({ DT_FLOAT }))
    .OP_END_FACTORY_REG(TensorArraySplit)

/**
*@brief Return the number of elements in a TensorArray. \n

*@par Inputs:
*The input handle must be type resource. Inputs include:
*@li handle: The handle to a TensorArray.
*@li flow_in: A float scalar that enforces proper chaining of operations. \n

*@par Outputs:
*size: The number of elements in a TensorArray.. \n

*@par Third-party framework compatibility
*Compatible with tensorflow TensorArraySize operator.
*/

REG_OP(TensorArraySize)
    .INPUT(handle, TensorType({ DT_RESOURCE }))
    .INPUT(flow_in, TensorType({ DT_FLOAT }))
    .OUTPUT(size, TensorType({ DT_INT32 }))
    .OP_END_FACTORY_REG(TensorArraySize)

/**
*@brief A queue implementation that dequeues elements in a random order. \n

*@par Attributes:
*@li component_types:A list of fully-defined Tensortype objects with
the same length as shapes, or None.
*@li shapes: (Optional.) A list of fully-defined TensorShape objects with
the same length as dtypes, or None.
*@li capacity: An integer. The upper bound on the number of elements that may
be stored in this queue.
*@li min_after_dequeue: An integer (described above).
*@li seed: An integer. Used to create a random seed.
*@li seed2: An integer. Used to create a random seed.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "". \n

*@par Outputs:
*handle: A Tensor of type resource. The handle to a stack. \n

*@par Third-party framework compatibility
*Compatible with tensorflow RandomShuffleQueue operator.
*/

REG_OP(RandomShuffleQueue)
    .OUTPUT(handle, TensorType({DT_RESOURCE}))
    .REQUIRED_ATTR(component_types, ListType)
    .ATTR(shapes, ListListInt, {})
    .ATTR(capacity, Int, -1)
    .ATTR(min_after_dequeue, Int, 0)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(RandomShuffleQueue)

/**
*@brief A queue that produces elements in first-in first-out order. \n

*@par Attributes:
*@li shapes: An optional list of shapes for each component of
a queue element. Defaults to {}. The length of this attr must be
either 0 or the same as the length of "component_types". Shapes of fixed
rank but variable size are allowed by setting any shape dimension to "-1".
In this case, the inputs' shape may vary along the given dimension,
and DequeueMany will pad the given dimension with zeros up to the maximum
shape of all elements in the given batch. If the length of this attr is "0",
different queue elements may have different ranks and shapes, but only one
element may be dequeued at a time.
*@li capacity: An optional int. Defaults to "-1". The upper bound on the number
of elements in this queue. Negative numbers mean no limit.
*@li container: An optional string. Defaults to "". If non-empty, this queue
is placed in the given container. Otherwise, a default container is used.
*@li shared_name: An optional string. Defaults to "". If non-empty, this queue
will be shared under the given name across multiple sessions. \n

*@par Outputs:
*handle: A Tensor of type DT_RESOURCE. \n

*@attention Constraints:
*PaddingFIFOQueue runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator PaddingFIFOQueue.
*/

REG_OP(PaddingFIFOQueue)
    .OUTPUT(handle, TensorType({DT_RESOURCE}))
    .REQUIRED_ATTR(component_types, ListType)
    .ATTR(shapes, ListListInt, {})
    .ATTR(capacity, Int, -1)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(PaddingFIFOQueue)

/**
*@brief A queue that produces elements sorted by the first component value. \n

*@par Attributes:
*@li component_types: An optional list of tf.DTypes. Defaults to {}.
The type of each component in a value.
*@li shapes: A list of shapes for each component of a queue element.
The length of this attr must be either 0 or the same as the length of
"component_types". If the length of this attr is 0, the shapes of queue
elements are not constrained, and only one element may be dequeued at a time.
*@li container: An optional string. Defaults to "". If non-empty, this queue
is placed in the given container. Otherwise, a default container is used.
*@li capacity:An integer. The upper bound on the number of elements that may be stored in this queue.
*@li shared_name: An optional string. Defaults to "". If non-empty, this
queue will be shared under the given name across multiple sessions. \n

*@par Outputs:
*handle: A Tensor of type DT_RESOURCE. \n

*@attention Constraints:
*PriorityQueue runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator PriorityQueue.
*/

REG_OP(PriorityQueue)
    .OUTPUT(handle, TensorType({DT_RESOURCE}))
    .ATTR(component_types, ListType, {})
    .ATTR(shapes, ListListInt, {})
    .ATTR(capacity, Int, -1)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(PriorityQueue)

/**
*@brief Multiplies the matrix "x1" by the matrix "x2". \n

*@par Inputs:
*Including:
*handle: A Tensor of type DT_RESOURCE. The handle to a queue. \n

*@par Attributes:
*cancel_pending_enqueues: An optional bool. Defaults to "False".
If true, all pending enqueue requests that are blocked on
the given queue will be canceled. \n

*@attention Constraints:
*QueueClose runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator QueueClose.
*/

REG_OP(QueueClose)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .ATTR(cancel_pending_enqueues, Bool, false)
    .OP_END_FACTORY_REG(QueueClose)

/**
*@brief Stage (key, values) in the underlying container which behaves like an ordered associative container. \n

*@par Inputs:
*Including:
* @li key: A Tensor of type DT_INT64.
* @li indices: A Tensor of type DT_INT32.
* @li values: A list of Must be one of the following types:
DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, DT_UINT32, DT_UINT64,
DT_RESOURCE, DT_STRING, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8,
DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32 that inserted
values should adhere to. It's a dynamic input.  \n

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to "0".
Maximum number of elements in the Staging Area.
If > 0, inserts on the container will block
when the capacity is reached.
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li dtypes: A list of DTypes.
*@li container: An optional string. Defaults to "".
If non-empty, this queue is placed in the given container.
Otherwise, a default container is used.
*@li shared_name: An optional string. Defaults to "".
It is necessary to match this name to the matching Unstage Op. \n

*@attention Constraints:
*OrderedMapStage runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator OrderedMapStage.
*/

REG_OP(OrderedMapStage)
    .INPUT(key, TensorType({DT_INT64}))
    .INPUT(indices, TensorType({DT_INT32}))
    .DYNAMIC_INPUT(values,
        TensorType({ DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, DT_UINT32, \
        DT_UINT64, DT_RESOURCE, DT_STRING, DT_COMPLEX64, DT_COMPLEX128, \
        DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32 }))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(OrderedMapStage)

/**
*@brief Returns the number of elements in the underlying container. \n

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to "0".
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li dtypes: A list of DTypes.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "". \n

*@par Outputs:
*size: A Tensor of type DT_INT32. \n

*@attention Constraints:
*OrderedMapSize runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator OrderedMapSize.
*/

REG_OP(OrderedMapSize)
    .OUTPUT(size, TensorType({DT_INT32}))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(OrderedMapSize)

/**
*@brief Removes all elements in the underlying container. \n

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to "0".
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li dtypes: A list of DTypes.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "". \n

*@attention Constraints:
*OrderedMapClear runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator OrderedMapClear.
*/

REG_OP(OrderedMapClear)
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(OrderedMapClear)

/**
*@brief FakeQueue, support tf api FixedLengthRecordReader. \n

*@par Inputs:
*Including:
* resource: A Tensor of type DT_RESOURCE.

*@par Outputs:
*handle: A Tensor of type DT_STRING ref. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator FakeQueue.
*/
REG_OP(FakeQueue)
    .INPUT(resource, TensorType({DT_RESOURCE}))
    .OUTPUT(handle, TensorType({DT_STRING}))
    .OP_END_FACTORY_REG(FakeQueue)

/**
*@brief Returns the number of incomplete elements in the underlying container. \n

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to "0".
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li dtypes: A list of DTypes.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "". \n

*@par Outputs:
*size: A Tensor of type DT_INT32. \n

*@attention Constraints:
*OrderedMapIncompleteSize runs on the Ascend AI CPU,
which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator OrderedMapIncompleteSize.
*/

REG_OP(OrderedMapIncompleteSize)
    .OUTPUT(size, TensorType({DT_INT32}))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(OrderedMapIncompleteSize)

/**
*@brief Peeks at the values at the specified key. \n

*@par Inputs:
*Including:
* @li key: A Tensor of type DT_INT64.
* @li indices: A Tensor of type DT_INT32. \n

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to "0".
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li dtypes: A list of DTypes that has length >= 1.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "". \n

*@par Outputs:
*values: A list of Tensor objects. Must be one of the following types:
DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
DT_INT64, DT_BOOL, DT_DOUBLE, DT_UINT32, DT_UINT64, DT_RESOURCE, DT_STRING,
DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32. \n

*@attention Constraints:
*OrderedMapPeek runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator OrderedMapPeek.
*/

REG_OP(OrderedMapPeek)
    .INPUT(key, TensorType({DT_INT64}))
    .INPUT(indices, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(values,
        TensorType({ DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, DT_UINT32, \
        DT_UINT64, DT_RESOURCE, DT_STRING, DT_COMPLEX64, DT_COMPLEX128, \
        DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32 }))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(OrderedMapPeek)

/**
*@brief Removes and returns the (key, value) element with the smallest. \n

*@par Inputs:
*Including:
* indices: A Tensor of type DT_INT32. \n

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to "0".
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li dtypes: A list of DTypes that has length >= 1.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "". \n

*@par Outputs:
*@li key: A Tensor of type DT_INT64.
*@li values: A list of Tensor objects. Must be one of the following types:
DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
DT_INT64, DT_BOOL, DT_DOUBLE, DT_UINT32, DT_UINT64, DT_RESOURCE, DT_STRING,
DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32. \n

*@attention Constraints:
*OrderedMapUnstageNoKey runs on the Ascend AI CPU,
which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator OrderedMapUnstageNoKey.
*/

REG_OP(OrderedMapUnstageNoKey)
    .INPUT(indices, TensorType({DT_INT32}))
    .OUTPUT(key, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(values,
        TensorType({ DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, DT_UINT32, \
        DT_UINT64, DT_RESOURCE, DT_STRING, DT_COMPLEX64, DT_COMPLEX128, \
        DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32 }))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(OrderedMapUnstageNoKey)

/**
*@brief Removes and returns the values associated with the key. \n

*@par Inputs:
*Including:
* @li key: A Tensor of type DT_INT64.
* @li indices: A Tensor of type DT_INT32. \n

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to "0".
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li dtypes: A list of tf.DTypes that has length >= 1.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "". \n

*@par Outputs:
*values: A list of Tensor objects. Must be one of the following types:
DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT,
DT_FLOAT16, DT_DOUBLE, DT_BOOL, DT_UINT32, DT_UINT64. \n

*@attention Constraints:
*OrderedMapUnstage runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator OrderedMapUnstage.
*/

REG_OP(OrderedMapUnstage)
    .INPUT(key, TensorType({DT_INT64}))
    .INPUT(indices, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(values, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16,
                                        DT_INT32, DT_INT64, DT_FLOAT, DT_FLOAT16,
                                        DT_DOUBLE, DT_BOOL, DT_UINT32, DT_UINT64}))
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(OrderedMapUnstage)

/**
*@brief A barrier represents a key-value map, where each key is a string,
and each value is a tuple of tensors. \n

*@par Attributes:
*@li component_types: The type of each component in a value.
*@li shapes: A list of shapes for each component of a queue element.
Each shape must be 1 in the first dimension.
The length of this attr must be the same as
the length of "component_types".
*@li capacity: The capacity of the barrier.
The default capacity is MAX_INT32,
which is the largest capacity of the underlying queue.
*@li container: If non-empty, this barrier is placed in the given container.
Otherwise, a default container is used.
*@li shared_name: If non-empty, this barrier will be shared under
the given name across multiple sessions. \n

*@par Outputs:
*handle: A Tensor of type DT_STRING_REF. The handle to the barrier. \n

*@attention Constraints:
*Barrier runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Barrier. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(Barrier)
    .OUTPUT(handle, TensorType({DT_STRING_REF}))
    .REQUIRED_ATTR(component_types, ListType)
    .ATTR(shapes, ListListInt, {})
    .ATTR(capacity, Int, -1)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(Barrier)

/**
*@brief For each key, assigns the respective value to the specified component. \n

*@par Inputs:
*Including:
* @li handle: A Tensor of type DT_STRING_REF. The handle to a barrier.
* @li keys: A Tensor of type DT_STRING. A 1D tensor of keys.
* @li values: An any-dimensional tensor of values, which are associated
with the respective keys. The 0th dimension must have length n
Must be one of the following types: DT_FLOAT, DT_FLOAT16, DT_INT8,
DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL,
DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128,  DT_RESOURCE, DT_STRING. \n

*@par Attributes:
*component_index: The component of the barrier elements that is being assigned. \n

*@attention Constraints:
*BarrierInsertMany runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator BarrierInsertMany. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(BarrierInsertMany)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .INPUT(keys, TensorType({DT_STRING}))
    .INPUT(values,
        TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, \
        DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING}))
    .REQUIRED_ATTR(component_index, Int)
    .OP_END_FACTORY_REG(BarrierInsertMany)

/**
*@brief Takes the given number of completed elements from a barrier. \n

*@par Inputs:
*Including:
* @li handle: A Tensor of type DT_STRING_REF. The handle to a barrier.
* @li num_elements: A Tensor of type DT_INT32.
A single-element tensor containing the number of elements to take. \n

*@par Attributes:
*@li component_types: The type of each component in a value.
*@li allow_small_batch: Allow to return less than "num_elements"
items if barrier is already closed.
*@li wait_for_incomplete: An any-dimensional tensor
for each component in the barrier element.
*@li timeout_ms: If the queue is empty, this operation will block for up to
"timeout_ms" milliseconds. Note: This option is not supported yet. \n

*@par Outputs:
*@li indices: A 1D tensor of type DT_INT64. The indices, with length "num_elems".
These indices refer to the batch in which the values were
placed into the barrier.
*@li keys: A 1D tensor of keys,
with length "num_elements" of type DT_STRING.
*@li values: A 1D tensor per component in a barrier element.
All values have length "num_elements" along the 0th dimension.
Must be one of the following types:
DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128,
DT_RESOURCE, DT_STRING. \n

*@attention Constraints:
*BarrierTakeMany runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator BarrierTakeMany. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(BarrierTakeMany)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .INPUT(num_elements, TensorType(DT_INT32))
    .OUTPUT(indices, TensorType({DT_INT64}))
    .OUTPUT(keys, TensorType({DT_STRING}))
    .DYNAMIC_OUTPUT(values,
        TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, \
        DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING}))
    .REQUIRED_ATTR(component_types, ListType)
    .ATTR(allow_small_batch, Bool, false)
    .ATTR(wait_for_incomplete, Bool, false)
    .ATTR(timeout_ms, Int, -1)
    .OP_END_FACTORY_REG(BarrierTakeMany)

/**
*@brief Closes the given barrier. \n

*@par Inputs:
*Including:
*handle: A Tensor of type DT_STRING_REF. The handle to a barrier. \n

*@par Attributes:
*cancel_pending_enqueues: If true, all pending enqueue requests
that are blocked on the barrier's queue will
be canceled. InsertMany will fail,
even if no new key is introduced. \n

*@attention Constraints:
*BarrierClose runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator BarrierClose. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(BarrierClose)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .ATTR(cancel_pending_enqueues, Bool, false)
    .OP_END_FACTORY_REG(BarrierClose)

/**
*@brief Computes the number of complete elements in the given barrier. \n

*@par Inputs:
*Including:
*handle: A Tensor of type DT_STRING_REF. The handle to a barrier. \n

*@par Outputs:
*size: A Tensor of type DT_INT32. The number of complete elements. \n

*@attention Constraints:
*BarrierReadySize runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator BarrierReadySize. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(BarrierReadySize)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .OUTPUT(size, TensorType(DT_INT32))
    .OP_END_FACTORY_REG(BarrierReadySize)

/**
*@brief Computes the number of incomplete elements in the given barrier. \n

*@par Inputs:
*Including:
*handle: A Tensor of type DT_STRING_REF. The handle to a barrier. \n

*@par Outputs:
*size: A Tensor of type DT_INT32. The number of incomplete elements in the barrier. \n

*@attention Constraints:
*BarrierIncompleteSize runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator BarrierIncompleteSize. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(BarrierIncompleteSize)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .OUTPUT(size, TensorType(DT_INT32))
    .OP_END_FACTORY_REG(BarrierIncompleteSize)

/**
*@brief Emits randomized records. \n

*@par Attributes:
*@li file_pattern: A string. Glob pattern for the data files.
*@li file_random_seed: An optional int. Defaults to 301. Random seeds used to
produce randomized records.
*@li file_shuffle_shift_ratio: An optional float. Defaults to 0. Shifts the
list of files after the list is randomly shuffled.
*@li file_buffer_size: An optional int. Defaults to 10000. The randomization
shuffling buffer.
*@li file_parallelism: An optional int. Defaults to 16. How many sstables are
opened and concurrently iterated over.
*@li batch_size: An optional int. Defaults to 32. The batch size.
*@li compression_type: An optional string. Defaults to "". The type of
compression for the file. Currently ZLIB and GZIP are supported. \n

*@par Outputs:
*records: A Tensor of type string. \n

*@par Third-party framework compatibility
*Compatible with tensorflow RecordInput operator.
*/

REG_OP(RecordInput)
    .OUTPUT(records, TensorType({DT_STRING}))
    .REQUIRED_ATTR(file_pattern, String)
    .ATTR(file_random_seed, Int, 301)
    .ATTR(file_shuffle_shift_ratio, Float, 0)
    .ATTR(file_buffer_size, Int, 10000)
    .ATTR(file_parallelism, Int, 16)
    .ATTR(batch_size, Int, 32)
    .ATTR(compression_type, String, "")
    .OP_END_FACTORY_REG(RecordInput)

/**
*@brief A conditional accumulator for aggregating gradients. \n

*@par Attributes:
*@li dtype: The type of the value being accumulated.
*@li shape: The shape of the values, can be [], in which case shape is unknown.
*@li container: If non-empty, this accumulator is placed in the given container.
Otherwise, a default container is used.
*@li shared_name: If non-empty, this accumulator will be shared under the given
name across multiple sessions.
*@li reduction_type: reduction operator type, default "MEAN". \n

*@par Outputs:
*handle: A Tensor of type DT_STRING_REF. The handle to the accumulator. \n

*@attention Constraints:
*ConditionalAccumulator runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ConditionalAccumulator. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(ConditionalAccumulator)
    .OUTPUT(handle, TensorType({DT_STRING_REF}))
    .REQUIRED_ATTR(dtype, Type)
    .REQUIRED_ATTR(shape, ListInt)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .ATTR(reduction_type, String, "MEAN")
    .OP_END_FACTORY_REG(ConditionalAccumulator)

/**
*@brief Applies a gradient to a given accumulator. \n

*@par Inputs:
*Does not add if "local_step" is lesser than the accumulator's "global_step".
* @li handle: A Tensor of type DT_STRING_REF. The handle to an accumulator.
* @li local_step: A Tensor of type DT_INT64.
The "local_step" value at which the gradient was computed. \n

* @li gradient: A tensor of the gradient to be accumulated.
Must be one of the following types:
DT_FLOAT16, DT_FLOAT, DT_DOUBLE

*@par Attributes:
*dtype: Must be one of the following types:
DT_FLOAT16, DT_FLOAT, DT_DOUBLE

*@attention Constraints:
*AccumulatorApplyGradient runs on the Ascend AI CPU,
which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator AccumulatorApplyGradient. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(AccumulatorApplyGradient)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .INPUT(local_step, TensorType({DT_INT64}))
    .INPUT(gradient, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(AccumulatorApplyGradient)

/**
*@brief Returns the number of gradients aggregated in the given accumulators. \n

*@par Inputs:
*Including:
*handle: A Tensor of type DT_STRING_REF. The handle to an accumulator. \n

*@par Outputs:
*y: A Tensor of type DT_INT32. The number of gradients aggregated
in the given accumulator. \n

*@attention Constraints:
*AccumulatorNumAccumulated runs on the Ascend AI CPU,
which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator AccumulatorNumAccumulated. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(AccumulatorNumAccumulated)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(AccumulatorNumAccumulated)

/**
*@brief Updates the accumulator with a new value for "global_step". \n

*@par Inputs:
*Input "new_global_step" is a scalar.
* @li handle: A Tensor of type DT_STRING_REF. The handle to an accumulator.
* @li new_global_step: The new "global_step" value to set A Tensor of type DT_INT64. \n

*@attention Constraints:
*AccumulatorSetGlobalStep runs on the Ascend AI CPU, which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator AccumulatorSetGlobalStep. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(AccumulatorSetGlobalStep)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .INPUT(new_global_step, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(AccumulatorSetGlobalStep)

/**
*@brief Extracts the average gradient in the given ConditionalAccumulator. \n

*@par Inputs:
* Input "num_required" is a scalar.
* @li handle: A Tensor of type DT_STRING_REF. The handle to an accumulator.
* @li num_required: A Tensor of type DT_INT32.
Number of gradients required before an aggregate is returned. \n

*@par Attributes:
*dtype: The data type of accumulated gradients.
Needs to correspond to the type of the accumulator. \n

*@par Outputs:
*y: The average of the accumulated gradients.
Must be one of the following types:
DT_FLOAT16, DT_FLOAT, DT_DOUBLE. \n

*@attention Constraints:
*AccumulatorTakeGradient runs on the Ascend AI CPU,
 which delivers poor performance.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator AccumulatorTakeGradient. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(AccumulatorTakeGradient)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .INPUT(num_required, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(AccumulatorTakeGradient)

/**
*@brief A conditional accumulator for aggregating sparse gradients. \n

*@par Attributes:
*@li shape: The shape of the values.
*@li dtype: The type of the value being accumulated.
*@li container: If non-empty, this accumulator is placed in the given
container. Otherwise, a default container is used.
*@li shared_name: If non-empty, this accumulator will be shared under the
given name across multiple sessions.
*@li reduction_type: The reduction method whose type is string,
default is "MEAN". \n

*@par Outputs:
*handle: The handle to the accumulator. \n

*@par Third-party framework compatibility
*Compatible with tensorflow SparseConditionalAccumulator operator. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(SparseConditionalAccumulator)
    .OUTPUT(handle, TensorType({DT_STRING_REF}))
    .REQUIRED_ATTR(shape, ListInt)
    .REQUIRED_ATTR(dtype, Type)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .ATTR(reduction_type, String, "MEAN")
    .OP_END_FACTORY_REG(SparseConditionalAccumulator)

/**
*@brief Applies a sparse gradient to a given accumulator. \n

*@par Inputs:
*The input handle must be type string_ref. Inputs include:
*@li handle: A Tensor of type mutable string. The handle to a accumulator.
*@li local_step: A Tensor of type int64. The local_step value at which the
sparse gradient was computed.
*@li indices: A Tensor of type int64. Indices of the sparse gradient to be
accumulated. Must be a vector.
*@li values: A Tensor. Values are the non-zero slices of the gradient,
and must have the same first dimension as indices, i.e., the nnz represented
by indices and values must be consistent.
*@li shape: A Tensor of type int64. \n

*@par Attributes:
*@li has_known_shape: A bool. Boolean indicating whether gradient_shape is
unknown, in which case the input is ignored during validation.
*@li dtype: The data type of accumulated gradients. Needs to correspond to
the type of the accumulator. \n

*@par Third-party framework compatibility
*Compatible with tensorflow SparseAccumulatorApplyGradient operator. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(SparseAccumulatorApplyGradient)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .INPUT(local_step, TensorType({DT_INT64}))
    .INPUT(indices, TensorType({DT_INT64}))
    .INPUT(values, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_UINT32, \
        DT_UINT64, DT_COMPLEX64, DT_COMPLEX128, DT_QINT16, DT_QUINT16, \
        DT_QINT8, DT_QUINT8, DT_QINT32}))
    .INPUT(shape, TensorType({DT_INT64}))
    .REQUIRED_ATTR(has_known_shape, Bool)
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(SparseAccumulatorApplyGradient)

/**
*@brief Extracts the average sparse gradient in a SparseConditionalAccumulator. \n

*@par Inputs:
*The input handle must be type string_ref. Inputs include:
*@li handle: The handle to a SparseConditionalAccumulator.
*@li num_required: Number of gradients required before we return an aggregate. \n

*@par Attributes:
*dtype: The data type of accumulated gradients. Needs to correspond to the
type of the accumulator. \n

*@par Outputs:
*@li indices: Indices of the average of the accumulated sparse gradients.
*@li values: Values of the average of the accumulated sparse gradients.
*@li shape: Shape of the average of the accumulated sparse gradients. \n

*@par Third-party framework compatibility
*Compatible with tensorflow SparseAccumulatorTakeGradient operator. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(SparseAccumulatorTakeGradient)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .INPUT(num_required, TensorType({DT_INT32}))
    .OUTPUT(indices, TensorType({DT_INT64}))
    .OUTPUT(values, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_DOUBLE, DT_FLOAT}))
    .OUTPUT(shape, TensorType({DT_INT64}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(SparseAccumulatorTakeGradient)

/**
*@brief A conditional accumulator for aggregating gradients. \n

*@par Attributes:
* @li dtype: The type of the value being accumulated.
* @li shape: The shape of the values, can be [], in which case shape is unknown.
* @li container: If non-empty, this accumulator is placed in the given container.
Otherwise, a default container is used.
* @li shared_name: If non-empty, this accumulator will be shared under the given
name across multiple sessions.
* @li reduction_type: reduction operator type, default "MEAN". \n

*@par Outputs:
*handle: A Tensor of type DT_RESOURCE. The handle to the accumulator. \n

*@attention Constraints:
*ResourceConditionalAccumulator runs on the Ascend AI CPU, which delivers poor performance. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ResourceConditionalAccumulator. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(ResourceConditionalAccumulator)
    .OUTPUT(handle, TensorType({DT_RESOURCE}))
    .REQUIRED_ATTR(dtype, Type)
    .REQUIRED_ATTR(shape, ListInt)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .ATTR(reduction_type, String, "MEAN")
    .OP_END_FACTORY_REG(ResourceConditionalAccumulator)

/**
*@brief Applies a gradient to a given accumulator.
Does not add if "local_step" is lesser than the accumulator's "global_step". \n

*@par Inputs:
* @li handle: The handle to an accumulator.
* @li local_step: The "local_step" value at which the gradient was computed.
* @li gradient: A tensor of the gradient to be accumulated.
Must be one of the following types:
DT_FLOAT16, DT_FLOAT, DT_DOUBLE

*@attention Constraints:
*ResourceAccumulatorApplyGradient runs on the Ascend AI CPU, which delivers poor performance. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ResourceAccumulatorApplyGradient. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(ResourceAccumulatorApplyGradient)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(local_step, TensorType({DT_INT64}))
    .INPUT(gradient, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(ResourceAccumulatorApplyGradient)

/**
*@brief Returns the number of gradients aggregated in the given accumulators. \n

*@par Inputs:
*handle: The handle to an accumulator. \n

*@par Outputs:
*num_accumulated: The number of gradients aggregated in the given accumulator. \n

*@attention Constraints:
*ResourceAccumulatorNumAccumulated runs on the Ascend AI CPU, which delivers poor performance. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ResourceAccumulatorNumAccumulated. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(ResourceAccumulatorNumAccumulated)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .OUTPUT(num_accumulated, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(ResourceAccumulatorNumAccumulated)

/**
*@brief Updates the accumulator with a new value for "global_step". \n

*@par Inputs:
* @li handle: The handle to an accumulator.
* @li new_global_step: The new "global_step" value to set. \n

*@attention Constraints:
*ResourceAccumulatorSetGlobalStep runs on the Ascend AI CPU, which delivers poor performance. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ResourceAccumulatorSetGlobalStep. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(ResourceAccumulatorSetGlobalStep)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(new_global_step, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(ResourceAccumulatorSetGlobalStep)

/**
*@brief Extracts the average gradient in the given ConditionalAccumulator. \n

*@par Inputs:
* @li handle: The handle to an accumulator.
* @li num_required: Number of gradients required before an aggregate is returned. \n

*@par Attributes:
*dtype: The data type of accumulated gradients.
Needs to correspond to the type of the accumulator. \n

*@par Outputs:
*average: The average of the accumulated gradients.
Must be one of the following types:
DT_FLOAT16, DT_FLOAT, DT_DOUBLE. \n

*@attention Constraints:
*ResourceAccumulatorTakeGradient runs on the Ascend AI CPU, which delivers poor performance. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ResourceAccumulatorTakeGradient. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(ResourceAccumulatorTakeGradient)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(num_required, TensorType({DT_INT32}))
    .OUTPUT(average, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(ResourceAccumulatorTakeGradient)

/**
*@brief Enqueue a Tensor on the computation outfeed. \n

*@par Inputs:
*Inputs include:
*x: A Tensor. Must be one of the following types: float16, float32,
float64, int8, int16, uint16, uint8, int32, int64, uint32, uint64,
bool, double, string. It's a dynamic input. \n

*@par Attributes:
*channel_name: name of operator channel, default "". \n

*@attention Constraints:
*The implementation for OutfeedEnqueueOp on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with tensorflow OutfeedEnqueueOp operator.
*/
REG_OP(OutfeedEnqueueOp)
  .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8,
      DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
      DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING}))
  .ATTR(channel_name, String, "")
  .OP_END_FACTORY_REG(OutfeedEnqueueOp)

/**
*@brief Enqueue a Tensor on the computation outfeed. \n

*@par Inputs:
*Inputs include:
*x: A Tensor. Must be one of the following types: float16, float32,
float64, int8, int16, uint16, uint8, int32, int64, uint32, uint64,
bool, double, string. It's a dynamic input. \n
*tensor_name: A Tensor. Must be string types. \n

*@par Attributes:
*channel_name: name of operator channel, default "". \n

*@attention Constraints:
*The implementation for OutfeedEnqueueOpV2 on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with tensorflow OutfeedEnqueueOpV2 operator.
*/
REG_OP(OutfeedEnqueueOpV2)
  .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8,
      DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
      DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING}))
  .INPUT(tensor_name, TensorType({DT_STRING}))
  .ATTR(channel_name, String, "")
  .OP_END_FACTORY_REG(OutfeedEnqueueOpV2)

/**
*@brief LruCache, create cache resource.
*@par Inputs:
*No input.
*@par Attributes:
*cache_size: cache size An optional "int64". Defaults to "100000".
*load_factor: rate which show if cache is full An optional "float", Defaults to "1".
*@par Outputs:
*cache: cache resource.
*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(LruCache)
  .OUTPUT(cache, TensorType({DT_RESOURCE}))
  .ATTR(container, String, "")
  .ATTR(shared_name, String, "LruCache")
  .ATTR(cache_size, Int, 100000)
  .ATTR(load_factor, Float, 1)
  .REQUIRED_ATTR(dtype, Type)
  .OP_END_FACTORY_REG(LruCache)

/**
*@brief CacheAdd, get id new come in cache and id get out of cache.
*@par Inputs:
*cache: resource data
*ids: Tensor stored id need to insert cache
*@par Outputs:
*swap_in_id: id come in cache.
*swap_in_idx: id in cache which come in cache
*swap_out_id: id get out of cache
*swap_out_idx: id in cache which get out of cache
*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(CacheAdd)
  .INPUT(cache, TensorType({DT_RESOURCE}))
  .INPUT(ids, TensorType({DT_INT64, DT_INT32, DT_UINT64, DT_UINT32}))
  .OUTPUT(swap_in_id, TensorType({DT_INT64, DT_INT32, DT_UINT64, DT_UINT32}))
  .OUTPUT(swap_in_idx, TensorType({DT_INT64, DT_INT32, DT_UINT64, DT_UINT32}))
  .OUTPUT(swap_out_id, TensorType({DT_INT64, DT_INT32, DT_UINT64, DT_UINT32}))
  .OUTPUT(swap_out_idx, TensorType({DT_INT64, DT_INT32, DT_UINT64, DT_UINT32}))
  .OP_END_FACTORY_REG(CacheAdd)

/**
*@brief CacheRemoteToLocalIndex, get id in cache from id.
*@par Inputs:
*cache: resource data
*ids: Tensor stored id need to insert cache
*@par Outputs:
*local_idx: id in cache.
*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(CacheRemoteIndexToLocal)
  .INPUT(cache, TensorType({DT_RESOURCE}))
  .INPUT(ids, TensorType({DT_INT64, DT_INT32, DT_UINT64, DT_UINT32}))
  .OUTPUT(local_idx, TensorType({DT_INT64, DT_INT32, DT_UINT64, DT_UINT32}))
  .OP_END_FACTORY_REG(CacheRemoteIndexToLocal)

/**
*@brief CacheAllToLocalIndex, get id in cache
*@par Inputs:
*cache: resource data
*local_idx: id in cache.
*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(CacheAllIndexToLocal)
  .INPUT(cache, TensorType({DT_RESOURCE}))
  .OUTPUT(local_idx, TensorType({DT_INT64, DT_INT32, DT_UINT64, DT_UINT32}))
  .REQUIRED_ATTR(dtype, Type)
  .OP_END_FACTORY_REG(CacheAllIndexToLocal)

/**
*@brief LRUCacheV2, aicore LRUCache.

*@par Inputs:
*index_list: exchange index list
*data: host data
*cache: gm cache
*tag: cache's tag
*is_last_call: if is last call write all cache to data

*@par Outputs:
*data: output data
*cache: gm cache
*tag: cache's tag
*index_offset_list: index_offset_list
*not_in_cache_index_list: output not in cache's index_list
*not_in_cache_number: scalar

*@par Attributes:
*pre_route_count: types of all outputs

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(LRUCacheV2)
    .INPUT(index_list, TensorType::BasicType())
    .INPUT(data, TensorType::BasicType())
    .INPUT(cache, TensorType::BasicType())
    .INPUT(tag, TensorType::BasicType())
    .INPUT(is_last_call, TensorType::BasicType())
    .OUTPUT(data, TensorType::BasicType())
    .OUTPUT(cache, TensorType::BasicType())
    .OUTPUT(tag, TensorType::BasicType())
    .OUTPUT(index_offset_list, TensorType::BasicType())
    .OUTPUT(not_in_cache_index_list, TensorType::BasicType())
    .OUTPUT(not_in_cache_number, TensorType::BasicType())
    .REQUIRED_ATTR(pre_route_count, Int)
    .OP_END_FACTORY_REG(LRUCacheV2)

/**
*@brief DynamicGetNext, dynamic get next data
*@par Inputs:
*x: the iterator, all types are available
*@par Outputs:
*y: the date in iterator, all types are available
*@par Attributes:
*output_types: types of all outputs
*output_shapes: shapes of all outputs
*_dynamic_graph_execute_mode: dynamic graph execution mode,
value is one of lazy_recompile and dynamic_execute
*_getnext_inputs_shape_range: shape ranges of outputs,
it works where _dynamic_graph_execute_mode is dynamic_execute
*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(DynamicGetNext)
  .INPUT(x, TensorType::ALL())
  .DYNAMIC_OUTPUT(y, TensorType::ALL())
  .ATTR(output_types, ListType, {})
  .ATTR(output_shapes, ListListInt, {{}, {}})
  .ATTR(_dynamic_graph_execute_mode, String, "lazy_recompile")
  .ATTR(_getnext_inputs_shape_range, String, "")
  .OP_END_FACTORY_REG(DynamicGetNext)

/**
@brief DynamicGetNextV2, dynamic get next data
* @par Inputs:
*x: the iterator, all types are available
* @par Outputs:
* y: the date in iterator, all types are available
* @par Attributes:
* output_types: types of all outputs
* output_shapes: shapes of all outputs
*_dynamic_graph_execute_mode: dynamic graph execution mode,
value is one of lazy_recompile and dynamic_execute
*_getnext_inputs_shape_range: shape ranges of outputs,
it works where _dynamic_graph_execute_mode is dynamic_execute
*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/

REG_OP(DynamicGetNextV2)
  .DYNAMIC_OUTPUT(y, TensorType::ALL())
  .ATTR(output_types, ListType, {})
  .ATTR(channel_name, String, "")
  .ATTR(output_shapes, ListListInt, {{}, {}})
  .ATTR(_dynamic_graph_execute_mode, String, "lazy_recompile")
  .ATTR(_getnext_inputs_shape_range, String, "")
  .OP_END_FACTORY_REG(DynamicGetNextV2)

/**
*@brief AdpGetNext
*@par Outputs:
*y: the data in iterator, all types are available
*@par Attributes:
*output_types: types of all outputs
*output_shapes: shapes of all outputs
*queue_name: cdqm queue name
*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(AdpGetNext)
  .DYNAMIC_OUTPUT(y, TensorType::ALL())
  .ATTR(output_types, ListType, {})
  .ATTR(output_shapes, ListListInt, {{}, {}})
  .ATTR(queue_name, String, "")
  .OP_END_FACTORY_REG(AdpGetNext)

/**
*@brief GetNextV2
*@par Outputs:
*y: the data in iterator, all types are available
*@par Attributes:
*output_types: types of all outputs
*output_shapes: shapes of all outputs
*queue_name: cdqm queue name
*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(GetNextV2)
  .DYNAMIC_OUTPUT(y, TensorType::ALL())
  .ATTR(output_types, ListType, {})
  .ATTR(output_shapes, ListListInt, {{}, {}})
  .ATTR(channel_name, String, "")
  .OP_END_FACTORY_REG(GetNextV2)

/**
*@brief GetNextFromQueue
*@par Inputs:
*x: the data, only support uint8
*@par Outputs:
*y: the data in iterator, all types are available
*@par Attributes:
*output_types: types of all outputs
*output_shapes: shapes of all outputs
*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(GetNextFromQueue)
  .INPUT(x, TensorType({DT_UINT8}))
  .DYNAMIC_OUTPUT(y, TensorType::ALL())
  .ATTR(output_types, ListType, {})
  .ATTR(output_shapes, ListListInt, {{}, {}})
  .OP_END_FACTORY_REG(GetNextFromQueue)

/**
*@brief Get the batch of data in data processing . \n

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

REG_OP(PeekData)
    .DYNAMIC_OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64,
                                   DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL}))
    .ATTR(output_types, ListType, {})
    .ATTR(output_shapes, ListListInt, {})
    .ATTR(channel_name, String, "")
    .OP_END_FACTORY_REG(PeekData)

/**
* @brief OptionalGetValue
* @par Inputs:
* optional: A tensor of type variant
* @par Outputs:
* components: A list of Tensor objects of output_types
* @par Attributes:
* output_types: types of all outputs
* output_shapes: shapes of all outputs
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(OptionalGetValue)
  .INPUT(optional, TensorType({DT_VARIANT}))
  .DYNAMIC_OUTPUT(components, TensorType::BasicType())
  .REQUIRED_ATTR(output_types, ListType)
  .REQUIRED_ATTR(output_shapes, ListListInt)
  .OP_END_FACTORY_REG(OptionalGetValue)

/**
* @brief User define function process. \n

* @par Inputs:
* @li x: A list of input tensor objects. It's a dynamic input. \n

* @par Outputs:
* @li y: A list of output tensor objects. It's a dynamic output. \n

* @par Attributes:
* @li bin_path: User's binary path.
* @li func_name: User defined function name.
* @li output_types: Types of outputs data.
* @li output_shapes: Shapes of outputs data.
* @li _flow_attr_process_node_engine_id: Default process node engine of FlowFunc.
*/
REG_OP(FlowFunc)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, \
        DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, \
        DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .REQUIRED_ATTR(bin_path, String)
    .REQUIRED_ATTR(func_name, String)
    .ATTR(output_shapes, ListListInt, {})
    .REQUIRED_ATTR(output_types, ListType)
    .OP_END_FACTORY_REG(FlowFunc)
} // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_DATA_FLOW_OPS_H_
