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

#ifndef GE_OP_DATA_FLOW_OPS_H_
#define GE_OP_DATA_FLOW_OPS_H_

#include <algorithm>
#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief This operation returns true if the queue is closed and false if \n
the queue is open.

*@par Inputs:
*The input handle must have the resource type. Inputs include: \n
*handle:A Tensor of type resource. The handle to a queue.

*@par Outputs:
*is_closed:A Tensor of type bool.

*/

REG_OP(QueueIsClosed)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .OUTPUT(is_closed, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(QueueIsClosed)

/**
*@brief Computes the number of elements in the given queue.

*@par Inputs:
*The input handle must have the resource type. Inputs include: \n
*handle:A Tensor of type mutable resource. The handle to a queue.

*@par Outputs:
*size:A Tensor of type int32.

*/

REG_OP(QueueSize)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .OUTPUT(size, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(QueueSize)

/**
*@brief A queue that produces elements in first-in first-out order.

*@par Attributes:
*@li component_types: A list of DType objects. The length of component_types \n
must equal the number of tensors in each queue element.
*@li shapes:(Optional.) A list of fully-defined TensorShape objects with the \n
same length as dtypes, or None.
*@li capacity:An integer. The upper bound on the number of elements that may \n
be stored in this queue.
*@li container: An optional string. Defaults to "". If non-empty, this queue \n
is placed in the given container. Otherwise, a default container is used.
*@li shared_name:(Optional.) If non-empty, this queue will be shared under \n
the given name across multiple sessions.

*@par Outputs:
*handle:A Tensor of type mutable resource. The handle to a queue.

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
*@brief Enqueues a tuple of one or more tensors in the given queue.

*@par Inputs:
*The input handle must have the resource type. Inputs include: \n
*@li handle:A Tensor of type mutable resource. The handle to a queue.
*@li components: A list of Tensor objects. One or more tensors from which \n
the enqueued tensors should be taken.

*@par Attributes:
*timeout_ms: An optional int. Defaults to -1. If the queue is full, this \n
operation will block for up to timeout_ms milliseconds. Note: This option \n
is not supported yet.

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
*@brief Enqueues zero or more tuples of one or more tensors in the given queue.

*@par Inputs:
*The input handle must have the resource type. Inputs include: \n
*@li handle:A Tensor of type mutable resource. The handle to a queue.
*@li components: A list of Tensor objects. One or more tensors from which \n
the enqueued tensors should be taken.

*@par Attributes:
*timeout_ms: An optional int. Defaults to -1. If the queue is full, this \n
operation will block for up to timeout_ms milliseconds. Note: This option \n
is not supported yet.

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
*@brief Dequeues n tuples of one or more tensors from the given queue.

*@par Inputs:
*The input handle must have the resource type. Inputs include: \n
*handle:A Tensor of type mutable resource. The handle to a queue.

*@par Attributes:
*@li timeout_ms: An optional int. Defaults to -1. If the queue is empty, this \n
operation will block for up to timeout_ms milliseconds. Note: This option is \n
not supported yet.
*@li component_types: A list of DTypes that has length >= 1. The type of each \n
component in a tuple.

*@par Outputs:
*components:A list of Tensor objects of type component_types.

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
*@brief Dequeues n tuples of one or more tensors from the given queue.

*@par Inputs:
*The input handle must have the resource type. Inputs include: \n
*@li handle:A Tensor of type mutable resource. The handle to a queue.
*@li n: A Tensor of type int32. The number of tuples to dequeue.

*@par Attributes:
*@li timeout_ms: An optional int. Defaults to -1. If the queue has fewer than \n
n elements, this operation will block for up to timeout_ms milliseconds. \n
Note: This option is not supported yet.
*@li component_types: A list of DTypes that has length >= 1. The type of each \n
component in a tuple.

*@par Outputs:
*components:A list of Tensor objects of type component_types.

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
*@brief Dequeues n tuples of one or more tensors from the given queue.

*@par Inputs:
*The input handle must have the resource type. Inputs include: \n
*@li handle:A Tensor of type mutable resource. The handle to a queue.
*@li n: A Tensor of type int32. The number of tuples to dequeue.

*@par Attributes:
*@li timeout_ms: An optional int. Defaults to -1. If the queue has fewer than \n
n elements, this operation will block for up to timeout_ms milliseconds. \n
Note: This option is not supported yet.
*@li component_types: A list of DTypes that has length >= 1. The type of each \n
component in a tuple.

*@par Outputs:
*components:A list of Tensor objects of type component_types.

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
*@brief Stage values similar to a lightweight Enqueue.

*@par Inputs:
*The input values must be a list of Tensor objects. Inputs include: \n
*values: A list of Tensor objects. A list of data types that inserted values \n
should adhere to.

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to 0. Maximum number of \n
elements in the Staging Area. If > 0, inserts on the container will block \n
when the capacity is reached.
*@li memory_limit: An optional int that is >= 0. Defaults to 0. The maximum \n
number of bytes allowed for Tensors in the Staging Area. If > 0, inserts will \n
block until sufficient space is available.
*@li container: An optional string. Defaults to "". If non-empty, this queue \n
is placed in the given container. Otherwise, a default container is used.
*@li shared_name: An optional string. Defaults to "". It is necessary to \n
match this name to the matching Unstage Op.

*@see Unstage

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
*@brief Op removes all elements in the underlying container.

*@par Attributes:
*@li capacity: A list of DTypes
*@li memory_limit: An optional int that is >= 0. Defaults to 0.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "".
*@li dtypes: A list of DTypes.

*@see Stage

*/

REG_OP(StageClear)
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .ATTR(dtypes, ListType, {})
    .OP_END_FACTORY_REG(StageClear)

/**
*@brief Op peeks at the values at the specified index. If the underlying \n
container does not contain sufficient elements this op will block until it does.

*@par Inputs:
*The input values must be type int32. Inputs include: \n
*values: A Tensor of type int32.

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to 0.
*@li memory_limit: An optional int that is >= 0. Defaults to 0.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "".
*@li dtypes: A list of DTypes that has length >= 1.

*@par Outputs:
*y:A list of Tensor objects of type dtypes.

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
*@brief Op returns the number of elements in the underlying container.

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to 0.
*@li memory_limit: An optional int that is >= 0. Defaults to 0.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "".
*@li dtypes: A list of DTypes that has length >= 1.

*@par Outputs:
*size:A Tensor of type int32.

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
*@brief Pop the element at the top of the stack.

*@par Inputs:
*The input handle must be type resource. Inputs include: \n
*handle: A Tensor of type resource. The handle to a stack.

*@par Attributes:
*elem_type: A DType. The type of the elem that is popped.

*@par Outputs:
*element:A Tensor of type elem_type.

*/

REG_OP(StackPop)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .OUTPUT(element, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, \
                     DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, \
                     DT_DOUBLE, DT_UINT32, DT_UINT64}))
    .REQUIRED_ATTR(elem_type, Type)
    .OP_END_FACTORY_REG(StackPop)

/**
*@brief Push an element onto the stack.

*@par Inputs:
*The input handle must be type resource. Inputs include: \n
*@li handle: A Tensor of type resource. The handle to a stack.
*@li elem: A Tensor. The tensor to be pushed onto the stack.

*@par Attributes:
*swap_memory: An optional bool. Defaults to False. Swap elem to CPU. Default \n
to false.

*@par Outputs:
*y:A Tensor. Has the same type as elem.

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
*@brief Close the stack.

*@par Inputs:
*The input handle must be type resource. Inputs include: \n
*handle: A Tensor of type resource. The handle to a stack.

*/

REG_OP(StackClose)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .OP_END_FACTORY_REG(StackClose)

/**
*@brief Create a stack.

*@par Inputs:
*The input max_size must be type int32. Inputs include: \n
*max_size: A Tensor of type int32. The number of elements of a stack.

*@par Attributes:
*@li stack_name: An optional string. Defaults to "".
*@li elem_type: The elements type of the created Stack.

*@par Outputs:
*handle: A Tensor of type resource. The handle to a stack.

*/

REG_OP(Stack)
    .INPUT(max_size, TensorType({DT_INT32}))
    .OUTPUT(handle, TensorType({DT_RESOURCE}))
    .ATTR(stack_name, String, "")
    .REQUIRED_ATTR(elem_type, Type)
    .OP_END_FACTORY_REG(Stack)

/**
*@brief Partitions "x" into "num_partitions" tensors using indices from "partitions".

*@par Inputs:
*Including: \n
* @li x: The Tensor to be sliced. Must be one of the following types: \n
DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \n
DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, \
DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING.
* @li partitions: A Tensor of type DT_INT32, with any shape. The indices.

*@par Attributes:
*num_partitions: The number of partitions to output.

*@par Outputs:
*y: A list of tensors of type DT_INT32.

*@attention Constraints:\n
*DynamicPartition runs on the Ascend AI CPU, which delivers poor performance.\n

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
*@brief Interleaves the values from the "x" tensors into a single tensor.

*@par Inputs:
*Including: \n
* @li indices: A list of at least 1 Tensor objects with type DT_INT32.
* @li x: A list with the same length as "indices" of Tensor objects. \n
Must be one of the following types: DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \n
DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_QINT32, \n
DT_QUINT8, DT_QINT8, DT_STRING, DT_COMPLEX64, DT_COMPLEX128.

*@par Attributes:
*N: An int that is >= 1. Defaults to "1".

*@par Outputs:
*y: A Tensor. Has the same type as "x".

*@attention Constraints:\n
*DynamicStitch runs on the Ascend AI CPU, which delivers poor performance.\n

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
*@brief Interleaves the values from the "x" tensors into a single tensor.

*@par Inputs:
*Including: \n
* @li indices: A list of at least 1 Tensor objects with type DT_INT32.
* @li x: A list with the same length as "indices" of Tensor objects. \n
Must be one of the following types: DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \n
DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_STRING, \n
DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT32.

*@par Attributes:
*N: An int that is >= 1. Defaults to "1".

*@par Outputs:
*y: A Tensor. Has the same type as "x".

*@attention Constraints:\n
*ParallelDynamicStitch runs on the Ascend AI CPU, which delivers poor performance.\n

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
*@brief Removes all elements in the underlying container.

*@par Attributes:An optional int that is >= 0. Defaults to "0".
*@li capacity: An optional int that is >= 0. Defaults to "0".
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "".

*@attention Constraints:\n
*MapClear runs on the Ascend AI CPU, which delivers poor performance.\n

*/

REG_OP(MapClear)
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(MapClear)

/**
*@brief Returns the number of incomplete elements in the underlying container.

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to "0".
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "".

*@par Outputs:
*size: A Tensor of type DT_INT32.

*@attention Constraints:\n
*MapIncompleteSize runs on the Ascend AI CPU, which delivers poor performance.\n

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
*@brief Unstage Op is similar to a lightweight Dequeue.

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to 0.
*@li memory_limit: An optional int that is >= 0. Defaults to 0.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "".
*@li dtypes: A list of DTypes that has length >= 1.

*@par Outputs:
*y: A list of Tensor objects of type dtypes.

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
*@brief Stage (key, values) in the underlying container which behaves like a hashtable.

*@par Inputs:
*Including: \n
* @li key: A Tensor of type DT_INT64.
* @li indices: A Tensor of type DT_INT32.
* @li values: A list of Tensor objects for tensor dtypes. \n
A list of data types that inserted values should adhere to of. \n
Must be one of the following types: DT_FLOAT, DT_FLOAT16, \n
DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, \n
DT_INT64, DT_BOOL, DT_DOUBLE, DT_UINT32, DT_UINT64, \n
DT_RESOURCE, DT_STRING, DT_COMPLEX64, DT_COMPLEX128, \n
DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32.

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to "0". \n
Maximum number of elements in the Staging Area. If > 0, \n
inserts on the container will block when the capacity is reached.
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li container: An optional string. Defaults to "". \n
If non-empty, this queue is placed in the given container. \n
Otherwise, a default container is used.
*@li shared_name: An optional string. Defaults to "". \n
It is necessary to match this name to the matching Unstage Op.

*@attention Constraints:\n
*MapStage runs on the Ascend AI CPU, which delivers poor performance.\n

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
*@brief Removes and returns the values associated with the key.

*@par Inputs:
*Including: \n
* @li key: A Tensor of type DT_INT64.
* @li indices: A Tensor of type DT_INT32.

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to "0".
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li dtypes: A list of DTypes that has length >= 1.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "".

*@par Outputs:
*values: A list of Tensor objects. Must be one of the following types: \n
DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, \n
DT_INT64, DT_BOOL, DT_DOUBLE, DT_UINT32, DT_UINT64, DT_RESOURCE, \n
DT_STRING, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8, \n
DT_QINT16, DT_QUINT16, DT_QINT32.

*@attention Constraints:\n
*MapUnstage runs on the Ascend AI CPU, which delivers poor performance.\n

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
*@brief Removes and returns a random (key, value).

*@par Inputs:
*Including: \n
*indices: A Tensor of type DT_INT32.

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to "0".
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li dtypes: A list of DTypes that has length >= 1.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "".

*@par Outputs:
*@li key: A Tensor of type DT_INT64.
*@li values: A list of Tensor objects. \n
Must be one of the following types: DT_FLOAT, DT_FLOAT16, DT_INT8, \n
DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, \n
DT_UINT32, DT_UINT64, DT_RESOURCE, DT_STRING, DT_COMPLEX64, DT_COMPLEX128, \n
DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32.

*@attention Constraints:\n
*MapUnstageNoKey runs on the Ascend AI CPU, which delivers poor performance.\n

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
*@brief Peeks at the values at the specified key.

*@par Inputs:
*Including: \n
* @li key: A Tensor of type DT_INT64.
* @li indices: A Tensor of type DT_INT32.

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to "0".
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "".

*@par Outputs:
*values: A list of Tensor objects of type "dtypes". \n
Must be one of the following types: DT_FLOAT, DT_FLOAT16, DT_INT8, \n
DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, \n
DT_DOUBLE, DT_UINT32, DT_UINT64, DT_RESOURCE, DT_STRING, DT_COMPLEX64, \n
DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32.

*@attention Constraints:\n
*MapPeek runs on the Ascend AI CPU, which delivers poor performance.\n

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
*@brief Returns the number of elements in the underlying container.

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to "0".
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "".

*@par Outputs:
*size: A Tensor of type DT_INT32.

*@attention Constraints:\n
*MatMul runs on the Ascend AI CPU, which delivers poor performance.\n

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
*@brief Class wrapping dynamic-sized, per-time-step, write-once Tensor arrays.

*@par Inputs:
*The input size must be type int32. Inputs include: \n
*@li size: int32 scalar Tensor: the size of the TensorArray. Required if \n
handle is not provided.

*@par Attributes:
*@li dtype: The data type of this TensorArray.
*@li element_shape: The TensorShape of elements in this TensorArray.
*@li dynamic_size: A boolean that determines whether writes to the \n
TensorArray are allowed to grow the size.
*@li clear_after_read: Boolean (optional, default: True). If True, clear \n
TensorArray values \n
after reading them. This disables read-many semantics, but allows early \n
release of memory.
*@li identical_element_shapes: If true (default is false), then all elements \n
in the TensorArray will be expected to have have identical shapes.
*@li tensor_array_name: String: the name of the TensorArray.

*@par Outputs:
*@li handle: The handle to the TensorArray.
*@li flow: A scalar used to control gradient flow.

*/

REG_OP(TensorArray)
    .INPUT(size, TensorType({DT_INT32}))
    .OUTPUT(handle, TensorType({DT_RESOURCE}))
    .OUTPUT(flow, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(dtype, Type)
    .ATTR(element_shape, ListInt, ge::UNKNOWN_SHAPE)
    .ATTR(dynamic_size, Bool, false)
    .ATTR(clear_after_read, Bool, true)
    .ATTR(identical_element_shapes, Bool, false)
    .ATTR(tensor_array_name, String, "")
    .OP_END_FACTORY_REG(TensorArray)

/**
*@brief Delete the TensorArray from its resource container.

*@par Inputs:
*The input handle must be type resource. Inputs include: \n
*handle: A Tensor of type resource. The handle to a TensorArray \n
(output of TensorArray or TensorArrayGrad).

*/

REG_OP(TensorArrayClose)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .OP_END_FACTORY_REG(TensorArrayClose)

/**
*@brief Concat the elements from the TensorArray into value value.

*@par Inputs:
*The input handle must be type resource. Inputs include: \n
*@li handle: The handle to a TensorArray.
*@li flow_in: A float scalar that enforces proper chaining of operations.

*@par Attributes:
*@li dtype: The type of the elem that is returned.
*@li element_shape_except0: The expected shape of an element, if known, \n
excluding the first dimension.

*@par Outputs:
*@li value: All of the elements in the TensorArray, concatenated along \n
the first axis.
*@li lengths: A vector of the row sizes of the original T elements in the \n
value output.

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
    .ATTR(element_shape_except0, ListInt, ge::UNKNOWN_SHAPE)
    .OP_END_FACTORY_REG(TensorArrayConcat)

/**
*@brief All elements selected by indices must have the same shape.

*@par Inputs:
*The input handle must be type resource. Inputs include: \n
*@li handle: The handle to a TensorArray.
*@li indices: The locations in the TensorArray from which to read tensor \n
elements.
*@li flow_in: A float scalar that enforces proper chaining of operations.

*@par Attributes:
*@li dtype: The type of the elem that is returned.
*@li element_shape: The expected shape of an element, if known. Used to \n
validate the shapes of TensorArray elements. If this shape is not fully \n
specified, gathering zero-size TensorArrays is an error.

*@par Outputs:
*value:  All of the elements in the TensorArray, concatenated along a new \n
axis (the new dimension 0).

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
    .ATTR(element_shape, ListInt, ge::UNKNOWN_SHAPE)
    .OP_END_FACTORY_REG(TensorArrayGather)

/**
*@brief Creates a TensorArray for storing the gradients of values in the \n
given handle.

*@par Inputs:
*The input handle must be type resource. Inputs include: \n
*@li handle: The handle to a TensorArray.
*@li flow_in: A float scalar that enforces proper chaining of operations.

*@par Attributes:
*source: The gradient source string, used to decide which gradient \n
TensorArray to return.

*@par Outputs:
*@li grad_handle: A Tensor of type resource.
*@li flow_out: A Tensor of type float.

*/

REG_OP(TensorArrayGrad)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(flow_in, TensorType({DT_FLOAT}))
    .OUTPUT(grad_handle, TensorType({DT_RESOURCE}))
    .OUTPUT(flow_out, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(source, String)
    .OP_END_FACTORY_REG(TensorArrayGrad)

/**
*@brief Push an element onto the tensor_array.

*@par Inputs:
*The input handle must be type resource. Inputs include: \n
*@li handle: The handle to a TensorArray.
*@li index: The position to write to inside the TensorArray.
*@li value: The tensor to write to the TensorArray.
*@li flow_in: A float scalar that enforces proper chaining of operations.

*@par Outputs:
*flow_out: A float scalar that enforces proper chaining of operations.

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
*@brief Creates a TensorArray for storing multiple gradients of values in \n
the given handle.

*@par Inputs:
*The input handle must be type resource. Inputs include: \n
*@li handle: A Tensor of type resource. The handle to the forward TensorArray.
*@li flow_in: A Tensor of type float. A float scalar that enforces proper \n
chaining of operations.
*@li shape_to_prepend: A Tensor of type int32. An int32 vector representing \n
a shape.

*@par Attributes:
*source: A string. The gradient source string, used to decide which gradient \n
TensorArray to return.

*@par Outputs:
*@li grad_handle: A Tensor of type resource.
*@li flow_out: A Tensor of type float.

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
*@brief Read an element from the TensorArray into output value.

*@par Inputs:
*The input handle must be type resource. Inputs include: \n
*@li handle: A Tensor of type resource. The handle to a TensorArray.
*@li index: A Tensor of type int32.
*@li flow_in: A Tensor of type float.

*@par Attributes:
*dtype: A DType.

*@par Outputs:
*y: A Tensor of type dtype.

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
*@brief Scatter the data from the input value into specific TensorArray \n
elements.

*@par Inputs:
*The input handle must be type resource. Inputs include: \n
*@li handle: The handle to a TensorArray.
*@li indices: The locations at which to write the tensor elements.
*@li value: The concatenated tensor to write to the TensorArray.
*@li flow_in: A float scalar that enforces proper chaining of operations.

*@par Outputs:
*flow_out: A float scalar that enforces proper chaining of operations.

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
*@brief Split the data from the input value into TensorArray elements.

*@par Inputs:
*The input handle must be type resource. Inputs include: \n
*@li handle: The handle to a TensorArray.
*@li value: The concatenated tensor to write to the TensorArray.
*@li lengths: The vector of lengths, how to split the rows of value into \n
the TensorArray.
*@li flow_in: A float scalar that enforces proper chaining of operations.

*@par Outputs:
*flow_out: A float scalar that enforces proper chaining of operations.

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
*@brief Return the number of elements in a TensorArray.

*@par Inputs:
*The input handle must be type resource. Inputs include: \n
*@li handle: The handle to a TensorArray.
*@li flow_in: A float scalar that enforces proper chaining of operations.

*@par Outputs:
*size: The number of elements in a TensorArray..

*/

REG_OP(TensorArraySize)
    .INPUT(handle, TensorType({ DT_RESOURCE }))
    .INPUT(flow_in, TensorType({ DT_FLOAT }))
    .OUTPUT(size, TensorType({ DT_INT32 }))
    .OP_END_FACTORY_REG(TensorArraySize)

/**
*@brief A queue implementation that dequeues elements in a random order.

*@par Attributes:
*@li shapes: (Optional.) A list of fully-defined TensorShape objects with \n
the same length as dtypes, or None.
*@li capacity: An integer. The upper bound on the number of elements that may \n
be stored in this queue.
*@li min_after_dequeue: An integer (described above).
*@li seed: An integer. Used to create a random seed.
*@li seed2: An integer. Used to create a random seed.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "".

*@par Outputs:
*handle: A Tensor of type resource. The handle to a stack.

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
*@brief A queue that produces elements in first-in first-out order.

*@par Attributes:
*@li shapes: An optional list of shapes for each component of \n
a queue element. Defaults to {}. The length of this attr must be \n
either 0 or the same as the length of "component_types". Shapes of fixed \n
rank but variable size are allowed by setting any shape dimension to "-1". \n
In this case, the inputs' shape may vary along the given dimension, \n
and DequeueMany will pad the given dimension with zeros up to the maximum \n
shape of all elements in the given batch. If the length of this attr is "0", \n
different queue elements may have different ranks and shapes, but only one \n
element may be dequeued at a time.
*@li capacity: An optional int. Defaults to "-1". The upper bound on the number \n
of elements in this queue. Negative numbers mean no limit.
*@li container: An optional string. Defaults to "". If non-empty, this queue \n
is placed in the given container. Otherwise, a default container is used.
*@li shared_name: An optional string. Defaults to "". If non-empty, this queue \n
will be shared under the given name across multiple sessions.

*@par Outputs:
*handle: A Tensor of type DT_RESOURCE.

*@attention Constraints:\n
*PaddingFIFOQueue runs on the Ascend AI CPU, which delivers poor performance.\n

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
*@brief A queue that produces elements sorted by the first component value.

*@par Attributes:
The type of each component in a value.
*@li shapes: A list of shapes for each component of a queue element.
The length of this attr must be either 0 or the same as the length of \n
"component_types". If the length of this attr is 0, the shapes of queue \n
elements are not constrained, and only one element may be dequeued at a time.
*@li container: An optional string. Defaults to "". If non-empty, this queue \n
is placed in the given container. Otherwise, a default container is used.
*@li shared_name: An optional string. Defaults to "". If non-empty, this \n
queue will be shared under the given name across multiple sessions.

*@par Outputs:
*handle: A Tensor of type DT_RESOURCE.

*@attention Constraints:\n
*PriorityQueue runs on the Ascend AI CPU, which delivers poor performance.\n

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
*@brief Multiplies the matrix "x1" by the matrix "x2".

*@par Inputs:
*Including: \n
*handle: A Tensor of type DT_RESOURCE. The handle to a queue.

*@par Attributes:
*cancel_pending_enqueues: An optional bool. Defaults to "False". \n
If true, all pending enqueue requests that are blocked on \n
the given queue will be canceled.

*@attention Constraints:\n
*QueueClose runs on the Ascend AI CPU, which delivers poor performance.\n

*/

REG_OP(QueueClose)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .ATTR(cancel_pending_enqueues, Bool, false)
    .OP_END_FACTORY_REG(QueueClose)

/**
*@brief Stage (key, values) in the underlying container which behaves like an ordered associative container.

*@par Inputs:
*Including: \n
* @li key: A Tensor of type DT_INT64.
* @li indices: A Tensor of type DT_INT32.
* @li values: A list of Must be one of the following types: \n
DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, \n
DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, DT_UINT32, DT_UINT64, \n
DT_RESOURCE, DT_STRING, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, \n
DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32 that inserted values should adhere to.

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to "0". \n
Maximum number of elements in the Staging Area. \n
If > 0, inserts on the container will block \n
when the capacity is reached.
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li dtypes: A list of DTypes.
*@li container: An optional string. Defaults to "". \n
If non-empty, this queue is placed in the given container. \n
Otherwise, a default container is used.
*@li shared_name: An optional string. Defaults to "". \n
It is necessary to match this name to the matching Unstage Op.

*@attention Constraints:\n
*OrderedMapStage runs on the Ascend AI CPU, which delivers poor performance.\n

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
*@brief Returns the number of elements in the underlying container.

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to "0".
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li dtypes: A list of DTypes.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "".

*@par Outputs:
*size: A Tensor of type DT_INT32.

*@attention Constraints:\n
*OrderedMapSize runs on the Ascend AI CPU, which delivers poor performance.\n

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
*@brief Removes all elements in the underlying container.

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to "0".
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li dtypes: A list of DTypes.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "".

*@attention Constraints:\n
*OrderedMapClear runs on the Ascend AI CPU, which delivers poor performance.\n

*/

REG_OP(OrderedMapClear)
    .ATTR(capacity, Int, 0)
    .ATTR(memory_limit, Int, 0)
    .ATTR(dtypes, ListType, {})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(OrderedMapClear)

/**
*@brief Returns the number of incomplete elements in the underlying container.

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to "0".
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li dtypes: A list of DTypes.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "".

*@par Outputs:
*size: A Tensor of type DT_INT32.

*@attention Constraints:\n
*OrderedMapIncompleteSize runs on the Ascend AI CPU, \n
which delivers poor performance.\n

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
*@brief Peeks at the values at the specified key.

*@par Inputs:
*Including: \n
* @li key: A Tensor of type DT_INT64.
* @li indices: A Tensor of type DT_INT32.

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to "0".
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li dtypes: A list of DTypes that has length >= 1.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "".

*@par Outputs:
*values: A list of Tensor objects. Must be one of the following types: \n
DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, \n
DT_INT64, DT_BOOL, DT_DOUBLE, DT_UINT32, DT_UINT64, DT_RESOURCE, DT_STRING, \n
DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32.

*@attention Constraints:\n
*OrderedMapPeek runs on the Ascend AI CPU, which delivers poor performance.\n

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
*@brief Removes and returns the (key, value) element with the smallest.

*@par Inputs:
*Including: \n
* @li indices: A Tensor of type DT_INT32.

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to "0".
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li dtypes: A list of DTypes that has length >= 1.
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "".

*@par Outputs:
*@li key: A Tensor of type DT_INT64.
*@li values: A list of Tensor objects. Must be one of the following types: \n
DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, \n
DT_INT64, DT_BOOL, DT_DOUBLE, DT_UINT32, DT_UINT64, DT_RESOURCE, DT_STRING, \n
DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32.

*@attention Constraints:\n
*OrderedMapUnstageNoKey runs on the Ascend AI CPU, \n
which delivers poor performance.\n

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
*@brief Removes and returns the values associated with the key.

*@par Inputs:
*Including: \n
* @li key: A Tensor of type DT_INT64.
* @li indices: A Tensor of type DT_INT32.

*@par Attributes:
*@li capacity: An optional int that is >= 0. Defaults to "0".
*@li memory_limit: An optional int that is >= 0. Defaults to "0".
*@li container: An optional string. Defaults to "".
*@li shared_name: An optional string. Defaults to "".

*@par Outputs:
*values: A list of Tensor objects. Must be one of the following types: \n
DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT, \n
DT_FLOAT16, DT_DOUBLE, DT_BOOL, DT_UINT32, DT_UINT64.

*@attention Constraints:\n
*OrderedMapUnstage runs on the Ascend AI CPU, which delivers poor performance.\n

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
*@brief A barrier represents a key-value map, where each key is a string, \n
and each value is a tuple of tensors.

*@par Attributes:
*@li component_types: The type of each component in a value.
*@li shapes: A list of shapes for each component of a queue element.
Each shape must be 1 in the first dimension. \n
The length of this attr must be the same as \n
the length of "component_types".
*@li capacity: The capacity of the barrier. \n
The default capacity is MAX_INT32, \n
which is the largest capacity of the underlying queue.
*@li container: If non-empty, this barrier is placed in the given container. \n
Otherwise, a default container is used.
*@li shared_name: If non-empty, this barrier will be shared under \n
the given name across multiple sessions.

*@par Outputs:
*handle: A Tensor of type DT_STRING_REF. The handle to the barrier.

*@attention Constraints:\n
*Barrier runs on the Ascend AI CPU, which delivers poor performance.\n

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
*@brief For each key, assigns the respective value to the specified component.

*@par Inputs:
*Including: \n
* @li handle: A Tensor of type DT_STRING_REF. The handle to a barrier.
* @li keys: A Tensor of type DT_STRING. A 1D tensor of keys.
* @li values: An any-dimensional tensor of values, which are associated \n
with the respective keys. The 0th dimension must have length n \n
Must be one of the following types: DT_FLOAT, DT_FLOAT16, DT_INT8, \n
DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, \n
DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128,  DT_RESOURCE, DT_STRING.

*@par Attributes:
*component_index: The component of the barrier elements that is being assigned.

*@attention Constraints:\n
*BarrierInsertMany runs on the Ascend AI CPU, which delivers poor performance.\n

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
*@brief Takes the given number of completed elements from a barrier.

*@par Inputs:
*Including: \n
* @li handle: A Tensor of type DT_STRING_REF. The handle to a barrier.
* @li num_elements: A Tensor of type DT_INT32. \n
A single-element tensor containing the number of elements to take.

*@par Attributes:
*@li component_types: The type of each component in a value.
*@li allow_small_batch: Allow to return less than "num_elements" \n
items if barrier is already closed.
*@li wait_for_incomplete: An any-dimensional tensor \n
for each component in the barrier element.
*@li timeout_ms: If the queue is empty, this operation will block for up to \n
"timeout_ms" milliseconds. Note: This option is not supported yet.

*@par Outputs:
*@li indices: A 1D tensor of type DT_INT64. The indices, with length "num_elems". \n
These indices refer to the batch in which the values were \n
placed into the barrier.
*@li keys: A 1D tensor of keys, \n
with length "num_elements" of type DT_STRING.
*@li values: A 1D tensor per component in a barrier element. \n
All values have length "num_elements" along the 0th dimension. \n
Must be one of the following types: \n
DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, \n
DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128, \n
DT_RESOURCE, DT_STRING.

*@attention Constraints:\n
*BarrierTakeMany runs on the Ascend AI CPU, which delivers poor performance.\n

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
*@brief Closes the given barrier.

*@par Inputs:
*Including: \n
*handle: A Tensor of type DT_STRING_REF. The handle to a barrier.

*@par Attributes:
*cancel_pending_enqueues: If true, all pending enqueue requests \n
that are blocked on the barrier's queue will \n
be canceled. InsertMany will fail, \n
even if no new key is introduced.

*@attention Constraints:\n
*BarrierClose runs on the Ascend AI CPU, which delivers poor performance.\n

*/

REG_OP(BarrierClose)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .ATTR(cancel_pending_enqueues, Bool, false)
    .OP_END_FACTORY_REG(BarrierClose)

/**
*@brief Computes the number of complete elements in the given barrier.

*@par Inputs:
*Including: \n
*handle: A Tensor of type DT_STRING_REF. The handle to a barrier.

*@par Outputs:
*size: A Tensor of type DT_INT32. The number of complete elements.

*@attention Constraints:\n
*BarrierReadySize runs on the Ascend AI CPU, which delivers poor performance.\n

*/

REG_OP(BarrierReadySize)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .OUTPUT(size, TensorType(DT_INT32))
    .OP_END_FACTORY_REG(BarrierReadySize)

/**
*@brief Computes the number of incomplete elements in the given barrier.

*@par Inputs:
*Including: \n
*handle: A Tensor of type DT_STRING_REF. The handle to a barrier.

*@par Outputs:
*size: A Tensor of type DT_INT32. The number of incomplete elements in the barrier.

*@attention Constraints:\n
*BarrierIncompleteSize runs on the Ascend AI CPU, which delivers poor performance.\n

*/

REG_OP(BarrierIncompleteSize)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .OUTPUT(size, TensorType(DT_INT32))
    .OP_END_FACTORY_REG(BarrierIncompleteSize)

/**
*@brief Emits randomized records.

*@par Attributes:
*@li file_pattern: A string. Glob pattern for the data files.
*@li file_random_seed: An optional int. Defaults to 301. Random seeds used to \n
produce randomized records.
*@li file_shuffle_shift_ratio: An optional float. Defaults to 0. Shifts the \n
list of files after the list is randomly shuffled.
*@li file_buffer_size: An optional int. Defaults to 10000. The randomization \n
shuffling buffer.
*@li file_parallelism: An optional int. Defaults to 16. How many sstables are \n
opened and concurrently iterated over.
*@li batch_size: An optional int. Defaults to 32. The batch size.
*@li compression_type: An optional string. Defaults to "". The type of \n
compression for the file. Currently ZLIB and GZIP are supported.

*@par Outputs:
*records: A Tensor of type string.

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
*@brief A conditional accumulator for aggregating gradients.

*@par Attributes:
*@li dtype: The type of the value being accumulated.
*@li shape: The shape of the values, can be [], in which case shape is unknown.
*@li container: If non-empty, this accumulator is placed in the given container. \n
Otherwise, a default container is used.
*@li shared_name: If non-empty, this accumulator will be shared under the given \n
name across multiple sessions.
*@li reduction_type: reduction operator type, default "MEAN".

*@par Outputs:
*handle: A Tensor of type DT_STRING_REF. The handle to the accumulator.

*@attention Constraints:\n
*ConditionalAccumulator runs on the Ascend AI CPU, which delivers poor performance.\n

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
*@brief Applies a gradient to a given accumulator.

*@par Inputs:
*Does not add if "local_step" is lesser than the accumulator's "global_step". \n
* @li handle: A Tensor of type DT_STRING_REF. The handle to an accumulator.
* @li local_step: A Tensor of type DT_INT64. \n
The "local_step" value at which the gradient was computed.

* @li gradient: A tensor of the gradient to be accumulated. \n
Must be one of the following types: \n
DT_FLOAT16, DT_FLOAT, DT_DOUBLE

*@par Attributes:
*dtype: Must be one of the following types: \n
DT_FLOAT16, DT_FLOAT, DT_DOUBLE

*@attention Constraints:\n
*AccumulatorApplyGradient runs on the Ascend AI CPU, \n
which delivers poor performance.\n

*/

REG_OP(AccumulatorApplyGradient)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .INPUT(local_step, TensorType({DT_INT64}))
    .INPUT(gradient, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(AccumulatorApplyGradient)

/**
*@brief Returns the number of gradients aggregated in the given accumulators.

*@par Inputs:
*Including: \n
*handle: A Tensor of type DT_STRING_REF. The handle to an accumulator.

*@par Outputs:
*y: A Tensor of type DT_INT32. The number of gradients aggregated \n
in the given accumulator.

*@attention Constraints:\n
*AccumulatorNumAccumulated runs on the Ascend AI CPU, \n
which delivers poor performance.\n

*/

REG_OP(AccumulatorNumAccumulated)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(AccumulatorNumAccumulated)

/**
*@brief Updates the accumulator with a new value for "global_step".

*@par Inputs:
*Input "new_global_step" is a scalar. \n
* @li handle: A Tensor of type DT_STRING_REF. The handle to an accumulator.
* @li new_global_step: The new "global_step" value to set A Tensor of type DT_INT64.

*@attention Constraints:\n
*AccumulatorSetGlobalStep runs on the Ascend AI CPU, which delivers poor performance.\n

*/

REG_OP(AccumulatorSetGlobalStep)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .INPUT(new_global_step, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(AccumulatorSetGlobalStep)

/**
*@brief Extracts the average gradient in the given ConditionalAccumulator.

*@par Inputs:
* Input "num_required" is a scalar. \n
* @li handle: A Tensor of type DT_STRING_REF. The handle to an accumulator.
* @li num_required: A Tensor of type DT_INT32. \n
Number of gradients required before an aggregate is returned.

*@par Attributes:
*dtype: The data type of accumulated gradients. \n
Needs to correspond to the type of the accumulator.

*@par Outputs:
*y: The average of the accumulated gradients. \n
Must be one of the following types:
DT_FLOAT16, DT_FLOAT, DT_DOUBLE.

*@attention Constraints:\n
*AccumulatorTakeGradient runs on the Ascend AI CPU,
\nwhich delivers poor performance.\n

*/

REG_OP(AccumulatorTakeGradient)
    .INPUT(handle, TensorType({DT_STRING_REF}))
    .INPUT(num_required, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(AccumulatorTakeGradient)

/**
*@brief A conditional accumulator for aggregating sparse gradients.

*@par Attributes:
*@li shape: The shape of the values.
*@li dtype: The type of the value being accumulated.
*@li container: If non-empty, this accumulator is placed in the given \n
container. Otherwise, a default container is used.
*@li shared_name: If non-empty, this accumulator will be shared under the \n
given name across multiple sessions.
*@li reduction_type: The reduction method whose type is string, \n
default is "MEAN".

*@par Outputs:
*handle: The handle to the accumulator.

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
*@brief Applies a sparse gradient to a given accumulator.

*@par Inputs:
*The input handle must be type string_ref. Inputs include: \n
*@li handle: A Tensor of type mutable string. The handle to a accumulator.
*@li local_step: A Tensor of type int64. The local_step value at which the \n
sparse gradient was computed.
*@li indices: A Tensor of type int64. Indices of the sparse gradient to be \n
accumulated. Must be a vector.
*@li values: A Tensor. Values are the non-zero slices of the gradient, \n
and must have the same first dimension as indices, i.e., the nnz represented \n
by indices and values must be consistent.
*@li shape: A Tensor of type int64.

*@par Attributes:
*@li has_known_shape: A bool. Boolean indicating whether gradient_shape is \n
unknown, in which case the input is ignored during validation.
*@li dtype: The data type of accumulated gradients. Needs to correspond to \n
the type of the accumulator.

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
*@brief Extracts the average sparse gradient in a SparseConditionalAccumulator.

*@par Inputs:
*The input handle must be type string_ref. Inputs include: \n
*@li handle: The handle to a SparseConditionalAccumulator.
*@li num_required: Number of gradients required before we return an aggregate.

*@par Attributes:
*dtype: The data type of accumulated gradients. Needs to correspond to the \n
type of the accumulator.

*@par Outputs:
*@li indices: Indices of the average of the accumulated sparse gradients.
*@li values: Values of the average of the accumulated sparse gradients.
*@li shape: Shape of the average of the accumulated sparse gradients.

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
*@brief A conditional accumulator for aggregating gradients.

*@par Attributes:
* @li dtype: The type of the value being accumulated.
* @li shape: The shape of the values, can be [], in which case shape is unknown.
* @li container: If non-empty, this accumulator is placed in the given container. \n
Otherwise, a default container is used.
* @li shared_name: If non-empty, this accumulator will be shared under the given \n
name across multiple sessions.
* @li reduction_type: reduction operator type, default "MEAN".

*@par Outputs:
*handle: A Tensor of type DT_RESOURCE. The handle to the accumulator.

*@attention Constraints:
*ResourceConditionalAccumulator runs on the Ascend AI CPU, which delivers poor performance.

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
*@brief Applies a gradient to a given accumulator. \n
Does not add if "local_step" is lesser than the accumulator's "global_step".

*@par Inputs:
* @li handle: The handle to an accumulator.
* @li local_step: The "local_step" value at which the gradient was computed.
* @li gradient: A tensor of the gradient to be accumulated. \n
Must be one of the following types: \n
DT_FLOAT16, DT_FLOAT, DT_DOUBLE

*@attention Constraints:
*ResourceAccumulatorApplyGradient runs on the Ascend AI CPU, which delivers poor performance.

*/

REG_OP(ResourceAccumulatorApplyGradient)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(local_step, TensorType({DT_INT64}))
    .INPUT(gradient, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(ResourceAccumulatorApplyGradient)

/**
*@brief Returns the number of gradients aggregated in the given accumulators.

*@par Inputs:
*handle: The handle to an accumulator.

*@par Outputs:
*num_accumulated: The number of gradients aggregated in the given accumulator.

*@attention Constraints:
*ResourceAccumulatorNumAccumulated runs on the Ascend AI CPU, which delivers poor performance.

*/

REG_OP(ResourceAccumulatorNumAccumulated)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .OUTPUT(num_accumulated, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(ResourceAccumulatorNumAccumulated)

/**
*@brief Updates the accumulator with a new value for "global_step".

*@par Inputs:
* @li handle: The handle to an accumulator.
* @li new_global_step: The new "global_step" value to set.

*@attention Constraints:
*ResourceAccumulatorSetGlobalStep runs on the Ascend AI CPU, which delivers poor performance.

*/

REG_OP(ResourceAccumulatorSetGlobalStep)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(new_global_step, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(ResourceAccumulatorSetGlobalStep)

/**
*@brief Extracts the average gradient in the given ConditionalAccumulator.

*@par Inputs:
* @li handle: The handle to an accumulator.
* @li num_required: Number of gradients required before an aggregate is returned.

*@par Attributes:
*dtype: The data type of accumulated gradients. \n
Needs to correspond to the type of the accumulator.

*@par Outputs:
*average: The average of the accumulated gradients. \n
Must be one of the following types: \n
DT_FLOAT16, DT_FLOAT, DT_DOUBLE.

*@attention Constraints:
*ResourceAccumulatorTakeGradient runs on the Ascend AI CPU, which delivers poor performance.

*/

REG_OP(ResourceAccumulatorTakeGradient)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(num_required, TensorType({DT_INT32}))
    .OUTPUT(average, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(ResourceAccumulatorTakeGradient)

/**
*@brief Enqueue a Tensor on the computation outfeed.

*@par Inputs:
*Inputs include: \n
*x: A Tensor. Must be one of the following types: float16, float32, \n
float64, int8, int16, uint16, uint8, int32, int64, uint32, uint64, \n
bool, double, string.

*@par Attributes:
*channel_name: name of operator channel, default "".

*@attention Constraints:\n
*-The implementation for OutfeedEnqueueOp on Ascend uses AICPU, with bad performance.\n

*/
REG_OP(OutfeedEnqueueOp)
  .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8,
      DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
      DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING}))
  .ATTR(channel_name, String, "")
  .OP_END_FACTORY_REG(OutfeedEnqueueOp)

}   // namespace ge

#endif  // GE_OP_DATA_FLOW_OPS_H_
