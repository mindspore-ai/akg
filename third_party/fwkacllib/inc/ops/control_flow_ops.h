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

#ifndef GE_CONTROL_FLOW_OPS_H_
#define GE_CONTROL_FLOW_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
 *@brief Forwards the value of an available tensor from input "x" to output "y". \n
 *       Merge waits for at least one of the input tensors to become available. \n
 *       It is usually combined with Switch to implement branching. \n
 *       Merge forwards the first tensor to become available to output "y", \n
 *       and sets "value_index" the index of the tensor in inputs.

 *@par Inputs:
 *x: The input tensors, one of which will become available. \n
 *   Must be one of the following types: float16, float32, float64, int8, \n
 *   int16, int32, int64, uint8, uint16, uint32, uint64, bool.

 *@par Outputs:
 *@li y: The available tensor. Has the same type as "x".
 *@li value_index: A scalar of type int32, for the index of the chosen input \n
 *                 tensor.

 *@see Switch()

 */
REG_OP(Merge)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OUTPUT(value_index, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(Merge)

/**
 *@brief Forwards the value of an available tensor from input "x" to output "y". \n
 *       Merge waits for at least one of the input tensors to become available. \n
 *       It is usually combined with Switch to implement branching. \n
 *       Merge forwards the first tensor to become available to output "y", \n
 *       and sets "value_index" the index of the tensor in inputs.

 *@par Inputs:
 *x: The input tensors, one of which will become available. \n
 *   Must be one of the following types: float16, float32, float64, int8, \n
 *   int16, int32, int64, uint8, uint16, uint32, uint64, bool.

 *@par Outputs:
 *@li y: The available tensor. Has the same type as "x".
 *@li value_index: A scalar of type int32, for the index of the chosen input \n
 *                 tensor.

 *@see Switch() | Merge()

 */
REG_OP(RefMerge)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OUTPUT(value_index, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(RefMerge)

/**
 *@brief Forwards "data" to the output port determined by "pred". \n
 *       If "pred" is "true", the data input is forwarded to "output_true". \n
 *       Otherwise, the data is forwarded to "output_false".

 *@par Inputs:
 *@li data: The tensor to be forwarded. \ n
 *          Must be one of the following types: float16, float32, float64, \n
 *          int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool.
 *@li pred: A boolean scalar. The output port that will receive data.

 *@par Outputs:
 *@li output_false: If "pred" is "false", data will be forwarded to this output. \n
 *                  Has the same type as "data".
 *@li output_true: If "pred" is "true", data will be forwarded to this output. \n
 *                 Has the same type as "data".

 *@see Merge()

 */
REG_OP(Switch)
    .INPUT(data, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .INPUT(pred, TensorType({DT_BOOL}))
    .OUTPUT(output_false, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OUTPUT(output_true, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OP_END_FACTORY_REG(Switch)

/**
 *@brief Forwards "data" to the output port determined by "pred". \n
 *       If "pred" is "true", the data input is forwarded to "output_true". \n
 *       Otherwise, the data is forwarded to "output_false".

 *@par Inputs:
 *@li data: The ref tensor to be forwarded. \n
 *          Must be one of the following types: float16, float32, float64, \n
 *          int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool.
 *@li pred: A boolean scalar. The output port that will receive data.

 *@par Outputs:
 *@li output_false: If "pred" is "false", data will be forwarded to this output. \n
 *                  Has the same type as "data".
 *@li output_true: If "pred" is "true", data will be forwarded to this output. \n
 *                 Has the same type as "data".

 *@see Merge() | Switch()

 */
REG_OP(RefSwitch)
    .INPUT(data, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .INPUT(pred, TensorType({DT_BOOL}))
    .OUTPUT(output_false, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OUTPUT(output_true, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OP_END_FACTORY_REG(RefSwitch)

REG_OP(SwitchN)
    .INPUT(data, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .INPUT(pred_value, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(output, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OP_END_FACTORY_REG(SwitchN)


/**
 *@brief Creates or finds a child frame, and makes "x" available to the child \n
 *       frame. This op is used together with Exit to create loops in the graph. \n
 *       The Executor uses the unique "frame_name" to identify frames. \n
 *       If "is_constant" is "true", output "y" is a constant in the child \n
 *       frame; otherwise it may be changed in the child frame.

 *@par Inputs:
 *x: The tensor to be made available to the child frame. \n
 *   Must be one of the following types: float16, float32, float64, int8, \n
 *   int16, int32, int64, uint8, uint16, uint32, uint64, bool.

 *@par Attributes:
 *@li frame_name: A required string. The name of the child frame.
 *@li is_constant: A required bool. If true, the output is constant in \n
 *                 the child frame.

 *@par Outputs:
 *y: A Tensor. Has the same type as "x".

 *@see Exit()

 */
REG_OP(Enter)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .REQUIRED_ATTR(frame_name, String)
    .REQUIRED_ATTR(is_constant, Bool)
    .OP_END_FACTORY_REG(Enter)

/**
 *@brief Creates or finds a child frame, and makes "x" available to the child \n
 *       frame. This op is used together with Exit to create loops in the graph. \n
 *       The Executor uses the unique "frame_name" to identify frames. \n
 *       If "is_constant" is "true", output "y" is a constant in the child \n
 *       frame; otherwise it may be changed in the child frame.

 *@par Inputs:
 *x: The tensor to be made available to the child frame. \n
 *   Must be one of the following types: float16, float32, float64, int8, \n
 *   int16, int32, int64, uint8, uint16, uint32, uint64, bool.

 *@par Attributes:
 *@li frame_name: A required string. The name of the child frame.
 *@li is_constant: A required bool. If true, the output is constant in \n
 *                 the child frame.

 *@par Outputs:
 *y: A tensor. Has the same type as "x".

 *@see Exit() | Enter()

 */
REG_OP(RefEnter)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .REQUIRED_ATTR(frame_name, String)
    .REQUIRED_ATTR(is_constant, Bool)
    .OP_END_FACTORY_REG(RefEnter)

/**
 *@brief Forwards the input to the output. This op represents the loop \n
 *       termination condition.

 *@par Inputs:
 *x: A boolean scalar. The condition of the Switch op.

 *@par Outputs:
 *y: The tensor "x".

 *@see Switch()

 */
REG_OP(LoopCond)
    .INPUT(x, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(LoopCond)

/**
 *@brief Makes the input available to the next iteration.

 *@par Inputs:
 *x: The tensor to be made available to the next iteration. \n
 *   Must be one of the following types: float16, float32, float64, int8, \n
 *   int16, int32, int64, uint8, uint16, uint32, uint64, bool.

 *@par Outputs:
 *y: A Tensor. Has the same type as "x".

 */
REG_OP(NextIteration)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OP_END_FACTORY_REG(NextIteration)

/**
 *@brief Makes the input available to the next iteration.

 *@par Inputs:
 *x: The tensor to be made available to the next iteration. \n
 *   Must be one of the following types: float16, float32, float64, int8, \n
 *   int16, int32, int64, uint8, uint16, uint32, uint64, bool.

 *@par Outputs:
 *y: A tensor. Has the same type as "x".

 */
REG_OP(RefNextIteration)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OP_END_FACTORY_REG(RefNextIteration)

/**
 *@brief Exits the current frame to its parent frame.

 *@par Inputs:
 *x: The tensor to be made available to the parent frame. \n
 *   Must be one of the following types: float16, float32, float64, int8, \n
 *   int16, int32, int64, uint8, uint16, uint32, uint64, bool.

 *@par Outputs:
 *y: A Tensor. Has the same type as "x".

 *@see Enter()

 */
REG_OP(Exit)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OP_END_FACTORY_REG(Exit)

/**
 *@brief Exits the current frame to its parent frame.

 *@par Inputs:
 *x: The tensor to be made available to the parent frame. \n
 *   Must be one of the following types: float16, float32, float64, int8, \n
 *   int16, int32, int64, uint8, uint16, uint32, uint64, bool.

 *@par Outputs:
 *y: A tensor. Has the same type as "x".

 *@see Enter() | Exit()

 */
REG_OP(RefExit)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OP_END_FACTORY_REG(RefExit)

/**
 *@brief Only useful as a placeholder for control edges. \n
 *       It is similar to a no-op that always produces a live control output \n
 *       even when some control inputs are dead.

 */
REG_OP(ControlTrigger)
    .OP_END_FACTORY_REG(ControlTrigger)
}  // namespace ge

#endif  // GE_CONTROL_FLOW_OPS_H_
