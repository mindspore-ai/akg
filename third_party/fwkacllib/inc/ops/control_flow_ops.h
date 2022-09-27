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
 * \file control_flow_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_CONTROL_FLOW_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_CONTROL_FLOW_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
 *@brief Forwards the value of an available tensor from input "x" to output "y".
 *       Merge waits for at least one of the input tensors to become available.
 *       It is usually combined with Switch to implement branching.
 *       Merge forwards the first tensor to become available to output "y",
 *       and sets "value_index" the index of the tensor in inputs . \n

 *@par Inputs:
 *x: The input tensors, one of which will become available.
 *   Must be one of the following types: float16, float32, float64, int8,
 *   int16, int32, int64, uint8, uint16, uint32, uint64, bool . It's a dynamic input. \n

 *@par Outputs:
 *@li y: The available tensor. Has the same type as "x".
 *@li value_index: A scalar of type int32, for the index of the chosen input
 *                 tensor . \n

 *@see Switch()

 *@par Third-party framework compatibility
 *@Compatible with the TensorFlow operator Merge.
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
 *@brief Forwards the value of an available tensor from input "x" to output "y".
 *       Merge waits for at least one of the input tensors to become available.
 *       It is usually combined with Switch to implement branching.
 *       Merge forwards the first tensor to become available to output "y",
 *       and sets "value_index" the index of the tensor in inputs . \n

 *@par Inputs:
 *x: The input tensors, one of which will become available.
 *   Must be one of the following types: float16, float32, float64, int8,
 *   int16, int32, int64, uint8, uint16, uint32, uint64, bool . It's a dynamic input. \n

 *@par Outputs:
 *@li y: The available tensor. Has the same type as "x".
 *@li value_index: A scalar of type int32, for the index of the chosen input
 *                 tensor . \n

 *@see Switch() | Merge()

 *@par Third-party framework compatibility
 *@Compatible with the TensorFlow operator RefMerge.
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
 *@brief Forwards "data" to the output port determined by "pred".
 *       If "pred" is "true", the data input is forwarded to "output_true".
 *       Otherwise, the data is forwarded to "output_false" . \n

 *@par Inputs:
 *@li data: The tensor to be forwarded.
 *          Must be one of the following types: float16, float32, float64,
 *          int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool.
 *@li pred: A boolean scalar. The output port that will receive data . \n

 *@par Outputs:
 *@li output_false: If "pred" is "false", data will be forwarded to this output.
 *                  Has the same type as "data".
 *@li output_true: If "pred" is "true", data will be forwarded to this output.
 *                 Has the same type as "data" . \n

 *@see Merge()

 *@par Third-party framework compatibility
 *@Compatible with the TensorFlow operator Switch.
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
 *@brief Forwards "data" to the output port determined by "pred".
 *       If "pred" is "true", the data input is forwarded to "output_true".
 *       Otherwise, the data is forwarded to "output_false" . \n

 *@par Inputs:
 *@li data: The ref tensor to be forwarded.
 *          Must be one of the following types: float16, float32, float64,
 *          int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool.
 *@li pred: A boolean scalar. The output port that will receive data . \n

 *@par Outputs:
 *@li output_false: If "pred" is "false", data will be forwarded to this output.
 *                  Has the same type as "data".
 *@li output_true: If "pred" is "true", data will be forwarded to this output.
 *                 Has the same type as "data" . \n

 *@see Merge() | Switch()

 *@par Third-party framework compatibility
 *@Compatible with the TensorFlow operator RefSwitch.
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

/**
 *@brief Forwards "data" to the output port determined by "pred_value" . \n

 *@par Inputs:
 *@li data: The tensor to be forwarded. \ n
 *          Must be one of the following types: float16, float32, float64,
 *          int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool.
 *@li pred_value: A int64 tensor which determines the output port that will receive data . \n

 *@par Outputs:
 *output: The output tensors, one of which will become available.
 *        Has the same type as "data".
 */
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
 *@brief Creates or finds a child frame, and makes "x" available to the child
 *       frame. This op is used together with Exit to create loops in the graph.
 *       The Executor uses the unique "frame_name" to identify frames.
 *       If "is_constant" is "true", output "y" is a constant in the child
 *       frame; otherwise it may be changed in the child frame . \n

 *@par Inputs:
 *x: The tensor to be made available to the child frame.
 *   Must be one of the following types: float16, float32, float64, int8,
 *   int16, int32, int64, uint8, uint16, uint32, uint64, bool . \n

 *@par Attributes:
 *@li frame_name: A required string. The name of the child frame.
 *@li is_constant: A required bool. If true, the output is constant in
 *                 the child frame . \n

 *@par Outputs:
 *y: A Tensor. Has the same type as "x" . \n

 *@see Exit()

 *@par Third-party framework compatibility
 *@Compatible with the TensorFlow operator Enter.
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
 *@brief Creates or finds a child frame, and makes "x" available to the child
 *       frame. This op is used together with Exit to create loops in the graph.
 *       The Executor uses the unique "frame_name" to identify frames.
 *       If "is_constant" is "true", output "y" is a constant in the child
 *       frame; otherwise it may be changed in the child frame . \n

 *@par Inputs:
 *x: The tensor to be made available to the child frame.
 *   Must be one of the following types: float16, float32, float64, int8,
 *   int16, int32, int64, uint8, uint16, uint32, uint64, bool . \n

 *@par Attributes:
 *@li frame_name: A required string. The name of the child frame.
 *@li is_constant: A required bool. If true, the output is constant in
 *                 the child frame . \n

 *@par Outputs:
 *y: A tensor. Has the same type as "x" . \n

 *@see Exit() | Enter()

 *@par Third-party framework compatibility
 *@Compatible with the TensorFlow operator RefEnter.
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
 *@brief Forwards the input to the output. This op represents the loop
 *       termination condition . \n

 *@par Inputs:
 *x: A boolean scalar. The condition of the Switch op . \n

 *@par Outputs:
 *y: The tensor "x" . \n

 *@see Switch()

 *@par Third-party framework compatibility
 *@Compatible with the TensorFlow operator LoopCond.
 */
REG_OP(LoopCond)
    .INPUT(x, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(LoopCond)

/**
 *@brief Makes the input available to the next iteration . \n

 *@par Inputs:
 *x: The tensor to be made available to the next iteration.
 *   Must be one of the following types: float16, float32, float64, int8,
 *   int16, int32, int64, uint8, uint16, uint32, uint64, bool . \n

 *@par Outputs:
 *y: A Tensor. Has the same type as "x" . \n

 *@par Third-party framework compatibility
 *@Compatible with the TensorFlow operator NextIteration.
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
 *@brief Makes the input available to the next iteration . \n

 *@par Inputs:
 *x: The tensor to be made available to the next iteration.
 *   Must be one of the following types: float16, float32, float64, int8,
 *   int16, int32, int64, uint8, uint16, uint32, uint64, bool . \n

 *@par Outputs:
 *y: A tensor. Has the same type as "x" . \n

 *@par Third-party framework compatibility
 *@Compatible with the TensorFlow operator RefNextIteration.
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
 *@brief Exits the current frame to its parent frame . \n

 *@par Inputs:
 *x: The tensor to be made available to the parent frame.
 *   Must be one of the following types: float16, float32, float64, int8,
 *   int16, int32, int64, uint8, uint16, uint32, uint64, bool . \n

 *@par Outputs:
 *y: A Tensor. Has the same type as "x" . \n

 *@see Enter()

 *@par Third-party framework compatibility
 *@Compatible with the TensorFlow operator Exit.
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
 *@brief Exits the current frame to its parent frame . \n

 *@par Inputs:
 *x: The tensor to be made available to the parent frame.
 *   Must be one of the following types: float16, float32, float64, int8,
 *   int16, int32, int64, uint8, uint16, uint32, uint64, bool . \n

 *@par Outputs:
 *y: A tensor. Has the same type as "x" . \n

 *@see Enter() | Exit()

 *@par Third-party framework compatibility
 *@Compatible with the TensorFlow operator RefExit.
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
 *@brief Only useful as a placeholder for control edges.
 *       It is similar to a no-op that always produces a live control output
 *       even when some control inputs are dead . \n

 *@par Third-party framework compatibility
 *@Compatible with the TensorFlow operator ControlTrigger.
 */
REG_OP(ControlTrigger)
    .OP_END_FACTORY_REG(ControlTrigger)

/**
*@brief Returns index of shape in the map.

*@par Inputs:
* Three inputs, including:
*@li x: One dimensional tensor of type int32, specifying queried shape, max size is 128.
*@li data_seq: One dimensional tensor of type int32, specifying the mapped table is queried.
*@li level_index: One dimensional tensor of type int32, specifying secondary index. \n

*@par Outputs:
*@li y: A Tensor with shape [8], of type int32, specifying index of shape in the map.
*@par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe.
*/
REG_OP(MapIndex)
    .INPUT(x, TensorType({DT_INT32}))
    .INPUT(data_seq, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(level_index, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(MapIndex)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_CONTROL_FLOW_OPS_H_
