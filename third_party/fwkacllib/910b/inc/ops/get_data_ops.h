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
 * \file get_data_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_GET_DATA_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_GET_DATA_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Binding dataset and GetNext
*@par Attributes: None
*@par Inputs: Dataset and GetNext operator
*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(MakeIterator)
    .INPUT(x, TensorType::ALL())
    .INPUT(x1, TensorType::ALL())
    .ATTR(_kernel, String, "dp")
    .OP_END_FACTORY_REG(MakeIterator)

/**
*@brief Dataset iterator
*@par Attributes:
*output_types: Data type of output
*output_shapes: Shapes of output
*container: Iterator container name
*shared_name: Iterator id
*@par Inputs: None
*@par Outputs: Dataset
*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(IteratorV2)
    .OUTPUT(y, TensorType::ALL())
    .ATTR(output_types, ListInt, {})
    .ATTR(output_shapes,ListListInt, {{}, {}})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(IteratorV2)

/**
*@brief Dataset GetNext iterator
*@par Attributes:
*output_types: Data type of output
*output_shapes: Shapes of output
*output_num: Num of output
*@par Inputs: Queue data
*@par Outputs: Input of computer graph
*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(IteratorGetNext)
    .INPUT(x, TensorType::ALL())
    .DYNAMIC_OUTPUT(y, TensorType::ALL())
    .ATTR(output_types, ListInt, {})
    .ATTR(output_shapes, ListListInt, {{},{}})
    .ATTR(output_num, Int, 1)
    .ATTR(_kernel, String, "dp")
    .OP_END_FACTORY_REG(IteratorGetNext)

/**
*@brief Device queue data area.
*@par Attributes:
*output_types: Data type of output
*output_shapes: Shapes of output
*channel_name: Channel ID corresponding to TDT
*@par Inputs: None
*@par Outputs: Dataset GetNext iterator
*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(DeviceQueueDataset)
    .OUTPUT(y, TensorType::ALL())
    .ATTR(output_types, ListInt, {})
    .ATTR(output_shapes, ListListInt, {{},{}})
    .ATTR(channel_name, String, "")
    .ATTR(_iterator_name, String, "IteratorV2")
    .OP_END_FACTORY_REG(DeviceQueueDataset)

} // namespace ge


#endif  // OPS_BUILT_IN_OP_PROTO_INC_GET_DATA_OPS_H_
