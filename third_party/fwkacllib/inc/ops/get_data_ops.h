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

#ifndef GE_OP_GET_DATA_OPS_H_
#define GE_OP_GET_DATA_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

REG_OP(MakeIterator)
    .INPUT(x, TensorType::ALL())
    .INPUT(x1, TensorType::ALL())
    .ATTR(_kernel, String, "dp")
    .OP_END_FACTORY_REG(MakeIterator)

REG_OP(IteratorV2)
    .OUTPUT(y, TensorType::ALL())
    .ATTR(output_types, ListInt, {})
    .ATTR(output_shapes,ListListInt, {{}, {}})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(IteratorV2)

REG_OP(IteratorGetNext)
    .INPUT(x, TensorType::ALL())
    .DYNAMIC_OUTPUT(y, TensorType::ALL())
    .ATTR(output_types, ListInt, {})
    .ATTR(output_shapes, ListListInt, {{},{}})
    .ATTR(output_num, Int, 1)
    .ATTR(_kernel, String, "dp")
    .OP_END_FACTORY_REG(IteratorGetNext)

REG_OP(DeviceQueueDataset)
    .OUTPUT(y, TensorType::ALL())
    .ATTR(output_types, ListInt, {})
    .ATTR(output_shapes, ListListInt, {{},{}})
    .ATTR(channel_name, String, "")
    .ATTR(_iterator_name, String, "IteratorV2")
    .OP_END_FACTORY_REG(DeviceQueueDataset)

} // namespace ge


#endif  // GE_OP_GET_DATA_OPS_H_
