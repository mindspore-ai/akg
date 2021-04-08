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
 * \file internal_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_INTERNAL_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_INTERNAL_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief aicpu assit help op for auxiliary matrix generation. \n

*@par Inputs:
*The input is dynamic for attribute func_name   \n

*@par Attributes:
*@li func_name:An required param, for example "topkv2".   \n

*@par Outputs:
*The output is dynamic for attribute func_name.
*/
REG_OP(AssistHelp)
    .DYNAMIC_INPUT(x, TensorType({ DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16,
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE }))
    .DYNAMIC_OUTPUT(y, TensorType({ DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16,
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    . REQUIRED_ATTR (func_name, String)
    . OP_END_FACTORY_REG(AssistHelp)

/**
*@brief aicpu cache help for lhisi cache flush. \n

*@par Inputs:
*The input is dynamic for attribute func_name   \n

*@par Outputs:
*The output is dynamic for attribute func_name.
*/
REG_OP(CacheUpdate)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(x, TensorType::BasicType())
    .OP_END_FACTORY_REG(CacheUpdate)

/**
*@brief transfer data from L1 buffer to DDR or DDR to L1. \n

*@par Inputs:
*The input is dynamic for attribute func_name   \n

*@par Outputs:
*The output is dynamic for attribute func_name.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(InternalDataMove)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .REQUIRED_ATTR(src_buf, String)
    .REQUIRED_ATTR(dst_buf, String)
    .OP_END_FACTORY_REG(InternalDataMove)

}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_INTERNAL_OPS_H_
