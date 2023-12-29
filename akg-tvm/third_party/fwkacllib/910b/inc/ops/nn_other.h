/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file nn_other.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_OTHER_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_OTHER_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Apply rotary position embedding.
 * @par Inputs:
 * x: A tensor. Must be one of the following types: float16, float, bfloat16.
 * r1: A tensor. Must be one of the following types: float16, float, bfloat16.
 * r2: A tensor. Must be one of the following types: float16, float, bfloat16. 
 * @par Outputs:
 * y: A Tensor. Has the same shape as "x".
 */
REG_OP(RotaryMul)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(r1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(r2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OP_END_FACTORY_REG(RotaryMul)

/**
 * @brief Calculate the inverse gradient of RotaryMul.
 * @par Inputs:
 * x: A tensor. Must be one of the following types: float16, float, bfloat16.
 * r1: A tensor. Must be one of the following types: float16, float, bfloat16.
 * r2: A tensor. Must be one of the following types: float16, float, bfloat16.
 * dy: A tensor. Data of grad increment.
 * @par Outputs:
 * dx: A Tensor. Has the same shape as "x".
 * dr1: A Tensor. Has the same shape as "r1".
 * dr2: A Tensor. Has the same shape as "r2".
 */
REG_OP(RotaryMulGrad)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .INPUT(r1, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .INPUT(r2, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .OUTPUT(dr1, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .OUTPUT(dr2, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .OP_END_FACTORY_REG(RotaryMulGrad)
} // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_OTHER_H_
