/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * 2021.7.9 - Add asin, acos, asinh, acosh.
 * 2021.7.11 - Add atan2, expm1
 */

#ifndef AKG_TOPI_ELEMWISE_H_
#define AKG_TOPI_ELEMWISE_H_

#include <string>

#include "topi/tags.h"
#include "tvm/ir.h"
#include "tvm/ir_pass.h"
#include "broadcast.h"

namespace topi {
using namespace air;

// Binary intrinsic operators
#define TOPI_DECLARE_BINARY_OP(OpName)                          \
  inline Tensor OpName(const Tensor& x,                         \
                       const Tensor& y,                         \
                       std::string name = "T_" #OpName,         \
                       std::string tag = kElementWise) {        \
    return compute(x->shape, [&](const Array<Var>& i) {         \
        return ::air::OpName(x(i), y(i));                       \
      }, name, tag);                                            \
  }

TOPI_DECLARE_UNARY_OP(asin);
TOPI_DECLARE_UNARY_OP(acos);
TOPI_DECLARE_UNARY_OP(isinf);
TOPI_DECLARE_UNARY_OP(isfinite);
TOPI_DECLARE_UNARY_OP(asinh);
TOPI_DECLARE_UNARY_OP(acosh);
TOPI_DECLARE_BINARY_OP(atan2);
TOPI_DECLARE_UNARY_OP(expm1);

}  // namespace topi
#endif  // AKG_TOPI_ELEMWISE_H_