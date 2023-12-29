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
 */

#ifndef AKG_EXPR_OPERATOR_H_
#define AKG_EXPR_OPERATOR_H_

#include <algorithm>
#include <type_traits>
#include "expr.h"
#include "ir.h"

namespace air {
TVM_DECLARE_INTRIN_UNARY(acos);
TVM_DECLARE_INTRIN_UNARY(asin);
TVM_DECLARE_INTRIN_UNARY(asinh);
TVM_DECLARE_INTRIN_UNARY(acosh);
TVM_DECLARE_INTRIN_UNARY(expm1);

#define AKG_DECLARE_INTRIN_BINARY(OpName)                               \
  inline Expr OpName(Expr x, Expr y) {                                  \
    return ir::Call::make(x.type(), #OpName, {x, y}, ir::Call::PureIntrinsic); \
  }                                                                     \

AKG_DECLARE_INTRIN_BINARY(atan2);
}  // namespace air
#endif  // AKG_EXPR_OPERATOR_H_
