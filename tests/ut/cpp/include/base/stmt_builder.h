/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef UT_BASE_STMT_BUILDER_H_
#define UT_BASE_STMT_BUILDER_H_
#include <list>
#include <string>
#include <vector>
#include "tvm/ir.h"
#include "base/expr_builder.h"

namespace akg {
class UTStmtBuilder {
 public:
  UTStmtBuilder() = default;
  ~UTStmtBuilder() = default;
  static air::Stmt CreateFor(
      air::Var loop_var,
      int32_t min,
      int32_t extent,
      air::Stmt body);
  static air::Stmt CreateRealizeByPlaceholderOp(
      const air::Operation &op,
      air::Stmt body);
  static air::Stmt CreateProvideAssign(
      air::ir::FunctionRef func_dst,
      air::Array<air::Expr> vars,
      air::Expr src,
      int value_index = 0);

  template <typename T>
  static air::Stmt CreateProvideBinary(
      air::ir::FunctionRef func_dst,
      air::Array<air::Expr> vars,
      air::Expr src1,
      air::Expr src2,
      int value_index = 0) {
    return air::ir::Provide::make(
        func_dst,
        value_index,
        T::make(src1, src2),
        vars);
  }
};  // class UTStmtBuilder
}  // namespace akg
#endif  // UT_BASE_STMT_BUILDER_H_
