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
#include "base/stmt_builder.h"

namespace akg {
air::Stmt UTStmtBuilder::CreateFor(
    const std::string &loop_var_name,
    int32_t min,
    int32_t extent,
    air::Stmt body) {
  return air::ir::For::make(
      UTExprBuilder::CreateVar(loop_var_name),
      UTExprBuilder::IntImm(min),
      UTExprBuilder::IntImm(extent),
      air::ir::ForType::Serial,
      air::ir::DeviceAPI::None,
      body);
}
air::Stmt UTStmtBuilder::CreateRealizeByPlaceholderOp(
    const air::Operation &op,
    air::Stmt body) {
  const air::PlaceholderOpNode *node = op.as<const air::PlaceholderOpNode>();
  CHECK(node);
  return air::ir::Realize::make(
      op,
      0,
      node->dtype,
      UTExprBuilder::CreateRegion(node->shape),
      UTExprBuilder::BoolImm(true),
      body);
}

air::Stmt UTStmtBuilder::CreateProvideAssign(
    air::ir::FunctionRef func_dst,
    const std::vector<std::string> &vars,
    air::Expr src,
    int value_index) {
  return air::ir::Provide::make(
      func_dst,
      value_index,
      src,
      UTExprBuilder::CreateVars(vars));
}
}  // namespace akg
