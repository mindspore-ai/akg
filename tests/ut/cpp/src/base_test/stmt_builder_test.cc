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
#include "gtest/gtest.h"
#include "base/dump_helper.h"
#include "base/expr_builder.h"
#include "base/stmt_builder.h"

namespace akg {
TEST(UTStmtBuilder, CreateProvideAssign) {
  // b(ax0) = a(ax0)
  UTVariablePool vp;
  vp.AddVar("ax0");
  air::Operation a = UTExprBuilder::PlaceholderOpNode("a", {1024}, air::Float(16));
  air::Operation b = UTExprBuilder::PlaceholderOpNode("b", {1024}, air::Float(16));
  air::Stmt stmt = UTStmtBuilder::CreateProvideAssign(
      b, vp.GetVars({"ax0"}), UTExprBuilder::ElementOf(a, vp.GetVars({"ax0"})));
  std::string dump_stmt = UTDumpHelper::Dump(stmt);
  EXPECT_EQ(dump_stmt, "b(ax0) = a(ax0)\n");
}

TEST(UTStmtBuilder, CreateProvideBinary) {
  // c(ax0) = a(ax0) + b(ax0)
  UTVariablePool vp;
  vp.AddVar("ax0");
  air::Operation a = UTExprBuilder::PlaceholderOpNode("a", {1024}, air::Float(16));
  air::Operation b = UTExprBuilder::PlaceholderOpNode("b", {1024}, air::Float(16));
  air::Operation c = UTExprBuilder::PlaceholderOpNode("c", {1024}, air::Float(16));
  air::Stmt stmt = UTStmtBuilder::CreateProvideBinary<air::ir::Add>(
      c, vp.GetVars({"ax0"}),
      UTExprBuilder::ElementOf(a, vp.GetVars({"ax0"})),
      UTExprBuilder::ElementOf(b, vp.GetVars({"ax0"})));
  std::string dump_stmt = UTDumpHelper::Dump(stmt);
  EXPECT_EQ(dump_stmt, "c(ax0) = (a(ax0) + b(ax0))\n");
}

TEST(UTStmtBuilder, CreateFor) {
  /*
   * for (i, 0, 1024) {
   *   c(i) = (a(i) + b(i))
   * }
   */
  UTVariablePool vp;
  vp.AddVar("i");
  air::Operation a = UTExprBuilder::PlaceholderOpNode("a", {1024}, air::Float(16));
  air::Operation b = UTExprBuilder::PlaceholderOpNode("b", {1024}, air::Float(16));
  air::Operation c = UTExprBuilder::PlaceholderOpNode("c", {1024}, air::Float(16));
  air::Stmt stmt_for = UTStmtBuilder::CreateFor(
      vp.GetVar("i"), 0, 1024,
      UTStmtBuilder::CreateProvideBinary<air::ir::Add>(
          c, vp.GetVars({"i"}),
          UTExprBuilder::ElementOf(a, vp.GetVars({"i"})),
          UTExprBuilder::ElementOf(b, vp.GetVars({"i"}))));
  std::string dump_stmt_for = UTDumpHelper::Dump(stmt_for);
  EXPECT_EQ(dump_stmt_for,
      "for (i, 0, 1024) {\n"
      "  c(i) = (a(i) + b(i))\n"
      "}\n");
}

TEST(UTStmtBuilder, CreateRealizeByPlaceholderOp) {
  UTVariablePool vp;
  vp.AddVar("i");
  air::Operation a = UTExprBuilder::PlaceholderOpNode("a", {1024}, air::Float(16));
  air::Operation b = UTExprBuilder::PlaceholderOpNode("b", {1024}, air::Float(16));
  air::Operation c = UTExprBuilder::PlaceholderOpNode("c", {1024}, air::Float(16));
  air::Stmt stmt_realize = UTStmtBuilder::CreateRealizeByPlaceholderOp(
      c,
      UTStmtBuilder::CreateFor(
          vp.GetVar("i"), 0, 1024,
          UTStmtBuilder::CreateProvideBinary<air::ir::Add>(
              c, vp.GetVars({"i"}),
              UTExprBuilder::ElementOf(a, vp.GetVars({"i"})),
              UTExprBuilder::ElementOf(b, vp.GetVars({"i"})))));
  std::string dump_stmt_realize = UTDumpHelper::Dump(stmt_realize);
  EXPECT_EQ(dump_stmt_realize,
      "realize c<float16>([0, 1024]) {\n"
      "  for (i, 0, 1024) {\n"
      "    c(i) = (a(i) + b(i))\n"
      "  }\n"
      "}\n");
}
}  // namespace akg
