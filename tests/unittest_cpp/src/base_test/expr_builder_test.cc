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

namespace akg {
TEST(UTExprBuilder, CreateShape) {
  ktvm::Array<ktvm::Expr> shape1 = UTExprBuilder::CreateShape({1024});
  std::string dump_shape1 = UTDumpHelper::Dump(shape1);
  EXPECT_EQ(dump_shape1, "[1024]");

  ktvm::Array<ktvm::Expr> shape2 = UTExprBuilder::CreateShape({32, 1024});
  std::string dump_shape2 = UTDumpHelper::Dump(shape2);
  EXPECT_EQ(dump_shape2, "[32, 1024]");

  ktvm::Array<ktvm::Expr> shape3 = UTExprBuilder::CreateShape({16, 32, 1024});
  std::string dump_shape3 = UTDumpHelper::Dump(shape3);
  EXPECT_EQ(dump_shape3, "[16, 32, 1024]");
}

TEST(UTExprBuilder, CreateVar) {
  ktvm::Var var = UTExprBuilder::CreateVar("ax0");
  std::string dump_var = UTDumpHelper::Dump(var);
  EXPECT_EQ(dump_var, "ax0");
}

TEST(UTExprBuilder, CreateVars) {
  ktvm::Array<ktvm::Expr> vars = UTExprBuilder::CreateVars({"ax0", "ax1", "ax2"});
  std::string dump_vars = UTDumpHelper::Dump(vars);
  EXPECT_EQ(dump_vars, "[ax0, ax1, ax2]");
}

TEST(UTExprBuilder, PlaceholderOpNode) {
  ktvm::Operation node = UTExprBuilder::PlaceholderOpNode("input", {16, 32, 1024}, ktvm::Float(16));
  std::string dump_node = UTDumpHelper::Dump(node);
  EXPECT_EQ(UTDumpHelper::RegxMatchPlaceholder(dump_node, "input"), true);
}

TEST(UTExprBuilder, TensorElement) {
  ktvm::Expr elem = UTExprBuilder::TensorElement("input", {16, 32, 1024}, {"ax0", "ax1", "ax2"}, ktvm::Float(16));
  std::string dump_elem = UTDumpHelper::Dump(elem);
  EXPECT_EQ(dump_elem, "input(ax0, ax1, ax2)");
}

TEST(UTTensorElementHelper, TensorElement) {
  UTTensorElementHelper helper({16, 32, 1024});
  std::string dump_elem1 = UTDumpHelper::Dump(helper.Elem("a", 3));
  EXPECT_EQ(dump_elem1, "a(ax0, ax1, ax2)");
  std::string dump_elem2 = UTDumpHelper::Dump(helper.Elem("b", 2));
  EXPECT_EQ(dump_elem2, "b(ax1, ax2)");
  std::string dump_elem3 = UTDumpHelper::Dump(helper.Elem("c", 1));
  EXPECT_EQ(dump_elem3, "c(ax2)");
}
}  // namespace akg
