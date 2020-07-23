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
TEST(UTExprBuilder, IntImm) {
  air::Expr int1 = UTExprBuilder::IntImm(1024);
  std::string dump_int1 = UTDumpHelper::Dump(int1);
  EXPECT_EQ(dump_int1, "1024");
  air::Expr int2 = UTExprBuilder::IntImm(1024, air::Int(64));
  std::string dump_int2 = UTDumpHelper::Dump(int2);
  EXPECT_EQ(dump_int2, "(int64)1024");
  air::Expr int3 = UTExprBuilder::IntImm(1024, air::Int(16));
  std::string dump_int3 = UTDumpHelper::Dump(int3);
  EXPECT_EQ(dump_int3, "(int16)1024");
}

TEST(UTExprBuilder, UIntImm) {
  air::Expr uint1 = UTExprBuilder::UIntImm(1024);
  std::string dump_uint1 = UTDumpHelper::Dump(uint1);
  EXPECT_EQ(dump_uint1, "(uint32)1024");
  air::Expr uint2 = UTExprBuilder::UIntImm(1024, air::UInt(64));
  std::string dump_uint2 = UTDumpHelper::Dump(uint2);
  EXPECT_EQ(dump_uint2, "(uint64)1024");
  air::Expr uint3 = UTExprBuilder::UIntImm(1024, air::UInt(16));
  std::string dump_uint3 = UTDumpHelper::Dump(uint3);
  EXPECT_EQ(dump_uint3, "(uint16)1024");
}

TEST(UTExprBuilder, Bool) {
  air::Expr bool_true = UTExprBuilder::BoolImm(true);
  std::string dump_bool_true = UTDumpHelper::Dump(bool_true);
  EXPECT_EQ(dump_bool_true, "(bool)1");
  air::Expr bool_false = UTExprBuilder::BoolImm(false);
  std::string dump_bool_false = UTDumpHelper::Dump(bool_false);
  EXPECT_EQ(dump_bool_false, "(bool)0");
}

TEST(UTExprBuilder, CreateShape) {
  air::Array<air::Expr> shape1 = UTExprBuilder::CreateShape({1024});
  std::string dump_shape1 = UTDumpHelper::Dump(shape1);
  EXPECT_EQ(dump_shape1, "[1024]");

  air::Array<air::Expr> shape2 = UTExprBuilder::CreateShape({32, 1024});
  std::string dump_shape2 = UTDumpHelper::Dump(shape2);
  EXPECT_EQ(dump_shape2, "[32, 1024]");

  air::Array<air::Expr> shape3 = UTExprBuilder::CreateShape({16, 32, 1024});
  std::string dump_shape3 = UTDumpHelper::Dump(shape3);
  EXPECT_EQ(dump_shape3, "[16, 32, 1024]");
}

TEST(UTExprBuilder, CreateVar) {
  air::Var var = UTExprBuilder::CreateVar("ax0");
  std::string dump_var = UTDumpHelper::Dump(var);
  EXPECT_EQ(dump_var, "ax0");
}

TEST(UTExprBuilder, CreateVars) {
  air::Array<air::Expr> vars = UTExprBuilder::CreateVars({"ax0", "ax1", "ax2"});
  std::string dump_vars = UTDumpHelper::Dump(vars);
  EXPECT_EQ(dump_vars, "[ax0, ax1, ax2]");
}

TEST(UTExprBuilder, CreateRange) {
  air::Range range = UTExprBuilder::CreateRange(0, 1024);
  std::string dump_range = UTDumpHelper::Dump(range);
  EXPECT_EQ(dump_range, "range(min=0, ext=1024)");
}

TEST(UTExprBuilder, PlaceholderOpNode) {
  air::Operation node = UTExprBuilder::PlaceholderOpNode("input", {16, 32, 1024}, air::Float(16));
  std::string dump_node = UTDumpHelper::Dump(node);
  EXPECT_EQ(UTDumpHelper::RegxMatchPlaceholder(dump_node, "input"), true);
}

TEST(UTExprBuilder, TensorElement) {
  air::Expr elem = UTExprBuilder::TensorElement("input", {16, 32, 1024}, {"ax0", "ax1", "ax2"}, air::Float(16));
  std::string dump_elem = UTDumpHelper::Dump(elem);
  EXPECT_EQ(dump_elem, "input(ax0, ax1, ax2)");
}

TEST(UTExprBuilder, ElememtOfPlaceholderOp) {
  air::Operation op = UTExprBuilder::PlaceholderOpNode("input", {16, 32, 1024}, air::Float(16));
  air::Expr elem = UTExprBuilder::ElementOfPlaceholderOp(op, {"ax0", "ax1", "ax2"});
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
