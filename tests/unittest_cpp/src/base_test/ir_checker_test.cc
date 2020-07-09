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
#include <gtest/gtest.h>
#include <string>
#include "base/ir_checker.h"
#include "base/expr_builder.h"
#include "base/stmt_builder.h"

namespace akg {
TEST(UTProvideChecker, CompareDump) {
  EXPECT_EQ(UTProvideChecker().CompareDump("a(i, j)", "a"), false);
  EXPECT_EQ(UTProvideChecker().CompareDump("a(i, j)", "a(i, j)"), true);
  EXPECT_EQ(UTProvideChecker(true).CompareDump("a(i, j)", "a"), true);
  EXPECT_EQ(UTProvideChecker(true).CompareDump("a(i, j)", "a(i, j)"), true);
}

class UTProvideCheckerTest : public testing::Test {
 public:
  UTProvideCheckerTest()
      : a_(UTExprBuilder::PlaceholderOpNode("a", {1024}, air::Float(16))),
        b_(UTExprBuilder::PlaceholderOpNode("b", {1024}, air::Float(16))),
        c_(UTExprBuilder::PlaceholderOpNode("c", {1024}, air::Float(16))) {}
  ~UTProvideCheckerTest() = default;
  air::Operation a_;
  air::Operation b_;
  air::Operation c_;
};  // class UTProvideCheckerTest

TEST_F(UTProvideCheckerTest, UTProvideCheckerForAssign) {
  // b(ax0) = a(ax0)
  air::Stmt stmt = UTStmtBuilder::CreateProvideAssign(
      b_, {"ax0"}, UTExprBuilder::ElementOf(a_, {"ax0"}));
  std::vector<std::tuple<std::string, const air::ir::Provide*, uint64_t>> infos_lhs =
      UTProvideCheckerForAssign().Find(stmt, "a(ax0)");
  ASSERT_EQ(infos_lhs.size(), 1);
  EXPECT_EQ(std::get<0>(infos_lhs[0]), "b(ax0)");
  EXPECT_EQ(std::get<2>(infos_lhs[0]), 1);
}

TEST_F(UTProvideCheckerTest, UTProvideCheckerForBinary) {
  // c(ax0) = (a(ax0) + b(ax0))
  air::Stmt stmt = UTStmtBuilder::CreateProvideBinary<air::ir::Add>(
      c_, {"ax0"},
      UTExprBuilder::ElementOf(a_, {"ax0"}),
      UTExprBuilder::ElementOf(b_, {"ax0"}));
  std::vector<std::tuple<std::string, const air::ir::Provide*, uint64_t>> infos_lhs =
      UTProvideCheckerForBinary().Find(stmt, UTProvideCheckerForBinary::BinaryOpType::kAdd, "a(ax0)", "b(ax0)");
  ASSERT_EQ(infos_lhs.size(), 1);
  EXPECT_EQ(std::get<0>(infos_lhs[0]), "c(ax0)");
  EXPECT_EQ(std::get<2>(infos_lhs[0]), 1);
}

class UTProvideCheckerTest2 : public testing::Test {
 public:
  UTProvideCheckerTest2()
      : a_(UTExprBuilder::PlaceholderOpNode("a", {16, 32, 1024}, air::Float(16))),
        b_(UTExprBuilder::PlaceholderOpNode("b", {16, 32, 1024}, air::Float(16))),
        c_(UTExprBuilder::PlaceholderOpNode("c", {16, 32, 1024}, air::Float(16))) {}
  ~UTProvideCheckerTest2() = default;
  air::Operation a_;
  air::Operation b_;
  air::Operation c_;
};  // class UTProvideCheckerTest

TEST_F(UTProvideCheckerTest2, UTProvideCheckerForBinary) {
  air::Stmt stmt = UTStmtBuilder::CreateFor(
      "i", 0, 16,
      UTStmtBuilder::CreateFor(
          "j", 0, 32,
          UTStmtBuilder::CreateFor(
              "k", 0, 1024,
              UTStmtBuilder::CreateProvideBinary<air::ir::Add>(
                  c_, {"i", "j", "k"},
                  UTExprBuilder::ElementOf(a_, {"i", "j", "k"}),
                  UTExprBuilder::ElementOf(b_, {"i", "j", "k"})))));
  std::string dump_stmt = UTDumpHelper::Dump(stmt);
  EXPECT_EQ(dump_stmt,
      "for (i, 0, 16) {\n"
      "  for (j, 0, 32) {\n"
      "    for (k, 0, 1024) {\n"
      "      c(i, j, k) = (a(i, j, k) + b(i, j, k))\n"
      "    }\n"
      "  }\n"
      "}\n");
  std::vector<std::tuple<std::string, const air::ir::Provide*, uint64_t>> infos_lhs =
      UTProvideCheckerForBinary().Find(
          stmt, UTProvideCheckerForBinary::BinaryOpType::kAdd, "a(i, j, k)", "b(i, j, k)");
  ASSERT_EQ(infos_lhs.size(), 1);
  EXPECT_EQ(std::get<0>(infos_lhs[0]), "c(i, j, k)");
  EXPECT_EQ(std::get<2>(infos_lhs[0]), 1024 * 32 * 16);
}
}  // namespace akg
