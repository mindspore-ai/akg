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
#include <tvm/ir.h>
#include "base/dump_helper.h"
#include "base/expr_builder.h"
#include "base/ir_checker.h"
#include "base/stmt_builder.h"
#define private public
#define protected public
#include "pass/to_three_address.cc"
#undef protected
#undef private

namespace akg {
class ToThreeAddressTest {
 public:
  ToThreeAddressTest() = default;
  ~ToThreeAddressTest() = default;
};

TEST(ToThreeAddressTest, BuildCase1) {
  UTTensorElementHelper th({16, 32, 1024});
  using Add = air::ir::Add;
  // a(ax1, ax2) + b(ax2) + c(ax0, ax1, ax2) + d(ax2)
  air::Expr expr = Add::make(Add::make(Add::make(th.Elem("a", 2), th.Elem("b", 1)), th.Elem("c", 3)), th.Elem("d", 1));
  std::string dump_expr = UTDumpHelper::Dump(expr);
  EXPECT_EQ(dump_expr, "(((a(ax1, ax2) + b(ax2)) + c(ax0, ax1, ax2)) + d(ax2))");
}

class ThreeAddressExprMutatorTest : public testing::Test {
 public:
  ThreeAddressExprMutatorTest()
      : mutator_(air::TensorNode::make(UTExprBuilder::CreateShape(shape_),               // shape
                                       dtype_,                                           // dtype
                                       UTExprBuilder::PlaceholderOpNode("out", shape_),  // op
                                       0),                                               // index
                 UTExprBuilder::CreateVars({"ax0", "ax1", "ax2"}),                       // args
                 UTExprBuilder::CreateVars({"ax0", "ax1", "ax2"}),                       // args
                 UTExprBuilder::CreateShape(shape_),                                     // shape
                 std::unordered_set<const Call *>(),                                     // broadcast
                 false,                                                                  // IsReductionOp
                 false) {}                                                               // cross_stmt_simplify
  ~ThreeAddressExprMutatorTest() = default;

  std::vector<int32_t> shape_ = {16, 32, 1024};
  air::DataType dtype_ = air::Float(16);
  ir::ThreeAddressExprMutator mutator_;
};  // ThreeAddressExprMutatorTest

TEST_F(ThreeAddressExprMutatorTest, MutateBinaryOp_Add) {
  UTTensorElementHelper th(shape_);
  using Add = air::ir::Add;
  air::Expr expr = Add::make(th.Elem("a", 2), th.Elem("b", 1));
  Expr expr_m = mutator_.Mutate(expr);
  EXPECT_NE(mutator_.imm_ops.size(), 0);
}

class PassTestToThreeAddress1 : public ::testing::Test {
 public:
  PassTestToThreeAddress1() { Construct(); }
  ~PassTestToThreeAddress1() = default;
  void Construct() {
    a_ = UTExprBuilder::PlaceholderOpNode("a", {1024}, air::Float(16));
    b_ = UTExprBuilder::PlaceholderOpNode("b", {32, 1024}, air::Float(16));
    c_ = UTExprBuilder::PlaceholderOpNode("c", {1024}, air::Float(16));
    out_ = UTExprBuilder::PlaceholderOpNode("out", {32, 1024}, air::Float(16));
    stmt = air::ir::AttrStmt::make(
      out_, "", UTExprBuilder::IntImm(1),
      UTStmtBuilder::CreateRealizeByPlaceholderOp(
        out_, air::ir::ProducerConsumer::make(
                out_, true,
                UTStmtBuilder::CreateFor(
                  "i", 0, 32,
                  UTStmtBuilder::CreateFor(
                    "j", 0, 1024,
                    UTStmtBuilder::CreateProvideBinary<air::ir::Add>(
                      out_, {"i", "j"},
                      air::ir::Add::make(UTExprBuilder::ElementOf(a_, {"j"}), UTExprBuilder::ElementOf(b_, {"i", "j"})),
                      UTExprBuilder::ElementOf(c_, {"j"})))))));
  }

  air::Operation a_;
  air::Operation b_;
  air::Operation c_;
  air::Operation out_;
  air::Stmt stmt;
};  // class PassTestToThreeAddress1

TEST_F(PassTestToThreeAddress1, CaseCheck) {
  std::vector<std::tuple<std::string, const air::ir::Provide *, uint64_t>> infos_lhs =
    UTProvideCheckerForAssign().Find(stmt, "((a(j) + b(i, j)) + c(j))");
  ASSERT_EQ(infos_lhs.size(), 1);
  EXPECT_EQ(std::get<0>(infos_lhs[0]), "out(i, j)");
  EXPECT_EQ(std::get<2>(infos_lhs[0]), 32 * 1024);
}

TEST_F(PassTestToThreeAddress1, TestPass) {
  Stmt stmt_out = ir::ToThreeAddress(stmt, false, 0, true);
  /* current implementation
   *   out_2(i, j) = b(i, j)
   *   out_3(i, j) = (a(j) + out_2(i, j))
   *   out(i, j) = (out_3(i, j) + c(j))
   */
  std::vector<std::tuple<std::string, const air::ir::Provide *, uint64_t>> info1 =
    UTProvideCheckerForAssign().Find(stmt_out, "b(i, j)");
  ASSERT_EQ(info1.size(), 1);
  std::string dump_b_target = std::get<0>(info1[0]);

  std::vector<std::tuple<std::string, const air::ir::Provide *, uint64_t>> info2 =
    UTProvideCheckerForBinary().Find(stmt_out, UTProvideCheckerForBinary::BinaryOpType::kAdd, "a(j)", dump_b_target);
  ASSERT_EQ(info2.size(), 1);
  std::string dump_sum1_target = std::get<0>(info2[0]);
  EXPECT_EQ(std::get<2>(info2[0]), 32 * 1024);

  std::vector<std::tuple<std::string, const air::ir::Provide *, uint64_t>> info3 =
    UTProvideCheckerForBinary().Find(stmt_out, UTProvideCheckerForBinary::BinaryOpType::kAdd, dump_sum1_target, "c(j)");
  ASSERT_EQ(info3.size(), 1);
  EXPECT_EQ(std::get<0>(info3[0]), "out(i, j)");
  EXPECT_EQ(std::get<2>(info3[0]), 32 * 1024);
}
}  // namespace akg
