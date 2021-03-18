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

/*
 * TestCase: Add3 with different dimension
 * input: a(ax2) + b(ax1, ax2) + c(ax2)
 * expect output: (a(ax2) + c(ax2)) + b(ax1, ax2)
 */
class PassTestToThreeAddress_Add3 : public ::testing::Test {
 public:
  PassTestToThreeAddress_Add3() { Construct(); }
  ~PassTestToThreeAddress_Add3() = default;
  void Construct() {
    vp_.AddVars({"ax1", "ax2"});
    a_ = UTExprBuilder::PlaceholderOpNode("a", {1024}, air::Float(16));
    b_ = UTExprBuilder::PlaceholderOpNode("b", {32, 1024}, air::Float(16));
    c_ = UTExprBuilder::PlaceholderOpNode("c", {1024}, air::Float(16));
    out_ = UTExprBuilder::PlaceholderOpNode("out", {32, 1024}, air::Float(16));
    stmt = air::ir::AttrStmt::make(
        out_, "", UTExprBuilder::IntImm(1),
        UTStmtBuilder::CreateRealizeByPlaceholderOp(
        out_,
        air::ir::ProducerConsumer::make(out_, true,
            UTStmtBuilder::CreateFor(
                vp_.GetVar("ax1"), 0, 32,
                UTStmtBuilder::CreateFor(
                    vp_.GetVar("ax2"), 0, 1024,
                    UTStmtBuilder::CreateProvideBinary<air::ir::Add>(
                        out_, vp_.GetVars({"ax1", "ax2"}),
                        air::ir::Add::make(
                            UTExprBuilder::ElementOf(a_, vp_.GetVars({"ax2"})),
                            UTExprBuilder::ElementOf(b_, vp_.GetVars({"ax1", "ax2"}))),
                        UTExprBuilder::ElementOf(c_, vp_.GetVars({"ax2"}))))))));
  }

  UTVariablePool vp_;
  air::Operation a_;
  air::Operation b_;
  air::Operation c_;
  air::Operation out_;
  air::Stmt stmt;
};  // class PassTestToThreeAddress_Add3

TEST_F(PassTestToThreeAddress_Add3, TestPass) {
  Stmt stmt_out = ir::ToThreeAddress(stmt, false, 0, true);
  // check1: out1(ax2) = a(ax2) + c(ax2)
  std::vector<std::tuple<std::string, const air::ir::Provide*, uint64_t>> info1 =
      UTProvideCheckerForBinary().Find(
          stmt_out, UTProvideCheckerForBinary::BinaryOpType::kAdd, "a(ax2)", "c(ax2)");
  ASSERT_EQ(info1.size(), 1);
  EXPECT_EQ(std::get<2>(info1[0]), 1024);
  std::string out1_name = std::get<0>(info1[0]);
  // check2: out(ax1, ax2) = out1(ax2) + b(ax1, ax2)
  std::vector<std::tuple<std::string, const air::ir::Provide*, uint64_t>> info2 =
      UTProvideCheckerForBinary().Find(
          stmt_out, UTProvideCheckerForBinary::BinaryOpType::kAdd, out1_name, "b(ax1, ax2)");
  ASSERT_EQ(info2.size(), 1);
  EXPECT_EQ(std::get<0>(info2[0]), "out(ax1, ax2)");
  EXPECT_EQ(std::get<2>(info2[0]), 32 * 1024);
}

/*
 * TestCase: Add4 with different dimension
 * input: a(ax3) + b(ax1, ax2, ax3) + c(ax2, ax3) + d(ax3)
 * expect output: ((a(ax3) + d(ax3)) + c(ax2, ax3)) + b(ax1, ax2, ax3)
 */
class PassTestToThreeAddress_Add4 : public ::testing::Test {
 public:
  PassTestToThreeAddress_Add4() {
    Construct();
  }
  ~PassTestToThreeAddress_Add4() = default;
  void Construct() {
    vp_.AddVars({"ax1", "ax2", "ax3"});
    a_ = UTExprBuilder::PlaceholderOpNode("a", {1024}, air::Float(16));
    b_ = UTExprBuilder::PlaceholderOpNode("b", {16, 32, 1024}, air::Float(16));
    c_ = UTExprBuilder::PlaceholderOpNode("c", {32, 1024}, air::Float(16));
    d_ = UTExprBuilder::PlaceholderOpNode("d", {1024}, air::Float(16));
    out_ = UTExprBuilder::PlaceholderOpNode("out", {16, 32, 1024}, air::Float(16));
    stmt = air::ir::AttrStmt::make(
        out_, "", UTExprBuilder::IntImm(1),
        UTStmtBuilder::CreateRealizeByPlaceholderOp(
        out_,
        air::ir::ProducerConsumer::make(out_, true,
            UTStmtBuilder::CreateFor(
                vp_.GetVar("ax1"), 0, 16,
                UTStmtBuilder::CreateFor(
                    vp_.GetVar("ax2"), 0, 32,
                    UTStmtBuilder::CreateFor(
                        vp_.GetVar("ax3"), 0, 1024,
                        UTStmtBuilder::CreateProvideBinary<air::ir::Add>(
                            out_, vp_.GetVars({"ax1", "ax2", "ax3"}),
                            air::ir::Add::make(
                                air::ir::Add::make(
                                    UTExprBuilder::ElementOf(a_, vp_.GetVars({"ax3"})),
                                    UTExprBuilder::ElementOf(b_, vp_.GetVars({"ax1", "ax2", "ax3"}))),
                                UTExprBuilder::ElementOf(c_, vp_.GetVars({"ax2", "ax3"}))),
                            UTExprBuilder::ElementOf(d_, vp_.GetVars({"ax3"})))))))));
  }

  UTVariablePool vp_;
  air::Operation a_;
  air::Operation b_;
  air::Operation c_;
  air::Operation d_;
  air::Operation out_;
  air::Stmt stmt;
};  // class PassTestToThreeAddress1

TEST_F(PassTestToThreeAddress_Add4, TestPass) {
  Stmt stmt_out = ir::ToThreeAddress(stmt, false, 0, true);
  // check1: out1(ax3) = a(ax3) + d(ax3)
  std::vector<std::tuple<std::string, const air::ir::Provide *, uint64_t>> info1 =
      UTProvideCheckerForBinary().Find(
          stmt_out, UTProvideCheckerForBinary::BinaryOpType::kAdd, "a(ax3)", "d(ax3)");
  ASSERT_EQ(info1.size(), 1);
  EXPECT_EQ(std::get<2>(info1[0]), 1024);
  std::string out1_name = std::get<0>(info1[0]);
  // check2: out2(ax2, ax3) = out1(ax3) + c(ax2, ax3)
  std::vector<std::tuple<std::string, const air::ir::Provide*, uint64_t>> info2 =
      UTProvideCheckerForBinary().Find(
          stmt_out, UTProvideCheckerForBinary::BinaryOpType::kAdd, out1_name, "c(ax2, ax3)");
  ASSERT_EQ(info2.size(), 1);
  EXPECT_EQ(std::get<2>(info2[0]), 32 * 1024);
  std::string out2_name = std::get<0>(info2[0]);
  // check3: out(ax1, ax2, ax3) = out2(ax2, ax3) + b(ax1, ax2, ax3)
  std::vector<std::tuple<std::string, const air::ir::Provide*, uint64_t>> info3 =
      UTProvideCheckerForBinary().Find(
          stmt_out, UTProvideCheckerForBinary::BinaryOpType::kAdd, out2_name, "b(ax1, ax2, ax3)");
  ASSERT_EQ(info3.size(), 1);
  EXPECT_EQ(std::get<2>(info3[0]), 16 * 32 * 1024);
  EXPECT_EQ(std::get<0>(info3[0]), "out(ax1, ax2, ax3)");
}

/*
 * TestCase: multiply two tensors with different dimension
 * input: a(ax1, ax2) * b(ax1)
 * expect output: broadcast b(ax1) first and multiply
 */
class PassTestToThreeAddress_Mul2 : public ::testing::Test {
 public:
  PassTestToThreeAddress_Mul2() {
    Construct();
  }
  ~PassTestToThreeAddress_Mul2() = default;

  void Construct() {
    vp_.AddVars({"ax1", "ax2"});
    a_ = UTExprBuilder::PlaceholderOpNode("a", {32, 1024}, air::Float(16));
    b_ = UTExprBuilder::PlaceholderOpNode("b", {32}, air::Float(16));
    out_ = UTExprBuilder::PlaceholderOpNode("out", {32, 1024}, air::Float(16));
    stmt = air::ir::AttrStmt::make(
        out_, "", UTExprBuilder::IntImm(1),
        UTStmtBuilder::CreateRealizeByPlaceholderOp(
            out_,
            air::ir::ProducerConsumer::make(out_, true,
                UTStmtBuilder::CreateFor(
                    vp_.GetVar("ax1"), 0, 32,
                    UTStmtBuilder::CreateFor(
                        vp_.GetVar("ax2"), 0, 1024,
                        UTStmtBuilder::CreateProvideBinary<air::ir::Mul>(
                            out_, vp_.GetVars({"ax1", "ax2"}),
                            UTExprBuilder::ElementOf(a_, vp_.GetVars({"ax1", "ax2"})),
                            UTExprBuilder::ElementOf(b_, vp_.GetVars({"ax1"}))))))));
  }

  UTVariablePool vp_;
  air::Operation a_;
  air::Operation b_;
  air::Operation out_;
  air::Stmt stmt;
};  // class PassTestToThreeAddress_Mul2

TEST_F(PassTestToThreeAddress_Mul2, TestPass) {
  Stmt stmt_out = ir::ToThreeAddress(stmt, false, 0, true);
  // check1: out1(ax1, ax2) = b(ax1)
  std::vector<std::tuple<std::string, const air::ir::Provide *, uint64_t>> info1 =
      UTProvideCheckerForAssign().Find(stmt_out, "b(ax1)");
  ASSERT_EQ(info1.size(), 1);
  EXPECT_EQ(std::get<2>(info1[0]), 32 * 1024);
  std::string out1_name = std::get<0>(info1[0]);
  // check2: out(ax1, ax2) = a(ax1, ax2) * out1(ax1, ax2)
  std::vector<std::tuple<std::string, const air::ir::Provide *, uint64_t>> info2 =
      UTProvideCheckerForBinary().Find(
          stmt_out, UTProvideCheckerForBinary::BinaryOpType::kMul, "a(ax1, ax2)", out1_name);
  ASSERT_EQ(info2.size(), 1);
  EXPECT_EQ(std::get<2>(info2[0]), 32 * 1024);
  EXPECT_EQ(std::get<0>(info2[0]), "out(ax1, ax2)");
}

/*
 * TestCast: FloorTest for fp16->int32, fp32->int32
 */
class PassTestToThreeAddress_FloorTest {
 public:
  PassTestToThreeAddress_FloorTest(const air::DataType &type_dst, const air::DataType &type_src)
      : type_dst_(type_dst), type_src_(type_src) {
    Construct();
  }
  ~PassTestToThreeAddress_FloorTest() = default;

  void Construct() {
    vp_.AddVars({"ax1"});
    a_ = UTExprBuilder::PlaceholderOpNode("a", {1024}, type_src_);
    out_ = UTExprBuilder::PlaceholderOpNode("out", {1024}, type_dst_);
    stmt = air::ir::AttrStmt::make(
        out_, "", UTExprBuilder::IntImm(1),
        UTStmtBuilder::CreateRealizeByPlaceholderOp(
            out_,
            air::ir::ProducerConsumer::make(out_, true,
                UTStmtBuilder::CreateFor(
                    vp_.GetVar("ax1"), 0, 1024,
                    UTStmtBuilder::CreateProvideAssign(
                        out_, vp_.GetVars({"ax1"}),
                        air::ir::Cast::make(
                            type_dst_,
                            air::ir::Call::make(
                                type_src_,
                                "floor",
                                {UTExprBuilder::ElementOf(a_, vp_.GetVars({"ax1"}))},
                                air::ir::Call::CallType::PureIntrinsic)))))));
  }

  air::DataType type_dst_;
  air::DataType type_src_;
  UTVariablePool vp_;
  air::Operation a_;
  air::Operation out_;
  air::Stmt stmt;
};  // class PassTestToThreeAddress_FloorTest1

/*
 * fp16->int32
 * expect: out(ax1) = floor(a(ax1)):int32:PI
 */
TEST(PassTestToThreeAddress_FloorTest, TestPass_fp16_to_int32) {
  PassTestToThreeAddress_FloorTest test(air::Int(32), air::Float(16));
  Stmt stmt_out = ir::ToThreeAddress(test.stmt, false, 0, true);
  std::vector<std::tuple<std::string, const air::ir::Provide *, uint64_t>> info =
      UTProvideCheckerForAssign().Find(stmt_out, "floor(a(ax1)):int32:PI");
  ASSERT_EQ(info.size(), 1);
  EXPECT_EQ(std::get<0>(info[0]), "out(ax1)");
  EXPECT_EQ(std::get<2>(info[0]), 1024);
}

/*
 * fp32->int32
 * expect: out(ax1) = floor(a(ax1)):int32:PI
 */
TEST(PassTestToThreeAddress_FloorTest, TestPass_fp32_to_int32) {
  PassTestToThreeAddress_FloorTest test(air::Int(32), air::Float(32));
  Stmt stmt_out = ir::ToThreeAddress(test.stmt, false, 0, true);
  std::vector<std::tuple<std::string, const air::ir::Provide *, uint64_t>> info =
      UTProvideCheckerForAssign().Find(stmt_out, "floor(a(ax1)):int32:PI");
  ASSERT_EQ(info.size(), 1);
  EXPECT_EQ(std::get<0>(info[0]), "out(ax1)");
  EXPECT_EQ(std::get<2>(info[0]), 1024);
}
}  // namespace akg
