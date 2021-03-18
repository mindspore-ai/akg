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
#include "pass_test_base/auto_poly_test_base.h"
#define private public
#define protected public
#include "ir_pass.h"
#undef protected
#undef private
#include "codegen/util.h"
#include "contrib/cce_parm/cceconf.h"

namespace akg {
/* AutoPolyTest1: test for to_three_address
 * Input pattern:
 * for (i0, 0, 32) {
 *   for (i1, 0, 1024) {
 *     out_0(i1) = b(i1) + c(i1)
 *     out(i0, i1) = out_0(i1) + a(i0, i1)
 *   }
 * }
 *
 * Expect output:
 * for (cc1, 0, 2) {
 *   for (cc2, 0, 16) {
 *     for (cc3, 0, 1024) {
 *       out_0_local_UB(cc2) = (b_local_UB(cc2) + c_local_UB(cc2))
 *       out_local_UB(cc3, cc2) = (out_0_local_UB(cc2) + a_local_UB(cc3, cc2))
 *     }
 *   }
 * }
 *
 * IR Check:
 *   count for (b_local_UB + c_local_UB): 32 * 1024
 */
class AutoPolyTest1 : public AutoPolyTestBase {
 public:
  AutoPolyTest1() {
    Construct();
  }
  ~AutoPolyTest1() = default;
  void Construct() {
    vp_.AddVars({"i0", "i1"});
    a_ = UTExprBuilder::PlaceholderOpNode("a", {32, 1024}, air::Float(16));
    b_ = UTExprBuilder::PlaceholderOpNode("b", {1024}, air::Float(16));
    c_ = UTExprBuilder::PlaceholderOpNode("c", {1024}, air::Float(16));
    out_ = UTExprBuilder::PlaceholderOpNode("out", {32, 1024}, air::Float(16));
    out_0_ = UTExprBuilder::PlaceholderOpNode("out_0", {1024}, air::Float(16));
    stmt_ = air::ir::AttrStmt::make(
        out_0_, "realize_scope", air::ir::StringImm::make(""),
        UTStmtBuilder::CreateRealizeByPlaceholderOp(
            out_0_,
            air::ir::AttrStmt::make(
                out_, "realize_scope", air::ir::StringImm::make(""),
                UTStmtBuilder::CreateRealizeByPlaceholderOp(
                    out_,
                    air::ir::ProducerConsumer::make(out_, true,
                        UTStmtBuilder::CreateFor(
                            vp_.GetVar("i0"), 0, 32,
                            UTStmtBuilder::CreateFor(
                                vp_.GetVar("i1"), 0, 1024,
                                air::ir::Block::make(
                                    UTStmtBuilder::CreateProvideBinary<air::ir::Add>(
                                        out_0_, vp_.GetVars({"i1"}),
                                        UTExprBuilder::ElementOf(b_, vp_.GetVars({"i1"})),
                                        UTExprBuilder::ElementOf(c_, vp_.GetVars({"i1"}))),
                                    UTStmtBuilder::CreateProvideBinary<air::ir::Add>(
                                        out_, vp_.GetVars({"i0", "i1"}),
                                        UTExprBuilder::ElementOf(out_0_, vp_.GetVars({"i1"})),
                                        UTExprBuilder::ElementOf(a_, vp_.GetVars({"i0", "i1"})))))))))));
    t_a_ = UTExprBuilder::CreateTensorByPlaceholder(a_);
    t_b_ = UTExprBuilder::CreateTensorByPlaceholder(b_);
    t_c_ = UTExprBuilder::CreateTensorByPlaceholder(c_);
    t_out_ = UTExprBuilder::CreateTensorByPlaceholder(out_);
    RegisterTensor(t_a_);
    RegisterTensor(t_b_);
    RegisterTensor(t_c_);
    RegisterTensor(t_out_);
  }

  UTVariablePool vp_;
  air::Operation a_;
  air::Operation b_;
  air::Operation c_;
  air::Tensor t_a_;
  air::Tensor t_b_;
  air::Tensor t_c_;
  air::Operation out_;
  air::Tensor t_out_;
  air::Operation out_0_;
  air::Stmt stmt_;
};  // class AutoPolyTest1

TEST_F(AutoPolyTest1, RunPass) {
  SetRunMode("cloud");
  air::Array<air::NodeRef> stmts_out = ir::AutoPoly(stmt_, binds_, "cce", global_attrs_, false, false);
  ASSERT_EQ(stmts_out.size(), 2);
  air::NodeRef stmt = stmts_out[0];
  std::vector<std::tuple<std::string, const air::ir::Provide*, uint64_t>> infos_lhs =
      UTProvideCheckerForBinary(true).Find(
          stmt, UTProvideCheckerForBinary::BinaryOpType::kAdd, "b_local_UB", "c_local_UB");
  ASSERT_EQ(infos_lhs.size(), 1);
  EXPECT_EQ(std::get<2>(infos_lhs[0]), 2 * 16 * 1024);
}

/* AutoPolyTest2: test for to_three_address
 * Input pattern:
 * for (i1, 0, 32) {
 *   out_0(i1) = b(i1) + c(i1)
 *   for (i0, 0, 1024) {
 *     out(i0, i1) = out_0(i1) + a(i0, i1)
 *   }
 * }
 *
 * Expect output:
 * for (cc1, 0, 2) {
 *   for (cc2, 0, 1024) {
 *     out_0_local_UB(cc2) = (b_local_UB(cc2) + c_local_UB(cc2))
 *   }
 *   for (cc2, 0, 1024) {
 *     for (cc3, 0, 16) {
 *       out_local_UB(cc3, cc2) = (out_0_local_UB(cc2) + a_local_UB(cc3, cc2))
 *     }
 *   }
 * }
 *
 * IR Check:
 *   count for (b_local_UB + c_local_UB): 2 * 1024
 */
class AutoPolyTest2 : public AutoPolyTestBase {
 public:
  AutoPolyTest2() {
    Construct();
  }
  ~AutoPolyTest2() = default;
  void Construct() {
    vp_.AddVars({"i0", "i1"});
    a_ = UTExprBuilder::PlaceholderOpNode("a", {32, 1024}, air::Float(16));
    b_ = UTExprBuilder::PlaceholderOpNode("b", {1024}, air::Float(16));
    c_ = UTExprBuilder::PlaceholderOpNode("c", {1024}, air::Float(16));
    out_ = UTExprBuilder::PlaceholderOpNode("out", {32, 1024}, air::Float(16));
    out_0_ = UTExprBuilder::PlaceholderOpNode("out_0", {1024}, air::Float(16));
    stmt_ = air::ir::AttrStmt::make(
        out_0_, "realize_scope", air::ir::StringImm::make(""),
        UTStmtBuilder::CreateRealizeByPlaceholderOp(
            out_0_,
            air::ir::AttrStmt::make(
                out_, "realize_scope", air::ir::StringImm::make(""),
                UTStmtBuilder::CreateRealizeByPlaceholderOp(
                    out_,
                    air::ir::ProducerConsumer::make(out_, true,
                        UTStmtBuilder::CreateFor(
                            vp_.GetVar("i1"), 0, 1024,
                            air::ir::Block::make(
                                UTStmtBuilder::CreateProvideBinary<air::ir::Add>(
                                    out_0_, vp_.GetVars({"i1"}),
                                    UTExprBuilder::ElementOf(b_, vp_.GetVars({"i1"})),
                                    UTExprBuilder::ElementOf(c_, vp_.GetVars({"i1"}))),
                                UTStmtBuilder::CreateFor(
                                    vp_.GetVar("i0"), 0, 32,
                                    UTStmtBuilder::CreateProvideBinary<air::ir::Add>(
                                        out_, vp_.GetVars({"i0", "i1"}),
                                        UTExprBuilder::ElementOf(out_0_, vp_.GetVars({"i1"})),
                                        UTExprBuilder::ElementOf(a_, vp_.GetVars({"i0", "i1"})))))))))));
    t_a_ = UTExprBuilder::CreateTensorByPlaceholder(a_);
    t_b_ = UTExprBuilder::CreateTensorByPlaceholder(b_);
    t_c_ = UTExprBuilder::CreateTensorByPlaceholder(c_);
    t_out_ = UTExprBuilder::CreateTensorByPlaceholder(out_);
    RegisterTensor(t_a_);
    RegisterTensor(t_b_);
    RegisterTensor(t_c_);
    RegisterTensor(t_out_);
  }

  UTVariablePool vp_;
  air::Operation a_;
  air::Operation b_;
  air::Operation c_;
  air::Tensor t_a_;
  air::Tensor t_b_;
  air::Tensor t_c_;
  air::Operation out_;
  air::Tensor t_out_;
  air::Operation out_0_;
  air::Stmt stmt_;
};  // class AutoPolyTest2

TEST_F(AutoPolyTest2, RunPass) {
  SetRunMode("cloud");
  air::Array<air::NodeRef> stmts_out = ir::AutoPoly(stmt_, binds_, "cce", global_attrs_, false, false);
  ASSERT_EQ(stmts_out.size(), 2);
  air::NodeRef stmt = stmts_out[0];
  std::vector<std::tuple<std::string, const air::ir::Provide*, uint64_t>> infos_lhs =
      UTProvideCheckerForBinary(true).Find(
          stmt, UTProvideCheckerForBinary::BinaryOpType::kAdd, "b_local_UB", "c_local_UB");
  ASSERT_EQ(infos_lhs.size(), 1);
  EXPECT_EQ(std::get<2>(infos_lhs[0]), 2 * 1024);
}

/* AutoPolyTest3: test for to_three_address
 * Input pattern:
 * for (i0, 0, 1024) {
 *   out_0(i0) = b(i0) + c(i0)
 * }
 * for (i1, 0, 32) {
 *   for (i0, 0, 1024) {
 *     out(i0, i1) = out_0(i1) + a(i0, i1)
 *   }
 * }
 *
 * Expect output:
 * for (cc1, 0, 2) {
 *   for (cc2, 0, 1024) {
 *     out_0_local_UB(cc2) = (b_local_UB(cc2) + c_local_UB(cc2))
 *   }
 *   for (cc2, 0, 1024) {
 *     for (cc3, 0, 16) {
 *       out_local_UB(cc3, cc2) = (out_0_local_UB(cc2) + a_local_UB(cc3, cc2))
 *     }
 *   }
 * }
 *
 * IR Check:
 *   count for (b_local_UB + c_local_UB): 2 * 1024
 */
class AutoPolyTest3 : public AutoPolyTestBase {
 public:
  AutoPolyTest3() {
    Construct();
  }
  ~AutoPolyTest3() = default;
  void Construct() {
    vp_.AddVars({"i0", "i1"});
    a_ = UTExprBuilder::PlaceholderOpNode("a", {32, 1024}, air::Float(16));
    b_ = UTExprBuilder::PlaceholderOpNode("b", {1024}, air::Float(16));
    c_ = UTExprBuilder::PlaceholderOpNode("c", {1024}, air::Float(16));
    out_ = UTExprBuilder::PlaceholderOpNode("out", {32, 1024}, air::Float(16));
    out_0_ = UTExprBuilder::PlaceholderOpNode("out_0", {1024}, air::Float(16));
    stmt_ = air::ir::AttrStmt::make(
        out_0_, "realize_scope", air::ir::StringImm::make(""),
        UTStmtBuilder::CreateRealizeByPlaceholderOp(
            out_0_,
            air::ir::AttrStmt::make(
                out_, "realize_scope", air::ir::StringImm::make(""),
                UTStmtBuilder::CreateRealizeByPlaceholderOp(
                    out_,
                    air::ir::ProducerConsumer::make(out_, true,
                        air::ir::Block::make(
                            UTStmtBuilder::CreateFor(
                                vp_.GetVar("i0"), 0, 1024,
                                UTStmtBuilder::CreateProvideBinary<air::ir::Add>(
                                    out_0_, vp_.GetVars({"i0"}),
                                    UTExprBuilder::ElementOf(b_, vp_.GetVars({"i0"})),
                                    UTExprBuilder::ElementOf(c_, vp_.GetVars({"i0"})))),
                            UTStmtBuilder::CreateFor(
                                vp_.GetVar("i0"), 0, 32,
                                UTStmtBuilder::CreateFor(
                                    vp_.GetVar("i1"), 0, 1024,
                                    UTStmtBuilder::CreateProvideBinary<air::ir::Add>(
                                        out_, vp_.GetVars({"i0", "i1"}),
                                        UTExprBuilder::ElementOf(out_0_, vp_.GetVars({"i1"})),
                                        UTExprBuilder::ElementOf(a_, vp_.GetVars({"i0", "i1"})))))))))));
    t_a_ = UTExprBuilder::CreateTensorByPlaceholder(a_);
    t_b_ = UTExprBuilder::CreateTensorByPlaceholder(b_);
    t_c_ = UTExprBuilder::CreateTensorByPlaceholder(c_);
    t_out_ = UTExprBuilder::CreateTensorByPlaceholder(out_);
    RegisterTensor(t_a_);
    RegisterTensor(t_b_);
    RegisterTensor(t_c_);
    RegisterTensor(t_out_);
  }

  UTVariablePool vp_;
  air::Operation a_;
  air::Operation b_;
  air::Operation c_;
  air::Tensor t_a_;
  air::Tensor t_b_;
  air::Tensor t_c_;
  air::Operation out_;
  air::Tensor t_out_;
  air::Operation out_0_;
  air::Stmt stmt_;
};  // class AutoPolyTest3

TEST_F(AutoPolyTest3, RunPass) {
  SetRunMode("cloud");
  air::Array<air::NodeRef> stmts_out = ir::AutoPoly(stmt_, binds_, "cce", global_attrs_, false, false);
  ASSERT_EQ(stmts_out.size(), 2);
  air::NodeRef stmt = stmts_out[0];
  std::vector<std::tuple<std::string, const air::ir::Provide*, uint64_t>> infos_lhs =
      UTProvideCheckerForBinary(true).Find(
          stmt, UTProvideCheckerForBinary::BinaryOpType::kAdd, "b_local_UB", "c_local_UB");
  ASSERT_EQ(infos_lhs.size(), 1);
  EXPECT_EQ(std::get<2>(infos_lhs[0]), 2 * 1024);
}
}  // namespace akg
