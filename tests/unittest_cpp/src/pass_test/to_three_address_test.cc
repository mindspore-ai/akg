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
#include "base/expr_builder.h"
#include "base/dump_helper.h"
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
  using Add = ktvm::ir::Add;
  // a(ax1, ax2) + b(ax2) + c(ax0, ax1, ax2) + d(ax2)
  ktvm::Expr expr =
      Add::make(
          Add::make(
              Add::make(th.Elem("a", 2), th.Elem("b", 1)),
              th.Elem("c", 3)),
          th.Elem("d", 1));
  std::string dump_expr = UTDumpHelper::Dump(expr);
  EXPECT_EQ(dump_expr, "(((a(ax1, ax2) + b(ax2)) + c(ax0, ax1, ax2)) + d(ax2))");
}

class ThreeAddressExprMutatorTest : public testing::Test {
 public:
  ThreeAddressExprMutatorTest()
      : mutator_(ktvm::TensorNode::make(
                    UTExprBuilder::CreateShape(shape_),               // shape
                    dtype_,                                           // dtype
                    UTExprBuilder::PlaceholderOpNode("out", shape_),  // op
                    0),                                               // index
                UTExprBuilder::CreateVars({"ax0", "ax1", "ax2"}),     // args
                UTExprBuilder::CreateShape(shape_),                   // shape
                std::unordered_set<const Call *>(),                   // broadcast
                false,                                                // IsReductionOp
                false) {}                                             // cross_stmt_simplify
  ~ThreeAddressExprMutatorTest() = default;

  std::vector<int32_t> shape_ = {16, 32, 1024};
  ktvm::DataType dtype_ = ktvm::Float(16);
  ir::ThreeAddressExprMutator mutator_;
};  // ThreeAddressExprMutatorTest

TEST_F(ThreeAddressExprMutatorTest, MutateBinaryOp_Add) {
  UTTensorElementHelper th(shape_);
  using Add = ktvm::ir::Add;
  ktvm::Expr expr = Add::make(th.Elem("a", 2), th.Elem("b", 1));
  Expr expr_m = mutator_.Mutate(expr);
  EXPECT_NE(mutator_.imm_ops.size(), 0);
}
}  // namespace akg
