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
#ifndef UT_IR_CHECKER_H_
#define UT_IR_CHECKER_H_
#include <string>
#include <tuple>
#include <vector>
#include <tvm/ir_visitor.h>
#include "base/dump_helper.h"
#include "base/expr_builder.h"

namespace akg {
class UTIRCheckHelper {
 public:
  UTIRCheckHelper() = default;
  ~UTIRCheckHelper() = default;
  static int64_t GetValueFromImm(const air::Expr &expr);
};  // class UTIRCheckHelper

class UTProvideChecker : public air::ir::IRVisitor {
 public:
  explicit UTProvideChecker(bool ignore_args = false)
      : ignore_args_(ignore_args) {}
  ~UTProvideChecker() = default;
  void Visit_(const air::ir::For *op) override;
  bool CompareDump(const std::string &dump, const std::string &target);

 protected:
  bool ignore_args_{false};
  std::vector<uint64_t> for_count_stack_;
};  // class UTProvideChecker

class UTProvideCheckerForAssign : public UTProvideChecker {
 public:
  explicit UTProvideCheckerForAssign(bool ignore_args = false)
      : UTProvideChecker(ignore_args) {}
  ~UTProvideCheckerForAssign() = default;
  std::vector<std::tuple<std::string, const air::ir::Provide*, uint64_t>> Find(
      const air::NodeRef &node,
      const std::string &dump_rhs);
  void Visit_(const air::ir::Provide *op) override;

 private:
  std::string dump_rhs_{""};
  std::vector<std::tuple<std::string, const air::ir::Provide*, uint64_t>> infos_lhs_;
};  // class UTProvideChecker

class UTProvideCheckerForBinary : public UTProvideChecker {
 public:
  enum BinaryOpType : int {
    kAdd,
    kSub,
    kMul,
    kDiv,
    kMod,
  };

  explicit UTProvideCheckerForBinary(bool ignore_args = false)
      : UTProvideChecker(ignore_args) {}
  ~UTProvideCheckerForBinary() = default;
  std::vector<std::tuple<std::string, const air::ir::Provide*, uint64_t>> Find(
      const air::NodeRef &node,
      BinaryOpType op_type,
      const std::string &dump_rhs1,
      const std::string &dump_rhs2);
  void Visit_(const air::ir::Provide *op) override;

  template <typename T>
  void CheckBinary(const air::ir::Provide *op) {
    const T *expr_binary = op->value.as<T>();
    if (expr_binary == nullptr) {
      return;
    }
    std::string dump_expr_a = UTDumpHelper::Dump(expr_binary->a);
    std::string dump_expr_b = UTDumpHelper::Dump(expr_binary->b);
    if ((dump_rhs1_.empty() || CompareDump(dump_expr_a, dump_rhs1_)) &&
        (dump_rhs2_.empty() || CompareDump(dump_expr_b, dump_rhs2_))) {
      air::Expr expr_call = UTExprBuilder::CreateCall(op->func, op->args);
      infos_lhs_.push_back(std::make_tuple(UTDumpHelper::Dump(expr_call), op, for_count_stack_.back()));
    }
  }

 private:
  BinaryOpType op_type_;
  std::string dump_rhs1_{""};
  std::string dump_rhs2_{""};
  std::vector<std::tuple<std::string, const air::ir::Provide*, uint64_t>> infos_lhs_;
};  // class UTProvideCheckerForBinary
}  // namespace akg
#endif  // UT_IR_CHECKER_H_
