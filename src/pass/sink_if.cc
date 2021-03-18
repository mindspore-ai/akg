/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

/**
 *
 * This pass sinks the if stmt into for loop reasonably.
 *
 * For example, cases like this:
 *
 * for (i, 0, 16) {
 *     if ((i * 2) == var) {
 *         for (c, 0, 16) {
 *             out(i, c) = in(i, c)
 *         }
 *     }
 * }
 *
 * will be transformed into:
 *
 * for (i, 0, 16) {
 *     for (c, 0, 16) {
 *         if ((i * 2) == var) {
 *             out(i, c) = in(i, c)
 *         }
 *     }
 * }
 *
 */

#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm.h>

namespace akg {
namespace ir {
namespace {
class IFSinker : public IRMutator {
 public:
  IFSinker() : cond_() {}
  ~IFSinker() override = default;

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    if (IsSimpleSelect(op)) {
      return Encapsulate(s);
    }

    cond_.push_back(op->condition);
    Stmt if_then = this->Mutate(op->then_case);
    bool then_same = if_then.same_as(op->then_case);

    Stmt if_else;
    bool else_same = true;
    if (op->else_case.defined()) {
      cond_.back() = Not::make(cond_.back());
      if_else = this->Mutate(op->else_case);
      else_same = if_else.same_as(op->else_case);
    }
    cond_.pop_back();

    if (then_same && else_same) {
      return s;
    } else if (then_same) {
      return Block::make(IfThenElse::make(op->condition, if_then), if_else);
    } else {
      return op->else_case.defined() ? Block::make(if_then, if_else) : if_then;
    }
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final { return Encapsulate(s); }

 private:
  static bool IsSimpleSelect(const IfThenElse *op) {
    if (!op->then_case.defined() || !op->then_case.as<Provide>()) {
      return false;
    }

    if (!op->else_case.defined() || !op->else_case.as<Provide>()) {
      return false;
    }

    auto then_case = op->then_case.as<Provide>();
    auto else_case = op->else_case.as<Provide>();
    CHECK(then_case);
    CHECK(else_case);

    if (then_case->func != else_case->func) {
      return false;
    }

    if (then_case->args.size() != else_case->args.size()) {
      return false;
    }

    for (size_t dim = 0; dim < then_case->args.size(); ++dim) {
      if (!then_case->args[dim].same_as(else_case->args[dim])) {
        return false;
      }
    }

    return true;
  }

  Stmt Encapsulate(const Stmt &s) {
    if (cond_.empty()) {
      return s;
    }

    auto rst = s;
    for (auto it = cond_.rbegin(); it != cond_.rend(); ++it) {
      rst = IfThenElse::make(*it, rst);
    }

    return rst;
  }

  std::vector<Expr> cond_;
};
}  // namespace

Stmt SinkIfStmt(const Stmt &stmt) { return IFSinker().Mutate(stmt); }
}  // namespace ir
}  // namespace akg
