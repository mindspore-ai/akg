/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * Delete condition of out most axis to enable multicore.
 * For example, cases like this:
 *
 * for (c, 0, 32) {
 *     for (i, 0, 4) {
 *         if (c == 0) {
 *             out2(i) = 3
 *         }
 *         out(c, i) = in(c, i)
 *     }
 * }
 *
 * will be transformed into:
 *
 * for (c, 0, 32) {
 *     for (i, 0, 4) {
 *         out2(i) = 3
 *         out(c, i) = in(c, i)
 *     }
 * }
 *
 */

#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <ir_pass.h>
#include <pass/ir_util.h>
#include <tvm/expr_operator.h>

namespace akg {
namespace ir {
class EleminateOutmostForCond : public IRMutator {
 public:
  EleminateOutmostForCond() = default;
  ~EleminateOutmostForCond() override = default;

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (should_eleminate_cond_) {
      return IRMutator::Mutate_(op, s);
    } else {
      should_eleminate_cond_ = true;
      outmost_for_ = op;
      return IRMutator::Mutate_(op, s);
    }
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    if (outmost_for_ != nullptr) {
      should_mod_if_expr = true;
      auto condition = this->Mutate(op->condition);
      should_mod_if_expr = false;
      auto then_case = this->Mutate(op->then_case);
      auto else_case = op->else_case;
      if (else_case.defined()) {
        else_case = this->Mutate(op->else_case);
      }
      return IfThenElse::make(condition, then_case, else_case);
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const EQ *op, const Expr &e) final {
    if (should_mod_if_expr && (op->a.same_as(outmost_for_->loop_var) || op->b.same_as(outmost_for_->loop_var))) {
      return make_const(Bool(), 1);
    }

    return IRMutator::Mutate_(op, e);
  }

 private:
  bool should_eleminate_cond_{false};
  bool should_mod_if_expr{false};
  const For *outmost_for_{nullptr};
};

Stmt PreProcess4Multicore(Stmt stmt) { return EleminateOutmostForCond().Mutate(std::move(stmt)); }
}  // namespace ir
}  // namespace akg
