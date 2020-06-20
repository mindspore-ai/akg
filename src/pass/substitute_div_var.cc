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

#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <ir_pass.h>

namespace akg {
namespace ir {
class DivVarMutator : public IRMutator {
 public:
  DivVarMutator() = default;
  ~DivVarMutator() override = default;

  Stmt Run(Stmt stmt) {
    auto new_stmt = IRMutator::Mutate(stmt);
    if (new_let_stmts_.empty()) {
      return new_stmt;
    }

    for (auto it = new_let_stmts_.rbegin(); it != new_let_stmts_.rend(); it++) {
      new_stmt = LetStmt::make(it->first, it->second, new_stmt);
    }
    return new_stmt;
  }

 private:
  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    auto call = op->value.as<Call>();
    if (call && call->name == "divide_var") {
      CHECK_EQ(call->args.size(), 2);
      auto ori_type = call->args[0].type();
      auto tmp_div = Div::make(FloatImm::make(Float(32), 1.0), Cast::make(Float(32), call->args[1]));
      auto tmp_var = Variable::make(Float(32), "DIVISOR_" + std::to_string(id_++));

      new_let_stmts_.push_back(std::make_pair(tmp_var, tmp_div));
      if (ori_type != Float(32)) {
        // scalar unit does not support float16
        CHECK(tmp_var.as<Variable>());
        auto cast_to_fp16 = Cast::make(Float(16), tmp_var);
        tmp_var = Variable::make(Float(16), tmp_var.as<Variable>()->name_hint + "_float16");
        new_let_stmts_.push_back(std::make_pair(tmp_var, cast_to_fp16));
      }
      return Provide::make(op->func, op->value_index, Mul::make(call->args[0], tmp_var), op->args);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  int id_{0};
  std::vector<std::pair<Var, Expr>> new_let_stmts_;
};

Stmt SubstituteDivVar(Stmt stmt) { return DivVarMutator().Run(stmt); }
}  // namespace ir
}  // namespace akg
