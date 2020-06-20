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

#include <tvm/ir_mutator.h>
#include <ir_pass.h>
#include <pass/ir_util.h>
#include <algorithm>

namespace akg {
namespace ir {
/*
 * Eliminate if which is in for loop
 * eg:
 * for (i, 0, 16)
 *   if (i >= 5)
 *     i + j
 * ==>
 * for (i, 0, 11)
 *   i + 5 + j
 */

enum CmpType { CmpInvalid = 0, CmpGE, CmpGT, CmpLE, CmpLT };

class IFEliminater : public IRMutator {
 public:
  IFEliminater() {}
  ~IFEliminater() override = default;

  Stmt Mutate(Stmt stmt) final {
    stmt = IRMutator::Mutate(stmt);
    return stmt;
  }

  Expr Mutate(Expr expr) final {
    expr = IRMutator::Mutate(expr);
    return expr;
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    auto cond = op->condition;
    auto then = op->then_case;
    auto els = op->else_case;

    if (!in_attr_stmt || els.defined()) {
      return IRMutator::Mutate_(op, s);
    }

    ++level;

    if_cond = true;
    static_cast<void>(this->Mutate(cond));
    current_var_ = nullptr;
    if_cond = false;

    for (auto iter = loop_vars_extents_.begin(); iter != loop_vars_extents_.end(); ++iter) {
      if (cond_vars_const_bounds_.count(iter->first) != 0) {
        auto extent = iter->second;
        if (cond->IsInstance<GE>()) {
          auto ge = cond.as<GE>();
          // if condition is not variable,  don't eliminate it
          if (ge->a->IsInstance<Variable>()) {
            iter->second = Simplify_cce(Sub::make(extent, cond_vars_const_bounds_[iter->first]));
            cmp_type = CmpGE;
          }
        } else if (cond->IsInstance<GT>()) {
          auto gt = cond.as<GT>();
          if (gt->a->IsInstance<Variable>()) {
            iter->second = Simplify_cce(Sub::make(extent, Min::make(cond_vars_const_bounds_[iter->first], Expr(1))));
            cmp_type = CmpGT;
          }
        } else if (cond->IsInstance<LE>()) {
          auto le = cond.as<LE>();
          if (le->a->IsInstance<Variable>()) {
            iter->second = Simplify_cce(Add::make(cond_vars_const_bounds_[iter->first], Expr(1)));
            cmp_type = CmpLE;
          }
        } else if (cond->IsInstance<LT>()) {
          auto lt = cond.as<LT>();
          if (lt->a->IsInstance<Variable>()) {
            iter->second = cond_vars_const_bounds_[iter->first];
            cmp_type = CmpLT;
          }
        }
      }
    }

    if_body = true;
    then = this->Mutate(then);
    if_body = false;

    std::vector<const Variable *> erase_list;
    for (auto iter : cond_level_) {
      if (iter.second == level) {
        erase_list.push_back(iter.first);
      }
    }

    for (auto var : erase_list) {
      cond_level_.erase(var);
      cond_vars_const_bounds_.erase(var);
    }

    --level;
    if (cmp_type == CmpInvalid) {
      return IRMutator::Mutate_(op, s);
    }

    return then;
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_emit_insn" && op->value.as<StringImm>() != nullptr &&
        op->value.as<StringImm>()->value == "dma_copy") {
      in_attr_stmt = true;
    }

    Stmt stmt = IRMutator::Mutate_(op, s);
    in_attr_stmt = false;

    return stmt;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (!in_attr_stmt) {
      return IRMutator::Mutate_(op, s);
    }

    loop_vars_extents_[op->loop_var.get()] = op->extent;

    auto body = this->Mutate(op->body);

    Stmt stmt =
      For::make(op->loop_var, op->min, loop_vars_extents_[op->loop_var.get()], op->for_type, op->device_api, body);
    loop_vars_extents_.erase(op->loop_var.get());

    return stmt;
  }

  Expr Mutate_(const Variable *op, const Expr &e) final {
    if (if_cond) {
      cond_vars_const_bounds_[op] = Expr(0);
      cond_level_[op] = level;
      current_var_ = op;
    } else if (if_body) {
      if (cond_vars_const_bounds_.count(op) != 0) {
        auto expr = e;
        switch (cmp_type) {
          case CmpGE:
            expr = Add::make(e, cond_vars_const_bounds_[op]);
            expr = IRMutator::Mutate_(op, expr);
            break;
          case CmpGT:
            expr = Add::make(e, Add::make(cond_vars_const_bounds_[op], Expr(1)));
            expr = IRMutator::Mutate_(op, expr);
            break;
          default:
            break;
        }

        return expr;
      }
    }

    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const IntImm *op, const Expr &e) final {
    if (if_cond && current_var_ != nullptr) {
      cond_vars_const_bounds_[current_var_] = e;
      current_var_ = nullptr;
    }
    return IRMutator::Mutate_(op, e);
  }

 private:
  bool if_cond{false};
  bool if_body{false};
  bool in_attr_stmt{false};
  int level{0};
  CmpType cmp_type{CmpInvalid};
  const Variable *current_var_{nullptr};
  std::unordered_map<const Variable *, Expr> loop_vars_extents_;
  std::unordered_map<const Variable *, Expr> cond_vars_const_bounds_;
  std::unordered_map<const Variable *, int> cond_level_;
};

Stmt EliminateIf(Stmt stmt) {
  stmt = IFEliminater().Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
