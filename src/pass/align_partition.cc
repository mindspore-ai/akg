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
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <arithmetic/int_set.h>
#include <runtime/thread_storage_scope.h>

#include <unordered_map>
#include <unordered_set>

#include "ir_pass.h"
#include "pass/zero_elimination.h"
#include "pass/utils.h"

namespace akg {
namespace ir {
class CandidateLoops : public IRVisitor {
 public:
  CandidateLoops() = default;
  ~CandidateLoops() override = default;

  void Visit_(const For *op) override {
    const Variable *var = op->loop_var.get();
    vrange_[var] = Range::make_by_min_extent(op->min, op->extent);

    loop_var_[op->loop_var->name_hint] = var;
    IRVisitor::Visit_(op);
    loop_var_.erase(op->loop_var->name_hint);

    if (loop_var_.empty() && select_) {
      candidates_[op] = select_;
      select_ = nullptr;
    }
  }

  void Visit_(const AttrStmt *op) override {
    if (op->attr_key == "reduce_update") {
      Array<IterVar> iv_arr = Downcast<Array<IterVar>>(op->node);
      const Var reduce_var = iv_arr[0]->var;
      auto it = loop_var_.find(reduce_var->name_hint);
      if (it != loop_var_.end()) {
        loop_var_.erase(it);
      }
    }

    IRVisitor::Visit_(op);
  }

  void Visit_(const Select *op) override {
    if (!select_ && !loop_var_.empty() && ProcCondition(op->condition)) {
      true_part_ = op->true_value;
      false_part_ = op->false_value;
      select_ = op;
    }

    IRVisitor::Visit_(op);
  }

  std::unordered_map<const Node *, const Node *> candidates_;
  const Node *select_{nullptr};
  const Variable *lhs_{nullptr};
  const Variable *rhs_{nullptr};
  Expr true_part_, false_part_;
  std::unordered_map<const Variable *, Range> vrange_;

 private:
  std::unordered_map<std::string, const Variable *> loop_var_;

  bool ProcCondition(const Expr &cond) {
    auto ProcEQ = [this](const Expr &a, const Expr &b) {
      // judge "a == b", both a and b are loop var.
      auto va = a.as<Variable>();
      auto vb = b.as<Variable>();
      if (va && loop_var_.count(va->name_hint) && vb && loop_var_.count(vb->name_hint)) {
        lhs_ = va;
        rhs_ = vb;

        return true;
      }

      return false;
    };

    auto ProcLT = [this](const Expr &a, const Expr &b) {
      // judge "a < b", a is loop var and b is imm.
      auto va = a.as<Variable>();
      if (va && (b.as<UIntImm>() || b.as<IntImm>())) {
        if ((va != lhs_) && (va != rhs_)) return false;

        if (CanProve(vrange_[va]->extent > b)) {
          vrange_[va] = Range::make_by_min_extent(vrange_[va]->min, b);
        }

        return true;
      }

      return false;
    };

    if (cond.as<EQ>()) {
      auto op = cond.as<EQ>();
      return ProcEQ(op->a, op->b);
    }

    if (cond.as<And>()) {
      auto op = cond.as<And>();
      // EQ must be processed before LT.
      if (op->a.as<EQ>() && op->b.as<LT>()) {
        auto eq_op = op->a.as<EQ>();
        auto lt_op = op->b.as<LT>();

        return ProcEQ(eq_op->a, eq_op->b) && ProcLT(lt_op->a, lt_op->b);
      }

      if (op->b.as<EQ>() && op->a.as<LT>()) {
        auto eq_op = op->b.as<EQ>();
        auto lt_op = op->a.as<LT>();

        return ProcEQ(eq_op->a, eq_op->b) && ProcLT(lt_op->a, lt_op->b);
      }
    }

    return false;
  }
};

// Replace the set of conditions given by ps with cond_value (true or false)
class SelectEliminatorCCE : public IRMutator {
 public:
  SelectEliminatorCCE(const Node *ps, const Expr cond_value) : ps_(ps), cond_value_(cond_value) {}
  ~SelectEliminatorCCE() override = default;

  using IRMutator::Mutate;
  Expr Mutate(Expr e) final {
    if (ps_ == e.get()) {
      return Mutate(cond_value_);
    }

    return IRMutator::Mutate(e);
  }

 private:
  const Node *ps_;
  Expr cond_value_;
};

class UpdateLoopRange : public IRMutator {
 public:
  UpdateLoopRange(const Variable *outer_var, const Range &outer_range, const Range &inner_range)
      : outer_var_(outer_var), outer_range_(outer_range), inner_range_(inner_range) {}
  ~UpdateLoopRange() override = default;

  Stmt Mutate_(const For *op, const Stmt &stmt) final {
    auto var = op->loop_var.as<Variable>();
    if (var == outer_var_) {
      auto r = RangeIntersect(outer_range_, inner_range_);
      if (CanProve((r->min != op->min) || (r->extent != op->extent))) {
        return For::make(op->loop_var, r->min, r->extent, op->for_type, op->device_api, op->body);
      }
    }

    return IRMutator::Mutate_(op, stmt);
  }

 private:
  Range RangeIntersect(const Range &a, const Range &b) {
    auto min = CanProve(a->min > b->min) ? a->min : b->min;
    auto extent = CanProve(a->extent < b->extent) ? a->extent : b->extent;

    return Range::make_by_min_extent(min, extent);
  }

  const Variable *outer_var_;
  const Range &outer_range_;
  const Range &inner_range_;
};

/* splits the loop based on the (==) conditions. Loops are splitted such that first entire matrix is computed and
 * in the next loop, only the diagonal elements are updated.
 *
 * === Example 1 ===
 * for i 0 16
 *   for j 0 16
 *     a[i, j] = D[i, j] * select(i==j, B[i, j] , C[i, j])
 * -->
 *  for i 0 16
 *    for j 0 16
 *       a[i, j] = D[i, j] * C[i, j]
 *  for i 0 16
 *    a[i, i] = D[i, i] * B[i, i]
 *
 * === Example 2 ===
 * for i 0 16
 *   for j 0 16
 *     a[i, j] = select(i==j && i<10, B[i, j] , C[i, j])
 * -->
 *  for i 0 16
 *    for j 0 16
 *       a[i, j] = C[i, j]
 *  for i 0 10
 *    a[i, i] = B[i, i]
 *
 * NOTE: Do not perform this transformation for reduction.
 * Presence of reduction is checked by checking `reduce_update` attr
 *
 * LIMITATIONS: Only works for one select inside the body of the for loop.
 *
 */
class AlignPartition : public IRMutator {
 public:
  AlignPartition() : selector_() {}
  ~AlignPartition() override = default;

  Stmt VisitAndMutate(const Stmt &stmt) {
    selector_.Visit(stmt);

    return Mutate(stmt);
  }

  Stmt Mutate_(const For *op, const Stmt &stmt) final {
    const Variable *var = op->loop_var.get();
    if (selector_.candidates_.count(op)) {
      for_op_ = op;
      Stmt new_stmt1 = SelectEliminatorCCE(selector_.candidates_[for_op_], selector_.false_part_).Mutate(stmt);
      inside_selector_for_ = true;
      if (!outer_var_ && ((var == selector_.lhs_) || (var == selector_.rhs_))) {
        outer_var_ = var;
        min_ = op->loop_var;
      }

      Stmt new_stmt2 = IRMutator::Mutate_(op, stmt);
      new_stmt2 =
        UpdateLoopRange(outer_var_, selector_.vrange_[outer_var_], selector_.vrange_[inner_var_]).Mutate(new_stmt2);
      inside_selector_for_ = false;
      for_op_ = nullptr;
      outer_var_ = nullptr;
      inner_var_ = nullptr;

      return air::ir::AppendStmts(new_stmt1, new_stmt2);
    } else if (inside_selector_for_) {
      if ((var == selector_.lhs_) || (var == selector_.rhs_)) {
        if (!outer_var_) {
          outer_var_ = var;
          min_ = op->loop_var;

          return IRMutator::Mutate_(op, stmt);
        } else if (!inner_var_) {
          inner_var_ = var;
          Stmt new_body = SelectEliminatorCCE(selector_.candidates_[for_op_], selector_.true_part_).Mutate(op->body);

          return Substitute(new_body, {{Var{op->loop_var}, min_}});
        }
      }
    }

    return IRMutator::Mutate_(op, stmt);
  }

 private:
  CandidateLoops selector_;
  const Node *for_op_{nullptr};
  bool inside_selector_for_{false};
  const Variable *outer_var_{nullptr};
  const Variable *inner_var_{nullptr};
  Expr min_;
};

Stmt AlignPartitionCCE(const Stmt stmt) { return AlignPartition().VisitAndMutate(stmt); }
}  // namespace ir
}  // namespace akg
