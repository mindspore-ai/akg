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
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/arithmetic.h>
#include <pass/ir_util.h>
#include "pass/utils.h"
#include "pass/rewrite_simplify_cce.h"

namespace akg {
namespace ir {
class LoopHoistDepender : public DataDepender {
 public:
  LoopHoistDepender(const IfThenElse *hoist_node, bool hoist_before, bool hoist_allocate = false)
      : hoist_node_(hoist_node), hoist_before_(hoist_before), hoist_allocate_(hoist_allocate) {}
  ~LoopHoistDepender() override = default;

  void Visit_(const Variable *op) override { VisitBuffer(op); }

  void Visit_(const Load *op) override { VisitBuffer(op); }

  void Visit_(const Store *op) override { VisitBuffer(op); }

  void Visit_(const Call *op) override { VisitBuffer(op); }

  void Visit_(const IfThenElse *op) override {
    if (op == hoist_node_) {
      has_found_ = true;
      return;
    }

    IRVisitor::Visit_(op);
  }

  void Visit_(const Allocate *op) override {
    // avoid use local buffer
    if (!hoist_allocate_) {
      def_.insert(op->buffer_var.get());
    }

    IRVisitor::Visit_(op);

    if (hoist_allocate_ && !has_found_) {
      def_.insert(op->buffer_var.get());
    }
  }

 private:
  template <typename T>
  void VisitBuffer(const T *op) {
    if ((hoist_before_ && !has_found_) || (!hoist_before_ && has_found_)) {
      DataDepender::Visit_(op);
    }
  }

  const IfThenElse *hoist_node_;
  bool hoist_before_;
  bool hoist_allocate_;
  bool has_found_{false};
};

class LoopSwitchHoister : public IRMutator {
 public:
  explicit LoopSwitchHoister(bool hoistAllocate = false) : hoist_allocate(hoistAllocate) {}
  ~LoopSwitchHoister() override = default;

  Stmt Mutate_(const For *op, const Stmt &s) final {
    loop_layer_[op->loop_var.get()] = loop_stack_.size();
    loop_stack_.emplace_back(LoopEntry{op, false});

    Stmt stmt = IRMutator::Mutate_(op, s);

    LoopEntry &entry = loop_stack_.back();
    if (!entry.hoist_before.empty()) {
      Stmt before = air::ir::MergeSeq(entry.hoist_before);
      stmt = Block::make(before, stmt);
    }

    if (!entry.hoist_after.empty()) {
      Stmt after = air::ir::MergeSeq(entry.hoist_after);
      stmt = Block::make(stmt, after);
    }

    if (hoist_allocate) {
      for (auto &allocate : entry.allocates) {
        stmt = Allocate::make(allocate->buffer_var, allocate->type, allocate->extents, allocate->condition, stmt);
        stmt = AttrStmt::make(allocate->buffer_var, "storage_scope", Expr("local.UB"), stmt);
      }
    }

    loop_stack_.pop_back();

    return stmt;
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    if (loop_stack_.empty()) {
      return IRMutator::Mutate_(op, s);
    }

    loop_stack_.back().branch_bound = true;
    Stmt stmt = IRMutator::Mutate_(op, s);
    loop_stack_.back().branch_bound = false;

    op = stmt.as<IfThenElse>();
    CHECK(op != nullptr);
    if (op->else_case.defined()) {
      return stmt;
    }

    global_layer = GetHoistLayer(op);
    if (global_layer < 0) {
      return stmt;
    }

    int dir = GetHoistDirection(op, global_layer);
    if (dir == 0) {
      return stmt;
    }

    bool hoist_before = dir > 0;
    LoopHoistDepender loop_dep(op, hoist_before, hoist_allocate);
    loop_dep.Visit(loop_stack_[global_layer].node->body);

    DataDepender if_dep;
    if_dep.Visit(op->then_case);
    if (if_dep.DependWith(loop_dep)) {
      return stmt;
    }

    Map<Var, Expr> vmap;
    for (size_t i = global_layer; i < loop_stack_.size(); ++i) {
      const For *loop = loop_stack_[i].node;
      vmap.Set(Var(loop->loop_var), hoist_before ? loop->min : loop->extent - 1);
    }

    Stmt hoist_stmt = air::ir::Substitute(op->then_case, vmap);
    if (hoist_before) {
      loop_stack_[global_layer].hoist_before.emplace_back(hoist_stmt);
    } else {
      loop_stack_[global_layer].hoist_after.emplace_back(hoist_stmt);
    }

    return Evaluate::make(0);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    // do not hoist in basic emit_insn block
    if (op->attr_key == "pragma_emit_insn") {
      return s;
    }

    if (op->attr_key == "storage_scope" && hoist_allocate) {
      if (op->value.as<StringImm>() && op->value.as<StringImm>()->value == "local.UB") {
        ub_allocate = true;
        auto stmt = IRMutator::Mutate_(op, s);
        ub_allocate = false;

        if (hoisted) {
          hoisted = false;

          CHECK(stmt.as<AttrStmt>());
          return stmt.as<AttrStmt>()->body;
        }

        return stmt;
      } else {
        bool before = ub_allocate;

        ub_allocate = false;
        auto stmt = IRMutator::Mutate_(op, s);
        ub_allocate = before;

        return stmt;
      }
    }

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Allocate *op, const Stmt &s) final {
    auto stmt = IRMutator::Mutate_(op, s);
    if (!hoist_allocate || !ub_allocate) {
      return stmt;
    }

    DataDepender allocate_dep;
    allocate_dep.Visit(stmt);
    if (allocate_dep.def_.count(op->buffer_var.get())) {
      hoisted = false;
      return stmt;
    }

    hoisted = true;
    loop_stack_[global_layer].allocates.push_back(op);

    CHECK(stmt.as<Allocate>());
    return stmt.as<Allocate>()->body;
  }

 private:
  int GetHoistLayer(const IfThenElse *op) {
    int target_layer = loop_stack_.size();
    auto get_target_layer = [&target_layer, this](const NodeRef &op) {
      if (target_layer < 0) return;

      const auto var = op.as<Variable>();
      if (var == nullptr) return;

      auto it = loop_layer_.find(var);
      if (it == loop_layer_.end()) {
        target_layer = -1;
      } else if (it->second < target_layer) {
        target_layer = it->second;
      }
    };

    PostOrderVisit(op->condition, get_target_layer);
    if (target_layer < 0 || (target_layer == static_cast<int>(loop_stack_.size()))) {
      return -1;
    }

    for (size_t i = target_layer; i < loop_stack_.size(); ++i) {
      if (loop_stack_[i].branch_bound) {
        return -1;
      }
    }

    return target_layer;
  }

  int GetHoistDirection(const IfThenElse *op, int layer) {
    Map<Var, Range> vmap;
    for (size_t i = layer; i < loop_stack_.size(); ++i) {
      const For *loop = loop_stack_[i].node;
      vmap.Set(Var(loop->loop_var), Range::make_by_min_extent(loop->min, loop->extent));
    }

    Expr not_hit_expr = op->condition == make_const(Int(32), 0);
    auto cannot_hit = [&vmap, &not_hit_expr](const For *loop, const Range &range) -> bool {
      Var loop_key = Var(loop->loop_var);
      Range prev = vmap[loop_key];
      vmap.Set(loop_key, range);
      Expr t = Simplify_cce(not_hit_expr, vmap);
      vmap.Set(loop_key, prev);

      return air::arith::Analyzer().CanProve(t);
    };

    bool pre_hoist = true;
    for (size_t i = layer; i < loop_stack_.size(); ++i) {
      const For *loop = loop_stack_[i].node;
      if (!cannot_hit(loop, Range::make_by_min_extent(loop->min + 1, loop->extent))) {
        pre_hoist = false;
        break;
      }
    }

    if (pre_hoist) return 1;

    for (size_t i = layer; i < loop_stack_.size(); ++i) {
      const For *loop = loop_stack_[i].node;
      if (!cannot_hit(loop, Range::make_by_min_extent(loop->min, loop->extent - 1))) {
        return 0;
      }
    }

    return -1;
  }

  struct LoopEntry {
    const For *node;
    bool branch_bound;
    std::vector<Stmt> hoist_before;
    std::vector<Stmt> hoist_after;
    std::vector<const Allocate *> allocates;
  };

  int global_layer{0};
  bool hoist_allocate{false};
  bool ub_allocate{false};
  bool hoisted{false};
  std::vector<LoopEntry> loop_stack_;
  std::unordered_map<const Variable *, int> loop_layer_;
};

Stmt LoopSwitchHoist(Stmt stmt, bool hoistAllocate) {
  Stmt prev;

  do {
    prev = stmt;
    stmt = LoopSwitchHoister(hoistAllocate).Mutate(stmt);
  } while (!stmt.same_as(prev));

  stmt = RemoveNoOp(stmt);

  return stmt;
}
}  // namespace ir
}  // namespace akg
