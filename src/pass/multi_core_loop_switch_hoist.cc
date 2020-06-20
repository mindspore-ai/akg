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
#include <tvm/ir_pass.h>
#include <ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/arithmetic.h>
#include <pass/ir_util.h>
#include <pass/utils.h>

namespace akg {
namespace ir {
class MultiCoreLoopHoistDepender : public DataDepender {
 public:
  MultiCoreLoopHoistDepender(const IfThenElse *hoist_node, bool hoist_before)
      : hoist_node_(hoist_node), hoist_before_(hoist_before) {}
  ~MultiCoreLoopHoistDepender() override = default;

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
    IRVisitor::Visit_(op);
    if (!has_found_) {
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
  bool has_found_{false};
};

class MultiCoreLoopSimplify : public IRMutator {
 public:
  explicit MultiCoreLoopSimplify(const std::unordered_map<const Variable *, Expr> &outer) : outer_table_(outer) {}
  ~MultiCoreLoopSimplify() override = default;

  Stmt Mutate_(const For *op, const Stmt &s) final {
    inner_table_[op->loop_var.get()] = op->extent;
    Stmt body = Mutate(op->body);
    auto extent = op->extent;
    if (NeedUpdate()) {
      extent = make_const(op->extent.type(), 1);
      if (outer_table_.count(unfind_var_) > 0) {
        extent = outer_table_[unfind_var_];
      }
      extent = Simplify(Mul::make(extent, op->extent));
      Init();
    }
    return For::make(op->loop_var, op->min, extent, op->for_type, op->device_api, body);
  }

  Expr Mutate_(const Load *op, const Expr &e) final {
    auto index = this->Mutate(op->index);
    return Load::make(op->type, op->buffer_var, Simplify(index), op->predicate);
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    auto value = Mutate(op->value);
    if (NeedUpdate()) {
      if (stores_.count(op->buffer_var->name_hint) == 0 && outer_table_.count(unfind_var_) > 0) {
        stores_[op->buffer_var->name_hint] = outer_table_[unfind_var_];
      }
    }
    return Store::make(op->buffer_var, value, op->index, op->predicate);
  }

  Expr Mutate_(const Variable *op, const Expr &e) final {
    if (inner_table_.count(op) == 0) {
      need_multiply_ = true;
      unfind_var_ = op;
      return Expr(0);
    }
    return IRMutator::Mutate_(op, e);
  }

  bool NeedUpdate() const {
    if (need_multiply_ && unfind_var_ != nullptr) {
      return true;
    }
    return false;
  }

  void Init() {
    need_multiply_ = false;
    unfind_var_ = nullptr;
  }

  bool need_multiply_{false};
  const Variable *unfind_var_{nullptr};
  std::unordered_map<std::string, Expr> stores_;
  std::unordered_map<const Variable *, Expr> outer_table_;
  std::unordered_map<const Variable *, Expr> inner_table_;
};

class MultiCoreLoopHoister : public IRMutator {
 public:
  MultiCoreLoopHoister() {}
  ~MultiCoreLoopHoister() override = default;

  Array<Expr> BroadCaseExtents(Array<Expr> extents, Expr mul) {
    Array<Expr> res;
    for (size_t i = 0; i < extents.size(); ++i) {
      if (i == extents.size() - 1) {
        res.push_back(Simplify(Mul::make(mul, extents[i])));
      } else {
        res.push_back(extents[i]);
      }
    }
    return res;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (multi_core_) {
      multicore_loop_.push_back(op);
      multi_core_ = false;
    }
    loop_layer_[op->loop_var.get()] = loop_stack_.size();
    loop_size_[op->loop_var.get()] = op->extent;
    loop_stack_.emplace_back(LoopEntry{op, false});
    Stmt stmt = IRMutator::Mutate_(op, s);
    LoopEntry &entry = loop_stack_.back();
    if (!entry.hoist_in.empty()) {
      Stmt before = ktvm::ir::MergeSeq(entry.hoist_in);
      MultiCoreLoopSimplify band_simplify(loop_size_);
      before = band_simplify.Mutate(before);
      stmt = Block::make(before, stmt);
      for (auto &allocate : entry.allocates) {
        auto extents = allocate->extents;
        if (band_simplify.stores_.count(allocate->buffer_var->name_hint) > 0) {
          extents = BroadCaseExtents(extents, band_simplify.stores_[allocate->buffer_var->name_hint]);
        }
        stmt = Allocate::make(allocate->buffer_var, allocate->type, extents, allocate->condition, stmt);
        stmt = AttrStmt::make(allocate->buffer_var, "storage_scope", Expr("local.UB"), stmt);
      }
    }
    loop_stack_.pop_back();
    return stmt;
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    if (loop_stack_.empty() || !need_hoisted_) {
      return IRMutator::Mutate_(op, s);
    }
    loop_stack_.back().branch_bound = true;
    Stmt stmt = IRMutator::Mutate_(op, s);
    loop_stack_.back().branch_bound = false;
    op = stmt.as<IfThenElse>();
    CHECK(op);
    if (op->else_case.defined()) {
      return stmt;
    }
    global_layer_ = GetHoistLayer(op);
    int layer = global_layer_;
    if (layer < 0 || static_cast<size_t>(layer) == loop_stack_.size() - 1) {
      return stmt;
    }
    bool hoist_in = IsMultiCoreLoop(layer);
    int dir = 0;
    if (hoist_in) {
      dir = 1;
      layer++;
      global_layer_++;
    }
    if (dir == 0) {
      return stmt;
    }
    bool hoist_before = dir > 0;
    MultiCoreLoopHoistDepender loop_dep(op, hoist_before);
    if (loop_stack_.size() > static_cast<unsigned int>(layer)) {
      loop_dep.Visit(loop_stack_[layer].node->body);
    }
    DataDepender if_dep;
    if_dep.Visit(op->then_case);
    if (if_dep.DependWith(loop_dep)) {
      return stmt;
    }
    Stmt hoist_stmt = op->then_case;
    if (hoist_in) {
      loop_stack_[layer].hoist_in.emplace_back(hoist_stmt);
    }
    return Evaluate::make(0);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    // do not hoist in basic emit_insn block
    if (op->attr_key == "pragma_emit_insn") {
      return s;
    } else if (op->attr_key == "storage_scope") {
      if (Equal(op->value, Expr("local.L1"))) {
        need_hoisted_ = false;
      }
      auto stmt = IRMutator::Mutate_(op, s);
      if (need_hoisted_ && hoisted_) {
        hoisted_ = false;
        CHECK(stmt.as<AttrStmt>());
        return stmt.as<AttrStmt>()->body;
      }
      return stmt;
    } else if (op->attr_key == "pragma_multi_core_depth") {
      multi_core_ = true;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Allocate *op, const Stmt &s) final {
    auto stmt = IRMutator::Mutate_(op, s);
    DataDepender allocate_dep;
    allocate_dep.Visit(stmt);
    if (!need_hoisted_ || allocate_dep.def_.count(op->buffer_var.get())) {
      hoisted_ = false;
      return stmt;
    }
    hoisted_ = true;
    loop_stack_[global_layer_].allocates.push_back(op);
    CHECK(stmt.as<Allocate>());
    return stmt.as<Allocate>()->body;
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->name == "copy_gm_to_ubuf") {
      need_hoisted_ = false;
    }
    return IRMutator::Mutate_(op, e);
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
    if (target_layer < 0 || target_layer == static_cast<int>(loop_stack_.size())) {
      return -1;
    }
    for (size_t i = target_layer; i < loop_stack_.size(); i++) {
      if (loop_stack_[i].branch_bound) {
        return -1;
      }
    }
    return target_layer;
  }

  bool IsMultiCoreLoop(int layer) {
    if (layer < static_cast<int>(loop_stack_.size())) {
      for (size_t i = 0; i < multicore_loop_.size(); ++i) {
        if (multicore_loop_[i] == loop_stack_[layer].node) {
          return true;
        }
      }
    }
    return false;
  }

  struct LoopEntry {
    const For *node;
    bool branch_bound;
    std::vector<Stmt> hoist_in;
    std::vector<const Allocate *> allocates;
    LoopEntry(const For *nd, bool bb) : node(nd), branch_bound(bb) {}
  };
  int global_layer_{0};
  bool hoisted_{false};
  bool multi_core_{false};
  bool need_hoisted_{true};
  std::vector<const For *> multicore_loop_;
  std::vector<LoopEntry> loop_stack_;
  std::unordered_map<const Variable *, int> loop_layer_;
  std::unordered_map<const Variable *, Expr> loop_size_;
};

Stmt MultiCoreLoopSwitchHoist(Stmt stmt) {
  Stmt prev;
  do {
    prev = stmt;
    stmt = MultiCoreLoopHoister().Mutate(stmt);
  } while (!stmt.same_as(prev));
  stmt = RemoveNoOp(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
