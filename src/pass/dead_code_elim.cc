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
#include <ir_pass.h>

namespace akg {
namespace ir {
class DcePlan : public IRVisitor {
 public:
  void Plan(const Stmt &stmt) {
    std::shared_ptr<Compound> root_comp = std::make_shared<Compound>();
    cur_comp_ = root_comp.get();
    cur_comp_->insn_begin = insns_.size();
    this->Visit(stmt);
    cur_comp_->insn_end = insns_.size();
    TouchRecord record;
    std::vector<Compound *> cond_loop;
    for (int i = insns_.size() - 1; i >= 0; i--) {
      InsnEntry *insn = insns_[i].get();
      if (IsUnused(insn, record)) {
        insn->removed = true;
        replace_[insn->node] = Evaluate::make(0);
        continue;
      }
      if (IsLoopDup(insn, cond_loop)) {
        CHECK(!cond_loop.empty());
        Expr cond;
        for (Compound *c : cond_loop) {
          const For *op = static_cast<const For *>(c->node);
          Expr expr = EQ::make(op->loop_var, op->extent - 1);
          cond = cond.defined() ? And::make(cond, expr) : expr;
        }
        Stmt cond_stmt = IfThenElse::make(cond, GetRef<Stmt>(insn->node));
        replace_[insn->node] = cond_stmt;
        std::swap(insn->cond_loop, cond_loop);
      }
      for (auto &a : insn->def) {
        record[a->buf].emplace_back(Touch{insn, a.get(), true});
      }
      for (auto &a : insn->use) {
        record[a->buf].emplace_back(Touch{insn, a.get(), false});
      }
    }
    MergeAdjacentCondElim();
  }

  void Visit_(const Allocate *op) override {
    alloc_comp_[op->buffer_var.get()] = cur_comp_;
    IRVisitor::Visit_(op);
  }

  void Visit_(const AttrStmt *op) override {
    if (op->attr_key == "pragma_emit_insn") {
      std::unique_ptr<InsnEntry> entry(new InsnEntry(op, cur_comp_));
      cur_insn_ = entry.get();
      insns_.emplace_back(std::move(entry));
      IRVisitor::Visit(op->body);
      cur_insn_ = nullptr;
      return;
    }
    if (op->attr_key == air::ir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      val_map_[iv->var.get()] = make_const(Int(32), 0);
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const For *op) override {
    if (cur_insn_ != nullptr) {
      insn_loop_.push_back(op);
      IRVisitor::Visit_(op);
      insn_loop_.pop_back();
    } else {
      std::unique_ptr<Compound> comp(new Compound(cur_comp_, op, Compound::FOR));
      Compound *parent = cur_comp_;
      cur_comp_ = comp.get();
      parent->children.emplace_back(std::move(comp));
      cur_comp_->insn_begin = insns_.size();
      IRVisitor::Visit_(op);
      cur_comp_->insn_end = insns_.size();
      cur_comp_ = parent;
    }
  }

  void Visit_(const IfThenElse *op) override {
    if (cur_insn_ != nullptr) {
      IRVisitor::Visit_(op);
      return;
    }
    std::unique_ptr<Compound> then_comp(new Compound(cur_comp_, op, Compound::IF_THEN));
    Compound *parent = cur_comp_;
    cur_comp_ = then_comp.get();
    parent->children.emplace_back(std::move(then_comp));
    cur_comp_->insn_begin = insns_.size();
    IRVisitor::Visit(op->then_case);
    cur_comp_->insn_end = insns_.size();
    if (op->else_case.defined()) {
      std::unique_ptr<Compound> else_comp(new Compound(cur_comp_, op, Compound::IF_ELSE));
      cur_comp_ = else_comp.get();
      parent->children.emplace_back(std::move(else_comp));
      cur_comp_->insn_begin = insns_.size();
      IRVisitor::Visit(op->else_case);
      cur_comp_->insn_end = insns_.size();
    }
    cur_comp_ = parent;
  }

  void Visit_(const Store *op) override {
    CHECK(cur_insn_ != nullptr);
    std::unique_ptr<InsnAccess> access(new InsnAccess(op->buffer_var.get()));
    BuildAccessInfo(access.get(), op->index);
    cur_insn_->def.emplace_back(std::move(access));
    IRVisitor::Visit_(op);
  }

  void Visit_(const Load *op) override {
    CHECK(cur_insn_ != nullptr);
    std::unique_ptr<InsnAccess> access(new InsnAccess(op->buffer_var.get()));
    BuildAccessInfo(access.get(), op->index);
    cur_insn_->use.emplace_back(std::move(access));
    IRVisitor::Visit_(op);
  }

  std::unordered_map<const Node *, Stmt> replace_;

 private:
  struct Compound {
    enum { ROOT, FOR, IF_THEN, IF_ELSE };
    Compound() : type(ROOT), layer(-1), node(nullptr), parent(nullptr) {}
    Compound(Compound *p, const Node *n, int t) : type(t), node(n), parent(p) {
      layer = p == nullptr ? -1 : p->layer + 1;
    }
    int type;
    int layer;
    const Node *node;
    Compound *parent;
    std::vector<std::unique_ptr<Compound>> children;
    int insn_begin{-1};
    int insn_end{-1};
  };

  struct InsnAccess {
    explicit InsnAccess(const Variable *b) : buf(b) {}
    const Variable *buf;
    air::arith::ConstIntBound bound;
    Array<Expr> linear_equ;
    std::vector<const For *> touch_axis;
    std::vector<Compound *> dup_axis;
  };

  struct InsnEntry {
    InsnEntry(const AttrStmt *n, Compound *c) : node(n), comp(c) {}
    const AttrStmt *node;
    Compound *comp;
    std::vector<std::unique_ptr<InsnAccess>> def;
    std::vector<std::unique_ptr<InsnAccess>> use;
    bool removed{false};
    std::vector<Compound *> cond_loop;
  };

  struct Touch {
    InsnEntry *insn;
    InsnAccess *access;
    bool is_def;
  };
  using TouchRecord = std::unordered_map<const Variable *, std::vector<Touch>>;

  void MergeAdjacentCondElim() {
    auto same_condition = [](InsnEntry *a, InsnEntry *b) -> bool {
      if ((a->cond_loop.size() != b->cond_loop.size()) || a->comp != b->comp) return false;
      for (size_t i = 0; i < a->cond_loop.size(); ++i) {
        if (a->cond_loop[i] != b->cond_loop[i]) return false;
      }
      return true;
    };
    int insn_size = insns_.size();
    int base = -1;
    for (int i = 0; i < insn_size; ++i) {
      if (base != -1 && same_condition(insns_[base].get(), insns_[i].get()) && (i != insn_size - 1)) continue;
      if (base != -1) {
        int end_cond = (i == insn_size - 1) ? i + 1 : i;
        Stmt body = GetRef<Stmt>(insns_[base]->node);
        for (int j = base + 1; j < end_cond; ++j) {
          body = Block::make(body, GetRef<Stmt>(insns_[j]->node));
          replace_[insns_[j]->node] = Evaluate::make(0);
        }
        const auto cond = replace_[insns_[base]->node].as<IfThenElse>();
        CHECK(cond);
        replace_[insns_[base]->node] = IfThenElse::make(cond->condition, body);
      }
      base = insns_[i]->cond_loop.empty() ? -1 : i;
    }
  }

  void BuildAccessInfo(InsnAccess *access, Expr index) {
    std::vector<const For *> loop_stack(insn_loop_.rbegin(), insn_loop_.rend());
    std::vector<Compound *> comp_stack;
    auto alloc_it = alloc_comp_.find(access->buf);
    Compound *alloc_comp = alloc_it == alloc_comp_.end() ? nullptr : alloc_it->second;
    int dup_end = -1;
    for (Compound *c = cur_comp_; c != nullptr; c = c->parent) {
      if (alloc_comp == c && dup_end == -1) {
        dup_end = loop_stack.size();
      }
      if (c->type == Compound::FOR) {
        loop_stack.push_back(static_cast<const For *>(c->node));
        comp_stack.push_back(c);
      } else if (dup_end == -1) {
        dup_end = loop_stack.size();
      }
    }
    std::vector<bool> touch_axis(loop_stack.size(), false);
    auto scan = [&loop_stack, &touch_axis](const NodeRef &op) {
      const auto var = op.as<Variable>();
      if (var == nullptr) return;
      for (size_t i = 0; i < loop_stack.size(); ++i) {
        if (loop_stack[i]->loop_var.get() == var) {
          touch_axis[i] = true;
        }
      }
    };
    index = air::ir::Substitute(index, val_map_);
    PostOrderVisit(index, scan);
    Array<Var> loop_var;
    air::arith::Analyzer analyzer;
    int insn_loop_size = insn_loop_.size();
    for (int i = 0; i < static_cast<int>(touch_axis.size()); ++i) {
      const For *op = loop_stack[i];
      if (touch_axis[i]) {
        loop_var.push_back(Var(op->loop_var));
        const auto min = op->min.as<IntImm>();
        const auto ext = op->extent.as<IntImm>();
        CHECK(min != nullptr && ext != nullptr);
        analyzer.const_int_bound.Update(Var(op->loop_var), air::arith::ConstIntBound(min->value, ext->value));
        access->touch_axis.push_back(op);
      } else if (i < dup_end && i >= insn_loop_size) {
        access->dup_axis.push_back(comp_stack[i - insn_loop_size]);
      }
    }
    access->linear_equ = air::arith::DetectLinearEquation(index, loop_var);
    access->bound = analyzer.const_int_bound(index);
  }

  bool IsUnused(InsnEntry *insn, TouchRecord &touched) {
    for (auto &def : insn->def) {
      InsnAccess *a = def.get();
      bool is_global = alloc_comp_.count(a->buf) == 0;
      auto itr = touched.find(a->buf);
      if (itr != touched.end()) {
        bool covered = false;
        bool overlap = false;
        for (auto it = itr->second.rbegin(); it != itr->second.rend(); ++it) {
          if (it->is_def) {
            if (AccessCovered(it->access, a) && MustReach(insn, it->insn)) {
              covered = true;
              break;
            }
            if (AccessOverlap(it->access, a) && MayReach(insn, it->insn)) {
              overlap = true;
            }
          } else {
            // 1. def->ref:  first def is used
            if (AccessOverlap(it->access, a) && MayReach(insn, it->insn)) return false;
          }
        }
        // 2. def->def: first def is not covered, and it overlap next used def or
        //    global def,first def is used
        if (!covered && (overlap || is_global)) return false;
      } else if (is_global) {
        // 3. global_def->null, global_def is used
        return false;
      }
      // 4. local_def->null: local_def is unused
    }
    return true;
  }

  bool IsLoopDup(InsnEntry *insn, std::vector<Compound *> &rm_cond) {
    CHECK(!insn->def.empty());
    std::list<Compound *> dup_axis(insn->def.front()->dup_axis.begin(), insn->def.front()->dup_axis.end());
    // get intersection of all def's dup_axis
    for (size_t i = 1; i < insn->def.size(); ++i) {
      auto &axis = insn->def[i]->dup_axis;
      for (auto it = dup_axis.begin(); it != dup_axis.end();) {
        if (std::find(axis.begin(), axis.end(), *it) == axis.end())
          it = dup_axis.erase(it);
        else
          ++it;
      }
    }
    if (dup_axis.empty()) return false;

    auto child_of = [](InsnEntry *e, Compound *c) -> bool {
      Compound *parent = e->comp;
      while (parent->layer > c->layer) {
        parent = parent->parent;
      }
      return parent == c;
    };
    auto is_depend = [this](InsnEntry *def_insn, InsnEntry *e) -> bool {
      for (auto &use : e->use) {
        for (auto &def : def_insn->def) {
          if (def->buf != use->buf) continue;
          if (this->AccessOverlap(use.get(), def.get()) && (this->MayReach(e, def_insn) || this->MayReach(def_insn, e)))
            return true;
        }
      }
      return false;
    };
    // delete ref axis
    int start_idx = dup_axis.front()->insn_begin;
    int end_idx = dup_axis.front()->insn_end;
    for (int i = start_idx; i < end_idx; ++i) {
      InsnEntry *entry = insns_[i].get();
      if (!entry->removed && is_depend(insn, entry)) {
        for (auto it = dup_axis.begin(); it != dup_axis.end();) {
          if (child_of(entry, *it) &&
              std::find(entry->cond_loop.begin(), entry->cond_loop.end(), *it) == entry->cond_loop.end())
            it = dup_axis.erase(it);
          else
            ++it;
        }
      }
    }
    if (dup_axis.empty()) return false;
    rm_cond = std::vector<Compound *>(dup_axis.begin(), dup_axis.end());
    return true;
  }

  bool AccessOverlap(InsnAccess *a1, InsnAccess *a2) const {
    return !((a1->bound->min_value >= a2->bound->max_value) || (a2->bound->min_value >= a1->bound->max_value));
  }

  bool AccessCovered(InsnAccess *large, InsnAccess *small) {
    if (!(small->bound->min_value >= large->bound->min_value && small->bound->max_value <= large->bound->max_value)) {
      return false;
    }
    if (large->touch_axis.size() != small->touch_axis.size() || large->linear_equ.size() != small->linear_equ.size()) {
      return false;
    }
    for (size_t i = 0; i < large->touch_axis.size(); ++i) {
      if (!Equal(large->touch_axis[i]->extent, small->touch_axis[i]->extent)) {
        return false;
      }
    }
    for (size_t i = 0; i < large->linear_equ.size(); ++i) {
      if (!Equal(large->linear_equ[i], small->linear_equ[i])) {
        return false;
      }
    }
    return true;
  }

  Compound *FirstBranch(const InsnEntry *entry) const {
    Compound *comp = entry->comp;
    while (comp != nullptr) {
      if (comp->type == Compound::IF_THEN || comp->type == Compound::IF_ELSE) {
        return comp;
      }
      comp = comp->parent;
    }
    return nullptr;
  }

  bool MustReach(const InsnEntry *from, const InsnEntry *to) const { return FirstBranch(from) == FirstBranch(to); }

  bool MayReach(const InsnEntry *from, const InsnEntry *to) const {
    auto b1 = FirstBranch(from);
    auto b2 = FirstBranch(to);
    return b1 == nullptr || b2 == nullptr || b1->node != b2->node;
  }

  std::vector<std::unique_ptr<InsnEntry>> insns_;
  InsnEntry *cur_insn_{nullptr};
  std::vector<const For *> insn_loop_;

  Compound root_comp_;
  Compound *cur_comp_{nullptr};
  std::unordered_map<const Variable *, Compound *> alloc_comp_;

  std::unordered_map<const Variable *, Expr> val_map_;
};

class DceSubstitute : public IRMutator {
 public:
  Stmt Substitute(Stmt stmt) {
    DcePlan dce;
    dce.Plan(stmt);
    if (dce.replace_.empty()) return stmt;
    replace_ = std::move(dce.replace_);
    return Mutate(stmt);
  }

  Stmt Mutate(Stmt stmt) override {
    const Node *node = stmt.as<Node>();
    auto it = replace_.find(node);
    if (it != replace_.end()) {
      return it->second;
    }
    return IRMutator::Mutate(stmt);
  }

 private:
  std::unordered_map<const Node *, Stmt> replace_;
};

Stmt DeadCodeElim(Stmt stmt) {
  Stmt prev = stmt;
  stmt = DceSubstitute().Substitute(stmt);
  if (!stmt.same_as(prev)) {
    stmt = LoopSwitchHoist(stmt);
  }
  return stmt;
}
}  // namespace ir
}  // namespace akg
