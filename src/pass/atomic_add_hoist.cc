/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <poly/poly_util.h>

namespace akg {
namespace ir {

struct HoistEntry {
  std::list<Stmt> hoist_before;
  std::list<Stmt> hoist_after;
  std::list<const Allocate *> allocates;
};

const Variable *GetBuffer(const Stmt &stmt) {
  std::unordered_set<const Variable *> buffers;
  PostOrderVisit(stmt, [&buffers](const NodeRef &op) {
    const Variable *buf = nullptr;
    if (auto store = op.as<Store>()) {
      buf = store->buffer_var.get();
    }
    if (auto load = op.as<Load>()) {
      buf = load->buffer_var.get();
    }
    if (buf && buf->name_hint.find("local_UB") != std::string::npos) {
      buffers.insert(buf);
    }
  });
  CHECK_EQ(buffers.size(), 1);
  return *buffers.begin();
}

class AtomicAddGetHoistEntry : public IRVisitor {
 public:
  std::unordered_map<const For *, HoistEntry> loop_map_hoist_entry_;

 private:
  std::vector<const For *> loop_stack_;
  std::unordered_map<const AttrStmt *, std::vector<const For *>> clean_attr_map_loop_stack_;
  std::unordered_map<const Variable *, const AttrStmt *> buf_map_clean_attr_;
  std::unordered_map<const Variable *, const For *> buf_map_hoist_loop_;

  void Visit_(const For *op) override {
    loop_stack_.emplace_back(op);
    IRVisitor::Visit_(op);
    loop_stack_.pop_back();
    auto kv_iter = buf_map_hoist_loop_.begin();
    while (kv_iter != buf_map_hoist_loop_.end()) {
      if (kv_iter->second == op) {
        kv_iter = buf_map_hoist_loop_.erase(kv_iter);
      } else {
        ++kv_iter;
      }
    }
  }

  void Visit_(const AttrStmt *op) override {
    // atomic clean_zero
    if (op->attr_key == ATTR_ATOMIC_CLEAN_ZERO) {
      clean_attr_map_loop_stack_[op] = loop_stack_;
      auto buf = GetBuffer(op->body);
      CHECK_EQ(buf_map_clean_attr_.count(buf), 0);
      buf_map_clean_attr_[buf] = op;
      return;
    }
    // atomic write
    if (op->attr_key == ATTR_ATOMIC_ADD || (op->attr_key == "pragma_emit_insn" && op->value->IsInstance<StringImm>() &&
                                            op->value.as<StringImm>()->value == "dma_atomic_add")) {
      // The atomic clean_zero and the atomic write should appear in pairs:
      auto buf = GetBuffer(op->body);
      CHECK(buf_map_clean_attr_.count(buf));
      auto clean_attr = buf_map_clean_attr_.at(buf);
      CHECK(clean_attr_map_loop_stack_.count(clean_attr));
      auto &clean_loop_stack = clean_attr_map_loop_stack_.at(clean_attr);
      auto &atomic_write_loop_stack = loop_stack_;
      CHECK_EQ(clean_loop_stack.size(), atomic_write_loop_stack.size());
      for (size_t i = 0; i < clean_loop_stack.size(); ++i) {
        CHECK(clean_loop_stack[i] == atomic_write_loop_stack[i]);
      }
      auto &common_loop_stack = clean_loop_stack;
      // Get hoist:
      auto hoist_loop = GetHoistLoop(clean_attr->body, op->body, common_loop_stack);
      auto hoist_before = GetRef<Stmt>(clean_attr);
      auto hoist_after = GetRef<Stmt>(op);
      if (hoist_loop) {
        if (loop_map_hoist_entry_.count(hoist_loop)) {
          loop_map_hoist_entry_.at(hoist_loop).hoist_before.push_back(hoist_before);
          loop_map_hoist_entry_.at(hoist_loop).hoist_after.push_back(hoist_after);
        } else {
          HoistEntry hoist_entry;
          hoist_entry.hoist_before.push_back(hoist_before);
          hoist_entry.hoist_after.push_back(hoist_after);
          loop_map_hoist_entry_[hoist_loop] = hoist_entry;
        }
        buf_map_hoist_loop_[buf] = hoist_loop;
      }
      // Clear information that is no longer used：
      buf_map_clean_attr_.erase(buf);
      clean_attr_map_loop_stack_.erase(clean_attr);
      return;
    }

    return IRVisitor::Visit_(op);
  }

  void Visit_(const Allocate *op) override {
    IRVisitor::Visit_(op);
    auto buf = op->buffer_var.get();
    if (buf_map_hoist_loop_.count(buf)) {
      auto hoist_loop = buf_map_hoist_loop_.at(buf);
      loop_map_hoist_entry_.at(hoist_loop).allocates.push_back(op);
    }
  }

  const For *GetHoistLoop(const Stmt &atomic_clean_stmt, const Stmt &atomic_write_stmt,
                          const std::vector<const For *> &common_loop_stack) {
    std::unordered_set<const Variable *> vars;
    std::vector<NodeRef> stmts = {atomic_clean_stmt, atomic_write_stmt};
    for (const auto &stmt : stmts) {
      PostOrderVisit(stmt, [&vars](const NodeRef &node) {
        if (auto var = node.as<Variable>()) {
          vars.insert(var);
        }
      });
    }
    const For *hoist_loop = nullptr;
    for (auto it = loop_stack_.rbegin(); it != loop_stack_.rend(); ++it) {
      auto loop_var = (*it)->loop_var.get();
      if (vars.count(loop_var)) {
        break;
      } else {
        hoist_loop = *it;
      }
    }
    return hoist_loop;
  }
};

class AtomicAddHoister : public IRMutator {
 public:
  explicit AtomicAddHoister(const std::unordered_map<const For *, HoistEntry> &loop_map_hoist_entry) {
    loop_map_hoist_entry_ = loop_map_hoist_entry;
  }
  ~AtomicAddHoister() override = default;

 private:
  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (loop_map_hoist_entry_.count(op)) {
      HoistEntry &entry = loop_map_hoist_entry_.at(op);
      // remove the hoisted stmt:
      hoist_before_ = entry.hoist_before;
      hoist_after_ = entry.hoist_after;
      hoist_allocate_ = entry.allocates;
      CHECK_EQ(hoist_before_.size(), hoist_after_.size());
      auto stmt = IRMutator::Mutate_(op, s);
      CHECK(hoist_before_.empty());
      CHECK(hoist_after_.empty());
      CHECK(hoist_allocate_.empty());
      // hoist:
      auto hoist_before = RemoveDuplicate(entry.hoist_before);
      auto hoist_after = RemoveDuplicate(entry.hoist_after);
      CHECK_EQ(hoist_before.size(), hoist_after.size());
      for (const auto &before : hoist_before) {
        stmt = Block::make(before, stmt);
      }
      for (const auto &after : hoist_after) {
        stmt = Block::make(stmt, after);
      }
      for (auto &allocate : entry.allocates) {
        stmt = Allocate::make(allocate->buffer_var, allocate->type, allocate->extents, allocate->condition, stmt);
        stmt = AttrStmt::make(allocate->buffer_var, "storage_scope", Expr("local.UB"), stmt);
      }
      loop_map_hoist_entry_.erase(op);
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == ATTR_ATOMIC_CLEAN_ZERO && !hoist_before_.empty() && hoist_before_.front() == s) {
      hoist_before_.pop_front();
      return Evaluate::make(0);
    }
    if (op->attr_key == ATTR_ATOMIC_ADD || (op->attr_key == "pragma_emit_insn" && op->value->IsInstance<StringImm>() &&
                                            op->value.as<StringImm>()->value == "dma_atomic_add")) {
      if (!hoist_after_.empty() && hoist_after_.front() == s) {
        hoist_after_.pop_front();
        return Evaluate::make(0);
      }
    }
    if (op->attr_key == "storage_scope" && op->node.as<Variable>() && op->value.as<StringImm>() &&
        op->value.as<StringImm>()->value == "local.UB") {
      auto allocate = op->body.as<Allocate>();
      auto stmt = IRMutator::Mutate_(op, s);
      if (allocate && !hoist_allocate_.empty() && hoist_allocate_.front() == allocate) {
        hoist_allocate_.pop_front();
        auto attr_stmt = stmt.as<AttrStmt>();
        CHECK(attr_stmt);
        return attr_stmt->body;
      }
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Allocate *op, const Stmt &s) final {
    auto stmt = IRMutator::Mutate_(op, s);
    if (!hoist_allocate_.empty() && hoist_allocate_.front() == op) {
      CHECK(stmt.as<Allocate>());
      return stmt.as<Allocate>()->body;
    }
    return stmt;
  }

  static std::list<Stmt> RemoveDuplicate(const std::list<Stmt> &hoist_item) {
    std::list<Stmt> hoist_item_new;
    std::unordered_set<const Variable *> buffers;
    for (const auto &stmt : hoist_item) {
      auto buffer = GetBuffer(stmt);
      if (!buffers.count(buffer)) {
        hoist_item_new.push_back(stmt);
        buffers.insert(buffer);
      }
    }
    return hoist_item_new;
  }

  std::unordered_map<const For *, HoistEntry> loop_map_hoist_entry_;
  std::list<const Allocate *> hoist_allocate_;
  std::list<Stmt> hoist_before_;
  std::list<Stmt> hoist_after_;
};

/*
  This pass tries to hoist the atomic_add stmt.
  For example:

  for (cc0, 0, 2) {
    // attr [input_1_local_UB] storage_scope = "local.UB"
    allocate input_1_local_UB[float32 * 64]
    // attr [0] pragma_emit_insn = "dma_copy"
    for (cc1, 0, 64) {
      input_1_local_UB[cc1] = input_1[((((blockIdx.x*2) + cc0)*64) + cc1)] if -2
    }
    // attr [input_1_red_local_UB] storage_scope = "local.UB"
    allocate input_1_red_local_UB[float32 * 8]
    // attr [0] atomic_clean_zero = 1
    // attr [0] pragma_emit_insn = "vector_dup"
    input_1_red_local_UB[0] = 0f
    // attr [0] pragma_emit_insn = "vec_binary_add"
    for (cc1, 0, 64) {
      input_1_red_local_UB[0] = (input_1_red_local_UB[0] if -2 + input_1_local_UB[cc1] if -2) if -2
    }
    // attr [0] atomic_add = 1
    // attr [0] pragma_emit_insn = "dma_atomic_add"
    input_1_red[0] = (input_1_red[0] + input_1_red_local_UB[0] if -2)
  }

  ====>

  // attr [input_1_red_local_UB] storage_scope = "local.UB"
  allocate input_1_red_local_UB[float32 * 8]
  // attr [0] atomic_clean_zero = 1
  // attr [0] pragma_emit_insn = "vector_dup"
  input_1_red_local_UB[0] = 0f
  for (cc0, 0, 2) {
    // attr [input_1_local_UB] storage_scope = "local.UB"
    allocate input_1_local_UB[float32 * 64]
    // attr [0] pragma_emit_insn = "dma_copy"
    for (cc1, 0, 64) {
      input_1_local_UB[cc1] = input_1[((((blockIdx.x*2) + cc0)*64) + cc1)] if -2
    }
    // attr [0] pragma_emit_insn = "vec_binary_add"
    for (cc1, 0, 64) {
      input_1_red_local_UB[0] = (input_1_red_local_UB[0] if -2 + input_1_local_UB[cc1] if -2) if -2
    }
  }
  // attr [0] atomic_add = 1
  // attr [0] pragma_emit_insn = "dma_atomic_add"
  input_1_red[0] = (input_1_red[0] + input_1_red_local_UB[0] if -2)
 */
Stmt AtomicAddHoist(Stmt stmt) {
  AtomicAddGetHoistEntry atomic_add_get;
  atomic_add_get.Visit(stmt);
  stmt = AtomicAddHoister(atomic_add_get.loop_map_hoist_entry_).Mutate(stmt);
  stmt = RemoveNoOp(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
