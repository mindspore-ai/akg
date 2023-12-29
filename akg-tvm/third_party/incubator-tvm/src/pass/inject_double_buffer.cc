/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \brief Inject double buffering optimization for data fetch.
 * \file inject_double_buffer.cc
 */

/*!
 * \ 2021.01.14
 * When shared memory is not enough for double buffer, new local memory can be used as the transfer
 * buffer to replace the second shared memory.
 * Combined with double buffer, different thread groups can be used respectively on promotion and
 * computation.
 * \ 2021.03.01
 * Add the tvm_storage_sync sentence after the first prefetch and remove the tvm_storage_sync sentence
 * between data movement and computation
 */

#include <tvm/expr_operator.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>

#include "../arithmetic/compute_expr.h"
#include "ir_util.h"

namespace air {
namespace ir {
constexpr auto TRANSFER_WRITE_INDEX = "transfer_write_index";
constexpr auto USE_THREAD_GROUP = "use_thread_group";

inline Expr BroadcastTo(Expr e, int lanes) {
  if (e.type().lanes() == lanes) return e;
  if (const Broadcast* op = e.as<Broadcast>()) {
    if (lanes % op->lanes == 0) {
      return Broadcast::make(op->value, lanes);
    }
  }
  CHECK_EQ(e.type().lanes(), 1)
      << "Cannot broadcast lane=" << e.type().lanes()
      << " to " << lanes;
  return Broadcast::make(e, lanes);
}

// Detect double buffer variables.
class DoubleBufferDetector : public IRVisitor {
 public:
  void Visit_(const AttrStmt* op) final {
    if (op->attr_key == attr::double_buffer_scope) {
      touched_.insert(op->node.as<Variable>());
      IRVisitor::Visit_(op);
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const Variable* op) final {
    if (touched_.count(op)) {
      touched_.erase(op);
    }
  }
  // The set of touched variable.
  std::unordered_set<const Variable*> touched_;
};

class StripSyncAndAllocs : public IRMutator {
 public:
  explicit StripSyncAndAllocs(bool use_double_shared)
  : use_double_buffer_(use_double_shared) {}

  Stmt Mutate_(const Block* op, const Stmt& s) final {
    Stmt first = Mutate(op->first);
    Stmt rest = Mutate(op->rest);
    if (const auto attr = first.as<AttrStmt>()) {
      if (attr->attr_key == "delete_this_sync"
        || (use_double_buffer_ && attr->attr_key == "delete_this_sync_for_db")) {
        return rest;
      }
    }
    return Block::make(first, rest);
  }

  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (op->attr_key == attr::storage_scope) {
      if (auto alloc = op->body.as<Allocate>()) {
        int fragment_size_ = (arith::ComputeReduce<Mul>(alloc->extents, Expr())
          * alloc->type.lanes() * alloc->type.bytes()).as<IntImm>()->value;
        auto it = fragment_info_.find(alloc->buffer_var.get());
        if ( it == fragment_info_.end()
          || ( it != fragment_info_.end() && (fragment_size_ > (it->second)) ) ) {
          fragment_info_[alloc->buffer_var.as<Variable>()] = fragment_size_;
          fragment_allocs_.emplace_back( AttrStmt::make(op->node, op->attr_key, op->value, Evaluate::make(0)) );
          fragment_allocs_.emplace_back(
            Allocate::make(alloc->buffer_var, alloc->type, alloc->extents, alloc->condition, Evaluate::make(0))
          );
        }
        return Mutate(alloc->body);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  std::vector<Stmt> GetFragmentAllocs() { return fragment_allocs_; }

 private:
  bool use_double_buffer_{false};
  std::vector<Stmt> fragment_allocs_;
  std::unordered_map<const Variable*, int> fragment_info_;
};

class StripDoubleBufferWrite : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (op->attr_key == attr::double_buffer_write) {
      return Mutate(op->body);
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }
};

class StripTransferWriteIndex : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (op->attr_key == TRANSFER_WRITE_INDEX) {
      return Mutate(op->body);
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }
};

class TransferBufferInjector : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (op->attr_key == TRANSFER_WRITE_INDEX) {
      const auto old_store = op->body.as<Store>();
      if (old_store) {
        const Var read_buffer = old_store->buffer_var;
        const Expr read_index = old_store->index;
        const Var write_buffer = Downcast<Var>(op->node);
        const Expr write_index = op->value;
        Expr transfer_load = Load::make(old_store->value.as<Load>()->type, read_buffer, read_index,
                                        old_store->value.as<Load>()->predicate);
        Stmt transfer_store =
            Store::make(write_buffer, transfer_load, write_index, old_store->predicate);
        return transfer_store;
      }
    }
    return IRMutator::Mutate_(op, s);
  }
};

class ThreadGroupDetector : public IRVisitor {
 public:
  void Visit_(const AttrStmt* op) {
    if (op->attr_key == USE_THREAD_GROUP) {
      if_enable_thread_group_ = true;
      thread_to_change_ = Downcast<Var>(op->node);
      thread_group_num_ = op->value;
    } else {
      return IRVisitor::Visit_(op);
    }
  }

  const bool IfEnableThreadGroup() { return if_enable_thread_group_; }

  const Var GetThreadToChange() { return thread_to_change_; }

  const Expr GetThreadGroupNum() { return thread_group_num_; }

 private:
  bool if_enable_thread_group_{false};
  Var thread_to_change_;
  Expr thread_group_num_;
};

class ThreadGroupInjector : public IRMutator {
 public:
  Stmt Inject(Stmt stmt) {
    ThreadGroupDetector detector;
    detector.Visit(stmt);
    if (detector.IfEnableThreadGroup()) {
      thread_to_change_ = detector.GetThreadToChange();
      thread_group_num_ = detector.GetThreadGroupNum();
      return this->Mutate(stmt);
    }
    return stmt;
  }

  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (op->attr_key == USE_THREAD_GROUP) {
      return Mutate(op->body);
    }
    if (op->attr_key == air::ir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      CHECK(thread_to_change_.defined());
      if (iv->var.same_as(thread_to_change_)) {
        thread_range_ = op->value;
        CHECK(thread_group_num_.defined());
        return AttrStmt::make(op->node, op->attr_key, op->value * thread_group_num_,
                              Mutate(op->body));
      } else {
        return IRMutator::Mutate_(op, s);
      }
    }
    if (op->attr_key == "thread_group_offset") {
      Expr thread_offset = op->value;
      Stmt body = Mutate(op->body);
      CHECK(thread_to_change_.defined());
      CHECK(thread_range_.defined());
      std::unordered_map<const Variable*, Expr> vmap;
      vmap[thread_to_change_.get()] = thread_to_change_ - thread_offset;
      body = Substitute(body, vmap);
      return IfThenElse::make(
          ((thread_to_change_ >= make_const(thread_offset.type(), 0) + thread_offset) &&
           (thread_to_change_ < thread_range_ + thread_offset)),
          body);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  Var thread_to_change_;
  Expr thread_group_num_;
  Expr thread_range_;
};

class DoubleBufferInjector : public IRMutator {
 public:
  explicit DoubleBufferInjector(int split_loop, bool use_double_shared)
      : split_loop_(split_loop), use_double_buffer_(use_double_shared) {}

  Stmt Inject(const Stmt& stmt) {
    DoubleBufferDetector detector;
    detector.Visit(stmt);
    if (detector.touched_.empty()) return stmt;
    for (const Variable* v : detector.touched_) {
      dbuffer_info_[v] = StorageEntry();
    }
    return ConvertSSA(this->Mutate(stmt));
  }

  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (op->attr_key == attr::storage_scope) {
      const Variable* buf = op->node.as<Variable>();
      auto it = dbuffer_info_.find(buf);
      if (it != dbuffer_info_.end()) {
        it->second.scope = op->value.as<StringImm>()->value;
        return Mutate(op->body);
      }
    } else if (op->attr_key == attr::double_buffer_scope) {
      return MakeProducer(op, s);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Allocate* op, const Stmt& s) final {
    auto it = dbuffer_info_.find(op->buffer_var.get());
    if (it != dbuffer_info_.end()) {
      it->second.stride = arith::ComputeReduce<Mul>(op->extents, Expr()) * op->type.lanes();
      it->second.type = op->type;
      it->second.condition = op->condition;
      it->second.transfer_buffer_scope = "local";
      it->second.transfer_buffer = Var(op->buffer_var->name_hint + "_transfer");
      it->second.transfer_buffer_extents.push_back(make_const(op->extents[0].type(), 1));
      Stmt stmt = IRMutator::Mutate_(op, s);
      if (const auto alloc = stmt.as<Allocate>()) {
        CHECK(it->second.loop != nullptr);
        auto& alloc_nest = loop_allocs_[it->second.loop];
        alloc_nest.emplace_back(AttrStmt::make(alloc->buffer_var, attr::storage_scope,
                                              StringImm::make(it->second.scope), Evaluate::make(0)));
        if (use_double_buffer_) {
          Array<Expr> new_extents{make_const(alloc->extents[0].type(), 2)};
          for (Expr e : alloc->extents) {
            new_extents.push_back(e);
          }
          alloc_nest.emplace_back(Allocate::make(alloc->buffer_var, alloc->type, new_extents, alloc->condition,
                                                Evaluate::make(0)));
        } else {
          alloc_nest.emplace_back(Allocate::make(alloc->buffer_var, alloc->type, alloc->extents, alloc->condition,
                                                Evaluate::make(0)));
        }
        alloc_nest.emplace_back(
            AttrStmt::make(it->second.transfer_buffer, air::ir::attr::storage_scope,
                            StringImm::make(it->second.transfer_buffer_scope), Evaluate::make(0)));
        alloc_nest.emplace_back(Allocate::make(it->second.transfer_buffer, it->second.type,
                                                it->second.transfer_buffer_extents,
                                                it->second.condition, Evaluate::make(0)));
        return alloc->body;
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For* op, const Stmt& s) final {
    loop_nest_.push_back(op);
    if (in_double_buffer_scope_) {
      transfer_loop_nest_.push_back(op);
    }
    Stmt stmt = IRMutator::Mutate_(op, s);
    const For* orig_loop = stmt.as<For>();
    auto iter = loop_transfer_.find(op);
    std::vector<Stmt> fragment_allocs;
    if (iter != loop_transfer_.end()) {
      stmt =
          For::make(orig_loop->loop_var, orig_loop->min, orig_loop->extent, orig_loop->for_type,
                    orig_loop->device_api, Block::make(orig_loop->body, MergeSeq(iter->second)));
    }
    auto it = loop_pre_.find(op);
    if (it != loop_pre_.end()) {
      const For* old_loop = stmt.as<For>();
      if (split_loop_ != 0) {
        // Explicitly unroll the loop
        CHECK(split_loop_ % 2 == 0 || split_loop_ == 1)
            << "It is better to split with multiple of 2";
        CHECK(is_zero(old_loop->min));
        Expr zero = old_loop->min;
        Expr new_ext = old_loop->extent - make_const(old_loop->loop_var.type(), 1);
        Expr factor = make_const(new_ext.type(), split_loop_);
        Expr outer_ext = new_ext / factor;
        Expr tail_base = outer_ext * factor;
        Var outer_var(old_loop->loop_var->name_hint + ".outer", old_loop->loop_var.type());
        std::unordered_map<const Variable*, Expr> vmap;
        std::vector<Stmt> loop_seq;
        StripSyncAndAllocs body_remover(use_double_buffer_);
        Stmt old_loop_body = body_remover.Mutate(old_loop->body);
        fragment_allocs = body_remover.GetFragmentAllocs();
        for (int32_t i = 0; i < split_loop_; ++i) {
          vmap[old_loop->loop_var.get()] = outer_var * factor + make_const(factor.type(), i);
          loop_seq.emplace_back(Substitute(old_loop_body, vmap));
        }
        // Add syncthreads at the end of main loop
        Stmt loop = For::make(outer_var, zero, outer_ext, old_loop->for_type, old_loop->device_api, 
          Block::make(MergeSeq(loop_seq), Evaluate::make(
                Call::make(Int(32), "tvm_storage_sync", {StringImm::make("shared")}, Call::Intrinsic)
          ))
        );
        // tail
        std::vector<Stmt> tail_seq;
        StripSyncAndAllocs tail_remover(false);
        old_loop_body = tail_remover.Mutate(old_loop->body);
        Stmt tail_body = StripDoubleBufferWrite().Mutate(old_loop_body);
        for (int32_t i = 0; i < split_loop_; ++i) {
          Expr idx = tail_base + make_const(tail_base.type(), i);
          vmap[old_loop->loop_var.get()] = idx;
          tail_seq.emplace_back(
              IfThenElse::make(idx < old_loop->extent, Substitute(tail_body, vmap)));
        }
        stmt = Block::make(loop, MergeSeq(tail_seq));
      }
      // Move fragment allocation statements to the top of the current loop
      Stmt loop_pre_stmt = MergeNest(fragment_allocs, MergeSeq(it->second));
      // Add syncthreads after the first prefetch
      loop_pre_stmt  = Block::make(
        loop_pre_stmt,
        Evaluate::make(Call::make(Int(32), "tvm_storage_sync", {StringImm::make("shared")}, Call::Intrinsic))
        );
      stmt = Block::make(loop_pre_stmt, stmt);
    }
    it = loop_allocs_.find(op);
    if (it != loop_allocs_.end()) {
      stmt = MergeNest(it->second, stmt);
    }
    loop_nest_.pop_back();
    return stmt;
  }

  Stmt Mutate_(const Store* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    if (const auto store = stmt.as<Store>()) {
      auto it = dbuffer_info_.find(store->buffer_var.get());
      if (it != dbuffer_info_.end()) {
        StorageEntry& e = it->second;
        CHECK(in_double_buffer_scope_);
        Expr transfer_index = make_const(e.loop->loop_var.type(), 0);
        Expr transfer_extent = make_const(e.transfer_buffer_extents[0].type(), 1);
        for (unsigned i = 0; i < transfer_loop_nest_.size(); i++) {
          if (i < transfer_loop_nest_.size() - 1) {
            transfer_index +=
                transfer_loop_nest_[i]->loop_var *
                (transfer_loop_nest_[i + 1]->extent - transfer_loop_nest_[i + 1]->min);
          } else {
            transfer_index += transfer_loop_nest_[i]->loop_var;
          }
          transfer_extent *= transfer_loop_nest_[i]->extent - transfer_loop_nest_[i]->min;
        }
        e.transfer_buffer_extents.push_back(transfer_extent);
        air::DataType transfer_type = store->value.as<Load>()->type;
        if (e.type != transfer_type) {
          e.type = transfer_type;
        }
        int lanes = std::max(store->value.type().lanes(), transfer_index.type().lanes());
        auto val = store->value.as<Load>()->index.as<Ramp>();
        if (val) {
          transfer_index = Ramp::make(transfer_index, val->stride, val->lanes);
        }
        Stmt transfer_store =
            Store::make(e.transfer_buffer, BroadcastTo(store->value, lanes),
                        BroadcastTo(transfer_index, lanes), BroadcastTo(store->predicate, lanes));
        if (use_double_buffer_) {
          CHECK(e.stride.defined());
          transfer_store =
            AttrStmt::make(store->buffer_var, TRANSFER_WRITE_INDEX, e.switch_write_var * e.stride + store->index, transfer_store);
        } else {
          transfer_store =
            AttrStmt::make(store->buffer_var, TRANSFER_WRITE_INDEX, store->index, transfer_store);
        }
        return transfer_store;
      }
    }
    return stmt;
  }

  Expr Mutate_(const Load* op, const Expr& e) final {
    Expr expr = IRMutator::Mutate_(op, e);
    if (const auto load = expr.as<Load>()) {
      auto it = dbuffer_info_.find(load->buffer_var.get());
      if ((use_double_buffer_ && it != dbuffer_info_.end())) {
        const StorageEntry& e = it->second;
        CHECK(e.stride.defined());
        CHECK(e.switch_read_var.defined());
        return Load::make(load->type, load->buffer_var, e.switch_read_var * e.stride + load->index, load->predicate);
      }
    }
    return expr;
  }

  Expr Mutate_(const Variable* op, const Expr& e) final {
    CHECK(!dbuffer_info_.count(op));
    return e;
  }

 private:
  Stmt MakeProducer(const AttrStmt* op, const Stmt& s) {
    const VarExpr buffer = Downcast<VarExpr>(op->node);
    CHECK_NE(loop_nest_.size(), 0U) << "Double buffer scope must be inside a loop";
    auto it = dbuffer_info_.find(buffer.get());
    if (it == dbuffer_info_.end()) {
      LOG(WARNING) << "Skip double buffer scope " << op->node;
      return Mutate(op->body);
    }
    StorageEntry& e = it->second;
    e.loop = loop_nest_.back();
    Expr zero = make_const(e.loop->loop_var.type(), 0);
    Expr one = make_const(e.loop->loop_var.type(), 1);
    Expr two = make_const(e.loop->loop_var.type(), 2);
    Expr loop_shift = e.loop->loop_var + one;
    e.switch_write_var = Var(e.loop->loop_var->name_hint + ".db", e.loop->loop_var.type());
    e.switch_read_var = indexmod(e.loop->loop_var, two);
    in_double_buffer_scope_ = true;
    Stmt body = Mutate(op->body);
    in_double_buffer_scope_ = false;
    std::unordered_map<const Variable*, Expr> vmap;
    vmap[e.loop->loop_var.get()] = zero;
    Stmt transfer_stmt;
    if (use_double_buffer_) {
      vmap[e.switch_write_var.get()] = zero;
    }
    transfer_stmt = TransferBufferInjector().Mutate(body);
    body = StripTransferWriteIndex().Mutate(body);
    loop_pre_[e.loop].emplace_back(Substitute(body, vmap));
    loop_pre_[e.loop].emplace_back(Substitute(transfer_stmt, vmap));
    vmap[e.loop->loop_var.get()] = loop_shift;
    if (use_double_buffer_) {
      vmap[e.switch_write_var.get()] = indexmod(loop_shift, two);
    } 
    body = Substitute(body, vmap);
    transfer_stmt = Substitute(transfer_stmt, vmap);
    body = AttrStmt::make(buffer, air::ir::attr::double_buffer_write, 1, body);
    body = IfThenElse::make(loop_shift < e.loop->extent, body);
    transfer_stmt = AttrStmt::make(e.transfer_buffer, attr::double_buffer_write, 1, transfer_stmt);
    transfer_stmt = IfThenElse::make(loop_shift < e.loop->extent, transfer_stmt);
    loop_transfer_[e.loop].emplace_back(transfer_stmt);
    transfer_loop_nest_.clear();
    return body;
  }
  // Storage entry for those who need double buffering.
  struct StorageEntry {
    // The size of the buffer
    Expr stride;
    // The loop we need
    const For* loop{nullptr};
    // The switch variable.
    VarExpr switch_write_var;
    // The switch variable for reading.
    Expr switch_read_var;
    // The storage scope.
    std::string scope;
    // The storage data type
    air::DataType type;
    // The storage condition
    Expr condition;
    //  The transfer scope
    std::string transfer_buffer_scope;
    //  The transfer buffer
    Var transfer_buffer;
    // The transfer buffer extent
    Array<Expr> transfer_buffer_extents;
  };
  // Whether split loop
  int32_t split_loop_;
  // Whether use transfer buffer to replace the second shared buffer
  bool use_double_buffer_{false};
  // Whether we are inside double buffer scope.
  bool in_double_buffer_scope_{false};
  // The current loop nest
  std::vector<const For*> loop_nest_;
  // The allocs to be appended before the loop
  std::unordered_map<const For*, std::vector<Stmt> > loop_allocs_;
  // The stmt to be appended before the loop
  std::unordered_map<const For*, std::vector<Stmt> > loop_pre_;
  // The allocation size of the buffer
  std::unordered_map<const Variable*, StorageEntry> dbuffer_info_;
  // The stmt to be appended inside the loop
  std::unordered_map<const For*, std::vector<Stmt> > loop_transfer_;
  // The loop nest for transfer
  std::vector<const For*> transfer_loop_nest_;
};

Stmt InjectDoubleBuffer(Stmt stmt, int split_loop, bool use_double_shared) {
  Stmt new_stmt = DoubleBufferInjector(split_loop, use_double_shared).Inject(stmt);
  new_stmt = ThreadGroupInjector().Inject(new_stmt);
  return new_stmt;
}
}  // namespace ir
}  // namespace air
