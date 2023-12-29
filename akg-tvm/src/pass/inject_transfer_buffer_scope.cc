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
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <arithmetic/compute_expr.h>
#include <ir_pass.h>
#include "build_module.h"
#include "utils.h"
#include "common/common_util.h"

namespace akg {
namespace ir {
constexpr auto PREFETCH_SCOPE = "double_buffer_scope";
constexpr auto THREAD_GROUP_OFFSET = "thread_group_offset";
constexpr auto WMMA_FACTOR_AB = 16;
constexpr auto WMMA_FACTOR_C = 32;
constexpr int REGISTER_FILE_SIZE_PER_SM = 256 * 1024;
constexpr int TOTAL_THREAD_NUM_PER_BLOCK = 1024;
constexpr int MIN_OUTER_LOOP = 2;
constexpr int MAX_OUTER_LOOP = 64;
constexpr int BIT32 = 32;
constexpr int BIT64 = 64;

class IfTensorCore : public IRVisitor {
 public:
  const bool IfUseTensorCore(Stmt stmt) {
    this->Visit(stmt);
    return if_use_tensor_core_;
  }

  void Visit_(const AttrStmt *op) {
    if (op->attr_key == "pragma_tensor_core") {
      if_use_tensor_core_ = true;
      return;
    } else {
      return IRVisitor::Visit_(op);
    }
  }

 private:
  bool if_use_tensor_core_{false};
};

class PrefetchScopeInjector : public IRMutator {
 public:
  bool HasShared(const Stmt &s) {
    if (auto store = s.as<Store>()) {
      is_nested_block_ = true;
      auto it = touched_.find(store->buffer_var.get());
      if (it != touched_.end()) {
        prefetch_var_ = store->buffer_var;
        return true;
      }
    } else if (auto loop = s.as<For>()) {
      if (HasShared(loop->body)) {
        return true;
      }
    } else if (auto attr = s.as<AttrStmt>()) {
      if (HasShared(attr->body)) {
        return true;
      }
    } else if (auto cond = s.as<IfThenElse>()) {
      if (HasShared(cond->then_case)) {
        return true;
      }
    }
    return false;
  }

  bool HasConstantOuterLoop() {
    return ((!loop_nest_.empty()) && (loop_nest_.back()->extent.as<IntImm>() != nullptr));
  }

  bool IsPrefetchBlock(const Stmt &s) {
    return (HasShared(s) && HasConstantOuterLoop());
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == air::ir::attr::storage_scope && op->value.as<StringImm>()->value == "shared") {
      touched_.insert(op->node.as<Variable>());
    } else if (op->attr_key == "shared_mem_promoted_complete") {
      if_shared_promoted_ = true;
    } else if (op->attr_key == "promote_register_to_global" || op->attr_key == "promote_register_to_shared") {
      if_shared_finished_ = true;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (IsPrefetchBlock(s)) {
      prefetch_outer_loop_ = (loop_nest_.back()->extent).as<IntImm>()->value;
      if_prefetch_injected_ = true;
      return AttrStmt::make(prefetch_var_, PREFETCH_SCOPE, 1, s);
    } else if (is_nested_block_) {
      return s;
    } else {
      loop_nest_.push_back(op);
      auto stmt = IRMutator::Mutate_(op, s);
      loop_nest_.pop_back();
      return stmt;
    }
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    if (IsPrefetchBlock(s)) {
      prefetch_outer_loop_ = (loop_nest_.back()->extent).as<IntImm>()->value;
      if_prefetch_injected_ = true;
      return AttrStmt::make(prefetch_var_, PREFETCH_SCOPE, 1, s);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Evaluate *op, const Stmt &s) final {
    if (is_const(op->value)) {
      return IRMutator::Mutate_(op, s);
    }
    if (const auto call = op->value.as<Call>()) {
      if (call->is_intrinsic(air::ir::intrinsic::tvm_storage_sync)) {
        if (if_prefetch_injected_ && (!if_shared_promoted_)) {
          return AttrStmt::make(Var(""), "delete_this_sync", 1, s);
        } else if (if_shared_promoted_ && (!if_shared_finished_)) {
          return AttrStmt::make(Var(""), "delete_this_sync_for_db", 1, s);
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    if (const auto ifthenelse = stmt.as<IfThenElse>()) {
      if (auto attr = ifthenelse->then_case.as<AttrStmt>()) {
        if (attr->attr_key == PREFETCH_SCOPE) {
          Stmt rotated_stmt = IfThenElse::make(ifthenelse->condition, attr->body, ifthenelse->else_case);
          rotated_stmt = AttrStmt::make(attr->node, attr->attr_key, attr->value, rotated_stmt);
          return rotated_stmt;
        }
      }
    }
    return stmt;
  }

  const bool GetIfPrefetchInjected() { return if_prefetch_injected_; }

  const int GetPrefetchOuterLoop() { return prefetch_outer_loop_; }

 private:
  std::unordered_set<const Variable *> touched_;
  VarExpr prefetch_var_;
  std::vector<const For *> loop_nest_;
  bool need_prefetch_{false};
  bool if_prefetch_injected_{false};
  bool if_shared_promoted_{false};
  bool if_shared_finished_{false};
  bool is_nested_block_{false};
  int prefetch_outer_loop_;
};

class IfResouceIsEnough : public IRVisitor {
 public:
  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == "bind_thread_x") {
      use_thread_y_ = false;
      return IRVisitor::Visit_(op);
    } else if (op->attr_key == air::ir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->var->name_hint == "threadIdx.x") {
        thread_x_var_ = iv->var;
        thread_x_value_ = op->value;
      }
      if (iv->var->name_hint == "threadIdx.y") {
        thread_y_var_ = iv->var;
        thread_y_value_ = op->value;
      }
      return IRVisitor::Visit_(op);
    } else if (op->attr_key == air::ir::attr::storage_scope) {
      if (auto alloc = op->body.as<Allocate>()) {
        if (op->value.as<StringImm>()->value == "shared") {
          shared_usage_ +=
            air::arith::ComputeReduce<Mul>(alloc->extents, Expr()) * alloc->type.lanes() * alloc->type.bytes();
        } else if (op->value.as<StringImm>()->value == "local") {
          promote_local_usage_ +=
            air::arith::ComputeReduce<Mul>(alloc->extents, Expr()) * alloc->type.lanes() * alloc->type.bytes();
        } else if (op->value.as<StringImm>()->value == "wmma.accumulator") {
          promote_local_usage_ += air::arith::ComputeReduce<Mul>(alloc->extents, Expr()) * alloc->type.lanes() /
                                  Expr(WMMA_FACTOR_C) * alloc->type.bytes();
        } else if (op->value.as<StringImm>()->value == "wmma.matrix_b" ||
                   op->value.as<StringImm>()->value == "wmma.matrix_a") {
          promote_local_usage_ += air::arith::ComputeReduce<Mul>(alloc->extents, Expr()) * alloc->type.lanes() /
                                  Expr(WMMA_FACTOR_AB) * alloc->type.bytes();
        }
      }
      return IRVisitor::Visit_(op);
    } else if (op->attr_key == PREFETCH_SCOPE) {
      in_prefetch_buffer_scope_ = true;
      Visit(op->body);
      if (transfer_loop_nest_.empty()) {
        auto store = op->body.as<Store>();
        if (store && store->index.as<Ramp>()) {
          prefetch_local_usage_ += (make_const(Int(BIT64), store->index.as<Ramp>()->lanes) *
                                    prefetch_data_type_.bytes());
        }
      } else {
        Expr current_local_usage = make_const(transfer_loop_nest_[0]->extent.type(), 1);
        for (unsigned i = 0; i < transfer_loop_nest_.size(); i++) {
          current_local_usage *= transfer_loop_nest_[i]->extent - transfer_loop_nest_[i]->min;
        }
        prefetch_local_usage_ += current_local_usage * prefetch_data_type_.bytes();
        transfer_loop_nest_.clear();
      }
      in_prefetch_buffer_scope_ = false;
    } else {
      return IRVisitor::Visit_(op);
    }
  }

  void Visit_(const For *op) {
    if (in_prefetch_buffer_scope_) transfer_loop_nest_.push_back(op);
    return IRVisitor::Visit_(op);
  }

  void Visit_(const Store *op) {
    if (in_prefetch_buffer_scope_) {
      if (const auto load = op->value.as<Load>()) {
        prefetch_data_type_ = load->type;
      }
    }
    return IRVisitor::Visit_(op);
  }

  const int GetBindThreadNum() { return (thread_x_value_ * thread_y_value_).as<IntImm>()->value; }

  const Var GetThreadGroupVar() {
    if (use_thread_y_)
      return thread_y_var_;
    else
      return thread_x_var_;
  }

  const Expr GetThreadGoupOffset() {
    if (use_thread_y_)
      return thread_y_value_;
    else
      return thread_x_value_;
  }
  const int GetTotalSharedUsage() { return shared_usage_.as<IntImm>()->value; }
  const int GetTotalLocalUsage() { return (promote_local_usage_ + prefetch_local_usage_).as<IntImm>()->value; }

 private:
  bool use_thread_y_{true};
  bool in_prefetch_buffer_scope_{false};
  Var thread_x_var_, thread_y_var_;
  Expr thread_x_value_, thread_y_value_;
  Expr shared_usage_{make_const(Int(BIT64), 0)};
  Expr promote_local_usage_{make_const(Int(BIT64), 0)};
  Expr prefetch_local_usage_{make_const(Int(BIT64), 0)};
  std::vector<const For *> transfer_loop_nest_;
  air::DataType prefetch_data_type_;
};

class ThreadGroupScopeInjector : public IRMutator {
 public:
  Stmt Inject(Stmt stmt, int thread_group, Var thread_var, Expr thread_offset) {
    thread_group_ = thread_group;
    thread_var_ = thread_var;
    thread_offset_ = thread_offset;
    return this->Mutate(stmt);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == air::ir::attr::thread_extent) {
      thread_extent_count_ += 1;
      Stmt body = Mutate(op->body);
      thread_extent_count_ -= 1;
      if (thread_extent_count_ == 0) {
        return AttrStmt::make(thread_var_, "use_thread_group", make_const(thread_offset_.type(), thread_group_),
                              AttrStmt::make(op->node, op->attr_key, op->value, body));
      } else {
        return AttrStmt::make(op->node, op->attr_key, op->value, body);
      }
    } else if (op->attr_key == PREFETCH_SCOPE) {
      Stmt body = Mutate(op->body);
      return AttrStmt::make(op->node, op->attr_key, op->value,
                            AttrStmt::make(thread_var_, THREAD_GROUP_OFFSET, thread_offset_, body));
    } else if (op->attr_key == "promote_register_to_shared" || op->attr_key == "promote_shared_to_global" ||
               op->attr_key == "promote_register_to_global" || op->attr_key == "shared_mem_promoted_complete") {
      Stmt body = Mutate(op->body);
      return AttrStmt::make(
        op->node, op->attr_key, op->value,
        AttrStmt::make(thread_var_, THREAD_GROUP_OFFSET, make_const(thread_offset_.type(), 0), body));
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

 private:
  int thread_group_;
  int thread_extent_count_ = 0;
  Var thread_var_;
  Expr thread_offset_;
};

Stmt InjectTransferBufferScope(Stmt stmt) {
  if (!IfTensorCore().IfUseTensorCore(stmt)) return stmt;
  PrefetchScopeInjector prefetch_injector;
  Stmt new_stmt = prefetch_injector.Mutate(stmt);
  const bool if_prefetch_injected = prefetch_injector.GetIfPrefetchInjected();
  if (!if_prefetch_injected) return stmt;
  const int thread_group = 2;
  const bool tuning{true};
  bool enable_double_buffer{false};
  bool enable_transfer_buffer{false};
  bool enable_thread_group{false};
  IfResouceIsEnough resource_calc;
  resource_calc.Visit(new_stmt);
  if (tuning) {
    // tuning: manually control by attrs
    enable_double_buffer = g_attrs.GetBool(kEnableDoubleBuffer, false);
    enable_transfer_buffer = g_attrs.GetBool(kEnableTransferBuffer, true);
    enable_thread_group = g_attrs.GetBool(kEnableThreadGroup, false);
  } else {
    // not tuning: auto-analyse
    const int total_shared_usage = resource_calc.GetTotalSharedUsage();
    float shared_mem_rate = float(total_shared_usage * 2) / float(common::ADVANCED_SHARED_MEMORY_SIZE);
    const int bind_thread_num = resource_calc.GetBindThreadNum();
    const int total_local_usage = resource_calc.GetTotalLocalUsage();
    float local_mem_rate = float(total_local_usage * bind_thread_num) / float(REGISTER_FILE_SIZE_PER_SM);
    if (shared_mem_rate >= 1 && local_mem_rate >= 1) return stmt;
    if (shared_mem_rate <= local_mem_rate) return new_stmt;
    int max_bind_thread_num = TOTAL_THREAD_NUM_PER_BLOCK / thread_group;
    const int prefetch_outer_loop = prefetch_injector.GetPrefetchOuterLoop();
    if ((prefetch_outer_loop > MIN_OUTER_LOOP) && (local_mem_rate < 1) &&
        ((bind_thread_num < max_bind_thread_num / 2) || (prefetch_outer_loop < MAX_OUTER_LOOP))) {
      enable_transfer_buffer = true;
      g_attrs.Set(kEnableTransferBuffer, air::make_const(Int(BIT32), true));
      if ((local_mem_rate < (1.0 / float(thread_group) / 2.0)) && (bind_thread_num < max_bind_thread_num)) {
        enable_thread_group = true;
      }
    }
  }
  // avoid enabling two modes
  if (enable_double_buffer) {
    enable_transfer_buffer = false;
  }
  Stmt stmt_after_prefetch = stmt;
  if (enable_transfer_buffer || enable_double_buffer) {
    if (enable_thread_group) {
      const Var thread_group_var = resource_calc.GetThreadGroupVar();
      const Expr thread_group_offset = resource_calc.GetThreadGoupOffset();
      stmt_after_prefetch =
        ThreadGroupScopeInjector().Inject(new_stmt, thread_group, thread_group_var, thread_group_offset);
    } else {
      stmt_after_prefetch = new_stmt;
    }
  }
  // add an attr of prefetch_mode
  int prefetch_mode = static_cast<int>(PrefetchMode::DEFAULT);
  if (enable_double_buffer && enable_thread_group) {
    prefetch_mode = static_cast<int>(PrefetchMode::DOUBLEBUFFER_THREADGROUP);
  } else if (enable_transfer_buffer && enable_thread_group) {
    prefetch_mode = static_cast<int>(PrefetchMode::TRANSFERBUFFER_THREADGROUP);
  } else if (enable_double_buffer) {
    prefetch_mode = static_cast<int>(PrefetchMode::DOUBLEBUFFER);
  } else if (enable_transfer_buffer) {
    prefetch_mode = static_cast<int>(PrefetchMode::TRANSFERBUFFER);
  }
  return AttrStmt::make(Expr("INFO"), ATTR_PREFETCH_MODE, prefetch_mode, stmt_after_prefetch);
}

}  // namespace ir
}  // namespace akg