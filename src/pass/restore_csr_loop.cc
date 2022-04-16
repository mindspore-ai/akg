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
#include <pass/storage_access.h>
#include "tvm.h"
#include "build_module.h"
#include "utils.h"

namespace akg {
namespace ir {

class MaxVarVisitor : public IRVisitor {
  void Visit_(const Variable *op) final {
    for (const auto &it : g_csr) {
      auto var = it.first.as<Variable>();
      if (var != nullptr && var->name_hint == op->name_hint) {
        csr_dynamic_extent_ = air::Downcast<Array<Expr>>(it.second);
      }
    }
  }

 public:
  Array<Expr> csr_dynamic_extent_;
};

Array<Expr> GetCsrDynamicExtent(const Expr &e) {
  auto max_var_visitor = MaxVarVisitor();
  max_var_visitor.Visit(e);
  return max_var_visitor.csr_dynamic_extent_;
}

class RecordInitStmt : public IRVisitor {
  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == "InitStmt") {
      has_init_stmt_ = true;
      auto provide = op->body.as<Provide>();
      CHECK(provide);
      if (provide->func.defined()) {
        init_tensor_ = provide->func;
      }
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const Block *op) final {
    IRVisitor::Visit(op->first);
    if (has_init_stmt_) {
      has_init_stmt_ = false;
      init_stmt_ = Block::make(op->first, Evaluate::make(Call::make(Int(32), "tvm_storage_sync",
                              {StringImm::make("shared")}, Call::Intrinsic)));
      return;
    }
    IRVisitor::Visit(op->rest);
    if (has_init_stmt_) {
      has_init_stmt_ = false;
      init_stmt_ = Block::make(op->rest, Evaluate::make(Call::make(Int(32), "tvm_storage_sync",
                              {StringImm::make("shared")}, Call::Intrinsic)));
      return;
    }
  }

  bool has_init_stmt_{false};

 public:
  Stmt init_stmt_;
  FunctionRef init_tensor_;
};

class InsertInitStmt : public IRMutator {
 public:
  explicit InsertInitStmt(Stmt init_stmt, FunctionRef init_tensor) : init_stmt_(init_stmt), init_tensor_(init_tensor) {}

 private:
  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    if (op->func.defined() && op->func->func_name() == init_tensor_->func_name()) {
      init_stmt_ = TensorSubstitute(init_stmt_, init_tensor_, op->func, op->value_index);
      in_init_ = true;
      init_stmt_ = IRMutator::Mutate(init_stmt_);
      auto body = Block::make(init_stmt_, op->body);
      return Realize::make(op->func, op->value_index, op->type, op->bounds, op->condition, body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    if (in_init_ && cur_loop_.defined()) {
      auto cond = And::make(op->condition, EQ::make(cur_loop_, Expr(0)));
      return IfThenElse::make(cond, op->then_case, op->else_case);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    Var tmp_loop = cur_loop_;
    cur_loop_ = op->loop_var;
    auto body = IRMutator::Mutate(op->body);
    cur_loop_ = tmp_loop;
    if (!body.same_as(op->body)) {
      return For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, body);
    }
    return s;
  }

  Stmt init_stmt_;
  FunctionRef init_tensor_;
  Var cur_loop_;
  bool in_init_{false};
};

class RemoveCsrBranch : public IRMutator {
  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (GetCsrDynamicExtent(op->extent).size() > 0) {
      csr_loop_ = op;
      check_extent_ = true;
      csr_extent_ = csr_loop_->extent;
      IRMutator::Mutate(csr_extent_);
      check_extent_ = false;
      auto body = IRMutator::Mutate(op->body);
      if (has_gm_read_) {
        has_gm_read_ = false;
        csr_loop_ = nullptr;
        return body;
      }
      return For::make(op->loop_var, Expr(0), csr_extent_, op->for_type, op->device_api, body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Variable *op, const Expr &e) final {
    if (check_extent_) {
      if (GetCsrDynamicExtent(e).size() > 0) {
        csr_extent_ = e;
      }
    }
    return e;
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (csr_loop_ != nullptr && op->attr_key == "GMRead") {
      has_gm_read_ = true;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Block *op, const Stmt &s) final {
    if (has_gm_read_) {
      if (auto eval = op->first.as<Evaluate>()) {
        if (auto call = eval->value.as<Call>()) {
          if (call->is_intrinsic(air::ir::intrinsic::tvm_storage_sync)) {
            CHECK(csr_loop_ != nullptr && csr_extent_.defined());
            auto body = IRMutator::Mutate(op->rest);
            auto for_stmt =
              For::make(csr_loop_->loop_var, Expr(0), csr_extent_, csr_loop_->for_type, csr_loop_->device_api, body);
            return Block::make(op->first, for_stmt);
          }
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    if (GetCsrDynamicExtent(op->condition).size() > 0) {
      return IRMutator::Mutate(op->then_case);
    }
    return IRMutator::Mutate_(op, s);
  }

  const For *csr_loop_{nullptr};
  Expr csr_extent_;
  bool has_gm_read_{false};
  bool check_extent_{false};
};

class VectorizedForLoop : public IRMutator {
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "vector_length") {
      vectorization_size_ = op->value;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (auto for_stmt = op->body.as<For>()) {
      if (for_stmt->for_type != ForType::Vectorized) {
        return IRMutator::Mutate_(op, s);
      }
      inner_loop_var_ = op->loop_var;
      const Variable *var = GetVariableFromCSR();
      max_var_ = Variable::make(var->type, var->name_hint);
      auto body = Mutate(op->body);
      Expr vec_loop_extent = Div::make(max_var_, vectorization_size_);
      return For::make(op->loop_var, Expr(0), vec_loop_extent + 1, op->for_type, op->device_api, body);
    } else if (op->for_type == ForType::Vectorized) {
      Expr vec_loop_extent = Div::make(max_var_, vectorization_size_);
      Stmt vectorized_for =
        For::make(op->loop_var, Expr(0), vectorization_size_, op->for_type, op->device_api, op->body);
      Stmt serial_for = For::make(op->loop_var, 0, max_var_ - inner_loop_var_ * vectorization_size_, ForType::Serial,
                                  op->device_api, op->body);
      Expr condition = (inner_loop_var_ != vec_loop_extent);
      Stmt body_with_tail = IfThenElse::make(condition, vectorized_for, serial_for);
      return body_with_tail;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final { return Mutate(op->then_case); }

  Expr vectorization_size_{};
  Var inner_loop_var_{};
  Var max_var_{};
};

class RestoreMaxVar : public IRMutator {
  Expr Mutate_(const Variable *op, const Expr &e) final {
    Array<Expr> extent_arr = GetCsrDynamicExtent(e);
    if (extent_arr.size() > 0) {
      int id = std::max(max_var_id_, 1);
      CHECK_LT(id, static_cast<int>(extent_arr.size()));
      return extent_arr[id];
    }
    return e;
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "max_var_id") {
      auto count = op->value.as<IntImm>();
      CHECK(count);
      max_var_id_ = count->value;
      return IRMutator::Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Block *op, const Stmt &s) final {
    int tmp_max_var_id = max_var_id_;
    auto first = IRMutator::Mutate(op->first);
    max_var_id_ = tmp_max_var_id;
    auto rest = IRMutator::Mutate(op->rest);
    if (!first.same_as(op->first) || !rest.same_as(op->rest)) {
      return Block::make(first, rest);
    }
    return s;
  }

  int max_var_id_{-1};
};

class SwapCsrLoopWithInner : public IRMutator {
  Stmt Mutate_(const For *op, const Stmt &s) final {
    bool is_outer = loops_.empty();
    loops_[op->loop_var.get()] = op;
    Stmt stmt;
    if (op->extent.as<Sub>() != nullptr) {
      bool need_swap = false;
      PostOrderVisit(op->extent, [this, &op, &need_swap] (const NodeRef &n) {
        if (n->IsInstance<Variable>()) {
          auto var = Downcast<Var>(n);
          if (var->name_hint.rfind("blockIdx", 0) != std::string::npos ||
              var->name_hint.rfind("threadIdx", 0) != std::string::npos) {
            return;
          }
          if (var.same_as(op->loop_var)) {
            swap_any_ = true;
            need_swap = true;
          } else if (loops_.count(var.get()) == 0) {
            swap_var_ = var;
            need_swap = true;
          }
        }
      });
      if (need_swap) {
        csr_var_ = op->loop_var;
        auto body = IRMutator::Mutate(op->body);
        CHECK(swap_var_.defined() && loops_.count(swap_var_.get()) > 0);
        auto swap_loop = loops_[swap_var_.get()];
        stmt = For::make(
          op->loop_var, swap_loop->min, swap_loop->extent, swap_loop->for_type, swap_loop->device_api, body);
        csr_var_ = Var();
        swap_var_ = Var();
        swap_any_ = false;
      } else {
        stmt = s;
      }
    } else if (op->loop_var.same_as(swap_var_) || swap_any_) {
      CHECK(csr_var_.defined() && loops_.count(csr_var_.get()) > 0);
      swap_var_ = op->loop_var;
      auto csr_loop = loops_[csr_var_.get()];
      Map<Var, Expr> vmap;
      vmap.Set(swap_var_, csr_var_);
      vmap.Set(csr_var_, swap_var_);
      auto body = Substitute(op->body, vmap);
      auto extent = Substitute(csr_loop->extent, vmap);
      stmt = For::make(op->loop_var, csr_loop->min, extent, csr_loop->for_type, csr_loop->device_api, body);
    } else {
      stmt = IRMutator::Mutate_(op, s);
    }
    if (is_outer) {
      loops_.clear();
    }
    return stmt;
  }

  std::unordered_map<const Variable *, const For *> loops_;
  Var csr_var_;
  Var swap_var_;
  bool swap_any_{false};
};

class VectorizationChecker : public IRVisitor {
 public:
  bool is_vectorized_{false};
  void Visit_(const For *op) final {
    is_vectorized_ |= (op->for_type == ForType::Vectorized);
    IRVisitor::Visit_(op);
  }
};

class CsrLoopStride : public IRMutator {
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) {
    if (op->attr_key == CSR_MAP_THREAD) {
      auto thread_num = op->value.as<IntImm>();
      CHECK(thread_num);
      stride_ = thread_num->value;
      return IRMutator::Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (op->extent.as<Sub>() != nullptr) {
      csr_loop_var_ = op->loop_var;
      auto body = IRMutator::Mutate(op->body);
      csr_loop_var_ = Expr();
      Stmt for_stmt;
      if (!body.same_as(op->body)) {
        for_stmt = For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, body);
      } else {
        for_stmt = s;
      }
      return AttrStmt::make(Expr("INFO"), "csr_dynamic_loop", Expr(stride_), for_stmt);
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Mul *op, const Expr &e) final {
    if (csr_loop_var_.defined()) {
      if (csr_loop_var_.same_as(op->a)) {
        return op->a;
      }
      if (csr_loop_var_.same_as(op->b)) {
        return op->b;
      }
    }
    return IRMutator::Mutate_(op, e);
  }

  int stride_{1};
  Expr csr_loop_var_;
};

Stmt RestoreCsrLoop(Stmt stmt, Map<Tensor, Buffer> extern_buffer, bool target_cuda) {
  auto v_checker = VectorizationChecker();
  v_checker.Visit(stmt);
  if (!target_cuda) {
    if (v_checker.is_vectorized_) {
      stmt = VectorizedForLoop().Mutate(stmt);
    }
    stmt = RestoreMaxVar().Mutate(stmt);
    g_csr.Clear();
    stmt = SwapCsrLoopWithInner().Mutate(stmt);
    return stmt;
  }
  stmt = RemoveCsrBranch().Mutate(stmt);
  stmt = RestoreMaxVar().Mutate(stmt);
  g_csr.Clear();
  stmt = SwapCsrLoopWithInner().Mutate(stmt);
  stmt = CsrLoopStride().Mutate(stmt);
  return stmt;
}

}  // namespace ir
}  // namespace akg
