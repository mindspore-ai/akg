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
    for (const auto &it: g_csr) {
      auto var = it.first.as<Variable>();
      if (var != nullptr && var->name_hint == op->name_hint) {
        csr_dynamic_extent_ = air::Downcast<Expr>(it.second);
      }
    }
  }

 public:
  Expr csr_dynamic_extent_;
};

Expr GetCsrDynamicExtent(const Expr &e) {
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
  explicit InsertInitStmt(Stmt init_stmt, FunctionRef init_tensor) :
    init_stmt_(init_stmt), init_tensor_(init_tensor) {}

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
    if (GetCsrDynamicExtent(op->extent).defined()) {
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
      if (GetCsrDynamicExtent(e).defined()) {
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
            auto for_stmt = For::make(
              csr_loop_->loop_var, Expr(0), csr_extent_, csr_loop_->for_type, csr_loop_->device_api, body);
            return Block::make(op->first, for_stmt);
          }
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    if (GetCsrDynamicExtent(op->condition).defined()) {
      return IRMutator::Mutate(op->then_case);
    }
    return IRMutator::Mutate_(op, s);
  }

  const For *csr_loop_{nullptr};
  Expr csr_extent_;
  bool has_gm_read_{false};
  bool check_extent_{false};
};

class CombineCsrBlock : public IRMutator {
  Stmt Mutate_(const Block *op, const Stmt &s) final {
    if (check_block_) {
      check_block_ = false;
      auto first = IRMutator::Mutate(op->first);
      if (csr_extent_) {
        csr_extent_ = false;
        return first;
      }
      auto rest = IRMutator::Mutate(op->rest);
      if (csr_extent_) {
        auto record_init = RecordInitStmt();
        record_init.Visit(first);
        if (record_init.init_stmt_.defined()) {
          rest = InsertInitStmt(record_init.init_stmt_, record_init.init_tensor_).Mutate(rest);
        }
        csr_extent_ = false;
        return rest;
      }
      if (!first.same_as(op->first) || !rest.same_as(op->rest)) {
        return Block::make(first, rest);
      }
      return s;
    }
    check_block_ = false;
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    check_block_ = false;
    auto min = op->min;
    auto extent = op->extent;
    if (!csr_extent_) {
      if (GetCsrDynamicExtent(op->extent).defined()) {
        csr_extent_ = true;
      }
    }
    auto body = IRMutator::Mutate(op->body);
    if (!min.same_as(op->min) || !extent.same_as(op->extent) || !body.same_as(op->body)) {
      return For::make(op->loop_var, min, extent, op->for_type, op->device_api, body);
    }
    return s;
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    check_block_ = false;
    return IRMutator::Mutate_(op, s);
  }

  bool check_block_{true};
  bool csr_extent_{false};
};

class RestoreMaxVar : public IRMutator {
  Expr Mutate_(const Call *op, const Expr &e) final {
    if (g_csr.count(e) > 0) {
      auto new_expr = g_csr.at(e);
      if (auto new_call = new_expr.as<Call>()) {
        Array<Expr> args;
        for (auto orig_arg: new_call->args) {
          args.push_back(IRMutator::Mutate(orig_arg));
        }
        return Call::make(
          new_call->type, new_call->name, args, new_call->call_type, new_call->func, new_call->value_index);
      }
    }
    return e;
  }

  Expr Mutate_(const Variable *op, const Expr &e) final {
    Expr extent = GetCsrDynamicExtent(e);
    if (extent.defined()) {
      return IRMutator::Mutate(air::Downcast<Expr>(extent));
    }
    return e;
  }
};

class CsrInnerLoopVar : public IRVisitor {
 public:
  void Visit(const NodeRef &node) final {
    if (in_csr_extent_) {
      if (node->IsInstance<ExprNode>()) {
        auto e = air::Downcast<Expr>(node);
        if (e.as<Variable>() != nullptr) {
          inner_loop_var_ = e;
          return;
        }
      }
    }
    IRVisitor::Visit(node);
  }

  bool need_swap_{false};
  Expr csr_loop_var_;
  Expr csr_extent_;
  Expr inner_loop_var_;
  Expr inner_extent_;

 private:
  void Visit_(const For *op) final {
    if (op->extent.as<Sub>() != nullptr) {
      csr_loop_var_ = op->loop_var;
      csr_extent_ = op->extent;
      in_csr_extent_ = true;
      IRVisitor::Visit(op->extent);
      in_csr_extent_ = false;
      in_csr_loop_ = true;
      IRVisitor::Visit(op->body);
      in_csr_loop_ = false;
      return;
    } else if (in_csr_loop_){
      if (inner_loop_var_.defined() && inner_loop_var_.same_as(op->loop_var)) {
        inner_extent_ = op->extent;
        need_swap_ = true;
        return;
      }
    }
    IRVisitor::Visit_(op);
  }

  bool in_csr_loop_{false};
  bool in_csr_extent_{false};
};

class SwapCsrLoopWithInner : public IRMutator {
 public:
  explicit SwapCsrLoopWithInner(Var csr_loop_var, Expr csr_extent, Var outer_loop_var, Expr outer_extent) :
    csr_loop_var_(outer_loop_var), csr_extent_(csr_extent),
    outer_loop_var_(csr_loop_var), outer_extent_(outer_extent) {}

 private:
  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (op->loop_var.same_as(csr_loop_var_)) {
      in_csr_loop_ = true;
      auto body = IRMutator::Mutate(op->body);
      auto extent = IRMutator::Mutate(csr_extent_);
      in_csr_loop_ = false;
      return For::make(csr_loop_var_, op->min, extent, op->for_type, op->device_api, body);
    } else if (op->loop_var.same_as(outer_loop_var_)) {
      auto body = IRMutator::Mutate(op->body);
      return For::make(outer_loop_var_, op->min, outer_extent_, op->for_type, op->device_api, body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Variable *op, const Expr &e) final {
    if (in_csr_loop_) {
      if (e.same_as(csr_loop_var_)) {
        return outer_loop_var_;
      }
      if (e.same_as(outer_loop_var_)) {
        return csr_loop_var_;
      }
    }
    return e;
  }

  bool in_csr_loop_{false};
  Var csr_loop_var_;
  Expr csr_extent_;
  Var outer_loop_var_;
  Expr outer_extent_;
};

class CsrLoopStride : public IRMutator {
 public:
  explicit CsrLoopStride(FunctionRef csr_sum_op) : csr_sum_op_(csr_sum_op) {}

 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) {
    if (op->node->IsInstance<IterVarNode>()) {
      auto iter_var = air::Downcast<IterVar>(op->node);
      if (iter_var->iter_type == air::kThreadIndex) {
        if (iter_var->thread_tag == "threadIdx.x") {
          if (auto thread_num = op->value.as<IntImm>()) {
            stride_ = thread_num->value;
            thread_var_ = iter_var->var;
          }
        } else if (iter_var->thread_tag == "blockIdx.y") {
          block_var_ = iter_var->var;
        }
      }
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
      auto stmt = AttrStmt::make(Expr("INFO"), "csr_dynamic_loop", Expr(stride_), for_stmt);
      if (csr_sum_op_.defined()) {
        CHECK(thread_var_.defined() && block_var_.defined());
        auto cond = EQ::make(thread_var_, Expr(0));
        auto provide = Provide::make(csr_sum_op_, 0, FloatImm::make(Float(32), 0.0), {block_var_});
        auto if_stmt = IfThenElse::make(cond, provide, Stmt());
        stmt = Block::make(if_stmt, stmt);
      }
      return stmt;
    } else {
      csr_sum_op_ = FunctionRef();
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
  Expr thread_var_;
  Expr block_var_;
  FunctionRef csr_sum_op_;
};

Stmt RestoreCsrLoop(Stmt stmt, Map<Tensor, Buffer> extern_buffer) {
  stmt = RemoveCsrBranch().Mutate(stmt);
  stmt = CombineCsrBlock().Mutate(stmt);
  stmt = RestoreMaxVar().Mutate(stmt);
  g_csr.Clear();
  auto csr_inner_loop = CsrInnerLoopVar();
  csr_inner_loop.Visit(stmt);
  if (csr_inner_loop.need_swap_) {
    CHECK(csr_inner_loop.csr_loop_var_.defined() && csr_inner_loop.csr_extent_.defined() &&
          csr_inner_loop.inner_loop_var_.defined() && csr_inner_loop.inner_extent_.defined());
    stmt = SwapCsrLoopWithInner(air::Downcast<Var>(csr_inner_loop.csr_loop_var_), csr_inner_loop.csr_extent_,
                                air::Downcast<Var>(csr_inner_loop.inner_loop_var_), csr_inner_loop.inner_extent_).Mutate(stmt);
  }
  FunctionRef csr_sum_op;
  for (auto pair: extern_buffer) {
    auto extern_op = pair.first->op.as<air::ExternOpNode>();
    if (extern_op != nullptr && extern_op->func_name().find("csr_reduce_sum") != std::string::npos) {
      csr_sum_op = pair.first->op;
      break;
    }
  }
  stmt = CsrLoopStride(csr_sum_op).Mutate(stmt);
  return stmt;
}

}  // namespace ir
}  // namespace akg