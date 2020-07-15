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
#include <arithmetic/compute_expr.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>

#include "pass/ir_util.h"
#include "ir_pass.h"

namespace akg {
namespace ir {
/**
 * ir transform of this pass:
 * 0. input IR sample:
 *   for (i, 0, n) {
 *     A = alloc(100)
 *     gm_to_ubuf(A, gm[i])
 *     calc(A)
 *   }
 * 1. double buffer unroll(AutoDoubleBufferInjector):
 *   for (i.db, 0, n/2) {
 *     A = alloc(100)
 *     gm_to_ubuf(A, gm[i.db*2])
 *     calc(A)
 *     A = alloc(100)
 *     gm_to_ubuf(A, gm[i.db*2+1])
 *     calc(A)
 *   }
 */
class DbFinder : public IRVisitor {
 public:
  DbFinder() {}
  ~DbFinder() override = default;
  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == air::ir::attr::double_buffer_scope) alreadyAdd_ = true;
  }
  bool alreadyAdd_{false};
};

class DetectSupportFor : public IRVisitor {
 public:
  DetectSupportFor() {}
  ~DetectSupportFor() override = default;
  void Visit_(const For *op) final {
    deq_outer_loops_.push_front(op);
    IRVisitor::Visit_(op);
    deq_outer_loops_.pop_front();
  }
  void Visit_(const Allocate *op) final {
    if (!deq_outer_loops_.empty()) {
      db_for_.insert(deq_outer_loops_[0]);
    }
    IRVisitor::Visit_(op);
  }

  std::unordered_set<const For *> db_for_;

 private:
  std::deque<const For *> deq_outer_loops_;
};

class AutoDoubleBufferInjector : public IRMutator {
 public:
  AutoDoubleBufferInjector() {}
  ~AutoDoubleBufferInjector() override = default;

  Stmt Inject(const Stmt &stmt) {
    DetectSupportFor dsf;
    dsf.Visit(stmt);
    db_loop_ = std::move(dsf.db_for_);
    if (db_loop_.empty()) {
      return stmt;
    }
    return Mutate(stmt);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (db_loop_.count(op) == 0) {
      return IRMutator::Mutate_(op, s);
    }
    Stmt body = IRMutator::Mutate(op->body);
    Expr factor = make_const(op->loop_var.type(), db_lane_);
    Var loop_var(op->loop_var->name_hint + ".db", op->loop_var.type());

    Expr tail_ext = Simplify_cce(op->extent % factor);
    air::arith::Analyzer analyzer_;
    bool need_tail = (!analyzer_.CanProve(tail_ext == 0));
    std::vector<Stmt> lane_body;
    for (int i = 0; i < db_lane_; i++) {
      std::unordered_map<const Variable *, Expr> vmap;
      Expr new_loop_var = loop_var * factor + make_const(factor.type(), i) + op->min;
      vmap[op->loop_var.get()] = new_loop_var;
      Stmt st = air::ir::Substitute(body, vmap);
      if (!is_const(op->extent) && need_tail && i != 0) {
        st = IfThenElse::make(new_loop_var < op->extent, st, Stmt());
      }
      lane_body.push_back(st);
    }
    Stmt stmt = air::ir::MergeSeq(lane_body);
    CHECK(factor.as<IntImm>());
    Expr new_ext = (is_const(op->extent) ? (op->extent / factor) : ((op->extent + factor - 1) / factor));
    auto for_type = (analyzer_.CanProve(new_ext < auto_unroll_bound_) && is_positive_const(new_ext)) ? ForType::Unrolled
                                                                                                     : ForType::Serial;
    stmt = For::make(loop_var, make_const(op->loop_var.type(), 0), new_ext, for_type, op->device_api, stmt);

    if (is_const(op->extent) && need_tail) {
      Var loop_var_tail(op->loop_var->name_hint + ".db.tail", op->loop_var.type());
      std::unordered_map<const Variable *, Expr> vmap;
      vmap[op->loop_var.get()] = op->extent / factor * factor + loop_var_tail + op->min;
      Stmt tail_stmt = air::ir::Substitute(body, vmap);
      auto for_type_ = (analyzer_.CanProve(tail_ext < auto_unroll_bound_) && is_positive_const(tail_ext))
                         ? ForType::Unrolled
                         : ForType::Serial;
      tail_stmt =
        For::make(loop_var_tail, make_const(op->loop_var.type(), 0), tail_ext, for_type_, op->device_api, tail_stmt);
      stmt = Block::make(stmt, tail_stmt);
    }
    return stmt;
  }

 private:
  std::unordered_set<const For *> db_loop_;
  // pipe buffer lane number
  int db_lane_{2};
  const int auto_unroll_bound_{2};
};

/**
 * Inject auto double buffer pass entry
 * @param [in] op    stmt The statement to be transformed
 * @return           Transformed stmt
 */
Stmt AutoDoubleBuffer(Stmt stmt) {
  DbFinder dbfinder;
  dbfinder.Visit(stmt);
  if (dbfinder.alreadyAdd_) {
    return stmt;
  }
  stmt = AutoDoubleBufferInjector().Inject(stmt);
  return air::ir::ConvertSSA(stmt);
}
}  // namespace ir
}  // namespace akg
