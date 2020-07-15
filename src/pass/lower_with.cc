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
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <ir_pass.h>
#include <pass/ir_util.h>
#include <pass/storage_access.h>

namespace akg {
namespace ir {
/*

rollback with

*/

// t_local_ub(0) = with()
// a() = t_local_ub(0)
// -->
// a() = with()

class CopyPropagation : public IRMutator {
 public:
  Stmt Run(Stmt stmt) {
    stmt = this->Mutate(stmt);
    rm_provide_ = true;
    stmt = this->Mutate(stmt);
    return stmt;
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == air::ir::attr::realize_scope && op->value.as<StringImm>()) {
      storage_scope_[op->node.get()] = op->value.as<StringImm>()->value;
    }
    if (op->attr_key == "pragma_emit_insn" && op->value.as<StringImm>() &&
        op->value.as<StringImm>()->value == "dma_copy") {
      in_dma_copy_ = true;
      Stmt stmt = IRMutator::Mutate_(op, s);
      in_dma_copy_ = false;
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }

  // disable optimize for control flow
  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    def_.clear();
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    def_.clear();
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (rm_provide_ && in_dma_copy_) {
      // remove this stmt
      if (provides_.count(op->func.get())) {
        return Evaluate::make(0);
      }
      return s;
    }
    in_provide_ = true;
    same_buf_mv_ = false;
    cur_dst_ = op->func.get();
    Stmt stmt = IRMutator::Mutate_(op, s);
    if (same_buf_mv_) {
      def_[op->func.get()] = op;
    }
    in_provide_ = false;
    return stmt;
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (in_provide_) {
      // CopyPropagation for the same buffer move
      auto src_buf = storage_scope_[op->func.get()];
      auto dst_buf = storage_scope_[cur_dst_];
      // minimize constraint for dma_copy, leave CP chance for a = b; c = a + f
      if (src_buf == dst_buf && in_dma_copy_) {
        same_buf_mv_ = true;
      }
      if (def_.count(op->func.get())) {
        // insert provide we need to remove
        provides_.insert(op->func.get());
        return def_[op->func.get()]->value;
      }
    }
    return IRMutator::Mutate_(op, e);
  }

 private:
  bool rm_provide_{false};
  bool in_provide_{false};
  std::unordered_map<const Node *, const Provide *> def_;
  std::unordered_set<const Node *> provides_;
  // Storage scope
  std::unordered_map<const Node *, std::string> storage_scope_;
  // cur provide
  const Node *cur_dst_{nullptr};
  bool same_buf_mv_{false};
  bool in_dma_copy_{false};
};

struct CallInfo {
  // func of lhs or rhs
  const Node *func;
  // index in args, and its related imm value
  std::map<size_t, Expr> idx_val;
  // index in args, and its original call
  std::map<size_t, Expr> idx_call;
};

class RollBackTensorIdx : public IRMutator {
 public:
  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    rhs_tensor_idx_.clear();
    lhs_tensor_idx_.clear();
    Stmt stmt = IRMutator::Mutate_(op, s);

    // return when this is not a tensor idx stmt
    if (rhs_tensor_idx_.size() == 0 && lhs_tensor_idx_.size() == 0) {
      return stmt;
    }

    CallInfo call_lhs;
    Array<Expr> lhs_args;
    for (size_t i = 0; i < op->args.size(); i++) {
      if (lhs_tensor_idx_.count(i)) {
        call_lhs.func = op->func.get();
        call_lhs.idx_call[i] = lhs_tensor_idx_[i];
        call_lhs.idx_val[i] = op->args[i];
        need_rebuild_.push_back(call_lhs);
        lhs_args.push_back(lhs_tensor_idx_[i]);
      } else {
        lhs_args.push_back(op->args[i]);
      }
    }
    const auto n = stmt.as<Provide>();
    CHECK(n);
    return Provide::make(op->func, op->value_index, n->value, lhs_args);
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    Expr val = IRMutator::Mutate_(op, e);
    const Call *opn = val.as<Call>();
    CHECK(opn);
    if (op->call_type == Call::PureIntrinsic) {
      CHECK(!opn->name.empty());
      if (opn->name == "lhs") {
        int pos = opn->args[1].as<IntImm>()->value;
        lhs_tensor_idx_[pos] = opn->args[0];
      }
      if (opn->name == "rhs") {
        int pos = opn->args[1].as<IntImm>()->value;
        rhs_tensor_idx_[pos] = opn->args[0];
      }
      if (opn->name == "orig") {
        rhs_ = opn->args[0];
        const Call *call = rhs_.as<Call>();
        if (call != nullptr) {
          CallInfo call_rhs;
          call_rhs.func = call->func.get();
          for (auto i : rhs_tensor_idx_) {
            call_rhs.idx_val[i.first] = call->args[i.first];
            call_rhs.idx_call[i.first] = i.second;
          }
          if (!RepeatCall(call_rhs)) {
            need_rebuild_.push_back(call_rhs);
          }
        }
      }
    }
    if (opn->call_type == Call::PureIntrinsic && opn->name == "with") {
      val = RebuildRHS(rhs_, rhs_tensor_idx_);
      rhs_tensor_idx_.clear();
    }
    return val;
  }

  // need rebuild
  std::vector<CallInfo> need_rebuild_;

 private:
  // rebuild rhs
  Expr RebuildRHS(const Expr &rhs_t, std::map<size_t, Expr> &rhs_tensor_idx_t) {
    Array<Expr> rhs_args;
    // roll back tensor expr
    Expr val = rhs_t;
    const Call *rhs = rhs_t.as<Call>();
    if (rhs && rhs->call_type == Call::Halide) {
      for (size_t i = 0; i < rhs->args.size(); i++) {
        if (rhs_tensor_idx_t.count(i)) {
          rhs_args.push_back(rhs_tensor_idx_t[i]);
        } else {
          rhs_args.push_back(rhs->args[i]);
        }
      }
      val = Call::make(rhs->type, rhs->name, rhs_args, rhs->call_type, rhs->func, rhs->value_index);
    }
    return val;
  }

  // for two objects, only if func, pos, val are the same, respectively
  bool RepeatCall(const CallInfo &v) {
    for (auto i : need_rebuild_) {
      if (i.func == v.func && i.idx_val.size() == v.idx_val.size()) {
        bool forall = false;
        for (auto j : v.idx_val) {
          if (i.idx_val.count(j.first) && Equal(i.idx_val[j.first], j.second)) {
            forall = true;
          } else {
            forall = false;
            break;
          }
        }
        if (forall) return forall;
      }
    }
    return false;
  }

  Expr rhs_;
  std::map<size_t, Expr> lhs_tensor_idx_;
  std::map<size_t, Expr> rhs_tensor_idx_;
};

Stmt LowerWith(Stmt stmt) {
  RollBackTensorIdx rbi;
  stmt = rbi.Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
