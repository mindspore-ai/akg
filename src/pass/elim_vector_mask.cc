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
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include "pass/common.h"

namespace akg {
namespace ir {
class VecMaskElimPlan : public IRVisitor {
 public:
  void Plan(const Stmt stmt) {
    this->Visit(stmt);
    TrimPending();
  }

  void Visit_(const Evaluate *op) final {
    if (const Call *call = op->value.as<Call>()) {
      if (call->name == "set_vector_mask") {
        if (first_.type == FT_DUP_PEND) {
          first_.type = FT_NONE;
          first_.vmask = Expr();
        }
        TrimPending();
        if (cur_vmask_.defined() && IsEqual(cur_vmask_, op->value)) {
          rm_vmask_.insert(call);
          if (first_.type == FT_NONE) {
            first_.vmask = op->value;
            first_.type = FT_DUP_PEND;
          }
        } else {
          cur_pend_ = op->value;
          if (!first_.vmask.defined()) {
            first_.vmask = cur_pend_;
          }
        }
      } else if (GetIntrinPipe(call->name) == PIPE_V && !excluded_vec_.count(call->name)) {
        if (cur_pend_.defined()) {
          cur_vmask_ = cur_pend_;
          cur_pend_ = Expr();
          if (first_.type == FT_NONE && first_.vmask.defined()) {
            first_.type = FT_MASKED;
          }
        }
        if (first_.type == FT_DUP_PEND) {
          first_.type = FT_DUP;
        }
        if (first_.type == FT_NONE) {
          first_.is_entry = false;
        }
      }
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const IfThenElse *op) final {
    FirstMask f1, f2;
    Expr init_pend = cur_pend_;
    Expr init_vmask = cur_vmask_;
    std::swap(first_, f1);
    Visit(op->then_case);
    if (cur_pend_.get() != init_pend.get()) {
      TrimPending();
    }
    std::swap(first_, f1);
    if (f1.type == FT_DUP_PEND) {
      f1.type = FT_NONE;
    }
    if (op->else_case.defined()) {
      cur_vmask_ = init_vmask;
      std::swap(first_, f2);
      Visit(op->else_case);
      if (cur_pend_.get() != init_pend.get()) {
        TrimPending();
      }
      std::swap(first_, f2);
      if (f2.type == FT_DUP_PEND) {
        f2.type = FT_NONE;
      }
    }
    if (!f1.is_entry || !f2.is_entry) {
      if (first_.type == FT_NONE) {
        if (first_.vmask.defined()) {
          first_.type = FT_MASKED;
        } else {
          first_.is_entry = false;
        }
      } else if (first_.type == FT_DUP_PEND) {
        first_.type = FT_DUP;
      }
    } else if (f1.type != FT_NONE || f2.type != FT_NONE) {
      if (first_.type == FT_DUP_PEND) {
        first_.type = FT_DUP;
      } else if (first_.type == FT_NONE) {
        first_.vmask = f1.type != FT_NONE ? f1.vmask : f2.vmask;
        first_.type = FT_CHILD;
        if (!f1.is_entry || !f2.is_entry) {
          first_.is_entry = false;
        }
        // for robust, recover first dup vmask
        if (f1.type == FT_DUP) rm_vmask_.erase(f1.vmask.as<Call>());
        if (f2.type == FT_DUP) rm_vmask_.erase(f2.vmask.as<Call>());
      }
    }
  }

  void Visit_(const For *op) final {
    Expr init_pend = cur_pend_;
    Expr init_vmask = cur_vmask_;
    FirstMask f;
    std::swap(first_, f);
    IRVisitor::Visit_(op);
    std::swap(first_, f);
    if (f.type == FT_NONE && f.is_entry) {
      return;
    }
    if (f.type == FT_DUP_PEND) {
      f.type = FT_NONE;
    }
    Expr entry;
    if (f.is_entry && (f.type == FT_MASKED || f.type == FT_CHILD)) {
      entry = f.vmask;
    } else if (init_pend.defined() && !rm_vmask_.count(init_pend.as<Call>())) {
      entry = init_pend;
    } else {
      entry = init_vmask;
    }
    bool coherent = f.is_entry && f.type != FT_DUP && entry.defined();
    if (!coherent && !cur_pend_.defined() && IsEqual(cur_vmask_, entry)) {
      coherent = true;
    }
    if (IsEqual(cur_pend_, entry)) {
      cur_vmask_ = cur_pend_;
      cur_pend_ = Expr();
      coherent = true;
    }
    if (cur_pend_.defined()) {
      const Call *vmask = cur_pend_.as<Call>();
      auto hoist = post_hoist_.find(vmask);
      if (hoist == post_hoist_.end()) {
        rm_vmask_.insert(vmask);
      } else if (IsEqual(cur_vmask_, entry)) {
        ins_after_.erase(hoist->second);
      }
      ins_after_[op] = cur_pend_;
      post_hoist_[vmask] = op;
      if (!coherent && IsEqual(cur_vmask_, entry)) {
        coherent = true;
      }
    }
    if (!coherent && f.is_entry && f.type == FT_DUP) {
      const Call *vmask = f.vmask.as<Call>();
      rm_vmask_.erase(vmask);
      coherent = true;
      entry = f.vmask;
      f.type = FT_MASKED;
    }
    if (!coherent) {
      LOG(WARNING) << "conerent is not satisfy, entry=" << entry << ", exit=" << cur_vmask_;
    }
    if (f.type == FT_MASKED && f.is_entry && IsEqual(f.vmask, cur_vmask_)) {
      const Call *vmask = f.vmask.as<Call>();
      auto hoist = prev_hoist_.find(vmask);
      if (hoist == prev_hoist_.end()) {
        rm_vmask_.insert(vmask);
      } else {
        ins_before_.erase(hoist->second);
      }
      ins_before_[op] = f.vmask;
      prev_hoist_[vmask] = op;
      if (first_.type == FT_NONE || first_.type == FT_DUP_PEND) {
        first_.vmask = f.vmask;
        first_.type = FT_MASKED;
      }
    }
    if (!f.is_entry) {
      if (first_.type == FT_NONE) {
        if (first_.vmask.defined()) {
          first_.type = FT_MASKED;
        } else {
          first_.is_entry = false;
        }
      } else if (first_.type == FT_DUP_PEND) {
        first_.type = FT_DUP;
      }
    } else if (f.type != FT_NONE) {
      if (first_.type == FT_DUP_PEND && f.type == FT_DUP) {
        first_.type = FT_DUP;
      } else if (first_.type == FT_DUP_PEND || first_.type == FT_NONE) {
        first_.type = f.type == FT_DUP ? FT_DUP : FT_CHILD;
        first_.vmask = f.vmask;
        if (!f.is_entry) first_.is_entry = false;
      }
    }
  }

  std::unordered_map<const For *, Expr> ins_before_;
  std::unordered_map<const For *, Expr> ins_after_;
  std::unordered_set<const Call *> rm_vmask_;

 private:
  void TrimPending() {
    if (!cur_pend_.defined()) return;
    const Call *vm = cur_pend_.as<Call>();
    rm_vmask_.insert(vm);
    auto itr = post_hoist_.find(vm);
    if (itr != post_hoist_.end()) {
      ins_after_.erase(itr->second);
    }
    cur_pend_ = Expr();
  }

  bool IsEqual(const Expr ea, const Expr eb) {
    const Call *ca = ea.as<Call>();
    const Call *cb = eb.as<Call>();
    if (ca == nullptr || cb == nullptr) {
      return ca == cb;
    }
    const Array<Expr> &a = ca->args;
    const Array<Expr> &b = cb->args;
    if (a.size() != b.size()) {
      return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
      if (!Equal(a[i], b[i])) {
        return false;
      }
    }
    return true;
  }

  enum FirstType {
    FT_NONE,
    FT_MASKED,
    FT_CHILD,
    FT_DUP,
    FT_DUP_PEND,
  };

  struct FirstMask {
    Expr vmask;
    FirstType type{FT_NONE};
    bool is_entry{true};
  };

  Expr cur_vmask_;
  Expr cur_pend_;
  FirstMask first_;
  std::unordered_map<const Call *, const For *> prev_hoist_;
  std::unordered_map<const Call *, const For *> post_hoist_;

  std::unordered_set<std::string> excluded_vec_{"copy_ubuf_to_ubuf", "copy_matrix_cc_to_ubuf"};
};

class VecMaskElim : public IRMutator {
 public:
  VecMaskElim() {}
  ~VecMaskElim() override = default;

  Stmt Elim(Stmt stmt) {
    plan_.Plan(stmt);
    if (plan_.rm_vmask_.empty()) {
      return stmt;
    }
    return Mutate(stmt);
  }

  Stmt Mutate_(const Evaluate *op, const Stmt &s) override {
    const Call *call = op->value.as<Call>();
    if (call != nullptr && plan_.rm_vmask_.find(call) != plan_.rm_vmask_.end()) {
      return Evaluate::make(0);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) override {
    Stmt stmt = IRMutator::Mutate_(op, s);
    auto it = plan_.ins_before_.find(op);
    if (it != plan_.ins_before_.end()) {
      Stmt vmask = Evaluate::make(it->second);
      stmt = Block::make(vmask, stmt);
    }
    it = plan_.ins_after_.find(op);
    if (it != plan_.ins_after_.end()) {
      Stmt vmask = Evaluate::make(it->second);
      stmt = Block::make(stmt, vmask);
    }
    return stmt;
  }

 private:
  VecMaskElimPlan plan_;
};

// we use set_vector_mask call pointer to index elim position, so
// it should be unique.
class UniqueVecMask : public IRMutator {
 public:
  Expr Mutate_(const Call *op, const Expr &e) override {
    if (op->name == "set_vector_mask") {
      if (vmask_.count(op)) {
        return Call::make(op->type, op->name, op->args, op->call_type);
      }
      vmask_.insert(op);
    }
    return IRMutator::Mutate_(op, e);
  }

 private:
  std::unordered_set<const Call *> vmask_;
};

Stmt ElimVectorMask(Stmt stmt) {
  stmt = UniqueVecMask().Mutate(stmt);
  stmt = VecMaskElim().Elim(stmt);
  return RemoveNoOp(stmt);
}
}  // namespace ir
}  // namespace akg
