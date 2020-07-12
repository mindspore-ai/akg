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
#include <tvm/operation.h>
#include <ir_pass.h>
#include <pass/ir_util.h>
#include <pass/storage_access.h>
#include "pass/utils.h"

namespace akg {
namespace ir {
class LoopNestOrdering : public IRVisitor {
 public:
  LoopNestOrdering(const std::map<const Variable *, int> &idmap, std::vector<int> &ordering)
      : idmap_(idmap), ordering_(ordering) {}
  ~LoopNestOrdering() override = default;

  void Visit_(const Variable *op) final {
    auto it = idmap_.find(op);
    if (it != idmap_.end()) {
      ordering_.push_back(it->second);
    } else {
      ordering_.push_back(-1);
    }
    IRVisitor::Visit_(op);
  }

 private:
  const std::map<const Variable *, int> &idmap_;
  std::vector<int> &ordering_;
};

class Replace : public IRMutator {
 public:
  Replace(const Node *from, const Operation &to) : from_(from), to_(to) {}
  ~Replace() override = default;

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    const auto n = stmt.as<Realize>();
    CHECK(n);
    if (op->func.get() == from_) {
      stmt = n->body;
    }
    return stmt;
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    const auto n = stmt.as<Provide>();
    CHECK(n);
    if (op->func.get() == from_) {
      stmt = Provide::make(to_, n->value_index, n->value, n->args);
    }
    return stmt;
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    Expr expr = IRMutator::Mutate_(op, e);
    const auto n = expr.as<Call>();
    CHECK(n);
    if (n->func.get() == from_) {
      expr = Call::make(n->type, to_->name, n->args, n->call_type, to_, n->value_index);
    }
    return expr;
  }

  Stmt Mutate_(const ProducerConsumer *op, const Stmt &s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    const auto n = stmt.as<ProducerConsumer>();
    CHECK(n);
    if (op->func.get() == from_) {
      stmt = n->body;
    }
    return stmt;
  }

 private:
  const Node *from_;
  const Operation &to_;
};

class MultiStageCSE : public IRMutator {
 public:
  explicit MultiStageCSE(const Map<Tensor, Buffer> &extern_buffer) : extern_buffer_(extern_buffer) {}
  ~MultiStageCSE() override = default;

  Stmt Mutate_(const For *op, const Stmt &s) final {
    loop_vars_[op->loop_var.get()] = op->min + op->extent;
    curr_for_id++;
    loop_vars_idmap[curr_for_id] = op->loop_var.get();
    loop_vars_idmap_reverse[op->loop_var.get()] = curr_for_id;
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    if (op->attr_key == air::ir::attr::realize_scope) {
      if (replace_.count(op->node.get())) {
        const auto n = stmt.as<AttrStmt>();
        CHECK(n);
        stmt = n->body;
      }
    }
    return stmt;
  }

  // we don't touch provide in if
  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final { return s; }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    shape_[op->func.get()] = op->bounds;
    Stmt stmt = IRMutator::Mutate_(op, s);
    if (replace_.count(op->func.get())) {
      stmt = Replace(op->func.get(), replace_[op->func.get()]).Mutate(stmt);
    }
    return stmt;
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Provide>();
    CHECK(op);
    bool found{false};
    for (auto i : defs_) {
      // make sure use has the same number of var
      if (CountVars(i.second.first->value) != CountVars(op->value)) {
        continue;
      }
      Expr a = Substitute(i.second.first->value, loop_vars_);
      Expr b = Substitute(op->value, loop_vars_);
      if (!Equal(a, b)) continue;

      Expr adef = Call::make(op->value.type(), "T", i.second.first->args, Call::Halide, op->func, op->value_index);
      Expr bdef = Call::make(op->value.type(), "T", op->args, Call::Halide, op->func, op->value_index);
      if (CountVars(adef) != CountVars(bdef)) {
        continue;
      }
      if (i.second.second == curr_for_id) {
        // if within the same "For" block then we compare directly the formulas
        if (!Equal(i.second.first->value, op->value)) {
          continue;
        }
      }
      bool fusable_ub_condition = CheckFusableCondition_(adef, bdef, shape_[op->func.get()]);
      adef = Substitute(adef, loop_vars_);
      bdef = Substitute(bdef, loop_vars_);
      // for adef = ause; bdef = buse, make sure ause == buse; and adef == bdef
      if (fusable_ub_condition && Equal(adef, bdef) && i.first != op->func.get() &&
          CheckUpdateRanges_(shape_[i.first], shape_[op->func.get()]) && i.second.second <= curr_for_id) {
        // do not eliminate the output statement.
        if (std::any_of(extern_buffer_.begin(), extern_buffer_.end(),
                        [op](const std::pair<Tensor, Buffer> &it) { return op->func.get() == it.first->op.get(); })) {
          auto old_op = i.second.first;
          auto new_call = Call::make(op->value.type(), old_op->func->func_name(), op->args, Call::CallType::Halide,
                                     old_op->func, old_op->value_index);
          stmt = Provide::make(op->func, op->value_index, new_call, op->args);
        } else {
          replace_[op->func.get()] = Operation(GetObjPtr(i.first));
          stmt = Evaluate::make(0);
          found = true;
        }
        break;
      }
    }

    CleanDefs(op);

    if (!found) {
      if (replace_.count(op->func.get())) {
        auto rep_op = replace_[op->func.get()].get();
        // The op is replaced by rep_op, so rep_op is rewritten in this Provide stmt,
        // therefore, we should not use the old rep_op to replace other statements again.
        static_cast<void>(defs_.erase(rep_op));
      }
      defs_[op->func.get()] = std::make_pair(op, curr_for_id);
    }
    return stmt;
  }

 private:
  bool CheckFusableCondition_(Expr a, Expr b, const Region rb) {
    std::vector<int> a_order, b_order;
    LoopNestOrdering(loop_vars_idmap_reverse, a_order).Visit(a);
    LoopNestOrdering(loop_vars_idmap_reverse, b_order).Visit(b);
    /* we want as much as CSE as possible so we will assume
       by default, loops are fusable, and only in condition below,
       it is non-fusable */
    if (a_order.size() == b_order.size() && a_order.size() > 0) {
      int diff = b_order[0] - a_order[0];
      for (unsigned int i = 0; i < a_order.size(); i++)
        if ((b_order[i] - a_order[i]) != diff) {
/**CSE across loop nests need to be done carefully.**
   Since CSE has the side-effect of making the live-range of a variable live
   across loop nests, so for non-fusable loop nests, we want to ensure the
   cse_var can be contained in UB.

loop-nest-a: i(0,64), j(0,32), k(0,16)
    cse_var[i][j][k] =

... cse_var is live ...

loop-nest-b: j(0,32), i(0,64), m(0,16)
                    = cse_var[i][j][m]

... the two loop nests are currently not fusable ...
*/
#define UB_SIZE 256000
          int tensor_memory_size = 1;
          for (unsigned int j = 0; j < rb.size(); j++) {
            const auto _ibx = (rb[j]->extent).as<IntImm>();
            if (_ibx && _ibx->value > 0) tensor_memory_size *= _ibx->value;
          }
          if (tensor_memory_size > UB_SIZE) {
            return false;
          }
        }
    }
    return true;
  }

  bool CheckUpdateRanges_(const Region ra, const Region rb) {
    if (ra.size() != rb.size())  // check shapes to be the same
      return false;
    if (ra.size() != 1) {
      // This means multi-dimension tensor, e.g. realize T1([0,32][0,64]),
      // then there are 2 dimensions to check for.  But since we only cache
      // the immediate enclosing for loop's extent, there can still be bugs.
      LOG(WARNING) << "StmtCSE: multiple dimensions but only comparing innermost for loop";
    } else {
      const auto _ibx = (rb[0]->extent).as<IntImm>();
      const auto _iby = loop_vars_[loop_vars_idmap[curr_for_id]].as<IntImm>();
      if (_ibx && _iby && _ibx->value != _iby->value) {
        return false;
      }
    }
    return true;
  }

  void CleanDefs(const Provide *op) {
    for (auto it = defs_.begin(); it != defs_.end();) {
      class Recorder : public IRVisitor {
       public:
        explicit Recorder(const Node *n) : calls(), node_(n) {}
        ~Recorder() override = default;

        void Visit_(const Call *op) override {
          if (op->func.get() == node_) {
            calls.insert(op);
          } else {
            IRVisitor::Visit_(op);
          }
        }
        std::set<const Call *> calls;

       private:
        const Node *node_{nullptr};
      };
      Recorder r(op->func.get());
      r.Visit(it->second.first->value);

      if (!r.calls.empty()) {
        it = defs_.erase(it);
      } else {
        ++it;
      }
    }
  }

  std::unordered_map<const Node *, std::pair<const Provide *, int>> defs_;
  // immediate enclosing for loop's id
  std::unordered_map<const Node *, Operation> replace_;
  std::unordered_map<const Variable *, Expr> loop_vars_;
  std::unordered_map<const Node *, Region> shape_;
  int curr_for_id{0};
  std::map<int, const Variable *> loop_vars_idmap;          // loop_id, for-loop-var
  std::map<const Variable *, int> loop_vars_idmap_reverse;  // for-loop-var, loop_id
  const Map<Tensor, Buffer> &extern_buffer_;
};

class Compact : public IRMutator {
 public:
  Compact() {}
  ~Compact() override = default;

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    Array<Expr> new_args = compact(op->args);
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Provide>();
    CHECK(op);
    return Provide::make(op->func, op->value_index, op->value, new_args);
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    Array<Expr> new_args = compact(op->args);
    Expr expr = IRMutator::Mutate_(op, e);
    op = expr.as<Call>();
    CHECK(op);
    if (op->call_type == Call::Halide) {
      expr = Call::make(op->type, op->name, new_args, op->call_type, op->func, op->value_index);
    }
    return expr;
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    Region new_bounds;
    Expr cone = make_const(Int(32), 1);
    air::arith::Analyzer analyzer_;
    for (size_t i = 0; i < op->bounds.size(); i++) {
      if (analyzer_.CanProve(op->bounds[i]->extent > cone) || i == 0 || i == op->bounds.size() - 1) {
        new_bounds.push_back(op->bounds[i]);
      }
    }
    Stmt stmt = IRMutator::Mutate_(op, s);
    if (op->bounds.size() == 1 || new_bounds.size() == 0) return stmt;
    op = stmt.as<Realize>();
    CHECK(op);
    return Realize::make(op->func, op->value_index, op->type, new_bounds, op->condition, op->body);
  }

 private:
  Array<Expr> compact(const Array<Expr> &in) {
    Array<Expr> out;
    if (in.size() == 1) {
      out = in;
      return out;
    }
    for (size_t i = 0; i < in.size(); i++) {
      if (!Equal(in[i], Expr(0)) || i == 0 || i == in.size() - 1) {
        out.push_back(in[i]);
      }
    }

    // for rank0 tensor
    if (out.size() == 0) {
      out = in;
    }

    return out;
  }
};

Stmt StmtCSE(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer) {
  Stmt prev = stmt;
  do {
    prev = stmt;
    stmt = MultiStageCSE(extern_buffer).Mutate(prev);
    stmt = RemoveNoOp(stmt);
  } while (!stmt.same_as(prev));
  return stmt;
}
}  // namespace ir
}  // namespace akg
