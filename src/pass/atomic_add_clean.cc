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
#include <pass/ir_util.h>
#include <pass/utils.h>
#include <poly/poly_util.h>

namespace akg {
namespace ir {

class AtomicAddCleanMutate : public IRMutator {
 public:
  explicit AtomicAddCleanMutate(const std::unordered_set<FunctionRef, ExprHash, ExprEqual> &atomic_add_buffers) {
    atomic_add_buffers_ = atomic_add_buffers;
  }

 private:
  static bool HasSameBuffer(const Stmt &stmt1, const Stmt &stmt2) {
    std::unordered_set<FunctionRef, ExprHash, ExprEqual> buffers;
    PostOrderVisit(stmt1, [&buffers](const NodeRef &node) {
      if (auto call = node.as<Call>()) {
        buffers.insert(call->func);
      } else if (auto provide = node.as<Provide>()) {
        buffers.insert(provide->func);
      }
    });
    bool has_same = false;
    PostOrderVisit(stmt2, [&buffers, &has_same](const NodeRef &node) {
      if (auto call = node.as<Call>()) {
        if (buffers.count(call->func)) {
          has_same = true;
        }
      } else if (auto provide = node.as<Provide>()) {
        if (buffers.count(provide->func)) {
          has_same = true;
        }
      }
    });
    return has_same;
  }

  Stmt Mutate_(const Block *op, const Stmt &s) override {
    auto first = this->Mutate(op->first);
    if (sum_blocks_.count(first)) {
      Stmt clean_zero = GetCleanZeroStmt(first);
      // The ATTR_ATOMIC_CLEAN_ZERO and the ATTR_ATOMIC_ADD should appear in pairs in the same Block:
      bool has_atomic_write = false;
      PostOrderVisit(op->rest, [clean_zero, &has_atomic_write](const NodeRef &node) {
        if (auto attr_op = node.as<AttrStmt>()) {
          if (!has_atomic_write && attr_op->attr_key == ATTR_ATOMIC_ADD && HasSameBuffer(clean_zero, attr_op->body)) {
            has_atomic_write = true;
          }
        }
      });
      CHECK(has_atomic_write);

      first = Block::make(clean_zero, first);
    }
    auto rest = this->Mutate(op->rest);
    return Block::make(first, rest);
  }

  Stmt Mutate_(const For *op, const Stmt &s) override {
    loop_stack_.push_back(op);
    auto stmt = IRMutator::Mutate_(op, s);
    loop_stack_.pop_back();
    return stmt;
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) override {
    // remove reduce init
    if (atomic_add_buffers_.count(op->func) && isZero(op->value)) {
      return Evaluate::make(0);
    }

    // sum: a = a + b
    if (atomic_add_buffers_.count(op->func) && op->value.as<air::ir::Add>()) {
      // check that op is a ReduceSum
      bool is_sum = false;
      PostOrderVisit(op->value, [op, &is_sum](const NodeRef &node) {
        if (auto call = node.as<Call>()) {
          if (call->func.same_as(op->func) && call->value_index == op->value_index) {
            is_sum = true;
          }
        }
      });
      CHECK(is_sum);

      // get stmt block of the ReduceSum
      std::unordered_set<const Variable *> sum_vars;
      PostOrderVisit(s, [&sum_vars](const NodeRef &node) {
        if (auto var = node.as<Variable>()) {
          sum_vars.insert(var);
        }
      });
      const For *for_op = nullptr;
      for (auto iter = loop_stack_.rbegin(); iter != loop_stack_.rend(); ++iter) {
        if (sum_vars.count((*iter)->loop_var.get())) {
          for_op = *iter;
        }
      }
      if (for_op) {
        sum_blocks_.insert(GetRef<const NodeRef>(for_op));
      } else {
        sum_blocks_.insert(s);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  static Stmt GetCleanZeroStmt(const Stmt &s) {
    const Provide *sum_op = nullptr;
    std::unordered_map<const Variable *, const For *> var_map_for;
    PostOrderVisit(s, [&sum_op, &var_map_for](const NodeRef &node) {
      if (auto for_op = node.as<For>()) {
        var_map_for[for_op->loop_var.get()] = for_op;
      } else if (auto provide_op = node.as<Provide>()) {
        CHECK(sum_op == nullptr);
        sum_op = provide_op;
      }
    });
    CHECK(sum_op);
    auto args = sum_op->args;
    std::vector<const Variable *> vars;
    for (auto arg : args) {
      PostOrderVisit(arg, [&vars](const NodeRef &node) {
        if (auto var = node.as<Variable>()) {
          vars.push_back(var);
        }
      });
    }
    std::vector<const For *> for_ops;
    for (auto var : vars) {
      CHECK(var_map_for.count(var));
      for_ops.push_back(var_map_for.at(var));
    }
    auto clean_zero_stmt =
      Provide::make(sum_op->func, sum_op->value_index, make_zero(sum_op->value.type()), args);
    for (auto iter = for_ops.rbegin(); iter != for_ops.rend(); ++iter) {
      auto for_op = *iter;
      clean_zero_stmt =
        For::make(for_op->loop_var, for_op->min, for_op->extent, for_op->for_type, for_op->device_api, clean_zero_stmt);
    }
    clean_zero_stmt = AttrStmt::make(make_zero(Int(32)), ATTR_ATOMIC_CLEAN_ZERO, Expr(1), clean_zero_stmt);
    return clean_zero_stmt;
  }

  std::unordered_set<NodeRef, ExprHash, ExprEqual> sum_blocks_;
  std::vector<const For *> loop_stack_;
  std::unordered_set<FunctionRef, ExprHash, ExprEqual> atomic_add_buffers_;
};

class AtomicAddGetter : public IRVisitor {
 public:
  AtomicAddGetter() = default;

  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == ATTR_ATOMIC_ADD) {
      in_atomic_add_ = true;
      IRVisitor::Visit_(op);
      in_atomic_add_ = false;
    } else {
      return IRVisitor::Visit_(op);
    }
  }

  void Visit_(const Call *op) final {
    if (in_atomic_add_) {
      if (op->func->func_name().find("local_UB") != std::string::npos) {
        atomic_add_buffers.insert(op->func);
      }
    }
    return IRVisitor::Visit_(op);
  }

  std::unordered_set<FunctionRef, ExprHash, ExprEqual> atomic_add_buffers;

 private:
  bool in_atomic_add_{false};
};

/*
  For the atomic add, every time the UB is copied to GM,
  the UB needs to be cleared to zero before the next sum.
  For example:

  realize input_1_red_local_UB<float32>([0, 1]) {
    for (cc0, 0, 64) {
      if ((cc0 == 0)) {
        input_1_red_local_UB(0) = 0f
      }
      // attr [placeholder(input_1_local_UB, 0x563ff88a63d0)] realize_scope = "local.UB"
      realize input_1_local_UB<float32>([0, 64]) {
        for (cc1, 0, 64) {
          input_1_local_UB(cc1) = input_1(((64*cc0) + cc1))
        }
        for (cc1, 0, 64) {
          input_1_red_local_UB(0) = (input_1_red_local_UB(0) + input_1_local_UB(cc1))
        }
        // attr [0] atomic_add = 1
        input_1_red(0) = (input_1_red(0) + input_1_red_local_UB(0))
      }
    }
  }

  ===>

  realize input_1_red_local_UB<float32>([0, 1]) {
    for (cc0, 0, 64) {
      // attr [placeholder(input_1_local_UB, 0x563ff88a63d0)] realize_scope = "local.UB"
      realize input_1_local_UB<float32>([0, 64]) {
        for (cc1, 0, 64) {
          input_1_local_UB(cc1) = input_1(((64*cc0) + cc1))
        }
        // attr [0] atomic_clean_zero = 1
        input_1_red_local_UB(0) = 0f
        for (cc1, 0, 64) {
          input_1_red_local_UB(0) = (input_1_red_local_UB(0) + input_1_local_UB(cc1))
        }
        // attr [0] atomic_add = 1
        input_1_red(0) = (input_1_red(0) + input_1_red_local_UB(0))
      }
    }
  }
*/
Stmt AtomicAddClean(Stmt stmt) {
  AtomicAddGetter atomic_add_getter;
  atomic_add_getter.Visit(stmt);
  stmt = AtomicAddCleanMutate(atomic_add_getter.atomic_add_buffers).Mutate(stmt);
  stmt = RemoveNoOp(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
