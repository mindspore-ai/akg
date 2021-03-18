/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm.h>
#include <unordered_set>

namespace akg {
namespace ir {
class SpecialValueRepPlan : public IRVisitor {
 public:
  void Visit_(const Allocate *op) override {
    if (op->new_expr.defined()) {
      offset_[op->buffer_var.get()] = truncdiv(op->new_expr, op->type.bytes());
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Call *op) final {
    if (op->name == "address_value") {
      CHECK_EQ(op->args.size(), 1);
      const Call *access_op = op->args[0].as<Call>();
      CHECK(access_op != nullptr);
      CHECK(access_op->is_intrinsic(air::ir::intrinsic::tvm_access_ptr));
      auto it = offset_.find(access_op->args[1].as<Variable>());
      CHECK(it != offset_.end());
      replace_map_[op] = cast(op->type, it->second + access_op->args[2]);
    } else {
      IRVisitor::Visit_(op);
    }
  }

  // store all relate vars exist in index
  std::unordered_map<const Node *, Expr> replace_map_;

 private:
  std::unordered_map<const Variable *, Expr> offset_;
};

// substitute relate var to zero
class NodeSubstutite : public IRMutator {
 public:
  explicit NodeSubstutite(std::unordered_map<const Node *, Expr> &replace_map) : replace_map_(replace_map) {}
  ~NodeSubstutite() override = default;

  using IRMutator::Mutate;
  Expr Mutate(Expr e) final {
    if (replace_map_.find(e.get()) != replace_map_.end()) {
      return replace_map_[e.get()];
    } else {
      return IRMutator::Mutate(e);
    }
  }

 private:
  std::unordered_map<const Node *, Expr> &replace_map_;
};

Stmt SpecialValueReplacer(const Stmt stmt) {
  SpecialValueRepPlan visitor;
  visitor.Visit(stmt);
  NodeSubstutite mutater(visitor.replace_map_);
  return mutater.Mutate(stmt);
}
}  // namespace ir
}  // namespace akg
