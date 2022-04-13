/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include <climits>
#include <unordered_map>
#include <tvm/arithmetic.h>
#include <tvm/ir_pass.h>
#include "composite/lower_tree/promote_datatype.h"
#include "composite/utils/util.h"
#include "poly/isl_emitter.h"
namespace akg {
namespace ir {
/*
Promote blockIdx.x from Int32 to Int64 if overflow is detected.

before:

if (blockIdx.x < 19836) {
  input[(blockIdx.x - 19265) * 144800] = ...
}

This statement will be further optimized in TVM arithmetic simplify pass like:
if (blockIdx.x < 19836) {
  input[blockIdx.x * 144800 - 19265 * 144800] = ...
}
both blockIdx.x * 144800 and 19265 * 144800 can exceed Int32, and cause overflow.

Therefore, this pass detects this pattern beforehand, and prevent index overflow:

after:

if (Int64(blockIdx.x) < 19836) {
  input[(Int64(blockIdx.x) - Int64(19265)) * Int64(144800)] = ...
}
*/
bool IsBlockAttr(const AttrStmt *op) {
  if (op->attr_key == air::ir::attr::thread_extent) {
    const IterVarNode *iv = op->node.as<IterVarNode>();
    CHECK(iv);
    std::string name = iv->var->name_hint;
    return name.compare(0, BLOCKIDX_LEN, BLOCKIDX) == 0;
  }
  return false;
}

void OverflowChecker::Visit_(const Store *op) {
  Expr after_simplify = CanonicalSimplify(op->index);
  this->Visit(after_simplify);
  IRVisitor::Visit_(op);
}

void OverflowChecker::Visit_(const Load *op) {
  Expr after_simplify = CanonicalSimplify(op->index);
  this->Visit(after_simplify);
  IRVisitor::Visit_(op);
}

void OverflowChecker::Visit_(const Add *op) {
  if (op->a->IsInstance<Variable>() && op->b->IsInstance<IntImm>()) {
    auto left_var = op->a.as<Variable>();
    auto right_int = op->b.as<IntImm>();
    if (left_var->name_hint == "blockIdx.x" && right_int->value + block_extent_ > INT_MAX) {
      need_promote_int64 = true;
      var_to_replace = left_var;
    }
  }
  IRVisitor::Visit_(op);
}

void OverflowChecker::Visit_(const Mul *op) {
  if (op->a->IsInstance<Variable>() && op->b->IsInstance<IntImm>()) {
    auto left_var = op->a.as<Variable>();
    auto right_int = op->b.as<IntImm>();
    if (left_var->name_hint == "blockIdx.x" && right_int->value * block_extent_ > INT_MAX) {
      need_promote_int64 = true;
      var_to_replace = left_var;
    }
  }
  IRVisitor::Visit_(op);
}

void OverflowChecker::Visit_(const AttrStmt *op) {
  if (IsBlockAttr(op)) {
    auto value_int_imm = op->value.as<IntImm>();
    CHECK(value_int_imm);
    block_extent_ = value_int_imm->value;
    IRVisitor::Visit_(op);
  }
  IRVisitor::Visit_(op);
}

void OverflowChecker::Visit(const NodeRef &node) {
  if (need_promote_int64) return;
  IRVisitor::Visit(node);
}

class MinMaxDtypeTrans : public IRMutator {
 public:
  Expr Mutate_(const Min *op, const Expr &s) {
    Expr a = CanonicalSimplify(op->a);
    Expr b = CanonicalSimplify(op->b);
    auto dtype_a = a.type();
    auto dtype_b = b.type();
    if (dtype_a.lanes() == dtype_b.lanes()) {
      return Min::make(IRMutator::Mutate(a), IRMutator::Mutate(b));
    }
    if (dtype_a.lanes() < dtype_b.lanes()) {
      Expr new_a = Cast::make(dtype_b, a);
      return Min::make(IRMutator::Mutate(new_a), IRMutator::Mutate(b));
    } else {
      Expr new_b = Cast::make(dtype_a, b);
      return Min::make(IRMutator::Mutate(a), IRMutator::Mutate(new_b));
    }
  }

  Expr Mutate_(const Max *op, const Expr &s) {
    Expr a = CanonicalSimplify(op->a);
    Expr b = CanonicalSimplify(op->b);
    auto dtype_a = a.type();
    auto dtype_b = b.type();
    if (dtype_a.lanes() == dtype_b.lanes()) {
      return Max::make(IRMutator::Mutate(a), IRMutator::Mutate(b));
    }
    if (dtype_a.lanes() < dtype_b.lanes()) {
      Expr new_a = Cast::make(dtype_b, a);
      return Max::make(IRMutator::Mutate(new_a), IRMutator::Mutate(b));
    } else {
      Expr new_b = Cast::make(dtype_a, b);
      return Max::make(IRMutator::Mutate(a), IRMutator::Mutate(new_b));
    }
  }
};

Stmt PromoteIndexDataType(Stmt stmt) {
  auto overflow_checker = OverflowChecker();
  overflow_checker.Visit(stmt);
  // Only promote blockIdx.x to Int64 when needed, to prevent performance loss
  // introduced by Int64 calculations.
  if (overflow_checker.need_promote_int64) {
    std::unordered_map<const Variable *, Expr> var_map{{}};
    var_map[overflow_checker.var_to_replace] =
      Cast::make(Int(poly::INT_64), GetRef<Expr>(overflow_checker.var_to_replace));
    stmt = Substitute(stmt, var_map);
    stmt = MinMaxDtypeTrans().Mutate(stmt);
  }
  return stmt;
}
}  // namespace ir
}  // namespace akg
