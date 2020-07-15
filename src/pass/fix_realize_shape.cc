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
#include <ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/arithmetic.h>
#include <pass/utils.h>

namespace akg {
namespace ir {
class RealizeShapeFixer : public IRMutator {
 private:
  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    // save realize shape in outer scope
    bool outer_realize_exist = false;
    Array<Range> outer_realize_shape;
    if (realize_shape.count(op->func) > 0) {
      outer_realize_exist = true;
      outer_realize_shape = realize_shape[op->func];
    }

    realize_shape[op->func] = op->bounds;

    Stmt body = Mutate(op->body);
    Stmt realize_stmt =
      Realize::make(op->func, op->value_index, op->type, realize_shape[op->func], op->condition, body);

    // restore realize shape in outer scope
    realize_shape.erase(op->func);
    if (outer_realize_exist) {
      realize_shape[op->func] = outer_realize_shape;
    }

    return realize_stmt;
  }

  void RemoveVarsInCond(const Expr &cond) {
    std::unordered_set<Var, air::NodeHash, air::NodeEqual> vars_in_cond;
    GatherVars(cond, &vars_in_cond);
    for (auto var_in_cond : vars_in_cond) {
      loop_var_bounds.erase(var_in_cond.get());
    }
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    auto backup_loop_var_bounds = loop_var_bounds;
    RemoveVarsInCond(op->condition);
    Stmt stmt = IRMutator::Mutate_(op, s);
    loop_var_bounds = backup_loop_var_bounds;
    return stmt;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    const Variable *loop_var = op->loop_var.get();
    CHECK_EQ(loop_var_bounds.count(loop_var), 0);
    Bound min_bound = InferBoundOfExpr(op->min, loop_var_bounds);
    Bound extent_bound = InferBoundOfExpr(op->extent, loop_var_bounds);
    loop_var_bounds[loop_var] = Range::make_by_min_extent(min_bound.min, extent_bound.max);

    Stmt stmt = IRMutator::Mutate_(op, s);

    loop_var_bounds.erase(loop_var);
    return stmt;
  }

  template <class T>
  void FixRealizeShapeFromCall(const T *op) {
    if (realize_shape.count(op->func) == 0) {
      return;
    }
    auto shape = realize_shape[op->func];
    size_t num_args = std::min(shape.size(), op->args.size());
    for (size_t i = 0; i < num_args; i++) {
      Bound shape_bound = Bound::make(shape[i]);
      Bound expr_bound = InferBoundOfExpr(op->args[i], loop_var_bounds);
      bool need_update_realize = false;
      if (is_positive_const(Simplify(expr_bound.max - shape_bound.max))) {
        shape_bound.max = expr_bound.max;
        need_update_realize = true;
      }
      if (is_positive_const(Simplify(shape_bound.min - expr_bound.min))) {
        shape_bound.min = expr_bound.min;
        need_update_realize = true;
      }

      if (need_update_realize) {
        auto extent = Simplify(shape_bound.max - shape_bound.min + 1);
        auto new_range = Range::make_by_min_extent(shape_bound.min, extent);
        shape.Set(i, new_range);
        realize_shape[op->func] = shape;
      }
    }
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    FixRealizeShapeFromCall(op);
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->call_type == Call::CallType::Halide) {
      FixRealizeShapeFromCall(op);
    }
    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const Select *op, const Expr &e) final {
    auto backup_loop_var_bounds = loop_var_bounds;
    RemoveVarsInCond(op->condition);
    Expr expr = IRMutator::Mutate_(op, e);
    loop_var_bounds = backup_loop_var_bounds;
    return expr;
  }

  std::unordered_map<FunctionRef, Array<Range>, NodeHash, NodeEqual> realize_shape;
  std::unordered_map<const Variable *, Range> loop_var_bounds;
};

Stmt FixRealizeShape(Stmt stmt) {
  stmt = RealizeShapeFixer().Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
