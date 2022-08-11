/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

/*
 * 2022.8.11
 *   Rescale realize_shape_ according to the args of the provide node.
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/arithmetic.h>
#include <pass/utils.h>

namespace akg {
namespace ir {
class RealizeShapeFixer : public IRMutator {
 private:
  class IsContainMod : public IRVisitor {
   public:
    explicit IsContainMod(const NodeRef &node) : node_(node) {}
    ~IsContainMod() override = default;

    void Visit_(const FloorMod *op) override { is_contain_mod_ = true; }

    bool IsTrue() {
      this->Visit(node_);
      return is_contain_mod_;
    }

   private:
    NodeRef node_;
    bool is_contain_mod_{false};
  };

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    // save realize shape in outer scope
    bool outer_realize_exist = false;
    Array<Range> outer_realize_shape;
    if (realize_shape_.count(op->func) > 0) {
      outer_realize_exist = true;
      outer_realize_shape = realize_shape_[op->func];
    }

    realize_shape_[op->func] = op->bounds;

    Stmt body = Mutate(op->body);
    Stmt realize_stmt =
      Realize::make(op->func, op->value_index, op->type, realize_shape_[op->func], op->condition, body);

    // restore realize shape in outer scope
    realize_shape_.erase(op->func);
    if (outer_realize_exist) {
      realize_shape_[op->func] = outer_realize_shape;
    }
    cur_realize_shape_.clear();

    return realize_stmt;
  }

  void RemoveVarsInCond(const Expr &cond) {
    std::unordered_set<Var, air::NodeHash, air::NodeEqual> vars_in_cond;
    GatherVars(cond, &vars_in_cond);
    for (auto var_in_cond : vars_in_cond) {
      loop_var_bounds_.erase(var_in_cond.get());
    }
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    auto backup_loop_var_bounds = loop_var_bounds_;
    RemoveVarsInCond(op->condition);
    Stmt stmt = IRMutator::Mutate_(op, s);
    loop_var_bounds_ = backup_loop_var_bounds;
    return stmt;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    const Variable *loop_var = op->loop_var.get();
    CHECK_EQ(loop_var_bounds_.count(loop_var), 0);
    Bound min_bound = InferBoundOfExpr(op->min, loop_var_bounds_);
    Bound extent_bound = InferBoundOfExpr(op->extent, loop_var_bounds_);
    loop_var_bounds_[loop_var] = Range::make_by_min_extent(min_bound.min, extent_bound.max);

    Stmt stmt = IRMutator::Mutate_(op, s);

    loop_var_bounds_.erase(loop_var);
    return stmt;
  }

  template <class T>
  void FixRealizeShapeFromCall(const T *op) {
    if (realize_shape_.count(op->func) == 0) {
      return;
    }
    auto shape = realize_shape_[op->func];
    size_t num_args = std::min(shape.size(), op->args.size());
    for (size_t i = 0; i < num_args; i++) {
      Bound shape_bound = Bound::make(shape[i]);
      if (IsContainMod(op->args[i]).IsTrue()) {
        continue;
      }
      Bound expr_bound = InferBoundOfExpr(op->args[i], loop_var_bounds_);
      bool need_update_realize = false;
      if (!Equal(shape_bound.max, expr_bound.max)) {
        shape_bound.max = expr_bound.max;
        need_update_realize = true;
      }
      if (!Equal(shape_bound.min, expr_bound.min)) {
        shape_bound.min = expr_bound.min;
        need_update_realize = true;
      }

      if (need_update_realize) {
        Expr new_bound_min = min(shape_bound.max, shape_bound.min);
        Expr new_bound_max = max(shape_bound.max, shape_bound.min);

        auto iter = cur_realize_shape_.find(op->func);
        if (iter != cur_realize_shape_.end()) {
          Bound cur_bound = Bound::make(iter->second[i]);
          new_bound_min = min(cur_bound.min, shape_bound.min);
          new_bound_max = max(cur_bound.max, shape_bound.max);
        }

        auto extent = Simplify(new_bound_max - new_bound_min + 1);
        auto new_range = Range::make_by_min_extent(new_bound_min, extent);
        shape.Set(i, new_range);
        realize_shape_[op->func] = shape;
        cur_realize_shape_[op->func] = shape;
      }
    }
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    FixRealizeShapeFromCall(op);
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Select *op, const Expr &e) final {
    auto backup_loop_var_bounds = loop_var_bounds_;
    RemoveVarsInCond(op->condition);
    Expr expr = IRMutator::Mutate_(op, e);
    loop_var_bounds_ = backup_loop_var_bounds;
    return expr;
  }

  std::unordered_map<FunctionRef, Array<Range>, NodeHash, NodeEqual> realize_shape_;
  std::unordered_map<FunctionRef, Array<Range>, NodeHash, NodeEqual> cur_realize_shape_;
  std::unordered_map<const Variable *, Range> loop_var_bounds_;
};

Stmt FixRealizeShape(Stmt stmt) {
  stmt = RealizeShapeFixer().Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
