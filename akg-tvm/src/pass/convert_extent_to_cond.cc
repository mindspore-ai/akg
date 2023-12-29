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

/**
 * Replace variables in loop extent with upper bound
 * and move the variable to if-then-else statement.
 * Make sure loop extent is an affine expr of constants and loop vars.
 */
#include <tvm/expr.h>
#include <tvm/operation.h>
#include <tvm/ir.h>
#include <tvm/tensor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include "pass/utils.h"

/*
 * Example before this pass:

  realize reduce0([0, 100]) {
    realize selected([0, 100], [0, 10]) {
      for (cc0, 0, 100) {
        for (cc1, 0, 10) {
          selected(cc0, cc1) = 0;
        }
      }
      for (cc0, 0, 10) {
        reduce0(cc0) = 0;
      }
      for (cc0, 0, 100) {
        for (cc1, var1(cc0), var2(cc0)) {
          selected(cc0, cc1) = var3(cc0, cc1);
        }
      }
      for (cc0, 0, 100) {
        for (cc1, 0, var1(cc0)) {
          reduce0(cc0) = reduce0(cc0) + var3(cc0, cc1);
        }
      }
    }
  }

  How this pass works:

  First, we need to determine the lower and upper bounds of the loop var
  according to the realize scope of tensor.
  For simplicity, we assume that the loop var is a tensor index
  (i.e. the tensor index is not a complicated expression).

  Second, we need to add an if statement for each provide statement inside the loop.
  The if statement checks whether the loop var is within the original loop range.

  After this pass:

  realize reduce0([0, 100]) {
    realize selected([0, 100], [0, 10]) {
      for (cc0, 0, 100) {
        for (cc1, 0, 10) {
          selected(cc0, cc1) = 0h;
        }
      }
      for (cc0, 0, 10) {
        reduce0(cc0) = 0h;
      }
      for (cc0, 0, 100) {
        for (cc1, 0, 10) {
          if (var1(cc0) <= cc1 && cc1 < var1(cc0) + var2(cc0)) {
            selected(cc0, cc1) = var3(cc0, cc1);
          }
        }
      }
      for (cc0, 0, 100) {
        for (cc1, 0, 10) {
          if (cc1 < var1(cc0)) {
            reduce0(cc0) = reduce0(cc0) + var3(cc0, cc1);
          }
        }
      }
    }
  }
 */

namespace akg {
namespace ir {
using Region = Array<Range>;

/*
 * Var substitute cc0' = cc0 + lower_bound.
 *
 * Function:
 * 1) Replace cc0 -> cc0 - lower_bound
 * 2) Replace cc0 + lower_bound -> cc0
 *
 * Params:
 * 1) op->loop_var = cc0
 * 2) replace_expr_to_arg = cc0 - lower_bound
 * 3) replace_arg_to_expr = cc0 + lower_bound
 *
 * Note:
 * Currently TVM cannot simplify "cc0 - lower_bound + lower_bound" to cc0, so we need to
 * explicitly replace "cc0 + lower_bound" to cc0.
 * We cannot use TVM substitute because we need to do the two subtitutions in one pass.
 */
class ReplaceLoopArgWithConversion : public IRMutator {
  Expr Mutate_(const Variable *op, const Expr &e) final {
    if (op == for_op_->loop_var.get()) {
      return replace_expr_;
    } else {
      return IRMutator::Mutate_(op, e);
    }
  }

  Expr Mutate_(const Add *expr, const Expr &e) final {
    if (auto compare_add = compare_expr_.as<Add>()) {
      if (Equal(expr->a, compare_add->a) && Equal(expr->b, compare_add->b)) {
        return for_op_->loop_var;
      } else if (Equal(expr->a, compare_add->b) && Equal(expr->b, compare_add->a)) {
        return for_op_->loop_var;
      }
    }
    return IRMutator::Mutate_(expr, e);
  }

 public:
  Stmt run(const Stmt &stmt, const For *op, const Expr &replace_expr_to_arg, const Expr &replace_arg_to_expr) {
    for_op_ = op;
    compare_expr_ = replace_expr_to_arg;
    replace_expr_ = replace_arg_to_expr;
    return Mutate(stmt);
  }

 private:
  const For *for_op_{nullptr};
  Expr compare_expr_;
  Expr replace_expr_;
};

// main mutator class of this pass
class ConvertExtentToCondMutator : public IRMutator {
 private:
  bool IsAffineExprOfLoopVars(const Expr &expr) { return IsAffineExprOfVars(expr, loop_vars); }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    tensor_shape[op->func] = op->bounds;
    tensor_type[op->func] = op->type;
    auto stmt = IRMutator::Mutate_(op, s);
    return stmt;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    auto isParamExtent = [](Expr extent) -> bool {
      if (const auto v = extent.as<Variable>()) {
        std::string name = v->name_hint;
        for (auto c : name) {
          if (c >= 'A' && c <= 'Z') return true;
        }
      }
      return false;
    };
    const Variable *loop_var = op->loop_var.get();
    loop_vars.insert(loop_var);
    var_ptr_to_expr_map[loop_var] = op->loop_var;
    bool is_non_static_range = false;
    if (is_const(op->min) && is_const(op->extent)) {
      loop_var_known_range.emplace(loop_var, Range::make_by_min_extent(op->min, op->extent));
      loop_var_known_range_is_primary.emplace(loop_var, true);
    } else if (!isParamExtent(op->extent)) {
      is_non_static_range = true;
      loop_vars_with_unknown_range.insert(loop_var);
      loop_var_unknown_range[loop_var] = Range::make_by_min_extent(op->min, op->extent);
      loop_var_known_range.erase(loop_var);
      loop_var_known_range_is_primary.erase(loop_var);
      loop_var_known_range_conversion.erase(loop_var);
      loop_var_known_range_to_replace.erase(loop_var);
      if (IsAffineExprOfLoopVars(op->min) && IsAffineExprOfLoopVars(op->extent)) {
        Expr simplified_min = InferBoundOfExpr(op->min, loop_var_known_range).min;
        Expr original_max = Simplify(op->min + op->extent);
        Expr simplified_max = InferBoundOfExpr(original_max, loop_var_known_range).max;
        Expr simplified_extent = Simplify(simplified_max - simplified_min);
        if (is_const(simplified_min) && is_const(simplified_extent)) {
          loop_var_known_range.emplace(loop_var, Range::make_by_min_extent(simplified_min, simplified_extent));
          loop_var_known_range_is_primary.emplace(loop_var, true);
        }
      }
    }

    auto stmt = IRMutator::Mutate_(op, s);

    if (is_non_static_range) {
      loop_vars_with_unknown_range.erase(loop_var);
      if (loop_var_known_range.count(loop_var) == 0) {
        LOG(FATAL) << "We cannot determine the range of loop var " << loop_var
                   << " from loop body, please modify DSL.\n"
                   << "The loop var must be either an affine expression of other loop vars and constants (e.g. 3*"
                   << loop_var << " + 100),\n"
                   << "or be a tensor index variable inside the loop body (e.g. var(" << loop_var
                   << ") = 1, or var(some_expression + " << loop_var << ") = 1).\n";
      }

      op = stmt.as<For>();
      CHECK(op != nullptr);
      auto body = op->body;
      if (!loop_var_known_range_is_primary[loop_var]) {
        body = ReplaceLoopArgWithConversion().run(body, op, loop_var_known_range_to_replace[loop_var],
                                                  loop_var_known_range_conversion[loop_var]);
      }

      auto range = loop_var_known_range[loop_var];
      return For::make(op->loop_var, range->min, range->extent, op->for_type, op->device_api, body);
    } else {
      return stmt;
    }
  }

  template <class T>
  void CheckLoopVarRange(const T *op, const FunctionRef &tensor) {
    if (tensor_shape.count(tensor)) {
      auto shape = tensor_shape.find(tensor)->second;
      size_t arg_count = op->args.size();
      for (size_t arg_idx = 0; arg_idx < arg_count; arg_idx++) {
        auto arg = op->args[arg_idx];
        bool found = false;
        bool is_primary = false;
        Expr loop_var_conversion;
        Expr loop_var_to_replace;
        const Variable *loop_var = nullptr;
        if (arg.template as<Variable>()) {
          found = true;
          is_primary = true;
          loop_var = arg.template as<Variable>();
        } else if (arg.template as<Add>()) {
          const Add *add_arg = arg.template as<Add>();
          CHECK(add_arg);
          if (add_arg->a.template as<Variable>() && !IsVarInExpr(add_arg->a, add_arg->b)) {
            found = true;
            loop_var = add_arg->a.template as<Variable>();
            loop_var_conversion = Sub::make(add_arg->a, add_arg->b);
          } else if (add_arg->b.template as<Variable>() && !IsVarInExpr(add_arg->b, add_arg->a)) {
            found = true;
            loop_var = add_arg->b.template as<Variable>();
            loop_var_conversion = Sub::make(add_arg->b, add_arg->a);
          }
          loop_var_to_replace = arg;
        }
        if (found) {
          CHECK(loop_var != nullptr);
          if (loop_vars_with_unknown_range.count(loop_var)) {
            if (loop_var_known_range.count(loop_var)) {
              bool old_is_primary = loop_var_known_range_is_primary[loop_var];
              if (is_primary && !old_is_primary) {
                loop_var_known_range[loop_var] = shape[arg_idx];
                loop_var_known_range_is_primary[loop_var] = true;
              } else if (is_primary && old_is_primary) {
                // if loop var used in multiple tensors, use the intersection of realize range
                auto old_shape = loop_var_known_range[loop_var];
                auto new_shape = shape[arg_idx];
                Expr min = Simplify(Min::make(old_shape->min, new_shape->min));
                Expr old_max = Add::make(old_shape->min, old_shape->extent);
                Expr new_max = Add::make(new_shape->min, new_shape->extent);
                Expr extent = Simplify(Sub::make(Min::make(old_max, new_max), min));
                if (isImm(min) && isImm(extent)) {
                  Range intersect = Range::make_by_min_extent(min, extent);
                  loop_var_known_range[loop_var] = intersect;
                } else {
                  loop_var_known_range[loop_var] = shape[arg_idx];
                }
              }
            } else {
              loop_var_known_range[loop_var] = shape[arg_idx];
              loop_var_known_range_is_primary[loop_var] = is_primary;
              loop_var_known_range_conversion[loop_var] = loop_var_conversion;
              loop_var_known_range_to_replace[loop_var] = loop_var_to_replace;
            }
          }
        }
      }
    }
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    CheckLoopVarRange(op, op->func);
    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    CheckLoopVarRange(op, op->func);

    auto stmt = IRMutator::Mutate_(op, s);
    if (!loop_vars_with_unknown_range.empty()) {
      Expr and_condition;
      bool is_first = true;
      for (auto loop_var_name : loop_vars_with_unknown_range) {
        const VarExpr loop_var = var_ptr_to_expr_map[loop_var_name];
        auto range = loop_var_unknown_range[loop_var.get()];
        Expr condition;
        // avoid generate redundant 0 <= cc1 in if condition
        if (range->min.as<IntImm>() && range->min.as<IntImm>()->value == 0) {
          condition = LT::make(loop_var, range->extent);
        } else {
          auto lower_bound_cond = LE::make(range->min, loop_var);
          auto upper_bound_cond = LT::make(loop_var, Add::make(range->min, range->extent));
          condition = And::make(lower_bound_cond, upper_bound_cond);
        }

        if (is_first) {
          and_condition = condition;
          is_first = false;
        } else {
          and_condition = And::make(and_condition, condition);
        }
      }

      Stmt if_stmt = IfThenElse::make(and_condition, stmt);
      return if_stmt;
    } else {
      return stmt;
    }
  }

 public:
  Stmt run(const Stmt &stmt, const Map<Tensor, Buffer> &extern_buffer) {
    tensor_shape.clear();
    tensor_type.clear();
    for (auto buffer : extern_buffer) {
      Region region;
      auto buffer_shape = buffer.second->shape;
      std::transform(buffer_shape.begin(), buffer_shape.end(), std::back_inserter(region.CopyOnWrite()->data),
                     [](const Expr &extent) { return (Range::make_by_min_extent(0, extent)); });
      tensor_shape[buffer.first->op] = region;
      tensor_type[buffer.first->op] = buffer.second->dtype;
    }
    var_ptr_to_expr_map.clear();
    loop_var_known_range.clear();
    loop_var_unknown_range.clear();
    loop_vars.clear();
    loop_vars_with_unknown_range.clear();
    loop_var_known_range_is_primary.clear();
    loop_var_known_range_conversion.clear();
    return Mutate(stmt);
  }

 private:
  std::unordered_map<FunctionRef, Region, NodeHash, NodeEqual> tensor_shape;
  std::unordered_map<FunctionRef, Type, NodeHash, NodeEqual> tensor_type;
  std::unordered_map<const Variable *, VarExpr> var_ptr_to_expr_map;
  std::unordered_map<const Variable *, Range> loop_var_unknown_range;
  std::unordered_map<const Variable *, Range> loop_var_known_range;
  // loop_var_known_range_is_primary indicates whether the inferred loop var range is constant. Constant loop var range
  // is primary. Non-constant range needs var replacement (e.g. realize(i + expr) -> i' = i + expr -> realize(i') -> i =
  // i' - expr)
  std::unordered_map<const Variable *, bool> loop_var_known_range_is_primary;
  // loop_var_known_range_conversion and loop_var_known_range_to_replace are meaningful only when
  // loop_var_known_range_is_primary is false loop_var_known_range_conversion is the var replacement i = i' - expr
  std::unordered_map<const Variable *, Expr> loop_var_known_range_conversion;
  // loop_var_known_range_to_replace is the var replacement i' = i + expr
  std::unordered_map<const Variable *, Expr> loop_var_known_range_to_replace;
  std::unordered_set<const Variable *> loop_vars;
  std::unordered_set<const Variable *> loop_vars_with_unknown_range;
};

Stmt ConvertExtentToCond(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer) {
  stmt = ConvertExtentToCondMutator().run(stmt, extern_buffer);
  return stmt;
}
}  // namespace ir
}  // namespace akg
