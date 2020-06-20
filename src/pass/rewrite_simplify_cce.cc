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
#include <tvm/arithmetic.h>
#include <tvm/expr_operator.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <floating.h>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include "pass/rewrite_simplify_cce.h"
#include "pass/utils.h"

namespace akg {
namespace ir {
using ktvm::arith::Analyzer;

Expr Simplify_cce(Expr expr, const Map<Var, Range> &vrange) {
  Analyzer analyzer;
  for (auto kv : vrange) {
    analyzer.Bind(kv.first, kv.second);
  }

  arith::RewriteSimplifierCCE rewrite_simplify_cce(&analyzer);
  if (is_const(expr)) return expr;
  auto res = rewrite_simplify_cce(expr);
  if (is_const(res)) return res;
  res = analyzer.rewrite_simplify(res);
  if (is_const(res)) return res;
  res = analyzer.canonical_simplify(res);
  return res;
}

Stmt Simplify_cce(const Stmt &stmt, const Map<Var, Range> &vrange) {
  Analyzer analyzer;
  for (auto kv : vrange) {
    analyzer.Bind(kv.first, kv.second);
  }

  arith::RewriteSimplifierCCE rewrite_simplify_cce(&analyzer);
  auto res = rewrite_simplify_cce(stmt);

  res = Simplify(res, vrange);
  return res;
}
}  // namespace ir

namespace arith {
/// REWRITE SIMPLIFIER FOR CUSTOM CCE SIMPLIFICATIONS

// macro for doing simple rewrite
#define TVM_TRY_REWRITE(SrcExpr, ResExpr) \
  if ((SrcExpr).Match(ret)) {             \
    return (ResExpr).Eval();              \
  }

// macro for rewrite + recursively rewrite ResExpr
#define TVM_TRY_RECURSIVE_REWRITE(SrcExpr, ResExpr) \
  if ((SrcExpr).Match(ret)) {                       \
    return RecursiveRewrite((ResExpr).Eval());      \
  }

// macro rewrite only if CondExor is true after match.
#define TVM_TRY_REWRITE_IF(SrcExpr, ResExpr, CondExpr) \
  if ((SrcExpr).Match(ret) && (CondExpr)) {            \
    return (ResExpr).Eval();                           \
  }

// macro rewrite + recursive_rewrite only if CondExor is true after match.
#define TVM_TRY_RECURSIVE_REWRITE_IF(SrcExpr, ResExpr, CondExpr) \
  if ((SrcExpr).Match(ret) && (CondExpr)) {                      \
    return RecursiveRewrite((ResExpr).Eval());                   \
  }

using ktvm::arith::Analyzer;
using ktvm::arith::ConstIntBound;
using ktvm::arith::IntervalSet;
using ktvm::arith::PVar;
using ktvm::arith::TryConstFold;

// try to prove x equals val
RewriteSimplifierCCE::Impl::CompareResult RewriteSimplifierCCE::Impl::TryCompare(const Expr &x, int64_t val) {
  Expr diff = Mutate(x);
  if (const auto *ptr = diff.as<IntImm>()) {
    if (ptr->value == val) {
      return kEQ;
    } else if (ptr->value > val) {
      return kGT;
    } else {
      return kLT;
    }
  }
  ConstIntBound dbound = parent_->const_int_bound(diff);
  if (dbound->min_value > val) {
    return kGT;
  }
  if (dbound->max_value < val) {
    return kLT;
  }
  if (dbound->min_value >= val) {
    return kGE;
  }
  if (dbound->max_value <= val) {
    return kLE;
  }
  if (val == 0) {
    ktvm::arith::ModularSet dmod = parent_->modular_set(diff);
    if (dmod->base != 0) {
      return kNE;
    }
  }
  return kUnknown;
}

template <typename T>
Expr ModSimplify(const Mod *op, const Expr &e) {
  if (const T *lhs = op->a.as<T>()) {
    Expr new_lhs_a = lhs->a, new_lhs_b = lhs->b;
    if (is_const(op->b)) {
      if (is_const(lhs->a)) {
        new_lhs_a = Mod::make(lhs->a, op->b);
      }
      if (is_const(lhs->b)) {
        new_lhs_b = Mod::make(lhs->b, op->b);
      }
      Expr new_lhs = Cast::make(op->b.type(), T::make(new_lhs_a, new_lhs_b));
      return Mod::make(new_lhs, op->b);
    }
  }
  return e;
}

// Loop Partition Specific Mutators
Stmt RewriteSimplifierCCE::Impl::Mutate_(const For *op, const Stmt &self) {
  const Variable *loop_var = op->loop_var.get();
  iteration_vars_[loop_var] = ktvm::IntSet::interval(op->min, op->extent + op->min - 1);
  in_for_bound_ = true;
  Expr min = this->Mutate(op->min);
  Expr extent = this->Mutate(op->extent);
  if (EvalSet(extent, iteration_vars_).can_prove_non_positive()) {
    return Evaluate::make(0);
  }
  in_for_bound_ = false;
  Stmt body = this->Mutate(op->body);
  iteration_vars_.erase(loop_var);
  return For::make(op->loop_var, min, extent, op->for_type, op->device_api, body);
}

// Rewrite simplify rules
Expr RewriteSimplifierCCE::Impl::Mutate_(const And *op, const Expr &self) {
  Expr ret = IRMutator::Mutate_(op, self);
  // Pattern var to match any expression
  PVar<Expr> x, y;
  // Pattern var match IntImm

  TVM_TRY_REWRITE((x >= y) && (x <= y), x == y);
  TVM_TRY_REWRITE((x >= y) && (y >= x), x == y);
  TVM_TRY_REWRITE((x <= y) && (y <= x), x == y);
  TVM_TRY_REWRITE((x <= y) && (x >= y), x == y);

  if (self.same_as(ret)) {
    return self;
  }

  return ret;
}

Expr RewriteSimplifierCCE::Impl::Mutate_(const Or *op, const Expr &self) {
  Expr ret = IRMutator::Mutate_(op, self);
  if (self.same_as(ret)) {
    return self;
  }

  return ret;
}

Expr RewriteSimplifierCCE::Impl::Mutate_(const LT *op, const Expr &self) {
  auto ret = RemoveSelectFromCond<LT>(op->a, op->b, self);
  if (!ret.same_as(self)) return ret;

  IntervalSet lhs_interval, rhs_interval;
  if (VarIntervalIsUseful(op->a, lhs_interval, op->b, rhs_interval)) {
    if (CanProve(lhs_interval.max() < rhs_interval.min())) {
      return make_const(self.type(), 1);
    } else if (CanProve(lhs_interval.min() >= rhs_interval.max())) {
      return make_zero(self.type());
    }
  }

  return IRMutator::Mutate_(op, self);
}

Expr RewriteSimplifierCCE::Impl::Mutate_(const LE *op, const Expr &self) {
  auto ret = RemoveSelectFromCond<LE>(op->a, op->b, self);
  if (!ret.same_as(self)) return ret;

  IntervalSet lhs_interval, rhs_interval;
  if (VarIntervalIsUseful(op->a, lhs_interval, op->b, rhs_interval)) {
    if (CanProve(lhs_interval.max() <= rhs_interval.min())) {
      return make_const(self.type(), 1);
    } else if (CanProve(lhs_interval.min() > rhs_interval.max())) {
      return make_zero(self.type());
    }
  }

  return IRMutator::Mutate_(op, self);
}

Expr RewriteSimplifierCCE::Impl::Mutate_(const GT *op, const Expr &self) {
  auto ret = RemoveSelectFromCond<GT>(op->a, op->b, self);
  if (!ret.same_as(self)) return ret;

  IntervalSet lhs_interval, rhs_interval;
  if (VarIntervalIsUseful(op->a, lhs_interval, op->b, rhs_interval)) {
    if (CanProve(lhs_interval.min() > rhs_interval.max())) {
      return make_const(self.type(), 1);
    } else if (CanProve(lhs_interval.max() <= rhs_interval.min())) {
      return make_zero(self.type());
    }
  }

  return IRMutator::Mutate_(op, self);
}

Expr RewriteSimplifierCCE::Impl::Mutate_(const GE *op, const Expr &self) {
  auto ret = RemoveSelectFromCond<GE>(op->a, op->b, self);
  if (!ret.same_as(self)) return ret;

  IntervalSet lhs_interval, rhs_interval;
  if (VarIntervalIsUseful(op->a, lhs_interval, op->b, rhs_interval)) {
    if (CanProve(lhs_interval.min() >= rhs_interval.max())) {
      return make_const(self.type(), 1);
    } else if (CanProve(lhs_interval.max() < rhs_interval.min())) {
      return make_zero(self.type());
    }
  }

  return IRMutator::Mutate_(op, self);
}

Expr RewriteSimplifierCCE::Impl::Mutate_(const EQ *op, const Expr &self) {
  Expr ret = IRMutator::Mutate_(op, self);
  // Pattern var to match any expression
  PVar<Expr> x;
  auto ctrue = ktvm::arith::PConst<Expr>(make_const(op->type, true));

  Expr const_res = TryConstFold<EQ>(op->a, op->b);
  if (const_res.defined()) return const_res;

  TVM_TRY_REWRITE(x == x, ctrue);

  // remove (i == j) conditions if they are always false
  IntervalSet lhs_interval, rhs_interval;
  if (VarIntervalIsUseful(op->a, lhs_interval, op->b, rhs_interval)) {
    if (Intersect(parent_, lhs_interval, rhs_interval)->IsEmpty()) {
      return make_zero(ret.type());
    }
  }

  return ret;
}

Expr RewriteSimplifierCCE::Impl::Mutate_(const NE *op, const Expr &self) {
  auto ret = RemoveSelectFromCond<NE>(op->a, op->b, self);
  if (!ret.same_as(self)) return ret;

  // return TRUE for (i != j) conditions if their intervals never intersect
  IntervalSet lhs_interval, rhs_interval;
  if (VarIntervalIsUseful(op->a, lhs_interval, op->b, rhs_interval)) {
    if (Intersect(parent_, lhs_interval, rhs_interval)->IsEmpty()) {
      return make_const(op->type, true);
    }
  }

  return self;
}

IntervalSet RewriteSimplifierCCE::Impl::BoundToIntervalSet(const ConstIntBound &bound) {
  Expr min, max;
  if (bound->min_value == ConstIntBound::kNegInf)
    min = ktvm::arith::neg_inf();
  else if (bound->min_value == ConstIntBound::kPosInf)
    min = ktvm::arith::pos_inf();
  else
    min = Expr(bound->min_value);

  if (bound->max_value == ConstIntBound::kNegInf)
    max = ktvm::arith::neg_inf();
  else if (bound->max_value == ConstIntBound::kPosInf)
    max = ktvm::arith::pos_inf();
  else
    max = Expr(bound->max_value);

  return IntervalSet(min, max);
}

bool RewriteSimplifierCCE::Impl::VarIntervalIsUseful(const Expr &e, IntervalSet &interval_e, const Expr &e2,
                                                     IntervalSet &interval_e2) {
  return VarIntervalIsUseful(e, interval_e) && VarIntervalIsUseful(e2, interval_e2);
}

bool RewriteSimplifierCCE::Impl::VarIntervalIsUseful(const Expr &e, IntervalSet &interval) {
  auto var = e.as<Variable>();
  if (var && iteration_vars_.find(var) == iteration_vars_.end()) {
    interval = BoundToIntervalSet(parent_->const_int_bound(e));
  } else {
    interval = Downcast<IntervalSet>(EvalSet(e, iteration_vars_));
  }
  Expr min_value = interval.min();
  Expr max_value = interval.max();
  if (min_value.type() != e.type()) {
    min_value = Cast::make(e.type(), min_value);
  }
  if (max_value.type() != e.type()) {
    max_value = Cast::make(e.type(), max_value);
  }
  interval = IntervalSet(min_value, max_value);
  return !interval.is_everything() && !interval.is_nothing();
}

Expr RewriteSimplifierCCE::Impl::Mutate_(const Sub *op, const Expr &self) {
  Expr ret = IRMutator::Mutate_(op, self);
  // Pattern var to match any expression
  PVar<Expr> x, y, z;
  // Pattern var match IntImm
  PVar<Integer> c1, c2, c3;
  PVar<Floating> f1;

  Expr const_res = TryConstFold<Sub>(op->a, op->b);
  if (const_res.defined()) return const_res;

  if (ktvm::arith::IsNumericType(op->type)) {
    // Any number type rules
    // cancelation rules
    TVM_TRY_REWRITE((x + y) - y, x);
    TVM_TRY_REWRITE((x + y) - x, y);
    TVM_TRY_REWRITE(x - (y + x), 0 - y);
    TVM_TRY_REWRITE(x - (x + y), 0 - y);

    TVM_TRY_REWRITE(min(x, y) - x, min(0, y - x));
    TVM_TRY_REWRITE(min(x, y) - y, min(x - y, 0));
    TVM_TRY_REWRITE(max(x, y) - x, max(0, y - x));
    TVM_TRY_REWRITE(max(x, y) - y, max(x - y, 0));

    TVM_TRY_REWRITE(x - max(x, y), min(0, x - y));
    TVM_TRY_REWRITE(y - max(x, y), min(y - x, 0));
    TVM_TRY_REWRITE(x - min(x, y), max(0, x - y));
    TVM_TRY_REWRITE(y - min(x, y), max(y - x, 0));

    // mul co-efficient folding
    TVM_TRY_REWRITE(x - x, ZeroWithTypeLike(x));
    TVM_TRY_REWRITE(x * y - x, x * (y - 1));
    TVM_TRY_REWRITE(y * x - x, x * (y - 1));
    TVM_TRY_REWRITE(x - y * x, x * (1 - y));
    TVM_TRY_REWRITE(x - x * y, x * (1 - y));
    TVM_TRY_REWRITE(x * y - x * z, x * (y - z));
    TVM_TRY_REWRITE(y * x - x * z, x * (y - z));
    TVM_TRY_REWRITE(x * y - z * x, x * (y - z));
    TVM_TRY_REWRITE(y * x - z * x, x * (y - z));

    // constant cancelation
    TVM_TRY_REWRITE((x + c1) - c2, x + (c1.Eval()->value - c2.Eval()->value));
    TVM_TRY_REWRITE((c1 - x) - (c2 - y), (y - x) + (c1.Eval()->value - c2.Eval()->value));

    // cancelization rule involving 4 operands
    TVM_TRY_REWRITE((x + y) - (x + z), y - z);
    TVM_TRY_REWRITE((x + y) - (z + x), y - z);
    TVM_TRY_REWRITE((y + x) - (z + x), y - z);
    TVM_TRY_REWRITE((y + x) - (x + z), y - z);

    TVM_TRY_REWRITE(min(x + y, z) - x, min(y, z - x));
    TVM_TRY_REWRITE(min(y + x, z) - x, min(y, z - x));
    TVM_TRY_REWRITE(min(z, x + y) - x, min(z - x, y));
    TVM_TRY_REWRITE(min(z, y + x) - x, min(z - x, y));

    TVM_TRY_REWRITE(max(x + y, z) - x, max(y, z - x));
    TVM_TRY_REWRITE(max(y + x, z) - x, max(y, z - x));
    TVM_TRY_REWRITE(max(z, x + y) - x, max(z - x, y));
    TVM_TRY_REWRITE(max(z, y + x) - x, max(z - x, y));

    TVM_TRY_REWRITE(x - min(x + y, z), max(0 - y, x - z));
    TVM_TRY_REWRITE(x - min(y + x, z), max(0 - y, x - z));
    TVM_TRY_REWRITE(x - min(z, x + y), max(x - z, 0 - y));
    TVM_TRY_REWRITE(x - min(z, y + x), max(x - z, 0 - y));

    TVM_TRY_REWRITE(min(x, y) - min(y, x), ZeroWithTypeLike(x));
    TVM_TRY_REWRITE(max(x, y) - max(y, x), ZeroWithTypeLike(x));
    // canonicalization rule
    // will try rewrite again after canonicalization.
    TVM_TRY_REWRITE(x - c1, x + (0 - c1));
    TVM_TRY_REWRITE(x - f1, x + (0 - f1));

    TVM_TRY_RECURSIVE_REWRITE((x + c1) - y, (x - y) + c1);
    TVM_TRY_RECURSIVE_REWRITE(x - (y - z), (x + z) - y);
    TVM_TRY_RECURSIVE_REWRITE(x - y * c1, x + y * (0 - c1));
  }

  TVM_TRY_REWRITE(c1 - select(x, c2, c3), select(x, c1 - c2, c1 - c3));
  TVM_TRY_REWRITE(select(x, c1, c2) - c3, select(x, c1 - c3, c2 - c3));

  if (CanProveEqual(op->a, 0)) {
    return -(op->b);
  }

  if (in_for_bound_) {
    TVM_TRY_REWRITE_IF(x - y, ZeroWithTypeLike(x),
                       EvalSet(y.Eval(), iteration_vars_).can_prove_non_negative() &&
                         EvalSet(x.Eval(), iteration_vars_).can_prove_non_positive());
  }

  return ret;
}

Expr RewriteSimplifierCCE::Impl::Mutate_(const Add *op, const Expr &self) {
  Expr ret = IRMutator::Mutate_(op, self);
  // Pattern var to match any expression
  PVar<Expr> x, y, z;
  // Pattern var match IntImm
  PVar<Integer> c1, c2, c3;
  PVar<Floating> f1, f2;

  Expr const_res = TryConstFold<Add>(op->a, op->b);
  if (const_res.defined()) return const_res;
  if (ktvm::arith::IsNumericType(op->type)) {
    // Any number type rules
    // cancelation rules
    TVM_TRY_REWRITE((x - y) + y, x);
    TVM_TRY_REWRITE(x + (y - x), y);

    TVM_TRY_REWRITE((x - y) + (y - z), x - z);
    TVM_TRY_REWRITE((x - y) + (z - x), z - y);

    TVM_TRY_REWRITE(min(x, y - z) + z, min(x + z, y));
    TVM_TRY_REWRITE(min(x - z, y) + z, min(x, y + z));
    TVM_TRY_REWRITE(max(x, y - z) + z, max(x + z, y));
    TVM_TRY_REWRITE(max(x - z, y) + z, max(x, y + z));

    TVM_TRY_REWRITE_IF(min(x, y + z * c1) + z * c2, min(x + z * c2, y), c1.Eval()->value == -c2.Eval()->value);
    TVM_TRY_REWRITE_IF(max(x, y + z * c1) + z * c2, max(x + z * c2, y), c1.Eval()->value == -c2.Eval()->value);
    TVM_TRY_REWRITE_IF(min(y + z * c1, x) + z * c2, min(x + z * c2, y), c1.Eval()->value == -c2.Eval()->value);
    TVM_TRY_REWRITE_IF(max(y + z * c1, x) + z * c2, max(x + z * c2, y), c1.Eval()->value == -c2.Eval()->value);

    TVM_TRY_REWRITE(max(x, y) + min(x, y), x + y);
    TVM_TRY_REWRITE(min(x, y) + max(x, y), x + y);
    TVM_TRY_REWRITE(max(x, y) + min(y, x), x + y);
    TVM_TRY_REWRITE(min(x, y) + max(y, x), x + y);

    TVM_TRY_REWRITE_IF(min(x, y + c1) + c2, min(x + c2, y), c1.Eval()->value == -c2.Eval()->value);
    TVM_TRY_REWRITE_IF(min(x + c1, y) + c2, min(x, y + c2), c1.Eval()->value == -c2.Eval()->value);
    TVM_TRY_REWRITE_IF(max(x, y + c1) + c2, max(x + c2, y), c1.Eval()->value == -c2.Eval()->value);
    TVM_TRY_REWRITE_IF(max(x + c1, y) + c2, max(x, y + c2), c1.Eval()->value == -c2.Eval()->value);

    // constant folding
    // NOTE: canonicalization might better at this.
    TVM_TRY_REWRITE((x + c1) + c2, x + (c1.Eval()->value + c2.Eval()->value));
    TVM_TRY_REWRITE((x - c1) + c2, x + (c2.Eval()->value - c1.Eval()->value));
    TVM_TRY_REWRITE(max(x, c2) + c1, max(x + c1, c1.Eval()->value + c2.Eval()->value));
    TVM_TRY_REWRITE(f1 + x, x + f1);

    // mul co-efficient folding
    TVM_TRY_REWRITE(x + x, x * 2);
    TVM_TRY_REWRITE(x * y + x, x * (y + 1));
    TVM_TRY_REWRITE(y * x + x, x * (y + 1));
    TVM_TRY_REWRITE(x + y * x, x * (1 + y));
    TVM_TRY_REWRITE(x + x * y, x * (1 + y));
    TVM_TRY_REWRITE(x * y + x * z, x * (y + z));
    TVM_TRY_REWRITE(y * x + x * z, x * (y + z));
    TVM_TRY_REWRITE(x * y + z * x, x * (y + z));
    TVM_TRY_REWRITE(y * x + z * x, x * (y + z));

    // DivMod rules
    // truc div
    TVM_TRY_REWRITE(truncdiv(x, c1) * c1 + truncmod(x, c1), x);
    TVM_TRY_REWRITE(truncdiv(x, y) * y + truncmod(x, y), x);
    // floor div
    TVM_TRY_REWRITE(floordiv(x, c1) * c1 + floormod(x, c1), x);
    TVM_TRY_REWRITE(floordiv(x, y) * y + floormod(x, y), y);

    // canonicalization rule
    // will try rewrite again after canonicalization.
    TVM_TRY_RECURSIVE_REWRITE(c1 + x, x + c1);
    TVM_TRY_RECURSIVE_REWRITE(x + (c1 - y), (x - y) + c1);
    TVM_TRY_RECURSIVE_REWRITE(x + c1 + y, (x + y) + c1);
    TVM_TRY_RECURSIVE_REWRITE(x + (c1 + y), (x + y) + c1);

    TVM_TRY_RECURSIVE_REWRITE(x + max(y, z), max(y, z) + x);
    TVM_TRY_RECURSIVE_REWRITE(x + min(y, z), min(y, z) + x);

    TVM_TRY_REWRITE_IF(((x * select(y, f1, f2)) + (select(z, f2, f1) * select(y, f2, f1))),
                       select(y, select(z, f2, f1), x), f1.Eval()->value == 0 && f2.Eval()->value == 1);
  }

  TVM_TRY_REWRITE(c1 + select(x, c2, c3), select(x, c1 + c2, c1 + c3));
  TVM_TRY_REWRITE(select(x, c1, c2) + c3, select(x, c1 + c3, c2 + c3));

  return ret;
}

Expr RewriteSimplifierCCE::Impl::Mutate_(const Mul *op, const Expr &self) {
  Expr ret = IRMutator::Mutate_(op, self);
  // Pattern var to match any expression
  PVar<Expr> x, y;
  // Pattern var match IntImm
  PVar<Integer> c1, c2, c3;
  // Pattern var match FloatImm
  PVar<Floating> f1, f2;

  Expr const_res = TryConstFold<Mul>(op->a, op->b);
  if (const_res.defined()) return const_res;

  if (ktvm::arith::IsNumericType(op->type)) {
    // constant simplification rule
    TVM_TRY_REWRITE((x + c1) * c2, x * c2 + c1 * c2);
    TVM_TRY_REWRITE((x * c1) * c2, x * (c1 * c2));
    TVM_TRY_REWRITE(min(x, y) * max(x, y), x * y);
    TVM_TRY_REWRITE(max(x, y) * min(x, y), x * y);
    TVM_TRY_REWRITE_IF(max(x, c2) * c1, max(x * c1, c1.Eval()->value * c2.Eval()->value), c1.Eval()->value > 0);

    // canonicalization
    TVM_TRY_REWRITE(c1 * x, x * c1);  // Unnecessary?
    TVM_TRY_RECURSIVE_REWRITE(c1 * x, x * c1);

    TVM_TRY_RECURSIVE_REWRITE(x * (c1 * y), (x * y) * c1);
    TVM_TRY_RECURSIVE_REWRITE(c1 * x, x * c1);
    TVM_TRY_RECURSIVE_REWRITE_IF((x - y) * c1, (y - x) * (0 - c1), c1.Eval()->value < 0);

    // Rules for float constants
    TVM_TRY_REWRITE_IF((x * f1) * f2, x * (f1 * f2), (akg::ir::LimitCheck<Floating>(f1, f2)));

    TVM_TRY_RECURSIVE_REWRITE(f1 * x, x * f1);

    TVM_TRY_REWRITE_IF(floordiv(x, y) * y, x, op->type.is_float());
    TVM_TRY_REWRITE_IF(floordiv(x, y) * x, y, op->type.is_float());
  }

  TVM_TRY_REWRITE_IF(c1 * select(x, c2, c3), select(x, c1 * c2, c1 * c3),
                     (akg::ir::LimitCheck<Integer>(c1, c2) && akg::ir::LimitCheck<Integer>(c1, c3)));
  TVM_TRY_REWRITE_IF(select(x, c1, c2) * c3, select(x, c1 * c3, c2 * c3),
                     (akg::ir::LimitCheck<Integer>(c1, c3) && akg::ir::LimitCheck<Integer>(c2, c3)));

  return ret;
}

Expr RewriteSimplifierCCE::Impl::Mutate_(const Div *op, const Expr &self) {
  Expr ret = IRMutator::Mutate_(op, self);
  // Pattern var to match any expression
  PVar<Expr> x, y;
  // Pattern var match IntImm
  PVar<Integer> c1, c2, c3;

  Expr const_res = TryConstFold<Div>(op->a, op->b);
  if (const_res.defined()) return const_res;

  TVM_TRY_REWRITE(truncdiv(c1, select(x, c2, c3)), select(x, truncdiv(c1, c2), truncdiv(c1, c3)));
  TVM_TRY_REWRITE(truncdiv(select(x, c1, c2), c3), select(x, truncdiv(c1, c3), truncdiv(c2, c3)));

  TVM_TRY_REWRITE_IF(floordiv((x * y), y), x, op->type.is_float());
  TVM_TRY_REWRITE_IF(floordiv((x * y), x), y, op->type.is_float());

  return ret;
}

Expr RewriteSimplifierCCE::Impl::Mutate_(const Mod *op, const Expr &self) {
  Expr ret = IRMutator::Mutate_(op, self);
  // Pattern var to match any expression
  PVar<Expr> x;
  // Pattern var match IntImm
  PVar<Integer> c1, c2, c3;

  Expr const_res = TryConstFold<Mod>(op->a, op->b);
  if (const_res.defined()) return const_res;

  // (x * c1) % c2 % c3 == (x % (c3 // c1)) * c1 if (c2 % c3 == 0) and (c3 % c1 == 0)
  TVM_TRY_REWRITE_IF(truncmod(truncmod((x * c1), c2), c3), truncmod(x, (truncdiv(c3, c1))) * c1,
                     CanProve(truncmod(c2.Eval(), c3.Eval()) == 0) && CanProve(truncmod(c3.Eval(), c1.Eval()) == 0));
  TVM_TRY_REWRITE(truncmod(c1, select(x, c2, c3)), select(x, truncmod(c1, c2), truncmod(c1, c3)));
  TVM_TRY_REWRITE(truncmod(select(x, c1, c2), c3), select(x, truncmod(c1, c3), truncmod(c2, c3)));

  TVM_TRY_REWRITE_IF(truncmod((truncmod(x, c1)), c2), truncmod(x, c2), CanProve(truncmod(c1.Eval(), c2.Eval()) == 0));

  // (x + y) % z == ((x % z) + (y % z)) % z
  Expr new_e = ModSimplify<Add>(op, ret);
  // (x * y) % z == ((x % z) * (y % z)) % z
  if (new_e.same_as(ret)) {
    new_e = ModSimplify<Mul>(op, ret);
  }
  // (x - y) % z == ((x % z) - (y % z)) % z
  if (new_e.same_as(ret)) {
    new_e = ModSimplify<Sub>(op, ret);
  }
  if (!new_e.same_as(ret)) {
    return this->Mutate(new_e);
  }

  return ret;
}

Expr RewriteSimplifierCCE::Impl::Mutate_(const Max *op, const Expr &self) {
  Expr ret = IRMutator::Mutate_(op, self);
  // Pattern var to match any expression
  PVar<Expr> x, y;
  PVar<Expr> tv, fv, tv2, fv2;

  TVM_TRY_REWRITE_IF(max(select(x, tv, fv), select(x, tv2, fv2)), max(tv, tv2), CanProve(x.Eval()));

  TVM_TRY_REWRITE_IF(max(select(x, tv, fv), select(x, tv2, fv2)), max(fv, fv2), CanProve((!x).Eval()));

  TVM_TRY_REWRITE(max(select(x, tv, fv), select(x, tv2, fv2)), select(x, max(tv, tv2), max(fv, fv2)));

  if ((max(x, y).Match(ret))) {
    IntervalSet lhs_interval, rhs_interval;
    if (VarIntervalIsUseful(x.Eval(), lhs_interval, y.Eval(), rhs_interval)) {
      if (CanProve(lhs_interval.min() >= rhs_interval.max())) {
        return x.Eval();
      } else if (CanProve(rhs_interval.min() >= lhs_interval.max())) {
        return y.Eval();
      }
    }
  }

  return ret;
}

Expr RewriteSimplifierCCE::Impl::Mutate_(const Min *op, const Expr &self) {
  Expr ret = IRMutator::Mutate_(op, self);
  // Pattern var to match any expression
  PVar<Expr> x, y;
  PVar<Expr> tv, fv, tv2, fv2;

  TVM_TRY_REWRITE_IF(min(select(x, tv, fv), select(x, tv2, fv2)), min(tv, tv2), CanProve(x.Eval()));

  TVM_TRY_REWRITE_IF(min(select(x, tv, fv), select(x, tv2, fv2)), min(fv, fv2), CanProve((!x).Eval()));

  TVM_TRY_REWRITE(min(select(x, tv, fv), select(x, tv2, fv2)), select(x, min(tv, tv2), min(fv, fv2)));

  if ((min(x, y).Match(ret))) {
    IntervalSet lhs_interval, rhs_interval;
    if (VarIntervalIsUseful(x.Eval(), lhs_interval, y.Eval(), rhs_interval)) {
      if (CanProve(lhs_interval.max() <= rhs_interval.min())) {
        return x.Eval();
      } else if (CanProve(rhs_interval.max() <= lhs_interval.min())) {
        return y.Eval();
      }
    }
  }

  return ret;
}

Expr RewriteSimplifierCCE::Impl::Mutate_(const Select *op, const Expr &self) {
  Expr ret = IRMutator::Mutate_(op, self);

  // Pattern var to match any expression
  PVar<Expr> x, y;
  TVM_TRY_REWRITE(select(x, y, y), y);
  PVar<Expr> tv, fv;
  TVM_TRY_REWRITE(select((x <= y), tv, fv), select(y < x, fv, tv));
  TVM_TRY_REWRITE(select(x != y, tv, fv), select(x == y, fv, tv));
  if (self.same_as(ret)) {
    return self;
  }

  return ret;
}

std::unordered_set<const Variable *> RewriteSimplifierCCE::Impl::GetExprVars(const Expr &expr) const {
  std::unordered_set<const Variable *> vars;
  PostOrderVisit(expr, [&vars](const NodeRef &node) {
    if (const auto v = node.as<Variable>()) {
      vars.insert(v);
    }
  });
  return vars;
}

/*!
 * \brief Checks if an expression is always True,
 * given the ranges of its variables.
 *
 *  This function evaluates the value of an expression \p expr with the boundaries
 *  of the Range of its (single) Variable. If the expression is True for both cases,
 *  this function returns True.
 *
 *  If \p expr has more than one Variable, this function returns false.
 *
 * \param expr The expression to evaluate.
 * \return True if \p expr is always True for one Variable, False otherwise.
 */
bool RewriteSimplifierCCE::Impl::Impl::IsConditionTrue(const Expr expr) const {
#if CHECK_IMPL_CONDITION
  // TRUE Find the single var in expr. Or return false.
  std::vector<std::unordered_map<const Variable *, Expr>> super_set;
  auto vars = GetExprVars(expr);
  if (vars.size() == 1) {
    const Variable *first = *(vars.begin());
    ConstIntBound a_bound = parent_->const_int_bound(Expr(akg::ir::GetObjPtr(first)));
    if (!a_bound.defined()) {
      return false;
    }

    super_set.push_back({{first, Expr((int32_t)(a_bound->min_value))}});
    super_set.push_back({{first, Expr((int32_t)(a_bound->max_value))}});
  } else if (vars.size() == 2) {
    const Variable *first = *(vars.begin());
    vars.erase(first);
    const Variable *second = *(vars.begin());

    ConstIntBound a_bound = parent_->const_int_bound(Expr(akg::ir::GetObjPtr(first)));
    ConstIntBound b_bound = parent_->const_int_bound(Expr(akg::ir::GetObjPtr(second)));
    if (!a_bound.defined() || !b_bound.defined()) {
      return false;
    }

    super_set.push_back({{first, Expr((int32_t)(a_bound->min_value))}, {second, Expr((int32_t)(b_bound->min_value))}});
    super_set.push_back({{first, Expr((int32_t)(a_bound->min_value))}, {second, Expr((int32_t)(b_bound->max_value))}});
    super_set.push_back({{first, Expr((int32_t)(a_bound->max_value))}, {second, Expr((int32_t)(b_bound->min_value))}});
    super_set.push_back({{first, Expr((int32_t)(a_bound->max_value))}, {second, Expr((int32_t)(b_bound->max_value))}});
  } else {
    return false;
  }

  for (auto it : super_set) {
    auto newExpr = Substitute(expr, it);
    if (!is_one(Simplify(newExpr))) {
      return false;
    }
  }
  return true;
#endif
  return false;
}

/**
 * Fills up the iteration_vars_ map with Intervals obtained from
 * the bounds captured by Analyzer when the Simplifier was created
 *
 * \param e The original expression to simplify.
 * */
void RewriteSimplifierCCE::Impl::SetIteratorsFromBounds(const Expr &e) {
  ConstIntBound e_bound;
  auto vars = GetExprVars(e);
  for (auto exp : vars) {
    e_bound = parent_->const_int_bound(Expr(akg::ir::GetObjPtr(exp)));
    Expr min_bound =
      (e_bound->min_value == e_bound.kNegInf ? ktvm::arith::SymbolicLimits::neg_inf_ : Expr(e_bound->min_value));
    Expr max_bound =
      (e_bound->min_value == e_bound.kPosInf ? ktvm::arith::SymbolicLimits::pos_inf_ : Expr(e_bound->max_value));
    iteration_vars_[exp] = ktvm::IntSet::interval(min_bound, max_bound);
  }
}

/*!
 * \brief Transforms a condition into a simplified form.
 *
 * Current patterns matched:
 * ###(expr1 <= expr2 && expr2 <= expr1) into (expr1 == expr2)
 * (expr1 < expr2 && expr2 == expr1) into (expr1 <= expr2)
 * (expr1 < expr2 && expr3 <= expr2) into (expr1 < expr2) if expr2 > expr3
 * or (expr3 <= expr2) otherwise
 *
 * \param cond The condition to transform.
 * \return A transformed expression if a pattern was matched,
 *  otherwise the original expression.
 */
Expr RewriteSimplifierCCE::Impl::ReduceCondition(Expr cond) {
  Expr ret = cond;
  if (auto op = cond.as<And>()) {
    Expr const_res = TryConstFold<And>(op->a, op->b);
    if (const_res.defined()) return const_res;
  }

  // Pattern var to match any expression
  PVar<Expr> x, y, z, a;

  TVM_TRY_REWRITE_IF(x < y && z <= y, x < y, CanProve(x.Eval() > z.Eval()));
  TVM_TRY_REWRITE_IF(x < y && z <= y, z <= y, CanProve(z.Eval() >= x.Eval()));

  TVM_TRY_REWRITE_IF(
    x < y || z == a, x <= z,
    CanProve((x.Eval() == z.Eval() && y.Eval() == a.Eval()) || (x.Eval() == a.Eval() && y.Eval() == z.Eval())));
  // Note:Could this rule above be just TVM_TRY_REWRITE(x < y || x == y, x <= y) ?
  return cond;
}

/*!
 * \brief Removes a select appearing as an operand of a condition.
 *
 * Converts an Expression of the form: (Expr CondOp Select(Cond, T, F))
 * To the form: ((Cond && Expr CondOp T) || (!Cond && Expr CondOp F))
 *
 * \param cond The condition to transform.
 * \return If possible, a new condition without the Select, otherwise it
 *  returns the original expression.
 */
template <typename T1>
Expr RewriteSimplifierCCE::Impl::RemoveSelectFromCond(const Expr &lhs, const Expr &rhs, const Expr &self) {
  using ktvm::ir::Simplify;
  if (auto select = rhs.as<Select>()) {  // If the Select is on the rhs
    auto orLhs = select->condition;
    auto orLhs2 = Simplify(T1::make(lhs, select->true_value));
    orLhs = ReduceCondition(And::make(orLhs, orLhs2));
    orLhs = ReduceCondition(Simplify(orLhs));
    auto orRhs = Simplify(Not::make(select->condition));
    auto orRhs2 = Simplify(T1::make(lhs, select->false_value));
    orRhs = ReduceCondition(And::make(orRhs, orRhs2));
    orRhs = ReduceCondition(Simplify(orRhs));
    auto newCond = Simplify(Or::make(orLhs, orRhs));
    newCond = ReduceCondition(newCond);
    return newCond;
  }
  if (auto select = lhs.as<Select>()) {  // If the Select is on the lhs
    auto orLhs = select->condition;
    auto orLhs2 = Simplify(T1::make(select->true_value, rhs));
    orLhs = Mutate(ReduceCondition(And::make(orLhs, orLhs2)));
    orLhs = Mutate(ReduceCondition(Simplify(orLhs)));

    auto orRhs = Simplify(Not::make(select->condition));
    auto orRhs2 = Simplify(T1::make(select->false_value, rhs));
    orRhs = Mutate(ReduceCondition(And::make(orRhs, orRhs2)));
    orRhs = Mutate(ReduceCondition(Simplify(orRhs)));

    auto newCond = Simplify(Or::make(orLhs, orRhs));
    newCond = Mutate(ReduceCondition(newCond));
    return newCond;
  }

  auto newLhs = Mutate(lhs);
  auto newRhs = Mutate(rhs);
  if (lhs.same_as(newLhs) && rhs.same_as(newRhs)) {
    return self;
  }

  return T1::make(newLhs, newRhs);
}

void RewriteSimplifierCCE::Impl::Update(const Var &var, const Expr &info, bool override) {
  if (!override) {
    auto it = var_map_.find(var);
    if (it != var_map_.end()) {
      CHECK(Equal(it->second, info)) << "Trying to update var \'" << var << "\'"
                                     << " with a different value: "
                                     << "original=" << it->second << ", new=" << info;
    }
  }
  var_map_[var] = info;
}

Expr RewriteSimplifierCCE::operator()(const Expr &expr) {
  impl_->SetIteratorsFromBounds(expr);
  // Do we need to apply this in a loop until no more changes?
  Expr res = expr;
  int max_iter = 2;
  for (int i = 0; i < max_iter; ++i) {
    Expr new_expr = impl_->Mutate(res);
    if (new_expr.same_as(res)) {
      return res;
    }
    res = new_expr;
  }
  return res;
}

Stmt RewriteSimplifierCCE::operator()(const Stmt &stmt) {
  Stmt res = stmt;
  int max_iter = 2;
  for (int i = 0; i < max_iter; ++i) {
    Stmt new_stmt = impl_->Mutate(res);
    if (new_stmt.same_as(res)) {
      return res;
    }
    res = new_stmt;
  }
  return res;
}

RewriteSimplifierCCE::RewriteSimplifierCCE(Analyzer *parent) : impl_(new Impl(parent)) {}

RewriteSimplifierCCE::~RewriteSimplifierCCE() { delete impl_; }
}  // namespace arith
}  // namespace akg
