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
#ifndef PASS_REWRITE_SIMPLIFY_CCE_H_
#define PASS_REWRITE_SIMPLIFY_CCE_H_
#define CHECK_IMPL_CONDITION false

#include <arithmetic/int_set.h>
#include <tvm/ir.h>
#include <tvm/arithmetic.h>
#include <tvm/expr_operator.h>
#include <tvm/ir_mutator.h>
#include <unordered_set>
#include <unordered_map>
#include "pass/utils.h"
#include "../src/arithmetic/const_fold.h"
#include "../src/arithmetic/pattern_match.h"

namespace akg {
namespace arith {
/*!
 * \brief Rewrite-rule based simplifier.
 */
class RewriteSimplifierCCE {
 public:
  /*!
   * \brief rewrites the expression in a simplified form
   * \param expr The expression of interest.
   * \return the result of the simplifciation.
   */
  Expr operator()(const Expr &expr);
  /*!
   * \brief rewrites the statement in a simplified form
   * \param expr The statement of interest.
   * \return the result of the simplifciation.
   */
  Stmt operator()(const Stmt &stmt);
  /*!
   * \brief Update binding of var to a new expression.
   *
   * \param var The variable of interest.
   * \param new_expr
   * \param override Whether do we allow override of existing information.
   */
  void Update(const Var &var, const Expr &new_expr, bool override = false);

  explicit RewriteSimplifierCCE(air::arith::Analyzer *parent);
  ~RewriteSimplifierCCE();

 private:
  friend class air::arith::Analyzer;
  friend class ConstraintContext;
  friend class CanonicalSimplifier;
  class Impl;
  /*! \brief Internal impl */
  Impl *impl_;
};

/*!
 * \brief Rewrite-based simplifier.
 *
 * This class can be inheritated for other simplifiers.
 */
class RewriteSimplifierCCE::Impl : public IRMutator {
 public:
  explicit Impl(air::arith::Analyzer *parent) : parent_(parent) {}
  ~Impl() override = default;

  void SetIteratorsFromBounds(const Expr &e);

  void Update(const Var &var, const Expr &info, bool override);
  Expr Mutate_(const Add *op, const Expr &self) override;
  Expr Mutate_(const And *op, const Expr &self) override;
  Expr Mutate_(const Or *op, const Expr &self) override;
  Expr Mutate_(const LT *op, const Expr &self) override;
  Expr Mutate_(const LE *op, const Expr &self) override;
  Expr Mutate_(const GT *op, const Expr &self) override;
  Expr Mutate_(const GE *op, const Expr &self) override;
  Expr Mutate_(const EQ *op, const Expr &self) override;
  Expr Mutate_(const Sub *op, const Expr &self) override;
  Expr Mutate_(const Mul *op, const Expr &self) override;
  Expr Mutate_(const Mod *op, const Expr &self) override;
  Expr Mutate_(const Max *op, const Expr &self) override;
  Expr Mutate_(const Min *op, const Expr &self) override;
  Expr Mutate_(const Select *op, const Expr &self) override;
  Expr Mutate_(const Div *op, const Expr &self) override;
  Expr Mutate_(const NE *op, const Expr &self) override;
  Stmt Mutate_(const For *op, const Stmt &self) override;
  /*
  Expr Mutate_(const Not* op, const Expr& self) override;
  Expr Mutate_(const Call* op, const Expr& self) override;
  Expr Mutate_(const Cast* op, const Expr& self) override;
  */

 protected:
  /*! \brief internal structure for comparison. */
  enum CompareResult { kUnknown, kEQ, kGT, kGE, kLT, kLE, kNE };
  // reference to the main analyzer
  air::arith::Analyzer *parent_;
  // counter to record recursive rewrite depth.
  int recur_depth_{0};
  // internal variable map
  std::unordered_map<Var, Expr, ExprHash, ExprEqual> var_map_;
  // maximum number of recursion allowed during a single pass.
  static const constexpr int kMaxRecurDepth = 5;

  /*!
   * \brief try to compare x against val.
   * \param x The expression to be evaluated.
   * \param val The constant value.
   * \return comparison result.
   */
  CompareResult TryCompare(const Expr &x, int64_t val);
  bool IsConditionTrue(Expr expr) const;
  std::unordered_set<const Variable *> GetExprVars(const Expr &expr) const;
  Expr ReduceCondition(Expr cond);
  template <typename T1>
  Expr RemoveSelectFromCond(const Expr &lhs, const Expr &rhs, const Expr &self);
  bool VarIntervalIsUseful(const Expr &e, air::arith::IntervalSet &interval);
  bool VarIntervalIsUseful(const Expr &e, air::arith::IntervalSet &interval_e, const Expr &e2,
                           air::arith::IntervalSet &interval_e2);
  air::arith::IntervalSet BoundToIntervalSet(const air::arith::ConstIntBound &bound);

 private:
  bool in_for_bound_{false};
  std::unordered_map<const Variable *, air::arith::IntSet> iteration_vars_;

  // Whether x is true
  bool CanProve(const Expr &x) { return parent_->CanProve(x); }
  // Whether x >= val
  bool CanProveGreaterEqual(const Expr &x, int64_t val) { return parent_->CanProveGreaterEqual(x, val); }
  // Whether x == val
  bool CanProveEqual(const Expr &x, int64_t val) { return TryCompare(x, val) == kEQ; }

  // Recursive rewrite x
  // we limit maximum depth of recursive rewrite allowed to
  // avoid infinite loop
  Expr RecursiveRewrite(const Expr &x) {
    if (recur_depth_ >= kMaxRecurDepth) return x;
    ++recur_depth_;
    Expr res = Mutate(x);
    --recur_depth_;
    return res;
  }

  template <typename TA>
  air::arith::PConstWithTypeLike<TA> ZeroWithTypeLike(const air::arith::Pattern<TA> &pattern) {
    return air::arith::PConstWithTypeLike<TA>(pattern.derived(), 0);
  }

  template <typename TA>
  air::arith::PConstWithTypeLike<TA> OneWithTypeLike(const air::arith::Pattern<TA> &pattern) {
    return air::arith::PConstWithTypeLike<TA>(pattern.derived(), 1);
  }
};
}  // namespace arith
}  // namespace akg

#endif  // PASS_REWRITE_SIMPLIFY_CCE_H_
