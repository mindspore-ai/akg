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

#ifndef AUTODIFF_ARITH_EXPR_SIMPLIFY_H
#define AUTODIFF_ARITH_EXPR_SIMPLIFY_H

#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include <ir_pass.h>

#include "pass/utils.h"
#include "canonical_form.h"

namespace akg {
namespace ir {
class ArithExprSimplifier {
 public:
  explicit ArithExprSimplifier(ktvm::DataType data_type = Int(32)) : data_type_(data_type) {}

  Expr Simplify(const Expr &e);

  Expr Simplify(const Expr &e, const vector<pair<Var, Var>> &div_mod_pair, const UnorderedVarMap &div_child,
                const UnorderedVarMap &floordiv_child);

  Expr ReduceInequality(const Expr &e, const Var &reduce_var);

  Expr ReduceInequality(const Expr &e, const vector<Var> &vars);

  Expr ReduceInequality(const Expr &e, const Var &reduce_var, int lcm);

  bool IsMonotonic(const Expr &e, const Var &var);

  int64_t GetSup(const Expr &e);

  bool Equals(const Expr &e1, const Expr &e2);

  bool IsDivisible(const Expr &e, const Expr &divisor);

  Expr Gcd(const Expr &e1, const Expr &e2);

  bool IsZeroExpr(const Expr &e);

  Expr ModSimplify(Expr &a, Expr &b);

  Expr DivSimplify(Expr &a, Expr &b);

  Array<Expr> GetPolynomial(const Expr &e1, const Expr &e2);

  Expr ScaleSubstitute(const Expr &e, const unordered_map<Var, vector<Expr>, NodeHash, NodeEqual> &substitute_map,
                       bool is_less, bool is_larger);

  int RangeWithPosvar(const Expr &e);

  ~ArithExprSimplifier() = default;

 private:
  template <typename R>
  Expr ReducedInequality(const Expr &e, const Var &reduce_var);

  Expr MoveToOneside(const Expr &e);

  bool CollectCoeffandOffsets(const set<Monomial> &norm_form, map<int, set<Monomial>> &coeffs, const Var &reduce_var);

  template <typename R>
  inline Expr CorrectFloorDiv(const Expr &coeff, int sign) const;

  inline int GetExprSign(const Expr &expr) const;

  vector<tuple<int, int, int>> CountSort(vector<int> &v);

 private:
  ktvm::DataType data_type_;
};
}  // namespace ir
}  // namespace akg

#endif  // AUTODIFF_ARITH_EXPR_SIMPLIFY_H
