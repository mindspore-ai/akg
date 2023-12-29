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

#ifndef AUTODIFF_CANONICAL_FORM_H
#define AUTODIFF_CANONICAL_FORM_H

#include <tvm/ir.h>
#include <tvm/ir_functor_ext.h>
#include <tvm.h>
#include <src/pass/ir_util.h>

namespace akg {
namespace ir {
using std::map;
using std::pair;
using std::set;
using std::string;
using std::tuple;
using std::unordered_map;
using std::vector;
static constexpr int INT32 = 32;

struct VarCompare {
  bool operator()(const Var &a, const Var &b) const { return a->name_hint.compare(b->name_hint) < 0; }
};

class Monomial;
using VarMap = map<const Var, int, VarCompare>;
using VarReplaceMap = unordered_map<Var, tuple<set<Monomial>, set<Monomial>, Expr>, air::NodeHash, air::NodeEqual>;
using UnorderedVarMap = unordered_map<Var, vector<Expr>, air::NodeHash, air::NodeEqual>;

class Monomial {
 public:
  VarMap degree_;
  mutable int64_t numerator_ = 1;
  mutable int64_t denominator_ = 1;

  int GetSign() const { return (-(numerator_ < 0)) ^ (-(denominator_ < 0)); }

  bool IsOne() const { return abs(numerator_) == 1 && abs(denominator_) == 1 && degree_.empty(); }

  bool IsConst() const { return degree_.empty(); }

  Monomial Add(const Monomial &monomial) const;

  Monomial Sub(const Monomial &monomial) const;

  Monomial &Mul(const Monomial &monomial);

  Monomial &Divide(const Monomial &monomial);

  Monomial Divisible(const Monomial &monomial) const;

  Expr ToExpr(const air::DataType data_type = Int(INT32), const bool is_negative = false) const;

  bool operator<(const Monomial &monomial) const;
  bool operator==(const Monomial &monomial) const;
};

class CanonicalForm : public air::ir::ExprFunctor<set<Monomial>(const air::Expr &n, const air::Expr &e)> {
 public:
  explicit CanonicalForm(air::DataType data_type = Int(INT32)) : data_type_(data_type) {}

  ~CanonicalForm() override = default;

  set<Monomial> ExprNormalForm(const air::Expr &e);

  set<Monomial> Gcd(const set<Monomial> &a, const set<Monomial> &b);

  set<Monomial> Addition(const set<Monomial> &a, const set<Monomial> &b);

  set<Monomial> Subtract(const set<Monomial> &a, const set<Monomial> &b);

  set<Monomial> Multiply(const set<Monomial> &a, const set<Monomial> &b);

  set<Monomial> Divide(const set<Monomial> &a, const set<Monomial> &b);

  void DivSimplify(set<Monomial> &a, set<Monomial> &b);

  Expr CreateMonomialsExpr(const set<Monomial> &monomials);

  Expr CreateMonomialsExpr(const set<Monomial> &monomials, int &sign);

  set<Monomial> DivAndModFilter(set<Monomial> &monomials, vector<pair<Var, Var>> div_mod_pair, VarReplaceMap var_child);

 private:
  pair<set<Monomial>, set<Monomial>> ComputeQuotientAndRemainder(const set<Monomial> &a, const set<Monomial> &b);

  set<Monomial> VisitExpr_(const Add *op, const air::Expr &e) final;

  set<Monomial> VisitExpr_(const Sub *op, const air::Expr &e) final;

  set<Monomial> VisitExpr_(const Mul *op, const air::Expr &e) final;

  set<Monomial> VisitExpr_(const Div *op, const air::Expr &e) final;

  set<Monomial> VisitExpr_(const FloorDiv *op, const air::Expr &e) final;

  set<Monomial> VisitExpr_(const Variable *op, const air::Expr &e) final;

  set<Monomial> VisitExpr_(const IntImm *op, const air::Expr &e) final;

  set<Monomial> VisitExpr_(const UIntImm *op, const air::Expr &e) final;

  air::DataType data_type_ = Int(INT32);
};
}  // namespace ir
}  // namespace akg

#endif  // AUTODIFF_CANONICAL_FORM_H
