/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "canonical_form.h"
#include <tvm/api_registry.h>

namespace akg {
namespace ir {
using std::get;
using std::to_string;

Expr CreateMonomialsExpr_(ktvm::DataType data_type, const set<Monomial> &monomials, int &sign) {
  if (monomials.empty()) {
    sign = 0;
    return make_const(data_type, 0);
  }

  Expr sumexpr;
  auto begin = monomials.begin();
  sign = begin->GetSign();
  for (const auto &item : monomials) {
    int is_negative = item.GetSign();
    sign = ((sign == is_negative) ? sign : 1);

    if (sumexpr.defined()) {
      if (is_negative) {
        sumexpr = Sub::make(sumexpr, item.ToExpr(data_type, true));
      } else {
        sumexpr = Add::make(sumexpr, item.ToExpr(data_type, false));
      }
    } else {
      sumexpr = item.ToExpr(data_type, false);
    }
  }
  return sumexpr;
}

Monomial Monomial::Add(const Monomial &monomial) const {
  auto a = numerator_;
  auto b = denominator_;
  auto c = monomial.numerator_;
  auto d = monomial.denominator_;

  numerator_ = a * d + b * c;
  auto gcd1 = ktvm::ir::gcd(numerator_, b);
  CHECK_NE(gcd1, 0);
  numerator_ = numerator_ / gcd1;
  denominator_ = b / gcd1;

  auto gcd2 = ktvm::ir::gcd(numerator_, d);
  CHECK_NE(gcd2, 0);
  numerator_ = numerator_ / gcd2;
  denominator_ = d / gcd2 * denominator_;

  auto gcd = ktvm::ir::gcd(numerator_, denominator_);
  CHECK_NE(gcd, 0);
  numerator_ = numerator_ / gcd;
  denominator_ = denominator_ / gcd;

  return *this;
}

Monomial Monomial::Sub(const Monomial &monomial) const {
  auto a = numerator_;
  auto b = denominator_;
  auto c = monomial.numerator_;
  auto d = monomial.denominator_;

  if (d < 0) {
    d *= -1;
  } else {
    c *= -1;
  }

  numerator_ = a * d + b * c;
  auto gcd1 = ktvm::ir::gcd(numerator_, b);
  CHECK_NE(gcd1, 0);
  numerator_ = numerator_ / gcd1;
  denominator_ = b / gcd1;

  auto gcd2 = ktvm::ir::gcd(numerator_, d);
  CHECK_NE(gcd2, 0);
  numerator_ = numerator_ / gcd2;
  denominator_ = d / gcd2 * denominator_;

  auto gcd = ktvm::ir::gcd(numerator_, denominator_);
  CHECK_NE(gcd, 0);
  numerator_ = numerator_ / gcd;
  denominator_ = denominator_ / gcd;

  return *this;
}

Monomial &Monomial::Mul(const Monomial &monomial) {
  auto gcd1 = ktvm::ir::gcd(numerator_, monomial.denominator_);
  auto gcd2 = ktvm::ir::gcd(denominator_, monomial.numerator_);
  CHECK_NE(gcd1, 0);
  CHECK_NE(gcd2, 0);
  numerator_ = (monomial.numerator_ / gcd2) * (numerator_ / gcd1);
  denominator_ = (monomial.denominator_ / gcd1) * (denominator_ / gcd2);

  auto gcd = ktvm::ir::gcd(numerator_, denominator_);
  CHECK_NE(gcd, 0);
  numerator_ /= gcd;
  denominator_ /= gcd;

  for (const auto &item : monomial.degree_) {
    if (degree_.count(item.first)) {
      degree_[item.first] += item.second;
    } else {
      degree_.emplace(item);
    }
  }

  return *this;
}

Monomial &Monomial::Divide(const Monomial &monomial) {
  auto gcd1 = ktvm::ir::gcd(numerator_, monomial.numerator_);
  auto gcd2 = ktvm::ir::gcd(denominator_, monomial.denominator_);
  CHECK_NE(gcd1, 0);
  CHECK_NE(gcd2, 0);
  numerator_ = (monomial.denominator_ / gcd2) * (numerator_ / gcd1);
  denominator_ = (monomial.numerator_ / gcd1) * (denominator_ / gcd2);

  auto gcd = ktvm::ir::gcd(numerator_, denominator_);
  CHECK_NE(gcd, 0);
  numerator_ /= gcd;
  denominator_ /= gcd;

  for (const auto &item : monomial.degree_) {
    if (degree_.count(item.first)) {
      degree_[item.first] -= item.second;
      if (degree_[item.first] == 0) {
        degree_.erase(item.first);
      }
    } else {
      degree_.emplace(item.first, -1 * item.second);
    }
  }

  return *this;
}

Monomial Monomial::Divisible(const Monomial &monomial) const {
  Monomial div;
  div.numerator_ = 0;
  if (degree_.empty() && monomial.degree_.empty()) {
    CHECK_NE(monomial.numerator_, 0) << "cannot divide by zero!";
    div.numerator_ = numerator_ / monomial.numerator_;
    CHECK_NE(denominator_, 0);
    div.denominator_ = monomial.denominator_ / denominator_;

    return div;
  }

  if (numerator_ % monomial.numerator_ != 0) {
    return div;
  }

  for (const auto &item : monomial.degree_) {
    auto it = degree_.find(item.first);
    if (it == degree_.end()) {
      return div;
    } else if (it->second < item.second) {
      return div;
    }
  }
  div = *this;

  return div.Divide(monomial);
}

Expr Monomial::ToExpr(ktvm::DataType data_type, bool is_negative) const {
  if (numerator_ == 0) {
    return make_const(data_type, 0);
  }

  if (numerator_ < 0 && denominator_ < 0) {
    numerator_ *= -1;
    denominator_ *= -1;
  }

  Expr div_a;
  Expr div_b;
  int64_t a = numerator_;
  int64_t b = denominator_;
  if (is_negative) {
    if (denominator_ < 0)
      b *= -1;
    else
      a *= -1;
  }

  for (const auto &item : degree_) {
    Expr item_expr = item.first;

    if (std::abs(item.second) == 0) {
      item_expr = make_const(data_type, 1);
    } else {
      for (int j = 1; j < std::abs(item.second); ++j) {
        item_expr = Mul::make(item_expr, item_expr);
      }
    }

    if (item.second > 0) {
      div_a = div_a.defined() ? Mul::make(div_a, item_expr) : item_expr;
    } else {
      div_b = div_b.defined() ? Mul::make(div_b, item_expr) : item_expr;
    }
  }

  if (div_a.defined()) {
    if (a != 1) {
      div_a = Mul::make(make_const(data_type, a), div_a);
    }
  } else {
    div_a = make_const(data_type, a);
  }

  if (div_b.defined()) {
    if (b != 1) {
      div_b = Mul::make(make_const(data_type, b), div_b);
    }
    return Div::make(div_a, div_b);
  } else if (denominator_ != 1) {
    return Div::make(div_a, make_const(data_type, b));
  }

  return div_a;
}

bool Monomial::operator==(const Monomial &monomial) const {
  if ((numerator_ != monomial.numerator_) || (denominator_ != monomial.denominator_) ||
      (degree_.size() != monomial.degree_.size())) {
    return false;
  }

  auto it_a = degree_.begin();
  auto it_b = monomial.degree_.begin();
  while (it_a != degree_.end()) {
    if (it_a->first.get() != it_b->first.get()) {
      return false;
    }

    it_a++;
    it_b++;
  }

  return true;
}

bool Monomial::operator<(const Monomial &monomial) const {
  if (degree_.empty()) {
    return false;
  } else if (monomial.degree_.empty()) {
    return true;
  }

  auto it_a = degree_.begin();
  auto a_end = degree_.end();
  auto it_b = monomial.degree_.begin();
  auto b_end = monomial.degree_.end();
  while ((it_a != a_end) && (it_b != b_end)) {
    int cmp = it_a->first->name_hint.compare(it_b->first->name_hint);
    if (cmp == 0) {
      if (it_a->second > it_b->second) {
        return true;
      } else if (it_a->second < it_b->second) {
        return false;
      }
    } else if (cmp < 0) {
      return true;
    } else if (cmp > 0) {
      return false;
    }

    it_a++;
    it_b++;
  }

  if (it_a != a_end) {
    return true;
  }

  if (it_b != b_end) {
    return false;
  }

  return false;
}

pair<set<Monomial>, set<Monomial>> CanonicalForm::ComputeQuotientAndRemainder(const set<Monomial> &a,
                                                                              const set<Monomial> &b) {
  CHECK(!b.empty()) << "cannot div by zero!";
  if (a.empty()) {
    return pair<set<Monomial>, set<Monomial>>(set<Monomial>(), set<Monomial>());
  }

  set<Monomial> q;
  set<Monomial> r = a;
  auto b_first = *b.begin();
  auto div = r.begin()->Divisible(b_first);
  while (div.numerator_ != 0) {
    q.emplace(div);
    auto mul = Multiply(b, set<Monomial>{div});
    r = Subtract(r, mul);
    if (r.empty()) {
      break;
    }

    div = r.begin()->Divisible(b_first);
  }

  return pair<set<Monomial>, set<Monomial>>(q, r);
}

set<Monomial> CanonicalForm::Gcd(const set<Monomial> &a, const set<Monomial> &b) {
  if (a.empty()) {
    return b;
  } else if (b.empty()) {
    return a;
  }

  auto m = a;
  auto n = b;
  pair<set<Monomial>, set<Monomial>> q_r;
  bool is_same = false;

  do {
    q_r = ComputeQuotientAndRemainder(m, n);
    if (is_same && q_r.first.empty()) {
      return set<Monomial>{Monomial()};
    }

    is_same = q_r.first.empty();
    m = n;
    n = q_r.second;
  } while (!q_r.second.empty());

  return m;
}

// A/B*B + A%B = A
set<Monomial> CanonicalForm::DivAndModFilter(set<Monomial> &monomials, vector<pair<Var, Var>> div_mod_pair,
                                             VarReplaceMap var_child) {
  for (auto &pair : div_mod_pair) {
    set<Monomial> div_coeff;
    set<Monomial> mod_coeff;
    set<Monomial> offset;
    for (auto item : monomials) {
      if (item.degree_.count(pair.first)) {
        auto coeff = item;
        if (item.degree_[pair.first] == 1) {
          coeff.degree_.erase(pair.first);
        } else if (item.degree_[pair.first] > 0) {
          item.degree_[pair.first] -= 1;
        } else {
          continue;
        }
        div_coeff.emplace(coeff);
      } else if (item.degree_.count(pair.second)) {
        auto coeff = item;
        if (item.degree_[pair.second] == 1) {
          coeff.degree_.erase(pair.second);
        } else if (item.degree_[pair.second] > 0) {
          item.degree_[pair.second] -= 1;
        } else {
          continue;
        }
        mod_coeff.emplace(coeff);
      } else {
        offset.emplace(item);
      }
    }

    auto q_r = ComputeQuotientAndRemainder(div_coeff, get<1>(var_child[pair.first]));
    // do not surpport "cancel match A/(B + C) * B + A%(B + C)"
    if (q_r.first == mod_coeff) {
      Monomial var;
      var.degree_.emplace(pair.first, 1);
      auto r = Multiply(q_r.second, set<Monomial>{var});
      auto new_items = Multiply(q_r.first, get<0>(var_child[pair.first]));
      monomials = Addition(r, new_items);
      monomials = Addition(monomials, offset);
    }
  }

  return monomials;
}

set<Monomial> CanonicalForm::ExprNormalForm(const ktvm::Expr &e) {
  auto ret = VisitExpr(e, e);
  return ret;
}

set<Monomial> CanonicalForm::Addition(const set<Monomial> &a, const set<Monomial> &b) {
  set<Monomial> ret = a;
  for (auto term_b : b) {
    auto it = ret.find(term_b);
    if (it != ret.end()) {
      static_cast<void>(it->Add(term_b));
      if (it->numerator_ == 0) {
        ret.erase(it);
      }
    } else {
      ret.emplace(term_b);
    }
  }

  return ret;
}

set<Monomial> CanonicalForm::Subtract(const set<Monomial> &a, const set<Monomial> &b) {
  set<Monomial> ret = a;
  for (auto term_b : b) {
    auto it = ret.find(term_b);
    if (it != ret.end()) {
      static_cast<void>(it->Sub(term_b));
      if (it->numerator_ == 0) {
        ret.erase(it);
      }
    } else {
      term_b.numerator_ *= -1;
      ret.emplace(term_b);
    }
  }

  return ret;
}

set<Monomial> CanonicalForm::Multiply(const set<Monomial> &a, const set<Monomial> &b) {
  set<Monomial> ret;
  for (const auto &item_a : a) {
    for (auto item_b : b) {
      static_cast<void>(item_b.Mul(item_a));
      auto it = ret.find(item_b);
      if (it != ret.end()) {
        static_cast<void>(it->Add(item_b));
        if (it->numerator_ == 0) {
          ret.erase(it);
        }
      } else {
        ret.emplace(item_b);
      }
    }
  }

  return ret;
}

set<Monomial> CanonicalForm::Divide(const set<Monomial> &a, const set<Monomial> &b) {
  set<Monomial> ret;
  for (const auto &item_b : b) {
    for (auto item_a : a) {
      static_cast<void>(item_a.Divide(item_b));
      ret.emplace(item_a);
    }
  }

  return ret;
}

void CanonicalForm::DivSimplify(set<Monomial> &a, set<Monomial> &b) {
  if (a.size() == 1 && a.begin()->IsConst() && b.size() == 1 && b.begin()->IsConst()) {
    auto q_r = ComputeQuotientAndRemainder(a, b);
    a = q_r.first;
    b = set<Monomial>{Monomial()};

    return;
  }

  set<Monomial> gcd = Gcd(a, b);
  if (gcd.size() == 1 && gcd.begin()->IsOne()) {
    return;
  }

  auto pair_a = ComputeQuotientAndRemainder(a, gcd);
  auto pair_b = ComputeQuotientAndRemainder(b, gcd);
  if (pair_a.second.empty() && pair_b.second.empty()) {
    a = pair_a.first;
    b = pair_b.first;
  }
}

set<Monomial> CanonicalForm::VisitExpr_(const Add *op, const ktvm::Expr &e) {
  auto ret_a = VisitExpr(op->a, op->a);
  auto ret_b = VisitExpr(op->b, op->b);

  for (auto &term_b : ret_b) {
    auto it = ret_a.find(term_b);
    if (it != ret_a.end()) {
      static_cast<void>(it->Add(term_b));
      if (it->numerator_ == 0) {
        ret_a.erase(it);
      }
    } else {
      ret_a.emplace(term_b);
    }
  }

  return ret_a;
}

set<Monomial> CanonicalForm::VisitExpr_(const Sub *op, const ktvm::Expr &e) {
  auto ret_a = VisitExpr(op->a, op->a);
  auto ret_b = VisitExpr(op->b, op->b);

  for (auto &term_b : ret_b) {
    auto it = ret_a.find(term_b);
    if (it != ret_a.end()) {
      static_cast<void>(it->Sub(term_b));
      if (it->numerator_ == 0) {
        ret_a.erase(it);
      }
    } else {
      term_b.numerator_ *= -1;
      ret_a.emplace(term_b);
    }
  }

  return ret_a;
}

set<Monomial> CanonicalForm::VisitExpr_(const Mul *op, const ktvm::Expr &e) {
  auto ret_a = VisitExpr(op->a, op->a);
  auto ret_b = VisitExpr(op->b, op->b);

  return Multiply(ret_a, ret_b);
}

set<Monomial> CanonicalForm::VisitExpr_(const Div *op, const ktvm::Expr &e) {
  auto ret_a = VisitExpr(op->a, op->a);
  auto ret_b = VisitExpr(op->b, op->b);

  return Divide(ret_a, ret_b);
}

set<Monomial> CanonicalForm::VisitExpr_(const FloorDiv *op, const ktvm::Expr &e) {
  auto expr = Div::make(op->a, op->b);

  return VisitExpr(expr, expr);
}

set<Monomial> CanonicalForm::VisitExpr_(const Variable *op, const ktvm::Expr &e) {
  Monomial term;
  term.degree_.emplace(Downcast<Var>(e), 1);

  return set<Monomial>{term};
}

set<Monomial> CanonicalForm::VisitExpr_(const IntImm *op, const ktvm::Expr &e) {
  if (op->value == 0) {
    return set<Monomial>();
  }

  Monomial term;
  term.numerator_ = op->value;

  return set<Monomial>{term};
}

set<Monomial> CanonicalForm::VisitExpr_(const UIntImm *op, const ktvm::Expr &e) {
  if (op->value == 0) {
    return set<Monomial>();
  }

  Monomial term;
  term.numerator_ = op->value;

  return set<Monomial>{term};
}

Expr CanonicalForm::CreateMonomialsExpr(const set<Monomial> &monomials, int &sign) {
  return CreateMonomialsExpr_(data_type_, monomials, sign);
}

Expr CanonicalForm::CreateMonomialsExpr(const set<Monomial> &monomials) {
  int sign;
  return CreateMonomialsExpr_(data_type_, monomials, sign);
}
}  // namespace ir
}  // namespace akg
