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

#include <tvm/api_registry.h>
#include "arith_expr_simplify.h"

namespace akg {
namespace ir {
using std::get;

bool ArithExprSimplifier::Equals(const Expr &e1, const Expr &e2) {
  if (e1.type() != e2.type()) {
    return false;
  }
  const auto &e = e1 - e2;
  if (auto imm = e.as<IntImm>()) {
    return imm->value == 0;
  }
  return CanonicalForm(data_type_).ExprNormalForm(e).empty();
}

bool ArithExprSimplifier::IsDivisible(const Expr &e, const Expr &divisor) {
  if (!divisor.as<IntImm>()) {
    LOG(FATAL) << "denominator should be integer.";
    return false;
  }
  CanonicalForm form(data_type_);
  set<Monomial> norm_form = form.ExprNormalForm(e);
  const int64_t value = divisor.as<IntImm>()->value;
  for (const auto &item : norm_form) {
    if (item.numerator_ % value || item.denominator_ != 1) {
      return false;
    }
  }
  return true;
}

bool ArithExprSimplifier::CollectCoeffandOffsets(const set<Monomial> &norm_form, map<int, set<Monomial>> &coeffs,
                                                 const Var &reduce_var) {
  for (auto item : norm_form) {
    int degree = 0;
    auto it = item.degree_.find(reduce_var);
    if (it != item.degree_.end()) {
      degree = it->second;
      item.degree_.erase(it);
    }
    if (degree == 0) {
      item.numerator_ *= -1;
    }
    if (coeffs.count(degree)) {
      coeffs[degree].emplace(item);
    } else {
      coeffs.emplace(degree, set<Monomial>{item});
    }
  }
  return true;
}

template <typename R>
inline Expr ArithExprSimplifier::CorrectFloorDiv(const Expr &coeff, const int sign) const {
  if (std::is_same<R, LT>::value || std::is_same<R, GE>::value) {
    if (sign == -1) {
      return Sub::make(make_const(data_type_, -1), coeff);
    } else {
      return Sub::make(coeff, make_const(data_type_, 1));
    }
  }
  return Expr(make_const(data_type_, 0));
}

inline int ArithExprSimplifier::GetExprSign(const Expr &expr) const {
  if (!expr.defined()) {
    return 1;
  }
  CanonicalForm form(data_type_);
  set<Monomial> norm_form = form.ExprNormalForm(expr);
  if (norm_form.empty()) {
    return 1;
  }
  int sign = norm_form.begin()->GetSign();
  for (const auto &it : norm_form) {
    int is_negative = it.GetSign();
    sign = (sign == is_negative) ? sign : 1;
  }
  return sign;
}

Expr ArithExprSimplifier::ScaleSubstitute(const Expr &e,
                                          const unordered_map<Var, vector<Expr>, NodeHash, NodeEqual> &substitute_map,
                                          bool is_less, bool is_larger) {
  // substitute div-var expr with proper real-div expr.
  // is_less: true when e is LE and LT expr
  // is_larger: true when expecting get larger approximation.
  if (!e.defined() || substitute_map.empty()) {
    return e;
  }

  Expr expr = e;
  if (e.as<LE>() || e.as<LT>() || e.as<GT>() || e.as<GE>()) {
    expr = MoveToOneside(e);
  }

  CanonicalForm form(data_type_);
  set<Monomial> norm_form = form.ExprNormalForm(expr);
  Expr sumexpr;
  std::unordered_map<const Variable *, Expr> substitute_tbl;
  for (const auto &item : norm_form) {
    bool is_neg = item.GetSign();
    auto item_expr = sumexpr.defined() ? item.ToExpr(data_type_, is_neg) : item.ToExpr(data_type_, false);
    substitute_tbl.clear();
    for (auto v : substitute_map) {
      if (IsVarInExpr(v.first, item_expr)) {
        if (!is_neg) {
          if (is_larger == is_less) {
            substitute_tbl.emplace(v.first.get(), v.second[0]);
          } else {
            substitute_tbl.emplace(v.first.get(), v.second[1]);
          }
        } else {
          if (is_larger == is_less) {
            substitute_tbl.emplace(v.first.get(), v.second[1]);
          } else {
            substitute_tbl.emplace(v.first.get(), v.second[0]);
          }
        }
      }
    }
    item_expr = Substitute(item_expr, substitute_tbl);
    if (is_neg) {
      sumexpr = sumexpr.defined() ? Sub::make(sumexpr, item_expr) : item_expr;
    } else {
      sumexpr = sumexpr.defined() ? Add::make(sumexpr, item_expr) : item_expr;
    }
  }
  return sumexpr;
}

template <typename R>
Expr ArithExprSimplifier::ReducedInequality(const Expr &e, const Var &reduce_var) {
  if (e.as<Variable>()) {
    return R::make(e, 0);
  }
  map<int, set<Monomial>> reduce_coeff;
  CanonicalForm form(data_type_);
  set<Monomial> norm_form = form.ExprNormalForm(e);
  if (!CollectCoeffandOffsets(norm_form, reduce_coeff, reduce_var)) {
    return e;
  }

  Expr lhs_expr;
  Expr rhs_expr;
  bool is_negative = false;

  Expr reduce_offset;
  CHECK(!reduce_coeff.empty());
  auto it = reduce_coeff.begin();
  if (it->first == 0) {
    rhs_expr = form.CreateMonomialsExpr(it->second);
    it++;
  }

  if (distance(it, reduce_coeff.end()) == 1) {
    Expr expr = reduce_var;
    for (int i = 1; i < it->first; i++) {
      expr = Mul::make(expr, reduce_var);
    }
    if (!lhs_expr.defined()) {
      lhs_expr = expr;
    }
    int sign = 0;
    auto coeff_expr = form.CreateMonomialsExpr(it->second, sign);
    if (sign != 1) {
      Expr correct_offset = CorrectFloorDiv<R>(coeff_expr, sign);
      Expr a = rhs_expr.defined() ? Add::make(rhs_expr, correct_offset) : correct_offset;
      Expr b = coeff_expr;
      auto simplified_expr = DivSimplify(a, b);
      rhs_expr = simplified_expr.defined() ? simplified_expr : Div::make(a, b);
      is_negative = sign == -1;
    }
  } else {
    for (; it != reduce_coeff.end(); it++) {
      Expr expr = reduce_var;
      for (int i = 1; i < it->first; i++) {
        expr = Mul::make(expr, reduce_var);
      }
      int sign = 0;
      auto coeff_expr = form.CreateMonomialsExpr(it->second, sign);
      auto item_expr = Mul::make(expr, coeff_expr);
      if (lhs_expr.defined()) {
        lhs_expr = Add::make(lhs_expr, item_expr);
      } else {
        lhs_expr = item_expr;
      }
    }
  }

  if (!rhs_expr.defined()) {
    rhs_expr = make_const(data_type_, 0);
  }
  if (is_negative) {
    if (std::is_same<R, GE>::value) {
      return LE::make(lhs_expr, rhs_expr);
    } else if (std::is_same<R, LE>::value) {
      return GE::make(lhs_expr, rhs_expr);
    } else if (std::is_same<R, GT>::value) {
      return LT::make(lhs_expr, rhs_expr);
    } else if (std::is_same<R, LT>::value) {
      return GT::make(lhs_expr, rhs_expr);
    }
  }
  return R::make(lhs_expr, rhs_expr);
}

Expr ArithExprSimplifier::ReduceInequality(const Expr &e, const Var &reduce_var) {
  Expr lhs, rhs;
  int sign = 0;
  if (auto le = e.as<LE>()) {
    if (auto div = le->a.as<Div>()) {
      bool varInB = IsVarInExpr(reduce_var, div->b);
      sign = GetExprSign(div->b);
      Expr offset = varInB ? Expr(make_const(data_type_, 0)) : div->b + make_const(data_type_, (sign == -1 ? 1 : -1));
      lhs = div->a;
      rhs = le->b * div->b + offset;
    } else if (auto floordiv = le->a.as<FloorDiv>()) {
      bool varInB = IsVarInExpr(reduce_var, floordiv->b);
      sign = GetExprSign(floordiv->b);
      Expr offset =
        varInB ? Expr(make_const(data_type_, 0)) : floordiv->b + make_const(data_type_, (sign == -1 ? 1 : -1));
      lhs = floordiv->a;
      rhs = le->b * floordiv->b + offset;
    }
    Expr linear_form = (lhs.defined() && rhs.defined()) ? lhs - rhs : le->a - le->b;
    if (sign == -1) {
      return ReducedInequality<GE>(linear_form, reduce_var);
    } else if (sign == 1) {
      return e;
    }
    return ReducedInequality<LE>(linear_form, reduce_var);
  } else if (auto lt = e.as<LT>()) {
    if (auto div = lt->a.as<Div>()) {
      sign = GetExprSign(div->b);
      lhs = div->a;
      rhs = lt->b * div->b;
    } else if (auto floordiv = lt->a.as<FloorDiv>()) {
      sign = GetExprSign(floordiv->b);
      lhs = floordiv->a;
      rhs = lt->b * floordiv->b;
    }
    Expr linear_form = (lhs.defined() && rhs.defined()) ? lhs - rhs : lt->a - lt->b;
    if (sign == -1) {
      return ReducedInequality<GT>(linear_form, reduce_var);
    } else if (sign == 1) {
      return e;
    }
    return ReducedInequality<LT>(linear_form, reduce_var);
  } else if (auto gt = e.as<GT>()) {
    if (auto div = gt->a.as<Div>()) {
      bool varInB = IsVarInExpr(reduce_var, div->b);
      sign = GetExprSign(div->b);
      Expr offset = varInB ? Expr(make_const(data_type_, 0)) : div->b + make_const(data_type_, (sign == -1 ? 1 : -1));
      lhs = div->a;
      rhs = gt->b * div->b + offset;
    } else if (auto floordiv = gt->a.as<FloorDiv>()) {
      bool varInB = IsVarInExpr(reduce_var, floordiv->b);
      sign = GetExprSign(floordiv->b);
      Expr offset =
        varInB ? Expr(make_const(data_type_, 0)) : floordiv->b + make_const(data_type_, (sign == -1 ? 1 : -1));
      lhs = floordiv->a;
      rhs = gt->b * floordiv->b + offset;
    }
    Expr linear_form = (lhs.defined() && rhs.defined()) ? lhs - rhs : gt->a - gt->b;
    if (sign == -1) {
      return ReducedInequality<LT>(linear_form, reduce_var);
    } else if (sign == 1) {
      return e;
    }
    return ReducedInequality<GT>(linear_form, reduce_var);
  } else if (auto ge = e.as<GE>()) {
    if (auto div = ge->a.as<Div>()) {
      sign = GetExprSign(div->b);
      lhs = div->a;
      rhs = ge->b * div->b;
    } else if (auto floordiv = ge->a.as<FloorDiv>()) {
      sign = GetExprSign(floordiv->b);
      lhs = floordiv->a;
      rhs = ge->b * floordiv->b;
    }
    Expr linear_form = (lhs.defined() && rhs.defined()) ? lhs - rhs : ge->a - ge->b;
    if (sign == -1) {
      return ReducedInequality<LE>(linear_form, reduce_var);
    } else if (sign == 1) {
      return e;
    }
    return ReducedInequality<GE>(linear_form, reduce_var);
  } else {
    LOG(FATAL) << "Only support to reduce LE, LT, GE, GT inequality. " << e;
    return Expr();
  }
}

Expr ArithExprSimplifier::MoveToOneside(const Expr &e) {
  if (e.as<LE>()) {
    return e.as<LE>()->a - e.as<LE>()->b;
  } else if (e.as<GE>()) {
    return e.as<GE>()->a - e.as<GE>()->b;
  } else if (e.as<LT>()) {
    return e.as<LT>()->a - e.as<LT>()->b;
  } else if (e.as<GT>()) {
    return e.as<GT>()->a - e.as<GT>()->b;
  } else {
    LOG(FATAL) << "Only support to reduce LE, LT, GE, GT inequality. " << e;
    return Expr();
  }
}

Expr ArithExprSimplifier::ReduceInequality(const Expr &e, const vector<Var> &reduce_vars) {
  if (reduce_vars.size() == 1) {
    return ReduceInequality(e, reduce_vars[0]);
  }

  Expr expr = MoveToOneside(e);
  CanonicalForm form(data_type_);
  set<Monomial> norm_form = form.ExprNormalForm(expr);
  Expr lhs;
  Expr rhs;
  for (auto &item : norm_form) {
    bool vars_in_item = false;
    for (auto &var : reduce_vars) {
      if (item.degree_.count(var)) {
        vars_in_item = true;
        break;
      }
    }
    bool is_negative = item.GetSign();
    auto item_expr = item.ToExpr(data_type_, is_negative);
    if (vars_in_item) {
      if (is_negative) {
        lhs = lhs.defined() ? Sub::make(lhs, item_expr) : item.ToExpr(data_type_, false);
      } else {
        lhs = lhs.defined() ? Add::make(lhs, item_expr) : item_expr;
      }
    } else {
      if (is_negative) {
        rhs = rhs.defined() ? Add::make(rhs, item_expr) : item_expr;
      } else {
        rhs = rhs.defined() ? Sub::make(rhs, item_expr) : item.ToExpr(data_type_, true);
      }
    }
  }
  if (!lhs.defined()) {
    lhs = make_const(data_type_, 0);
  }
  if (!rhs.defined()) {
    rhs = make_const(data_type_, 0);
  }

  if (e.as<LE>()) {
    return LE::make(lhs, rhs);
  } else if (e.as<GE>()) {
    return GE::make(lhs, rhs);
  } else if (e.as<LT>()) {
    return LT::make(lhs, rhs);
  } else if (e.as<GT>()) {
    return GT::make(lhs, rhs);
  }
  return Expr();
}

Expr ArithExprSimplifier::ReduceInequality(const Expr &e, const Var &reduce_var, int lcm) {
  Expr expr;
  expr = MoveToOneside(e);
  Expr reduced_expr = Mul::make(expr, lcm);
  CanonicalForm form(data_type_);
  set<Monomial> norm_form = form.ExprNormalForm(reduced_expr);
  Expr lhs;
  for (auto &item : norm_form) {
    if (item.denominator_ != 1) {
      CHECK_NE(item.denominator_, 0);
      int scale = lcm / item.denominator_;
      item.denominator_ = 1;
      item.numerator_ = item.numerator_ / lcm * scale;
    }
    bool is_negative = (item.numerator_ * item.denominator_) < 0;
    Expr item_expr = item.ToExpr(data_type_, is_negative);
    if (is_negative) {
      lhs = lhs.defined() ? Sub::make(lhs, item_expr) : item.ToExpr(data_type_, false);
    } else {
      lhs = lhs.defined() ? Add::make(lhs, item_expr) : item_expr;
    }
  }
  if (e.as<LE>()) {
    return ReducedInequality<LE>(lhs, reduce_var);
  } else if (e.as<LT>()) {
    return ReducedInequality<LT>(lhs, reduce_var);
  } else if (e.as<GE>()) {
    return ReducedInequality<GE>(lhs, reduce_var);
  } else if (e.as<GT>()) {
    return ReducedInequality<GT>(lhs, reduce_var);
  } else {
    LOG(FATAL) << "Only support to reduce LE, LT, GE, GT inequality. " << e;
    return Expr();
  }
}

bool ArithExprSimplifier::IsMonotonic(const Expr &e, const Var &var) {
  CanonicalForm form(data_type_);
  set<Monomial> norm_form = form.ExprNormalForm(e);
  Expr derivative;
  for (auto item : norm_form) {
    if (item.degree_.empty()) {
      continue;
    }
    bool is_negative = item.GetSign();
    auto it = item.degree_.find(var);
    item.numerator_ = item.numerator_ * it->second;
    it->second = it->second - 1;
    auto item_expr = item.ToExpr(data_type_, is_negative);

    if (is_negative) {
      derivative = derivative.defined() ? Sub::make(derivative, item_expr) : item.ToExpr(data_type_, false);
    } else {
      derivative = derivative.defined() ? Add::make(derivative, item_expr) : item_expr;
    }
  }

  if (!derivative.defined()) {
    derivative = make_const(data_type_, 0);
  }

  return GetSign(derivative) != static_cast<int>(Sign::UNK);
}

int64_t ArithExprSimplifier::GetSup(const Expr &e) {
  CanonicalForm form(data_type_);
  set<Monomial> norm_form = form.ExprNormalForm(e);
  CHECK(!norm_form.empty());
  CHECK(!norm_form.begin()->degree_.empty());
  int64_t highest = norm_form.begin()->degree_.begin()->second;
  int64_t highest_coef = 1;
  float cst = 0;

  for (auto &item : norm_form) {
    if (item.degree_.empty()) {
      cst = static_cast<float>(item.numerator_ * -1);
    }
    if (item.numerator_ == 0 || item.denominator_ == 0) {
      continue;
    }
    highest_coef = item.degree_.begin()->second > highest ? item.denominator_ / item.numerator_ : highest_coef;
    highest = item.degree_.begin()->second > highest ? item.degree_.begin()->second : highest;
  }
  return static_cast<int64_t>(pow(cst * static_cast<float>(highest_coef), 1.0 / highest)) + 1;
}

bool ArithExprSimplifier::IsZeroExpr(const Expr &e) {
  if (auto imm = e.as<IntImm>()) {
    return imm->value == 0;
  }
  CanonicalForm form(data_type_);
  set<Monomial> norm_form = form.ExprNormalForm(e);
  return norm_form.empty();
}

Expr ArithExprSimplifier::Gcd(const Expr &e1, const Expr &e2) {
  if (e1.as<IntImm>() && e1.as<IntImm>()->value == 0) return e2;
  if (e2.as<IntImm>() && e2.as<IntImm>()->value == 0) return e1;
  if (e1.as<IntImm>() && e1.as<IntImm>()->value == 1) return e1;
  if (e2.as<IntImm>() && e2.as<IntImm>()->value == 1) return e2;
  if (e1.as<IntImm>() && e2.as<IntImm>()) {
    auto gcd = ktvm::ir::gcd(e1.as<IntImm>()->value, e2.as<IntImm>()->value);
    return Expr(gcd);
  }

  CanonicalForm form(data_type_);
  set<Monomial> normal_form2e1 = form.ExprNormalForm(e1);
  set<Monomial> normal_form2e2 = form.ExprNormalForm(e2);
  set<Monomial> gcdNormForm = form.Gcd(normal_form2e1, normal_form2e2);

  return form.CreateMonomialsExpr(gcdNormForm);
}

Expr ArithExprSimplifier::ModSimplify(Expr &a, Expr &b) {
  CanonicalForm form(data_type_);
  set<Monomial> normal_form1 = form.ExprNormalForm(a);

  if (b.as<IntImm>() && b.as<IntImm>()->value != 0) {
    // for case like (4*m + n)%2 = n
    for (auto it = normal_form1.begin(); it != normal_form1.end();) {
      if (it->numerator_ % b.as<IntImm>()->value == 0) {
        normal_form1.erase(it++);
      } else {
        it++;
      }
    }
    a = form.CreateMonomialsExpr(normal_form1);
  } else {
    set<Monomial> normal_form2 = form.ExprNormalForm(b);
    form.DivSimplify(normal_form1, normal_form2);
    a = form.CreateMonomialsExpr(normal_form1);
    b = form.CreateMonomialsExpr(normal_form2);
    CHECK(!IsZero(b)) << "cannot mod by zero! ";
  }

  if (a.as<IntImm>() && b.as<IntImm>()) {
    int64_t value_a = a.as<IntImm>()->value;
    int64_t value_b = b.as<IntImm>()->value;
    return make_const(data_type_, value_a % value_b);
  }
  return normal_form1.empty() || is_const_int(b, 1) ? make_const(data_type_, 0) : Expr();
}

Expr ArithExprSimplifier::DivSimplify(Expr &a, Expr &b) {
  CanonicalForm form(data_type_);
  set<Monomial> lhsNormalForm = form.ExprNormalForm(a);
  set<Monomial> rhsNormalForm = form.ExprNormalForm(b);
  form.DivSimplify(lhsNormalForm, rhsNormalForm);
  a = form.CreateMonomialsExpr(lhsNormalForm);
  b = form.CreateMonomialsExpr(rhsNormalForm);
  CHECK(!IsZero(b)) << "cannot div by zero! ";
  if (is_const_int(b, 1)) {
    return a;
  } else if (a.as<IntImm>() && b.as<IntImm>()) {
    int64_t value_a = a.as<IntImm>()->value;
    int64_t value_b = b.as<IntImm>()->value;
    CHECK_NE(value_b, 0);
    return make_const(data_type_, value_a / value_b);
  } else {
    return Expr();
  }
}

vector<tuple<int, int, int>> ArithExprSimplifier::CountSort(vector<int> &v) {
  // return vector of tuple with <count_num, numerator, first index for numerator.
  CHECK(!v.empty());
  vector<int> ori_v = v;
  std::sort(v.begin(), v.end());
  vector<tuple<int, int, int>> sort_pair;
  std::unordered_set<int> v_set(v.begin(), v.end());
  int size = static_cast<int>(v.size());
  int cur_v = v[0];
  int count = 1;
  sort_pair.emplace_back(count, v[0], 0);
  for (int i = 1; i < size; i++) {
    if (v[i] != cur_v) {
      auto it = std::find(ori_v.begin(), ori_v.end(), v[i]);
      int index = std::distance(ori_v.begin(), it);
      sort_pair.emplace_back(count, v[i], index);
      count = 1;
      cur_v = v[i];
    } else {
      count += 1;
    }
  }
  sort(sort_pair.begin(), sort_pair.end(), [](std::tuple<int, int, int> a, std::tuple<int, int, int> b) -> bool {
    return (std::get<0>(a) > std::get<0>(b));
  });
  return sort_pair;
}

Array<Expr> ArithExprSimplifier::GetPolynomial(const Expr &e1, const Expr &e2) {
  // get polynomial without 0-degree monomial then extract common factor.
  // e1 is target expr, e2 is the potential matching expr
  Array<Expr> exprs;
  CanonicalForm form(data_type_);
  set<Monomial> normal_form1 = form.ExprNormalForm(e1);
  set<Monomial> normal_form2 = form.ExprNormalForm(e2);
  if (normal_form1.empty() || normal_form2.empty() || normal_form1.size() > normal_form2.size()) {
    exprs.push_back(e1);
    exprs.push_back(e2);
    return exprs;
  }
  vector<int> numerator_vec1;
  vector<int> numerator_vec2;
  Expr const1;
  Expr const2;
  auto rbegin = normal_form1.rbegin();
  if (rbegin->degree_.empty()) {
    const1 = rbegin->ToExpr(data_type_);
    normal_form1.erase(*rbegin);
  }
  auto rbegin2 = normal_form2.rbegin();
  if (rbegin2->degree_.empty()) {
    const2 = rbegin2->ToExpr(data_type_);
    normal_form2.erase(*rbegin2);
  }
  for (const auto &item : normal_form1) {
    auto it = normal_form2.find(item);
    if (it == normal_form2.end() || item.denominator_ != 1 || it->denominator_ != 1) {
      exprs.push_back(e1);
      exprs.push_back(e2);
      return exprs;
    }
    numerator_vec1.push_back(item.numerator_);
    numerator_vec2.push_back(it->numerator_);
  }

  vector<tuple<int, int, int>> sort_vec1 = CountSort(numerator_vec1);
  vector<tuple<int, int, int>> sort_vec2 = CountSort(numerator_vec2);
  if (sort_vec1.size() == 1 && sort_vec2.size() == 2) {
    auto it = normal_form2.begin();
    advance(it, get<2>(sort_vec2[0]));
    Monomial extraterm = *it;
    extraterm.numerator_ = get<1>(sort_vec2[0]) - get<1>(sort_vec2[1]);
    extraterm.denominator_ = 1;
    set<Monomial> sub_normal_form2;
    set<Monomial> sub_normalForm3;

    set_intersection(normal_form1.begin(), normal_form1.begin(), normal_form2.begin(), normal_form2.begin(),
                     inserter(sub_normal_form2, sub_normal_form2.begin()));
    set_difference(normal_form1.begin(), normal_form1.begin(), normal_form2.begin(), normal_form2.begin(),
                   inserter(sub_normalForm3, sub_normalForm3.begin()));

    sub_normalForm3.emplace(extraterm);
    Expr expr1 = form.CreateMonomialsExpr(normal_form1);
    Expr expr2 = form.CreateMonomialsExpr(sub_normal_form2);
    Expr extra_expr2 = form.CreateMonomialsExpr(sub_normalForm3);
    expr2 = Mul::make(numerator_vec2[0], expr2);
    Expr new_expr1 = Mul::make(numerator_vec1[0], expr1);
    Expr new_expr2 = Add::make(expr2, extra_expr2);
    if (const1.defined()) {
      new_expr1 = Add::make(new_expr1, const1);
    }
    if (const2.defined()) {
      new_expr2 = Add::make(new_expr2, const2);
    }
    exprs.push_back(new_expr1);
    exprs.push_back(new_expr2);
    return exprs;
  } else {
    exprs.push_back(e1);
    exprs.push_back(e2);
    return exprs;
  }
}

int ArithExprSimplifier::RangeWithPosvar(const Expr &e) {
  CanonicalForm form(data_type_);
  set<Monomial> normal_form = form.ExprNormalForm(e);
  if (static_cast<int>(normal_form.size()) > 1) {
    return static_cast<int>(Interval_1::UNK);
  }
  CHECK(!normal_form.empty());
  Monomial first_term = *normal_form.begin();
  CHECK_NE(first_term.denominator_, 0);
  if (e.as<Div>()) {
    if (first_term.numerator_ == 1 && first_term.denominator_ == 1) {
      return static_cast<int>(Interval_1::GEZERO);
    } else if ((first_term.numerator_ / first_term.denominator_) >= 1) {
      return static_cast<int>(Interval_1::GEONE);
    } else if ((first_term.numerator_ / first_term.denominator_) >= 0) {
      return static_cast<int>(Interval_1::GEZERO);
    } else {
      return static_cast<int>(Interval_1::LTZERO);
    }
  } else {
    if (first_term.numerator_ >= 1) {
      return static_cast<int>(Interval_1::GEONE);
    } else if (first_term.numerator_ >= 0) {
      return static_cast<int>(Interval_1::GEZERO);
    } else {
      return static_cast<int>(Interval_1::LTZERO);
    }
  }
}

Expr ArithExprSimplifier::Simplify(const Expr &e) {
  if (e.as<GE>() || e.as<GT>() || e.as<LE>() || e.as<LT>()) return e;
  CanonicalForm form(data_type_);
  set<Monomial> normal_form = form.ExprNormalForm(e);
  Expr normal_expr = form.CreateMonomialsExpr(normal_form);
  return normal_expr;
}

Expr ArithExprSimplifier::Simplify(const Expr &e, const vector<pair<Var, Var>> &div_mod_pair,
                                   const UnorderedVarMap &div_child, const UnorderedVarMap &floordiv_child) {
  if (e.as<GE>() || e.as<GT>() || e.as<LE>() || e.as<LT>()) return e;
  CanonicalForm form(data_type_);
  VarReplaceMap var_replace;
  for (const auto &pair : div_mod_pair) {
    auto it = div_child.find(pair.first);
    if (it != div_child.end()) {
      CHECK_GE(it->second.size(), 2);
      auto a = form.ExprNormalForm(it->second[0]);
      auto b = form.ExprNormalForm(it->second[1]);
      var_replace[pair.first] =
        tuple<set<Monomial>, set<Monomial>, Expr>(a, b, Div::make(it->second[0], it->second[1]));
      var_replace[pair.second] =
        tuple<set<Monomial>, set<Monomial>, Expr>(a, b, Div::make(it->second[0], it->second[1]));
    } else {
      it = floordiv_child.find(pair.first);
      if (it != floordiv_child.end()) {
        CHECK_GE(it->second.size(), 2);
        auto a = form.ExprNormalForm(it->second[0]);
        auto b = form.ExprNormalForm(it->second[1]);
        var_replace[pair.first] =
          tuple<set<Monomial>, set<Monomial>, Expr>(a, b, Div::make(it->second[0], it->second[1]));
        var_replace[pair.second] =
          tuple<set<Monomial>, set<Monomial>, Expr>(a, b, Div::make(it->second[0], it->second[1]));
      } else {
        LOG(WARNING) << "can not found Div and Mod match, please check.";
      }
    }
  }
  set<Monomial> normal_form = form.ExprNormalForm(e);
  static_cast<void>(form.DivAndModFilter(normal_form, div_mod_pair, var_replace));
  Expr normal_expr = form.CreateMonomialsExpr(normal_form);
  return normal_expr;
}
}  // namespace ir
}  // namespace akg
