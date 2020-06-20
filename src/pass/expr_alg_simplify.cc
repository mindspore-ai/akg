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

#include <tvm/api_registry.h>
#include "expr_alg_simplify.h"
#include "arith_expr_simplify.h"
#include "utils.h"

namespace akg {
namespace ir {
struct VariableLess {
  bool operator()(const Variable *l, const Variable *r) const { return l < r; }
};

void TypeChecker(const ktvm::Expr &e, ktvm::DataType &outer_type, ktvm::DataType &highest_type, bool &is_cast_exists) {
  PackedFunc pre_order_visit;
  bool is_first_cast_op = true;
  pre_order_visit = PackedFunc([&pre_order_visit, &is_first_cast_op, &highest_type, &outer_type, &is_cast_exists](
                                 const TVMArgs args, TVMRetValue *ret) {
    Expr e = args[0];
    if (e.as<IntImm>() || e.as<UIntImm>() || e.as<FloatImm>() || e.as<Variable>()) {
      if (!is_first_cast_op) {
        if (e.type().bits() > highest_type.bits()) {
          highest_type = e.type();
        }
      } else {
        highest_type = e.type();
      }
    } else if (auto load = e.as<Load>()) {
      static_cast<void>(pre_order_visit(load->index));
    } else if (auto call = e.as<Call>()) {
      for (auto a : call->args) {
        static_cast<void>(pre_order_visit(a));
      }
    } else if (auto cast = e.as<Cast>()) {
      if (is_first_cast_op) {
        outer_type = cast->type;
        highest_type = cast->type;
        is_first_cast_op = false;
        is_cast_exists = true;
      } else {
        if (cast->type.bits() > highest_type.bits()) highest_type = cast->type;
      }
      static_cast<void>(pre_order_visit(cast->value));
    } else {
      Array<Expr> child = GetBinaryOpExprChildren(e);
      if (!child.empty()) {
        static_cast<void>(pre_order_visit(child[0]));
        static_cast<void>(pre_order_visit(child[1]));
      }
    }
    *ret = TVMRetValue();
  });

  static_cast<void>(pre_order_visit(e));
}

Expr ExprSimplifier::Simplify(const ktvm::Expr &e) {
  if (!e.type().is_int() && !e.type().is_uint()) {
    return e;
  }
  old_vars_ = GetVarsInExpr(e, false);
  ktvm::DataType outer_type;
  ktvm::DataType highest_type;
  bool is_cast_exists = false;
  TypeChecker(e, outer_type, highest_type, is_cast_exists);
  highest_cast_type_ = highest_type;

  Expr castForm = is_cast_exists ? CastNormalize(e, highest_type) : e;

  auto pure_expr = Mutate(castForm);
  ArithExprSimplifier simplifier(highest_cast_type_);
  Expr expr;
  if (div_mod_pair_.empty()) {
    expr = simplifier.Simplify(pure_expr);
  } else {
    expr = simplifier.Simplify(pure_expr, div_mod_pair_, div_child_, floordiv_child_);
  }
  auto ret = Retrieval(expr);
  return is_cast_exists ? Cast::make(outer_type, ret) : ret;
}

Expr ExprSimplifier::Simplify(const ktvm::Expr &e, const vector<Expr> &conds) {
  info_ = conds;
  return Simplify(e);
}

vector<Expr> ExprSimplifier::GatherRetroTerm(const Expr &e) {
  vector<Expr> e_vec;
  CHECK(e.as<Add>());
  auto add_a = e.as<Add>()->a;
  auto add_b = e.as<Add>()->b;
  PostOrderVisit(add_a, [&add_b, &e_vec, this](const NodeRef &node) {
    if (node.as<FloorDiv>() && node.as<FloorDiv>()->a.as<Sub>()) {
      auto new_numerator = Simplify(add_b * node.as<FloorDiv>()->b + node.as<FloorDiv>()->a);
      e_vec.push_back(FloorDiv::make(new_numerator, node.as<FloorDiv>()->b));
    }
    if (node.as<Div>() && node.as<Div>()->a.as<Sub>()) {
      auto new_numerator = Simplify(add_b * node.as<Div>()->b + node.as<Div>()->a);
      e_vec.push_back(Div::make(new_numerator, node.as<Div>()->b));
    }
  });
  return e_vec;
}

Expr ExprSimplifier::SubstituteDiv(const Expr &e, const Expr &substitute) {
  if (e.as<FloorDiv>() || e.as<Div>()) {
    return substitute;
  } else if (e.as<Add>()) {
    return Add::make(SubstituteDiv(e.as<Add>()->a, substitute), SubstituteDiv(e.as<Add>()->b, substitute));
  } else if (e.as<Sub>()) {
    return Sub::make(SubstituteDiv(e.as<Sub>()->a, substitute), SubstituteDiv(e.as<Sub>()->b, substitute));
  } else if (e.as<Mul>()) {
    return Mul::make(SubstituteDiv(e.as<Mul>()->a, substitute), SubstituteDiv(e.as<Mul>()->b, substitute));
  } else {
    return e;
  }
}

Expr ExprSimplifier::RetroConstToMin(const Expr &e) {
  if (e.as<Add>() && e.as<Add>()->a.as<IntImm>() && e.as<Add>()->b.as<Min>()) {
    Expr min_a = Add::make(e.as<Add>()->b.as<Min>()->a, e.as<Add>()->a);
    Expr min_b = Add::make(e.as<Add>()->b.as<Min>()->b, e.as<Add>()->a);
    vector<Expr> e_vec_1 = GatherRetroTerm(min_a);
    vector<Expr> e_vec_2 = GatherRetroTerm(min_b);
    if (e_vec_1.size() == 1 && e_vec_2.size() == 0) {
      Expr substitute_e = Simplify(SubstituteDiv(e.as<Add>()->b.as<Min>()->a, e_vec_1[0]));
      return Min::make(substitute_e, Simplify(min_b));
    } else if (e_vec_1.size() == 0 && e_vec_2.size() == 1) {
      Expr substitute_e = Simplify(SubstituteDiv(e.as<Add>()->b.as<Min>()->b, e_vec_2[0]));
      return Min::make(Simplify(min_a), substitute_e);
    } else {
      return e;
    }
  } else if (e.as<Add>() && e.as<Add>()->b.as<IntImm>() && e.as<Add>()->a.as<Min>()) {
    Expr min_a = Add::make(e.as<Add>()->a.as<Min>()->a, e.as<Add>()->b);
    Expr min_b = Add::make(e.as<Add>()->a.as<Min>()->b, e.as<Add>()->b);
    vector<Expr> e_vec_1 = GatherRetroTerm(min_a);
    vector<Expr> e_vec_2 = GatherRetroTerm(min_b);
    if (e_vec_1.size() == 1 && e_vec_2.size() == 0) {
      Expr substitute_e = Simplify(SubstituteDiv(e.as<Add>()->a.as<Min>()->a, e_vec_1[0]));
      return Min::make(substitute_e, Simplify(min_b));
    } else if (e_vec_1.size() == 0 && e_vec_2.size() == 1) {
      Expr substitute_e = Simplify(SubstituteDiv(e.as<Add>()->a.as<Min>()->b, e_vec_2[0]));
      return Min::make(Simplify(min_a), substitute_e);
    } else {
      return e;
    }
  } else {
    return e;
  }
}

Expr ExprSimplifier::ExtraDivVar(const Expr &expr, const Var &div_var) {
  Expr a;
  if (auto le = expr.as<LE>()) {
    a = le->a;
  } else if (auto lt = expr.as<LT>()) {
    a = lt->a;
  } else if (auto gt = expr.as<GT>()) {
    a = gt->a;
  } else if (auto ge = expr.as<GE>()) {
    a = ge->a;
  }
  if (auto var = a.as<Variable>()) {
    if (var != div_var.get()) {
      return expr;
    }
  }

  Expr ret = expr;
  Expr lhs;
  if (floordiv_child_.count(div_var)) {
    if (floordiv_child_[div_var][1].as<IntImm>() && floordiv_child_[div_var][1].as<IntImm>()->value == 1) {
      lhs = floordiv_child_[div_var][0];
    } else {
      lhs = FloorDiv::make(floordiv_child_[div_var][0], floordiv_child_[div_var][1]);
    }
  } else if (div_child_.count(div_var)) {
    if (div_child_[div_var][1].as<IntImm>() && div_child_[div_var][1].as<IntImm>()->value == 1) {
      lhs = div_child_[div_var][0];
    } else {
      lhs = Div::make(div_child_[div_var][0], div_child_[div_var][1]);
    }
  }
  if (lhs.defined()) {
    if (expr.as<LE>()) {
      Expr b = Mutate(expr.as<LE>()->b);
      ret = LE::make(lhs, b);
    } else if (expr.as<LT>()) {
      Expr b = Mutate(expr.as<LT>()->b);
      ret = LT::make(lhs, b);
    } else if (expr.as<GT>()) {
      Expr b = Mutate(expr.as<GT>()->b);
      ret = GT::make(lhs, b);
    } else if (expr.as<GE>()) {
      Expr b = Mutate(expr.as<GE>()->b);
      ret = GE::make(lhs, b);
    }
    ArithExprSimplifier simplifier(highest_cast_type_);
    ret = simplifier.ReduceInequality(ret, reduce_var_);
  }
  return ret;
}

int ExprSimplifier::VisitDivWithLcm(const Expr &expr) const {
  int lcm = 1;
  int gcd = 1;
  PostOrderVisit(expr, [&lcm, &gcd](const NodeRef &node) {
    if (auto f_div = node.as<FloorDiv>()) {
      int denominator = f_div->b.as<IntImm>()->value;
      CHECK(denominator != 0) << "denominator is zero!";
      gcd = gcd == 1 ? denominator : ktvm::ir::gcd(gcd, denominator);
      lcm = lcm == 1 ? gcd : lcm * denominator / gcd;
    } else if (auto div = node.as<Div>()) {
      int denominator = div->b.as<IntImm>()->value;
      CHECK(denominator != 0) << "denominator is zero!";
      gcd = gcd == 1 ? denominator : ktvm::ir::gcd(gcd, denominator);
      lcm = lcm == 1 ? gcd : lcm * denominator / gcd;
    }
  });
  return lcm;
}

Expr ExprSimplifier::ReduceInequality(const ktvm::Expr &e, const Var &reduce_var, bool scale, bool get_larger) {
  old_vars_ = GetVarsInExpr(e, false);
  ktvm::DataType outer_type;
  ktvm::DataType highest_type;
  bool is_cast_exists = false;
  TypeChecker(e, outer_type, highest_type, is_cast_exists);
  highest_cast_type_ = highest_type;
  Expr castForm = is_cast_exists ? CastNormalize(e, highest_type) : e;
  is_less_than_ = (e.as<LE>() || e.as<LT>());
  reduce_var_ = reduce_var;
  var_with_reduce_.push_back(reduce_var);
  if (scale) {
    is_scale_ = true;
  }
  auto pure_expr = Mutate(castForm);
  vector<Var> reduce_vars;
  for (auto var : var_with_reduce_) {
    if (IsVarInExpr(var, pure_expr)) {
      reduce_vars.push_back(var);
    }
  }
  Expr expr = pure_expr;
  auto size = reduce_vars.size();
  bool reduce_in_expr = IsVarInExpr(reduce_var, pure_expr);

  ArithExprSimplifier simplifier(highest_cast_type_);
  if (size > 1) {
    if (scale && (size == divvar_with_reduce_.size() || (size == divvar_with_reduce_.size() + 1 && reduce_in_expr))) {
      expr = ReduceIneqlWithScale(pure_expr, reduce_var, is_less_than_, get_larger);
    } else {
      expr = simplifier.ReduceInequality(pure_expr, reduce_vars);
    }
  } else if (size == 1 && reduce_in_expr) {
    expr = simplifier.ReduceInequality(pure_expr, reduce_var);
  } else if (size == 1) {
    int var_count = 0;
    PostOrderVisit(pure_expr, [&var_count, &reduce_vars](const NodeRef &node) {
      if (node.as<Variable>() && node.as<Variable>() == reduce_vars[0].get()) {
        var_count += 1;
      }
    });
    if (var_count == 1) {
      expr = simplifier.ReduceInequality(pure_expr, reduce_vars[0]);
      expr = ExtraDivVar(expr, reduce_vars[0]);
    } else {
      if (scale) {
        expr = ReduceIneqlWithScale(pure_expr, reduce_var, is_less_than_, get_larger);
      }
    }
  }

  auto ret = Retrieval(expr);
  ret = Simplify(ret);
  // Solving Polynomial Inequality, monotonically increasing.
  // Cast expr if there exists Cast.
  if (auto le = ret.as<LE>()) {
    if (!Equal(le->a, reduce_var)) {
      ret = HighDegIneqlSolver(ret, reduce_var, get_larger);
    }
    return is_cast_exists ? LE::make(Cast::make(outer_type, ret.as<LE>()->a), Cast::make(outer_type, ret.as<LE>()->b))
                          : ret;
  } else if (auto lt = ret.as<LT>()) {
    if (!Equal(lt->a, reduce_var)) {
      ret = HighDegIneqlSolver(ret, reduce_var, get_larger);
    }
    return is_cast_exists ? LT::make(Cast::make(outer_type, ret.as<LT>()->a), Cast::make(outer_type, ret.as<LT>()->b))
                          : ret;
  } else if (auto gt = ret.as<GT>()) {
    if (!Equal(gt->a, reduce_var)) {
      ret = HighDegIneqlSolver(ret, reduce_var, get_larger);
    }
    return is_cast_exists ? GT::make(Cast::make(outer_type, ret.as<GT>()->a), Cast::make(outer_type, ret.as<GT>()->b))
                          : ret;
  } else if (auto ge = ret.as<GE>()) {
    if (!Equal(ge->a, reduce_var)) {
      ret = HighDegIneqlSolver(ret, reduce_var, get_larger);
    }
    return is_cast_exists ? GE::make(Cast::make(outer_type, ret.as<GE>()->a), Cast::make(outer_type, ret.as<GE>()->a))
                          : ret;
  } else {
    // result of no solution input cannot be reconstructed to above ops, return original input
    return e;
  }
}

Expr ExprSimplifier::ReduceIneqlWithScale(const Expr &e, const Var &reduce_var, bool is_less, bool get_larger) {
  ArithExprSimplifier simplifier(highest_cast_type_);
  Expr expr = simplifier.ScaleSubstitute(e, div_scale_range_, is_less, get_larger);
  expr = simplifier.Simplify(expr);
  if (e.as<LE>()) {
    expr = LE::make(expr, 0);
  } else if (e.as<LT>()) {
    expr = LT::make(expr, 0);
  } else if (e.as<GT>()) {
    expr = GT::make(expr, 0);
  } else if (e.as<GE>()) {
    expr = GE::make(expr, 0);
  }

  int lcm = VisitDivWithLcm(expr);
  if (lcm == 1) {
    expr = simplifier.ReduceInequality(expr, reduce_var);
  } else {
    expr = simplifier.ReduceInequality(expr, reduce_var, lcm);
  }
  return expr;
}

Expr ExprSimplifier::HighDegIneqlSolver(const Expr &expr, const Var &tar_var, bool get_larger) {
  CHECK(expr.as<LT>() || expr.as<LE>() || expr.as<GT>() || expr.as<GE>()) << "Input is not Inequality";
  ArithExprSimplifier simplifier(highest_cast_type_);
  std::unordered_set<Var, NodeHash, NodeEqual> var_set;

  // only for one variable case.
  GatherVars(expr, &var_set);
  if (var_set.size() != static_cast<int>(1)) {
    return expr;
  }

  // Only for Polynominal case.
  bool polynominal = true;
  PostOrderVisit(expr, [&polynominal](const NodeRef &node) {
    if (node.as<Div>() || node.as<FloorDiv>()) {
      polynominal = false;
    }
  });

  if (!polynominal) {
    return expr;
  }

  Expr oneside;
  if (auto le = expr.as<LE>()) {
    oneside = le->a - le->b;
  } else if (auto lt = expr.as<LT>()) {
    oneside = lt->a - lt->b;
  } else if (auto gt = expr.as<GT>()) {
    oneside = gt->a - gt->b;
  } else {
    auto ge = expr.as<GE>();
    oneside = ge->a - ge->b;
  }

  if (!simplifier.IsMonotonic(oneside, *var_set.begin())) {
    return expr;
  }

  Var var = *var_set.begin();

  int64_t sup = simplifier.GetSup(oneside);
  int64_t inf = 0;
  std::unordered_map<Var, Expr, NodeHash, NodeEqual> substitute_map_1;
  std::unordered_map<Var, Expr, NodeHash, NodeEqual> substitute_map_2;
  substitute_map_1.emplace(var, make_const(oneside.type(), inf));
  substitute_map_2.emplace(var, make_const(oneside.type(), sup));
  auto inf_v = Simplify(Substitute(oneside, substitute_map_1));
  auto sup_v = Simplify(Substitute(oneside, substitute_map_2));
  if ((inf_v.as<IntImm>()->value <= 0 && sup_v.as<IntImm>()->value >= 0) ||
      (inf_v.as<IntImm>()->value >= 0 && sup_v.as<IntImm>()->value <= 0)) {
    vector<int64_t> ret = BisectSolver(oneside, var, inf, sup);
    if (get_larger == is_less_than_) {
      if (expr.as<LE>()) {
        return LE::make(var, make_const(oneside.type(), ret[1]));
      } else if (expr.as<LT>()) {
        return LT::make(var, make_const(oneside.type(), ret[1]));
      } else if (expr.as<GT>()) {
        return GT::make(var, make_const(oneside.type(), ret[0]));
      } else {
        return GE::make(var, make_const(oneside.type(), ret[0]));
      }
    } else {
      if (expr.as<LE>()) {
        return LE::make(var, make_const(oneside.type(), ret[0]));
      } else if (expr.as<LT>()) {
        return LT::make(var, make_const(oneside.type(), ret[0]));
      } else if (expr.as<GT>()) {
        return GT::make(var, make_const(oneside.type(), ret[1]));
      } else {
        return GE::make(var, make_const(oneside.type(), ret[1]));
      }
    }
  } else if (inf_v.as<IntImm>()->value >= 0 && sup_v.as<IntImm>()->value >= 0) {
    if (expr.as<LE>()) {
      return LE::make(var, make_const(oneside.type(), 0));
    } else if (expr.as<LT>()) {
      return LT::make(var, make_const(oneside.type(), 0));
    } else if (expr.as<GT>()) {
      return GT::make(var, make_const(oneside.type(), 0));
    } else {
      return GE::make(var, make_const(oneside.type(), 0));
    }
  } else if (inf_v.as<IntImm>()->value <= 0 && sup_v.as<IntImm>()->value <= 0) {
    if (expr.as<LE>()) {
      return GE::make(var, make_const(oneside.type(), 0));
    } else if (expr.as<LT>()) {
      return GT::make(var, make_const(oneside.type(), 0));
    } else if (expr.as<GT>()) {
      return LT::make(var, make_const(oneside.type(), 0));
    } else {
      return LE::make(var, make_const(oneside.type(), 0));
    }
  } else {
    return expr;
  }
}

vector<int64_t> ExprSimplifier::BisectSolver(Expr &e, Var &var, int64_t inf, int64_t sup) {
  auto func = [](Expr expr, Var var, int64_t value) -> Expr {
    std::unordered_map<Var, Expr, NodeHash, NodeEqual> substitute_map;
    substitute_map.emplace(var, make_const(expr.type(), value));
    return Substitute(expr, substitute_map);
  };

  int64_t cnt = sup - inf;
  while (cnt > 1) {
    Expr lexpr = Simplify(func(e, var, inf));
    Expr rexpr = Simplify(func(e, var, sup));
    int64_t left = lexpr.as<IntImm>()->value;
    int64_t right = rexpr.as<IntImm>()->value;
    vector<int64_t> ret;
    if (left == 0) {
      ret.push_back(inf);
      ret.push_back(inf);
      return ret;
    }

    if (right == 0) {
      ret.push_back(sup);
      ret.push_back(sup);
      return ret;
    }

    int64_t mid = (inf + sup) / 2;
    Expr mexpr = Simplify(func(e, var, mid));
    if (left * mexpr.as<IntImm>()->value < 0) {
      sup = mid;
    } else {
      inf = mid;
    }
    cnt = sup - inf;
  }
  vector<int64_t> ret;
  ret.push_back(inf);
  ret.push_back(sup);
  return ret;
}

bool ExprSimplifier::Equals(const ktvm::Expr &e1, const ktvm::Expr &e2) {
  auto pure_expr = Mutate(e1 - e2);
  ArithExprSimplifier simplifier;
  // is zero expr
  return simplifier.IsZeroExpr(pure_expr);
}

bool ExprSimplifier::CanProveWithParam(const Expr &e) {
  if (ktvm::arith::Analyzer().CanProve(e)) return true;
  if (ktvm::arith::Analyzer().CanProve(!e)) return false;
  CHECK(e.as<LE>() || e.as<LT>() || e.as<GT>() || e.as<GE>() || e.as<EQ>()) << "Unsupported expr " << e;

  ArithExprSimplifier simplifier;
  is_scale_ = true;
  if (auto eq = e.as<EQ>()) {
    auto pure_expr = eq->a - eq->b;
    pure_expr = Mutate(pure_expr);
    return simplifier.IsZeroExpr(pure_expr);
  } else if (auto le = e.as<LE>()) {
    auto pure_expr = le->a - le->b;
    pure_expr = Mutate(pure_expr);
    pure_expr = simplifier.ScaleSubstitute(pure_expr, div_scale_range_, true, false);
    pure_expr = simplifier.Simplify(pure_expr);
    return (GetSign(pure_expr) == static_cast<int>(Sign::NEG) || GetSign(pure_expr) == static_cast<int>(Sign::ZERO));
  } else if (auto lt = e.as<LT>()) {
    auto pure_expr = lt->a - lt->b;
    pure_expr = Mutate(pure_expr);
    pure_expr = simplifier.ScaleSubstitute(pure_expr, div_scale_range_, true, false);
    pure_expr = simplifier.Simplify(pure_expr);
    return GetSign(pure_expr) == static_cast<int>(Sign::NEG);
  } else if (auto ge = e.as<GE>()) {
    auto pure_expr = ge->a - ge->b;
    pure_expr = Mutate(pure_expr);
    pure_expr = simplifier.ScaleSubstitute(pure_expr, div_scale_range_, false, false);
    pure_expr = simplifier.Simplify(pure_expr);
    return (GetSign(pure_expr) == static_cast<int>(Sign::POS) || GetSign(pure_expr) == static_cast<int>(Sign::ZERO));
  } else {
    auto gt = e.as<GT>();
    auto pure_expr = gt->a - gt->b;
    pure_expr = Mutate(pure_expr);
    pure_expr = simplifier.ScaleSubstitute(pure_expr, div_scale_range_, false, false);
    pure_expr = simplifier.Simplify(pure_expr);
    return GetSign(pure_expr) == static_cast<int>(Sign::POS);
  }
}

bool ExprSimplifier::CanProveWithPosParam(const Expr &e) {
  if (ktvm::arith::Analyzer().CanProve(e)) return true;
  if (ktvm::arith::Analyzer().CanProve(!e)) return false;
  CHECK(e.as<LE>() || e.as<LT>() || e.as<GT>() || e.as<GE>() || e.as<EQ>()) << "Unsupported expr " << e;

  ArithExprSimplifier simplifier;
  is_scale_ = true;
  if (auto eq = e.as<EQ>()) {
    auto pure_expr = eq->a - eq->b;
    pure_expr = Mutate(pure_expr);
    return simplifier.IsZeroExpr(pure_expr);
  } else if (auto le = e.as<LE>()) {
    auto pure_expr = le->a - le->b;
    pure_expr = Mutate(pure_expr);
    pure_expr = simplifier.ScaleSubstitute(pure_expr, div_scale_range_, true, false);
    pure_expr = simplifier.Simplify(pure_expr);
    auto expr_sign = GetSign(pure_expr);
    if (expr_sign == static_cast<int>(Sign::UNK)) {
      if (auto sub = pure_expr.as<Sub>()) {
        if (!is_const_int(pure_expr.as<Sub>()->a, 0)) {
          auto quotient = simplifier.Simplify(Div::make(sub->b, sub->a));
          if (simplifier.RangeWithPosvar(quotient) == static_cast<int>(Interval_1::GEONE)) {
            return true;
          }
        }
      } else if (auto add = pure_expr.as<Add>()) {
        auto add_a = GetSign(add->a);
        auto neg_item = add_a < 0 ? -1 * add->a : -1 * add->b;
        auto pos_item = add_a < 0 ? add->b : add->a;
        auto quotient = simplifier.Simplify(Div::make(neg_item, pos_item));
        if (simplifier.RangeWithPosvar(quotient) == static_cast<int>(Interval_1::GEONE)) {
          return true;
        }
      }
    }
    return (expr_sign == static_cast<int>(Sign::NEG) || expr_sign == static_cast<int>(Sign::ZERO));
  } else if (auto lt = e.as<LT>()) {
    auto pure_expr = lt->a - lt->b;
    pure_expr = Mutate(pure_expr);
    pure_expr = simplifier.ScaleSubstitute(pure_expr, div_scale_range_, true, false);
    pure_expr = simplifier.Simplify(pure_expr);

    auto expr_sign = GetSign(pure_expr);
    if (expr_sign == static_cast<int>(Sign::UNK)) {
      if (auto sub = pure_expr.as<Sub>()) {
        if (!is_const_int(pure_expr.as<Sub>()->a, 0)) {
          auto quotient = simplifier.Simplify(Div::make(sub->b, sub->a));
          if (simplifier.RangeWithPosvar(quotient) == static_cast<int>(Interval_1::GEONE)) {
            return true;
          }
        }
      } else if (auto add = pure_expr.as<Add>()) {
        auto add_a = GetSign(add->a);
        auto neg_item = add_a < 0 ? -1 * add->a : -1 * add->b;
        auto pos_item = add_a < 0 ? add->b : add->a;
        auto quotient = simplifier.Simplify(Div::make(neg_item, pos_item));
        if (simplifier.RangeWithPosvar(quotient) == static_cast<int>(Interval_1::GEONE)) {
          return true;
        }
      }
    }
    return expr_sign == static_cast<int>(Sign::NEG);
  } else if (auto ge = e.as<GE>()) {
    auto pure_expr = ge->a - ge->b;
    pure_expr = Mutate(pure_expr);
    pure_expr = simplifier.ScaleSubstitute(pure_expr, div_scale_range_, false, false);
    pure_expr = simplifier.Simplify(pure_expr);
    auto expr_sign = GetSign(pure_expr);
    if (expr_sign == static_cast<int>(Sign::UNK)) {
      if (auto sub = pure_expr.as<Sub>()) {
        if (!is_const_int(pure_expr.as<Sub>()->b, 0)) {
          auto quotient = simplifier.Simplify(Div::make(sub->a, sub->b));
          if (simplifier.RangeWithPosvar(quotient) == static_cast<int>(Interval_1::GEONE)) {
            return true;
          }
        } else if (auto add = pure_expr.as<Add>()) {
          auto add_a = GetSign(add->a);
          auto neg_item = add_a < 0 ? -1 * add->a : -1 * add->b;
          auto pos_item = add_a < 0 ? add->b : add->a;
          auto quotient = simplifier.Simplify(Div::make(pos_item, neg_item));
          if (simplifier.RangeWithPosvar(quotient) == static_cast<int>(Interval_1::GEONE)) {
            return true;
          }
        }
      }
    }
    return (expr_sign == static_cast<int>(Sign::POS) || expr_sign == static_cast<int>(Sign::ZERO));
  } else {
    auto pure_expr = e.as<GT>()->a - e.as<GT>()->b;
    pure_expr = Mutate(pure_expr);
    pure_expr = simplifier.ScaleSubstitute(pure_expr, div_scale_range_, false, false);
    pure_expr = simplifier.Simplify(pure_expr);
    auto expr_sign = GetSign(pure_expr);
    if (expr_sign == static_cast<int>(Sign::UNK)) {
      if (auto sub = pure_expr.as<Sub>()) {
        if (!is_const_int(pure_expr.as<Sub>()->b, 0)) {
          auto quotient = simplifier.Simplify(Div::make(sub->a, sub->b));
          if (simplifier.RangeWithPosvar(quotient) == static_cast<int>(Interval_1::GEONE)) {
            return true;
          }
        }
      } else if (auto add = pure_expr.as<Add>()) {
        auto add_a = GetSign(add->a);
        auto neg_item = add_a < 0 ? -1 * add->a : -1 * add->b;
        auto pos_item = add_a < 0 ? add->b : add->a;
        auto quotient = simplifier.Simplify(Div::make(pos_item, neg_item));
        if (simplifier.RangeWithPosvar(quotient) == static_cast<int>(Interval_1::GEONE)) {
          return true;
        }
      }
    }
    return expr_sign == static_cast<int>(Sign::POS);
  }
}

bool ExprSimplifier::IsDivisible(const ktvm::Expr &e1, const ktvm::Expr &e2) {
  if (!e2.as<IntImm>()) {
    LOG(FATAL) << "denominator should be integer.";
    return false;
  }
  auto pure_expr = Mutate(e1);
  ArithExprSimplifier simplifier;
  return simplifier.IsDivisible(pure_expr, e2);
}

Expr ExprSimplifier::Gcd(const ktvm::Expr &e1, const ktvm::Expr &e2) {
  auto pure_expr1 = Simplify(e1);
  auto pure_expr2 = Simplify(e2);
  old_vars_ = GetVarsInExpr(e1 + e2, false);
  pure_expr1 = Mutate(pure_expr1);
  pure_expr2 = Mutate(pure_expr2);
  ArithExprSimplifier simplifier;
  auto expr = simplifier.Gcd(pure_expr1, pure_expr2);
  return Retrieval(expr);
}

Expr ExprSimplifier::Retrieval(const Expr &e) {
  Expr expr = e;
  is_retrieval_ = true;
  auto newVars = GetVarsInExpr(expr, false);
  vector<Variable *> newVarsPtr;
  vector<Variable *> oldVarsPtr;
  for (auto it : newVars) {
    newVarsPtr.emplace_back(const_cast<Variable *>(it.get()));
  }
  for (auto it : old_vars_) {
    oldVarsPtr.emplace_back(const_cast<Variable *>(it.get()));
  }

  sort(newVarsPtr.begin(), newVarsPtr.end(), VariableLess());
  sort(oldVarsPtr.begin(), oldVarsPtr.end(), VariableLess());

  vector<Variable *> diffVars;
  std::set_difference(newVarsPtr.begin(), newVarsPtr.end(), oldVarsPtr.begin(), oldVarsPtr.end(),
                      std::back_inserter(diffVars));

  while (!diffVars.empty()) {
    expr = Mutate(expr);
    newVars = GetVarsInExpr(expr, false);
    newVarsPtr.clear();
    for (auto it : newVars) {
      newVarsPtr.emplace_back(const_cast<Variable *>(it.get()));
    }
    sort(newVarsPtr.begin(), newVarsPtr.end(), VariableLess());
    diffVars.clear();
    std::set_difference(newVarsPtr.begin(), newVarsPtr.end(), oldVarsPtr.begin(), oldVarsPtr.end(),
                        std::back_inserter(diffVars));
  }
  is_retrieval_ = false;
  return expr;
}

Array<Expr> ExprSimplifier::GetPolynomial(const ktvm::Expr &e1, const ktvm::Expr &e2) {
  Array<Expr> exprs;
  ArithExprSimplifier simplifier;
  exprs = simplifier.GetPolynomial(e1, e2);
  return exprs;
}

Expr ExprSimplifier::SimplifyWithInfo(const ktvm::Expr &e, const vector<Expr> &conds) const {
  if (auto min = e.as<Min>()) {
    Bound cmp_bound = InferVarBound((min->a - min->b), conds);
    if (GetSign(cmp_bound.max) == static_cast<int>(Sign::NEG) ||
        GetSign(cmp_bound.max) == static_cast<int>(Sign::ZERO)) {
      return min->a;
    } else if (GetSign(cmp_bound.min) == static_cast<int>(Sign::POS) ||
               GetSign(cmp_bound.min) == static_cast<int>(Sign::ZERO)) {
      return min->b;
    } else {
      return e;
    }
  } else if (auto max = e.as<Max>()) {
    Bound cmp_bound = InferVarBound((max->a - max->b), conds);
    if (GetSign(cmp_bound.max) == static_cast<int>(Sign::NEG) ||
        GetSign(cmp_bound.max) == static_cast<int>(Sign::ZERO)) {
      return max->b;
    } else if (GetSign(cmp_bound.min) == static_cast<int>(Sign::POS) ||
               GetSign(cmp_bound.min) == static_cast<int>(Sign::ZERO)) {
      return max->a;
    } else {
      return e;
    }
  } else {
    return e;
  }
}

Expr ExprSimplifier::Mutate_(const Variable *op, const Expr &e) {
  if (!is_retrieval_) return e;
  Var var = ktvm::Downcast<Var>(e);

  if (min_map_.count(op)) {
    auto kv = min_child_.find(var);
    CHECK(kv != min_child_.end());
    return Min::make(Mutate(kv->second[0]), Mutate(kv->second[1]));
  }

  if (max_map_.count(op)) {
    auto kv = max_child_.find(var);
    CHECK(kv != max_child_.end());
    if (kv->first.get() == op) {
      return Max::make(Mutate(kv->second[0]), Mutate(kv->second[1]));
    }
  }

  if (mod_map_.count(op)) {
    auto kv = mod_child_.find(var);
    CHECK(kv != mod_child_.end());
    if (kv->first.get() == op) {
      if (kv->second[1].as<IntImm>() && kv->second[1].as<IntImm>()->value == 1) {
        return make_const(highest_cast_type_, 0);
      } else {
        return Mod::make(Mutate(kv->second[0]), Mutate(kv->second[1]));
      }
    }
  }

  if (cast_map_.count(op)) {
    auto kv = cast_child_.find(var);
    CHECK(kv != cast_child_.end());
    if (kv->first.get() == op) {
      return highest_cast_type_ == Int(64) ? Cast::make(highest_cast_type_, Mutate(kv->second))
                                           : Cast::make(cast_map_[op], Mutate(kv->second));
    }
  }

  if (floordiv_map_.count(op)) {
    auto kv = floordiv_child_.find(var);
    CHECK(kv != floordiv_child_.end());
    if (kv->first.get() == op) {
      return FloorDiv::make(Mutate(kv->second[0]), Mutate(kv->second[1]));
    }
  }

  if (div_map_.count(op)) {
    auto kv = div_child_.find(var);
    CHECK(kv != div_child_.end());
    if (kv->first.get() == op) {
      return Div::make(Mutate(kv->second[0]), Mutate(kv->second[1]));
    }
  }

  if (select_map_.count(op)) {
    auto kv = select_child_.find(var);
    CHECK(kv != select_child_.end());
    return Select::make(Mutate(kv->second[0]), Mutate(kv->second[1]), Mutate(kv->second[2]));
  }

  if (and_map_.count(op)) {
    auto kv = and_child_.find(var);
    CHECK(kv != and_child_.end());
    return And::make(Mutate(kv->second[0]), Mutate(kv->second[1]));
  }

  if (or_map_.count(op)) {
    auto kv = or_child_.find(var);
    CHECK(kv != or_child_.end());
    return Or::make(Mutate(kv->second[0]), Mutate(kv->second[1]));
  }

  if (not_map_.count(op)) {
    auto kv = not_child_.find(var);
    CHECK(kv != not_child_.end());
    return Not::make(Mutate(kv->second[0]));
  }

  if (load_map_.count(op)) {
    auto kv = load_child_.find(var);
    CHECK(kv != load_child_.end());
    if (kv->first.get() == op) {
      return Load::make(kv->second.type, kv->second.buffer_var, Mutate(kv->second.index), Mutate(kv->second.predicate));
    }
  }

  if (call_map_.count(op)) {
    auto kv = call_child_.find(var);
    CHECK(kv != call_child_.end());
    if (kv->first.get() == op) {
      Array<Expr> args_;
      for (auto a : kv->second.args) {
        args_.push_back(Mutate(a));
      }
      return Call::make(kv->second.type, kv->second.name, args_, kv->second.call_type, kv->second.func,
                        kv->second.value_index);
    }
  }

  if (string_map_.count(op)) {
    auto kv = string_child_.find(var);
    CHECK(kv != string_child_.end());
    if (kv->first.get() == op) {
      return StringImm::make(kv->second);
    }
  }
  return e;
}

Expr ExprSimplifier::Mutate_(const Min *op, const Expr &e) {
  auto ret_a = Mutate(op->a);
  auto ret_b = Mutate(op->b);
  Expr mutate_e = Min::make(ret_a, ret_b);
  if (is_retrieval_) return mutate_e;

  // simplify each child
  ArithExprSimplifier simplifier(highest_cast_type_);
  ret_a = simplifier.Simplify(ret_a);
  ret_b = simplifier.Simplify(ret_b);
  auto sub_ret = simplifier.Simplify(ret_a - ret_b);
  if (is_const_true(sub_ret)) {
    return ret_b;
  }

  if (is_const(sub_ret) && !is_const_true(sub_ret)) {
    return ret_a;
  }

  // in case that lhs equals rhs
  if (Equal(ret_a, ret_b)) {
    return ret_a;
  }

  // check whether the min-op already exists. If so, replace the min op with new variable.
  for (const auto &kv : min_child_) {
    if ((Equal(kv.second[0], ret_a) && Equal(kv.second[1], ret_b) && op->type == min_map_[kv.first.get()]) ||
        (Equal(kv.second[0], ret_b) && Equal(kv.second[1], ret_a) && op->type == min_map_[kv.first.get()])) {
      return kv.first;
    }
  }

  // check whether the min op is already simplified to avoid repeat compute.
  for (const auto &cache_entry : cache_min_) {
    if (Equal(cache_entry[0], mutate_e)) {
      return cache_entry[1];
    }
  }

  if (info_.size() > 0) {
    Expr ret_e = SimplifyWithInfo(mutate_e, info_);
    if (!ret_e.as<Min>()) {
      Array<Expr> min_pair;
      ret_e = Mutate(ret_e);
      min_pair.push_back(mutate_e);
      min_pair.push_back(ret_e);
      cache_min_.push_back(min_pair);
      return ret_e;
    }
  }

  string name = "min_" + to_string(min_op_count_++);
  Var x(name, op->type);
  min_map_.emplace(x.get(), op->type);
  min_child_[x] = {ret_a, ret_b};
  if (IsVarsInExpr(var_with_reduce_, ret_a - ret_b)) {
    var_with_reduce_.push_back(x);
  }
  return x;
}

Expr ExprSimplifier::Mutate_(const Max *op, const Expr &e) {
  auto ret_a = Mutate(op->a);
  auto ret_b = Mutate(op->b);
  Expr mutate_e = Max::make(ret_a, ret_b);
  if (is_retrieval_) return mutate_e;

  // simplify each child
  ArithExprSimplifier simplifier(highest_cast_type_);
  ret_a = simplifier.Simplify(ret_a);
  ret_b = simplifier.Simplify(ret_b);
  auto sub_ret = simplifier.Simplify(ret_a - ret_b);
  if (is_const_true(sub_ret)) {
    return ret_a;
  }

  if (is_const(sub_ret) && !is_const_true(sub_ret)) {
    return ret_b;
  }
  // in case that lhs equals rhs
  if (Equal(ret_a, ret_b)) {
    return ret_a;
  }

  // check whether the max-op already exists. If so, replace the max-op with new variable.
  for (const auto &kv : max_child_) {
    if ((Equal(kv.second[0], ret_a) && Equal(kv.second[1], ret_b) && op->type == max_map_[kv.first.get()]) ||
        (Equal(kv.second[0], ret_b) && Equal(kv.second[1], ret_a) && op->type == max_map_[kv.first.get()])) {
      return kv.first;
    }
  }

  // check whether the max-op is already simplified.
  for (const auto &cache_entry : cache_max_) {
    if (Equal(cache_entry[0], mutate_e)) {
      return cache_entry[1];
    }
  }

  if (info_.size() > 0) {
    Expr ret_e = SimplifyWithInfo(e, info_);
    if (!ret_e.as<Max>()) {
      Array<Expr> max_pair;
      ret_e = Mutate(ret_e);
      max_pair.push_back(mutate_e);
      max_pair.push_back(ret_e);
      cache_max_.push_back(max_pair);
      return ret_e;
    }
  }

  string name = "max_" + to_string(max_op_count_++);
  Var x(name, op->type);
  max_map_.emplace(x.get(), op->type);
  max_child_[x] = {ret_a, ret_b};
  if (IsVarsInExpr(var_with_reduce_, ret_a - ret_b)) {
    var_with_reduce_.push_back(x);
  }
  return x;
}

Expr ExprSimplifier::Mutate_(const Cast *op, const Expr &e) {
  if (is_retrieval_) return e;
  ArithExprSimplifier simplifier(highest_cast_type_);
  // replace the min op with new variable
  for (const auto &kv : cast_child_) {
    if (Equal(kv.second, op->value) && op->type == cast_map_[kv.first.get()]) {
      return kv.first;
    }
  }

  string name = "cast_" + to_string(cast_op_count_++);
  Var x(name, highest_cast_type_);
  cast_map_.emplace(x.get(), op->type);
  cast_child_[x] = {op->value};
  if (IsVarsInExpr(var_with_reduce_, op->value)) {
    var_with_reduce_.push_back(x);
  }
  return x;
}

Expr ExprSimplifier::Mutate_(const StringImm *op, const Expr &e) {
  if (is_retrieval_) return e;
  ArithExprSimplifier simplifier(highest_cast_type_);

  for (const auto &kv : string_child_) {
    if (Equal(kv.second, op->value) && op->type == string_map_[kv.first.get()]) {
      return kv.first;
    }
  }

  string name = "string_" + to_string(string_op_count_++);
  Var x(name, op->type);
  string_map_.emplace(x.get(), op->type);
  string_child_[x] = op->value;
  if (IsVarsInExpr(var_with_reduce_, op->value)) {
    var_with_reduce_.push_back(x);
  }
  return x;
}

Expr ExprSimplifier::Mutate_(const FloorDiv *op, const Expr &e) {
  auto ret_a = Mutate(op->a);
  auto ret_b = Mutate(op->b);
  if (is_retrieval_) return FloorDiv::make(ret_a, ret_b);
  // simplify each child
  ArithExprSimplifier simplifier(highest_cast_type_);
  auto expr = simplifier.DivSimplify(ret_a, ret_b);
  if (expr.defined()) {
    return expr;
  }

  for (const auto &kv : floordiv_child_) {
    if ((Equal(kv.second[0], ret_a) && Equal(kv.second[1], ret_b) && op->type == floordiv_map_[kv.first.get()]))
      return kv.first;
  }

  string name = "floorDiv_" + to_string(floor_div_op_count_++);
  Var x(name, op->type);
  floordiv_map_.emplace(x.get(), op->type);
  floordiv_child_[x] = {ret_a, ret_b};

  if (is_scale_ && op->b.as<IntImm>() && var_with_reduce_.empty()) {
    div_scale_range_[x] = {Div::make(op->a - op->b + 1, op->b), Div::make(op->a, op->b)};
  }
  if (IsVarsInExpr(var_with_reduce_, ret_a - ret_b)) {
    if (is_scale_ && op->b.as<IntImm>()) {
      div_scale_range_[x] = {Div::make(op->a - op->b + 1, op->b), Div::make(op->a, op->b)};
    }
    var_with_reduce_.push_back(x);
    divvar_with_reduce_.push_back(x);
  }
  for (const auto &kv : mod_child_) {
    if (Equal(kv.second[0], ret_a) && Equal(kv.second[1], ret_b)) {
      div_mod_pair_.emplace_back(x, kv.first);
    }
  }
  return x;
}

Expr ExprSimplifier::Mutate_(const Div *op, const Expr &e) {
  auto ret_a = Mutate(op->a);
  auto ret_b = Mutate(op->b);
  if (is_retrieval_) return Div::make(ret_a, ret_b);
  // simplify each child
  ArithExprSimplifier simplifier(highest_cast_type_);
  auto expr = simplifier.DivSimplify(ret_a, ret_b);
  if (expr.defined()) {
    return expr;
  }

  for (const auto &kv : div_child_) {
    if ((Equal(kv.second[0], ret_a) && Equal(kv.second[1], ret_b) && op->type == div_map_[kv.first.get()]))
      return kv.first;
  }

  string name = "div_" + to_string(div_op_count_++);
  Var x(name, op->type);
  div_map_.emplace(x.get(), op->type);
  div_child_[x] = {ret_a, ret_b};

  if (is_scale_ && op->b.as<IntImm>() && var_with_reduce_.empty()) {
    div_scale_range_[x] = {Div::make(op->a - op->b + 1, op->b), Div::make(op->a, op->b)};
  }

  if (IsVarsInExpr(var_with_reduce_, ret_a - ret_b)) {
    if (is_scale_ && op->b.as<IntImm>()) {
      div_scale_range_[x] = {Div::make(op->a - op->b + 1, op->b), Div::make(op->a, op->b)};
    }
    var_with_reduce_.push_back(x);
    divvar_with_reduce_.push_back(x);
  }
  for (const auto &kv : mod_child_) {
    if (Equal(kv.second[0], ret_a) && Equal(kv.second[1], ret_b)) {
      div_mod_pair_.emplace_back(x, kv.first);
    }
  }
  return x;
}

Expr ExprSimplifier::Mutate_(const Select *op, const Expr &e) {
  auto cond = Mutate(op->condition);
  auto true_value = Mutate(op->true_value);
  auto false_value = Mutate(op->false_value);
  if (is_retrieval_) return Select::make(cond, true_value, false_value);

  ArithExprSimplifier simplifier(highest_cast_type_);
  cond = simplifier.Simplify(cond);
  if (is_const(cond)) {
    if (GetIntConst(cond)) {
      return simplifier.Simplify(true_value);
    } else {
      return simplifier.Simplify(false_value);
    }
  }

  true_value = simplifier.Simplify(true_value);
  false_value = simplifier.Simplify(false_value);

  string name = "select_" + to_string(select_op_count_++);
  Var x(name, op->type);
  select_map_.emplace(x.get(), op->type);
  select_child_[x] = {cond, true_value, false_value};
  if (IsVarsInExpr(var_with_reduce_, cond + true_value + false_value)) {
    var_with_reduce_.push_back(x);
  }
  return x;
}

template <class T>
Expr ExprSimplifier::BinaryMutate(const T *op, const Expr &e) {
  auto ret_a = Mutate(op->a);
  auto ret_b = Mutate(op->b);
  if (is_retrieval_) return T::make(ret_a, ret_b);
  // simplify each child
  ArithExprSimplifier simplifier(highest_cast_type_);
  ret_a = simplifier.Simplify(ret_a);
  ret_b = simplifier.Simplify(ret_b);
  return T::make(ret_a, ret_b);
}

Expr ExprSimplifier::Mutate_(const GE *op, const Expr &e) { return BinaryMutate(op, e); }

Expr ExprSimplifier::Mutate_(const GT *op, const Expr &e) { return BinaryMutate(op, e); }

Expr ExprSimplifier::Mutate_(const LT *op, const Expr &e) { return BinaryMutate(op, e); }

Expr ExprSimplifier::Mutate_(const LE *op, const Expr &e) { return BinaryMutate(op, e); }

Expr ExprSimplifier::Mutate_(const EQ *op, const Expr &e) { return BinaryMutate(op, e); }

Expr ExprSimplifier::Mutate_(const NE *op, const Expr &e) { return BinaryMutate(op, e); }

template <class T>
Expr ExprSimplifier::BinaryBoolMutate(const T *op, const Expr &e) {
  auto ret_a = Mutate(op->a);
  auto ret_b = Mutate(op->b);
  if (is_retrieval_) return T::make(ret_a, ret_b);
  // simplify each child
  ArithExprSimplifier simplifier(op->type);
  ret_a = simplifier.Simplify(ret_a);
  ret_b = simplifier.Simplify(ret_b);
  return T::make(ret_a, ret_b);
}

Expr ExprSimplifier::Mutate_(const And *op, const Expr &e) {
  Expr new_e = BinaryBoolMutate(op, e);
  auto new_op = new_e.as<And>();
  CHECK(new_op);
  if (is_const_false(new_op->a)) {
    return new_op->a;
  } else if (is_const_false(new_op->b)) {
    return new_op->b;
  } else if (is_const_true(new_op->a)) {
    return new_op->b;
  } else if (is_const_true(new_op->b)) {
    return new_op->a;
  } else {
    string name = "and_" + to_string(and_op_count_++);
    Var x(name, op->type);
    and_map_.emplace(x.get(), op->type);
    and_child_[x] = {new_op->a, new_op->b};
    if (IsVarsInExpr(var_with_reduce_, new_op->a - new_op->b)) {
      var_with_reduce_.push_back(x);
    }
    return x;
  }
}

Expr ExprSimplifier::Mutate_(const Or *op, const Expr &e) {
  Expr new_e = BinaryBoolMutate(op, e);
  auto new_op = new_e.as<Or>();
  CHECK(new_op);
  if (is_const_true(new_op->a)) {
    return new_op->a;
  } else if (is_const_true(new_op->b)) {
    return new_op->b;
  } else if (is_const_false(new_op->a)) {
    return new_op->b;
  } else if (is_const_false(new_op->b)) {
    return new_op->a;
  } else {
    string name = "or_" + to_string(or_op_count_++);
    Var x(name, op->type);
    or_map_.emplace(x.get(), op->type);
    or_child_[x] = {new_op->a, new_op->b};
    if (IsVarsInExpr(var_with_reduce_, new_op->a - new_op->b)) {
      var_with_reduce_.push_back(x);
    }
    return x;
  }
}

Expr ExprSimplifier::Mutate_(const Not *op, const Expr &e) {
  auto ret_a = Mutate(op->a);
  if (is_retrieval_) return Not::make(ret_a);
  // simplify each child
  ArithExprSimplifier simplifier(highest_cast_type_);
  ret_a = simplifier.Simplify(ret_a);
  if (is_const(ret_a)) {
    if (is_const_false(ret_a))
      return make_const(e.type(), 1);
    else
      return make_const(e.type(), 0);
  }

  string name = "not_" + to_string(not_op_count_++);
  Var x(name, op->type);
  not_map_.emplace(x.get(), op->type);
  not_child_[x] = {ret_a};
  if (IsVarsInExpr(var_with_reduce_, ret_a)) {
    var_with_reduce_.push_back(x);
  }
  return x;
}

Expr ExprSimplifier::Mutate_(const Load *op, const Expr &e) {
  auto ret_a = Mutate(op->index);
  if (is_retrieval_) return Load::make(op->type, op->buffer_var, ret_a, op->predicate);
  // simplify each child
  ArithExprSimplifier simplifier(highest_cast_type_);
  ret_a = simplifier.Simplify(ret_a);

  load_op_count_++;
  for (const auto &kv : load_child_) {
    if ((Equal(kv.second.index, ret_a) && Equal(kv.second.buffer_var, op->buffer_var) &&
         Equal(kv.second.predicate, op->predicate) && kv.second.type == op->type)) {
      return kv.first;
    }
  }

  string name = "load_" + to_string(load_op_count_);
  Var x(name, op->type);
  load_map_.emplace(x.get(), op->type);
  load_child_[x] = {op->buffer_var, ret_a, op->predicate, op->type};
  if (IsVarsInExpr(var_with_reduce_, ret_a)) {
    var_with_reduce_.push_back(x);
  }
  return x;
}

Expr ExprSimplifier::Mutate_(const Call *op, const Expr &e) {
  if (is_retrieval_) return e;
  // simplify each child
  ArithExprSimplifier simplifier(highest_cast_type_);
  Array<Expr> ret;
  for (const auto &it : e.as<Call>()->args) {
    ret.push_back(Mutate(it));
  }

  call_op_count_++;
  string name = "call_" + to_string(call_op_count_);
  Var x(name, op->type);
  call_map_.emplace(x.get(), op->type);
  call_child_[x] = {op->type, op->name, ret, op->call_type, op->func, op->value_index};
  return x;
}

Expr ExprSimplifier::Mutate_(const FloorMod *op, const Expr &e) {
  // transform FloorMod to Mod
  return Mutate(Mod::make(op->a, op->b));
}

Expr ExprSimplifier::Mutate_(const Mod *op, const Expr &e) {
  auto ret_a = Mutate(op->a);
  auto ret_b = Mutate(op->b);
  if (is_retrieval_) return Mod::make(ret_a, ret_b);
  // simplify each child
  ArithExprSimplifier simplifier(highest_cast_type_);

  auto expr = simplifier.ModSimplify(ret_a, ret_b);
  if (expr.defined()) return expr;
  mod_op_count_++;
  for (const auto &kv : mod_child_) {
    if ((Equal(kv.second[0], ret_a) && Equal(kv.second[1], ret_b) && op->type == mod_map_[kv.first.get()])) {
      return kv.first;
    }
  }

  string name = "mod_" + to_string(mod_op_count_);
  Var x(name, op->type);
  mod_map_.emplace(x.get(), op->type);

  mod_child_[x] = {ret_a, ret_b};

  if (IsVarsInExpr(var_with_reduce_, ret_a - ret_b)) {
    var_with_reduce_.push_back(x);
  }
  for (const auto &kv : div_child_) {
    if (Equal(kv.second[0], ret_a) && Equal(kv.second[1], ret_b)) {
      div_mod_pair_.emplace_back(kv.first, x);
    }
  }
  for (const auto &kv : floordiv_child_) {
    if (Equal(kv.second[0], ret_a) && Equal(kv.second[1], ret_b)) {
      div_mod_pair_.emplace_back(kv.first, x);
    }
  }
  return x;
}

Expr ExprSimplifier::DivMutator::Mutate_(const Mul *op, const Expr &e) {
  // Transfer Floordiv into float div:
  // For int denominator, eliminate the denominator if multiplier is divisible by it, e.g. a/8 * 8 = a;
  // For other cases, keep the e;
  if (auto a_div = op->a.as<Div>()) {
    if (op->b.as<IntImm>() && a_div->b.as<IntImm>()) {
      CHECK_NE(a_div->b.as<IntImm>()->value, 0);
      if (op->b.as<IntImm>()->value % a_div->b.as<IntImm>()->value == 0) {
        Expr scale = make_const(e.type(), op->b.as<IntImm>()->value / a_div->b.as<IntImm>()->value);
        return is_const_int(scale, 1) ? a_div->a : Mul::make(a_div->a, scale);
      }
    }
  } else if (auto b_div = op->b.as<Div>()) {
    if (op->a.as<IntImm>() && b_div->b.as<IntImm>()) {
      CHECK_NE(b_div->b.as<IntImm>()->value, 0);
      if (op->a.as<IntImm>()->value % b_div->b.as<IntImm>()->value == 0) {
        Expr scale = make_const(e.type(), op->a.as<IntImm>()->value / b_div->b.as<IntImm>()->value);
        return is_const_int(scale, 1) ? b_div->a : Mul::make(b_div->a, scale);
      }
    }
  } else if (auto a_f_div = op->a.as<FloorDiv>()) {
    if (op->b.as<IntImm>() && a_f_div->b.as<IntImm>()) {
      CHECK_NE(a_f_div->b.as<IntImm>()->value, 0);
      if (op->b.as<IntImm>()->value % a_f_div->b.as<IntImm>()->value == 0) {
        Expr scale = make_const(e.type(), op->b.as<IntImm>()->value / a_f_div->b.as<IntImm>()->value);
        return is_const_int(scale, 1) ? a_f_div->a : Mul::make(a_f_div->a, scale);
      }
    }
  } else if (auto b_f_div = op->b.as<FloorDiv>()) {
    if (op->a.as<IntImm>() && b_f_div->b.as<IntImm>()) {
      CHECK_NE(b_f_div->b.as<IntImm>()->value, 0);
      if (op->a.as<IntImm>()->value % b_f_div->b.as<IntImm>()->value == 0) {
        Expr scale = make_const(e.type(), op->a.as<IntImm>()->value / b_f_div->b.as<IntImm>()->value);
        return is_const_int(scale, 1) ? b_f_div->a : Mul::make(b_f_div->a, scale);
      }
    }
  }
  return e;
}

Stmt TestReduceInequality(const ktvm::Expr &e, const Var &reduce_var, bool scale, bool getlarger) {
  Expr result = ExprSimplifier().ReduceInequality(e, reduce_var, scale, getlarger);
  Stmt ret = Evaluate::make(0);
  ret = AttrStmt::make(make_const(Int(32), 0), "ReduceInequality", result, ret);
  return ret;
}

Stmt TestSimplify(const ktvm::Expr &e) {
  Expr result = ExprSimplifier().Simplify(e);
  Stmt ret = Evaluate::make(0);
  ret = AttrStmt::make(make_const(Int(32), 0), "Simplify", result, ret);
  return ret;
}

Stmt TestCanProveWithPosParam(const ktvm::Expr &e) {
  bool res = ExprSimplifier().CanProveWithPosParam(e);
  Stmt ret = Evaluate::make(0);
  ret = AttrStmt::make(make_const(Int(32), 0), "CanProveWithParam", res, ret);
  return ret;
}

void TestExprCompuationSimplify() {
  ArithExprSimplifier alg;
  Var T_0_0("T_0_0", Int(32)), T_0_1("T_0_1", Int(32)), T_0_2("T_0_2", Int(32));
  Var a("a", Int(32)), b("b", Int(32)), c("c", Int(32));
}
}  // namespace ir
}  // namespace akg
