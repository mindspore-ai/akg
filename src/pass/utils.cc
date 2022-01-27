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
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/api_registry.h>

#include <vector>
#include <algorithm>
#include <regex>
#include <limits>
#include <cmath>
#include <map>

#include "ir_pass.h"
#include "pass/utils.h"
#include "pass/expr_alg_simplify.h"
#include "poly/tiling/tiling_algorithm.h"

namespace akg {
namespace ir {
using std::list;
using std::make_pair;
using std::max;
using std::min;

using UnorderSet = std::unordered_set<Var, NodeHash, NodeEqual>;
using UnorderMap = std::unordered_map<Var, Expr, NodeHash, NodeEqual>;

template <typename T>
std::vector<T> ArrayToVector(const Array<T> &array) {
  std::vector<T> result;
  for (auto &a : array) {
    result.push_back(a);
  }

  return result;
}

bool IsCover(const Array<Expr> &big, const Array<Expr> &small) {
  std::vector<Expr> vbig = ArrayToVector(big);
  std::vector<Expr> vsmall = ArrayToVector(small);

  std::vector<Expr> result;

  std::set_union(vbig.begin(), vbig.end(), vsmall.begin(), vsmall.end(), std::back_inserter(result), Compare);

  return result.size() == vbig.size();
}

int64_t Log2(uint64_t value) {
  int64_t ret = 0;
  while (value > 1) {
    value >>= 1;
    ret += 1;
  }
  return static_cast<int>(ret);
}

class RmEmptyEmit : public IRMutator {
 public:
  RmEmptyEmit() {}
  ~RmEmptyEmit() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_emit_insn") {
      // #AttrStmt 0
      if (Equal(op->body, Evaluate::make(0))) {
        return Evaluate::make(0);
      }
      Stmt body = this->Mutate(op->body);
      // #AttrStmt if(){}
      if (body.as<IfThenElse>()) {
        return body;
      }
      if (body.as<Block>()) {
        return body;
      }
      if (const auto attr = body.as<AttrStmt>()) {
        if (attr->attr_key == "pragma_emit_insn") return body;
      }
    }
    return IRMutator::Mutate_(op, s);
  }
};

Stmt RmEmptyEmitAttr(const Stmt stmt) { return RmEmptyEmit().Mutate(stmt); }

// NonLinear index, the components can be undefined.
struct NonLinearEqEntry {
  Expr base;
  Expr coeff;
};

/// Class IndexVarDetector is used by DetectNonLinearIndex
class IndexVarDetector : public air::ir::ExprFunctor<NonLinearEqEntry(const Expr &, const Expr &)> {
 public:
  explicit IndexVarDetector(Array<Expr> &vars) : loopVars(vars) {}
  ~IndexVarDetector() override = default;
  bool Detect(const Expr &e) {
    static_cast<void>(VisitExpr(e, e));
    if (fail_) return false;
    return true;
  }

  NonLinearEqEntry VisitExpr_(const Add *op, const Expr &e) final {
    if (fail_) return NonLinearEqEntry();
    NonLinearEqEntry a = VisitExpr(op->a, op->a);
    NonLinearEqEntry b = VisitExpr(op->b, op->b);
    NonLinearEqEntry ret;

    auto GetAddResult = [](Expr lhs, Expr rhs) {
      if (!lhs.defined()) {
        return rhs;
      } else if (!rhs.defined()) {
        return lhs;
      }
      return lhs + rhs;
    };

    // assign add op
    ret.base = GetAddResult(a.base, b.base);
    ret.coeff = GetAddResult(a.coeff, b.coeff);
    return ret;
  }

  NonLinearEqEntry VisitExpr_(const Sub *op, const Expr &e) final {
    if (fail_) return NonLinearEqEntry();
    NonLinearEqEntry a = VisitExpr(op->a, op->a);
    NonLinearEqEntry b = VisitExpr(op->b, op->b);
    NonLinearEqEntry ret;

    auto GetSubResult = [](Expr lhs, const Expr rhs) {
      if (!rhs.defined()) {
        return lhs;
      } else if (!lhs.defined()) {
        return -rhs;
      }
      return lhs - rhs;
    };

    // assign sub op
    ret.base = GetSubResult(a.base, b.base);
    ret.coeff = GetSubResult(a.coeff, b.coeff);
    return ret;
  }

  NonLinearEqEntry VisitExpr_(const Mul *op, const Expr &e) final {
    if (fail_) return NonLinearEqEntry();
    NonLinearEqEntry a = VisitExpr(op->a, op->a);
    NonLinearEqEntry b = VisitExpr(op->b, op->b);
    NonLinearEqEntry ret;

    auto GetMulResult = [](Expr lhs, Expr rhs) {
      if (!lhs.defined()) {
        return rhs;
      } else if (!rhs.defined()) {
        return lhs;
      }
      return lhs * rhs;
    };

    // assign mul op
    ret.base = GetMulResult(a.base, b.base);
    ret.coeff = GetMulResult(a.base, b.coeff);
    return ret;
  }
  NonLinearEqEntry VisitExpr_(const FloorMod *op, const Expr &e) final {
    NonLinearEqEntry ret;
    ret.coeff = make_const(op->type, 1);
    loopVars.push_back(e);
    return ret;
  }
  NonLinearEqEntry VisitExpr_(const Mod *op, const Expr &e) final {
    NonLinearEqEntry ret;
    ret.coeff = make_const(op->type, 1);
    loopVars.push_back(e);
    return ret;
  }
  NonLinearEqEntry VisitExpr_(const FloorDiv *op, const Expr &e) final {
    NonLinearEqEntry ret;
    ret.coeff = make_const(op->type, 1);
    loopVars.push_back(e);
    return ret;
  }
  NonLinearEqEntry VisitExpr_(const Div *op, const Expr &e) final {
    NonLinearEqEntry ret;
    ret.coeff = make_const(op->type, 1);
    loopVars.push_back(e);
    return ret;
  }
  NonLinearEqEntry VisitExpr_(const Variable *op, const Expr &e) final {
    NonLinearEqEntry ret;
    ret.coeff = make_const(op->type, 1);
    loopVars.push_back(e);
    return ret;
  }
  NonLinearEqEntry VisitExprDefault_(const Node *op, const Expr &e) final {
    if (fail_) return NonLinearEqEntry();
    NonLinearEqEntry ret;
    ret.base = e;
    return ret;
  }

 private:
  Array<Expr> &loopVars;
  bool fail_{false};
};

bool IsConstVar(const Expr &var, const Array<Expr> &constVars) {
  for (Expr constVar : constVars) {
    if (Equal(var, constVar)) {
      return true;
    }
  }
  return false;
}

bool IsInVarArray(const Expr &var, const Array<Expr> &loopVars) {
  for (auto tmpVar : loopVars) {
    if (Equal(var, tmpVar)) {
      return true;
    }
  }
  return false;
}

/// Detect non-linear index, return its loop vars and related coefficient.
/// Expr contains Mod and Div will be treated as a loop var.
/// Example:
///        input index: (((((((v0 % 2)*b) + (v1/2))*8) + v2) + 1)*2)
///             return: [(v0 % 2), (v1/2), v2] and [(2*(8*b)), 16, 2, 2]
/// \param index The index to be detected.
/// \param constVars The vars should be considered as constants.
/// \return  Array<Expr>[0] is the detected loop vars.
///          Array<Expr>[1] is the detected coefficients.
Array<Array<Expr>> DetectNonLinearIndex(const Expr &index, const Array<Expr> &constVars) {
  Array<Expr> coeffs;
  Array<Expr> rawLoopVars;
  Array<Expr> loopVars;
  Array<Expr> cLoopVars;
  Array<Expr> tmpCLoopVars;
  Array<Array<Expr>> res;

  Var tmpVar;
  Expr newE = index;
  Expr newCoeff;
  Array<Var> pureVars;
  if (!IndexVarDetector(rawLoopVars).Detect(index)) {
    return Array<Array<Expr>>();
  }

  int tmpVarIndex = 0;
  std::string tmpVarNameBase = "tmp";
  std::string tmpVarName;
  for (auto cVar : rawLoopVars) {
    // check const var expr, which is used in offset
    if (IsConstVar(cVar, constVars)) {
      if (cVar->IsInstance<Variable>()) {
        continue;
      } else {
        tmpVarName = tmpVarNameBase + std::to_string(tmpVarIndex);
        tmpVar = Var(tmpVarName);
        newE = substitute(cVar, tmpVar, newE);
        cLoopVars.push_back(cVar);
        tmpCLoopVars.push_back(tmpVar);
        ++tmpVarIndex;
        continue;
      }
    }
    // check repeat var expr
    if (IsInVarArray(cVar, loopVars)) {
      continue;
    }
    // get pureVars
    loopVars.push_back(cVar);
    if (cVar->IsInstance<Variable>()) {
      pureVars.push_back(Downcast<Var>(cVar));
    } else {
      tmpVarName = tmpVarNameBase + std::to_string(tmpVarIndex);
      tmpVar = Var(tmpVarName);
      newE = substitute(cVar, tmpVar, newE);
      pureVars.push_back(tmpVar);
      ++tmpVarIndex;
    }
  }

  coeffs = air::arith::DetectLinearEquation(newE, pureVars);
  // detect error, such as a * a * 16
  if (coeffs.empty()) {
    return Array<Array<Expr>>();
  } else if (!cLoopVars.empty()) {
    for (uint32_t coeffId = 0; coeffId < coeffs.size(); ++coeffId) {
      if (coeffs[coeffId].as<IntImm>() || coeffs[coeffId].as<UIntImm>()) continue;

      newCoeff = coeffs[coeffId];

      for (uint32_t i = 0; i < cLoopVars.size(); ++i) {
        newCoeff = substitute(tmpCLoopVars[i], cLoopVars[i], newCoeff);
      }

      coeffs.Set(coeffId, Simplify(newCoeff));
    }
  }

  res.push_back(loopVars);
  res.push_back(coeffs);

  return res;
}

class GetLinearCoefOfVarMutator : public IRMutator {
 public:
  explicit GetLinearCoefOfVarMutator(const Var &var) : var_(var) {}
  ~GetLinearCoefOfVarMutator() override = default;

 private:
  Expr Mutate_(const Variable *op, const Expr &e) final { return make_const(op->type, (op == var_.get() ? 1 : 0)); }

  Expr Mutate_(const IntImm *op, const Expr &e) final { return make_const(op->type, 0); }
  Expr Mutate_(const UIntImm *op, const Expr &e) final { return make_const(op->type, 0); }
  Expr Mutate_(const FloatImm *op, const Expr &e) final { return make_const(op->type, 0); }

  Expr Mutate_(const Add *op, const Expr &e) final {
    auto a = Mutate(op->a);
    auto b = Mutate(op->b);
    if (!a.defined() || !b.defined()) {
      return Expr();
    }
    return Simplify(a + b);
  }

  Expr Mutate_(const Sub *op, const Expr &e) final {
    auto a = Mutate(op->a);
    auto b = Mutate(op->b);
    if (!a.defined() || !b.defined()) {
      return Expr();
    }
    return Simplify(a - b);
  }

  Expr Mutate_(const Mul *op, const Expr &e) final {
    auto a = Mutate(op->a);
    auto b = Mutate(op->b);
    if (!a.defined() || !b.defined()) {
      return Expr();
    }
    if (is_const_int(a, 0) && is_const_int(b, 0)) {
      return 0;
    }
    if (!is_const_int(a, 0)) {
      return Simplify(a * op->b);
    }
    if (!is_const_int(b, 0)) {
      return Simplify(b * op->a);
    }
    return Expr();
  }

  template <class DivLike>
  Expr DivMutate(const DivLike *op, const Expr &e) {
    auto a = Mutate(op->a);
    auto b = Mutate(op->b);
    if (!a.defined() || !b.defined()) {
      return Expr();
    }
    if (!is_const_int(b, 0)) {
      return Expr();
    }
    return Simplify(DivLike::make(a, op->b));
  }

  Expr Mutate_(const Div *op, const Expr &e) final { return DivMutate(op, e); }
  Expr Mutate_(const FloorDiv *op, const Expr &e) final { return DivMutate(op, e); }
  Expr Mutate_(const Mod *op, const Expr &e) final { return DivMutate(op, e); }
  Expr Mutate_(const FloorMod *op, const Expr &e) final { return DivMutate(op, e); }

  Expr Mutate_(const Load *op, const Expr &e) final {
    auto index = Mutate(op->index);
    if (!index.defined() || !is_const_int(index, 0)) {
      return Expr();
    }
    return make_const(op->type, 0);
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    for (auto arg : op->args) {
      auto index = Mutate(arg);
      if (!index.defined() || !is_const_int(index, 0)) {
        return Expr();
      }
    }
    return make_const(op->type, 0);
  }

  template <class UnknownOp>
  Expr UnknownMutate(const UnknownOp *op, const Expr &e) {
    auto a = Mutate(op->a);
    auto b = Mutate(op->b);
    if (!a.defined() || !b.defined()) {
      return Expr();
    }
    if (is_const_int(a, 0) && is_const_int(b, 0)) {
      return make_const(op->type, 0);
    }
    return Expr();
  }

  Expr Mutate_(const Min *op, const Expr &e) { return UnknownMutate(op, e); }
  Expr Mutate_(const Max *op, const Expr &e) { return UnknownMutate(op, e); }
  Expr Mutate_(const Select *op, const Expr &e) { return Expr(); }

  const Var &var_;
};

Expr GetLinearCoefOfVar(const Expr &e, const Var &var) {
  // cannot get linear coef of bool expr
  if (e.type().is_bool()) return Expr();

  ExprSimplifier simplifier;
  return GetLinearCoefOfVarMutator(var).Mutate(simplifier.Simplify(e));
}

class CTensorSubstitute : public IRMutator {
 public:
  CTensorSubstitute(const FunctionRef &a, const FunctionRef &b, int b_value_index)
      : a_{a}, b_{b}, b_value_index_{b_value_index} {}
  ~CTensorSubstitute() override = default;
  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->call_type == Call::Halide) {
      if (op->func.defined() && op->func == a_) {
        return Call::make(op->type, b_->func_name(), op->args, Call::CallType::Halide, b_, b_value_index_);
      }
    }
    return IRMutator::Mutate_(op, e);
  }
  Stmt Mutate_(const ProducerConsumer *op, const Stmt &s) final {
    if (op->func.defined() && op->func == a_) {
      auto body = this->Mutate(op->body);
      return ProducerConsumer::make(b_, op->is_producer, body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (op->func.defined() && op->func == a_) {
      auto value = this->Mutate(op->value);
      return Provide::make(b_, b_value_index_, value, op->args);
    }
    return IRMutator::Mutate_(op, s);
  }
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->node == a_) {
      CHECK(op->body.defined());
      auto body = this->Mutate(op->body);
      return AttrStmt::make(b_, op->attr_key, op->value, body);
    }
    return IRMutator::Mutate_(op, s);
  }
  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    if (op->func.defined() && op->func == a_) {
      auto body = this->Mutate(op->body);
      return Realize::make(b_, b_value_index_, op->type, op->bounds, op->condition, body);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  FunctionRef a_, b_;
  int b_value_index_{0};
};

class CTensorStringSubstitute : public IRMutator {
 public:
  CTensorStringSubstitute(const std::string &a, const FunctionRef b, int b_value_index)
      : a_{a}, b_{b}, b_value_index_{b_value_index} {}
  ~CTensorStringSubstitute() override = default;
  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->call_type == Call::Halide) {
      if (op->func.defined() && op->func->func_name() == a_) {
        return Call::make(op->type, b_->func_name(), op->args, Call::CallType::Halide, b_, b_value_index_);
      }
    }
    return IRMutator::Mutate_(op, e);
  }
  Stmt Mutate_(const ProducerConsumer *op, const Stmt &s) final {
    if (op->func.defined() && op->func->func_name() == a_) {
      auto body = this->Mutate(op->body);
      return ProducerConsumer::make(b_, op->is_producer, body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (op->func.defined() && op->func->func_name() == a_) {
      auto value = this->Mutate(op->value);
      return Provide::make(b_, b_value_index_, value, op->args);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  std::string a_;
  FunctionRef b_;
  int b_value_index_{0};
};

Stmt TensorSubstitute(const Stmt &stmt, const FunctionRef &a, const FunctionRef &b, int b_value_index) {
  return CTensorSubstitute(a, b, b_value_index).Mutate(stmt);
}

Stmt TensorStringSubstitute(const Stmt &stmt, const std::string &a, const FunctionRef &b, int b_value_index) {
  return CTensorStringSubstitute(a, b, b_value_index).Mutate(stmt);
}

Expr TensorSubstitute(const Expr &e, const FunctionRef &a, const FunctionRef &b, int b_value_index) {
  return CTensorSubstitute(a, b, b_value_index).Mutate(e);
}

Expr TensorStringSubstitute(const Expr &e, const std::string &a, const FunctionRef &b, int b_value_index) {
  return CTensorStringSubstitute(a, b, b_value_index).Mutate(e);
}

Stmt SubstituteLoopVar(Stmt &s, const Variable *old_var, const Expr &new_var) {
  if (!old_var) {
    return s;
  }
  std::unordered_map<const Variable *, Expr> vmap;
  vmap.emplace(old_var, new_var);
  s = Substitute(s, vmap);
  return s;
}

class CheckVarInExpr : public IRVisitor {
 private:
  void Visit_(const Variable *var) override {
    if (var->name_hint == to_find) {
      found = true;
    }
  }

 public:
  CheckVarInExpr() = default;
  ~CheckVarInExpr() override = default;
  bool run(const Expr &needle, const Expr &haystack) {
    CHECK(needle.as<Variable>());
    to_find = needle.as<Variable>()->name_hint;
    found = false;
    IRVisitor::Visit(haystack);
    return found;
  }

 private:
  std::string to_find;
  bool found{false};
};

bool IsVarInExpr(const Expr &needle, const Expr &haystack) { return CheckVarInExpr().run(needle, haystack); }

bool IsVarsInExpr(const vector<Var> &vars, const Expr &haystack) {
  for (auto &var : vars) {
    if (CheckVarInExpr().run(var, haystack)) {
      return true;
    }
  }
  return false;
}

class CheckFlexVarInIf : public IRVisitor {
 private:
  void Visit_(const Variable *var) override {
    if (var->name_hint == to_find_ && (inLoad_ || inEQ_)) {
      isFlex_ = false;
    }
  }

  void Visit_(const EQ *op) override {
    inEQ_ = true;
    IRVisitor::Visit(op->a);
    IRVisitor::Visit(op->b);
    inEQ_ = false;
  }

  void Visit_(const Load *load) override {
    inLoad_ = true;
    IRVisitor::Visit_(load);
    inLoad_ = false;
  }

 public:
  bool run(const Expr &var, const Expr &expr) {
    CHECK(var.as<Variable>());
    to_find_ = var.as<Variable>()->name_hint;
    isFlex_ = true;
    IRVisitor::Visit(expr);
    return isFlex_;
  }

 private:
  std::string to_find_;
  bool isFlex_{true};
  bool inLoad_{false};
  bool inEQ_{false};
};

bool IsFlexVarInIf(const Expr &var, const Array<Stmt> &ops) {
  bool isFlexVar = true;
  for (const auto &op : ops) {
    if (op.as<IfThenElse>()) {
      isFlexVar = isFlexVar && CheckFlexVarInIf().run(var, op.as<IfThenElse>()->condition);
    }
  }
  return isFlexVar;
}

bool CanProve(const Expr &e) {
  CHECK(e.type().is_bool()) << "Argument to can_prove is not a boolean Expr: " << e << "\n";
  return is_one(Simplify(e));
}

bool IsZero(const Expr &e) {
  if (e->IsInstance<FloatImm>()) {
    return fabs(e.as<FloatImm>()->value) <= 1e-15;
  } else if (e->IsInstance<IntImm>()) {
    return e.as<IntImm>() == 0;
  } else if (e->IsInstance<UIntImm>()) {
    return e.as<UIntImm>() == 0u;
  }

  return false;
}

/// Get int value of an expr
/// \param expr - Input expr
/// \return int - Int value
int64_t GetIntConst(const Expr &expr) {
  CHECK(expr.defined());
  // Check the expr should be either IntImm or UIntImm
  if (!(expr->IsInstance<IntImm>() || expr->IsInstance<UIntImm>() || expr->IsInstance<FloatImm>())) {
    LOG(FATAL) << "\n\n" << expr << "is not a const\n";
  }

  if (expr->IsInstance<IntImm>()) {
    return expr.as<IntImm>()->value;
  } else if (expr->IsInstance<UIntImm>()) {
    return static_cast<int64_t>(expr.as<UIntImm>()->value);
  }

  // otherwise it is always a FloatImm
  return static_cast<int64_t>(expr.as<FloatImm>()->value);
}

double GetFloatConst(const Expr &expr) {
  CHECK(expr.defined());
  if (expr.as<FloatImm>()) {
    return static_cast<double>(expr.as<FloatImm>()->value);
  }
  return static_cast<double>(GetIntConst(expr));
}

int GetInt32Const(const Expr &expr) { return static_cast<int>(GetIntConst(expr)); }

/// Get uint value of an expr
/// \param expr  - Input expr
/// \return uint - Uint value
uint64_t GetUIntConst(const Expr &expr) {
  CHECK(expr.as<UIntImm>());
  return expr.as<UIntImm>()->value;
}

std::ostream &operator<<(std::ostream &os, const Bound &bound) {
  os << "Bound(min=" << bound.min << ", max=" << bound.max << ")";
  return os;
}

class InferBoundOfExprClass {
 public:
  Bound infer_range(const Expr &expr) {
    CHECK(expr.defined()) << "Cannot infer range of undefined expr.";
    if (expr.as<IntImm>() || expr.as<UIntImm>() || expr.as<FloatImm>()) {
      return Bound::make(expr, expr);
    } else if (const auto var = expr.as<Variable>()) {
      if (binds.count(var) > 0) {
        Bound var_min_range = infer_range(binds[var].min);
        Bound var_max_range = infer_range(binds[var].max);
        if (!var_max_range.defined() || !var_min_range.defined()) return Bound();
        return Bound::make(var_min_range.min, var_max_range.max);
      } else {
        return Bound::make(expr, expr);
      }
    } else if (const auto add = expr.as<Add>()) {
      Bound bound_a = infer_range(add->a);
      Bound bound_b = infer_range(add->b);
      if (!bound_a.defined() || !bound_b.defined()) return Bound();
      return Bound::make(Simplify(bound_a.min + bound_b.min), Simplify(bound_a.max + bound_b.max));
    } else if (const auto sub = expr.as<Sub>()) {
      Bound bound_a = infer_range(sub->a);
      Bound bound_b = infer_range(sub->b);
      if (!bound_a.defined() || !bound_b.defined()) return Bound();
      return Bound::make(Simplify(bound_a.min - bound_b.max), Simplify(bound_a.max - bound_b.min));
    } else if (const auto mul = expr.as<Mul>()) {
      Bound bound_a = infer_range(mul->a);
      Bound bound_b = infer_range(mul->b);
      if (!bound_a.defined() || !bound_b.defined()) return Bound();
      Bound bound;
      if (CanProve(bound_a.min >= 0) && CanProve(bound_b.min >= 0)) {
        bound.min = Simplify(bound_a.min * bound_b.min);
      } else {
        bound.min = expr;
      }
      if (CanProve(bound_a.max >= 0) && CanProve(bound_b.max >= 0)) {
        bound.max = Simplify(bound_a.max * bound_b.max);
      } else {
        bound.max = expr;
      }
      return bound;
    } else if (const auto f_div = expr.as<FloorDiv>()) {
      Bound bound_a = infer_range(f_div->a);
      Bound bound_b = infer_range(f_div->b);
      if (!bound_a.defined() || !bound_b.defined()) return Bound();
      Bound bound;
      if (CanProve(bound_a.min >= 0) && CanProve(bound_b.max > 0)) {
        bound.min = Simplify(floordiv(bound_a.min, bound_b.max));
      } else {
        bound.min = expr;
      }
      if (CanProve(bound_a.max >= 0) && CanProve(bound_b.min > 0)) {
        bound.max = Simplify(floordiv(bound_a.max, bound_b.min));
      } else {
        bound.max = expr;
      }
      return bound;
    } else if (const auto div = expr.as<Div>()) {
      Bound bound_a = infer_range(div->a);
      Bound bound_b = infer_range(div->b);
      if (!bound_a.defined() || !bound_b.defined()) return Bound();
      Bound bound;
      if (CanProve(bound_a.min >= 0) && CanProve(bound_b.max > 0)) {
        bound.min = Simplify(floordiv(bound_a.min, bound_b.max));
      } else {
        bound.min = expr;
      }
      if (CanProve(bound_a.max >= 0) && CanProve(bound_b.min > 0)) {
        bound.max = Simplify(floordiv(bound_a.max, bound_b.min));
      } else {
        bound.max = expr;
      }
      return bound;
    } else if (const auto min_expr = expr.as<Min>()) {
      Bound bound_a = infer_range(min_expr->a);
      Bound bound_b = infer_range(min_expr->b);
      if (!bound_a.defined() || !bound_b.defined()) return Bound();
      return Bound::make(Simplify(min(bound_a.min, bound_b.min)), Simplify(min(bound_a.max, bound_b.max)));
    } else if (const auto max_expr = expr.as<Max>()) {
      Bound bound_a = infer_range(max_expr->a);
      Bound bound_b = infer_range(max_expr->b);
      if (!bound_a.defined() || !bound_b.defined()) return Bound();
      return Bound::make(Simplify(max(bound_a.min, bound_b.min)), Simplify(max(bound_a.max, bound_b.max)));
    } else {
      LOG(INFO) << "constrain expr is invalid " << expr;
    }
    return {};
  }

  Bound run(const Expr &expr, const std::unordered_map<const Variable *, Range> &dom_map) {
    set_dom_map(dom_map);
    return infer_range(expr);
  }

  Bound run(const Expr &expr, const std::unordered_map<const Variable *, Bound> &dom_map) {
    for (auto bind : dom_map) {
      binds.emplace(bind.first, bind.second);
    }
    return infer_range(expr);
  }

  void set_dom_map(const std::unordered_map<const Variable *, Range> &dom_map) {
    for (auto bind : dom_map) {
      binds.emplace(bind.first, Bound::make(bind.second));
    }
  }

 private:
  std::unordered_map<const Variable *, Bound> binds;
};

Bound InferBoundOfExpr(const Expr &expr, const std::unordered_map<const Variable *, Range> &dom_map) {
  return InferBoundOfExprClass().run(expr, dom_map);
}

class SimplifyConditionExprClass {
 public:
  Expr run(const Expr &expr, const std::unordered_map<const Variable *, Range> &var_bound_map) {
    bound_checker.set_dom_map(var_bound_map);
    return simplify(expr);
  }

 private:
  Expr simplify(const Expr &expr) {
    if (auto and_op = expr.as<And>()) {
      Expr a = simplify(and_op->a);
      Expr b = simplify(and_op->b);
      if (is_const_true(a) && is_const_true(b)) {
        return true;
      } else if (is_const_false(a) || is_const_false(b)) {
        return false;
      }
    } else if (auto or_op = expr.as<Or>()) {
      Expr a = simplify(or_op->a);
      Expr b = simplify(or_op->b);
      if (is_const_true(a) || is_const_true(b)) {
        return true;
      } else if (is_const_false(a) && is_const_false(b)) {
        return false;
      }
    } else if (auto not_op = expr.as<Not>()) {
      Expr a = simplify(not_op->a);
      if (is_const_true(a)) {
        return false;
      } else if (is_const_false(a)) {
        return true;
      }
    } else if (auto lt = expr.as<LT>()) {
      Bound lhs = bound_checker.infer_range(lt->a);
      Bound rhs = bound_checker.infer_range(lt->b);
      if (CanProve(lhs.max < rhs.min)) {
        return true;
      } else if (CanProve(lhs.min >= rhs.max)) {
        return false;
      }
    } else if (auto le = expr.as<LE>()) {
      Bound lhs = bound_checker.infer_range(le->a);
      Bound rhs = bound_checker.infer_range(le->b);
      if (CanProve(lhs.max <= rhs.min)) {
        return true;
      } else if (CanProve(lhs.min > rhs.max)) {
        return false;
      }
    } else if (auto gt = expr.as<GT>()) {
      Bound lhs = bound_checker.infer_range(gt->a);
      Bound rhs = bound_checker.infer_range(gt->b);
      if (CanProve(lhs.max > rhs.min)) {
        return true;
      } else if (CanProve(lhs.min <= rhs.max)) {
        return false;
      }
    } else if (auto ge = expr.as<GE>()) {
      Bound lhs = bound_checker.infer_range(ge->a);
      Bound rhs = bound_checker.infer_range(ge->b);
      if (CanProve(lhs.max >= rhs.min)) {
        return true;
      } else if (CanProve(lhs.min < rhs.max)) {
        return false;
      }
    } else if (auto eq = expr.as<EQ>()) {
      Bound lhs = bound_checker.infer_range(eq->a);
      Bound rhs = bound_checker.infer_range(eq->b);
      if (CanProve(lhs.max == rhs.min)) {
        return true;
      } else if (CanProve(lhs.max != rhs.min)) {
        return false;
      }
    } else if (auto ne = expr.as<NE>()) {
      Bound lhs = bound_checker.infer_range(ne->a);
      Bound rhs = bound_checker.infer_range(ne->b);
      if (CanProve(lhs.max != rhs.min)) {
        return true;
      } else if (CanProve(lhs.max == rhs.min)) {
        return false;
      }
    }

    return Simplify(expr);
  }

  InferBoundOfExprClass bound_checker;
};

Expr SimplifyConditionExpr(const Expr &expr, const std::unordered_map<const Variable *, Range> &var_bound_map) {
  return SimplifyConditionExprClass().run(expr, var_bound_map);
}

int GetRangeWithParam(const Expr &expr) {
  if (expr.as<IntImm>() || expr.as<UIntImm>() || expr.as<FloatImm>()) {
    if (CanProve(expr == 0))
      return static_cast<int>(Interval::ZERO);
    else if (CanProve(expr > 0))
      return static_cast<int>(Interval::GTZERO);
    else
      return static_cast<int>(Interval::LTZERO);
  } else if (expr.as<Variable>()) {
    // we assume all the variables are non-negative values
    return static_cast<int>(Interval::GEZERO);
  } else if (auto add = expr.as<Add>()) {
    auto sign_a = GetRangeWithParam(add->a);
    auto sign_b = GetRangeWithParam(add->b);
    if (sign_a == 0 || sign_b == 0) {
      return sign_b == 0 ? sign_a : sign_b;
    } else if (sign_a * sign_b > 0) {
      return sign_a > 0 ? (sign_a + sign_b + 1) / 2 : (sign_a + sign_b - 1) / 2;
    } else {
      return static_cast<int>(Interval::UNKNOWN);
    }
  } else if (auto sub = expr.as<Sub>()) {
    auto sign_a = GetRangeWithParam(sub->a);
    auto sign_b = GetRangeWithParam(sub->b);
    if (sign_a == 0 || sign_b == 0) {
      return sign_b == 0 ? sign_a : sign_b * -1;
    } else if (sign_a * sign_b < 0) {
      return sign_a > 0 ? (sign_a - sign_b + 1) / 2 : (sign_a - sign_b - 1) / 2;
    } else {
      return static_cast<int>(Interval::UNKNOWN);
    }
  } else if (auto mul = expr.as<Mul>()) {
    auto sign_a = GetRangeWithParam(mul->a);
    auto sign_b = GetRangeWithParam(mul->b);
    if (sign_a == 0 || sign_b == 0) {
      return static_cast<int>(Interval::ZERO);
    } else if (sign_a * sign_b > 0) {
      return (sign_a * sign_b + 1) / 2;
    } else if (sign_a * sign_b < 0) {
      return (sign_a * sign_b - 1) / 2;
    } else {
      return static_cast<int>(Interval::UNKNOWN);
    }
  } else if (auto div = expr.as<Div>()) {
    auto sign_a = GetRangeWithParam(div->a);
    auto sign_b = GetRangeWithParam(div->b);
    CHECK(sign_b != 0) << "cannot divide by zero: ";
    if (sign_a == static_cast<int>(Interval::ZERO)) {
      return static_cast<int>(Interval::ZERO);
    } else if (sign_a * sign_b > 0) {
      return (sign_a * sign_b + 1) / 2;
    } else if (sign_a * sign_b < 0) {
      return (sign_a * sign_b - 1) / 2;
    } else {
      return static_cast<int>(Interval::UNKNOWN);
    }
  } else if (auto f_div = expr.as<FloorDiv>()) {
    auto sign_a = GetRangeWithParam(f_div->a);
    auto sign_b = GetRangeWithParam(f_div->b);
    CHECK(sign_b != 0) << "cannot divide by zero: ";
    if (sign_a == static_cast<int>(Interval::ZERO)) {
      return static_cast<int>(Interval::ZERO);
    } else if (sign_a * sign_b > 0) {
      return (sign_a * sign_b + 1) / 2;
    } else if (sign_a * sign_b < 0) {
      return (sign_a * sign_b - 1) / 2;
    } else {
      return static_cast<int>(Interval::UNKNOWN);
    }
  } else if (auto min_op = expr.as<Min>()) {
    auto sign_a = GetRangeWithParam(min_op->a);
    auto sign_b = GetRangeWithParam(min_op->b);
    if (sign_a != static_cast<int>(Interval::UNKNOWN) && sign_b != static_cast<int>(Interval::UNKNOWN)) {
      return min(sign_a, sign_b);
    } else if (sign_a == static_cast<int>(Interval::LTZERO) || sign_b == static_cast<int>(Interval::LTZERO)) {
      return static_cast<int>(Interval::LTZERO);
    } else if (min_op->a.as<IntImm>() && sign_b == static_cast<int>(Interval::UNKNOWN)) {
      return sign_a;
    } else if (min_op->b.as<IntImm>() && sign_a == static_cast<int>(Interval::UNKNOWN)) {
      return sign_b;
    } else {
      return static_cast<int>(Interval::UNKNOWN);
    }
  } else if (auto max_op = expr.as<Max>()) {
    auto sign_a = GetRangeWithParam(max_op->a);
    auto sign_b = GetRangeWithParam(max_op->b);
    if (sign_a != static_cast<int>(Interval::UNKNOWN) && sign_b != static_cast<int>(Interval::UNKNOWN)) {
      return max(sign_a, sign_b);
    } else if (sign_a == static_cast<int>(Interval::GTZERO) || sign_b == static_cast<int>(Interval::GTZERO)) {
      return static_cast<int>(Interval::GTZERO);
    } else if (max_op->a.as<IntImm>() && sign_b == static_cast<int>(Interval::UNKNOWN)) {
      return sign_a;
    } else if (max_op->b.as<IntImm>() && sign_a == static_cast<int>(Interval::UNKNOWN)) {
      return sign_b;
    } else {
      return static_cast<int>(Interval::UNKNOWN);
    }
  } else if (expr.as<Mod>()) {
    return static_cast<int>(Interval::UNKNOWN);
  } else if (expr.as<Call>()) {
    return static_cast<int>(Interval::UNKNOWN);
  } else {
    LOG(INFO) << "cannot deal with the computation type: " << expr;
  }
  return static_cast<int>(Interval::UNKNOWN);
}

int GetSign(const Expr &bound) {
  auto b_range = GetRangeWithParam(bound);
  if (b_range == 0) {
    return static_cast<int>(Sign::ZERO);
  } else if (b_range > 0 && b_range != static_cast<int>(Interval::UNKNOWN)) {
    return static_cast<int>(Sign::POS);
  } else if (b_range < 0) {
    return static_cast<int>(Sign::NEG);
  }
  return static_cast<int>(Sign::UNK);
}

UnorderSet IntersectionSet(const UnorderSet &vars_set, const UnorderSet &vars) {
  UnorderSet intersection;
  for (auto cur_var : vars) {
    if (vars_set.find(cur_var) != vars_set.end()) {
      intersection.emplace(cur_var);
    }
  }
  return intersection;
}

UnorderSet DifferenceSet(const UnorderSet &vars_set, const UnorderSet &vars) {
  UnorderSet diffset;
  for (auto cur_var : vars) {
    if (vars_set.find(cur_var) == vars_set.end()) {
      diffset.emplace(cur_var);
    }
  }
  return diffset;
}

class InferBoundOfExprWithCondClass {
 public:
  bool CheckConstExpr(const Expr &expr) {
    if (expr.as<IntImm>() || expr.as<UIntImm>() || expr.as<FloatImm>()) {
      return true;
    }
    return false;
  }

  bool NotPolynomial(const Expr &expr) {
    if (expr.as<Mul>() || expr.as<Variable>()) {
      return true;
    }
    return false;
  }

  template <typename T>
  Expr GetRecurRes(const T *op, const Expr &constraint, const UnorderSet &vars_set) {
    Expr l_subexpr = RecurTarExpr(op->a, constraint, vars_set);
    Expr r_subexpr = RecurTarExpr(op->b, constraint, vars_set);
    if (!Equal(l_subexpr, constraint) && !Equal(r_subexpr, constraint)) {
      UnorderSet lset;
      UnorderSet rset;
      GatherVars(l_subexpr, &lset);
      GatherVars(r_subexpr, &rset);
      if (IntersectionSet(lset, vars_set).size() > IntersectionSet(rset, vars_set).size())
        return l_subexpr;
      else if (IntersectionSet(lset, vars_set).size() < IntersectionSet(rset, vars_set).size())
        return r_subexpr;
      else
        return l_subexpr;
    } else if (!Equal(l_subexpr, constraint)) {
      return l_subexpr;
    } else if (!Equal(r_subexpr, constraint)) {
      return r_subexpr;
    }
    return constraint;
  }
  Expr RecurTarExpr(const Expr &expr, const Expr &constraint, const UnorderSet &vars_set) {
    Expr substitute_expr = DetectSubstituteExpr(expr, constraint, vars_set);
    if (!Equal(substitute_expr, constraint)) {
      return substitute_expr;
    }
    if (expr.as<Add>()) {
      return GetRecurRes(expr.as<Add>(), constraint, vars_set);
    } else if (expr.as<Sub>()) {
      return GetRecurRes(expr.as<Sub>(), constraint, vars_set);
    } else if (expr.as<Mul>()) {
      return GetRecurRes(expr.as<Mul>(), constraint, vars_set);
    } else if (expr.as<Div>()) {
      return GetRecurRes(expr.as<Div>(), constraint, vars_set);
    } else if (expr.as<FloorDiv>()) {
      return GetRecurRes(expr.as<FloorDiv>(), constraint, vars_set);
    } else {
      return constraint;
    }
  }

  bool IsPoly(const Expr &expr) {
    bool poly = true;
    if (!expr.as<Add>() && !expr.as<Sub>()) {
      poly = false;
    }
    PostOrderVisit(expr, [&poly](const NodeRef &node) {
      if (!node.as<Mul>() && !node.as<Add>() && !node.as<IntImm>() && !node.as<Variable>() && !node.as<Sub>()) {
        poly = false;
      }
    });
    return poly;
  }

  Expr PolySubstitute(const Expr &expr, const Expr &common, const Var &var) {
    if (Equal(expr, common)) {
      return var;
    }
    if (expr.as<Add>()) {
      Expr add_a = expr.as<Add>()->a;
      Expr add_b = expr.as<Add>()->b;
      return Add::make(PolySubstitute(add_a, common, var), PolySubstitute(add_b, common, var));
    } else if (expr.as<Mul>()) {
      Expr mul_a = expr.as<Mul>()->a;
      Expr mul_b = expr.as<Mul>()->b;
      return Mul::make(PolySubstitute(mul_a, common, var), PolySubstitute(mul_b, common, var));
    } else {
      return expr;
    }
  }

  Expr DetectSubstituteExpr(const Expr &expr, const Expr &constraint, const UnorderSet &vars_set) {
    // expr is the target expression, constraint is the source expression. We will rewrite constraint if detect expr.
    // recursively match expr or part of expr with constraint. If match success, return;
    if (!CheckConstExpr(ExprSimplifier().Simplify(expr)) && !CheckConstExpr(ExprSimplifier().Simplify(constraint))) {
      if ((expr.as<Variable>() && vars_set.count(Downcast<Var>(expr))) || !expr.as<Variable>()) {
        air::DataType dtype = expr.type();
        if (Equal(expr, constraint)) {
          std::unordered_map<const Variable *, Expr>::iterator iter;
          iter = std::find_if(substitute_map.begin(), substitute_map.end(),
                              [&expr](const std::unordered_map<const Variable *, Expr>::value_type &vt) {
                                return Equal(vt.second, expr);
                              });
          if (iter == substitute_map.end()) {
            Var new_var("v_" + to_string(substitute_map.size()), dtype);
            substitute_map.emplace(new_var.get(), expr);
            var_map.emplace(new_var.get(), new_var);
            return Expr(new_var);
          } else {
            return var_map[iter->first];
          }
        }
        if (NotPolynomial(expr) && CheckConstExpr(ExprSimplifier().Simplify(div(constraint, expr)))) {
          std::unordered_map<const Variable *, Expr>::iterator iter;
          iter = std::find_if(substitute_map.begin(), substitute_map.end(),
                              [&expr](const std::unordered_map<const Variable *, Expr>::value_type &vt) {
                                return Equal(vt.second, expr);
                              });
          if (iter == substitute_map.end()) {
            Var new_var("v_" + to_string(substitute_map.size()), dtype);
            substitute_map.emplace(new_var.get(), expr);
            var_map.emplace(new_var.get(), new_var);
            return Mul::make(Expr(new_var), ExprSimplifier().Simplify(div(constraint, expr)));
          } else {
            return Mul::make(var_map[iter->first], ExprSimplifier().Simplify(div(constraint, expr)));
          }
        }
        UnorderSet expr_var;
        UnorderSet cond_var;
        GatherVars(expr, &expr_var);
        GatherVars(constraint, &cond_var);
        if (IsPoly(expr) && IsPoly(constraint) && expr_var.size() > 1 && cond_var.size() == expr_var.size()) {
          Array<Expr> exprs = ExprSimplifier().GetPolynomial(expr, constraint);
          if (!Equal(exprs[0], expr) && (exprs[0].as<Mul>() || exprs[0].as<Add>()->a.as<Mul>())) {
            detectpoly = true;
            polyexpr = exprs[0];
            Expr common = exprs[0].as<Mul>() ? exprs[0].as<Mul>()->b : exprs[0].as<Add>()->a.as<Mul>()->b;
            std::unordered_map<const Variable *, Expr>::iterator iter;
            iter = std::find_if(substitute_map.begin(), substitute_map.end(),
                                [&common](const std::unordered_map<const Variable *, Expr>::value_type &vt) {
                                  return Equal(vt.second, common);
                                });
            if (iter == substitute_map.end()) {
              Var new_var("v_" + to_string(substitute_map.size()), dtype);
              substitute_map.emplace(new_var.get(), common);
              var_map.emplace(new_var.get(), new_var);
              return PolySubstitute(exprs[1], common, new_var);
            } else {
              return PolySubstitute(exprs[1], common, var_map[iter->first]);
            }
          }
        }
      }
    }
    if (constraint.as<IntImm>() || constraint.as<UIntImm>() || constraint.as<FloatImm>() || constraint.as<Variable>()) {
      return constraint;
    } else if (auto add = constraint.as<Add>()) {
      Expr expr_a = DetectSubstituteExpr(expr, add->a, vars_set);
      Expr expr_b = DetectSubstituteExpr(expr, add->b, vars_set);
      return Add::make(expr_a, expr_b);
    } else if (auto sub = constraint.as<Sub>()) {
      Expr expr_a = DetectSubstituteExpr(expr, sub->a, vars_set);
      Expr expr_b = DetectSubstituteExpr(expr, sub->b, vars_set);
      return Sub::make(expr_a, expr_b);
    } else if (auto mul = constraint.as<Mul>()) {
      Expr expr_a = DetectSubstituteExpr(expr, mul->a, vars_set);
      Expr expr_b = DetectSubstituteExpr(expr, mul->b, vars_set);
      return Mul::make(expr_a, expr_b);
    } else if (auto fdiv = constraint.as<FloorDiv>()) {
      Expr expr_a = DetectSubstituteExpr(expr, fdiv->a, vars_set);
      Expr expr_b = DetectSubstituteExpr(expr, fdiv->b, vars_set);
      return FloorDiv::make(expr_a, expr_b);
    } else if (auto div = constraint.as<Div>()) {
      Expr expr_a = DetectSubstituteExpr(expr, div->a, vars_set);
      Expr expr_b = DetectSubstituteExpr(expr, div->b, vars_set);
      return Div::make(expr_a, expr_b);
    } else {
      return constraint;
    }
  }

  void InsertPair(const pair<const Expr, Bound> &cur_pair) {
    std::vector<pair<const Expr, Bound>>::iterator iter;
    iter = std::find_if(
      conds_var_combine.begin(), conds_var_combine.end(),
      [&cur_pair](const pair<const Expr, Bound> &element) { return Equal(element.first, cur_pair.first); });
    if (iter == conds_var_combine.end()) {
      conds_var_combine.push_back(cur_pair);
    } else {
      iter->second = cur_pair.second;
    }
  }

  void VisitCmpExpr(const EQ *op, const UnorderSet &vars_set) {
    Expr lexpr = op->a;
    Expr rexpr = op->b;
    if (lexpr.as<Variable>()) {
      if (conds_var.count(lexpr.as<Variable>()) == 0) {
        conds_var.emplace(lexpr.as<Variable>(), Bound::make(rexpr, rexpr));
      } else {
        conds_var[lexpr.as<Variable>()] =
          GetTightBound(lexpr, Bound::make(rexpr, rexpr), conds_var[lexpr.as<Variable>()]);
      }
    } else {
      pair<const Expr, Bound> cur_pair = make_pair(lexpr, Bound::make(rexpr, rexpr));
      InsertPair(cur_pair);
    }
  }

  void VisitCmpExpr(const LE *op, const UnorderSet &vars_set) {
    Expr lexpr = op->a;
    Expr rexpr = op->b;
    if (lexpr.as<Variable>()) {
      if (conds_var.count(lexpr.as<Variable>()) == 0) {
        conds_var.emplace(lexpr.as<Variable>(), Bound::make(0, rexpr));
      } else {
        conds_var[lexpr.as<Variable>()] = GetTightBound(lexpr, Bound::make(0, rexpr), conds_var[lexpr.as<Variable>()]);
      }
    } else {
      // when lexpr is not a variable, store a bound with update tight bound.
      Bound lbound = GetExprBoundWithCond(lexpr);
      Bound onesidebound;
      onesidebound = Bound::make(lexpr, rexpr);
      Bound intersectbound = GetTightBound(lexpr, onesidebound, lbound);
      pair<const Expr, Bound> cur_pair = make_pair(lexpr, intersectbound);
      InsertPair(cur_pair);
    }
  }

  void VisitCmpExpr(const LT *op, const UnorderSet &vars_set) {
    Expr lexpr = op->a;
    Expr rexpr = op->b;
    if (lexpr.as<Variable>()) {
      if (conds_var.count(lexpr.as<Variable>()) == 0) {
        conds_var.emplace(lexpr.as<Variable>(), Bound::make(0, Simplify(rexpr - 1)));
      } else {
        conds_var[lexpr.as<Variable>()] =
          GetTightBound(lexpr, Bound::make(0, Simplify(rexpr - 1)), conds_var[lexpr.as<Variable>()]);
      }
    } else {
      Bound lbound = GetExprBoundWithCond(lexpr);
      Bound onesidebound;
      onesidebound = Bound::make(lexpr, Simplify(rexpr - 1));
      Bound intersectbound = GetTightBound(lexpr, onesidebound, lbound);
      pair<const Expr, Bound> cur_pair = make_pair(lexpr, intersectbound);
      InsertPair(cur_pair);
    }
  }

  void VisitCmpExpr(const GE *op, const UnorderSet &vars_set) {
    Expr lexpr = op->a;
    Expr rexpr = op->b;
    if (lexpr.as<Variable>()) {
      if (conds_var.count(lexpr.as<Variable>()) == 0) {
        conds_var.emplace(lexpr.as<Variable>(), Bound::make(rexpr, lexpr));
      } else {
        conds_var[lexpr.as<Variable>()] =
          GetTightBound(lexpr, Bound::make(rexpr, lexpr), conds_var[lexpr.as<Variable>()]);
      }
    } else {
      Bound onesidebound;
      onesidebound = Bound::make(rexpr, lexpr);
      pair<const Expr, Bound> cur_pair = make_pair(lexpr, onesidebound);
      InsertPair(cur_pair);
    }
  }

  void VisitCmpExpr(const GT *op, const UnorderSet &vars_set) {
    Expr lexpr = op->a;
    Expr rexpr = op->b;
    if (lexpr.as<Variable>()) {
      if (conds_var.count(lexpr.as<Variable>()) == 0) {
        conds_var.emplace(lexpr.as<Variable>(), Bound::make(Simplify(rexpr + 1), lexpr));
      } else {
        conds_var[lexpr.as<Variable>()] =
          GetTightBound(lexpr, Bound::make(Simplify(rexpr + 1), lexpr), conds_var[lexpr.as<Variable>()]);
      }
    } else {
      Bound onesidebound;
      onesidebound = Bound::make(Simplify(rexpr + 1), lexpr);
      pair<const Expr, Bound> cur_pair = make_pair(lexpr, onesidebound);
      InsertPair(cur_pair);
    }
  }

  void InsertCallBound(const Expr &e) {
    PostOrderVisit(e, [this](const NodeRef &node) {
      const auto call = node.as<Call>();
      if (call == nullptr) return;
      if (call->name.find("FL") != std::string::npos) {
        if (call->name == tiling_algorithm::intrinsic::FL_find_divisible_tiling_factor) {
          CHECK_GE(call->args.size(), 2U);
          auto find_a = call->args[0];
          auto find_b = call->args[1];
          if (find_a.as<IntImm>() && find_b.as<Variable>()) {
            Expr bound_max;
            if (conds_var.count(find_b.as<Variable>()) > 0) {
              bound_max = conds_var[find_b.as<Variable>()].max;
              bound_max = GetExprBoundWithCond(Min::make(find_a, bound_max)).max;
            } else {
              bound_max = find_a;
            }
            pair<const Expr, Bound> cur_pair = make_pair(Downcast<Expr>(node), Bound::make(1, bound_max));
            InsertPair(cur_pair);
          } else if (Equal(find_a, find_b)) {
            LOG(WARNING) << "[Warning]: Get Call with same args: " << call;
          }
        } else if (call->name == tiling_algorithm::intrinsic::FL_get_gcd) {
          CHECK_GE(call->args.size(), 2U);
          auto gcd_a = call->args[0];
          auto gcd_b = call->args[1];
          if (gcd_a.as<IntImm>() && gcd_a.as<IntImm>()->value >= 1 && !gcd_b.as<IntImm>()) {
            pair<const Expr, Bound> cur_pair = make_pair(Downcast<Expr>(node), Bound::make(1, gcd_a));
            InsertPair(cur_pair);
          } else if (gcd_b.as<IntImm>() && gcd_b.as<IntImm>()->value >= 1 && !gcd_a.as<IntImm>()) {
            pair<const Expr, Bound> cur_pair = make_pair(Downcast<Expr>(node), Bound::make(1, gcd_b));
            InsertPair(cur_pair);
          }
        }
      }
    });
  }

  template <typename T>
  bool IsParaCond(const T *op, const UnorderSet &vars_set) {
    UnorderSet Paraset1;
    UnorderSet Paraset2;
    GatherVars(op->a, &Paraset1);
    GatherVars(op->b, &Paraset2);
    if (!IntersectionSet(Paraset1, vars_set).size() && !IntersectionSet(Paraset2, vars_set).size())
      return true;
    else
      return false;
  }

  template <typename T>
  Expr PreProcessExpr(const T *op, const Expr &expr, const UnorderSet &vars_set) {
    Expr ori_lexpr = op->a;
    Expr ori_rexpr = op->b;

    // Using more check to avoid recursive operations.
    if (ori_lexpr.as<Variable>()) {
      return T::make(op->a, op->b);
    }

    if (ori_rexpr.as<Min>() || ori_rexpr.as<Max>() || ori_rexpr.as<Call>()) return T::make(op->a, op->b);

    UnorderSet var_set;
    Expr cur_expr = T::make(op->a, op->b);
    GatherVars(cur_expr, &var_set);
    if (static_cast<int>(var_set.size()) == 1) {
      return ExprSimplifier().ReduceInequality(cur_expr, *var_set.begin(), true, true);
    }
    UnorderSet cond_set;
    UnorderSet lexpr_set;
    GatherVars(expr, &cond_set);
    GatherVars(ori_lexpr, &lexpr_set);
    if (IntersectionSet(lexpr_set, cond_set).size() == lexpr_set.size()) {
      return cur_expr;
    }

    Expr sub_expr = RecurTarExpr(expr, op->a - op->b, vars_set);
    Expr moving_expr = T::make(sub_expr, 0);
    if (substitute_map.size() > 0) {
      for (auto sub_var : var_map) {
        if (CheckVarInExpr().run(sub_var.second, moving_expr)) {
          moving_expr = ExprSimplifier().ReduceInequality(moving_expr, sub_var.second, true, true);
          moving_expr = Substitute(moving_expr, substitute_map);
        }
      }
    }
    return moving_expr;
  }

  void GetExprBound(const Expr &cond, const Expr &expr, const UnorderSet &vars_set) {
    if (cond.as<EQ>()) {
      Expr movexpr = PreProcessExpr(cond.as<EQ>(), expr, vars_set);
      VisitCmpExpr(movexpr.as<EQ>(), vars_set);
    } else if (cond.as<LE>()) {
      Expr movexpr = PreProcessExpr(cond.as<LE>(), expr, vars_set);
      if (movexpr.as<GE>()) {
        VisitCmpExpr(movexpr.as<GE>(), vars_set);
      } else {
        VisitCmpExpr(movexpr.as<LE>(), vars_set);
      }
    } else if (cond.as<LT>()) {
      Expr movexpr = PreProcessExpr(cond.as<LT>(), expr, vars_set);
      if (movexpr.as<GT>()) {
        VisitCmpExpr(movexpr.as<GT>(), vars_set);
      } else {
        VisitCmpExpr(movexpr.as<LT>(), vars_set);
      }
    } else if (cond.as<GE>()) {
      Expr movexpr = PreProcessExpr(cond.as<GE>(), expr, vars_set);
      if (movexpr.as<LE>()) {
        VisitCmpExpr(movexpr.as<LE>(), vars_set);
      } else {
        VisitCmpExpr(movexpr.as<GE>(), vars_set);
      }
    } else if (cond.as<GT>()) {
      Expr movexpr = PreProcessExpr(cond.as<GT>(), expr, vars_set);
      if (movexpr.as<LT>()) {
        VisitCmpExpr(movexpr.as<LT>(), vars_set);
      } else {
        VisitCmpExpr(movexpr.as<GT>(), vars_set);
      }
    } else {
      LOG(INFO) << "constraint expr is invalid" << cond;
    }
  }

  void GetVarBound(const Expr &cond, const UnorderSet &vars_set) {
    if (cond.as<EQ>()) {
      VisitCmpExpr(cond.as<EQ>(), vars_set);
    } else if (cond.as<LE>()) {
      VisitCmpExpr(cond.as<LE>(), vars_set);
    } else if (cond.as<LT>()) {
      VisitCmpExpr(cond.as<LT>(), vars_set);
    } else if (cond.as<GE>()) {
      VisitCmpExpr(cond.as<GE>(), vars_set);
    } else if (cond.as<GT>()) {
      VisitCmpExpr(cond.as<GT>(), vars_set);
    } else {
      LOG(INFO) << "constraint expr is invalid" << cond;
    }
  }

  Bound GetTightBound(const Expr &e, const Bound &exist_bound, const Bound &cur_bound) {
    Expr min_bound;
    Expr max_bound;
    ExprSimplifier expr_compute;
    CHECK(exist_bound.min.defined());
    CHECK(cur_bound.min.defined());
    int min_sign = GetSign(expr_compute.Simplify(exist_bound.min - cur_bound.min));
    if (min_sign == static_cast<int>(Sign::POS) || min_sign == static_cast<int>(Sign::ZERO)) {
      min_bound = exist_bound.min;
    } else if (min_sign == static_cast<int>(Sign::NEG)) {
      min_bound = cur_bound.min;
    } else if (Equal(e, exist_bound.min) && !Equal(e, cur_bound.min)) {
      min_bound = cur_bound.min;
    } else if (!Equal(e, exist_bound.min) && Equal(e, cur_bound.min)) {
      min_bound = exist_bound.min;
    } else if (Equal(e, exist_bound.min) && Equal(e, cur_bound.min)) {
      min_bound = e;
    } else {
      min_bound = Max::make(exist_bound.min, cur_bound.min);
    }

    CHECK(exist_bound.max.defined());
    CHECK(cur_bound.max.defined());
    auto max_sign = GetSign(expr_compute.Simplify(exist_bound.max - cur_bound.max));
    if (max_sign == static_cast<int>(Sign::POS) || max_sign == static_cast<int>(Sign::ZERO)) {
      max_bound = cur_bound.max;
    } else if (max_sign == static_cast<int>(Sign::NEG)) {
      max_bound = exist_bound.max;
    } else if (Equal(e, exist_bound.max) && !Equal(e, cur_bound.max)) {
      max_bound = cur_bound.max;
    } else if (!Equal(e, exist_bound.max) && Equal(e, cur_bound.max)) {
      max_bound = exist_bound.max;
    } else if (Equal(e, exist_bound.max) && Equal(e, cur_bound.max)) {
      max_bound = e;
    } else {
      max_bound = Min::make(exist_bound.max, cur_bound.max);
    }

    return Bound::make(min_bound, max_bound);
  }

  template <typename T>
  Expr BinaryOpWithMaxMin(T *op) {
    ExprSimplifier spl;
    CHECK(op);
    Expr lhs = op->a;
    Expr rhs = op->b;
    if (lhs.as<Min>()) {
      Expr min_a = T::make(lhs.as<Min>()->a, rhs);
      Expr min_b = T::make(lhs.as<Min>()->b, rhs);
      return Min::make(spl.Simplify(min_a), spl.Simplify(min_b));
    } else if (lhs.as<Max>()) {
      Expr max_a = T::make(lhs.as<Max>()->a, rhs);
      Expr max_b = T::make(lhs.as<Max>()->b, rhs);
      return Max::make(spl.Simplify(max_a), spl.Simplify(max_b));
    } else if (rhs.as<Min>()) {
      Expr min_a = T::make(lhs, rhs.as<Min>()->a);
      Expr min_b = T::make(lhs, rhs.as<Min>()->b);
      if (min_a.as<Sub>() || min_a.as<Div>() || min_a.as<FloorDiv>()) {
        return Max::make(spl.Simplify(min_a), spl.Simplify(min_b));
      } else {
        return Min::make(spl.Simplify(min_a), spl.Simplify(min_b));
      }
    } else if (rhs.as<Max>()) {
      Expr max_a = T::make(lhs, rhs.as<Max>()->a);
      Expr max_b = T::make(lhs, rhs.as<Max>()->b);
      if (max_a.as<Sub>() || max_a.as<Div>() || max_a.as<FloorDiv>()) {
        return Min::make(spl.Simplify(max_a), spl.Simplify(max_b));
      } else {
        return Max::make(spl.Simplify(max_a), spl.Simplify(max_b));
      }
    } else {
      return spl.Simplify(T::make(spl.Simplify(lhs), spl.Simplify(rhs)));
    }
  }

  Bound GetExprBoundWithCond(const Expr &expr) {
    ExprSimplifier spl;
    if (expr.as<Variable>() && conds_var.count(expr.as<Variable>()) > 0) {
      return conds_var[expr.as<Variable>()];
    }

    for (auto cond_expr : conds_var_combine) {
      if (Equal(expr, cond_expr.first)) {
        return cond_expr.second;
      }
      if (expr.as<Call>() && cond_expr.first.as<Call>()) {
        auto expr_call = expr.as<Call>();
        auto cond_call = cond_expr.first.as<Call>();
        CHECK(expr_call);
        CHECK(cond_call);
        if (expr_call->name == cond_call->name &&
            ((Equal(expr_call->args[0], cond_call->args[0]) && Equal(expr_call->args[1], cond_call->args[1])) ||
             (Equal(expr_call->args[0], cond_call->args[1]) && Equal(expr_call->args[1], cond_call->args[0])))) {
          return cond_expr.second;
        }
      }

      if (!expr.as<IntImm>() && !cond_expr.first.as<IntImm>()) {
        auto scale = spl.Simplify(div(expr, cond_expr.first));
        if (scale.as<IntImm>()) {
          if (is_const_true(scale)) {
            return Bound::make(spl.Simplify(scale * cond_expr.second.min), spl.Simplify(scale * cond_expr.second.max));
          }
          return Bound::make(spl.Simplify(scale * cond_expr.second.max), spl.Simplify(scale * cond_expr.second.min));
        }

        if (!is_const_int(expr, 0)) {
          auto scale1 = spl.Simplify(div(cond_expr.first, expr));
          if (scale1.as<IntImm>() && static_cast<int>(scale1.as<IntImm>()->value) != 0) {
            if (is_const_true(scale1)) {
              return Bound::make(spl.Simplify(cond_expr.second.min / scale1),
                                 spl.Simplify(cond_expr.second.max / scale1));
            }
            return Bound::make(spl.Simplify(cond_expr.second.max / scale1),
                               spl.Simplify(cond_expr.second.min / scale1));
          }
        }
      }
    }

    if (expr.as<IntImm>() || expr.as<UIntImm>() || expr.as<FloatImm>()) {
      return Bound::make(expr, expr);
    } else if (expr.as<Variable>()) {
      auto var = expr.as<Variable>();
      if (conds_var.count(var) > 0) {
        Bound var_min_range = GetExprBoundWithCond(conds_var[var].min);
        Bound var_max_range = GetExprBoundWithCond(conds_var[var].max);
        return Bound::make(var_min_range.min, var_max_range.max);
      } else {
        return Bound::make(expr, expr);
      }
    } else if (auto add = expr.as<Add>()) {
      Bound bound_a = GetExprBoundWithCond(add->a);
      Bound bound_b = GetExprBoundWithCond(add->b);
      return Bound::make(BinaryOpWithMaxMin(Add::make(bound_a.min, bound_b.min).as<Add>()),
                         BinaryOpWithMaxMin(Add::make(bound_a.max, bound_b.max).as<Add>()));
    } else if (auto sub = expr.as<Sub>()) {
      if ((sub->a.as<Min>() || sub->a.as<Max>()) && !sub->b.as<Min>() && !sub->b.as<Max>()) {
        return GetExprBoundWithCond(BinaryOpWithMaxMin(expr.as<Sub>()));
      }
      if (!sub->a.as<Min>() && !sub->a.as<Max>() && (sub->b.as<Min>() || sub->b.as<Max>())) {
        return GetExprBoundWithCond(BinaryOpWithMaxMin(expr.as<Sub>()));
      }
      Bound bound_a = GetExprBoundWithCond(sub->a);
      Bound bound_b = GetExprBoundWithCond(sub->b);
      return Bound::make(BinaryOpWithMaxMin(Sub::make(bound_a.min, bound_b.max).as<Sub>()),
                         BinaryOpWithMaxMin(Sub::make(bound_a.max, bound_b.min).as<Sub>()));
    } else if (auto mul = expr.as<Mul>()) {
      Bound bound_a = GetExprBoundWithCond(mul->a);
      Bound bound_b = GetExprBoundWithCond(mul->b);
      Bound bound;

      auto a_min_sign = GetSign(bound_a.min);
      auto a_max_sign = GetSign(bound_a.max);
      auto b_min_sign = GetSign(bound_b.min);
      auto b_max_sign = GetSign(bound_b.max);

      if ((a_min_sign == static_cast<int>(Sign::POS) || a_min_sign == static_cast<int>(Sign::ZERO)) &&
          (b_min_sign == static_cast<int>(Sign::POS) || b_min_sign == static_cast<int>(Sign::ZERO))) {
        bound.min = BinaryOpWithMaxMin(Mul::make(bound_a.min, bound_b.min).as<Mul>());
        bound.max = BinaryOpWithMaxMin(Mul::make(bound_a.max, bound_b.max).as<Mul>());
      } else if ((a_max_sign == static_cast<int>(Sign::NEG) || a_max_sign == static_cast<int>(Sign::ZERO)) &&
                 (b_max_sign == static_cast<int>(Sign::NEG) || b_max_sign == static_cast<int>(Sign::ZERO))) {
        bound.min = BinaryOpWithMaxMin(Mul::make(bound_a.max, bound_b.max).as<Mul>());
        bound.max = BinaryOpWithMaxMin(Mul::make(bound_a.min, bound_b.min).as<Mul>());
      } else if (b_max_sign == static_cast<int>(Sign::POS) && a_max_sign == static_cast<int>(Sign::POS)) {
        bound.max = BinaryOpWithMaxMin(Mul::make(bound_b.max, bound_a.max).as<Mul>());
      } else if ((a_max_sign == static_cast<int>(Sign::NEG) || a_max_sign == static_cast<int>(Sign::ZERO)) &&
                 (b_min_sign == static_cast<int>(Sign::POS) || b_min_sign == static_cast<int>(Sign::ZERO))) {
        bound.min = BinaryOpWithMaxMin(Mul::make(bound_a.min, bound_b.max).as<Mul>());
        bound.max = BinaryOpWithMaxMin(Mul::make(bound_a.max, bound_b.min).as<Mul>());
      } else if ((b_max_sign == static_cast<int>(Sign::NEG) || b_max_sign == static_cast<int>(Sign::ZERO)) &&
                 (a_min_sign == static_cast<int>(Sign::POS) || a_min_sign == static_cast<int>(Sign::ZERO))) {
        bound.min = BinaryOpWithMaxMin(Mul::make(bound_b.min, bound_a.max).as<Mul>());
        bound.max = BinaryOpWithMaxMin(Mul::make(bound_b.max, bound_a.min).as<Mul>());
      } else if ((a_max_sign == static_cast<int>(Sign::POS) || a_max_sign == static_cast<int>(Sign::ZERO)) &&
                 a_min_sign == static_cast<int>(Sign::NEG) &&
                 (b_min_sign == static_cast<int>(Sign::POS) || b_min_sign == static_cast<int>(Sign::ZERO))) {
        bound.min = BinaryOpWithMaxMin(Mul::make(bound_a.min, bound_b.max).as<Mul>());
        bound.max = BinaryOpWithMaxMin(Mul::make(bound_a.max, bound_b.max).as<Mul>());
      } else if ((b_max_sign == static_cast<int>(Sign::POS) || b_max_sign == static_cast<int>(Sign::ZERO)) &&
                 b_min_sign == static_cast<int>(Sign::NEG) &&
                 (a_min_sign == static_cast<int>(Sign::POS) || a_min_sign == static_cast<int>(Sign::ZERO))) {
        bound.min = BinaryOpWithMaxMin(Mul::make(bound_b.min, bound_a.max).as<Mul>());
        bound.max = BinaryOpWithMaxMin(Mul::make(bound_b.max, bound_a.max).as<Mul>());
      } else if ((b_max_sign == static_cast<int>(Sign::POS) || b_max_sign == static_cast<int>(Sign::ZERO)) &&
                 b_min_sign == static_cast<int>(Sign::NEG) &&
                 (a_max_sign == static_cast<int>(Sign::NEG) || a_max_sign == static_cast<int>(Sign::ZERO))) {
        bound.min = BinaryOpWithMaxMin(Mul::make(bound_b.max, bound_a.min).as<Mul>());
        bound.max = BinaryOpWithMaxMin(Mul::make(bound_b.min, bound_a.min).as<Mul>());
      } else if ((a_max_sign == static_cast<int>(Sign::POS) || a_max_sign == static_cast<int>(Sign::ZERO)) &&
                 a_min_sign == static_cast<int>(Sign::NEG) &&
                 (b_max_sign == static_cast<int>(Sign::NEG) || b_max_sign == static_cast<int>(Sign::ZERO))) {
        bound.min = BinaryOpWithMaxMin(Mul::make(bound_a.max, bound_b.min).as<Mul>());
        bound.max = BinaryOpWithMaxMin(Mul::make(bound_a.min, bound_b.min).as<Mul>());
      }
      if (!bound.min.defined()) bound.min = expr;
      if (!bound.max.defined()) bound.max = expr;
      return bound;
    } else if (expr.as<FloorDiv>()) {
      Expr simpdiv = spl.Simplify(expr);
      if (!simpdiv.as<FloorDiv>()) {
        return GetExprBoundWithCond(simpdiv);
      }
      auto div = simpdiv.as<FloorDiv>();
      if ((div->a.as<Min>() || div->a.as<Max>()) && !div->b.as<Min>() && !div->b.as<Max>()) {
        return GetExprBoundWithCond(BinaryOpWithMaxMin(expr.as<FloorDiv>()));
      }
      Bound bound_a = GetExprBoundWithCond(div->a);
      Bound bound_b = GetExprBoundWithCond(div->b);
      Bound bound;

      auto a_min_sign = GetSign(bound_a.min);
      auto a_max_sign = GetSign(bound_a.max);
      auto b_min_sign = GetSign(bound_b.min);
      auto b_max_sign = GetSign(bound_b.max);

      if ((a_min_sign == static_cast<int>(Sign::POS) || a_min_sign == static_cast<int>(Sign::ZERO)) &&
          b_min_sign == static_cast<int>(Sign::POS)) {
        if (!is_const_int(bound_b.max, 0)) {
          bound.min = BinaryOpWithMaxMin(FloorDiv::make(bound_a.min, bound_b.max).as<FloorDiv>());
        }
        if (!is_const_int(bound_b.min, 0)) {
          bound.max = BinaryOpWithMaxMin(FloorDiv::make(bound_a.max, bound_b.min).as<FloorDiv>());
        }
      } else if ((a_min_sign == static_cast<int>(Sign::POS) || a_min_sign == static_cast<int>(Sign::ZERO)) &&
                 b_max_sign == static_cast<int>(Sign::NEG)) {
        if (!is_const_int(bound_b.max, 0)) {
          bound.min = BinaryOpWithMaxMin(FloorDiv::make(bound_a.max, bound_b.max).as<FloorDiv>());
        }
        if (!is_const_int(bound_b.min, 0)) {
          bound.max = BinaryOpWithMaxMin(FloorDiv::make(bound_a.min, bound_b.min).as<FloorDiv>());
        }
      } else if ((a_min_sign == static_cast<int>(Sign::POS) || a_min_sign == static_cast<int>(Sign::ZERO)) &&
                 b_max_sign == static_cast<int>(Sign::POS) && b_max_sign == static_cast<int>(Sign::NEG)) {
        if (!is_const_int(bound_b.min, 0)) {
          bound.min = BinaryOpWithMaxMin(FloorDiv::make(bound_a.min, bound_b.min).as<FloorDiv>());
        }
        if (!is_const_int(bound_b.max, 0)) {
          bound.max = BinaryOpWithMaxMin(FloorDiv::make(bound_a.max, bound_b.max).as<FloorDiv>());
        }
      } else if ((a_max_sign == static_cast<int>(Sign::NEG) || a_max_sign == static_cast<int>(Sign::ZERO)) &&
                 b_max_sign == static_cast<int>(Sign::NEG)) {
        if (!is_const_int(bound_b.min, 0)) {
          bound.min = BinaryOpWithMaxMin(FloorDiv::make(bound_a.max, bound_b.min).as<FloorDiv>());
        }
        if (!is_const_int(bound_b.max, 0)) {
          bound.max = BinaryOpWithMaxMin(FloorDiv::make(bound_a.min, bound_b.max).as<FloorDiv>());
        }
      } else if (a_min_sign == static_cast<int>(Sign::NEG) && a_max_sign == static_cast<int>(Sign::POS) &&
                 b_min_sign == static_cast<int>(Sign::POS)) {
        if (!is_const_int(bound_b.min, 0)) {
          bound.min = BinaryOpWithMaxMin(FloorDiv::make(bound_a.min, bound_b.min).as<FloorDiv>());
          bound.max = BinaryOpWithMaxMin(FloorDiv::make(bound_a.max, bound_b.min).as<FloorDiv>());
        }
      } else if ((a_max_sign == static_cast<int>(Sign::NEG) || a_max_sign == static_cast<int>(Sign::ZERO)) &&
                 b_min_sign == static_cast<int>(Sign::POS)) {
        if (!is_const_int(bound_b.min, 0)) {
          bound.min = BinaryOpWithMaxMin(FloorDiv::make(bound_a.min, bound_b.min).as<FloorDiv>());
        }
        if (!is_const_int(bound_b.max, 0)) {
          bound.max = BinaryOpWithMaxMin(FloorDiv::make(bound_a.max, bound_b.max).as<FloorDiv>());
        }
      } else if (a_max_sign == static_cast<int>(Sign::POS) && a_min_sign == static_cast<int>(Sign::NEG) &&
                 b_max_sign == static_cast<int>(Sign::NEG)) {
        if (!is_const_int(bound_b.min, 0)) {
          bound.min = BinaryOpWithMaxMin(FloorDiv::make(bound_a.max, bound_b.min).as<FloorDiv>());
        }
        if (!is_const_int(bound_b.max, 0)) {
          bound.max = BinaryOpWithMaxMin(FloorDiv::make(bound_a.min, bound_b.max).as<FloorDiv>());
        }
      } else {
        bound = Bound::make(expr, expr);
      }
      if (!bound.min.defined()) bound.min = expr;
      if (!bound.max.defined()) bound.max = expr;
      return bound;
    } else if (expr.as<Div>()) {
      Expr simpdiv = spl.Simplify(expr);
      if (!simpdiv.as<Div>()) {
        return GetExprBoundWithCond(simpdiv);
      }
      auto div = simpdiv.as<Div>();
      if ((div->a.as<Min>() || div->a.as<Max>()) && !div->b.as<Min>() && !div->b.as<Max>()) {
        return GetExprBoundWithCond(BinaryOpWithMaxMin(expr.as<Div>()));
      }
      Bound bound_a = GetExprBoundWithCond(div->a);
      Bound bound_b = GetExprBoundWithCond(div->b);
      Bound bound;

      auto a_min_sign = GetSign(bound_a.min);
      auto a_max_sign = GetSign(bound_a.max);
      auto b_min_sign = GetSign(bound_b.min);
      auto b_max_sign = GetSign(bound_b.max);

      if ((a_min_sign == static_cast<int>(Sign::POS) || a_min_sign == static_cast<int>(Sign::ZERO)) &&
          b_min_sign == static_cast<int>(Sign::POS)) {
        if (!is_const_int(bound_b.max, 0)) {
          bound.min = BinaryOpWithMaxMin(FloorDiv::make(bound_a.min, bound_b.max).as<FloorDiv>());
        }
        if (!is_const_int(bound_b.min, 0)) {
          bound.max = BinaryOpWithMaxMin(FloorDiv::make(bound_a.max, bound_b.min).as<FloorDiv>());
        }
      } else if ((a_min_sign == static_cast<int>(Sign::POS) || a_min_sign == static_cast<int>(Sign::ZERO)) &&
                 b_max_sign == static_cast<int>(Sign::NEG)) {
        if (!is_const_int(bound_b.max, 0)) {
          bound.min = BinaryOpWithMaxMin(FloorDiv::make(bound_a.max, bound_b.max).as<FloorDiv>());
        }
        if (!is_const_int(bound_b.min, 0)) {
          bound.max = BinaryOpWithMaxMin(FloorDiv::make(bound_a.min, bound_b.min).as<FloorDiv>());
        }
      } else if ((a_min_sign == static_cast<int>(Sign::POS) || a_min_sign == static_cast<int>(Sign::ZERO)) &&
                 b_max_sign == static_cast<int>(Sign::POS) && b_max_sign == static_cast<int>(Sign::NEG)) {
        if (!is_const_int(bound_b.min, 0)) {
          bound.min = BinaryOpWithMaxMin(FloorDiv::make(bound_a.min, bound_b.min).as<FloorDiv>());
        }
        if (!is_const_int(bound_b.max, 0)) {
          bound.max = BinaryOpWithMaxMin(FloorDiv::make(bound_a.max, bound_b.max).as<FloorDiv>());
        }
      } else if ((a_max_sign == static_cast<int>(Sign::NEG) || a_max_sign == static_cast<int>(Sign::ZERO)) &&
                 b_max_sign == static_cast<int>(Sign::NEG)) {
        if (!is_const_int(bound_b.min, 0)) {
          bound.min = BinaryOpWithMaxMin(FloorDiv::make(bound_a.max, bound_b.min).as<FloorDiv>());
        }
        if (!is_const_int(bound_b.max, 0)) {
          bound.max = BinaryOpWithMaxMin(FloorDiv::make(bound_a.min, bound_b.max).as<FloorDiv>());
        }
      } else if (a_min_sign == static_cast<int>(Sign::NEG) && a_max_sign == static_cast<int>(Sign::POS) &&
                 b_min_sign == static_cast<int>(Sign::POS)) {
        if (!is_const_int(bound_b.min, 0)) {
          bound.min = BinaryOpWithMaxMin(FloorDiv::make(bound_a.min, bound_b.min).as<FloorDiv>());
          bound.max = BinaryOpWithMaxMin(FloorDiv::make(bound_a.max, bound_b.min).as<FloorDiv>());
        }
      } else if ((a_max_sign == static_cast<int>(Sign::NEG) || a_max_sign == static_cast<int>(Sign::ZERO)) &&
                 b_min_sign == static_cast<int>(Sign::POS)) {
        if (!is_const_int(bound_b.min, 0)) {
          bound.min = BinaryOpWithMaxMin(FloorDiv::make(bound_a.min, bound_b.min).as<FloorDiv>());
        }
        if (!is_const_int(bound_b.max, 0)) {
          bound.max = BinaryOpWithMaxMin(FloorDiv::make(bound_a.max, bound_b.max).as<FloorDiv>());
        }
      } else if (a_max_sign == static_cast<int>(Sign::POS) && a_min_sign == static_cast<int>(Sign::NEG) &&
                 b_max_sign == static_cast<int>(Sign::NEG)) {
        if (!is_const_int(bound_b.min, 0)) {
          bound.min = BinaryOpWithMaxMin(FloorDiv::make(bound_a.max, bound_b.min).as<FloorDiv>());
        }
        if (!is_const_int(bound_b.max, 0)) {
          bound.max = BinaryOpWithMaxMin(FloorDiv::make(bound_a.min, bound_b.max).as<FloorDiv>());
        }
      } else {
        bound = Bound::make(expr, expr);
      }
      if (!bound.min.defined()) bound.min = expr;
      if (!bound.max.defined()) bound.max = expr;
      return bound;
    } else if (expr.as<Min>()) {
      auto min_expr = expr.as<Min>();
      Bound bound;
      Bound a_bound = GetExprBoundWithCond(spl.Simplify(min_expr->a));
      Bound b_bound = GetExprBoundWithCond(spl.Simplify(min_expr->b));
      auto a_min_sign = GetSign(a_bound.min);
      auto b_min_sign = GetSign(b_bound.min);

      if ((a_bound.min - b_bound.min).as<IntImm>()) {
        bound.min = CanProve(a_bound.min - b_bound.min >= 0) ? b_bound.min : a_bound.min;
      } else if ((a_min_sign == static_cast<int>(Sign::POS) || a_min_sign == static_cast<int>(Sign::ZERO)) &&
                 (b_min_sign == static_cast<int>(Sign::POS) || b_min_sign == static_cast<int>(Sign::ZERO))) {
        bound.min = 0;
      } else if (a_min_sign == static_cast<int>(Interval::GTZERO) && b_min_sign == static_cast<int>(Interval::GTZERO)) {
        bound.min = 1;
      }

      auto prefered_bound = [](Bound bound, Expr e) {
        if (bound.max.as<IntImm>()) {
          return true;
        } else if (e.as<Variable>()) {
          return true;
        }
        return false;
      };

      auto a_get_prefer = prefered_bound(a_bound, min_expr->a);
      auto b_get_prefer = prefered_bound(b_bound, min_expr->b);
      if (a_get_prefer && !b_get_prefer) {
        bound.max = a_bound.max;
      } else if (!a_get_prefer && b_get_prefer) {
        bound.max = b_bound.max;
      }

      if (bound.max.defined()) {
        if (!bound.min.defined()) {
          bound.min = expr;
        }
        pair<const Expr, Bound> min_pair = std::make_pair(expr, bound);
        InsertPair(min_pair);
        return bound;
      }

      Bound compare_min = GetExprBoundWithCond(spl.Simplify(a_bound.min - b_bound.min));
      Bound compare_max = GetExprBoundWithCond(spl.Simplify(a_bound.max - b_bound.max));

      auto min_min_sign = GetSign(compare_min.min);
      auto min_max_sign = GetSign(compare_min.max);
      auto max_min_sign = GetSign(compare_max.min);
      auto max_max_sign = GetSign(compare_max.max);

      if (min_min_sign == static_cast<int>(Sign::POS) || min_min_sign == static_cast<int>(Sign::ZERO)) {
        bound.min = b_bound.min;
      } else if (min_max_sign == static_cast<int>(Sign::NEG)) {
        bound.min = a_bound.min;
      }

      if (max_min_sign == static_cast<int>(Sign::POS) || max_min_sign == static_cast<int>(Sign::ZERO)) {
        bound.max = b_bound.max;
      } else if (max_max_sign == static_cast<int>(Sign::NEG)) {
        bound.max = a_bound.max;
      }

      if (!bound.min.defined()) bound.min = expr;
      if (!bound.max.defined()) bound.max = expr;
      pair<const Expr, Bound> min_pair = std::make_pair(expr, bound);
      InsertPair(min_pair);
      return bound;
    } else if (expr.as<Max>()) {
      auto max_expr = expr.as<Max>();
      Bound bound;
      Bound a_bound = GetExprBoundWithCond(max_expr->a);
      Bound b_bound = GetExprBoundWithCond(max_expr->b);
      auto a_max_sign = GetSign(a_bound.max);
      auto b_max_sign = GetSign(b_bound.max);

      if ((a_bound.max - b_bound.max).as<IntImm>()) {
        bound.max = CanProve(a_bound.max - b_bound.max >= 0) ? a_bound.min : b_bound.min;
      } else if ((a_max_sign == static_cast<int>(Sign::NEG) || a_max_sign == static_cast<int>(Sign::ZERO)) &&
                 (b_max_sign == static_cast<int>(Sign::NEG) || b_max_sign == static_cast<int>(Sign::ZERO))) {
        bound.max = 0;
      } else if (a_max_sign == static_cast<int>(Interval::LTZERO) && b_max_sign == static_cast<int>(Interval::LTZERO)) {
        bound.max = -1;
      }

      auto prefered_bound = [](Bound bound, Expr e) {
        if (bound.min.as<IntImm>()) {
          return true;
        } else if (e.as<Variable>()) {
          return true;
        }
        return false;
      };

      auto a_get_prefer = prefered_bound(a_bound, max_expr->a);
      auto b_get_prefer = prefered_bound(b_bound, max_expr->b);
      if (a_get_prefer && !b_get_prefer) {
        bound.min = a_bound.min;
      } else if (!a_get_prefer && b_get_prefer) {
        bound.min = b_bound.min;
      }

      if (bound.min.defined()) {
        if (!bound.max.defined()) {
          bound.max = expr;
        }
        pair<const Expr, Bound> max_pair = std::make_pair(expr, bound);
        InsertPair(max_pair);
        return bound;
      }

      Bound compare_min = GetExprBoundWithCond(spl.Simplify(a_bound.min - b_bound.min));
      Bound compare_max = GetExprBoundWithCond(spl.Simplify(a_bound.max - b_bound.max));

      if (GetSign(compare_min.min) == static_cast<int>(Sign::POS) ||
          GetSign(compare_min.min) == static_cast<int>(Sign::ZERO)) {
        bound.min = a_bound.min;
      } else if (GetSign(compare_min.max) == static_cast<int>(Sign::NEG)) {
        bound.min = b_bound.min;
      }

      if (GetSign(compare_max.min) == static_cast<int>(Sign::POS) ||
          GetSign(compare_max.min) == static_cast<int>(Sign::ZERO)) {
        bound.max = a_bound.max;
      } else if (GetSign(compare_max.max) == static_cast<int>(Sign::NEG)) {
        bound.max = b_bound.max;
      }

      if (!bound.min.defined()) bound.min = expr;
      if (!bound.max.defined()) bound.max = expr;
      pair<const Expr, Bound> max_pair = std::make_pair(expr, bound);
      InsertPair(max_pair);
      return bound;
    } else if (auto op = expr.as<Call>()) {
      if (op->name.find("shift_right") != std::string::npos && op->args[1].as<IntImm>()) {
        int degree = op->args[1].as<IntImm>()->value;
        return GetExprBoundWithCond(Div::make(op->args[0], make_const(expr.type(), pow(2.0, degree))));
      } else {
        return Bound::make(expr, expr);
      }
    } else {
      return Bound::make(expr, expr);
    }
  }

  bool IsValidCond(const Expr &e1, const Expr &e2, const UnorderSet &e1_set, const UnorderSet &var_set) {
    // e1 is the target expr, e2 is the constraint for it.
    // valid constraint: if all vars in constraint also appear in target expr or all vars in difference set are paras.
    if (!e2.as<LT>() && !e2.as<LE>() && !e2.as<GE>() && !e2.as<GT>()) {
      return false;
    }

    UnorderSet cond_set;
    GatherVars(e2, &cond_set);
    if (cond_set.size() == 1) {
      return true;
    }
    UnorderSet diff_set = DifferenceSet(e1_set, cond_set);
    if (diff_set.size() > 0) {
      for (auto diffvar : diff_set) {
        if (conds_var.count(diffvar.get()) == 0 && var_set.count(diffvar) > 0) {
          return false;
        }
      }
      return true;
    }
    return true;
  }

  Bound PostCompBound(const Bound &bound) {
    Bound final_b = Bound::make(GetExprBoundWithCond(bound.min).min, GetExprBoundWithCond(bound.max).max);
    if (Equal(final_b.min, bound.max) && !final_b.min.as<IntImm>()) {
      auto b_min_sign = GetRangeWithParam(bound.min);
      auto b_max_sign = GetRangeWithParam(bound.max);
      if (b_min_sign == static_cast<int>(Interval::GEZERO)) {
        final_b.min = 0;
      } else if (b_min_sign == static_cast<int>(Interval::GTZERO)) {
        final_b.min = 1;
      } else if (b_max_sign == static_cast<int>(Interval::LEZERO)) {
        final_b.max = 0;
      } else if (b_max_sign == static_cast<int>(Interval::LTZERO)) {
        final_b.max = -1;
      }
    }
    return final_b;
  }

  Bound InferBoundWithCond(const Expr &expr, const Array<Expr> &constraints) {
    conds_var.clear();
    conds_var_combine.clear();
    substitute_map.clear();
    UnorderSet expr_var;
    GatherVars(expr, &expr_var);
    UnorderSet vars_set;
    GatherVars(expr, &vars_set);
    for (auto constraint : constraints) {
      PostOrderVisit(constraint, [&vars_set](const NodeRef &node) {
        if (node.as<Variable>()) {
          vars_set.emplace(Downcast<Var>(node));
        }
      });
    }
    Array<Expr> constraints_sort = GetSortedConstraint(constraints, vars_set);
    for (auto constraint : constraints_sort) {
      if (IsValidCond(expr, constraint, expr_var, vars_set) || constraints.size() == 1) {
        InsertCallBound(constraint);
        GetExprBound(constraint, expr, vars_set);
      } else {
        continue;
      }
    }

    if (detectpoly) {
      InsertCallBound(polyexpr);
      return GetExprBoundWithCond(polyexpr);
    }
    InsertCallBound(expr);
    Bound tar_bound = GetExprBoundWithCond(expr);
    // PostProcess for bound;
    return PostCompBound(tar_bound);
  }

  Bound InferBoundWithCond(const Expr &expr, const Array<Expr> &var_cst, const Array<Expr> &constraints,
                           const UnorderSet &vars_set) {
    conds_var.clear();
    conds_var_combine.clear();
    substitute_map.clear();
    UnorderSet expr_var;
    GatherVars(expr, &expr_var);

    for (auto var_c : var_cst) {
      InsertCallBound(var_c);
      GetVarBound(var_c, vars_set);
    }

    if (constraints.size() > 0) {
      Array<Expr> constraints_sort = GetSortedConstraint(constraints, vars_set);
      for (auto constraint : constraints_sort) {
        if (IsValidCond(expr, constraint, expr_var, vars_set) || constraints.size() == 1) {
          InsertCallBound(constraint);
          GetExprBound(constraint, expr, vars_set);
        } else {
          continue;
        }
      }
    }

    if (detectpoly) {
      InsertCallBound(polyexpr);
      return GetExprBoundWithCond(polyexpr);
    }
    InsertCallBound(expr);
    Bound tar_bound = GetExprBoundWithCond(expr);
    return PostCompBound(tar_bound);
  }

  Bound InferBoundWithSortedCond(const Expr &expr, const Array<Expr> &constraints) {
    conds_var.clear();
    conds_var_combine.clear();
    substitute_map.clear();
    UnorderSet vars_set;
    GatherVars(expr, &vars_set);
    for (auto constraint : constraints) {
      PostOrderVisit(constraint, [&vars_set](const NodeRef &node) {
        if (node.as<Variable>()) {
          vars_set.emplace(Downcast<Var>(node));
        }
      });
      GetVarBound(constraint, vars_set);
    }

    if (detectpoly) {
      InsertCallBound(polyexpr);
      return GetExprBoundWithCond(polyexpr);
    }
    InsertCallBound(expr);
    Bound tar_bound = GetExprBoundWithCond(expr);
    // PostProcess for bound;
    return PostCompBound(tar_bound);
  }

 private:
  bool detectpoly = false;
  bool detectmin = false;
  Expr polyexpr;
  std::unordered_map<const Variable *, Bound> conds_var;
  std::vector<pair<const Expr, Bound>> conds_var_combine;
  std::unordered_map<const Variable *, Expr> substitute_map;
  std::unordered_map<const Variable *, Var> var_map;
};

Expr GetConstIntUpBound(const Expr &e) {
  Expr one_bound = e;
  if (one_bound.as<Min>()) {
    auto min_a = GetConstIntUpBound(one_bound.as<Min>()->a);
    auto min_b = GetConstIntUpBound(one_bound.as<Min>()->b);
    if (GetSign(min_a - min_b) == static_cast<int>(Sign::POS) ||
        GetSign(min_a - min_b) == static_cast<int>(Sign::ZERO)) {
      return min_b;
    } else if (GetSign(min_a - min_b) == static_cast<int>(Sign::NEG)) {
      return min_a;
    } else if (is_const(min_a) && !is_const(min_b)) {
      return min_a;
    } else if (is_const(min_b) && !is_const(min_a)) {
      return min_b;
    } else {
      return one_bound;
    }
  } else if (auto mul = one_bound.as<Mul>()) {
    return Simplify(GetConstIntUpBound(mul->a) * GetConstIntUpBound(mul->b));
  } else if (auto div = one_bound.as<Div>()) {
    return Simplify(Div::make(GetConstIntUpBound(div->a), GetConstIntUpBound(div->b)));
  } else if (auto f_div = one_bound.as<FloorDiv>()) {
    return Simplify(FloorDiv::make(GetConstIntUpBound(f_div->a), GetConstIntUpBound(f_div->b)));
  } else if (auto add = one_bound.as<Add>()) {
    return Simplify(GetConstIntUpBound(add->a) + GetConstIntUpBound(add->b));
  } else if (auto sub = one_bound.as<Sub>()) {
    return Simplify(GetConstIntUpBound(sub->a) - GetConstIntUpBound(sub->b));
  } else {
    return one_bound;
  }
}

Expr GetConstIntLowBound(const Expr &e) {
  Expr one_bound = e;
  if (one_bound.as<Max>()) {
    auto max_a = GetConstIntLowBound(one_bound.as<Max>()->a);
    auto max_b = GetConstIntLowBound(one_bound.as<Max>()->b);
    if (GetSign(max_a - max_b) == static_cast<int>(Sign::POS) ||
        GetSign(max_a - max_b) == static_cast<int>(Sign::ZERO)) {
      return max_a;
    } else if (GetSign(max_a - max_b) == static_cast<int>(Sign::NEG)) {
      return max_b;
    } else if (is_const(max_a) && !is_const(max_b)) {
      return max_a;
    } else if (is_const(max_b) && !is_const(max_a)) {
      return max_b;
    } else {
      return one_bound;
    }
  } else if (auto mul = one_bound.as<Mul>()) {
    return Simplify(GetConstIntUpBound(mul->a) * GetConstIntUpBound(mul->b));
  } else if (auto div = one_bound.as<Div>()) {
    return Simplify(Div::make(GetConstIntUpBound(div->a), GetConstIntUpBound(div->b)));
  } else if (auto f_div = one_bound.as<FloorDiv>()) {
    return Simplify(FloorDiv::make(GetConstIntUpBound(f_div->a), GetConstIntUpBound(f_div->b)));
  } else if (auto add = one_bound.as<Add>()) {
    return Simplify(GetConstIntUpBound(add->a) + GetConstIntUpBound(add->b));
  } else if (auto sub = one_bound.as<Sub>()) {
    return Simplify(GetConstIntUpBound(sub->a) - GetConstIntUpBound(sub->b));
  } else {
    return one_bound;
  }
}

Bound InferVarBound(const Expr &expr, const Array<Expr> &constraints) {
  /// Inferbound for algebra simplify pass for speeding up computation.
  return InferBoundOfExprWithCondClass().InferBoundWithSortedCond(expr, constraints);
}

Bound InferBoundOfExprWithCond(const Expr &expr, const Array<Expr> &constraints) {
  /// Simple version of Inferbound. Use this will regard all Vars in expr as 'Variable'.
  /// \param expr: target expr to get its bound.
  /// \param constrints: constraints for inferring bound.
  return InferBoundOfExprWithCondClass().InferBoundWithCond(expr, constraints);
}

Bound InferBoundOfExprWithCond(const Expr &expr, const Array<Expr> &var_cst, const Array<Expr> &constraints,
                               const UnorderSet &vars_set) {
  /// Inferbound for input expr with vars_set.
  /// \param expr: target expr to get its bound.
  /// \param constrints: constraints for inferring bound.
  /// \param vars_set: in order to distinguish 'variables' from 'parameters', store all variables as vars_set.
  return InferBoundOfExprWithCondClass().InferBoundWithCond(expr, var_cst, constraints, vars_set);
}

Stmt TestInferBoundWithCond(const Expr &expr, const Array<Expr> &constraints) {
  Bound bound = InferBoundOfExprWithCondClass().InferBoundWithCond(expr, constraints);
  Stmt res = Evaluate::make(0);
  res = AttrStmt::make(make_zero(Int(32)), "Min", bound.min, res);
  res = AttrStmt::make(make_zero(Int(32)), "Max", bound.max, res);
  return res;
}

Expr RemoveCast(const Expr &expr) {
  Expr simp_expr = ExprSimplifier().Simplify(expr);
  if (simp_expr.as<Cast>())
    return simp_expr.as<Cast>()->value;
  else
    return simp_expr;
}

Expr SplitCompOp(const Expr &expr) {
  Expr oneside_e;
  if (expr.as<EQ>()) {
    oneside_e = Sub::make(RemoveCast(expr.as<EQ>()->a), RemoveCast(expr.as<EQ>()->a));
  } else if (expr.as<LT>()) {
    oneside_e = Sub::make(RemoveCast(expr.as<LT>()->a), RemoveCast(expr.as<LT>()->a));
  } else if (expr.as<LE>()) {
    oneside_e = Sub::make(RemoveCast(expr.as<LE>()->a), RemoveCast(expr.as<LE>()->a));
  } else if (expr.as<GT>()) {
    oneside_e = Sub::make(RemoveCast(expr.as<GT>()->a), RemoveCast(expr.as<GT>()->a));
  } else if (expr.as<GE>()) {
    oneside_e = Sub::make(RemoveCast(expr.as<GE>()->a), RemoveCast(expr.as<GE>()->a));
  } else {
    oneside_e = expr;
  }
  return oneside_e;
}

CondGraph::CondGraph(int vertices) {
  this->vertices = vertices;
  adj = new list<int>[vertices];
  for (int i = 0; i < vertices; ++i) {
    indegree.push_back(0);
  }
}

CondGraph::~CondGraph() { delete[] adj; }

void CondGraph::AddEdge(int v, int w) {
  adj[v].push_back(w);
  ++indegree[w];
}

bool CondGraph::TopoSort() {
  for (int i = 0; i < vertices; ++i) {
    if (indegree[i] == 0) {
      zero_set.push(i);
    }
  }

  int count = 0;
  while (!zero_set.empty()) {
    int v = zero_set.front();
    zero_set.pop();
    sort_res.push_back(v);
    ++count;

    for (auto it = adj[v].begin(); it != adj[v].end(); ++it) {
      if (!(--indegree[*it])) zero_set.push(*it);
    }
  }

  return count >= vertices;
}

void CondGraph::TopoSortConstraintByVar(const Array<Expr> &constraints, const UnorderSet &vars_set) {
  UnorderSet vars;
  // use tuple to store constraints index, number of vars, expr
  int size = constraints.size();
  var_constraint.clear();
  for (int i = 0; i < size; ++i) {
    vars.clear();
    GatherVars(constraints[i], &vars);
    UnorderSet intersec_set = IntersectionSet(vars, vars_set);
    int count = intersec_set.size();
    var_constraint.push_back(std::make_tuple(i, count, constraints[i]));
  }
  sort(var_constraint.begin(), var_constraint.end(),
       [](std::tuple<int, int, Expr> a, std::tuple<int, int, Expr> b) -> bool {
         return (std::get<1>(a) < std::get<1>(b));
       });
}

void CondGraph::TopoSortConstraint(const Array<Expr> &constraints, const UnorderSet &vars_set) {
  int size = constraints.size();
  TopoSortConstraintByVar(constraints, vars_set);
  for (int j = 1; j < size; ++j) {
    AddEdgeInExpr(j, std::get<2>(var_constraint[j]));
  }
}

void CondGraph::AddEdgeByDetectOp(const int index, const Expr &expr) {
  if (expr.as<Add>()) {
    auto add = expr.as<Add>();
    AddEdgeInExpr(index, add->a);
    AddEdgeInExpr(index, add->b);
  } else if (expr.as<Sub>()) {
    auto sub = expr.as<Sub>();
    AddEdgeInExpr(index, sub->a);
    AddEdgeInExpr(index, sub->b);
  } else if (expr.as<Mul>()) {
    auto mul = expr.as<Mul>();
    AddEdgeInExpr(index, mul->a);
    AddEdgeInExpr(index, mul->b);
  } else if (expr.as<FloorDiv>()) {
    auto div = expr.as<FloorDiv>();
    AddEdgeInExpr(index, div->a);
    AddEdgeInExpr(index, div->b);
  } else if (expr.as<Div>()) {
    auto div = expr.as<Div>();
    AddEdgeInExpr(index, div->a);
    AddEdgeInExpr(index, div->b);
  } else if (expr.as<Min>()) {
    auto min_expr = expr.as<Min>();
    AddEdgeInExpr(index, min_expr->a);
    AddEdgeInExpr(index, min_expr->b);
  } else if (expr.as<Max>()) {
    auto max_expr = expr.as<Max>();
    AddEdgeInExpr(index, max_expr->a);
    AddEdgeInExpr(index, max_expr->b);
  } else {
    return;
  }
}

void CondGraph::AddEdgeInExpr(const int index, const Expr &expr) {
  // use label to store whether comparative op in expr.
  Expr oneside_e = SplitCompOp(expr);
  for (int i = 0; i < index; ++i) {
    Expr previous_e = SplitCompOp(std::get<2>(var_constraint[i]));
    if (Equal(oneside_e, previous_e)) AddEdge(std::get<0>(var_constraint[i]), std::get<0>(var_constraint[index]));
  }
  AddEdgeByDetectOp(index, oneside_e);
}

Array<Expr> GetSortedConstraint(const Array<Expr> &constraints, const UnorderSet &vars_set) {
  CondGraph g_constraints(static_cast<int>(constraints.size()));
  g_constraints.TopoSortConstraint(constraints, vars_set);
  if (g_constraints.TopoSort()) {
    std::vector<int> &sort_index = g_constraints.sort_res;
    Array<Expr> sorted_constraints;
    for (int i = 0; i < static_cast<int>(constraints.size()); ++i) {
      sorted_constraints.push_back(constraints[sort_index[i]]);
    }
    return sorted_constraints;
  } else {
    LOG(INFO) << "The constraints are cyclic, cannot infer the bound!";
  }

  return constraints;
}

void SimplifyIfCondClass::GetCondBound(const EQ *op) {
  Expr expr = ExprSimplifier().Simplify(op->a - op->b);
  expr = expr.as<Cast>() ? expr.as<Cast>()->value : expr;
  this->cond_bound = std::make_pair(expr, Bound::make(0, 0));
}

void SimplifyIfCondClass::GetCondBound(const NE *op) {
  Expr expr = ExprSimplifier().Simplify(op->a - op->b);
  expr = expr.as<Cast>() ? expr.as<Cast>()->value : expr;
  this->cond_bound = std::make_pair(expr, Bound::make(0, 0));
}

void SimplifyIfCondClass::GetCondBound(const LT *op) {
  Expr expr = ExprSimplifier().Simplify(op->a - op->b);
  expr = expr.as<Cast>() ? expr.as<Cast>()->value : expr;
  this->cond_bound = std::make_pair(expr, Bound::make(expr, -1));
}

void SimplifyIfCondClass::GetCondBound(const LE *op) {
  Expr expr = ExprSimplifier().Simplify(op->a - op->b);
  expr = expr.as<Cast>() ? expr.as<Cast>()->value : expr;
  this->cond_bound = std::make_pair(expr, Bound::make(expr, 0));
}

void SimplifyIfCondClass::GetCondBound(const GE *op) {
  Expr expr = ExprSimplifier().Simplify(op->a - op->b);
  expr = expr.as<Cast>() ? expr.as<Cast>()->value : expr;
  this->cond_bound = std::make_pair(expr, Bound::make(0, expr));
}

void SimplifyIfCondClass::GetCondBound(const GT *op) {
  Expr expr = ExprSimplifier().Simplify(op->a - op->b);
  expr = expr.as<Cast>() ? expr.as<Cast>()->value : expr;
  this->cond_bound = std::make_pair(expr, Bound::make(1, expr));
}

bool SimplifyIfCondClass::CanProveValid(const Expr &cond, const Array<Expr> &constraints) {
  // if 1, the cond can be inferred from constraints. That means the condition is redundant.
  CHECK(cond.as<EQ>() || cond.as<NE>() || cond.as<LT>() || cond.as<LE>() || cond.as<GT>() || cond.as<GE>() ||
        cond.as<And>() || cond.as<Or>())
    << "Cannot support this comparative op: " << cond;
  Bound compute_bound;
  ExprSimplifier spl;
  if (auto eq = cond.as<EQ>()) {
    // To prove expr bound is [0,0]
    auto expr = spl.Simplify(eq->a - eq->b);
    compute_bound = InferBoundOfExprWithCond(expr, constraints);
    return (Equal(compute_bound.min, compute_bound.max) && is_const_int(compute_bound.min, 0));
  } else if (auto ne = cond.as<NE>()) {
    // To prove expr bound is [-expr,0)U(0,expr)]
    auto expr = spl.Simplify(ne->a - ne->b);
    compute_bound = InferBoundOfExprWithCond(expr, constraints);
    Bound complement = Bound::make(0, 0);
    Bound tight_bound = InferBoundOfExprWithCondClass().GetTightBound(expr, compute_bound, complement);
    if (Equal(tight_bound.min, tight_bound.max) && is_const_int(tight_bound.min, 0)) {
      return false;
    } else if (tight_bound.min.as<Max>() || tight_bound.max.as<Min>()) {
      return false;
    } else {
      return true;
    }
  } else if (auto lt = cond.as<LT>()) {
    auto expr = spl.Simplify(lt->a - lt->b);
    compute_bound = InferBoundOfExprWithCond(expr, constraints);
    return (GetSign(Simplify(compute_bound.max + 1)) == static_cast<int>(Sign::NEG) ||
            CanProve(Simplify(compute_bound.max + 1) == 0));
  } else if (auto le = cond.as<LE>()) {
    auto expr = spl.Simplify(le->a - le->b);
    compute_bound = InferBoundOfExprWithCond(expr, constraints);
    return (GetSign(Simplify(compute_bound.max)) == static_cast<int>(Sign::NEG) ||
            CanProve(Simplify(compute_bound.max) == 0));
  } else if (auto gt = cond.as<GT>()) {
    auto expr = spl.Simplify(gt->a - gt->b);
    compute_bound = InferBoundOfExprWithCond(expr, constraints);
    return (GetSign(Simplify(compute_bound.min - 1)) == static_cast<int>(Sign::POS) ||
            GetSign(Simplify(compute_bound.min - 1)) == static_cast<int>(Sign::ZERO));
  } else if (auto ge = cond.as<GE>()) {
    auto expr = spl.Simplify(ge->a - ge->b);
    compute_bound = InferBoundOfExprWithCond(expr, constraints);
    return (GetSign(Simplify(compute_bound.min)) == static_cast<int>(Sign::POS) ||
            GetSign(Simplify(compute_bound.min)) == static_cast<int>(Sign::ZERO));
  } else if (auto and_op = cond.as<And>()) {
    return CanProveValid(and_op->a, constraints) && CanProveValid(and_op->b, constraints);
  } else if (auto or_op = cond.as<Or>()) {
    return CanProveValid(or_op->a, constraints) || CanProveValid(or_op->b, constraints);
  } else {
    LOG(INFO) << "Cannot support this comparative op: " << cond;
    return false;
  }
}

bool CanProve(const Expr &expr, const std::unordered_map<const Variable *, Range> &var_bound_map) {
  return is_const_true(SimplifyConditionExpr(expr, var_bound_map));
}

class SimplifyExprClass {
 public:
  Expr run(const Expr &expr, const std::unordered_map<const Variable *, Range> &var_bound_map_) {
    var_bound_map = var_bound_map_;
    return simplify(expr);
  }

  Expr simplify(const Expr &expr) {
    if (auto min = expr.as<Min>()) {
      Bound lhs = InferBoundOfExpr(min->a, var_bound_map);
      Bound rhs = InferBoundOfExpr(min->b, var_bound_map);
      if (CanProve(lhs.max <= rhs.min)) {
        return min->a;
      } else if (CanProve(lhs.min >= rhs.max)) {
        return min->b;
      }
    } else if (auto max = expr.as<Max>()) {
      Bound lhs = InferBoundOfExpr(max->a, var_bound_map);
      Bound rhs = InferBoundOfExpr(max->b, var_bound_map);
      if (CanProve(lhs.max <= rhs.min)) {
        return max->b;
      } else if (CanProve(lhs.min >= rhs.max)) {
        return max->a;
      }
    }

    if (expr.type().is_bool()) {
      return SimplifyConditionExpr(expr, var_bound_map);
    }
    return Simplify(expr);
  }

 private:
  std::unordered_map<const Variable *, Range> var_bound_map;
};

Expr SimplifyExpr(const Expr &expr, const std::unordered_map<const Variable *, Range> &var_bound_map) {
  return SimplifyExprClass().run(expr, var_bound_map);
}

class CheckAffineExprOfVars {
 public:
  bool run(const Expr &expr, const std::unordered_set<const Variable *> &_vars) {
    vars = _vars;
    return IsAffineExprOfVars(expr);
  }

 private:
  bool IsAffineExprOfVars(const Expr &expr) {
    if (auto opAdd = expr.as<Add>()) {
      return IsAffineExprOfVars(opAdd->a) && IsAffineExprOfVars(opAdd->b);
    } else if (auto opSub = expr.as<Sub>()) {
      return IsAffineExprOfVars(opSub->a) && IsAffineExprOfVars(opSub->b);
    } else if (auto opMul = expr.as<Mul>()) {
      return (IsAffineExprOfVars(opMul->a) && IsConstExpr(opMul->b)) ||
             (IsConstExpr(opMul->a) && IsAffineExprOfVars(opMul->b));
    } else if (auto opDiv = expr.as<Div>()) {
      return IsAffineExprOfVars(opDiv->a) && IsConstExpr(opDiv->b);
    } else if (auto opFloorDiv = expr.as<FloorDiv>()) {
      return IsAffineExprOfVars(opFloorDiv->a) && IsConstExpr(opFloorDiv->b);
    } else if (auto opMod = expr.as<Mod>()) {
      return IsAffineExprOfVars(opMod->a) && IsConstExpr(opMod->b);
    } else if (auto opFloorMod = expr.as<FloorMod>()) {
      return IsAffineExprOfVars(opFloorMod->a) && IsConstExpr(opFloorMod->b);
    } else if (auto opVar = expr.as<Variable>()) {
      return vars.count(opVar) > 0;
    } else {
      return IsConstExpr(expr);
    }
  }

  std::unordered_set<const Variable *> vars;
};

bool IsAffineExprOfVars(const Expr &expr, const std::unordered_set<const Variable *> &vars) {
  return CheckAffineExprOfVars().run(expr, vars);
}

Range InferSimpleExprRange(Expr e, std::unordered_map<const Variable *, Range> *rmap) {
  // make sure no expression as:
  // varA + varA * 3
  // 3 * 7
  e = Simplify_cce(e);
  if (const auto varOp = e.as<Variable>()) {
    CHECK(rmap->count(varOp));
    auto it = rmap->find(varOp);
    return it->second;
  } else if (e.as<IntImm>()) {
    return Range::make_by_min_extent(e, Expr(1));
  } else if (const auto addOp = e.as<Add>()) {
    Range a = InferSimpleExprRange(addOp->a, rmap);
    Range b = InferSimpleExprRange(addOp->b, rmap);
    if (!a.defined() || !b.defined()) return Range();
    return Range::make_by_min_extent(Simplify_cce(a->min + b->min), Simplify_cce(a->extent + b->extent - 1));

  } else if (const auto subOp = e.as<Sub>()) {
    Range a = InferSimpleExprRange(subOp->a, rmap);
    Range b = InferSimpleExprRange(subOp->b, rmap);
    if (!a.defined() || !b.defined()) return Range();
    return Range::make_by_min_extent(Simplify_cce(a->min - (b->min + b->extent - 1)),
                                     Simplify_cce(a->extent + b->extent - 1));

  } else if (const auto mulOp = e.as<Mul>()) {
    if (const auto imma = mulOp->a.as<IntImm>()) {
      Range b = InferSimpleExprRange(mulOp->b, rmap);
      int value = static_cast<int>(imma->value);
      if (!b.defined()) return Range();
      if (value > 0) {
        return Range::make_by_min_extent(Simplify_cce(b->min * value),
                                         Simplify_cce((b->min + b->extent - 1) * value - b->min * value + 1));
      } else if (value < 0) {
        return Range::make_by_min_extent(Simplify_cce((b->min + b->extent - 1) * value),
                                         Simplify_cce(b->min * value - (b->min + b->extent - 1) * value + 1));
      } else {
        LOG(INFO) << "unsupported expression " << e;
      }
    } else if (const auto immb = mulOp->b.as<IntImm>()) {
      Range a = InferSimpleExprRange(mulOp->a, rmap);
      if (!a.defined()) return Range();
      int value = static_cast<int>(immb->value);
      if (value > 0) {
        return Range::make_by_min_extent(Simplify_cce(a->min * value),
                                         Simplify_cce((a->min + a->extent - 1) * value - a->min * value + 1));
      } else if (value < 0) {
        return Range::make_by_min_extent(Simplify_cce((a->min + a->extent - 1) * value),
                                         Simplify_cce(a->min * value - (a->min + a->extent - 1) * value + 1));
      } else {
        LOG(INFO) << "unsupported expression " << e;
      }
    } else {
      LOG(INFO) << "unsupported expression " << e;
    }
  } else {
    LOG(INFO) << "unsupported expression " << e;
  }

  return Range();
}

/* Match expr against a pattern.
 *
 * Pattern expr: An expression to match, with Variables that represent wildcard match:
 *   - Var("any"): any sub-expression
 *   - Var("int"): IntImm
 *   - Var("var"): Variable
 *   - Var("varOrInt"): Variable or IntImm
 * The pattern matching assumes commutativity, i.e. (a + b) matches (b + a).
 * However, it does not assume associativity, i.e. ((a + b) + c) does not match (a + (b + c)).
 *
 * Output arg: the list of wildcard matches.
 * Return value: match or not.
 */
bool ExprPatternMatch(const Expr &expr, const Expr &pattern, std::vector<Expr> *matches) {
  std::vector<Expr> matches_a, matches_b;
  bool flag = false;
  if (pattern.as<Variable>() && pattern.as<Variable>()->name_hint == "any") {
    matches_a.push_back(expr);
    flag = true;
  } else if (auto i = expr.as<IntImm>()) {
    if (auto pattern_i = pattern.as<IntImm>()) {
      return i->value == pattern_i->value;
    } else if (auto pattern_var = pattern.as<Variable>()) {
      if (pattern_var->name_hint == "int" || pattern_var->name_hint == "varOrInt") {
        matches_a.push_back(expr);
        flag = true;
      }
    }
  } else if (auto var = expr.as<Variable>()) {
    if (auto x = pattern.as<Variable>()) {
      if (var == x) return true;
      if (x->name_hint == "var" || x->name_hint == "varOrInt") {
        matches_a.push_back(expr);
        flag = true;
      }
    }
  } else if (auto add = expr.as<Add>()) {
    if (auto x = pattern.as<Add>()) {
      flag = (ExprPatternMatch(add->a, x->a, &matches_a) && ExprPatternMatch(add->b, x->b, &matches_b)) ||
             (ExprPatternMatch(add->a, x->b, &matches_b) && ExprPatternMatch(add->b, x->a, &matches_a));
    }
  } else if (auto sub = expr.as<Sub>()) {
    if (auto x = pattern.as<Sub>()) {
      flag = ExprPatternMatch(sub->a, x->a, &matches_a) && ExprPatternMatch(sub->b, x->b, &matches_b);
    }
  } else if (auto mul = expr.as<Mul>()) {
    if (auto x = pattern.as<Mul>()) {
      flag = (ExprPatternMatch(mul->a, x->a, &matches_a) && ExprPatternMatch(mul->b, x->b, &matches_b)) ||
             (ExprPatternMatch(mul->a, x->b, &matches_b) && ExprPatternMatch(mul->b, x->a, &matches_a));
    }
  } else if (auto div = expr.as<Div>()) {
    if (auto x = pattern.as<Div>()) {
      flag = ExprPatternMatch(div->a, x->a, &matches_a) && ExprPatternMatch(div->b, x->b, &matches_b);
    }
  } else if (auto f_div = expr.as<FloorDiv>()) {
    if (auto x = pattern.as<FloorDiv>()) {
      flag = ExprPatternMatch(f_div->a, x->a, &matches_a) && ExprPatternMatch(f_div->b, x->b, &matches_b);
    }
  } else if (auto mod = expr.as<Mod>()) {
    if (auto x = pattern.as<Mod>()) {
      flag = ExprPatternMatch(mod->a, x->a, &matches_a) && ExprPatternMatch(mod->b, x->b, &matches_b);
    }
  } else if (auto f_mod = expr.as<FloorMod>()) {
    if (auto x = pattern.as<FloorMod>()) {
      flag = ExprPatternMatch(f_mod->a, x->a, &matches_a) && ExprPatternMatch(f_mod->b, x->b, &matches_b);
    }
  } else if (auto min = expr.as<Min>()) {
    if (auto x = pattern.as<Min>()) {
      flag = (ExprPatternMatch(min->a, x->a, &matches_a) && ExprPatternMatch(min->b, x->b, &matches_b)) ||
             (ExprPatternMatch(min->a, x->b, &matches_b) && ExprPatternMatch(min->b, x->a, &matches_a));
    }
  } else if (auto max = expr.as<Max>()) {
    if (auto x = pattern.as<Max>()) {
      flag = (ExprPatternMatch(max->a, x->a, &matches_a) && ExprPatternMatch(max->b, x->b, &matches_b)) ||
             (ExprPatternMatch(max->a, x->b, &matches_b) && ExprPatternMatch(max->b, x->a, &matches_a));
    }
  }

  if (flag && matches) {
    matches->clear();
    for (auto it : matches_a) matches->push_back(it);
    for (auto it : matches_b) matches->push_back(it);
  }
  return flag;
}

template <class T>
bool LimitCheck(const air::arith::PVar<T> &n1, const air::arith::PVar<T> &n2) {
  if (n1.Eval().template as<FloatImm>() && n2.Eval().template as<FloatImm>()) {
    auto f1_eval = n1.Eval().template as<FloatImm>();
    auto f2_eval = n2.Eval().template as<FloatImm>();
    int f1_bit = f1_eval->type.bits();
    int f2_bit = f2_eval->type.bits();
    double f1_v = f1_eval->value;
    double f2_v = f2_eval->value;

    switch ((f1_bit > f2_bit) ? f1_bit : f2_bit) {
      case 16:
        return HALF_MIN <= (f1_v * f2_v) && (f1_v * f2_v) <= HALF_MAX;
      case 32:
        return std::numeric_limits<float>::min() <= (f1_v * f2_v) && (f1_v * f2_v) <= std::numeric_limits<float>::max();
      case 64:
        if (std::fabs(f2_v) <= std::numeric_limits<double>::epsilon()) return true;
        return (std::numeric_limits<double>::min() / f2_v) <= f1_v &&
               f1_v <= (std::numeric_limits<double>::max() / f2_v);
      default:
        break;
    }
    return false;
  } else if (n1.Eval().template as<IntImm>() && n2.Eval().template as<IntImm>()) {
    auto c1_eval = n1.Eval().template as<IntImm>();
    auto c2_eval = n2.Eval().template as<IntImm>();
    int c1_bit = c1_eval->type.bits();
    int c2_bit = c2_eval->type.bits();
    int64_t c1_v = c1_eval->value;
    int64_t c2_v = c2_eval->value;

    int64_t int8_min = std::numeric_limits<int8_t>::min(), int8_max = std::numeric_limits<int8_t>::max();
    int64_t int16_min = std::numeric_limits<int16_t>::min(), int16_max = std::numeric_limits<int16_t>::max();
    int64_t int32_min = std::numeric_limits<int32_t>::min(), int32_max = std::numeric_limits<int32_t>::max();
    int64_t int64_min = std::numeric_limits<int64_t>::min(), int64_max = std::numeric_limits<int64_t>::max();

    switch ((c1_bit > c2_bit) ? c1_bit : c2_bit) {
      case 8:
        return int8_min <= (c1_v * c2_v) && (c1_v * c2_v) <= int8_max;
      case 16:
        return int16_min <= (c1_v * c2_v) && (c1_v * c2_v) <= int16_max;
      case 32:
        return int32_min <= (c1_v * c2_v) && (c1_v * c2_v) <= int32_max;
      case 64:
        if (c2_v == 0) return true;
        return (int64_min / c2_v) <= c1_v && c1_v <= (int64_max / c2_v);
      default:
        break;
    }
    return false;
  }

  return false;
}

template bool LimitCheck(const air::arith::PVar<Floating> &, const air::arith::PVar<Floating> &);
template bool LimitCheck(const air::arith::PVar<Integer> &, const air::arith::PVar<Integer> &);

std::vector<Expr> ExtractSubExprs(const Expr &e) {
  std::vector<Expr> exprs;
  if (auto add = e.as<Add>()) {
    exprs.push_back(add->a);
    exprs.push_back(add->b);
  } else if (auto sub = e.as<Sub>()) {
    exprs.push_back(sub->a);
    exprs.push_back(sub->b);
  } else if (auto mul = e.as<Mul>()) {
    exprs.push_back(mul->a);
    exprs.push_back(mul->b);
  } else if (auto div = e.as<Div>()) {
    exprs.push_back(div->a);
    exprs.push_back(div->b);
  } else if (auto f_div = e.as<FloorDiv>()) {
    exprs.push_back(f_div->a);
    exprs.push_back(f_div->b);
  } else if (auto mod = e.as<Mod>()) {
    exprs.push_back(mod->a);
    exprs.push_back(mod->b);
  } else if (auto f_mod = e.as<FloorMod>()) {
    exprs.push_back(f_mod->a);
    exprs.push_back(f_mod->b);
  } else if (auto cast = e.as<Cast>()) {
    exprs.push_back(cast->value);
  } else if (auto eq = e.as<EQ>()) {
    exprs.push_back(eq->a);
    exprs.push_back(eq->b);
  } else if (auto ne = e.as<NE>()) {
    exprs.push_back(ne->a);
    exprs.push_back(ne->b);
  } else if (auto le = e.as<LE>()) {
    exprs.push_back(le->a);
    exprs.push_back(le->b);
  } else if (auto ge = e.as<GE>()) {
    exprs.push_back(ge->a);
    exprs.push_back(ge->b);
  } else if (auto lt = e.as<LT>()) {
    exprs.push_back(lt->a);
    exprs.push_back(lt->b);
  } else if (auto gt = e.as<GT>()) {
    exprs.push_back(gt->a);
    exprs.push_back(gt->b);
  } else if (auto and_op = e.as<And>()) {
    exprs.push_back(and_op->a);
    exprs.push_back(and_op->b);
  } else if (auto or_op = e.as<Or>()) {
    exprs.push_back(or_op->a);
    exprs.push_back(or_op->b);
  } else if (auto not_op = e.as<Not>()) {
    exprs.push_back(not_op->a);
  } else if (auto sel = e.as<Select>()) {
    exprs.push_back(sel->condition);
    exprs.push_back(sel->true_value);
    exprs.push_back(sel->false_value);
  } else if (auto load = e.as<Load>()) {
    exprs.push_back(load->index);
  } else if (auto store = e.as<Store>()) {
    exprs.push_back(store->index);
  } else if (auto call = e.as<Call>()) {
    for (auto arg : call->args) {
      exprs.push_back(arg);
    }
  }
  return exprs;
}

std::string ExprToString(const Expr &expr) {
  std::ostringstream os;
  os << expr;
  return os.str();
}

std::string ExprToVarName(const Expr &expr) {
  std::string name = ExprToString(expr);
  // replace special chars with '_'
  std::replace_if(
    name.begin(), name.end(), [](const char c) -> bool { return !std::isalnum(c); }, '_');
  // remove redundant '_'
  std::regex rx("_+");
  name = std::regex_replace(name, rx, "_");
  if (name.empty() || (*(name.begin()) >= '0' && *(name.begin()) <= '9')) {
    name = "_" + name;
  }
  return name;
}

/// Get the children of expr of binary operation
/// \param e - Expr to be processed
/// \return Array<Expr> e.a and e.b - If e is not binary op, then return empty Array.
Array<Expr> GetBinaryOpExprChildren(const Expr &e) {
  Array<Expr> children;
  if (auto add = e.as<Add>()) {
    children.push_back(add->a);
    children.push_back(add->b);
    return children;
  } else if (auto sub = e.as<Sub>()) {
    children.push_back(sub->a);
    children.push_back(sub->b);
    return children;
  } else if (auto mul = e.as<Mul>()) {
    children.push_back(mul->a);
    children.push_back(mul->b);
    return children;
  } else if (auto div = e.as<Div>()) {
    children.push_back(div->a);
    children.push_back(div->b);
    return children;
  } else if (auto f_div = e.as<FloorDiv>()) {
    children.push_back(f_div->a);
    children.push_back(f_div->b);
    return children;
  } else if (auto mod = e.as<Mod>()) {
    children.push_back(mod->a);
    children.push_back(mod->b);
    return children;
  } else if (auto f_mod = e.as<FloorMod>()) {
    children.push_back(f_mod->a);
    children.push_back(f_mod->b);
    return children;
  } else if (auto min = e.as<Min>()) {
    children.push_back(min->a);
    children.push_back(min->b);
    return children;
  } else if (auto max = e.as<Max>()) {
    children.push_back(max->a);
    children.push_back(max->b);
    return children;
  } else if (auto eq = e.as<EQ>()) {
    children.push_back(eq->a);
    children.push_back(eq->b);
    return children;
  } else if (auto ne = e.as<NE>()) {
    children.push_back(ne->a);
    children.push_back(ne->b);
    return children;
  } else if (auto lt = e.as<LT>()) {
    children.push_back(lt->a);
    children.push_back(lt->b);
    return children;
  } else if (auto le = e.as<LE>()) {
    children.push_back(le->a);
    children.push_back(le->b);
    return children;
  } else if (auto gt = e.as<GT>()) {
    children.push_back(gt->a);
    children.push_back(gt->b);
    return children;
  } else if (auto ge = e.as<GE>()) {
    children.push_back(ge->a);
    children.push_back(ge->b);
    return children;
  } else if (auto and_op = e.as<And>()) {
    children.push_back(and_op->a);
    children.push_back(and_op->b);
    return children;
  } else if (auto or_op = e.as<Or>()) {
    children.push_back(or_op->a);
    children.push_back(or_op->b);
    return children;
  } else {
    return children;
  }
}

/// Get the calculate op name of expr of binary operation
/// \param e - Expr to be processed
/// \return Expr op name - If e is not binary op, then return empty Expr.
Expr GetBinaryOpName(const Expr &e) {
  if (e.as<Add>()) {
    return Expr("add");
  } else if (e.as<Sub>()) {
    return Expr("sub");
  } else if (e.as<Mul>()) {
    return Expr("mul");
  } else if (e.as<Div>()) {
    return Expr("div");
  } else {
    return Expr();
  }
}

/// Get all Var in expr
/// \param expr - Expr to be processed
/// \return Array<VarExpr> - List of var in expr
Array<VarExpr> GetVarsInExpr(const Expr &expr, bool exclude_upper_case_vars) {
  class VariableMutator : public IRMutator {
   public:
    explicit VariableMutator(Array<Var> &ivar_set, bool exclude_upper = false)
        : ivar_set_(ivar_set), exclude_upper_(exclude_upper) {}
    ~VariableMutator() override = default;

    Expr Mutate_(const Variable *op, const Expr &e) final {
      bool find_var = true;
      if (exclude_upper_) {
        for (auto c : op->name_hint) {
          if (c >= 'A' && c <= 'Z') {
            find_var = false;
            break;
          }
        }
      }
      if (find_var) {
        bool find = false;
        for (auto iter = ivar_set_.begin(); iter != ivar_set_.end(); ++iter) {
          if ((*iter).get() == op) {
            find = true;
            break;
          }
        }
        if (!find) {
          ivar_set_.push_back(Downcast<Var>(e));
        }
      }
      return e;
    }
    Array<Var> &ivar_set_;
    bool exclude_upper_{false};
  };

  Array<Var> ivar_set;
  VariableMutator(ivar_set, exclude_upper_case_vars).Mutate(expr);
  return ivar_set;
}

/// Checks Expr is a halide call or a binaryop that contains halide call
/// \param e - Expr to be processed
/// \return true if Expr is a halide call
bool IsHalideCall(const Expr &e) {
  auto call = e.as<Call>();
  if (call && call->call_type == Call::Halide) {
    return true;
  }
  if (GetBinaryOpName(e).defined()) {
    return ContainsHalideCall(GetBinaryOpExprChildren(e));
  }
  return false;
}

/// Checks if any Expr in Array is a halide call or a binaryop that contains halide call
/// \param args - Array of Expr to be processed
/// \return true if Array contains a halide call
bool ContainsHalideCall(const Array<Expr> args) {
  return std::any_of(args.begin(), args.end(), IsHalideCall);
}

/// Get expr's reduce type
/// \param args - Expr to be processed
/// \return std::string of reduce type
std::string GetOpReduceType(const Expr value) {
  if (value.as<Max>()) return AKG_REDUCE_MAX;
  if (value.as<Min>()) return AKG_REDUCE_MIN;
  if (value.as<Add>()) return AKG_REDUCE_SUM;
  if (value.as<Mul>()) return AKG_REDUCE_PROD;
  if (value.as<And>()) return AKG_REDUCE_AND;
  if (value.as<Or>()) return AKG_REDUCE_OR;
  return AKG_REDUCE_UNSUPPORTED;
}

#ifdef USE_AKG_COMPILE_STUB
std::string GetProductName() { return "default"; }

const char *const LOCAL_BUF = "LOCAL_BUF";
const char *const LOCAL_C1 = "LOCAL_C1";
const char *const LOCAL_C0B = "LOCAL_C0B";
const char *const LOCAL_C0C = "LOCAL_C0C";
const char *const LOCAL_C1_LOCAL_C0A = "LOCAL_C1_LOCAL_C0A";
const char *const LOCAL_C1_LOCAL_C0B = "LOCAL_C1_LOCAL_C0B";
const char *const LOCAL_BUF_LOCAL_C0C = "LOCAL_BUF_LOCAL_C0C";
const char *const FRACTAL_C1 = "FRACTAL_C1";
const char *const FRACTAL_C1_LOCAL_C0B = "FRACTAL_C1_LOCAL_C0B";
#endif

void ConstructAtomicReturnFuncName(const std::string &reduce_lib, const std::string &reduce_op,
                                   std::string &akg_atomic_api, std::string &akg_atomic_template_arg) {
  std::string reduce_lib_namespace = "";
  std::string reduce_return_name = "";
  if (reduce_lib == REDUCE_LIB_TYPE_ORIGIN) {
    reduce_lib_namespace = AKG_REDUCE_LIB_SPACE;
    reduce_return_name = AKG_REDUCE_RETURN_NAME;
  } else if (reduce_lib == REDUCE_LIB_TYPE_PARIS) {
    reduce_lib_namespace = PARIS_REDUCE_LIB_SPACE;
    reduce_return_name = PARIS_REDUCE_RETURN_NAME;
  } else {
    CHECK(false) << "reduce lib type is invalid!"
                 << "\n";
  }
  akg_atomic_api = reduce_lib_namespace + "::" + reduce_return_name;
  akg_atomic_template_arg = reduce_op;
}

Stmt MakeAtomicStmt(const AtomicReturnData &atomic_data) {
  std::string func_name = atomic_data.akg_atomic_api;

  Expr template_arg0 = make_const(atomic_data.output_tensor_data_type_info, 1);
  CHECK(!atomic_data.akg_atomic_template_arg.empty());
  Expr template_arg1 = StringImm::make(atomic_data.akg_atomic_template_arg);

  Expr arg0 = atomic_data.atomic_rhs;

  auto p = atomic_data.gm_write_stmt.as<Provide>();
  CHECK(p);
  Expr arg1 = Call::make(p->value.type(), p->func->func_name(), p->args, Call::Halide, p->func, 0);
  arg1 = Call::make(arg1.type(), "&", {arg1}, Call::Extern);

  Array<Expr> args;
  Expr arg2 = Call::make(Int(32), atomic_data.reduce_op, args, Call::Extern);

  return Evaluate::make(Call::make(Int(32), func_name, {template_arg0, template_arg1, arg0, arg1, arg2}, Call::Extern));
}
}  // namespace ir

#ifdef USE_AKG_COMPILE_STUB
std::string GetBufScope(const std::string &name) { return "buffer"; }
#endif
}  // namespace akg