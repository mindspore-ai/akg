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
#include <dmlc/base.h>
#include <dmlc/parameter.h>
#include <tvm/api_registry.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>
#include <tvm/arithmetic.h>
#include <arithmetic/const_fold.h>
#include <op/op_util.h>

#include <algorithm>
#include <map>
#include <set>
#include <utility>
#include <vector>
#include <unordered_set>
#include <tuple>

#include "pass/utils.h"
#include "pass/zero_elimination.h"
#include "pass/autodiff_cce.h"
#include "pass/rewrite_simplify_cce.h"

namespace akg {
namespace ir {
using air::IterVarType;
using air::arith::EvalSet;
using air::arith::IntSet;
using air::ir::HasSideEffect;

struct ExprLess {
  bool operator()(const Expr &l, const Expr &r) const { return Compare(l, r) < 0; }
};

struct ExprEq {
  bool operator()(const Expr &l, const Expr &r) const { return Compare(l, r) == 0; }
};

// Merge two maps, prefer the right one on conflict
template <class K, class V>
Map<K, V> Merge(Map<K, V> original, const Map<K, V> &update) {
  for (const auto &p : update) {
    original.Set(p.first, p.second);
  }

  return std::move(original);
}

// Concatenate two arrays
template <class T>
Array<T> Concat(Array<T> a, const Array<T> &b) {
  for (const auto &x : b) {
    a.push_back(x);
  }

  return std::move(a);
}

// Combine all expressions from the container using &&.
template <class container>
Expr All(const container &c) {
  Expr res;
  for (const auto &e : c) {
    if (res.get()) {
      res = res && e;
    } else {
      res = e;
    }
  }

  if (res.get()) {
    return res;
  } else {
    return const_true();
  }
}

// Create a select statement of the form cond ? on_true : 0
Expr SelectElseZero(const Expr &cond, const Expr &on_true) {
  return Select::make(cond, on_true, make_zero(on_true.type()));
}

// Simplify_cce the expression as thoroughly as possible by using all available simplifiers.
Expr SuperSimplify(Expr e, const Map<Var, Range> &vranges) {
  // For some reason no simplifier can detect that there is only one value of the variable
  std::unordered_map<const Variable *, Expr> vmap;
  for (const auto &var_range : vranges) {
    if (is_const_int(var_range.second->extent, 1)) {
      vmap[var_range.first.get()] = var_range.second->min;
    }
  }

  if (!vmap.empty()) {
    e = Substitute(e, vmap);
  }

  e = SimplifyMad().Mutate(e);
  return AutodiffSimplify().Mutate(CanonicalSimplify(Simplify_cce(CanonicalSimplify(e, vranges), vranges), vranges));
}

// Provability check that uses SuperSimplify
bool CanProve(const Expr &e, const Map<Var, Range> &vranges) { return is_one(SuperSimplify(e, vranges)); }

class ExprFreeVarsVisitor : public IRVisitor {
 public:
  std::vector<Var> free_array;
  std::unordered_set<const Variable *> bound;
  std::unordered_set<const Variable *> free;

  void Visit(const NodeRef &node) override {
    if (const auto v = node.as<Variable>()) {
      if (!bound.count(v) && !free.count(v)) {
        free.insert(v);
        free_array.push_back(Downcast<Var>(node));
      }
    } else {
      IRVisitor::Visit(node);
    }
  }

  void Visit_(const Variable *op) override { CHECK(false) << "This case shouldn't happen"; }

  void Visit_(const LetStmt *op) override {
    bound.insert(op->var.get());
    IRVisitor::Visit_(op);
  }

  void Visit_(const For *op) override {
    bound.insert(op->loop_var.get());
    IRVisitor::Visit_(op);
  }

  void Visit_(const Let *op) override {
    bound.insert(op->var.get());
    IRVisitor::Visit_(op);
  }

  void Visit_(const Reduce *op) override {
    for (const auto &iv : op->axis) {
      bound.insert(iv->var.get());
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Store *op) override {
    Visit(op->buffer_var);
    IRVisitor::Visit_(op);
  }

  void Visit_(const Allocate *op) override {
    Visit(op->buffer_var);
    IRVisitor::Visit_(op);
  }

  void Visit_(const Free *op) override {
    Visit(op->buffer_var);
    IRVisitor::Visit_(op);
  }

  void Visit_(const Load *op) override {
    Visit(op->buffer_var);
    IRVisitor::Visit_(op);
  }
};

// Get free variables of an expression
Array<Var> ExprFreeVars(const Expr &expr) {
  ExprFreeVarsVisitor visitor;
  visitor.Visit(expr);
  return visitor.free_array;
}

DomainTransformation ComposeDomainTransformations(const DomainTransformation &first,
                                                  const DomainTransformation &second) {
  CHECK(second->old_domain.same_as(first->new_domain));
  Map<Var, Expr> new_to_old;
  Map<Var, Expr> old_to_new;
  for (auto p : second->new_to_old) {
    new_to_old.Set(p.first, Substitute(p.second, first->new_to_old));
  }

  for (auto p : first->old_to_new) {
    old_to_new.Set(p.first, Substitute(p.second, second->old_to_new));
  }

  return DomainTransformationNode::make(second->new_domain, first->old_domain, new_to_old, old_to_new);
}

DomainTransformation DomainTransformation::operator+=(const DomainTransformation &other) {
  *this = ComposeDomainTransformations(*this, other);
  return *this;
}

DomainTransformation EmptyDomainTransformation(const Domain &domain) {
  Map<Var, Expr> new_to_old;
  Map<Var, Expr> old_to_new;
  for (const Var &v : domain->variables) {
    old_to_new.Set(v, make_zero(v.type()));
  }
  Domain new_domain = DomainNode::make({}, {make_zero(Bool())}, {});
  return DomainTransformationNode::make(new_domain, domain, new_to_old, old_to_new);
}

DomainTransformation IdDomainTransformation(const Domain &domain) {
  Map<Var, Expr> new_to_old;
  for (const Var &v : domain->variables) {
    new_to_old.Set(v, v);
  }

  return DomainTransformationNode::make(domain, domain, new_to_old, new_to_old);
}

// Convert an array of itervars to an array of inequalities
Array<Expr> IterVarsToInequalities(const Array<IterVar> &itervars) {
  Array<Expr> res;
  for (const IterVar &v : itervars) {
    res.push_back(GE::make(v->var, v->dom->min));
    res.push_back(LT::make(v->var, v->dom->min + v->dom->extent));
  }

  return res;
}

// Convert an array of itervars to a map from vars to ranges
Map<Var, Range> IterVarsToMap(const Array<IterVar> &itervars) {
  Map<Var, Range> res;
  for (const IterVar &v : itervars) {
    res.Set(v->var, v->dom);
  }

  return res;
}

// Convert an array of itervars to an array of vars
Array<Var> IterVarsToVars(const Array<IterVar> &itervars) {
  Array<Var> res;
  for (const IterVar &v : itervars) {
    res.push_back(v->var);
  }

  return res;
}

// Given a map from vars to ranges create an array of itervars
Array<IterVar> IterVarsFromMap(const Array<Var> &vars, const Map<Var, Range> &vranges,
                               IterVarType iter_type = IterVarType::kDataPar, const std::string &thread_tag = "") {
  Array<IterVar> res;
  for (const Var &v : vars) {
    CHECK(vranges.count(v)) << "A range for the variable " << v << " was not provided in map " << vranges;
    res.push_back(IterVarNode::make(vranges[v], v, iter_type, thread_tag));
  }

  return res;
}

Expr SimplifyCombiner(const Expr &expr, bool prune_unused_components) {
  const auto op = expr.as<Reduce>();

  // First Simplify the results
  Array<Expr> simplified_result;
  for (const auto &res : op->combiner->result) {
    simplified_result.push_back(SuperSimplify(res, IterVarsToMap(op->axis)));
  }

  // Which components to keep
  std::vector<int> used(op->combiner->result.size(), false);

  if (prune_unused_components) {
    // This function recursively marks the used components starting from
    // the index idx
    std::function<void(int)> mark_used;
    mark_used = [&used, &simplified_result, op, &mark_used](size_t idx) {
      // if the idx-th component was mark as used before, do nothing
      if (used[idx]) return;
      used[idx] = true;

      // check if the idx-th result expr uses some lhs or rhs variables
      // and recursively mark the corresponding components
      for (size_t i = 0; i < simplified_result.size(); ++i)
        if (!used[i]) {
          if (ExprUseVar(simplified_result[idx], op->combiner->lhs[i]) ||
              ExprUseVar(simplified_result[idx], op->combiner->rhs[i]))
            mark_used(static_cast<int>(i));
        }
    };

    // mark all used components starting from the value_index
    mark_used(op->value_index);
  } else {
    // if prunning was not requested, keep all components
    used.assign(used.size(), true);
  }

  int new_value_index = op->value_index;
  Array<Expr> new_result;
  Array<Expr> new_identity;
  Array<Var> new_lhs;
  Array<Var> new_rhs;
  Array<Expr> new_source;

  // Create the new expressions based on the used ones.
  for (size_t i = 0; i < used.size(); ++i) {
    if (used[i]) {
      // We Simplify_cce the result of the combiner and the identity element
      new_result.push_back(simplified_result[i]);
      new_identity.push_back(SuperSimplify(op->combiner->identity_element[i], IterVarsToMap(op->axis)));
      new_lhs.push_back(op->combiner->lhs[i]);
      new_rhs.push_back(op->combiner->rhs[i]);
      new_source.push_back(op->source[i]);
    } else if (static_cast<int>(i) < op->value_index) {
      // value_index should also be adjusted
      new_value_index--;
    }
  }

  CommReducer new_combiner = CommReducerNode::make(new_lhs, new_rhs, new_result, new_identity);
  return Reduce::make(new_combiner, new_source, op->axis, op->condition, new_value_index);
}

// Clone iter vars and return both the new vars and the substitution from old to new.
std::pair<Array<IterVar>, std::unordered_map<const Variable *, Expr>> CloneIterVars(const Array<IterVar> &vars) {
  Array<IterVar> new_vars;
  std::unordered_map<const Variable *, Expr> vmap;
  for (const IterVar &iv : vars) {
    IterVar new_v = IterVarNode::make(iv->dom, iv->var.copy_with_suffix(""), iv->iter_type, iv->thread_tag);
    new_vars.push_back(new_v);
    vmap[iv->var.get()] = new_v;
  }

  return std::make_pair(std::move(new_vars), std::move(vmap));
}

// Clone reduction by cloning the axis variables.
Expr CloneReduction(const Expr &expr) {
  if (const auto red = expr.as<Reduce>()) {
    Array<IterVar> new_axis;
    std::unordered_map<const Variable *, Expr> vmap;
    std::tie(new_axis, vmap) = CloneIterVars(red->axis);

    Array<Expr> src_with_newaxis;
    for (const auto &src : red->source) {
      src_with_newaxis.push_back(Substitute(src, vmap));
    }

    return Reduce::make(red->combiner, src_with_newaxis, new_axis, Substitute(red->condition, vmap), red->value_index);
  } else {
    return expr;
  }
}

// Return true if this combiner is just a sum.
bool IsSumCombiner(const CommReducer &combiner, const Map<Var, Range> &vranges) {
  if (combiner->result.size() != 1) {
    return false;
  }

  if (!is_const_value(SuperSimplify(combiner->identity_element[0], vranges), 0)) {
    return false;
  }

  Expr should_be_zero = SuperSimplify(combiner->result[0] - (combiner->lhs[0] + combiner->rhs[0]), vranges);
  return is_const_value(should_be_zero, 0);
}

// Return true if zero may be factored out of a reduction with this combiner.
bool CanFactorZeroFromCombiner(const CommReducer &combiner, int value_index, const Map<Var, Range> &vranges) {
  if (!is_const_value(SuperSimplify(combiner->identity_element[value_index], vranges), 0)) {
    return false;
  }

  Expr zero = make_zero(combiner->result[value_index].type());
  Expr in =
    Substitute(combiner->result[value_index], {{combiner->lhs[value_index], zero}, {combiner->rhs[value_index], zero}});
  in = SuperSimplify(in, vranges);

  return is_const_value(in, 0);
}

// If expr is a Call node, perform inlining, otherwise do nothing
Expr InlineThisCall(const Expr &expr) {
  if (const auto op = expr.as<Call>()) {
    if (op->call_type == Call::CallType::Halide) {
      if (const auto op_comp = op->func.as<ComputeOpNode>()) {
        Array<Var> tensor_axes;
        for (const auto &var : op_comp->axis) {
          tensor_axes.push_back(var->var);
        }

        Stmt inlined = Inline(Evaluate::make(expr), op->func, tensor_axes, op_comp->body[op->value_index]);
        if (const auto ev = inlined.as<Evaluate>()) {
          // If it is a reduction, clone it
          return CloneReduction(ev->value);
        }
      }
    }
  }

  return expr;
}

Tensor InlineTailCall(const Tensor &tensor) { return TransformBody(tensor, InlineThisCall); }

// Implements InlineTensors by trying to inline every Call of the given Expr
class InlineTensorsMutator : public IRMutator {
 public:
  explicit InlineTensorsMutator(const Array<Tensor> &inlineable, bool inline_reductions = false)
      : inline_reductions_(inline_reductions) {
    for (const Tensor &tensor : inlineable) {
      inlineable_.emplace(tensor->op.operator->(), tensor->value_index);
    }
  }
  ~InlineTensorsMutator() override = default;

  Expr Mutate_(const Call *op, const Expr &e) override {
    if (op->call_type == Call::CallType::Halide) {
      if (const auto op_comp = op->func.as<ComputeOpNode>()) {
        // Inline only if the array of inlineable tensors is empty or contains this tensor
        if (inlineable_.empty() || inlineable_.count({op_comp, op->value_index})) {
          // Inline only compute nodes that are not reductions (unless inline reductions is allowed)
          if (inline_reductions_ || !op_comp->body[0].as<Reduce>()) {
            // Inline this call and then try to perform further inlining
            return Mutate(InlineThisCall(e));
          }
        }
      }
    }

    // If we cannot inline this call, we should try to do inlining in its arguments
    return IRMutator::Mutate_(op, e);
  }

 private:
  // Tensors which are allowed to be inlined, represented as pairs (op_node, value_index)
  std::set<std::pair<const OperationNode *, int>> inlineable_;
  bool inline_reductions_;
};

Expr InlineTensors(const Expr &expr, const Array<Tensor> &inlineable, bool inline_reductions) {
  return InlineTensorsMutator(inlineable, inline_reductions).Mutate(expr);
}

Tensor InlineTensors(const Tensor &tensor, const Array<Tensor> &inlineable, bool inline_reductions) {
  auto transformation = [inlineable, inline_reductions](const Expr &e) {
    return InlineTensorsMutator(inlineable, inline_reductions).Mutate(e);
  };

  return TransformBody(tensor, transformation);
}

struct NonzeronessConditionResult {
  Expr cond;
  Expr value;

  Expr to_expr() const { return SelectElseZero(cond, value); }
};

// The implementation of NonzeronessCondition
class NonzeronessConditionFunctor
    : public air::ir::ExprFunctor<NonzeronessConditionResult(const Expr &, const Expr &)> {
 public:
  NonzeronessConditionResult NonzeronessCondition(const Expr &e) {
    if (e.type().is_bool()) {
      // Boolean expressions are non-zero whenever they are true themselves
      return {e, const_true()};
    } else {
      return VisitExpr(e, e);
    }
  }

  // Most of the cases are implemented using helpers below
  result_type VisitExpr_(const Variable *, const Expr &e) final { return Default_(e); }
  result_type VisitExpr_(const IntImm *op, const Expr &e) final { return Const_(op, e); }
  result_type VisitExpr_(const UIntImm *op, const Expr &e) final { return Const_(op, e); }
  result_type VisitExpr_(const FloatImm *op, const Expr &e) final { return Const_(op, e); }
  result_type VisitExpr_(const StringImm *, const Expr &e) final { return Default_(e); }
  result_type VisitExpr_(const Add *op, const Expr &e) final { return BinOpAddLike_(op, e); }
  result_type VisitExpr_(const Sub *op, const Expr &e) final { return BinOpAddLike_(op, e); }
  result_type VisitExpr_(const Mul *op, const Expr &e) final { return BinOpMulLike_(op, e); }
  result_type VisitExpr_(const Div *op, const Expr &e) final { return BinOpDivLike_(op, e); }
  result_type VisitExpr_(const FloorDiv *op, const Expr &e) final { return BinOpDivLike_(op, e); }
  result_type VisitExpr_(const Mod *op, const Expr &e) final { return BinOpDivLike_(op, e); }
  result_type VisitExpr_(const FloorMod *op, const Expr &e) final { return BinOpDivLike_(op, e); }
  result_type VisitExpr_(const Min *op, const Expr &e) final { return BinOpAddLike_(op, e); }
  result_type VisitExpr_(const Max *op, const Expr &e) final { return BinOpAddLike_(op, e); }

  result_type VisitExpr_(const Cast *op, const Expr &e) final {
    auto nz_a = NonzeronessCondition(op->value);
    if (nz_a.value.same_as(op->value)) {
      return {nz_a.cond, e};
    } else {
      return {nz_a.cond, Cast::make(op->type, nz_a.value)};
    }
  }

  result_type VisitExpr_(const Select *op, const Expr &e) final {
    Expr cond = op->condition, true_val = op->true_value, false_val = op->false_value;
    auto nz_a = NonzeronessCondition(true_val);
    auto nz_b = NonzeronessCondition(false_val);
    // If the false part is zero, we can get rid of the select
    if (is_const_value(nz_b.value, 0)) {
      Expr new_cond = SuperSimplify(nz_a.cond && cond);
      return {new_cond, nz_a.value};
    }

    // If the true part is zero, we can also get rid of the select
    if (is_const_value(nz_a.value, 0)) {
      Expr new_cond = SuperSimplify(nz_b.cond && !cond);
      return {new_cond, nz_b.value};
    }

    // Otherwise we retain the select and combine the conditions into this
    Expr new_cond = SuperSimplify((cond && nz_a.cond) || (!cond && nz_b.cond));
    if (nz_a.value.same_as(true_val) && nz_b.value.same_as(false_val)) {
      return {new_cond, e};
    } else {
      return {new_cond, Select::make(cond, nz_a.value, nz_b.value)};
    }
  }

  result_type VisitExpr_(const Call *op, const Expr &e) final {
    if (op->name == air::ir::intrinsic::tvm_if_then_else) {
      Expr cond = op->args[0], true_val = op->args[1], false_val = op->args[2];
      auto nz_a = NonzeronessCondition(true_val);
      auto nz_b = NonzeronessCondition(false_val);

      // We don't have as much freedom here as in the select case
      // since the `if` must be preserved in any case
      Expr new_cond = SuperSimplify((cond && nz_a.cond) || (!cond && nz_b.cond));
      if (nz_a.value.same_as(true_val) && nz_b.value.same_as(false_val)) {
        return {new_cond, e};
      } else {
        return {new_cond, if_then_else(cond, nz_a.value, nz_b.value)};
      }
    } else {
      return Default_(e);
    }
  }

  NonzeronessConditionResult Default_(const Expr &e) const {
    // This is always correct, so it's the default
    return {const_true(), e};
  }

  template <class TNode>
  NonzeronessConditionResult Const_(const TNode *op, const Expr &e) {
    if (op->value == 0) {
      return {const_false(), e};
    } else {
      return {const_true(), e};
    }
  }

  template <class TNode>
  NonzeronessConditionResult BinOpAddLike_(const TNode *op, const Expr &e) {
    auto nz_a = NonzeronessCondition(op->a);
    auto nz_b = NonzeronessCondition(op->b);
    // For addition and similar ops the result may be nonzero if either of the arguments is
    // nonzero, so we combine the conditions with Or.
    if (Equal(nz_a.cond, nz_b.cond)) {
      // If the conditions are the same, we don't need Or
      if (nz_a.value.same_as(op->a) && nz_b.value.same_as(op->b)) {
        return {nz_a.cond, e};
      } else {
        return {nz_a.cond, TNode::make(nz_a.value, nz_b.value)};
      }
    } else {
      // Otherwise use Or
      Expr new_cond = SuperSimplify(nz_a.cond || nz_b.cond);
      // A little optimization: if the combined condition is the same as one of the inner
      // conditions, we don't need to guard the inner value with a select, otherwise
      // we create a select in the `to_expr` call.
      Expr new_a = Equal(nz_a.cond, new_cond) ? nz_a.value : nz_a.to_expr();
      Expr new_b = Equal(nz_b.cond, new_cond) ? nz_b.value : nz_b.to_expr();
      Expr new_expr = TNode::make(new_a, new_b);

      return {new_cond, new_expr};
    }
  }

  template <class TNode>
  NonzeronessConditionResult BinOpMulLike_(const TNode *op, const Expr &e) {
    auto nz_a = NonzeronessCondition(op->a);
    auto nz_b = NonzeronessCondition(op->b);

    // For multiplication and similar ops the result may be nonzero if
    // both the arguments are nonzero, so we combine with And.
    Expr new_cond = SuperSimplify(nz_a.cond && nz_b.cond);

    if (nz_a.value.same_as(op->a) && nz_b.value.same_as(op->b)) {
      return {new_cond, e};
    } else {
      return {new_cond, TNode::make(nz_a.value, nz_b.value)};
    }
  }

  template <class TNode>
  NonzeronessConditionResult BinOpDivLike_(const TNode *op, const Expr &e) {
    auto nz_a = NonzeronessCondition(op->a);
    // For Div we simply use the condition of the numerator.
    if (nz_a.value.same_as(op->a)) {
      return {nz_a.cond, e};
    } else {
      return {nz_a.cond, TNode::make(nz_a.value, op->b)};
    }
  }
};

// Transform expr into a pair (condition, new_expr) such that the old expr is equivalent to
// `select(condition, new_expr, 0)`. The pair is represented as a struct for clarity.
NonzeronessConditionResult NonzeronessCondition(const Expr &expr) {
  return NonzeronessConditionFunctor().NonzeronessCondition(expr);
}

Expr LiftNonzeronessCondition(const Expr &expr) { return NonzeronessCondition(expr).to_expr(); }

class NormalizeComparisonsMutator : public IRMutator {
 public:
  Expr Mutate_(const EQ *op, const Expr &e) override { return Make<EQ>(op->a, op->b); }
  Expr Mutate_(const NE *op, const Expr &e) override { return Make<NE>(op->a, op->b); }
  Expr Mutate_(const LT *op, const Expr &e) override { return Make<LT>(op->a, op->b); }
  Expr Mutate_(const LE *op, const Expr &e) override { return Make<LE>(op->a, op->b); }
  Expr Mutate_(const GT *op, const Expr &e) override { return Make<LT>(op->b, op->a); }
  Expr Mutate_(const GE *op, const Expr &e) override { return Make<LE>(op->b, op->a); }

 private:
  template <class TNode>
  Expr Make(const Expr &a, const Expr &b) {
    // rewrite LT to LE for ints
    if (std::is_same<TNode, LT>::value && (a.type().is_int() || a.type().is_uint())) {
      return LE::make(SuperSimplify(a - b + 1), make_zero(a.type()));
    }

    return TNode::make(SuperSimplify(a - b), make_zero(a.type()));
  }
};

// Rewrite every comparison into the form a == 0, a != 0, a <= 0, and sometimes for floats a < 0
Expr NormalizeComparisons(const Expr &expr) { return NormalizeComparisonsMutator().Mutate(expr); }

struct FactorOutAtomicFormulasResult {
  std::vector<Expr> atomic_formulas;
  Expr rest;

  Expr to_expr() const {
    Expr res = rest;
    for (const Expr &e : atomic_formulas) {
      res = And::make(e, res);
    }

    return res;
  }

  Array<Expr> to_array() const {
    Array<Expr> res = atomic_formulas;
    res.push_back(rest);

    return res;
  }
};

// The implementation of FactorOutAtomicFormulas
class FactorOutAtomicFormulasFunctor
    : public air::ir::ExprFunctor<FactorOutAtomicFormulasResult(const Expr &, const Expr &)> {
 public:
  result_type Atomic_(const Expr &e) {
    // For atomic expressions the result is the expr itself with True as the residual
    return {{e}, make_const(e.type(), 1)};
  }

  // This is basically the list of expression kinds that are considered atomic
  result_type VisitExpr_(const Variable *, const Expr &e) final { return Atomic_(e); }
  result_type VisitExpr_(const Call *, const Expr &e) final { return Atomic_(e); }
  result_type VisitExpr_(const IntImm *, const Expr &e) final { return Atomic_(e); }
  result_type VisitExpr_(const UIntImm *, const Expr &e) final { return Atomic_(e); }
  result_type VisitExpr_(const EQ *, const Expr &e) final { return Atomic_(e); }
  result_type VisitExpr_(const NE *, const Expr &e) final { return Atomic_(e); }
  result_type VisitExpr_(const LE *, const Expr &e) final { return Atomic_(e); }
  result_type VisitExpr_(const LT *, const Expr &e) final { return Atomic_(e); }
  result_type VisitExpr_(const GE *, const Expr &e) final { return Atomic_(e); }
  result_type VisitExpr_(const GT *, const Expr &e) final { return Atomic_(e); }

  result_type VisitExpr_(const And *op, const Expr &e) final {
    auto res_a = VisitExpr(op->a, op->a);
    auto res_b = VisitExpr(op->b, op->b);

    // For the And case we return the union of the sets of atomic formulas
    std::vector<Expr> res;
    res.reserve(res_a.atomic_formulas.size() + res_b.atomic_formulas.size());
    std::set_union(res_a.atomic_formulas.begin(), res_a.atomic_formulas.end(), res_b.atomic_formulas.begin(),
                   res_b.atomic_formulas.end(), std::back_inserter(res), ExprLess());

    // And the residuals are combined with &&
    return {res, res_a.rest && res_b.rest};
  }

  result_type VisitExpr_(const Mul *op, const Expr &e) final {
    auto res_a = VisitExpr(op->a, op->a);
    auto res_b = VisitExpr(op->b, op->b);

    // For multiplication we do the same thing as for And
    std::vector<Expr> res;
    res.reserve(res_a.atomic_formulas.size() + res_b.atomic_formulas.size());
    std::set_union(res_a.atomic_formulas.begin(), res_a.atomic_formulas.end(), res_b.atomic_formulas.begin(),
                   res_b.atomic_formulas.end(), std::back_inserter(res), ExprLess());

    return {res, res_a.rest * res_b.rest};
  }

  result_type VisitExpr_(const Or *op, const Expr &e) final {
    auto res_a = VisitExpr(op->a, op->a);
    auto res_b = VisitExpr(op->b, op->b);

    // For the Or case we intersect the sets of atomic formulas
    std::vector<Expr> res;
    res.reserve(std::min(res_a.atomic_formulas.size(), res_b.atomic_formulas.size()));
    std::set_intersection(res_a.atomic_formulas.begin(), res_a.atomic_formulas.end(), res_b.atomic_formulas.begin(),
                          res_b.atomic_formulas.end(), std::back_inserter(res), ExprLess());

    // Computing the residual is more complex: we have to compute the sets of atomic formulas
    // which are left behind, and then combine them with the residuals into the new residual.

    std::vector<Expr> new_cond_a;
    new_cond_a.reserve(res_a.atomic_formulas.size() - res.size());
    std::set_difference(res_a.atomic_formulas.begin(), res_a.atomic_formulas.end(), res.begin(), res.end(),
                        std::back_inserter(new_cond_a), ExprLess());

    std::vector<Expr> new_cond_b;
    new_cond_b.reserve(res_b.atomic_formulas.size() - res.size());
    std::set_difference(res_b.atomic_formulas.begin(), res_b.atomic_formulas.end(), res.begin(), res.end(),
                        std::back_inserter(new_cond_b), ExprLess());

    res_a.atomic_formulas = std::move(new_cond_a);
    res_b.atomic_formulas = std::move(new_cond_b);

    Expr new_rest = (res_a.to_expr() || res_b.to_expr());

    return {res, new_rest};
  }
};

// Transform the given formula into a conjunction of atomic formulas (represented as an array)
// and a non-atomic residual. Atomic formulas are consts, calls, variables and comparisons (a <= b,
// etc), i.e. formulas which are not logical operators (||, &&, !) on the top level.
FactorOutAtomicFormulasResult FactorOutAtomicFormulas(const Expr &e) {
  return FactorOutAtomicFormulasFunctor().VisitExpr(e, e);
}

class RemoveRedundantInequalitiesMutator : public IRMutator {
 public:
  explicit RemoveRedundantInequalitiesMutator(Array<Expr> known) {
    for (const Expr &cond : known) {
      known_.push_back(SuperSimplify(cond));
    }
  }
  ~RemoveRedundantInequalitiesMutator() override = default;

  Expr Mutate_(const Select *op, const Expr &e) override {
    bool has_side_effect = HasSideEffect(e);
    Expr new_cond = SuperSimplify(Mutate(op->condition));
    if (is_one(new_cond) && !has_side_effect) {
      return Mutate(op->true_value);
    } else if (is_zero(new_cond) && !has_side_effect) {
      return Mutate(op->false_value);
    } else {
      Array<Expr> new_known = known_;
      for (const Expr &atomic : FactorOutAtomicFormulas(new_cond).atomic_formulas) {
        new_known.push_back(atomic);
      }
      RemoveRedundantInequalitiesMutator new_mutator(new_known);

      // Note that we mutate only the true value with the new mutator
      return Select::make(new_cond, new_mutator.Mutate(op->true_value), Mutate(op->false_value));
    }
  }

  Expr Mutate_(const Call *op, const Expr &e) override {
    if (op->name == air::ir::intrinsic::tvm_if_then_else) {
      Expr new_cond = SuperSimplify(Mutate(op->args[0]));
      if (is_one(new_cond)) {
        return Mutate(op->args[1]);
      } else if (is_zero(new_cond)) {
        return Mutate(op->args[2]);
      } else {
        Array<Expr> new_known = known_;
        for (const Expr &atomic : FactorOutAtomicFormulas(new_cond).atomic_formulas) {
          new_known.push_back(atomic);
        }
        RemoveRedundantInequalitiesMutator new_mutator(new_known);

        // Note that we mutate only the true value with the new mutator
        return if_then_else(new_cond, new_mutator.Mutate(op->args[1]), Mutate(op->args[2]));
      }
    } else {
      return IRMutator::Mutate_(op, e);
    }
  }

  Expr Mutate_(const Reduce *op, const Expr &e) override {
    Array<Expr> known_with_axes = known_;
    for (const Expr &axis_cond : IterVarsToInequalities(op->axis)) {
      known_with_axes.push_back(axis_cond);
    }
    RemoveRedundantInequalitiesMutator mutator_with_axes(known_with_axes);

    Expr new_cond = mutator_with_axes.Mutate(op->condition);

    Array<Expr> new_known = known_with_axes;
    for (const Expr &atomic : FactorOutAtomicFormulas(new_cond).atomic_formulas) {
      new_known.push_back(atomic);
    }
    RemoveRedundantInequalitiesMutator new_mutator(new_known);

    Array<Expr> new_source;
    for (const Expr &src : op->source) {
      new_source.push_back(new_mutator.Mutate(src));
    }

    return Reduce::make(op->combiner, new_source, op->axis, new_cond, op->value_index);
  }

  Expr Mutate_(const EQ *op, const Expr &e) override { return MutateAtomic_(e); }
  Expr Mutate_(const NE *op, const Expr &e) override { return MutateAtomic_(e); }
  Expr Mutate_(const LT *op, const Expr &e) override { return MutateAtomic_(e); }
  Expr Mutate_(const LE *op, const Expr &e) override { return MutateAtomic_(e); }
  Expr Mutate_(const GT *op, const Expr &e) override { return MutateAtomic_(e); }
  Expr Mutate_(const GE *op, const Expr &e) override { return MutateAtomic_(e); }

  Expr Mutate_(const And *op, const Expr &e) override { return (Mutate(op->a) && Mutate(op->b)); }

 private:
  Expr MutateAtomic_(const Expr &e) {
    Expr simplified = SuperSimplify(e);
    for (const Expr &other : known_) {
      if (Equal(simplified, other)) {
        return const_true();
      }
    }

    return simplified;
  }

  Array<Expr> known_;
};

// Propagate information from conditions and remove redundant inequalities
Expr RemoveRedundantInequalities(const Expr &expr, const Array<Expr> &known) {
  return RemoveRedundantInequalitiesMutator(known).Mutate(expr);
}

struct EliminateDivModResult {
  Expr expr;
  Map<Var, Expr> substitution;
  Array<Var> new_variables;
  Array<Expr> conditions;
  Map<Var, Range> ranges;
};

class EliminateDivModMutator : public IRMutator {
 public:
  Map<Var, Expr> substitution;
  Array<Var> new_variables;
  Array<Expr> conditions;
  Map<Var, Range> ranges;

  explicit EliminateDivModMutator(const Map<Var, Range> &ranges) : ranges(ranges) {}
  ~EliminateDivModMutator() override = default;

  Expr Mutate_(const Div *op, const Expr &e) override {
    const auto imm = op->b.as<IntImm>();
    if (imm && (imm->value > 0)) {
      // Try to find the already existing variables for this expression
      auto it = expr_to_vars_.find({op->a, imm->value});
      if (it != expr_to_vars_.end()) {
        return it->second.first;
      }

      // Otherwise recursively mutate the left hand side, and create new variables
      Expr mutated_a = Mutate(op->a);
      if (auto var_pair_opt = AddNewVarPair(op->a, mutated_a, imm->value)) {
        return var_pair_opt.value().first;
      } else {
        return truncdiv(mutated_a, op->b);
      }
    }

    return Mutate(op->a) / Mutate(op->b);
  }

  Expr Mutate_(const Mod *op, const Expr &e) override {
    const auto imm = op->b.as<IntImm>();
    if (imm && (imm->value > 0)) {
      // Try to find the already existing variables for this expression
      auto it = expr_to_vars_.find({op->a, imm->value});
      if (it != expr_to_vars_.end()) {
        return it->second.second;
      }

      // Otherwise recursively mutate the left hand side, and create new variables
      Expr mutated_a = Mutate(op->a);
      if (auto var_pair_opt = AddNewVarPair(op->a, mutated_a, imm->value)) {
        return var_pair_opt.value().second;
      } else {
        return mutated_a % op->b;
      }
    }

    return Mutate(op->a) % Mutate(op->b);
  }

 private:
  dmlc::optional<std::pair<Var, Var>> AddNewVarPair(const Expr &e, const Expr &mut, int64_t val) {
    using tresult = dmlc::optional<std::pair<Var, Var>>;

    // Try to find the variables using the mutated expressions
    if (!e.same_as(mut)) {
      auto it = expr_to_vars_.find({mut, val});
      if (it != expr_to_vars_.end()) {
        return tresult(it->second);
      }
    }

    Expr val_e = make_const(e.type(), val);
    idx_ += 1;

    // Convert `ranges` to IntSets
    std::unordered_map<const Variable *, IntSet> var_intsets;
    for (const auto &p : ranges) {
      var_intsets[p.first.get()] = IntSet::range(p.second);
    }

    // Infer ranges for the expressions we want to replace with variables
    Range div_range = EvalSet(mut / val_e, var_intsets).cover_range(Range());
    Range mod_range = EvalSet(mut % val_e, var_intsets).cover_range(Range());
    // We don't want to add unbounded variables
    if (!div_range.get() || !mod_range.get()) {
      LOG(WARNING) << "EliminateDivMod: won't eliminate div or mod of expr " << e
                   << "  because its bounds cannot be inferred";
      return tresult();
    }

    // Create new variables for the expressions
    auto div = Var("div" + std::to_string(idx_), e.type());
    auto mod = Var("mod" + std::to_string(idx_), e.type());

    new_variables.push_back(div);
    new_variables.push_back(mod);

    substitution.Set(div, mut / val_e);
    substitution.Set(mod, mut % val_e);

    ranges.Set(div, div_range);
    ranges.Set(mod, mod_range);

    // This additional condition works as a definition for the new variables
    conditions.push_back(mut == div * val_e + mod);

    if (!CanProve(mod_range->extent <= val_e)) {
      // Since we use the C/C++ definition of mod, there may be multiple values of `mod`
      // satisfying the added condition if the expr `e` may change its sign, so we
      // have to add another condition.
      LOG(WARNING) << "EliminateDivMod: cannot fully eliminate div or mod of expr " << e
                   << "  (probably it may change its sign)";
      conditions.push_back(Select::make(e >= 0, mod >= 0, mod <= 0));
    }

    auto p = std::make_pair(div, mod);
    expr_to_vars_[{e, val}] = p;
    if (!e.same_as(mut)) {
      expr_to_vars_[{mut, val}] = p;
    }

    return tresult(p);
  }

  // A custom comparison function for pairs of exprs and numbers. Compares exprs deeply.
  struct Compare_ {
    bool operator()(const std::pair<Expr, int64_t> &p1, const std::pair<Expr, int64_t> &p2) const {
      if (p1.second < p2.second) {
        return true;
      } else if (p1.second == p2.second) {
        return Compare(p1.first, p2.first) < 0;
      } else {
        return false;
      }
    }
  };

  // A counter for naming new variables
  int idx_{0};
  // A map from pairs of exprs and numbers (e, n) to pairs of new vars (div, mod)
  // such that `div = e / n` and `mod = e % n`
  std::map<std::pair<Expr, int64_t>, std::pair<Var, Var>, Compare_> expr_to_vars_;
};

// Replace every subexpr of the form e/const and e % const with a new variable.
// Syntactically equal expressions will be mapped to the same variable.
EliminateDivModResult EliminateDivMod(const Expr &expr, Map<Var, Range> ranges) {
  EliminateDivModResult res;
  EliminateDivModMutator mutator(std::move(ranges));

  res.expr = mutator.Mutate(expr);
  res.conditions = std::move(mutator.conditions);
  res.new_variables = std::move(mutator.new_variables);
  res.substitution = std::move(mutator.substitution);
  res.ranges = std::move(mutator.ranges);

  return res;
}

// run EliminateDivMod from the conditions of a domain
DomainTransformation EliminateDivModFromDomainConditions(const Domain &domain) {
  auto elim_res = EliminateDivMod(All(domain->conditions), domain->ranges);

  Map<Var, Range> new_vranges = elim_res.ranges;
  Array<Var> new_axis = Concat(domain->variables, elim_res.new_variables);
  Expr new_cond = elim_res.expr && All(elim_res.conditions);

  Domain new_domain = DomainNode::make(new_axis, FactorOutAtomicFormulas(new_cond).to_array(), new_vranges);

  Map<Var, Expr> old_to_new;
  Map<Var, Expr> new_to_old = elim_res.substitution;
  for (const Var &v : domain->variables) {
    old_to_new.Set(v, v);
    new_to_old.Set(v, v);
  }

  return DomainTransformationNode::make(new_domain, domain, new_to_old, old_to_new);
}

std::tuple<int64_t, int64_t, int64_t> xgcd(int64_t a, int64_t b) {
  int64_t s = 0, old_s = 1;
  int64_t t = 1, old_t = 0;
  int64_t r = b, old_r = a;

  while (r != 0) {
    int64_t q = old_r / r;
    std::swap(r, old_r);
    r -= q * old_r;
    std::swap(s, old_s);
    s -= q * old_s;
    std::swap(t, old_t);
    t -= q * old_t;
  }

  CHECK_NE(old_r, 0);
  CHECK_EQ(a % old_r, 0);
  CHECK_EQ(b % old_r, 0);
  CHECK(old_r == old_s * a + old_t * b);

  return std::make_tuple(old_r, old_s, old_t);
}

DomainTransformation SolveSystemOfEquations(const Domain &domain) {
  // Conditions we don't know what to do with
  std::vector<Expr> rest;
  // Matrix represented as a vector of rows, each row is an array of coefficients
  std::vector<std::vector<int64_t>> matrix;
  // A column of right hand sides
  std::vector<Expr> rhs;
  // A map from old vars to new vars represented as a matrix, each row of this matrix corresponds to
  // an old variable (from domain->variables) and represents a vector of coefficients
  std::vector<std::vector<int64_t>> old_to_new;
  // A map from new vars to old vars represented directly as an array of expressions
  std::vector<Expr> new_to_old;

  size_t vars_size = domain->variables.size();

  // Initialize the old_to_new matrix with the identity matrix
  for (size_t i = 0; i < vars_size; ++i) {
    old_to_new.emplace_back(vars_size);
    old_to_new.back()[i] = 1;
    new_to_old.push_back(domain->variables[i]);
  }

  // Transform formulas into rows of the matrix
  for (const Expr &formula : domain->conditions) {
    if (const auto eq = formula.as<EQ>()) {
      Array<Expr> coefs =
        air::arith::DetectLinearEquation(SuperSimplify(eq->a - eq->b, domain->ranges), domain->variables);
      if (!coefs.empty()) {
        std::vector<int64_t> row;
        for (size_t j = 0; j < coefs.size() - 1; ++j) {
          Expr c = coefs[j];
          if (const auto intimm = c.as<IntImm>()) {
            row.push_back(intimm->value);
          } else {
            row.clear();
            break;
          }
        }

        if (!row.empty()) {
          matrix.push_back(row);
          rhs.push_back(-coefs[coefs.size() - 1]);
          continue;
        }
      }
    }

    // otherwise
    rest.push_back(formula);
  }

  // Diagonalize the matrix
  for (size_t index = 0; index < std::min(matrix.size(), vars_size); ++index) {
    // Here the matrix is partially diagonalized, that is matrix[i, j] is zero for all i, j
    // such that (i < index) or (j < index), unless (i == j).
    // That is, now we are diagonalizing the submatrix with i >= index and j >= index

    // Find a row with a nonzero element in the index-th column
    // (We also prefer rows where this element has minimal abs value)
    size_t best_i = index;
    for (auto i = best_i; i < matrix.size(); ++i) {
      int64_t m_old = matrix[best_i][index];
      int64_t m_new = matrix[i][index];
      if (m_new != 0) {
        if ((m_old == 0) || (std::abs(m_new) < std::abs(m_old))) {
          best_i = i;
        }
      }
    }
    // Move the row we found to the index-th position
    std::swap(matrix[index], matrix[best_i]);
    std::swap(rhs[index], rhs[best_i]);

    // If the index-th diagonal element is still zero, try to find a column with nonzero index-th
    // element and move it to the index-th position
    if (matrix[index][index] == 0) {
      for (size_t j = index + 1; j < vars_size; ++j) {
        if (matrix[index][j] != 0) {
          for (size_t i = index; i < matrix.size(); ++i) {
            std::swap(matrix[i][index], matrix[i][j]);
          }
          // swapping columns corresponds to swapping the corresponding new variables
          std::swap(new_to_old[index], new_to_old[j]);
          for (size_t i = 0; i < old_to_new.size(); ++i) {
            std::swap(old_to_new[i][index], old_to_new[i][j]);
          }
          break;
        }
      }
    }

    // If the index-th diagonal element is still zero, then both the index-th row and the index-th
    // column are completely zero, and we don't need to do anything; just go to the next index
    if (matrix[index][index] == 0) {
      continue;
    }

    // Now the index-th diagonal element is non-zero and we can zero all the index-th column
    // below it by subtracting rows from each other
    for (auto i = index + 1; i < matrix.size(); ++i) {
      if (matrix[i][index] != 0) {
        int64_t g, a, b;
        // g = a*matrix[index][index] + b*matrix[i][index]
        CHECK_NE(matrix[index][index], 0);
        if (matrix[i][index] % matrix[index][index] != 0) {
          std::tie(g, a, b) = xgcd(matrix[index][index], matrix[i][index]);
        } else {
          // Explicitly avoid changing the index-th row. This is important to avoid infinite
          // loop.
          g = matrix[index][index];
          a = 1;
          b = 0;
        }

        // Let m = matrix[index][index], n = matrix[i][index], then the following is true:
        //
        // [ a   n/g ][ m/g  n/g ] = [ 1  0 ]
        // [ b  -m/g ][ b    -a  ] = [ 0  1 ]
        //
        // Note that the two matrices are integer (since g = gcd(m, n)).
        // We will essentially multiply our matrix on the left by a dilated and transposed version
        // of the first of these two matrices. The second matrix is not needed here, however we will
        // use it while zeroing the index-th row.

        CHECK_NE(g, 0);
        int64_t m_g = matrix[index][index] / g;
        int64_t n_g = matrix[i][index] / g;

        // Note that j is the index of the column, not the row
        for (size_t j = index; j < matrix[i].size(); ++j) {
          // Multiply index-th row by a and add the i-th row multiplied by b
          // This will make the index-th diagonal element equal to the gcd
          int64_t new_index_j = a * matrix[index][j] + b * matrix[i][j];
          // This transformation performs zeroing of matrix[i][index]
          int64_t new_i_j = n_g * matrix[index][j] - m_g * matrix[i][j];
          matrix[index][j] = new_index_j;
          matrix[i][j] = new_i_j;
        }
        // We have to do the same with rhs
        Expr ea = make_const(rhs[index].type(), a);
        Expr eb = make_const(rhs[i].type(), b);
        Expr e_m_g = make_const(rhs[i].type(), m_g);
        Expr e_n_g = make_const(rhs[index].type(), n_g);
        Expr new_index_rhs = ea * rhs[index] + eb * rhs[i];
        Expr new_i_rhs = e_n_g * rhs[index] - e_m_g * rhs[i];
        rhs[index] = new_index_rhs;
        rhs[i] = new_i_rhs;
      }
    }

    bool changed = false;

    // Now we have to zero the elements of the index-th row by manipulating columns.
    // This is more difficult because column manipulation corresponds to variable manipulation,
    // but the algorithm is essentially the same as before.
    for (size_t j = index + 1; j < vars_size; ++j) {
      if (matrix[index][j] != 0) {
        int64_t g, a, b;
        // g = a*matrix[index][index] + b*matrix[index][j]
        CHECK_NE(matrix[index][index], 0);
        if (matrix[index][j] % matrix[index][index] != 0) {
          std::tie(g, a, b) = xgcd(matrix[index][index], matrix[index][j]);
          // During this phase we may disrupt the zeroness of the index-th column, so we will
          // have to take some action if this might have happened.
          changed = true;
        } else {
          // Explicitly avoid changing the index-th column. This is important to avoid infinite
          // loop. Note that here we don't have to set `changed` to true since we don't change the
          // index-th column.
          g = matrix[index][index];
          a = 1;
          b = 0;
        }

        // Let m = matrix[index][index], n = matrix[index][j], then the following is true:
        //
        // [ a   n/g ][ m/g  n/g ] = [ 1  0 ]
        // [ b  -m/g ][ b    -a  ] = [ 0  1 ]
        //
        // Now we are going to multiply our matrix on the right (to manipulate columns instead of
        // rows), we will also transform the old_to_new matrix the same way, and we will use the
        // second matrix to transform new_to_old.

        CHECK_NE(g, 0);
        int64_t m_g = matrix[index][index] / g;
        int64_t n_g = matrix[index][j] / g;

        for (size_t i = index; i < matrix.size(); ++i) {
          int64_t new_i_index = a * matrix[i][index] + b * matrix[i][j];
          int64_t new_i_j = n_g * matrix[i][index] - m_g * matrix[i][j];
          matrix[i][index] = new_i_index;
          matrix[i][j] = new_i_j;
        }

        // We do exactly the same transformations with old_to_new
        for (size_t i = 0; i < old_to_new.size(); ++i) {
          int64_t new_i_index = a * old_to_new[i][index] + b * old_to_new[i][j];
          int64_t new_i_j = n_g * old_to_new[i][index] - m_g * old_to_new[i][j];
          old_to_new[i][index] = new_i_index;
          old_to_new[i][j] = new_i_j;
        }

        // And apply reverse transformations to new_to_old.
        Expr ea = make_const(new_to_old[j].type(), a);
        Expr eb = make_const(new_to_old[index].type(), b);
        Expr e_m_g = make_const(new_to_old[index].type(), m_g);
        Expr e_n_g = make_const(new_to_old[j].type(), n_g);
        Expr new_index = e_m_g * new_to_old[index] + e_n_g * new_to_old[j];
        Expr new_j = eb * new_to_old[index] - ea * new_to_old[j];
        new_to_old[index] = new_index;
        new_to_old[j] = new_j;
      }
    }

    if (changed) {
      // We might have changed the first column, so we have to zero it once more (or at least check
      // if it's zero), so just perform this iteration once more.
      index -= 1;
    }
  }

  // Set the signs for new variables to have positive sign on 1st dependent input variable
  for (size_t ii = 0; ii < new_to_old.size(); ii++) {
    Array<Expr> coefs =
      air::arith::DetectLinearEquation(SuperSimplify(new_to_old[ii], domain->ranges), domain->variables);
    if (!coefs.empty()) {
      for (size_t jj = 0; jj < coefs.size() - 1; ++jj) {
        Expr c = coefs[jj];
        if (const auto intimm = c.as<IntImm>()) {
          if (intimm->value == 0) continue;

          if (intimm->value < 0) {
            new_to_old[ii] = Simplify_cce(-new_to_old[ii], domain->ranges);
            for (size_t kk = 0; kk < old_to_new.size(); kk++) {
              old_to_new[kk][ii] = -old_to_new[kk][ii];
            }
          } else {
            break;
          }
        }
      }
    }
  }

  Array<Var> new_vars;
  Map<Var, Expr> new_to_old_map;
  std::vector<Expr> solution;
  Array<Expr> conditions;

  // Simplify_cce right hand sides
  for (Expr &r : rhs) {
    r = SuperSimplify(r, domain->ranges);
  }

  // Create the conditions of the existence of a solution
  for (size_t j = 0; j < matrix.size(); ++j) {
    Expr new_cond;
    if ((j >= vars_size) || (matrix[j][j] == 0)) {
      // The row of matrix is zero. A solution exists only if the rhs[j] is also zero
      new_cond = (rhs[j] == 0);
    } else {
      // The diagonal element is non-zero. A solution exists only if the diagonal element
      // is a divisor of the rhs[j]
      new_cond = (truncmod(rhs[j], static_cast<int>(std::abs(matrix[j][j]))) == 0);
    }

    new_cond = SuperSimplify(new_cond, domain->ranges);
    if (is_const_int(new_cond, 0)) {
      return EmptyDomainTransformation(domain);
    } else if (!is_const_int(new_cond, 1)) {
      conditions.push_back(new_cond);
    }
  }

  // Now create new variables or directly solve the equations
  for (size_t j = 0; j < vars_size; ++j) {
    if ((j >= matrix.size()) || (matrix[j][j] == 0)) {
      // The j-th variable can take any integer value, create a tvm variable for it
      Expr to_old = SuperSimplify(new_to_old[j], domain->ranges);
      std::string name_hint = "n" + std::to_string(new_vars.size());
      if (const auto v_old = to_old.as<Variable>()) {
        name_hint += "_" + v_old->name_hint;
      }

      Var v = Var(name_hint, new_to_old[j].type());
      solution.push_back(v);
      new_vars.push_back(v);
      new_to_old_map.Set(v, to_old);
    } else {
      // The j-th variable is just a single value, don't create a tvm variable
      if (matrix[j][j] >= 0) {
        Expr a = make_const(rhs[j].type(), matrix[j][j]);
        solution.push_back(SuperSimplify(rhs[j] / a, domain->ranges));
      } else {
        // This is required because some simplifiers have problems with dividing by negative numbers
        Expr a = make_const(rhs[j].type(), -matrix[j][j]);
        solution.push_back(SuperSimplify((-rhs[j]) / a, domain->ranges));
      }
    }
  }

  // Convert the old_to_new matrix to map
  Map<Var, Expr> old_to_new_map;
  for (size_t i = 0; i < vars_size; ++i) {
    Expr e = make_zero(domain->variables[i].type());
    for (size_t j = 0; j < vars_size; ++j) {
      e = e + make_const(e.type(), old_to_new[i][j]) * solution[j];
    }
    e = SuperSimplify(e);
    old_to_new_map.Set(domain->variables[i], e);
  }

  // The resulting ranges
  Map<Var, Range> ranges;

  // First of all, fill the new ranges with outer variable ranges
  std::unordered_set<const Variable *> vset;
  for (const Var &v : domain->variables) {
    vset.insert(v.get());
  }

  for (const auto &p : domain->ranges) {
    if (!vset.count(p.first.get())) {
      ranges.Set(p.first, p.second);
    }
  }

  // Convert original ranges to IntSets
  std::unordered_map<const Variable *, IntSet> var_intsets;
  for (const auto &p : domain->ranges) {
    var_intsets[p.first.get()] = IntSet::range(p.second);
  }

  // Infer ranges for the new variables and add them to the resulting ranges
  for (const auto &p : new_to_old_map) {
    Range range = EvalSet(p.second, var_intsets).cover_range(Range());
    if (range.defined()) {
      ranges.Set(p.first, range);
    }
  }

  // We have to transform ranges of the old variables into conditions over new variables because new
  // ranges are not enough usually.
  for (const auto &p : domain->ranges) {
    if (old_to_new_map.count(p.first)) {
      Expr in_terms_of_new = old_to_new_map[p.first];
      Expr lower_cond = SuperSimplify(p.second->min <= in_terms_of_new, ranges);
      Expr upper_cond = SuperSimplify(in_terms_of_new < p.second->min + p.second->extent, ranges);

      if (!is_const_int(lower_cond, 1)) {
        conditions.push_back(lower_cond);
      }

      if (!is_const_int(upper_cond, 1)) {
        conditions.push_back(upper_cond);
      }
    }
  }

  // Add the rest conditions
  for (const Expr &cond : rest) {
    conditions.push_back(Substitute(cond, old_to_new_map));
  }

  Domain new_domain = DomainNode::make(new_vars, conditions, ranges);
  return DomainTransformationNode::make(new_domain, domain, new_to_old_map, old_to_new_map);
}

VarBounds VarBounds::substitute(const Map<Var, Expr> &subst) const {
  auto apply_fun = [&subst](const Expr &e) { return Substitute(e, subst); };

  return {Substitute(coef, subst), air::ir::UpdateArray(lower, apply_fun), air::ir::UpdateArray(equal, apply_fun),
          air::ir::UpdateArray(upper, apply_fun)};
}

Array<Expr> SolveSystemOfInequalitiesResult::as_conditions() const {
  Array<Expr> res;
  for (const Var &v : variables) {
    auto it = bounds.find(v.get());
    CHECK(it != bounds.end());
    const VarBounds &bnds = it->second;
    Expr lhs = bnds.coef * v;

    for (const Expr &rhs : bnds.equal) {
      res.push_back(EQ::make(lhs, rhs));
    }

    for (const Expr &rhs : bnds.lower) {
      res.push_back(GE::make(lhs, rhs));
    }

    for (const Expr &rhs : bnds.upper) {
      res.push_back(LE::make(lhs, rhs));
    }
  }

  for (const Expr &e : other_conditions) {
    res.push_back(e);
  }

  return res;
}

Expr SimplifyRemainder(const Expr &expr) {
  // Simplifying condition of the form: (((0 - var) % value) == 0)
  if (const EQ *op = expr.as<EQ>()) {
    if (const Mod *mod = op->a.as<Mod>()) {
      if (const Sub *subA = mod->a.as<Sub>()) {
        if (is_zero(subA->a)) {
          return EQ::make(Mod::make(subA->b, mod->b), op->b);
        }
      }
    }
  }

  return expr;
}

// Rewrite the system of inequalities using Fourier-Motzkin elimination
// Note that variable ranges help a lot, so this parameter is even non-optional
SolveSystemOfInequalitiesResult SolveSystemOfInequalities(const Array<Expr> &inequalities, const Array<Var> &variables,
                                                          const Map<Var, Range> &vranges) {
  SolveSystemOfInequalitiesResult res;
  res.variables = variables;

  // The algorithm consists in doing the following things for each variable v
  // - Take formulas from `current` and classify them according to polarity wrt v
  // - Combine each formula of positive polarity (wrt v) with each formula of negative polarity
  // - Put the resulting combinations into `new_current` along with unclassifiable formulas
  // - Replace `current` with `new_current` and move to the next variable

  // current and new_current are sorted to enable some heuristics
  std::set<Expr, ExprLess> current;
  std::set<Expr, ExprLess> new_current;
  // A vector of pairs (c, e), c > 0, representing formulas of the form c*v + e <= 0
  std::vector<std::pair<int64_t, Expr>> coef_pos;
  // A vector of pairs (c, e), c < 0, representing formulas of the form c*v + e <= 0
  std::vector<std::pair<int64_t, Expr>> coef_neg;

  // formulas we don't know what to do with
  std::vector<Expr> rest;

  // A helper that adds an inequality to new_current if it's not obviously redundant
  auto add_to_new_current = [&new_current, &vranges](const Expr &new_ineq) {
    if (CanProve(new_ineq, vranges)) {
      // redundant: follows from the vranges
      return;
    }

    if (const LE *new_le = new_ineq.as<LE>()) {
      // A heuristic: check if the new inequality is a consequence of one
      // of its future neighbors (in this case don't add it) or if a future neighbor is
      // a consequence of the new ineq (in which case remove the neighbor)
      auto it_neighbor = new_current.lower_bound(new_ineq);
      if (it_neighbor != new_current.begin()) {
        const LE *le = std::prev(it_neighbor)->as<LE>();
        if (le && CanProve(new_le->a - le->a <= 0, vranges)) {
          return;
        } else if (le && CanProve(le->a - new_le->a <= 0, vranges)) {
          new_current.erase(std::prev(it_neighbor));
        }
      }

      // Check the other neighbor
      if (it_neighbor != new_current.end()) {
        const LE *le = it_neighbor->as<LE>();
        if (le && CanProve(new_le->a - le->a <= 0, vranges)) {
          return;
        } else if (le && CanProve(le->a - new_le->a <= 0, vranges)) {
          it_neighbor = new_current.erase(it_neighbor);
        }
      }

      new_current.insert(it_neighbor, new_ineq);
    } else {
      new_current.insert(new_ineq);
    }
  };

  // Simplify_cce each inequality into the form `expr <= 0` and add to new_current formulas
  for (const Expr &ineq : inequalities) {
    add_to_new_current(NormalizeComparisons(SuperSimplify(ineq, vranges)));
  }

  std::swap(current, new_current);

  for (const Var &v : variables) {
    CHECK(!res.bounds.count(v.get())) << "Variable " << v
                                      << " appears several times in the `variables` which might be a bug";

    new_current.clear();
    coef_pos.clear();
    coef_neg.clear();

    // Add bounds from vranges
    if (vranges.count(v)) {
      const Range &range = vranges[v];
      Expr range_lbound = SuperSimplify(range->min, vranges);
      Expr range_ubound = SuperSimplify(range->min + range->extent - 1, vranges);
      coef_neg.emplace_back(std::pair<int64_t, Expr>{-1, range_lbound});
      coef_pos.emplace_back(std::pair<int64_t, Expr>{1, -range_ubound});
    }

    // Take formulas from `current` and classify them according to polarity wrt v
    for (const Expr &ineq : current) {
      if (const LE *le = ineq.as<LE>()) {
        Array<Expr> coef = air::arith::DetectLinearEquation(le->a, {v});
        if (!coef.empty() && is_const(coef[0])) {
          CHECK(as_const_int(coef[0]));
          int64_t coef0 = *as_const_int(coef[0]);
          if (coef0 == 0) {
            // zero polarity, straight to new_current
            add_to_new_current(ineq);
          } else if (coef0 > 0) {
            coef_pos.emplace_back(std::pair<int64_t, Expr>{coef0, coef[1]});
          } else {
            coef_neg.emplace_back(std::pair<int64_t, Expr>{coef0, coef[1]});
          }
          continue;
        }
      } else if (const EQ *eq = ineq.as<EQ>()) {
        Array<Expr> coef = air::arith::DetectLinearEquation(eq->a, {v});
        if (!coef.empty() && is_const(coef[0])) {
          CHECK(as_const_int(coef[0]));
          int64_t coef0 = *as_const_int(coef[0]);
          if (coef0 == 0) {
            // zero polarity, straight to new_current
            add_to_new_current(ineq);
          } else if (coef0 > 0) {
            // Equalities may be considered as pairs of two inequalities
            coef_pos.emplace_back(std::pair<int64_t, Expr>{coef0, coef[1]});
            coef_neg.emplace_back(std::pair<int64_t, Expr>{-coef0, -coef[1]});
          } else {
            coef_pos.emplace_back(std::pair<int64_t, Expr>{-coef0, -coef[1]});
            coef_neg.emplace_back(std::pair<int64_t, Expr>{coef0, coef[1]});
          }
          continue;
        }
      }

      // if nothing worked, put it in rest
      rest.push_back(ineq);
    }

    // Combine each positive inequality with each negative one (by adding them together)
    for (const auto &pos : coef_pos) {
      for (const auto &neg : coef_neg) {
        auto first_gcd = air::ir::gcd(static_cast<int>(pos.first), static_cast<int>(-neg.first));
        CHECK_NE(first_gcd, 0);
        Expr c_pos = make_const(v.type(), neg.first / first_gcd);
        Expr c_neg = make_const(v.type(), pos.first / first_gcd);
        Expr new_lhs = c_neg * neg.second - c_pos * pos.second;
        Expr new_ineq = LE::make(new_lhs, make_zero(pos.second.type()));
        new_ineq = NormalizeComparisons(SuperSimplify(new_ineq, vranges));
        add_to_new_current(new_ineq);
      }
    }

    // Now we have to generate resulting (in)equalities for the variable v

    // Find the common denominator in a sense
    // We will generate formulas of the form coef_lcm*v <= bound
    int64_t coef_lcm = 1;
    for (const auto &pos : coef_pos) {
      coef_lcm = air::ir::lcm(coef_lcm, pos.first);
    }
    for (const auto &neg : coef_neg) {
      coef_lcm = air::ir::lcm(coef_lcm, -neg.first);
    }

    // The resulting lower and upper bounds stored in sorted vectors
    std::vector<Expr> upper_bounds;
    std::vector<Expr> lower_bounds;
    upper_bounds.reserve(coef_pos.size());
    lower_bounds.reserve(coef_neg.size());

    for (const auto &pos : coef_pos) {
      CHECK_NE(pos.first, 0);
      Expr bound = make_const(v.type(), -coef_lcm / pos.first) * pos.second;
      bound = SuperSimplify(bound, vranges);
      // Don't add if any of the existing bounds is better
      if (std::any_of(upper_bounds.begin(), upper_bounds.end(),
                      [&bound, &vranges](const Expr &o) { return CanProve(o - bound <= 0, vranges); })) {
        continue;
      }

      // Erase all worse bounds
      upper_bounds.erase(
        std::remove_if(upper_bounds.begin(), upper_bounds.end(),
                       [&bound, &vranges](const Expr &o) { return CanProve(o - bound >= 0, vranges); }),
        upper_bounds.end());

      // Add
      upper_bounds.push_back(bound);
    }

    for (const auto &neg : coef_neg) {
      CHECK_NE(neg.first, 0);
      Expr bound = make_const(v.type(), -coef_lcm / neg.first) * neg.second;

      bound = SuperSimplify(bound, vranges);
      // Don't add if any of the existing bounds is better
      if (std::any_of(lower_bounds.begin(), lower_bounds.end(),
                      [&bound, &vranges](const Expr &o) { return CanProve(o - bound >= 0, vranges); })) {
        continue;
      }

      // Erase all worse bounds
      lower_bounds.erase(
        std::remove_if(lower_bounds.begin(), lower_bounds.end(),
                       [&bound, &vranges](const Expr &o) { return CanProve(o - bound <= 0, vranges); }),
        lower_bounds.end());

      // Add
      lower_bounds.push_back(bound);
    }

    // Sort the vectors and remove duplicates
    for (std::vector<Expr> *bounds : {&upper_bounds, &lower_bounds}) {
      std::sort(bounds->begin(), bounds->end(), ExprLess());
      bounds->erase(std::unique(bounds->begin(), bounds->end(), ExprEq()), bounds->end());
    }

    // Bounds which are both lower and upper should go to equal...
    std::vector<Expr> equal;
    equal.reserve(std::min(upper_bounds.size(), lower_bounds.size()));
    std::set_intersection(upper_bounds.begin(), upper_bounds.end(), lower_bounds.begin(), lower_bounds.end(),
                          std::back_inserter(equal), ExprLess());

    // ...and be removed from upper bounds...
    std::vector<Expr> new_upper;
    new_upper.reserve(upper_bounds.size() - equal.size());
    std::set_difference(upper_bounds.begin(), upper_bounds.end(), equal.begin(), equal.end(),
                        std::back_inserter(new_upper), ExprLess());

    // ...and from lower bounds.
    std::vector<Expr> new_lower;
    new_lower.reserve(lower_bounds.size() - equal.size());
    std::set_difference(lower_bounds.begin(), lower_bounds.end(), equal.begin(), equal.end(),
                        std::back_inserter(new_lower), ExprLess());

    // Write it to the result.
    auto &bnds = res.bounds[v.get()];
    bnds.coef = make_const(v.type(), coef_lcm);
    bnds.equal = equal;
    bnds.lower = new_lower;
    bnds.upper = new_upper;

    std::swap(current, new_current);
  }

  // Everything that is left goes to res.other_conditions
  for (const Expr &e : current) {
    Expr e_simp = SuperSimplify(e, vranges);
    if (is_const_int(e_simp, 0)) {
      // contradiction detected
      res.other_conditions = {const_false()};
      return res;
    } else if (is_const_int(e_simp, 1)) {
      continue;
    } else {
      res.other_conditions.push_back(e_simp);
    }
  }

  for (const Expr &e : rest) res.other_conditions.push_back(e);

  return res;
}

// Deskew the given domain
DomainTransformation DeskewDomain(const Domain &domain) {
  // Resulting ranges will contain ranges for the new variables and for the variables that are
  // not in the domain->variables but are in domain->ranges
  Map<Var, Range> res_ranges;

  // vars are variables from domain's variables followed by all the other variables from its ranges
  Array<Var> vars = domain->variables;
  for (const auto &pair : domain->ranges) {
    bool already = false;
    for (const Var &v : vars) {
      already = already || v.same_as(pair.first);
    }
    if (!already) {
      vars.push_back(pair.first);
      // Also populate the resulting ranges with ranges of outer variables
      res_ranges.Set(pair.first, pair.second);
    }
  }

  auto solved_system = SolveSystemOfInequalities(domain->conditions, vars, domain->ranges);

  Map<Var, Expr> res_old_to_new;
  Map<Var, Expr> res_new_to_old;
  Array<Var> res_variables;
  Array<Expr> res_conditions;
  std::unordered_map<const Variable *, IntSet> new_var_intsets;

  Map<Var, Range> vranges = domain->ranges;

  // Initialize new_var_intsets with the old var intsets
  for (const auto &pair : domain->ranges) {
    new_var_intsets[pair.first.get()] = IntSet::range(pair.second);
  }

  // We process variables in the reverse direction to start with the most independent one.
  // This order is needed to compute new ranges.
  for (auto it = domain->variables.rbegin(); it != domain->variables.rend(); ++it) {
    const Var &var = *it;
    auto &bnd = solved_system.bounds[var.get()];
    // Note that we replace old vars with new ones
    bnd = bnd.substitute(res_old_to_new);
    if (is_one(bnd.coef) && !bnd.equal.empty()) {
      // There is an equation of the form `v == expr`, so this variable can be completely removed.
      // Note that we use the 0-th expression because they are ordered by complexity, so it must be
      // the simplest one.
      res_old_to_new.Set(var, bnd.equal[0]);
    } else {
      Array<Expr> lowers = Concat(bnd.equal, bnd.lower);
      Array<Expr> uppers = Concat(bnd.equal, bnd.upper);

      // Here we will try all pairs of lower and upper bounds and find the best pair, that is, the
      // pair with the minimal difference between the upper and the lower.
      // Note that the bounds are for v*coef, not for v (because we don't want complex expressions
      // involving division).

      // The lower bound of the best pair so far
      Expr best_lower = vranges[var]->min * bnd.coef;
      // The difference between the upper and the lower of the best pair so far
      Expr best_diff = (vranges[var]->extent - 1) * bnd.coef;
      // The overapproximation of the best difference
      Expr best_diff_over = best_diff;

      for (const Expr &low : lowers) {
        for (const Expr &upp : uppers) {
          Expr diff = SuperSimplify(upp - low, vranges);
          // Since diff may depend on some other variables, we compute its overapproximation
          Expr diff_over = EvalSet(diff, new_var_intsets).max();
          if (air::arith::is_pos_inf(diff_over)) {
            continue;
          }

          // If it is provable that the new one is strictly better than the current best one,
          // then replace it. Note that we are biased towards earlier pairs which should be simpler.
          if (CanProve(diff_over - best_diff_over < 0, vranges)) {
            best_lower = low;
            best_diff = diff;
            best_diff_over = diff_over;
          }
        }
      }

      if (is_const_int(best_diff, 0)) {
        // In this case coef*iv = best_lower
        // Don't create an itervar, just replace it everywhere with its min
        res_old_to_new.Set(var, SuperSimplify(best_lower / bnd.coef, vranges));
        // To assure correctness, we have to add a condition that best_lower can be divided by coef
        res_conditions.push_back(SuperSimplify(best_lower % bnd.coef == 0, vranges));
      } else {
        std::string suffix = Equal(best_lower, vranges[var]->min * bnd.coef) ? "" : "_shifted";
        Var new_var = var.copy_with_suffix(suffix);

        // We will replace our iv with new_var + shift.
        // We use rounding-up division to compute shift. Since we want to use a single formula
        // without selects in as many cases as possible, we try to prove conditions manually.
        Expr shift;
        if (CanProve(best_lower <= 0, vranges)) {
          shift = best_lower / bnd.coef;
        } else if (CanProve(best_lower > -bnd.coef, vranges)) {
          shift = (best_lower + bnd.coef - 1) / bnd.coef;
        } else {
          shift = Select::make(best_lower <= -bnd.coef, best_lower / bnd.coef, (best_lower + bnd.coef - 1) / bnd.coef);
        }
        shift = SuperSimplify(shift, vranges);

        Expr diff = SuperSimplify(best_diff_over / bnd.coef, vranges);
        if (is_const_int(diff, 0)) {
          // Don't create an itervar, just replace it everywhere with its min
          res_old_to_new.Set(var, shift);
        } else {
          res_old_to_new.Set(var, new_var + shift);
          // Note that we are substituting old with new, so best_lower contains new var,
          // that is we have to substitute new with old in best_lower here
          res_new_to_old.Set(new_var, SuperSimplify(var - Substitute(shift, res_new_to_old), vranges));

          new_var_intsets[new_var.get()] = IntSet::interval(make_zero(new_var.type()), diff);

          // Add the new var to the resulting axis
          auto range = Range(make_zero(new_var.type()), SuperSimplify(diff + 1, vranges));
          res_variables.push_back(new_var);
          res_ranges.Set(new_var, range);
          vranges.Set(new_var, range);
        }
      }
    }
  }

  for (const Expr &old_cond : solved_system.as_conditions()) {
    auto simpleOldCond = SuperSimplify(Substitute(old_cond, res_old_to_new), vranges);
    simpleOldCond = SimplifyRemainder(simpleOldCond);
    bool exists = false;
    for (const Expr &check_cond : res_conditions) {
      if (CanProve(check_cond == simpleOldCond)) {
        exists = true;
        break;
      }
    }
    if (!exists) {
      res_conditions.push_back(simpleOldCond);
    }
  }

  // Reverse the axis so that it matches the order of the original variables
  res_variables = Array<Var>(res_variables.rbegin(), res_variables.rend());

  Domain new_domain = DomainNode::make(res_variables, res_conditions, res_ranges);
  return DomainTransformationNode::make(new_domain, domain, res_new_to_old, res_old_to_new);
}

// Simplify_cce an iteration domain.
DomainTransformation SimplifyDomain(const Domain &domain, bool eliminate_div_mod, bool keep_dims) {
  DomainTransformation transf = IdDomainTransformation(domain);

  if (eliminate_div_mod) {
    transf += EliminateDivModFromDomainConditions(transf->new_domain);
  }

  // Repeating the following steps has a positive effect.
  for (size_t i = 0; i < N_REPEAT_TRANSFORM; ++i) {
    DomainTransformation tr;
    tr = SolveSystemOfEquations(transf->new_domain);

    // SolveSystemOfEquations might lead to unexpected iterator variables
    // such as the case of roi_align_ad. To pass this case while making most of the optimization
    // in zero elimination, we check whether all the var in new domain are included in the range.
    // If not, we will not update the domain transformation.
    DomainTransformation old_tf = transf;

    transf += tr;
    Map<Var, Range> vranges = transf->new_domain->ranges;
    bool unexpect_var = false;

    for (auto it = transf->new_domain->variables.begin(); it != transf->new_domain->variables.end(); ++it) {
      const Var &var = *it;
      if (vranges.find(var) == vranges.end()) {
        unexpect_var = true;
        break;
      }
    }

    if (!unexpect_var) {
      tr = DeskewDomain(transf->new_domain);
      transf += tr;
    } else {
      transf = old_tf;
    }
  }
  return transf;
}

// Use the condition of a reduction op to Simplify_cce its domain (axis)
Expr SimplifyReductionDomain(const Expr &expr, const Map<Var, Range> &outer_vranges) {
  if (const auto red = expr.as<Reduce>()) {
    Domain domain = DomainNode::make(IterVarsToVars(red->axis), FactorOutAtomicFormulas(red->condition).to_array(),
                                     Merge(outer_vranges, IterVarsToMap(red->axis)));
    auto res = SimplifyDomain(domain);

    Array<Expr> new_source;
    for (const Expr &src : red->source) {
      new_source.push_back(Substitute(src, res->old_to_new));
    }

    Array<IterVar> new_axis =
      IterVarsFromMap(res->new_domain->variables, res->new_domain->ranges, IterVarType::kCommReduce);

    // Perform simplification mainly to remove a possibly empty reduction.
    return Simplify_cce(
      Reduce::make(red->combiner, new_source, new_axis, All(res->new_domain->conditions), red->value_index));
  } else {
    return expr;
  }
}

// Restore the reduced dims from the expr
class RestoreDimsTensor : public IRMutator {
 public:
  RestoreDimsTensor(const Map<Var, Range> vranges, const Array<Var> &used_res_variables,
                    const Map<Var, Expr> new_to_old)
      : new_vranges(vranges), used_res_variables_(used_res_variables), new_to_old_(new_to_old) {}
  ~RestoreDimsTensor() override = default;

  Expr Mutate_(const Call *op, const Expr &e) override {
    Expr expr = IRMutator::Mutate_(op, e);
    const Call *n = expr.as<Call>();
    CHECK(n);
    if (n->call_type == Call::Halide) {
      size_t counter_int = 0, counter_var = 0;
      bool previous_exist = (new_var_exprs.size() > 0);
      bool supported = true;
      for (size_t i = 0; i < n->args.size(); i++) {
        if ((n->args[i]->GetTypeKey() == "IntImm") && (n->args[i].as<IntImm>()->value == 0)) {
          std::string new_name = "kd_" + std::to_string(counter_int++);
          if (previous_exist) {
            if (new_vars[i].get()->name_hint != new_name) {
              supported = false;
              break;
            }
          } else {
            auto new_v = Var(new_name);
            new_var_exprs.push_back(new_v);
            new_vars.push_back(new_v);
            new_vranges.Set(new_v, Range::make_by_min_extent(0, 1));
            new_call_exprs.push_back(Expr(0));
          }
        } else {
          if (previous_exist) {
            if (!new_var_exprs[i].same_as(used_res_variables_[counter_var])) {
              supported = false;
              break;
            }
            counter_var++;
          } else {
            new_var_exprs.push_back(used_res_variables_[counter_var]);
            new_vars.push_back(used_res_variables_[counter_var]);
            new_call_exprs.push_back(new_to_old_[used_res_variables_[counter_var]]);
            counter_var++;
          }
        }
      }
      if (!supported) {
        return e;
      }
      auto new_expr = Call::make(n->type, n->name, new_var_exprs, n->call_type, n->func, n->value_index);
      return new_expr;
    }
    return expr;
  }

 public:
  Array<Expr> new_var_exprs;
  Array<Expr> new_call_exprs;
  Array<Var> new_vars;
  Map<Var, Range> new_vranges;

 private:
  Array<Var> used_res_variables_;
  Map<Var, Expr> new_to_old_;
};

void ExtractUsedVariables(const Expr &e, const Expr &cond, const Array<Var> &outer_axis, const Map<Var, Range> &vranges,
                          DomainTransformation &simplified_domain, Expr &new_expr, Array<Var> &used_res_variables) {
  Domain domain = DomainNode::make(outer_axis, FactorOutAtomicFormulas(cond).to_array(), vranges);

  simplified_domain = SimplifyDomain(domain);
  new_expr = SuperSimplify(Substitute(e, simplified_domain->old_to_new), simplified_domain->new_domain->ranges);

  // This is mostly done to Simplify_cce if_then_else which is not known by the Halide simplifier
  new_expr = RemoveRedundantInequalities(new_expr, simplified_domain->new_domain->conditions);

  // Keep only those variables of the new vars which are used in the new_expr
  for (const Var &var : simplified_domain->new_domain->variables) {
    if (ExprUseVar(new_expr, var)) {
      used_res_variables.push_back(var);
    }
  }
}

bool CheckIfVolumeIncreased(const Array<Var> &outer_axis, const Map<Var, Range> &vranges,
                            const Array<Var> &used_res_variables, const DomainTransformation &simplified_domain) {
  // Compute volumes before and after
  Expr old_volume = make_const(Int(64), 1);
  for (const Var &var : outer_axis) {
    old_volume = old_volume * vranges[var]->extent;
  }

  Expr new_volume = make_const(Int(64), 1);
  for (const Var &var : used_res_variables) {
    new_volume = new_volume * simplified_domain->new_domain->ranges[var]->extent;
  }

  // if we can prove that the old volume is not greater than the new volume then
  // prefer the old expression.
  if (CanProve(old_volume <= new_volume, vranges)) {
    return true;
  } else {
    return false;
  }
}

void CheckReduceExpr(const DomainTransformation &simplified_domain, Expr &new_expr) {
  if (auto isRed = new_expr.as<Reduce>()) {
    Array<Expr> newSource;
    bool changed = false;
    for (size_t i = 0; i < isRed->source.size(); i++) {
      if (auto isSelect = isRed->source[i].as<Select>()) {
        auto simplerCond =
          Simplify_cce(isSelect->condition, Merge(simplified_domain->new_domain->ranges, IterVarsToMap(isRed->axis)));
        if (!simplerCond.same_as(isSelect->condition)) {
          changed = true;
        }
        newSource.push_back(Select::make(simplerCond, isSelect->true_value, isSelect->false_value));
      } else {
        newSource.push_back(isRed->source[i]);
      }
    }
    if (changed) {
      new_expr = Reduce::make(isRed->combiner, newSource, isRed->axis, isRed->condition, isRed->value_index);
    }
  }

  return;
}

// Extract the given expr under the given condition as a separate tensor if the volume of the
// extracted tensor will be less than the volume of the outer_axis
Expr ExtractAsTensorMaybe(const Expr &e, const Expr &cond, const Array<Var> &outer_axis, const Map<Var, Range> &vranges,
                          bool keep_dims) {
  DomainTransformation res;
  Expr new_expr;
  Array<Var> used_res_variables;
  ExtractUsedVariables(e, cond, outer_axis, vranges, res, new_expr, used_res_variables);

  // If the expression does not use vars then it is probably better to keep it inlined
  if (used_res_variables.empty()) {
    // We can return the new_expr here instead of the old e because it doesn't use variables
    // otherwise we would need to replace the new vars or create a let-expression
    return new_expr;
  }

  // If it's already a call to a tensor then extracting it will probably be useless
  if (const Call *call = new_expr.as<Call>()) {
    if (call->call_type == Call::CallType::Halide) {
      return e;
    }
  }

  if (CheckIfVolumeIncreased(outer_axis, vranges, used_res_variables, res)) {
    return e;
  }

  CheckReduceExpr(res, new_expr);

  static int new_tensor_counter = 0;
  std::string new_tensor_name("extracted_tensor_" + std::to_string(new_tensor_counter));
  new_tensor_counter++;

  if (keep_dims) {
    RestoreDimsTensor restore_dims(res->new_domain->ranges, used_res_variables, res->new_to_old);
    Expr new_expr_keep_dims = restore_dims.Mutate(new_expr);
    if (!new_expr_keep_dims.same_as(new_expr)) {
      Tensor new_tensor = TensorFromExpr(
        new_expr_keep_dims, IterVarsFromMap(restore_dims.new_vars, restore_dims.new_vranges), new_tensor_name);
      return Call::make(e.type(), new_tensor->op->name, restore_dims.new_call_exprs, Call::CallType::Halide,
                        new_tensor->op, new_tensor->value_index);
    }
  }

  Tensor tensor =
    TensorFromExpr(new_expr, IterVarsFromMap(used_res_variables, res->new_domain->ranges), new_tensor_name);
  Array<Expr> args;
  for (const Var &var : used_res_variables) {
    args.push_back(res->new_to_old[var]);
  }

  return Call::make(e.type(), tensor->op->name, args, Call::CallType::Halide, tensor->op, tensor->value_index);
}

// Extract from cond an implication of cond not containing vars
std::pair<Expr, Expr> ImplicationNotContainingVars(const Expr &cond, const std::unordered_set<const Variable *> &vars) {
  CHECK(cond.type().is_bool()) << "The type of cond must be bool";
  if (const And *and_op = cond.as<And>()) {
    auto pair_a = ImplicationNotContainingVars(and_op->a, vars);
    auto pair_b = ImplicationNotContainingVars(and_op->b, vars);

    return {pair_a.first && pair_b.first, pair_a.second && pair_b.second};
  } else if (const Or *or_op = cond.as<Or>()) {
    auto pair_a = ImplicationNotContainingVars(or_op->a, vars);
    auto pair_b = ImplicationNotContainingVars(or_op->b, vars);

    return {(pair_a.first || pair_b.first),
            (pair_a.first || pair_b.second) && (pair_b.first || pair_a.second) && (pair_a.second || pair_b.second)};
  } else if (!ExprUseVar(cond, vars)) {
    return {cond, const_true()};
  } else {
    return {const_true(), cond};
  }
}

// Factor conditions out of a reduction by applying Fourier-Motzkin elimination and moving out
// (in)equalities which do not depend on the reduction variables.
std::pair<Expr, Expr> LiftConditionsThroughReduction(const Expr &cond, const Array<IterVar> &red_axis,
                                                     const Array<IterVar> &outer_axis) {
  // Factor out atomics so that we can consider this as a system of inequalities
  auto factoratomic_res = FactorOutAtomicFormulas(cond);
  Array<Expr> atomics = factoratomic_res.atomic_formulas;
  const Expr &rest = factoratomic_res.rest;

  Array<Var> allvars;
  for (const IterVar &v : red_axis) {
    allvars.push_back(v->var);
  }

  for (const IterVar &v : outer_axis) {
    allvars.push_back(v->var);
  }

  auto vranges = Merge(IterVarsToMap(red_axis), IterVarsToMap(outer_axis));
  // start from reduction vars, so that input vars don't depend on them
  atomics = SolveSystemOfInequalities(atomics, allvars, vranges).as_conditions();

  // Append the rest part
  Expr rewritten_cond = (All(atomics) && rest);

  std::unordered_set<const Variable *> vset;
  for (const IterVar &v : red_axis) {
    vset.insert(v->var.get());
  }

  // The outer (first) condition does not contain reduction vars,
  // the inner (second) condition is everything else
  return ImplicationNotContainingVars(rewritten_cond, vset);
}

class ExtractReductionsMutator : public IRMutator {
 public:
  ExtractReductionsMutator(const Array<Var> &outer_axis, Map<Var, Range> vranges,
                           std::string name = "extracted_reduction")
      : outer_axis_(outer_axis), vranges_(std::move(vranges)), name_(std::move(name)) {}
  ~ExtractReductionsMutator() override = default;

  Expr Mutate_(const Reduce *op, const Expr &e) override {
    ExtractReductionsMutator new_mutator(Concat(IterVarsToVars(op->axis), outer_axis_),
                                         Merge(vranges_, IterVarsToMap(op->axis)), name_);

    Array<Expr> new_source;
    for (const Expr &src : op->source) {
      new_source.push_back(new_mutator.Mutate(src));
    }

    auto simplerCond = Simplify_cce(op->condition, Merge(vranges_, IterVarsToMap(op->axis)));
    Expr new_reduce = Reduce::make(op->combiner, new_source, op->axis, simplerCond, op->value_index);

    ExprFreeVarsVisitor fv_visitor;
    fv_visitor.Visit(new_reduce);

    // Vars of the tensor we are going to create for this reduction
    Array<Var> vars;
    for (const Var &v : outer_axis_) {
      // We take variables from the outer_axis_ which are also present in the new reduction
      if (fv_visitor.free.count(v.get())) {
        vars.push_back(v);
      }
    }

    auto newaxis_vmap_pair = CloneIterVars(IterVarsFromMap(vars, vranges_));
    Array<IterVar> new_axis = newaxis_vmap_pair.first;
    new_reduce = SuperSimplify(Substitute(new_reduce, newaxis_vmap_pair.second), IterVarsToMap(new_axis));

    Tensor tensor = TensorFromExpr(new_reduce, new_axis, name_, tag_, attrs_);

    Array<Expr> args;
    for (const Var &v : vars) {
      args.push_back(v);
    }

    return Call::make(e.type(), tensor->op->name, args, Call::CallType::Halide, tensor->op, tensor->value_index);
  }

 private:
  Array<Var> outer_axis_;
  Map<Var, Range> vranges_;
  std::string name_;
  std::string tag_;
  Map<std::string, NodeRef> attrs_;
};

// Extract reductions as separate tensors.
Expr ExtractReductions(const Expr &expr, const Array<Var> &outer_axis, const Map<Var, Range> &vranges) {
  return ExtractReductionsMutator(outer_axis, vranges).Mutate(expr);
}

Expr ExtractNonTopReductions(const Expr &expr, const Array<Var> &outer_axis, const Map<Var, Range> &vranges) {
  if (const auto red = expr.as<Reduce>()) {
    Array<Var> new_outer_axis = Concat(IterVarsToVars(red->axis), outer_axis);
    Map<Var, Range> new_vranges = Merge(vranges, IterVarsToMap(red->axis));
    Array<Expr> new_source;
    for (const Expr &src : red->source) {
      new_source.push_back(ExtractReductions(src, new_outer_axis, new_vranges));
    }
    Expr new_condition = ExtractReductions(red->condition, new_outer_axis, new_vranges);

    return Reduce::make(red->combiner, new_source, red->axis, new_condition, red->value_index);
  } else {
    return ExtractReductions(expr, outer_axis, vranges);
  }
}

Expr OptimizeAndLiftNonzeronessConditionsImpl(const Expr &expr, const Array<IterVar> &axis,
                                              const Map<Var, Range> &vranges, bool keep_dims = false) {
  Expr result;
  Map<Var, Range> combined_vranges = Merge(vranges, IterVarsToMap(axis));
  if (const auto *red = expr.as<Reduce>()) {
    bool is_sum = IsSumCombiner(red->combiner, vranges);
    if (is_sum || CanFactorZeroFromCombiner(red->combiner, red->value_index, vranges)) {
      Expr new_red = expr;

      // Here we Simplify_cce the reduction
      {
        Expr cond = red->condition;
        Array<Expr> source = red->source;

        // If it is a summation then we can lift nonzeroness conditions from the source
        // and add them to the reduction conditions
        if (is_sum) {
          auto nz = NonzeronessCondition(red->source[red->value_index]);
          cond = nz.cond && cond;
          source.Set(0, nz.value);
        }

        new_red = Reduce::make(red->combiner, source, red->axis, cond, red->value_index);
        new_red = SimplifyReductionDomain(new_red, combined_vranges);
        red = new_red.as<Reduce>();
        // If the reduction disappears completely then transform the result as a non-reduction
        if (!red) {
          // For reduction, the keep_dims should be False
          return OptimizeAndLiftNonzeronessConditionsImpl(new_red, axis, vranges);
        }
      }

      Expr new_outer_cond, new_reduce_cond;
      Array<Expr> new_source = red->source;

      // Partially lift conditions from the reduce condition
      std::tie(new_outer_cond, new_reduce_cond) = LiftConditionsThroughReduction(red->condition, red->axis, axis);

      // If it's not sum then we haven't yet lifted nonzeroness cond from the source
      if (!is_sum) {
        Expr outer_nz_cond, nz_cond, nz_source;
        auto nz = NonzeronessCondition(red->source[red->value_index]);
        // Append conditions from the reduction
        nz_cond = new_reduce_cond && nz.cond;
        nz_source = nz.value;
        std::tie(outer_nz_cond, nz_cond) = LiftConditionsThroughReduction(nz_cond, red->axis, axis);
        new_outer_cond = new_outer_cond && outer_nz_cond;
        new_source.Set(static_cast<size_t>(static_cast<uint>(red->value_index)), SelectElseZero(nz_cond, nz_source));
      }

      Expr new_reduce = Reduce::make(red->combiner, new_source, red->axis, new_reduce_cond, red->value_index);
      new_reduce = ExtractAsTensorMaybe(new_reduce, new_outer_cond, IterVarsToVars(axis), combined_vranges, keep_dims);
      result = SelectElseZero(new_outer_cond, new_reduce);
    } else {
      return SimplifyReductionDomain(expr, combined_vranges);
    }
  } else {
    auto nz = NonzeronessCondition(expr);
    Expr new_expr = ExtractAsTensorMaybe(nz.value, nz.cond, IterVarsToVars(axis), combined_vranges, keep_dims);
    result = SelectElseZero(nz.cond, new_expr);
  }

  // Note that RemoveRedundantInequalities can sometimes propagate equalities which
  // other simplifiers cannot, like (i % 3) == 0.
  Array<Expr> axis_conds = IterVarsToInequalities(axis);
  result = RemoveRedundantInequalities(result, axis_conds);

  // Sometimes ExtractAsTensorMaybe doesn't perform extraction, so there may be some non-top
  // reductions left, take care of them

  auto candidate = SuperSimplify(ExtractReductions(result, IterVarsToVars(axis), combined_vranges), combined_vranges);
  if (auto select = candidate.as<Select>()) {
    auto simplerCond = Simplify_cce(select->condition, IterVarsToMap(axis));
    if (!simplerCond.same_as(select->condition)) {
      candidate = Select::make(simplerCond, select->true_value, select->false_value);
    }
  }

  return candidate;
}

Tensor OptimizeAndLiftNonzeronessConditions(const Tensor &tensor, bool keep_dims, const Map<Var, Range> &vranges) {
  auto transform_func = [&vranges, &keep_dims](const Expr &expr, const Array<IterVar> &axis) {
    return OptimizeAndLiftNonzeronessConditionsImpl(expr, axis, vranges, keep_dims);
  };

  return TransformBody(tensor, transform_func);
}

Domain DomainNode::make(Array<Var> variables, Array<Expr> conditions, Map<Var, Range> ranges) {
  auto n = make_node<DomainNode>();
  n->variables = std::move(variables);
  n->conditions = std::move(conditions);
  n->ranges = std::move(ranges);

  return Domain(n);
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable).set_dispatch<DomainNode>([](const ObjectRef &node, IRPrinter *p) {
  auto *d = static_cast<const DomainNode *>(node.get());
  p->stream << "Domain(variables=" << d->variables << ", conditions=" << d->conditions << ", ranges=" << d->ranges
            << ')';
});

TVM_REGISTER_NODE_TYPE(DomainNode);

DomainTransformation DomainTransformationNode::make(Domain new_domain, Domain old_domain, Map<Var, Expr> new_to_old,
                                                    Map<Var, Expr> old_to_new) {
  auto n = make_node<DomainTransformationNode>();
  n->new_domain = std::move(new_domain);
  n->old_domain = std::move(old_domain);
  n->new_to_old = std::move(new_to_old);
  n->old_to_new = std::move(old_to_new);

  return DomainTransformation(n);
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
  .set_dispatch<DomainTransformationNode>([](const ObjectRef &node, IRPrinter *p) {
    auto *d = static_cast<const DomainTransformationNode *>(node.get());
    p->stream << "DomainTransformation(new_domain=" << d->new_domain << ", old_domain=" << d->old_domain
              << ", new_to_old=" << d->new_to_old << ", old_to_new=" << d->old_to_new << ')';
  });

TVM_REGISTER_NODE_TYPE(DomainTransformationNode);

TVM_REGISTER_API("arith._make_Domain").set_body([](const TVMArgs args, TVMRetValue *ret) {
  if (args[1].IsObjectRef<Expr>()) {
    *ret = DomainNode::make(args[0], FactorOutAtomicFormulas(args[1]).to_array(), args[2]);
  } else {
    *ret = DomainNode::make(args[0], args[1], args[2]);
  }
});

TVM_REGISTER_API("ir_pass.ComposeDomainTransformations").set_body([](const TVMArgs args, TVMRetValue *ret) {
  *ret = ComposeDomainTransformations(args[0], args[1]);
});

TVM_REGISTER_API("ir_pass.EmptyDomainTransformation").set_body([](const TVMArgs args, TVMRetValue *ret) {
  *ret = EmptyDomainTransformation(args[0]);
});

TVM_REGISTER_API("ir_pass.IdDomainTransformation").set_body([](const TVMArgs args, TVMRetValue *ret) {
  *ret = IdDomainTransformation(args[0]);
});

TVM_REGISTER_API("ir_pass.SolveSystemOfEquations").set_body([](const TVMArgs args, TVMRetValue *ret) {
  *ret = SolveSystemOfEquations(args[0]);
});

TVM_REGISTER_API("ir_pass.IsSumCombiner").set_body([](const TVMArgs args, TVMRetValue *ret) {
  if (args.size() >= 2) {
    *ret = IsSumCombiner(args[0], args[1]);
  } else {
    *ret = IsSumCombiner(args[0]);
  }
});

TVM_REGISTER_API("ir_pass.CanFactorZeroFromCombiner").set_body([](const TVMArgs args, TVMRetValue *ret) {
  if (args.size() >= 3) {
    *ret = CanFactorZeroFromCombiner(args[0], args[1], args[2]);
  } else {
    *ret = CanFactorZeroFromCombiner(args[0], args[1]);
  }
});

TVM_REGISTER_API("ir_pass.LiftNonzeronessCondition").set_body([](const TVMArgs args, TVMRetValue *ret) {
  *ret = LiftNonzeronessCondition(args[0]);
});

TVM_REGISTER_API("ir_pass.InlineTailCall").set_body([](const TVMArgs args, TVMRetValue *ret) {
  *ret = InlineTailCall(args[0]);
});

TVM_REGISTER_API("ir_pass.InlineTensors").set_body([](const TVMArgs args, TVMRetValue *ret) {
  if (args[0].IsObjectRef<Expr>()) {
    Expr e = args[0];
    if (args.size() == 1) {
      *ret = InlineTensors(e);
    } else if (args.size() == 2) {
      *ret = InlineTensors(e, args[1]);
    } else if (args.size() >= 3) {
      *ret = InlineTensors(e, args[1], args[2]);
    }
  } else if (args[0].IsObjectRef<Tensor>()) {
    Tensor t = args[0];
    if (args.size() == 1) {
      *ret = InlineTensors(t);
    } else if (args.size() == 2) {
      *ret = InlineTensors(t, args[1]);
    } else if (args.size() >= 3) {
      *ret = InlineTensors(t, args[1], args[2]);
    }
  }
});

TVM_REGISTER_API("ir_pass.SolveSystemOfInequalities").set_body([](const TVMArgs args, TVMRetValue *ret) {
  *ret = SolveSystemOfInequalities(args[0], args[1], args[2]).as_conditions();
});

TVM_REGISTER_API("ir_pass.SimplifyDomain").set_body([](const TVMArgs args, TVMRetValue *ret) {
  if (args.size() == 1) {
    *ret = SimplifyDomain(args[0]);
  } else {
    *ret = SimplifyDomain(args[0], args[1]);
  }
});

TVM_REGISTER_API("ir_pass.SimplifyReductionDomain").set_body([](const TVMArgs args, TVMRetValue *ret) {
  *ret = SimplifyReductionDomain(args[0], args[1]);
});

TVM_REGISTER_API("ir_pass.ExtractAsTensorMaybe").set_body([](const TVMArgs args, TVMRetValue *ret) {
  CHECK(args.size() >= 4) << "Not enough args.";
  if (args.size() == 4) {
    *ret = ExtractAsTensorMaybe(args[0], args[1], args[2], args[3]);
  } else {
    *ret = ExtractAsTensorMaybe(args[0], args[1], args[2], args[3], args[4]);
  }
});

TVM_REGISTER_API("ir_pass.ExtractReductions").set_body([](const TVMArgs args, TVMRetValue *ret) {
  *ret = ExtractReductions(args[0], args[1], args[2]);
});

TVM_REGISTER_API("ir_pass.ExtractNonTopReductions").set_body([](const TVMArgs args, TVMRetValue *ret) {
  *ret = ExtractNonTopReductions(args[0], args[1], args[2]);
});

TVM_REGISTER_API("ir_pass.OptimizeAndLiftNonzeronessConditions").set_body([](const TVMArgs args, TVMRetValue *ret) {
  CHECK(args.size()) << "No given args.";
  if (args.size() >= 3) {
    *ret = OptimizeAndLiftNonzeronessConditions(args[0], args[1], args[2]);
  } else {
    if (args.size() >= 2) {
      *ret = OptimizeAndLiftNonzeronessConditions(args[0], args[1]);
    } else {
      *ret = OptimizeAndLiftNonzeronessConditions(args[0]);
    }
  }
});

Tensor TensorFromExpr(const Expr &expr, const Array<IterVar> &axis, const std::string &name, const std::string &tag,
                      const Map<std::string, NodeRef> &attrs) {
  Array<Expr> new_bodies;
  int new_value_index = 0;

  // If this is a reduction then we have to clone its body
  if (const auto red = expr.as<Reduce>()) {
    new_value_index = red->value_index;

    for (size_t i = 0; i < red->source.size(); ++i) {
      Expr ith_red = Reduce::make(red->combiner, red->source, red->axis, red->condition, static_cast<int>(i));
      new_bodies.push_back(ith_red);
    }
  } else {
    new_value_index = 0;
    new_bodies.push_back(expr);
  }

  return ComputeOpNode::make(name, tag, attrs, axis, new_bodies)
    .output(static_cast<size_t>(static_cast<uint>(new_value_index)));
}

Tensor TransformBody(const Tensor &tensor, const std::function<Expr(const Expr &, const Array<IterVar> &)> &func) {
  if (const auto op = tensor->op.as<ComputeOpNode>()) {
    // Transform only one body
    Expr new_body = func(op->body[tensor->value_index], op->axis);
    // If the body didn't change then we can return the same tensor
    if (new_body.same_as(op->body[tensor->value_index])) return tensor;

    return TensorFromExpr(new_body, op->axis, op->name, op->tag, op->attrs);
  } else {
    return tensor;
  }
}

Tensor TransformBody(const Tensor &tensor, const std::function<Expr(const Expr &)> &func) {
  return TransformBody(tensor, [func](const Expr &e, const Array<IterVar> &) { return func(e); });
}
}  // namespace ir
}  // namespace akg
