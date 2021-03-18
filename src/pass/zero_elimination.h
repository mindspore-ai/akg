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
#ifndef PASS_ZERO_ELIMINATION_H_
#define PASS_ZERO_ELIMINATION_H_

#include <tvm/ir.h>
#include <tvm/tensor.h>
#include <tvm.h>
#include <string>
#include <unordered_map>

namespace akg {
namespace ir {
class Domain;

/*! \brief Node representing a domain of iteration.
 *
 *   A domain of iteration represents a set of values of `variables` within `ranges` such that all
 *   of the `conditions` are true. Conditions are represented as an array of conditions for
 *   convenience, many domain transformations require them to be atomic formulas. Ranges
 *   overapproximate the domain like a bounding box, many transformations are much more efficient
 *   when we have this information.
 */
class DomainNode : public Node {
 public:
  /*! \brief Iteration variables */
  Array<Var> variables;
  /*! \brief Conditions */
  Array<Expr> conditions;
  /*! \brief A map from variables (both iteration variables and parameters) to the corresponding
   *   ranges. Ranges may be integer or symbolic, the main thing is that we should be able to
   *   create IterVars with them. */
  Map<Var, Range> ranges;
  /*! \brief constructor */
  DomainNode() {}
  ~DomainNode() {}

  void VisitAttrs(AttrVisitor *v) {
    CHECK(v);
    v->Visit("variables", &variables);
    v->Visit("conditions", &conditions);
    v->Visit("ranges", &ranges);
  }
  TVM_DLL static Domain make(Array<Var> variables, Array<Expr> conditions, Map<Var, Range> ranges);

  static constexpr const char *_type_key = "Domain";
  TVM_DECLARE_NODE_TYPE_INFO(DomainNode, Node);
};

TVM_DEFINE_NODE_REF(Domain, DomainNode);

class DomainTransformation;

/*! \brief Node representing the result of domain transformation. */
class DomainTransformationNode : public Node {
 public:
  /*! \brief New domain */
  Domain new_domain;
  /*! \brief Old domain */
  Domain old_domain;
  /*! \brief A map from new variables to the corresponding expressions in terms of old variables */
  Map<Var, Expr> new_to_old;
  /*! \brief A map from old variables to the corresponding expressions in terms of new variables */
  Map<Var, Expr> old_to_new;
  /*! \brief constructor */
  DomainTransformationNode() {}
  ~DomainTransformationNode() {}

  void VisitAttrs(AttrVisitor *v) {
    CHECK(v);
    v->Visit("new_domain", &new_domain);
    v->Visit("old_domain", &old_domain);
    v->Visit("new_to_old", &new_to_old);
    v->Visit("old_to_new", &old_to_new);
  }
  TVM_DLL static DomainTransformation make(Domain new_domain, Domain old_domain, Map<Var, Expr> new_to_old,
                                           Map<Var, Expr> old_to_new);

  static constexpr const char *_type_key = "DomainTransformation";
  TVM_DECLARE_NODE_TYPE_INFO(DomainTransformationNode, Node);
};

class DomainTransformation : public NodeRef {
 public:
  DomainTransformation() {}
  explicit DomainTransformation(const ObjectPtr<Object> &n) : NodeRef(n) {}
  ~DomainTransformation() {}
  const DomainTransformationNode *operator->() const { return static_cast<const DomainTransformationNode *>(get()); }
  using ContainerType = DomainTransformationNode;

  /*! \brief Compose this domain transformation with another domain transformation. */
  DomainTransformation operator+=(const DomainTransformation &other);
};

/*!
 * \brief Simplify the expression as thoroughly as possible by using all available simplifiers.
 * Including: CanonicalSimplify, Simplify_cce and the AutoDiffSimplify.
 */
TVM_DLL Expr SuperSimplify(Expr e, const Map<Var, Range> &vranges = Map<Var, Range>());

/*!
 * \brief Provability check that uses SuperSimplify
 */
TVM_DLL bool CanProve(const Expr &e, const Map<Var, Range> &vranges = Map<Var, Range>());

/*!
 * \brief Compose two domain transformations into one.
 */
TVM_DLL DomainTransformation ComposeDomainTransformations(const DomainTransformation &first,
                                                          const DomainTransformation &second);

/*!
 * \brief Create a domain transformation transforming the given domain to the empty domain (with no
 *  variables and a single false condition).
 */
TVM_DLL DomainTransformation EmptyDomainTransformation(const Domain &domain);

/*!
 * \brief Create a domain transformation that transforms the given domain to itself.
 */
TVM_DLL DomainTransformation IdDomainTransformation(const Domain &domain);

/*!
 * \brief Clone the reduction by cloning its iteration variables.
 */
TVM_DLL Expr CloneReduction(const Expr &expr);

/*!
 * \brief Check if the given combiner represents summation.
 */
TVM_DLL bool IsSumCombiner(const CommReducer &combiner, const Map<Var, Range> &vranges = Map<Var, Range>());

/*!
 * \brief Check if zero may be factored out of a reduction with this combiner when it is in
 *  the \p value_index position.
 *
 *  For example, if the combiner works on tuples of two elements and `value_index = 1`,
 *  check that `(a, 0) combine (b, 0) = (c, 0)` for any a, b and some c.
 *  Note that all combiners generated by akg have this property.
 */
TVM_DLL bool CanFactorZeroFromCombiner(const CommReducer &combiner, int value_index,
                                       const Map<Var, Range> &vranges = Map<Var, Range>());

/*!
 * \brief Transform the expression into `c ? e : 0`, that is lift the condition of being
 *  possible to be non-zero to the top level.
 */
TVM_DLL Expr LiftNonzeronessCondition(const Expr &expr);

/*!
 * \brief If the body of the tensor consists of a single tensor call (indexing) expression,
 *  inline it.
 */
TVM_DLL Tensor InlineTailCall(const Tensor &tensor);

/*!
 * \brief Inline tensors recursively.
 *
 *  This function will inline tensors recursively until it reaches a tensor which is impossible to
 *  inline (a reduction if \p inline_reductions is false, a non-compute tensor, a tensor which is
 *  not from \p inlineable). It won't descend into non-inlinable tensors' bodies.
 *
 * \param expr The expression to transform.
 * \param inlineable A list of tensors which are allowed to be inlined. If empty, try
 *  to inline all tensors.
 * \param inline_reductions Whether to inline reductions (this may result in top-level reduction
 *  nodes).
 */
TVM_DLL Expr InlineTensors(const Expr &expr, const Array<Tensor> &inlineable = Array<Tensor>(),
                           bool inline_reductions = false);

/*!
 * \brief Inline tensors recursively.
 *
 *  This function will inline tensors recursively until it reaches a tensor which is impossible to
 *  inline (a reduction if \p inline_reductions is false, a non-compute tensor, a tensor which is
 *  not from \p inlineable). It won't descend into non-inlinable tensors' bodies.
 *
 * \param tensor The tensor whose body to transform.
 * \param inlineable A list of tensors which are allowed to be inlined. If empty, try
 *  to inline all tensors.
 * \param inline_reductions Whether to inline reductions (this may result in top-level reduction
 *  nodes).
 */
TVM_DLL Tensor InlineTensors(const Tensor &tensor, const Array<Tensor> &inlineable = Array<Tensor>(),
                             bool inline_reductions = false);

/*!
 * \brief A struct representing a set of inequalities describing bounds of a variable.
 *
 *  Given a variable x, this struct represents the following (in)equalities:
 *  - `coef*x >= low` for each `low` in `lower`
 *  - `coef*x == eq` for each `eq` in `equal`
 *  - `coef*x <= upp` for each `upp` in `upper`
 *
 *  Note that every array is supposed to be sorted in the order of increasing expression
 *  complexity.
 */
struct VarBounds {
  Expr coef;
  Array<Expr> lower;
  Array<Expr> equal;
  Array<Expr> upper;

  /*!
   * \brief Perform substitution on all components of the struct.
   */
  VarBounds substitute(const Map<Var, Expr> &subst) const;
};

/*!
 * \brief A struct representing a system of inequalities resulted from Fourier-Motzkin elimination.
 */
struct SolveSystemOfInequalitiesResult {
  Array<Var> variables;
  std::unordered_map<const Variable *, VarBounds> bounds;
  Array<Expr> other_conditions;

  /*!
   * \brief Combine the information into an array of (in)equalities.
   */
  Array<Expr> as_conditions() const;
};

/*!
 * \brief Rewrite the system of inequalities using Fourier-Motzkin elimination.
 *
 *  This function takes an array of (in)equalities and an array of variables, and essentially
 *  rewrites the (in)equalities into an array of (in)equalities of the following form:
 *
 *      x0 >= f0(x1, x2, ..., xn)
 *      x0 <= g0(x1, x2, ..., xn)
 *      x1 >= f1(x2, ..., xn)
 *      x1 <= g1(x2, ..., xn)
 *      ...
 *      xn >= fn()  // just a constant
 *      xn <= gn()  // just a constant
 *
 *  This array is represented in a more structural way using SolveSystemOfInequalitiesResult.
 *
 *  Note that the algorithm is extremely slow, it is super-exponential, so please provide variable
 *  ranges to aid the removal of redundant inequalities.
 *
 * \param inequalities The original (in)equalities.
 * \param variables The variables x0, ..., xn
 * \param vranges A map from variables to the corresponding value ranges. Extremely important for
 *   efficiency.
 */
TVM_DLL SolveSystemOfInequalitiesResult SolveSystemOfInequalities(const Array<Expr> &inequalities,
                                                                  const Array<Var> &variables,
                                                                  const Map<Var, Range> &vranges);

// Number of times to solve system of equiation and transform the domain.
const int N_REPEAT_TRANSFORM = 2;

/*!
 * \brief Simplify an iteration domain.
 *
 *  An iteration domain is basically an array of variables and a condition. The function will do the
 *  following:
 *  - Replace div and mod operations with new variables (optional).
 *  - Extract (in)equalities from the condition.
 *  - Perform Fourier-Motzkin elimination.
 *  - Shear the domain of iteration (e.g. if `y <= x <= y + 2` then x will be replaced with `y + d`
 *    where `d` is a new variable such that `0 <= d <= 2`).
 *  - Remove redundant variables.
 *  - Infer new variable ranges (hopefully more precise).
 *
 * \param domain The original domain.
 * \param eliminate_div_mod Whether to eliminate div and mod by introducing new variables.
 * \param keep_dims Whether to keep the dims equal 1
 */
TVM_DLL DomainTransformation SimplifyDomain(const Domain &domain, bool eliminate_div_mod = true,
                                            bool keep_dims = false);

/*!
 * \brief Simplify the iteration domain of a reduction expression using SimplifyDomain.
 */
TVM_DLL Expr SimplifyReductionDomain(const Expr &expr, const Map<Var, Range> &outer_vranges);

/*!
 * \brief Extract the given expression under the given condition as a separate tensor if the volume
 *  of the extracted tensor will be less than the volume of the \p outer_axis.
 *
 * \param expr The expression to extract.
 * \param cond A condition which is assumed to be true.
 * \param outer_axis Some variables, usually input variables of the enclosing tensor.
 * \param vranges Information about ranges of variables.
 * \param keep_dims Select to keep or not the dimensions equal 1 in newly created Jacobians
 * \return Either a call to an extracted tensor or the original expression.
 */
TVM_DLL Expr ExtractAsTensorMaybe(const Expr &expr, const Expr &cond, const Array<Var> &outer_axis,
                                  const Map<Var, Range> &vranges, bool keep_dims = false);

/*!
 * \brief Extract reductions as separate tensors. This may be needed when non-top-level reductions
 *  are created.
 *
 * \param expr The expression from which to extract reductions.
 * \param outer_axis Input variables of the enclosing tensor.
 * \param vranges Information about ranges of variables.
 * \return An expression without non-top-level reductions.
 */
TVM_DLL Expr ExtractReductions(const Expr &expr, const Array<Var> &outer_axis, const Map<Var, Range> &vranges);

/*!
 * \brief Extract reductions as separate tensors, but if the expr itself is a reduction, leave it
 *  intact.
 *
 * \param expr The expression from which to extract reductions.
 * \param outer_axis Input variables of the enclosing tensor.
 * \param vranges Information about ranges of variables.
 * \return An expression without non-top-level reductions.
 */
TVM_DLL Expr ExtractNonTopReductions(const Expr &expr, const Array<Var> &outer_axis, const Map<Var, Range> &vranges);

/*!
 * \brief Perform lifting of conditions of being possible to be non-zero together with
 *  applying some transformations like simplifying the reduction domain. Works only with
 *  this particular tensor's body, i.e. doesn't perform inlining.
 *
 * \param tensor The original tensor;
 * \param keep_dims Select to keep or not the dimensions equal 1 in newly created Jacobians
 * \param vranges Optional map from free variables to their value ranges.
 * \return An optimized tensor.
 */
TVM_DLL Tensor OptimizeAndLiftNonzeronessConditions(const Tensor &tensor, bool keep_dims = false,
                                                    const Map<Var, Range> &vranges = Map<Var, Range>());

/*!
 * \brief Pretty print the tensor with all its dependencies.
 */
TVM_DLL std::string PrintTensorRecursively(const Tensor &tensor);

/*!
 * \brief Pretty print the tensors with all their dependencies.
 */
TVM_DLL std::string PrintTensorsRecursively(const Array<Tensor> &tensor);

/*!
 * \brief Check if the given expr is a const of any type equal to the given integer value.
 * \param e The expression.
 * \param value The value to compare to.
 * \return Whether the expression is a const equal to the value.
 * \tparam ValueType The value type
 */
template <typename ValueType>
inline bool is_const_value(const Expr &e, ValueType value);

template <typename ValueType>
inline bool is_const_value(const Expr &e, ValueType value) {
  static_assert(std::is_integral<ValueType>::value, "Comparison to non-integer values is forbidden.");
  if (const auto *i = e.as<IntImm>()) {
    return i->value == value;
  } else if (const auto *ui = e.as<UIntImm>()) {
    return (value >= 0) && (ui->value == static_cast<uint32_t>(value));
  } else if (const auto *fi = e.as<FloatImm>()) {
    return fi->value == value;
  } else if (const auto *c = e.as<Cast>()) {
    return is_const_value(c->value, value);
  } else if (const auto *b = e.as<Broadcast>()) {
    return is_const_value(b->value, value);
  } else {
    return false;
  }
}
Tensor TensorFromExpr(const Expr &expr, const Array<IterVar> &axis, const std::string &name = "tensor",
                      const std::string &tag = "", const Map<std::string, NodeRef> &attrs = {});

/*!
 * \brief Transform the body of a tensor if it is a compute tensor, otherwise return it
 *  unchanged. Note that if the compute returns a tuple, it transforms only one element,
 *  other elements are discarded.
 *
 * \param tensor The tensor to transform.
 * \param func The transformation function working on expressions and additionally taking
 *  the array of the tensor's itervars.
 * \return The transformed tensor.
 */
Tensor TransformBody(const Tensor &tensor, const std::function<Expr(const Expr &, const Array<IterVar> &)> &func);

/*!
 * \brief Transform the body of a tensor if it is a compute tensor, otherwise return it
 *  unchanged. Note that if the compute returns a tuple, it transforms only one element,
 *  other elements are discarded.
 *
 * \param tensor The tensor to transform.
 * \param func The transformation function (working on expressions).
 * \return The transformed tensor.
 */
Tensor TransformBody(const Tensor &tensor, const std::function<Expr(const Expr &)> &func);

Map<Var, Range> IterVarsToMap(const Array<IterVar> &itervars);
}  // namespace ir
}  // namespace akg

#endif  // PASS_ZERO_ELIMINATION_H_
