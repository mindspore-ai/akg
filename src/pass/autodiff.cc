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
#include "pass/autodiff.h"
#include <tvm/ir_mutator.h>
#include "ir_pass.h"
#include "pass/autodiff_cce.h"
#include "pass/zero_elimination.h"

namespace akg {
namespace ir {
DifferentiationResult DifferentiationResultNode::make(Array<Tensor> result, Map<Tensor, Tensor> adjoints,
                                                      Map<Tensor, Map<Tensor, Tensor>> summands) {
  auto n = make_node<DifferentiationResultNode>();
  n->result = std::move(result);
  n->adjoints = std::move(adjoints);
  n->adjoint_summands = std::move(summands);
  return DifferentiationResult(n);
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
  .set_dispatch<DifferentiationResultNode>([](const ObjectRef &node, IRPrinter *p) {
    auto r = static_cast<const DifferentiationResultNode *>(node.get());
    p->stream << "DifferentiationResult(result=" << r->result << ", adjoints=" << r->adjoints
              << ", adjoint_summands=" << r->adjoint_summands << ')';
  });

TVM_REGISTER_NODE_TYPE(DifferentiationResultNode);

/*! \brief Differentiate an expression wrt a variable or a tensor element */
class JacobianMutator : public IRMutator {
 public:
  /*!
   * \brief Differentiate wrt `input(indices)`.
   * \param input The input tensor.
   * \param indices The indices of the element with respect to which to differentiate.
   */
  JacobianMutator(Tensor input, Array<Expr> indices) : input_(std::move(input)), indices_(std::move(indices)) {}
  /*!
   * \brief Differentiate wrt the input variable.
   * \param input The input variable.
   */
  explicit JacobianMutator(VarExpr input) : input_var_(std::move(input)) {}

  ~JacobianMutator() override = default;

  Expr Mutate(Expr e) override {
    if ((e.type().is_int() || e.type().is_uint()) && is_const(e)) {
      // Assume that the derivative of any integer const is always 0
      return make_zero(e.type());
    } else {
      return IRMutator::Mutate(e);
    }
  }

  Expr Mutate_(const Variable *op, const Expr &e) override {
    if (input_var_.operator->() && input_var_.get() == op && op->type.is_float()) {
      return FloatImm::make(op->type, 1.0);
    } else {
      return make_zero(op->type);
    }
  }

  Expr Mutate_(const Call *op, const Expr &e) override {
    if (op->call_type == Call::CallType::Halide) {
      if (input_.operator->() && op->func.same_as(input_->op) && op->value_index == input_->value_index) {
        Expr condition = const_true();
        for (size_t i = 0; i < input_.ndim(); ++i) {
          condition = And::make(condition, EQ::make(indices_[i], op->args[i]));
        }
        return Cast::make(op->type, condition);
      } else {
        return make_zero(op->type);
      }
    } else if (op->call_type == Call::CallType::PureIntrinsic) {
      static std::unordered_set<std::string> piecewise_const = {"floor", "ceil", "trunc", "round"};
      if (op->name == "exp") {
        return Mul::make(Mutate(op->args[0]), e);
      } else if (op->name == "mad") {
        Expr a = Mutate(op->args[0]);
        Expr b = Mutate(op->args[1]);
        if (is_const_ad(a) && is_const_ad(b)) return a + b;
        return Call::make(e.type(), "mad", {Mutate(op->args[0]), Mutate(op->args[1])}, Call::PureIntrinsic);
      } else if (op->name == "cos") {
        return Mul::make(FloatImm::make(e.type(), -1.000),
                         Mul::make(Mutate(op->args[0]), Call::make(e.type(), "sin", op->args, Call::PureIntrinsic)));
      } else if (op->name == "log") {
        return Div::make(Mutate(op->args[0]), op->args[0]);
      } else if (op->name == "sigmoid") {
        return Mul::make(Mutate(op->args[0]), Mul::make(e, Sub::make(FloatImm::make(e.type(), 1.0), e)));
      } else if (op->name == "sqrt") {
        return Div::make(Mutate(op->args[0]), Mul::make(e, FloatImm::make(e.type(), 2.0)));
      } else if (op->name == "tanh") {
        return Mul::make(Mutate(op->args[0]), Sub::make(FloatImm::make(e.type(), 1.0), Mul::make(e, e)));
      } else if (op->name == "pow") {
        auto x = op->args[0], y = op->args[1];
        return e * (Mutate(y) * log(x) + Mutate(x) * y / x);
      } else if (op->name == "fabs") {
        auto type = op->args[0].type();
        return Mul::make(Mutate(op->args[0]), Select::make(GE::make(op->args[0], make_zero(type)),
                                                           FloatImm::make(type, 1.0), FloatImm::make(type, -1.0)));
      } else if (op->name == "rsqrt") {
        auto type = op->args[0].type();
        return Mul::make(Mutate(op->args[0]), Div::make(Mul::make(FloatImm::make(type, -0.5), e), op->args[0]));
      } else if (op->name == air::ir::intrinsic::tvm_if_then_else) {
        Array<Expr> new_args = {op->args[0], Mutate(op->args[1]), Mutate(op->args[2])};
        return Call::make(op->type, op->name, new_args, op->call_type, op->func, op->value_index);
      } else if (piecewise_const.count(op->name)) {
        return FloatImm::make(e.type(), 0.0);
      } else {
        LOG(FATAL) << "Derivative of this intrinsic is not implemented: " << op->name;
      }
    }
    LOG(FATAL) << "Derivative of this expr is not implemented";
    return e;
  }

  Expr Mutate_(const Add *op, const Expr &e) override { return air::ir::Add::make(Mutate(op->a), Mutate(op->b)); }

  Expr Mutate_(const Sub *op, const Expr &e) override { return air::ir::Sub::make(Mutate(op->a), Mutate(op->b)); }

  Expr Mutate_(const Mul *op, const Expr &e) override {
    return Add::make(Mul::make(Mutate(op->a), op->b), Mul::make(op->a, Mutate(op->b)));
  }

  Expr Mutate_(const Div *op, const Expr &e) override {
    if (op->b.as<IntImm>() || op->b.as<UIntImm>() || op->b.as<FloatImm>()) {
      // When the divisor is a const, Simplify_cce this case to avoid b*b
      return Div::make(Mutate(op->a), op->b);
    } else {
      return Div::make(Sub::make(Mul::make(Mutate(op->a), op->b), Mul::make(op->a, Mutate(op->b))),
                       Mul::make(op->b, op->b));
    }
  }

  Expr Mutate_(const Min *op, const Expr &e) override {
    return Select::make(LT::make(op->a, op->b), Mutate(op->a), Mutate(op->b));
  }

  Expr Mutate_(const Max *op, const Expr &e) override {
    return Select::make(GT::make(op->a, op->b), Mutate(op->a), Mutate(op->b));
  }

  Expr Mutate_(const Reduce *op, const Expr &e) override {
    // This case is relatively difficult because a reduction expression
    // may use an arbitrary combiner.
    // The resulting reduction expression will return a tuple containing
    // both derivatives and the original results (in exactly this order).

    // We have to clone the reduction axes because otherwise the original expression
    // cannot be used together with the derivative (it will lead to errors during lowering)
    Expr expr_with_new_axes = CloneReduction(e);
    op = expr_with_new_axes.as<Reduce>();

    // New lhs and rhs variables of the new combiner consist of variables
    // representing derivatives followed by the original variables.
    Array<Var> new_lhs;
    CHECK(op);
    std::transform(op->combiner->lhs.begin(), op->combiner->lhs.end(), std::back_inserter(new_lhs.CopyOnWrite()->data),
                   [](const Var &var) { return (var.copy_with_suffix(".der")); });
    std::copy(op->combiner->lhs.begin(), op->combiner->lhs.end(), std::back_inserter(new_lhs.CopyOnWrite()->data));

    Array<Var> new_rhs;
    std::transform(op->combiner->rhs.begin(), op->combiner->rhs.end(), std::back_inserter(new_rhs.CopyOnWrite()->data),
                   [](const Var &var) { return (var.copy_with_suffix(".der")); });
    std::copy(op->combiner->rhs.begin(), op->combiner->rhs.end(), std::back_inserter(new_rhs.CopyOnWrite()->data));

    // The new combiner result also consists of the resulting derivatives
    // followed by the original results.
    Array<Expr> new_result;
    for (const auto &res : op->combiner->result) {
      // Each resulting derivative is computed as a sum of derivatives
      // wrt lhs and rhs multiplied by the derivatives of lhs and rhs
      Expr new_res = make_zero(res.type());
      for (size_t i = 0; i < op->combiner->lhs.size(); ++i) {
        Expr res_di = Derivative(res, op->combiner->lhs[i]);
        // new_lhs[i] is the derivative of lhs[i] (wrt our input tensor)
        new_res = Add::make(new_res, Mul::make(new_lhs[i], res_di));
      }
      for (size_t i = 0; i < op->combiner->rhs.size(); ++i) {
        Expr res_di = Derivative(res, op->combiner->rhs[i]);
        new_res = Add::make(new_res, Mul::make(new_rhs[i], res_di));
      }
      new_result.push_back(new_res);
    }
    std::copy(op->combiner->result.begin(), op->combiner->result.end(),
              std::back_inserter(new_result.CopyOnWrite()->data));

    // The identity is transformed in a similar way
    Array<Expr> new_identity;
    std::transform(op->combiner->identity_element.begin(), op->combiner->identity_element.end(),
                   std::back_inserter(new_identity.CopyOnWrite()->data), [this](const Expr &id) { return Mutate(id); });
    std::copy(op->combiner->identity_element.begin(), op->combiner->identity_element.end(),
              std::back_inserter(new_identity.CopyOnWrite()->data));

    Array<Expr> new_source;
    std::transform(op->source.begin(), op->source.end(), std::back_inserter(new_source.CopyOnWrite()->data),
                   [this](const Expr &src) { return Mutate(src); });
    std::copy(op->source.begin(), op->source.end(), std::back_inserter(new_source.CopyOnWrite()->data));

    CommReducer new_combiner = CommReducerNode::make(new_lhs, new_rhs, new_result, new_identity);
    // Also Simplify_cce the resulting combiner (mostly to get rid of unused components)
    return SimplifyCombiner(Reduce::make(new_combiner, new_source, op->axis, op->condition, op->value_index));
  }

  Expr Mutate_(const Cast *op, const Expr &e) override {
    if (op->type.is_float()) {
      return Cast::make(op->value.type(), Mutate(op->value));
    } else {
      return make_zero(op->value.type());
    }
  }

  Expr Mutate_(const Select *op, const Expr &e) override {
    return Select::make(op->condition, Mutate(op->true_value), Mutate(op->false_value));
  }

  Expr Mutate_(const IntImm *op, const Expr &e) override { return air::ir::IntImm::make(op->type, 0); }
  Expr Mutate_(const UIntImm *op, const Expr &e) override { return air::ir::UIntImm::make(op->type, 0); }

  Expr Mutate_(const FloatImm *op, const Expr &e) override { return air::ir::FloatImm::make(op->type, 0); }

 private:
  Tensor input_;
  Array<Expr> indices_;
  VarExpr input_var_;
};

Expr Jacobian(const Expr &expr, const Tensor &input, const Array<Expr> &indices) {
  return JacobianMutator(input, indices).Mutate(expr);
}

Expr Derivative(const Expr &expr, const VarExpr &var) { return JacobianMutator(var).Mutate(expr); }

Tensor Jacobian(const Tensor &output, const Tensor &input, bool &used_head, bool optimize, bool keep_dims,
                const Tensor &head) {
  if (const auto op = output->op.as<ComputeOpNode>()) {
    bool is_input_tensor = false;
    auto array = op->InputTensors();
    if (std::any_of(array.begin(), array.end(), [&](const Tensor &child) { return (input == child); })) {
      is_input_tensor = true;
    }
    CHECK(is_input_tensor) << "Jacobian is called on a pair of tensors such that the output "
                           << "does not depend on the input. This is probably a mistake.";

    // We have to clone the iteration axes because otherwise the original expression
    // cannot be used together with the derivative (it will lead to errors during lowering)
    Array<IterVar> new_axis;
    std::unordered_map<const Variable *, Expr> vmap;
    for (IterVar iv : op->axis) {
      IterVar new_v = IterVarNode::make(iv->dom, iv->var.copy_with_suffix(""), iv->iter_type, iv->thread_tag);
      new_axis.push_back(new_v);
      vmap[iv->var.operator->()] = new_v;
    }

    // Generate new itervars for the input
    Array<Expr> input_itervars;
    size_t i = 0;
    for (Expr ext : input->shape) {
      IterVar new_v = IterVarNode::make(Range(0, ext), Var("jac_i" + std::to_string(i)), air::IterVarType::kDataPar);
      // Append them to new_axis
      new_axis.push_back(new_v);
      // We also need a separate array of these itervars
      input_itervars.push_back(new_v);
      ++i;
    }

    // The differentiation itself happens here
    Expr new_body = Jacobian(air::ir::Substitute(op->body[output->value_index], vmap), input, input_itervars);
    new_body = Simplify_cce(new_body);

    int value_index = 0;
    Array<Expr> new_bodies;
    Array<Expr> new_shape = output->shape;
    std::copy(input->shape.begin(), input->shape.end(), std::back_inserter(new_shape.CopyOnWrite()->data));

    // If this is a reduction then it may return a tuple and we have
    // to repeat the body several times
    const auto red = new_body.as<Reduce>();
    if (red) {
      value_index = red->value_index;
      for (size_t j = 0; j < red->source.size(); ++j) {
        new_bodies.push_back(Reduce::make(red->combiner, red->source, red->axis, red->condition, static_cast<int>(j)));
      }
    } else {
      new_bodies.push_back(new_body);
    }

    auto new_op = ComputeOpNode::make(op->name + "_jacobian", op->tag, op->attrs, new_axis, new_bodies);
    Tensor tensor = TensorNode::make(new_shape, output->dtype, new_op, value_index);

    if (red) {
      tensor = OptimizeReduction(tensor, op, red, output, input, new_shape, new_axis, head, used_head);
    }

    if (optimize) {
      tensor = OptimizeAndLiftNonzeronessConditions(tensor, keep_dims);
    }

    return tensor;
  } else {
    LOG(FATAL) << "Derivative of this op is not implemented: " << output->op;
    return Tensor();
  }
}

Tensor DiffBuildingBlock(const Tensor &output, const Tensor &input, const Tensor &head,
                         const Map<std::string, NodeRef> &attrs, const Array<Tensor> &new_pld_array) {
  AttrMap in_attrs;
  if (attrs.defined()) {
    in_attrs = attrs;
  }
  bool ad_conv_enable_ = (in_attrs.GetIntAttr("ad_conv_enable", 0) != 0);
  bool keep_dims = (in_attrs.GetIntAttr("keep_dims", 0) != 0);

  if (ad_conv_enable_) {
    Tensor back_conv = DiffConv(output, input, head, attrs, new_pld_array);
    if (back_conv.defined()) {
      return back_conv;
    }
  }

  bool hasmad = HasMad(output);
  bool used_head = false;
  Tensor jac_output_input = Jacobian(output, input, used_head, true, keep_dims, head);
  Tensor result;
  Tensor head_cast;

  if (used_head) {
    result = jac_output_input;
  } else {
    if (head->dtype == jac_output_input->dtype) {
      head_cast = head;
    } else {
      head_cast = topi::cast(head, jac_output_input->dtype);
    }
    result = TensorDot(head_cast, jac_output_input, static_cast<int>(output->shape.size()),
                       output->op->name + "_" + input->op->name + "_grad", hasmad);
  }

  result = InlineTensors(result, {jac_output_input});
  result = OptimizeAndLiftNonzeronessConditions(result, keep_dims);
  result = InlineTailCall(result);

  return result;
}

DifferentiationResult Differentiate(const Tensor &output, const Array<Tensor> &inputs, const Tensor &head_or_null,
                                    const Map<std::string, NodeRef> &attrs, const Array<Tensor> &new_pld_array,
                                    const FDiffBuildingBlock &fdiff, const Map<Tensor, Array<Tensor>> &override_deps) {
  if (!output.get()) {
    LOG(FATAL) << "output is a null pointer.";
    return DifferentiationResult();
  }

  Tensor head = head_or_null;

  // If the head is a null pointer, create an identity tensor
  if (!head.get()) {
    Array<Expr> shape = output->shape;
    std::copy(output->shape.begin(), output->shape.end(), std::back_inserter(shape.CopyOnWrite()->data));
    auto func = [&output](const Array<air::Var> &input_indices) {
      Expr res = const_true();
      for (size_t i = 0; i < output->shape.size(); ++i) {
        res = res && (Expr(input_indices[i]) == Expr(input_indices[output->shape.size() + i]));
      }

      Expr res_cast = Select::make(res, make_const(output->dtype, 1), make_const(output->dtype, 0));
      return res_cast;
    };
    head = air::compute(shape, func, "identity");
  }

  // This map maps a tensor to the list of tensors immediately depending on it (using it in their
  // bodies)
  std::unordered_map<Tensor, std::vector<Tensor>> reverse_dependencies;

  // Map doesn't work correctly for Tensors, so convert it to std::unordered_map
  std::unordered_map<Tensor, Array<Tensor>> override_deps_map;
  for (auto pair : override_deps) {
    override_deps_map.insert(pair);
  }

  // Collect reverse dependencies
  std::vector<Tensor> stack({output});
  while (!stack.empty()) {
    Tensor tensor = stack.back();
    stack.pop_back();

    auto it = override_deps_map.find(tensor);
    Array<Tensor> deps = it != override_deps_map.end() ? it->second : tensor->op->InputTensors();

    for (const Tensor &child : deps) {
      if (!reverse_dependencies.count(child)) {
        stack.push_back(child);
      }
      reverse_dependencies[child].push_back(tensor);
    }
  }

  // Individual summands of the adjoints
  std::unordered_map<Tensor, Map<Tensor, Tensor>> summands;

  // This map maps tensors to the corresponding adjoints (dLoss/dTensor)
  std::unordered_map<Tensor, Tensor> adjoints;
  // head is the adjoint of output by definition
  adjoints[output] = head;

  // This is a recursive function that does all the work. It computes the adjoint for a given
  // tensor, adds it to the map, and returns it
  std::function<Tensor(const Tensor &)> compute_adjoint;
  compute_adjoint = [&compute_adjoint, &adjoints, &summands, &reverse_dependencies, &fdiff, &attrs, &new_pld_array,
                     &head, &output](const Tensor &tensor) {
    if (!adjoints.count(tensor)) {
      // Here the adjoint hasn't been computed yet
      Tensor res_adjoint;
      std::vector<Tensor> deps = reverse_dependencies[tensor];
      if (deps.empty()) {
        // No reverse dependencies means that the output does not depend on this tensor,
        // return a zero tensor of the appropriate shape
        Array<Expr> result_shape(head->shape.begin(),
                                 head->shape.end() + static_cast<size_t>(0 - output->shape.size()));
        std::copy(tensor->shape.begin(), tensor->shape.end(), std::back_inserter(result_shape.CopyOnWrite()->data));
        res_adjoint = topi::full(result_shape, output->dtype, make_zero(output->dtype));
      } else {
        // The new adjoint is computed as a sum of the reverse dependencies' adjoints multiplied
        // by the corresponding "local" jacobians (dDep/dTensor). The computation of the jacobian
        // and the multiplication is done in the function fdiff (DiffBuildingBlock by default).
        for (const Tensor &dep : deps) {
          Tensor part = fdiff(dep, tensor, compute_adjoint(dep), attrs, new_pld_array);
          if (res_adjoint.get()) {
            if (res_adjoint->dtype != part->dtype) {
              res_adjoint = topi::cast(res_adjoint, part->dtype);
            }
          }
          res_adjoint = res_adjoint.get() ? topi::add(res_adjoint, part) : part;

          // Add this part to summands
          auto &summands_of_adjoint = summands[tensor];
          if (summands_of_adjoint.get()) {
            summands_of_adjoint.Set(dep, part);
          } else {
            summands_of_adjoint = Map<Tensor, Tensor>({{dep, part}});
          }
        }
      }

      adjoints[tensor] = res_adjoint;
      return res_adjoint;
    } else {
      return adjoints[tensor];
    }
  };

  // Adjoints corresponding to inputs
  Array<Tensor> result;

  // If inputs is empty, compute adjoints for all tensors, on which output depends
  if (inputs.empty()) {
    for (const auto &dep : reverse_dependencies) {
      static_cast<void>(compute_adjoint(dep.first));
    }
  }

  // Compute an adjoint for each input
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(result.CopyOnWrite()->data), compute_adjoint);

  AttrMap in_attrs;
  if (attrs.defined()) {
    in_attrs = attrs;
  }

  bool tensor_optimize_ = (in_attrs.GetIntAttr("tensor_optimize", 0) != 0);
  if (!tensor_optimize_) {
    return DifferentiationResultNode::make(result, adjoints, summands);
  } else {  // Running TIL optimization passes
    Array<Tensor> optimized_result;
    ADOptimizePasses(result, optimized_result, attrs, new_pld_array);
    // AD FINISHED... Returning to Poly
    return DifferentiationResultNode::make(optimized_result, adjoints, summands);
  }
}

TVM_REGISTER_API("akg.autodiff.Jacobian").set_body([](const TVMArgs args, TVMRetValue *ret) {
  bool used_head = false;
  if (args.size() >= 4) {
    *ret = Jacobian(args[0], args[1], used_head, args[2].operator bool(), args[3].operator bool());
  } else {
    if (args.size() >= 3) {
      *ret = Jacobian(args[0], args[1], used_head, args[2].operator bool());
    } else {
      *ret = Jacobian(args[0], args[1], used_head);
    }
  }
});

TVM_REGISTER_API("akg.autodiff.Derivative").set_body([](const TVMArgs args, TVMRetValue *ret) {
  *ret = Derivative(args[0], args[1]);
});

TVM_REGISTER_API("akg.autodiff.DiffBuildingBlock").set_body([](const TVMArgs args, TVMRetValue *ret) {
  *ret = DiffBuildingBlock(args[0], args[1], args[2], args[3], args[4]);
});

TVM_REGISTER_API("akg.autodiff.Differentiate").set_body([](const TVMArgs args, TVMRetValue *ret) {
  CHECK(args.size()) << "No input args.";
  if (args.size() == 1) {
    *ret = Differentiate(args[0]);
  } else if (args.size() == 2) {
    *ret = Differentiate(args[0], args[1]);
  } else if (args.size() == 3) {
    *ret = Differentiate(args[0], args[1], args[2]);
  } else if (args.size() == 4) {
    *ret = Differentiate(args[0], args[1], args[2], args[3]);
  } else if (args.size() == 5) {
    *ret = Differentiate(args[0], args[1], args[2], args[3], args[4]);
  } else if (args.size() >= 6) {
    auto pfunc = args[5].operator PackedFunc();
    auto fdiff = [pfunc](const Tensor &o, const Tensor &i, const Tensor &h, const Map<std::string, NodeRef> &attrs,
                         const Array<Tensor> &new_pld) { return pfunc(o, i, h, attrs, new_pld); };

    if (args.size() >= 7) {
      *ret = Differentiate(args[0], args[1], args[2], args[3], args[4], fdiff, args[6]);
    } else {
      *ret = Differentiate(args[0], args[1], args[2], args[3], args[4], fdiff);
    }
  }
});
}  // namespace ir
}  // namespace akg
