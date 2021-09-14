/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "pass/utils.h"
#include "pass/autodiff_cce.h"
#include "pass/zero_elimination.h"

namespace akg {
namespace ir {

class TmpTensorChecker : public IRVisitor {
 public:
  explicit TmpTensorChecker(std::map<FunctionRef, bool> tmpTensorMap) : tmpTensorMap_(std::move(tmpTensorMap)) {}
  void Visit_(const Call *op) override {
    if (tmpTensorMap_.find(op->func) != tmpTensorMap_.end()) {
      tmpTensorMap_[op->func] = true;
    }
    IRVisitor::Visit_(op);
  }

  std::map<FunctionRef, bool> getTmpTensor() const { return tmpTensorMap_; }

 private:
  std::map<FunctionRef, bool> tmpTensorMap_;
};

/*! \brief Differentiate an expression wrt a variable or a tensor element */
class JacobianMutator : public IRMutator {
 public:
  /*!
   * \brief Differentiate wrt `input(indices)`.
   * \param input The input tensor.
   * \param indices The indices of the element with respect to which to differentiate.
   */
  JacobianMutator(Tensor input, Array<Expr> indices) : input_(std::move(input)), indices_(std::move(indices)) {}
  JacobianMutator(Tensor input, Tensor old_output, Array<Expr> indices, Array<IterVar> jac_indices,
                  std::map<FunctionRef, Tensor> tensor_mapping, std::map<FunctionRef, bool> tmp_result = {})
      : input_(std::move(input)),
        old_output_(std::move(old_output)),
        indices_(std::move(indices)),
        jac_indices_(std::move(jac_indices)),
        tensor_mapping_(std::move(tensor_mapping)),
        tmp_result_(std::move(tmp_result)) {}
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

  Stmt Mutate(Stmt stmt) override {
    if (outer_stmt_) {
      if (tmp_result_.find(old_output_->op) != tmp_result_.end() && tmp_result_[old_output_->op]) {
        Region region;
        for (auto dim : old_output_->shape) {
          region.push_back(Range::make_by_min_extent(0, dim));
        }
        stmt = Realize::make(old_output_->op, old_output_->value_index, old_output_->dtype, region, const_true(), stmt);
        stmt = AttrStmt::make(old_output_->op, "realize_scope", StringImm::make("local"), stmt);
      }

      for (auto index = jac_indices_.rbegin(); index != jac_indices_.rend(); ++index) {
        stmt = For::make((*index)->var, (*index)->dom->min, (*index)->dom->extent, air::ir::ForType::Serial,
                         air::ir::DeviceAPI::None, stmt);
      }
      outer_stmt_ = false;
    }

    stmt = IRMutator::Mutate(stmt);

    return stmt;
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
      } else if (tensor_mapping_.find(op->func) != tensor_mapping_.end()) {
        Array<Expr> call_args;

        if (old_output_ == Downcast<Operation>(op->func).output(op->value_index)) {
          for (auto index : jac_indices_) {
            call_args.push_back(index);
          }
        }

        for (auto index : op->args) {
          call_args.push_back(index);
        }
        return Call::make(op->type, op->name + "_" + input_->op->name + "_jac", call_args, op->call_type,
                          tensor_mapping_[op->func]->op);
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

  Stmt Mutate_(const Realize *op, const Stmt &s) {
    realize_node_[Operation(GetObjPtr(op->func.get())).output(op->value_index)] = op;
    Stmt body = IRMutator::Mutate(op->body);
    realize_body_[op->func] = body;
    return Realize::make(op->func, op->value_index, op->type, op->bounds, op->condition, body);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) {
    FunctionRef func = Downcast<FunctionRef>(op->node);
    attr_node_[func] = op;

    Stmt ret = IRMutator::Mutate_(op, s);

    if (tmp_result_.find(func) != tmp_result_.end() && !tmp_result_[func]) {
      ret = realize_body_[func];
    }

    if (tensor_mapping_.find(func) != tensor_mapping_.end() && func != old_output_->op) {
      // find new tensor added for reduction on the node of this attr
      // TODO(ZICHUN): check this hard coded 0
      Tensor output = Downcast<Operation>(func).output(0);

      const Realize *ref_real = realize_node_[output];
      const AttrStmt *ref_attr = attr_node_[output->op];
      auto x = tensor_mapping_[func];

      ret = Realize::make(x->op, x->value_index, x->dtype, ref_real->bounds, ref_real->condition, ret);
      ret = AttrStmt::make(x->op, ref_attr->attr_key, ref_attr->value, ret);
    }

    return ret;
  }

  Stmt Mutate_(const For *op, const Stmt &s) {
    Stmt body = this->Mutate(op->body);
    return For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, body);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) {
    if (tensor_mapping_.find(op->func) == tensor_mapping_.end()) {
      Tensor tmp_tensor = Downcast<Operation>(op->func).output(op->value_index);
      Tensor tmp_tensor_jac = PlaceholderOpNode::make(tmp_tensor->op->name + "_" + input_->op->name + "_jac",
                                                      tmp_tensor->shape, tmp_tensor->dtype)
                                .output(0);
      tensor_mapping_[tmp_tensor->op] = tmp_tensor_jac;
    }

    if (tensor_mapping_.find(op->func) == tensor_mapping_.end()) {
      tmp_result_[op->func] = false;
    }

    auto new_value = this->Mutate(op->value);
    Array<Expr> provide_arg;

    if (old_output_ == Downcast<Operation>(op->func).output(op->value_index)) {
      for (auto index : jac_indices_) {
        provide_arg.push_back(index);
      }
    }

    for (auto index : op->args) {
      provide_arg.push_back(index);
    }

    Stmt jac_body = Provide::make(tensor_mapping_[op->func]->op, op->value_index, new_value, provide_arg);

    if (tmp_result_[op->func]) {
      jac_body = Block::make(jac_body, s);
    }

    return jac_body;
  }

  std::map<FunctionRef, bool> getTmpTensor() const { return tmp_result_; }

 private:
  Tensor input_;
  Tensor old_output_;
  VarExpr input_var_;
  Array<Expr> indices_;
  Array<IterVar> jac_indices_;
  std::map<FunctionRef, Tensor> tensor_mapping_;
  std::map<FunctionRef, bool> tmp_result_;

  bool outer_stmt_ = true;
  std::unordered_map<Tensor, const Realize *> realize_node_;
  std::unordered_map<FunctionRef, Stmt, air::NodeHash, air::NodeEqual> realize_body_;
  std::unordered_map<FunctionRef, const AttrStmt *, air::NodeHash, air::NodeEqual> attr_node_;
};

Expr Jacobian(const Expr &expr, const Tensor &input, const Array<Expr> &indices) {
  return JacobianMutator(input, indices).Mutate(expr);
}

Stmt Jacobian(const Stmt &stmt, const Tensor &input, const Array<Expr> &indices, const Array<IterVar> &jac_indices,
              const std::map<FunctionRef, Tensor> tensor_mapping, const Tensor &old_output) {
  auto autodiff_mutator = JacobianMutator(input, old_output, indices, jac_indices, tensor_mapping);
  Stmt result = autodiff_mutator.Mutate(stmt);

  auto tmp_check = TmpTensorChecker(autodiff_mutator.getTmpTensor());
  tmp_check.Visit(result);

  autodiff_mutator = JacobianMutator(input, old_output, indices, jac_indices, tensor_mapping, tmp_check.getTmpTensor());

  result = autodiff_mutator.Mutate(stmt);
  return result;
}

Expr Derivative(const Expr &expr, const VarExpr &var) { return JacobianMutator(var).Mutate(expr); }

Tensor Jacobian(const Tensor &output, const Tensor &input, bool &used_head, bool optimize, bool keep_dims,
                const Tensor &head) {
  const auto op = output->op.as<ComputeOpNode>();
  CHECK(op) << "A Compute op is expected but get : " << output->op;

  auto input_array = op->InputTensors();
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
}

Tensor HybridTensorDot(const Tensor &jac, const Tensor &head, const Array<IterVar> &itervars,
                       const Array<Expr> &reduction_extent) {
  const auto op = jac->op.as<air::HybridOpNode>();
  Tensor output_jac = op->outputs[0];
  Stmt jac_body = op->body;
  CHECK(op) << "A Hybrid op is expected but get : " << jac->op;

  // here we collect the arg for all tensors
  // itervar for the output/grad: jac_i0_grad, jac_i1_grad, ...
  // itervar for reduction: i0_red, i1_red, ...
  // call arg for jac: [jac_i0_grad, jac_i1_grad, ..., i0_red, i1_red, ...]
  // call arg for head: [i0_red, i1_red, ...]
  // call arg for output/grad: [jac_i0_grad, jac_i1_grad, ...]
  Array<Expr> grad_arg;
  Array<Expr> jac_arg;
  Array<Expr> head_arg;
  Array<Expr> output_shape;
  Array<IterVar> tensor_dot_itervars;

  for (auto iv : itervars) {
    IterVar new_v = IterVarNode::make(iv->dom, iv->var.copy_with_suffix("_grad"), iv->iter_type, iv->thread_tag);
    output_shape.push_back(iv->dom->extent);
    tensor_dot_itervars.push_back(new_v);

    jac_arg.push_back(new_v->var);
    grad_arg.push_back(new_v->var);
  }

  // create the placeholder for the output of the tensor dot, namely the grad
  Tensor output_grad =
    PlaceholderOpNode::make(output_jac->op->name + "_grad", output_shape, output_jac->dtype).output(0);

  Array<IterVar> reduction_axis;
  size_t count = 0;
  for (auto ext : reduction_extent) {
    IterVar new_v =
      IterVarNode::make(Range(0, ext), Var("i" + std::to_string(count) + "_red"), air::IterVarType::kDataPar);
    reduction_axis.push_back(new_v);

    jac_arg.push_back(new_v->var);
    head_arg.push_back(new_v->var);

    count++;
  }

  // create expr: head[i0_red, i1_red, ...] * jac[jac_i0_grad, jac_i1_grad, ..., i0_red, i1_red, ...]
  Expr head_call = Call::make(head->dtype, head->op->name, head_arg, Call::CallType::Halide, head->op);
  Expr jac_call = Call::make(output_jac->dtype, output_jac->op->name, jac_arg, Call::CallType::Halide, output_jac->op);
  Expr prod_head_jac = Mul::make(head_call, jac_call);

  // create provide stmt:
  // grad[jac_i0_grad, ...] = grad[jac_i0_grad, ...] + head[i0_red, ...] * jac[jac_i0_grad, ..., i0_red, ...]
  Expr grad_call =
    Call::make(output_grad->dtype, output_grad->op->name, grad_arg, Call::CallType::Halide, output_grad->op);
  Expr reduction_incre = Add::make(grad_call, prod_head_jac);
  Stmt reduction_body = Provide::make(output_grad->op, output_grad->value_index, reduction_incre, grad_arg);

  for (auto index = reduction_axis.rbegin(); index != reduction_axis.rend(); ++index) {
    reduction_body = For::make((*index)->var, (*index)->dom->min, (*index)->dom->extent, air::ir::ForType::Serial,
                               air::ir::DeviceAPI::None, reduction_body);
  }

  Stmt tensorDot_init =
    Provide::make(output_grad->op, output_grad->value_index, air::ir::FloatImm::make(output_grad->dtype, 0), grad_arg);
  Stmt tensorDot_body = Block::make(tensorDot_init, reduction_body);

  for (auto index = tensor_dot_itervars.rbegin(); index != tensor_dot_itervars.rend(); ++index) {
    tensorDot_body = For::make((*index)->var, (*index)->dom->min, (*index)->dom->extent, air::ir::ForType::Serial,
                               air::ir::DeviceAPI::None, tensorDot_body);
  }

  auto result_stmt = Block::make(jac_body, tensorDot_body);

  // previous output_jac becomes a local variable now
  // add realize node for this tensor
  Region region;
  for (auto dim : output_jac->shape) {
    region.push_back(Range::make_by_min_extent(0, dim));
  }
  result_stmt =
    Realize::make(output_jac->op, output_jac->value_index, output_jac->dtype, region, const_true(), result_stmt);
  result_stmt = AttrStmt::make(output_jac->op, "realize_scope", StringImm::make("local"), result_stmt);

  // add head as the first input of the result tensor
  Array<Tensor> new_input = {head};
  for (auto old_input : op->inputs) {
    new_input.push_back(old_input);
  }

  Tensor tensor = air::HybridOpNode::make(op->name + "_grad", op->tag, op->attrs, new_input, {output_grad}, {}, {}, {},
                                          {}, result_stmt)
                    .output(0);
  return tensor;
}

Tensor JacobianHybrid(const Tensor &output, const Tensor &input, const Tensor &head) {
  const auto op = output->op.as<air::HybridOpNode>();
  CHECK(op) << "A Hybrid op is expected but get : " << output->op;

  // We have to clone the iteration axes because otherwise the original expression
  // cannot be used together with the derivative (it will lead to errors during lowering)
  Array<IterVar> new_axis;
  std::unordered_map<const Variable *, Expr> vmap;
  for (IterVar iv : op->axis) {
    IterVar new_v = IterVarNode::make(iv->dom, iv->var.copy_with_suffix(""), iv->iter_type, iv->thread_tag);
    new_axis.push_back(new_v);
    vmap[iv->var.operator->()] = new_v;
  }

  Array<Expr> new_shape = output->shape;
  std::copy(input->shape.begin(), input->shape.end(), std::back_inserter(new_shape.CopyOnWrite()->data));

  // Generate new itervars for the input
  Array<Expr> input_itervars;
  Array<IterVar> jac_itervars;
  size_t i = 0;
  for (Expr ext : input->shape) {
    IterVar new_v = IterVarNode::make(Range(0, ext), Var("jac_i" + std::to_string(i)), air::IterVarType::kDataPar);
    // Append them to new_axis
    new_axis.push_back(new_v);
    // We also need a separate array of these itervars
    input_itervars.push_back(new_v);
    jac_itervars.push_back(new_v);
    ++i;
  }

  Tensor output_jac =
    PlaceholderOpNode::make(op->outputs[0]->op->name + "_" + input->op->name + "_jac", new_shape, op->outputs[0]->dtype)
      .output(0);
  std::map<FunctionRef, Tensor> tensor_mapping = {{op->outputs[0]->op, output_jac}};

  auto new_body = air::ir::Substitute(op->body, vmap);

  new_body = Jacobian(new_body, input, input_itervars, jac_itervars, tensor_mapping, op->outputs[0]);

  Tensor tensor = air::HybridOpNode::make(op->name + "_" + input->op->name + "_jac", op->tag, op->attrs, op->inputs,
                                          {output_jac}, {}, {}, {}, {}, new_body)
                    .output(0);
  ;

  // if head is not null, grad = tensordot(jac, head)
  if (head.get()) {
    tensor = HybridTensorDot(tensor, head, jac_itervars, op->outputs[0]->shape);
  }

  return tensor;
}

}  // namespace ir
}  // namespace akg