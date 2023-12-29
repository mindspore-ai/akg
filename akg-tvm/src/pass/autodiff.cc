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

Tensor HeadGenerator(const Tensor &output, const Tensor &head_or_null) {
  if (head_or_null.get()) {
    return head_or_null;
  }

  // If the head is a null pointer, create an identity tensor
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
  return air::compute(shape, func, "identity");
}

std::unordered_map<Tensor, std::vector<Tensor>> ReverseDependenciesGenerator(
  const Tensor &output, const Map<Tensor, Array<Tensor>> &override_deps) {
  // Map doesn't work correctly for Tensors, so convert it to std::unordered_map
  std::unordered_map<Tensor, Array<Tensor>> override_deps_map;
  std::unordered_map<Tensor, std::vector<Tensor>> reverse_dependencies;
  for (auto pair : override_deps) {
    override_deps_map.insert(pair);
  }

  // Collect reverse dependencies
  std::vector<Tensor> stack({output});
  while (!stack.empty()) {
    Tensor tensor = stack.back();
    tensor = SplitTensor(tensor);
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
  return reverse_dependencies;
}

DifferentiationResult Differentiate(const Tensor &output, const Array<Tensor> &inputs, const Tensor &head_or_null,
                                    const Map<std::string, NodeRef> &attrs, const Array<Tensor> &new_pld_array,
                                    const FDiffBuildingBlock &fdiff, const Map<Tensor, Array<Tensor>> &override_deps) {
  if (!output.get()) {
    LOG(FATAL) << "output is a null pointer.";
    return DifferentiationResult();
  }

  Tensor output_split = SplitTensor(output);
  Tensor head = HeadGenerator(output, head_or_null);

  // This map maps a tensor to the list of tensors immediately depending on it (using it in their
  // bodies)
  std::unordered_map<Tensor, std::vector<Tensor>> reverse_dependencies =
    ReverseDependenciesGenerator(output_split, override_deps);

  // Individual summands of the adjoints
  std::unordered_map<Tensor, Map<Tensor, Tensor>> summands;

  // This map maps tensors to the corresponding adjoints (dLoss/dTensor)
  std::unordered_map<Tensor, Tensor> adjoints;
  // head is the adjoint of output by definition
  adjoints[output_split] = head;

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
          if (res_adjoint.get() && res_adjoint->dtype != part->dtype) {
            res_adjoint = topi::cast(res_adjoint, part->dtype);
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

  bool tensor_optimize_ = (in_attrs.GetInt("tensor_optimize", 0) != 0);
  if (!tensor_optimize_) {
    return DifferentiationResultNode::make(result, adjoints, summands);
  } else {  // Running TIL optimization passes
    Array<Tensor> optimized_result;
    ADOptimizePasses(result, optimized_result, attrs, new_pld_array);
    // AD FINISHED... Returning to Poly
    return DifferentiationResultNode::make(optimized_result, adjoints, summands);
  }
}

Tensor DiffBuildingBlock(const Tensor &output, const Tensor &input, const Tensor &head,
                         const Map<std::string, NodeRef> &attrs, const Array<Tensor> &new_pld_array) {
  AttrMap in_attrs;
  if (attrs.defined()) {
    in_attrs = attrs;
  }
  bool ad_conv_enable_ = (in_attrs.GetInt("ad_conv_enable", 0) != 0);
  bool keep_dims = (in_attrs.GetInt("keep_dims", 0) != 0);

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

bool OutputsFromSameHybridOp(const Array<Tensor> &outputs, const air::HybridOpNode *&op) {
  if (static_cast<size_t>(op->num_outputs()) != outputs.size()) {
    return false;
  }

  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs[i]->op.as<air::HybridOpNode>() != op) {
      return false;
    }
    if (static_cast<size_t>(outputs[i]->value_index) != i) {
      return false;
    }
  }
  return true;
}

Array<Tensor> SingleHybridOpDifferentiate(const Array<Tensor> &outputs, const Array<Tensor> &inputs,
                                          const Array<Tensor> &heads) {
  if (inputs.empty()) {
    LOG(FATAL) << "op with no input cannot be differentiated.";
  }

  for (size_t i = 0; i < inputs.size(); i++) {
    if (!inputs[i].get()) {
      LOG(FATAL) << "inputs[" << std::to_string(i) << "] is a null pointer.";
    }
  }

  if (outputs.empty()) {
    LOG(FATAL) << "op with no output cannot be differentiated.";
  }

  for (size_t i = 0; i < outputs.size(); i++) {
    if (!outputs[i].get()) {
      LOG(FATAL) << "outputs[" << std::to_string(i) << "] is a null pointer.";
    }
  }

  if (heads.size() != outputs.size()) {
    LOG(FATAL) << "outputs' num is inconsistent with heads.";
  }

  for (size_t i = 0; i < heads.size(); i++) {
    if (!heads[i].get()) {
      LOG(FATAL) << "heads[" << std::to_string(i) << "] is a null pointer.";
    }
  }

  const air::HybridOpNode *op = outputs[0]->op.as<air::HybridOpNode>();
  if (!op || !OutputsFromSameHybridOp(outputs, op)) {
    LOG(FATAL) << "Check Failed! All outputs must come from the same Hybrid Op.";
  }

  if (outputs.size() == 1) {
    // when there is only one output, go back to Differentiate to enjoy all existing opt for autodiff
    return Differentiate(outputs[0], inputs, heads[0])->result;
  }

  auto DiffHybridOp = [&op, &heads](const Tensor &input) {
    Tensor input_grad;
    Array<Tensor> jacs = JacobianHybrid(op, input);

    for (size_t i = 0; i < jacs.size(); i++) {
      Tensor output = op->outputs[i];
      Tensor jac_output_input = jacs[i];
      Tensor head_cast =
        heads[i]->dtype == jac_output_input->dtype ? heads[i] : topi::cast(heads[i], jac_output_input->dtype);
      Tensor part = TensorDot(
        head_cast, jac_output_input, static_cast<int>(output->shape.size()),
        output->op->name + "_" + std::to_string(output->value_index) + "_" + input->op->name + "_grad", false);
      if (input_grad.get()) {
        // when input_grad is defined, cast it to the dtype of newly computed part and add this part to it
        if (input_grad->dtype != part->dtype) {
          input_grad = topi::cast(input_grad, part->dtype);
        }
        input_grad = topi::add(input_grad, part);
      } else {
        // when input_grad is undefined, assign newly computed part to it
        input_grad = part;
      }
    }

    return input_grad;
  };

  // Adjoints corresponding to inputs
  Array<Tensor> result;

  // Compute an adjoint for each input
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(result.CopyOnWrite()->data), DiffHybridOp);

  return result;
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

TVM_REGISTER_API("akg.autodiff.SingleHybridOpDifferentiate").set_body([](const TVMArgs args, TVMRetValue *ret) {
  if (args.size() == 3) {
    *ret = SingleHybridOpDifferentiate(args[0], args[1], args[2]);
  } else {
    LOG(FATAL) << "arg num must be 3, but given " << args.size();
  }
});
}  // namespace ir
}  // namespace akg
