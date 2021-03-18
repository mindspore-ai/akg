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

#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <arithmetic/const_fold.h>
#include <op/op_util.h>
#include "pass/autodiff_cce.h"
#include "pass/zero_elimination.h"

namespace akg {
namespace ir {
/*!
 * \brief Create mmad computation node
 *
 * \param source: input expr
 * \param rdom: space of iter vars1
 *
 * \return created mmad node
 */
Expr Mmad(Expr source, const Array<IterVar> &rdom) {
  Var x("x", source.type()), y("y", source.type());
  Expr result = Call::make(source.type(), "mad", {x, y}, Call::PureIntrinsic);
  Expr identity_element = make_zero(source.type());
  CommReducer combiner = CommReducerNode::make({x}, {y}, {result}, {identity_element});
  return Reduce::make(combiner, {source}, rdom, make_const(Bool(1), true), 0);
}

/*!
 * \brief The function check convolution type of a given tensor
 *
 * \param tensor: the tensor to be checked
 *
 * \return The enum value of convolution type
 */
ADConvType ConvType(const Tensor &tensor) {
  auto op = tensor->op.as<ComputeOpNode>();
  CHECK(op);
  CHECK(op->InputTensors().size() >= 2) << "Convolution has at least 2 inputs";
  const Tensor &op_data = op->InputTensors()[0];
  const Tensor &op_weight = op->InputTensors()[1];

  const auto k_h = op->attrs["pragma_conv_kernel_h"].as<IntImm>()->value;
  const auto k_w = op->attrs["pragma_conv_kernel_w"].as<IntImm>()->value;
  const auto hw = k_h * k_w;

  CHECK_NE(hw, 0);
  const auto data_Cin = op_data->shape[1].as<IntImm>()->value * op_data->shape[4].as<IntImm>()->value;
  const auto weight_Cin = op_weight->shape[0].as<IntImm>()->value * op_weight->shape[3].as<IntImm>()->value / hw;
  const auto weight_Cout = op_weight->shape[1].as<IntImm>()->value * op_weight->shape[2].as<IntImm>()->value / hw;
  const auto output_C = tensor->shape[1].as<IntImm>()->value * tensor->shape[4].as<IntImm>()->value;
  if (output_C == weight_Cin * weight_Cout) {
    return ADConvType::DEPTHWISE;
  }
  if (weight_Cin < data_Cin) {
    return ADConvType::GROUP;
  }
  if (weight_Cin == data_Cin) {
    return ADConvType::NORMAL;
  }
  return ADConvType::UNKNOWN;
}

/*!
 * \brief The function check if a given tensor is a Conv op or not. If it's a Conv op, then it check in the given name
 * matches the name of data input or the name of weight input.
 *
 * \param tensor: the tensor to be checked if it is a Conv op
 * \param is_name_data: set to True if tensor is a Conv op AND name is the name of data input of the Conv
 * \param is_name_weight: set to True if tensor is a Conv op AND name is the name of the weight (filter/kernel) input of
 * the Conv
 * \param name: the name to be checked if matches the data or weight input name of a Conv op
 *
 * \return The logic value false if tensor is not a Conv op (in this case both is_name_data and is_name_weight are false
 * too) The logic value true if tensor is a Conv op
 */
bool IsReduceConv(const Tensor &tensor, bool &is_name_data, bool &is_name_weight, const std::string &name) {
  const ComputeOpNode *op = nullptr;
  const Reduce *reduce_op = nullptr;
  const Cast *cast_op = nullptr;

  is_name_data = false;
  is_name_weight = false;

  if ((op = tensor->op.as<ComputeOpNode>()) && (op->InputTensors().size() == 2) && (!op->body.empty()) &&
      (reduce_op = op->body[0].as<Reduce>())) {
    const Mul *mul_op{nullptr};
    if ((!reduce_op->source.empty()) && reduce_op->source[0].as<Mul>()) {
      mul_op = reduce_op->source[0].as<Mul>();
    } else {
      if ((!reduce_op->source.empty()) && (cast_op = reduce_op->source[0].as<Cast>()) == nullptr) {
        return false;
      }
      if ((mul_op = cast_op->value.as<Mul>()) == nullptr) {
        return false;
      }
    }

    if ((mul_op != nullptr) && (mul_op->a.as<Call>()) && (mul_op->a.as<Call>()->args.size() == 3) &&
        (mul_op->b.as<Call>())) {
      // Can add more attributes for checking, these attrs are used directly in AD
      if ((op->attrs.count("pragma_conv_fm_n") == 0) || (op->attrs.count("pragma_conv_fm_c") == 0) ||
          (op->attrs.count("pragma_conv_fm_h") == 0) || (op->attrs.count("pragma_conv_fm_w") == 0) ||
          (op->attrs.count("pragma_conv_h_cut") == 0) || (op->attrs.count("pragma_conv_w_cut") == 0) ||
          (op->attrs.count("pragma_conv_co_cut") == 0) || (op->attrs.count("pragma_conv_m_cut") == 0) ||
          (op->attrs.count("pragma_conv_k_cut") == 0) || (op->attrs.count("pragma_conv_n_cut") == 0)) {
        return false;
      }
      const Tensor &op_data = op->InputTensors()[0];
      const Tensor &op_weight = op->InputTensors()[1];
      if (name != "") {
        CHECK(mul_op->a.as<Call>()->args[2].as<Call>());
        if ((name == mul_op->a.as<Call>()->args[2].as<Call>()->name) && (op_data->op.defined()) &&
            (name == op_data->op->name)) {
          is_name_data = true;
        } else {
          if ((name == mul_op->b.as<Call>()->name) && (op_weight->op.defined()) && (name == op_weight->op->name)) {
            is_name_weight = true;
          }
        }
      }
      return true;
    }
  }
  return false;
}

/*!
 * \brief Creates the backward computation of a Convolution wrt [Data OR Weight].
 *
 *  This function evaluates the value of an expression \p expr with the boundaries
 *  of the Range of its (single) Variable. If the expression is True for both cases,
 *  this function returns True.
 *
 *  If \p expr has more than one Variable, this function return false.
 *
 * \param output The Tensor compute of the requested Backward kernel.
 * \param input The Tensor of the Data input to the Convolution.
 * \param head The Tensor of the Input gradient (Gradient of the Loss wrt output of the Convolution).
 * \param attrs The additional set of attributes used for the AD algorithms
 * \param new_pld_array Array of Tensors for the placeholders.
 *
 * \return A Tensor compute with the backward kernel if it could be generated, an empty Tensor otherwise.
 */
Tensor DiffConv(const Tensor &output, const Tensor &input, const Tensor &head, const Map<std::string, NodeRef> &attrs,
                const Array<Tensor> &new_pld_array) {
  Tensor result;
  bool is_name_data = false, is_name_weight = false;

  if (IsReduceConv(output, is_name_data, is_name_weight, input->op->name)) {
    // Set to work with Depthwise
    const auto Tile0 = output->op->attrs["pragma_conv_h_cut"].as<IntImm>()->value;
    const auto Tile1 = output->op->attrs["pragma_conv_co_cut"].as<IntImm>()->value;
    const auto Tile2 = output->op->attrs["pragma_conv_m_cut"].as<IntImm>()->value;
    const auto Tile3 = output->op->attrs["pragma_conv_k_cut"].as<IntImm>()->value;
    const auto Tile4 = output->op->attrs["pragma_conv_n_cut"].as<IntImm>()->value;

    CHECK_GE(head->shape.size(), 5);
    const auto head_n = head->shape[0].as<IntImm>()->value;
    const auto head_c = head->shape[1].as<IntImm>()->value * head->shape[4].as<IntImm>()->value;
    const auto head_h = head->shape[2].as<IntImm>()->value;
    const auto head_w = head->shape[3].as<IntImm>()->value;

    const auto s_h = output->op->attrs["pragma_conv_stride_h"].as<IntImm>()->value;
    const auto s_w = output->op->attrs["pragma_conv_stride_w"].as<IntImm>()->value;
    const auto d_h = output->op->attrs["pragma_conv_dilation_h"].as<IntImm>()->value;
    const auto d_w = output->op->attrs["pragma_conv_dilation_w"].as<IntImm>()->value;

    const auto p_top = output->op->attrs["pragma_conv_padding_top"].as<IntImm>()->value;
    const auto p_bottom = output->op->attrs["pragma_conv_padding_bottom"].as<IntImm>()->value;
    const auto p_left = output->op->attrs["pragma_conv_padding_left"].as<IntImm>()->value;
    const auto p_right = output->op->attrs["pragma_conv_padding_right"].as<IntImm>()->value;

    const auto k_n = output->op->attrs["pragma_conv_kernel_n"].as<IntImm>()->value;
    // k_c = in_c
    const auto k_c = output->op->attrs["pragma_conv_fm_c"].as<IntImm>()->value;
    const auto k_h = output->op->attrs["pragma_conv_kernel_h"].as<IntImm>()->value;
    const auto k_w = output->op->attrs["pragma_conv_kernel_w"].as<IntImm>()->value;

    ADConvType conv_type = ConvType(output);

    CHECK_GT(conv_type, ADConvType::UNKNOWN) << "Unsupported type of Convolution";
    CHECK_NE(conv_type, ADConvType::DEPTHWISE) << "Depthwise AD not supported yet!";

    const auto in_n = output->op->attrs["pragma_conv_fm_n"].as<IntImm>()->value;
    const auto in_c = output->op->attrs["pragma_conv_fm_c"].as<IntImm>()->value;
    const auto in_h = output->op->attrs["pragma_conv_fm_h"].as<IntImm>()->value;
    const auto in_w = output->op->attrs["pragma_conv_fm_w"].as<IntImm>()->value;

    AttrMap in_attrs;
    if (attrs.defined()) {
      in_attrs = attrs;
    }

    if (is_name_data) {
      Array<Expr> NCHW_strided_head_shape = {Expr(head_n), Expr(head_c), Expr((head_h - 1) * s_h + 1),
                                             Expr((head_w - 1) * s_w + 1)};
      Array<Expr> NCHW_flipped_weight_shape = {Expr(k_c), Expr(k_n), Expr(k_h), Expr(k_w)};

      if (conv_type == ADConvType::NORMAL) {
        if (in_attrs.GetIntAttr("ad_conv_reuse_conv", 0) != 0) {
          Array<Integer> Stride = {1, 1};
          Array<Integer> Dilation = {1, 1};
          Array<Integer> Pad = {Integer(k_h - 1 - p_top), Integer(k_h - 1 - p_bottom), Integer(k_w - 1 - p_left),
                                Integer(k_w - 1 - p_right)};
          const PackedFunc *f = air::runtime::Registry::Get("akg.autodiff.conv_compute_forward");
          CHECK(f);
          CHECK_GE(new_pld_array.size(), 3);
          result = (*f)(NCHW_strided_head_shape, NCHW_flipped_weight_shape, Pad, Stride, Dilation, new_pld_array[0],
                        new_pld_array[1], new_pld_array[2], Tile0, Tile1, Tile2, Tile3, Tile4, true);
          return result;
        } else {
          Array<Integer> Stride = {Integer(s_h), Integer(s_w)};
          Array<Integer> Dilation = {Integer(d_h), Integer(d_w)};
          Array<Integer> Pad = {Integer(p_top), Integer(p_bottom), Integer(p_left), Integer(p_right)};
          const PackedFunc *f_input_ad = air::runtime::Registry::Get("akg.autodiff.conv_input_ad_tensor");
          CHECK(f_input_ad);
          Array<Tensor> dx_data;
          // new_pld_array contains 3 tensors: Head, rounded_forward_weight (in fractal) and original_forward_weight (in
          // NCHW)
          CHECK_GE(new_pld_array.size(), 3);
          // new_pld_array [2] contains original_forward_weight in 4D NCHW
          CHECK_GE(new_pld_array[2]->shape.size(), 4);

          dx_data.push_back(new_pld_array[0]);
          dx_data.push_back(new_pld_array[1]);

          const auto orig_Cout = new_pld_array[2]->shape[0].as<IntImm>()->value;
          const auto orig_Cin = new_pld_array[2]->shape[1].as<IntImm>()->value;
          Array<Expr> NCHW_data_shape = {Expr(in_n), Expr(orig_Cin), Expr(in_h), Expr(in_w)};
          Array<Expr> NCHW_weight_shape = {Expr(orig_Cout), Expr(orig_Cin), Expr(k_h), Expr(k_w)};

          result = (*f_input_ad)(dx_data, NCHW_data_shape, NCHW_weight_shape, Pad, Stride, Dilation);
          return result;
        }
      }
      if (conv_type == ADConvType::GROUP) {
        const int block_size = 16;
        const auto op = output->op.as<ComputeOpNode>();
        CHECK(op);
        const Tensor &op_weight = op->InputTensors()[1];
        const auto hw = k_h * k_w;
        CHECK_NE(hw, 0);
        CHECK_GE(op_weight->shape.size(), 4);
        const auto weight_Cin = op_weight->shape[0].as<IntImm>()->value * op_weight->shape[3].as<IntImm>()->value / hw;
        CHECK_NE(weight_Cin, 0);
        const auto group = k_c / weight_Cin;
        const auto in_h_ = output->op->attrs["pragma_conv_fm_h"].as<IntImm>()->value;
        const auto in_w_ = output->op->attrs["pragma_conv_fm_w"].as<IntImm>()->value;

        const auto cutH_e = (head_h - 1) * s_h + 1;
        int cutCo_e = block_size;
        const auto cutM_e = ((in_h_ * in_w_ + block_size - 1) / block_size) * block_size;
        int cutK_e = AD_GROUP_CONV_DEFAULT_cutK, cutN_e = block_size;

        const PackedFunc *f_group = air::runtime::Registry::Get("akg.autodiff.group_conv_compute_forward");
        CHECK(f_group);
        CHECK_GE(new_pld_array.size(), 3);
        result = (*f_group)(head_n, (head_h - 1) * s_h + 1, (head_w - 1) * s_w + 1, k_n, k_c, group, k_h, k_w,
                            new_pld_array[0], new_pld_array[1], new_pld_array[2], k_h - 1 - p_top, k_w - 1 - p_left, 1,
                            1, cutH_e, cutCo_e, cutM_e, cutK_e, cutN_e, block_size);
        return result;
      }
    }
    if (is_name_weight) {
      Array<Expr> NCHW_trans_data_shape = {Expr(in_c), Expr(in_n), Expr(in_h), Expr(in_w)};
      Array<Expr> NCHW_trans_head_shape = {Expr(head_c), Expr(head_n), Expr(head_h), Expr(head_w)};
      Array<Integer> Pad = {Integer(p_top), Integer(p_bottom), Integer(p_left), Integer(p_right)};
      if (conv_type == ADConvType::NORMAL) {
        if (in_attrs.GetIntAttr("ad_conv_reuse_conv", 0) != 0) {
          Array<Integer> Stride = {1, 1};
          Array<Integer> Dilation = {Integer(s_h), Integer(s_w)};
          const PackedFunc *f = air::runtime::Registry::Get("akg.autodiff.conv_compute_forward");
          CHECK(f);
          CHECK_GE(new_pld_array.size(), 3);
          result = (*f)(NCHW_trans_data_shape, NCHW_trans_head_shape, Pad, Stride, Dilation, new_pld_array[0],
                        new_pld_array[1], new_pld_array[2], Tile0, Tile1, Tile2, Tile3, Tile4, true);
          return result;
        } else {
          Array<Integer> Stride = {Integer(s_h), Integer(s_w)};
          Array<Integer> Dilation = {Integer(d_h), Integer(d_w)};
          const PackedFunc *f_filter_ad = air::runtime::Registry::Get("akg.autodiff.conv_filter_ad_tensor");
          CHECK(f_filter_ad);
          Array<Tensor> dw_data;

          // new_pld_array contains 3 tensors: Head, rounded_forward_input (in 5D) and original_forward_weight (in NCHW)
          CHECK_GE(new_pld_array.size(), 3);
          // new_pld_array [2] contains original_forward_weight in 4D NCHW
          CHECK_GE(new_pld_array[2]->shape.size(), 4);

          dw_data.push_back(new_pld_array[0]);
          dw_data.push_back(new_pld_array[1]);

          const auto orig_Cout = new_pld_array[2]->shape[0].as<IntImm>()->value;
          const auto orig_Cin = new_pld_array[2]->shape[1].as<IntImm>()->value;

          Array<Expr> NCHW_data_shape = {Expr(in_n), Expr(orig_Cin), Expr(in_h), Expr(in_w)};
          Array<Expr> NCHW_weight_shape = {Expr(orig_Cout), Expr(orig_Cin), Expr(k_h), Expr(k_w)};

          result = (*f_filter_ad)(dw_data, NCHW_data_shape, NCHW_weight_shape, Pad, Stride, Dilation);
          return result;
        }
      }
      if (conv_type == ADConvType::GROUP) {
        const int block_size = 16;
        const auto op = output->op.as<ComputeOpNode>();
        CHECK(op);
        const Tensor &op_weight = op->InputTensors()[1];
        const auto hw = k_h * k_w;
        CHECK_NE(hw, 0);
        CHECK_GE(op_weight->shape.size(), 4);
        const auto weight_Cin = op_weight->shape[0].as<IntImm>()->value * op_weight->shape[3].as<IntImm>()->value / hw;
        CHECK_NE(weight_Cin, 0);
        const auto group = k_c / weight_Cin;

        const auto cutH_e = in_h;
        const auto cutCo_e = block_size;
        const auto cutM_e = ((hw + block_size - 1) / block_size) * block_size;
        int cutK_e = AD_GROUP_CONV_DEFAULT_cutK, cutN_e = block_size;
        const PackedFunc *f_group = air::runtime::Registry::Get("akg.autodiff.group_conv_compute_forward");
        CHECK(f_group);
        CHECK_GE(new_pld_array.size(), 3);
        result = (*f_group)(weight_Cin, in_h, in_w, in_n * group, head_c, group, head_h, head_w, new_pld_array[0],
                            new_pld_array[1], new_pld_array[2], p_top, p_left, 1, 1, cutH_e, cutCo_e, cutM_e, cutK_e,
                            cutN_e, block_size);
        return result;
      }
    }
  }
  return result;
}

std::unordered_set<const Variable *> AutodiffSimplify::GetExprVars(const Expr &expr) const {
  std::unordered_set<const Variable *> vars;
  PostOrderVisit(expr, [&vars](const NodeRef &node) {
    if (const auto v = node.as<Variable>()) {
      vars.insert(v);
    }
  });
  return vars;
}

template <typename T1>
Expr AutodiffSimplify::SimplifyMaxMin(Expr lhs, Expr rhs, const Expr &e) {
  if (auto selLhs = lhs.as<Select>()) {
    if (auto selRhs = rhs.as<Select>()) {
      if (Equal(selLhs->condition, selRhs->condition)) {
        return Select::make(selLhs->condition, T1::make(Mutate(selLhs->true_value), Mutate(selRhs->true_value)),
                            T1::make(Mutate(selLhs->false_value), Mutate(selRhs->false_value)));
      }
    }
  }

  auto newLhs = Mutate(lhs);
  auto newRhs = Mutate(rhs);
  if (lhs.same_as(newLhs) && rhs.same_as(newRhs)) {
    return e;
  }

  return T1::make(newLhs, newRhs);
}

/*!
 * \brief Finds the range of a Var in Map given its name.
 *  This function loops over a Map of <Var, Range> looking
 *  for a Var that is named var_name.
 *  If no Range is found for Var in the Map an empty Range is returned.
 *
 * \param var_name The name of the Var to find in cond_ranges_
 * \return The Range of the Var named var_name
 */
Range AutodiffSimplify::GetRangeOfVar(const std::string &var_name) {
  auto it = std::find_if(cond_ranges_.begin(), cond_ranges_.end(), [&var_name](const std::pair<Var, Range> &var_range) {
    return var_range.first.get()->name_hint == var_name;
  });
  if (it != cond_ranges_.end()) {
    return (*it).second;
  }

  return Range();
}

Expr AutodiffSimplify::Mutate_(const Mod *op, const Expr &e) {
  const auto imm = op->b.as<IntImm>();
  if (imm && imm->value < 0) {
    auto mod = Mutate(op->a) % Mutate(-(op->b));
    return mod;
  }

  return e;
}

Expr AutodiffSimplify::Mutate_(const EQ *op, const Expr &e) {
  if (air::arith::IsNumericType(op->a.type()) && air::arith::IsNumericType((op->b.type()))) {
    return op->a >= op->b && op->a <= op->b;
  }

  return e;
}

Expr SimplifyMad::Mutate_(const Call *op, const Expr &e) {
  if (op->name == "mad") {
    CHECK_GE(op->args.size(), 2);
    return op->args[0] + op->args[1];
  }
  return e;
}

/*!
 * \brief check if has mad Call
 *
 * \param output input tensor
 * \return true if input tensor is mad call, otherwise returns false.
 */
bool HasMad(const Tensor &output) {
  if (const auto op = output->op.as<ComputeOpNode>()) {
    bool found = false;
    auto CheckMad = [&found](const NodeRef &op) {
      const auto v = op.as<Call>();
      if (v && v->name == "mad") {
        found = true;
      }
    };
    for (auto i : op->body) {
      if (const auto r = i.as<Reduce>()) {
        for (auto j : r->combiner->result) {
          air::ir::PostOrderVisit(j, CheckMad);
          if (found) return true;
        }
      }
    }
  }
  return false;
}

/*!
 * \brief A generalization of matrix multiplication to tensors.
 *
 * \param A The tensor A
 * \param B The tensor B
 * \param axes The number of the dimensions to reduce over
 * \param name The name of the operation
 * \param has_mad The flag to mark has_mad operation
 *
 * \return A Tensor computing the result
 */
Tensor TensorDot(const Tensor &A, const Tensor &B, int axes = 2, std::string name = "T_tensordot",
                 bool has_mad = false) {
  CHECK_GE(A->shape.size(), axes);
  CHECK_GE(B->shape.size(), axes);

  Array<Expr> output_shape(A->shape.begin(), A->shape.end() + (-axes));
  for (auto it = B->shape.begin() + axes; it != B->shape.end(); ++it) output_shape.push_back(*it);

  Array<IterVar> iter_vars;
  for (int i = 0; i < axes; ++i) iter_vars.push_back(reduce_axis(Range(0, B->shape[i]), "k" + std::to_string(i)));

  auto func = [&A, &B, &iter_vars, axes, has_mad](const Array<Var> &input_indices) {
    Array<Expr> A_indices(input_indices.begin(), input_indices.begin() + (static_cast<int>(A->shape.size()) - axes));

    std::transform(iter_vars.begin(), iter_vars.end(), std::back_inserter(A_indices.CopyOnWrite()->data),
                   [](const IterVar &v) { return static_cast<Expr>(v); });

    Array<Expr> B_indices;
    std::transform(iter_vars.begin(), iter_vars.end(), std::back_inserter(B_indices.CopyOnWrite()->data),
                   [](const IterVar &v) { return static_cast<Expr>(v); });

    auto it = input_indices.begin() + (static_cast<int>(A->shape.size()) - axes);
    for (; it != input_indices.end(); ++it) B_indices.push_back(*it);

    // Some passes don't like reductions with empty axis, so avoid it here
    if (iter_vars.empty())
      return A(A_indices) * B(B_indices);
    else
      return has_mad ? Mmad(A(A_indices) * B(B_indices), iter_vars) : air::sum(A(A_indices) * B(B_indices), iter_vars);
  };

  return compute(output_shape, func, name, topi::kMatMul);
}

// Functions for optimization passes for AD

bool CompareTensors(const Tensor &left_tensor, const Tensor &right_tensor,
                    const std::unordered_set<Tensor> &equal_set) {
  if ((equal_set.find(left_tensor) != equal_set.end()) && (equal_set.find(right_tensor) != equal_set.end())) {
    return true;
  }
  return false;
}

bool CompareTensors(const Tensor &left_tensor, const Tensor &right_tensor,
                    const std::vector<std::unordered_set<Tensor>> &all_equal_sets) {
  if (left_tensor == right_tensor) {
    return true;
  }
  for (auto it_set : all_equal_sets) {
    if (CompareTensors(left_tensor, right_tensor, it_set)) {
      return true;
    }
  }
  return false;
}

bool DeepCompareTensorsNodes(const Tensor &left_tensor, const Tensor &right_tensor) {
  if (left_tensor == right_tensor) {
    return true;
  }
  // checking dtype
  if (left_tensor->dtype != right_tensor->dtype) {
    return false;
  }
  // checking output shape
  if (left_tensor->shape.size() != right_tensor->shape.size()) {
    return false;
  }
  for (size_t i = 0; i < left_tensor->shape.size(); i++) {
    if (left_tensor->shape[i].as<IntImm>()->value != right_tensor->shape[i].as<IntImm>()->value) {
      return false;
    }
  }
  // checking value index
  if (left_tensor->value_index != right_tensor->value_index) {
    return false;
  }
  // Checking the op
  if (left_tensor->op.as<PlaceholderOpNode>() || right_tensor->op.as<PlaceholderOpNode>()) {
    // if both are Placeholder, the test should pass at the begin of the function already
    return false;
  }
  auto left_op = left_tensor->op.as<ComputeOpNode>();
  auto right_op = right_tensor->op.as<ComputeOpNode>();
  if ((left_op == nullptr) || (right_op == nullptr)) {
    return false;
  }
  // Checking the size of input_tensors
  if (left_op->InputTensors().size() != right_op->InputTensors().size()) {
    return false;
  }
  // Checking the axis
  if (left_op->axis.size() != right_op->axis.size()) {
    return false;
  }
  for (size_t i = 0; i < left_op->axis.size(); i++) {
    if (!Equal(left_op->axis[i]->dom->min, right_op->axis[i]->dom->min)) {
      return false;
    }
    if (!Equal(left_op->axis[i]->dom->extent, right_op->axis[i]->dom->extent)) {
      return false;
    }
  }
  // Checking the reduce_axis
  if (left_op->reduce_axis.size() != right_op->reduce_axis.size()) {
    return false;
  }
  for (size_t i = 0; i < left_op->reduce_axis.size(); i++) {
    if (!Equal(left_op->reduce_axis[i]->dom->min, right_op->reduce_axis[i]->dom->min)) {
      return false;
    }
    if (!Equal(left_op->reduce_axis[i]->dom->extent, right_op->reduce_axis[i]->dom->extent)) {
      return false;
    }
  }
  return true;
}

bool CheckEqualAndBothCommutative(const ComputeOpNode *left_op, const ComputeOpNode *right_op, bool &commutative) {
  commutative = false;
  if ((left_op == nullptr) || (right_op == nullptr)) {
    return false;
  }
  CHECK(!left_op->body.empty());
  CHECK(!right_op->body.empty());
  if (left_op->body[0].as<Add>() != nullptr) {
    if (right_op->body[0].as<Add>() == nullptr) {
      return false;
    }
    commutative = true;
    return true;
  }
  if (left_op->body[0].as<Mul>() != nullptr) {
    if (right_op->body[0].as<Mul>() == nullptr) {
      return false;
    }
    commutative = true;
    return true;
  }
  if (left_op->body[0].as<Sub>() != nullptr) {
    if (right_op->body[0].as<Sub>() == nullptr) {
      return false;
    }
    return true;
  }
  if (left_op->body[0].as<Div>() != nullptr) {
    if (right_op->body[0].as<Div>() == nullptr) {
      return false;
    }
  }
  // Add more Ops if needed
  return true;
}

bool CheckEqualConst(const ComputeOpNode *left_op, const ComputeOpNode *right_op) {
  if ((left_op == nullptr) || (right_op == nullptr)) {
    return false;
  }
  if ((left_op->InputTensors().size() != 1) || (right_op->InputTensors().size() != 1)) {
    return false;
  }

  Expr left_const, right_const;
  CHECK(!left_op->body.empty());
  CHECK(!right_op->body.empty());
  if (left_op->body[0]->GetTypeKey() == "Mul") {
    if (is_const_ad(left_op->body[0].as<Mul>()->a)) {
      left_const = left_op->body[0].as<Mul>()->a;
    } else {
      if (is_const_ad(left_op->body[0].as<Mul>()->b)) {
        left_const = left_op->body[0].as<Mul>()->b;
      } else {
        return false;
      }
    }
    if (is_const_ad(right_op->body[0].as<Mul>()->a)) {
      right_const = right_op->body[0].as<Mul>()->a;
    } else {
      if (is_const_ad(right_op->body[0].as<Mul>()->b)) {
        right_const = right_op->body[0].as<Mul>()->b;
      } else {
        return false;
      }
    }
  } else {
    if (left_op->body[0]->GetTypeKey() == "Add") {
      if (is_const_ad(left_op->body[0].as<Add>()->a)) {
        left_const = left_op->body[0].as<Add>()->a;
      } else {
        if (is_const_ad(left_op->body[0].as<Add>()->b)) {
          left_const = left_op->body[0].as<Add>()->b;
        } else {
          return false;
        }
      }
      if (is_const_ad(right_op->body[0].as<Add>()->a)) {
        right_const = right_op->body[0].as<Add>()->a;
      } else {
        if (is_const_ad(right_op->body[0].as<Add>()->b)) {
          right_const = right_op->body[0].as<Add>()->b;
        } else {
          return false;
        }
      }
    }
  }
  return Equal(right_const, left_const);
}

bool DeepCompareTensors(const Tensor &left_tensor, const Tensor &right_tensor) {
  if (left_tensor == right_tensor) {
    return true;
  }

  if (!DeepCompareTensorsNodes(left_tensor, right_tensor)) {
    return false;
  }

  auto left_op = left_tensor->op.as<ComputeOpNode>();
  auto right_op = right_tensor->op.as<ComputeOpNode>();

  if (left_op != nullptr && !left_op->body.empty() && right_op != nullptr && !right_op->body.empty()) {
    if (auto left_call = left_op->body[0].as<Call>()) {
      auto right_call = right_op->body[0].as<Call>();
      if (right_call == nullptr) {
        return false;
      }
      if (left_call->call_type != right_call->call_type) {
        return false;
      }
      if (left_call->call_type == Call::PureIntrinsic) {
        if (left_call->name != right_call->name) {
          return false;
        }
      }
      // Call::Halide or Call::PureIntrinsic
      for (size_t i = 0; i < left_op->InputTensors().size(); i++) {
        if (!DeepCompareTensors(left_op->InputTensors()[i], right_op->InputTensors()[i])) {
          return false;
        }
      }
      return true;
    }
  }

  bool commutative_op = false;
  if (!CheckEqualAndBothCommutative(left_op, right_op, commutative_op)) {
    return false;
  }
  if (!commutative_op) {
    for (size_t i = 0; i < left_op->InputTensors().size(); i++) {
      if (!DeepCompareTensors(left_op->InputTensors()[i], right_op->InputTensors()[i])) {
        return false;
      }
      return true;
    }
  } else {
    if (!DeepCompareTensors(left_op->InputTensors()[0], right_op->InputTensors()[0])) {
      if (right_op->InputTensors().size() < 2) {
        return false;
      }
      if (!DeepCompareTensors(left_op->InputTensors()[0], right_op->InputTensors()[1])) {
        return false;
      }
      if (!DeepCompareTensors(left_op->InputTensors()[1], right_op->InputTensors()[0])) {
        return false;
      }
      return true;
    }
    if (right_op->InputTensors().size() < 2) {
      return CheckEqualConst(left_op, right_op);
    }
    if (!DeepCompareTensors(left_op->InputTensors()[1], right_op->InputTensors()[1])) {
      return false;
    }
    return true;
  }
  return false;
}

bool ShallowCompareTensors(const Tensor &left_tensor, const Tensor &right_tensor,
                           std::vector<std::unordered_set<Tensor>> &equal_tensors_at_distance) {
  if (left_tensor == right_tensor) {
    return true;
  }
  if (!DeepCompareTensorsNodes(left_tensor, right_tensor)) {
    return false;
  }
  auto left_op = left_tensor->op.as<ComputeOpNode>();
  auto right_op = right_tensor->op.as<ComputeOpNode>();

  if (left_op != nullptr && !left_op->body.empty() && right_op != nullptr && !right_op->body.empty()) {
    if (auto left_call = left_op->body[0].as<Call>()) {
      auto right_call = right_op->body[0].as<Call>();
      if (right_call == nullptr) {
        return false;
      }
      if (left_call->call_type != right_call->call_type) {
        return false;
      }
      if (left_call->call_type == Call::PureIntrinsic) {
        if (left_call->name != right_call->name) {
          return false;
        }
        if (left_op->InputTensors()[0] == right_op->InputTensors()[0]) {
          return true;
        }
      }
      // Call::Halide or Call::PureIntrinsic
      for (size_t i = 0; i < left_op->InputTensors().size(); i++) {
        if (!CompareTensors(left_op->InputTensors()[i], right_op->InputTensors()[i], equal_tensors_at_distance)) {
          return false;
        }
      }
      return true;
    }
  }

  bool commutative_op = false;
  if (!CheckEqualAndBothCommutative(left_op, right_op, commutative_op)) {
    return false;
  }
  if (!commutative_op) {
    for (size_t i = 0; i < left_op->InputTensors().size(); i++) {
      if (!CompareTensors(left_op->InputTensors()[i], right_op->InputTensors()[i], equal_tensors_at_distance)) {
        return false;
      }
      return true;
    }
  } else {
    if (!CompareTensors(left_op->InputTensors()[0], right_op->InputTensors()[0], equal_tensors_at_distance)) {
      if (right_op->InputTensors().size() < 2) {
        return false;
      }
      if (!CompareTensors(left_op->InputTensors()[0], right_op->InputTensors()[1], equal_tensors_at_distance)) {
        return false;
      }
      if (!CompareTensors(left_op->InputTensors()[1], right_op->InputTensors()[0], equal_tensors_at_distance)) {
        return false;
      }
      return true;
    }
    if (right_op->InputTensors().size() < 2) {
      return CheckEqualConst(left_op, right_op);
    }
    if (!CompareTensors(left_op->InputTensors()[1], right_op->InputTensors()[1], equal_tensors_at_distance)) {
      return false;
    }
    return true;
  }
  return false;
}

void CollectAllTensorsByDistance(const Tensor &tensor, std::unordered_set<int> &distances,
                                 std::unordered_map<int, std::unordered_set<Tensor>> &all_tensors_at_distance) {
  for (auto inp : tensor->op->InputTensors()) {
    std::unordered_set<int> child_distances;
    CollectAllTensorsByDistance(inp, child_distances, all_tensors_at_distance);
    for (auto it : child_distances) {
      distances.insert(it + 1);
      all_tensors_at_distance[it + 1].insert(tensor);
    }
  }
  if (tensor->op.as<PlaceholderOpNode>() != nullptr) {
    (void)all_tensors_at_distance[0].emplace(tensor);
    (void)distances.emplace(0);
  }
  return;
}

void FindEqualTensorsByDistance(const std::unordered_map<int, std::unordered_set<Tensor>> &all_tensors_at_distance,
                                std::unordered_map<int, std::vector<std::unordered_set<Tensor>>> &equal_tensors) {
  size_t max_distance = all_tensors_at_distance.size();
  for (size_t distance = 1; distance < max_distance; distance++) {
    // Find all sets of equal tensors at distance "distance" to placeholders
    // At distance = 0, there are placeholders only
    std::vector<Tensor> tmp_tensors_at_distance;
    std::vector<int> assigned;
    if (all_tensors_at_distance.find(distance) == all_tensors_at_distance.end()) {
      continue;
    }
    for (auto it : all_tensors_at_distance.at(distance)) {
      tmp_tensors_at_distance.push_back(it);
      assigned.push_back(0);
    }
    if (tmp_tensors_at_distance.empty()) {
      continue;
    }

    for (size_t i = 0; i < tmp_tensors_at_distance.size() - 1; i++) {
      if (assigned[i] == 1) {
        continue;
      }
      std::unordered_set<Tensor> tmp_equal_tensors;
      size_t t_distance = distance;
      for (size_t j = i + 1; j < tmp_tensors_at_distance.size(); j++) {
        if (assigned[j] == 1) {
          continue;
        }
        // Comparing tmp_tensors_at_distance[i] with tmp_tensors_at_distance[j]
        if (ShallowCompareTensors(tmp_tensors_at_distance[i], tmp_tensors_at_distance[j],
                                  equal_tensors[t_distance - 1])) {
          if (assigned[i] == 0) {
            tmp_equal_tensors.insert(tmp_tensors_at_distance[i]);
            assigned[i] = 1;
          }
          tmp_equal_tensors.insert(tmp_tensors_at_distance[j]);
          assigned[j] = 1;
        }
      }
      if (tmp_equal_tensors.size() > 0) {
        equal_tensors[t_distance].push_back(tmp_equal_tensors);
        tmp_equal_tensors.clear();
      }
    }
  }
}

// The replacer of Operations: replace all operations in an Expr according to a given Map
class OperationReplacer : public IRMutator {
 public:
  explicit OperationReplacer(const std::unordered_map<Operation, Operation> &replace_operations_map)
      : replace_operations_map_(replace_operations_map) {}
  ~OperationReplacer() override = default;

  Expr Mutate_(const Call *op, const Expr &e) override {
    if (op->call_type == Call::Halide) {
      Operation t_op = Downcast<Operation>(op->func).output(op->value_index)->op;
      auto it = replace_operations_map_.find(t_op);
      if (it != replace_operations_map_.end()) {
        Expr ret = Call::make(op->type, it->second->name, op->args, op->call_type, it->second, op->value_index);
        found = true;
        CHECK(ret.as<Call>());
        return IRMutator::Mutate_(ret.as<Call>(), ret);
      }
    }
    return IRMutator::Mutate_(op, e);
  }

  bool HasFound() const { return found; }

 private:
  bool found{false};
  const std::unordered_map<Operation, Operation> &replace_operations_map_;
};

Expr ReplaceOperation(const Expr &expr, const std::unordered_map<Operation, Operation> &replace) {
  OperationReplacer repl(replace);
  Expr ret = repl.Mutate(expr);
  return repl.HasFound() ? ret : expr;
}

Operation ReplaceInputs(const Operation &self, const std::unordered_map<Operation, Operation> &rmap) {
  const auto _this = self.as<ComputeOpNode>();
  if (_this == nullptr) {
    return self;
  }
  Array<Expr> arr;
  arr = air::ir::UpdateArray(_this->body, [&rmap](const Expr &e) { return ReplaceOperation(e, rmap); });
  if (!arr.same_as(_this->body)) {
    return ComputeOpNode::make(_this->name, _this->tag, _this->attrs, _this->axis, arr);
  } else {
    return self;
  }
}

Operation ReplaceTensorRecursively(const Tensor &tensor, std::unordered_map<Operation, Operation> &replace_map,
                                   std::unordered_set<Tensor> &visited,
                                   std::unordered_set<Operation> &new_operators_marked) {
  bool all_child_updated = true;
  for (auto inp : tensor->op->InputTensors()) {
    if (visited.find(inp) != visited.end()) {
      continue;
    }
    if ((replace_map.find(inp->op) == replace_map.end()) || (replace_map[inp->op] == inp->op)) {
      auto replaced_op = ReplaceTensorRecursively(inp, replace_map, visited, new_operators_marked);
      if (visited.find(inp) != visited.end()) {
        if (!replaced_op.same_as(inp->op)) {
          // The operation was updated for actual input inp
          replace_map[inp->op] = replaced_op;
        }
      } else {
        all_child_updated = false;
      }
    } else {
      bool dest_visited = false;
      auto dest_operation = replace_map[inp->op];
      if (new_operators_marked.find(dest_operation) != new_operators_marked.end()) {
        dest_visited = true;
      } else {
        for (auto it_tensor : visited) {
          if (it_tensor->op == dest_operation) {
            dest_visited = true;
            if (replace_map.find(dest_operation) != replace_map.end()) {
              // destination operation was already visited and updated
              replace_map[inp->op] = replace_map[dest_operation];
              new_operators_marked.insert(replace_map[dest_operation]);
            } else {
              replace_map[inp->op] = dest_operation;
            }
          }
        }
      }
      if (!dest_visited) {
        all_child_updated = false;
      }
    }
  }

  if (!all_child_updated) {
    return tensor->op;
  }

  // After all children visited, update the tensor
  auto new_op = ReplaceInputs(tensor->op, replace_map);
  visited.insert(tensor);
  if (!tensor->op.same_as(new_op)) {
    return new_op;
  } else {
    return tensor->op;
  }
}

Tensor ADPassReplaceTensorsUsingOperations(const Tensor &root, std::unordered_map<Tensor, Tensor> &replace_map,
                                           bool internal_replace) {
  std::unordered_map<Operation, Operation> operation_replace_map;
  for (auto it : replace_map) {
    operation_replace_map[it.first->op] = it.second->op;
  }

  std::unordered_set<Operation> new_operators_marked;
  std::unordered_set<Tensor> visited;
  // When replacing tensors with external tensors, these external tensors are already "visited",
  // marking them to be ready to use, when replace tensors with internal tensors, they have to wait
  // for "deepest" to be replaced first to avoid duplications of tensors
  if (!internal_replace) {
    for (auto it : replace_map) {
      visited.insert(it.second);
      if (it.second->op->InputTensors().size() > 0) {
        // Cast case: it.second = Cast (pld)
        visited.insert(it.second->op->InputTensors()[0]);
      }
      operation_replace_map[it.second->op] = it.second->op;
    }
  }

  Operation replaced_op;
  while (visited.find(root) == visited.end()) {
    replaced_op = ReplaceTensorRecursively(root, operation_replace_map, visited, new_operators_marked);
  }
  if (!root->op.same_as(replaced_op)) {
    auto new_tensor = TensorNode::make(root->shape, root->dtype, replaced_op, root->value_index);
    return new_tensor;
  }
  return root;
}

void CollectReverseDependencies(const Array<Tensor> &input_tensors,
                                std::unordered_map<Tensor, std::unordered_set<Tensor>> &reverse_dependencies) {
  std::vector<Tensor> stack;
  for (auto it : input_tensors) {
    stack.push_back(it);
  }
  while (!stack.empty()) {
    Tensor tensor = stack.back();
    stack.pop_back();
    for (const Tensor &child : tensor->op->InputTensors()) {
      if (!reverse_dependencies.count(child)) {
        stack.push_back(child);
      }
      (void)reverse_dependencies[child].emplace(tensor);
    }
  }
  return;
}

void ADPassReplaceArrayTensorsUsingOperations(Array<Tensor> &input_tensors, Array<Tensor> &output_tensors,
                                              std::unordered_map<Tensor, Tensor> &replace_map) {
  if (replace_map.size() == 0) {
    output_tensors.assign(input_tensors.begin(), input_tensors.end());
    return;
  }
  std::unordered_map<Tensor, std::unordered_set<Tensor>> reverse_dependencies;
  CollectReverseDependencies(input_tensors, reverse_dependencies);
  std::unordered_map<Tensor, int> dfs_order;
  for (auto it : input_tensors) {
    CollectDFSOrder(it, dfs_order);
  }
  // Defining a lambda function to compare two tensors according to orders from dfs_order
  std::function<bool(const Tensor &, const Tensor &)> comp_func = [&dfs_order](const Tensor &t1, const Tensor &t2) {
    return dfs_order[t1] < dfs_order[t2];
  };
  std::set<Tensor, std::function<bool(Tensor, Tensor)>> ordered_queue(comp_func);
  std::unordered_map<Operation, Operation> operation_replace_map;
  for (auto it : replace_map) {
    operation_replace_map[it.first->op] = it.second->op;
    (void)ordered_queue.emplace(it.first);
  }
  while (ordered_queue.size() >= 1) {
    auto it_first = ordered_queue.begin();
    Tensor act_tensor = *it_first;
    (void)ordered_queue.erase(it_first);
    ordered_queue.insert(reverse_dependencies[act_tensor].begin(), reverse_dependencies[act_tensor].end());
    if (operation_replace_map.find(act_tensor->op) != operation_replace_map.end()) {
      continue;
    }
    auto new_op = ReplaceInputs(act_tensor->op, operation_replace_map);
    for (auto it_replace : operation_replace_map) {
      if (it_replace.second == act_tensor->op) {
        operation_replace_map[it_replace.first] = new_op;
      }
    }
    if (!act_tensor->op.same_as(new_op)) {
      operation_replace_map[act_tensor->op] = new_op;
    }
  }
  for (auto it : input_tensors) {
    if (operation_replace_map.find(it->op) == operation_replace_map.end()) {
      output_tensors.push_back(it);
    } else {
      auto new_tensor = TensorNode::make(it->shape, it->dtype, operation_replace_map[it->op], it->value_index);
      output_tensors.push_back(new_tensor);
    }
  }
  return;
}

Tensor ADPassReplaceExternalTensorsUsingOperations(const Tensor &root, const Array<Tensor> &matching_array) {
  std::unordered_map<Tensor, Tensor> replace_map;
  for (size_t i = 0; i < matching_array.size() / 2; i++) {
    replace_map.insert({matching_array[2 * i], matching_array[2 * i + 1]});
    // Checking if output tensor is from a Cast
    if (matching_array[2 * i]->op->InputTensors().size() == 1) {
      const ComputeOpNode *op{nullptr};
      if ((op = matching_array[2 * i]->op.as<ComputeOpNode>()) && (op->body[0].as<Cast>())) {
        // Found a Cast
        Tensor new_tensor =
          topi::cast(matching_array[2 * i + 1], (matching_array[2 * i]->op->InputTensors()[0])->dtype);
        // Adding a new position to dictionary
        replace_map.insert({matching_array[2 * i]->op->InputTensors()[0], new_tensor});
      }
    }
  }
  return ADPassReplaceTensorsUsingOperations(root, replace_map, false);
}

void ADPassMergeInternalArrayTensors(Array<Tensor> &input_tensors, Array<Tensor> &output_tensors) {
  std::unordered_map<int, std::vector<std::unordered_set<Tensor>>> equal_tensors;
  std::unordered_map<int, std::unordered_set<Tensor>> all_tensors_at_distance;
  std::unordered_map<Tensor, Tensor> replace_map;
  std::unordered_set<int> distances;

  for (size_t i = 0; i < input_tensors.size(); i++) {
    CollectAllTensorsByDistance(input_tensors[i], distances, all_tensors_at_distance);
    distances.clear();
  }

  std::unordered_map<Tensor, int> dfs_order;
  for (auto it : input_tensors) {
    CollectDFSOrder(it, dfs_order);
  }

  FindEqualTensorsByDistance(all_tensors_at_distance, equal_tensors);
  for (auto it : equal_tensors) {
    if (it.second.size() == 0) {
      continue;
    }
    for (auto it2 : it.second) {
      std::vector<Tensor> tmp_equal_tensors;
      for (auto it3 : it2) {
        if (std::find(tmp_equal_tensors.begin(), tmp_equal_tensors.end(), it3) == tmp_equal_tensors.end()) {
          tmp_equal_tensors.push_back(it3);
        }
      }
      size_t min_indx = 0;
      for (size_t i = 0; i < tmp_equal_tensors.size(); i++) {
        if (dfs_order[tmp_equal_tensors[i]] < dfs_order[tmp_equal_tensors[min_indx]]) {
          min_indx = i;
        }
      }
      for (size_t i = 0; i < tmp_equal_tensors.size(); i++) {
        if (i == min_indx) {
          continue;
        }
        // Add mapping tmp_equal_tensors[i] --> tmp_equal_tensors[min_indx] to the replace_map
        // Check if a reverse mapping (tmp_equal_tensors[min_indx] --> tmp_equal_tensors[i]) was
        // already in the replace_map
        if (replace_map.find(tmp_equal_tensors[min_indx]) != replace_map.end()) {
          if (replace_map[tmp_equal_tensors[min_indx]] == tmp_equal_tensors[i]) {
            continue;
          }
        }
        replace_map.insert({tmp_equal_tensors[i], tmp_equal_tensors[min_indx]});
      }
    }
  }

  ADPassReplaceArrayTensorsUsingOperations(input_tensors, output_tensors, replace_map);
  return;
}

bool IsBroadcast(const Tensor &tensor) {
  auto output_shape = tensor->shape;
  if (tensor->op->InputTensors().size() == 0) {
    return false;
  }
  for (auto it : tensor->op->InputTensors()) {
    auto input_shape = it->shape;
    if (input_shape.size() != output_shape.size()) {
      return false;
    }
    int broadcast_count = 0;
    for (size_t i = 0; i < output_shape.size(); i++) {
      const IntImm *input_shape_val{nullptr};
      const IntImm *output_shape_val{nullptr};
      if ((input_shape_val = input_shape[i].as<IntImm>()) == nullptr) {
        return false;
      }
      if ((output_shape_val = output_shape[i].as<IntImm>()) == nullptr) {
        return false;
      }

      if ((output_shape_val->value > 1) && (input_shape_val->value == 1)) {
        broadcast_count++;
      }
    }
    if (broadcast_count == 0) {
      return false;
    }
  }
  return true;
}

bool IsBroadcastAt(const Tensor &tensor, const std::vector<size_t> &axes) {
  auto output_shape = tensor->shape;
  if ((tensor->op->InputTensors().size() == 0) || (axes.size() == 0)) {
    return false;
  }
  size_t max_axis = axes[0];
  for (auto i : axes) {
    if (i > max_axis) {
      max_axis = i;
    }
  }

  for (auto it : tensor->op->InputTensors()) {
    auto input_shape = it->shape;
    if ((input_shape.size() != output_shape.size()) || (input_shape.size() < max_axis)) {
      return false;
    }
    for (size_t i = 0; i < output_shape.size(); i++) {
      const IntImm *input_shape_val{nullptr};
      const IntImm *output_shape_val{nullptr};
      if ((input_shape_val = input_shape[i].as<IntImm>()) == nullptr) {
        return false;
      }
      if ((output_shape_val = output_shape[i].as<IntImm>()) == nullptr) {
        return false;
      }

      if (std::find(axes.begin(), axes.end(), i) != axes.end()) {
        if (input_shape_val->value != 1) {
          return false;
        }
      } else {
        if (output_shape_val->value != input_shape_val->value) {
          return false;
        }
      }
    }
  }
  return true;
}

bool IsSameBroadcast(const Tensor &left_tensor, const Tensor &right_tensor) {
  if (!IsBroadcast(left_tensor) || !IsBroadcast(right_tensor)) {
    return false;
  }

  auto left_output_shape = left_tensor->shape;
  auto right_output_shape = right_tensor->shape;
  if (left_output_shape.size() != right_output_shape.size()) {
    return false;
  }
  for (size_t i = 0; i < left_output_shape.size(); i++) {
    if ((left_output_shape[i].as<IntImm>() == nullptr) || (right_output_shape[i].as<IntImm>() == nullptr)) {
      return false;
    }
    if (left_output_shape[i].as<IntImm>()->value != right_output_shape[i].as<IntImm>()->value) {
      return false;
    }
  }

  if (left_tensor->op->InputTensors().size() != right_tensor->op->InputTensors().size()) {
    return false;
  }
  for (size_t i = 0; i < left_tensor->op->InputTensors().size(); i++) {
    auto left_input_shape = left_tensor->op->InputTensors()[i]->shape;
    auto right_input_shape = right_tensor->op->InputTensors()[i]->shape;
    if (left_input_shape.size() != right_input_shape.size()) {
      return false;
    }
    for (size_t j = 0; j < left_input_shape.size(); j++) {
      if ((left_input_shape[j].as<IntImm>() == nullptr) || (right_input_shape[j].as<IntImm>() == nullptr)) {
        return false;
      }
      if (left_input_shape[j].as<IntImm>()->value != right_input_shape[j].as<IntImm>()->value) {
        return false;
      }
    }
  }
  return true;
}

Tensor IsolateTensor(const Tensor &tensor) {
  auto lambda_reload = [&](const Array<Var> &i) { return tensor(Array<Expr>(i.begin(), i.end())); };
  std::unordered_map<std::string, NodeRef> attrs;
  attrs["no_inline"] = Expr(1);
  Tensor reload_tensor = compute(tensor->shape, lambda_reload, tensor->op->name + "_no_inline", tensor->op->tag, attrs);
  return reload_tensor;
}

void CollectDFSOrder(const Tensor &root, std::unordered_map<Tensor, int> &dfs_order) {
  for (auto inp : root->op->InputTensors()) {
    CollectDFSOrder(inp, dfs_order);
  }
  if (dfs_order.find(root) == dfs_order.end()) {
    // To make sure the size is before calling dfs_order[root]
    int size = static_cast<int>(dfs_order.size());
    (void)dfs_order.emplace(root, size);
  }
  return;
}

void ReplaceAndIsolateArrayTensors(Array<Tensor> &input_tensors, Array<Tensor> &output_tensors,
                                   std::unordered_set<Tensor> &tensors_to_isolate) {
  std::unordered_map<Tensor, std::unordered_set<Tensor>> reverse_dependencies;
  CollectReverseDependencies(input_tensors, reverse_dependencies);

  std::unordered_map<Tensor, int> dfs_order;
  for (auto it : input_tensors) {
    CollectDFSOrder(it, dfs_order);
  }
  // Defining a lambda function to compare two tensors according to orders from dfs_order
  std::function<bool(const Tensor &, const Tensor &)> comp_func = [&dfs_order](const Tensor &t1, const Tensor &t2) {
    return dfs_order[t1] < dfs_order[t2];
  };

  std::set<Tensor, std::function<bool(Tensor, Tensor)>> ordered_queue(tensors_to_isolate.begin(),
                                                                      tensors_to_isolate.end(), comp_func);
  std::unordered_map<Tensor, Tensor> replace_map;
  std::unordered_map<Operation, Operation> operation_replace_map;

  while (ordered_queue.size() >= 1) {
    auto it_first = ordered_queue.begin();
    Tensor act_tensor = *it_first;
    (void)ordered_queue.erase(it_first);
    auto new_op = ReplaceInputs(act_tensor->op, operation_replace_map);
    if (tensors_to_isolate.find(act_tensor) != tensors_to_isolate.end()) {
      auto new_tensor = TensorNode::make(act_tensor->shape, act_tensor->dtype, new_op, act_tensor->value_index);
      auto new_isolated_tensor = IsolateTensor(new_tensor);
      new_op = new_isolated_tensor->op;
      if (std::find(input_tensors.begin(), input_tensors.end(), act_tensor) != input_tensors.end()) {
        replace_map[act_tensor] = new_isolated_tensor;
      }
    } else {
      if (std::find(input_tensors.begin(), input_tensors.end(), act_tensor) != input_tensors.end()) {
        auto new_tensor = TensorNode::make(act_tensor->shape, act_tensor->dtype, new_op, act_tensor->value_index);
        replace_map[act_tensor] = new_tensor;
      }
    }
    if (!act_tensor->op.same_as(new_op)) {
      operation_replace_map[act_tensor->op] = new_op;
    }
    for (auto it_parent : reverse_dependencies[act_tensor]) {
      (void)ordered_queue.emplace(it_parent);
    }
  }
  for (auto it : input_tensors) {
    if (replace_map.find(it) == replace_map.end()) {
      output_tensors.push_back(it);
    } else {
      output_tensors.push_back(replace_map[it]);
    }
  }
  return;
}

void ADPassIsolateTensors(Array<Tensor> &input_tensors, Array<Tensor> &output_tensors) {
  std::unordered_map<int, std::unordered_set<Tensor>> tmp_all_tensors_at_distance;
  std::unordered_map<Tensor, std::set<int>> tensor_distances;
  std::unordered_set<int> distances;
  for (size_t i = 0; i < input_tensors.size(); i++) {
    tmp_all_tensors_at_distance.clear();
    distances.clear();
    CollectAllTensorsByDistance(input_tensors[i], distances, tmp_all_tensors_at_distance);
    for (auto it : tmp_all_tensors_at_distance) {
      for (auto it2 : it.second) {
        (void)tensor_distances[it2].emplace(it.first);
      }
    }
  }

  std::unordered_map<Tensor, std::unordered_set<Tensor>> reverse_dependencies;
  CollectReverseDependencies(input_tensors, reverse_dependencies);

  std::unordered_set<Tensor> tensors_to_isolate;
  for (auto it : reverse_dependencies) {
    if ((it.second.size() >= AD_PASS_MIN_DEPENDANTS_TO_ISOLATE) && (tensor_distances[it.first].size() > 0) &&
        (*(tensor_distances[it.first].begin()) >= AD_PASS_MIN_DISTANCE_TO_ISOLATE)) {
      (void)tensors_to_isolate.emplace(it.first);
    }
  }

  if (tensors_to_isolate.size() == 0) {
    for (auto it : input_tensors) {
      output_tensors.push_back(it);
    }
    return;
  }
  Array<Tensor> tmp_result;
  ReplaceAndIsolateArrayTensors(input_tensors, tmp_result, tensors_to_isolate);
  // Running pass MergeInternalArrayTensors again because newly equal tensors could be generated
  ADPassMergeInternalArrayTensors(tmp_result, output_tensors);
  return;
}

bool IsReduceSum(const Tensor &root, std::vector<size_t> &reduction_axes) {
  const ComputeOpNode *comp_op{nullptr};
  const Reduce *reduce_op{nullptr};
  if ((comp_op = root->op.as<ComputeOpNode>()) == nullptr) {
    return false;
  }
  if (!comp_op->body.empty() && (reduce_op = comp_op->body[0].as<Reduce>()) == nullptr) {
    return false;
  }
  CHECK(reduce_op->combiner.get());
  CHECK(!reduce_op->combiner.get()->result.empty());
  auto reduction_type = reduce_op->combiner.get()->result[0];
  if (!reduction_type.as<Add>()) {
    return false;
  }
  if (!reduce_op->source.empty() && reduce_op->source[0].as<Call>() == nullptr) {
    return false;
  }
  for (size_t i = 0; i < reduce_op->source[0].as<Call>()->args.size(); i++) {
    for (size_t j = 0; j < reduce_op->axis.size(); j++) {
      if (Equal(reduce_op->source[0].as<Call>()->args[i], reduce_op->axis[j])) {
        reduction_axes.push_back(i);
      }
    }
  }
  return true;
}

bool IsReducePattern_1(const Tensor &root, std::vector<Tensor> &result) {
  // reduce(A * (B * broadcast(C)))
  static int counter = 1;
  std::vector<size_t> reduction_axes;
  if (!IsReduceSum(root, reduction_axes)) {
    return false;
  }
  CHECK(root->op.defined());
  if (root->op->InputTensors().size() != 1) {
    return false;
  }
  if (root->op->InputTensors()[0]->op.as<ComputeOpNode>() == nullptr) {
    return false;
  }
  if (!root->op->InputTensors()[0]->op.as<ComputeOpNode>()->body.empty() &&
      root->op->InputTensors()[0]->op.as<ComputeOpNode>()->body[0]->GetTypeKey() != "Mul") {
    return false;
  }
  if (root->op->InputTensors()[0]->op->InputTensors().size() != 2) {
    return false;
  }
  const Tensor &A = root->op->InputTensors()[0]->op->InputTensors()[0];
  if (root->op->InputTensors()[0]->op->InputTensors()[1]->op.as<ComputeOpNode>() == nullptr) {
    return false;
  }
  if (!root->op->InputTensors()[0]->op->InputTensors()[1]->op.as<ComputeOpNode>()->body.empty() &&
      root->op->InputTensors()[0]->op->InputTensors()[1]->op.as<ComputeOpNode>()->body[0]->GetTypeKey() != "Mul") {
    return false;
  }
  if (root->op->InputTensors()[0]->op->InputTensors()[1]->op->InputTensors().size() != 2) {
    return false;
  }
  const Tensor &B = root->op->InputTensors()[0]->op->InputTensors()[1]->op->InputTensors()[0];
  if (!IsBroadcastAt(root->op->InputTensors()[0]->op->InputTensors()[1]->op->InputTensors()[1], reduction_axes)) {
    return false;
  }
  const Tensor &C = root->op->InputTensors()[0]->op->InputTensors()[1]->op->InputTensors()[1]->op->InputTensors()[0];
  // Push back new tensor into result
  std::string name1 = std::string("T_mul_r1_") + std::to_string(counter++);
  std::string name2 = std::string("T_mul_r1_") + std::to_string(counter++);
  Array<air::Integer> red_axes;
  for (auto i : reduction_axes) {
    red_axes.push_back(i);
  }
  result.push_back(topi::multiply(topi::sum(topi::multiply(A, B, name1), red_axes, true, true), C, name2));
  return true;
}

bool IsReducePattern_2(const Tensor &root, std::vector<Tensor> &result) {
  // reduce((A * broadcast(B))*broadcast(C))
  static int counter = 1;
  std::vector<size_t> reduction_axes;
  if (!IsReduceSum(root, reduction_axes)) {
    return false;
  }
  CHECK(root->op.defined());
  if (root->op->InputTensors().size() != 1) {
    return false;
  }
  if (root->op->InputTensors()[0]->op.as<ComputeOpNode>() == nullptr) {
    return false;
  }
  if (!root->op->InputTensors()[0]->op.as<ComputeOpNode>()->body.empty() &&
      root->op->InputTensors()[0]->op.as<ComputeOpNode>()->body[0]->GetTypeKey() != "Mul") {
    return false;
  }
  if (root->op->InputTensors()[0]->op->InputTensors().size() != 2) {
    return false;
  }
  if (!IsBroadcastAt(root->op->InputTensors()[0]->op->InputTensors()[1], reduction_axes)) {
    return false;
  }
  const Tensor &C = root->op->InputTensors()[0]->op->InputTensors()[1]->op->InputTensors()[0];
  if (root->op->InputTensors()[0]->op->InputTensors()[0]->op.as<ComputeOpNode>() == nullptr) {
    return false;
  }
  if (!root->op->InputTensors()[0]->op->InputTensors()[0]->op.as<ComputeOpNode>()->body.empty() &&
      root->op->InputTensors()[0]->op->InputTensors()[0]->op.as<ComputeOpNode>()->body[0]->GetTypeKey() != "Mul") {
    return false;
  }
  if (root->op->InputTensors()[0]->op->InputTensors()[0]->op->InputTensors().size() != 2) {
    return false;
  }
  const Tensor &A = root->op->InputTensors()[0]->op->InputTensors()[0]->op->InputTensors()[0];
  if (!IsBroadcastAt(root->op->InputTensors()[0]->op->InputTensors()[0]->op->InputTensors()[1], reduction_axes)) {
    return false;
  }
  const Tensor &B = root->op->InputTensors()[0]->op->InputTensors()[0]->op->InputTensors()[1]->op->InputTensors()[0];
  // Push back new tensor into result
  std::string name1 = std::string("T_mul_r2_") + std::to_string(counter++);
  std::string name2 = std::string("T_mul_r2_") + std::to_string(counter++);
  Array<air::Integer> red_axes;
  for (auto i : reduction_axes) {
    red_axes.push_back(Integer(i));
  }
  result.push_back(topi::multiply(topi::sum(A, red_axes, true, true), topi::multiply(B, C, name1), name2));
  return true;
}

bool IsReducePattern_3(const Tensor &root, std::vector<Tensor> &result) {
  // reduce((A * broadcast(B))*C)
  static int counter = 0;
  std::vector<size_t> reduction_axes;
  if (!IsReduceSum(root, reduction_axes)) {
    return false;
  }
  CHECK(root->op.defined());
  if (root->op->InputTensors().size() != 1) {
    return false;
  }
  if (root->op->InputTensors()[0]->op.as<ComputeOpNode>() == nullptr) {
    return false;
  }
  if (!root->op->InputTensors()[0]->op.as<ComputeOpNode>()->body.empty() &&
      root->op->InputTensors()[0]->op.as<ComputeOpNode>()->body[0]->GetTypeKey() != "Mul") {
    return false;
  }
  if (root->op->InputTensors()[0]->op->InputTensors().size() != 2) {
    return false;
  }
  const Tensor &C = root->op->InputTensors()[0]->op->InputTensors()[1];
  if (root->op->InputTensors()[0]->op->InputTensors()[0]->op.as<ComputeOpNode>() == nullptr) {
    return false;
  }
  if (!root->op->InputTensors()[0]->op->InputTensors()[0]->op.as<ComputeOpNode>()->body.empty() &&
      root->op->InputTensors()[0]->op->InputTensors()[0]->op.as<ComputeOpNode>()->body[0]->GetTypeKey() != "Mul") {
    return false;
  }
  if (root->op->InputTensors()[0]->op->InputTensors()[0]->op->InputTensors().size() != 2) {
    return false;
  }
  const Tensor &A = root->op->InputTensors()[0]->op->InputTensors()[0]->op->InputTensors()[0];
  if (!IsBroadcastAt(root->op->InputTensors()[0]->op->InputTensors()[0]->op->InputTensors()[1], reduction_axes)) {
    return false;
  }
  const Tensor &B = root->op->InputTensors()[0]->op->InputTensors()[0]->op->InputTensors()[1]->op->InputTensors()[0];
  // Push back new tensor into result
  std::string name1 = std::string("T_mul_r3_") + std::to_string(counter++);
  std::string name2 = std::string("T_mul_r3_") + std::to_string(counter++);
  Array<air::Integer> red_axes;
  for (auto i : reduction_axes) {
    red_axes.push_back(Integer(i));
  }
  result.push_back(topi::multiply(topi::sum(topi::multiply(A, C, name1), red_axes, true, true), B, name2));
  return true;
}

void CollectAllReduce(const Tensor &tensor, std::vector<Tensor> &result,
                      std::unordered_map<Tensor, Tensor> &map_new_reduce) {
  std::vector<size_t> reduction_axis;
  if (IsReduceSum(tensor, reduction_axis)) {
    if (std::find(result.begin(), result.end(), tensor) == result.end()) {
      result.push_back(tensor);
    }
    std::vector<Tensor> result_patern;
    if (IsReducePattern_1(tensor, result_patern)) {
      map_new_reduce.insert({tensor, result_patern.back()});
    } else {
      if (IsReducePattern_2(tensor, result_patern)) {
        map_new_reduce.insert({tensor, result_patern.back()});
      } else {
        if (IsReducePattern_3(tensor, result_patern)) {
          map_new_reduce.insert({tensor, result_patern.back()});
        }
      }
    }
  }
  for (auto inp : tensor->op->InputTensors()) {
    CollectAllReduce(inp, result, map_new_reduce);
  }
}

void ADPassReduceBroadcastSimplify(Array<Tensor> &input_tensors, Array<Tensor> &output_tensors) {
  std::vector<Tensor> result;
  std::unordered_map<Tensor, Tensor> map_new_reduce;
  for (auto it : input_tensors) {
    CollectAllReduce(it, result, map_new_reduce);
  }
  Array<Tensor> tmp_result;
  ADPassReplaceArrayTensorsUsingOperations(input_tensors, tmp_result, map_new_reduce);
  // Running pass MergeInternalArrayTensors again because newly equal tensors could be generated
  ADPassMergeInternalArrayTensors(tmp_result, output_tensors);
  return;
}

bool IsPullSupportedMul(const Tensor &tensor) {
  const ComputeOpNode *comp_op{nullptr};
  const Mul *mul_op{nullptr};
  if ((comp_op = tensor->op.as<ComputeOpNode>()) == nullptr) {
    return false;
  } else {
    if ((mul_op = comp_op->body[0].as<Mul>()) == nullptr) {
      return false;
    }
  }
  if ((tensor->op->InputTensors().size() == 1) && !mul_op->a.as<FloatImm>() && !mul_op->b.as<FloatImm>()) {
    return false;
  }
  return true;
}

void PullConstFromMul(const Tensor &tensor, float &new_const,
                      std::unordered_map<Tensor, std::pair<Tensor, float>> &map_new_muls) {
  static int counter = 1;
  CHECK(tensor->op.defined());
  if (!IsPullSupportedMul(tensor) || (tensor->op->InputTensors().size() < 1)) {
    for (auto it : tensor->op->InputTensors()) {
      if (map_new_muls.find(it) == map_new_muls.end()) {
        PullConstFromMul(it, new_const, map_new_muls);
      } else {
        new_const = std::get<1>(map_new_muls[it]);
      }
    }
    new_const = 1.0;
    map_new_muls[tensor] = std::make_pair(tensor, 1.0);
    return;
  }

  if (tensor->op->InputTensors().size() == 2) {
    float new_const_left, new_const_right;
    PullConstFromMul(tensor->op->InputTensors()[0], new_const_left, map_new_muls);
    PullConstFromMul(tensor->op->InputTensors()[1], new_const_right, map_new_muls);
    bool free_left =
      ((map_new_muls.find(tensor->op->InputTensors()[0]) == map_new_muls.end()) ||
       (std::get<0>(map_new_muls[tensor->op->InputTensors()[0]])->op.same_as(tensor->op->InputTensors()[0]->op)));
    bool free_right =
      ((map_new_muls.find(tensor->op->InputTensors()[1]) == map_new_muls.end()) ||
       (std::get<0>(map_new_muls[tensor->op->InputTensors()[1]])->op.same_as(tensor->op->InputTensors()[1]->op)));
    if (free_left && free_right && (fabs(new_const_left - 1.0) < __FLT_EPSILON__) &&
        (fabs(new_const_right - 1.0) < __FLT_EPSILON__)) {
      new_const = 1.0;
      map_new_muls[tensor] = std::make_pair(tensor, 1.0);
      return;
    }
    new_const = new_const_left * new_const_right;
    Tensor new_mul = topi::multiply(std::get<0>(map_new_muls[tensor->op->InputTensors()[0]]),
                                    std::get<0>(map_new_muls[tensor->op->InputTensors()[1]]),
                                    std::string("T_mul_pc_") + std::to_string(counter++));
    map_new_muls[tensor] = std::make_pair(new_mul, new_const);
    return;
  } else {
    PullConstFromMul(tensor->op->InputTensors()[0], new_const, map_new_muls);
    // Checked for validity of mul_op in IsPullSupportedMul(tensor)
    CHECK(tensor->op.as<ComputeOpNode>());
    CHECK(!tensor->op.as<ComputeOpNode>()->body.empty());
    auto mul_op = tensor->op.as<ComputeOpNode>()->body[0].as<Mul>();
    CHECK(mul_op);
    float new_const_const = 0;
    if (mul_op->a.as<FloatImm>()) {
      new_const_const = mul_op->a.as<FloatImm>()->value;
    } else {
      if (mul_op->b.as<FloatImm>()) {
        new_const_const = mul_op->b.as<FloatImm>()->value;
      }
    }
    new_const *= new_const_const;
    map_new_muls[tensor] = std::make_pair(std::get<0>(map_new_muls[tensor->op->InputTensors()[0]]), new_const);
  }
}

void ADPassSimplifyConstMultiply(Array<Tensor> &input_tensors, Array<Tensor> &output_tensors) {
  std::unordered_map<Tensor, std::pair<Tensor, float>> map_new_muls;
  std::unordered_map<Tensor, Tensor> replace_map;
  static int counter = 1;
  for (auto it : input_tensors) {
    float new_const = 0.0;
    PullConstFromMul(it, new_const, map_new_muls);
    // Constructing new maps
    for (auto it2 : map_new_muls) {
      if (map_new_muls.find(std::get<0>(it2.second)) == map_new_muls.end()) {
        if (std::get<1>(it2.second) != 1.0) {
          std::string name = std::string("T_mul_pc_gen_") + std::to_string(counter++);
          auto new_expr = Expr(FloatImm::make(it2.first->dtype, std::get<1>(it2.second)));
          (void)replace_map.emplace(it2.first, topi::multiply(std::get<0>(it2.second), new_expr, name));
        } else {
          (void)replace_map.emplace(it2.first, std::get<0>(it2.second));
        }
      }
    }
  }
  Array<Tensor> tmp_result;
  ADPassReplaceArrayTensorsUsingOperations(input_tensors, tmp_result, replace_map);
  // Running pass MergeInternalArrayTensors again because newly equal tensors could be generated
  ADPassMergeInternalArrayTensors(tmp_result, output_tensors);
  return;
}

bool IsMulWithTwoInputs(const Tensor &root) {
  const auto comp_op = root->op.as<ComputeOpNode>();
  if (comp_op == nullptr) {
    return false;
  }
  const auto mul_op = comp_op->body[0].as<Mul>();
  if (mul_op == nullptr) {
    return false;
  }
  if (root->op->InputTensors().size() != 2) {
    return false;
  }
  return true;
}

void CollectAllMulWithTwoInputs(const Tensor &tensor, std::unordered_set<Tensor> &result) {
  if (IsMulWithTwoInputs(tensor)) {
    if (std::find(result.begin(), result.end(), tensor) == result.end()) {
      (void)result.emplace(tensor);
    }
  }
  for (auto inp : tensor->op->InputTensors()) {
    CollectAllMulWithTwoInputs(inp, result);
  }
}

bool FindMul(const std::unordered_set<Tensor> &all_mul, const Tensor &left_tensor, const Tensor &right_tensor) {
  for (auto it : all_mul) {
    if (it->op->InputTensors().size() != 2) {
      continue;
    }
    if (it->op->InputTensors()[0]->op.same_as(left_tensor->op) &&
        it->op->InputTensors()[1]->op.same_as(right_tensor->op)) {
      return true;
    }
    if (it->op->InputTensors()[1]->op.same_as(left_tensor->op) &&
        it->op->InputTensors()[0]->op.same_as(right_tensor->op)) {
      return true;
    }
  }
  return false;
}

Tensor GetMul(const std::unordered_set<Tensor> &all_mul, const Tensor &left_tensor, const Tensor &right_tensor) {
  for (auto it : all_mul) {
    if (it->op->InputTensors().size() != 2) {
      continue;
    }
    if (it->op->InputTensors()[0]->op.same_as(left_tensor->op) &&
        it->op->InputTensors()[1]->op.same_as(right_tensor->op)) {
      return it;
    }
    if (it->op->InputTensors()[1]->op.same_as(left_tensor->op) &&
        it->op->InputTensors()[0]->op.same_as(right_tensor->op)) {
      return it;
    }
  }
  // Return as dummy when inputs are not matched
  return left_tensor;
}

void ADPassSwapMultiplyOrder(Array<Tensor> &input_tensors, Array<Tensor> &output_tensors) {
  std::unordered_map<Tensor, std::unordered_set<Tensor>> reverse_dependencies;
  CollectReverseDependencies(input_tensors, reverse_dependencies);

  std::unordered_set<Tensor> all_mul;
  for (auto it : input_tensors) {
    CollectAllMulWithTwoInputs(it, all_mul);
  }
  std::unordered_map<Tensor, Tensor> replace_map;
  static int counter = 1;
  for (auto it_A : all_mul) {
    Tensor B = it_A->op->InputTensors()[0];
    Tensor C = it_A->op->InputTensors()[1];
    Tensor new_A;
    std::string name = std::string("T_mul_sw_") + std::to_string(counter);
    if ((all_mul.find(B) != all_mul.end()) && (reverse_dependencies[B].size() == 1)) {
      Tensor D = B->op->InputTensors()[0];
      Tensor E = B->op->InputTensors()[1];
      if (FindMul(all_mul, D, C)) {
        counter++;
        new_A = topi::multiply(GetMul(all_mul, D, C), E, name);
        replace_map[it_A] = new_A;
        continue;
      }
      if (FindMul(all_mul, E, C)) {
        counter++;
        new_A = topi::multiply(GetMul(all_mul, E, C), D, name);
        replace_map[it_A] = new_A;
        continue;
      }
    }
    if ((all_mul.find(C) != all_mul.end()) && (reverse_dependencies[C].size() == 1)) {
      Tensor D = C->op->InputTensors()[0];
      Tensor E = C->op->InputTensors()[1];
      if (FindMul(all_mul, B, D)) {
        counter++;
        new_A = topi::multiply(GetMul(all_mul, B, D), E, name);
        replace_map[it_A] = new_A;
        continue;
      }
      if (FindMul(all_mul, B, E)) {
        counter++;
        new_A = topi::multiply(GetMul(all_mul, B, E), D, name);
        replace_map[it_A] = new_A;
      }
    }
  }
  Array<Tensor> tmp_result;
  ADPassReplaceArrayTensorsUsingOperations(input_tensors, tmp_result, replace_map);
  // Running pass MergeInternalArrayTensors again because newly equal tensors may be generated
  ADPassMergeInternalArrayTensors(tmp_result, output_tensors);
  return;
}

void ADPassMergeMultipleBroadcast(Array<Tensor> &input_tensors, Array<Tensor> &output_tensors) {
  std::unordered_set<Tensor> all_mul;
  for (auto it : input_tensors) {
    CollectAllMulWithTwoInputs(it, all_mul);
  }
  std::unordered_map<Tensor, Tensor> replace_map;
  static int counter = 1;
  for (auto it_A : all_mul) {
    Tensor B = it_A->op->InputTensors()[0];
    Tensor C = it_A->op->InputTensors()[1];
    Tensor new_A;
    std::string name_mul1 = std::string("T_mul_mb1_") + std::to_string(counter);
    std::string name_mul2 = std::string("T_mul_mb2_") + std::to_string(counter);
    std::string name_bc = std::string("T_bc_mb_") + std::to_string(counter);
    if (IsSameBroadcast(B, C)) {
      counter++;
      new_A = topi::broadcast_to(topi::multiply(B->op->InputTensors()[0], C->op->InputTensors()[0], name_mul1),
                                 it_A->shape, name_bc);
      replace_map[it_A] = new_A;
      continue;
    }
    if ((all_mul.find(B) != all_mul.end()) && IsBroadcast(C)) {
      Tensor D = B->op->InputTensors()[0];
      Tensor E = B->op->InputTensors()[1];
      if (IsSameBroadcast(C, D)) {
        counter++;
        new_A = topi::multiply(
          topi::broadcast_to(topi::multiply(D->op->InputTensors()[0], C->op->InputTensors()[0], name_mul1), it_A->shape,
                             name_bc),
          E, name_mul2);
        replace_map[it_A] = new_A;
        continue;
      }
      if (IsSameBroadcast(C, E)) {
        counter++;
        new_A = topi::multiply(
          topi::broadcast_to(topi::multiply(E->op->InputTensors()[0], C->op->InputTensors()[0], name_mul1), it_A->shape,
                             name_bc),
          D, name_mul2);
        replace_map[it_A] = new_A;
        continue;
      }
    }
    if ((all_mul.find(C) != all_mul.end()) && IsBroadcast(B)) {
      Tensor D = C->op->InputTensors()[0];
      Tensor E = C->op->InputTensors()[1];
      if (IsSameBroadcast(B, D)) {
        counter++;
        new_A = topi::multiply(
          topi::broadcast_to(topi::multiply(B->op->InputTensors()[0], D->op->InputTensors()[0], name_mul1), it_A->shape,
                             name_bc),
          E, name_mul2);
        replace_map[it_A] = new_A;
        continue;
      }
      if (IsSameBroadcast(B, E)) {
        counter++;
        new_A = topi::multiply(
          topi::broadcast_to(topi::multiply(B->op->InputTensors()[0], E->op->InputTensors()[0], name_mul1), it_A->shape,
                             name_bc),
          D, name_mul2);
        replace_map[it_A] = new_A;
        continue;
      }
    }
  }
  Array<Tensor> tmp_result;
  ADPassReplaceArrayTensorsUsingOperations(input_tensors, tmp_result, replace_map);
  // Running pass MergeInternalArrayTensors again because newly equal tensors may be generated
  ADPassMergeInternalArrayTensors(tmp_result, output_tensors);
  return;
}

void ADRunAllPasses(Array<Tensor> &input_tensors, Array<Tensor> &output_tensors, AttrMap &in_attrs,
                    const Array<Tensor> &new_pld_array, const std::string &DOT_prefix) {
  bool export_DOT_ = (in_attrs.GetIntAttr("export_DOT", 0) != 0);
  auto f_group = air::runtime::Registry::Get("akg.autodiff.export_to_DOT");
  if (f_group == nullptr) {
    export_DOT_ = false;
  }
  // Pass 1: Replace selected tensors with placeholders with values from forward
  Array<Tensor> result_pld_pass1;
  for (const auto &it : input_tensors) {
    result_pld_pass1.push_back(ADPassReplaceExternalTensorsUsingOperations(it, new_pld_array));
  }
  // Pass 2: Recursively find "equal" tensors and merge them together
  Array<Tensor> result_pld_pass2;
  ADPassMergeInternalArrayTensors(result_pld_pass1, result_pld_pass2);
  // Pass 3: Reduce-Broadcast simplify
  Array<Tensor> result_pld_pass3;
  ADPassReduceBroadcastSimplify(result_pld_pass2, result_pld_pass3);
  // Pass 4: Pulling const
  Array<Tensor> result_pld_pass4;
  ADPassSimplifyConstMultiply(result_pld_pass3, result_pld_pass4);
  // Pass 5: Merge broadcasted Muls
  Array<Tensor> result_pld_pass5;
  ADPassMergeMultipleBroadcast(result_pld_pass4, result_pld_pass5);
  // Pass 6: Swap tensors' order to find more reuse of Mul
  Array<Tensor> result_pld_pass6;
  ADPassSwapMultiplyOrder(result_pld_pass5, result_pld_pass6);
  bool disable_isolating_ = (in_attrs.GetIntAttr("disable_isolating", 0) != 0);
  if (!disable_isolating_) {
    // Pass 7: Automatic finding of common computation nodes and isolating them
    Array<Tensor> result_pld_pass7;
    ADPassIsolateTensors(result_pld_pass6, result_pld_pass7);
    if (export_DOT_) {
      (void)(*f_group)(result_pld_pass7, DOT_prefix + "_7.dot");
    }
  }
  output_tensors.assign(result_pld_pass6.begin(), result_pld_pass6.end());
  if (export_DOT_) {
    (void)(*f_group)(input_tensors, DOT_prefix + "_inputs.dot");
    (void)(*f_group)(result_pld_pass1, DOT_prefix + "_1.dot");
    (void)(*f_group)(result_pld_pass2, DOT_prefix + "_2.dot");
    (void)(*f_group)(result_pld_pass3, DOT_prefix + "_3.dot");
    (void)(*f_group)(result_pld_pass4, DOT_prefix + "_4.dot");
    (void)(*f_group)(result_pld_pass5, DOT_prefix + "_5.dot");
    (void)(*f_group)(result_pld_pass6, DOT_prefix + "_6.dot");
    (void)(*f_group)(output_tensors, DOT_prefix + "_outputs.dot");
  }
  return;
}

void ADOptimizePasses(Array<Tensor> &input_tensors, Array<Tensor> &output_tensors,
                      const Map<std::string, NodeRef> &attrs, const Array<Tensor> &new_pld_array) {
  AttrMap in_attrs;
  if (attrs.defined()) {
    in_attrs = attrs;
  }
  bool separate_output_ = (in_attrs.GetIntAttr("separate_output", 0) != 0);
  if (!separate_output_) {
    ADRunAllPasses(input_tensors, output_tensors, in_attrs, new_pld_array, "ad_pass_grouped");
  } else {
    std::string attr_key = "disable_isolating";
    in_attrs.Set(attr_key, Expr(1));
    for (size_t i = 0; i < input_tensors.size(); i++) {
      Array<Tensor> tmp_in, tmp_out;
      tmp_in.push_back(input_tensors[i]);
      ADRunAllPasses(tmp_in, tmp_out, in_attrs, new_pld_array, "ad_pass_split_" + std::to_string(i));
      output_tensors.push_back(tmp_out[0]);
    }
  }
  return;
}

Array<IterVar> DuplicateIterArray(const Array<IterVar> &axis, const std::string &suffix) {
  Array<IterVar> new_iter_arr;
  for (IterVar iv : axis) {
    IterVar new_iv = IterVarNode::make(iv->dom, iv->var.copy_with_suffix(suffix), iv->iter_type, iv->thread_tag);
    new_iter_arr.push_back(new_iv);
  }
  return new_iter_arr;
}

Array<Expr> FindInputArgs(const Array<Expr> &exprs, const Array<IterVar> &fw_comp_itervars) {
  std::unordered_map<const Variable *, Expr> arg_map;
  for (auto expr : exprs) {
    PostOrderVisit(expr, [&fw_comp_itervars, &arg_map](const NodeRef &node) {
      if (const Variable *v = node.as<Variable>()) {
        // Go through fw comp itervars and save its vars in map when name_hint matches
        // These will be replaced later when the new call to source is created.
        CHECK(v);
        for (auto fw_it : fw_comp_itervars) {
          if (v->name_hint == fw_it->var->name_hint) {
            arg_map[v] = fw_it->var;
          }
        }
      }
    });
  }

  Array<Expr> input_vars_fw;
  for (auto expr : exprs) {
    input_vars_fw.push_back(air::ir::Substitute(expr, arg_map));
  }

  return input_vars_fw;
}

Tensor BroadcastToCond(const Tensor &t, const Expr &condition, const Array<IterVar> &iter_vars,
                       const Array<IterVar> &fw_axis, const Expr &else_val, const std::string &name,
                       const std::string &tag) {
  Array<air::Expr> expr_vars;
  std::vector<IterVar> axis;
  for (auto v : iter_vars) {
    axis.emplace_back(IterVarNode::make(v->dom, v->var, air::IterVarType::kDataPar));
  }

  // Building fw compute indexes in corresponding order.
  for (auto fw : fw_axis) {
    for (auto v : axis) {
      if (fw->var->name_hint == v->var->name_hint) {
        expr_vars.push_back(v->var);
      }
    }
  }

  auto true_value = Call::make(t->dtype, t->op->name, expr_vars, Call::Halide, t->op, t->value_index);
  auto sel = Select::make(condition, true_value, else_val);

  Expr body;
  if (CanProve(condition)) {
    body = true_value;
  } else {
    // Creating compute with select(bd_tens == inp, Head (or 1), 0)
    body = sel;
  }

  return ComputeOpNode::make("broadcast_select_" + t->op->name, tag, {}, axis, {body}).output(0);
}

// Transforms a source comparison of reduction AD into a range condition
// Three cases:
// 1: jac_i == (iterator * constant) + reduction_iterator --->
//    (iterator * constant) <= jac_i < (iterator * constant) + reduction_iterator.lenght()
// 2: jac_i == iterator + reduction_iterator ---> iterator <= jac_i < iterator + reduction_iterator.lenght()
// 3: jac_i == reduction_iterator ---> 0 <= jac_i < reduction_iterator.lenght()
// extra case: if the EQ op is jac_i == iterator (no reduction), we return true
//             the condition is no needed anymore since this dimension will be removed later
class RedAxisToRangeMutator : public IRMutator {
 public:
  RedAxisToRangeMutator()
      : changes(0), reduce_iterators_count(0), all_lhs_are_jac_iterators(true), valid_eqs(true), red_axis() {}
  ~RedAxisToRangeMutator() override = default;

  Expr RemoveAxisFromExpr(const Array<IterVar> &red_ax, const Expr &e, const Array<IterVar> &new_axis) {
    this->red_axis = red_ax;
    this->CountReduceIterators(e);
    this->ValidateLHS(e);
    Expr condition = this->Mutate(e);
    condition = SuperSimplify(condition, IterVarsToMap(new_axis));
    return condition;
  }

  Expr Mutate_(const EQ *op, const Expr &e) {
    auto lhs = op->a.as<Variable>();
    if (op->type.is_bool() && lhs) {
      if (auto add = op->b.as<Add>()) {
        if (auto red_it = add->b.as<Variable>()) {
          if (auto mul = add->a.as<Mul>()) {
            if (mul->a.as<Variable>()) {
              if (mul->b.as<IntImm>() || mul->b.as<FloatImm>()) {  // case 1
                for (auto it : this->red_axis) {
                  if (red_it->name_hint == it->var->name_hint) {
                    auto get = GE::make(op->a, add->a + it->dom->min);
                    auto lt = LT::make(op->a, add->a + it->dom->extent);
                    this->changes++;
                    return And::make(get, lt);
                  }
                }
              }
            }
          } else if (add->a.as<Variable>()) {  // case 2
            for (auto it : this->red_axis) {
              if (red_it->name_hint == it->var->name_hint) {
                auto get = GE::make(op->a, add->a + it->dom->min);
                auto lt = LT::make(op->a, add->a + it->dom->extent);
                this->changes++;
                return And::make(get, lt);
              }
            }
          }
        }
      } else if (auto rhs = op->b.as<Variable>()) {  // case 3
        for (auto it : this->red_axis) {
          if (rhs->name_hint == it->var->name_hint) {
            auto get = GE::make(op->a, it->dom->min);
            auto lt = LT::make(op->a, it->dom->extent);
            this->changes++;
            return And::make(get, lt);
          }
        }

        // extra case
        // Remove redundant axis and conditions (jac_i0 == n)
        // Save the jac_ vars in list to remove them later from tensor
        iter_to_remove[rhs->name_hint] = lhs->name_hint;
        rev_iter_to_remove[lhs->name_hint] = rhs->name_hint;
        return make_const(op->type, true);
      }
    }

    // Unexpected condition reached, mutator is no longer valid
    this->valid_eqs = false;
    return e;
  }

  // Check how many reduce iterators there are inside of the expression
  void CountReduceIterators(const Expr &e) {
    if (this->red_axis.empty()) {
      return;
    }

    PostOrderVisit(e, [this](const NodeRef &node) {
      if (const auto var = node.as<Variable>()) {
        for (auto it : this->red_axis) {
          if (var->name_hint == it->var->name_hint) {
            this->reduce_iterators_count++;
          }
        }
      }
    });
  }

  // Sanity check: check if all lhs of EQ ops are iterators of the jacobian (jac_...)
  void ValidateLHS(const Expr &e) {
    PostOrderVisit(e, [this](const NodeRef &node) {
      if (const auto eq = node.as<EQ>()) {
        if (auto lhs = eq->a.as<Variable>()) {
          if (lhs->name_hint.rfind("jac_", 0) != 0) {
            this->all_lhs_are_jac_iterators = false;
          }
        }
      }
    });
  }

  std::unordered_set<std::string> FindNewRedAxis(Array<IterVar> full_axis) {
    std::unordered_set<std::string> new_axis;
    for (auto ax : full_axis) {
      auto red = rev_iter_to_remove.find(ax->var->name_hint);
      if (red == rev_iter_to_remove.end()) {
        new_axis.insert(ax->var->name_hint);
      }
    }

    return new_axis;
  }

  bool IsValid() const {
    return this->changes == this->reduce_iterators_count && this->all_lhs_are_jac_iterators && this->valid_eqs;
  }

  std::unordered_map<std::string, std::string> GetIterToRemove() { return iter_to_remove; }

 private:
  int changes;
  int reduce_iterators_count;
  bool all_lhs_are_jac_iterators = false;
  bool valid_eqs = false;
  Array<IterVar> red_axis;
  std::unordered_map<std::string, std::string> iter_to_remove;
  std::unordered_map<std::string, std::string> rev_iter_to_remove;
};

void RemoveRedundantDimensions(const std::unordered_map<std::string, std::string> &remove,
                               const Array<IterVar> &orig_axis, const ComputeOpNode *op, Expr &condition,
                               Array<IterVar> &jac_axis, Array<IterVar> &bcast_axis, Array<IterVar> &fw_axis,
                               Array<Expr> &args, Array<Expr> &input_vars_eqsel) {
  std::unordered_map<const Variable *, Expr> arg_map;
  Array<IterVar> input_iter_vars;

  for (auto ax : orig_axis) {
    if (remove.find(ax->var->name_hint) == remove.end()) {
      IterVar new_ax = IterVarNode::make(ax->dom, ax->var.copy_with_suffix(""), ax->iter_type, ax->thread_tag);
      jac_axis.push_back(new_ax);
      args.push_back(new_ax);
      bcast_axis.push_back(new_ax);
      if (ax->var->name_hint.rfind("jac_", 0) == 0) {
        input_iter_vars.push_back(new_ax);
        input_vars_eqsel.push_back(new_ax);
      }
      arg_map[ax->var.get()] = new_ax->var;
    }
  }

  for (size_t i = 0; i < op->axis.size(); i++) {
    bool found = false;
    auto ax = op->axis[i];
    for (auto jac_ax : jac_axis) {
      if (ax->var->name_hint == jac_ax->var->name_hint) {
        found = true;
        fw_axis.push_back(jac_ax);
        break;
      }
    }

    if (!found) {
      fw_axis.push_back(input_iter_vars[i]);
    }
  }

  condition = air::ir::Substitute(condition, arg_map);
}

Tensor GetForwardCompute(const Reduce *red, const ComputeOpNode *op, const Tensor &input, const Tensor &output,
                         const Call *source_call) {
  // First re-create the forward expression
  CHECK(red);
  CHECK(source_call);
  CommReducer new_combiner = CommReducerNode::make({red->combiner->lhs[1]}, {red->combiner->rhs[1]},
                                                   {red->combiner->result[1]}, {red->combiner->identity_element[1]});
  // Since we're creating a new compute op node, we need to create a new Call to the source.
  auto fw_comp_itervars = DuplicateIterArray(op->axis);

  // This new call must have args updated because the compute will not know
  // about the original iterators
  // Find expr vars in original source call
  auto input_vars_fw = FindInputArgs(source_call->args, fw_comp_itervars);
  auto inp_tens =
    Call::make(input->dtype, input->op->name, input_vars_fw, Call::CallType::Halide, input->op, input->value_index);
  auto new_reduce = Reduce::make(new_combiner, {inp_tens}, red->axis, red->condition, 0);
  auto new_op = ComputeOpNode::make("fw_" + op->name, op->tag, op->attrs, fw_comp_itervars, {new_reduce});

  // Forward compute
  return TensorNode::make(output->shape, output->dtype, new_op, 0);
}

Tensor BuildSelectFromBdcast(const ComputeOpNode *op, const Tensor &input, const Tensor &fw_bdc,
                             const Array<IterVar> &jac_axis, const Array<Expr> &args,
                             const Array<Expr> &input_vars_eqsel, const Array<IterVar> &fw_axis, const Tensor &head,
                             bool &used_head) {
  Array<Expr> head_vars_eqsel;

  // Creating args (indexing) for head tensor
  for (auto ax : fw_axis) {
    head_vars_eqsel.push_back(ax);
  }

  auto inp_tens_select =
    Call::make(input->dtype, input->op->name, input_vars_eqsel, Call::CallType::Halide, input->op, input->value_index);
  auto bd_tens =
    Call::make(fw_bdc->dtype, fw_bdc->op->name, args, Call::CallType::Halide, fw_bdc->op, fw_bdc->value_index);

  // Head is given, then directly compute the gradient by using it as the
  // select value instead of 1.
  Expr true_value, false_value;
  if (head.defined()) {
    true_value =
      Call::make(head->dtype, head->op->name, head_vars_eqsel, Call::CallType::Halide, head->op, head->value_index);
    false_value = make_const(head->dtype, 0);
    used_head = true;
  } else {
    true_value = make_const(input->dtype, 1);
    false_value = make_const(input->dtype, 0);
  }

  auto condition = inp_tens_select == bd_tens;
  Expr body;
  if (CanProve(condition)) {
    body = true_value;
  } else {
    // Creating compute with select(bd_tens == inp, Head (or 1), 0)
    body = Select::make(condition, true_value, false_value);
  }

  return ComputeOpNode::make(op->name + "_jacobian", "", {}, jac_axis, {body}).output(0);
}

Tensor BroadcastAndSelect(const Tensor &input, const Expr &condition, const Tensor &head, bool &used_head,
                          const Array<IterVar> &fw_axis, const Array<IterVar> &jac_axis) {
  Array<Expr> head_vars_eqsel;
  // Creating args (indexing) for head tensor
  for (auto ax : fw_axis) {
    head_vars_eqsel.push_back(ax);
  }

  Expr true_value, false_value;
  if (head.defined()) {
    true_value =
      Call::make(head->dtype, head->op->name, head_vars_eqsel, Call::CallType::Halide, head->op, head->value_index);
    false_value = make_const(head->dtype, 0);
    used_head = true;
  } else {
    true_value = make_const(input->dtype, 1);
    false_value = make_const(input->dtype, 0);
  }

  Expr body;
  if (CanProve(condition)) {
    body = true_value;
  } else {
    // Creating compute with select(bd_tens == inp, Head (or 1), 0)
    body = Select::make(condition, true_value, false_value);
  }

  return ComputeOpNode::make(input->op->name + "_broadcast", topi::kBroadcast, {}, jac_axis, {body}).output(0);
}

bool CheckCombiner(const CommReducer &combiner) {
  if (combiner->result.size() == 2) {
    auto forward_combiner = combiner->result[1];
    auto max = forward_combiner.as<Max>();
    auto min = forward_combiner.as<Min>();
    // If we have a max or min reduction
    if ((max && max->a.as<Variable>() && max->b.as<Variable>()) ||
        (min && min->a.as<Variable>() && min->b.as<Variable>())) {
      return true;
    }
  } else if (combiner->result.size() == 1) {
    auto new_combiner = combiner->result[0];
    auto add = new_combiner.as<Add>();
    if (add && add->a.as<Variable>() && add->b.as<Variable>()) {
      return true;
    }
  }
  return false;
}

Tensor OptimizeReduction(Tensor &tensor, const ComputeOpNode *op, const Reduce *red, const Tensor &output,
                         const Tensor &input, const Array<Expr> &new_shape, const Array<IterVar> &new_axis,
                         const Tensor &head, bool &used_head) {
  CHECK(red);
  if (!CheckCombiner(red->combiner)) {
    return tensor;
  }

  // If the source at position 0 is a cast like (float(condition))
  // if not, return original tensor
  auto cast_cond = red->source[0].as<Cast>();
  if (!cast_cond) {
    return tensor;
  }

  auto remove_axis = RedAxisToRangeMutator();
  auto condition = remove_axis.RemoveAxisFromExpr(red->axis, cast_cond->value, new_axis);
  // If the new expression is not valid, we return the tensor with no changes
  if (!remove_axis.IsValid()) {
    return tensor;
  }

  // Arrys to store iterators and indexes for input, broadcast, forward compute and jacobian
  Array<Expr> input_vars_eqsel;
  Array<Expr> args;
  Array<IterVar> jac_axis;
  Array<IterVar> opt_bcast_axis;
  Array<IterVar> fw_axis;
  RemoveRedundantDimensions(remove_axis.GetIterToRemove(), new_axis, op, condition, jac_axis, opt_bcast_axis, fw_axis,
                            args, input_vars_eqsel);
  if (red->combiner->result[1].as<Min>() || red->combiner->result[1].as<Max>()) {
    auto source_call = red->source[1].as<Call>();
    if (!source_call) {
      return tensor;
    }

    auto fw_comp = GetForwardCompute(red, op, input, output, source_call);
    // create a broadcast from oh, ow -> oh, ow, h, w
    auto fw_bdc = BroadcastToCond(fw_comp, condition, opt_bcast_axis, fw_axis, red->combiner->identity_element[1]);
    tensor = BuildSelectFromBdcast(op, input, fw_bdc, jac_axis, args, input_vars_eqsel, fw_axis, head, used_head);
  } else if (red->combiner->result[0].as<Add>()) {
    tensor = BroadcastAndSelect(input, condition, head, used_head, fw_axis, jac_axis);
  }

  // If the head was used, then reduce sum over the head indexes
  if (used_head) {
    Array<Integer> sum_indexes;
    auto new_red_axis = remove_axis.FindNewRedAxis(op->axis);

    for (size_t i = 0; i < jac_axis.size(); i++) {
      if (new_red_axis.find(jac_axis[i]->var->name_hint) != new_red_axis.end()) {
        sum_indexes.push_back(i);
      }
    }
    // sum the indexes that are not part of the input shape.
    if (!sum_indexes.empty()) {
      tensor = topi::sum(tensor, sum_indexes);
    }
  }

  return tensor;
}
}  // namespace ir
}  // namespace akg
