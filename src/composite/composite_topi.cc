/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "topi/elemwise.h"
#include "topi/reduction.h"
#include "topi/broadcast.h"
#include "pass/utils.h"
#include "composite/util.h"

namespace akg {
#define TOPI_TWO_INPUTS_CALL(ins, rv, fn)                                             \
  do {                                                                                \
    auto inputs = ins[0].operator Array<NodeRef>();                                   \
    CHECK_EQ(inputs.size(), 2);                                                       \
    if (inputs[0]->IsInstance<TensorNode>() && inputs[1]->IsInstance<TensorNode>()) { \
      *rv = fn(Downcast<Tensor>(inputs[0]), Downcast<Tensor>(inputs[1]));             \
    } else if (inputs[0]->IsInstance<TensorNode>()) {                                 \
      *rv = fn(Downcast<Tensor>(inputs[0]), Downcast<Expr>(inputs[1]));               \
    } else if (inputs[1]->IsInstance<TensorNode>()) {                                 \
      *rv = fn(Downcast<Expr>(inputs[0]), Downcast<Tensor>(inputs[1]));               \
    } else {                                                                          \
      *rv = fn(Downcast<Expr>(inputs[0]), Downcast<Expr>(inputs[1]));                 \
    }                                                                                 \
  } while (0);

#define TOPI_ONE_INPUT_CALL(ins, rv, fn)            \
  do {                                              \
    auto inputs = ins[0].operator Array<NodeRef>(); \
    CHECK_EQ(inputs.size(), 1);                     \
    CHECK(inputs[0]->IsInstance<TensorNode>());     \
    *rv = fn(Downcast<Tensor>(inputs[0]));          \
  } while (0);

#define TOPI_ONE_INPUT_ONE_ATTR_CALL(ins, rv, fn, get_attr)    \
  do {                                                         \
    auto inputs = ins[0].operator Array<NodeRef>();            \
    CHECK_EQ(inputs.size(), 1);                                \
    CHECK(inputs[0]->IsInstance<TensorNode>());                \
    auto attrs = ins[1].operator Array<NodeRef>();             \
    CHECK_GE(attrs.size(), 1);                                 \
    *rv = fn(Downcast<Tensor>(inputs[0]), get_attr(attrs[0])); \
  } while (0);

Array<Integer> ArrayOrInt(const NodeRef &arg) {
  if (arg->IsInstance<IntImm>() || arg->IsInstance<UIntImm>()) {
    Array<Integer> result;
    result.push_back(Downcast<Integer>(arg));
    return result;
  } else {
    return Downcast<Array<Integer>>(arg);
  }
}

std::string GetString(const NodeRef &arg) {
  auto val = arg.as<StringImm>();
  CHECK(val) << "Input arg is not a string";
  return val->value;
}

void CommonCompare(TVMArgs args, TVMRetValue *rv, const std::string &cmp) {
  CHECK_GE(args.size(), 1);
  auto inputs = args[0].operator Array<NodeRef>();
  CHECK_GE(inputs.size(), 2);

  std::string name = "T_" + cmp + "_";
  Expr true_expr = make_const(Float(32), 1);
  Expr false_expr = make_const(Float(32), 0);
  air::FCompute fcompute;

  if (inputs[0]->IsInstance<TensorNode>()) {
    auto tensor0 = Downcast<Tensor>(inputs[0]);
    true_expr = make_const(tensor0->dtype, 1);
    false_expr = make_const(tensor0->dtype, 0);
    if (inputs[1]->IsInstance<TensorNode>()) {
      auto tensor1 = Downcast<Tensor>(inputs[1]);
      (void)name.append(tensor0->op->name).append("_").append(tensor1->op->name);
      if (cmp == "GreaterEqual") {
        fcompute = [&](const Array<Var> &indices) {
          return Select::make(tensor0(indices) >= tensor1(indices), true_expr, false_expr);
        };
      } else if (cmp == "LessEqual") {
        fcompute = [&](const Array<Var> &indices) {
          return Select::make(tensor0(indices) <= tensor1(indices), true_expr, false_expr);
        };
      }
      *rv = compute(tensor0->shape, fcompute, name);
    } else {
      CHECK(inputs[1]->IsInstance<ExprNode>());
      auto expr1 = Downcast<Expr>(inputs[1]);
      (void)name.append(tensor0->op->name);
      if (cmp == "GreaterEqual") {
        fcompute = [&](const Array<Var> &indices) {
          return Select::make(tensor0(indices) >= expr1, true_expr, false_expr);
        };
      } else if (cmp == "LessEqual") {
        fcompute = [&](const Array<Var> &indices) {
          return Select::make(tensor0(indices) <= expr1, true_expr, false_expr);
        };
      }
      *rv = compute(tensor0->shape, fcompute, name);
    }
  } else if (inputs[1]->IsInstance<TensorNode>()) {
    auto tensor1 = Downcast<Tensor>(inputs[1]);
    true_expr = make_const(tensor1->dtype, 1);
    false_expr = make_const(tensor1->dtype, 0);
    CHECK(inputs[0]->IsInstance<ExprNode>());
    auto expr0 = Downcast<Expr>(inputs[0]);
    (void)name.append(tensor1->op->name);
    if (cmp == "GreaterEqual") {
      fcompute = [&](const Array<Var> &indices) {
        return Select::make(expr0 >= tensor1(indices), true_expr, false_expr);
      };
    } else if (cmp == "LessEqual") {
      fcompute = [&](const Array<Var> &indices) {
        return Select::make(expr0 <= tensor1(indices), true_expr, false_expr);
      };
    }
    *rv = compute(tensor1->shape, fcompute, name);
  } else {
    CHECK(inputs[0]->IsInstance<ExprNode>());
    CHECK(inputs[1]->IsInstance<ExprNode>());
    // scaler >= scaler
    auto expr0 = Downcast<Expr>(inputs[0]);
    auto expr1 = Downcast<Expr>(inputs[1]);
    (void)name.append("scalar");
    if (cmp == "GreaterEqual") {
      fcompute = [&](const Array<Var> &indices) { return Select::make(expr0 >= expr1, true_expr, false_expr); };
    } else if (cmp == "LessEqual") {
      fcompute = [&](const Array<Var> &indices) { return Select::make(expr0 <= expr1, true_expr, false_expr); };
    }
    *rv = compute({Expr(1)}, fcompute, name);
  }
}

void CommonSelect(NodeRef a, NodeRef b, NodeRef c, NodeRef d, TVMRetValue *rv, bool ge) {
  bool a_is_expr = a->IsInstance<ExprNode>();
  bool a_is_tensor = a->IsInstance<TensorNode>();
  bool b_is_expr = b->IsInstance<ExprNode>();
  bool b_is_tensor = b->IsInstance<TensorNode>();
  bool c_is_tensor = c->IsInstance<TensorNode>();
  bool c_is_expr = c->IsInstance<ExprNode>();
  bool d_is_defined = d.defined();
  bool d_is_tensor = (d_is_defined && d->IsInstance<TensorNode>());
  bool d_is_expr = (d_is_defined && d->IsInstance<ExprNode>());
  CHECK(a_is_expr || a_is_tensor) << "Input1 should be of type Expr or Tensor";
  CHECK(b_is_expr || b_is_tensor) << "Input2 should be of type Expr or Tensor";
  CHECK(c_is_expr || c_is_tensor) << "Input3 should be of type Expr or Tensor";
  CHECK(d_is_expr || d_is_tensor) << "Input4 should be of type Expr or Tensor";
  CHECK((!d_is_defined && c_is_tensor) || !(c_is_expr && d_is_expr)) << "Input3 or input4 should be of type Tensor";

  Tensor a_tensor;
  Tensor b_tensor;
  Tensor c_tensor;
  Tensor d_tensor;
  auto shape = Downcast<Tensor>(c_is_tensor ? c : d)->shape;
  auto dtype = Downcast<Tensor>(c_is_tensor ? c : d)->dtype;
  c_tensor =
    c_is_tensor ? Downcast<Tensor>(c) : compute(shape, [&](const Array<Var> &indices) { return Downcast<Expr>(c); });
  if (d_is_tensor) {
    d_tensor = Downcast<Tensor>(d);
  } else if (d_is_defined) {
    d_tensor = compute(shape, [&](const Array<Var> &indices) { return Downcast<Expr>(d); });
  }

  if (a_is_expr) {
    a_tensor = compute(shape, [&](const Array<Var> &indices) { return Downcast<Expr>(a); });
  } else {
    a_tensor = topi::broadcast_to(Downcast<Tensor>(a), shape);
  }

  if (b_is_expr) {
    b_tensor = compute(shape, [&](const Array<Var> &indices) { return Downcast<Expr>(b); });
  } else {
    b_tensor = topi::broadcast_to(Downcast<Tensor>(b), shape);
  }

  if (dtype == Float(16)) {
    a_tensor = topi::cast(a_tensor, Float(32));
    b_tensor = topi::cast(b_tensor, Float(32));
    c_tensor = topi::cast(c_tensor, Float(32));
    if (d_tensor.defined()) {
      d_tensor = topi::cast(d_tensor, Float(32));
    }
  }

  Tensor cmp_value;
  auto sub_ab = ge ? topi::subtract(a_tensor, b_tensor) : topi::subtract(b_tensor, a_tensor);
  if (dtype == Int(32)) {
    auto add_min = topi::add(sub_ab, make_const(dtype, 1));
    auto vmax_zero = topi::maximum(add_min, make_const(dtype, 0));
    cmp_value = topi::minimum(vmax_zero, make_const(dtype, 1));
  } else {
    CHECK_EQ(dtype, Float(32));
    auto min_value = make_const(dtype, pow(2, -126));
    auto max_value = make_const(dtype, pow(2, 62));
    auto mid_value = make_const(dtype, pow(2, 2));
    auto data_zero =
      ge ? topi::multiply(a_tensor, make_const(dtype, 0)) : topi::multiply(b_tensor, make_const(dtype, 0));
    auto add_min = topi::add(sub_ab, min_value);
    auto vmax_zero = topi::maximum(add_min, data_zero);
    auto vmin_min = topi::minimum(vmax_zero, min_value);
    auto vmul_max = topi::multiply(vmin_min, max_value);
    vmul_max = topi::multiply(vmul_max, max_value);
    cmp_value = topi::multiply(vmul_max, mid_value);
  }

  Tensor res;
  if (d_tensor.defined()) {
    auto cmp_value_invert = topi::subtract(make_const(cmp_value->dtype, 1), cmp_value);
    auto res1 = topi::multiply(c_tensor, cmp_value);
    auto res2 = topi::multiply(d_tensor, cmp_value_invert);
    res = topi::add(res1, res2);
  } else {
    res = topi::multiply(c_tensor, cmp_value);
  }

  if (dtype == Float(16)) {
    res = topi::cast(res, Float(16));
  }

  *rv = res;
}

void CommonMaximumGrad(TVMArgs args, TVMRetValue *rv, bool ge) {
  CHECK_GE(args.size(), 1);
  auto inputs = args[0].operator Array<NodeRef>();
  CHECK_GE(inputs.size(), 3);
  CHECK(inputs[2]->IsInstance<TensorNode>());
  CommonSelect(inputs[0], inputs[1], inputs[2], make_const(Int(32), 0), rv, ge);
}

TVM_REGISTER_GLOBAL("Abs").set_body([](TVMArgs args, TVMRetValue *rv) { TOPI_ONE_INPUT_CALL(args, rv, topi::abs); });

TVM_REGISTER_GLOBAL("Round").set_body([](TVMArgs args, TVMRetValue *rv) {
  auto call = [](const air::Tensor &tensor) {
    std::string name = "T_round_" + tensor->op->name;
    return compute(
      tensor->shape, [&](const Array<Var> &i) { return air::cast(air::Int(32), air::round(tensor(i))); }, name,
      topi::kElementWise);
  };
  TOPI_ONE_INPUT_CALL(args, rv, call);
});

TVM_REGISTER_GLOBAL("Neg").set_body([](TVMArgs args, TVMRetValue *rv) {
  auto call = [](const Tensor &tensor) {
    std::string name = "T_negative_" + tensor->op->name;
    return topi::negative(tensor, name);
  };
  TOPI_ONE_INPUT_CALL(args, rv, call);
});

TVM_REGISTER_GLOBAL("Exp").set_body([](TVMArgs args, TVMRetValue *rv) { TOPI_ONE_INPUT_CALL(args, rv, topi::exp); });

TVM_REGISTER_GLOBAL("TensorAdd").set_body([](TVMArgs args, TVMRetValue *rv) {
  TOPI_TWO_INPUTS_CALL(args, rv, topi::add);
});

TVM_REGISTER_GLOBAL("RealDiv").set_body([](TVMArgs args, TVMRetValue *rv) {
  TOPI_TWO_INPUTS_CALL(args, rv, topi::divide);
});

TVM_REGISTER_GLOBAL("Mul").set_body([](TVMArgs args, TVMRetValue *rv) {
  TOPI_TWO_INPUTS_CALL(args, rv, topi::multiply);
});

TVM_REGISTER_GLOBAL("Minimum").set_body([](TVMArgs args, TVMRetValue *rv) {
  TOPI_TWO_INPUTS_CALL(args, rv, topi::minimum);
});

TVM_REGISTER_GLOBAL("Maximum").set_body([](TVMArgs args, TVMRetValue *rv) {
  TOPI_TWO_INPUTS_CALL(args, rv, topi::maximum);
});

TVM_REGISTER_GLOBAL("Log").set_body([](TVMArgs args, TVMRetValue *rv) { TOPI_ONE_INPUT_CALL(args, rv, topi::log); });

TVM_REGISTER_GLOBAL("ReduceSum").set_body([](TVMArgs args, TVMRetValue *rv) {
  CHECK_GE(args.size(), 2);
  auto attrs = args[1].operator air::Array<air::NodeRef>();
  CHECK_GE(attrs.size(), 2);
  air::Array<air::Integer> axis = ArrayOrInt(attrs[0]);
  auto keepdims = static_cast<bool>(ir::GetInt32Const(Downcast<Expr>(attrs[1])));

  if (attrs.size() == 3) {
    Map<std::string, NodeRef> com_attrs;
    com_attrs.Set("atomic_add", attrs[2]);
    auto name = GetString(attrs[2]);
    auto call = [&axis, &keepdims, &name, &com_attrs](const air::Tensor &tensor) {
      auto ndim = tensor->shape.size();
      CHECK_NE(ndim, 0) << "Cannot reduce a 0 dim Tensor";
      auto reduce_axes = topi::GetRealAxis(static_cast<int>(ndim), axis);
      auto squeeze_axes = keepdims ? std::vector<int>() : reduce_axes;
      auto target_shape = topi::MakeReduceTargetShape(reduce_axes, tensor, keepdims, true);
      auto r_axes = topi::MakeReduceAxes(reduce_axes, tensor);

      auto reduce_compute = [&](const Array<Var> &indices) {
        Array<Expr> eval_range;
        Array<Var> eval_indices;
        int arg_counter = 0;
        int red_counter = 0;

        for (size_t i = 0; i < tensor->shape.size(); ++i) {
          bool squeeze_i = std::find(squeeze_axes.begin(), squeeze_axes.end(), i) != squeeze_axes.end();
          if (std::find(reduce_axes.begin(), reduce_axes.end(), i) != reduce_axes.end()) {
            // real_axis contains i
            eval_range.push_back(r_axes[red_counter]);
            eval_indices.push_back(r_axes[red_counter]->var);
            red_counter++;
            squeeze_i ? arg_counter : ++arg_counter;
            continue;
          }
          eval_range.push_back(indices[arg_counter]);
          arg_counter++;
        }
        return air::sum(tensor(eval_range), r_axes);
      };

      return compute(target_shape, reduce_compute, name, topi::kCommReduce, com_attrs);
    };
    TOPI_ONE_INPUT_CALL(args, rv, call);
  } else {
    auto call = [&axis, &keepdims](const air::Tensor &tensor) { return topi::sum(tensor, axis, keepdims); };
    TOPI_ONE_INPUT_CALL(args, rv, call);
  }
});

TVM_REGISTER_GLOBAL("Pow").set_body([](TVMArgs args, TVMRetValue *rv) { TOPI_TWO_INPUTS_CALL(args, rv, topi::power); });

TVM_REGISTER_GLOBAL("Sub").set_body([](TVMArgs args, TVMRetValue *rv) {
  TOPI_TWO_INPUTS_CALL(args, rv, topi::subtract);
});

TVM_REGISTER_GLOBAL("Rsqrt").set_body([](TVMArgs args, TVMRetValue *rv) {
  TOPI_ONE_INPUT_CALL(args, rv, topi::rsqrt);
});

TVM_REGISTER_GLOBAL("Sqrt").set_body([](TVMArgs args, TVMRetValue *rv) { TOPI_ONE_INPUT_CALL(args, rv, topi::sqrt); });

TVM_REGISTER_GLOBAL("ExpandDims").set_body([](TVMArgs args, TVMRetValue *rv) {
  auto ref = [](NodeRef attr) -> int {
    auto axis = ir::GetInt32Const(Downcast<Expr>(attr));
    return axis;
  };

  TOPI_ONE_INPUT_ONE_ATTR_CALL(args, rv, topi::expand_dims, ref);
});

TVM_REGISTER_GLOBAL("Reshape").set_body([](TVMArgs args, TVMRetValue *rv) {
  auto ref = [](NodeRef attr) -> Array<Expr> {
    auto shape = Downcast<Array<Integer>>(attr);
    CHECK(!shape.empty());
    Array<Expr> newshape;
    for (auto s : shape) {
      newshape.push_back(s);
    }
    return newshape;
  };

  TOPI_ONE_INPUT_ONE_ATTR_CALL(args, rv, topi::reshape, ref);
});

TVM_REGISTER_GLOBAL("Cast").set_body([](TVMArgs args, TVMRetValue *rv) {
  auto type_mapping_copy = type_mapping;
  auto ref = [&type_mapping_copy](NodeRef attr) -> Type {
    CHECK(attr->IsInstance<StringImm>());
    std::string dtype_str = attr.as<StringImm>()->value;
    if (type_mapping_copy.find(dtype_str) == type_mapping_copy.end()) {
      LOG(FATAL) << "Not support dtype: " << dtype_str;
    }
    return type_mapping_copy[dtype_str];
  };

  auto call = [](const Tensor &tensor, Type type) {
    std::string name = "T_cast_" + tensor->op->name;
    if (tensor->dtype == air::Bool() && type == air::Float(32)) {
      return topi::cast(topi::cast(tensor, air::Float(16), name), type, name);
    } else if (tensor->dtype == air::Float(32) && type == air::Bool()) {
      const char *runtime_mode = std::getenv("RUNTIME_MODE");
      if (runtime_mode == nullptr || (runtime_mode != nullptr && std::strstr(runtime_mode, "cloud") == nullptr)) {
        auto tmp = topi::cast(tensor, air::Float(16), name + "tmp");
        auto zero = make_zero(air::Float(16));
        auto res = topi::not_equal(tmp, zero);
        return topi::cast(res, type, name);
      } else {
        auto zero = make_zero(tensor->dtype);
        return topi::not_equal(tensor, zero);
      }
    }

    return topi::cast(tensor, type, name);
  };
  TOPI_ONE_INPUT_ONE_ATTR_CALL(args, rv, call, ref);
});

TVM_REGISTER_GLOBAL("Tile").set_body([](TVMArgs args, TVMRetValue *rv) {
  auto ref = [](NodeRef attr) -> Array<Integer> {
    auto multiples = Downcast<Array<Integer>>(attr);
    CHECK(!multiples.empty());
    return multiples;
  };

  auto call = [](const Tensor &tensor, const Array<Integer> &multiples) {
    std::string name = "T_tile_" + tensor->op->name;
    return topi::tile(tensor, multiples, name);
  };
  TOPI_ONE_INPUT_ONE_ATTR_CALL(args, rv, call, ref);
});

TVM_REGISTER_GLOBAL("AddN").set_body([](TVMArgs args, TVMRetValue *rv) {
  CHECK_GE(args.size(), 1);
  auto arr_t = args[0].operator Array<Tensor>();
  CHECK(!arr_t.empty());
  *rv = topi::elemwise_sum(arr_t);
});

TVM_REGISTER_GLOBAL("ReduceMax").set_body([](TVMArgs args, TVMRetValue *rv) {
  CHECK_GE(args.size(), 2);
  auto attrs = args[1].operator Array<NodeRef>();
  CHECK_GE(attrs.size(), 2);
  auto axis = ArrayOrInt(attrs[0]);
  CHECK(attrs[1]->IsInstance<ExprNode>());
  auto keepdims = static_cast<bool>(ir::GetInt32Const(Downcast<Expr>(attrs[1])));

  auto call = [&axis, &keepdims](const Tensor &tensor) { return topi::max(tensor, axis, keepdims); };
  TOPI_ONE_INPUT_CALL(args, rv, call);
});

TVM_REGISTER_GLOBAL("ReduceMin").set_body([](TVMArgs args, TVMRetValue *rv) {
  CHECK_GE(args.size(), 2);
  auto attrs = args[1].operator Array<NodeRef>();
  CHECK_GE(attrs.size(), 2);
  auto axis = ArrayOrInt(attrs[0]);
  CHECK(attrs[1]->IsInstance<ExprNode>());
  auto keepdims = static_cast<bool>(ir::GetInt32Const(Downcast<Expr>(attrs[1])));

  auto call = [&axis, &keepdims](const Tensor &tensor) { return topi::min(tensor, axis, keepdims); };
  TOPI_ONE_INPUT_CALL(args, rv, call);
});

TVM_REGISTER_GLOBAL("OneHot").set_body([](TVMArgs args, TVMRetValue *rv) {
  CHECK_GE(args.size(), 2);
  auto inputs = args[0].operator Array<NodeRef>();
  CHECK_GE(inputs.size(), 3);
  CHECK(inputs[0]->IsInstance<TensorNode>());
  CHECK(inputs[1]->IsInstance<ExprNode>());
  CHECK(inputs[2]->IsInstance<ExprNode>());
  auto indices = Downcast<Tensor>(inputs[0]);
  auto on_value = Downcast<Expr>(inputs[1]);
  auto off_value = Downcast<Expr>(inputs[2]);

  auto attrs = args[1].operator Array<NodeRef>();
  CHECK_GE(attrs.size(), 2);
  CHECK(attrs[0]->IsInstance<ExprNode>());
  CHECK(attrs[1]->IsInstance<ExprNode>());
  auto depth = ir::GetInt32Const(Downcast<Expr>(attrs[0]));
  auto axis = ir::GetInt32Const(Downcast<Expr>(attrs[1]));

  *rv = topi::one_hot(indices, on_value, off_value, depth, axis, indices->dtype);
});

TVM_REGISTER_GLOBAL("Equal").set_body([](TVMArgs args, TVMRetValue *rv) {
  CHECK_GE(args.size(), 1);
  auto inputs = args[0].operator Array<NodeRef>();
  CHECK(inputs[0]->IsInstance<TensorNode>());
  CHECK(inputs[1]->IsInstance<TensorNode>());
  auto data1 = Downcast<Tensor>(inputs[0]);
  auto data2 = Downcast<Tensor>(inputs[1]);

  CHECK_EQ(data1->shape.size(), data2->shape.size())
    << "x and y must have the same shape. Got different number of dimension: " << data1->shape.size() << " vs "
    << data2->shape.size();
  CHECK_EQ(data1->dtype, data2->dtype) << "x and y must have the same dtype: " << data1->dtype << " vs "
                                       << data2->dtype;
  Expr true_expr = make_const(data1->dtype, true);
  Expr false_expr = make_const(data1->dtype, false);

  std::string name = "T_equal_";
  (void)name.append(data1->op->name).append("_").append(data2->op->name);
  *rv = compute(
    data1->shape,
    [&](const Array<Var> &indices) { return Select::make(data1(indices) == data2(indices), true_expr, false_expr); },
    name, topi::kBroadcast);
});

TVM_REGISTER_GLOBAL("Reciprocal").set_body([](TVMArgs args, TVMRetValue *rv) {
  auto call = [](const Tensor &tensor) { return topi::divide(make_const(tensor->dtype, 1.0), tensor); };
  TOPI_ONE_INPUT_CALL(args, rv, call);
});

TVM_REGISTER_GLOBAL("GreaterEqual").set_body([](TVMArgs args, TVMRetValue *rv) {
  CommonCompare(args, rv, "GreaterEqual");
});

TVM_REGISTER_GLOBAL("LessEqual").set_body([](TVMArgs args, TVMRetValue *rv) {
  CHECK_GE(args.size(), 1);
  auto inputs = args[0].operator Array<NodeRef>();
  CHECK(inputs[0]->IsInstance<TensorNode>());
  CHECK(inputs[1]->IsInstance<TensorNode>());
  auto data1 = Downcast<Tensor>(inputs[0]);
  auto data2 = Downcast<Tensor>(inputs[1]);

  CHECK_EQ(data1->shape.size(), data2->shape.size())
    << "x and y must have the same shape. Got different number of dimension: " << data1->shape.size() << " vs "
    << data2->shape.size();
  CHECK_EQ(data1->dtype, data2->dtype) << "x and y must have the same dtype: " << data1->dtype << " vs "
                                       << data2->dtype;
  Expr true_expr = make_const(data1->dtype, 1);
  Expr false_expr = make_const(data1->dtype, 0);

  std::string name = "T_less_equal_";
  (void)name.append(data1->op->name).append("_").append(data2->op->name);
  *rv = compute(
    data1->shape,
    [&](const Array<Var> &indices) { return Select::make(data1(indices) <= data2(indices), true_expr, false_expr); },
    name, topi::kBroadcast);
});

TVM_REGISTER_GLOBAL("ZerosLike").set_body([](TVMArgs args, TVMRetValue *rv) {
  CHECK_GE(args.size(), 1);
  auto inputs = args[0].operator Array<NodeRef>();
  CHECK(inputs[0]->IsInstance<TensorNode>());
  auto data = Downcast<Tensor>(inputs[0]);
  std::string name = "T_zero_like_";
  // In some case, zeros like can use scalar 0 directly
  *rv = compute(
    data->shape, [&](const Array<Var> &indices) { return make_const(data->dtype, 0); }, name, topi::kBroadcast);
});

TVM_REGISTER_GLOBAL("Select").set_body([](TVMArgs args, TVMRetValue *rv) {
  CHECK_GE(args.size(), 1);
  auto inputs = args[0].operator Array<NodeRef>();
  CHECK(inputs[0]->IsInstance<TensorNode>());
  CHECK(inputs[1]->IsInstance<TensorNode>());
  CHECK(inputs[2]->IsInstance<TensorNode>());
  auto condition = Downcast<Tensor>(inputs[0]);
  auto x = Downcast<Tensor>(inputs[1]);
  auto y = Downcast<Tensor>(inputs[2]);
  *rv = topi::where(condition, x, y);
});

TVM_REGISTER_GLOBAL("SelectGE").set_body([](TVMArgs args, TVMRetValue *rv) {
  CHECK_GE(args.size(), 1);
  auto inputs = args[0].operator Array<NodeRef>();
  CHECK_EQ(inputs.size(), 4);

  if ((inputs[1]->IsInstance<ExprNode>() && akg::ir::IsZero(Downcast<Expr>(inputs[1]))) &&
      (inputs[3]->IsInstance<ExprNode>() && akg::ir::IsZero(Downcast<Expr>(inputs[3]))) &&
      inputs[0]->IsInstance<TensorNode>() && inputs[2]->IsInstance<TensorNode>()) {
    // Rewrite relu grad
    air::FCompute fcompute;
    Tensor x_tensor = Downcast<Tensor>(inputs[0]);
    Tensor dout_tensor = Downcast<Tensor>(inputs[2]);

    Expr help_min = 1;
    Expr help_rec_one = 1;
    Expr help_rec_sec = 1;
    if (x_tensor->dtype == Float(32)) {
      help_min = make_const(x_tensor->dtype, pow(2, -126));
      help_rec_one = make_const(x_tensor->dtype, pow(2, 38));
      help_rec_sec = make_const(x_tensor->dtype, pow(2, 44));
    } else if (x_tensor->dtype == Float(16)) {
      help_min = make_const(x_tensor->dtype, pow(2, -24));
      help_rec_one = make_const(x_tensor->dtype, pow(2, 12));
      help_rec_sec = make_const(x_tensor->dtype, pow(2, 12));
    }

    auto res = topi::minimum(x_tensor, help_min);
    res = topi::maximum(res, make_zero(x_tensor->dtype));
    res = topi::multiply(res, help_rec_one);
    if (x_tensor->dtype == Float(32)) {
      res = topi::multiply(res, help_rec_sec);
    }
    res = topi::multiply(res, help_rec_sec);
    if (res->dtype != dout_tensor->dtype) {
      res = topi::cast(res, dout_tensor->dtype, "T_cast_GE");
    }
    *rv = topi::multiply(dout_tensor, res);
  } else if (inputs[3]->IsInstance<ExprNode>() && akg::ir::IsZero(Downcast<Expr>(inputs[3]))) {
    CommonMaximumGrad(args, rv, true);
  } else {
    LOG(FATAL) << "Common select ge has not been implemented yet.";
  }
});

TVM_REGISTER_GLOBAL("SelectLE").set_body([](TVMArgs args, TVMRetValue *rv) {
  CHECK_GE(args.size(), 1);
  auto inputs = args[0].operator Array<NodeRef>();
  CHECK_EQ(inputs.size(), 4);
  if (inputs[3]->IsInstance<ExprNode>() && akg::ir::IsZero(Downcast<Expr>(inputs[3]))) {
    CommonMaximumGrad(args, rv, false);
  } else {
    LOG(FATAL) << "Common select le has not been implemented yet.";
  }
});

TVM_REGISTER_GLOBAL("SelectGT").set_body([](TVMArgs args, TVMRetValue *rv) {
  CHECK_GE(args.size(), 1);
  auto inputs = args[0].operator Array<NodeRef>();
  CHECK_EQ(inputs.size(), 4);
  CommonSelect(inputs[0], inputs[1], inputs[3], inputs[2], rv, false);
});

TVM_REGISTER_GLOBAL("SelectLT").set_body([](TVMArgs args, TVMRetValue *rv) {
  CHECK_GE(args.size(), 1);
  auto inputs = args[0].operator Array<NodeRef>();
  CHECK_EQ(inputs.size(), 4);
  CommonSelect(inputs[0], inputs[1], inputs[3], inputs[2], rv, true);
});

TVM_REGISTER_GLOBAL("InplaceAssign").set_body([](TVMArgs args, TVMRetValue *rv) {
  CHECK_GE(args.size(), 1);
  auto inputs = args[0].operator Array<NodeRef>();
  CHECK_GE(inputs.size(), 2);
  bool in2_is_expr = inputs[1]->IsInstance<ExprNode>();
  bool in2_is_tensor = inputs[1]->IsInstance<TensorNode>();
  CHECK(inputs[0]->IsInstance<TensorNode>()) << "Input1 should be of type Tensor";
  CHECK(in2_is_expr || in2_is_tensor) << "Input2 should be of type Expr or Tensor";
  auto ref = Downcast<Tensor>(inputs[0]);
  auto val = in2_is_expr ? compute(ref->shape, [&](const Array<Var> &indices) { return Downcast<Expr>(inputs[1]); })
                         : Downcast<Tensor>(inputs[1]);
  auto buf = decl_buffer(val->shape, val->dtype, ref->op->name);
  *rv = Map<Tensor, Buffer>({{ref, buf}, {val, buf}});
});

TVM_REGISTER_GLOBAL("EquivFormat").set_body([](TVMArgs args, TVMRetValue *rv) {
  CHECK_GE(args.size(), 1);
  auto inputs = args[0].operator Array<NodeRef>();
  CHECK(inputs[0]->IsInstance<TensorNode>());
  auto ref = Downcast<Tensor>(inputs[0]);
  *rv = ref;
});

TVM_REGISTER_GLOBAL("AddMinValue").set_body([](TVMArgs args, TVMRetValue *rv) {
  CHECK_GE(args.size(), 1);
  auto inputs = args[0].operator Array<NodeRef>();
  CHECK(inputs[0]->IsInstance<TensorNode>());
  auto tensor = Downcast<Tensor>(inputs[0]);
  Expr min_value = 0;
  if (tensor->dtype == Float(32)) {
    min_value = make_const(Float(32), pow(2, -126));
    *rv = topi::add(tensor, min_value);
  } else if (tensor->dtype == Float(16)) {
    min_value = make_const(Float(16), pow(2, -24));
    *rv = topi::add(tensor, min_value);
  } else {
    *rv = tensor;
  }
});

TVM_REGISTER_GLOBAL("TransData").set_body([](TVMArgs args, TVMRetValue *rv) {
  CHECK_GE(args.size(), 2);
  auto inputs = args[0].operator Array<NodeRef>();
  auto attrs = args[1].operator Array<NodeRef>();
  CHECK_GE(inputs.size(), 1);
  CHECK(inputs[0]->IsInstance<TensorNode>());
  auto input_data = Downcast<Tensor>(inputs[0]);
  CHECK_GE(attrs.size(), 2);
  auto src_format = GetString(attrs[0]);
  auto dst_format = GetString(attrs[1]);
  auto input_shape = input_data->shape;
  auto cube_size = 16;
  // FRACTAL_NZ: zN fractal format
  if (src_format == "DefaultFormat" && dst_format == "FRACTAL_NZ") {
    if (input_data->dtype != Float(16) && input_data->dtype != Float(32)) {
      LOG(FATAL) << "dtype of input should be float16 or float32";
    }
    if (input_data->dtype == Float(32)) {
      input_data = topi::cast(input_data, Float(16));
    }
    CHECK_GE(input_shape.size(), 2);
    auto batch_dim = input_shape.size() - 2;
    auto m = input_shape[batch_dim];
    auto n = input_shape[batch_dim + 1];
    Array<Expr> output_shape;
    for (size_t i = 0; i < batch_dim; ++i) {
      output_shape.push_back(input_shape[i]);
    }
    auto m1 = truncdiv(m + cube_size - 1, cube_size);
    auto n1 = truncdiv(n + cube_size - 1, cube_size);
    output_shape.push_back(n1);
    output_shape.push_back(m1);
    output_shape.push_back(cube_size);
    output_shape.push_back(cube_size);
    auto fcompute = [&input_data, &m, &n, &batch_dim, &cube_size](const Array<Var> &indices) {
      Array<Expr> input_indice;
      for (size_t i = 0; i < batch_dim; ++i) {
        input_indice.push_back(indices[i]);
      }
      auto n1_indice = indices[batch_dim];
      auto m1_indice = indices[batch_dim + 1];
      auto m0_indice = indices[batch_dim + 2];
      auto n0_indice = indices[batch_dim + 3];
      auto m_indice = m1_indice * cube_size + m0_indice;
      auto n_indice = n1_indice * cube_size + n0_indice;
      input_indice.push_back(m_indice);
      input_indice.push_back(n_indice);
      auto res = if_then_else(m_indice >= m || n_indice >= n, make_zero(input_data->dtype), input_data(input_indice));
      return res;
    };
    auto name = "T_transdata_" + input_data->op->name;
    *rv = compute(output_shape, fcompute, name);
  } else if (src_format == "FRACTAL_NZ" && dst_format == "DefaultFormat") {
    if (input_data->dtype != Float(16) && input_data->dtype != Float(32)) {
      LOG(FATAL) << "dtype of input should be float16 or float32";
    }
    CHECK_GE(input_shape.size(), 4);
    auto batch_dim = input_shape.size() - 4;
    CHECK_GE(attrs.size(), 3);
    auto original_shape = Downcast<Array<Expr>>(attrs[2]);
    CHECK_EQ(original_shape.size(), batch_dim + 2);
    auto output_shape = original_shape;
    auto name = "T_transdata_" + input_data->op->name;
    auto fcompute = [&input_data, &batch_dim, &cube_size](const Array<Var> &indices) {
      Array<Expr> input_indice;
      for (size_t i = 0; i < batch_dim; ++i) {
        input_indice.push_back(indices[i]);
      }
      auto m_indice = indices[batch_dim];
      auto n_indice = indices[batch_dim + 1];
      auto m1_indice = truncdiv(m_indice, cube_size);
      auto m0_indice = truncmod(m_indice, cube_size);
      auto n1_indice = truncdiv(n_indice, cube_size);
      auto n0_indice = truncmod(n_indice, cube_size);
      input_indice.push_back(n1_indice);
      input_indice.push_back(m1_indice);
      input_indice.push_back(m0_indice);
      input_indice.push_back(n0_indice);
      return input_data(input_indice);
    };
    *rv = compute(output_shape, fcompute, name);
  } else {
    LOG(FATAL) << "TransData for src_format " << src_format << "and dst_format" << dst_format << " is not supported";
  }
});
}  // namespace akg
